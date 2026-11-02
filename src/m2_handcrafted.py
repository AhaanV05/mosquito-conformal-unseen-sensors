"""M2: pitch / harmonic hand-built features + gradient-boosted classifier.

Features are chosen to avoid obvious recording-setup cues (no clip duration, no absolute level): a foreground-minus-
background whitened spectrum (removes the channel colour, like the challenge winner's idea), the wingbeat fundamental
estimated by harmonic salience, harmonic amplitudes relative to the fundamental, and a few spectral-shape numbers.
Outputs match the M1 format: results/phase1/preds/m2_seed<seed>_<split>.npz plus the feature matrix for later steps.
"""
import argparse
import os
import random
import time
from math import gcd
from multiprocessing import Pool

import numpy as np
import soundfile as sf
from scipy.signal import get_window, resample_poly

from common import FEAT, RAW, RES, ensure_dirs, log_to, parse_id, read_ids, unseen_domain_by_species_id

SR = 8000
NFFT = 1024
HOP = 256
DF = SR / NFFT
WIN = get_window('hann', NFFT).astype(np.float32)
LO_BIN, HI_BIN = int(100 / DF), int(1500 / DF)      # energy band for foreground / background selection
FEAT_NAMES = (['wband%02d' % i for i in range(40)] + ['f0_hz', 'salience_max'] + ['h1_db'] +
              ['h%d_minus_h1' % k for k in range(2, 7)] + ['hnr_db', 'centroid_hz', 'spread_hz', 'rolloff_hz', 'am_cv'])
N_FEAT = len(FEAT_NAMES)


def load8k(path):
    x, sr = sf.read(path, dtype='float32', always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    peak = float(np.abs(x).max()) if x.size else 0.0
    return x / peak if peak > 0 else x


def clip_features(path):
    out = np.full(N_FEAT, np.nan, dtype=np.float32)
    try:
        x = load8k(path)
        if x.size < 256 or not np.any(x):
            return out
        if x.size < NFFT:
            x = np.pad(x, (0, NFFT - x.size))
        frames = np.lib.stride_tricks.sliding_window_view(x, NFFT)[::HOP]
        P = np.abs(np.fft.rfft(frames * WIN, axis=1)) ** 2
        e = P[:, LO_BIN:HI_BIN].sum(axis=1)
        order = np.argsort(e)
        n = len(order)
        n_fg = max(3, int(round(0.4 * n)))
        n_bg = max(2, int(round(0.3 * n)))
        fg_idx, bg_idx = order[-n_fg:], order[:n_bg]
        L = 10.0 * np.log10(P + 1e-12)
        W = L[fg_idx].mean(axis=0) - L[bg_idx].mean(axis=0)            # whitened spectrum in dB
        Wp = np.maximum(W, 0.0)
        Ws = np.maximum.reduce([np.roll(Wp, 1), Wp, np.roll(Wp, -1)])  # tolerate +-1 bin misalignment
        pooled = W[5:205].reshape(40, 5).mean(axis=1)
        cand = np.arange(20, 101)
        sal = np.zeros(len(cand))
        for k in range(1, 6):
            sal += Ws[np.minimum(cand * k, len(Ws) - 1)]
        c = int(cand[int(np.argmax(sal))])
        harm = np.array([Ws[min(int(round(k * c)), len(Ws) - 1)] for k in range(1, 7)])
        band = np.arange(13, 193)
        wb = Wp[band]
        tot = wb.sum() + 1e-9
        freqs = band * DF
        centroid = float((freqs * wb).sum() / tot)
        spread = float(np.sqrt(((freqs - centroid) ** 2 * wb).sum() / tot))
        csum = np.cumsum(wb) / tot
        rolloff = float(freqs[int(np.searchsorted(csum, 0.85))])
        harm_bins = {min(int(round(k * c)) + d, len(W) - 1) for k in range(1, 7) for d in (-1, 0, 1)}
        mask = np.zeros(len(W), dtype=bool)
        mask[list(harm_bins)] = True
        hnr = float(W[mask & (np.arange(len(W)) >= 13) & (np.arange(len(W)) < 193)].mean() -
                    W[~mask & (np.arange(len(W)) >= 13) & (np.arange(len(W)) < 193)].mean())
        ef = e[fg_idx]
        am_cv = float(ef.std() / (ef.mean() + 1e-12))
        out[:] = np.concatenate([pooled, [c * DF, sal.max()], [harm[0]], harm[1:] - harm[0],
                                 [hnr, centroid, spread, rolloff, am_cv]]).astype(np.float32)
    except Exception:
        pass
    return out


def _worker(fid):
    return clip_features(os.path.join(RAW, fid + '.wav'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cap', type=int, default=3000, help='max training clips per (species, domain) pair')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args()
    log = log_to('m2_handcrafted.log')
    ensure_dirs(FEAT, os.path.join(RES, 'phase1', 'preds'))

    train_all = read_ids('Training')
    rnd = random.Random(0)
    groups = {}
    for fid in train_all:
        groups.setdefault(parse_id(fid), []).append(fid)
    train_ids = []
    for key in sorted(groups):
        g = groups[key]
        train_ids += g if len(g) <= args.cap else rnd.sample(g, args.cap)
    val_ids, test_ids = read_ids('Validation'), read_ids('Test')
    log('train pool %d (cap %d per pair), validation %d, test %d' % (len(train_ids), args.cap, len(val_ids), len(test_ids)))

    feat_path = os.path.join(FEAT, 'm2_features.npz')
    if os.path.exists(feat_path):
        z = np.load(feat_path, allow_pickle=True)
        ids_all, X_all = list(z['ids']), z['X']
        log('loaded cached features %s' % (X_all.shape,))
    else:
        ids_all = train_ids + val_ids + test_ids
        t0 = time.time()
        feats = []
        with Pool(args.workers) as pool:
            for i, f in enumerate(pool.imap(_worker, ids_all, chunksize=64), 1):
                feats.append(f)
                if i % 10000 == 0:
                    log('features %d/%d (%.0f s)' % (i, len(ids_all), time.time() - t0))
        X_all = np.stack(feats)
        np.savez_compressed(feat_path, ids=np.array(ids_all), X=X_all, names=np.array(FEAT_NAMES))
        log('features done %s in %.0f s, NaN rows: %d' % (X_all.shape, time.time() - t0, int(np.isnan(X_all).all(axis=1).sum())))

    index = {fid: i for i, fid in enumerate(ids_all)}
    from sklearn.ensemble import HistGradientBoostingClassifier
    unseen = unseen_domain_by_species_id()

    def block(ids):
        X = X_all[[index[f] for f in ids]]
        sp = np.array([parse_id(f)[0] for f in ids])
        dom = np.array([parse_id(f)[1] for f in ids])
        return X, sp - 1, dom

    Xtr, ytr, _ = block(train_ids)
    Xva, yva, dva = block(val_ids)
    Xte, yte, dte = block(test_ids)
    part_te = np.array([1 if unseen.get(y + 1) == d else 0 for y, d in zip(yte, dte)])
    counts = np.bincount(ytr, minlength=9).astype(float)
    sw = (len(ytr) / (9 * np.maximum(counts, 1)))[ytr]
    for seed in range(args.seeds):
        t0 = time.time()
        clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=300, random_state=seed,
                                             early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)
        clf.fit(Xtr, ytr, sample_weight=sw)
        for split, X, y, d, part, ids in [('validation', Xva, yva, dva, np.full(len(yva), -1), val_ids),
                                          ('test', Xte, yte, dte, part_te, test_ids)]:
            probs = np.zeros((len(y), 9), dtype=np.float32)
            probs[:, clf.classes_] = clf.predict_proba(X)
            np.savez_compressed(os.path.join(RES, 'phase1', 'preds', 'm2_seed%d_%s.npz' % (seed, split)),
                                file_id=np.array(ids), probs=probs, y=y, domain=d, partition=part, feat=X)
            if split == 'test':
                for name, m in [('seen', part == 0), ('unseen', part == 1)]:
                    rec = [(probs[m & (y == c)].argmax(1) == c).mean() for c in range(9) if (m & (y == c)).any()]
                    log('seed %d test BA_%s = %.4f (n=%d)' % (seed, name, float(np.mean(rec)), int(m.sum())))
        log('seed %d trained in %.0f s (iterations %d)' % (seed, time.time() - t0, clf.n_iter_))
    log('ALL DONE')


if __name__ == '__main__':
    main()
