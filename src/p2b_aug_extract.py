"""Phase 2b (audio-level): augmented AST-padded-1024 embeddings (pooler_output, as Phase 1's M3) for D5 training clips.

Each view: random spectral tilt + low/high cut, then a 0.62 s background snippet from microphone recordings added at SNR 0-25 dB.
Background bank = quietest 0.62 s window of each microphone Training+Validation clip (>= 1 s). Per pseudo-unseen fold the bank
EXCLUDES the held-out group's recordings (leak-free). Variants: 'all' (1000 D5 clips/species x 2 views) and 9 folds (500 x 1 view).
Output: data/cdmsc/features/p2b_aug_<variant>.npz (emb, y, src_id)
"""
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.signal import resample_poly

from common import FEAT, RAW, ensure_dirs, log_to, parse_id, read_ids
from p2a_extract_ast import SNIP, SOS, load8k
from scipy.signal import sosfilt

MODEL_ID = 'MIT/ast-finetuned-audioset-10-10-0.4593'
log = log_to('p2b_aug_extract.log')
# fold name -> (species index 0-based, domains or None = all microphone domains)
FOLDS = {'Ae.aeg-mic': (0, None), 'Ae.alb-mic': (1, None), 'Cx.pip-mic': (6, None),
         'An.min-D1': (7, (1,)), 'An.min-D3': (7, (3,)), 'An.min-D4': (7, (4,)),
         'An.ste-D1': (8, (1,)), 'An.ste-D2': (8, (2,)), 'An.ste-D3': (8, (3,))}


def build_bank():
    ids = [i for i in read_ids('Training') + read_ids('Validation') if parse_id(i)[1] != 5]
    bank = []
    for fid in ids:
        x = load8k(os.path.join(RAW, fid + '.wav'))
        if x.size < 8000:
            continue
        e = sosfilt(SOS, x) ** 2
        c = np.concatenate([[0.0], np.cumsum(e, dtype=np.float64)])
        st = np.arange(0, x.size - SNIP + 1, SNIP // 2)
        j = st[np.argmin(c[st + SNIP] - c[st])]
        seg = x[j:j + SNIP].astype(np.float32)
        r = float(np.sqrt((seg ** 2).mean()))
        if r > 1e-6:
            bank.append((parse_id(fid)[0] - 1, parse_id(fid)[1], seg / r))
    log('noise bank: %d snippets from %d microphone clips' % (len(bank), len(ids)))
    return bank


def augment(x, noise, rnd):
    n = x.size
    f = np.fft.rfftfreq(n, 1 / 8000.0)[1:]
    tilt = rnd.uniform(-3, 3)                                   # dB per octave around 1 kHz
    g = 10 ** (tilt * np.log2(f / 1000.0) / 20)
    lc, hc = rnd.uniform(50, 200), rnd.uniform(2500, 3900)
    g = g / np.sqrt(1 + (lc / f) ** 4) / np.sqrt(1 + (f / hc) ** 8)
    X = np.fft.rfft(x)
    X[1:] *= g
    y = np.fft.irfft(X, n).astype(np.float32)
    r = float(np.sqrt((y ** 2).mean())) or 1.0
    snr = rnd.uniform(0, 25)
    noise = np.resize(noise, n)
    y = y + noise * (r / 10 ** (snr / 20))
    p = float(np.abs(y).max())
    return y / p if p > 0 else y


def main():
    import torch
    from transformers import ASTFeatureExtractor, ASTModel
    ensure_dirs(FEAT)
    z = np.load(os.path.join(FEAT, 'm3_emb_train.npz'), allow_pickle=True)
    d5 = [i for i in z['ids'] if parse_id(i)[1] == 5]
    rnd0 = random.Random(0)
    per = {}
    for i in d5:
        per.setdefault(parse_id(i)[0] - 1, []).append(i)
    pick = {c: rnd0.sample(v, 1000) for c, v in per.items()}
    bank = build_bank()
    model = ASTModel.from_pretrained(MODEL_ID).cuda().half().eval()
    fe = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    pool = ThreadPoolExecutor(4)
    variants = [('all', None, 1000, 2)] + [(k, k, 500, 1) for k in FOLDS]
    for name, fold, nper, nviews in variants:
        path = os.path.join(FEAT, 'p2b_aug_%s.npz' % name)
        if os.path.exists(path):
            continue
        if fold is None:
            nb = bank
        else:
            sp, doms = FOLDS[fold]
            nb = [b for b in bank if not (b[0] == sp and (doms is None or b[1] in doms))]
        log('%s: bank %d, %d clips/species x %d views' % (name, len(nb), nper, nviews))
        jobs = [(i, v) for c in sorted(pick) for i in pick[c][:nper] for v in range(nviews)]
        rnd = random.Random(sum(map(ord, name)))
        seeds = [rnd.random() for _ in jobs]

        def make(job_seed):
            (fid, v), s = job_seed
            r = random.Random(s)
            x = load8k(os.path.join(RAW, fid + '.wav'))
            x = augment(x - x.mean(), nb[r.randrange(len(nb))][2], r)
            w = resample_poly(x, 2, 1).astype(np.float32)
            p = float(np.abs(w).max())
            return fe(w / p if p > 0 else w, sampling_rate=16000, return_tensors='np')['input_values'][0]
        out, t0 = [], time.time()
        for s in range(0, len(jobs), 32):
            feats = list(pool.map(make, list(zip(jobs[s:s + 32], seeds[s:s + 32]))))
            with torch.no_grad():
                o = model(input_values=torch.from_numpy(np.stack(feats)).cuda().half()).pooler_output
            out.append(o.float().cpu().numpy())
        emb = np.concatenate(out)
        np.savez(path, emb=emb, y=np.array([parse_id(j[0])[0] - 1 for j in jobs]), src=np.array([j[0] for j in jobs]))
        log('saved %s %s in %.0f s' % (name, emb.shape, time.time() - t0))
    log('ALL DONE')


if __name__ == '__main__':
    sys.exit(main())
