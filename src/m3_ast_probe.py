"""M3: logistic-regression probe on frozen AST (AudioSet) embeddings. Best effort: skips cleanly if a step is unavailable.

Needs: the public Hugging Face weights MIT/ast-finetuned-audioset-10-10-0.4593 (no login), a CUDA GPU.
Output format matches M1/M2: results/phase1/preds/m3_seed0_{validation,test}.npz (feat = embedding).
"""
import argparse
import os
import random
import sys
import time
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from common import FEAT, RAW, RES, ensure_dirs, log_to, parse_id, read_ids, unseen_domain_by_species_id

MODEL_ID = 'MIT/ast-finetuned-audioset-10-10-0.4593'
SR = 16000
log = log_to('m3_ast_probe.log')


def load16k(path):
    x, sr = sf.read(path, dtype='float32', always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    peak = float(np.abs(x).max()) if x.size else 0.0
    return x / peak if peak > 0 else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cap', type=int, default=1500, help='max training clips per (species, domain) pair')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()
    if args.workers > 0:
        log('forcing DataLoader workers from %d to 0: free RAM is very low and spawn workers cannot pickle a local class' % args.workers)
        args.workers = 0
    try:
        import torch
        from transformers import ASTFeatureExtractor, ASTModel
        if not torch.cuda.is_available():
            log('M3 skipped: no CUDA device visible to torch (%s)' % torch.__version__)
            return
        device = torch.device('cuda')
        log('loading %s' % MODEL_ID)
        model = ASTModel.from_pretrained(MODEL_ID).to(device).half().eval()
    except Exception as e:  # noqa: BLE001
        log('M3 skipped: could not set up the model: %r' % (e,))
        return

    class ClipDS(torch.utils.data.Dataset):
        def __init__(self, ids):
            self.ids = ids
            self.fe = None

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            if self.fe is None:
                self.fe = ASTFeatureExtractor.from_pretrained(MODEL_ID)
            x = load16k(os.path.join(RAW, self.ids[i] + '.wav'))
            if x.size < 400:
                x = np.pad(x, (0, 400 - x.size))
            v = self.fe(x, sampling_rate=SR, return_tensors='np')['input_values'][0]
            return torch.from_numpy(v.astype(np.float32))

    def embed(name, ids):
        path = os.path.join(FEAT, 'm3_emb_%s.npz' % name)
        if os.path.exists(path):
            z = np.load(path, allow_pickle=True)
            if list(z['ids']) == list(ids):
                log('%s embeddings cached' % name)
                return z['emb']
        dl = torch.utils.data.DataLoader(ClipDS(ids), batch_size=args.batch, num_workers=args.workers, shuffle=False)
        out = []
        t0 = time.time()
        with torch.no_grad():
            for i, b in enumerate(dl):
                o = model(input_values=b.to(device).half())
                out.append(o.pooler_output.float().cpu().numpy())
                if i % 100 == 0:
                    log('%s batch %d/%d (%.0f s)' % (name, i, len(dl), time.time() - t0))
        emb = np.concatenate(out)
        np.savez_compressed(path, ids=np.array(ids), emb=emb)
        log('%s embeddings %s in %.0f s' % (name, emb.shape, time.time() - t0))
        return emb

    ensure_dirs(FEAT, os.path.join(RES, 'phase1', 'preds'))
    rnd = random.Random(0)
    groups = {}
    for fid in read_ids('Training'):
        groups.setdefault(parse_id(fid), []).append(fid)
    train_ids = []
    for key in sorted(groups):
        g = groups[key]
        train_ids += g if len(g) <= args.cap else rnd.sample(g, args.cap)
    val_ids, test_ids = read_ids('Validation'), read_ids('Test')
    log('train pool %d, validation %d, test %d' % (len(train_ids), len(val_ids), len(test_ids)))
    try:
        E_tr, E_va, E_te = embed('train', train_ids), embed('validation', val_ids), embed('test', test_ids)
    except Exception as e:  # noqa: BLE001
        log('M3 skipped: embedding extraction failed: %r' % (e,))
        return

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    unseen = unseen_domain_by_species_id()
    ytr = np.array([parse_id(f)[0] - 1 for f in train_ids])
    sc = StandardScaler().fit(E_tr)
    clf = LogisticRegression(C=0.5, max_iter=400, class_weight='balanced')
    clf.fit(sc.transform(E_tr), ytr)
    log('probe trained')
    for split, E, ids in [('validation', E_va, val_ids), ('test', E_te, test_ids)]:
        y = np.array([parse_id(f)[0] - 1 for f in ids])
        d = np.array([parse_id(f)[1] for f in ids])
        probs = np.zeros((len(y), 9), dtype=np.float32)
        probs[:, clf.classes_] = clf.predict_proba(sc.transform(E))
        part = np.array([1 if unseen.get(a + 1) == b else 0 for a, b in zip(y, d)]) if split == 'test' \
            else np.full(len(y), -1)
        np.savez_compressed(os.path.join(RES, 'phase1', 'preds', 'm3_seed0_%s.npz' % split), file_id=np.array(ids),
                            probs=probs, y=y, domain=d, partition=part, feat=E)
        if split == 'test':
            for name, m in [('seen', part == 0), ('unseen', part == 1)]:
                rec = [(probs[m & (y == c)].argmax(1) == c).mean() for c in range(9) if (m & (y == c)).any()]
                log('test BA_%s = %.4f (n=%d)' % (name, float(np.mean(rec)), int(m.sum())))
    log('ALL DONE')


if __name__ == '__main__':
    sys.exit(main())
