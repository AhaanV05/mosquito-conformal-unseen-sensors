"""Perch 2.0 (community PyTorch port of the official weights) embeddings, padded-input recipe identical to Bird-MAE.

Clip -> 32 kHz -> zero-padded into one 5 s window (first 5 s) -> 1536-d embedding. Clean sets follow the id order of
m3_emb_{train,validation,test}.npz; augmented views reuse the noise-swap/EQ pipeline and per-fold noise banks of p2b_aug_extract.
Outputs: data/cdmsc/features/pc_{train,validation,test}.npz and pc_aug_<variant>.npz   (emb (N,1,1536) fp16)
  python p2g_perch_extract.py --check
"""
import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import FEAT, RAW, ensure_dirs, log_to, parse_id
from p2a_extract_ast import load8k
from p2b_aug_extract import FOLDS, augment, build_bank
from p2b_bird_extract import WIN, wav32

log = log_to('p2g_perch_extract.log')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    import torch
    from perchv2_pytorch import PerchV2
    ensure_dirs(FEAT)
    model = PerchV2.from_pretrained(embeddings_only=True).cuda().eval()
    pool = ThreadPoolExecutor(4)

    def embed(waves):
        b = np.zeros((len(waves), WIN), np.float32)
        for i, w in enumerate(waves):
            b[i, :min(len(w), WIN)] = w[:WIN]
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            out = model(torch.from_numpy(b).cuda())['embedding']
        return out.float().cpu().numpy().astype(np.float16)[:, None, :]

    def stream(items, loader, name):
        res, t0, bs = [], time.time(), 32
        for s in range(0, len(items), bs):
            res.append(embed(list(pool.map(loader, items[s:s + bs]))))
            if (s // bs) % 50 == 0:
                log('%s %d/%d %.1f clips/s' % (name, s + bs, len(items), (s + bs) / (time.time() - t0)))
        return np.concatenate(res)

    clean = lambda fid: wav32(load8k(os.path.join(RAW, fid + '.wav')))
    if args.check:
        ids = list(np.load(os.path.join(FEAT, 'm3_emb_validation.npz'), allow_pickle=True)['ids'][:300])
        t0 = time.time(); e = stream(ids, clean, 'check')
        log('CHECK 300 clips %.1f s, shape %s, finite %s, mean abs %.3f' % (time.time() - t0, e.shape, bool(np.isfinite(e.astype(np.float32)).all()), float(np.abs(e).mean())))
        return
    for split in ('train', 'validation', 'test'):
        path = os.path.join(FEAT, 'pc_%s.npz' % split)
        if os.path.exists(path):
            continue
        ids = list(np.load(os.path.join(FEAT, 'm3_emb_%s.npz' % split), allow_pickle=True)['ids'])
        np.savez(path, ids=np.array(ids), emb=stream(ids, clean, split))
        log('saved %s' % path)
    zt = np.load(os.path.join(FEAT, 'm3_emb_train.npz'), allow_pickle=True)
    rnd0, per = random.Random(0), {}
    for i in [i for i in zt['ids'] if parse_id(i)[1] == 5]:
        per.setdefault(parse_id(i)[0] - 1, []).append(i)
    pick = {c: rnd0.sample(v, 1000) for c, v in per.items()}
    bank = build_bank()
    for name, fold, nper, nviews in [('all', None, 1000, 2)] + [(k, k, 500, 1) for k in FOLDS]:
        path = os.path.join(FEAT, 'pc_aug_%s.npz' % name)
        if os.path.exists(path):
            continue
        if fold is None:
            nb = bank
        else:
            sp, doms = FOLDS[fold]
            nb = [b for b in bank if not (b[0] == sp and (doms is None or b[1] in doms))]
        jobs = [(i, v) for c in sorted(pick) for i in pick[c][:nper] for v in range(nviews)]
        rnd = random.Random(sum(map(ord, name)) + 11)
        jobs = [(j, rnd.random()) for j in jobs]

        def make(js):
            (fid, v), s = js
            r = random.Random(s)
            x = load8k(os.path.join(RAW, fid + '.wav'))
            return wav32(augment(x - x.mean(), nb[r.randrange(len(nb))][2], r))
        emb = stream(jobs, make, 'aug_' + name)
        np.savez(path, emb=emb, y=np.array([parse_id(j[0][0])[0] - 1 for j in jobs]), src=np.array([j[0][0] for j in jobs]))
        log('saved %s %s' % (path, emb.shape))
    log('ALL DONE')


if __name__ == '__main__':
    sys.exit(main())
