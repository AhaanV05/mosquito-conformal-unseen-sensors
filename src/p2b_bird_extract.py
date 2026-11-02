"""Phase 2 (Bird-MAE): padded-input embeddings, 4 layer taps (3, 6, 9, 12), masked mean over valid time patches.

Mirrors the AST padded recipe that won in 2b: the raw clip (first 5 s) at 32 kHz is zero-padded to 5 s, no tiling.
Clean sets use the same id order as m3_emb_{train,validation,test}.npz so labels/domains line up; augmented views reuse the
noise-swap/EQ pipeline and the per-fold noise banks of p2b_aug_extract.
Outputs: data/cdmsc/features/bm_{train,validation,test}.npz and bm_aug_<variant>.npz  (emb (N,4,768) fp16)
  python p2b_bird_extract.py --check    # 300 clips timing
"""
import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.signal import resample_poly

from common import FEAT, RAW, ensure_dirs, log_to, parse_id
from p2a_extract_ast import load8k
from p2b_aug_extract import FOLDS, augment, build_bank

MODEL_ID = 'DBD-research-group/Bird-MAE-Base'
TAPS = (3, 6, 9, 12)
WIN = 160000
log = log_to('p2b_bird_extract.log')


def wav32(x8):
    w = resample_poly(x8, 4, 1).astype(np.float32)
    p = float(np.abs(w).max())
    return (w / p if p > 0 else w)[:WIN]


def embed(waves, fe, model):
    import torch
    n = [min(len(w), WIN) for w in waves]
    batch = np.zeros((len(waves), WIN), np.float32)
    for i, w in enumerate(waves):
        batch[i, :n[i]] = w[:WIN]
    x = fe(batch, return_tensors='pt').cuda().half()
    with torch.no_grad():
        hs = model(input_values=x, output_hidden_states=True, return_dict=True).hidden_states
    cols = torch.tensor([min(32, max(1, -(-(((m - 400) // 320) + 1) // 16))) for m in n], device='cuda')
    mask = (torch.arange(32, device='cuda')[None] < cols[:, None]).float()          # (B, 32 time patches)
    out = []
    for t in TAPS:
        p = hs[t][:, 1:].float().reshape(len(waves), 32, 8, 768).mean(2)            # time-major tokens
        out.append((p * mask[..., None]).sum(1) / mask.sum(1, keepdim=True))
    return torch.stack(out, 1).cpu().numpy().astype(np.float16)


def run(jobs_wave, fe, model, name, bs=32):
    out, t0 = [], time.time()
    for s in range(0, len(jobs_wave), bs):
        out.append(embed([w for w in jobs_wave[s:s + bs]], fe, model))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    from transformers import AutoFeatureExtractor, AutoModel
    ensure_dirs(FEAT)
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).cuda().half().eval()
    pool = ThreadPoolExecutor(4)

    def stream(ids_or_jobs, loader, name):
        res, t0, bs = [], time.time(), 32
        for s in range(0, len(ids_or_jobs), bs):
            waves = list(pool.map(loader, ids_or_jobs[s:s + bs]))
            res.append(embed(waves, fe, model))
            if (s // bs) % 50 == 0:
                log('%s %d/%d %.1f clips/s' % (name, s + bs, len(ids_or_jobs), (s + bs) / (time.time() - t0)))
        return np.concatenate(res)

    clean = lambda fid: wav32(load8k(os.path.join(RAW, fid + '.wav')))
    if args.check:
        ids = list(np.load(os.path.join(FEAT, 'm3_emb_validation.npz'), allow_pickle=True)['ids'][:300])
        t0 = time.time(); e = stream(ids, clean, 'check')
        log('CHECK 300 clips %.1f s, finite %s, mean abs %.3f' % (time.time() - t0, bool(np.isfinite(e.astype(np.float32)).all()), float(np.abs(e).mean())))
        return
    for split in ('train', 'validation', 'test'):
        path = os.path.join(FEAT, 'bm_%s.npz' % split)
        if os.path.exists(path):
            continue
        ids = list(np.load(os.path.join(FEAT, 'm3_emb_%s.npz' % split), allow_pickle=True)['ids'])
        np.savez(path, ids=np.array(ids), emb=stream(ids, clean, split))
        log('saved %s' % path)
    # augmented views: same picks as p2b_aug_extract
    zt = np.load(os.path.join(FEAT, 'm3_emb_train.npz'), allow_pickle=True)
    d5 = [i for i in zt['ids'] if parse_id(i)[1] == 5]
    rnd0, per = random.Random(0), {}
    for i in d5:
        per.setdefault(parse_id(i)[0] - 1, []).append(i)
    pick = {c: rnd0.sample(v, 1000) for c, v in per.items()}
    bank = build_bank()
    for name, fold, nper, nviews in [('all', None, 1000, 2)] + [(k, k, 500, 1) for k in FOLDS]:
        path = os.path.join(FEAT, 'bm_aug_%s.npz' % name)
        if os.path.exists(path):
            continue
        if fold is None:
            nb = bank
        else:
            sp, doms = FOLDS[fold]
            nb = [b for b in bank if not (b[0] == sp and (doms is None or b[1] in doms))]
        jobs = [(i, v) for c in sorted(pick) for i in pick[c][:nper] for v in range(nviews)]
        rnd = random.Random(sum(map(ord, name)) + 7)
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
