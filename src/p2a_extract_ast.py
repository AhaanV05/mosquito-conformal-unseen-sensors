"""Phase 2a: AST embeddings from canonical 0.62 s snippets tiled to a 128-frame (1.28 s) window.

Per clip: up to K loudest non-overlapping 0.62 s snippets (100-2000 Hz energy), each tiled with 20 ms cross-fades,
peak-normalised, resampled 8k->16k, kaldi fbank (128 mel), AST with the time axis of the position embeddings cropped
(centre 12 of 101 columns, as the AST authors do). Output per clip: mean over snippets of masked-free patch-token mean
at layers 3, 6, 9, 12 -> (4, 768) fp16. Resumable in chunks.

  python p2a_extract_ast.py --set check       # 2000 validation clips, prints throughput
  python p2a_extract_ast.py --set all         # train (D5 capped) + validation + test
"""
import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

from common import FEAT, RAW, ensure_dirs, log_to, parse_id, read_ids

MODEL_ID = 'MIT/ast-finetuned-audioset-10-10-0.4593'
SNIP = 4960            # 0.62 s at 8 kHz
TARGET8 = 10240        # 1.28 s at 8 kHz
OV = 160               # 20 ms cross-fade
TAPS = (3, 6, 9, 12)
FRAMES = 128
MEAN, STD = -4.2677393, 4.5689974
log = log_to('p2a_extract_ast.log')
SOS = butter(4, [100, 2000], btype='band', fs=8000, output='sos')


def load8k(path):
    x, sr = sf.read(path, dtype='float32', always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 8000:
        g = gcd(int(sr), 8000)
        x = resample_poly(x, 8000 // g, int(sr) // g).astype(np.float32)
    return x


def snippets(x, k=4):
    if x.size <= SNIP:
        return [x]
    e = sosfilt(SOS, x) ** 2
    c = np.concatenate([[0.0], np.cumsum(e, dtype=np.float64)])
    starts = np.arange(0, x.size - SNIP + 1, SNIP // 2)
    en = c[starts + SNIP] - c[starts]
    out, taken = [], []
    for i in np.argsort(-en):
        if all(abs(int(starts[i]) - t) >= SNIP for t in taken):
            taken.append(int(starts[i]))
            out.append(x[starts[i]:starts[i] + SNIP])
            if len(out) == k:
                break
    return out


def tile(seg, target=TARGET8):
    seg = seg - seg.mean()
    if seg.size < 2 * OV + 8:
        seg = np.pad(seg, (0, 2 * OV + 8 - seg.size))
    out = seg.copy()
    fi = np.linspace(0, 1, OV, dtype=np.float32)
    while out.size < target:
        out = np.concatenate([out[:-OV], out[-OV:] * (1 - fi) + seg[:OV] * fi, seg[OV:]])
    out = out[:target]
    p = float(np.abs(out).max())
    return out / p if p > 0 else out


_FE = None


def fbank_batch(wavs16):
    import torch
    global _FE
    if _FE is None:
        from transformers import ASTFeatureExtractor
        _FE = ASTFeatureExtractor.from_pretrained(MODEL_ID, max_length=FRAMES)
    v = _FE(list(wavs16), sampling_rate=16000, return_tensors='np')['input_values']
    return torch.from_numpy(v.astype(np.float32))


def prep(fid):
    x = load8k(os.path.join(RAW, fid + '.wav'))
    segs = snippets(x if x.size else np.zeros(SNIP, np.float32))
    return [resample_poly(tile(s), 2, 1).astype(np.float32) for s in segs]


def build_model(device):
    import torch
    from transformers import ASTConfig, ASTModel
    full = ASTModel.from_pretrained(MODEL_ID)
    cfg = ASTConfig.from_pretrained(MODEL_ID)
    cfg.max_length = FRAMES
    m = ASTModel(cfg)
    sd = full.state_dict()
    key = 'embeddings.position_embeddings'
    pe = sd[key]                                   # (1, 2 + 12*101, 768)
    cls, grid = pe[:, :2], pe[:, 2:].reshape(1, 12, 101, -1)
    t = (FRAMES - 16) // 10 + 1
    a = 50 - t // 2
    sd[key] = torch.cat([cls, grid[:, :, a:a + t].reshape(1, 12 * t, -1)], 1)
    m.load_state_dict(sd)
    return m.to(device).half().eval()


def embed(ids, model, device, pool, batch=64):
    import torch
    out = np.zeros((len(ids), len(TAPS), 768), np.float16)
    t0 = time.time()
    for s in range(0, len(ids), batch):
        chunk = ids[s:s + batch]
        segs = list(pool.map(prep, chunk))
        flat = [w for sg in segs for w in sg]
        owner = np.concatenate([[i] * len(sg) for i, sg in enumerate(segs)])
        res = []
        with torch.no_grad():
            for b in range(0, len(flat), 128):
                x = fbank_batch(flat[b:b + 128]).to(device).half()
                hs = model(input_values=x, output_hidden_states=True).hidden_states
                res.append(torch.stack([hs[t][:, 2:].float().mean(1) for t in TAPS], 1).cpu().numpy())
        res = np.concatenate(res)
        for i in range(len(chunk)):
            out[s + i] = res[owner == i].mean(0)
        if (s // batch) % 20 == 0:
            log('%d/%d clips, %.1f clips/s' % (s + len(chunk), len(ids), (s + len(chunk)) / (time.time() - t0)))
    return out


def clip_sets(cap):
    rnd = random.Random(0)
    groups = {}
    for fid in read_ids('Training'):
        groups.setdefault(parse_id(fid), []).append(fid)
    train = []
    for key in sorted(groups):
        g = groups[key]
        train += rnd.sample(g, cap) if (key[1] == 5 and len(g) > cap) else g
    return {'train': train, 'validation': read_ids('Validation'), 'test': read_ids('Test')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=['check', 'all'], default='check')
    ap.add_argument('--cap', type=int, default=4000)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    import torch
    device = torch.device('cuda')
    ensure_dirs(FEAT)
    model = build_model(device)
    pool = ThreadPoolExecutor(args.threads)
    if args.set == 'check':
        ids = random.Random(1).sample(read_ids('Validation'), 2000)
        t0 = time.time()
        e = embed(ids, model, device, pool)
        dt = time.time() - t0
        log('CHECK: 2000 clips in %.0f s = %.1f clips/s; finite=%s; mean abs %.3f' % (dt, 2000 / dt, bool(np.isfinite(e.astype(np.float32)).all()), float(np.abs(e).mean())))
        return
    CH = 5000
    for name, ids in clip_sets(args.cap).items():
        for c in range(0, len(ids), CH):
            path = os.path.join(FEAT, 'p2a_ast128_%s_%03d.npz' % (name, c // CH))
            if os.path.exists(path):
                continue
            sub = ids[c:c + CH]
            e = embed(sub, model, device, pool)
            np.savez(path, ids=np.array(sub), emb=e)
            log('saved %s' % path)
    log('ALL DONE')


if __name__ == '__main__':
    sys.exit(main())
