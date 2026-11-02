"""Phase 2b: sensor compensation on cached AST embeddings, scored on the same 9 pseudo-unseen folds as 2a.

Configs: m3_linear (Phase 1 padded-1024 embeddings, linear head: fair baseline), lin_nap2/lin_nap3 (linear head + nuisance-direction
projection with k=2/3), mlp_nap3, mlp_align (per-species mean+covariance alignment of D5 vs microphone embeddings).
NAP: per tap, top-k directions of (species mean in a microphone domain - same species D5 mean) over bridge species, projected out.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import p2a_head as H
from common import FEAT, RES, log_to, parse_id, unseen_domain_by_species_id

log = H.log = log_to('p2b_comp.log')
dev = H.dev


def nap_dirs(X, y, d, k):
    """X (N,T,768) -> list of (768,k) projections, one per tap."""
    out = []
    X = X.astype(np.float32)
    for t in range(X.shape[1]):
        V = []
        for c in range(9):
            m5 = X[(y == c) & (d == 5), t].mean(0) if ((y == c) & (d == 5)).sum() >= 10 else None
            if m5 is None:
                continue
            for dm in (1, 2, 3, 4):
                s = (y == c) & (d == dm)
                if s.sum() >= 10:
                    V.append(X[s, t].mean(0) - m5)
        if not V:
            out.append(np.zeros((X.shape[2], 0), np.float32)); continue
        vt = torch.linalg.svd(torch.from_numpy(np.array(V, dtype=np.float32)).to(dev), full_matrices=False)[2].cpu().numpy()
        out.append(np.ascontiguousarray(vt[:k].T))
    return out


def apply_nap(X, U):
    X = torch.from_numpy(X.astype(np.float32)).to(dev)
    for t, u in enumerate(U):
        if u.shape[1]:
            ut = torch.from_numpy(u).to(dev)
            X[:, t] -= (X[:, t] @ ut) @ ut.T
    return X.cpu().numpy()


def train_b(X, y, d, kind, align, seed, epochs=30):
    torch.manual_seed(seed); np.random.seed(seed)
    Xt = torch.from_numpy(X.astype(np.float32)).to(dev); yt = torch.from_numpy(y).to(dev); dt = torch.from_numpy(d).to(dev)
    u, inv, cnt = np.unique(y * 10 + d, return_inverse=True, return_counts=True)
    sw = 1.0 / np.sqrt(cnt[inv])
    cw = np.array([sw[y == c].sum() if (y == c).any() else 1.0 for c in range(9)])
    cw = torch.tensor((1 / cw) / (1 / cw).mean(), dtype=torch.float32, device=dev)
    swt = torch.from_numpy(sw).float().to(dev)
    net = H.Net(kind, X.shape[2], X.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    B = 512 if align else 256
    total = epochs * max(len(y) // 256, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=total, pct_start=0.07)
    for _ in range(total):
        idx = torch.multinomial(swt, B, replacement=True)
        xb = Xt[idx] + 0.1 * Xt.std() * torch.randn(B, Xt.shape[1], Xt.shape[2], device=dev)
        net.train()
        if align:
            h = (net.ln(xb) * torch.softmax(net.w, 0)[None, :, None]).sum(1)
            z = F.normalize(net.fuse(net.proj(h)), dim=-1)
            lg = net.s * z @ F.normalize(net.cls, dim=-1).T
        else:
            lg = net(xb)
        loss = F.cross_entropy(lg, yt[idx], weight=cw)
        if align:
            pen, n = 0.0, 0
            yb, db = yt[idx], dt[idx]
            for c in range(9):
                a, b = z[(yb == c) & (db == 5)], z[(yb == c) & (db != 5)]
                if len(a) >= 8 and len(b) >= 4:
                    ca, cb = torch.cov(a.T), torch.cov(b.T)
                    pen = pen + ((a.mean(0) - b.mean(0)) ** 2).sum() + ((ca - cb) ** 2).sum()
                    n += 1
            if n:
                loss = loss + 0.1 * pen / n
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval()
    return net


def m3_arrays():
    def one(n):
        z = np.load(os.path.join(FEAT, 'm3_emb_%s.npz' % n), allow_pickle=True)
        ids = z['ids']
        return (z['emb'][:, None, :].astype(np.float32), np.array([parse_id(i)[0] - 1 for i in ids]),
                np.array([parse_id(i)[1] for i in ids]))
    return one('train'), one('validation'), one('test')


def main():
    unseen = unseen_domain_by_species_id()
    (_, Xtr, ytr, dtr), (_, Xva, yva, dva), (_, Xte, yte, dte) = (H.load('train'), H.load('validation'), H.load('test'))
    part = np.array([1 if unseen.get(a + 1) == b else 0 for a, b in zip(yte, dte)])
    mic = dva != 5
    pool = lambda a, b: np.concatenate([a, b[mic]])
    folds = [('Ae.aeg-mic', lambda y, d: (y == 0) & (d != 5)), ('Ae.alb-mic', lambda y, d: (y == 1) & (d != 5)),
             ('Cx.pip-mic', lambda y, d: (y == 6) & (d != 5))]
    folds += [('An.min-D%d' % k, (lambda k: lambda y, d: (y == 7) & (d == k))(k)) for k in (1, 3, 4)]
    folds += [('An.ste-D%d' % k, (lambda k: lambda y, d: (y == 8) & (d == k))(k)) for k in (1, 2, 3)]
    sets = {'ast128': (Xtr, ytr, dtr, pool(Xtr, Xva), pool(ytr, yva), pool(dtr, dva), Xte)}
    (m3tr, m3y, m3d), (m3va, m3yv, m3dv), (m3te, _, _) = m3_arrays()
    mm = m3dv != 5
    sets['m3pad'] = (m3tr, m3y, m3d, np.concatenate([m3tr, m3va[mm]]), np.concatenate([m3y, m3yv[mm]]),
                     np.concatenate([m3d, m3dv[mm]]), m3te)
    # (name, data, head kind, nap k, align)
    configs = [('m3_linear', 'm3pad', 'linear_last', 0, False), ('lin_nap0', 'ast128', 'linear_last', 0, False),
               ('lin_nap2', 'ast128', 'linear_last', 2, False), ('lin_nap3', 'ast128', 'linear_last', 3, False),
               ('mlp_nap3', 'ast128', 'mlp_mix', 3, False), ('mlp_align', 'ast128', 'mlp_mix', 0, True)]
    seeds = [0, 1]
    rp = os.path.join(RES, 'phase2a', 'p2b_results.json')
    out = json.load(open(rp)) if os.path.exists(rp) else {}
    for name, ds, kind, k, align in configs:
        if name in out:
            continue
        X, y, d, Xp, yp, dp, Xt = sets[ds]
        if ds == 'm3pad':                       # linear_last reads tap index 3: replicate the single vector into 4 taps
            rep = lambda a: np.repeat(a, 4, axis=1)
            X, Xp, Xt = rep(X), rep(Xp), rep(Xt)

        def fit(Xa, ya, da):
            U = nap_dirs(Xa, ya, da, k) if k else None
            Xn = apply_nap(Xa, U) if k else Xa
            return U, [train_b(Xn, ya, da, kind, align, s) for s in seeds]

        def pr(U, nets, Xq):
            Xq = apply_nap(Xq, U) if k else Xq
            return np.mean([H.predict(n, Xq) for n in nets], 0)
        U, nets = fit(X, y, d)
        P = pr(U, nets, Xt)
        res = {'test_BA_seen': H.ba(P, yte, part == 0), 'test_BA_unseen': H.ba(P, yte, part == 1)}
        fr = {}
        for fname, sel in folds:
            held = sel(yp, dp)
            Uf, nf = fit(Xp[~held], yp[~held], dp[~held])
            pf = pr(Uf, nf, Xp[held])
            fr[fname] = (float((pf.argmax(1) == yp[held]).mean()), int(held.sum()), float((H.GEN[pf.argmax(1)] == H.GEN[yp[held]]).mean()))
        res['folds'] = fr
        res['fold_macro_acc'] = float(np.mean([v[0] for v in fr.values()]))
        res['fold_bridge_acc'] = float(np.mean([fr[f][0] for f in ('Ae.aeg-mic', 'Ae.alb-mic', 'Cx.pip-mic')]))
        out[name] = res
        log('%s: test BA seen %.3f unseen %.3f | fold macro %.3f bridge-only %.3f' % (
            name, res['test_BA_seen'], res['test_BA_unseen'], res['fold_macro_acc'], res['fold_bridge_acc']))
        json.dump(out, open(os.path.join(RES, 'phase2a', 'p2b_results.json'), 'w'), indent=1)


if __name__ == '__main__':
    sys.exit(main())
