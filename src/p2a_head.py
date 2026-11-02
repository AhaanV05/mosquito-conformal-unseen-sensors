"""Phase 2a: train heads on cached AST-128 embeddings; compare configs on the official test and on 9 pseudo-unseen folds.

Configs: linear_last (linear probe, tap 12), mlp_mix (4-tap mix + adapter + fusion + cosine head), mlp_mix_hier (+ genus/complex loss).
Fold groups (Train+Val microphone clips held out with all their data removed from training):
LSMO: all mic clips of Ae.aeg, Ae.alb, Cx.pip. LODO: An.min D1/D3/D4, An.ste D1/D2/D3.
Model selection uses only the folds; the official unseen test is reported for reference.
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import FEAT, RES, SP_SHORT, ensure_dirs, log_to, parse_id, unseen_domain_by_species_id

log = log_to('p2a_head.log')
dev = torch.device('cuda')
GEN = np.array([0, 0, 1, 2, 2, 2, 1, 2, 2])           # Aedes, Culex, Anopheles
CPX = np.array([0, 1, 2, 3, 3, 4, 5, 6, 7])           # An. gam + An. ara merged
G_GEN = torch.zeros(9, 3); G_GEN[range(9), GEN] = 1
G_CPX = torch.zeros(9, 8); G_CPX[range(9), CPX] = 1


def load(name):
    ids, emb = [], []
    for f in sorted(glob.glob(os.path.join(FEAT, 'p2a_ast128_%s_*.npz' % name))):
        z = np.load(f)
        ids += list(z['ids']); emb.append(z['emb'])
    ids = np.array(ids)
    y = np.array([parse_id(i)[0] - 1 for i in ids]); d = np.array([parse_id(i)[1] for i in ids])
    return ids, np.concatenate(emb), y, d


class Net(nn.Module):
    def __init__(self, kind, d=768, taps=4):
        super().__init__()
        self.kind = kind
        if kind == 'linear_last':
            self.ln = nn.LayerNorm(d); self.fc = nn.Linear(d, 9)
        else:
            self.ln = nn.LayerNorm(d); self.w = nn.Parameter(torch.zeros(taps))
            self.proj = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2))
            self.fuse = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 256))
            self.cls = nn.Parameter(torch.randn(9, 256) * 0.05); self.s = nn.Parameter(torch.tensor(16.0))

    def forward(self, x):                      # x (B, 4, 768)
        if self.kind == 'linear_last':
            return self.fc(self.ln(x[:, -1]))
        h = (self.ln(x) * torch.softmax(self.w, 0)[None, :, None]).sum(1)
        z = F.normalize(self.fuse(self.proj(h)), dim=-1)
        return self.s * z @ F.normalize(self.cls, dim=-1).T


def train(X, y, d, kind, hier, seed, epochs=30):
    torch.manual_seed(seed); np.random.seed(seed)
    X = torch.from_numpy(X.astype(np.float32)).to(dev); yt = torch.from_numpy(y).to(dev)
    pair = y * 10 + d
    u, inv, cnt = np.unique(pair, return_inverse=True, return_counts=True)
    sw = 1.0 / np.sqrt(cnt[inv])
    cw = np.zeros(9)
    for c in range(9):
        cw[c] = sw[y == c].sum() if (y == c).any() else 1.0
    cw = torch.tensor((1 / cw) / (1 / cw).mean(), dtype=torch.float32, device=dev)
    swt = torch.from_numpy(sw).float().to(dev)
    net = Net(kind).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    steps = max(len(y) // 256, 1); total = epochs * steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=total, pct_start=0.07)
    gg, gc = G_GEN.to(dev), G_CPX.to(dev)
    for _ in range(total):
        idx = torch.multinomial(swt, 256, replacement=True)
        xb = X[idx] + 0.1 * X.std() * torch.randn(256, 4, 768, device=dev)
        net.train(); lg = net(xb); lp = F.log_softmax(lg, -1)
        loss = F.nll_loss(lp, yt[idx], weight=cw)
        if hier:
            p = lp.exp()
            loss = loss + 0.3 * -(torch.log((p @ gg).clamp_min(1e-8)) * gg[yt[idx]]).sum(1).mean() \
                        + 0.3 * -(torch.log((p @ gc).clamp_min(1e-8)) * gc[yt[idx]]).sum(1).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval()
    return net


def predict(net, X):
    with torch.no_grad():
        return torch.softmax(net(torch.from_numpy(X.astype(np.float32)).to(dev)), -1).cpu().numpy()


def ba(p, y, mask):
    r = [(p[mask & (y == c)].argmax(1) == c).mean() for c in range(9) if (mask & (y == c)).any()]
    return float(np.mean(r))


def main():
    ensure_dirs(os.path.join(RES, 'phase2a'))
    _, Xtr, ytr, dtr = load('train'); _, Xva, yva, dva = load('validation'); _, Xte, yte, dte = load('test')
    log('train %s val %s test %s' % (Xtr.shape, Xva.shape, Xte.shape))
    unseen = unseen_domain_by_species_id()
    part = np.array([1 if unseen.get(a + 1) == b else 0 for a, b in zip(yte, dte)])
    mic_va = dva != 5
    # fold pool = Train + Val microphone clips
    Xp = np.concatenate([Xtr, Xva[mic_va]]); yp = np.concatenate([ytr, yva[mic_va]]); dp = np.concatenate([dtr, dva[mic_va]])
    folds = [('Ae.aeg-mic', lambda y, d: (y == 0) & (d != 5)), ('Ae.alb-mic', lambda y, d: (y == 1) & (d != 5)),
             ('Cx.pip-mic', lambda y, d: (y == 6) & (d != 5))]
    folds += [('An.min-D%d' % k, (lambda k: lambda y, d: (y == 7) & (d == k))(k)) for k in (1, 3, 4)]
    folds += [('An.ste-D%d' % k, (lambda k: lambda y, d: (y == 8) & (d == k))(k)) for k in (1, 2, 3)]
    configs = [('linear_last', 'linear_last', False), ('mlp_mix', 'mlp_mix', False), ('mlp_mix_hier', 'mlp_mix', True)]
    seeds = [0, 1]
    out = {}
    for name, kind, hier in configs:
        P = np.mean([predict(train(Xtr, ytr, dtr, kind, hier, s), Xte) for s in seeds], 0)
        res = {'test_BA_seen': ba(P, yte, part == 0), 'test_BA_unseen': ba(P, yte, part == 1),
               'genus_acc_unseen': float((np.array([GEN[i] for i in P[part == 1].argmax(1)]) == GEN[yte[part == 1]]).mean())}
        fr = {}
        for fname, sel in folds:
            held = sel(yp, dp)
            nets = [train(Xp[~held], yp[~held], dp[~held], kind, hier, s) for s in seeds]
            pf = np.mean([predict(n, Xp[held]) for n in nets], 0)
            fr[fname] = (float((pf.argmax(1) == yp[held]).mean()), int(held.sum()),
                         float((GEN[pf.argmax(1)] == GEN[yp[held]]).mean()))
        res['folds'] = fr
        res['fold_macro_acc'] = float(np.mean([v[0] for v in fr.values()]))
        res['fold_macro_genus_acc'] = float(np.mean([v[2] for v in fr.values()]))
        out[name] = res
        log('%s: test BA seen %.3f unseen %.3f | fold macro acc %.3f genus %.3f' % (
            name, res['test_BA_seen'], res['test_BA_unseen'], res['fold_macro_acc'], res['fold_macro_genus_acc']))
        np.savez_compressed(os.path.join(RES, 'phase2a', 'probs_%s.npz' % name), probs=P, y=yte, domain=dte, partition=part)
    json.dump(out, open(os.path.join(RES, 'phase2a', 'p2a_head_results.json'), 'w'), indent=1)


if __name__ == '__main__':
    sys.exit(main())
