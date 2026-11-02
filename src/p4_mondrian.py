"""Phase 4: sensor-conditional (Mondrian) conformal sets, evaluated on the WHOLE dev test (seen + unseen).

A sensor detector (logistic regression on AST embeddings, fit on Validation to tell D5 from microphone clips) routes each clip.
D5 branch: threshold from Validation D5 clips (full model). Microphone branch: threshold from the pseudo-unseen groups (ensemble of
fold models; C1 pooled / C2b second-worst / C2 worst). All unseen official groups are microphone domains. Compared with one global
threshold (C0 Validation, C1 pooled). Reports coverage per official test group, overall, and the coverage-versus-size frontier.
Detector uses no test labels. Output: results/phase4/.
"""
import json
import os
import sys

import numpy as np
import torch

from common import RES, SP_SHORT, ensure_dirs, unseen_domain_by_species_id
from p2b_comp import m3_arrays
from p2b_aug_extract import FOLDS
from p2c_calibrate import qhat

OUT = os.path.join(RES, 'phase4')
ALPHAS = tuple(round(x, 3) for x in [0.02, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.40])


def detector(Xv, dv, Xt):
    dev = torch.device('cuda')
    Xv, Xt = torch.from_numpy(Xv[:, 0]).to(dev), torch.from_numpy(Xt[:, 0]).to(dev)
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-6
    Xv, Xt = (Xv - mu) / sd, (Xt - mu) / sd
    y = torch.from_numpy((dv != 5).astype(np.float32)).to(dev)
    w = torch.zeros(Xv.shape[1], device=dev, requires_grad=True); b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200)
    pw = torch.where(y > 0, (1 - y.mean()) / y.mean(), torch.tensor(1.0, device=dev))

    def closure():
        opt.zero_grad()
        l = (torch.nn.functional.binary_cross_entropy_with_logits(Xv @ w + b, y, reduction='none') * pw).mean() + 1e-3 * (w ** 2).sum()
        l.backward(); return l
    opt.step(closure)
    return torch.sigmoid(Xt @ w + b).detach().cpu().numpy()


def main():
    ensure_dirs(OUT)
    unseen = unseen_domain_by_species_id()
    _, (Xva, yva, dva), (Xte, yte, dte) = m3_arrays()
    part = np.array([1 if unseen.get(a + 1) == b else 0 for a, b in zip(yte, dte)])
    z = np.load(os.path.join(RES, 'phase2c', 'oof.npz'), allow_pickle=True)
    P_ens, P_full, P_val = z['P_ens'], z['P_full_te'], z['P_full_va']
    oof_s = {k: 1 - z['oof_' + k][np.arange(len(z['y_' + k])), z['y_' + k]] for k in FOLDS}
    pooled = np.concatenate(list(oof_s.values()))
    sv = 1 - P_val[np.arange(len(yva)), yva]
    sv5 = sv[dva == 5]
    pm = detector(Xva, dva, Xte)
    flag = pm > 0.5
    mic = dte != 5
    from sklearn.metrics import roc_auc_score
    print('detector: balanced acc %.3f, AUC %.4f' % (0.5 * ((flag & mic).sum() / mic.sum() + (~flag & ~mic).sum() / (~mic).sum()), roc_auc_score(mic, pm)))
    names = {c: SP_SHORT[c + 1] for c in range(9)}
    rows, frontier = [], {}

    def evaluate(S, tag, a, use):
        cov = S[np.arange(len(yte)), yte]
        macro = float(np.mean([cov[yte == c].mean() for c in range(9)]))
        grp = {}
        for c in range(9):
            for lab, m in (('seen', (part == 0)), ('unseen', (part == 1))):
                mm = m & (yte == c)
                if mm.any():
                    grp['%s-%s' % (names[c], lab)] = float(cov[mm].mean())
        r = dict(rule=tag, alpha=a, overall_cov=float(cov.mean()), macro_species_cov=macro, size=float(S.sum(1).mean()),
                 D5_cov=float(cov[~mic].mean()), D5_size=float(S[~mic].sum(1).mean()), mic_cov=float(cov[mic].mean()), mic_size=float(S[mic].sum(1).mean()),
                 unseen_cov=float(cov[part == 1].mean()), unseen_size=float(S[part == 1].sum(1).mean()),
                 seen_cov=float(cov[part == 0].mean()), vacuous=float((S.sum(1) == 9).mean()),
                 unseen_groups_ge80=int(sum(v >= 0.8 for k, v in grp.items() if k.endswith('unseen'))), groups=grp)
        rows.append(r); frontier.setdefault(tag, []).append((r['size'], r['overall_cov']))
    for a in ALPHAS:
        q5 = qhat(sv5, a)
        q_mic = {'C1': qhat(pooled, a), 'C2b': sorted(qhat(s, a) for s in oof_s.values())[-2], 'C2': max(qhat(s, a) for s in oof_s.values())}
        evaluate((1 - P_full) <= qhat(sv, a), 'global C0 (validation)', a, None)
        evaluate((1 - P_ens) <= q_mic['C1'], 'global C1 (pooled pseudo-unseen)', a, None)
        for k, qm in q_mic.items():
            for lab, route in (('detected', flag), ('oracle', mic)):
                S = np.where(route[:, None], (1 - P_ens) <= qm, (1 - P_full) <= q5)
                evaluate(S, 'Mondrian by sensor (%s) mic=%s' % (lab, k), a, None)
    json.dump(rows, open(os.path.join(OUT, 'mondrian_results.json'), 'w'), indent=1)
    import csv
    with open(os.path.join(OUT, 'mondrian_results.csv'), 'w', newline='') as fh:
        keys = [k for k in rows[0] if k != 'groups']
        wr = csv.DictWriter(fh, fieldnames=keys + sorted({g for r in rows for g in r['groups']}))
        wr.writeheader()
        for r in rows:
            wr.writerow({**{k: r[k] for k in keys}, **r['groups']})
    L = ['# Phase 4: sensor-conditional (Mondrian) conformal sets on the whole dev test', '',
         '| rule | alpha | overall cov | species-macro cov | mean size | D5 cov / size | mic cov / size | unseen cov / size | seen cov | unseen groups >= 0.80 |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        if r['alpha'] in (0.10, 0.20):
            L.append('| %s | %.2f | %.3f | %.3f | %.2f | %.3f / %.2f | %.3f / %.2f | %.3f / %.2f | %.3f | %d/9 |' % (
                r['rule'], r['alpha'], r['overall_cov'], r['macro_species_cov'], r['size'], r['D5_cov'], r['D5_size'], r['mic_cov'],
                r['mic_size'], r['unseen_cov'], r['unseen_size'], r['seen_cov'], r['unseen_groups_ge80']))
    open(os.path.join(OUT, 'mondrian_report.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    sys.exit(main())
