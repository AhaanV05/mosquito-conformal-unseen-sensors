"""Phase 3 + figures. (a) Few-shot site recalibration on the 2b/2c AST classifier: k labelled clips per unseen species from the
new domain set the conformal threshold; coverage measured on the remaining clips. (b) Coverage-vs-set-size curves for the
calibration rules, with the 'random subset' reference (a random set of size s covers s/9). Output: results/phase3/, results/figs/.
"""
import json
import os
import sys

import numpy as np

import p2a_head as H
from common import RES, SP_SHORT, ensure_dirs, unseen_domain_by_species_id
from p2b_comp import m3_arrays
from p2b_aug_extract import FOLDS
from p2c_calibrate import evaluate, gen_probs, qhat

OUT, FIG = os.path.join(RES, 'phase3'), os.path.join(RES, 'figs')
ALPHA_GRID = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70]


def main():
    ensure_dirs(OUT, FIG)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    unseen = unseen_domain_by_species_id()
    _, _, (Xte, yte, dte) = m3_arrays()
    part = np.array([1 if unseen.get(a + 1) == b else 0 for a, b in zip(yte, dte)])
    z = np.load(os.path.join(RES, 'phase2c', 'oof.npz'), allow_pickle=True)
    P_ens, P_full, P_val = z['P_ens'], z['P_full_te'], z['P_full_va']
    oof = {k: z['oof_' + k] for k in FOLDS}; yo = {k: z['y_' + k] for k in FOLDS}
    ug = {SP_SHORT[c + 1]: np.where((part == 1) & (yte == c))[0] for c in range(9) if ((part == 1) & (yte == c)).any()}
    ugm = {k: (np.isin(np.arange(len(yte)), v)) for k, v in ug.items()}

    # ---- (a) few-shot recalibration (alpha = 0.10), 30 random draws
    rng = np.random.default_rng(0)
    res = []
    for k in (0, 5, 10, 20, 50):
        covs, sizes, vac, gcov, gsize = [], [], [], [], []
        for rep in range(30 if k else 1):
            sc, gsc, held = [], [], {}
            for name, idx in ug.items():
                n_shot = min(k, len(idx) // 2)
                shots = rng.choice(idx, n_shot, replace=False) if n_shot else np.array([], int)
                held[name] = np.setdiff1d(idx, shots)
                sc += list(1 - P_ens[shots, yte[shots]])
                gsc += list(1 - gen_probs(P_ens[shots])[np.arange(len(shots)), H.GEN[yte[shots]]])
            if k:
                q, gq = qhat(np.array(sc), 0.10), qhat(np.array(gsc), 0.10)
            else:                                                     # k = 0: pseudo-unseen pooled rule (C1)
                q = qhat(np.concatenate([1 - oof[f][np.arange(len(yo[f])), yo[f]] for f in FOLDS]), 0.10)
                gq = qhat(np.concatenate([1 - gen_probs(oof[f])[np.arange(len(yo[f])), H.GEN[yo[f]]] for f in FOLDS]), 0.10)
            c, s, v, gc, gs = [], [], [], [], []
            for name, idx in held.items():
                S = (1 - P_ens[idx]) <= q
                c.append(S[np.arange(len(idx)), yte[idx]].mean()); s.append(S.sum(1).mean()); v.append((S.sum(1) == 9).mean())
                Gp = gen_probs(P_ens[idx]); GS = (1 - Gp) <= gq
                gc.append(GS[np.arange(len(idx)), H.GEN[yte[idx]]].mean()); gs.append(GS.sum(1).mean())
            covs.append(np.mean(c)); sizes.append(np.mean(s)); vac.append(np.mean(v)); gcov.append(np.mean(gc)); gsize.append(np.mean(gs))
        res.append(dict(k=k, species_cov=float(np.mean(covs)), species_cov_sd=float(np.std(covs)), species_size=float(np.mean(sizes)),
                        vacuous=float(np.mean(vac)), genus_cov=float(np.mean(gcov)), genus_size=float(np.mean(gsize))))
    json.dump(res, open(os.path.join(OUT, 'fewshot_results.json'), 'w'), indent=1)
    L = ['# Phase 3: few-shot site recalibration (alpha = 0.10, AST mlp_aug ensemble, 30 draws; k = 0 is the pseudo-unseen pooled rule)', '',
         '| shots per species | species coverage (macro) | species set size (of 9) | vacuous | genus coverage | genus set size (of 3) |', '|---|---|---|---|---|---|']
    for r in res:
        L.append('| %d | %.3f ± %.3f | %.2f | %.2f | %.3f | %.2f |' % (r['k'], r['species_cov'], r['species_cov_sd'], r['species_size'],
                                                                      r['vacuous'], r['genus_cov'], r['genus_size']))
    open(os.path.join(OUT, 'fewshot_report.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')

    # ---- (b) coverage vs set size curves
    sc_oof = {k: 1 - oof[k][np.arange(len(yo[k])), yo[k]] for k in FOLDS}
    pooled = np.concatenate(list(sc_oof.values()))
    _, (Xva, yva, dva), _ = m3_arrays()
    sv = 1 - P_val[np.arange(len(yva)), yva]
    curves = {'C0 validation': [], 'C1 pooled pseudo-unseen': [], 'C2b second-worst group': [], 'C2 worst group': []}
    for a in ALPHA_GRID:
        for name, q, P in [('C0 validation', qhat(sv, a), P_full), ('C1 pooled pseudo-unseen', qhat(pooled, a), P_ens),
                           ('C2b second-worst group', sorted(qhat(s, a) for s in sc_oof.values())[-2], P_ens),
                           ('C2 worst group', max(qhat(s, a) for s in sc_oof.values()), P_ens)]:
            u = evaluate(P, yte, ugm, q)
            curves[name].append((u['size'], u['macro_cov']))
    json.dump(curves, open(os.path.join(OUT, 'curves.json'), 'w'), indent=1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, pts in curves.items():
        ax[0].plot(*zip(*pts), marker='o', label=name)
    xs = np.linspace(0, 9, 50)
    ax[0].plot(xs, xs / 9, 'k--', label='random subset of that size')
    ax[0].axhline(0.9, color='grey', lw=0.8, ls=':')
    ax[0].set_xlabel('mean set size (species, of 9)'); ax[0].set_ylabel('coverage on unseen groups (species-averaged)')
    ax[0].set_title('Coverage versus set size, official unseen groups'); ax[0].legend(fontsize=7, loc='lower right'); ax[0].set_ylim(0, 1.02)
    ks = [r['k'] for r in res]
    ax[1].plot(ks, [r['species_cov'] for r in res], 'o-', label='coverage'); ax[1].plot(ks, [r['species_size'] / 9 for r in res], 's--', label='set size / 9')
    ax[1].axhline(0.9, color='grey', lw=0.8, ls=':'); ax[1].set_xlabel('labelled shots per species at the new site (0 = pseudo-unseen rule)')
    ax[1].set_title('Few-shot recalibration, alpha = 0.10'); ax[1].legend(); ax[1].set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'phase2_coverage_vs_size.png'), dpi=140)
    # per-group coverage heat at alpha 0.10 for the 3 key rules
    names = list(ugm)
    rows = {}
    for name, q, P in [('C0 validation', qhat(sv, 0.1), P_full), ('C1 pooled', qhat(pooled, 0.1), P_ens),
                       ('C2b 2nd-worst', sorted(qhat(s, 0.1) for s in sc_oof.values())[-2], P_ens)]:
        rows[name] = [evaluate(P, yte, ugm, q)['per_group'][n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 2.6))
    im = ax.imshow(list(rows.values()), vmin=0, vmax=1, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(names)), names, rotation=45, fontsize=8); ax.set_yticks(range(len(rows)), list(rows), fontsize=8)
    for i, r in enumerate(rows.values()):
        for j, v in enumerate(r):
            ax.text(j, i, '%.2f' % v, ha='center', va='center', color='w' if v < 0.6 else 'k', fontsize=7)
    ax.set_title('Per-group unseen coverage at alpha = 0.10'); fig.colorbar(im); fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'phase2_per_group_coverage.png'), dpi=140)
    print('\n'.join(L)); print({k: [(round(a, 2), round(b, 2)) for a, b in v] for k, v in curves.items()})


if __name__ == '__main__':
    sys.exit(main())
