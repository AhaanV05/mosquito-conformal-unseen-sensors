"""Phase 2c: domain-aware conformal calibration + taxonomy sets on the 2b model (mlp_aug, padded AST + noise-swap views).

Out-of-fold (OOF) probabilities for the 9 pseudo-unseen groups come from fold models that never saw the group.
Rules: C0 (Phase 1 style: Validation, full model), C1 (pooled OOF), C2 (max over groups of per-group quantile),
C2b (second largest). Predictor for C1/C2: ensemble of all fold models. Sets: species (LAC score), lifted to taxonomy nodes,
and genus-level conformal. Evaluated on the 9 official unseen groups (dev test). Output: results/phase2c/.
"""
import json
import math
import os
import sys

import numpy as np

import p2a_head as H
from common import RES, SP_SHORT, ensure_dirs, log_to, unseen_domain_by_species_id
from p2b_aug_eval import aug, rep4
from p2b_aug_extract import FOLDS
from p2b_comp import m3_arrays, train_b

log = log_to('p2c_calibrate.log')
OUT = os.path.join(RES, 'phase2c')
ALPHAS = (0.05, 0.10, 0.20)
GEN = H.GEN
AN = [3, 4, 5, 7, 8]                                  # Anopheles species indices


def qhat(s, a):
    n = len(s)
    k = math.ceil((n + 1) * (1 - a))
    return np.inf if k > n else float(np.sort(s)[k - 1])


def gen_probs(P):
    return np.stack([P[:, GEN == g].sum(1) for g in range(3)], 1)


def lift(S):
    """S: bool (9,). Widen to the whole Anopheles genus if >= 3 of its 5 species are in S. Returns (leaf mask, node count, coarse)."""
    L = S.copy()
    if S[AN].sum() >= 3:
        L[AN] = True
    nodes = 0
    coarse = True
    for g, members in enumerate([[0, 1], [2, 6], AN]):
        k = int(L[members].sum())
        if k == 0:
            continue
        if k == len(members) or (g == 2 and set(np.where(L)[0]) >= {3, 4} and k == 2 and L[[3, 4]].all()):
            nodes += 1                                  # genus node, or the gambiae complex when only gam+ara
        else:
            nodes += k; coarse = False
    return L, nodes, coarse


def evaluate(P, y, groups, q, level='species'):
    """Coverage (per group, macro, micro), mean set size, empty and vacuous shares."""
    if level == 'genus':
        P, y = gen_probs(P), GEN[y]
    S = (1 - P) <= q
    cov = S[np.arange(len(y)), y]
    per = {g: float(cov[m].mean()) for g, m in groups.items()}
    return {'macro_cov': float(np.mean(list(per.values()))), 'micro_cov': float(cov.mean()),
            'size': float(S.sum(1).mean()), 'empty': float((S.sum(1) == 0).mean()),
            'vacuous': float((S.sum(1) == P.shape[1]).mean()), 'per_group': per}


def main():
    ensure_dirs(OUT)
    unseen = unseen_domain_by_species_id()
    (Xtr, ytr, dtr), (Xva, yva, dva), (Xte, yte, dte) = m3_arrays()
    part = np.array([1 if unseen.get(a + 1) == b else 0 for a, b in zip(yte, dte)])
    mic = dva != 5
    Xp = rep4(np.concatenate([Xtr, Xva[mic]])); yp = np.concatenate([ytr, yva[mic]]); dp = np.concatenate([dtr, dva[mic]])
    Xtr4, Xte4, Xva4 = rep4(Xtr), rep4(Xte), rep4(Xva)
    sel = {k: (lambda sp, doms: lambda y, d: (y == sp) & ((d != 5) if doms is None else np.isin(d, doms)))(*v) for k, v in FOLDS.items()}
    cache = os.path.join(OUT, 'oof.npz')
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        oof = {k: (z['oof_' + k], z['y_' + k]) for k in FOLDS}
        P_ens, P_full_te, P_full_va = z['P_ens'], z['P_full_te'], z['P_full_va']
    else:
        def fit(X, y, d, vname):
            ax, ay, ad = aug(vname)
            return [train_b(np.concatenate([X, ax]), np.concatenate([y, ay]), np.concatenate([d, ad]), 'mlp_mix', False, s) for s in (0, 1)]
        full = fit(Xtr4, ytr, dtr, 'all')
        P_full_te = np.mean([H.predict(n, Xte4) for n in full], 0)
        P_full_va = np.mean([H.predict(n, Xva4) for n in full], 0)
        oof, te_list = {}, []
        for fname, f in sel.items():
            held = f(yp, dp)
            nets = fit(Xp[~held], yp[~held], dp[~held], fname)
            oof[fname] = (np.mean([H.predict(n, Xp[held]) for n in nets], 0), yp[held])
            te_list.append(np.mean([H.predict(n, Xte4) for n in nets], 0))
            log('fold %s done (n=%d)' % (fname, held.sum()))
        P_ens = np.mean(te_list, 0)
        np.savez_compressed(cache, P_ens=P_ens, P_full_te=P_full_te, P_full_va=P_full_va,
                            **{'oof_' + k: v[0] for k, v in oof.items()}, **{'y_' + k: v[1] for k, v in oof.items()})
    log('BA unseen: ensemble %.3f full %.3f' % (H.ba(P_ens, yte, part == 1), H.ba(P_full_te, yte, part == 1)))
    ug = {SP_SHORT[c + 1]: (part == 1) & (yte == c) for c in range(9) if ((part == 1) & (yte == c)).any()}
    sg = {SP_SHORT[c + 1]: (part == 0) & (yte == c) for c in range(9) if ((part == 0) & (yte == c)).any()}
    sc_oof = {k: 1 - v[0][np.arange(len(v[1])), v[1]] for k, v in oof.items()}
    gsc_oof = {k: 1 - gen_probs(v[0])[np.arange(len(v[1])), GEN[v[1]]] for k, v in oof.items()}
    sc_val = 1 - P_full_va[np.arange(len(yva)), yva]
    rows = []
    for a in ALPHAS:
        rules = {'C0 validation': (qhat(sc_val, a), P_full_te),
                 'C1 pooled OOF': (qhat(np.concatenate(list(sc_oof.values())), a), P_ens),
                 'C2 max group': (max(qhat(s, a) for s in sc_oof.values()), P_ens),
                 'C2b 2nd-largest group': (sorted(qhat(s, a) for s in sc_oof.values())[-2], P_ens)}
        grules = {'G0 validation': (qhat(1 - gen_probs(P_full_va)[np.arange(len(yva)), GEN[yva]], a), P_full_te),
                  'G2 max group': (max(qhat(s, a) for s in gsc_oof.values()), P_ens)}
        for name, (q, P) in rules.items():
            u, s_ = evaluate(P, yte, ug, q), evaluate(P, yte, sg, q)
            S = (1 - P[part == 1]) <= q
            lifts = [lift(s) for s in S]
            Lm = np.array([l[0] for l in lifts])
            lc = Lm[np.arange(len(Lm)), yte[part == 1]]
            rows.append(dict(alpha=a, rule=name, level='species', q=q, unseen_macro=u['macro_cov'], unseen_micro=u['micro_cov'],
                             seen_macro=s_['macro_cov'], size=u['size'], empty=u['empty'], vacuous=u['vacuous'],
                             lifted_macro_cov=float(np.mean([lc[yte[part == 1] == c].mean() for c in np.unique(yte[part == 1])])),
                             lifted_size=float(Lm.sum(1).mean()), nodes=float(np.mean([l[1] for l in lifts])),
                             coarse_share=float(np.mean([l[2] for l in lifts])), groups_ge80=int(sum(v >= 0.8 for v in u['per_group'].values())),
                             per_group=u['per_group']))
        for name, (q, P) in grules.items():
            u = evaluate(P, yte, ug, q, 'genus')
            rows.append(dict(alpha=a, rule=name, level='genus', q=q, unseen_macro=u['macro_cov'], unseen_micro=u['micro_cov'],
                             seen_macro=evaluate(P, yte, sg, q, 'genus')['macro_cov'], size=u['size'], empty=u['empty'],
                             vacuous=u['vacuous'], groups_ge80=int(sum(v >= 0.8 for v in u['per_group'].values())), per_group=u['per_group']))
    json.dump(rows, open(os.path.join(OUT, 'phase2c_results.json'), 'w'), indent=1)
    L = ['# Phase 2c results (dev test, 9 official unseen groups; classifier = mlp_aug, ensemble of 9 fold models x 2 seeds)', '',
         'Unseen BA: ensemble %.3f, full model %.3f. Pseudo-unseen calibration groups: %s.' % (
             H.ba(P_ens, yte, part == 1), H.ba(P_full_te, yte, part == 1), {k: len(v[1]) for k, v in oof.items()}), '',
         '| alpha | rule | level | q | unseen macro cov | seen macro cov | set size | vacuous | groups >= 0.80 | lifted macro cov | lifted size | coarse share |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        L.append('| %.2f | %s | %s | %.3f | %.3f | %.3f | %.2f | %.2f | %d/9 | %s | %s | %s |' % (
            r['alpha'], r['rule'], r['level'], r['q'], r['unseen_macro'], r['seen_macro'], r['size'], r['vacuous'], r['groups_ge80'],
            '%.3f' % r['lifted_macro_cov'] if 'lifted_macro_cov' in r else '', '%.2f' % r['lifted_size'] if 'lifted_size' in r else '',
            '%.2f' % r['coarse_share'] if 'coarse_share' in r else ''))
    L += ['', '## Per-group unseen coverage at alpha = 0.10', '']
    for r in rows:
        if r['alpha'] == 0.10:
            L.append('- %s (%s): %s' % (r['rule'], r['level'], ', '.join('%s %.2f' % kv for kv in r['per_group'].items())))
    open(os.path.join(OUT, 'phase2c_report.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    log('wrote report')


if __name__ == '__main__':
    sys.exit(main())
