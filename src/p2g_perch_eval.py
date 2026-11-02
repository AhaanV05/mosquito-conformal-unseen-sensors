"""Perch 2.0: head on the padded embedding with noise-swap views (mlp_aug and lin_aug recipes), 9 fold models, 3 seeds each,
then late fusion with AST and Bird-MAE and the 2c calibration rules. Output: results/phase2g/ (report + csv).
"""
import csv
import json
import os
import sys

import numpy as np

import p2a_head as H
from common import FEAT, RES, SP_SHORT, ensure_dirs, log_to, parse_id, unseen_domain_by_species_id
from p2b_aug_extract import FOLDS
from p2b_comp import train_b
from p2c_calibrate import ALPHAS, evaluate, gen_probs, qhat

log = log_to('p2g_perch_eval.log')
OUT = os.path.join(RES, 'phase2g')
SEEDS = (0, 1, 2)


def pc(split):
    z = np.load(os.path.join(FEAT, 'pc_%s.npz' % split), allow_pickle=True)
    return z['emb'].astype(np.float32), np.array([parse_id(i)[0] - 1 for i in z['ids']]), np.array([parse_id(i)[1] for i in z['ids']])


def pc_aug(name):
    z = np.load(os.path.join(FEAT, 'pc_aug_%s.npz' % name))
    return z['emb'].astype(np.float32), z['y'], np.full(len(z['y']), 6)


def main():
    ensure_dirs(OUT)
    unseen = unseen_domain_by_species_id()
    (Xtr, ytr, dtr), (Xva, yva, dva), (Xte, yte, dte) = pc('train'), pc('validation'), pc('test')
    part = np.array([1 if unseen.get(a + 1) == b else 0 for a, b in zip(yte, dte)])
    mic = dva != 5
    Xp = np.concatenate([Xtr, Xva[mic]]); yp = np.concatenate([ytr, yva[mic]]); dp = np.concatenate([dtr, dva[mic]])
    sel = {k: (lambda sp, doms: lambda y, d: (y == sp) & ((d != 5) if doms is None else np.isin(d, doms)))(*v) for k, v in FOLDS.items()}
    cache = os.path.join(OUT, 'oof_pc.npz')
    res = {}
    for kind in ('linear_last', 'mlp_mix'):
        def fit(X, y, d, vname, s):
            ax, ay, ad = pc_aug(vname)
            return train_b(np.concatenate([X, ax]), np.concatenate([y, ay]), np.concatenate([d, ad]), kind, False, s)
        per_seed = []
        for s in SEEDS:
            Pt = H.predict(fit(Xtr, ytr, dtr, 'all', s), Xte)
            fr = {}
            for k, f in sel.items():
                held = f(yp, dp)
                fr[k] = float((H.predict(fit(Xp[~held], yp[~held], dp[~held], k, s), Xp[held]).argmax(1) == yp[held]).mean())
            per_seed.append(dict(seed=s, unseen=H.ba(Pt, yte, part == 1), seen=H.ba(Pt, yte, part == 0), fold=float(np.mean(list(fr.values()))),
                                 bridge=float(np.mean([fr[k] for k in ('Ae.aeg-mic', 'Ae.alb-mic', 'Cx.pip-mic')])), groups=fr))
            log('perch %s seed %d: unseen %.3f fold %.3f bridge %.3f' % (kind, s, per_seed[-1]['unseen'], per_seed[-1]['fold'], per_seed[-1]['bridge']))
        res[kind] = per_seed
    json.dump(res, open(os.path.join(OUT, 'perch_seeds.json'), 'w'), indent=1)
    # ensemble + OOF for calibration and fusion (mlp head)
    if os.path.exists(cache):
        z = np.load(cache); oof = {k: z['oof_' + k] for k in FOLDS}; Pe, Pf_te, Pf_va = z['P_ens'], z['P_full_te'], z['P_full_va']
    else:
        kind = 'mlp_mix'
        def fit2(X, y, d, vname):
            ax, ay, ad = pc_aug(vname)
            return [train_b(np.concatenate([X, ax]), np.concatenate([y, ay]), np.concatenate([d, ad]), kind, False, s) for s in (0, 1)]
        full = fit2(Xtr, ytr, dtr, 'all')
        Pf_te = np.mean([H.predict(n, Xte) for n in full], 0); Pf_va = np.mean([H.predict(n, Xva) for n in full], 0)
        oof, te = {}, []
        for k, f in sel.items():
            held = f(yp, dp)
            nets = fit2(Xp[~held], yp[~held], dp[~held], k)
            oof[k] = np.mean([H.predict(n, Xp[held]) for n in nets], 0); te.append(np.mean([H.predict(n, Xte) for n in nets], 0))
        Pe = np.mean(te, 0)
        np.savez_compressed(cache, P_ens=Pe, P_full_te=Pf_te, P_full_va=Pf_va, **{'oof_' + k: v for k, v in oof.items()})
    a = np.load(os.path.join(RES, 'phase2c', 'oof.npz'), allow_pickle=True)
    b = np.load(os.path.join(RES, 'phase2d', 'oof_bm.npz'))
    yo = {k: a['y_' + k] for k in FOLDS}
    S = {'Perch': (oof, Pe, Pf_te, Pf_va), 'AST': ({k: a['oof_' + k] for k in FOLDS}, a['P_ens'], a['P_full_te'], a['P_full_va'])}
    S['AST+Perch'] = ({k: (S['AST'][0][k] + oof[k]) / 2 for k in FOLDS}, (a['P_ens'] + Pe) / 2, (a['P_full_te'] + Pf_te) / 2, (a['P_full_va'] + Pf_va) / 2)
    S['AST+Bird+Perch'] = ({k: (S['AST'][0][k] + b['oof_' + k] + oof[k]) / 3 for k in FOLDS}, (a['P_ens'] + b['P_ens'] + Pe) / 3,
                           (a['P_full_te'] + b['P_full_te'] + Pf_te) / 3, (a['P_full_va'] + b['P_full_va'] + Pf_va) / 3)
    ug = {SP_SHORT[c + 1]: (part == 1) & (yte == c) for c in range(9) if ((part == 1) & (yte == c)).any()}
    L = ['# Perch 2.0 (community PyTorch port) and fusion', '', '## Perch alone, per seed (single-seed models; 3 seeds)', '',
         '| head | unseen BA | seen BA | fold accuracy | bridge folds |', '|---|---|---|---|---|']
    for kind, ps in res.items():
        g = lambda m: np.array([p[m] for p in ps])
        L.append('| %s + aug | %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f |' % (kind, g('unseen').mean(), g('unseen').std(ddof=1), g('seen').mean(),
                                                                                         g('seen').std(ddof=1), g('fold').mean(), g('fold').std(ddof=1), g('bridge').mean(), g('bridge').std(ddof=1)))
    L += ['', '## Ensembles and fusion (mlp head; AST and Bird-MAE from 2c/2d)', '', '| model | BA unseen (ens / full) | BA seen (full) | fold accuracy | bridge folds |', '|---|---|---|---|---|']
    rows = []
    for name, (o, P_ens, P_te, P_va) in S.items():
        accs = {k: float((o[k].argmax(1) == yo[k]).mean()) for k in FOLDS}
        L.append('| %s | %.3f / %.3f | %.3f | %.3f | %.3f |' % (name, H.ba(P_ens, yte, part == 1), H.ba(P_te, yte, part == 1), H.ba(P_te, yte, part == 0),
                                                           np.mean(list(accs.values())), np.mean([accs[k] for k in ('Ae.aeg-mic', 'Ae.alb-mic', 'Cx.pip-mic')])))
        sc = {k: 1 - o[k][np.arange(len(yo[k])), yo[k]] for k in FOLDS}
        sv = 1 - P_va[np.arange(len(yva)), yva]
        for al in (0.10, 0.20):
            for rule, q, P in [('C0 validation', qhat(sv, al), P_te), ('C1 pooled', qhat(np.concatenate(list(sc.values())), al), P_ens),
                               ('C2b second-worst', sorted(qhat(s_, al) for s_ in sc.values())[-2], P_ens), ('C2 worst', max(qhat(s_, al) for s_ in sc.values()), P_ens)]:
                u = evaluate(P, yte, ug, q)
                rows.append(dict(model=name, alpha=al, rule=rule, q=q, unseen_cov=u['macro_cov'], size=u['size'], vacuous=u['vacuous'],
                                 groups_ge80=int(sum(v >= .8 for v in u['per_group'].values()))))
    L += ['', '## Calibration on the official unseen groups', '', '| model | alpha | rule | unseen coverage | set size | vacuous | groups >= 0.80 |', '|---|---|---|---|---|---|---|']
    for r in rows:
        L.append('| %s | %.2f | %s | %.3f | %.2f | %.2f | %d/9 |' % (r['model'], r['alpha'], r['rule'], r['unseen_cov'], r['size'], r['vacuous'], r['groups_ge80']))
    open(os.path.join(OUT, 'perch_report.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    with open(os.path.join(OUT, 'perch_calibration.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    log('wrote report')


if __name__ == '__main__':
    sys.exit(main())
