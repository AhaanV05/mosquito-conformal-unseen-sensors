"""Phase 2d: Bird-MAE head (mlp_aug recipe), late fusion with the AST model, and the 2c calibration rules on the fused classifier.

Bird-MAE: padded-input 4-tap embeddings + noise-swap views, 9 fold models x 2 seeds + a full model, as in 2c for AST.
Fusion: mean of AST and Bird-MAE probabilities (OOF rows, test, validation all align). Output: results/phase2d/.
"""
import json
import os
import sys

import numpy as np

import p2a_head as H
from common import FEAT, RES, SP_SHORT, ensure_dirs, log_to, parse_id, unseen_domain_by_species_id
from p2b_aug_extract import FOLDS
from p2b_comp import train_b
from p2c_calibrate import ALPHAS, evaluate, gen_probs, lift, qhat

log = log_to('p2d_bird_fuse.log')
OUT = os.path.join(RES, 'phase2d')


def bm(split):
    z = np.load(os.path.join(FEAT, 'bm_%s.npz' % split), allow_pickle=True)
    ids = z['ids']
    return z['emb'].astype(np.float32), np.array([parse_id(i)[0] - 1 for i in ids]), np.array([parse_id(i)[1] for i in ids])


def bm_aug(name):
    z = np.load(os.path.join(FEAT, 'bm_aug_%s.npz' % name))
    return z['emb'].astype(np.float32), z['y'], np.full(len(z['y']), 6)


def main():
    ensure_dirs(OUT)
    unseen = unseen_domain_by_species_id()
    (Xtr, ytr, dtr), (Xva, yva, dva), (Xte, yte, dte) = bm('train'), bm('validation'), bm('test')
    part = np.array([1 if unseen.get(a + 1) == b else 0 for a, b in zip(yte, dte)])
    mic = dva != 5
    Xp = np.concatenate([Xtr, Xva[mic]]); yp = np.concatenate([ytr, yva[mic]]); dp = np.concatenate([dtr, dva[mic]])
    sel = {k: (lambda sp, doms: lambda y, d: (y == sp) & ((d != 5) if doms is None else np.isin(d, doms)))(*v) for k, v in FOLDS.items()}
    cache = os.path.join(OUT, 'oof_bm.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        oof = {k: z['oof_' + k] for k in FOLDS}
        Pe, Pf_te, Pf_va = z['P_ens'], z['P_full_te'], z['P_full_va']
    else:
        def fit(X, y, d, vname):
            ax, ay, ad = bm_aug(vname)
            return [train_b(np.concatenate([X, ax]), np.concatenate([y, ay]), np.concatenate([d, ad]), 'mlp_mix', False, s) for s in (0, 1)]
        full = fit(Xtr, ytr, dtr, 'all')
        Pf_te = np.mean([H.predict(n, Xte) for n in full], 0)
        Pf_va = np.mean([H.predict(n, Xva) for n in full], 0)
        oof, te = {}, []
        for k, f in sel.items():
            held = f(yp, dp)
            nets = fit(Xp[~held], yp[~held], dp[~held], k)
            oof[k] = np.mean([H.predict(n, Xp[held]) for n in nets], 0)
            te.append(np.mean([H.predict(n, Xte) for n in nets], 0))
            log('bm fold %s done' % k)
        Pe = np.mean(te, 0)
        np.savez_compressed(cache, P_ens=Pe, P_full_te=Pf_te, P_full_va=Pf_va, **{'oof_' + k: v for k, v in oof.items()})
    a = np.load(os.path.join(RES, 'phase2c', 'oof.npz'), allow_pickle=True)
    yo = {k: a['y_' + k] for k in FOLDS}
    sets = {'AST': ({k: a['oof_' + k] for k in FOLDS}, a['P_ens'], a['P_full_te'], a['P_full_va']),
            'BirdMAE': (oof, Pe, Pf_te, Pf_va)}
    sets['Fused'] = ({k: (sets['AST'][0][k] + oof[k]) / 2 for k in FOLDS}, (a['P_ens'] + Pe) / 2,
                     (a['P_full_te'] + Pf_te) / 2, (a['P_full_va'] + Pf_va) / 2)
    ug = {SP_SHORT[c + 1]: (part == 1) & (yte == c) for c in range(9) if ((part == 1) & (yte == c)).any()}
    L = ['# Phase 2d: Bird-MAE and late fusion (dev test, 9 official unseen groups)', '',
         '| model | BA unseen (ens / full) | BA seen (full) | fold accuracy (macro) | bridge folds | genus acc unseen |', '|---|---|---|---|---|---|']
    rows = []
    for name, (o, P_ens, P_te, P_va) in sets.items():
        accs = {k: float((o[k].argmax(1) == yo[k]).mean()) for k in FOLDS}
        gacc = float((H.GEN[P_te[part == 1].argmax(1)] == H.GEN[yte[part == 1]]).mean())
        L.append('| %s | %.3f / %.3f | %.3f | %.3f | %.3f | %.3f |' % (
            name, H.ba(P_ens, yte, part == 1), H.ba(P_te, yte, part == 1), H.ba(P_te, yte, part == 0), np.mean(list(accs.values())),
            np.mean([accs[k] for k in ('Ae.aeg-mic', 'Ae.alb-mic', 'Cx.pip-mic')]), gacc))
        sc = {k: 1 - o[k][np.arange(len(yo[k])), yo[k]] for k in FOLDS}
        gsc = {k: 1 - gen_probs(o[k])[np.arange(len(yo[k])), H.GEN[yo[k]]] for k in FOLDS}
        sv = 1 - P_va[np.arange(len(yva)), yva]
        for al in ALPHAS:
            for rule, q, P in [('C0 validation', qhat(sv, al), P_te), ('C1 pooled OOF', qhat(np.concatenate(list(sc.values())), al), P_ens),
                               ('C2 max group', max(qhat(s, al) for s in sc.values()), P_ens),
                               ('C2b 2nd-largest', sorted(qhat(s, al) for s in sc.values())[-2], P_ens)]:
                u = evaluate(P, yte, ug, q)
                rows.append(dict(model=name, alpha=al, rule=rule, level='species', q=q, cov=u['macro_cov'], size=u['size'],
                                 vacuous=u['vacuous'], ge80=int(sum(v >= .8 for v in u['per_group'].values())), per_group=u['per_group']))
            for rule, q, P in [('G0 validation', qhat(1 - gen_probs(P_va)[np.arange(len(yva)), H.GEN[yva]], al), P_te),
                               ('G2 max group', max(qhat(s, al) for s in gsc.values()), P_ens)]:
                u = evaluate(P, yte, ug, q, 'genus')
                rows.append(dict(model=name, alpha=al, rule=rule, level='genus', q=q, cov=u['macro_cov'], size=u['size'],
                                 vacuous=u['vacuous'], ge80=int(sum(v >= .8 for v in u['per_group'].values())), per_group=u['per_group']))
    L += ['', '| model | alpha | rule | level | q | unseen macro cov | set size | vacuous | groups >= 0.80 |', '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        L.append('| %s | %.2f | %s | %s | %.3f | %.3f | %.2f | %.2f | %d/9 |' % (r['model'], r['alpha'], r['rule'], r['level'], r['q'],
                                                                              r['cov'], r['size'], r['vacuous'], r['ge80']))
    json.dump(rows, open(os.path.join(OUT, 'phase2d_results.json'), 'w'), indent=1)
    open(os.path.join(OUT, 'phase2d_report.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    log('wrote report')


if __name__ == '__main__':
    sys.exit(main())
