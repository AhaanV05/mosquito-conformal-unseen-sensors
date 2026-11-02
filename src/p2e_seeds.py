"""Phase 2e: 5-seed repeat of the key AST comparison (padded input): lin_clean vs lin_aug vs mlp_aug.
Per-seed metrics from single-seed models (not ensembles): test BA unseen/seen, fold accuracy over the 9 pseudo-unseen groups,
bridge-fold accuracy. Reports mean +- sd over seeds and the paired difference to lin_clean. Output: results/phase2e/.
"""
import json
import os
import sys

import numpy as np

import p2a_head as H
from common import RES, ensure_dirs, log_to, unseen_domain_by_species_id
from p2b_aug_eval import aug, rep4
from p2b_aug_extract import FOLDS
from p2b_comp import m3_arrays, train_b

log = log_to('p2e_seeds.log')
OUT = os.path.join(RES, 'phase2e')
SEEDS = [0, 1, 2, 3, 4]


def main():
    ensure_dirs(OUT)
    unseen = unseen_domain_by_species_id()
    (Xtr, ytr, dtr), (Xva, yva, dva), (Xte, yte, dte) = m3_arrays()
    part = np.array([1 if unseen.get(a + 1) == b else 0 for a, b in zip(yte, dte)])
    mic = dva != 5
    Xp = rep4(np.concatenate([Xtr, Xva[mic]])); yp = np.concatenate([ytr, yva[mic]]); dp = np.concatenate([dtr, dva[mic]])
    Xtr, Xte = rep4(Xtr), rep4(Xte)
    sel = {k: (lambda sp, doms: lambda y, d: (y == sp) & ((d != 5) if doms is None else np.isin(d, doms)))(*v) for k, v in FOLDS.items()}
    rp = os.path.join(OUT, 'seed_results.json')
    out = json.load(open(rp)) if os.path.exists(rp) else {}
    for name, kind, use_aug in [('lin_clean', 'linear_last', False), ('lin_aug', 'linear_last', True), ('mlp_aug', 'mlp_mix', True)]:
        out.setdefault(name, {})
        for s in SEEDS:
            if str(s) in out[name]:
                continue

            def fit(X, y, d, vname):
                if use_aug:
                    ax, ay, ad = aug(vname)
                    X, y, d = np.concatenate([X, ax]), np.concatenate([y, ay]), np.concatenate([d, ad])
                return train_b(X, y, d, kind, False, s)
            P = H.predict(fit(Xtr, ytr, dtr, 'all'), Xte)
            fr = {}
            for fname, f in sel.items():
                held = f(yp, dp)
                pf = H.predict(fit(Xp[~held], yp[~held], dp[~held], fname), Xp[held])
                fr[fname] = float((pf.argmax(1) == yp[held]).mean())
            out[name][str(s)] = {'unseen': H.ba(P, yte, part == 1), 'seen': H.ba(P, yte, part == 0), 'fold': float(np.mean(list(fr.values()))),
                                 'bridge': float(np.mean([fr[k] for k in ('Ae.aeg-mic', 'Ae.alb-mic', 'Cx.pip-mic')])), 'groups': fr}
            json.dump(out, open(rp, 'w'), indent=1)
            log('%s seed %d: unseen %.3f fold %.3f bridge %.3f' % (name, s, out[name][str(s)]['unseen'], out[name][str(s)]['fold'], out[name][str(s)]['bridge']))
    L = ['# Phase 2e: 5 seeds, AST padded input (single-seed models, mean ± sd over seeds)', '',
         '| config | test unseen BA | test seen BA | fold accuracy | bridge folds | paired fold diff vs lin_clean |', '|---|---|---|---|---|---|']
    base = np.array([out['lin_clean'][str(s)]['fold'] for s in SEEDS])
    for name in out:
        g = lambda m: np.array([out[name][str(s)][m] for s in SEEDS])
        d = g('fold') - base
        L.append('| %s | %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f | %+.3f ± %.3f |' % (
            name, g('unseen').mean(), g('unseen').std(ddof=1), g('seen').mean(), g('seen').std(ddof=1), g('fold').mean(), g('fold').std(ddof=1),
            g('bridge').mean(), g('bridge').std(ddof=1), d.mean(), d.std(ddof=1)))
    open(os.path.join(OUT, 'seed_report.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    log('wrote report')


if __name__ == '__main__':
    sys.exit(main())
