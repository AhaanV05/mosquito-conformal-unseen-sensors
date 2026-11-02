"""Phase 2b (audio-level): same 9 pseudo-unseen folds, Phase 1 padded AST embeddings, clean vs clean + augmented D5 views.
Augmented views are tagged domain 6 (pseudo-microphone). Fold models use the fold-specific noise bank views (leak-free).
Configs: lin_clean (reference), lin_aug, mlp_aug. Output: results/phase2a/p2b_aug_results.json
"""
import json
import os
import sys

import numpy as np

import p2a_head as H
from common import FEAT, RES, log_to, unseen_domain_by_species_id
from p2b_aug_extract import FOLDS
from p2b_comp import m3_arrays, train_b

log = log_to('p2b_aug_eval.log')


def aug(name):
    z = np.load(os.path.join(FEAT, 'p2b_aug_%s.npz' % name))
    return np.repeat(z['emb'][:, None, :].astype(np.float32), 4, axis=1), z['y'], np.full(len(z['y']), 6)


def rep4(a):
    return np.repeat(a, 4, axis=1)


def main():
    unseen = unseen_domain_by_species_id()
    (Xtr, ytr, dtr), (Xva, yva, dva), (Xte, yte, dte) = m3_arrays()
    part = np.array([1 if unseen.get(a + 1) == b else 0 for a, b in zip(yte, dte)])
    mic = dva != 5
    Xp = rep4(np.concatenate([Xtr, Xva[mic]])); yp = np.concatenate([ytr, yva[mic]]); dp = np.concatenate([dtr, dva[mic]])
    Xtr, Xte = rep4(Xtr), rep4(Xte)
    sel = {k: (lambda sp, doms: lambda y, d: (y == sp) & ((d != 5) if doms is None else np.isin(d, doms)))(*v) for k, v in FOLDS.items()}
    rp = os.path.join(RES, 'phase2a', 'p2b_aug_results.json')
    out = json.load(open(rp)) if os.path.exists(rp) else {}
    seeds = [0, 1]
    for name, kind, use_aug in [('lin_clean', 'linear_last', False), ('lin_aug', 'linear_last', True), ('mlp_aug', 'mlp_mix', True)]:
        if name in out:
            continue

        def fit(X, y, d, vname):
            if use_aug:
                ax, ay, ad = aug(vname)
                X, y, d = np.concatenate([X, ax]), np.concatenate([y, ay]), np.concatenate([d, ad])
            return [train_b(X, y, d, kind, False, s) for s in seeds]
        P = np.mean([H.predict(n, Xte) for n in fit(Xtr, ytr, dtr, 'all')], 0)
        res = {'test_BA_seen': H.ba(P, yte, part == 0), 'test_BA_unseen': H.ba(P, yte, part == 1),
               'genus_acc_unseen': float((H.GEN[P[part == 1].argmax(1)] == H.GEN[yte[part == 1]]).mean())}
        fr = {}
        for fname, f in sel.items():
            held = f(yp, dp)
            nets = fit(Xp[~held], yp[~held], dp[~held], fname)
            pf = np.mean([H.predict(n, Xp[held]) for n in nets], 0)
            fr[fname] = (float((pf.argmax(1) == yp[held]).mean()), int(held.sum()), float((H.GEN[pf.argmax(1)] == H.GEN[yp[held]]).mean()))
        res['folds'] = fr
        res['fold_macro_acc'] = float(np.mean([v[0] for v in fr.values()]))
        res['fold_bridge_acc'] = float(np.mean([fr[f][0] for f in ('Ae.aeg-mic', 'Ae.alb-mic', 'Cx.pip-mic')]))
        out[name] = res
        np.savez_compressed(os.path.join(RES, 'phase2a', 'probs_aug_%s.npz' % name), probs=P, y=yte, domain=dte, partition=part)
        log('%s: test BA seen %.3f unseen %.3f genus %.3f | fold macro %.3f bridge-only %.3f' % (
            name, res['test_BA_seen'], res['test_BA_unseen'], res['genus_acc_unseen'], res['fold_macro_acc'], res['fold_bridge_acc']))
        json.dump(out, open(rp, 'w'), indent=1)


if __name__ == '__main__':
    sys.exit(main())
