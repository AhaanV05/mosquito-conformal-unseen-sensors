"""Phase 2f: bootstrap 95% intervals for the 2c calibration rules (AST mlp_aug, alpha = 0.10 and 0.20).
Resamples clips within each pseudo-unseen group (calibration scores), within the validation split (C0) and within each official
unseen group (test). Captures clip-sampling noise only, NOT classifier training variance (see 5-seed results). CPU only.
"""
import json
import os
import sys

import numpy as np

from common import RES, SP_SHORT, ensure_dirs, unseen_domain_by_species_id
from p2b_comp import m3_arrays
from p2b_aug_extract import FOLDS
from p2c_calibrate import qhat

B = 1000


def main():
    out = os.path.join(RES, 'phase2e'); ensure_dirs(out)
    unseen = unseen_domain_by_species_id()
    _, (Xva, yva, dva), (Xte, yte, dte) = m3_arrays()
    part = np.array([1 if unseen.get(a + 1) == b else 0 for a, b in zip(yte, dte)])
    z = np.load(os.path.join(RES, 'phase2c', 'oof.npz'), allow_pickle=True)
    P_ens, P_full, P_val = z['P_ens'], z['P_full_te'], z['P_full_va']
    oof_s = {k: 1 - z['oof_' + k][np.arange(len(z['y_' + k])), z['y_' + k]] for k in FOLDS}
    sv = 1 - P_val[np.arange(len(yva)), yva]
    groups = {c: np.where((part == 1) & (yte == c))[0] for c in range(9) if ((part == 1) & (yte == c)).any()}
    rng = np.random.default_rng(0)
    res = {}
    for a in (0.10, 0.20):
        acc = {r: ([], [], []) for r in ('C0 validation', 'C1 pooled', 'C2b second-worst', 'C2 worst')}
        for _ in range(B):
            cal = {k: rng.choice(v, len(v)) for k, v in oof_s.items()}
            qs = [qhat(v, a) for v in cal.values()]
            q = {'C0 validation': (qhat(rng.choice(sv, len(sv)), a), P_full), 'C1 pooled': (qhat(np.concatenate(list(cal.values())), a), P_ens),
                 'C2b second-worst': (sorted(qs)[-2], P_ens), 'C2 worst': (max(qs), P_ens)}
            tst = {c: rng.choice(ix, len(ix)) for c, ix in groups.items()}
            for r, (thr, P) in q.items():
                cov, size, ge = [], [], 0
                for c, ix in tst.items():
                    S = (1 - P[ix]) <= thr
                    cv = S[np.arange(len(ix)), yte[ix]].mean()
                    cov.append(cv); size.append(S.sum(1).mean()); ge += cv >= 0.8
                acc[r][0].append(np.mean(cov)); acc[r][1].append(np.mean(size)); acc[r][2].append(ge)
        for r, (c, s, g) in acc.items():
            res['%s alpha=%.2f' % (r, a)] = dict(cov=[float(np.mean(c)), *map(float, np.percentile(c, [2.5, 97.5]))],
                                                size=[float(np.mean(s)), *map(float, np.percentile(s, [2.5, 97.5]))],
                                                groups_ge80=[float(np.mean(g)), *map(float, np.percentile(g, [2.5, 97.5]))])
    json.dump(res, open(os.path.join(out, 'bootstrap_ci.json'), 'w'), indent=1)
    L = ['# Phase 2f: bootstrap 95%% intervals, clip sampling only (%d resamples; group-averaged over the 9 official unseen groups)' % B, '',
         '| rule | alpha | unseen coverage | set size (mean over groups, of 9) | groups >= 0.80 (of 9) |', '|---|---|---|---|---|']
    for k, v in res.items():
        r, a = k.rsplit(' alpha=', 1)
        L.append('| %s | %s | %.3f [%.3f, %.3f] | %.2f [%.2f, %.2f] | %.1f [%.0f, %.0f] |' % ((r, a) + tuple(v['cov']) + tuple(v['size']) + tuple(v['groups_ge80'])))
    open(os.path.join(out, 'bootstrap_report.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    sys.exit(main())
