"""Phase 1: does standard split-conformal calibration keep its promise on unseen domains?
Plus Phase 1b repair previews that need no retraining (temperature scaling, few-shot target calibration,
shift-weighted calibration, taxonomy-level sets).

Reads results/phase1/preds/<family>_seed<k>_{validation,test}.npz (families m1, m2, m3) and writes
results/phase1/phase1_report.md, CSV tables and figures.
"""
import argparse
import glob
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from common import GENUS, GROUP, RES, SP_SHORT, ensure_dirs, log_to  # noqa: E402

PRED = os.path.join(RES, 'phase1', 'preds')
OUT = os.path.join(RES, 'phase1')
FIGS = os.path.join(OUT, 'figs')
ALPHAS = [0.05, 0.10, 0.20]
FAMILY_LABEL = {'m1': 'M1 released baseline CNN', 'm2': 'M2 pitch/harmonic + boosting', 'm3': 'M3 AST embedding probe'}
log = log_to('phase1_conformal.log')
ensure_dirs(FIGS)


# ---------------------------------------------------------------- loading
def load_family(prefix):
    seeds = sorted({int(os.path.basename(p).split('seed')[1].split('_')[0])
                    for p in glob.glob(os.path.join(PRED, prefix + '_seed*_test.npz'))})
    fam = {}
    for s in seeds:
        v = np.load(os.path.join(PRED, '%s_seed%d_validation.npz' % (prefix, s)), allow_pickle=True)
        t = np.load(os.path.join(PRED, '%s_seed%d_test.npz' % (prefix, s)), allow_pickle=True)
        fam[s] = dict(vp=v['probs'].astype(np.float64), vy=v['y'], vd=v['domain'], vid=v['file_id'],
                      tp=t['probs'].astype(np.float64), ty=t['y'], td=t['domain'], tpart=t['partition'],
                      tid=t['file_id'],
                      vfeat=v['feat'] if 'feat' in v.files else None, tfeat=t['feat'] if 'feat' in t.files else None)
    return fam


def ensemble(fam):
    ks = sorted(fam)
    e = dict(fam[ks[0]])
    e['vp'] = np.mean([fam[k]['vp'] for k in ks], axis=0)
    e['tp'] = np.mean([fam[k]['tp'] for k in ks], axis=0)
    return e


# ---------------------------------------------------------------- conformal
def lac_q(scores, alpha):
    s = np.sort(scores)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else s[k - 1]


def sets_marginal(probs, cal_p, cal_y, alpha):
    q = lac_q(1 - cal_p[np.arange(len(cal_y)), cal_y], alpha)
    return (1 - probs) <= q


def sets_classcond(probs, cal_p, cal_y, alpha, n_classes=9):
    S = np.zeros_like(probs, dtype=bool)
    for c in range(n_classes):
        m = cal_y == c
        q = lac_q(1 - cal_p[m, c], alpha) if m.any() else np.inf
        S[:, c] = (1 - probs[:, c]) <= q
    return S


def set_metrics(S, y, mask, n_classes=9):
    if mask.sum() == 0:
        return None
    Sm, ym = S[mask], y[mask]
    cov = Sm[np.arange(len(ym)), ym]
    per = [cov[ym == c].mean() for c in range(n_classes) if (ym == c).any()]
    size = Sm.sum(1)
    return dict(n=int(mask.sum()), micro_cov=float(cov.mean()), macro_cov=float(np.mean(per)),
                set_size=float(size.mean()), empty=float((size == 0).mean()), singleton=float((size == 1).mean()))


def boot_ci(S, y, mask, reps=500, seed=0, n_classes=9):
    rng = np.random.default_rng(seed)
    cov = S[mask][np.arange(mask.sum()), y[mask]].astype(float)
    ym = y[mask]
    micro, macro = [], []
    for _ in range(reps):
        idx = rng.integers(0, len(ym), len(ym))
        c, yy = cov[idx], ym[idx]
        cnt = np.bincount(yy, minlength=n_classes)
        tot = np.bincount(yy, weights=c, minlength=n_classes)
        micro.append(c.mean())
        macro.append((tot[cnt > 0] / cnt[cnt > 0]).mean())
    return (np.percentile(micro, [2.5, 97.5]).tolist(), np.percentile(macro, [2.5, 97.5]).tolist())


def softmax_T(p, T):
    z = np.log(np.clip(p, 1e-12, 1)) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(p, y):
    best, bt = np.inf, 1.0
    for T in np.exp(np.linspace(np.log(0.5), np.log(8), 40)):
        nll = -np.log(np.clip(softmax_T(p, T)[np.arange(len(y)), y], 1e-12, 1)).mean()
        if nll < best:
            best, bt = nll, T
    return bt


def weighted_q(scores, w, level):
    order = np.argsort(scores)
    s, ww = scores[order], w[order]
    cum = np.cumsum(ww) / (ww.sum() + ww.mean())      # one extra "test point" worth of average weight at +inf
    idx = np.searchsorted(cum, level)
    return np.inf if idx >= len(s) else s[idx]


def agg_matrix(mapping):
    names = sorted(set(mapping.values()))
    M = np.zeros((9, len(names)))
    for sid, g in mapping.items():
        M[sid - 1, names.index(g)] = 1
    return M, names


# ---------------------------------------------------------------- analysis
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--families', default='m1,m2,m3')
    args = ap.parse_args()
    fams = {}
    for f in args.families.split(','):
        fam = load_family(f)
        if fam:
            fams[f] = fam
            log('loaded %s: %d seeds, validation %d, test %d' % (f, len(fam), len(fam[min(fam)]['vy']), len(fam[min(fam)]['ty'])))
        else:
            log('family %s not available, skipped' % f)
    if not fams:
        raise SystemExit('no predictions found')
    ref = fams[sorted(fams)[0]]
    ref_k = min(ref)
    tid_ref = ref[ref_k]['tid']
    # sanity: all families must be in the same clip order
    for f, fam in fams.items():
        for k, d in fam.items():
            if not (len(d['tid']) == len(tid_ref) and np.array_equal(d['tid'], tid_ref)):
                raise SystemExit('clip order mismatch in %s seed %s' % (f, k))
    y = ref[ref_k]['ty']
    dom = ref[ref_k]['td']
    part = ref[ref_k]['tpart']
    seen_m, unseen_m = part == 0, part == 1
    log('test clips: %d seen, %d unseen' % (seen_m.sum(), unseen_m.sum()))

    rows = []
    for f, fam in fams.items():
        members = [(str(k), d) for k, d in fam.items()] + [('ens', ensemble(fam))]
        for tag, d in members:
            for alpha in ALPHAS:
                for method in ['marginal', 'classcond']:
                    S = (sets_marginal if method == 'marginal' else sets_classcond)(d['tp'], d['vp'], d['vy'], alpha)
                    for pname, m in [('seen', seen_m), ('unseen', unseen_m)]:
                        r = set_metrics(S, y, m)
                        r.update(family=f, member=tag, alpha=alpha, method=method, partition=pname)
                        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, 'phase1_results_all.csv'), index=False)
    seedrows = res[res['member'] != 'ens']
    agg = (seedrows.groupby(['family', 'method', 'alpha', 'partition'])
           [['micro_cov', 'macro_cov', 'set_size', 'empty', 'singleton']].agg(['mean', 'std']))
    agg.columns = ['_'.join(c) for c in agg.columns]
    agg.to_csv(os.path.join(OUT, 'phase1_results_by_family.csv'))

    # gate at alpha = 0.10, marginal split conformal, species-averaged coverage on unseen clips
    gate_rows = []
    for f in fams:
        g = seedrows[(seedrows['family'] == f) & (seedrows['alpha'] == 0.10) & (seedrows['method'] == 'marginal')]
        su = g[g['partition'] == 'unseen']['macro_cov']
        ss = g[g['partition'] == 'seen']['macro_cov']
        gate_rows.append(dict(family=f, seen_macro=float(ss.mean()), unseen_macro=float(su.mean()),
                              unseen_macro_sd=float(su.std() if len(su) > 1 else 0.0),
                              shortfall=float(0.90 - su.mean()), premise_holds=bool(su.mean() < 0.85),
                              sanity_seen_ok=bool(abs(ss.mean() - 0.90) <= 0.03)))
    gate = pd.DataFrame(gate_rows)
    n_hold = int(gate['premise_holds'].sum())
    need = 2 if len(gate) >= 2 else 1
    verdict = ('PREMISE SUPPORTED' if n_hold >= need else 'PREMISE NOT SUPPORTED') + \
        ': standard calibration falls short by more than 5 points on unseen domains in %d of %d classifier families ' \
        '(rule: at least %d)' % (n_hold, len(gate), need)
    gate.to_csv(os.path.join(OUT, 'phase1_gate.csv'), index=False)
    log('GATE: ' + verdict)
    log(gate.to_string())

    # bootstrap CIs, per-species and per-domain coverage for the ensembles at alpha = 0.10
    ci_rows, sp_rows, dom_rows = [], [], []
    for f, fam in fams.items():
        d = ensemble(fam)
        S = sets_marginal(d['tp'], d['vp'], d['vy'], 0.10)
        for pname, m in [('seen', seen_m), ('unseen', unseen_m)]:
            mi, ma = boot_ci(S, y, m)
            r = set_metrics(S, y, m)
            ci_rows.append(dict(family=f, partition=pname, micro=r['micro_cov'], micro_lo=mi[0], micro_hi=mi[1],
                                macro=r['macro_cov'], macro_lo=ma[0], macro_hi=ma[1], set_size=r['set_size'], n=r['n']))
        for c in range(9):
            for pname, m in [('seen', seen_m), ('unseen', unseen_m)]:
                mm = m & (y == c)
                if mm.any():
                    sp_rows.append(dict(family=f, species=SP_SHORT[c + 1], partition=pname, n=int(mm.sum()),
                                        coverage=float(S[mm, c].mean()), set_size=float(S[mm].sum(1).mean()),
                                        top1_acc=float((d['tp'][mm].argmax(1) == c).mean())))
        for dd in range(1, 6):
            mm = unseen_m & (dom == dd)
            if mm.any():
                dom_rows.append(dict(family=f, domain='D%d' % dd, n=int(mm.sum()),
                                     coverage=float(S[mm, y[mm]].mean()), set_size=float(S[mm].sum(1).mean())))
    ci, sp, dm = pd.DataFrame(ci_rows), pd.DataFrame(sp_rows), pd.DataFrame(dom_rows)
    ci.to_csv(os.path.join(OUT, 'phase1_ci_ensemble_alpha010.csv'), index=False)
    sp.to_csv(os.path.join(OUT, 'phase1_per_species_alpha010.csv'), index=False)
    dm.to_csv(os.path.join(OUT, 'phase1_per_domain_unseen_alpha010.csv'), index=False)

    # figures
    fig, axes = plt.subplots(1, len(fams), figsize=(5 * len(fams), 4), squeeze=False)
    grid = np.linspace(0.02, 0.5, 25)
    for ax, (f, fam) in zip(axes[0], fams.items()):
        d = ensemble(fam)
        for pname, m, col in [('seen', seen_m, 'tab:blue'), ('unseen', unseen_m, 'tab:red')]:
            ys = [set_metrics(sets_marginal(d['tp'], d['vp'], d['vy'], a), y, m)['macro_cov'] for a in grid]
            ax.plot(1 - grid, ys, label=pname, color=col)
        ax.plot([0.5, 1], [0.5, 1], 'k--', lw=0.8)
        ax.set_title(FAMILY_LABEL.get(f, f), fontsize=9)
        ax.set_xlabel('target coverage (1 - alpha)')
        ax.set_ylabel('species-averaged coverage')
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, 'coverage_vs_target.png'), dpi=150)
    plt.close()

    # ---- Phase 1b previews (ensembles, alpha = 0.10)
    prev = {}
    # R5 temperature scaling
    r5 = []
    for f, fam in fams.items():
        d = ensemble(fam)
        T = fit_temperature(d['vp'], d['vy'])
        vp2, tp2 = softmax_T(d['vp'], T), softmax_T(d['tp'], T)
        for label, (a, b) in [('raw', (d['vp'], d['tp'])), ('temp-scaled', (vp2, tp2))]:
            S = sets_marginal(b, a, d['vy'], 0.10)
            for pname, m in [('seen', seen_m), ('unseen', unseen_m)]:
                r = set_metrics(S, y, m)
                r5.append(dict(family=f, calibration=label, T=round(T, 2), partition=pname, macro_cov=r['macro_cov'],
                               micro_cov=r['micro_cov'], set_size=r['set_size']))
    prev['R5 temperature scaling before conformal'] = pd.DataFrame(r5)

    # R3 few-shot target calibration: k labelled unseen-domain clips per species recalibrate the threshold
    r3 = []
    rng = np.random.default_rng(1)
    for f, fam in fams.items():
        d = ensemble(fam)
        idx_u = np.where(unseen_m)[0]
        yu, pu = y[idx_u], d['tp'][idx_u]
        for k in [5, 10, 20, 50]:
            elig = [c for c in range(9) if (yu == c).sum() >= k + 10]
            if not elig:
                continue
            macros, micros, sizes = [], [], []
            for _ in range(50):
                cal_idx = np.concatenate([rng.choice(np.where(yu == c)[0], k, replace=False) for c in elig])
                q = lac_q(1 - pu[cal_idx, yu[cal_idx]], 0.10)
                rest = np.ones(len(yu), dtype=bool)
                rest[cal_idx] = False
                rest &= np.isin(yu, elig)
                S = (1 - pu) <= q
                cov = S[np.arange(len(yu)), yu]
                per = [cov[rest & (yu == c)].mean() for c in elig]
                macros.append(np.mean(per))
                micros.append(cov[rest].mean())
                sizes.append(S[rest].sum(1).mean())
            r3.append(dict(family=f, k_per_species=k, species_used=len(elig), macro_cov=float(np.mean(macros)),
                           macro_cov_sd=float(np.std(macros)), micro_cov=float(np.mean(micros)),
                           set_size=float(np.mean(sizes))))
    prev['R3 few-shot target calibration (k labelled unseen-domain clips per species)'] = pd.DataFrame(r3)

    # R2 shift-weighted conformal (heuristic): weight calibration clips by how much they look like unseen-domain clips
    r2 = []
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler
    for f, fam in fams.items():
        d = ensemble(fam)
        if d.get('vfeat') is not None and d.get('tfeat') is not None:
            Xc, Xu = np.nan_to_num(d['vfeat'], nan=0.0), np.nan_to_num(d['tfeat'][unseen_m], nan=0.0)
            kind = 'hand-built features'
        else:
            Xc, Xu = np.log(np.clip(d['vp'], 1e-6, 1)), np.log(np.clip(d['tp'][unseen_m], 1e-6, 1))
            kind = 'model log-probabilities'
        X = np.vstack([Xc, Xu])
        lab = np.r_[np.zeros(len(Xc)), np.ones(len(Xu))]
        X = StandardScaler().fit_transform(X)
        clf = LogisticRegression(C=0.5, max_iter=500, class_weight='balanced')
        pr = cross_val_predict(clf, X, lab, cv=3, method='predict_proba')[:, 1]
        auc = roc_auc_score(lab, pr)
        clf.fit(X, lab)
        pc = clf.predict_proba(X[:len(Xc)])[:, 1]
        w = np.clip(pc / np.clip(1 - pc, 1e-3, None), 0.01, 100)
        s_cal = 1 - d['vp'][np.arange(len(d['vy'])), d['vy']]
        q_w = weighted_q(s_cal, w, 0.90)
        S = (1 - d['tp']) <= q_w
        r = set_metrics(S, y, unseen_m)
        r2.append(dict(family=f, shift_features=kind, seen_vs_unseen_AUC=float(auc), macro_cov=r['macro_cov'],
                       micro_cov=r['micro_cov'], set_size=r['set_size'], eff_sample_size=float(w.sum() ** 2 / (w ** 2).sum())))
    prev['R2 shift-weighted calibration (heuristic)'] = pd.DataFrame(r2)

    # R6 taxonomy-level sets
    r6 = []
    for level, mapping in [('species (9)', None), ('sibling group (8)', GROUP), ('genus (3)', GENUS)]:
        for f, fam in fams.items():
            d = ensemble(fam)
            if mapping is None:
                M, names = np.eye(9), [SP_SHORT[i] for i in range(1, 10)]
            else:
                M, names = agg_matrix(mapping)
            vp, tp = d['vp'] @ M, d['tp'] @ M
            vy, ty = (M[d['vy']].argmax(1)), (M[y].argmax(1))
            G = M.shape[1]
            S = sets_marginal(tp, vp, vy, 0.10)
            for pname, m in [('seen', seen_m), ('unseen', unseen_m)]:
                r = set_metrics(S, ty, m, n_classes=G)
                acc = float(np.mean([(tp[m & (ty == c)].argmax(1) == c).mean() for c in range(G) if (m & (ty == c)).any()]))
                r6.append(dict(level=level, family=f, partition=pname, macro_cov=r['macro_cov'], micro_cov=r['micro_cov'],
                               set_size=r['set_size'], balanced_acc=acc))
    prev['R6 conformal sets at coarser taxonomy levels'] = pd.DataFrame(r6)
    for k, v in prev.items():
        v.to_csv(os.path.join(OUT, 'preview_' + k.split()[0] + '.csv'), index=False)

    # ---------------------------------------------------------------- report
    def md(df, floatfmt='.3f'):
        return df.to_markdown(index=False, floatfmt=floatfmt)
    top = ci.copy()
    top['family'] = top['family'].map(lambda x: FAMILY_LABEL.get(x, x))
    lines = ['# Phase 1: does standard split-conformal calibration hold on unseen domains?', '',
             '**Gate verdict:** ' + verdict, '',
             'Setup: LAC score (1 - p_true), threshold from the Validation split, target coverage 90%. "seen" and "unseen" '
             'test clips follow the official definition (each species has one held-out domain). Species-averaged ("macro") '
             'coverage is the gate metric: below 0.85 means the promise falls short by more than 5 points.', '',
             '## Gate table (alpha = 0.10, marginal LAC, mean over seeds)', '', md(gate), '',
             '## Ensemble coverage with bootstrap 95% CIs (alpha = 0.10)', '', md(top), '',
             '## Seed-averaged results for all targets', '', md(agg.reset_index().round(3)), '',
             '## Per-species coverage (ensemble, alpha = 0.10)', '', md(sp), '',
             '## Per-domain coverage on unseen clips (ensemble, alpha = 0.10)', '', md(dm), '',
             '![coverage](figs/coverage_vs_target.png)', '',
             '# Phase 1b repair previews (no retraining, ensembles, alpha = 0.10, PREVIEW ONLY)', '']
    for k, v in prev.items():
        lines += ['## ' + k, '', md(v), '']
    lines += ['## Caveats', '',
              '- The Validation split was also used to choose the released M1 checkpoints, so M1 calibration is slightly optimistic.',
              '- Unseen-clip counts per species are small for some species; their coverage estimates are noisy (see n).',
              '- Sets can reach coverage trivially by being large; always read coverage together with set size.',
              '- R2 is a heuristic weighted-quantile; R3 assumes labelled clips from the new domain are available.']
    with open(os.path.join(OUT, 'phase1_report.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    log('wrote phase1_report.md')


if __name__ == '__main__':
    main()
