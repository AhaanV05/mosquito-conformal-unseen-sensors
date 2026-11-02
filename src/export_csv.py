"""Export every Phase 2/3 result JSON to tidy CSV files in results/csv/ (like Phase 1). Safe to re-run."""
import json
import os

import pandas as pd

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
OUT = os.path.join(R, 'csv')
os.makedirs(OUT, exist_ok=True)


def load(*p):
    f = os.path.join(R, *p)
    return json.load(open(f)) if os.path.exists(f) else None


summ, folds = [], []
for stage, path, fam in [('2a', ('phase2a', 'p2a_head_results.json'), 'AST tiled 128-frame'),
                         ('2b', ('phase2a', 'p2b_results.json'), 'AST (2b compensation)'),
                         ('2b-aug', ('phase2a', 'p2b_aug_results.json'), 'AST padded + augmentation')]:
    d = load(*path)
    for name, r in (d or {}).items():
        summ.append(dict(stage=stage, family=fam, config=name, test_BA_seen=r.get('test_BA_seen'), test_BA_unseen=r.get('test_BA_unseen'),
                         genus_acc_unseen=r.get('genus_acc_unseen'), fold_macro_acc=r.get('fold_macro_acc'),
                         fold_bridge_acc=r.get('fold_bridge_acc'), seeds=2))
        for fn, v in r.get('folds', {}).items():
            folds.append(dict(stage=stage, config=name, fold=fn, accuracy=v[0], n_clips=v[1], genus_accuracy=v[2]))
pd.DataFrame(summ).to_csv(os.path.join(OUT, 'phase2_config_summary.csv'), index=False)
pd.DataFrame(folds).to_csv(os.path.join(OUT, 'phase2_fold_accuracy.csv'), index=False)

rows = []
for src, path in [('2c AST', ('phase2c', 'phase2c_results.json')), ('2d', ('phase2d', 'phase2d_results.json'))]:
    for r in load(*path) or []:
        base = {k: v for k, v in r.items() if k != 'per_group'}
        base.setdefault('model', 'AST')
        base['source'] = src
        for g, v in r['per_group'].items():
            base['cov_' + g] = v
        rows.append(base)
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'phase2_calibration_rules.csv'), index=False)

fs = load('phase3', 'fewshot_results.json')
if fs:
    pd.DataFrame(fs).to_csv(os.path.join(OUT, 'phase3_fewshot.csv'), index=False)
cv = load('phase3', 'curves.json')
if cv:
    pd.DataFrame([dict(rule=k, point=i, set_size=a, unseen_coverage=b) for k, v in cv.items() for i, (a, b) in enumerate(v)]).to_csv(
        os.path.join(OUT, 'phase2_coverage_vs_size_curves.csv'), index=False)
sd = load('phase2e', 'seed_results.json')
if sd:
    pd.DataFrame([dict(config=c, seed=int(s), **{k: v for k, v in m.items() if k != 'groups'}, **{'fold_' + g: a for g, a in m['groups'].items()})
                  for c, ss in sd.items() for s, m in ss.items()]).to_csv(os.path.join(OUT, 'phase2e_seed_results.csv'), index=False)
pp = load('phase2g', 'perch_seeds.json')
if pp:
    pd.DataFrame([dict(head=h, **{k: v for k, v in m.items() if k != 'groups'}, **{'fold_' + g: a for g, a in m['groups'].items()}) for h, ms in pp.items() for m in ms]).to_csv(os.path.join(OUT, 'phase2g_perch_seeds.csv'), index=False)
import shutil
for src in ('phase4/mondrian_results.csv', 'phase2g/perch_calibration.csv'):
    if os.path.exists(os.path.join(R, src)):
        shutil.copy(os.path.join(R, src), os.path.join(OUT, os.path.basename(src)))
print(sorted(os.listdir(OUT)))
