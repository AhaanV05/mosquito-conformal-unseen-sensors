"""Figure for Phase 4: whole-test coverage vs mean set size, global rules vs sensor-conditional (Mondrian) rules."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
rows = json.load(open(os.path.join(R, 'phase4', 'mondrian_results.json')))
fam = {}
for r in rows:
    fam.setdefault(r['rule'], []).append((r['alpha'], r['size'], r['overall_cov'], r['unseen_cov'], r['unseen_size']))
show = ['global C0 (validation)', 'global C1 (pooled pseudo-unseen)', 'Mondrian by sensor (detected) mic=C1',
        'Mondrian by sensor (detected) mic=C2b', 'Mondrian by sensor (detected) mic=C2']
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for name in show:
    pts = sorted(fam[name])
    ax[0].plot([p[1] for p in pts], [p[2] for p in pts], marker='o', label=name.replace('Mondrian by sensor', 'Mondrian'))
    ax[1].plot([p[4] for p in pts], [p[3] for p in pts], marker='o', label=name.replace('Mondrian by sensor', 'Mondrian'))
for a, t in zip(ax, ('Whole dev test (seen + unseen)', 'Unseen groups only')):
    a.axhline(0.9, color='grey', lw=0.8, ls=':'); a.set_xlabel('mean set size (species, of 9)'); a.set_ylabel('coverage'); a.set_title(t); a.set_ylim(0, 1.02)
ax[0].legend(fontsize=6.5, loc='lower right')
fig.tight_layout(); fig.savefig(os.path.join(R, 'figs', 'phase4_mondrian_frontier.png'), dpi=140)
print('ok')
