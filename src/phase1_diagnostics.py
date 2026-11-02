"""Phase 1 follow-up diagnostics on the saved predictions: where do unseen clips go, can the embedding tell sensors apart,
and is the sibling pair separable inside one sensor? Writes results/phase1/diagnostics.md. Evidence for the Phase 2 design."""
import glob
import os

import numpy as np

from common import RES, SP_SHORT, ensure_dirs

P = os.path.join(RES, 'phase1', 'preds')
OUT = os.path.join(RES, 'phase1', 'diagnostics.md')
lines = []


def say(s=''):
    print(s, flush=True)
    lines.append(s)


def table(rows, head):
    say('| ' + ' | '.join(head) + ' |')
    say('|' + '|'.join(['---'] * len(head)) + '|')
    for r in rows:
        say('| ' + ' | '.join(str(c) for c in r) + ' |')
    say()


def load(prefix, split):
    z = np.load(os.path.join(P, '%s_%s.npz' % (prefix, split)), allow_pickle=True)
    return {k: z[k] for k in z.files}


def pred_mix(probs, mask, k=3):
    pr = probs[mask].argmax(1)
    cnt = np.bincount(pr, minlength=9) / max(mask.sum(), 1)
    top = np.argsort(-cnt)[:k]
    return ', '.join('%s %.0f%%' % (SP_SHORT[t + 1], 100 * cnt[t]) for t in top if cnt[t] > 0)


def main():
    ensure_dirs(os.path.dirname(OUT))
    m3 = load('m3_seed0', 'test')
    m3v = load('m3_seed0', 'validation')
    m1 = [load(os.path.basename(f)[:-9], 'test') for f in sorted(glob.glob(os.path.join(P, 'm1_seed*_test.npz')))]
    say('# Phase 1 diagnostics (saved predictions, no retraining)')
    say()
    say('## 1. Where do the unseen clips go? (top predicted species, test split)')
    say()
    y, d, part = m3['y'], m3['domain'], m3['partition']
    rows = []
    for c in range(9):
        for dom in sorted(set(d[(y == c) & (part == 1)])):
            m = (y == c) & (d == dom) & (part == 1)
            rec3 = (m3['probs'][m].argmax(1) == c).mean()
            m1p = np.mean([p['probs'] for p in m1], axis=0) if m1 else None
            rec1 = (m1p[m].argmax(1) == c).mean() if m1 else float('nan')
            rows.append((SP_SHORT[c + 1], 'D%d' % dom, int(m.sum()), '%.3f' % rec3, pred_mix(m3['probs'], m),
                         '%.3f' % rec1, pred_mix(m1p, m) if m1 else ''))
    table(rows, ['species', 'unseen domain', 'n', 'M3 recall', 'M3 predicted as', 'M1 recall', 'M1 predicted as'])

    say('## 2. Seen test clips: per-species recall (is the sibling pair separable inside the optical-like sensor?)')
    say()
    rows = []
    for c in range(9):
        m = (y == c) & (part == 0)
        if m.sum():
            rows.append((SP_SHORT[c + 1], int(m.sum()), '%.3f' % (m3['probs'][m].argmax(1) == c).mean(), pred_mix(m3['probs'], m)))
    table(rows, ['species', 'n seen', 'M3 recall', 'M3 predicted as'])

    say('## 3. Can the AST embedding tell the sensor apart? (logistic regression, fit on Validation, scored on Test)')
    say()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(m3v['feat'])
    ytr, yte = (m3v['domain'] == 5).astype(int), (d == 5).astype(int)
    clf = LogisticRegression(C=0.5, max_iter=300, class_weight='balanced').fit(sc.transform(m3v['feat']), ytr)
    pr = clf.predict_proba(sc.transform(m3['feat']))[:, 1]
    say('D5 vs microphone domains: balanced accuracy %.3f, AUC %.3f (test n=%d, microphone clips=%d).'
        % (balanced_accuracy_score(yte, pr > 0.5), roc_auc_score(yte, pr), len(yte), int((yte == 0).sum())))
    say()

    say('## 4. Ceiling check inside one microphone domain (D1 test clips, 5-fold CV, optimistic: clips of one recording leak)')
    say()
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    m = (d == 1) & np.isin(y, [2, 3, 4])      # An.gam=3, An.ara=4 (0-based), Cx.qui=2
    X, yy = sc.transform(m3['feat'][m]), y[m]
    pred = cross_val_predict(LogisticRegression(C=0.5, max_iter=300, class_weight='balanced'), X, yy,
                             cv=StratifiedKFold(5, shuffle=True, random_state=0))
    rec = {SP_SHORT[c + 1]: float((pred[yy == c] == c).mean()) for c in np.unique(yy)}
    say('3-class (Cx.qui, An.gam, An.ara) inside D1: balanced accuracy %.3f; recall %s'
        % (balanced_accuracy_score(yy, pred), {k: round(v, 3) for k, v in rec.items()}))
    pair = np.isin(yy, [3, 4])
    p2 = cross_val_predict(LogisticRegression(C=0.5, max_iter=300, class_weight='balanced'), X[pair], yy[pair],
                           cv=StratifiedKFold(5, shuffle=True, random_state=0))
    say('An.gam vs An.ara only, inside D1: balanced accuracy %.3f' % balanced_accuracy_score(yy[pair], p2))
    say()
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
