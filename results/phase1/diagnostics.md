# Phase 1 diagnostics (saved predictions, no retraining)

## 1. Where do the unseen clips go? (top predicted species, test split)

| species | unseen domain | n | M3 recall | M3 predicted as | M1 recall | M1 predicted as |
|---|---|---|---|---|---|---|
| Ae.aeg | D3 | 192 | 0.115 | Ae.alb 69%, Ae.aeg 11%, An.min 10% | 0.000 | An.ste 73%, An.min 25%, Ae.alb 2% |
| Ae.alb | D2 | 419 | 0.702 | Ae.alb 70%, An.min 13%, An.dir 8% | 0.014 | An.ste 47%, Ae.aeg 28%, An.dir 22% |
| Cx.qui | D1 | 672 | 0.109 | An.min 40%, An.dir 27%, Ae.aeg 13% | 0.000 | An.ste 49%, An.dir 29%, Ae.aeg 15% |
| An.gam | D1 | 818 | 0.022 | An.min 39%, Ae.alb 33%, An.ste 14% | 0.000 | An.ste 88%, Ae.aeg 9%, An.min 2% |
| An.ara | D1 | 1820 | 0.001 | An.min 31%, Ae.aeg 31%, Cx.pip 13% | 0.000 | An.min 51%, Ae.aeg 38%, An.ste 6% |
| An.dir | D4 | 40 | 0.000 | An.min 68%, Ae.aeg 18%, An.ste 10% | 0.000 | An.min 55%, An.ste 45% |
| Cx.pip | D3 | 68 | 0.912 | Cx.pip 91%, Ae.aeg 3%, An.ste 3% | 0.824 | Cx.pip 82%, An.min 7%, Ae.aeg 4% |
| An.min | D2 | 99 | 0.677 | An.min 68%, An.ste 30%, Cx.pip 1% | 0.263 | An.ste 72%, An.min 26%, Cx.pip 2% |
| An.ste | D4 | 74 | 0.054 | An.min 50%, Cx.qui 22%, Ae.aeg 16% | 0.500 | An.ste 50%, Ae.aeg 18%, An.min 16% |

## 2. Seen test clips: per-species recall (is the sibling pair separable inside the optical-like sensor?)

| species | n seen | M3 recall | M3 predicted as |
|---|---|---|---|
| Ae.aeg | 7967 | 0.654 | Ae.aeg 65%, Cx.pip 11%, Ae.alb 8% |
| Ae.alb | 1433 | 0.783 | Ae.alb 78%, Ae.aeg 8%, Cx.pip 5% |
| Cx.qui | 6534 | 0.852 | Cx.qui 85%, Ae.aeg 8%, An.ara 3% |
| An.gam | 3882 | 0.675 | An.gam 67%, An.ara 22%, Ae.alb 5% |
| An.ara | 292 | 0.647 | An.ara 65%, An.gam 21%, Ae.aeg 4% |
| Cx.pip | 2907 | 0.764 | Cx.pip 76%, Ae.aeg 10%, An.ara 5% |

## 3. Can the AST embedding tell the sensor apart? (logistic regression, fit on Validation, scored on Test)

D5 vs microphone domains: balanced accuracy 0.989, AUC 0.999 (test n=27217, microphone clips=4238).

## 4. Ceiling check inside one microphone domain (D1 test clips, 5-fold CV, optimistic: clips of one recording leak)

3-class (Cx.qui, An.gam, An.ara) inside D1: balanced accuracy 0.954; recall {'Cx.qui': 0.951, 'An.gam': 0.933, 'An.ara': 0.977}
An.gam vs An.ara only, inside D1: balanced accuracy 0.962

