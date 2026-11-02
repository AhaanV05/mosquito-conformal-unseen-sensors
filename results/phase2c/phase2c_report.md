# Phase 2c results (dev test, 9 official unseen groups; classifier = mlp_aug, ensemble of 9 fold models x 2 seeds)

Unseen BA: ensemble 0.264, full model 0.270. Pseudo-unseen calibration groups: {'Ae.aeg-mic': 131, 'Ae.alb-mic': 90, 'Cx.pip-mic': 119, 'An.min-D1': 200, 'An.min-D3': 200, 'An.min-D4': 51, 'An.ste-D1': 200, 'An.ste-D2': 200, 'An.ste-D3': 200}.

| alpha | rule | level | q | unseen macro cov | seen macro cov | set size | vacuous | groups >= 0.80 | lifted macro cov | lifted size | coarse share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | C0 validation | species | 0.992 | 0.383 | 0.952 | 2.01 | 0.00 | 2/9 | 0.400 | 2.32 | 0.05 |
| 0.05 | C1 pooled OOF | species | 1.000 | 0.984 | 1.000 | 8.41 | 0.60 | 9/9 | 0.998 | 8.94 | 0.95 |
| 0.05 | C2 max group | species | 1.000 | 1.000 | 1.000 | 9.00 | 1.00 | 9/9 | 1.000 | 9.00 | 1.00 |
| 0.05 | C2b 2nd-largest group | species | 1.000 | 0.995 | 1.000 | 8.78 | 0.81 | 9/9 | 0.999 | 8.98 | 0.98 |
| 0.05 | G0 validation | genus | 0.969 | 0.785 | 0.956 | 1.42 | 0.05 | 6/9 |  |  |  |
| 0.05 | G2 max group | genus | 1.000 | 1.000 | 1.000 | 2.98 | 0.98 | 9/9 |  |  |  |
| 0.10 | C0 validation | species | 0.945 | 0.325 | 0.900 | 1.54 | 0.00 | 2/9 | 0.330 | 1.69 | 0.03 |
| 0.10 | C1 pooled OOF | species | 1.000 | 0.934 | 1.000 | 6.81 | 0.13 | 8/9 | 0.989 | 8.60 | 0.77 |
| 0.10 | C2 max group | species | 1.000 | 1.000 | 1.000 | 9.00 | 1.00 | 9/9 | 1.000 | 9.00 | 1.00 |
| 0.10 | C2b 2nd-largest group | species | 1.000 | 0.968 | 1.000 | 8.01 | 0.44 | 9/9 | 0.998 | 8.89 | 0.92 |
| 0.10 | G0 validation | genus | 0.808 | 0.732 | 0.914 | 1.19 | 0.01 | 4/9 |  |  |  |
| 0.10 | G2 max group | genus | 1.000 | 1.000 | 1.000 | 2.95 | 0.95 | 9/9 |  |  |  |
| 0.20 | C0 validation | species | 0.676 | 0.282 | 0.808 | 1.13 | 0.00 | 1/9 | 0.282 | 1.19 | 0.01 |
| 0.20 | C1 pooled OOF | species | 1.000 | 0.730 | 0.998 | 4.57 | 0.01 | 3/9 | 0.818 | 6.85 | 0.30 |
| 0.20 | C2 max group | species | 1.000 | 0.999 | 1.000 | 8.92 | 0.92 | 9/9 | 1.000 | 8.99 | 0.99 |
| 0.20 | C2b 2nd-largest group | species | 1.000 | 0.926 | 1.000 | 6.61 | 0.10 | 8/9 | 0.985 | 8.53 | 0.74 |
| 0.20 | G0 validation | genus | 0.378 | 0.632 | 0.826 | 0.91 | 0.00 | 4/9 |  |  |  |
| 0.20 | G2 max group | genus | 1.000 | 0.994 | 1.000 | 2.81 | 0.83 | 9/9 |  |  |  |

## Per-group unseen coverage at alpha = 0.10

- C0 validation (species): Ae.aeg 0.08, Ae.alb 0.65, Cx.qui 0.14, An.gam 0.12, An.ara 0.01, An.dir 0.00, Cx.pip 0.94, An.min 0.83, An.ste 0.15
- C1 pooled OOF (species): Ae.aeg 0.96, Ae.alb 1.00, Cx.qui 0.98, An.gam 0.84, An.ara 0.67, An.dir 0.97, Cx.pip 1.00, An.min 1.00, An.ste 0.97
- C2 max group (species): Ae.aeg 1.00, Ae.alb 1.00, Cx.qui 1.00, An.gam 1.00, An.ara 1.00, An.dir 1.00, Cx.pip 1.00, An.min 1.00, An.ste 1.00
- C2b 2nd-largest group (species): Ae.aeg 0.98, Ae.alb 1.00, Cx.qui 1.00, An.gam 0.92, An.ara 0.86, An.dir 0.97, Cx.pip 1.00, An.min 1.00, An.ste 0.97
- G0 validation (genus): Ae.aeg 0.99, Ae.alb 0.65, Cx.qui 0.10, An.gam 0.77, An.ara 0.43, An.dir 0.93, Cx.pip 0.96, An.min 1.00, An.ste 0.76
- G2 max group (genus): Ae.aeg 1.00, Ae.alb 1.00, Cx.qui 1.00, An.gam 1.00, An.ara 1.00, An.dir 1.00, Cx.pip 1.00, An.min 1.00, An.ste 1.00
