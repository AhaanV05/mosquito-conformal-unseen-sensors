# Phase 2d: Bird-MAE and late fusion (dev test, 9 official unseen groups)

| model | BA unseen (ens / full) | BA seen (full) | fold accuracy (macro) | bridge folds | genus acc unseen |
|---|---|---|---|---|---|
| AST | 0.264 / 0.270 | 0.765 | 0.456 | 0.139 | 0.438 |
| BirdMAE | 0.221 / 0.216 | 0.811 | 0.309 | 0.003 | 0.469 |
| Fused | 0.240 / 0.256 | 0.810 | 0.390 | 0.036 | 0.471 |

| model | alpha | rule | level | q | unseen macro cov | set size | vacuous | groups >= 0.80 |
|---|---|---|---|---|---|---|---|---|
| AST | 0.05 | C0 validation | species | 0.992 | 0.383 | 2.01 | 0.00 | 2/9 |
| AST | 0.05 | C1 pooled OOF | species | 1.000 | 0.984 | 8.41 | 0.60 | 9/9 |
| AST | 0.05 | C2 max group | species | 1.000 | 1.000 | 9.00 | 1.00 | 9/9 |
| AST | 0.05 | C2b 2nd-largest | species | 1.000 | 0.995 | 8.78 | 0.81 | 9/9 |
| AST | 0.05 | G0 validation | genus | 0.969 | 0.785 | 1.42 | 0.05 | 6/9 |
| AST | 0.05 | G2 max group | genus | 1.000 | 1.000 | 2.98 | 0.98 | 9/9 |
| AST | 0.10 | C0 validation | species | 0.945 | 0.325 | 1.54 | 0.00 | 2/9 |
| AST | 0.10 | C1 pooled OOF | species | 1.000 | 0.934 | 6.81 | 0.13 | 8/9 |
| AST | 0.10 | C2 max group | species | 1.000 | 1.000 | 9.00 | 1.00 | 9/9 |
| AST | 0.10 | C2b 2nd-largest | species | 1.000 | 0.968 | 8.01 | 0.44 | 9/9 |
| AST | 0.10 | G0 validation | genus | 0.808 | 0.732 | 1.19 | 0.01 | 4/9 |
| AST | 0.10 | G2 max group | genus | 1.000 | 1.000 | 2.95 | 0.95 | 9/9 |
| AST | 0.20 | C0 validation | species | 0.676 | 0.282 | 1.13 | 0.00 | 1/9 |
| AST | 0.20 | C1 pooled OOF | species | 1.000 | 0.730 | 4.57 | 0.01 | 3/9 |
| AST | 0.20 | C2 max group | species | 1.000 | 0.999 | 8.92 | 0.92 | 9/9 |
| AST | 0.20 | C2b 2nd-largest | species | 1.000 | 0.926 | 6.61 | 0.10 | 8/9 |
| AST | 0.20 | G0 validation | genus | 0.378 | 0.632 | 0.91 | 0.00 | 4/9 |
| AST | 0.20 | G2 max group | genus | 1.000 | 0.994 | 2.81 | 0.83 | 9/9 |
| BirdMAE | 0.05 | C0 validation | species | 0.915 | 0.256 | 1.55 | 0.00 | 1/9 |
| BirdMAE | 0.05 | C1 pooled OOF | species | 1.000 | 0.958 | 8.58 | 0.67 | 8/9 |
| BirdMAE | 0.05 | C2 max group | species | 1.000 | 1.000 | 9.00 | 1.00 | 9/9 |
| BirdMAE | 0.05 | C2b 2nd-largest | species | 1.000 | 1.000 | 9.00 | 1.00 | 9/9 |
| BirdMAE | 0.05 | G0 validation | genus | 0.838 | 0.722 | 1.24 | 0.02 | 6/9 |
| BirdMAE | 0.05 | G2 max group | genus | 1.000 | 0.999 | 2.99 | 0.99 | 9/9 |
| BirdMAE | 0.10 | C0 validation | species | 0.791 | 0.236 | 1.25 | 0.00 | 1/9 |
| BirdMAE | 0.10 | C1 pooled OOF | species | 1.000 | 0.897 | 7.80 | 0.27 | 7/9 |
| BirdMAE | 0.10 | C2 max group | species | 1.000 | 1.000 | 9.00 | 1.00 | 9/9 |
| BirdMAE | 0.10 | C2b 2nd-largest | species | 1.000 | 0.996 | 8.97 | 0.97 | 9/9 |
| BirdMAE | 0.10 | G0 validation | genus | 0.640 | 0.695 | 1.07 | 0.00 | 6/9 |
| BirdMAE | 0.10 | G2 max group | genus | 1.000 | 0.998 | 2.97 | 0.97 | 9/9 |
| BirdMAE | 0.20 | C0 validation | species | 0.544 | 0.214 | 0.98 | 0.00 | 1/9 |
| BirdMAE | 0.20 | C1 pooled OOF | species | 1.000 | 0.674 | 5.78 | 0.01 | 4/9 |
| BirdMAE | 0.20 | C2 max group | species | 1.000 | 0.983 | 8.83 | 0.85 | 9/9 |
| BirdMAE | 0.20 | C2b 2nd-largest | species | 1.000 | 0.970 | 8.70 | 0.75 | 9/9 |
| BirdMAE | 0.20 | G0 validation | genus | 0.323 | 0.653 | 0.87 | 0.00 | 5/9 |
| BirdMAE | 0.20 | G2 max group | genus | 1.000 | 0.994 | 2.95 | 0.95 | 9/9 |
| Fused | 0.05 | C0 validation | species | 0.900 | 0.344 | 1.63 | 0.00 | 2/9 |
| Fused | 0.05 | C1 pooled OOF | species | 1.000 | 0.976 | 8.40 | 0.57 | 9/9 |
| Fused | 0.05 | C2 max group | species | 1.000 | 1.000 | 9.00 | 1.00 | 9/9 |
| Fused | 0.05 | C2b 2nd-largest | species | 1.000 | 0.999 | 8.97 | 0.97 | 9/9 |
| Fused | 0.05 | G0 validation | genus | 0.784 | 0.785 | 1.27 | 0.01 | 6/9 |
| Fused | 0.05 | G2 max group | genus | 1.000 | 1.000 | 2.99 | 0.99 | 9/9 |
| Fused | 0.10 | C0 validation | species | 0.762 | 0.313 | 1.31 | 0.00 | 1/9 |
| Fused | 0.10 | C1 pooled OOF | species | 1.000 | 0.917 | 6.84 | 0.10 | 8/9 |
| Fused | 0.10 | C2 max group | species | 1.000 | 0.998 | 8.95 | 0.95 | 9/9 |
| Fused | 0.10 | C2b 2nd-largest | species | 1.000 | 0.984 | 8.60 | 0.69 | 9/9 |
| Fused | 0.10 | G0 validation | genus | 0.595 | 0.728 | 1.08 | 0.00 | 6/9 |
| Fused | 0.10 | G2 max group | genus | 1.000 | 0.999 | 2.95 | 0.95 | 9/9 |
| Fused | 0.20 | C0 validation | species | 0.541 | 0.257 | 0.98 | 0.00 | 1/9 |
| Fused | 0.20 | C1 pooled OOF | species | 1.000 | 0.686 | 4.52 | 0.01 | 4/9 |
| Fused | 0.20 | C2 max group | species | 1.000 | 0.992 | 8.83 | 0.85 | 9/9 |
| Fused | 0.20 | C2b 2nd-largest | species | 1.000 | 0.937 | 7.29 | 0.18 | 8/9 |
| Fused | 0.20 | G0 validation | genus | 0.369 | 0.613 | 0.84 | 0.00 | 3/9 |
| Fused | 0.20 | G2 max group | genus | 1.000 | 0.977 | 2.75 | 0.76 | 9/9 |
