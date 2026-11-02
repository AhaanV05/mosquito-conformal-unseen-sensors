# Perch 2.0 (community PyTorch port) and fusion

## Perch alone, per seed (single-seed models; 3 seeds)

| head | unseen BA | seen BA | fold accuracy | bridge folds |
|---|---|---|---|---|
| linear_last + aug | 0.261 ± 0.001 | 0.774 ± 0.002 | 0.387 ± 0.003 | 0.021 ± 0.001 |
| mlp_mix + aug | 0.244 ± 0.005 | 0.778 ± 0.004 | 0.432 ± 0.008 | 0.022 ± 0.007 |

## Ensembles and fusion (mlp head; AST and Bird-MAE from 2c/2d)

| model | BA unseen (ens / full) | BA seen (full) | fold accuracy | bridge folds |
|---|---|---|---|---|
| Perch | 0.236 / 0.244 | 0.785 | 0.432 | 0.018 |
| AST | 0.264 / 0.270 | 0.765 | 0.456 | 0.139 |
| AST+Perch | 0.236 / 0.246 | 0.808 | 0.465 | 0.073 |
| AST+Bird+Perch | 0.231 / 0.238 | 0.829 | 0.418 | 0.019 |

## Calibration on the official unseen groups

| model | alpha | rule | unseen coverage | set size | vacuous | groups >= 0.80 |
|---|---|---|---|---|---|---|
| Perch | 0.10 | C0 validation | 0.300 | 1.41 | 0.00 | 1/9 |
| Perch | 0.10 | C1 pooled | 0.905 | 7.40 | 0.27 | 7/9 |
| Perch | 0.10 | C2b second-worst | 0.966 | 8.52 | 0.67 | 9/9 |
| Perch | 0.10 | C2 worst | 1.000 | 9.00 | 1.00 | 9/9 |
| Perch | 0.20 | C0 validation | 0.257 | 1.05 | 0.00 | 1/9 |
| Perch | 0.20 | C1 pooled | 0.778 | 5.42 | 0.02 | 6/9 |
| Perch | 0.20 | C2b second-worst | 0.901 | 7.31 | 0.25 | 7/9 |
| Perch | 0.20 | C2 worst | 0.998 | 8.98 | 0.98 | 9/9 |
| AST | 0.10 | C0 validation | 0.325 | 1.54 | 0.00 | 2/9 |
| AST | 0.10 | C1 pooled | 0.934 | 6.81 | 0.13 | 8/9 |
| AST | 0.10 | C2b second-worst | 0.968 | 8.01 | 0.44 | 9/9 |
| AST | 0.10 | C2 worst | 1.000 | 9.00 | 1.00 | 9/9 |
| AST | 0.20 | C0 validation | 0.282 | 1.13 | 0.00 | 1/9 |
| AST | 0.20 | C1 pooled | 0.730 | 4.57 | 0.01 | 3/9 |
| AST | 0.20 | C2b second-worst | 0.926 | 6.61 | 0.10 | 8/9 |
| AST | 0.20 | C2 worst | 0.999 | 8.92 | 0.92 | 9/9 |
| AST+Perch | 0.10 | C0 validation | 0.338 | 1.38 | 0.00 | 2/9 |
| AST+Perch | 0.10 | C1 pooled | 0.916 | 6.27 | 0.08 | 7/9 |
| AST+Perch | 0.10 | C2b second-worst | 0.944 | 6.99 | 0.18 | 8/9 |
| AST+Perch | 0.10 | C2 worst | 1.000 | 8.99 | 0.99 | 9/9 |
| AST+Perch | 0.20 | C0 validation | 0.254 | 0.99 | 0.00 | 1/9 |
| AST+Perch | 0.20 | C1 pooled | 0.742 | 4.31 | 0.01 | 5/9 |
| AST+Perch | 0.20 | C2b second-worst | 0.876 | 5.72 | 0.05 | 7/9 |
| AST+Perch | 0.20 | C2 worst | 0.996 | 8.82 | 0.85 | 9/9 |
| AST+Bird+Perch | 0.10 | C0 validation | 0.305 | 1.26 | 0.00 | 1/9 |
| AST+Bird+Perch | 0.10 | C1 pooled | 0.893 | 6.32 | 0.07 | 7/9 |
| AST+Bird+Perch | 0.10 | C2b second-worst | 0.961 | 7.68 | 0.31 | 8/9 |
| AST+Bird+Perch | 0.10 | C2 worst | 0.999 | 8.98 | 0.98 | 9/9 |
| AST+Bird+Perch | 0.20 | C0 validation | 0.220 | 0.92 | 0.00 | 1/9 |
| AST+Bird+Perch | 0.20 | C1 pooled | 0.709 | 4.30 | 0.01 | 5/9 |
| AST+Bird+Perch | 0.20 | C2b second-worst | 0.821 | 5.64 | 0.03 | 6/9 |
| AST+Bird+Perch | 0.20 | C2 worst | 0.993 | 8.79 | 0.83 | 9/9 |
