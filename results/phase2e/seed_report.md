# Phase 2e: 5 seeds, AST padded input (single-seed models, mean ± sd over seeds)

| config | test unseen BA | test seen BA | fold accuracy | bridge folds | paired fold diff vs lin_clean |
|---|---|---|---|---|---|
| lin_clean | 0.259 ± 0.001 | 0.716 ± 0.001 | 0.414 ± 0.006 | 0.134 ± 0.007 | +0.000 ± 0.000 |
| lin_aug | 0.261 ± 0.003 | 0.727 ± 0.001 | 0.439 ± 0.002 | 0.184 ± 0.005 | +0.025 ± 0.007 |
| mlp_aug | 0.261 ± 0.008 | 0.756 ± 0.001 | 0.447 ± 0.009 | 0.118 ± 0.014 | +0.033 ± 0.012 |
