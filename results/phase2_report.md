# Phase 2 and 3 results (2026-10-28 to 2026-10-30)

All numbers are from scripts in `src/` (`p2a_*`, `p2b_*`, `p2c_*`, `p2d_*`, `p3_*`), logs in `results/logs/`, tables and figures in `results/phase2a|2c|2d|3` and `results/figs/`. Two seeds per configuration unless stated, so differences of a point or two are noise. "Unseen" is the official dev-test unseen split (4,202 clips, 9 groups). "Fold accuracy" is accuracy on the nine pseudo-unseen groups built from Training+Validation (3 leave-one-species'-microphones-out folds, 6 leave-one-microphone-domain-out folds; 1,391 clips). "Bridge folds" are the three leave-species-microphones-out folds, the closest match to the official worst case (D5 to microphone). Fold models never see the held-out group.

## Headline

No classifier reached the 0.32 unseen balanced-accuracy target (best 0.270). Calibration on held-out groups restores coverage on the unseen groups, but only by returning most of the 9 species. Every calibration rule lies on one coverage-versus-set-size curve; the classifier decides the curve and the rule only picks a point on it. Few-shot recalibration with labelled clips from the new site does not change this.

## 2a: input and head (AST, 4 layer taps)

| Config | Test seen | Test unseen | Fold accuracy | Bridge folds |
|---|---|---|---|---|
| Phase 1 padded 1024-frame input, linear head (reference) | 0.718 | **0.259** | **0.413** | **0.134** |
| Canonical tiled 128-frame input, linear, last layer | 0.736 | 0.210 | 0.351 | 0.106 |
| Canonical tiled input, 4-tap MLP + cosine head | 0.829 | 0.206 | 0.367 | 0.097 |
| + genus and complex loss | 0.827 | 0.198 | 0.364 | 0.093 |

The canonical tiled input lost to the padded input. The MLP head, layer taps and hierarchical loss help seen domains and not unseen ones. Gate (unseen balanced accuracy at least 0.32): **missed**.

## 2b: sensor compensation

| Config (AST) | Test unseen | Fold accuracy | Bridge folds |
|---|---|---|---|
| Nuisance-direction projection k=2, linear | 0.207 | 0.326 | 0.121 |
| Projection k=3, linear | 0.212 | 0.327 | 0.124 |
| Projection k=3, MLP | 0.224 | 0.372 | 0.101 |
| Per-species alignment loss, MLP | 0.206 | 0.379 | 0.087 |
| Padded input, linear, clean (reference) | 0.259 | 0.413 | 0.134 |
| Padded + noise-swap/EQ views, linear | 0.265 | 0.437 | **0.186** |
| Padded + noise-swap/EQ views, MLP | **0.270** | **0.456** | 0.139 |

Embedding-space tools (projection, alignment) gave no gain on the right folds. Audio-level augmentation (microphone background from a fold-specific bank that excludes the held-out group, random spectral tilt and band limits) gave +2.5 ± 0.7 points (linear head) and +3.3 ± 1.2 (MLP head) on fold accuracy over 5 seeds (Phase 2e table below; the two-seed figures in this table were optimistic), the only gain, and none on the official unseen test. Gate (+3 points on pseudo-unseen groups): borderline: met by the MLP head's mean (+3.3 ± 1.2), not by the linear head (+2.5 ± 0.7), and the MLP head is worse on the bridge folds.

## 2d: Bird-MAE and fusion

| Model | Unseen BA (ensemble / full) | Seen BA | Fold accuracy | Bridge folds |
|---|---|---|---|---|
| AST (2b recipe) | 0.264 / 0.270 | 0.765 | 0.456 | 0.139 |
| Bird-MAE | 0.221 / 0.216 | 0.811 | 0.309 | 0.003 |
| Late fusion (mean probabilities) | 0.240 / 0.256 | 0.810 | 0.390 | 0.036 |

Bird-MAE is better on seen domains and near zero on held-out bridge species. Fusion does worse than AST alone. Perch 2.0 is in its own section below.

## 2c: calibration (AST 2b model, alpha = 0.10, official unseen groups)

| Rule | Unseen coverage (species-avg) | Seen coverage | Set size (of 9) | Vacuous | Groups >= 0.80 |
|---|---|---|---|---|---|
| C0: Validation split (Phase 1 style) | 0.325 | 0.900 | 1.54 | 0% | 2/9 |
| C1: pooled pseudo-unseen | 0.934 | 1.000 | 6.81 | 13% | 8/9 |
| C2b: second-worst group | 0.968 | 1.000 | 8.01 | 44% | 9/9 |
| C2: worst group | 1.000 | 1.000 | 9.00 | 100% | 9/9 |
| G2: worst group, genus level | 1.000 | 1.000 | 2.95 of 3 | 95% | 9/9 |

At alpha = 0.20: C0 0.282 coverage; C1 0.730 (size 4.57); C2b 0.926 (size 6.61); C2 0.999 (size 8.92). Lifting sets to taxonomy nodes did not shrink them. Gate (macro coverage at least 0.85 with non-vacuous sets): coverage met by C1, C2b and C2, **non-vacuity missed**.

The worst-group threshold is the maximum possible score because at least one pseudo-unseen group contains clips where the model gives the true species almost no probability. No threshold can cover those without admitting everything.

## Coverage versus set size (`results/figs/phase2_coverage_vs_size.png`)

C1, C2b and C2 trace the same curve; C0 lies slightly below it. All sit above the "random subset of that size" line (coverage = size/9) by only about 0.1 to 0.2, for example 0.93 coverage at 6.8 of 9 species where a random set of that size covers 0.76.

## Phase 3: few-shot site recalibration (alpha = 0.10, 30 draws, mean over the 9 unseen groups)

| Shots per species | Species coverage | Species set size (of 9) | Vacuous | Genus coverage | Genus set size (of 3) |
|---|---|---|---|---|---|
| 0 (pooled pseudo-unseen rule) | 0.934 | 7.50 | 26% | 0.929 | 2.23 |
| 5 | 0.919 ± 0.032 | 7.40 | 26% | 0.925 | 2.21 |
| 10 | 0.900 ± 0.031 | 7.10 | 20% | 0.903 | 2.06 |
| 20 | 0.901 ± 0.022 | 7.12 | 19% | 0.900 | 2.04 |
| 50 | 0.910 ± 0.020 | 7.22 | 21% | 0.913 | 2.12 |

Labelled clips from the new site hold coverage near 0.90 but leave sets at about 7 of 9 species. (Set sizes here average over groups, so the k=0 row differs from the clip-averaged 6.81 above.) Gate (sets at least 20% smaller at equal coverage): **missed**. Unlabeled adaptation was not built.

## Phase 2e: 5 seeds for the key comparison (`results/phase2e/seed_report.md`, single-seed models, mean ± sd)

| Config (AST padded input) | Test unseen BA | Test seen BA | Fold accuracy | Bridge folds | Paired fold difference vs lin_clean |
|---|---|---|---|---|---|
| lin_clean | 0.259 ± 0.001 | 0.716 ± 0.001 | 0.414 ± 0.006 | 0.134 ± 0.007 | |
| lin_aug | 0.261 ± 0.003 | 0.727 ± 0.001 | 0.439 ± 0.002 | **0.184 ± 0.005** | +0.025 ± 0.007 |
| mlp_aug | 0.261 ± 0.008 | 0.756 ± 0.001 | **0.447 ± 0.009** | 0.118 ± 0.014 | +0.033 ± 0.012 |

This corrects the two-seed numbers above (mlp_aug 0.270 unseen, 0.456 fold, which were seed-lucky). Augmentation gives a consistent gain on held-out fold accuracy (all five paired differences positive for the linear head) and a larger gain on the bridge folds for the linear head. The MLP head helps seen domains (0.727 to 0.756) and the other folds, but is worse than the clean linear head on the bridge folds (0.118 against 0.134). Official unseen balanced accuracy does not move (0.259 to 0.261).

## Perch 2.0 (`results/phase2g/perch_report.md`, `perch_calibration.csv`)

Community PyTorch port of the official weights (maintainer reports float-round-off agreement with the TensorFlow model; we did not verify this ourselves). Same recipe as Bird-MAE: clip padded into one 5 s window at 32 kHz, 1,536-d embedding, noise-swap views with fold-specific noise banks, 9 fold models.

| Perch head (3 seeds, single-seed models) | Unseen BA | Seen BA | Fold accuracy | Bridge folds |
|---|---|---|---|---|
| Linear + augmentation | 0.261 ± 0.001 | 0.774 ± 0.002 | 0.387 ± 0.003 | 0.021 ± 0.001 |
| MLP + augmentation | 0.244 ± 0.005 | 0.778 ± 0.004 | 0.432 ± 0.008 | 0.022 ± 0.007 |

Ensembles: Perch 0.236 / 0.244 unseen (ensemble / full model), AST + Perch 0.236 / 0.246, AST 0.264 / 0.270. Calibration at alpha = 0.10 on the unseen groups: Perch C1 0.905 coverage with 7.4 species, C2b 0.966 with 8.5; AST + Bird-MAE + Perch C1 0.893 with 6.3, C2b 0.961 with 7.7. Perch gives higher seen-domain accuracy than AST (0.77 to 0.78 against 0.72 to 0.76) but is worse on the held-out bridge folds (0.02 against 0.13 to 0.18), so it relies on the sensor shortcut at least as much. The winner's 0.318 for Perch alone was not reproduced; they used alignment methods and a tuning protocol we cannot see in full.

## Bootstrap intervals (`results/phase2e/bootstrap_report.md`, clip sampling only, 1000 resamples)

| Rule | alpha | Unseen coverage | Set size (mean over groups, of 9) | Groups >= 0.80 |
|---|---|---|---|---|
| C0 validation | 0.10 | 0.325 [0.309, 0.341] | 1.49 | 1.8 |
| C1 pooled | 0.10 | 0.933 [0.907, 0.948] | 7.51 [7.19, 7.77] | 8.0 |
| C2b second-worst | 0.10 | 0.976 [0.949, 0.995] | 8.41 | 8.9 |
| C2 worst | 0.10 | 1.000 [0.999, 1.000] | 8.99 | 9.0 |
| C1 pooled | 0.20 | 0.732 [0.687, 0.783] | 5.30 | 3.5 |
| C2b second-worst | 0.20 | 0.932 [0.869, 0.964] | 7.53 | 8.0 |

These capture clip-sampling noise only; classifier training variance is in the 5-seed table.

## Phase 4: sensor-conditional (Mondrian) calibration on the whole dev test (`results/phase4/`, `results/figs/phase4_mondrian_frontier.png`)

A sensor detector (logistic regression on AST embeddings, fit on Validation; test balanced accuracy 0.981, AUC 0.999) routes clips. Optical (D5) branch: threshold from Validation D5 clips. Microphone branch: threshold from the nine pseudo-unseen groups (C1, C2b or C2). All unseen groups are microphone domains.

| Rule (alpha = 0.10) | Overall cov | Mean size | D5 cov / size | Unseen cov / size |
|---|---|---|---|---|
| Global C0 (validation) | 0.785 | 1.54 | 0.900 / 1.52 | 0.158 / 1.65 |
| Global C1 (pooled pseudo-unseen) | 0.973 | 6.81 | 1.000 / 6.60 | 0.822 / 7.94 |
| Mondrian, mic = C1 (detected) | 0.886 | 2.48 | 0.900 / 1.52 | 0.805 / 7.71 |
| Mondrian, mic = C2b (detected) | 0.901 | 2.58 | 0.900 / 1.52 | 0.903 / 8.30 |
| Mondrian, mic = C2 worst group (detected, pre-specified headline) | 0.912 | 2.64 | 0.900 / 1.52 | 0.980 / 8.72 |
| Mondrian, mic = C2 (oracle sensor) | 0.916 | 2.69 | 0.900 / 1.52 | 1.000 / 9.00 |

Matched whole-test coverage (linear interpolation on an alpha sweep, mean set size in species): at 0.85, global C1 2.46 against 2.27 to 2.35 (Mondrian); at 0.90, 3.37 against 2.57; at 0.95, 5.49 against 2.97 to 3.01. The standard Validation threshold does not reach 0.90 at any alpha tried. On the unseen groups alone the curves coincide (right panel of the figure): the gain is that sets are small on optical clips, not that unseen sets shrink.

## Caveats

- Most ablations are two seeds; the key augmentation comparison has 5 seeds (Phase 2e). Calibration intervals are clip-bootstrap only.
- The unseen test predictions were looked at in Phase 1 and the diagnostics, then used in reporting 2c/2d. Model selection used only pseudo-unseen folds, but the unseen test is not a pristine hold-out.
- The pseudo-unseen groups are small (51 to 200 clips) and only three of the nine mimic the official D5-to-microphone shift.
- The padded AST input was chosen after the tiled input lost, so that choice used fold and test results.
- Unlabeled adaptation, more augmentation draws and fine-tuning were not tried. The challenge winner reports 0.318 (Perch alone), above our best 0.27 and above our Perch result (0.26), and we cannot say why. The Perch weights come from a community port.
- Evaluation labels have not been released, so there is no clean confirmation set.
- Whole-test coverage is dominated by D5 (84% of test clips); the Phase 4 gain does not apply to the unseen groups alone.
- Phase 4 was designed after seeing the unseen-test failure pattern; the three microphone-branch rules were fixed before it was run and all are reported, with the worst-group rule as the pre-specified headline.
