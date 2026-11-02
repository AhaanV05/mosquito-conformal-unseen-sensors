# Overnight report: Phase 0 and Phase 1 (2026-10-08)

Everything below was produced by scripts in `src/`; logs are in `results/logs/`, tables and figures in `results/phase0/` and `results/phase1/`. Full Phase 1 tables for all three classifier families: `results/phase1/phase1_report.md` (the earlier M1+M2-only version is `phase1_report_m1_m2.md`).

## Headline

1. **Phase 0 passed.** The data is intact and the official baseline reproduces: balanced accuracy on unseen domains **0.175 ± 0.019** and on seen domains **0.883** over the 10 released seeds (README 0.185 ± 0.014 and 0.879; baseline paper 0.175 and 0.881).
2. **Phase 1 passed, by a huge margin, in all three classifier families.** At a 90% target, standard split-conformal calibration falls far short on unseen domains. Species-averaged coverage is **0.165 ± 0.019** for the released baseline CNN (M1), **0.362 ± 0.009** for a pitch/harmonic-feature model (M2) and **0.355** for a probe on pretrained AST embeddings (M3, the best classifier of the three, 0.288 balanced accuracy on unseen domains). On seen domains it hits the target (0.876, 0.900 and 0.904), so the pipeline is sound. The shortfalls are 74, 54 and 55 points, against a gate of 5.
3. **Few-shot recalibration restores coverage, but only by returning nearly every species.** With 5 to 50 labelled clips per species from the unseen domain, coverage returns to about 0.90 for every family, but the average set holds about 8 of 9 species, even for the best classifier (M3). The base classifier is the real bottleneck.

## What ran

| Step | Result |
|---|---|
| Download | `Development_data.zip` 3.8 GB and `Evaluation_data.zip` 471 MB, both extracted into `data\cdmsc\`; originals untouched |
| Audit (`src/phase0_audit.py`) | 271,380 clips, 0 missing; splits Training 213,647, Validation 30,516, Test 27,217 (23,015 seen, 4,202 unseen) |
| Environment | project `.venv` (system packages reused) with CUDA torch 2.8.0 on the RTX 4060, soundfile, librosa, torchlibrosa |
| Baseline features (`src/baseline_features.py`) | memory-light version of the official script; mean/std from a seeded 40,000-clip subsample of Training (2.65 M frames), not all 213,647 |
| M1 (`src/m1_predictions.py`) | 10 released checkpoints, probabilities on Validation and Test |
| M2 (`src/m2_handcrafted.py`) | 53 pitch/harmonic features (foreground minus background spectrum, fundamental, harmonic amplitudes), gradient boosting, 3 seeds, D5 capped at 3,000 clips per species-domain pair |
| M3 (`src/m3_ast_probe.py`) | frozen AST (AudioSet) pooled embeddings from the public Hugging Face weights, logistic-regression probe on 10,308 training clips (cap 1,500 per species-domain pair), single seed, no domain alignment; 51 minutes on the GPU |
| Analysis (`src/phase1_conformal.py`) | split-conformal, LAC score, calibrated on Validation, plus four repair previews |

## Phase 1 in numbers (alpha = 0.10, mean over seeds)

| Model | Seen coverage (species-avg) | Unseen coverage (species-avg) | Unseen overall coverage | Avg set size, unseen | Empty sets, unseen | Balanced acc., unseen |
|---|---|---|---|---|---|---|
| M1 baseline CNN | 0.876 | **0.165 ± 0.019** | 0.030 | 0.76 | 25% | 0.175 |
| M2 pitch/harmonic + boosting | 0.900 | **0.362 ± 0.009** | 0.214 | 2.46 | 0% | 0.169 |
| M3 AST embedding probe | 0.904 | **0.355** (CI 0.340 to 0.371) | 0.175 | 1.69 | 0% | 0.288 |

Bootstrap 95% intervals (ensemble) for species-averaged unseen coverage: M1 0.137 to 0.174, M2 0.334 to 0.376, M3 0.340 to 0.371.

**Where it breaks.** The failure is concentrated in species whose unseen domain is a different sensor type. For M1, unseen coverage is 0.000 for *Ae. aegypti*, *Ae. albopictus*, *Cx. quinquefasciatus*, *An. gambiae*, *An. arabiensis* and *An. dirus*. For M3 it is 0.21 (*Ae. aegypti*), 0.21 (*Cx. quinquefasciatus*), 0.045 (*An. gambiae*), 0.005 (*An. arabiensis*), 0.000 (*An. dirus*) and 0.14 (*An. stephensi*). Species whose unseen domain is another microphone set do much better: *Ae. albopictus* 0.86, *Cx. pipiens* 0.93 and *An. minimus* 0.81 for M3. Of the 4,202 unseen clips, 3,310 are D1 microphone clips of species trained only on the 0.62 s D5 clips (apparently an optical sensor). Per-domain unseen coverage on D1 is 0.00 (M1), 0.13 (M2) and 0.06 (M3). Two species, *An. arabiensis* and *An. gambiae* in D1, make up 2,638 of the 4,202 unseen clips, and every family recognises them at almost zero (M3 top-1 accuracy 0.1% and 2.2%).

## Repair previews (no retraining, ensembles, alpha = 0.10; PREVIEW ONLY)

| Idea | M1 unseen coverage (species-avg) | M2 | M3 | Verdict |
|---|---|---|---|---|
| Plain split-conformal (reference) | 0.153 | 0.354 | 0.355 | |
| Temperature scaling first | 0.153 | 0.364 | 0.354 | no help |
| Shift-weighted calibration | 0.216 | 0.410 | 0.331 | small help at best; seen-vs-unseen AUC 0.99, 0.98, 0.99 (the shift is trivially detectable); M3's weights degenerate (effective sample size 22) |
| Few-shot target calibration (5 to 50 labelled clips per species) | 0.90 to 0.92 | 0.89 to 0.92 | 0.90 to 0.91 | coverage restored, but sets hold about 8 of 9 species |
| Genus-level sets (3 genera) | 0.238 | **0.659** (set 1.57 of 3) | 0.567 (set 1.25 of 3) | best partial repair, still under 0.90 |
| Sibling-group sets | 0.111 | 0.379 | 0.381 | little change: only *An. gambiae* and *An. arabiensis* merge |

## What this means for Phase 2

- The promise breaks for a structural reason: most unseen clips are a different sensor type from the calibration data (about 98% D5). A better threshold cannot fix classifiers that essentially cannot recognise the unseen species-sensor pairs.
- The bottleneck is **cross-sensor discrimination**. The AST probe is the best base so far (unseen balanced accuracy 0.288 against 0.17 for the others, with no domain alignment), but its coverage is no better, because its errors are concentrated in the same unseen species. Sets only become informative once the classifier carries real signal on unseen sensors.
- Candidate directions, in order of expected payoff:
  1. A stronger base classifier: pretrained embeddings (AST, Perch 2.0, Bird-MAE) with domain alignment (CORAL, MMD), the pitch/harmonic features, and ensembling, evaluated per species-domain.
  2. Calibration that sees a domain change: calibrate on held-out sensors or domains. This needs a classifier trained leave-one-domain-out, so the calibration data really reflects unseen behaviour.
  3. Taxonomy-aware sets (genus first) as the fallback answer when species is uncertain.
  4. A small labelled "site calibration kit" (5 to 20 clips per species), but only once the classifier is good enough to make the resulting sets small.

## Caveats and things to know

- The Validation split was also used to choose the released M1 checkpoints, so M1 calibration is slightly optimistic.
- Feature statistics came from a 40,000-clip subsample, not all training clips. The baseline still reproduces within seed-to-seed spread (0.175 against 0.185), so the effect is small.
- Several species have few unseen clips (*An. dirus* 40, *Cx. pipiens* 68, *An. stephensi* 74, *An. minimus* 99), so their per-species numbers are noisy. *An. dirus*, *An. minimus* and *An. stephensi* have no seen test clips.
- "Seen" test clips are mostly D5 (optical-like) too, so seen coverage says little about microphone behaviour.
- M3 is a single seed and its probe used a 1,500-per-pair training cap, so its numbers carry no seed spread. M2 and M3 calibrate on the full Validation split without model selection on it.
- A bug fix along the way: the first M1 run crashed its DataLoader workers on Windows (the script lacked a main guard), so the script was fixed and rerun without workers.
- Machine note: this laptop has very little free RAM (Chrome and VS Code), so jobs ran one at a time with few workers.
