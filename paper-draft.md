# When the Sensor Changes, the Promise Breaks: Conformal Species Sets for Mosquito Sound under Cross-Domain Shift

*Draft v1, 2026-10-30. Measurement and benchmark study, intended for a workshop or applied bioacoustics venue. Author list and affiliations to be added.*

## Abstract

Mosquito species can be recognised from wingbeat sound, and models score above 88% balanced accuracy on recordings like their training data. On recordings from a sensor they never saw, the official BioDCASE 2026 cross-domain baseline scores 17.5%. We ask whether a classifier can at least be honest about this, by returning a *set* of species that contains the truth 90% of the time, and whether that promise survives a sensor change. It does not. With standard split-conformal calibration, species-averaged coverage on unseen domains is 0.17 to 0.36 against a 0.90 target for three classifier families, while the same method hits 0.88 to 0.90 on seen domains. We trace the failure to a sensor-to-species shortcut: unseen microphone recordings of three species are labelled as the three *Anopheles* species that exist only in microphone training data, which make up 76% of the microphone training clips. We build a leak-free calibration protocol from held-out species-sensor groups inside the training data. It restores coverage on the official unseen groups (0.93 to 1.00), but only by returning 7 to 9 of the 9 species, because the classifier gives essentially no probability to the truth for some held-out clips. All calibration rules lie on one coverage-versus-size curve that sits only 0.1 to 0.2 above a random-subset baseline on the unseen groups, and a few labelled clips from the new site do not shrink the sets. Calibrating separately per detected sensor type does not change the unseen curve but confines its cost: on the whole test it reaches 0.912 coverage at 2.64 species on average, 24 to 46% smaller sets than a global threshold at matched coverage. Audio-level augmentation that mixes microphone background into optical-sensor clips is the only intervention that helped (+2.5 ± 0.7 points of held-out accuracy for a linear head over 5 seeds, and no change on the official unseen test); embedding-space projection, alignment, a second encoder and fusion did not. The honest summary is that calibration cannot rescue a classifier that is near chance on the new sensor, and that this holds for a real, severe shift in bioacoustics.

## 1. Introduction

Acoustic surveillance could widen mosquito monitoring if a cheap sensor could name the species it hears. Laboratory accuracy is high, but deployments change the device, room and background. The 2026 BioDCASE challenge (Task 5, cross-domain mosquito species classification) measures this directly: its baseline drops from 88.1% to 17.5% balanced accuracy on unseen species-domain pairs, and the best of 13 teams reached 31.3%.

When accuracy on new sensors is that low, a deployed system should say how sure it is. Conformal prediction offers a set-valued answer with a coverage promise that needs the calibration and test data to be exchangeable, which a sensor change breaks. We measure how badly, diagnose why, and test what can be repaired. Our contributions are:

1. **A measurement.** Standard conformal calibration undercovers unseen domains by 54 to 74 points at a 90% target across three classifier families, on the official benchmark.
2. **A diagnosis.** The failure is a sensor-to-species shortcut that follows from how the benchmark's species and sensors overlap, with evidence from prediction tables, base rates and sensor separability.
3. **A protocol.** A leak-free pseudo-unseen calibration protocol built from held-out species-sensor groups in the training data.
4. **A bounded positive.** Calibrating separately per detected sensor type confines the cost of honesty to the clips that need it: on the whole dev test, 0.912 coverage at 2.64 species on average, 24 to 46% smaller sets than one global threshold at matched coverage (0.90 to 0.95).
5. **A negative result stated plainly.** Held-out-group calibration restores coverage only by returning most species. Embedding-space compensation, a second encoder, late fusion, taxonomy-lifted sets and few-shot site recalibration did not make the sets informative.

## 2. Related work

*Cross-domain mosquito audio.* HumBugDB provides 36 species of labelled audio; DR-BioL (2025) shows the domain shortcut and reduces it with contrastive and species-conditional alignment, reporting accuracy only. The BioDCASE 2026 challenge report and its winner (frozen Perch 2.0 and Bird-MAE features with CORAL and MMD alignment) reach 0.313 unseen balanced accuracy; the winner states that no validation signal for unseen domains existed and that sibling species overlap.

*Conformal prediction under shift.* Standard split conformal needs exchangeability. Zou and Liu (2024) show on simulated data that it loses coverage on unseen target domains and propose a repair; Siahkali et al. (2026) inflate thresholds under bounded label-conditional covariate shift; Wang et al. (2025) handle unknown subpopulation shifts. A worst-domain maximum threshold is a known conservative baseline. Taxonomy-aware conformal sets exist (Mortier et al., 2025; den Hengst et al., 2025, which includes an audio benchmark). Closest to our few-shot result, Audited Conformal Prediction (arXiv 2606.14909, 2026) uses a small labelled target sample to train an audit model, and a weighted conformal classifier for covariate shift with posterior drift (arXiv 2502.17744) uses source and target samples; a medical-imaging study (arXiv 2506.18162) also finds that coverage breaks under input and label shift and suggests recalibrating on a sample from each new domain. We use these ideas and do not claim them. Searches on 2026-11-02 (conformal prediction with bioacoustics, audio species classification and domain shift) found no work combining conformal sets with species classification from sound, but the searches were not exhaustive.

## 3. Setting

**Data.** CD-MSC development data: 271,380 clips, 60.66 h, 9 species, 5 domains. D5 holds 98% of clips (0.62 s at 8 kHz, apparently the optical-sensor Wingbeats data, inferred from its species list and clip length); D1 to D4 are microphone sets (5,728 clips; mean clip lengths 2.7 to 50 s). Each species has one official unseen domain at test time: the species-domain pair is absent from Training and Validation. The unseen test split has 4,202 clips; 3,310 are D1 microphone clips of species that appear in training only through D5.

**Metrics.** Balanced accuracy on seen and unseen pairs. For conformal sets: species-averaged (macro) coverage, coverage of each of the 9 official unseen groups, mean set size, empty and vacuous (all 9 species) shares.

**Classifiers.** (M1) the 10 released baseline checkpoints (log-mel CNN); (M2) 53 pitch and harmonic features with gradient boosting; (M3) frozen AST (AudioSet) embeddings with a linear probe. Later: AST layer taps with an MLP and cosine head; Bird-MAE; late fusion.

## 4. Standard calibration fails (Phase 1)

With the LAC score (1 minus the true-class probability) and the Validation split as calibration data, at a 90% target:

| Model | Seen coverage | Unseen coverage | Unseen BA |
|---|---|---|---|
| M1 baseline CNN | 0.876 | 0.165 ± 0.019 | 0.175 |
| M2 pitch features | 0.900 | 0.362 ± 0.009 | 0.169 |
| M3 AST probe | 0.904 | 0.355 | 0.288 |

The method hits its target on seen domains, so the pipeline is sound. Simple repairs fail: temperature scaling does nothing, shift-weighted calibration helps slightly (M2 0.354 to 0.410, M3 worse), and genus-level sets reach only 0.57 to 0.66.

## 5. Why: a sensor-to-species shortcut

- **Where the errors go.** Unseen D1 clips of *Cx. quinquefasciatus*, *An. gambiae* and *An. arabiensis* are labelled as *An. minimus*, *An. dirus* or *An. stephensi*: the AST probe sends at least 67% of *Cx. quinquefasciatus* and at least 53% of *An. gambiae* there, and the released baseline sends 88% of *An. gambiae* to *An. stephensi*.
- **Why the data teaches it.** 1,138 of the 1,490 microphone clips in Training+Validation (76%) belong to those three microphone-only species; 99.4% of Training clips are D5. Only three bridge species (*Ae. aegypti*, *Ae. albopictus*, *Cx. pipiens*) have both sensors, with 340 microphone clips.
- **The sensor is easy to read.** A linear probe on the AST embedding tells D5 from microphone clips at 0.989 balanced accuracy.
- **Siblings are secondary.** Inside D5, *An. gambiae* and *An. arabiensis* confuse each other about 21 to 22% of the time but each keeps about 65% recall; in D1 they are mostly labelled as other species.

## 6. Pseudo-unseen calibration protocol

The official Validation split contains only seen species-domain pairs, so it cannot show behaviour on a new sensor. From Training+Validation we hide groups and train fold models that never see them: three leave-one-species'-microphones-out folds (*Ae. aegypti* 131, *Ae. albopictus* 90, *Cx. pipiens* 119 clips) that mimic D5-to-microphone, and six leave-one-domain-out folds (*An. minimus* D1, D3, D4; *An. stephensi* D1, D2, D3; 51 to 200 clips) that mimic microphone-to-microphone. That gives 9 groups and 1,391 clips. Out-of-fold scores give thresholds: pooled over groups (C1), second-worst group (C2b), worst group (C2). If the new group's threshold is exchangeable with the nine, the chance it falls at or below the worst is G/(G+1) = 0.9 (an assumption we cannot check, not a theorem). Noise banks used for augmentation exclude the held-out group's recordings.

## 7. Experiments

**Input.** A canonical 0.62 s snippet tiled to the encoder window lost to the Phase 1 padded input (unseen 0.210 against 0.259; fold accuracy 0.351 against 0.413). We keep the padded input.

**Compensation.** Nuisance-direction projection (k = 2, 3) and a per-species alignment loss gave no gain on the folds that matter (bridge-fold accuracy 0.087 to 0.124 against 0.134). Audio-level augmentation (microphone background from a fold-specific bank, random spectral tilt, band limits) raised held-out fold accuracy from 0.414 to 0.439 ± 0.002 with a linear head (+2.5 ± 0.7 points over 5 seeds, all five paired differences positive) and bridge-fold accuracy from 0.134 to 0.184; an MLP head reached 0.447 ± 0.009 overall but only 0.118 on the bridge folds. Official unseen balanced accuracy did not move (0.259 to 0.261).

**Second encoder.** Bird-MAE reached 0.221 unseen and 0.003 on the bridge folds; late fusion with AST (0.240) was worse than AST alone (0.264).

**Perch 2.0.** Using a community PyTorch port of the official weights (1,536-d embedding, the same padded-input and augmentation recipe as for AST and Bird-MAE; the port was not verified against the official model) produced no gain: unseen balanced accuracy 0.261 ± 0.001 (linear head) and 0.244 ± 0.005 (MLP head) over 3 seeds, held-out bridge folds only 0.02 against 0.13 to 0.18 for AST, and no benefit from fusion. We did not reproduce the winner's 0.318 for Perch alone.

Best classifier: AST with audio augmentation, 0.261 ± 0.008 unseen balanced accuracy over 5 seeds (ensembles reach 0.264 to 0.270), below the 0.313 challenge result.

## 8. Calibration results

At alpha = 0.10 on the 9 official unseen groups:

| Rule | Unseen coverage | Set size (of 9) | Vacuous | Groups >= 0.80 |
|---|---|---|---|---|
| C0 validation | 0.325 | 1.54 | 0% | 2/9 |
| C1 pooled pseudo-unseen | 0.934 | 6.81 | 13% | 8/9 |
| C2b second-worst | 0.968 | 8.01 | 44% | 9/9 |
| C2 worst group | 1.000 | 9.00 | 100% | 9/9 |

Held-out-group calibration repairs coverage, but through vacuous sets. The worst-group threshold is the maximum score because some held-out clips get essentially zero probability on the truth. Genus-level worst-group sets hold 2.95 of 3 genera. Clip-level bootstrap 95% intervals (clip sampling only, not training variance): C0 0.325 [0.309, 0.341]; C1 0.933 [0.907, 0.948] with 7.5 [7.2, 7.8] species per group on average; C2b 0.976 [0.949, 0.995]; C2 1.000 [0.999, 1.000] (`results/phase2e/bootstrap_report.md`). Figure 1 shows C1, C2b and C2 on one curve, 0.1 to 0.2 above the random-subset line; Figure 2 shows per-group coverage. Few-shot recalibration with 5 to 50 labelled clips per species keeps coverage at about 0.90 to 0.92 with sets of 7.1 to 7.4 species.

## 9. Sensor-conditional calibration (the one positive result)

Because the vacuous sets arise only on microphone clips of species the model knows through the optical sensor, we also calibrate per sensor type (Mondrian conformal). A sensor detector (a linear probe on AST embeddings fit on Validation, balanced accuracy 0.981 and AUC 0.999 on the test split, no test labels used) routes each clip. Optical (D5) clips use a threshold calibrated on Validation D5 clips; microphone clips use a threshold from the nine pseudo-unseen groups. Every official unseen group is a microphone domain. All three microphone-branch rules were fixed in advance (pooled, second-worst, worst group); we report the pre-specified worst-group rule as the headline.

At alpha = 0.10 the whole dev test (seen and unseen) is covered at **0.912 with mean set size 2.64** (optical clips 0.900 at 1.52 species; unseen groups 0.980 at 8.72). Against one global threshold at matched whole-test coverage, the sensor-conditional rule needs sets **24% smaller at 0.90 coverage (2.57 against 3.37 species) and 46% smaller at 0.95 (2.98 against 5.49)**; the standard Validation threshold cannot reach 0.90 at any alpha we tried. Replacing the detector by the true sensor type changes little (0.916 at 2.69).

This is an honest, bounded gain: on the unseen groups alone the global and sensor-conditional curves coincide (Figure 3), so unseen sets still hold 8 to 9 of 9 species. The whole-test number is dominated by D5, 84% of test clips. The result is that the cost of honesty can be confined to the clips where the classifier cannot be trusted, not that the classifier crosses the gap.

## 10. Discussion and limits

The calibration rule chooses a point on a coverage-versus-size curve; the classifier determines the curve. Under a sensor change that the classifier cannot cross, the only way to keep the promise is to say very little. The result is a calibrated statement of ignorance, which is useful as a warning against deploying such systems on new sensors, and as a benchmark for better classifiers.

**Limits.** Two seeds for most ablations (5 for the augmentation comparison); the unseen test was looked at during diagnosis and then used in reporting, though configuration choices used pseudo-unseen folds; pseudo-unseen groups are small and only three mimic the official shift; unlabeled adaptation, fine-tuning and more augmentation were not tried, and our Perch weights are a community port; our best classifier is below the challenge winner and we cannot say why; evaluation labels are unreleased, so there is no clean test; D5's identity is inferred.

## 11. Conclusion

On a real, severe sensor shift in mosquito audio, standard conformal prediction does not hold its promise, the cause is a shortcut the training data teaches, and calibrating on held-out groups restores coverage only at the price of near-vacuous answers. Progress needs a classifier that crosses the sensor gap; we release the diagnosis, the protocol and the curve as the yardstick.

## Figures and tables to finalise

- Fig. 1: `results/figs/phase2_coverage_vs_size.png`
- Fig. 2: `results/figs/phase2_per_group_coverage.png`
- Fig. 3: `results/figs/phase4_mondrian_frontier.png`
- Tables from `results/phase2_report.md`, `results/phase1/phase1_report.md`, `results/phase1/diagnostics.md`
- Not done (the project was closed on 2026-11-02, so these are open caveats, not a to-do list): checking the Perch port against the official model; a still fuller literature search; confirming D5's identity; evaluation labels were not requested. Done 2026-11-02: citations checked, 5-seed comparison (`results/phase2e/`), CSV exports (`results/csv/`).
