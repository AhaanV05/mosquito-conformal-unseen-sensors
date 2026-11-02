# Final results statement

Project: trustworthy mosquito identification from sound on unseen sensors. Benchmark: BioDCASE 2026 Cross-Domain Mosquito Species Classification (CD-MSC). Closed 2026-11-02. Every number below is in `results/` (tables, CSVs in `results/csv/`, figures in `results/figs/`) and traces to a script in `src/`.

## The question

A mosquito-sound classifier can say how sure it is by returning a set of species that contains the truth at least 90% of the time (conformal prediction). Does that promise survive a recording sensor the model has never seen, why not if it does not, and what can be repaired?

## What the results state

1. **The data and the baseline are sound.** 271,380 clips, none missing. The 10 released baseline checkpoints reproduce at 0.175 ± 0.019 balanced accuracy on unseen domains and 0.883 on seen (README: 0.185 and 0.879).

2. **The promise breaks, badly.** At a 90% target, species-averaged coverage on unseen domains is 0.165 (released baseline CNN), 0.362 (pitch-feature model) and 0.355 (AST probe), against 0.876 to 0.904 on seen domains. The shortfall is 54 to 74 points in three classifier families. Clip-bootstrap interval for the standard rule on the best model: 0.325 [0.309, 0.341].

3. **The cause is a sensor-to-species shortcut.** Unseen microphone clips of *Cx. quinquefasciatus*, *An. gambiae* and *An. arabiensis* are labelled as the three *Anopheles* species that exist only in microphone training data (*minimus*, *dirus*, *stephensi*). Those three make up 76% of the microphone training clips, 99.4% of training clips come from the optical-like sensor D5, and the sensor is easy to read from the embedding (0.989 balanced accuracy). Sibling confusion (*An. gambiae* vs *An. arabiensis*, about 21 to 22% inside D5) is real but secondary.

4. **No classifier we built crossed the sensor gap.** Best unseen balanced accuracy is 0.26 to 0.27 (challenge winner 0.313; target 0.32). Over 5 seeds the AST linear probe on the padded input gets 0.259 ± 0.001; with audio augmentation 0.261 ± 0.003; with an MLP head 0.261 ± 0.008.

5. **What was tried, and what happened.**

   | Idea | Result |
   |---|---|
   | Fixed-shape tiled input | worse than the padded input (unseen 0.210 vs 0.259) |
   | Removing sensor directions; per-species alignment | no gain on the held-out groups |
   | Audio augmentation (microphone background from a bank that excludes the held-out group, random filtering) | the only gain: held-out accuracy +2.5 ± 0.7 points (linear head), +3.3 ± 1.2 (MLP head, but worse on the bridge folds), 5 seeds; no change on the official unseen test |
   | Bird-MAE; late fusion | worse (0.22 unseen; fusion 0.24 to 0.26) |
   | Perch 2.0 via a community port | no gain (0.261 linear, 0.244 MLP; bridge folds 0.02); the winner's 0.318 was not reproduced |
   | Genus-level and taxonomy-lifted sets | also near-vacuous at a 90% promise |

6. **Calibrating on held-out groups restores coverage but not usefulness.** Nine pseudo-unseen groups built from Training+Validation (1,391 clips, leak-free) give coverage on the official unseen groups of 0.933 [0.907, 0.948] (pooled rule, 7.5 species per group on average), 0.976 (second-worst group) and 1.000 (worst group, all 9 species). The rules all lie on one coverage-versus-set-size curve only 0.1 to 0.2 above a random subset of the same size. A few labelled clips from the new site (5 to 50 per species) keep coverage near 0.90 with sets of about 7 of 9 species. A calibration rule picks a point on the curve; the classifier sets the curve.

7. **One bounded positive.** Calibrating separately per detected sensor type (detector balanced accuracy 0.981) gives, on the whole dev test, 0.912 coverage at 2.64 species on average, and at matched coverage 24% smaller sets at 0.90 (2.57 vs 3.37 species) and 46% smaller at 0.95 (2.98 vs 5.49) than one global threshold. On the unseen groups alone nothing changes (8 to 9 species). The whole-test number is dominated by D5, 84% of test clips. This confines the cost of honesty to where the classifier cannot be trusted; it does not make the classifier work on the new sensor.

## Gate summary

| Gate | Target | Outcome |
|---|---|---|
| 0 | baseline reproduces | passed |
| 1 | standard coverage short by more than 5 points | passed (54 to 74 points) |
| 2a | unseen balanced accuracy at least 0.32 | missed (best 0.27) |
| 2b | +3 points on held-out groups | borderline (+2.5 ± 0.7 linear; +3.3 ± 1.2 MLP; official unseen test unchanged) |
| 2c | coverage at least 0.85 with non-vacuous sets | coverage met, non-vacuity missed |
| 3 | sets at least 20% smaller at equal coverage with a little labelled new-site audio | missed |

## Honest limits

- Most ablations use two seeds; the key augmentation comparison and Perch use 5 and 3. Bootstrap intervals cover clip sampling, not training variance.
- The official unseen test was inspected during diagnosis and then used in reporting. Configuration choices used only the held-out groups from Training+Validation, but claims about the unseen test are exploratory. Evaluation labels were never released and were not requested.
- The nine held-out groups are small (51 to 200 clips) and only three mimic the official shift (optical to microphone).
- The sensor-conditional result was designed after seeing where the failures were; its three microphone-branch rules were fixed in advance and all are reported.
- Perch weights come from a community port, not verified against the official model. D5's identity as the Wingbeats optical data is inferred. Unlabeled adaptation and fine-tuning were not tried.
- Nothing here shows a deployable system.

## Novelty, stated carefully

Taxonomy-aware conformal sets, conformal prediction under unseen domains, and a worst-domain maximum threshold all exist in the literature (see `paper-draft.md`). To our searches, no earlier work combines conformal sets with species classification from sound. What this project contributes is the measurement on a real, severe sensor shift, the shortcut diagnosis with its evidence, a leak-free held-out-group calibration protocol, the coverage-versus-size yardstick, and the bounded sensor-conditional result.

## Where everything is

| What | File |
|---|---|
| Paper draft | `paper-draft.md` |
| This statement | `FINAL_RESULTS.md` |
| Plain-language problem and status | `problem-statement.md` |
| All Phase 2 to 4 tables and caveats | `results/phase2_report.md` |
| Phase 0 and 1 | `results/overnight_report.md`, `results/phase1/`, `results/phase0/` |
| Seeds, bootstrap, Perch, Mondrian | `results/phase2e/`, `results/phase2g/`, `results/phase4/` |
| CSV exports | `results/csv/` |
| Design that was followed (with an outcome banner) | `architecture-next-phases.md` |
| Presentation | `presentation-outline.md` |
| Overview and reproduction steps | `README.md` |
