# Final presentation outline

Audience: a technical audience that knows machine learning but not mosquito bioacoustics or conformal prediction. About 12 minutes, 10 slides. Numbers are from `results/phase2_report.md` and `paper-draft.md`; figures are in `results/figs/`.

## The story in one sentence

Sound could track disease-carrying mosquitoes, but a model that is "90% sure" on one recording sensor is wrong most of the time on a new one; we measured how badly, found why, tried to repair it, and found what can and cannot be fixed.

## Slides

1. **Why listen to mosquitoes.** Species differ in wingbeat tone; cheap sensors could widen surveillance (dengue, malaria, filariasis). The catch: it must work on a new device.
2. **The benchmark and the gap.** BioDCASE 2026 CD-MSC: 271,380 clips, 9 species, 5 domains. Official baseline 88% on familiar recordings, **17.5%** on new ones (chance 11%); best challenge system 31.3%.
3. **The question.** Can the system at least be honest? A set of species with the promise "the truth is inside at least 90% of the time", and does that promise survive a new sensor? (One-line explanation of conformal prediction.)
4. **The promise breaks.** Standard calibration at a 90% target: coverage 0.88 to 0.90 on familiar domains, **0.17 to 0.36** on new ones, in three classifier families. Per-group heat map: `phase2_per_group_coverage.png`.
5. **Why: a sensor-to-species shortcut.** New-sensor clips of *Cx. quinquefasciatus*, *An. gambiae*, *An. arabiensis* are labelled as the three *Anopheles* that exist only in microphone data (76% of microphone training clips; 99.4% of training clips are the optical sensor D5). The sensor is easy to read (0.989 balanced accuracy). Siblings are a secondary problem.
6. **A fair way to test repairs.** Official validation contains only seen species-sensor pairs, so we hide species-sensor groups inside training: 9 groups, 1,391 clips, leak-free (noise banks exclude the held-out group).
7. **What we tried.** Table: tiled input (worse), projection and alignment (nothing), Bird-MAE and fusion (worse), audio augmentation (+2.5 ± 0.7 points held-out, 5 seeds, nothing on the official unseen test). Best classifier 0.26 unseen versus winner 0.31.
8. **Calibration repairs coverage, with a price.** Held-out-group calibration: 0.93 to 1.00 coverage but 7 to 9 of 9 species per answer. All rules sit on one coverage-versus-size curve, only 0.1 to 0.2 above random subsets (`phase2_coverage_vs_size.png`); a few labelled clips from the new site do not shrink the sets.
9. **The bounded positive.** Calibrate per detected sensor type: whole-test coverage 0.912 at 2.64 species; at matched coverage 24% (0.90) to 46% (0.95) smaller sets than a global threshold (`phase4_mondrian_frontier.png`). Unseen groups still get 8 to 9 species: the cost is confined, not removed.
10. **Take-aways and limits.** Calibration cannot rescue a classifier that is near chance on the new sensor; confine the cost of honesty to the clips that need it; the yardstick (curve, protocol) is reusable. Limits: 2 to 5 seeds, unseen test inspected during diagnosis, Perch 2.0 tried via a community port (no gain), no evaluation labels.

## Questions to prepare for

- *Is this just a weak classifier?* Partly: the best we built is below the winner. The calibration result still shows the structure: coverage can only be bought with set size when the classifier is near chance.
- *Why not fine-tune?* Few microphone clips, 8 GB GPU, and fine-tuning is known to hurt out-of-distribution accuracy; untested here.
- *What is new?* Hierarchical conformal sets and conformal under shift both exist (cite them). New here: the measurement on a real sensor shift in bioacoustics, the shortcut diagnosis, the leak-free held-out-group protocol, and the sensor-conditional result.
- *Is the Mondrian result inflated by the easy optical clips?* Yes, whole-test numbers are dominated by D5 (84%); we show unseen-only curves next to it.
- *Can I trust the unseen numbers?* Configuration choices used only held-out folds, but the unseen test was inspected during diagnosis, so treat them as exploratory until labels for the evaluation set exist.
- *Does 5 seeds change the story?* It corrected the augmentation gain from +4.3 (two seeds) to +2.5 ± 0.7.

## Demo or backup material

Per-group coverage table (`results/csv/phase2_calibration_rules.csv`), seed table (`results/phase2e/seed_report.md`), bootstrap intervals (`results/phase2e/bootstrap_report.md`), the diagnostics table (`results/phase1/diagnostics.md`).
