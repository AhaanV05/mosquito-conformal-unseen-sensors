# Problem Statement

**Project:** Trustworthy mosquito identification from sound, on recordings the model has never seen
**Benchmark:** BioDCASE 2026 Cross-Domain Mosquito Species Classification (CD-MSC)
**Status:** Final (v3), closed 2026-11-02 after Phases 0 to 4 were run; see `FINAL_RESULTS.md`. Earlier: updated 2026-10-30. v1 was written before any experiment, v2 after Phases 0 and 1.
**Where we are:** the premise held, but the repairs we tried did not make the answers useful. The project is now a measurement and benchmark study. The paper draft is `paper-draft.md` and the Phase 2 and 3 numbers are in `results/phase2_report.md`. `architecture-next-phases.md` is the plan that was followed; its banner says what happened.

---

## 1. The short version

Cheap microphones could help track disease-carrying mosquitoes, because every mosquito species has a slightly different flight sound. Computer models can already tell species apart very well on recordings that look like the ones they trained on. But when the same model hears a recording from a **different place, microphone or room**, it falls apart. On the 2026 benchmark we use, the official baseline scores 88% on familiar recordings and only **17.5% on new ones** (random guessing would give 11%). The best system in the 2026 challenge reached only **31.3%**.

We want to make this more *honest and more useful*, not just a few points more accurate. Our model will:

1. say **how sure it is** and give a **set of possible species** with a measurable promise ("the true answer is in this set at least 90% of the time"),
2. keep that promise **even on recordings from a new device or place**, and
3. answer at the **right level of detail**. If two species sound identical, it should say "Anopheles gambiae complex" instead of guessing one.

**What Phases 2 and 3 added (2026-10-30).** We could not build a classifier that crosses the sensor gap: the best reaches 0.27 balanced accuracy on new domains (target 0.32; the challenge winner reports 0.31). Calibrating on held-out groups does restore the promise on new domains (coverage 0.93 to 1.00 at a 90% target), but only by listing 7 to 9 of the 9 species, because for some held-out recordings the model gives the true species almost no probability. A few labelled clips from the new site do not shrink the sets. The honest finding is that a promise about uncertainty cannot rescue a classifier that is near chance on the new sensor.

**What the first two phases showed.** The promise does break, badly. With a 90% target, standard methods cover only **17% to 36%** of the true species on new domains, while on familiar domains they hit 88% to 90%. The cause is not only weak confidence estimates: the models learn to recognise the *recording device* instead of the mosquito, so recordings from a new microphone get labelled as one of three species that were only ever recorded on microphones. Fixing that is the next job.

---

## 2. What Phase 0 and Phase 1 have answered

Everything here was measured on the real data on a laptop GPU (RTX 4060), using scripts in `src/`. Full numbers are in `results/overnight_report.md`, `results/phase1/phase1_report.md` and `results/phase1/diagnostics.md`.

### 2.1 Questions that now have an answer

| Question | Answer | Evidence |
|---|---|---|
| Is the data usable and complete? | **Yes.** 271,380 clips, none missing. Splits: Training 213,647, Validation 30,516, Test 27,217 (23,015 seen, 4,202 unseen) | Phase 0 audit |
| Does the official "unseen" label mean what we thought? | **Yes.** It agrees 100% with "this species was never recorded in this domain in Training or Validation", and no unseen clip appears in Training or Validation | Phase 0 audit |
| Can we reproduce the official baseline? | **Yes.** 10 released checkpoints: **0.175 ± 0.019** on unseen domains and **0.883** on seen (README: 0.185 ± 0.014 and 0.879) | Phase 0 |
| **RQ1: does the promise break on new domains?** | **Yes, by a very wide margin.** At a 90% target, species-averaged coverage on unseen domains is **0.165** (official baseline CNN), **0.362** (pitch and harmonic features) and **0.355** (probe on a pretrained audio model, AST). On seen domains the same methods give 0.876, 0.900 and 0.904. The shortfall is 54 to 74 points; our gate was 5 | Phase 1 |
| Is that just because the baseline is weak? | **No.** The best of the three classifiers (AST probe, 0.288 balanced accuracy on unseen domains) fails just as badly, so the failure is not a quirk of one model | Phase 1, three families |
| Is the sensor shift hard to see? | **No, it is easy to see**: a simple test tells seen from unseen clips with AUC 0.98 to 0.99, and the pretrained embedding separates the D5 sensor from microphone clips with 0.989 balanced accuracy. Detecting the shift is not the problem; coping with it is | Phase 1, diagnostics |
| Where exactly does it break? | In species whose unseen domain is a **different kind of sensor**. 3,310 of the 4,202 unseen test clips are D1 microphone clips of species the model only heard through D5. *An. arabiensis* and *An. gambiae* in D1 make up 2,638 of those and are recognised almost never (0.1% and 2.2%). Species whose unseen domain is another microphone do far better (*Cx. pipiens* 0.91, *Ae. albopictus* 0.70, *An. minimus* 0.68 for the AST probe) | Phase 1 |
| **Why** does it break? | **A sensor-to-species shortcut.** Unseen D1 clips of *Cx. quinquefasciatus*, *An. gambiae* and *An. arabiensis* are labelled as the three *Anopheles* species that exist only in microphone data (*minimus*, *dirus*, *stephensi*): the released baseline sends 88% of *An. gambiae* to *An. stephensi* alone. The training data explains it: **76% of microphone clips in Training+Validation belong to those three species**, and 99.4% of Training clips are D5 | Diagnostics |
| Are the sibling species the main problem? | **Not the main one.** In the D5 sensor, *An. gambiae* and *An. arabiensis* confuse each other about 21% to 22% of the time but each keeps about 65% recall. In D1 they mostly get labelled as *other* species. The sibling problem is real but secondary | Diagnostics |
| Can simple repairs fix the promise without retraining? | **No.** Temperature scaling does nothing. Re-weighting for the shift helps a little (pitch-feature model 0.354 to 0.410; AST probe gets worse, 0.355 to 0.331). **A few labelled clips from the new domain** (5 to 50 per species) restore about 0.90 coverage, but only by listing about 8 of the 9 species, which says almost nothing. **Genus-level answers** are the best partial repair (0.659 for the pitch-feature model, 0.567 for the AST probe) but still fall short of 0.90 | Phase 1 previews |
| So what is the real bottleneck? | **Telling species apart across sensors.** A better threshold cannot rescue a classifier that is near chance on the unseen species-sensor pairs. The classifier has to improve first, then calibration has something to work with | Phase 1 conclusion |
| Is there headroom? | **Yes.** A plain probe on pretrained AST features, with no alignment at all, reaches 0.288 on unseen domains, close to the challenge winner's 0.313 | Phase 1 |
| Does the data fit on our machine? | **Yes.** About 9 GB extracted, a GPU with 8 GB, little spare memory. Jobs must run one at a time | Phase 0 |
| What is the data really like? | See section 10. The pieces v1 could only guess at (domain sizes, clip lengths, species per domain) are now measured | Phase 0 audit |
| Is the novelty claim still valid? | **Narrower than v1 thought.** A literature check found that taxonomy-aware conformal sets already exist (2025) and that conformal prediction losing coverage on unseen domains is already known (2024). What we still have is the *combination*, on a real, severe sensor shift in bioacoustics, with a diagnosis of the cause and a calibration protocol built from the benchmark's own structure. See section 5 | Literature check |
| Can we get the pretrained models? | **Partly.** AST and Bird-MAE are public. The official Perch 2.0 path returned a login error, so it needs your action | Checked |

### 2.2 What is still open

- Does a stronger classifier plus sensor compensation fix the shortcut? (Phase 2a and 2b)
- Does calibration learned from held-out *groups* transfer to the official unseen groups? (Phase 2c)
- How informative can the sets be, at species, group and genus level? (RQ3)
- How much does a little new-site audio help? (RQ4, Phase 3)
- Which real source is each domain? Will evaluation labels be released?

### 2.3 Caveats on the Phase 1 numbers

- The Validation split was also used to choose the released baseline's checkpoints, so the baseline's calibration is slightly optimistic.
- Feature statistics for the baseline came from a subsample of 40,000 training clips. The baseline still reproduces within its seed spread.
- Several species have few unseen clips (*An. dirus* 40, *Cx. pipiens* 68, *An. stephensi* 74, *An. minimus* 99), so their individual numbers are noisy.
- "Seen" test clips are mostly D5 too, so seen coverage says little about microphone behaviour.
- The AST probe is a single run (no seed spread), trained on a capped subset of 10,308 clips.
- A probe trained on D1 clips themselves separates the three D1 species at 0.954 balanced accuracy, but that number is optimistic (clips from one recording leak and different species were probably recorded in different sessions). We do not read it as proof that the sibling species can be told apart by sound.
- The unseen test predictions have now been looked at. Later design choices are therefore made on pseudo-unseen groups built from Training and Validation (see the architecture document), and the unseen test is reserved for the final report.

---

## 2.4 What Phases 2 and 3 answered (2026-10-30)

| Question | Answer |
|---|---|
| Does a cleaner, fixed-shape input (loudest 0.62 s pieces, tiled) remove the sensor clue? | **No, it was worse.** Unseen balanced accuracy 0.210 against 0.259 for the Phase 1 padded input, and 0.351 against 0.413 on held-out groups |
| Do embedding-space fixes (removing sensor directions, per-species alignment) help? | **No.** No gain on the held-out species-sensor groups that matter |
| What helped? | **Audio-level augmentation**: mixing microphone background noise (from a bank that excludes the held-out group) and random filtering into D5 clips. +2.5 ± 0.7 points on held-out groups with a linear head (0.414 to 0.439, 5 seeds; +3.3 ± 1.2 with an MLP head, which is worse on the bridge folds), the only gain; no change on the official unseen test (0.259 to 0.261) |
| Does a second pretrained model (Bird-MAE) or fusing it help? | **No.** Bird-MAE 0.22 unseen and near zero on held-out species-sensor groups; fusion 0.24 to 0.26, worse than AST alone (0.26 to 0.27) |
| Can calibration on held-out groups repair the promise? (RQ2) | **Coverage yes, usefulness no.** Coverage on the 9 official unseen groups: 0.93 (pooled rule, sets of 6.8 of 9 species), 0.97 (second-worst group, 8.0), 1.00 (worst group, all 9). Standard calibration gets 0.33 |
| How precise can answers be? (RQ3) | **Not very.** Taxonomy-lifted and genus-level sets are also near-vacuous at a 90% promise (worst-group genus sets hold 2.95 of 3 genera). All rules lie on one coverage-versus-size curve only 0.1 to 0.2 above a random subset of the same size |
| Does a little labelled new-site audio help? (RQ4, labelled part) | **Not enough.** 5 to 50 labelled clips per species keep coverage near 0.90 with sets of about 7.1 of 9 species. Unlabeled adaptation was not built |
| Can we keep sets small where it is safe? | **Yes, in a bounded way.** Calibrating separately per detected sensor type (detector balanced accuracy 0.981) gives 0.912 coverage at 2.64 species on the whole dev test, against 3.37 species for a global threshold at 0.90 and 5.49 at 0.95. On the unseen groups alone nothing changes (8 to 9 species). Whole-test numbers are dominated by D5, 84% of test clips |
| How certain are these results? | Clip-bootstrap 95% intervals: standard calibration 0.325 [0.309, 0.341] against pooled held-out-group calibration 0.933 [0.907, 0.948]. Five seeds for the augmentation comparison corrected the two-seed numbers downward (see `results/phase2_report.md`) |
| Is Perch 2.0 better? | **No.** unseen balanced accuracy 0.261 ± 0.001 with a linear head and 0.244 ± 0.005 with an MLP head (3 seeds), seen 0.77 to 0.78, held-out fold accuracy 0.387 ± 0.003 and 0.432 ± 0.008, bridge folds only 0.02; fusion with AST (0.236 unseen) and with AST and Bird-MAE did not beat AST alone, and calibration behaved the same (pooled held-out-group rule 0.905 coverage with 7.4 species). The winner's 0.318 for Perch alone was not reproduced |

Details and caveats: `results/phase2_report.md`.

---

## 3. Background

### 3.1 Why mosquitoes matter

Mosquito-borne diseases such as malaria, dengue, chikungunya, Zika and filariasis affect more than a billion people every year and cause close to a million deaths (WHO figures quoted in the challenge paper). Which mosquito is present matters a lot:

- *Aedes aegypti* and *Aedes albopictus* spread dengue, chikungunya and Zika.
- *Anopheles* species spread malaria.
- *Culex quinquefasciatus* spreads lymphatic filariasis.
- Many other species bite but spread little or no disease.

Health teams track mosquitoes with traps and then identify each catch by hand. That is slow, expensive, needs experts, and exposes field workers to infection. It also covers very few places.

### 3.2 Why listen to them

A flying mosquito beats its wings hundreds of times per second, which makes a thin, steady tone. The tone's pitch and its overtones depend on the species, the sex (males are higher), body size and even the temperature. A phone microphone can pick this up. If a cheap sensor could name the species it hears, surveillance could cover many more places at much lower cost.

### 3.3 What the computer actually sees

We turn each short audio clip into a picture of sound (a spectrogram), or into a list of numbers (an embedding) made by a large pretrained audio model, and train a classifier to name the species. This part is mostly solved: within one recording setup, accuracy is high.

---

## 4. The problem

### 4.1 It works in the lab and fails in the field

A model trained on recordings from some devices and places is usually tested on recordings from the *same* devices and places. That hides a big weakness. In real use, the sensor is new, the room is different and the background noise is different.

### 4.2 What is a "domain"?

In this project, a **domain** means one recording condition: the place, the device and the acoustic environment together. **Domain shift** means the test recordings come from a condition the model never trained on. In this benchmark the biggest shift is a change of **sensor type**: D5 appears to be an optical sensor, D1 to D4 are microphones.

### 4.3 The numbers

| Evidence | Result |
|---|---|
| Official CD-MSC baseline, familiar domains (reproduced by us) | 88.3% balanced accuracy |
| Official CD-MSC baseline, **new** domains (reproduced by us) | **17.5%** (chance for 9 species is 11.1%) |
| Best challenge system (JKU Linz), new domains | **31.3%** |
| Our probe on pretrained AST features, new domains | **28.8%** (no alignment, single run) |
| One earlier study (DR-BioL, 2025), same species on familiar vs new data | 99.8% vs 41.4% |
| Share of the "90% promise" actually delivered on new domains (our Phase 1) | **17% to 36%** |

Even the winner scored far below what we would call "useful in the field".

### 4.4 Why it fails

1. **Shortcut learning, now measured.** A model can score well by recognising the *microphone or room* instead of the mosquito. In our data the shortcut is visible: new-sensor clips of three species get labelled as the three species that were only ever recorded on microphones.
2. **Very lopsided data.** About 98% of the benchmark's clips come from a single source (D5, appears to be the optical-sensor *Wingbeats* data). The four microphone sources hold 5,728 clips in total. The model mostly learns the big source.
3. **Species and domain are tangled.** Six species have D5 recordings. Three species (*An. dirus*, *An. minimus*, *An. stephensi*) exist only in microphone data. Only three "bridge" species (*Ae. aegypti*, *Ae. albopictus*, *Cx. pipiens*) appear in both worlds, with just 340 microphone clips in Training+Validation. So "microphone-like" becomes a clue for three *Anopheles* species.
4. **The training input itself leaks the sensor.** In our Phase 1 probe, every 0.62 s D5 clip sat in a 10-second window that was 94% empty, while microphone clips filled most of it. Clip length and padding can identify the sensor too. (This is a likely contributor, still to be tested.)

### 4.5 A second problem: some species sound the same

The winner's own report shows two sibling species, *Anopheles gambiae* and *Anopheles arabiensis*, scoring **5% and 19%**. They are members of the same species complex and sound almost identical. Our diagnostics agree in part: even inside the optical sensor, a probe confuses them with each other about 21% to 22% of the time. The benchmark scores a 9-way guess and punishes the model for an acoustically near-impossible choice. A trustworthy system should say "one of these two" when that is all the sound can support. This is a secondary problem; section 4.4 is the main one.

---

## 5. What has been done, and what hasn't

| Work | What it did | What it left open |
|---|---|---|
| HumBugDB (2021) | 20 hours of labelled mosquito audio, 36 species, baseline neural networks | Tested mostly on matched recordings |
| BEANS (2022) | Put HumBugDB into a standard animal-sound benchmark | Within-distribution splits |
| MosquitoSong+ (2024) | Noise-robust species and sex classification | Not tested on new devices or places |
| DR-BioL (2025) | Showed the shortcut problem and trained models to reduce it | Accuracy only, no guarantees |
| BioDCASE 2026 CD-MSC challenge | Official cross-domain benchmark, 13 teams, public reports | Best result is still 31.3% |
| Winning system (JKU Linz) | Frozen Perch 2.0 and Bird-MAE features, domain alignment, a rule that trusts several device-independent features only when they agree | Its report says there is no way to check performance on unseen domains and that some species overlap in sound |
| Hierarchical conformal prediction (Mortier et al., Jan 2025; den Hengst et al., Aug 2025) | Prediction sets made of taxonomy nodes, with coverage guarantees; the second includes an audio benchmark | Assumes calibration data resembles test data; not tested under sensor shift |
| Conformal prediction for unseen domains (Zou and Liu 2024; Siahkali et al. 2026; Wang et al. 2025) | Shows (on simulated data) that standard conformal loses coverage on unseen domains, proposes repairs under assumptions | Not tested on audio species classification with a real sensor change |

**Our claim, narrowed.** To the best of our searches, none of the BioDCASE system summaries describes set-valued, guaranteed answers or adapting to unlabeled new-site audio. But the *pieces* are not new: taxonomy-aware sets exist, and conformal failure under domain shift is known. What we offer is:

- the measurement of how badly standard calibration fails on a real, severe sensor shift in bioacoustics,
- the diagnosis that the failure is a sensor-to-species shortcut, with the evidence,
- a calibration protocol that builds "pseudo-unseen" groups from the benchmark's own structure, and
- a combined system, with an honest account of where its promise holds and where it does not.

That is a strong workshop or applied-venue paper, not a new-theory paper. These searches were not exhaustive; a fuller literature check is still needed before the paper is written.

---

## 6. What we are solving

**Main question:** Can a mosquito sound classifier give *honest, set-valued answers* that stay correct on recordings from devices and places it has never seen?

We break it into four questions:

- **RQ1. Does the promise break?** **Answered: yes.** Standard methods give 17% to 36% real coverage on new domains at a 90% target (section 2.1). A 2026 study on machine sounds and a 2024 study on simulated data saw the same kind of failure, but nobody had checked it for mosquitoes.
- **RQ2. Can we repair it?** **Partly (Phases 2 to 4).** Calibrating on held-out groups restores the promise on the unseen groups (coverage 0.93 to 1.00) but only with sets of 7 to 9 of the 9 species, because the classifier is near chance on the new sensor. Audio-level augmentation softens the sensor shortcut a little (+2.5 ± 0.7 points on held-out groups over 5 seeds, nothing on the official unseen test); projection, alignment, a second encoder and fusion did not help. What does work, in a bounded way: calibrating separately per detected sensor type gives 0.912 coverage at 2.64 species on the whole test, 24% to 46% smaller sets than one global threshold at matched coverage, while unseen groups still get 8 to 9 species.
- **RQ3. How precise can we be?** **Answered: not very on a new sensor.** At a 90% promise, genus-level and taxonomy-lifted sets are also near-vacuous on the unseen groups (worst-group genus sets hold 2.95 of 3 genera). On the sensor the model knows (D5), sets average 1.5 species at 0.90 coverage.
- **RQ4 (stretch). How much does a little new-site audio help?** **Labelled part answered: not enough.** 5 to 50 labelled clips per species keep coverage near 0.90 with sets of about 7 of 9 species. Unlabeled adaptation was not built.

**Outcome.** The premise held and the full repair did not work, so the project is a measured, diagnosed benchmark study of how far calibrated honesty can go under sensor shift, with one bounded practical result (sensor-conditional calibration).

---

## 7. Our approach in plain words

The full specification, with layer shapes, parameter counts and the reason for every choice, is in `architecture-next-phases.md`. In short:

1. **Same-shaped input for every clip.** Cut every recording to the loudest 0.62 s pieces (the size of a D5 clip), so length and padding cannot give away the sensor.
2. **Strong frozen features.** Use pretrained audio models (AST, Bird-MAE, and Perch 2.0 if we can get it) as fixed feature extractors, plus 53 pitch and harmonic measurements. We train only a small network (1.36 million parameters) on top.
3. **Sensor compensation, aimed at the measured shortcut.** (a) Add microphone background noise and filtering to D5 recordings during training, so "microphone noise" no longer implies three particular species. (b) Subtract the directions in feature space that mostly encode "which sensor", estimated from the three bridge species. (c) Penalise differences between sensors *within the same species*. The challenge winner's plain alignment (CORAL, MMD) also aligns the species mix, which is wrong here, so we only keep it as a comparison.
4. **Set-valued answers.** Use *conformal prediction*: a method that, using a calibration set, decides how large the answer set must be so the true species is inside it at least 90% of the time.
5. **Calibration on held-out groups.** Build nine "pseudo-unseen" groups from Training and Validation by hiding all of a species' microphone clips, or one microphone domain of a species, then calibrate on how the model does on them. Use the worst group's threshold, which gives a rough "about 9 times in 10" statement about a new group if groups behave alike. That is an assumption, not a theorem.
6. **Taxonomy levels.** Let the answer be a species, a sibling group (for example "An. gambiae complex") or a genus, built by widening the species set up the family tree, which keeps the promise intact.
7. **Fair testing.** Every choice is made on the pseudo-unseen groups, never on the official unseen test. Report every number over several random seeds, and report coverage for each of the nine official unseen groups, not only an average.

**Methods we compare against:**

- the official CD-MSC baseline with plain conformal calibration (done in Phase 1),
- the AST probe with plain conformal calibration (done in Phase 1),
- a plain Perch probe, if we can get Perch,
- a reproduction attempt of the winning pipeline (not the exact system),
- standard (clip-level) conformal prediction.

---

## 8. Why we are doing it

- **Public health.** Mosquito surveillance is costly and sparse. A sound-based system is only valuable if it works on a new phone in a new village, and only trustworthy if it says when it is unsure.
- **Honesty.** A system that guesses one species when two sound identical gives false confidence. A set with a measured promise is a more truthful answer.
- **Science.** It tests, on a real benchmark, a general claim: that promises about uncertainty break under sensor shift, why they break, and how much a careful design can repair.
- **Practical for us.** The benchmark is ready-made, has a public leaderboard and public baseline code, and our approach needs only small models on frozen features. Phase 0 and 1 already ran on a laptop GPU.

---

## 9. Scope

**In scope**

- Species classification from short audio clips on the CD-MSC data.
- Set-valued predictions with calibration across domains.
- Analysis of which species can and cannot be separated by sound.
- (Stretch) adaptation with unlabeled or few labelled clips from a new domain.

**Out of scope**

- Building hardware, apps or any deployment.
- Predicting whether a mosquito carries a disease. Species is not infection.
- Disease-risk maps or outbreak prediction.
- Claiming results "for India". The data includes India-relevant species (*An. stephensi*, *An. minimus*, *Ae. aegypti*, *Ae. albopictus*, *Cx. quinquefasciatus*) but has no Indian recordings. At most we can say "India-relevant species, not India-validated".

**Rules we follow** (so results stay comparable to the challenge): pretrained models are allowed, external *labelled mosquito* data is not, and the evaluation set is used only for final testing.

---

## 10. The data (now measured)

- **Benchmark:** BioDCASE 2026 CD-MSC. Development set on Zenodo (record 19095788), downloaded and checked (3.8 GB zip). The evaluation set (471 MB zip) is also downloaded; its file names carry no labels. The baseline README links Zenodo record 20478577 for evaluation data, but that record number is still not confirmed.
- **Size:** 271,380 clips, 60.66 hours, 9 species, 5 domains (D1 to D5). All clips on disk.
- **Species and clips:** *Ae. aegypti* 81,587, *Ae. albopictus* 18,517, *Cx. quinquefasciatus* 72,056, *An. gambiae* 46,998, *An. arabiensis* 21,117, *Cx. pipiens* 29,754, *An. minimus* 550, *An. stephensi* 674, *An. dirus* 127.
- **Domains:** D5 holds 265,652 clips (about 98%), D1 to D4 hold 4,065, 784, 679 and 200.

| Domain | Clip length (mean, sampled) | Sampling rates seen | Species present (all splits) |
|---|---|---|---|
| D1 | 7.5 s (0.04 to 54 s) | 8 kHz, some 44.1 kHz | all 9 species |
| D2 | 2.7 s (0.4 to 35 s) | 8 kHz | *Ae. albopictus*, *An. minimus*, *An. stephensi*, *Cx. pipiens* |
| D3 | 12.4 s (2 to 36 s) | 8 kHz | *Ae. aegypti*, *Ae. albopictus*, *An. minimus*, *An. stephensi*, *Cx. pipiens* |
| D4 | 50.5 s (0.2 to 294 s) | 8, 44.1, 48 kHz | *Ae. aegypti*, *An. dirus*, *An. minimus*, *An. stephensi*, *Cx. quinquefasciatus* |
| D5 | exactly 0.62 s | 8 kHz | six species: *Ae. aegypti*, *Ae. albopictus*, *An. arabiensis*, *An. gambiae*, *Cx. pipiens*, *Cx. quinquefasciatus* |

- **D5's identity.** Its six species and its clip length (0.62 s, which is 5,000 samples at 8 kHz) match, as far as I recall, the optical-sensor *Wingbeats* dataset. That is an inference, not confirmed.
- **Development test set:** deliberately includes combinations the training set never saw. The official unseen domain per species is *Ae. aegypti* D3, *Ae. albopictus* D2, *Cx. quinquefasciatus* D1, *An. gambiae* D1, *An. arabiensis* D1, *An. dirus* D4, *Cx. pipiens* D3, *An. minimus* D2, *An. stephensi* D4.
- **Training microphone clips:** D1 634, D2 230, D3 364, D4 80 (1,308 in all, of which 996 are the three microphone-only *Anopheles*).
- **Metrics used by the challenge:** balanced accuracy on seen domains, on unseen domains (the main score), and the gap between them.

**Not yet confirmed:** which real dataset D1 to D4 are, and whether the evaluation labels will be published.

---

## 11. How we will judge success

These are targets, not promises. Status as of 2026-10-08.

| # | Target | Status |
|---|---|---|
| 1 | **Reproduce the baseline.** | **Done.** 0.175 ± 0.019 unseen (README 0.185). The planned "frozen Perch probe near 0.3" was replaced by an AST probe at 0.288, because Perch needed a login at the time (it was later run through a community port: 0.26) |
| 2 | **Test RQ1.** With a 90% target, ordinary calibration covers visibly less than 90% on unseen domains (a shortfall of more than 5 points). | **Done, passed:** shortfalls of 54 to 74 points in three of three classifier families |
| 3 | **Test RQ2.** Our domain-aware version gets within about 2 to 3 points of 90% coverage on unseen domains. | Open. Revised target in the architecture document: macro coverage at least 0.85, with at least 7 of the 9 official unseen groups at 0.80 or higher |
| 4 | **Keep answers useful.** The sets stay small enough to be informative (average set size, and how often answers are at species, group or genus level). | Open. Phase 1 shows the trap: coverage can be bought with sets of 8 of 9 species |
| 5 | **Be honest where it fails.** Report which species cannot be resolved. | Partly seen: sibling confusion is real inside D5 |

---

## 12. Risks and honest limits

- **Only five domains.** A guarantee across domains needs many domains. Our nine pseudo-unseen groups are the best we can build, so claims are empirical rather than theorems.
- **Lopsided data.** The microphone domains are tiny, and the pseudo-unseen groups hold only 1,391 clips between them. Results can be noisy, and we show per-group intervals.
- **The pseudo-unseen groups may not look like the official ones.** Official unseen pairs are mostly D5-to-microphone, the same shift as only 3 of our 9 groups. Whether calibration transfers is the central question of Phase 2c.
- **Evaluation labels may stay hidden.** Then we cannot compare directly with the leaderboard and must rely on the development test set, which we have already looked at. The winner's own split gave 0.362 while the official score was 0.313, so the two are not identical.
- **Competition.** The organisers and challenge teams may publish follow-ups. We found no set-valued approach, but cannot rule it out.
- **Sibling species.** Some species may be inseparable by sound. That is a finding, not a failure, but it limits headline accuracy.
- **Perch.** Run later through a community PyTorch port (no login); it did not help. The port was not verified against the official model.
- **Lab to field.** Even a good result here does not prove it works in a village.

---

## 13. Plan

| Phase | What happens | Gate to continue | Status |
|---|---|---|---|
| 0. Setup and reproduce | Download the data, run the official baseline | Baseline reproduces | **Done, passed** |
| 1. Test the premise | Apply standard calibration and measure real coverage on unseen domains | Coverage falls short by more than 5 points | **Done, passed** (54 to 74 points) |
| 2a. Better features | Same-shaped input; AST at 128 frames; Bird-MAE; Perch if available; pitch features; fusion head | Unseen balanced accuracy at least 0.32 (otherwise continue and report) | **Done, missed** (best 0.27). Tiled input lost to padded; Bird-MAE and fusion no better; Perch and pitch-feature fusion not run |
| 2b. Sensor compensation | Noise-swap augmentation, nuisance-direction projection, per-species alignment | Gain of at least 3 points on pseudo-unseen groups | **Done, borderline** (5 seeds: +2.5 ± 0.7 linear head, +3.3 ± 1.2 MLP head on held-out groups, official unseen test unchanged; projection and alignment gave nothing) |
| 2c. Calibration and sets | Worst-group thresholds, taxonomy-lifted sets, comparison with plain calibration | Macro coverage at least 0.85 on unseen, sets not vacuous (otherwise report the coverage-versus-size curve) | **Done: coverage met, non-vacuity missed.** Curve reported |
| 2d. Full comparison | All systems, 5 seeds, per-group analysis | | **Partly done**: AST, Bird-MAE and fusion compared with 2 seeds; no winner-style reproduction; 5 seeds not run |
| 3. Stretch | Adaptation with unlabeled or few labelled clips from a new domain | Measurable gain | **Labelled part done, gain missed** (sets stay at about 7 of 9). Unlabeled adaptation not built |
| 4. Write-up | Results, ablations, honest limitations | | **Draft written** (`paper-draft.md`); needs more seeds, intervals and a literature check |

---

## 14. Things we still need to find out

**Settled since v1**

- Download size and Colab or Drive space: not needed, because everything runs locally (about 9 GB of data plus a 7 GB Python environment).
- Whether the data holds what the papers say: checked in Phase 0.
- Whether a literature check turns up anything we missed: partly done, and it narrowed our claim (section 5).
- Access: AST and Bird-MAE are public; Perch 2.0 was run through a community port (no login needed).

**Still open**

- Which real recording source each domain (D1 to D4) is, and whether D5 is the Wingbeats data.
- Whether the evaluation labels will be released (not pursued; the project closed without them, so claims about the official unseen test are exploratory).
- Whether the community Perch port matches the official model (the maintainer says it does; unverified here).
- A fuller literature search before the paper.

---

## 15. Glossary

- **Domain / domain shift:** the recording condition (place, device, environment), and what happens when it changes between training and testing.
- **Balanced accuracy:** accuracy averaged over species, so rare species count as much as common ones.
- **Seen / unseen domain:** for a given species, a domain that is in / not in its training data.
- **Embedding:** a list of numbers a large pretrained model produces to describe a clip.
- **Frozen features / probe:** using a pretrained model as is and training only a small classifier on top.
- **Shortcut learning:** getting the right answer for the wrong reason, such as recognising the microphone.
- **Conformal prediction:** a way to give a set of answers with a measurable promise about how often the truth is inside it.
- **Coverage:** how often the true species really is inside the answer set.
- **Set size:** how many species are in the answer set; smaller is more informative.
- **Species complex:** a group of closely related species that look and sound almost identical.
- **Optical sensor:** a device that records wingbeats as changes in light instead of sound pressure.
- **Leave-one-domain-out:** training without one domain and testing only on it.
- **Bridge species:** a species that appears in both the optical-like D5 and at least one microphone domain (*Ae. aegypti*, *Ae. albopictus*, *Cx. pipiens*). They are our only direct evidence of what a sensor change does to a species.
- **Pseudo-unseen group:** a species-domain combination that we deliberately hide from training (using only Training and Validation data) so we can see how the model and its promise behave on something new.
- **Worst-group threshold:** the strictest of the per-group thresholds, so the promise holds for the hardest group we have seen.
- **Nuisance-direction projection:** removing from the feature vectors the directions that mostly carry "which sensor" rather than "which species". Borrowed from speaker recognition, where the "channel" must not be confused with the speaker.

---

## 16. References

- BioDCASE 2026 CD-MSC baseline paper: https://arxiv.org/abs/2603.20118
- Challenge task page: https://biodcase.github.io/challenge2026/task5
- Challenge results and technical reports: https://biodcase.github.io/challenge2026/task5-results
- Baseline code: https://github.com/Yuanbo2020/CD-MSC
- Winning technical report (JKU Linz): https://biodcase.github.io/documents/challenge2026/technical_reports/Shestakov_task5.technical_report.pdf
- HumBugDB: https://arxiv.org/abs/2110.07607
- DR-BioL (domain-robust learning): https://arxiv.org/abs/2510.00346
- MosquitoSong+: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0310121
- BEANS benchmark: https://arxiv.org/abs/2210.12300
- Audit of risk control on signal detectors under domain shift: https://arxiv.org/abs/2606.15153
- Perch 2.0: https://arxiv.org/abs/2508.04665
- Bird-MAE ("Can Masked Autoencoders Also Listen to Birds?"): https://arxiv.org/abs/2504.12880
- Hierarchical conformal prediction, Mortier et al. 2025: https://arxiv.org/abs/2501.19038
- Hierarchical Conformal Classification, den Hengst et al. 2025: https://arxiv.org/abs/2508.13288
- Conformal prediction for out-of-distribution data, Zou and Liu 2024: https://arxiv.org/abs/2403.19950
- Pseudo-calibrated conformal prediction under distribution shift, Siahkali et al. 2026: https://arxiv.org/abs/2602.14913
- Conformal prediction under unknown subpopulation shifts, Wang et al. 2025: https://arxiv.org/abs/2506.05583
- Audited Conformal Prediction under unknown distribution shift, 2026: https://arxiv.org/abs/2606.14909
- Weighted conformal classification with covariate shift and posterior drift: https://arxiv.org/abs/2502.17744
- Fine-tuning can distort pretrained features, Kumar et al. 2022: https://arxiv.org/abs/2202.10054
- Class-conditional conformal prediction with many classes, Ding et al. 2023: https://arxiv.org/abs/2306.09335
- Abuzz (phone recordings of mosquitoes): https://datadryad.org/dataset/doi:10.5061/dryad.98d7s
- Our results: `results/overnight_report.md`, `results/phase1/phase1_report.md`, `results/phase1/diagnostics.md`
