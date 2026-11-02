# Architecture of the Next Phases (Phase 2 to Phase 4)

**Project:** Trustworthy mosquito identification from sound, on recordings from sensors the model has never seen
**Benchmark:** BioDCASE 2026 Cross-Domain Mosquito Species Classification (CD-MSC)
**Status:** Design v1, written 2026-10-08 after Phase 0 and Phase 1, before any Phase 2 code. **Outcome (2026-10-30): this is the plan that was followed, not a description of a working system.**
**Read with:** `problem-statement.md` (what and why), `results/overnight_report.md` (Phase 0 and 1 numbers), `results/phase1/diagnostics.md` (evidence for the design, produced by `src/phase1_diagnostics.py`)

> **What actually happened (2026-10-30).** Built and run: the AST branch (canonical-tiled 128-frame input and the Phase 1 padded input), a 4-tap MLP + cosine head, the genus loss, nuisance-direction projection, per-species alignment, noise-swap and EQ augmentation with per-fold noise banks, Bird-MAE, late fusion, the nine pseudo-unseen folds, the calibration rules C0, C1, C2, C2b, genus-level rules and few-shot recalibration. **Not built:** Perch 2.0 (login needed), the pitch-feature branch in the fusion head, concatenation fusion (the AST and Bird-MAE augmented views used different random noise, so fusion was late), the adversarial sensor head, class-conditional C3, unlabeled re-centering, and 5 seeds (2 were used). Results are in `results/phase2_report.md`.
>
> **What changed from the plan.** (1) The canonical tiled input, the central choice of section 3.1, **lost** to the padded input, which was kept. (2) Projection (5.2) and alignment (5.3) gave no gain; audio-level augmentation (5.1) was the only tool that helped. (3) Group calibration (section 7) restores coverage but yields near-vacuous sets, because the classifier is near chance on some held-out groups. (4) Gates 2a (0.32) and 3 were missed, 2b was borderline (5 seeds), and 2c met coverage but not non-vacuity. The sections below are the original design and are unchanged.

**How to read the tags.** Every justification below carries one of these, so you can see what is solid:

- **[M]** measured by us in Phase 0 or 1, numbers are in the files above
- **[L]** taken from a paper or model card (references in section 14, with how well each was checked)
- **[H]** a hypothesis we will test, not a fact
- **[A]** an assumption we are making and will state in the paper

---

## 1. What the architecture has to fix

Phase 0 and 1 told us where the system breaks. Each row is evidence and the design decision it forces.

| # | What we saw | Tag | What it forces |
|---|---|---|---|
| 1 | At a 90% target, standard conformal sets cover only 16.5% (baseline CNN), 36.2% (pitch features) and 35.5% (AST probe) of unseen-domain clips, while seen-domain coverage is on target (0.88 to 0.90) | M | Calibration must see a domain change, **and** the classifier must carry real signal on unseen sensors |
| 2 | Unseen D1 microphone clips of *Cx. quinquefasciatus*, *An. gambiae* and *An. arabiensis* are labelled as the **microphone-only Anopheles** (*minimus*, *dirus*, *stephensi*): the AST probe sends at least 67% of *Cx. qui* and at least 53% of *An. gam* there, and the released baseline sends 88% of *An. gam* to *An. stephensi* alone | M | The model learned "microphone-like sound means one of three Anopheles". We must break the link between sensor and species |
| 3 | 1,138 of the 1,490 microphone clips in Train+Validation (76%) belong to three species that have **no** D5 recordings (*An. dirus*, *An. minimus*, *An. stephensi*). 99.4% of Training clips are D5 | M | The label mix differs by sensor. Aligning all D5 and microphone clips together (plain CORAL or MMD) would also align the label mixes, which is wrong. Alignment has to be per species, or learned from paired species |
| 4 | Only **three bridge species** (*Ae. aegypti*, *Ae. albopictus*, *Cx. pipiens*) have both D5 and microphone clips: 340 microphone clips in Train+Validation (*Cx. quinquefasciatus* adds 12). These are our only direct evidence of what a sensor change does to a species | M | Use them twice: to estimate the sensor effect, and to build "pseudo-unseen" test folds |
| 5 | The AST embedding separates D5 from microphone clips with 0.989 balanced accuracy (AUC 0.999) | M | The sensor is strongly present, and linearly readable, in the features. It can be measured and partly projected out |
| 6 | The Phase 1 AST probe put every 0.62 s D5 clip into a 10.24 s window (about 94% padding), while microphone clips (mean 2.7 to 50 s) filled most of it | M for the arithmetic, H for the effect | Give every clip the same canonical input, so length and padding cannot identify the sensor |
| 7 | Genus-level sets cover far more than species-level sets: 0.659 against 0.362 (pitch features) and 0.567 against 0.355 (AST probe) | M | Taxonomy-aware answers, and a genus-level training signal |
| 8 | A handful of labelled clips per species restored 0.90 coverage, but the sets held about 8 of 9 species | M | The classifier must improve first, or few-shot calibration only produces useless sets |
| 9 | Inside D5, *An. gambiae* and *An. arabiensis* are confused with each other about 21 to 22% of the time but each keeps about 65% recall. In D1 they mostly go to *other* species (*An. minimus*, *Ae. albopictus*, *Ae. aegypti*) | M | The sibling pair is a real but secondary problem. The main failure is the sensor shortcut (row 2) |
| 10 | A probe trained *inside* D1 separates *Cx. qui*, *An. gam* and *An. ara* at 0.954 balanced accuracy (5-fold CV) | M, but optimistic: clips from one recording leak, and different species were probably recorded in different sessions | The information may already be in the features. The gain may come from sensor compensation more than from new features [H]. We will not read this as proof that the sibling pair is acoustically separable |

---

## 2. The system at a glance

```
 audio clip (8 kHz, 0.04 s to 294 s)
        |
  [Stage 0]  canonical input: loudest 0.62 s snippets (up to 4), tiled to each encoder's window
        |
  [Stage 1]  FROZEN encoder bank (no training; embeddings cached to disk)
        |        AST (12-layer ViT)  | Bird-MAE (12-layer ViT)  | Perch 2.0 (EfficientNet-B3) | 53 pitch/harmonic features
        |        4 layer taps each   | 4 layer taps each        | 1 embedding                  |
        |
  [Stage 2]  sensor compensation (training time + embedding space)
        |        background-swap + EQ augmentation   |   nuisance-direction projection (NAP)
        |
  [Stage 3]  SMALL TRAINABLE HEAD (1.36 M parameters)
        |        per-encoder adapters -> fusion MLP -> 256-d species embedding -> cosine classifier (9 species)
        |        losses: species + genus + complex + class-conditional alignment
        |        5 seeds, 9 fold-models each -> ensemble
        |
  probabilities over 9 species  (genus, complex and root probabilities by summing)
        |
  [Stage 4]  CALIBRATION on pseudo-unseen groups (not on ordinary validation clips)
        |        score -> worst-group threshold -> species set -> lifted to taxonomy nodes
        |
  answer: a set of taxonomy nodes, e.g. {An. gambiae complex}  or  {Aedes}  or  {Cx. pipiens}
        |
  [Stage 5]  (stretch) site adaptation: unlabeled re-centering, few-shot prototypes + recalibration
```

**In plain words.** Big pretrained audio models act as ears: they turn a clip into a list of numbers. We never change them. A small network that we do train reads those numbers and names the species. The usual way this goes wrong is that the small network learns the *microphone* instead of the *mosquito*. So we (a) make every clip look structurally identical on input, (b) show the network D5 recordings with microphone-style noise and filtering added, (c) subtract the directions in feature space that mostly encode "which sensor", and (d) penalise differences between sensors within the same species. Last, a separate calibration step decides how many species to list so that the true one is inside the answer 90% of the time, and it learns that from held-out *sensor changes*, not from ordinary clips.

---

## 3. Stage 0: Data handling and folds

### 3.1 Canonical input (fixes row 6)

1. Load the wav at 8 kHz, resample with a polyphase filter to each encoder's rate (AST 16 kHz, Bird-MAE 32 kHz, Perch 32 kHz). Content above 4 kHz is empty, which does no harm.
2. Peak-normalise the clip (as in Phase 1).
3. D5 clips are exactly 0.62 s, so the snippet is the whole clip. For longer clips take sliding 0.62 s windows (hop 0.31 s), score each by energy in 100 to 2,000 Hz, and keep the **top 4 non-overlapping windows**. At test time the clip embedding is the mean of the snippet embeddings. Four is a cap that bounds compute: the longest clips run to 294 s.
4. Fill each encoder's window by **tiling** the snippet with 20 ms cross-fades. The wingbeat is a steady tone, so tiling is a natural way to fill a window. We test it against zero-padding (ablation A1).
5. AST window: **128 frames (1.28 s)** instead of 1024 (10.24 s). We crop the time axis of the learned position embeddings from 101 to 12 patch columns, which is how the AST authors handle shorter inputs [L, verify the exact recipe when coding]. Tokens fall from 1,212 to 144, roughly an 8x saving.
6. Bird-MAE and Perch use fixed 5 s windows (tiling fills them).

**Why:** a model must not be able to tell the sensor from clip length or padding [M for the arithmetic, H for the effect]. This also matches the benchmark's native unit, the 0.62 s D5 clip.

### 3.2 Splits and the pseudo-unseen folds (fixes the missing validation signal)

The official Validation split contains only *seen* species-domain pairs, so it cannot show how a model behaves on a new sensor. The challenge winner's report names exactly this gap. We build our own signal from Training+Validation, which holds 1,490 microphone clips, using the official definition of "unseen": the species is present in other domains but this species-domain pair is absent.

| Fold type | Held-out group (all its clips removed from training) | Clips | Mimics |
|---|---|---|---|
| **LSMO** (leave one species' microphones out), 3 folds | all microphone clips of *Ae. aegypti* (D1 111 + D4 20) | 131 | the species is known only from D5, then heard on a microphone, as with *An. gam*, *An. ara*, *Cx. qui* in D1 |
| | all microphone clips of *Ae. albopictus* (D1 73 + D3 17) | 90 | |
| | all microphone clips of *Cx. pipiens* (D1 59 + D2 60) | 119 | |
| **LODO** (leave one domain out), 6 folds | *An. minimus* in D1 / D3 / D4 | 200 / 200 / 51 | a species known from other microphones, heard on a new one (microphone to microphone) |
| | *An. stephensi* in D1 / D2 / D3 | 200 / 200 / 200 | |

That is **9 groups and 1,391 pseudo-unseen clips**. Not usable: *An. dirus* (only D1 in Train+Val) and *Cx. quinquefasciatus* (12 clips). The 3 LSMO folds are the closest we can get to the official worst case (D5 to microphone), but the 6 LODO groups are a different kind of shift (microphone to microphone), and the official unseen pairs are mostly the first kind. Whether calibration learned on these 9 groups transfers to the official ones is the central empirical question of Phase 2c [A, H].

**Official dev test.** The unseen test pairs (4,202 clips, 9 official groups) are used for final reporting. **Disclosure:** we already looked at their predictions in Phase 1 and in `diagnostics.md`, so that set is no longer untouched. To limit leakage, all configuration choices are made on the 9 pseudo-unseen groups only, and the pipeline is fixed in this document before it is run on the official unseen test clips. If the organisers release evaluation labels, they become the clean confirmation set.

### 3.3 Sampling and class balance

- **Pair-balanced sampling.** A batch picks a (species, domain) pair with probability proportional to the square root of its size, then a clip from that pair. This lifts microphone clips from 0.6% of the data to about 25% of each batch, without drowning the D5 pairs.
- **Species-balanced loss.** Because the microphone-only species (*An. dirus*, *An. minimus*, *An. stephensi*) get 2 to 7 times less sampling weight than the others, each species gets a loss weight that makes every species contribute equally.
- **D5 cap.** 4,000 D5 clips per species for training (24,000 in total). D5 is 98% of the data and extra clips add little.

---

## 4. Stage 1: The frozen encoder bank

All encoders are frozen. We run each once, in fp16 on the RTX 4060, and cache the outputs.

| Encoder | Architecture | Input it expects | Pretraining | Output we keep | Role | Access |
|---|---|---|---|---|---|---|
| **AST** (`MIT/ast-finetuned-audioset-10-10-0.4593`) | ViT-B: 12 layers, hidden 768, 12 heads, 16x16 patches with stride 10, about 87 M parameters | 128-mel log fbank at 16 kHz, 1024 frames by default (we use 128) | supervised on AudioSet, 527 classes, including **Insect, Mosquito, Buzz** [M: read from the model config] | masked mean of patch tokens at layers **3, 6, 9, 12**, 4 x 768 | already works (balanced accuracy 0.288 unseen); the AudioSet "Mosquito" class means it has heard insect buzzing | public, no login, already cached on this machine |
| **Bird-MAE** (`DBD-research-group/Bird-MAE-Base`) | ViT-B/16 masked autoencoder, 12 layers, hidden 768 [L] | 32 kHz audio (model card), mel front end from its feature extractor; the 5 s window length is from memory, verify | self-supervised on the large BirdSet bird dataset [L] | patch-token mean at layers 3, 6, 9, 12, 4 x 768 | self-supervised features keep low-level spectral detail; the winner used it | public on Hugging Face, loads with `trust_remote_code` (the model card says to check it for errors) |
| **Perch 2.0** | EfficientNet-B3, about 12 M parameters, embedding of 1,536 dimensions [L] | 32 kHz, 5 s, 128 mel bins, 500 frames [L] | supervised multi-taxa bioacoustics [L, check insect coverage] | the 1,536-d mean embedding | the winner's best single encoder (0.318 alone) | **not yet confirmed**: the official path on Hugging Face returned a login error; the official release is on Kaggle; an unofficial PyTorch port exists but is unverified. **Needs you** (section 14) |
| **Pitch/harmonic features** (`src/m2_handcrafted.py`) | no network: 53 numbers (fundamental, 6 harmonic amplitudes, harmonic-to-noise ratio, spectral shape, amplitude-modulation depth) | the same 0.62 s snippet at 8 kHz | physics of the wingbeat | 53-d vector | independent view, cheap | already built |

**Why frozen, not fine-tuned?**
- The microphone data is tiny (1,308 training clips) and 8 GB of GPU memory is small. [M]
- Fine-tuning a whole network on data dominated by one sensor is the easiest way to learn the sensor shortcut, and fine-tuning is known to distort pretrained features and hurt out-of-distribution accuracy (Kumar et al., ICLR 2022) [L]. The challenge winner also used frozen encoders [L].
- Frozen encoders let us cache every embedding once. All later experiments (45 models per configuration) then train in minutes.

**Why three different encoders?** Different pretraining gives different errors: supervised on general sound events (AST), self-supervised on birds (Bird-MAE), supervised multi-taxa (Perch). Combining them reduced variance for the winning team [L]. If one adds nothing in ablation A3, we drop it.

**Why four layer taps?** The last layer of an AudioSet-trained network is tuned to its 527 labels. Middle layers of a ViT keep more of the raw spectral structure, and a wingbeat is a low-level periodicity cue [H]. A learned weighted sum over taps costs 4 numbers per encoder and tells us which depth matters, which is a result in itself. Ablation A2 tests it against the last layer alone.

**Why mean of patch tokens?** The global token of a model trained for 527 labels carries label-specific bias. A mean over valid (non-padding) patch tokens is the standard probing choice for masked autoencoders. A patch-level attention head (the "prototypical probe" in the Bird-MAE paper helps a lot on birds [L]) is a Stage-3 upgrade held back until the cheap version is shown to plateau, because caching patch tokens would cost tens of gigabytes.

**Storage.** 4 taps x 768 x 2 bytes is 6 KB per clip per ViT encoder, so about 150 k passes cost under 1 GB per encoder.

---

## 5. Stage 2: Sensor compensation

Three tools, each aimed at one row of section 1. They are tested one at a time (ablations B1 to B3), not stacked on faith.

### 5.1 Augmentation that simulates a sensor change (rows 2 and 3)

Each D5 training clip gets 4 augmented versions *before* the encoder (so the encoder sees the change):

- **Background swap.** Add 0.62 s of background taken from a microphone recording, at a random signal-to-noise ratio of 0 to 25 dB. Backgrounds come from the lowest-energy 30% of frames of microphone Training clips, tagged by domain (the same idea M2 already uses to separate foreground from background). **This directly targets the shortcut in row 2.** At present D1 background noise appears only with *Ae. aegypti*, *Ae. albopictus*, *An. dirus*, *An. minimus*, *An. stephensi* and *Cx. pipiens*. If we mix it into *An. gambiae*, *An. arabiensis* and *Cx. quinquefasciatus* training clips, "D1 background" stops implying those six species.
- **Frequency response.** Random spectral tilt of up to 6 dB, a random low cut (50 to 200 Hz), a random high cut (2.5 to 4 kHz), as a stand-in for microphone and optical responses.
- **No large pitch shifts.** The wingbeat frequency is the species cue, and temperature moves it by a few percent at most. We allow at most 3% [A].

**Leakage rule.** In every pseudo-unseen fold, the background bank excludes recordings of the held-out group. So views are generated per fold, using AST on 500 D5 clips per species x 2 views (about 6,000 passes per fold-model), and a final set of 2,000 per species x 4 views (48,000 passes). Augmentation ablations run on AST first and move to the other encoders only if they pay off.

### 5.2 Nuisance-direction projection, "NAP" (rows 3, 4 and 5)

This borrows a standard idea from speaker verification, where a *channel* (the phone line) must not be confused with the *speaker*. Here the sensor is the channel and the species is the speaker. Nuisance Attribute Projection removes the feature-space directions that vary with the channel within one class [L: Solomonoff, Campbell and Boardman, Odyssey 2004 and ICASSP 2005; checked 2026-11-02].

1. For each bridge species s and each microphone domain d where it exists, compute the mean embedding in that domain minus its mean embedding in D5: `delta(s,d) = mean(s,d) - mean(s,D5)`. That gives 6 vectors (*Ae. aeg* D1 and D4, *Ae. alb* D1 and D3, *Cx. pip* D1 and D2).
2. Take the top **k** principal directions of these vectors (k = 0, 2 or 3, chosen on the pseudo-unseen folds). Six vectors cap the rank, so k is tiny by design.
3. Project every embedding with `P = I - U U^T`, per encoder and per tap, before the adapter.

**Why this and not plain CORAL or MMD:** the deltas are computed *within a species*, so they do not include the label-mix difference from row 3 [A: the sensor offset is roughly the same for all species]. It has no trainable parameters, so it cannot overfit the 340 bridge clips beyond choosing k. **In an LSMO fold** the held-out species is removed from the estimate too, leaving only 4 delta vectors. That makes the fold a fair test of whether the sensor effect transfers to a *new species*.

### 5.3 Class-conditional alignment loss (row 3)

During training, for each species that has both D5 and microphone clips in the batch, penalise the difference between the mean and covariance of its D5 and microphone embeddings in the 256-d space (a CORAL-style loss, applied per species). Variants tested in B3: none, plain (all clips together; the winner's choice, expected to be harmful here), class-conditional (ours), and an adversarial sensor head with gradient reversal (16,578 extra parameters).

---

## 6. Stage 3: The trainable head

### 6.1 Layer by layer

Shapes are per clip. "Params" counts weights and biases.

| # | Layer | Input to output | Params | Notes |
|---|---|---|---|---|
| **AST branch** | | | **198,404** | |
| A1 | LayerNorm per tap (shared) | 4 x 768 to 4 x 768 | 1,536 | |
| A2 | Learned weighted sum over 4 taps | 4 x 768 to 768 | 4 | softmax weights; reports which depth matters |
| A3 | Linear | 768 to 256 | 196,864 | |
| A4 | GELU, Dropout 0.2 | 256 | 0 | |
| **Bird-MAE branch** | same as A1 to A4 | | **198,404** | |
| **Perch branch** | | | **396,544** | |
| P1 | LayerNorm | 1,536 to 1,536 | 3,072 | |
| P2 | Linear | 1,536 to 256 | 393,472 | |
| P3 | GELU, Dropout 0.2 | 256 | 0 | |
| **Pitch branch** | | | **3,456** | |
| H1 | Standardise (fixed training mean and std) | 53 | 0 | |
| H2 | Linear, GELU | 53 to 64 | 3,456 | |
| **Fusion** | | | **559,488** | |
| F1 | Concatenate the branches | 256 + 256 + 256 + 64 = 832 | 0 | a missing encoder is dropped from the concatenation |
| F2 | LayerNorm | 832 | 1,664 | |
| F3 | Linear, GELU, Dropout 0.3 | 832 to 512 | 426,496 | |
| F4 | Linear | 512 to 256 | 131,328 | output **z**, then L2-normalised |
| **Classifier** | | | **2,305** | |
| C1 | Cosine classifier | 256 to 9 | 2,305 | normalised class weights, one learnable scale `s`; logits = `s * cos(z, w_c)` |
| | **Total trainable** | | **1,358,601** | about 1.36 M, against roughly 185 M frozen parameters |

Counting along one path: 3 weight layers before the classifier (branch Linear, F3, F4) plus the cosine layer. Without Perch the fusion input shrinks to 576 and the total is 830,473.

### 6.2 Why this shape

- **Small and shallow.** There are 1,308 microphone training clips. A deep head memorises the sensor, so the capacity stays modest: 4 weight layers, 1.36 M parameters, dropout 0.2 and 0.3, weight decay 0.05 [M for the data size, H for the size choice]. Ablation A4 compares it with a plain linear probe (what Phase 1 used) to show whether the MLP and fusion earn their keep.
- **One adapter per encoder, then fusion.** Separate adapters stop one encoder's scale from dominating the others, and keep ablations clean (drop a branch, nothing else changes).
- **256-d bottleneck with a cosine classifier.** Bounded logits make the model less prone to extreme overconfidence under shift, and the class weights double as **prototypes** for the few-shot step in Stage 5 [H].
- **No attention head yet.** Held back until the cheap version plateaus (section 4).

### 6.3 Output hierarchy

Taxonomy used (14 nodes):

```
root
 |-- Aedes ........ Ae. aegypti, Ae. albopictus
 |-- Culex ........ Cx. quinquefasciatus, Cx. pipiens
 |-- Anopheles .... [An. gambiae complex: An. gambiae, An. arabiensis], An. dirus, An. minimus, An. stephensi
```

The network emits 9 species probabilities `p`. The probability of any internal node is the sum of its species, so the hierarchy is consistent by construction and needs no extra parameters. *An. gambiae complex* is the only true sibling group in the dataset; the other four *Anopheles* are not sisters of each other in this grouping [A, from the project's own species list].

### 6.4 Loss

```
L = CE_species(p, y)
  + 0.3 * NLL_genus(p, y)        -log of the total probability inside the true genus
  + 0.3 * NLL_complex(p, y)      same, for the 8-cell partition that merges An. gam + An. ara
  + 0.1 * CORAL_class_conditional(z)          (variant B3)
  [+ 0.1 * adversarial sensor loss]           (variant B3, optional)
```

The coarse terms reward putting probability into the right genus even when the species is unclear. Genus is the most transferable signal we measured (row 7). The weights 0.3 and 0.1 are starting values, tuned on the pseudo-unseen folds.

### 6.5 Training recipe

AdamW, learning rate 1e-3, weight decay 0.05, cosine schedule with 2 warm-up epochs, 40 epochs, batch 256, gradient clipping 1.0, fp32 (the head is tiny). Feature-level noise (Gaussian, 0.1 of the feature standard deviation) on the cached embeddings, in addition to the audio-level augmentation. **Early stopping and every hyper-parameter choice use macro balanced accuracy on the held-out pseudo-unseen group**, never the official unseen test. Five seeds per fold.

### 6.6 Ensembling

For each of the 9 folds and 5 seeds we train one model: **45 models per configuration**. The final predictor is the average of the fold models' probabilities. A temperature (one number) is fitted on pseudo-unseen out-of-fold predictions and applied before calibration. Each calibration score comes from the fold model that did **not** see that group (out-of-fold), while the final predictor averages all fold models. Calibration is therefore a little mismatched to prediction, as in cross-conformal methods. We check this empirically on the official unseen groups and report it [A].

---

## 7. Stage 4: Calibration and set construction (the main contribution)

### 7.1 Score

`s(x, y) = 1 - p_y(x)` (the "LAC" score used in Phase 1). Alternatives tested: APS and RAPS scores, which tend to give larger but better-spread sets.

### 7.2 Threshold from pseudo-unseen groups

For each of the G = 9 groups, compute the `(1 - alpha)` quantile `q_g` of its out-of-fold scores, with the usual finite-sample correction.

| Rule | Threshold | What it means |
|---|---|---|
| **C0** (Phase 1) | one quantile over ordinary validation clips | fails: 0.16 to 0.36 coverage on unseen groups |
| **C1** | one quantile over all pseudo-unseen clips pooled | simple; gives larger groups more weight |
| **C2** | `q* = max over g of q_g` | **worst-group** threshold |
| **C2'** | the second-largest of the 9 values | less conservative |
| **C3** | per-species thresholds, shrunk toward the global one (in the spirit of class-conditional conformal prediction with many classes, Ding et al. 2023, which clusters classes with similar scores rather than shrinking toward a global threshold; checked 2026-11-02 [L]) | species-level coverage, which is what macro metrics reward |

**What C2 buys, and what it does not.** If the held-out group's own threshold is exchangeable with the 9 calibration groups' thresholds, the chance that it falls at or below the largest of them is `G/(G+1) = 9/10`. This is the usual order-statistic argument, and it ignores each group's own estimation noise. So the statement we can make is: *"for a new group, with probability about 0.9, the worst-group threshold yields at least 90% coverage"*. It is only as good as the exchangeability assumption **[A]**, which we cannot check with 9 groups, 3 of them from one kind of shift. We state it as a design guide, **not a theorem**, and the evidence we offer is empirical: how many of the 9 official unseen groups reach near-target coverage (section 9).

### 7.3 Taxonomy-aware sets

Two constructions:

- **H1, lift (ours, valid by construction).** Build the species set `S(x)` at the chosen threshold. Return the smallest collection of taxonomy nodes whose species cover `S(x)`: if the set contains both *Ae. aegypti* and *Ae. albopictus*, answer "*Aedes*". A superset can only increase coverage, so the guarantee from 7.2 survives. A rule that switches level based on `x` alone (choose the finest level whose set is small) does **not** keep the guarantee, because the choice depends on the input, so we do not use it.
- **H2, optimised sets** from the hierarchical conformal literature, which pick node sets that minimise size under the coverage constraint (Mortier et al. 2025; den Hengst et al. 2025 [L]). We use their algorithms as comparison and as an upper bound on how small hierarchical sets can get.

Reported for each: coverage, average number of species covered, share of answers that are a single species, resolved to complex, resolved to genus, or the whole root (vacuous).

### 7.4 What is new here and what is not

Taxonomy-aware conformal sets already exist [L: Mortier et al., Jan 2025; den Hengst et al., Aug 2025, which includes an audio benchmark]. So do conformal methods for unseen domains [L: Zou and Liu 2024; Siahkali et al. 2026 on pseudo-calibration; Wang et al. 2025 on unknown subpopulation shifts], and a worst-domain "max" threshold is a known conservative baseline. **We claim none of the pieces.** The contribution is (i) the measurement that standard calibration fails by 54 to 74 points on a real, severe sensor shift in bioacoustics, (ii) the diagnosis that the failure is a sensor-to-species shortcut, (iii) a leak-free pseudo-unseen calibration protocol built from the benchmark's own structure, (iv) the combined system, with an honest account of where the guarantee holds and where it does not. This is closer to a strong workshop or applied-venue paper than a new-theory paper.

---

## 8. Stage 5 (stretch): Site adaptation

Only started if Stage 4 yields informative sets.

- **Unlabeled re-centering.** Use unlabeled audio from the new site (the unlabeled Evaluation_data.zip is on disk) to estimate and remove that site's mean offset in embedding space. **Hazard:** a new site records only a few species, so its mean embedding mixes the sensor offset with a *species mix* offset, exactly the confound of row 3. Guards: estimate the offset only from background frames, or only along the NAP directions, and compare against doing nothing. Results are reported as "transductive" and kept apart from the "inductive" numbers.
- **Few-shot prototypes.** With k = 5, 10 or 20 labelled clips for a species at the new site, move that species' classifier weight toward the mean of those clips' embeddings (a convex combination), then recalibrate the threshold on those clips. Phase 1 showed recalibration alone restores coverage with vacuous sets; with a stronger classifier the sets should shrink.
- **Not planned:** end-to-end test-time training (such as entropy minimisation of the whole network). With 3 species present at a site it tends to collapse.

---

## 9. Evaluation protocol

### 9.1 Systems compared

| ID | System |
|---|---|
| S0 | Official baseline CNN (10 released seeds) + plain split-conformal (Phase 1) |
| S1 | AST probe (Phase 1's M3) + plain split-conformal |
| S2 | Plain Perch probe, if access works |
| S3 | "Winner-style" pipeline: frozen Perch + Bird-MAE, plain CORAL and MMD alignment, no agreement gate. A **reproduction attempt**, not the exact system |
| S4 | **Ours**: Stages 0 to 4 |
| S5 | Ours + Stage 5 variants |

### 9.2 Ablation ladder (one factor at a time, run in this order, each judged on the pseudo-unseen folds)

| ID | Question | Options |
|---|---|---|
| A1 | Does canonical input help? | padded 1024 frames (Phase 1) / padded 128 / tiled 128 |
| A2 | Do layer taps help? | last layer / 4-tap weighted sum |
| A3 | Which encoders pay off? | AST / Bird-MAE / Perch / AST+Bird-MAE / all |
| A4 | Is the MLP worth it? | linear probe / MLP + cosine head |
| A5 | Does the hierarchical loss help? | species only / + genus + complex |
| B1 | Does sensor-style augmentation help? | none / frequency response / + background swap |
| B2 | Does NAP help? | k = 0, 2, 3 |
| B3 | Which alignment? | none / plain CORAL / class-conditional / adversarial |
| C1 | Which calibration? | C0 / C1 / C2 / C2' / C3 |
| C2 | Which sets? | species / H1 lift / H2 |

Choices are greedy and sequential, not a full grid, so about 45 models per configuration at a few minutes each is affordable.

### 9.3 Metrics

- **Classification:** balanced accuracy on seen and unseen pairs, domain shift gap (DSG), so we stay comparable with the challenge.
- **Coverage:** macro (species-averaged) and micro, **plus the coverage of each of the 9 official unseen groups**, because 9 numbers show whether the worst-group threshold really protects the weak groups.
- **Informativeness:** average species-set size, share of singleton answers, share resolved to complex, genus or root, empty-set rate.
- **Calibration of probabilities:** log-loss and expected calibration error, for context only.
- **Uncertainty:** bootstrap over clips within groups, 5 seeds. Several groups are tiny (*An. dirus* 40 test clips, *Cx. pipiens* 68, *An. stephensi* 74), so we show per-group intervals and never headline a tiny group.

### 9.4 Success and failure targets

These are targets, not promises.

| Gate | Target | If missed |
|---|---|---|
| 2a (features) | unseen balanced accuracy at least 0.32 on the dev test (the leaderboard top is 0.313 on the evaluation set; the winner's own dev split gave 0.362) | continue anyway, because the paper is about trustworthy sets and the frontier; report honestly |
| 2b (compensation) | pseudo-unseen balanced accuracy up by at least 3 points over the best 2a model, and not all three D1-unseen species (*Cx. qui*, *An. gam*, *An. ara*) left under 5% recall | keep whichever tool helped; if none, the finding is that frozen features plus these tools do not cross the sensor gap, and we move the weight to Stage 5 |
| 2c (calibration) | macro unseen coverage at least 0.85 at alpha = 0.10, with at least 7 of 9 official unseen groups at 0.80 or higher, and sets not vacuous (average under 6 of 9 species, or under 1.8 of 3 genera) | report the **coverage versus set-size curve** as the main result |
| 3 (adaptation) | few-shot or unlabeled adaptation shrinks sets by at least 20% at equal coverage | report as negative |

---

## 10. Compute and storage budget (this laptop: RTX 4060 8 GB, about 15 GB RAM with little free)

| Item | Estimate | Note |
|---|---|---|
| Encoder passes per encoder | about 150 k clean (24 k train D5 + 1.3 k microphone train + 30.5 k validation + 27.2 k test + up to 4 snippets for long microphone clips + 48 k augmented) plus about 54 k for per-fold views | **estimate**, to be measured on 2,000 clips first |
| Throughput | Phase 1 ran about 22 clips/s for AST (68 k clips in 51 min) | the 128-frame window should be faster; CPU feature extraction was probably the limit (unverified) |
| Time | about 2.5 h per encoder at the Phase 1 rate (about 204 k passes), 7 to 8 h for three; the per-fold views start with AST only, so less in practice | one encoder at a time, because of RAM |
| Cached embeddings | under 1 GB per ViT encoder | fp16 |
| Head training | seconds to minutes per model, 45 models per configuration | GPU or even CPU |
| Calibration and sets | seconds | CPU |

Run one job at a time. Scripts need a main guard and zero DataLoader workers (lessons from Phase 0).

---

## 11. Phases and their order

| Phase | Work | Output |
|---|---|---|
| **2a** Features | canonical input; AST re-extraction at 128 frames; Bird-MAE; (Perch if access); pitch features; ablations A1 to A5 | best fused classifier by pseudo-unseen score |
| **2b** Sensor compensation | augmentation (AST first), NAP, alignment; ablations B1 to B3 | a sensor-robust classifier; the "bridge" evidence |
| **2c** Calibration and sets | C0 to C3, H1 and H2 on the 9 groups; evaluation on the 9 official unseen groups | the main result table and curves |
| **2d** Full comparison | S0 to S4, 5 seeds, intervals, per-group analysis | the paper's central tables |
| **3** Adaptation (stretch) | Stage 5 | RQ4 answer |
| **4** Write-up | ablations, limits, figures | paper draft |

---

## 12. Risks and what would change our mind

| Risk | Effect | Mitigation |
|---|---|---|
| The 9 pseudo-unseen groups are too small or too unlike the official ones | calibration does not transfer | report per-group coverage; this is itself a result |
| The sensor effect differs by species (so NAP and augmentation fail) | little gain over Phase 1 | the ablations will show it; fall back on genus-level answers |
| Perch access fails | lose the winner's best encoder | AST + Bird-MAE are public |
| Sibling species are inseparable | headline accuracy is capped | the complex-level answer is the designed fallback |
| Evaluation labels never released | no clean test | label the dev-test use as exploratory, as in section 3.2 |
| Window crop for AST breaks the pretrained position embeddings | worse features | A1 compares against the 1024-frame setting |
| Guarantee not valid | the claim shrinks | stated as empirical from the outset |
| Results are negative | less exciting paper | still a measured, diagnosed, honest benchmark study (see problem statement) |

---

## 13. Alternatives considered and set aside

- **Fine-tune a spectrogram network end to end.** The microphone data is too small and the shortcut is too easy to learn (section 4).
- **Plain CORAL and MMD over all clips (the winner's alignment).** Aligns label mixes as well as sensors (row 3). Kept only as comparison S3 and ablation B3.
- **Domain-adversarial training only.** Needs enough microphone data per domain to train a discriminator, which we lack; kept as one variant.
- **Transformer or attention head from the start.** More capacity than the data supports. Revisit after the cheap head plateaus.
- **Agreement gate over harmonic features** (the winner's trick). Its benefit was within seed noise [L].
- **Choosing the answer level per clip.** Breaks the coverage guarantee (section 7.3).
- **Zero-shot or text-prompted audio models.** Not suited to this narrow species task, and low payoff.

---

## 14. Open items that need you

1. **Perch 2.0 weights.** The official path on Hugging Face returned a login error. The official release is on Kaggle and may need a Kaggle login. If you want Perch in, please log in or download it yourself and tell me where the files are. Do not paste any key or token into chat. An unofficial PyTorch port exists, but I would not trust it without a numerical check against the official model.
2. **Evaluation labels.** (Not pursued; the project closed without them.) Ask the organisers whether the labels of the evaluation set will be released. A clean test changes how strong our claims can be.
3. **Go-ahead for 2a.** The first real compute is the 128-frame AST re-extraction, with a throughput check on 2,000 clips before the full run.
4. **Which real datasets D1 to D5 are.** Still unconfirmed. It matters for how we describe the shift (is D5 the optical Wingbeats data?).

---

## 15. References (with how well each was checked)

Checked by fetching the page in this session:

- Mortier, Javanmardi, Sale et al. *Conformal Prediction in Hierarchical Classification with Constrained Representation Complexity.* arXiv 2501.19038 (Jan 2025)
- den Hengst, Blin, Mohammadi et al. *Hierarchical Conformal Classification.* arXiv 2508.13288 (Aug 2025); abstract mentions audio, image and text benchmarks
- Zou and Liu. *Coverage-Guaranteed Prediction Sets for Out-of-Distribution Data.* arXiv 2403.19950 (2024); shows split conformal loses coverage on unseen target domains
- Siahkali, Verma and Gupta. *Coverage Guarantees for Pseudo-Calibrated Conformal Prediction under Distribution Shift.* arXiv 2602.14913 (Feb 2026)
- Wang, Yaldiz, Bakman et al. *Conformal Prediction Adaptive to Unknown Subpopulation Shifts.* arXiv 2506.05583 (2025)
- Rauch et al. *Can Masked Autoencoders Also Listen to Birds?* (Bird-MAE) arXiv 2504.12880; Hugging Face model card read
- Perch 2.0 paper: arXiv 2508.04665 (architecture details taken from search results, not read in full)
- AST: `MIT/ast-finetuned-audioset-10-10-0.4593` config read from the local cache (12 layers, hidden 768, 12 heads, 1024 frames, 128 mel bins, labels include Insect, Mosquito, Buzz)

Checked against search results on 2026-11-02:

- Kumar, Raghunathan, Jones, Ma and Liang, *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*, ICLR 2022 (oral), arXiv 2202.10054
- Solomonoff, Campbell and Boardman, *Channel compensation for SVM speaker recognition*, Odyssey 2004, and *Advances in channel compensation for SVM speaker recognition*, ICASSP 2005 (Nuisance Attribute Projection); the original papers were not opened, only MIT Lincoln Laboratory listings and follow-ups
- Ding, Angelopoulos, Bates, Jordan and Tibshirani, *Class-Conditional Conformal Prediction with Many Classes*, NeurIPS 2023, arXiv 2306.09335 (it clusters classes with similar scores)

Still unverified: the AST authors' recipe for cropping position embeddings for shorter inputs, and Bird-MAE's 5 s window length (the model card was read; the feature extractor config says a 512-frame target length at 10 ms hop).
