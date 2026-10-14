# Step 6 — Describe the response: threshold benchmarks (version 2)

Each figure varies one setting of step 6 (the fit of components and background to an
average, and the features read from it); every other setting keeps the preset value.

## Method

- **Data.** CCEP: OpenNeuro ds004080 v1.2.4, 30 patients (`benchmarks/seuils/ds004080_v2_subjects.tsv`).
  They were selected with the rule of `validation/ds004080_subjects.txt`: subjects in
  alphabetical order, the first run at 2048 Hz with a stimulation pair of at least 10 trials.
  ERP CORE: 40 subjects × 7 components, the published epoched files. TEP: ds002094, 20
  subjects, the published TESA cleaning.
- **Known truth.** Components of known shape (Gaussian or skewed), latency, width and
  amplitude are added to real noise: the plus-minus average of the same subject's trials, which
  cancels the response and keeps the noise of the average.
  - CCEP: 60 waves per patient, each with 0 to 3 components (N1-, P-, N2-like) at 2 to 20 times
    the noise SD, and an exponential background in half of them.
  - ERP: 10 waves per subject and component, with a neighbouring component of opposite
    polarity in half of them.
  - Trial by trial: the subject's single-trial residuals plus a component with a known shift per
    trial.
- **Real noise alone.** The plus-minus average of every channel: any component found is false.
- **Real data.** The trial averages. Reliability: the same fit on the odd and on the even
  trials.
- **No circularity.** The channels are all good channels (CCEP), the published channel (ERP)
  or the channel of largest response from 38 to 140 ms (TEP). None of them depends on the
  setting under test.
- **Statistics.** The subject is the unit: each metric is averaged within a subject first.
  - Each value gets a mean over subjects with a 95 % bootstrap CI (10 000 resamples of
    subjects).
  - Each value is compared with the proposed one by a paired Wilcoxon signed-rank test over
    subjects, Holm-corrected across the values of a panel. In the figures, * marks p_Holm < 0.05.
- **Choose on one half, confirm on the other.** The subjects are split once at random (seed 2026).
  - The best value (blue triangle) is the one with the highest F1 on known truth in the
    development half. F1 = 2·TP / (2·TP + FP + FN): TP are the true components recovered, FP
    the false components, FN the missed components.
  - Every number shown comes from the confirmation half (CCEP 15, ERP CORE 20, TEP 10).
- **Reproduce.** Run `python benchmarks/seuils/etape6_all.py`. Every number of every figure is in
  `etape6_numbers.csv`; the captions, software versions and dataset versions are in
  `etape6_meta.json`.

Failed fits: none in CCEP. In ERP CORE, 2.2 % of the targeted fits have no labelled component;
they count as missing.

## Results

Numbers are confirmation-half means.

| Figure | Setting (proposed) | What the data say | Verdict |
|---|---|---|---|
| 6-01-shapes | Gaussian + skewed | Highest F1: 75.2, against 72.8 (Gaussian only) and 73.9 (skewed only); both differences significant. Also chosen on the development half. | Supported |
| 6-02-background | exponential (CCEP) | Highest F1, and chosen on the development half. Without a background, the latency error rises from 3.0 to 4.2 ms (significant). | Supported |
| 6-03a-max-components | 6 | Plateau from 3 to 10; 1 or 2 lose components (significant). | Supported (not binding) |
| 6-03b-max-fwhm | 0.25 s | Plateau from 0.15 to 0.5 s. At 0.1 s or less, components are missed and false ones added (significant). | Supported |
| 6-04-delta-bic | 10 (CCEP) | F1 plateau from 3 to 10 (about 75 to 76), with no significant difference from 10 below 25. The development half chose 8. False components on real noise fall from 0.24 (ΔBIC 0) to 0.12 (10) to 0.03 (60) per wave, while recall falls from 73 % to 67 % to 51 %. "calibrated" is worse (F1 67). | Supported (on the plateau) |
| 6-04b-delta-bic-erp | 10 (ERP) | F1 plateau from 1 to 12, peak at 3 to 4 (76.0); the development half chose 3. From 15 upwards it is significantly worse. 10 sits at the strict edge of the plateau: fewer false components (0.15 per wave, against 0.26 at 3) and lower recall (65 %, against 71 %). | Supported, but a lower value (4 to 6) is equally good on F1 |
| 6-05-candidate-threshold | 2 σ | Plateau from 0.5 to 2.5 σ; from 3 σ, recall drops (significant). Lower values are slower. | Supported (fastest value without loss) |
| 6-06-merge | c = 0.5 | No significant difference anywhere from 0 to 2. | No effect on these data |
| 6-07a / 6-07b cancelling pairs | no (CCEP), yes (ERP) | CCEP: no significant difference. ERP: rejecting lowers the latency error (6.7 against 7.1 ms, significant). The development half chose "yes" in both. | ERP supported; CCEP could use "yes" too |
| 6-08-latency-definition | model 50 % area (ERP) | On known truth, the 50 % area latencies (raw or model) err least and the peak latencies err most (P3: about 19 against 28 ms). On real data the 50 % area latencies are also the most reliable between odd and even trials (P3: 25 against 70 to 77 ms). The model 50 % area is within ± 10 ms of the published latency for 89 to 100 % of subjects. | Supported |
| 6-09-onset-offset | 10 % of the peak | The error against the truth and the odd/even difference both fall steadily as the fraction rises: onset error 37 ms at 10 %, 30 ms at 20 %, 21 ms at 50 % (significant). | **Not supported: a higher fraction (20 to 50 %) is more accurate and more reliable** |
| 6-10-targeted-measure | ± 25 ms, 3 parts; tolerance ± 10 ms | The margin barely matters. With 2 or more parts, agreement with the published latency on real data reaches 91 to 96 %, against 82 to 86 % with one part. On known truth all cells give 72 to 80 %. The 95th percentile of the measurement error is 24 ms: ± 10 ms holds only about 80 % of correct measurements. | Margin and parts supported; **the ± 10 ms tolerance is stricter than the measurement error** |
| 6-11-tep-windows | N45 38–55 ms, N100 80–140 ms | Only 10 % of the confirmation subjects have a negative component in the N45 window, whatever the window. N100: 50 % (proposed) and 80 % with the window derived from the development half (76–193 ms). Only 10 subjects per half, so the CIs are wide. | Inconclusive (too few subjects); the N45 problem is not the window |
| 6-12-trial-by-trial | 0.5 × FWHM (scalp) | With 59 trials per subject (median), the shifts found are dominated by noise. With a true jitter of 0 ms, the SD found is 42 ms at 0.5 × FWHM, and it grows with the bound. The error of the shifts grows with the bound; the share of trials at the bound falls from 82 % (0.1) to 37 % (0.5) to 3 % (2). | **Single-trial latencies are not recoverable at this SNR with any bound** |
