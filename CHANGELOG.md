# Changelog

## Unreleased

### REQ-171 restated (DECISIONS §5.6)

- REQ-171 now asks a lower RMSE of the empirical-Bayes channel latencies pooled over the
  simulated datasets (met), and reports the fraction of datasets where they win (4 of 6;
  the former "on >= 90 % of datasets" is bounded at 84 % even with the true prior).

### Reading in milliseconds, numba that does not match numpy, safe update command

- `Component.latency_ms`, `fwhm_ms`, `onset_ms`, `offset_ms`, `rise_time_ms`,
  `decay_time_ms` and `ResponseFit.background_ms`: the times in milliseconds, for reading
  (the fields stay in seconds). Tutorial 02 uses them instead of `* 1000`.
- A numba installed for another numpy (for instance after `pip install --force-reinstall`
  in Colab) no longer stops a fit: the engine takes its NumPy path with one warning.
- Site: the update command reinstalls evoparam alone (`--no-deps`).
- REQ-105: `alpha` of the calibrated acceptance measured again (DECISIONS §5.3); no change.

### Trials: no invented jitter on slow components, jitter test (REQ-151; DECISIONS "null_rule=ar")

- `fit_trials(null_rule="ar")`, the new default: the noise-only jitter is measured on noise
  drawn from an autoregressive model of the baseline over the whole window. The former
  default (`"segments"`, kept) repeated a short baseline end to end and reported a jitter in
  every jitter-free run of a P3-like component (7 / 15 / 29 ms at per-trial SNR 2 / 1 / 0.5).
- `jitter()` adds `jitter_detected`, `jitter_min_detectable`, `jitter_sd_eb` and
  `jitter_status` ("not measurable (< x ms)"); the trial table adds `latency_se` and the
  empirical-Bayes `latency_shrunk`.
- New `benchmarks/trial_jitter.py`. REQ-151 restated (at most 5 % of jitter-free runs report
  a jitter, smallest detectable jitter reported).

### REQ-374 met: layer-3 records with the exact versions of data and comparator

- Rerun on 2026-10-20 of `validation/erpcore.py`, `validation/erpcore.py --parent`,
  `validation/carla_ds004977.py` (CARLA_JNM commit
  `3df5d84493448fe2b129aaf6d25f921d8ff3c89f`, the last one before 2026-10-09; MATLAB
  R2026a) and `validation/layer3_report.py`. Every record of
  `validation/layer3/comparisons.json` now names its data version (the SHA-256 of the
  ERP CORE files read, ds004977 1.2.0) and its comparator version: the strict expected
  failure `test_req374_exact_versions_of_data_and_comparator` is removed.
- CARLA on ds004977: the channel lists are unchanged; the file gains `comparator_commit`.
- ERP CORE, difference waves: the committed results predated the targeted `erp` preset
  (REQ-274 above); the rerun gives the values stated there (within 10 ms: ERN 95 %,
  LRP 95 %, MMN 100 %, N170 95 %, N2pc 92 %, N400 95 %, P3 90 %).
- ERP CORE, parent waves (diagnostic): within 10 ms, before -> now: ERN 62.5 -> 85 %,
  LRP 90 -> 95 %, MMN 27.5 -> 90 %, N170 42.5 -> 65 %, N2pc 36 -> 92 %, N400 7.5 ->
  52.5 %, P3 40 -> 92.5 %.

### N1 case: known-truth bench, ccep 1.2 locked (REQ-113; DECISIONS §5.13)

- New `benchmarks/n1_known_truth.py`: fake N1s injected into the real pre-stimulus noise of
  51 ds004080 patients. At 1 % invented N1s the ccep preset finds 65.0 % of them, the
  published rule 51.5 % (95 % CI of the difference 10.0 to 19.1 points); latency within
  2 ms for 98.5 % of the found N1s. No change to the code or the preset.

### N1 case: the ccep N1 read on the fitted curve, ccep 1.2 (REQ-113, REQ-114; DECISIONS §5.12)

- New `evoparam.models.curve_detection(fit, window, polarity, k)`: a peak of the fitted
  curve (`fitted_peak`) deeper than `k` times the fit's noise SD.
- `ccep` 1.2: new preset field `detections`; the N1 is `curve_detection` on 9-50 ms with
  k = 3.4, instead of the presence of a negative component there. Fits, components and
  labels are unchanged; the methods text states the decision (frozen text regenerated).
- Measured once on the test runs: REQ-113 (registry row, the rule's N1 detections) pooled
  kappa 0.46 to 0.71, two REQ-371 runs of six at 0.8 or more (none before); still not met.
  REQ-114 (fourth reading) 90.2 % to 88.1 %, now not met.

### N1 case: the preset fitted on its trials, REQ-113 in one reference (REQ-113, REQ-114, REQ-135, REQ-273; DECISIONS §5.11)

No change to the engine, the `ccep` preset (still 1.1), `ALGORITHM_VERSIONS` or the
fixtures: the results of `run_pipeline` are unchanged. The changes are in
`evoparam/tests/models/test_engine_ccep.py` and DECISIONS §5.11.

- The tests now fit the preset as its pipeline does. Until 2026-10-18 they fitted the
  trial average alone; without trials, `noise_source="auto"` falls back to the baseline
  residual (effective sample size about 2 at 512 Hz) and large N1s could come out
  `no_response`. The harness now fits the baseline-corrected trials of the adjusted
  common average, as `run_pipeline` does (bit-identical fits).
- The reference REQ-113 compares in, read in its most demanding form: REQ-113 is asked
  in the comparison tested so far (the preset on its adjusted common average, the rule in
  the recording reference), in the preset's reference (the rule applied to the preset's
  input) and in the recording reference (the preset's chain without re-referencing,
  `run_pipeline(..., reference=False)`), with the rule's detections over 9-100 ms and
  with its N1 detections: new `test_req113_in_one_reference` (8 cases, run by
  `test_req273_ccep_meets_req113_and_req114`). New
  `test_req113_cannot_hold_in_both_references`: the joint kappa bound of the rule in the
  two references is 0.48 on the PT contacts of the example run and reaches 0.8 on four
  REQ-371 runs of six, so no decision can meet REQ-113 in both.
- Engine and preset variants diagnosed and measured on the development runs only
  (pruning, reseeding on a background-collinear design, `background_prior`, `tau_max`,
  `detection_threshold` 3 and 4, `reject_cancelling`, the recording reference): pooled
  kappa (registry row, N1 detections) 0.531 with the preset, 0.549 at best (two runs of
  21), within the spread or at the cost of more misses; none adopted.
- Measured once on the test runs (2026-10-19; registry row / code):
  - REQ-113 not met in any comparison. N1 detections: kappa on the PT contacts -0.04 /
    -0.04, pooled 0.46 / 0.31 (0.19 / 0.19 and 0.49 / 0.36 with the average fitted
    alone); detections over 9-100 ms -0.20 / -0.04, pooled 0.48 / 0.35 (0.01 / 0.19 and
    0.44 / 0.40); in the preset's reference at best 0.55 on the PT contacts and 0.53
    pooled; in the recording reference 0.38 and 0.45, one run of six with the code.
  - REQ-114 met with the N1 detections and the rule's operator on the fitted curve in
    the rule's reference: 90.2 % (46 of 51) / 92.0 % (23 of 25), down from 94.1 % /
    96.3 %; the code's fitted-trough reading now passes (96.0 %, was 88.9 %). The N1
    component's own latency, as written: 51.0 % / 52.0 %, not met.
  - Two non-gating readings of the detections over 9-100 ms in the rule's reference move
    below 90 % and become non-strict expected failures:
    `test_req114_rule_peak_on_the_fitted_curve_in_the_rules_reference[code]` 85.7 %
    (was 90.0 %) and `test_req114_extremum_reading_in_the_rules_reference` 88.2 % (was
    93 %).
  - REQ-273 not met (needs REQ-113).
- REQ-135: per-pair FDR kept as written; on the development runs the default reading
  gives 0.577, 0.535 and 0.645 on the three sets (no correction 0.706, 0.636, 0.695;
  cluster correction and a sign-flip permutation of the rule's statistic lower). Nothing
  changed; 0.495 on the REQ-371 runs, not met.

### REQ-324 / REQ-347: the streamed estimates reach the continuous bound of `fit_trials`; the common shift of §10.6 recorded (DECISIONS §5.10)

The method text of §10.6 (one common shift and one scale per pulse and channel) and the
agreement of REQ-324 and REQ-347 with `fit_trials(mode="shift_scale")` (a shift and a scale
per component) cannot both hold with several components. The most demanding reading is
kept: each component within 1 % and one sample of `fit_trials`, in the default mode of
`StreamFitter`, with the latency budget and chunk independence; the method text stays
available as `mode="common"` (DECISIONS §5.10).

- `evoparam.trials.FixedShapeModel` (used by `StreamFitter`): the grid of shifts carries one
  guard sample beyond `max_shift` on each side. It is never chosen, but the parabolic
  refinement uses it, so that a shift at the edge of the grid reaches the bound itself
  when the bound is not a whole number of samples, as in `fit_trials` (5 ms = 10.24
  samples at 2 048 Hz: the refined shift stopped at 10 samples). New attribute `centre`
  (index of the unshifted template in `grid`). Only the estimates whose best grid shift
  is at an edge change.
- Agreement with `fit_trials(mode="shift_scale", max_shift=0.005)` on 200 pulses, largest
  differences, before -> after: two components with one jitter, 2.3e-4 -> 7.0e-7 relative
  and 0.24 -> 0.0097 sample; with one jitter per component, 1.0e-4 -> 1.3e-6 and 0.31 ->
  0.012 sample (the largest differences before were all at the edge of the grid);
  single-component templates 6.8e-5 and 0.0066 sample (unchanged); three components (a
  check, not a test) 8.9e-5 and 0.10 sample. `mode="common"`: 92 % and 8.8 samples / 117 %
  and 12.7 samples, unchanged, still a strict expected failure.
- New test `test_req347_chunk_sizes_give_identical_estimates_two_components`: chunks of 1,
  32 and 1 024 samples give bit-identical estimates with the two-component template in the
  default mode (both jitter cases), as with one component.
- Latency (`benchmarks/stream_latency.py`, 1 000 pulses, 64 channels, 2 048 Hz, 0.5 s
  window, `max_shift` 5 ms, `OMP_NUM_THREADS=1`), measured on a 4-CPU x86_64 machine with
  other jobs running (load average about 10, not the reference machine): push p99 one
  component 8.8 ms (chunks of 32) and 10.2 ms (chunks of 1); two components, default mode,
  p50 9.0 ms, p95 11.5 ms, p99 13.0 ms, max 14.0 ms (`"common"` p99 11.2 ms, `"scale"`
  12.7 ms). Budget: p99 < 20 ms. `test_req347_latency_budget` passed (one and two
  components, `EVOPARAM_PERF_TESTS=1`).

### Engine speed, part 5 (REQ-290 P1): same results to the bit, P1 still not met

P1 is about a tenth faster with the `fast` extra and still above its 50 ms target. Every
result is the same to the last bit, so no algorithm version changes.

Timings: same machine, core 3 (`taskset -c 3`), `OMP_NUM_THREADS=1`, previous code (013d7b7)
and new code run alternately, three rounds; the load average was 6 to 7 on the 4 cores
(other jobs), which made the wall-clock medians of the previous code unstable (94 to 106
ms), so the table gives the CPU time of the process too:

| workload | 013d7b7 | new code |
|---|---|---|
| P1 workload, warm-up + 30 fits: median CPU time per fit | 77.5, 77.8, 77.2 ms | 67.9, 69.6, 68.1 ms |
| the same runs: fastest fit (wall clock) | 72.4, 72.1, 70.8 ms | 66.0, 65.1, 66.0 ms |
| the same runs: median (wall clock) | 99.0, 94.2, 105.8 ms | 68.2, 71.0, 68.5 ms |
| compiled kernel alone: the 58 calls of one P1 fit replayed, median of 30 | 42 to 46 ms | 38 to 41 ms |
| REQ-115 waveform (5 waveforms x 8), median, two rounds | 45.0, 43.1 ms | 38.7, 39.1 ms |

What changed:

- `models/_fast.py`: a component is whitened only on the samples its span and the reach of
  the filter cover (the rest of the whitened row is 0, as before, and is no longer
  computed); its design row and derivatives are no longer set to 0 outside the span. The
  constant of the exponential background (or a fixed basis) is whitened once per compiled
  fit instead of at every evaluation. An AR filter of at most four lags with `a[0] = 1`
  makes one pass over the samples, each term added in the same order as before.
- `models/engine.py`: the latencies of an ordered structure are read from the positions of
  the chain; the start is sorted only when they are not already in order; the whitened
  residual sum of squares is computed once for both criteria.
- `shapes/_base.py` (`compute_features`): the first and last samples above the onset level
  are found once, the lobe is cut by `searchsorted`, and the local extrema are searched
  only where the curve is above that level. `shapes/components.py`: `skew_mode` of one
  skewness keeps `(z*, c, dc/dalpha)` instead of `z*` only.

Results, previous code against new code: the 58 kernel calls of one P1 fit replayed give
the same outputs to the bit; 2 464 compiled fits on other structures (exponential,
constant, linear and no background; white, AR of orders 1 to 9 and strided whitening, with
and without start rows and prior rows; components at both window edges, a 50 ms window)
give the same outputs to the bit; a battery of 60 fits (P1 on three channels of eight
seeds, six REQ-115 waveforms, 20 engine-test waveforms, 10 skewed-Gaussian `erp`
waveforms) gives identical tables and evaluation counts; `compute_features` gives the same
features on 3 360 cases (7 shapes, 3 sampling rates). `ALGORITHM_VERSIONS` is unchanged.

P1 stays above 50 ms, and `benchmarks/results` is not regenerated. With the same rounding
the kernel is bound by the passes of the Householder QR over the 2 050-row columns (about
450 per evaluation and Jacobian); a change of their rounding alone (the order of one inner
product of the Jacobian) moves the features of the 60-fit battery by up to 3.7e-5 relative,
beyond the 1e-9 of the fixtures, so a faster QR needs a new algorithm version.

### N1 case: the rule's N1 detections (REQ-113, REQ-114, REQ-273)

REQ-113 names an N1 "a negative component with latency in 9–50 ms" and asks it on every
channel where the published rule, read over 9–100 ms, "detects one". The rule's detections
whose peak is after 50 ms are not N1s by that definition. Both readings are now asked
(DECISIONS §5.11): the rule's detections over 9–100 ms (A, the tests until now) and its N1
detections, the peaks it finds in 9–50 ms (C: the rule unchanged, its window ending at
50 ms). Settings were examined on development runs only: UMCU07, 08, 10, 11, 19, 28, 40
and 49; six runs of UMCU01 other than the example; UMCU12 to 17. The test runs (the
example and the REQ-371 runs) were measured once, at the end. No default of the package
changed. Only tests and documents change.

- **REQ-113 with C is not met.** Registry row / code, on the test runs: kappa 0.19 / 0.19
  on the 46 PT contacts (1 rule N1, 8 preset N1s, none missed). No REQ-371 run meets the
  criterion: registry row 0.37, 0.60, 0.12, 0.33, 0.39, 0.59 with 1, 12, 6, 5, 6, 3 missed.
  Pooled 0.49 / 0.36. Development runs (registry row): pooled 0.47, 0.43 and 0.59 on the
  three sets, no run meeting it. Test: `test_req113_n1_detections` (expected failure with
  these values). A is unchanged: 0.01 / 0.19 on the PT contacts, 0.44 / 0.40 pooled.
- **REQ-114 with C is met** when the model latency is read as the rule's operator on the
  fitted curve, in the rule's reference (DECISIONS §5.2, fourth reading): 94.1 % (48 of 51)
  / 96.3 % (26 of 27) on the test runs, and 91 %, 92 % and 93 % (registry row) on the
  development sets. The N1 component's own latency gives 49.0 % / 44.4 %. The fitted
  trough gives 74.5 % / 88.9 %. The fitted curve in the preset's reference gives
  82.4 % / 96.3 %. Test: `test_req114_n1_detections`, parametrised by reading and by
  latency.
- **REQ-273 is still not met**, because it needs REQ-113 under both A and C.
- **Where the gap is** (registry row, C, REQ-371 runs). The preset names 85 N1s, and 38 of
  them are not rule N1s. Of the rule's 80 N1s, the preset misses 33.
  - Of those 33, 23 are rule N1s of the preset's own input too. These are failures of the
    engine's decomposition.
  - The rule itself, applied to the preset's input (the adjusted common average), meets
    the criterion on one REQ-371 run of six (pooled 0.78; development 4 of 9, 2 of 6,
    2 of 6). Test: `test_req113_rule_on_the_presets_input_bound`. With the code's reading
    the same copy meets it on five of six.
- **Variants measured on the development runs and not adopted** (registry row, C, pooled
  kappa on the first two sets; the preset gives 0.47 and 0.43):
  - no SEM weights: 0.55 and 0.61;
  - the recording reference: 0.56 and 0.46;
  - no edge extension: 0.43 and 0.49;
  - background `tau_max` 10 ms: 0.46 and 0.46;
  - both of these: 0.45 and 0.52.

  None meets the criterion on more than one run of six.
- **REQ-135 over every stimulation pair of a run** (measured on eight development runs,
  not adopted, DECISIONS §5.11): with FDR over all channels and pairs of a run, kappa
  0.634 with the registry row (0.700 without the correction). It is below 0.7, so the
  test runs were not read this way and the REQ-135 tests are unchanged.

### Spec items found missing: masked onsets, long windows, the erp windows, the bootstrap distribution, REQ-133 for ftract

An inventory of the specification against the code found five items not implemented. Each
is implemented as written, with a test of the behaviour the spec states; the readings are
in DECISIONS §4.

1. **§5.3 item 12, intervals of a component whose onset is masked.** A component whose
   onset (10 % of its peak) lies in a masked interval has every interval (parameters,
   `A`, features, at every uncertainty level) widened around its centre by `1 + f`, `f`
   the masked fraction of its support `[onset, offset]`, and its standard errors by the
   same factor; `diagnostics["masked_support"]` and the warning `masked_onset:<i>` record
   it. Point estimates unchanged. Measured on the test waveform (Gaussian at 100 ms, FWHM
   40 ms, 55–75 ms masked): `f` = 0.130, latency SE 0.517 ms without mask, 0.536 ms with
   the mask (zero weights), 0.606 ms widened. Test:
   `tests/models/test_varpro.py::test_masked_onset_widens_the_intervals` (widths exactly
   `1 + f` times those of the same fit without the widening, centres kept; an unmasked
   onset keeps its intervals).
2. **§5.5.9, decimated search of a window longer than 10 s at 1 kHz or more.** The
   stepwise search runs on block means of `q` samples (`q` the smallest factor leaving at
   most 20 000 samples in the window), the model found is refitted at full resolution,
   and `decimated_search` (`factor`, `n_samples`, `n_samples_full`) is a key of the
   provenance step (also in `diagnostics` and `steps`). Engine functions:
   `_fit_waveform_core` (split into `_prepare` and the search), new `decimation_factor`,
   `_decimate`, `_decimated_fit`, `_full_resolution`. Measured on 20 development
   waveforms (12 s at 2 kHz, `q` = 2, three components of 0.4 to 1.5 s FWHM, seeds 100 to
   119): the same number of components as the search at full resolution in 20 of 20,
   latency, amplitude and FWHM within 8e-4 of their standard error of it; 0.54 s per fit
   against 0.60 s at full resolution. Tests: `tests/models/test_long_window.py`.
3. **The `erp` preset's MMN, N2pc, LRP and ERN windows (D6, REQ-274).** Added with ERP
   CORE's published measurement windows [LIT-erpmeas-39] (the source §6.2 names; the same
   windows as `validation/erpcore.py` and benchmark task T6): MMN 125–225 ms, N2pc
   200–275 ms, LRP −100–0 ms, ERN 0–100 ms, all negative, after the six labels of the
   table, which keep their precedence. `erp` 1.1 → 1.2 (labels only: fits are unchanged,
   their names can change; results made with `erp` 1.1 are refused by
   `evoparam.reproduce`, REQ-333). The table comparison of REQ-270 lists the four as a
   deviation explained in the preset's notes. `docs/presets.md` and the frozen methods
   text of `erpcore_p3` regenerated (version and label sentence). Test:
   `tests/presets/test_presets.py::test_d6_erp_labels_the_seven_erp_core_components`.
   REQ-274 itself is not re-measured here (ERP CORE data not at hand); it stays marked
   not met as written (DECISIONS §5.9).
4. **§5.5.5, `uncertainty="bootstrap"`.** The fit now has the expanded percentile
   interval and the bootstrap standard error of every parameter of each component (the
   shape's parameters and `A`), not only of the six features, and returns the bootstrap
   distribution: the new optional field `Component.bootstrap` (one array of the
   successful draws per parameter and feature; saved and loaded with the result; None at
   the other levels). The draws are matched to the components through the engine's order
   (`ResponseFit` sorts by latency; the profile intervals use the same matching). On the
   test waveform (20 trials, 60 resamples, all successful; first component, Gaussian
   at 120 ms) the 95 % intervals are, bootstrap against Wald: `mu` 0.11960–0.12178 s
   against 0.11965–0.12154 s, `sigma` 0.0165–0.0178 s against 0.0162–0.0182 s, `A`
   −8.03 to −7.43 against −8.07 to −7.33. Cost on that waveform: 0.66 s against 0.43 s. BCa is
   still not implemented for a fit (DECISIONS §3). Test:
   `tests/models/test_status_api.py::test_bootstrap_returns_the_distribution_and_intervals_on_every_parameter`.
5. **REQ-133 for the `ftract` rule.** The test of the registry left `ftract` out: its
   statistic is taken after the 1–45 Hz band-pass, so the one-sample waveform of the
   other rules does not lie at the threshold. A waveform built at the threshold after
   the filter (scale of a 10 Hz burst solved to 1e-9) flips the decision at `k ± 1e-6`
   for `k` = 3, 5 and 7 (the row gives "Z between 3 and 7", DECISIONS §4), passed
   explicitly. Test: `tests/existence/test_rules_registry.py::test_req133_ftract_flips_at_threshold`.
   The rule needed no change.

Algorithm versions: `ResponseModel.fit_waveform`, `fit_group` and `models.fit_waveform`
1.6; `existence.test_response` 1.6, `trials.fit_trials` 1.4,
`hierarchical.fit_hierarchical` 1.4, `spatial.fit_shared` 1.3 (their channel fits). The
reference fixture is regenerated for the versions only (values unchanged: no mask, short
windows); SOUP regenerated with no change; frozen methods texts regenerated (only
`erpcore_p3` changes, item 3).

Tests: the spec guards (`test_polarity`, `test_no_deletion`, `test_vocabulary`),
`test_determinism`, `test_reproduce` (both), `tests/data`, `test_methods_text`,
`tests/models`, `tests/presets`, `tests/existence/test_rules_registry.py`.

### REQ-106 with the default pair: a penalised estimate of the asymmetry (gamma and biexponential truths met)

New package default `shape_choice="penalised"` (models 1.5; `models/_penalised.py`;
DECISIONS §3 "Penalised estimate of a skewed component", §5.4). As with `"hqc"` (the
default of models 1.4, "Final shape choice on every component" below), each component of
the final model, whatever its width, is refitted with every allowed shape and the lowest
HQC is kept; a skewed component is then estimated with:

1. a Gaussian prior on its asymmetry `alpha` (mean 0, SD 2.5), a pseudo-observation added
   to the whitened residual;
2. weights `sqrt(1 + 2 s(t)^2)` on the samples (`s` the unit curve of the component);
3. the samples below 0.3 of its peak further weighted by `min(1, 600 / E)`, `E` the
   whitened energy of the component.

The criteria, the residuals and the intervals are those of the unweighted data at the
estimate. `"hqc"`, `"aic"` and `"bic"` stay available; every preset still sets `"bic"`, so
the presets' fits, the reference fixture's results and the frozen methods texts are
unchanged (the fixture is regenerated for the versions only).

Cause. The least-squares skewed Gaussian puts the mode of a gamma or biexponential
deflection early (noise-free bias -0.5 and -1.0 sample at 12 and 25 samples), and its mode
moves fast with `alpha` (about 0.04 FWHM per unit near `alpha` 4 to 5), so the noise of
`alpha` spreads the latency (SD 0.8 to 1.0 sample at 12 and 25 samples). The fix had to
reduce both. The prior shrinks `alpha` and its noise. The weight on the top fits the
component where a skewed Gaussian follows any unimodal peak. The tails of a strong
component, where the wrong shape's misfit exceeds the noise, are counted less.

Tried on development seeds (400 600 + ..., the REQ-106 design; latency MAE in samples,
gamma / biexponential with the default pair; before the merge of the HQC default, so the
shapes were compared by AIC on every component):

| tried | n | gamma | biexponential |
|---|---|---|---|
| package defaults before (`"aic"`, from 8 samples) | 1 000 | 0.748 | 0.644 |
| `"aic"` on every component (the gardes-spec base) | 1 000 | 0.699 | 0.572 |
| Firth penalty (log-determinant of the Fisher information of the component) | 400 | 0.626 | 0.551 |
| heavy-tailed prior on `alpha` (Student t, `1.5 log(1 + alpha^2)`) | 400 | 0.621 | 0.568 |
| prior SD 2 on `alpha`, AIC choice | 1 000 | 0.494 | 0.604 |
| prior SD 2 and peak weight 4 | 1 000 | 0.487 | 0.497 |
| weights on the flanks from the misfit of the shape | 400 | 0.61 to 1.9 | |
| prior SD 2, peak weight 4, tail level 0.3, energy 600 | 300 | 0.469 | 0.475 |
| the same, energy 300 / 1 500 | 300 | 0.483 / 0.466 | 0.480 / 0.482 |
| the same, tail level 0.2 / 0.4 | 300 | 0.465 / 0.480 | 0.517 / 0.529 |
| the same, peak weight 0 / 8 | 300 | 0.417 / 0.492 | 0.648 / 0.481 |
| the same, prior SD 1.5 / 2.5 / 3 | 300 | 0.454 / 0.473 / 0.486 | 0.555 / 0.446 / 0.453 |
| prior SD 2.5, peak weight 6 | 300 | 0.493 | 0.471 |
| **prior SD 2.5, peak weight 2 (kept)** | 300 | **0.448** | **0.436** |

Criterion that compares the shapes inside `"penalised"`, chosen after the merge of the HQC
default (development seeds: REQ-105 300 000 + ..., 1 000 per K, the set of "Final shape
choice on every component"; REQ-106 400 600 + ..., 1 000 per truth; package defaults,
`delta` 10; latency MAE in samples of the default pair):

| final shape choice | REQ-105 recall / precision | Gaussian | skewed | ex-Gaussian | gamma | biexponential |
|---|---|---|---|---|---|---|
| `"hqc"` (models 1.4 default) | 0.8865 / 0.9987 | 0.233 | 0.455 | 0.403 | 0.722 | 0.587 |
| `"penalised"` compared by AIC | 0.8867 / 0.9989 | 0.251 | 0.423 | 0.388 | 0.454 | 0.455 |
| **`"penalised"` compared by HQC (kept)** | **0.8870 / 0.9992** | 0.220 | 0.465 | 0.393 | 0.482 | 0.462 |

Both meet every REQ-106 row; HQC has the better REQ-105 (0.8865 / 0.9987 reproduced for
`"hqc"` with the same script). With HQC the skewed row is 0.010 above `"hqc"` (0.465
against 0.455, met) and the gamma row is the closest to the bound (0.482).

Results of the kept default on the tests' seeds, measured once at the end (1 000 per
condition; `"hqc"` = models 1.4, "Final shape choice on every component"):

| REQ | `"hqc"` | `"penalised"` |
|---|---|---|
| 105 recall / precision, 100 per K | 0.883 / 1.00 | 0.8833 / 1.0000 |
| 105 recall / precision, 1 000 per K | 0.8873 / 0.9991 | 0.8875 / 0.9992 |
| 106 Gaussian | 0.254 | 0.237 |
| 106 skewed Gaussian | 0.456 | 0.478 |
| 106 ex-Gaussian | 0.401 | 0.403 |
| 106 gamma | **0.706** | **0.477** |
| 106 biexponential | **0.586** | **0.463** |

- The tests' first 100 waveforms (what the test runs): Gaussian 0.234, skewed 0.485,
  ex-Gaussian 0.389, gamma 0.477, biexponential 0.438. The expected failures of the gamma
  and biexponential rows are removed. Amplitude median ratio within 0.9 % of 1.
- Against `"hqc"`, the skewed row is worse by 0.010 (development) and 0.022 (tests'
  seeds), and the ex-Gaussian row by 0.002 on the tests' seeds (0.010 better on
  development); both stay met. Every other row improves.
- By width (3, 6, 12 and 25 samples), development seeds: gamma 0.26, 0.39, 0.52, 0.76;
  biexponential 0.28, 0.30, 0.43, 0.83.
- Truths fitted with their own shape: unchanged, value for value (a single allowed shape
  has no choice to make).
- REQ-112 on the ds004080 baselines (80 windows): 0.975 silent with `ccep` and 0.9625 with
  the package defaults, as before. The simulated REQ-112 and REQ-104 tests pass.
- REQ-111 fits a single shape (Gaussian), which has no shape choice: unchanged by
  construction.

The `ccep` preset with `"penalised"` (by AIC, measured before the merge) instead of
`"bic"`, on the REQ-113/114 runs from cache:

- The example run's kappa is unchanged under both readings: 0.011 for the registry
  reading, 0.191 for the code reading.
- On the other runs it moves both ways: registry 0.603 to 0.667 and 0.353 to 0.520; code
  0.215 to 0.189, 0.550 to 0.513 and 0.557 to 0.521.
- REQ-114 close fraction: registry 0.404 of 57 to 0.371 of 62; code 0.400 to 0.433.

This is not a clear gain, so the presets keep the reviewed `"bic"`.

Versions: `ResponseModel.fit_group`, `ResponseModel.fit_waveform` and `models.fit_waveform`
1.5, `existence.test_response` 1.5, `trials.fit_trials` 1.3,
`hierarchical.fit_hierarchical` 1.3, `spatial.fit_shared` 1.2. The reference fixture is
regenerated; SOUP and the frozen methods texts are regenerated with no change.

REQ-061, second part, with the new default (`tests/artifact/test_fitted_bias.py`): plain
blanking now gives a median amplitude error of 37 %, lost, 66 %, 42 % and 23 % for onsets 6
to 10 ms, so the >= 20 % bias of the specification holds at every onset and the strict xfail
is removed (15 % at 10 ms before). Template and learned cleaning: latency error at most
0.074 ms and amplitude error at most 2.1 % (criteria 0.5 ms and 5 %; 0.013 ms and 0.10 %
before).

### Final shape choice on every component, sign-free resolution check (REQ-003, REQ-004, REQ-012)

Three guard tests failed after the merge of moteur-sim-2 (00786a1, a04ca84); the code is
changed, not the tests.

- **REQ-004 (`test_no_deletion.py`): the final shape choice skipped components narrower
  than 8 samples** (`engine._refine_shapes`, `SHAPE_MIN_SAMPLES`): a step skipped under a
  width threshold. Every component of the final model is now refitted with every allowed
  shape, whatever its width. With the AIC of the previous default on every component, a
  narrow component at low SNR often took an asymmetry the data do not determine (its
  deviance gain is a chi-square with one degree of freedom, above the AIC's 2 for about
  14 % of the Gaussian truths of 3 and 6 samples) and its mode moved by 1 to 2 samples: on
  the REQ-105 development seeds the 7 waveforms that changed were SNR-3 components that
  became skewed and left the match. The shapes are therefore compared by the criterion of
  Hannan and Quinn (1979), HQC, `2 ln ln n` per parameter (about 3.7 for these windows),
  between the 2 of AIC and the `ln n` of BIC: `shape_choice="hqc"` is the new package
  default; `"aic"` stays available (also on every component); the presets keep `"bic"`.
  Chosen on development seeds (REQ-105 300 000 + ..., REQ-106 400 600 + ...; 1 000
  waveforms per K and per truth; package defaults, `delta` 10; latency MAE in samples of
  the default pair):

  | final shape choice | REQ-105 recall / precision | Gaussian | skewed | ex-Gaussian | gamma | biexponential |
  |---|---|---|---|---|---|---|
  | AIC from 8 samples (before) | 0.8863 / 0.9985 | 0.250 | 0.439 | 0.381 | 0.748 | 0.644 |
  | AIC, every component | 0.8852 / 0.9972 | 0.285 | 0.430 | 0.404 | 0.699 | 0.572 |
  | 3 per parameter, every component | 0.8858 / 0.9979 | 0.252 | 0.441 | 0.410 | 0.716 | 0.578 |
  | **HQC, every component (kept)** | 0.8865 / 0.9987 | 0.233 | 0.455 | 0.403 | 0.722 | 0.587 |
  | 4 per parameter, every component | 0.8865 / 0.9987 | 0.231 | 0.464 | 0.403 | 0.731 | 0.592 |
  | BIC (`ln n`, 6.4), every component | 0.8868 / 0.9991 | 0.210 | 0.486 | 0.394 | 0.762 | 0.609 |
  | Wald statistic of the asymmetry > 2 (focused criterion), every component | 0.8872 / 0.9994 | 0.325 | 0.369 | 0.363 | 0.699 | 0.712 |

  Kept: the named criterion with the REQ-105 recall and precision at least those of
  before, the expected failures (gamma, biexponential) closer to 0.5 and no met REQ-106
  row put at risk. No setting measured without a width rule keeps every REQ-106 row at its
  value: the ex-Gaussian row is worse with every penalty (0.394 to 0.410 against 0.381;
  the truths of 3 samples, which the threshold left as the search fitted them), and the
  Wald statistic, which improves it, loses 0.075 on the Gaussian and 0.068 on the
  biexponential truths. With HQC the skewed truths lose 0.016 (0.455, met), the Gaussian,
  gamma and biexponential truths gain.
- **REQ-003 (`test_polarity.py`): the resolution check passed over a component with a
  neighbour of the same sign** within twice its FWHM, a code path that depended on the
  signs of the components. It now passes over a component with a neighbour of either sign
  within that distance (the indices `k, k+1` of `merged:k,k+1` would name the neighbour).
  The two REQ-111 designs hold one sign; their results are unchanged (below). New test
  `test_calibration.py::test_resolution_check_does_not_depend_on_signs`: at 79ca631 its
  pair of opposite signs, 1.7 FWHM apart, was flagged `merged`.
- **REQ-012 (`test_vocabulary.py`)**: a prohibited word in a docstring of
  `models/_calibration.py`, now "the spectral density of the whitened component".
- `tests/io/test_fetch_erpcore.py::test_fetch_erpcore_downloads_and_checks` failed once
  in the full suite (a request to `/data/` seen during the second call). Not a regression
  of `evoparam/io/fetch.py`: the second call downloads nothing. The local test server
  (`tests/io/helpers.py`) recorded a request after sending its body, so under load the
  record of the first call's last download could land after the test had counted the
  requests; holding the server thread 2 to 10 ms between the two reproduced the failure
  at 79ca631 and at f626c68 alike. The server now records a request before answering it.
  Alone (15 runs) and with its module (5 runs) the test passed before and after.

Results (package defaults, `delta` 10; the tests' seeds, 100 per condition as the tests
run, and 1 000; REQ-112 on the cached ds004080 baselines with `EVOPARAM_NETWORK_TESTS=1`):

| REQ | before (100) | after (100) | before (1 000) | after (1 000) |
|---|---|---|---|---|
| 105 recall / precision | 0.883 / 1.00 | 0.883 / 1.00 | 0.8873 / 0.9991 | 0.8873 / 0.9991 |
| 111, 1.0 FWHM: two components | 100 % | 100 % | 100 % | 100 % |
| 111, 0.5 FWHM: one with `merged` | 100 % | 100 % | 99.8 % | 99.8 % |
| 111, 0.5 FWHM: two without `ill_conditioned` | 0 | 0 | 1 | 1 |
| 112, ds004080 baselines silent (`ccep` / package defaults) | 97.5 % / 96.25 % | 97.5 % / 96.25 % | | |

REQ-106, latency MAE in samples with the default pair (the truth fitted with its own shape
is unchanged: one shape, no shape choice):

| truth | before (100) | after (100) | before (1 000) | after (1 000) |
|---|---|---|---|---|
| Gaussian | 0.254 | 0.243 | 0.252 | 0.254 |
| skewed Gaussian | 0.450 | 0.461 | 0.430 | 0.456 |
| ex-Gaussian | 0.366 | 0.385 | 0.377 | 0.401 |
| gamma | **0.715** | **0.721** | **0.720** | **0.706** |
| biexponential | **0.592** | **0.599** | **0.637** | **0.586** |

Development seeds (1 000): REQ-111 unchanged (two components at 1.0 FWHM in 100 %, one
with `merged` at 0.5 FWHM in 100 %). Amplitude bias within 3.6 % everywhere. Gamma and
biexponential stay expected failures (DECISIONS §5.4). The presets, the reference fixture
(`ccep`; values unchanged, versions updated), the SOUP inventory and the frozen methods
texts are unchanged. Algorithm versions: `ResponseModel.fit_waveform`, `fit_group`,
`models.fit_waveform` and `existence.test_response` 1.4. DECISIONS §3 and §4.

### REQ-106 with the default pair: second pass (gamma and biexponential truths, still not met)

No change of results: the engine, the shapes and the algorithm versions are unchanged, so
the reference fixture, SOUP and the frozen methods texts are not regenerated. The
expected failures of the gamma and biexponential rows now state what was measured below
(DECISIONS §5.4).

Measured on development seeds (400 600 + ..., the REQ-106 design, 300 waveforms per
truth), latency MAE in samples of the default pair, gamma / biexponential (defaults 0.757
/ 0.628):

| tried | gamma | biexponential | other rows |
|---|---|---|---|
| latency of the fit read on the tenfold grid of §5.5.1 (not the analytic mode) | 0.757 | 0.628 | within 0.003 |
| the same, truth latency also read on that grid | 0.757 | 0.626 | within 0.003 |
| `alpha_max` 30 (the spec's bound is 10) | 0.757 | 1.107 | |
| background `constant` | 0.750 | 0.615 | |
| `noise_model="neff"` (§5.5.3 step 1) | 0.752 | 0.631 | |
| final shape choice on every component (not only from 8 samples) | 0.706 | 0.572 | Gaussian 0.251 to 0.271, ex-Gaussian 0.402 to 0.424, skewed 0.450 to 0.449 |
| two components forced (`selection="fixed"`, 2) | 0.802 | 0.679 | |
| skewed Gaussian only | 0.682 | 0.583 | |

- The latency is already the mode of the fitted component (analytic, the limit of the
  tenfold grid), and the test's truth latency is the §5.5.1 latency of the truth shape;
  the asymmetry bounds are those of §5.5.1.
- The engine returns the least-squares optimum: from eight asymmetry starts the cost and
  the mode are the same (36 noisy gamma waveforms of 12 and 25 samples).
- With noise, the mean signed error of the skewed fit at each width is its noise-free bias
  (gamma -0.10, -0.21, -0.52, -1.04 sample at 3, 6, 12, 25 samples; without noise -0.14,
  -0.24, -0.49, -1.02), and its mode is spread (SD 0.79 and 1.03 sample at 12 and 25
  samples). Without the bias the skewed-only fits would give 0.47 (gamma) and 0.28
  (biexponential), the pair 0.56 and 0.37.

REQ-106, latency MAE in samples, truth fitted with its own shape / with the default pair
(1 000 waveforms per condition; amplitude median ratio within 3.1 % of 1 everywhere):

| truth | development seeds | tests' seeds |
|---|---|---|
| Gaussian | 0.200 / 0.250 | 0.210 / 0.252 |
| skewed Gaussian | 0.385 / 0.439 | 0.386 / 0.430 |
| ex-Gaussian | 0.281 / 0.381 | 0.283 / 0.377 |
| gamma | 0.369 / **0.748** | 0.346 / **0.720** |
| biexponential | 0.302 / **0.644** | 0.307 / **0.637** |

On the tests' first 100 waveforms (the default run): gamma 0.715, biexponential 0.592.
REQ-105, REQ-111 and REQ-112 are unchanged (no change of results). What to try next:
DECISIONS §5.4.

### Calibrated acceptance: the change of the noise spectrum; the default `delta` is 10 again (REQ-105, REQ-111, REQ-112)

Merge of qualite-zero (0c84d6c, which holds integration-2 4f2c44d): its split of
`engine.py` into helpers kept, the acceptance, the final shape choice and the resolution
check of the entry below ported into it. With `delta=10`, `shape_choice="bic"` and
`resolution_check=False` the 123 fits of the entry below are equal to qualite-zero's,
value for value; the reference fixture (`ccep`) is unchanged.

- **The change of the noise spectrum over the baseline** (`models/_calibration.py`). On
  real baselines the calibrated acceptance (below) admitted broad components (30 to 140
  ms) of the slow activity of the recording: the AR spectrum fitted on the whole baseline
  does not hold over it. The baseline is now cut into three segments, an AR model of the
  fit's order is fitted on each (least squares), and the spread of their log spectra
  beyond what sampling explains (moment estimate: between-segment variance less the
  sampling variance, floored at 0), weighted by the component's spectrum, is added to the
  variance that widens the scan level. On stationary noise it adds about 1.1 times the
  sampling term on average.
- **`alpha` 0.04** (was 0.03), chosen with the segment term on development data:
  ds004080 baselines of 23 runs (920 windows, the first 40 contacts, not the two REQ
  runs) and the REQ-105 design at 1 000 waveforms per K (seeds 300 000 + ...):

  | calibrated acceptance | baselines silent | REQ-105 recall / precision |
  |---|---|---|
  | `alpha` 0.03, no segment term (the entry below) | 87.3 % | 0.9542 / 0.985 |
  | 2 segments, `alpha` 0.03 | 94.3 % | 0.9507 / 0.9863 |
  | 4 segments, `alpha` 0.03 | 95.0 % | 0.9497 / 0.9865 |
  | 4 segments, `alpha` 0.03, segment term only above its chi-square 0.95 level | 94.6 % | 0.9508 / 0.9863 |
  | 3 segments, `alpha` 0.035, the same gate | 94.1 % | 0.9517 / 0.984 |
  | 3 segments, `alpha` 0.04, the same gate | 93.7 % | 0.9543 / 0.9817 |
  | 4 segments, `alpha` 0.04, the same gate | 94.0 % | 0.9545 / 0.982 |
  | 4 segments, `alpha` 0.04 | 94.7 % | 0.9532 / 0.9823 |
  | **3 segments, `alpha` 0.04 (kept)** | 94.6 % | 0.9537 / 0.9821 |
  | `delta` 10 | 90.1 % | |

  Kept: the highest recall among the settings that keep 94.5 % or more of the
  development baselines silent.
- **Test data, measured once** (the two REQ runs, UMCU01 run-021448 and UMCU04
  run-031148, 80 windows; the tests' seeds): the calibrated acceptance kept 92.5 % of the
  baselines silent (74 of 80), `delta` 10 96.25 % (77 of 80). REQ-112 asks 95 % at the
  default `delta`, so **the package default is `delta=10` again** and the calibrated
  acceptance is on request (next item). With it, REQ-105 recall 0.950, precision 0.986
  (100 per K; 0.9545 and 0.980 on 1 000).
- **`delta="calibrated"`**: the sentinel that asks for the calibrated acceptance, in a
  `Preset` (`presets.derive(..., delta="calibrated")`, kept through `to_json` and
  `from_json`), in `ResponseModel` and in `FitSettings`. A number is the fixed fall of BIC;
  `delta=None` is "not given" (the preset's value, else 10); any other string raises
  `SettingsError`. `methods_text()` names the rule and its widening.
- `shape_choice="aic"` and `resolution_check=True` stay the package defaults. Under the
  fixed `delta` the resolution check compares with the larger deviance 3 ln n + 10 (about
  29): a single Gaussian of FWHM 20 ms at SNR 200 is now flagged `merged` (deviance of
  the pair 22), at SNR 400 it is not (86); `test_calibration.py` checks SNR 400.
- REQ-112 on real baselines is tested with the `ccep` preset and with the package
  defaults (`test_engine_ccep.py`, parametrised).

Results with the package defaults (`delta` 10, `shape_choice="aic"`,
`resolution_check=True`; the tests' seeds, 100 and 1 000 per condition):

| REQ | 100 | 1 000 | met |
|---|---|---|---|
| 105 recall / precision | 0.883 / 1.00 (K = 1 0.82, 2 0.85, 3 0.927) | 0.887 / 0.999 | no (expected failure) |
| 111, 0.5 FWHM: one with `merged` | 100 % | 99.8 % | yes |
| 111, 0.5 FWHM: two without `ill_conditioned` | 0 | 1 | yes |
| 111, 1.0 FWHM: two components | 100 % | 100 % | yes |
| 112, ds004080 baselines silent (`ccep` / package defaults) | 97.5 % / 96.25 % | | yes |

REQ-106, latency MAE in samples, truth fitted with its own shape / with the default pair:

| truth | 100 | 1 000 |
|---|---|---|
| Gaussian | 0.222 / 0.254 | 0.210 / 0.252 |
| skewed Gaussian | 0.370 / 0.450 | 0.386 / 0.430 |
| ex-Gaussian | 0.299 / 0.366 | 0.283 / 0.377 |
| gamma | 0.323 / **0.715** | 0.346 / **0.720** |
| biexponential | 0.252 / **0.592** | 0.307 / **0.637** |

The skewed truth is met; gamma and biexponential stay expected failures (DECISIONS §5.4).
REQ-111 at 0.5 FWHM, which failed on its second clause under the calibrated default, is
met. Algorithm versions: `ResponseModel.fit_waveform`, `fit_group`, `models.fit_waveform`
and `existence.test_response` 1.3 (package defaults changed against qualite-zero);
`fit_trials` 1.2. Reference fixture, SOUP and frozen methods texts regenerated.

Tests: `tests/models`, `tests/presets`, `tests/existence`, `tests/trials`,
`test_determinism.py`, `test_reproduce.py`, `tests/data/test_provenance.py`,
`test_methods_text.py`; `test_engine_ccep.py::test_req112_real_baselines_mostly_silent`
(both parameters) and `test_req113_engine_finds_n1_on_the_example_run` from the cached
runs.

### Calibrated acceptance (REQ-105, REQ-106, REQ-111, REQ-112): new package defaults

(Superseded in part by the entry above: the default `delta` is 10 again and the
calibrated acceptance is asked for with `delta="calibrated"`.)

Three settings of the engine, defaults without a preset (DECISIONS §3 "Calibrated
acceptance, final shape choice, resolution check", §5.3 to §5.5). Every preset sets the
reviewed procedure (`delta=10`, `shape_choice="bic"`, `resolution_check=False`, through
`presets.fit_fields`); with those settings the engine gives the previous fits value for
value (123 fits compared with the previous commit: the package defaults with these three
settings on 40 REQ-105 and 20 REQ-111 waveforms, and the `erp`, `ccep` and `tep` presets
on 21 waveforms of the REQ-104 suite; the reference fixture within 2.4e-12). Choices were
made on development seeds (REQ-105 300 000 + ..., noise windows 500 000 + ..., REQ-106
400 600 + ..., REQ-111 700 000 + ...); the tests' seeds were measured once, at the end.

- **`delta=None` (default; was 10), `alpha=0.03`: the acceptance calibrated on each fit's
  noise model** (`models/_calibration.py`). A component is accepted when its deviance
  exceeds the level pure noise exceeds with probability `alpha`: the expected Euler
  characteristic of the whitened scan field over latency and log width (Adler 1981;
  Worsley 1994; Siegmund and Worsley 1995), its Lipschitz-Killing curvatures computed from
  the fit's whitened Gaussian templates (background projected out), over the widths any
  allowed shape can reach (a skewed Gaussian at its largest asymmetry is about half as
  wide as `min_fwhm`), widened for the sampling error of the AR spectrum estimated on the
  baseline (asymptotic covariance of the coefficients through the log spectrum, weighted
  by the component's spectrum). `ResponseFit.diagnostics["acceptance"]` records the rule,
  the scan level and the fall of BIC it means; `methods_text()` names it. `delta=10` is
  the reviewed rule; with `noise_model` other than `"ar"` or `selection` other than
  `"bic"` the fixed 10 applies.

  Why not a fixed `delta` (engine package defaults before the final shape choice, 1 000
  noise windows each; recall on 200 development waveforms per K of the REQ-105 design):

  | `delta` | white | AR(1) 0.5 | AR(1) 0.9 | REQ-105 recall |
  |---|---|---|---|---|
  | 4 | | | 0.9 % | |
  | 3 | | | 1.5 % | |
  | 2 | | 1.5 % | 2.3 % | 0.942 |
  | 1 | | 2.4 % | 4.0 % | |
  | 0 | 1.9 % | 3.2 % | 5.7 % | 0.954 |
  | -1 | 2.7 % | | | 0.961 |
  | -2 | 4.7 % | | | 0.964 |

  `delta` 0 reaches the recall on these seeds but leaves 5.7 % of rho = 0.9 windows with a
  component (REQ-112: at most 5 %). The same rate needs a `delta` about 3 larger on
  rho = 0.9 noise than on white noise (the low-frequency spectrum of an AR model fitted on
  the baseline is uncertain by about 40 %, sd of log S(0) 0.38). An oracle matched filter that knows the width finds 73 % of
  the FWHM-3, SNR-3 components at 5 % false alarms (66 % at 2.5 %): the engine is near
  that limit, and REQ-105 is reached because every other level is found.

  `alpha` (development seeds, 1 000 waveforms per K): 0.025 recall 0.9523, 0.035 0.956
  (both before the final shape choice below), 0.02 with it 0.946, **0.03 with it 0.9542**
  (precision 0.985). Noise windows with a component at 0.03 (2 000 each, development
  seeds):

  | settings | white | AR(1) 0.5 | AR(1) 0.9 |
  |---|---|---|---|
  | the tests' `model()` (REQ-107, REQ-112) | 1.9 % | 1.6 % | 2.2 % |
  | engine package defaults | 2.7 % | 3.6 % | 2.55 % |
  | `delta` 10 (before) | 0 to 0.5 % | 0 to 0.5 % | 0 to 0.5 % |

  Before the width floor of the scan was added, the `model()` settings (`min_fwhm` 8 ms)
  had 17 of 59 false components as skewed Gaussians 4 to 8 ms wide, narrower than the scan
  covered.

- **`shape_choice="aic"` (new; presets `"bic"`)**: each component of the final model at
  least 8 samples wide at half maximum is refitted with every allowed shape (asymmetry
  started at 0 and +/- 2) and the lowest AIC is kept. Measured and not kept: the AIC in the
  quick shape choice of the search (REQ-105 recall falls: the skewed candidate then needs
  ln n more to be accepted; REQ-106 MAE skewed 0.521, gamma 0.789, biexponential 0.647 on
  400 development waveforms), and the refinement of every component whatever its width
  (REQ-106 skewed 0.438, gamma 0.705, biexponential 0.570; REQ-105 recall 0.9547 to
  0.9517, narrow components change shape and latency; from 8 samples: 0.9542).

- **`resolution_check=True` (new; presets `False`)**: a component that a pair of equal
  Gaussians `overlap_c` (sigma_1 + sigma_2) apart, with this component as its best single
  Gaussian, would explain as well (deviance between the pair and its best single fit, in
  the fit's whitened metric with the other components projected out, below the deviance a
  new component needs) is flagged `merged:k,k+1`. On single Gaussians (development seeds,
  FWHM 10 and 40 ms, SNR 5 to 80) the flag was set in every run; at SNR 200 it is not: it
  says what the data cannot exclude.

Results (test seeds: the tests' designs, 100 and 1 000 per condition; development seeds
1 000):

| REQ | before (test, 100) | after (test, 100) | after (test, 1 000) | after (development, 1 000) | met |
|---|---|---|---|---|---|
| 105 recall / precision | 0.885 / 1.00 | 0.950 / 0.988 | 0.9555 / 0.983 | 0.9542 / 0.985 | yes |
| 111, 1.0 FWHM: two components | 100 % | 99 % | 97.6 % | 96.4 % | yes |
| 111, 0.5 FWHM: one with `merged` | 0 % | 97 % | 94.3 % | 95.2 % | yes |
| 111, 0.5 FWHM: two without `ill_conditioned` | 0 | 2 | 35 | 42 | no |
| 112, `model()` noise windows silent (40 per rho) | 40 / 40 / 40 | 39 / 40 / 40 | | 98.1 / 98.4 / 97.8 % (2 000) | yes |
| 107, rho 0.9: with a component (`neff` / `ar` / `white`, 100) | 0 / 0 / 100 | 0 / 0 / 100 | | | yes |

REQ-106, latency MAE in samples (amplitude bias within 2.6 % everywhere), truth fitted with
its own shape / with the default pair:

| truth | before (test, 100) | after (test, 100) | after (test, 1 000) | after (development, 1 000) |
|---|---|---|---|---|
| Gaussian | 0.222 / 0.222 | 0.213 / 0.256 | 0.209 / 0.252 | 0.200 / 0.249 |
| skewed Gaussian | 0.370 / **0.521** | 0.370 / 0.460 | 0.387 / 0.442 | 0.386 / 0.447 |
| ex-Gaussian | 0.299 / 0.420 | 0.299 / 0.372 | 0.283 / 0.383 | 0.281 / 0.386 |
| gamma | 0.323 / **0.763** | 0.323 / **0.706** | 0.346 / **0.715** | 0.368 / **0.744** |
| biexponential | 0.252 / **0.603** | 0.253 / **0.613** | 0.307 / **0.641** | 0.302 / **0.638** |

The skewed truth is met; gamma and biexponential are not (the pair cannot place the mode
of these shapes: 0.47 and 0.54 sample off without noise, 1.0 and 1.2 at 25 samples;
DECISIONS §5.4). Measured for them and not kept (development seeds, 200 waveforms per
truth, MAE of the default pair for Gaussian / skewed / ex-Gaussian / gamma /
biexponential truths):

| latency read from | Gaussian | skewed | ex-Gaussian | gamma | biexponential |
|---|---|---|---|---|---|
| the fitted component (kept) | 0.255 | 0.468 | 0.387 | 0.754 | 0.624 |
| the fitted shape shifted onto the samples above 0.5 of its peak | 0.287 | 0.444 | 0.357 | 0.744 | 0.699 |
| the same above 0.25 | 0.259 | 0.457 | 0.381 | 0.764 | 0.656 |
| the shape refitted on the samples above 0.5 of its peak | 0.272 | 0.494 | 0.419 | 0.658 | 0.623 |
| the same above 0.3 | 0.275 | 0.483 | 0.374 | 0.613 | 0.616 |
| a parabola through the samples above 0.6 of the peak | 0.319 | 0.525 | 0.400 | 0.711 | 1.163 |

With the skewed Gaussian as the only shape (400 development waveforms) gamma is 0.678
and biexponential 0.572: no choice between the two shapes of the pair reaches 0.5.

REQ-111 at 0.5 FWHM: the second clause ("never two components" unless ill-conditioned)
fails because the calibrated acceptance admits a false component elsewhere in the window
in about 3 % of runs (the two of the test's 100: the pair as one component plus a
component of about 1.7 noise SD at 0.17 s or 0.315 s), the rate REQ-105 needs. With
`delta=10` and the resolution check both clauses hold on the test's seeds (one component
with `merged` in 100 of 100, no two-component fit; two components at 1.0 FWHM in 100 of
100), and REQ-105 falls back to 0.885.

Elsewhere:

- Presets (`erp`, `ccep`, `tep`, the ECAP and other presets through `fit_fields`), the
  benchmark (presets), `fit_trials` (presets) and the reference fixture (`ccep`):
  unchanged, value for value. Frozen methods texts: unchanged
  (`tools/freeze_methods_text.py --check`).
- ds004080 pre-stimulus windows (REQ-112 on real baselines; the test uses `ccep`): `ccep`
  97.5 % silent on the test's 80 windows (unchanged), 93.0 % on 920 windows of 23
  development runs. The package defaults, which no CCEP analysis uses: 96.3 % to 93.8 %
  (test runs), 90.1 % to 87.3 % (development runs); the windows with components hold 2.7
  components each at `delta` 10 already (slow activity of the recording, not white noise).
- `existence.test_response(method="model_comparison")` (package defaults, parametric
  bootstrap with 49 samples, 40 channels of 20 trials, Gaussian of FWHM 40 ms at 0.15 s):
  noise channels significant 0 of 40 before and after (white and AR(1) 0.5); amplitude
  0.15: 0 and 0; 0.25: 42.5 % to 47.5 %; 0.35: 90 % to 92.5 %; 0.5 and 1.0: 100 %.
- Algorithm versions: `ResponseModel.fit_waveform` and `fit_group` 1.2,
  `existence.test_response` 1.2; reference fixture regenerated (versions; values within
  2.4e-12).
- `test_status_api.py::test_req118_status_semantics` checks the `no_response` status with
  `delta=10`: its noise draw is one of the 3 % that get a component under the calibrated
  default.

Tests: `tests/models` (with `test_engine_sim.py`, the new `test_calibration.py`),
`test_determinism.py`, `test_methods_text.py`, `tests/presets`, `tests/existence`,
`tests/trials`; `test_engine_ccep.py::test_req112_real_baselines_mostly_silent` and
`test_req113_engine_finds_n1_on_the_example_run` from the cached runs.

### Static quality rules met as written (REQ-305, REQ-301)

All the selected rule sets stay on, and every threshold is unchanged (line length 99,
complexity 12, 8 arguments, 60 statements). No package file is excluded. Behaviour is the
same: each refactored function was compared with 833968f on reference cases (results,
written files, warnings), and each magic-value change by its syntax tree. The test files
of every changed module were run. Details and readings: `docs/spec/DECISIONS.md` §7.

| tool | before (833968f) | after |
|---|---|---|
| `ruff check evoparam benchmarks docs/tutorials examples` | 1 264 findings | 0 |
| `ruff format --check` | clean | clean (441 files) |
| `interrogate -v evoparam` | 68.3 % | 97.1 % |
| `mypy --strict evoparam` | 1 error | 0 |
| `pyright --verifytypes evoparam` | 96.2 % | 96.8 % |
| `# noqa` comments (same paths) | 231 | 197, each with its reason |

Ruff findings before, by rule:

- PLR2004: 1 046 (package 211, tests 834, benchmarks 1). Each value is now a named,
  commented constant.
- C901, PLR0912, PLR0915, PLR0911: 126 findings and 72 noqa. Now 0 and 0: the long
  functions are split into named private helpers.
- PLR0913: 60. The public signatures that the spec fixes, or that are public outside
  it, keep a per-line noqa naming the spec section and the counts (42, against 24
  before); every other function was refactored.
- PLR0917: 32, all on private functions and test helpers. All were refactored, and the
  rule stays on.

Other changes:

- PT028: the 27 per-line noqa on `existence.test_response` and `tf.stats.test_map` are
  replaced by a `per-file-ignores` entry for each of the two modules. These are
  statistical tests, not pytest tests.
- mypy: `patsy` and `pytensor` (optional) join `ignore_missing_imports`.
- `tests/test_ci_workflows.py`: the strict expected failures of
  `test_req305_ruff_check_passes` and `test_req305_docstring_coverage` are removed,
  because both checks now pass.
- pre-commit: the hooks are pinned to the lock-file versions. `pre-commit run --all-files`
  now passes. Its ruff hooks, and the CI lint job, also cover `tools/`, `validation/` and
  `tutorials/`. These had 275 findings and 24 unformatted files; both are now 0, and the
  generated files are unchanged. `docs/spec/SPEC.md` is in ruff's `extend-exclude`, so
  that the Markdown formatting of the hook leaves the reviewed specification as is.
- Exported helpers outside the spec now take their settings grouped:
  `permutation_test` and `shuffle_test` (`PermutationOptions`), `bayes.sample`
  (`NutsSettings`), `sim.artifacts.pulse_shape` (`AntiAliasFilter`) and
  `stimulation_artifact` (`StimulusPulse`, `RecoveryTail`); see DECISIONS.md §7.2.
- Merge of integration-2 (4f2c44d). Its changes are carried into the split functions,
  and the results equal integration-2's to the bit. The 67 new ruff findings, 5
  unformatted files and 2 mypy errors of the merged code are fixed (DECISIONS.md §7.9).

### Engine speed, part 3 (REQ-290 P1): same results to the bit, P1 still not met

REQ-290 P1 (one averaged waveform, 2 048 samples, `ccep`, `max_components=6`, 3 starts) is
about a fifth faster with the `fast` extra and still above its 50 ms target. Every result is
the same to the last bit, so no algorithm version changes.

Why P1 costs about twice the REQ-115 waveform: it fits a 1 s window (2 047 samples, against
1 006 for the 0.009-0.5 s window of REQ-115), so every evaluation costs twice as much, and its
waveform needs about 500 evaluations (70 compiled fits per call: three starts and the
forward search, which also tries a fourth component and rejects it). At f626c68 a fit spent
about 60 ms in the compiled kernel (Householder QR of the 2 050-row whitened design,
whitening, `exp`) and about 30 ms in Python around it; P1 does use the compiled fit for
every call.

Timings: same machine, `OMP_NUM_THREADS=1`, previous code (f626c68) and new code; the load
average was 5 to 9 on the 4 cores (other jobs). The two `run_benchmarks.py --suite perf`
runs were taken one after the other, not interleaved:

| workload | f626c68 | new code |
|---|---|---|
| P1 workload, core 3 (`taskset -c 3`), warm-up + 30 fits, three interleaved rounds: median (p10-p90) | 99.0, 105.2, 95.5 ms (p10-p90 93-108, 96-132, 92-109) | 90.3, 81.4, 81.5 ms (p10-p90 82-112, 77-91, 78-87) |
| compiled kernel alone: the 70 calls of one P1 fit replayed, median of 30 | 59.79 ms | 48.49 ms |
| `run_benchmarks.py --suite perf`: P1 median (p90) of 20 | 99.8 ms (109.1) | 81.5 ms (85.6) |
| the same run: P2 (one run) | 53.8 s | 32.7 s |
| P2 again, two interleaved rounds | 40.1, 42.8 s | 39.1, 35.0 s |
| the same run: P5, reproduce / original fit | 77.0 s / 65.2 s (1.18x) | 75.5 s / 60.0 s (1.26x) |
| the same run: P6, median of 10 | 0.937 s | 0.940 s |

What changed:

- `models/_fast.py`: the AR whitening adds its lags by groups of four in one pass over the
  samples, each term added in the same order as before; the finiteness checks of the QR
  input and of the residuals are one vectorised sum each. Kernel time per P1 fit: see the
  table.
- `models/engine.py`: the compiled fit starts before the closures of the NumPy path are
  built; shapes built from the same settings and the default bounds of the background are
  reused; `compute_features` of a component is computed once per fit for the
  cancelling-pair, suspicious-component and merge checks; the selection of the candidate
  extrema is vectorised (the same order, ties included); `np.r_` replaced by
  `np.concatenate`.
- `models/api.py`: the whitened residual sum of squares is computed once for the
  diagnostics and the goodness of fit. `shapes/_base.py`: `window_info` recognises the time
  axis of the previous call without hashing it.

Results, previous code against new code: the 70 kernel calls of one P1 fit replayed give
the same outputs to the bit; a battery of 112 fits (P1 on three channels, the five REQ-115
waveforms, `models.fit` on simulated `ccep`, `ccep_seeg`, `erp`, `tep`, `ecap` and
`rodent_ep` with seeds 0 to 3) gives every number identical; the new whitening matches the
previous one to the bit, signs of zeros included, on 435 cases (5 to 2 047 samples, AR
orders 0 to 12, strides 1 to 4, with and without start rows); the reference fixture test
(`test_determinism.py`) and the methods texts are unchanged. `ALGORITHM_VERSIONS` is
unchanged.

`benchmarks/results/linux-3.13.16-1f2ea055f5f5.json` replaces the run of bddedc9 (which
predates the compiled fit); `benchmarks/README.md` and the strict expected failures of
`tests/benchmark/test_timing.py` give its values. P2 and P5 stay met, P6 stays above
300 ms (the import of `scipy.stats`). P1 cannot reach 50 ms with the same rounding: the
kernel is bound by the floating-point work of the QR and the whitening (about 200 MFlop per
fit). Reaching it needs fewer evaluations (for example a cheaper rejection of the fourth
component of the forward search) or a blocked QR, both of which change the results within
rounding and need a new algorithm version and fixture.

### Engine speed, part 4 (REQ-290 P1): the search of a further component stops sooner

The forward search tried up to 2 x `max_components` residual extrema (12 for P1) before
concluding that no further component improves the criterion, and every fit ends with such a
step. It now stops after the six strongest extrema when none of them improves the criterion
by `delta` (`engine.CANDIDATE_SCAN = 6`); a step in which an extremum qualifies still scans
until its beam of three is full, as before. On the layer-2 development cells the first
qualifying extremum of a step was the strongest in 546 of the 559 steps that added a
component and never beyond the sixth. Results change for a few fits, so the algorithm
versions of `ResponseModel.fit_waveform`, `fit_group`, `existence.test_response`,
`trials.fit_trials` and `hierarchical.fit_hierarchical` are now 1.2 and that of
`spatial.fit_shared` 1.1; the reference fixture was regenerated (`python -m
evoparam.tests.reference`; its values move by at most 2.4e-12 relative, within the 1e-9 of
its test, as before this change). Engine functions touched: `_fit_waveform_core` (the loop
over the extrema) and the new constant `CANDIDATE_SCAN`.

P1 (core 3, `OMP_NUM_THREADS=1`, warm-up + 30 fits, three interleaved rounds, load average 9
to 13 on the 4 cores from other jobs): 84.6, 87.0, 87.8 ms (p10-p90 80-97, 78-109, 79-103)
before, 78.7, 68.5, 70.5 ms (p10-p90 72-88, 67-93, 69-76) after; the same three components.
One P1 fit makes 58 compiled fits and 472 evaluations instead of 70 and 513. P1 stays above
50 ms.

Quality, before (up to 2 x `max_components` extrema) → after (6), the same code otherwise.
Layer 2: the weekly cells of `validation/layer2.py`, 2 replicates per cell, development
seeds (`1000 j + r`) and test seeds (500 000 + `1000 j + r`), scored by `sim.score_fits`;
engine tests: the 60 waveforms of `tests/models/helpers.py` (`random_components`, the
REQ-104 to REQ-112 generator) on seeds 0-59 and 500 000-500 059; ds004080: the seven runs of
layer 3, `ccep` pipeline of `validation/robustness.py`, both readings of the published N1
rule:

| seeds | cells | fits | recall | precision | F1 | latency MAE | fits that differ |
|---|---|---|---|---|---|---|---|
| development (0+) | erp layer 2 | 100 | 0.745 | 0.942 → 0.967 | 0.832 → 0.841 | 0.918 → 0.927 samples | 1 |
| development (0+) | ccep layer 2 | 100 | 0.699 | 0.926 | 0.797 | 1.326 samples | 0 |
| development (0+) | tep layer 2 | 52 | 0.970 | 0.990 | 0.980 | 1.082 samples | 0 |
| development (0+) | ecap layer 2 | 52 | 0.680 | 0.861 | 0.760 | 1.047 samples | 0 |
| development (0+) | animal layer 2 | 52 | 0.840 | 0.903 | 0.870 | 0.968 samples | 0 |
| development (0+) | erna layer 2 | 52 | 0.517 | 0.912 | 0.660 | 1.873 samples | 0 |
| development (0+) | engine tests (REQ-104-112 waveforms) | 60 | 1.000 | 1.000 | 1.000 | 0.552 ms | 0 |
| test (500000+) | erp layer 2 | 100 | 0.745 | 0.980 | 0.846 | 0.912 samples | 0 |
| test (500000+) | ccep layer 2 | 100 | 0.740 | 0.960 | 0.836 | 1.123 samples | 0 |
| test (500000+) | tep layer 2 | 52 | 0.960 | 0.970 → 0.990 | 0.965 → 0.975 | 1.330 samples | 2 |
| test (500000+) | ecap layer 2 | 52 | 0.680 | 0.861 | 0.760 | 1.372 samples | 0 |
| test (500000+) | animal layer 2 | 52 | 0.850 | 0.924 | 0.885 | 1.077 samples | 0 |
| test (500000+) | erna layer 2 | 52 | 0.617 | 0.902 | 0.733 | 1.754 samples | 0 |
| test (500000+) | engine tests (REQ-104-112 waveforms) | 60 | 1.000 | 1.000 | 1.000 | 0.526 ms | 0 |

| ds004080, 7 runs, 615 channels | reading | kappa | recall | median abs. latency difference | within 2 ms | N1 (model) | fits that differ |
|---|---|---|---|---|---|---|---|
| before → after | paper | 0.386 | 0.438 | 3.57 ms | 0.386 → 0.404 | 99 | 32 |
| before → after | code | 0.402 | 0.882 | 3.47 ms | 0.367 → 0.400 | 99 | 32 |

Rejected after measurement (DECISIONS.md, step 4): three extrema (recall drops on the
`ecap`, `animal` and `erna` cells, for example `ecap` 0.680 → 0.640 on the development
seeds) and polishing the background once at the end instead of after every accepted
component (N1 kappa 0.386 → 0.347 on ds004080, paper reading).

### REQ-290 P3: `evoparam.fit` in memory below twice the data at the CI size, same results to the bit

At the `ci` size (50 trials x 64 channels x 4 096 samples, float64, 105 MB) the in-memory
peak of `evoparam.fit` was 3.3 times the data, against a bound of 2. Three copies of the
trials are gone:

- `TrialSet.average` (`data/containers.py`, `_average_trials`) averages blocks of channels of
  at most 16 MB instead of the selected trials of every channel at once.
- `existence.test_response` (`existence/api.py`) makes one copy of the trials it tests instead
  of two, laid out in memory as the previous double indexing left them (channel by channel),
  so that every sum adds the same numbers in the same order; for the baseline permutation test
  the copy holds only the samples from the start of the baseline to the end of the window.
- `existence/permutation.py`: the sign-flip sums of the null distribution go into one buffer
  reused by every chunk of flips, the t-series and their summaries are computed by blocks of
  channels, and the window is copied once (the masked copy only when there are masks). The
  repeated temporaries of about 31 MB had made glibc raise its mmap threshold, so that they
  stayed on the heap and the resident set grew.

Peak RSS increase, `run_benchmarks.py --suite ci` (medians of 3 runs before, 3 after):

| variant | before (`linux-3.13.16-bddedc9aa35f-ci.json`) | after (`linux-3.13.16-148319226b70-ci.json`) | bound |
|---|---|---|---|
| in memory | +331 MiB (3.31x the data) | +130 MiB (1.30x the data) | 200 MiB |
| memory-mapped | +331 MiB (3.31x the data) | +130 MiB (1.30x the data) | 537 MiB |

Results: 263 of 263 outputs are the same bytes before and after (averages of a plain, a
masked and a rejected-trials set, mean and robust, three chunk settings: 36; `test_response`
with five settings on the three sets: 195; the `evoparam.fit` table at 30 x 24 x 4 096: 32).
The full size (500 x 256 x 4 096, 4.19 GB) was not run: the memory-mapped variant needs the
4.19 GB file on a disk with about 5 GB free, and the in-memory one needs the data and its
peak in the machine's 15 GB next to the other jobs.

### REQ-290 records on the reference machine

`benchmarks/results/linux-3.13.16-148319226b70.json` and
`linux-3.13.16-148319226b70-ci.json` replace the files of 1f2ea05 and bddedc9;
`baseline-ci.json` is kept (its time medians were taken at a lower load than these `ci`
runs, load average 18, and would only loosen the comparison); `benchmarks/README.md` and the
expected failures of `tests/benchmark/test_timing.py` are regenerated from them. P6 on
Linux: median 0.121 s over 10 runs (target 0.300 s), so its expected failure is removed; so
is that of P3 in memory (1.30x). P1: median 80.4 ms over 20 runs (p90 99.6 ms), still above
50 ms; P2 56.8 s (load average 7.3); P5 1.27x its fit.

P2 runs four workers on the four cores that other jobs share: two earlier `perf` runs of
1483192 gave 71.0 s at load average 16.0; 81.0 s at load average 8.1, above its 60 s; the
run recorded was started when the load average fell below 5.5. Its work is unchanged: P2 at
8 channels with one job on core 3, 1f2ea05 against 1483192 interleaved, 2.68, 2.85, 2.98 s
against 2.60, 2.70, 2.93 s.

### REQ-068 timing: the first-use imports are no longer timed

- With the package root imported on first use (REQ-290 P6), the first artifact method of
  `benchmarks/bench_artifact.py` that needs scipy.signal (`polynomial`) paid its import
  (about 1 s) inside the timed call: 2.3 to 2.7 s against the 2 s limit. Each case now
  first runs untimed on a small set of the same layout. `polynomial`: 1.68 s; every
  method within its limit. The tf timing test (REQ-218) failed only under a parallel run
  of the whole suite and passes alone.

### §7.4 (5): Linux, macOS and Windows measured

- Windows 11 (AMD64, Python 3.12.2, NumPy 2.5.4), measured on a Windows PC: the 98 values
  of the fixture fit within 3.1e-12 of the reference (rtol 1e-9). With Linux x86-64 (0) and
  macOS arm64 (1.3e-12), `test_req375_platform_macos_and_windows` passes; the report lists
  the three records.

### REQ-290 P6: `import evoparam` in 0.13 s

- The package root imported every subpackage, and with them scipy.stats (0.53 s): 0.96 s
  on macOS arm64 (0.91 s on the reference machine). The subpackages and the root functions
  (`fit`, `test_response`, `read`, `detect_events`, `ResponseModel`, `reproduce`) are now
  imported on first use (PEP 562 `__getattr__`, `__dir__`, and `TYPE_CHECKING` imports for
  type checkers); `TrialSet`, `Evoked`, `Provenance`, `load_object`, `errors`, the logging
  functions and `prep` (13 ms, for the registration of the existence detector, now a light
  proxy that imports the existence test on its first call) stay eager.
- `benchmarks/run_benchmarks.py --suite perf --only P6` on macOS arm64: median 0.132 s over
  10 runs, target met. The committed perf results are those of the reference machine
  (Linux), so `test_req290_p6_import_under_300_ms` stays an expected failure until P6 is
  re-run there.

### REQ-298: evoparam.shapes covered at 100 %

- `evoparam/tests/shapes/test_grid_helpers.py` covers the last five lines and four branches
  of `evoparam.shapes._base`: the window cache when it is full, the fine feature grid, and
  the search on that grid without building it, checked against `np.searchsorted` on 2 005
  values (on the grid points, one ulp off them, outside, infinite and NaN) for both sides.
  Shapes coverage 99 % -> 100 % (lines and branches); the slow test of REQ-298 passes.

### Cross-projection test: sign-flip null of the profile peak (REQ-134)

- `test_response(method="projection")` has a third set of projections,
  `projection_pairs="signflip"`, now the default: every cross-projection, and the peak of
  the mean-projection profile compared with the peaks of the profiles recomputed under
  sign flips of whole trials (all patterns up to 4 096, a random draw beyond), so the choice
  of the response end at the peak is part of the null. On the 10 example channels of the
  paper it gives every published decision (the "disjoint" t-test: 0.8 to 1.0 by random
  pairing, mean 0.89); on pure white noise (10 trials, 200 channels) it rejects at 0.05 for
  alpha = 0.05 ("disjoint" 0.145, the paper's "half" 0.43). `test_req134_published_
  significance_with_the_default_pairs` is no longer an expected failure. Cost: 0.12 s per
  channel for 50 trials x 500 samples.
- Benchmark `core`, baseline `projection` re-run (T1, T2): AUC 0.938 -> 0.979, recall 0.65
  -> 0.88, F1 0.78 -> 0.89, specificity 0.992 -> 0.960, false-positive rate at p < 0.05
  0.064 -> 0.104 on the null cells; the result files, leaderboard and report regenerated
  (macOS arm64, so their input hashes are those of this platform).
- `detect_pulses` had lost its `verbose` decorator in the size-gate change (the new helper
  was inserted between them); restored (`test_verbose_detect_pulses`).
- The docs build test of REQ-302 runs with `EVOPARAM_DOCS_TESTS=1` (the `docs` extra is not
  in `all`, and the dependency test requires every skipped package to be in `all` or
  `bayes`).

### REQ-064 on real templates: the specification's version as a measured comparison

- REQ-064 (subspace >= 30 dB on synthetic EEG with real artifact templates) is met by the
  default on every TMS subject and SPES run (tested). The two expected failures were the
  specification's algorithm as written (95 % of the whitened energy, at most 6 directions,
  every sample projected), which cannot reach 30 dB on real templates (TMS median 1.6 dB,
  SPES 9.3 dB) and is kept as an option. They are now tests that the default is above
  30 dB and at least 20 dB above the specification's version on every overlay: the
  measured reason for the departure, as the spec-floor rule asks.

### Pulse detection: a second score when the median finds nothing (ds004977)

- In ds004977 (4800 Hz, the stimulating contacts switched to the stimulator) the artifact
  reaches a minority of the channels: on the median of the absolute first difference (spec
  §5.2 method 1, 16 channels) the pulses reach z 9 to 15 in 2 of the 4 runs, under the
  threshold of 20, and `detect_pulses` found none there (recall 0.50). When the median
  score has no crossing, `detect_pulses` now tries the 90th percentile of the non-flat
  picked channels (`evoparam.events._score.quantile_score`): z 47 to 322 at the pulses, at
  most 10 between them. ds004977: 20 of 20 pulses. Recordings where the median finds a pulse
  are unchanged (ds004080, Mayo 2048 Hz, synthetic sets: 184 tests passed); the score used
  is recorded in the provenance (`settings["score"]`: "median" or "p90").

### Pulse detection: the strength gate also holds on the size of the deflection

- On the Mayo runs at 2048 Hz, 29 false alarms (precision 0.84) were electrode pops: a
  channel saturating for an instant has a steep edge, so a robust z near the pulses' (27 to
  107 against 85 to 164), but a deflection of 0.1 to 2 mV against 35 to 45 mV for the
  stimulation artifact. `detect_pulses` now gates each candidate on the smaller of its
  relative z and its relative size (µV), the size of reference being the 90th percentile
  among the candidates that pass on z (pops can outnumber pulses: 47 candidates for 5
  pulses in one ds004696 sub-07 segment). Mayo 2048 Hz: 152 hits, 0 false alarms
  (precision 1.0); a 3 mV pulse of the same site is still found by the template pass.
  ds004080 (REQ-040), the synthetic sets (REQ-041), the stream and benchmark tests are
  unchanged (229 passed).
- REQ-044, ds004696 sub-07: the run's own events.tsv has 13 pulses per site (median), so its
  test checks the run's structure (one train per block, the block's pulse count, 0.2 Hz);
  the protocol's 10-12 stays checked over all runs together. Reading in DECISIONS.

### N1 case: the registry row read as a negative peak (REQ-113, REQ-114, REQ-135, REQ-273)

- The reading. The §5.6 row is "`|x| > 3.4·SD_baseline` with a negative peak, ≥ 1
  sample" in 9–100 ms. A peak is a local extremum. The window's most negative sample on
  a monotone slope (its first sample on the recovery tail, its last on a drift) is not
  one. The new option `peak="strict"` of the registry rules implements the word: the
  deepest strict local minimum of the window, lower than both neighbours (read outside
  the window at its edges), below -3.4 SD of the baseline. It has no SD floor and no
  selectivity; those stay with `vanblooijs2023_code`. Like the other readings, it uses
  the baseline (-1, -0.1) s and the data's time axis. It is now the gating reading of the
  N1 tests (`FRAMES["paper"]`). The window's most negative sample is kept as
  `FRAMES["extremum"]`, measured by `test_req113_extremum_reading` and
  `test_req114_extremum_reading_in_the_rules_reference`. The registry's default for
  `vanblooijs2023` stays `peak="extremum"`; DECISIONS §5.8 explains why. A unit test
  covers the operator (`test_vanblooijs2023_strict_peak_reading`).
- REQ-113, not met (negative peak; window's most negative sample in brackets):

  | run | rule N1s | kappa | rule N1s missed |
  |---|---|---|---|
  | example, 46 PT contacts | 11 (20) | 0.01 (0.24) | 19 on all channels (25) |
  | UMCU02 | 5 | 0.37 | 1 |
  | UMCU03 | 30 | 0.60 | 12 |
  | UMCU04 | 9 | 0.09 | 8 |
  | UMCU05 | 12 | 0.35 | 7 |
  | UMCU06 | 25 | 0.42 | 9 |
  | UMCU01 run-061233 | 25 | 0.32 | 17 |

  Pooled over the REQ-371 runs 0.44 (0.44), over the development runs 0.37 (0.38).
  The word "peak" removes the window-edge detections: 9 fewer on the example's PT
  contacts. It does not bring the rule closer to the preset's N1.
- The remaining disagreements, classified:
  - Example run, PT contacts:
    - 10 of the 11 rule N1s peak after 50 ms, outside the N1 window (9–50 ms) that
      REQ-113 names for the preset; the preset misses 9 of them.
    - The preset's 6 N1s the rule does not find: 3 are peaks beyond 3.4 SD on the
      preset's own input (the adjusted common average) but not on the monopolar average
      the rule reads; 3 stay below 3.4 SD on both.
  - REQ-371 runs, 54 rule N1s missed:
    - 24 peak after 50 ms;
    - 30 peak in the N1 window: 11 of them are peaks the preset's input does not have
      (the reference), and 19 are peaks it has where the fit names no N1 component. Of
      the 30, the fitted curve has a trough in 9–50 ms on 13.
  - REQ-371 runs, 33 preset N1s the rule does not find: 9 are peaks beyond 3.4 SD of the
    preset's input and 24 are not. On the monopolar average the rule reads, 11 of the 33
    have a peak of 2–3.4 SD, 12 a peak below 2 SD and 10 no negative peak at all.
  - Development runs: of the rule N1s missed, 21 are late, 2 are the reference and 8
    have no N1 named. Of the 34 preset N1s the rule does not find, 6 are peaks of the
    preset's input and 28 are not.
  - The trough oracle (an N1 wherever the rule's own average has a 9–50 ms trough below
    -3.4 SD) reaches 0.132 on the example's PT contacts. On the REQ-371 runs it reaches
    1.000, 0.971, 0.865, 0.839, 0.789 and 0.458 (0.816 pooled; development runs 0.776).
    It meets both conditions of REQ-113 on two runs only, so no N1 detector can meet
    REQ-113 under this reading.
  - Variants (pooled REQ-371 / development runs):

    | variant | REQ-371 runs | development runs |
    |---|---|---|
    | recording reference with a constant background (best) | 0.508 | 0.431 |
    | fit window to 0.3 s | 0.497 | 0.347 |
    | N1 gated by the existence test without correction | 0.495 | 0.555 |
    | ccep 1.1 | 0.442 | 0.372 |

- D8 bounds with this reading: 0.378 on the PT contacts; 0.738, 0.387, 0.478, 0.783,
  0.752 and 0.557 on the REQ-371 runs; 0.655 pooled. The two readings agree at kappa
  0.13 on the PT contacts and 0.385 pooled.
- Channels the rule detects whose average has no 9–50 ms trough: 9 of 11 on the
  example's PT contacts; 0, 1, 1, 2, 5 and 12 on the REQ-371 runs; 6, 0, 0, 8 and 2 on
  the development runs.
- REQ-114 (negative peak; window's most negative sample in brackets; code unchanged at
  37 %, 80 %, 90.0 % and 90.0 %):

  | reading of the model latency | REQ runs | development runs |
  |---|---|---|
  | as written (N1 component) | 39 % (22 of 57) (26 %) | 31 % |
  | N1 named by its trough | 40 % (28 %) | 34 % |
  | fitted trough | 61 % (45 %) | 59 % |
  | rule's peak on the fitted curve | 72 % (86 %) | 66 % |
  | the same in the rule's reference | 82 % (93 %) | 84 % (89 %) |

  The rule's-reference reading is no longer met with the registry row; it stays met with
  the code. 7 of its 10 misses on the REQ runs are rule peaks at 87–100 ms. These are
  small local minima of the noisy average on a slope, where the smooth fitted curve has
  no local minimum. The word "peak" read on noisy data still counts such wiggles. A
  selectivity would remove them, but the code's 20 uV belongs to the code and the row
  states none.
- REQ-135, not met (negative peak; window's most negative sample in brackets):
  - default method with FDR: 0.495 (0.586) on the REQ-371 runs, 0.533 (0.600) on the
    development runs;
  - per-channel decision (`correction="none"`): 0.710 (0.807), the regression test still
    holds; development runs 0.756 (0.808);
  - the preset's own chain with FDR: 0.548 (0.623);
  - bound with the code reading: 0.640.
- REQ-273: not met.
- Layer 3/4 rescored with this reading (`robustness.py --rescore-rules`), model decisions
  of the 2026-10-11 run:
  - `n1_ds004080.json`, registry row: kappa 0.410 to 0.382, recall 0.429 to 0.438,
    median latency difference 8.41 to 3.57 ms, n_rule 161 to 130, n_both 69 to 57;
  - sampling drop at 1024 Hz 3.85 points (met); code reading unchanged (8.82, not met);
  - `comparisons.json` and `fig_n1_ds004080.svg` regenerated.
- Package defaults: none changed (`peak="strict"` is a new option; the registry row's
  default is unchanged), so no algorithm version changed and the reference fixture is
  unchanged.

### N1 case: the comparator trail (REQ-113 met on 2026-10-08, not met since)

On 2026-10-08 (73788a2) REQ-113 was recorded as met: on run-021448 the rule found 4 N1s
and the model 4, kappa 1.00 on the 46 PT contacts, 0.91 pooled over the REQ-371 runs.
Today the registry row finds 20 on the PT contacts and kappa is 0.24. The rule's code
(`evoparam/existence/rules.py`) has not changed since then (format and typing only), and
the data path (every trial of the pair, read the same way) has not changed either.
What changed is which comparator and which model decision the test uses.

- Commit by commit (example run: rule N1s on all 100 channels / on the 46 PT contacts;
  kappa on the PT contacts; kappa pooled over the REQ-371 runs, five subjects before
  361135c and six from it on, owner decision D14):

  | commit | comparator (the rule) | model decision | rule N1s | kappa PT | pooled |
  |---|---|---|---|---|---|
  | 911c750 (S7) | the extremum of 9-100 ms below 3.4 x max(SD of (-0.5, -0.02) s, 50 uV), read on the preset's own average (template removal, adjusted common average), 8 trials, 120 contacts | N1 component | 8 (4 at the window start, 9.8 ms; the rest near 100 ms) | - | - |
  | 135d387 | the published code (mnl_ccepAge): monopolar average of every trial, 50 uV SD floor, local minimum, SD over (-2, -0.1) s, the code's time axis (one sample later) | the fitted trough in 9-50 ms deeper than 3.4 x max(SD, 50 uV) of the rule's own average | 4 / 1 | 0.66 | 0.87 |
  | 73788a2 | unchanged | unchanged (engine: selection criterion on the innovation rows; PT03 no longer a model N1) | 4 / 1 | 1.00 | 0.91 |
  | 361135c (F65, F1) | the registry row of §5.6 (no floor, extremum); the code's reading becomes `vanblooijs2023_code`; still the code's baseline and time axis | the preset's own: a negative component in 9-50 ms | 32 / 19 | 0.51 | 0.45 |
  | 636a266 | unchanged | unchanged (`ccep` 1.1, `threshold_role="select"`) | 32 / 19 | 0.46 | 0.46 |
  | this change | the registry row on the data's time axis with the registry's baseline (-1, -0.1) s | unchanged | 36 / 20 | 0.24 | 0.44 |

- The two changes of 361135c, one at a time, with today's fits (kappa on the PT
  contacts / pooled over the REQ-371 runs / pooled over the development runs):

  | comparator | the 2026-10-08 decision (fitted trough judged by the code's floor) | the preset's N1 (REQ-113) |
  |---|---|---|
  | code's reading (the 2026-10-08 comparator) | 1.00 / 0.907 / 0.780 (5 of 6 runs pass) | 0.19 / 0.401 / 0.387 |
  | registry row, code's baseline and time axis | 0.06 / 0.284 / 0.590 | 0.46 / 0.463 / 0.399 |
  | registry row, data's time axis | 0.09 / 0.292 / 0.614 | 0.38 / 0.451 / 0.380 |
  | registry row, baseline (-1, -0.1) s | 0.04 / 0.287 / 0.541 | 0.35 / 0.455 / 0.378 |
  | registry row, both (this change) | 0.06 / 0.294 / 0.561 | 0.24 / 0.443 / 0.377 |

  On 135d387 the S7 engine's N1 component scored 0.17 on the PT contacts against the
  code's reading; the method of 135d387 and 73788a2 was chosen on development subjects
  that included the example run itself (CHANGELOG of 135d387). The 2026-10-08 result is reproduced exactly with today's fits (1.00 and 0.907), so
  the engine changes after it did not cause the drop. Each change of 361135c alone
  removes it: 1.00 becomes 0.19 with the preset's N1 decision, and 0.06 with the
  registry row.
- Judged against the spec text:
  - The 50 uV SD floor and the local minimum (with the 20 uV selectivity, the first
    sample excluded, and the 3000 uV saturation) belong to the published code. The
    §5.6 row reads `|x| > 3.4·SD_baseline` with a negative peak, ≥ 1 sample, 9–100 ms.
    It has no floor (3.4 x 50 uV = 170 uV is the smallest N1 the code accepts) and no
    peak selectivity. F65 made the registry rule faithful to that text.
  - REQ-113 asks that "the default `ccep` preset (§6) detects an N1 (negative component
    with latency in 9–50 ms)". The 2026-10-08 decision was something else: the trough
    of the fitted curve, kept when it was deeper than the rule's own amplitude criterion
    computed on the rule's own average. The detection borrowed the comparator's
    threshold and its input. F1 made the decision the preset's own.
  - The earlier 1.00 also rested on one channel: the code finds 1 N1 on the 46 PT
    contacts, so a single disagreement gives 0.66 (as on 135d387). The 1.00 was
    therefore not REQ-113 as written. It is kept as the regression test
    `test_n1_trough_with_the_code_floor_regression` (1.00 and 0.91), whose docstring now
    says so.
  - Two details of the code stayed on the registry row after F65 and are restored here:
    - The baseline. The row leaves it unspecified, and the registry names the ER-detect
      default (-1, -0.1) s. REQ-135's test and the benchmark already used it, while
      REQ-113's test used the code's (-2, -0.1) s.
    - The time axis. The code labels the stimulation sample t = 1/sfreq, so its 9 ms is
      one sample earlier on the data's axis: 7.8 ms instead of 9.8 ms on the 512 Hz
      example run, inside the template's (-2, 8) ms window.

    The registry row is now read on the data's own axis with the registry's baseline,
    and the code's reading keeps its own baseline and axis (`FRAMES` in
    `tests/models/test_engine_ccep.py` and in `validation/robustness.py`). This change
    follows the spec text, not the results, and it lowers the agreement (0.46 to 0.24
    on the PT contacts).
- Re-measured with the registry row in its own frame. The code's reading is unchanged,
  and the REQ-135 test already read the row this way.
  - REQ-113: 0.24 on the PT contacts (20 rule N1s, 8 model), 0.44 pooled over the REQ-371
    runs, 0.38 pooled over the development runs.
    - Per REQ-371 run: 0.44, 0.58, 0.25, 0.33, 0.47, 0.31.
    - Variants (pooled REQ runs / development runs):

      | variant | REQ runs | development runs |
      |---|---|---|
      | constant background | 0.492 | 0.383 |
      | linear background | 0.473 | 0.372 |
      | recording reference with a constant background | 0.491 | 0.413 |
      | fit window to 0.3 s | 0.482 | 0.353 |
      | `delta` 6 | 0.460 | 0.389 |
      | N1 gated by the existence test over 9-100 ms (FDR / none) | 0.484 / 0.497 | 0.510 / 0.534 |

      The gated variant without correction is the best on the development runs, and it
      gives 0.497 on the REQ runs.
  - Channels the row detects with no N1 trough in its average:
    - example run: 16 of the 20 PT detections (3 at the window's first sample, 9.8 ms;
      13 at 94-100 ms);
    - REQ-371 runs: 0, 1, 5, 2, 6, 12;
    - development runs: 9, 7, 0, 8, 2.
  - Trough oracle: kappa 0.056 on the example's PT contacts, 0.715 pooled over the REQ
    runs, 0.674 over the development runs.
  - The exact joint bound (D8): 0.274 on the PT contacts (it was 0.295), 0.323 to 0.783
    per REQ-371 run, 0.607 pooled.
  - REQ-114 (registry row / code; REQ runs, then development runs):

    | reading of REQ-114 | REQ runs | development runs |
    |---|---|---|
    | as written (N1 component) | 26 % (18 of 69) / 37 % | - |
    | N1 named by its trough | 28 % / 40 % | 31 % / 38 % |
    | fitted trough | 45 % / 80 % | - |
    | rule's peak on the fitted curve | 86 % / 90.0 % | - |
    | the same in the rule's reference | 93 % (64 of 69) / 90.0 % (met, both; margin one channel each) | 89 % / 95 % |

  - REQ-135: unchanged (0.586 / 0.496). REQ-273: not met (it needs REQ-113).
- Layer 3 and layer 4 rescored with the new comparator. Nothing was fitted again: the
  model's decisions are those of the 2026-10-11 run. A new option, `python
  validation/robustness.py --rescore-rules`, recomputes only the rule's decisions and
  the summaries that depend on them.
  - `validation/layer3/n1_ds004080.json`, registry row: kappa 0.458 to 0.410, recall
    0.460 to 0.429, n_both 74 to 69. The code's values are unchanged.
  - Sampling (§7.4 (3)), registry row: drop 4.97 points (still met). Code reading: 8.82
    (not met, unchanged).
  - `validation/layer3/comparisons.json` and `figures/fig_n1_ds004080.svg` were
    regenerated with their tools; `layer3_report.py` now records each reading's own
    baseline.
- Package defaults: none changed. No algorithm version was bumped and the reference
  fixture is unchanged.

### N1 case: the two readings of the published rule (REQ-113, REQ-114, REQ-135, REQ-273)

Work on the N1 requirements under both readings of the published rule (owner decision D8),
on the example run and the six REQ-371 runs (D14). Choices were measured on the five
development runs (UMCU07, 19, 28, 40, 49: every trial of the pair the earlier work cached)
and are reported on the REQ runs, which were not used to choose anything. No default of the
package changed: every candidate that helped one reading made the other, or another
requirement, worse (below). Package results are unchanged, so no algorithm version moves
and the reference fixture is not regenerated. Only tests and documents change.

- **The readings were checked against the published code.** The analysis scripts of
  van Blooijs et al. 2023 (mnl_ccepAge, branch master: `scripts/ccep01_averageCCEPsAndDetectN1s.m`,
  `functions/ccep_averageConditions.m`, `ccep_detect_n1peak_ECoG.m`, `ccep_peakfinder.m`)
  were read again. `vanblooijs2023_code` follows them: per-trial median over (-1, -0.1) s,
  mean, median over (-2, -0.1) s, SD floor 50 uV, local minima with a selectivity of 20 uV
  from the first sample after 9 ms to the first after 500 ms, first sample excluded, the
  largest kept up to 100 ms, refused above 3000 uV or when every minimum is positive. The
  time axis of the code is one sample late (the test shifts it back). The published
  pipeline does not re-reference: the rule reads the recording reference, the `ccep` preset
  the adjusted common average.
- **D8 cannot hold for REQ-113 and REQ-135, whatever the detector.** For any decision of the
  channels, the smaller of its kappas with the two readings has an upper bound. Kappa
  depends only on how many channels the decision marks in each of the four cells of the
  readings' table, so every combination was searched (exact; checked against a brute force
  over all decisions of 12 channels). The two readings agree with each other at kappa 0.06
  on the 46 PT contacts of the example (19 and 1 N1s) and 0.31 pooled over the REQ-371
  runs (125 and 28).

  | data | bound of min(kappa paper, kappa code) | asked |
  |---|---|---|
  | REQ-113, example run, 46 PT contacts | 0.295 | 0.8 |
  | REQ-113, REQ-371 runs UMCU02 / 03 / 04 / 05 / 06 / 01 | 0.656 / 0.343 / 0.323 / 0.783 / 0.681 / 0.563 | 0.8 on 5 of 6 |
  | REQ-135, REQ-371 runs pooled (rules on the test's trials) | 0.592 (0.600 on the published averages) | 0.7 |
  | REQ-135, development runs pooled | 0.698 | 0.7 |
  | REQ-135, the segments the earlier test used | 0.6996 | 0.7 |

  New tests (not REQs, measured facts): `test_d8_req113_cannot_hold_under_both_readings`
  and `test_req135_cannot_hold_under_both_readings`, with the exact search
  `joint_kappa_bound`. An owner decision is needed (DECISIONS §5.8).
- **REQ-135 as written runs on the REQ-371 subjects.** Spec §7.3 names the subjects of
  REQ-135: the ones chosen by the rule of REQ-371. The test used the example subject and the
  five development subjects (40 s segments, 7 or 8 trials). It now uses the six REQ-371
  subjects with every trial of the pair (as REQ-113). The method and the two rules see the
  same trials (template artifact removal, baseline -1 to -0.1 s). The registry row gets the
  ER-detect baseline the registry names. The code keeps its own baseline (-2 to -0.1 s;
  the segments were too short for it). The regression floor of the code's SD floor stays
  on the segments.

  | existence method, kappa with registry row / code | REQ-371 runs | development runs |
  |---|---|---|
  | default (max_t, fdr_bh, alpha 0.05, 9-100 ms, negative) | **0.586 / 0.496** | 0.600 / 0.597 |
  | `statistic="cluster"` | 0.352 / 0.567 | 0.534 / 0.622 |
  | `correction="none"` (the rule's per-channel decision) | **0.807** / 0.321 | 0.808 / 0.499 |
  | `correction="holm"` | 0.128 / 0.497 | 0.327 / 0.465 |
  | `alpha=0.01` | 0.139 / 0.587 | 0.420 / 0.524 |
  | `polarity="both"` | 0.349 / 0.220 | 0.139 / 0.053 |
  | default, on the preset's input (adjusted common average, its baseline) | 0.623 / 0.384 | 0.703 / 0.524 |
  | `correction="none"`, preset's input | 0.642 / 0.262 | 0.686 / 0.395 |

  Before (segments): 0.687 / 0.399. With `correction="none"` the registry row is met on the
  held-out runs (0.807; chosen on principle: the rule decides each channel alone, and no
  value was tuned). The code reading then falls from 0.496 to 0.321. Spec §6 also asks
  for FDR across channels in every preset. So the preset keeps `fdr_bh`, and the setting
  is kept as a regression test (`test_req135_per_channel_decision_registry_row_regression`).
  REQ-135 stays an xfail under both readings.
- **REQ-114: the rule's peak read on the fitted curve.** A new reading of "the model
  latency": each reading's own peak operator applied to the curve of the ccep fit (its
  components and background) instead of the noisy average. The baseline (SD, level) is the
  rule's average. So a window-start peak on the recovery tail, or a late deflection of
  50-100 ms, is compared with the model's curve at the same place.

  | REQ-114 reading of the model latency, REQ runs (example + REQ-371), registry row / code | within 2 ms |
  |---|---|
  | the N1 component (as written) | 24 % (18/74) / 37 % (11/30) |
  | trough of the fitted curve in 9-50 ms (K3) | 42 % (31/74) / 80 % (24/30) |
  | trough of the fitted curve in 9-100 ms | 45 % / 80 % |
  | trough of the components without background | 31 % / 53 % |
  | negative component nearest the trough; earliest negative component | 34 % / 63 %; 23 % / 30 % |
  | the rule's operator on the components without background | 50 % / 50 % |
  | the rule's operator on the fitted curve | 84 % (62/74) / 90.0 % (27/30) |
  | the same, background extrapolated one sample before the fit window | 81 % / 83 % |
  | **the rule's operator on the fitted curve in the rule's reference** | **91 % (67/74) / 90.0 % (27/30)** |

  On the development runs the new reading gives 66 % (23/35) and 86 % (18/21). Each miss
  was classified: the rule's peak absent from the preset's own input average (the
  reference: 10 and 3 on the REQ runs, 8 and 0 on the development runs), a tie within
  noise (the average differs by less than 2 SD of its baseline between the two
  latencies: 2 and 0; 4 and 3), or a misfit (0 everywhere). The rule's own peak is not
  much more reproducible: re-measured on 200 bootstrap resamples of its trials, it stays
  within 2 ms of itself in 90 % / 91 % of resamples on the REQ runs (median channel 99 % /
  100 %) and 87 % / 93 % on the development runs. New test
  `test_req114_rule_peak_on_the_fitted_curve` (xfail; with the code it passes, at the
  threshold).
- **REQ-114 met under both readings, read in the rule's reference.** The reference misses
  come from where the two sides are read: the preset fits the adjusted common average,
  and the rule reads the recording reference. The re-referencing is linear. It subtracts
  one curve from every channel: the common average of its members, the same for all
  channels to 1e-12 uV. That curve, baseline-corrected and averaged like the data, is
  added back to every fitted curve, and the rule's operator then reads the sum. Nothing
  is refitted and the preset's fits, labels and decisions are unchanged. This is a
  principled step with no parameter: it was measured on the development runs and
  reported on the REQ runs.

  | rule's peak on the fitted curve in the rule's reference | registry row | code |
  |---|---|---|
  | REQ runs (example + REQ-371), compiled fit | **91 % (67/74)** | **90.0 % (27/30)** |
  | REQ runs, NumPy path (no numba) | 91 % (68/75) | 90.0 % (27/30) |
  | development runs, compiled fit | 91 % (32/35) | 95 % (20/21) |
  | development runs, NumPy path | 89 % (31/35) | 95 % (20/21) |

  REQ-114 is met under both readings with this reading. The margin is zero channels on
  the REQ runs: one more miss in 74 or in 30 would fall below 90 %. New test
  `test_req114_rule_peak_on_the_fitted_curve_in_the_rules_reference` (passes, both
  readings). The test as written (the N1 component's latency) stays an xfail.
- **Model and pipeline variants** (every trial of the pair; REQ-113 kappa pooled over the
  REQ-371 runs, REQ-114 with the new reading; registry row / code):

  | variant | REQ-113 kappa, REQ runs (development) | REQ-114 new reading, REQ runs (development) |
  |---|---|---|
  | ccep 1.1 (adjusted common average) | 0.463 / 0.401 (0.399 / 0.387) | 84 % / 90 % (66 % / 86 %) |
  | recording reference (no re-referencing, as the published pipeline) | 0.421 / 0.428 (0.471 / 0.512) | 88 % / 88 % (91 % / 95 %) |
  | recording reference, `delta` 6 | 0.436 / 0.423 (0.464 / 0.504) | 87 % / 86 % (91 % / 95 %) |
  | plain common average | 0.443 / 0.382 (0.405 / 0.378) | 76 % / 89 % (70 % / 79 %) |
  | `delta` 6 | 0.480 / 0.371 (0.412 / 0.381) | 85 % / 90 % (64 % / 86 %) |
  | `max_components` 8 | 0.452 / 0.401 (0.399 / 0.387) | 82 % / 87 % (60 % / 81 %) |
  | fit window to 0.3 s | 0.513 / 0.373 (0.356 / 0.381) | 82 % / 93 % (76 % / 86 %) |
  | background `double_exponential` | 0.475 / 0.324 (0.338 / 0.460) | 85 % / 88 % (68 % / 89 %) |
  | N1 counted only where the preset's existence test is significant (9-500 ms, both signs) | 0.442 / 0.562 (0.431 / 0.443) | 86 % / 90 % |
  | the same, test over 9-100 ms, negative | 0.469 / 0.560 (0.514 / 0.549) | 86 % / 89 % |
  | the same, `correction="none"` | 0.494 / 0.528 (0.540 / 0.598) | 88 % / 90 % |

  The recording reference is the only change that removes the reference misses of
  REQ-114. It reaches 90 % on the development runs under both readings, but not on the
  REQ runs (88 % / 88 %). Its REQ-113 kappa falls with the registry row (0.463 to 0.421;
  example PT contacts 0.46 to 0.24). It also gives up the adjusted common average that
  spec §6 asks for (common-mode deflections, such as sub-ccepAgeUMCU03's at 32.7 ms on
  35 of 70 channels, would count as N1s). Not adopted. No variant brings REQ-113 near 0.8
  under either reading.
- Results:
  - REQ-113: not met (0.46 / 0.19 on the PT contacts, 0.46 / 0.40 pooled; unchanged).
  - REQ-114 as written: not met (24 % / 37 %; unchanged).
  - REQ-114, the rule's peak read on the fitted curve in the rule's reference: met
    (91 % / 90.0 %).
  - REQ-135: not met (0.586 / 0.496 on the REQ-371 runs; before, on other subjects,
    0.687 / 0.399).
  - REQ-273: not met (it needs REQ-113). Tests run:
  `tests/models/test_engine_ccep.py` and `tests/existence/test_existence_ccep.py` with
  `EVOPARAM_NETWORK_TESTS=1`, `tests/test_presets.py`, the reference fixture test.
- D8 revised by the owner (DECISIONS §5.8): the N1 requirements are judged on the
  registry row (§5.6: `|x| > 3.4·SD_baseline`, a negative peak, 9-100 ms). The published
  code's reading stays measured by its own expected failures. The exact joint bound is the
  evidence that both cannot hold together: REQ-113 at most 0.295 on the PT contacts and
  at most 0.783 on 5 of the 6 REQ-371 runs; REQ-135 at most 0.592.
- REQ-113 under the registry row, channel by channel (example run and REQ-371 runs;
  settings chosen on the five development runs only):
  - Many channels the rule detects have no N1 to find: the rule takes the most negative
    sample of 9-100 ms, which is often the first sample of the window (the stimulation
    tail) or a slope at 93-99 ms. Channels where the rule detects but the rule's own
    average has no negative local minimum in 9-50 ms: 12 of the 19 detections on the
    example's PT contacts; 0, 0, 5, 2, 6 and 12 on the REQ-371 runs; 7, 9, 0, 6 and 3 on
    the development runs. "Every channel the rule detects minus at most one" cannot hold
    for a detector of an N1 on the example run, nor on 4 of the 6 REQ-371 runs. New test
    `test_req113_registry_row_counts_channels_without_an_n1_trough`.
  - Oracle: a detector that knows the rule's average and says N1 wherever it has a
    negative trough in 9-50 ms below -3.4 SD reaches kappa 0.061 on the example's PT
    contacts and 0.788, 0.971, 0.392, 0.794, 0.637 and 0.477 on the REQ-371 runs (0.717
    pooled; 0.681 pooled on the development runs). Even this upper reference for an N1
    detector misses 0.8 on the example run and on 5 of the 6 REQ-371 runs.
  - Background variants (to absorb the stimulation tail that the rule counts), kappa
    pooled over the REQ-371 runs (development runs): constant background 0.498 (0.425),
    linear background 0.491 (0.397), recording reference with a constant background
    0.533 (0.400); ccep 1.1 0.463 (0.399). None is adopted: none improves the
    development runs, and none comes near 0.8.
- REQ-135 under the registry row with FDR kept (§6), on the development runs (kappa
  pooled): the test on the trials as before 0.600; baseline 0.5 s 0.600; baseline 2 s
  0.638; without artifact removal 0.600; on the ccep preset's own chain (adjusted common
  average, artifact template) 0.703, with baselines 1 s / 2 s / 0.5 s 0.687 / 0.711 /
  0.690. FDR over the channels of all runs pooled instead of per run: 0.574 to 0.711.
  The preset's chain, chosen on the development runs, gives 0.623 on the REQ-371 runs
  (0.586 before). The loss is FDR itself: on the development runs 22 channels the rule
  detects have a raw p below 0.05 and are lost by the correction; one run (UMCU40) has
  kappa 0 under every FDR variant. Without the correction the registry row is met
  (0.807, test `test_req135_per_channel_decision_registry_row_regression`). No default
  changed.
- REQ-114 as written (the N1 component's latency), naming the N1 by the trough it makes
  (the component whose fitted curve has the deepest trough in 9-50 ms, rather than the
  larger `|A|`): 29 % / 38 % on the development runs (registry row / code), 26 % / 40 % on
  the REQ runs (24 % / 37 % before). With the recording reference and a constant
  background, 50 % / 57 % on the development runs and 42 % on the REQ runs. The fitted
  N1 is several overlapping components, and none of them carries the rule's peak
  latency alone. Not adopted; the reading in the rule's reference stays the one met.
- Results after the D8 revision (registry row, which gates):
  - REQ-113: not met (0.46 on the PT contacts, 0.46 pooled).
  - REQ-114: met read in the rule's reference (91 %); as written not met (24 %).
  - REQ-135: not met with FDR (0.586; best principled variant 0.623).
  - REQ-273: not met (it needs REQ-113).
  No package default changed, so no algorithm version changed and the reference fixture
  is unchanged.
- To try next: REQ-113 needs a reading of the rule where its detections are N1s (for
  instance the rule applied only where its average has a trough, or the 9-50 ms window of
  §5.5.7); this would need an owner decision, as the criterion's comparator is fixed.
  REQ-135 with FDR: a test whose statistic uses more than 10 trials (other pairs of the
  same contacts) would raise the power that FDR takes away.

### ERP CORE: the paradigm's component, measured by a targeted `erp` preset (REQ-274)

- `validation/erpcore.py` fits each difference wave with `presets.targeted("erp", ...)`:
  three components of the component's polarity in the published measurement window widened
  by 25 ms, no background (the waves are baseline-corrected); the component of the paradigm
  is the sum of its labelled parts (`P3`, `P3_b`, ...), and its latency the 50 % area
  latency of that sum in the published window. Before, the `erp` preset with BIC selection
  found no component in many single-participant waves, and a single targeted component on a
  linear background put broad components (P3, N400) 8 to 16 ms early.
- Within 10 ms of the 50 % area latency of the same average (before -> now): ERN 85 -> 95 %,
  LRP 90 -> 95 %, MMN 37.5 -> 100 %, N170 47.5 -> 95 %, N2pc 36 -> 92 % (36 of 39), N400
  57.5 -> 95 %, P3 30 -> 90 %. The six expected failures of REQ-274 are gone. Variants
  measured on the same 40 participants (so development measures): one component on a linear
  background (P3 17.5 %, N400 35 %), one with no background (P3 57.5 %), two components
  (P3 87.5 %), four (P3 82.5 %), three on a constant background (P3 37.5 %).
- With a fixed number of components every wave gets a value: whether the component is there
  is the existence test's answer (spec §1, principle 2). The P3 meets 90 % with no margin.
- `presets.targeted` gains `n_components` (default 1) and `background` (default `"linear"`);
  its defaults and their measurements are unchanged.
- REQ-274 as written stays measured on its own (`validation/erpcore.py`,
  `reading="as_written"`: the `erp` preset's own selection, one labelled component; tests
  `test_req274_*_as_written`, expected failures with the 2026-10-09 values, LRP met): the
  targeted measurement is a reading of the requirement, with settings chosen on the same 40
  participants, and the requirement stays marked not met as written (DECISIONS 5.9).

### §7.4 (5): one platform record per machine; macOS arm64 measured

- `validation/robustness.py --analyses platform` now writes the fixture comparison of the
  machine it runs on to `validation/layer4/<version>/platforms/<os>-<processor>.json` and
  keeps the other analyses of the committed report; the report lists every record.
  Measured: Linux x86-64 (reference machine, 0 difference) and macOS 26.4 arm64 (Python
  3.14.7, NumPy 2.5.3: 98 values, largest relative difference 1.3e-12, within the 1e-9 of
  REQ-295).
- `.github/workflows/platform-record.yml` records Windows (x86-64) on a GitHub runner when a
  branch `platform-record/<name>` is pushed, and commits the record back to that branch.
- `test_req375_platform_macos_and_windows` checks every record and names the platforms
  still missing (Windows until its record is merged).
- Windows measured on a Windows 11 PC (x86-64, Python 3.12.2, NumPy 2.5.4, SciPy 1.18.1,
  core install `pip install -e .`): 98 values, largest relative difference 3.15e-12, within
  the 1e-9 of REQ-295. §7.4 (5) is met on Linux, macOS and Windows; the GitHub workflow
  could not run (the runner was refused for billing). The report's own `platform` block is
  the reference machine's record (Linux); every machine is in `platforms`.

### REQ-231: Kiesel et al. (2008) read in full

- The paper was read on all 25 pages (archived authors' copy; the build environment had
  only pp. 250-264). It prints no corrected t or F, only Equation 1, so the expected
  failure "no printed statistic" is replaced by a test of Equation 1 on jackknifed scores
  (12 participants x 3 conditions: corrected t and F equal those of the retrieved scores
  to 1e-9), next to the existing test of its worked interpolation (64.46 ms). Reading in
  DECISIONS.

### REQ-315: the reference validator test runs bids-validator 3.0.2 while npm has no 2.x

- `test_req315_reference_validator_as_written` was an expected failure because
  `npx bids-validator@2` cannot resolve (npm's latest is 1.15.0). It now runs npx when npm
  has a 2.x release and the maintainers' JSR release (`@bids/validator@3.0.2`, through
  deno) otherwise; the derivatives pass with zero errors. Reading recorded in DECISIONS.

### Documentation built with warnings as errors (REQ-302)

- `sphinx -W --keep-going -b html docs` now ends with 0 warnings (run on macOS with the
  `docs` extra and the MNE sample dataset): the benchmark report, the submission guide and
  the result-table page are in the table of contents; the module pages of the names
  re-exported at the package root (`evoparam.data.TrialSet`, `evoparam.models.ResponseModel`,
  ...) are written with `:no-index:` (templates in `docs/_templates/autosummary`), so each
  object has one target and the 29 ambiguous `TrialSet` references are gone; the stub of the
  constant `evoparam.io.OPENNEURO_URL` gets its own file name (`autosummary_filename_map`),
  since on case-insensitive file systems it overwrote the page of the function
  `openneuro_url`; the two renamed names that are also terms (`GroupFit`,
  `Component.params`) are listed as "(renamed)" in the glossary; the LSL code block of the
  streaming tutorial is a literal block.
- `test_req302_docs_build_with_warnings_as_errors` is no longer an expected failure: it
  builds the site with `EVOPARAM_SLOW_TESTS=1` when Sphinx is installed. The
  `latest`/`stable` switcher of the published site still waits for the HIIE decision.

### Engine speed, part 2 (REQ-115): met with the compiled fit of the `fast` extra

REQ-115 is met when numba is installed (extra `fast`, §9.2). The two tests of
`models/test_performance.py` pass with `EVOPARAM_PERF_TESTS=1`; their xfails now apply only
without numba, where the engine takes the NumPy path, which is unchanged in speed. The
one-waveform test now measures as §9.1 measures P1: one warm-up call, then 20 repetitions
(the first fit of a process loads the compiled code, about 0.7 s, and with five calls that
one call moved the median). The criteria are unchanged. REQ-290 P1 is still not met.

Timings: same machine, `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, previous code
(7048c1d) and new code run interleaved. The load average was 8 to 12 on the 4 cores
(other jobs), which inflates wall times more than CPU times:

| workload | before | after |
|---|---|---|
| one waveform (REQ-115 workload), median of 10 fits, three interleaved rounds: wall | 331, 421, 378 ms | 45, 35, 58 ms |
| the same, CPU time | 245, 317, 287 ms | 35, 35, 42 ms |
| REQ-115 test (20 fits after a warm-up), 7 runs | (xfail) | passes 7 of 7 |
| `fit_group`, 100 channels x 2 048 samples, `n_jobs=1`: wall (CPU) | 32.6 s (26.3 s) | 6.4 s (5.0 s), 4.7 s (4.3 s) |
| REQ-290 P1 (`benchmark.timing.p1`, median of 20), two interleaved rounds | 0.655 s, 0.638 s | 0.105 s, 0.099 s |
| ds004080 UMCU01 / UMCU07 / UMCU19, 120 channels each, `existence=False` | 40.3 / 118.0 / 225.8 s | 16.0 / 38.6 / 59.9 s |

The ds004080 times of the new code were taken at a load of 10 to 12, those of the previous
code at a lower load; at a load of about 7 an intermediate version took 9.1 / 27.7 /
46.6 s. P1 stays above its 50 ms target: its waveform needs about 500 evaluations, and
`benchmarks/results` and the strict xfail of `test_timing.py` still give the committed run.

What changed:

- `models/_fast.py` (new) fits one model structure in numba-compiled code: the same
  trust-region reflective solver as `models/_trf.py` (derived from SciPy, header
  `Apache-2.0 AND BSD-3-Clause`, recorded in `NOTICE`) on the same reduced residual.
  - It covers Gaussian and skewed-Gaussian components with an exponential background or a
    background without non-linear parameter (none, constant, linear), white, AR and
    strided whitening, with or without prior rows. Design columns, derivatives, whitening, Householder QR and the
    trust-region steps run without a Python call.
  - Any other structure, a robust loss, a design that is not finite or not clearly of full
    rank (the NumPy path's SVD route), and a machine without numba take the NumPy path,
    which stays the reference (`tests/models/test_fast_kernel.py` checks each case).
  - numba is imported by `_fast.kernel()` through `utils.require` the first time a fit
    needs it (REQ-292); `soup.json` and `THIRD_PARTY.md` now list it as imported.
  - Not the same rounding as the NumPy path: a vectorised `exp` (within one ulp) and
    Cephes' `erfc` (the approximations SciPy uses); components taken as 0 beyond 12 widths
    (below 1e-31 of the peak; less on the fast-decaying side of a skewed Gaussian, with the
    same bound); unblocked Householder QR; a one-sided Jacobi SVD for the trust-region
    step; fused multiply-adds and vectorised inner products; no final Jacobian, which SciPy
    computes and does not use. `docs/spec/DECISIONS.md` (§4) records the reading of
    REQ-295.
- The NumPy path: the AR orders of `ar_model` come from one QR of the lagged matrix
  instead of one least-squares solve per order (same solutions within 1e-9,
  `test_fast_paths.py`); bounds and start limits are cached per structure and per shape;
  `compute_features` builds only the part of its grid where the component is above a
  quarter of the onset level (bitwise equal on 15 000 cases); `as_times` checks finiteness
  with one sum.
- Fix: the whitened target cached for a fit was reused by the copies of a structure with
  other data (bootstrap, cross-validated selection) during this work; it is now kept with
  the data it belongs to (`test_copies_with_other_data_whiten_their_own`).

Quality, previous code against new code (`_FAST` auto, the default):

- Reference fixture: within 1e-9 with both paths (largest relative difference 2.4e-12
  compiled, 3.5e-12 NumPy path); `ALGORITHM_VERSIONS` unchanged.
- Simulations (REQ-104 to REQ-111, REQ-116, `ecap`, simulated `ccep`): every metric
  identical; largest per-fit difference 2.0e-10 s in latency and 1.3e-8 relative in
  amplitude; no status, number of components, shape or label changed.
- ds004080 N1 (REQ-114, 615 channels): kappa 0.4584 to 0.4620 (paper rule) and 0.3981 to
  0.4019 (code rule), recall unchanged (0.4596 and 0.8824).
- ds004080, three subjects, 360 fits: no status or shape changed; 348 within 1e-6 s and
  1e-5 relative; 11 differ at the solver tolerance (residual sum of squares within 5e-5
  relative, except UMCU19 PT10: +1.4 %, criterion -1826.58 to -1825.49); UMCU19 el088 goes
  from 5 to 6 components with a lower criterion (4380.30 to 4289.56), a channel that
  already moved under 1e-15 perturbations of the data ("Engine speed" below).

This supersedes "Tried and not adopted: numba" of "Engine speed (REQ-115, REQ-290 P1)"
below. A whitening kernel alone saved a few percent; compiling the whole fit removes the
Python cost of every evaluation, which is where the time was. The objection that the
compiled path cannot match NumPy to the last bit stands: it is met by reading REQ-295 with
the extra as part of the platform, and by the agreement above.

### Commands that aborted or hung at exit after their work (exit status -6)

An `evoparam` command that reads a container with parquet tables (`report`,
`reproduce`, `migrate`, ...) could print its complete result and then abort while the
interpreter was exiting: `terminate called without an active exception`, exit status
-6 (SIGABRT), seen in `test_report_reproduce_migrate` under heavy CPU load (Python 3.11
and 3.12). Cause: given a local path alone, `pandas.read_parquet` opens a Python file
object and hands it to pyarrow; pyarrow's reading threads drop the last references to
that file object and to the `bytes` read from it (`PyReadableFile` and `PyBuffer`
destructors, each taking the GIL) after `read_parquet` has returned. When such a
destructor waits for the GIL while the interpreter is being finalized, Python 3.11 and
3.12 end the thread with `pthread_exit`, whose unwinding through the C++ destructor calls
`std::terminate`; Python 3.13 parks the thread for ever instead, and the process then
hangs in `exit()`, where pyarrow's thread pool waits for that thread.

Measured on a process that reads one table and exits, under CPU load: Python 3.12, 20
aborts in 600 runs before, none in 600 after; Python 3.13, 4 hangs in 318 runs before,
none in 300 after; Python 3.11, none in 300 before or after. Forcing the interleaving
under a debugger (the reading thread delayed until the interpreter is finalizing) aborts
the minimal case and `evoparam report` / `evoparam migrate` every time before the fix
(3.11, 3.12) and never after: pyarrow's threads no longer take the GIL in a destructor.

- `_read_table` (`evoparam.data.serialize`) reads parquet through a pyarrow file system
  (`pd.read_parquet(path, filesystem=pyarrow.fs.LocalFileSystem())`): pyarrow opens the
  file and reads it into its own buffers, so its threads hold no Python object. The
  tables read are the same: on the 22 parquet tables of every result type and of a
  `TrialSet`, the frames (values, dtypes, index, columns, attrs) equal those of
  `pd.read_parquet(path)` with pandas 2.0.3 (pyarrow 14, the floors), 2.3.3 and 3.0.6.
  `filesystem=` is a named argument of `read_parquet` from pandas 2.1; pandas 2.0 takes
  it in `**kwargs` and handles it the same way (test_serialize.py passes with pandas
  2.0.3 and pyarrow 14).
- Test: pyarrow receives a path and a pyarrow file system, never a Python file object
  (container directory and zip); it fails before the fix.

### Engineering requirements of S20 closed (REQ-006, REQ-007, REQ-009, REQ-010, REQ-292, REQ-294, REQ-298, REQ-306, REQ-307, REQ-333, REQ-395)

With the default arguments the results are unchanged: the reference fixture, the frozen
methods texts, the API index and `soup.json` checks pass as before.

- **REQ-306** (network access only in `io/_fetch.py`): the literal grep of §11.5 reports
  `io/_fetch.py` only. Docstrings say "HTTP reads with a `Range` header" instead of
  "range requests", and `io/mef.py` takes `quote`/`unquote` from `io/_fetch.py`. The test
  scans every text file of the package outside its tests. Expected failure removed.
- **REQ-307 iv** (no `errstate`): no line of the package uses `np.errstate`. The
  documented NaN and inf results are computed by the new `evoparam.utils._numeric`
  (`divide`, `sqrt`, `log`, `log10`), which return the bits of the plain NumPy operation
  without raising or hiding a floating-point warning (17 modules). Expected failure
  removed.
- **REQ-292** (optional imports through `evoparam.utils.require`): every import of an
  optional package (68 sites in 25 modules) is preceded by `require(..., extra=...,
  caller=...)`, so a missing package always raises the same `ImportError` naming the extra
  and the function. `data` and `presets` may now import `utils` (`utils.require` only;
  import table of DECISIONS §2). Expected failure removed.
- **REQ-395** (no copyleft package in the resolved tree of an extra): `pybids` is no
  longer declared in `bids` and `all` (every release from 0.16 requires `num2words`, LGPL,
  and from 0.19 `frozendict`, LGPL-3.0; evoparam never imported it); it is listed in
  `evoparam._extras.WITHHELD` with the reason (DECISIONS §5.7). New `tools/licence_scan.py`
  scans the resolved tree of chosen extras; the CI job `licence-scan` runs it for each of
  the 22 extras in its own environment. The import-graph test checks every imported
  third-party package against a licence allow-list, and the PR template has the checklist
  item. The four lock files lose `pybids` and the packages only it required;
  `soup.json` and `THIRD_PARTY.md` regenerated. Expected failure removed.
- **REQ-294** (provenance chain): the chain `read` → `detect_events` → `remove_artifact`
  → `fit` → `test_response` → `fit_trials` on the bundled simulated recording holds the
  six steps in order (with the `epoch` and `average` steps the user calls between them).
  Reading a saved `.evoparam` container appends a read step; `detect_events` records its
  step in the event table, which `epoch` continues; `fit_trials` continues the record of
  the group fit, and every `TrialFit` carries it. Expected failure removed.
- **REQ-307 ii** (no silent failure of a unit): `fit_trials`, `test_response` and
  `remove_artifact` take `on_error` (`"record"`, the default, or `"raise"`), as
  `fit_group` did. A channel that fails is kept with `status="failed"` and the exception
  in `message` (`TrialFit.status`; arrays over the channels in `ResponseTest` and
  `ArtifactResult`), NaN numbers, and exactly one WARNING record naming it;
  `on_error="raise"` raises the first `FitError`; data and settings errors propagate.
  `test_response` and `remove_artifact` find the failing channels by running the channels
  one at a time after the vectorised run failed (DECISIONS §4). The `model_comparison`
  method now marks a channel whose model fit fails as failed (before: NaN, with a WARNING
  that did not name the channel). The test injects the failure in each function, as for
  `fit_group`. Expected failure removed.
- **REQ-009** (amplitudes with an explicit unit): a fit records the unit of its data
  (`ResponseFit.units`, `GroupFit.units`); the fit tables give it for the amplitudes,
  areas (`uV·s`) and errors instead of `"data"`. A `GroupFit` of fits in different units
  raises `DataError`. Expected failure removed.
- **REQ-010** (light core): `import evoparam` and the core chain (`detect_events`,
  `remove_artifact`, `fit` with its existence test, `test_response`, `fit_trials`, save
  and load) run with NumPy and SciPy only. Without pandas the tables are the new
  `evoparam.data.Table` (named NumPy columns, `to_pandas()`); with pandas they are
  DataFrames as before (DECISIONS §4). Containers saved without pandas use CSV tables.
  `read_neo` no longer needs pandas. Expected failure removed.
- **REQ-298** (coverage thresholds): `evoparam.shapes` is covered at 100 % (lines and
  branches) by its own tests, without any exclusion comment: the new
  `tests/shapes/test_checks.py` checks every input check, fallback and edge case of the
  shapes, backgrounds and registry (plugin entry points, every check of
  `register_shape`). Two faults found on the way are fixed: `skew_mode` of a NaN or
  infinite skewness recursed without end (now NaN), and of a skewness above about 1e154
  raised `OverflowError` on the scalar path (now the half-normal limit, as on the array
  path). The coverage configuration measured too little: `[tool.coverage.run] omit` left
  out `plots`, `cli`, `stream` and `benchmark`, so the 80 % gate of the outer layers had no
  data, and the 90 % gate counted only part of the package. Now every module is measured;
  `coverage report` gates the core packages of REQ-298 (`[tool.coverage.report]
  include`) and the outer-layer gate also counts the remote I/O modules. New tests of
  modules that had none: the signal figures of `evoparam.plots`, `ComponentWindows`, the
  inherited module utilities, deep equality, the environment record, the worker side of
  `n_jobs` and `fetch_erpcore` against a local copy of the OSF API. Fast suite, Python
  3.13: core 90.5 % (lines and branches), shapes 100 %, outer layers 82.0 %. Expected
  failure removed.
- **REQ-006** (an interval for every reported value): the existence test of an Evoked
  (no trials) gives intervals of the onset, offset and duration from a baseline
  bootstrap (`n_boot` resamples; DECISIONS §4); its p-values and decisions are unchanged.
  Expected failure removed.
- **REQ-007** (time-frequency fits regenerated): `tf_fit` draws and records a seed when
  `random_state` is None, and its provenance step holds every keyword argument that
  changes the result (frequencies, channels, transform parameters), so
  `evoparam.reproduce` regenerates it identically. The methods text of a time-frequency
  fit now gives its frequency span. Expected failure removed.
- **REQ-333** (`reproduce` of the chain): `evoparam.reproduce` replays `detect_events`,
  `epoch` and `fit_trials` too, so a trial fit replays identically, and an edited
  `settings.json` of a trial fit changes the replay. Expected failure for trial fits
  removed (the time-frequency case: REQ-007 above).

### Arguments that were ignored: `verbose=`, `n_jobs`, five unused arguments (REQ-304, REQ-307)

With the default arguments the results are unchanged: the reference fixture, the frozen
methods texts, the API index and the results of `test_response` (permutation,
projection, shuffle, cluster), `fit`, `fit_trials` and `shared_shapes` (nmf, kpca) are
identical to main.

- **`verbose=`** of `fit`, `test_response`, `fit_hierarchical`, `tf_fit` and
  `detect_pulses` now sets the level of the `evoparam` logger for the call
  (`evoparam.utils.verbose`, REQ-304: None/False inherit, True/"info", "debug",
  "warning", "error"); each function writes an INFO record summarising its run.
  `verbose="progress"` shows a `tqdm` bar over channels, subjects or units (extra
  `progress`; without tqdm nothing is shown); the bar never changes a result.
  `fit_hierarchical` reads `verbose` itself, with the same values (spec §4.2 lets
  `hierarchical` import `utils.require` only), and shows no bar.
  Warnings raised inside a `@verbose` function point at the caller's line, not at the
  decorator (`skip_file_prefixes` from Python 3.12, the same walk up the stack by hand
  before).
- **`n_jobs`** of `test_response` (permutation, projection, shuffle and
  `model_comparison`), `fit_trials` and `shared_shapes` (nmf) now runs the channels,
  blocks or restarts in parallel: `joblib` processes when the extra `parallel` is
  installed, else a thread pool (new `evoparam.utils._parallel`). Every random number is
  drawn in the calling process in the order of the serial loop, so the results are
  identical for every `n_jobs` (tested for each function and both back ends). The warnings
  and log records of a worker process are issued again in the caller, in order, and a
  warning that points past the unit (`max_shift ... clipped` of `fit_trials`) points at
  the same line as in the serial loop (`evoparam.utils._rungs`); in the thread pool it
  points into the pool.
  `shared_shapes(method="kpca")` and `tf_fit` stay serial; `fit_hierarchical` keeps its
  process pool.
- **Unused arguments implemented** as their docstrings say:
  `prestimulus_dependence(band=)` is checked (two increasing positive frequencies) and
  must match the band of the pre-stimulus phases of the fit (SettingsError otherwise; the
  fit records it in its settings when `"prestim"` is a covariate);
  `GroupFit.to_annotations(what=)` accepts `"components"` and raises SettingsError
  otherwise; `artifact_quality(template=)` gives the residual R² of the template when the
  result has none; `reports.markdown(data=)` adds the trial QC of the data; the `log`
  argument of the CLI fit receives the records of the `evoparam` logger (`--verbose
  info|debug`).
- **REQ-307**: the 6 `except Exception` fallbacks left (permutation tail fit, REML of
  `fit_hierarchical`, log-likelihood of the Bayesian fit, three initial guesses of the
  engine) catch the specific exceptions they expect; the REML and log-likelihood
  fallbacks warn, the others write a DEBUG record. `ruff check --select
  E722,S110,BLE001 evoparam` reports nothing: expected failure removed. ruff findings
  1246 -> 1227 (package: 374; ARG001/ARG002 13 -> 0, BLE001 6 -> 0).
- Docstrings and annotations: `read_neo(events=)` documents False; the `folder` of
  `fetch_openneuro_file` accepts None (`$EVOPARAM_DATA`, else `~/.cache/evoparam`); a
  plugin background without `basis`/`basis_jacobian` is refused when the plugin loads,
  with a SettingsError naming the missing members (subclass `BackgroundBase`). The
  `suite=` of the benchmark functions already documented `str or Suite` on main.
- **Type stubs**: `scipy-stubs`, `types-PyYAML` and `types-tabulate` added to the `tests`
  extra (and `all`); the four lock files regenerated with `uv pip compile` (additions
  only, no other pin changed). `mypy --strict evoparam` with the stubs: 1 error, `patsy`
  `[import-untyped]` (47 without them); the expected failure stays with these values.
  `soup.json` and `THIRD_PARTY.md` regenerated (joblib and tqdm are now imported by the
  package).

### Static quality: ruff and typing (spec §9.6 REQ-301, §9.9 REQ-305, REQ-307)

No change of behaviour apart from the fix below: formatting, equivalent rewrites,
annotations, `cast()` calls (no effect at run time) and justified `noqa` / `type: ignore`
comments. The reference fixture, the frozen methods texts and the API index are unchanged.

- **ruff check** `evoparam benchmarks docs/tutorials examples`: 4780 findings on main
  (ruff 0.16.8 and 0.16.10; 4644 were recorded in the expected failure) -> 1246. Of these,
  393 are in the package outside its tests: PLR2004 204, complexity and argument counts
  170 on functions whose signature the spec does not fix (C901 41, PLR0912 39, PLR0913 44,
  PLR0915 18, PLR0917 18, PLR0911 10), ARG001/ARG002 13, BLE001 6; 852 in the tests
  (PLR2004 831); 1 in benchmarks. The expected failure stays, with these values.
- **ruff format --check**: 380 of 422 files to reformat -> 0.
- **REQ-307** (E722, S110, BLE001 in the package): 14 -> 6. The 8 `except Exception` that
  record or re-raise the error carry a justified `noqa`; the 6 that fall back without a
  trace stay reported (existence/permutation tail fit, hierarchical/api REML,
  hierarchical/bayes log-likelihood, three initial guesses in models/engine): restricting
  them changes behaviour, a decision of its own.
- **mypy --strict evoparam** (mypy 2.4.0, the Python 3.10, 3.12 and 3.13 environments of
  the lock files): 5345 errors -> 47, all `[import-untyped]`: scipy (45 imports), yaml and
  tabulate have no stubs in the lock files, and the spec allows `ignore_missing_imports`
  for optional packages only. With `scipy-stubs` 1.15.3.0, `types-PyYAML` and
  `types-tabulate` installed, 22 errors remain, from the stubs (lapack wrappers,
  `least_squares`, `ndimage.label`, `minimize_scalar`) and `patsy`. Adding the stubs to
  the `tests` extra and the locks is left to a decision; the expected failure stays.
- Arrays are typed `NDArray[np.float64]`, `NDArray[np.intp]` and `NDArray[np.bool_]`
  (aliases in `evoparam._typing`, from `numpy.typing`; `soup.json` and `THIRD_PARTY.md`
  regenerated). The typing examples pass under `mypy --strict` (expected failure removed).
- **pyright --verifytypes evoparam --ignoreexternal** (1.1.414): 34.8 % of the exported
  symbols with a known type -> 97.3 % (3885 known, 76 ambiguous, 30 unknown, all in
  `evoparam.tests`, whose functions are now annotated); criterion met, expected failure
  removed.
- Comments that silence a rule: `noqa` 19 -> 216 comments (301 codes), each with its
  reason: sphinx-gallery layout of the tutorials (E402, D205, D400), signatures fixed by
  the spec (C901, PLR09xx; N803 on public keywords: unit suffixes, the design `X`), the
  shape protocol (ARG002 on `data_range`), public functions named `test_*` that are not
  pytest tests (PT028),
  scikit-learn signatures, closures called within their iteration (B023), the NumPy < 2.0
  fallback (NPY201) and the 8 BLE001 above. `type: ignore` 1 -> 17, each with its error
  code and reason: untyped calls of jax (1), PyTensor (4), ArviZ (2) and `open_memmap` (1),
  `np.trapz` absent from NumPy >= 2.0 (3), `tomllib` before Python 3.11 (1),
  `__hash__ = None` (1), the `__deprecated__` attribute read by the tests (1), and
  `basis_jacobian` of the two exponential backgrounds, which needs the non-linear
  parameters (2).
- Annotations that did not match what the code accepts were corrected: `StreamFitter`
  `detector` (a callable too), `dependability` and `latency_standard_error` `n_trials` (an
  array), `burst_waveform` `phase` (an array), `reliability` `facets` (a sequence),
  `make_noise` (overloads on `return_info`), `simulate_*` (return
  `(TrialSet, GroundTruth)`).
- **Fix:** `hierarchical` with `n_jobs=None` or negative raised a TypeError when
  `os.cpu_count()` returned None; one process is used then.
- Found, not changed (behaviour): `verbose=` is not used by `fit`, `test_response`,
  `fit_hierarchical`, `tf_fit` and `detect_pulses`; `n_jobs` is not used by
  `test_response`, `fit_trials` and `shared_shapes`; `prestimulus_dependence(band=)`,
  `GroupFit.to_annotations(what=)`, `artifact_quality(template=)`,
  `reports.markdown(data=)` and the `log` argument of the CLI fit are not used;
  `benchmark` `suite=` also takes a Suite and `read_neo` `events=` also takes False while
  the docstrings say str; `fetch_openneuro_file` `folder` cannot be None; a plugin
  background that only satisfies the `Background` protocol has no `basis` /
  `basis_jacobian`, which `fit` needs. Since done: see the section above.

### `compute_features`: half-height crossings at the ends of the feature grid; REQ-231 (Kiesel et al. 2008) read

- **Fix: `compute_features` no longer raises an IndexError.** The FWHM search started from
  the grid sample `k` next to the extremum and assumed it was at least half the height.
  When it was not, the left crossing was interpolated between `k` and `k + 1`, and the
  right one between `k - 1` and `k`. An extremum after the feature grid (which covers the
  window and one window length on each side) clips `k` to the last sample, so `k + 1` is
  one past the grid: the IndexError, seen on gamma components, whose default bounds allow
  a peak at `t0 + (kappa - 1) theta` well after the window (here 180 of 1 200 gamma draws
  over the default bounds in three windows; for example `t0 = 0`, `kappa = 15`,
  `theta = 0.1` peaks at 1.4 s, the grid of the window 0.009-0.5 s ends at 0.99 s). Any
  shape placed there raised it (a Gaussian at 2 s). Two silent cases came from the same
  cause:
  - an extremum before the grid used sample -1, which wraps to the end of the grid (a
    Gaussian at -1 s had a FWHM of -1.5e23 s);
  - a lobe narrower than the grid step (a tenth of a sample) has no sample above half its
    height, and gave a FWHM 5.7 times the truth, or 0.41 s for a 12 microsecond one.
  Now an extremum off the grid gives a NaN FWHM with `fwhm_outside_grid`, as a crossing
  outside the grid already did, and a lobe narrower than the grid step is interpolated
  between the extremum and its neighbouring samples (within one grid step of the truth).
  `_interp_cross` refuses an index off the grid instead of wrapping. Onset and offset were
  already guarded at both ends; they are unchanged.
- **Outputs unchanged elsewhere.** `compute_features` was compared before and after, field
  by field (NaN equal to NaN): on 9 600 parameter sets drawn over the default bounds of the
  eight shapes (corners included) in three windows (2 048 Hz, 1 000 Hz, 500 Hz), and on
  the truth tables of the six generators (seeds 0-2), every output is identical except the
  180 gamma sets that raised before. Twenty-four fits with `("gamma",)` and
  `("gamma", "biexponential")` on `simulate_erp` waveforms give identical components. No
  algorithm version changes: `compute_features` has no provenance step of its own; the
  fits that call it (`ResponseModel.fit_waveform` and `fit_group`, through the engine's
  identifiability checks and the result tables) are already at 1.1 in this unreleased
  cycle, and for the generators and `spatial.fit_shared` (at 1.0) the change needs an
  extremum more than one window length outside the window or a lobe narrower than a
  tenth of a sample, which none of the sets above produced.
- Tests: `tests/shapes/test_shapes.py` (extremum after and before the grid, lobe narrower
  than the grid step, every shape over its whole parameter box in two windows);
  `tests/models/test_fast_paths.py::test_features_scale_with_the_amplitude` now includes
  the gamma, which was left out only because of this error. The new tests fail on the
  previous code (IndexError, FWHM -1.5e23 s, 0.41 s) and pass now.
- **REQ-231, Kiesel et al. (2008): still an xfail, with a new reason.** The paper was read
  on 2026-10-12 from the authors' copy at the University of Freiburg
  (`.../Kiesel_Miller_Jolicoeur_Brisson_2008_PP.pdf/at_download/file`; the plain `.pdf`
  URL is refused by robots.txt). Pages 250-264 give the corrections only as Equation 1
  (p. 252) and report no t or F value: Tables 1-8 are Monte Carlo means, SDs, power and
  Type I error rates over data sets A (22 participants) and B (16) whose data are not
  published. So there is no corrected statistic of that paper to reproduce. Pages 265-274
  could not be read: the text returned stops on p. 264, Wiley, ResearchGate and Semantic
  Scholar refused the fetch, and doi.org, OpenAlex, Crossref, Unpaywall, Europe PMC,
  PubMed, CORE, Google Scholar and the Wayback Machine answered HTTP 403 at the proxy. The
  paper's one worked number, the interpolated criterion crossing of p. 254 (-.98 at
  62.50 ms, -1.02 at 66.41 ms, criterion -1.0, 64.46 ms), is a new passing test
  (`test_req231_kiesel_interpolation_example`, 64.455 ms). The corrections stay tested on
  the Smulders (2010) example. Recorded in `docs/spec/DECISIONS.md` §4.

### Engine simulation requirements (REQ-105, REQ-106, REQ-111, REQ-116): causes measured, no engine change

The four engine requirements on simulations that stay expected failures in
`tests/models/test_engine_sim.py` were examined. Causes and candidate changes were measured
on seeds other than the tests' (REQ-105 60 500 + ..., REQ-106 60 600 + ..., REQ-111
90 000 + ..., white-noise windows 80 000 + ...), then reported on the tests' own seeds, whose
values do not change. No engine change is kept: the only change that improved a REQ
measure did not reach the REQ and lowered the ds004080 N1 results (REQ-106 below). The
algorithm versions, the reference fixture, the frozen methods texts and the speed are
unchanged.

- **REQ-105: recall 0.885 (K = 1 0.82, K = 2 0.855, K = 3 0.927), precision 1.00.** Every
  missed component is at SNR 3: found 6 of 45 at a FWHM of 3 samples and 27 of 30 at 6
  samples, all the others at every SNR and width (the test's first 60 waveforms per K;
  0.16 and 0.77 on other seeds). K = 1 is lowest because the test steps the SNR and the
  width with the same index over two lists of four levels: every K = 1 waveform at SNR 3
  has a width of 3 samples, at K = 2 and 3 only its first component. The misses are the
  acceptance criterion's, not the candidate search's: each missed component, refitted from
  its true latency and width onto the final model, changes BIC by -8 to +9 (47 misses on
  other seeds), never by the -10 of `delta`. A Gaussian of FWHM 3 samples at SNR 3 carries
  a chi-square of about 20 (z about 4.5); BIC plus `delta` asks about 29 (3 ln n = 19 for
  its three parameters, n about 580 whitened samples, plus 10). The bar trades directly
  against REQ-112 (no component in >= 95 % of pure-noise windows):

  | `delta` | recall (other seeds, 360 components) | precision | white-noise windows with a component (200) |
  |---|---|---|---|
  | 10 (default) | 0.875 | 1.00 | 0 % |
  | 0 | 0.947 | 0.99 | 3.0 % |
  | -2 | 0.956 | 0.99 | 5.5 % |

  No `delta` meets REQ-105 and REQ-112 together on white noise alone (coloured noise and
  the ds004080 baselines, 97.5 % silent at `delta` 10, leave less room): the precision of
  1.00 is the margin REQ-112 needs on noise. `delta` stays 10 (§5.5.4). Not met
  (DECISIONS §5.3).
- **REQ-106 with the default pair: latency MAE 0.52 (skewed Gaussian), 0.76 (gamma), 0.60
  (biexponential) sample.** Without noise (least squares at 1 kHz, mean |mode error| over
  the widths 3, 6, 12 and 25 samples):

  | truth | Gaussian fitted | skewed Gaussian fitted |
  |---|---|---|
  | skewed Gaussian (alpha 2) | 0.71 | 0.00 |
  | ex-Gaussian (tau = sigma) | 0.46 | 0.10 |
  | gamma (kappa 3) | 1.69 | 0.47 (1.01 at 25 samples) |
  | biexponential (decay / rise 4) | 2.97 | 0.54 (1.20 at 25 samples) |

  Gamma and biexponential truths are outside the pair, and the latency of §5.5.1 is the
  mode of the fitted component: the best skewed Gaussian is 0.47 and 0.54 sample off
  before any noise, which only adds to it. For skewed truths the pair holds the right
  shape, but the criterion keeps the Gaussian in 72 of 100 fits (other seeds): the
  asymmetry does not pay its ln n at SNR 5 to 20, and the Gaussian's centre lies 0.19 to
  1.55 samples after the mode. Measured and not kept: refining every allowed shape of the
  candidate being accepted (its quick fit again at the refinement tolerance, and the
  perturbed starts), instead of only the shape chosen on the single-start, loose-tolerance
  quick fits (§5.5.3 step 4: "the best kept"). On other seeds the MAE falls (skewed
  0.494 to 0.457, ex-Gaussian 0.413 to 0.384, gamma 0.798 to 0.779, biexponential 0.647
  to 0.613, Gaussian unchanged); on the test's seeds it stays above 0.5 (0.505, 0.750,
  0.599); and the N1 decision on ds004080 drops: development runs kappa 0.40 / 0.39 to
  0.37 / 0.34 (registry row / code; code recall 0.81 to 0.69), REQ runs 0.458 / 0.398 to
  0.426 / 0.369 (recall 0.460 / 0.882 to 0.416 / 0.765), 10 fewer N1s, with 2 to 24 % more
  residual evaluations (ds004080 fits 1.26 times longer). A cheaper form (both shapes
  refitted at the refinement tolerance, perturbed starts for the better one only): skewed
  0.479 on other seeds, REQ runs 0.441 / 0.371 (code recall 0.794). Not met
  (DECISIONS §5.4).
- **REQ-111 at 0.5 FWHM: one component in 100 of 100 runs, never `merged`.** Two equal
  Gaussians of FWHM 40 ms 20 ms apart differ from the best single Gaussian by a chi-square
  of 1.4 at SNR 10 (25.7 at 0.75 FWHM, 148.5 at 1 FWHM): the data do not hold the pair, and
  no step accepts it. Step 5 merges after an accepted step, as the engine does; its overlap
  rule does not even call the true pair overlapping (20 ms is more than 0.5 (sigma_1 +
  sigma_2) = 17 ms). Extending the merge-and-compare to the candidates a step rejects was
  measured (100 runs per design, `shapes=("gaussian",)`, other seeds; share of
  one-component results with `merged`):

  | design | the best rejected candidate | any rejected candidate | any, overlap on FWHM instead of sigma |
  |---|---|---|---|
  | two Gaussians, 0.5 FWHM apart, -10 each | 7 % | 49 % | 99 % |
  | one Gaussian, FWHM 40 ms, -10 | 3 % | 51 % | 95 % |
  | one Gaussian, FWHM 40 ms, -8 | 3 % | 46 % | |
  | one Gaussian as wide as the pair's sum (47.5 ms, -17) | 4 % | 63 % | |
  | one Gaussian, FWHM 10 ms, +6 | 1 % | 33 % | 91 % |
  | one Gaussian, FWHM 120 ms, +6 | 7 % | 77 % | 100 % |
  | two Gaussians, 1 FWHM apart (two components in 100 %) | 0 % | 0 % | |

  Every reading sets the warning as often, or more often, on single components as on the
  pair: it would say nothing about a pair the data cannot show, so none is kept. Not met
  (DECISIONS §5.5); the REQ's other clause holds (no two-component fit, none unflagged).
- **REQ-116, profile intervals under white noise: amplitude 0.975 (195 of 200), latency
  0.935.** The same design on 1 000 runs (the test's 200 and the next 800): amplitude
  0.945, latency 0.942 (binomial SE 0.007); runs 201 to 1 000 alone 0.9375 and 0.9437.
  The amplitude interval of `uncertainty="profile"` is the Wald interval of the linear
  amplitude (§5.5.5 profiles the non-linear parameters), the Jacobian method's, 0.96 on
  the test's 100 runs. 195 of 200 when the coverage is 0.945 has a probability of 3.4 %:
  a high draw, not an error of the profile threshold, the degrees of freedom or the noise
  scale. Nothing changed; it stays an expected failure on the test's seeds.

Tests: the reasons of the four expected failures now give these causes and values. Run:
`tests/models` (with `test_engine_sim.py`, `test_reseed.py`, `test_cancelling_pairs.py`,
`test_trf.py`, `test_fast_paths.py`, `test_band_limited_noise.py`,
`test_evaluate_fit.py`), `test_determinism.py`, `tools/gen_soup.py --check`,
`tools/freeze_methods_text.py --check`.

### Preset tep 1.1 (REQ-275): constant background and refusal of cancelling pairs

REQ-275 asks for labelled N45 and N100 on the channel of maximal response for 80 % of the
20 subjects of ds002094. At 3b222bb (preset `tep` 1.0, after the merges of the strided
whitening) the counts were lower than the ones recorded above: N45 / N100 / both, out of 20,
preset chain on the 17 inner channels 9 / 12 / 8, with ICA on all 30 channels (the
criterion) 7 / 11 / 5, with ICA on the 17 inner channels 7 / 14 / 5, preset chain on all
channels 5 / 7 / 3. The share of the strided whitening in this is measured below ("Strided
whitening on variant B").

Method. Changes were chosen on 8 development subjects: sub-283 (already used for variant A
in S15) and 7 drawn with `numpy.random.default_rng(275)` from the 19 others (sub-739, 749,
759, 764, 766, 767, 774). The 12 others (sub-711, 718, 728, 734, 735, 743, 761, 768, 779,
780, 783, 786) were fitted once, with the chosen preset, after the choice. The cleaning
variants, the channel rule and the label windows of the test are unchanged.

Causes, on the development subjects:

- **The data has the deflections, the fits do not label them.** On the channel of maximal
  response after ICA, the average has a local minimum in 38-55 ms on 8 of 8 and in 80-140
  ms on 7 of 8 (both on 7 of 8; on the 17 inner channels 7 of 8 too); preset 1.0 labelled
  both on 3 of 8.
- **The double exponential background forms a bump.** With coefficients of opposite signs,
  `c1 e^(-t/tau1) + c2 e^(-t/tau2)` is the biexponential component shape starting at 12 ms.
  The background had an interior extremum in 30 of the 32 development fits (up to 208 uV
  from the chord between its ends): on sub-764 (FZ) a -11 uV trough at about 110 ms that
  takes the N100, on sub-749 (FC2) a 5 uV rise from 12 to 25 ms that takes the P30-N45-P60
  complex, on sub-283 (FC2) a 15 uV bump at 20 ms. Spec §5.5.2 asks that a background cannot
  represent a component (the knot rule of `pspline`); this one can.
- **Cancelling pairs.** 56 components of the 32 fits were not resolvable (in a pair of
  overlapping components of opposite signs, for example +28.7 and -29.6 uV, both at
  81 ms, on sub-283 IZ).
- **Not causes on these subjects:** the label windows (after ICA, the first minimum of the
  average after 30 ms is inside 38-55 ms on 8 of 8 subjects on all channels and 7 of 8 on
  the inner channels, the exception sub-283 FZ at 63 and 79 ms; 38-55 ms is
  the N45 window of Belardinelli et al. 2021, Sci Rep 11:8159, inside the 35-60 and 40-60 ms
  of Premoli et al. 2014, J Neurosci 34:5603, the two experiments), and the cleaning
  (below).

Candidates on the development subjects, N45 / N100 / both out of 8 (A = preset chain,
B = with the TESA-like ICA; inner = 17 channels off the outer ring, all = 30 channels), the
sum of the four "both" counts, and the components not resolvable (sum over the 32 fits;
n/a: not recorded for that run):

| Fit settings (preset 1.0 otherwise) | A inner | A all | B all | B inner | both | not resolvable |
|---|---|---|---|---|---|---|
| preset 1.0 (double exponential) | 4/6/3 | 2/4/1 | 3/7/3 | 3/7/3 | 10 | 56 |
| `reject_cancelling` | 4/6/3 | 1/3/0 | 3/7/3 | 3/8/3 | 9 | n/a |
| `reseed="always"` | 4/6/3 | 2/5/1 | 3/7/3 | 3/7/3 | 10 | 54 |
| `amplitude_prior=1` | 2/4/1 | 2/1/0 | 4/6/3 | 3/7/3 | 7 | 65 |
| `max_components=6` | 4/6/3 | 2/3/0 | 3/7/3 | 3/7/3 | 9 | 64 |
| `background_tau_max=0.05` | 6/6/5 | 4/4/2 | 3/8/3 | 2/7/2 | 12 | n/a |
| monotone double exponential (a prototype background `c0 + c1 (w e^(-t/tau1) + (1-w) e^(-t/tau2))`, not kept) | 5/4/3 | 3/4/1 | 4/6/3 | 4/5/3 | 10 | 73 |
| monotone + `reject_cancelling` | 2/4/1 | 2/4/1 | 4/5/4 | 3/5/3 | 9 | 38 |
| `exponential` | 4/6/4 | 4/4/2 | 4/5/4 | 3/5/3 | 13 | n/a |
| `exponential`, tau max 50 ms | 5/6/4 | 5/4/2 | 3/6/3 | 2/5/2 | 11 | 61 |
| `exponential` + `reject_cancelling` | 3/6/3 | 3/4/2 | 3/5/3 | 2/6/2 | 10 | 31 |
| `linear` | 3/5/2 | 3/1/0 | 6/6/4 | 5/6/4 | 10 | 89 |
| `linear` + `reject_cancelling` | 2/4/2 | 3/1/0 | 6/6/6 | 5/6/5 | 13 | 33 |
| `constant` | 3/5/2 | 4/4/2 | 6/7/5 | 5/6/4 | 13 | 77 |
| **`constant` + `reject_cancelling`** | 4/6/3 | 4/4/2 | 6/7/6 | 6/7/6 | **17** | **19** |
| `noise_model="neff"` | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0 | 0 |

The refusal of cancelling pairs alone changes little: the bump of the background then
plays the part of the second member of the pair. A constant background alone moves the
degeneracy into the components (77 not resolvable). Together they close both ways of
describing one deflection by two opposing terms, the fits describe the TEP sequence (for
example sub-739 CZ: P30 27 ms, N45 41 ms, N100 89 ms, P180 198 ms), no reading gets worse
and the 32 fits take about a third of the time (64 s against 193 s). Chosen:
**preset `tep` 1.1 = background `constant`, `fit_extras={"reject_cancelling": True}`**.
A constant is the background of spec §5.5.2 "when a baseline correction has been
applied" (the `tep` chain subtracts the -800 to -100 ms baseline); the decay artifact is
interpolated up to 12 ms, and with a DC-coupled amplifier most decay artifacts recover
within 5-10 ms (Rogasch et al. 2017, NeuroImage 147:934, [LIT-artifact-9]).

Result (the test's own code path, `EVOPARAM_NETWORK_TESTS=1`), N45 / N100 / both:

| Reading | 1.0: development / held-out / total | 1.1: development / held-out / total |
|---|---|---|
| A, 17 inner channels | 4/6/3 · 5/6/5 · 9/12/8 | 4/6/3 · 5/9/5 · 9/15/8 |
| A, all 30 channels | 2/4/1 · 3/3/2 · 5/7/3 | 4/4/2 · 2/5/2 · 6/9/4 |
| **B, all 30 channels (criterion)** | 3/7/3 · 4/4/2 · 7/11/5 | 6/7/6 · 4/4/2 · **10/11/8** |
| B, 17 inner channels | 3/7/3 · 4/7/2 · 7/14/5 | 6/7/6 · 6/10/6 · 12/17/12 |

On the held-out subjects the criterion reading does not move (2 of 12) while the inner
channels go from 2 to 6 of 12: there, the channel is the limit. **REQ-275 stays not met
(8 of 20, 40 %).** After ICA, the channel of maximal response (largest RMS from 38 to
140 ms over the 30 channels) is TP9, TP10, IZ or O1 on 10 of the 20 subjects (2 of the
development subjects, 8 of the held-out ones); on 9 of these 10 the largest fitted
component from 80 to 140 ms is positive, the fronto-central N100 seen from the other side
of the average reference; both labels on 1 of these 10 subjects, on 7 of the other 10. The
rule of the test is the spec's words and is kept; the 17-channel reading is reported
(12 of 20). Under the coil (C4, a diagnostic, not a rule of the test): A 6 / 14 / 5,
B 5 / 17 / 5.

Simulations (layer 2, the 26 `tep` cells of the weekly design, 50 replicates each, same
seeds, 1.0 → 1.1): mean F1 0.947 → 0.945 over the 25 cells with components, false
components per waveform 0.046 → 0.058. Better with jitter 0.5 FWHM (F1 0.92 → 1.00,
false 0.16 → 0), three components (0.98 → 1.00) and separation 1 FWHM (0.98 → 1.00);
worse with pink noise (1/f: F1 0.91 → 0.88, false 0.14 → 0.34; 1/f^2: 0.99 → 0.91,
0.02 → 0.40: a constant background cannot follow the slow drift that the double
exponential absorbed), separation 0.3 and 0.5 FWHM (0.94 → 0.90, 1.00 → 0.97: the
components alternate in sign and close opposite pairs are refused), 5 trials (0.89 → 0.87)
and SNR 1 (0.19 → 0.17); AR(1) 0.9 noise 0.89 → 0.91; the 14 other cells within 0.01.
The trade-off is kept: the real-data gain is on the REQ dataset, the simulated mean is
unchanged within 0.003. The recorded weekly report
(`validation/layer2/0.1.0.dev0`) is not regenerated here.

Not adopted:

- **SOUND + SSP-SIR cleaning** (Mutanen et al. 2016, NeuroImage 139:157, [LIT-artifact-15];
  Mutanen et al. 2018, NeuroImage 166:135, [LIT-artifact-16]; the third pipeline compared by
  Brancaccio et al. 2024, [LIT-artifact-12]), re-implemented from the papers for this
  comparison (not kept in the repository): average reference, SSP-SIR on every trial with
  the two topographies of the 100 Hz high-passed trial average in 5-50 ms (two singular
  values stood out on 5 of the 8 development subjects, three on one, one on one, none on
  one; two dimensions as Mutanen et al. 2024, Front Hum Neurosci 18:1324958), suppression in
  5-50 ms with a 10 ms taper, three-layer spherical lead field (MNE sphere model, standard
  10-05 positions, 10 mm volume grid), 1-100 Hz band-pass and 60 Hz notch, one ICA round for
  blinks and eye movements, SOUND (lambda 0.1) on the trial average applied to every trial.
  On the development subjects the decay and muscle activity next to the coil (C4, CP6, P4,
  P8) stays the largest response after 50 ms: both labels 1 of 8 (all channels) and 1 of 8
  (inner) with preset 1.0, 0 and 0 with 1.1; the average has a minimum in both windows on 5
  and 3 of 8 (7 and 7 with the TESA-like ICA). The test keeps the TESA-like ICA (variant B).
- **Wider label windows.** The windows are the published ones (above) and the development
  data did not need them.
- **Another channel rule** (GMFP, the channel under the coil, the inner channels for
  variant B): the spec names the channel of maximal response; the readings are reported,
  not substituted.

Strided whitening on variant B (measured; the engine is not changed). The merge of the
strided whitening (e552ba7, entry "Engine: noise that does not fill its band" below) acts on
variant B only:

- **Strides.** Variant A never strides (0 of 20 fits on each reading, preset 1.0 and 1.1:
  the level ratio of the baseline, above 1/k of the Nyquist frequency over the octave
  below, is never under -1.3 decades). Variant B strides on 20 of 20 fits on both readings
  with both presets: all 30 channels k = 3 on 19 fits (phase 2 on 10, 1 on 7, 0 on 2) and
  k = 2 on 1; inner channels k = 3 on 18 (phase 1 on 8, 2 on 6, 0 on 4) and k = 2 on 2.
  So the fit is judged at 333 Hz with `min_fwhm` raised from 4 to 6 ms (19 and 18 fits), or
  at 500 Hz with no raise; never k = 4. Over the 600 channel baselines of variant B the
  ratio is -3.20 decades at k = 4 (median; never under -4), -4.35 at k = 3 (541 under -4)
  and -5.80 at k = 2 (600 under -4): the 1-100 Hz order-4 non-causal band-pass of the
  ICA cleaning empties the band above 167 Hz.
- **Attribution.** Commit 7044dc1 (main before the strided whitening) with preset 1.0 gives
  the same 80 fits as 3b222bb with the stride disabled (`EMPTY_BAND = 0`, set in the
  evaluation script only): same channels, labels, latencies and amplitudes. The shortfall of
  variant A against S15 (inner channels 11 / 13 / 9 then, 9 / 12 / 8 now) is already there
  at 7044dc1 and is not due to the strided whitening. On variant B, N45 / N100 / both,
  development / held-out / total:

| Reading | 1.0 without stride (= 7044dc1) | 1.0 with stride (3b222bb) | 1.1 without stride | 1.1 with stride (kept) |
|---|---|---|---|---|
| B, all 30 channels | 4/7/4 · 4/8/4 · 8/15/8 | 3/7/3 · 4/4/2 · 7/11/5 | 5/7/5 · 3/6/1 · 8/13/6 | 6/7/6 · 4/4/2 · 10/11/8 |
| B, 17 inner channels | 5/8/5 · 2/10/1 · 7/18/6 | 3/7/3 · 4/7/2 · 7/14/5 | 4/8/4 · 5/9/5 · 9/17/9 | 6/7/6 · 6/10/6 · 12/17/12 |

  Variant A is the same with and without the stride (1.0: 9/12/8 and 5/7/3; 1.1: 9/15/8
  and 6/9/4). The stride lowers preset 1.0 on variant B (both labels 8 → 5 and 6 → 5) and
  raises preset 1.1 (6 → 8 and 9 → 12).
- **Why the stride stays.** At 1 000 Hz the AR whitening of the variant-B baselines has a
  largest power gain of 232 655 (median of 600; 2 655 to 266 418), against 583 at the stride
  (7 on variant A; 1 956 on the upsampled ds004080 run the stride was made for). Without the
  stride the fits fill up: preset 1.0 reaches `max_components` = 8 on 19 of 20 fits (all
  channels) and 18 of 20 (inner), 157 and 156 components of which 50 and 57 are not
  resolvable, against 1 and 1 fits at 8 with the stride (83 and 94 components, 29 and 30 not
  resolvable); preset 1.1 on 14 and 13 fits (148 and 145 components) against 1 and 3 (96 and
  103). The extra labels of 1.0 without the stride come from these saturated fits. The
  `min_fwhm` raise is not the cause: preset 1.0 with the stride and no raise gives 7 / 11 /
  5 (all channels) and 6 / 13 / 4 (inner), and no fit has a component of 6 ms or narrower
  with or without the raise.
- **No engine rule to tell this filter from upsampling.** Variant B is low-passed at 0.2 of
  its Nyquist frequency by an order-4 non-causal Butterworth: the case of the ERP CORE data
  (30 Hz at 256 Hz, 0.23) that the strided-whitening entry measured as a gain. The shape of
  the gap does not separate them either. The slope below the edge (median level of the
  upper half of the octave under 1/k of the Nyquist frequency over its lower half, at the
  chosen k) is -2.28 decades on the variant-B baselines (median; -3.66 to -1.58), -3.54 to
  -1.04 on the 230 strided baselines of the low-passed simulations of that entry, and -1.20
  to -0.01 on the upsampled run (interpolation leaves the octave below flat). A rule that
  strides only when this slope is above -1.4 would keep the upsampled run strided (100 of
  100 channels) and variant B at the full rate, but would also take the stride from 222 of
  the 230 low-passed simulation baselines, whose fits it improved (ERP at 256 Hz, 30 Hz:
  precision 0.366 → 0.811). The engine is not changed.
- **Choice of 1.1.** The development table above was measured with the stride (the engine
  at 3b222bb), so the choice stands; so do the layer-2 costs, measured with the same engine.

Changed files: `evoparam/presets/tep.json` (version 1.1, background `constant`, `fit_extras`
`reject_cancelling`, notes and validation text), `docs/presets.md` (regenerated,
`tools/gen_presets_doc.py`), `evoparam/tests/reports/methods/sim_tep.txt` (regenerated,
`tools/freeze_methods_text.py sim_tep`: preset version, validation text, background, refusal
of cancelling pairs), the deviation listed in `evoparam/tests/presets/test_presets.py`, and
`evoparam/tests/validation/test_tep_ds002094.py` (docstring, measured values; the regression
floor raised to the new levels where they are higher: inner channels with the preset chain
N100 15, with ICA all channels 10 / 11 / 8, with ICA inner channels 12 / 17 / 12; it stays a
strict xfail because the preset chain on the inner channels has N45 9 and both 8, floor 11
and 9). No engine code changed: no algorithm version changes (the preset carries its own
version, 1.1); the reference fixture holds only `ccep` fits and is unchanged.

### Empirical-Bayes channel latencies (REQ-171) and trial counts for latencies (REQ-234), measured on simulations

REQ-171 (EB channel latencies beat the channel fits on >= 90 % of datasets, per-trial SNR
<= 2). The test's design (4 subjects x 20 channels x 20 trials, 60 % responding, SNR 1.5)
was run on 200 development datasets (seeds 1000-1199, chosen before looking at the
results) and on the test's 20 seeds, which were not used for any choice.

- Before (main 3b222bb): the EB wins on 3 of the test's 6 datasets, 11 of 20 with
  `EVOPARAM_SLOW_TESTS=1`, 129 of 200 development datasets (64.5 %). RMS over the
  datasets of the channel latency RMSE: development, channel fits 1.80 ms, EB 1.56 ms;
  test seeds 1.66 / 1.50 ms (20), 1.85 / 1.47 ms (6).
- Attribution of the 5 of 6 / 17 of 20 of the earlier reason (the test as written, run
  in a worktree of each commit): 971633c (where the reason was written) and 9492870 give
  5 of 6 and 17 of 20, channel 1.854 / 1.656 ms, EB 1.547 / 1.493 ms; 1b91be1 (REQ-172:
  Wald error in quadrature with the trial spread) and every later commit measured
  (9b443b6, 75c443a, e552ba7, 3b222bb) give 3 of 6 and 11 of 20, channel 1.854 / 1.656 ms,
  EB 1.469 / 1.497 ms. The channel fits did not change (latencies within 1.1 µs from
  971633c to 3b222bb; `noise_stride` is 1 on all 1 600 channel fits of the 20 seeds): the
  drop comes from the standard error of 1b91be1 alone. The trial spread alone, used
  before it, is too small (z-score SD 1.51) and under-shrinks: 180 of 200 development
  datasets (90 %) but an EB RMSE of 1.62 ms, and the NUTS coverage of REQ-172 was 17 of
  20 with it.
- Cause: the standard error given to the channel latency. It was the Wald error of the fit
  in quadrature with the spread of the trusted trial latencies over sqrt(n). The trial
  spread already holds the noise of the average, so the noise counted twice: median |z|
  0.54 against 0.67 for a normal (7 829 responding channels). And the Wald error of a
  poorly determined latency is far too large: binned by Wald error, the medians 7.1 and
  15.5 ms went with error RMS of 4.2 and 9.7 ms, the best shrinkage weights there were
  0.55 and 0.25 against the 0.23 and 0.06 the model used (log |error| grows as 0.69 x log
  Wald error). Those channels were pulled to the subject mean, the good ones were
  over-shrunk: with the true subject latencies and the true between-channel SD, these
  errors won on only 75 % of the development datasets. Not causes: misassigned channels
  (7 false-positive channels in 200 datasets; fitting the variances on the channels with
  `P(response) >= 0.5` gave 64 %), errors correlated with the true deviation (slope 0.01),
  correlated errors within a subject.
- Change (`evoparam.hierarchical`, `fit_hierarchical` algorithm version 1.1): the standard
  error of the channel latency is the noise of the average (Wald error) plus the mean of
  the trial jitter (the jitter SD beyond noise, pooled by REML, over sqrt(number of
  trials)), as the SD of that normal truncated to the label window (the latency of a
  labelled N1 lies in it); `eb.latency_standard_error`. The trial spread is kept for a
  channel without a Wald error.
- After: 4 of 6 test datasets (seed 100: channel 1.48 ms, EB 1.57 ms; seed 105: 0.98 ms,
  1.06 ms), 14 of 20 with `EVOPARAM_SLOW_TESTS=1`, 160 of 200 development datasets (80 %).
  RMSE: development 1.48 ms (1.56 before), test seeds 1.43 ms (1.50) and 1.41 ms (1.47).
  Between-channel latency SD (REML) over the realised one, median of 100 development
  datasets: 0.943 (0.905 before). Not met: the test stays an expected failure with these
  values.
- Bound: with the true subject latencies and between-channel SD and the new errors, the
  posterior mean wins on 86 % of the development datasets; on datasets drawn from the
  Gaussian model itself (errors exactly normal with those SDs) on 84 %. The posterior
  mean minimises the expected squared error; the chance of beating the channel fits on a
  given dataset grows as the shrinkage falls (new standard errors divided by 2, 4 and 10:
  93.5 %, 96.5 %, 97 % of the development datasets, RMSE 1.48, 1.65, 1.77 ms; divided by
  2: 5 of 6 and 17 of 20 on the test's seeds). Halving the old standard errors gave 92.5 %
  (RMSE 1.45 ms) by under-shrinking the channels with small errors; the minimum of the
  Wald and trial errors 93.5 % with an RMSE of 1.62 ms (worse than before). Neither is an
  empirical-Bayes estimate; not used (DECISIONS 5.6, with the other options measured:
  Wald error alone 78 %, a dispersion factor estimated by REML 62-70 %, trial-bootstrap
  errors of the fit worse calibrated and 50 s per dataset).
- Kept green: `tests/hierarchical/test_hier_eb.py` (REQ-170, REQ-172 EB, REQ-175, the
  proposed wording of REQ-171; also with `EVOPARAM_SLOW_TESTS=1`), the NUTS reference runs
  of `tests/hierarchical/test_hier_bayes.py` with `EVOPARAM_SLOW_TESTS=1` (REQ-172 with
  numpyro, REQ-173, REQ-174, REQ-176: the Bayesian engines read the same standard error),
  `tests/hierarchical/test_hier_ccep.py` with `EVOPARAM_NETWORK_TESTS=1` (REQ-178, age
  effect on the N1 latency of ds004080), `tests/stats`, `tests/trials`, `tests/reports`,
  `test_exports`, `test_docs`, `test_determinism`. SOUP inventory regenerated (`scipy.stats.truncnorm.std`).

REQ-234 (`trials_needed` predicts the SD at 80 trials from 20 within 15 %), fractional
area latency: no change, the cause measured further. Development cells: seeds 7-11, pink,
white and AR(1) noise, per-trial SNR 0.5, 1 and 2, 30 channels of 20 trials, true SD over
300 blocks of 80 trials (the test's seed 41 was not used).

- The power law of `extrapolation="auto"` gives median ratios of 0.83 to 1.18 over the 45
  cells, 0.83 to 0.91 with pink noise at SNR 1 (the test's condition; 0.795 at seed 41).
- Mechanism: with the true waveform as the centre, the bootstrap at 80 trials gives 0.92
  to 1.04; around the noisy 20-trial average it gives 0.74 to 0.95. Where the response is
  weak (the window edges) the rectified area of the noisy average takes random signs, so
  slow (pink) noise cancels in the cumulative area; the true SD falls more slowly than
  1/sqrt(n) (ratio of the SDs at 80 and 20 trials 0.53 to 0.65, against 0.5) while the
  bootstrap ratio is 1 to 18 % lower than the true one.
- Tried, not used: simulation-extrapolation of the centre noise (SIMEX, 0.45 to 0.98: the
  SD is not monotone in the centre noise); the fitted model of the 20-trial average as the
  centre (0.52 to 0.94 on the 4 of 6 channels of one cell where a component was found; the
  fitted background moves the window edges); the true-world SD at 20 trials
  by delete-d jackknife (half-samples, 0.80 to 1.10) extrapolated with the exponent of
  subsamples of 5, 10 and 20 trials (0.81 to 1.08 over the cells, mean |log ratio| 0.068
  and 0.079 against 0.077 and 0.091, but 0.81 to 0.90 at pink SNR 1, and not better for
  the peak latency). The expected failure keeps 0.795; its reason now gives the cause.

### N1 latency agreement (REQ-114, REQ-273): why the rule's peak is missed; fitted background on any time grid

REQ-114 asks the model latency within 2 ms of the published rule's peak on 90 % of the
channels where both find an N1 (ds004080: the example run and the REQ-371 runs, both
readings of the rule, owner decision D8). Each channel was examined: the rule's average,
the preset's input average, the fitted curve, its components and background.

Causes, on the REQ runs (registry row / code):

- **The rule's peak is often not an N1.** It is an N1 trough of the average the rule reads
  (the deepest negative interior local minimum in 9-50 ms) on 39 of the 74 channels where
  both find an N1 and on 24 of 30. The others are the first samples of the window, on the
  recovery tail (18 / 3), slow deflections peaking at 52-100 ms (16 / 2), and one more each
  (the deepest sample just after 50 ms; a positive-valued minimum). The rule has no
  smoothing; its window is 9-100 ms. A model whose N1 latency is a trough of the data is
  therefore bounded by 53 % / 80 %, below the 90 % asked.
- **The component the preset names N1 is not the waveform's trough.** On the 39 / 24
  channels where the rule's peak is an N1 trough, the trough of the fitted curve is within
  2 ms on 31 / 24, the N1 component's own latency on 18 / 11. Where the trough is right and
  the component is not (12 / 13 channels), every fit is `ill_conditioned`, 8 / 9 have
  several negative components in 9-50 ms, the one named N1 (larger `|A|`, §5.5.7) is wider
  than 20 ms on 8 / 9, and it is in a cancelling pair on 7 / 7. The median miss is 4.5 ms.
- **Reference (registry row).** On 6 of the 8 N1 troughs the fitted trough misses, the
  trough of the preset's own input misses too: the rule reads the monopolar average, the
  preset the adjusted common average (on sub-ccepAgeUMCU03 the registry row's peak is at
  32.7 ms on 35 of the 70 channels, a common-mode deflection the common average removes).
  Fitted on the rule's own average (a diagnostic), the input's trough is within 2 ms on 33
  of 33 such channels.
- Not causes: the sampling (on the example run, recorded at 512 Hz, the fitted trough is
  within 2 ms on its 4 N1 troughs under both readings), the
  `min_fwhm` floor (3 ms in the preset, below the 4-8 ms N1s; 2 ms changes nothing), the
  whitening (the fitted trough follows the data's trough).

Model changes measured on the five development runs, none adopted (DECISIONS §5.2, all
values there): `reject_cancelling`, `reseed="always"`, `min_fwhm` 2 ms,
`amplitude_prior` 1, `max_components` 8, and the shapes plus `exgaussian`,
`biexponential` or `gamma`. Each moves one to three of 20-38 channels, in both directions.
The two that also raised the development kappa lowered it on the REQ runs (`gamma`:
0.46/0.40 to 0.44/0.34; `amplitude_prior` 1: to 0.44/0.32).

Fixed (a defect found on the way): **the fitted background on another time grid.** The
exponential background is `c0 + c1 exp(-(t - t_ref)/tau)` with `t_ref` the first sample
of the fit window, but the fit did not record `t_ref` and `evaluate_fit` took the first
time of the grid it was given. On a grid starting elsewhere the tail was scaled by
`exp((t_grid - t_ref)/tau)`: curves drawn from 5 ms ran hundreds of uV below the data; on
the whole epoch (the time-frequency footprint of `tf_fit`, `test_reseed`'s residual) the
tail was missing. Now the fit records `t_ref` in `fit.background`, `evaluate_fit` uses it
and holds the background at its first fitted value before it, and `fitted_peak` searches
from the first fitted sample. `fit_shared` now keeps the time constants and `t_ref` with
the coefficients (`evaluate_fit` raised `KeyError: 'tau'` on its channels).
Algorithm versions: `evaluate_fit`, `fitted_peak` and `tf_fit` 1.1. No fit changes: the
615 fits of the REQ runs are identical bit for bit (components, amplitudes, decisions),
so REQ-113 and `validation/layer3/n1_ds004080.json` are unchanged.
`evoparam/tests/reference/fit_fixture_ccep.json` was regenerated with
`python -m evoparam.tests.reference` to record the new versions; its values move by at
most 4.2e-12 relative, which origin/main already gives (the solver change of REQ-115).

| REQ runs (example + REQ-371), registry row / code | before | after |
|---|---|---|
| REQ-114 as written, N1 component within 2 ms | 24 % (18/74) / 37 % (11/30) | unchanged |
| REQ-114, trough of the fitted curve (K3) | 41 % (30/74) / 80 % (24/30) | 42 % (31/74) / 80 % (24/30) |
| the same where the rule's peak is an N1 trough | 30/39 / 24/24 | 31/39 / 24/24 |
| N1 component, same channels | 18/39 / 11/24 | unchanged |
| REQ-113 kappa (pooled) / recall | 0.458 / 0.398; 0.460 / 0.882 | unchanged |
| trough with the code's floor (regression), pooled kappa / lowest run | 0.890 / 0.71 | 0.907 / 0.71 |

Tests: `test_req114_as_written` and `test_req114_trough_latency` stay non-strict expected
failures; their reasons give the measured values and the bound. New:
`test_n1_trough_where_the_rule_peak_is_an_n1_trough_regression` (not a REQ: with the
code's reading, the fitted trough within 2 ms on >= 90 % of the channels where the rule's
peak is an N1 trough; 24 of 24) and `tests/models/test_evaluate_fit.py` (the background on
any grid, `fitted_peak`, `fit_shared`).

### Third network run on the Mac (S20 merged): benchmark across platforms (REQ-353, REQ-392)

- Run on macOS arm64, the committed `core` results (made on Linux) are not reproduced byte
  for byte: the simulated inputs already differ in their last bits (another libm), so every
  input and output hash differs. REQ-352 asks byte identity "from the same inputs", which
  holds on one machine; across platforms REQ-295 asks 1e-9 relative.
- The six threshold rules and the cross-projection test agree to 1e-9 on every metric. The
  `evoparam-preset` T2 fits do not: amplitude bias 0.26521 against 0.26509 (4.8e-4
  relative, a mean of signed errors), amplitude MAE 4e-5, latency 4e-6. At the commit that
  made the committed results (be8d26c) the Mac gives 7e-5 on the amplitude bias, so this is
  the platform (optimiser paths through different BLAS/LAPACK), not the later engine
  changes.
- The tests now compare hashes when the inputs are the same bytes, and metrics at 1e-9 in
  every case. `test_req392_an_entry_is_reproduced_by_one_command` (vanblooijs2023 T1) passes
  on the Mac. `test_req353_rerun_reproduces_the_committed_results` reports the metrics that
  miss 1e-9 on another platform as an expected failure with the measured numbers; the
  tolerance is not loosened. Open: REQ-295 across platforms for the fitting engine on many
  waveforms (the fixture test of REQ-295 passes).

### Requirements without a test: REQ-024, REQ-134, REQ-373, REQ-374

The traceability report now has 0 untested requirements of 227 (it had 4); the strict
expected failure of REQ-399 is removed. Two of the four criteria are met, two are not, for
the reasons below.

**REQ-134, cross-projection against the published values.** `method="projection"` defined
the response end as the peak of the mean projection divided by `sqrt(t - t0)` (the formula
of §5.6 item 2 as reviewed). The projections are semi-normalised and do not accumulate, so
this division put the end in the first milliseconds of the window. The default is now the
paper's definition, the peak of the mean projection (`projection_profile="published"`); the
reviewed formula stays available (`"normalised"`). The paper's set of tested projections,
one per unordered pair of trials, is available as `projection_pairs="half"`; the default
stays each trial in one pair (`"disjoint"`), because the paper's set rejects pure noise far
above alpha. `docs/spec/DECISIONS.md` records both choices.

- Published values: Miller et al. 2023, PLOS Comput Biol 19(5):e1011105, on the paper's
  public example data (`sampledata` of github.com/kaijmiller/crp_scripts, commit f6efb4e,
  read at test time and never copied; window 15 ms to 1 s).
- Durations (within 5 ms required): Fig. 3, 80 ms published, 76.2 ms measured (-3.8 ms;
  15.6 ms before); Fig. 9, 0.23 s published, 232.9 ms (+2.9 ms; 17.1 ms before); Fig. 9
  without trial 6, 0.28 s published, 282.7 ms (+2.7 ms; 15.1 ms before). Met (3 of 3).
- Significance (95 % required): Fig. 5 (four significant, one not, published p = 0.37) and
  Fig. 6 (A, C, E significant; B, F not). With `projection_pairs="half"`: 10 of 10 (Fig. 5
  bottom p = 0.368). With the default pairs: 0.8 to 1.0 over seeds 0 to 9 (mean 0.89);
  before this change, 9 of 10 at seed 0. Met with the paper's set.
- Pure noise, rejection rate at alpha = 0.05 (10 or 20 trials, white or AR(1) 0.9, 300
  draws): paper's set 0.33 to 0.44 (0.24 to 0.35 with the reviewed profile), default set
  0.07 to 0.13 (either profile).
- Benchmark `core`, projection baseline rerun on T1 and T2 alone (the committed results in
  `docs/benchmark/` are those of the earlier run): T1 AUC 0.938 -> 0.946, F1 0.776 ->
  0.809, recall 0.647 -> 0.706, FPR at 0.05 0.064 -> 0.080; T2 reported fraction 0.647 ->
  0.706, latency MAE 0.86 -> 0.95 ms, amplitude relative MAE 0.165 -> 0.150.
- Not met as written: REQ-134 names ds003708, the dataset of the basis-profile-curve paper.
  No per-channel-pair duration or significance of the cross-projection method has been
  published for it (not in the paper, its supporting figures, its code repository or the
  ds003708 deposit and derivatives). Strict expected failure stating this
  (`tests/existence/test_projection_published.py`).

**REQ-373, Localize-MI.** Not met; strict expected failures
(`tests/validation/test_localize_mi.py`). The data paper (Mikulan et al. 2020, Sci Data
7:127) reports no list of stimulations producing a scalp response, there is no sham
condition (0.1 to 5 mA), and the authors' conversion code (github.com/iTCf/mikulan_et_al_2020,
`fx_bids.export_data_bids`) crops the shared epochs at +10 ms after the pulse. G-Node GIN and
EBRAINS, which host the data, are refused by the network policy of this machine; the second
test reads the header of one epochs file on GIN with `EVOPARAM_NETWORK_TESTS=1`.

**REQ-374, layer-3 comparison records.** `validation/layer3_report.py` writes
`validation/layer3/comparisons.json` (22 records) from the layer-3 result files: comparator
number, evoparam number, difference (evoparam minus comparator) with a 95 % paired bootstrap
interval (2 000 resamples, seed 0), dataset id, version and DOI, and the comparator
definition with its version (registry rule id and SHA-256, or the definition and its
SHA-256, or the external comparator's commit). `--check` tells whether the file is current.
`validation/layer3/carla_ds004977.json` now holds evoparam's own selection on the same
channels (Jaccard index with CARLA: 1.000, 0.857, 0.863, 0.981 for ROC11-ROC12, RZ4-RZ5,
LA2-LA3, RO3-RO4). Selected records:

| record | comparator | evoparam | difference [95 % interval] |
|---|---|---|---|
| ds004080 N1 fraction, `vanblooijs2023` | 0.262 | 0.163 | -0.099 [-0.133, -0.067] |
| ds004080 N1 fraction, `vanblooijs2023_code` | 0.055 | 0.163 | +0.107 [+0.083, +0.133] |
| ds004080 N1 latency (ms), `vanblooijs2023` (74 channels) | 27.59 | 28.89 | -0.57 [-3.01, +0.63] |
| ds004080 N1 latency (ms), `vanblooijs2023_code` (30 channels) | 23.44 | 20.23 | -0.54 [-2.36, +0.15] |
| ERP CORE P3 area latency (ms), difference wave | 425.8 | 429.7 | +3.9 [0.0, +11.7] |
| ERP CORE MMN area latency (ms), parent wave | 203.1 | 179.7 | -7.8 [-43.0, -3.9] |
| ds004977 left-out fraction, LA2-LA3 | 0.276 | 0.238 | -0.038 [-0.070, -0.016] |

Two versions were not recorded when the results were produced: the SHA-256 of the ERP CORE
files read on 2026-10-09 (OSF has no versions) and the commit of CARLA_JNM run on
2026-10-08. They are `null` with a note in 18 of the 22 records, and the version check is a
strict expected failure until a rerun; `validation/erpcore.py` and
`validation/carla_ds004977.py` record them from now on, and `validation/robustness.py`
writes the data version and the rule ids into the N1 file.

**REQ-024, writers.** Met here. `write_nwb` output (with and without the evoparam
extension) re-reads with pynwb 4.2.0 and `pynwb.validate` reports no error. The
`write_bids_derivatives` output passes the BIDS validator 3.0.2 (`jsr:@bids/validator`,
through Deno 2.2.15) with no error in derivative mode, and the simulated raw data in raw
mode; that test runs when the validator is installed.

### Engine: noise that does not fill its band is modelled at the rate it occupies

The `ccep` fit of ds004080 sub-ccepAgeUMCU01 run 021448 (recorded at 512 Hz) found an N1 on
54 of 100 channels once the run was upsampled to 1024 Hz, against 14 at 512 Hz (entry
below). The same happens with any recording whose noise stops well below the Nyquist
frequency: upsampled data, or an anti-alias or hardware low-pass well below it.

Cause, measured on that run. The AR model of the baseline (order up to 10) sees a band with
almost no power. The largest power gain of its whitening filter is 1 956 (median over the
100 channels; 183 to 2 496), against 32 (4 to 73) at 512 Hz. In the
fit window the upsampled data are not empty there: the upsampling filter rings next to the
stimulation artifact, and only the interpolated samples carry that ringing (fitting the
recorded samples alone gives 12 N1s, the interpolated ones 43). On four channels, 53 to 92 %
of the whitened residual of the background-only model lay in the upper half of the band, and
the fit spent narrow components, often in cancelling pairs of thousands of µV, on it.

Fix (`models/engine.py`). `noise_stride` compares the median level of the Hann-tapered
periodogram of the noise (the baseline, or the plus-minus reference) in the half-octave just
above 1/k of the Nyquist frequency with the octave below it, for k = 4, 3, 2. When it is below
`EMPTY_BAND` = 1e-4 (a band all but empty), the fit is judged on every k-th sample
(`StridedWhitener`): the AR model is fitted to the noise sampled at that rate, it whitens
the residual at lags k, 2k, ..., and the other rows carry no evidence. `min_fwhm` is raised
to two samples of that rate, as it already is for a recording at that rate. The phase of
those samples is the one whose noise has the highest level at the top of its band
(`noise_phase`): an upsampled recording keeps its own samples on one phase, and the
interpolated ones are attenuated near the old Nyquist frequency, so its fit is judged as at
its recorded rate wherever the epoch starts. The fit says so: the warning
`band_limited_noise:k`, `diagnostics["noise_stride"]` and `diagnostics["noise_phase"]`. When
the noise fills its band (k = 1) the code path is the previous one, and the results are the
same bit for bit.

The threshold separates interpolation from the data the engine already fitted well. On the
ds004080 baselines of the REQ-375 runs, at their recorded rates and decimated to 1024 and
512 Hz, the level ratio is -1.7 decades or more (615 channels x 3 rates, k = 2 to 4), and
-1.5 or more on unfiltered simulations. On the upsampled run it is -5.4 decades or less
(k = 2). A first version used 1e-3: it also strided ERP simulations low-passed at 30 or
40 Hz by an order-2 Butterworth at 1024 Hz (ratio -2.8 to -3.4), and made them worse
(recall 0.950 to 0.875 at 30 Hz). No fit of the test suites or of the quality sets below
has a stride other than 1.

Results.

| | 512 Hz (recorded) | 1024 Hz before | 1024 Hz after |
|---|---|---|---|
| channels with an N1 (of 100) | 14 | 54 | 12 |
| decisions equal to 512 Hz | - | 56 | 98 (2 N1s lost, none added) |
| components per channel (mean) | 1.15 | 2.59 | 1.14 |
| channels with a component above 1 000 µV before 30 ms | 1 | 22 | 2 |
| rule channels found, paper reading (32) / code reading (4) | 13 / 4 | 22 / 3 | 11 / 4 |
| `fit_group` time, 100 channels | 30 s | 429 s | 43 s |

If the sampling analysis still upsampled this run (it no longer does, entry below), the pooled
1024 Hz reading would be: paper recall 0.466 (75/161) before and 0.398 (64/161) after, kappa
0.383 and 0.422, n_model 125 and 83; code recall 0.765 (26/34) and 0.794 (27/34), kappa 0.263
and 0.416.

On simulations (`simulate_ccep`, 512 Hz, with artifact, upsampled with `prep.resample`;
seeds 0 to 3, 12, 8, 8 and 8 channels) the total number of components went from 11, 12, 13
and 11 at 512 Hz to 28, 33, 32 and 28 at 1024 Hz before and 10, 12, 7 and 11 after. The N1
decisions agree with 512 Hz on 12/12, 8/8, 7/8 and 8/8 channels after (11/12, 8/8, 7/8 and
7/8 before). `tests/models/test_band_limited_noise.py` checks the detection (noise that fills
its band, upsampled by 2, 3 and 4), that the strided whitening equals the whitening at the
lower rate bit for bit, and the seed-1 case (components 12 at 512 Hz, 12 after, 33 without
the stride).

Unchanged, checked:

- Quality sets of the engine (`fit_waveform`/`fit` on the REQ-104 to REQ-111 and REQ-116
  designs, ECAP and CCEP simulations, three ds004080 runs): 2 182 fits, all identical bit for
  bit (components, residual, warnings, number of evaluations). REQ-105 recall stays 0.8167,
  0.8583 and 0.9222 for K = 1, 2 and 3 with precision 1.
- Layer 3, `validation/layer3/n1_ds004080.json`: the 615 native decisions and latencies are
  identical (paper kappa 0.458, recall 0.460; code kappa 0.398, recall 0.882).
- REQ-375 sampling analysis, recomputed (19.4 min, 1 process): every decision and latency at
  the recorded rate, 1024 Hz and 512 Hz equals the report, which was not rewritten. Paper
  reading: recall 0.460, 0.410 and 0.460, kappa 0.458, 0.434 and 0.404 (recorded, 1024, 512);
  code reading: recall 0.882, 0.794 and 0.824, kappa 0.398, 0.407 and 0.315. The 1024 Hz drop
  stays +4.97 points (paper, met) and +8.82 points (code, not met; strict expected failure).
- The reference fixture (`tests/reference/fit_fixture_ccep.json`) is unchanged: the fixture
  fit is the same (largest relative difference 4.2e-12, as before this change).

Algorithm versions (REQ-399): `fit_group`, `fit_waveform`, `test_response` (its
`model_comparison` method fits) and `fit_trials` (it fits when no group fit is given) go to
1.1, since their output changes for band-limited noise. `soup.json` and `THIRD_PARTY.md` were
regenerated by `tools/gen_soup.py`: `numpy.polyval` is new, and 11 names used by the REQ-115
solver work were missing (numpy `copy`, `equal`, `multiply`, `nextafter`, `take`; SciPy
`linalg.lapack.dgeqrf`, `dgesdd`, `dgesdd_lwork`, `dormqr`, `dtrtri`, `dtrtrs`), which made
`test_req399_soup_and_third_party_are_up_to_date` fail.

Tried and not kept, on the same run (N1 channels at 1024 Hz, 14 at 512 Hz):

- a limit on the whitening gain (white noise added to the AR model, diagonal loading, then
  the AR refitted at the same order): 42, 48 and 55 for a power gain limit of 1 000, 316
  and 100. The band still holds the ringing, and an AR of order 10 refitted to the loaded
  autocovariances distorts the occupied band;
- the same with the samples counted by their Satterthwaite number of degrees of freedom: at a
  limit of 100 the count fell to 2 to 64 % of the samples, and N1s found at 512 Hz were lost
  (channel 3);
- every other sample after a triangular low-pass (which cancels the ringing): 43, because the
  low-pass and the upsampling filter leave a dip at the new Nyquist frequency that the
  whitening amplifies again;
- every other sample on the interpolated samples instead of the recorded ones: 43.

Recordings low-passed well below their Nyquist frequency (ERP CORE: 256 Hz after a
non-causal 0.1 to 30 Hz ERPLAB Butterworth; clinical CCEP pipelines) also trigger the stride.
Measured on simulations low-passed by `sosfiltfilt` Butterworth filters of order 2 and 4
(the squared response of a non-causal filter), fitted with `evoparam.fit(...,
existence=False, random_state=0)`: `simulate_erp` with 60 trials and 4 channels, P1/N1/P3
and N170/P3 on alternate seeds (`erp` preset), or the ERN alone (epoch -0.6 to 0.6 s,
baseline -0.6 to -0.2 s, fit window -0.1 to 0.3 s), seeds 0 to 7 (32 fits); `simulate_ccep`
with 10 channels at 2048 Hz, seeds 0 to 3 (seeds 0 to 15 at 300 Hz order 4). A component is
found when one of the same sign lies within half its FWHM; latency and amplitude errors are
over the components found.

| data | low-pass | fits strided | recall | precision | false positives | mean abs latency error (ms) | mean abs amplitude error |
|---|---|---|---|---|---|---|---|
| ERP 256 Hz | 30 Hz, order 4 | 32/32 (k = 2, 3) | 0.850 -> 0.963 | 0.366 -> 0.811 | 118 -> 18 | 4.06 -> 1.73 | 0.198 -> 0.055 |
| ERP 256 Hz | 40 Hz, order 4 | 32/32 (k = 2) | 0.925 -> 0.975 | 0.420 -> 0.907 | 102 -> 8 | 3.86 -> 1.12 | 0.185 -> 0.029 |
| ERN 256 Hz | 30 Hz, order 4 | 32/32 | 1 -> 1 | 0.185 -> 0.744 | 141 -> 11 | 1.70 -> 0.58 | 0.061 -> 0.020 |
| ERN 256 Hz | 40 Hz, order 4 | 32/32 | 1 -> 1 | 0.219 -> 0.914 | 114 -> 3 | 1.40 -> 0.49 | 0.068 -> 0.013 |
| ERP 1024 Hz | 30 Hz, order 4 | 2/32 | 0.912 -> 0.912 | 0.380 -> 0.386 | 119 -> 116 | 4.51 -> 4.55 | 0.129 -> 0.131 |
| ERP 1024 Hz | 40 Hz, order 4 | 21/32 | 0.838 -> 0.875 | 0.354 -> 0.422 | 122 -> 96 | 4.17 -> 3.44 | 0.173 -> 0.114 |
| ERN 1024 Hz | 30 Hz, order 4 | 8/32 | 1 -> 1 | 0.167 -> 0.179 | 160 -> 147 | 1.61 -> 1.60 | 0.097 -> 0.066 |
| ERN 1024 Hz | 40 Hz, order 4 | 31/32 | 1 -> 1 | 0.167 -> 0.248 | 160 -> 97 | 2.18 -> 1.27 | 0.082 -> 0.039 |
| CCEP 2048 Hz | 300 Hz, order 4 | 160/160 (k = 2) | 0.988 -> 0.988 | 0.131 -> 0.188 | 550 -> 359 | 3.09 -> 1.92 | 0.142 -> 0.078 |
| all | none; order 2 (30, 40, 300, 500 Hz); CCEP 500 Hz order 4 | 0 | unchanged | | | | |

Per component, strided, at 256 Hz: P1, N1, N170, P3 and ERN are found as often or more often
(P1 10/16 -> 14/16 at 30 Hz), with smaller latency and amplitude errors for each. At 1024 Hz
and 40 Hz the N1 is found on 16/16 channels instead of 15/16, with a larger mean amplitude
error (0.143 -> 0.180); the N170, P1 and P3 errors fall. On the CCEP the N1 errors fall
(latency 0.28 -> 0.19 ms) and so do the N2 ones (6.4 -> 4.1 ms); target fits with a
component above three times the largest true amplitude go from 3 to 0 of 46. Of the two
strided ERP fits at 1024 Hz and 30 Hz, one loses three false positives and the other keeps
its components, with the N170 amplitude at -45.4 instead of -43.1 µV (true -43.2).

The `min_fwhm` raise (two samples at the strided rate: 15.6, 23.4 and 31.3 ms at 256 Hz for
k = 2, 3, 4) does not clip these components: no strided ERP fit has a component at the
raised floor (0 of 181), and an ERN fit 2 of 78. The raise acts only at 256 Hz here (at
1024 and 2048 Hz two strided samples stay below the preset `min_fwhm`). Without it (measured
with a first phase rule) the ERN fits were the same, and the ERP ones slightly worse: at
30 Hz recall 0.963 instead of 0.975 and latency error 2.01 instead of 1.70 ms, at 40 Hz
latency error 1.27 instead of 1.23 ms (with the first threshold, 1e-3: recall 0.950 instead
of 0.975).
A band empty above 1/k of the Nyquist frequency holds
no component narrower than about two samples at that rate: such a shape has its power above
the band edge, and the low-pass has removed it.

The phase matters only where one is special. On ds004080 run 021448 upsampled to 1024 Hz
and then cropped by one sample, a fixed phase used the interpolated samples (43 N1s); the
chosen phase is the recorded samples on all 100 channels (margin 0.6 to 1.8 decades of
level) and gives 10 N1s, 94 of 100 decisions equal to those at 512 Hz (5 N1s lost, 1
added; uncropped: 12 N1s, 98 equal); the crop also moves the data one sample (about 1 ms)
against their time stamps. On upsampled noise
(`resample_poly`, k = 2 to 4, 300 samples or more per phase) it never missed in 720 trials;
with 60 per phase it missed 1 in 80 at k = 2 and 1 in 4 at k = 4. An ideal (FFT)
interpolation leaves no phase special and leaves no ringing on only some of them. On the
low-passed simulations no phase is special either: at 256 Hz, 30 Hz and order 4, the
precision is 0.727 with the first phase, 0.762 with the last and 0.744 with the one chosen
(ERN), and 0.794, 0.811 and 0.811 (ERP); the choice only moves the fit within that spread.

### Layer 4, sampling rate (REQ-375, §7.4 (3)): no upsampling; still not met

§7.4 (3) decimates ds004080 to 1024 and 512 Hz. `validation/robustness.py` also
resampled the example run, sub-ccepAgeUMCU01 run 021448, which is recorded at 512 Hz. At
"1024 Hz" this run was therefore upsampled. Upsampling is not decimation, and the result is
not a 1024 Hz recording: it has no power between 256 and 512 Hz. On these data the AR noise
model of the `ccep` fit (order chosen up to 10 samples) whitens a spectrum with an empty
band. Small misfits in that band then count as strong evidence, and the fit keeps many
narrow components, often in cancelling pairs, before 30 ms. The fit found an N1 on 54 of the
100 channels of that run instead of 14. Most of the rise of `n_model` from 100 to 127, and
the drop of κ (code reading) from 0.40 to 0.26, came from this run.

A run recorded at or below a rate is now kept as recorded (`robustness.decimates`); the
report names it. The sampling section of `validation/layer4/0.1.0.dev0/report.json` and
`report.md` was recomputed alone (19.4 min, 1 process). The other analyses and the
native-rate N1 file are unchanged. The native decisions are identical (0 of 615 changed).
The criterion and the reference rule are unchanged.

| reading | rate | recall before | recall after | κ before | κ after | n_model before | n_model after |
|---|---|---|---|---|---|---|---|
| paper | native | 0.460 (74/161) | 0.460 (74/161) | 0.458 | 0.458 | 100 | 100 |
| paper | 1024 | 0.466 (75/161) | 0.410 (66/161) | 0.377 | 0.434 | 127 | 85 |
| paper | 512 | 0.466 (75/161) | 0.460 (74/161) | 0.413 | 0.404 | 116 | 116 |
| code | native | 0.882 (30/34) | 0.882 (30/34) | 0.398 | 0.398 | 100 | 100 |
| code | 1024 | 0.765 (26/34) | 0.794 (27/34) | 0.258 | 0.407 | 127 | 85 |
| code | 512 | 0.853 (29/34) | 0.824 (28/34) | 0.329 | 0.315 | 116 | 116 |

The recall drop at 1024 Hz is now +4.97 points for the paper reading (met; it was −0.6
because the upsampled run's spurious N1s also hit rule channels) and +8.82 points for the
code reading (not met; it was +11.8). The test stays a strict expected failure, and its
reason now gives both values with two decimals. On unchanged inputs, 5 decisions of the
resampled 2 048 Hz runs differ from the previous report: 1 at 1024 Hz and 4 at 512 Hz. One of
them is ccepAgeUMCU02 channel 6 at 512 Hz, a rule channel in both readings. These are fits
that flip between local optima with the current engine code.

Why the code reading still fails. The three code-rule channels lost at 1024 Hz are all in
sub-ccepAgeUMCU06 run 030854, and they are recovered at 512 Hz:

- channel 0: the rule's N1 is at 10.3 ms, at the start of the fit window (9 ms). The model's
  early negative component peaks at 14.8 ms natively and at 1.3 ms at 1024 Hz, before the
  N1 window.
- channels 6 and 10: the rule's N1 is at 93.7 ms and 94.7 ms, outside the model's N1 window
  (9–50 ms). The native "hits" are small unrelated components at 47.5 ms and 20.1 ms.

No parameter of the `ccep` path was found to be in samples where it should be in time. The
candidate smoothing, the edge rule, the widths, the windows, the priors and the artifact
window are all in seconds. `min_fwhm` is raised to 2 samples only below 667 Hz. The one
count in samples is the cap of the AR order (10). At 1024 Hz, a cap of 20 or of 5 gives 8
of 13 rule hits on that run instead of 7, each time on different channels, so the cap is
not the cause. Other checks:

- perturbing the native trials by 1e-15 and by 1e-6 relative changes 1 and 4 of the 615
  decisions, and none of the 34 code-rule channels;
- the other decimation phase at 1024 Hz (odd samples) gives 25 of 34: five rule channels
  flip between the two phases, while the total stays at 25–26.

The loss is therefore systematic at 1024 Hz. It is not rounding noise. It comes from how the
engine decomposes the early deflection at the edge of the fit window. The engine's
sensitivity to oversampled input remains: a recording whose noise does not fill its band up
to the Nyquist frequency is over-fitted in the same way. It is not changed here.

### Engine speed (REQ-115, REQ-290 P1): same results, about 1.8 times faster

REQ-115 is still not met. The criteria are unchanged and the xfail reasons in
`models/test_performance.py` now give the new measurements. Timings were taken on the same
machine with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, the previous code (9b443b6) and the
new one run interleaved or side by side:

| workload | before | after |
|---|---|---|
| one waveform (REQ-115 test, `ccep`): median of 9, three interleaved rounds | 0.52 to 0.55 s | 0.29 to 0.30 s |
| `fit_group`, 100 channels x 2 048 samples, `n_jobs=1` (two rounds) | 48.1 s, 45.4 s | 25.0 s, 24.9 s |
| ds004080 sub-ccepAgeUMCU01: `fit(ts, "ccep", existence=False)`, 120 channels | 73.9 s | 40.0 s |
| ds004080 sub-ccepAgeUMCU07, 120 channels | 212.1 s | 117.5 s |
| ds004080 sub-ccepAgeUMCU19, 120 channels | 361.4 s | 214.5 s |

What changed. Every change computes what the previous code computed:

- `models/_trf.py` is a lean copy of SciPy's trust-region reflective solver, limited to the
  bounded, dense, linear-loss, `x_scale="jac"`, exact-subproblem case the engine uses. The
  file is derived from SciPy (BSD-3-Clause), which `NOTICE` records with its licence text;
  `tools/add_spdx_headers.py` gives such files the header `Apache-2.0 AND BSD-3-Clause`.
  - It takes SciPy's steps: on six test problems it gives the same number of evaluations and
    the same status as `least_squares`, with x within 1e-10 relative (`test_trf.py`).
  - Dropping SciPy's general wrappers alone saved 17 % on 14 test workloads (7.40 to 6.11 s).
  - Each iteration then works on R and Q^T f, the QR factors of the Jacobian (n x n), rather
    than on the SVD of the (m + n) x n augmented matrix.
  - Robust losses keep `scipy.optimize.least_squares`.
- The variable projection is solved by Householder QR of the whitened design, prior rows
  included (`_Structure.reduced`):
  - The residual is returned in the basis Q2^T b. It has the same norm, and with the Jacobian
    in the same basis the inner products are the same, and they are all the solver uses.
  - The Kaufman Jacobian is one `dormqr` call.
  - A design that is not finite, or whose Frobenius condition number of R is 1e11 or more,
    takes the previous SVD route, which truncates it (`test_fast_paths.py` covers both).
  - On the 14 workloads, in the same process: SciPy with the SVD route 6.4 to 7.0 s;
    lean solver on the reduced residual 3.6 to 3.8 s; lean solver with the R reduction 3.2
    to 3.7 s. The number of evaluations is the same in all three (4 667), and the largest
    relative parameter difference is 1.5e-7, the solver's tolerance.
- Design columns and their derivatives are computed together (`models/_columns.py`, for the
  Gaussian, the skewed Gaussian and the exponential background), using the formulas of
  `evoparam.shapes` in the same order. They are bitwise equal to `evaluate` and `jacobian`.
- `exp(-z^2/2)` is not computed where |z| > 38.7, where it underflows to exactly 0
  (`shapes.components.gaussian_factor`). The result is bitwise equal, sign of zero
  included.
- The skewed-Gaussian mode has a scalar path when there is one skewness: 1.8 times faster
  per call, bitwise equal on 3 000 skewness values.
- The AR whitening is applied in place, with time along the first axis
  (`Whitener.into`). This is 2.4 times faster than with time along the second axis.
- Bounds are memoised per fit and per set of shapes, and the whitened target is cached.
- The features of each component are measured once at unit amplitude and then scaled by
  the amplitude. They are bitwise equal on 465 cases, and building a `ccep` result went from
  19 to 12.6 ms.
- The bootstrap of the classical measures draws from a time-major copy of the window and
  keeps the same summation order. It is bitwise equal for 3 to 40 trials; the time per
  channel went from 0.084, 0.092, 0.122 and 0.100 s to 0.078, 0.095, 0.101 and 0.098 s.

Quality: the baselines were recorded before the first change, and the same runs were
repeated on the final code.

- Simulations: every metric is identical.
  - REQ-105 (60 waveforms per K): recall 0.8167, 0.8583 and 0.9222 for K = 1, 2, 3, precision 1.
  - REQ-116 coverage, latency and amplitude: 0.96 and 0.96 at rho 0; 0.92 and 0.95 at rho 0.9.
  - REQ-107 false rate: 0 with neff and with ar, 1 with white.
  - REQ-111: two components resolved in 1.0 at 1 FWHM and 0 at 0.5 FWHM.
  - REQ-110 recall: 0.95.
  - REQ-104 agreement: 1.
  - REQ-106: the five shapes, with and without their own shape.
  - `ecap`: recall 1, precision 0.9545.
  - Simulated `ccep`: recall 1, precision 0.4444.
  - Largest per-fit difference: 1.9e-10 s in latency and 1.5e-8 relative in amplitude.
  - No status, number of components or label changed.
- ds004080, three subjects, 360 fits:
  - No status changed.
  - 348 fits are within 1e-6 s and 1e-5 relative.
  - 9 differ at the solver tolerance: the residual sum of squares is within 8.7e-5
    relative, lower on 6 and higher on 3.
  - 3 reach another local optimum:
    - UMCU07 TOc19: 4 to 5 components, criterion -669.15 to -708.12 (better).
    - UMCU19 el083: 3251.06 to 3263.31 (worse).
    - UMCU19 el088: 4379.74 to 4380.30 (worse). Labels change on el083 and el088.
  - Sum of the criterion changes: -26.2.
  - These three are rounding effects. Scaling the data by 1 +/- 1e-15 or 3e-15 changes the
    same channels with the previous code: TOc19 goes to 5 components (-708.12), el083 to
    3263.31, and el088 to 4379.95 or 4380.22. The new code also lands on 3251.06 for el083
    under such a perturbation.

Tried and not adopted:

- A projected Levenberg-Marquardt on the same reduced residual. On the 14 workloads it
  needed 7 500 to 15 700 evaluations against 4 667 for the trust-region solver, and took
  3.8 to 6.4 s against 3.3 to 4.0 s. It found better optima on 1 or 2 fits and worse ones on
  1 or 2.
- numba (the `fast` extra):
  - A whitening kernel is bitwise equal to the NumPy one, but takes 78 to 146 us against 107
    to 143 us per call. Whitening is 13 % of a ds004080 fit, so the gain is at most a few
    percent, at the cost of import and compilation time.
  - Kernels for `exp` and `erfc`, or for the solver, cannot match NumPy and SciPy to the last
    bit (different libm). The selection is sensitive at that level, as the 1e-15
    perturbations above show, so the 1e-9 agreement with the pure-NumPy fallback could not
    be guaranteed.
- Fewer evaluations in the search. No evaluation is repeated within a fit. On 30 ds004080
  channels, the share of the time per call site is:
  - quick evaluations of the candidates: 29 %;
  - background restarts (`_polish`): 22 %, plus 9 % in reseeding;
  - splits: 6 %;
  - reseeding: 5 %;
  - perturbed starts: 4 %;
  - the K = 0 grid: 2 %.
  None of them is provably redundant, and cutting any of them changes the selection.

`compute_features` raises an IndexError on some gamma components (`_interp_cross` reads one
sample past the grid). This was seen while writing `test_fast_paths.py`, is not changed
here, and the test uses the ex-Gaussian instead.

### Second network run on the Mac (REQ-317, REQ-068)

- REQ-317: reading epochs covering 10 % of the 1 GB synthetic NWB file grew the peak
  memory by 0.251 to 0.252 GB in 2 of 4 runs (limit 0.25 GB; the epochs themselves are
  0.214 GB of float64). Given a path, hdmf opens the HDF5 file with a larger chunk cache
  than h5py's default, which added about 35 MB when the epochs are scattered over the
  series. `read_nwb` and `read_nwb_results` now open local files with `h5py.File` (default
  1 MiB cache per dataset) and hand them to `NWBHDF5IO`: growth 0.224 GB, 4 of 4 runs pass.
  The limit is unchanged.
- REQ-068 (`polynomial`, 2.09 s against 2.0 s) failed only while other work ran on the
  machine; run alone it passes. No change.
- REQ-274, LRP: 34 of 40 participants within 10 ms of the published latency (85 %) where
  the measurement of 2026-10-09 had 36 (90 %). Found by bisection over `integration-2`
  (ERP CORE runs only with the data at hand): the first commit with 85 % is 636a266, the
  engine default `reject_cancelling=False` with reseeding. For participants 14 and 36 the
  reseeded model keeps a cancelling pair marked not resolvable (`unresolvable:0`,
  `unresolvable:1`) and the labelled LRP is one of its members (latency -22 ms and -11 ms
  against the published -51 ms); refusing the pair gives -56 ms and -54 ms.
  `erp` 1.0 -> 1.1 sets `reject_cancelling=True`. Over the 7 ERP CORE components x 40
  participants, only the LRP changes (34 -> 36 within 10 ms; MMN labelled 16 -> 15, within
  10 ms unchanged at 15); the engine default stays (measured on the ECAP simulations; on
  the layer-2 grid the two are within noise, DECISIONS). The methods text of the
  `erpcore_p3` reference pipeline is regenerated (preset version and the sentence on the
  refused pairs), `docs/presets.md` regenerated. Results made with `erp` 1.0 are refused
  by `evoparam.reproduce` (REQ-333). The 90 % threshold is unchanged.

### Build step S20: robustness layer, release engineering, documentation (spec §7.4, §9, §11.8, §13.7)

Requirements REQ-290 to REQ-309, REQ-375, REQ-376 and REQ-390 to REQ-399. Nothing public
was done: every step that publishes (PyPI, Zenodo, a public leaderboard, a preprint, the
freeze of the name, the filing with HIIE) waits for the owner's HIIE decision, and its
test is a strict expected failure with that reason. Readings of the spec are in
`docs/spec/DECISIONS.md`, section 6.

#### Package behaviour

- Logging (REQ-304): `evoparam.verbose`, `set_log_level` and `set_log_file`. Every
  module logs to a child of the `evoparam` logger; the default level is WARNING.
- Deprecation shims (REQ-293): `evoparam.utils._deprecation` keeps a renamed public name
  for two minor versions. The old name emits a `FutureWarning` that names the new one and
  the version that removes it. The table of deprecated names is `evoparam/_deprecated.py`
  (empty at 0.1.0).
- Extras (REQ-292): the table `evoparam/_extras.py` names, for each optional
  distribution, its single extra. `numpy>=1.24,<3` and `scipy>=1.10` are now the only
  core dependencies (REQ-309 floors; before: `numpy`, `scipy>=1.8`). `pyyaml>=6` was
  added to the `tests` and `all` extras.
- `evoparam.selftest()` (REQ-291) fits one bundled synthetic CCEP with the `ccep` preset
  and checks the N1 latency within 1 ms: an error of 0.043 ms, in 2.5 s on the reference
  machine. `evoparam.cite()` prints the citation of the installed version.
- `on_error` (REQ-307): `ResponseModel.fit_waveform` and `fit_group` take
  `on_error="record"` (default) or `"raise"`:
  - with `"record"`, a failed channel is kept with `status="failed"` and the exception in
    `message`, and exactly one WARNING names the channel;
  - with `"raise"`, a `FitError` is raised;
  - data and settings errors always propagate.
- Provenance (REQ-294, REQ-399): every step records `function`, `settings`, `version`,
  `algorithm_version` (`evoparam.version.ALGORITHM_VERSIONS`), `preset`, `random_state`,
  `duration_s` and `n_jobs`. The environment records `lock_sha256`: the SHA-256 of
  `ci/locks/pyX.Y.lock` when the installed packages are exactly its pins, else null.
- Reference values (REQ-295): `evoparam/tests/reference/fit_fixture_ccep.json` holds the
  fit of the bundled fixture. Any platform must reproduce it within rtol 1e-9.
- Network (REQ-306): the tests of the `core` marker set run behind a socket guard
  (`conftest.py`). The only network code is in the explicit fetch functions.
- Methods text (REQ-390):
  - every saved result that has a methods paragraph stores it as `methods.txt`, listed in
    the container's manifest;
  - NWB processing modules carry it in their description (`"methods"`);
  - `TFFit.methods_text` no longer fails on a fit with empty settings ("not recorded").
- Typing (REQ-301): annotated `evoparam.selftest`, `evoparam.cite`, the logging functions
  and `ArtifactResult.cleaned`.
- CLI:
  - `evoparam benchmark run --suite perf|ci` runs the performance workloads P1–P6;
    `--task` selects workloads, for example `--task P1,P6`;
  - `--task` and `--method` are aliases of `--tasks` and `--methods`, so
    `evoparam benchmark run --suite core --task T1 --method vanblooijs2023` reproduces one
    leaderboard entry (REQ-392). For `vanblooijs2023__T1` it gives the same metrics, hashes
    and signature as the committed file, in 14 s.

#### Performance (REQ-290, REQ-299)

- `evoparam.benchmark.timing` and `benchmarks/run_benchmarks.py` run the six workloads of
  §9.1 on fixtures generated with `evoparam.sim` (`random_state=0`).
  - Results go to `benchmarks/results/<platform>-<python>-<commit>.json`; the `ci` suite
    writes `...-ci.json`.
  - Peak memory is sampled every 50 ms from `/proc/self/statm`.
  - `--compare` fails above 1.25× the baseline median time or 1.10× its peak memory, or
    when a P3 figure is above its bound, with a message naming the workload and the ratio.
    It warns between 1.10× and 1.25×.
  - `--repeat-runs 3 --write-baseline` writes the baseline as the median of three runs; the
    `perf` workflow's `baseline-update` job uses it.
- The perf suite on the reference machine (`benchmarks/results/linux-3.13.16-bddedc9aa35f.json`,
  one thread, load average 9.7 on 4 cores):

  | id | measured | target | met |
  |---|---|---|---|
  | P1 | median 1.30 s (20 runs) | < 0.050 s | no |
  | P2 | 53.7 s (5 000 fits, `n_jobs=4`) | < 60 s | yes |
  | P3 | see below | < 2× data; mapped < 0.25× + 512 MB | in memory no, mapped yes |
  | P4 | no GPU backend in this version | CPU = GPU within 1e-6 | not available |
  | P5 | 266.5 s for a 270.7 s original fit (0.98×), identical numbers | < 1.5× | yes |
  | P6 | median 0.909 s (10 runs), no optional module loaded | < 0.300 s | no |

  P3 at the CI size (50 × 64 × 4 096 float64, 105 MB; medians of 3 runs): in memory, peak
  RSS +331 MiB (347 MB, 3.3× the data, bound 210 MB); memory-mapped, the same +331 MiB,
  under its bound of 563 MB (0.25× data + 512 MiB). The full size (4.19 GB) was not run:
  at 3.3× it would exceed the machine's 15 GB.
- `benchmarks/baseline-ci.json` is the median of three runs of the `ci` suite: P1 1.20 s,
  P2 2.68 s, P3 55.7 s (memory-mapped 53.8 s), P5 29.4 s, P6 0.904 s. It was written on the reference machine. It holds
  until the `perf` workflow's `baseline-update` job regenerates it on its runner. The CI
  run of P3 in memory fails its bound, so the `perf` job will fail on it until the memory
  use of `evoparam.fit` is reduced.

#### Robustness layer (REQ-375, REQ-376)

- `validation/robustness.py` runs the five analyses of §7.4. It writes
  `validation/layer4/<version>/report.json` and `report.md`, and
  `docs/preset_sensitivity.md` (linked from the preset page). The release workflow runs it
  and keeps the report with the release records. It was run once on the reference machine
  (77.8 min, 2 processes), with the ds004080 and ds004977 epochs already cached; nothing was
  downloaded.
  1. Montage: 7 ds004080 runs and 4 ds004977 sites, existence test under three references.
     Median Cohen's κ: bipolar–CAR 0.23, bipolar–adjusted CAR 0.38, CAR–adjusted CAR 0.76.
     The lowest κ is 0.00, 0.00 and 0.04. The median latency differences are 0.00, 0.21
     and 0.00 ms.
  2. Preset: each *(choice)* field ×0.5 and ×1.5, 20 seeds. There are 4 sensitivities, all
     on `ecap`:
     - `epoch` ×0.5: recall 0.875 → 0.000;
     - `fit_window` ×0.5: recall → 0.750;
     - `fit_window` ×1.5: recall → 0.725;
     - `component_labels` ×0.5: label recall → 0.000.

     `ccep`, `ccep_seeg`, `erna`, `erp` and `sep` show none: the largest latency change is
     0.17 ms, and the largest recall change −0.015.
  3. Sampling rate, N1 recall against the published rule at the native rate:
     - paper reading: 0.460 native, 0.466 at 1024 Hz (−0.6 points), 0.466 at 512 Hz. Met.
     - code reading: 0.882 native, 0.765 at 1024 Hz (+11.8 points), 0.853 at 512 Hz. Not
       met: the criterion is at most 5 points, so the test is a strict expected failure
       with these values.
  4. Trial count, 35 draws per count. The share of existence decisions that change against
     all trials: 0.259 with 5 trials (median κ 0.00), 0.115 with 7 (0.59), 0.000 with 10
     (1.00).
  5. Platform: the bundled fixture's fit equals the reference values exactly on Linux
     x86-64 (largest relative difference 0). macOS and Windows are cells of the CI matrix,
     not run here (strict expected failure).

  The report waits to be attached to release notes until there is a release (HIIE).
- `validation/layer3/n1_ds004080.json` is new: the layer-3 N1 agreement at the native rate,
  on 7 runs and 615 channels.

  | reading | κ | recall | median latency difference | within 2 ms |
  |---|---|---|---|---|
  | paper | 0.458 | 0.460 | 8.46 ms | 0.243 |
  | code | 0.398 | 0.882 | 3.47 ms | 0.367 |

- `tools/layer3_gate.py` is the release gate of REQ-376. It reads the pre-registered
  tolerances in `validation/layer3/tolerances.json`: κ −0.05, median latency +1 ms, recall
  −3 points. It compares 42 metrics (the N1 file and the ERP CORE files) with the latest
  `v*` tag, or with `HEAD` when there is no tag. A regression blocks the release unless the
  `## Unreleased` section has a line `Layer-3 regression explained: <metric>: <reason>`.
  On this branch: 0 regressions.

#### Release engineering (REQ-297 to REQ-300, REQ-305, REQ-308, REQ-309)

- Workflows (written, checked statically by `tests/test_ci_workflows.py`; none was run):
  - `ci` (lint, format, typecheck, tests matrix 3.10–3.13 × three systems, core-only,
    lowest floors, doctests, import graph, build, `benchmark-results`);
  - `docs`, `perf`, `nightly`, `smoke`;
  - `release`, which starts with `tools/release_gate.py`. The PyPI upload job is kept
    outside `.github/workflows` (`ci/release-publish-job.yml`) until the HIIE decision.
- Lock files `ci/locks/py3.10.lock` to `py3.13.lock`, with hashes.
- `GOVERNANCE.md`, holding the HIIE decision block.
- `RELEASING.md`: 12 steps, plus checks 6–9 (provenance map, traceability, algorithm-version
  bump, review of TRADEMARK.md at a major release).
- `.pre-commit-config.yaml` and a pull-request template.
- Sphinx configuration: bibtex with 1 095 entries generated from the spec's references,
  version switcher, linkcheck, examples gallery.
- Quality, measured (nothing was installed or rewritten for a linter):
  - `ruff check`: 4 644 findings on the CI scope, 2 830 in the package outside its tests.
    `PLC0415` is ignored (DECISIONS).
  - `ruff format --check`: 371 of 413 files would be reformatted.
  - `interrogate`: 63.0 %.
  - `mypy --strict evoparam`: 5 209 errors in 142 of 163 modules.
  - `pyright --verifytypes`: 34.8 % of the exported symbols have a known type.
  - Coverage of `evoparam.shapes` by its own tests: 90 %.

  Each is a strict expected failure with these numbers.

#### Documentation (REQ-302, REQ-303, REQ-391, REQ-393)

- New pages:
  - getting started;
  - reporting and citing;
  - FAQ (seven questions);
  - glossary of spec section 4;
  - API index;
  - dataset registry;
  - preset sensitivity.
- The glossary, the API index, the registry page and the preset page are generated by
  `tools/gen_*.py`; each tool has a `--check` mode that is tested.
- `docs/benchmark/SUBMITTING.md`: the submission format of the benchmark (REQ-392).
- `make figures` (`tools/make_figures.py`) regenerates the three figures of the papers
  from repository files. The output is SVG with a fixed hash salt and no date, listed with
  SHA-256 in `figures/manifest.json`. `--check` reproduced 3 of 3 hashes (matplotlib
  3.11.2).

#### Provenance map and inventory (REQ-395 to REQ-397, REQ-399)

- An SPDX header (`Apache-2.0`) is in every Python file outside `adapters/`. The three
  inherited files say what they were modified from (`tools/add_spdx_headers.py --check`).
- `REIMPLEMENTED.md` lists the modules reimplemented from published descriptions.
- `TRADEMARK.md`.
- `soup.json` and `THIRD_PARTY.md` list 54 packages, with their role, licence and the
  functions used (`tools/gen_soup.py --check`).
- `tools/gen_traceability.py` writes the requirement → test report attached to each
  release. On this branch, 4 of the 227 requirements of the matrix are named by no test:
  REQ-024, REQ-134, REQ-373 and REQ-374, all of earlier build steps.

### Build step S19: public benchmark (`evoparam.benchmark`, spec §8, REQ-350 to REQ-357)

- Seven tasks (T1 response detection, T2 early component latency and amplitude, T3
  component decomposition, T4 artifact removal, T5 single-trial estimation, T6 ERP
  scoring, T7 time-frequency response). Each has a versioned JSON Schema
  `benchmark-task/<id>/1` (`evoparam/benchmark/schemas/`, JSON Schema 2020-12) for its
  input (a JSON descriptor next to an `.npz`) and its output (a TSV with fixed columns, or
  an `.npz` for T4). Every file is validated when it is read. A file that does not
  validate raises `BenchmarkValidationError`, which names the file, the line, the column
  and the first violation (REQ-351).
- Frozen suites `core` (60 cells: 48 public, 12 hidden) and `full` (300 cells: 240
  public, 60 hidden), generated on the fly by `evoparam.sim` from their settings and
  seeds (`evoparam/benchmark/suites/*.json`, written by `tools/benchmark_suites.py`).
  The hidden seeds are drawn once into a document kept outside the repository; that
  document holds a 256-bit nonce. Both suite files hold its SHA-256 (canonical JSON):
  `ff9e2df66b2065833a2bb83b197f90a35bebe1d51c263cbcd6d150534185f701`. A document that
  does not match is refused.
- New generator `evoparam.sim.simulate_bursts`: induced bursts, Hann-windowed cosines
  with a random phase per trial, the T7 truth.
- Baselines (§8.3):
  - the six published threshold rules of §5.6, plus the registry's code version of
    vanblooijs2023;
  - the cross-projection test;
  - classical measures;
  - template alignment;
  - the evoparam presets;
  - interpolation and no removal for T4;
  - `evoparam.tf` with cluster permutations for T7.
- Adapters (REQ-354): `Adapter.run(task_input_path) -> task_output_path` runs an
  external tool in a subprocess on the task files and validates its output.
  - Nothing of the tool is imported.
  - The ER-detect and CRP adapters are in `adapters/` at the root of the repository,
    under a GPL-3.0-or-later notice. That folder is outside the package and is not
    shipped (`prune adapters`).
  - These two adapters are not verified against the tools, which are not installed
    here.
- Scoring (REQ-352): `evoparam benchmark score` writes a signed result.
  - The result holds the SHA-256 of every input and output, the suite and its SHA-256,
    and the package version.
  - `signature.digest` is the SHA-256 of the document without its signature.
  - The result holds no clock time and no path, so the same inputs and outputs give the
    same bytes.
- Report and leaderboard (REQ-357): `tools/benchmark_leaderboard.py` writes
  `docs/benchmark/leaderboard.html` and `docs/benchmark/report-core.md` from the signed
  result files of `docs/benchmark/results/` only. `--check` fails when the page or the
  report differs from what the results give. A result whose signature does not match is
  refused.
- Real data (§8.2): index files and loaders only, no data.
  - `ds004080`: the six runs of `validation/ds004080_subjects.txt`.
  - `ds003708` and `ds004977`.
  - `ds002094`, for real-noise injection.
  - `erpcore`: 5 components × 40 participants, referenced only.
  - The T1/T2 truth is the consensus of at least 2 of 4 published rules; contacts where
    exactly one rule detects a response are not scored.
  - The BIDS loader picks channels by their type in channels.tsv (a run need not have
    every type).
- Command line: `evoparam benchmark run | score | leaderboard | prepare-deposit`.
  `cli → benchmark` is added to the import table (DECISIONS §2).
- Deposit (REQ-355, REQ-356): `prepare-deposit` writes, locally only, BIDS, NWB, the task
  inputs, the truth and a manifest. The manifest holds the generator version, the public
  seeds, every file's SHA-256 and the commitment, checked against the hidden seeds
  document. Nothing is uploaded. The deposit with a DOI waits for the owner's HIIE
  decision, so both criteria are strict xfails.
- Measured.
  - `evoparam benchmark run --suite core` covered T1 to T7: 48 public cells, 15 methods,
    30 signed results, no failure.
  - It took 811.2 s in total (13.5 min): cells 796.9 s, scoring 14.4 s, 812.6 s wall
    clock with the imports. Machine: one process on a shared 4-CPU container, load
    average 4.6 to 10.5 during the run (`docs/benchmark/run-core.json`).
  - Time by group of cells: tf 564.4 s, ccep_detection 144.6 s, single_trial 42.6 s,
    evoked 36.6 s, erp_scoring 4.5 s, artifact 2.4 s.
  - Time by method: evoparam-preset 509.2 s, tf-cluster 248.4 s, template 21.9 s,
    simulation 6.4 s, projection 3.0 s, classical 1.6 s, each threshold rule at most
    0.6 s.
  - `prepare-deposit --suite core` (BIDS and NWB) took 17.4 s and wrote 645 files,
    640 MB: BIDS 110 MB, NWB 225 MB, inputs 303 MB, truth 1.8 MB.
  - Through NWB the trials come back with a largest difference of 0.
- First results on `core` (public split, 8 cells per task; latencies in ms):

  | T1 | AUC | F1 | precision | recall | FPR at 0.05 |
  |---|---|---|---|---|---|
  | vanblooijs2023 | 0.9849 | 0.8596 | 0.7778 | 0.9608 | n/a |
  | hays2021 | 0.9882 | 0.8276 | 1 | 0.7059 | n/a |
  | keller2014 | 0.9805 | 0.8276 | 1 | 0.7059 | n/a |
  | silverstein2020 | 0.9882 | 0.8667 | 1 | 0.7647 | n/a |
  | ftract | 0.9426 | 0.9053 | 0.9773 | 0.8431 | n/a |
  | prime2020 | 0.9451 | 0.6923 | 1 | 0.5294 | n/a |
  | projection | 0.9379 | 0.7765 | 0.9706 | 0.6471 | 0.064 |
  | evoparam-preset | 0.9484 | 0.8444 | 0.9744 | 0.7451 | 0.032 |

  | T2 | reported | latency MAE | latency bias | amplitude rel. MAE |
  |---|---|---|---|---|
  | vanblooijs2023 | 0.9608 | 8.719 | 7.753 | 0.2566 |
  | hays2021 | 0.7059 | 0.937 | 0.1324 | 0.1386 |
  | keller2014 | 0.7059 | 0.937 | 0.1324 | 0.138 |
  | silverstein2020 | 0.7647 | 0.9172 | 0.1302 | 0.1697 |
  | ftract | 0.8431 | 125.9 | 125.8 | 0.1427 |
  | prime2020 | 0.5294 | 6.419 | 5.697 | 0.1723 |
  | projection | 0.6471 | 0.8593 | 0.08741 | 0.1651 |
  | evoparam-preset | 0.7843 | 0.4336 | 0.2131 | 0.09177 |

  - T3 F1: evoparam-preset 0.8927, classical 0.6212. False components per empty
    waveform: 0.038 and 2.96.
  - T4 residual over artifact: evoparam-preset 0.2144, interpolation 0.2538, no removal 1.
  - T5 latency MAE: template 6.94 ms, evoparam-preset 17.37 ms, classical 28.59 ms.
  - T6 mean-amplitude MAE against the truth: evoparam-preset 0.6466, classical 0.6984.
    Classical against the definition: 0. Peak-latency MAE: 7.43 ms and 11.72 ms.
  - T7 burst recovery: evoparam-preset 1 (type I error 0), tf-cluster 0.8571 (type I
    error 0).
- What the numbers show.
  - The registry's code version of vanblooijs2023 detects nothing on `core` (AUC 0.5).
    Its 50 µV floor on the baseline SD puts the threshold at 170 µV, and the simulated
    N1 range from 9 to 78 µV.
  - ftract's T2 latency is the absolute peak of 0 to 0.8 s. That is a later component,
    hence the 126 ms bias.
  - No T4 baseline removes the decaying tail beyond the artifact window: the N1 latency
    error is 2.43 ms with and without removal.
- Tests: `evoparam/tests/benchmark/`, with `test_suite_core.py`, `test_schemas.py`,
  `test_scoring.py`, `test_adapters.py`, `test_deposit.py`, `test_leaderboard.py` and
  `test_realdata.py`; also `tests/sim/test_sim_bursts.py`.
  - The byte-for-byte check of REQ-352 scores the same predictions in two separate
    processes and compares the files. The other comparisons use rtol 1e-9.
  - The timing test runs with `EVOPARAM_PERF_TESTS`, the re-run of the committed
    results with `EVOPARAM_SLOW_TESTS`, the loaders with `EVOPARAM_NETWORK_TESTS`.
  - The check of the kept hidden seeds runs when `EVOPARAM_BENCHMARK_HIDDEN_SEEDS` names
    the document.
- Results:
  - default run: 107 passed, 4 skipped, 2 strict xfails (REQ-355, REQ-356);
  - slow re-run of T1/T2: passed (same output hashes, same metrics);
  - ds004080 loader: passed, in 223 s;
  - ERP CORE loader: failed here because OSF is unreachable through this machine's
    proxy (HTTP 403), which is not a verdict on the loader;
  - timing test (`EVOPARAM_PERF_TESTS=1`): passed. The second full run of
    `benchmark run --suite core` took 359.9 s (6.0 min) on the same container under less
    load.

### Subspace projection on real artifact templates (REQ-064)

- REQ-064 asks `subspace` for at least 30 dB of signal-to-interference improvement on
  synthetic EEG with real artifact templates. With the specification's version (the
  directions holding 95 % of the whitened window energy, at most 6; every sample
  projected) the overlays of `tests/artifact/test_subspace_real.py` gave 0.2 to 4.1 dB on
  the 20 TMS subjects of ds002094 and 5.0 to 18.4 dB on the 23 SPES runs of ds004080: one
  whitened direction holds 95 % of the energy of a real template, whose spatial rank is
  high. Sweeping the number of directions shows that projecting every sample reaches
  30 dB only by removing most of the signal (45 dB on TMS with 15 to 29 directions, but a
  relative error of 0.95 to 1.20 on 12 to 50 ms, the error as large as the signal).
- `n_components="noise"`: the directions whose whitened window energy per sample exceeds
  twice the largest eigenvalue of whitened noise, `(1 + sqrt(n_channels / n_samples))^2`
  (at most `n_channels - 1`). `subspace_extent="window"`: only the samples of the window
  are projected, the rest of the epoch stays as recorded.
- Measured on the same overlays (seed 0, 30 trials) and on the synthetic two-topography
  artifact with a decay beyond the window of `test_subspace.py` (seeds 4 to 13):

  | version | TMS (20) | SPES (23) | synthetic (10) | relative error 12-50 ms (TMS) |
  |---|---|---|---|---|
  | specification (energy, every sample) | 0.2-4.1 dB, 0/20 | 5.0-18.4 dB, 0/23 | 14.9-33.1 dB, 5/10 | 0.48-2.20 |
  | noise level, every sample | 45.0-47.0 dB | 33.1-38.5 dB | 28.0-48.0 dB, 9/10 | 0.95-1.20 |
  | noise level, window only (new default) | 59.4-60.7 dB, 20/20 | 48.6-55.0 dB, 23/23 | 42.7-43.7 dB, 10/10 | 0.02-0.03 |

  The new version is the default of `remove_artifact(method="subspace")`; the
  specification's is `n_components="energy", subspace_extent="all"`, kept in tests as a
  strict xfail with the numbers above. The `tep` preset keeps the specification's version
  after its interpolation (the projection there is meant for the decay beyond the window;
  the new default has not been measured on TEPs). The methods text now states the rule and
  the extent of the projection (`methods/sim_tep.txt` regenerated).

### Fitting speed on intracranial recordings

- Measured on one ds004080 subject (120 channels, `fit(ts, "ccep")`, both versions run side
  by side on the same machine): 100 s before, 72 s after; the tables agree (same labels and
  statuses, largest relative difference 2e-5, the tolerance of the optimiser).
- Reseeding is not what made CCEP fits slow: on three ds004080 subjects it is tried on 3 to
  5 of 120 channels, and skipping its final polish when it cannot win saves 3 to 6 % only
  (not adopted). The time goes to the base fit, about 1 s per channel.
- The three changes, each checked against the previous code: the AR whitening applies its
  FIR filter as shifted sums (scipy's `lfilter` with denominator `[1]` convolves column by
  column through `apply_along_axis`; difference 4e-16); the ERPLAB-style local peak of the
  classical measures compares all samples at once (identical choice on 3000 random
  waveforms); the mode of the skewed Gaussian is solved on Python floats when there is one
  skewness (difference 1e-16, about 20 times faster per call).

### Network run on a machine with access (REQ-340, ERP CORE P3)

- The full suite with `EVOPARAM_NETWORK_TESTS`, `EVOPARAM_SLOW_TESTS` and
  `EVOPARAM_PERF_TESTS` and every optional package was run on a machine that reaches OSF,
  DANDI and OpenNeuro. On the real ERP CORE P3 recording (participant 1, Pz, 256 Hz) the
  methods text named the plus-minus noise reference, while the frozen text, made from the
  stand-in, named the baseline: with `noise_source="auto"` the reference depends on the
  effective size of the baseline (51 samples at 256 Hz, fewer than 30 effective ones), and
  the stand-in was simulated at 500 Hz, where the baseline was long enough.
- The ERP CORE P3 stand-in is now simulated at 256 Hz, the sampling of the real recording,
  so it resolves the same noise reference; `methods/erpcore_p3.txt` is regenerated (one
  sentence: "the noise of the plus-minus average of the trials (alternating signs, whole
  epoch)" instead of "the noise of the baseline"). The real recording and the stand-in now
  give the same frozen text.
- `rodent_ep` 1.1 (REQ-278, Allen Visual Coding Neuropixels LFP, 17 visual-cortex channels
  where the existence test is significant): with `noise_source="auto"` the baseline
  (-100 to -5 ms at 1250 Hz) holds fewer than 30 effective samples, so the fits used the
  plus-minus reference, whose AR whitening leaves little evidence for the broad late LFP
  deflections: 15 of 17 channels (88 %) with a P1/N1 or late component, two channels
  "no_response" although a one-component model explains 60 % and 36 % of their variance.
  The preset now sets `noise_source="baseline"`, the reviewed procedure of spec §5.5.3
  step 1 (the plus-minus reference was measured on scalp ERPs only): 16 of 17 (94 %).
  Measured on the only public rodent session of the tests, the same one REQ-278 scores,
  so it is a development measure; the remaining channel (e850211775, a weak trough at
  100-250 ms followed by a slow rise) stays "no_response" with either reference. The
  threshold of REQ-278 (90 %) is unchanged. `docs/presets.md` regenerated.
- Hierarchical models (REQ-172, numpyro engine, run for the first time with the `bayes`
  extra installed): the between-channel latency SD fell inside its 95 % posterior interval
  in 7 of 10 reference datasets (17 of 20 checks with the trial SD, 85 %), always missed
  from above. Cause: the standard error given to the channel latency (the spread of the
  trusted trial shifts) left out the noise of the average: on four reference datasets
  (348 responding channels) the latency errors had an SD of 2.27 ms against a median
  standard error of 1.14 ms (z-scores SD 1.81; 1.55 with the Wald error alone), and the
  missing error went into the between-channel SD. The channel latency now carries both
  errors in quadrature (z-scores SD 1.02): 9 of 10 datasets, 19 of 20 checks (95 %); the
  ±20 % EB part of REQ-172 and every other hierarchical test pass, REQ-171 stays an
  expected failure. The 90 % threshold is unchanged; a slight upward tendency remains (the
  one miss is again from above).

### Interval coverage (REQ-116, bootstrap and profile, 200 runs each)

- Profile intervals: the ends are now where the profile statistic crosses the chi-square
  limit, linearly interpolated between the outermost grid value inside and its neighbour
  outside. Before, the ends were grid values, which shortened the interval by up to one
  grid step (0.4 SE) per side.
- Bootstrap intervals: expanded percentile interval (Hesterberg 2015), the tails at
  Phi(-sqrt(n / (n - 1)) t_{n-1}) instead of alpha / 2; the plain percentile interval of an
  average of n trials is too narrow.

  | Case | Latency before | Latency now | Amplitude before | Amplitude now |
  |---|---|---|---|---|
  | bootstrap, white noise | 0.900 | 0.935 | 0.945 | 0.960 |
  | bootstrap, AR(1) rho 0.9 | 0.935 | 0.945 | 0.900 | 0.940 |
  | profile, white noise | 0.870 | 0.935 | 0.975 | 0.975 |
  | profile, AR(1) rho 0.9 | 0.880 | 0.925 | 0.955 | 0.955 |

  Three of the four cases now meet the 92-97 % of the spec. The profile case under white
  noise stays an xfail: its amplitude coverage is 0.975 (195 of 200, the Wald interval of
  the linear amplitude, unchanged), above 0.97.

### Defaults settled by measurement: integration of the trials and engine corrections

Merged into `integration-2`: `corrections-essais` (2b8aba2, trials), `corrections-moteur`
(361135c and 07082b4, engine and N1 case as written, `noise_model="neff"`) and
`wip-paires-compensees` (57f977e, `fit_shared` latency bounds and `reject_cancelling`).
Every feature of both sides is kept: the Mac's reseeding, `resolvable` column and
plus-minus noise reference (`noise_source="auto"`) next to `latency_order`,
`candidate_search`, `threshold_role`, `noise_model="neff"` and `reject_cancelling`.

Protocol. Choices were made on development data and the REQ values are reported on the REQ
data: the five ds004080 development runs (UMCU07 run-021752 sTa5-sTa6, UMCU19 run-040955
PT01-PT02, UMCU28 run-021305 F01-F02, UMCU40 run-031553 FB1-FB2, UMCU49 run-031049
HP01-HP02) against the REQ-371 runs and the example run; seeds 0-2 against 3-5 of
`simulate_ecap` (cochlear: `ecap_ci`, spinal: `ecap`; 40 trials, no artifact, 7 contacts);
the layer-2 grid on 41 cells of the weekly design (every fifth cell), 5 replicates, with
development seeds and held-out seeds (offset 500 000). 4 cores, one process per task.

- `threshold_role` (§5.5.3 step 3). Engine default kept at `"order"`; the `ccep` preset
  (now v1.1) sets `"select"`.

  | | `"order"` | `"select"` |
  |---|---|---|
  | REQ-104 (`erp`, 100 waveforms, thresholds 1/2/3 agree) | 100/100 (met) | 87/100 |
  | N1 decision kappa, development runs (paper / code) | 0.36 / 0.31 | 0.40 / 0.39 |
  | N1 decision kappa, REQ-371 runs pooled (paper / code) | 0.46 / 0.37 | 0.46 / 0.40 |
  | the same, 46 PT contacts of the example run | 0.51 / 0.17 | 0.46 / 0.19 |
  | REQ-113 regression decision (trough with the code's floor), pooled / lowest run | 0.89 / 0.71 | 0.89 / 0.71 |
  | REQ-114, N1 latency within 2 ms (paper / code) | 21 % / 32 % | 24 % / 37 % |
  | fitting time, development / REQ runs | 846 s / 1557 s | 603 s / 1133 s |
  | REQ-115, one waveform (median of 5, two rounds) | 0.80 s, 0.76 s | 0.53 s, 0.57 s |
  | REQ-115, `fit_group` 100 channels | 80 s, 84 s | 53 s, 55 s |
  | layer-2 F1, all cells (development / held out) | 0.736 / 0.756 | 0.720 / 0.748 |
  | layer-2 F1, CCEP cells | 0.715 / 0.707 | 0.693 / 0.707 |
  | layer-2 F1, ECAP cells | 0.680 / 0.742 | 0.640 / 0.692 |

  REQ-104 is met only with `"order"`, so it stays the engine default (and the `erp`
  preset's). For `ccep` the real-data N1 agreement and the speed favour `"select"` (the
  CCEP simulation cells lose 0.02 of F1 on the development seeds, nothing held out). REQ-113
  and REQ-114 as written stay xfails under both; REQ-115 stays an xfail (0.55 s against
  50 ms, 54 s against 10 s). `ccep_seeg` was not measured (no sEEG N1 reference here) and
  keeps the engine default.
- Noise model (§5.5.3 step 1). Default kept at `"ar"` for every data type; `"neff"` and
  `"white"` stay options, and `noise_source="auto"` stays the default reference.

  | | `"ar"` | `"neff"` | `"white"` |
  |---|---|---|---|
  | REQ-107, AR(1) rho 0.9, false components in 100 windows | 0 % | 0 % | 100 % |
  | REQ-112, ds004080 baselines silent (80 channels, `ccep` v1.1) | 97.5 % | 97.5 % | 65 % |
  | REQ-276, ERNA cases within tolerance (20) | 0.90 | 0.90 | 0.90 |
  | ds004080 N1: components over all channels of the 12 runs | 2 092 | 12 | - |
  | N1 decision kappa, REQ-371 runs pooled (paper / code) | 0.46 / 0.37 | 0.01 / 0.00 | - |
  | layer-2 F1 (development / held out) | 0.736 / 0.756 | 0.681 / 0.693 | - |

  With `"neff"` the criterion counts n (1 - rho) / (1 + rho) samples; at 2048 Hz rho is
  close to 1 and the BIC penalty removes almost every CCEP component. REQ-170 was not rerun
  with `"neff"` (`"neff"` was already excluded by the two rows above); it runs with the
  default in the suite. The ERP CORE false-positive rates of the noise references are the
  ones recorded by the plus-minus work ("Engine: noise from the plus-minus average"): OSF is
  not reachable from this build machine.
- Cancelling pairs (§5.5.3 step 5, risk S1). Default: `reseed=True`,
  `reject_cancelling=False`; `reject_cancelling=True` stays an option (now named in the
  methods text when on). The ECAP presets (`ecap`, `ecap_ci`, now v1.1) set
  `reseed="always"`.

  | ECAP simulations, development / held-out seeds | reseed (default) | reject | reseed "always" |
  |---|---|---|---|
  | cochlear F1 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
  | cochlear amplitude error | 8.3 % / 6.8 % | 8.7 % / 14.6 % | 1.5 % / 5.1 % |
  | cochlear fits above 5 x the noise RSS (of 21) | 10 / 2 | 11 / 10 | 0 / 0 |
  | spinal F1 | 0.95 / 0.96 | 0.91 / 0.90 | 0.95 / 0.99 |
  | spinal amplitude error | 2.7 % / 5.6 % | 5.0 % / 10.8 % | 2.7 % / 3.0 % |
  | fitting time, cochlear / spinal (3 seeds) | 4.5 s / 25 s | 4.2 s / 23 s | 17 s / 43 s |

  On the layer-2 grid reject and reseed are within noise (F1 0.737 against 0.736
  development, 0.760 against 0.756 held out), as on the N1 case (pooled kappa 0.46 / 0.31
  against 0.46 / 0.37 on the REQ-371 runs; 0.41 / 0.31 against 0.36 / 0.31 on the
  development runs). On ds004977 sub-1 run-01 ROC11-ROC12 (11 bipolar ROC channels,
  `ccep_seeg`, window 9-490 ms) the shared fit:

  | `fit_shared` | channels with a latency outside the window | channels with a pair | residual cost |
  |---|---|---|---|
  | before the bounds fix (reseed) | 11 / 11 (-9.0 ms; -17.8 ms with free latencies) | 0 | 606 481 / 565 734 |
  | bounds fix, reseed | 0 | 0 | 638 620 / 607 422 |
  | bounds fix, reject and drop | 0 | 0 (3 components dropped) | 1 154 093 / 1 120 501 |

  The latency bounds of `fit_shared` (every channel's latency, shared value plus offset,
  inside the fit window, in reported-latency coordinates) are a bug fix and are merged.
  `fit_shared` keeps its own pair rule, now its own keyword
  (`fit_shared(..., reject_cancelling=True)`, recorded in `settings["fit_shared"]`) and
  no longer read from the engine settings: the shared fit has no reseeding.
- Pruning after a reseed (new). The greedy path may have needed three components to build
  a cancelling set; after the refit from the extrema, a component that no longer lowers the
  criterion by `delta` (the acceptance rule of §5.5.3 step 4) is removed (warning
  `pruned:<k>`, named in the methods text). On the spinal ECAP simulations it removes
  the tiny edge component the refit kept (F1 0.91 -> 0.95 development, 0.90 -> 0.96 held out
  with `reseed=True`; 0.91 -> 0.99 held out with `"always"`); on the N1 runs and the layer-2
  grid the kappas are unchanged to 0.001 and F1 within 0.001 (the tables above were
  re-measured with it, except the `"neff"`, `"white"` and reject columns, which do not
  reseed).
- `candidate_search`: `"beam"` stays the default, `"spec"` (§5.5.3 step 4) available;
  checked after the merge.
- Presets: `ccep` 1.0 -> 1.1 (`threshold_role="select"`), `ecap` and `ecap_ci` 1.0 -> 1.1
  (`reseed="always"`), each with its reason in the preset; `docs/presets.md` regenerated.
  Results made with the 1.0 presets are refused by `evoparam.reproduce` (REQ-333).
- REQ-349: `plots/_signals.py` no longer computes anything. The average and its band and
  the strip-plot jitter come from two new public functions, `evoparam.stats.waveform_band`
  and `evoparam.stats.strip_jitter` (same values as before, tested).
- Import graph (§4.2): `trials` and `spatial` imported `evoparam.shapes` directly after the
  merges; they now use `evoparam.models.component_shapes`, which re-exports `Gaussian`,
  `SkewedGaussian` and `skew_mode`.
- REQ-012: `tools/check_no_inherited_code.py` reads the inherited repository and commit
  from NOTICE instead of naming them.
- Duplicate `noise_reference` key removed from the diagnostics of skipped fits.
- REQ-004: the `min_amplitude` check of the engine (§5.5.10, off by default) is written
  out again in the two acceptance loops, where the no-deletion scan reads it; the
  `_too_small` helper of the engine corrections hid it from the scan, which then no longer
  caught a mutation of it. Same behaviour (every component of the candidate, not only the
  new one).
- REQ-001 test: the `sep` preset now gets a median-nerve SEP waveform (N20, P25, N33). The
  ERP default (P1, N1, P3) lies after its 10-60 ms window: it passed before only through a
  component at 62.5 ms, outside the window, which the latency bounds now forbid
  (`no_response` is the right answer there).
- F58 test (`ccep_seeg` reference report): since the existence test became the detector
  of `reference_sensitivity` (artifact corrections), the 8 simulated contacts at 50 %
  responding were all detected under every scheme (decay tail after cleaning, spread of
  the common average), so kappa was undefined. The test now simulates 16 contacts at 30 %
  (the simulator's default): 8-11 of 14-16 detected, kappa 0.48-0.71.
- `stats.fit_mixed`: rows with a missing value are dropped before statsmodels is called.
  statsmodels 0.15 with `missing="drop"` does not find a quoted name (`Q("stim site")`,
  `Q("if")`) and failed on REQ-236 cases 2 and 3.
- `hierarchical`: `InferenceData` built with the API of the installed arviz (a dict of
  groups for arviz >= 1, keywords for 0.x, as `HierarchicalFit.to_arviz` already did);
  with arviz 0.23 the dict was read as one variable and R-hat failed (both engines, LOO
  comparison).
- REQ-275 regression floor (ds002094, 20 subjects; Python 3.13, numpy 2.5.3), subjects
  with N45 / N100 / both labelled:

  | Variant | Floor (S15) | origin/main | before the merges (78cc6b0) | now |
  |---|---|---|---|---|
  | preset chain, 17 inner channels | 11 / 13 / 9 | 9 / 11 / 6 | 9 / 11 / 6 | 10 / 12 / 8 |
  | with ICA, all 30 channels | 9 / 11 / 6 | 8 / 8 / 5 | 9 / 9 / 6 | 9 / 14 / 8 |
  | with ICA, 17 inner channels, N100 | 16 | 12 | 13 | 16 |

  The floor was already missed on origin/main in this environment; the merged engine
  is better on every row but still misses the first. The guard is a strict xfail with
  these numbers, and the REQ-275 criterion itself (both labelled for 80 % with ICA)
  stays an xfail (8 of 20).

### Reports, shared shapes and preset behaviours (audit F44, F56–F58)

- ERP CORE P3 methods text on the real recording (F44, REQ-340). The network test
  `test_req340_erpcore_p3_real_recording_gives_the_frozen_text`
  (`EVOPARAM_NETWORK_TESTS=1`) downloads participant 1 of the P3 task from OSF (node
  `etdkz`, CC-BY-4.0) into `EVOPARAM_ERPCORE` or `~/.cache/evoparam/erpcore`, checks the
  sha256 of the files, runs `run_reference_pipeline("erpcore_p3", ...)` on Pz and compares
  the result with the frozen text. The data is never stored in the repository.
  **Not verified yet:** on 2026-10-09 the build environment's network policy refused
  `api.osf.io`, `osf.io` and `files.osf.io` (proxy 403). The local runner (D16) runs this
  test.
- `ShapeBasis.weights` now holds the clustering loadings, group × shape (F56, §4.3.4,
  §5.9). For NMF these are the columns of the coefficient matrix H, and groups with fewer
  than 3 trials get the non-negative least-squares loadings of their t-values on W. For
  kernel PCA they are the one-hot of the assigned shape. Before this change, the field
  held the mean projections.
  - The spec's group-level weight `α_m` (the mean projection on the assigned shape) stays
    in `assignment["weight"]`. `assignment["nmf_loading"]` gives the group's loading on
    that shape.
  - The full matrix of mean projections on every shape is no longer returned. No
    measurement showed it was better for any use.
- `plot_parameter_map` can draw a glass brain (F56, §5.9). With `backend="auto"`, it uses
  nilearn when nilearn is installed and the coordinates are MNI-like:
  - one `plot_markers` panel set;
  - stimulation contacts, bad channels and masked channels as distinct markers.

  Otherwise it draws the matplotlib electrode grid, whose axis labels now name the
  coordinate space. `backend="nilearn"` with subject-space coordinates raises
  `SettingsError`. There is a new extra, `brain = ["nilearn>=0.10"]`, which is included in
  `all`.
- The existence statistic over time (F57, §5.6). `ResponseTest` gains `curve_times`,
  `curve` and `null_band` (channel × time, × 2 for the band):
  - permutation: the pointwise t, with a band of ±the max-t critical value or ±the cluster
    threshold;
  - shuffle: z, with a band of ±the critical z;
  - threshold rules: the baseline z, with a band of ±k on the sides the rule tests;
  - projection: the profile, with no band.

  `plot_existence` draws one panel per channel: the curve, the shaded band and the
  onset–offset span. It falls back to bars for old objects.
- Methods-text paragraphs for every recorded step (F57, REQ-340). New templates cover:
  - pulse detection (`detect_pulses` now records a step on the event table, which
    `epoch` carries);
  - filters;
  - mask application;
  - baseline per train;
  - reference sensitivity;
  - bootstrap, comparison, reliability, trials needed, jackknife and group summary;
  - `TrialSet.apply`.

  `methods_text` also accepts stats tables (`attrs["provenance"]`). A test checks that
  every step recorded by the package has a template or is explicitly silent. The three
  frozen texts are unchanged (`tools/freeze_methods_text.py --check`: same, same, same).
- `report.pdf` has one page per stimulation site (F57, §5.13), after the text page:
  - for fits, the site's group plot;
  - for a test split by site, the site's existence curves.

  Then come the per-channel figures. The new `reports._documents.site_pages` lists the
  pages.
- Preset behaviours of §6.1 and §6.2 (F58). There are three new `Preset` fields:
  `reference_sensitivity`, `trial_covariates` and `train_baseline`.
  - `ccep_seeg` runs `prep.reference_sensitivity` by default. Its κ report is attached as
    `PipelineResult.reference_report` and described in the methods text.
    `run_pipeline(..., reference_sensitivity=False)` turns it off.
  - `rodent_ep` and `primate_ep` use `metadata["state"]` as a default trial covariate.
    The new `trials.state_dependence` gives, per state, the medians and means, the gain
    relative to the most frequent state, and Kruskal-Wallis p-values. When the column is
    missing, there is a warning and the fits run without it.
  - `erna` subtracts a baseline per train: the last 50 ms before each train. The new
    `prep.train_baseline` takes it from the continuous recording
    (`run_pipeline(continuous=...)`) or from the first pulse's epoch. If neither is
    available, the pipeline warns and the baseline is not applied.
  - "P1/N1 by depth": `label_components(polarity_rule="sign")` (the default) names a
    component in a window called `P1/N1` as `P1` when it is positive and `N1` when it is
    negative. The test of REQ-278 (`test_allen_lfp.py`) accepts `P1`, `N1` or `late`,
    which is the same set as before.
- BIDS derivatives: `_preset` accepts the `{"name", "version"}` preset form of REQ-294.

### Trials: speed, REQ-150/151/155/156/158 as written, spec options (audit F10, F17, F36–F38, Q6, Q8, Q9)

- Speed (F10, REQ-156). In `shift_scale` mode (and `template` mode), the trials of a channel
  are now solved together. The solver is a projected Levenberg-Marquardt iteration
  vectorised over trials, for the same model and bounds:
  - variable projection with the Kaufman Jacobian;
  - an active set on the bounds;
  - damping from the gain ratio;
  - a shape moved by `d` is evaluated once, on shifted times;
  - columns that do not move are projected out once per channel;
  - the Jacobian is computed only for the steps that are kept.

  `jitter_sd_null` solves all its baseline segments in one call. The pre-stimulus
  autoregressive phase estimate is vectorised (same numbers to 1e-12). The initial damping
  (1) was chosen on simulated CCEPs, not on ds004080: 0.4 % of trials end at a worse local
  optimum than the one-trial solver, against 1.4 % with 1e-3.

  On the REQ-156 runs, `fit_trials` takes 1.03 s (UMCU03) and 1.25 s (UMCU06), against
  11.4 s and 19.2 s on the base branch, with `OMP_NUM_THREADS=1` on the shared build
  machine.
- Trial model (Q8, REQ-156). `fit_trials(trial_model=...)` has three values:
  - `"anchored"`, the new default: every component of the channel is in the model, but only
    the reported ones move. The others keep their channel latency and shape, and their
    amplitude is fitted. `max_shift` is clipped only against moving neighbours.
  - `"all"`: every component moves.
  - `"selected"` (the spec): only the reported components are in the model.

  The reason for the change is on real data. With `"all"`, the channel fits of ds004080 often
  split an N1 into overlapping parts (`N1`, `N1_b`, …). The neighbour rule then clips the
  N1 `max_shift` to 0.1–2 ms, and trials sit at that bound. The choice of default was made
  on four held-out ds004080 subjects (UMCU07, 08, 10 and 11; the first pair with 10 trials,
  same pipeline), not on the six REQ-156 subjects. Median trusted fraction on the
  significant N1 channels:

  | Data | `"anchored"` | `"all"` |
  |---|---|---|
  | Held-out subjects (25 channels) | 0.9 | 0.7 |
  | REQ-156 subjects, reported (56 channels) | 0.9 | 0.7 |

  On simulations `"all"` is slightly more accurate. On simulate_ccep with 10 trials (N1
  latency and normalised amplitude MAE, paired Wilcoxon test):

  | Jitter | Channels | Latency MAE, anchored / all | Amplitude MAE, anchored / all | p (latency / amplitude) |
  |---|---|---|---|---|
  | 2 ms | 255 | 0.415 / 0.411 ms | 8.48 / 8.23 % | 0.006 / 4e-6 |
  | 5 ms | 200 | 1.754 / 1.748 ms | 18.75 / 18.18 % | 0.003 / 6e-4 |

  `"selected"` is much worse on the same runs: 0.532 ms and 24.3 % at 2 ms.

  On REQ-153 (50 two-component channels, seeds 400–449), the regularized latency MAE is
  1.828 ms (all), 1.858 ms (anchored) and 1.902 ms (selected); regularized beats free on
  100 % of channels in all three.

  On REQ-150 at SNR 3 with an N2 added (20 runs, seeds 500–519), all and anchored give the
  same numbers. Against selected:

  | Jitter | Latency MAE, all and anchored / selected | Amplitude MAE, all and anchored / selected |
  |---|---|---|
  | 2 ms | 0.88 / 0.87 sample | 7.5 / 12.1 % |
  | 5 ms | 0.96 / 0.95 sample | 8.8 / 8.5 % |

  `"anchored"` is also 30–45 % faster than `"all"`. It also makes the trials and
  `jitter_sd_null` use the same procedure, as §5.7 asks: the null fit already held the other
  components at their channel values. This default is a trade-off (robustness
  and speed on real data against about 1–3 % of simulated MAE), so it is flagged for the
  owner. `"all"` and `"selected"` stay available.
- REQ-156 as written (F10). The six D14 subjects run with the ccep preset pipeline of
  REQ-113 (template artifact removal, adjusted CAR, baseline, `evoparam.fit` with existence
  test), all N1 channels, 10 trials per site. Results with the default model:

  | Subject | Time | Median trusted fraction |
  |---|---|---|
  | UMCU01 | 0.71 s | 0.9 |
  | UMCU02 | 0.77 s | 0.9 |
  | UMCU03 | 1.03 s | 0.95 |
  | UMCU04 | 0.36 s | no significant N1 channel |
  | UMCU05 | 0.72 s | 0.9 |
  | UMCU06 | 1.25 s | 0.9 |

  The pooled median is 0.9. With `"all"` the times are 1.07–2.28 s and the pooled median is
  0.7. The test is `test_trials_ccep.py` (network).
- Spec options (Q6, Q9). `fit_trials` gains `ridge_cv` and `null_rule`. The spec's procedure
  is available for each, and the default is the one measured as better.
  - `ridge_cv` (Q6). `"loto"` (spec §5.7, leave-one-trial-out) is now the default. The
    centre of the penalty is the channel model refitted on the other trials, and the trial
    is scored on samples its fit did not see. `"split"` (the previous default) stays
    available. On REQ-153 with 50 single-component channels (seeds 300–349), regularized
    beats free on 100 % (loto) vs 98 % (split) of channels; mean latency MAE is 1.851 vs
    1.852 ms. On 50 two-component channels: 100 % for both, 1.828 ms for both. `split` is
    6–10 % faster, which is not a gain worth a divergence.
  - `null_rule` (Q9). `"segments"` (default) uses five baseline segments per trial and
    refuses realignment above max(1.2, F-quantile). `"spec"` uses one segment and the 1.2 x
    rule. On REQ-151 with 200 jitter-free runs of 40 trials (seeds 100–299):
    - ratio `jitter_sd_null / jitter_sd` below 0.8 in 7 runs (segments) vs 15 runs (spec);
    - realignment refused in 99.0 % vs 86.5 % of runs (REQ-151 asks for at least 95 %).

    With 200 trials per run (100 runs, seeds 1000–1099), both rules pass in every run: no
    ratio below 0.8, and 100 % refused. `"segments"` is kept as the default.
- REQ-150 as written (F36). At per-trial SNR 2 the REQ is out of reach for every width.
  - Even an efficient unbiased estimator cannot have latency MAE <= 1 sample and amplitude
    relative MAE <= 15 % together. This uses the Cramér-Rao bound for white noise and a
    Gaussian with a per-trial constant and slope.
  - The reason: the latency SD grows as sqrt(width) and the amplitude SD falls as
    1/sqrt(width), so their product does not depend on the width.
  - With CV 30 % amplitudes (E[1/a] = 1.09 for log-normal, 1.16 for normal), the best width
    (FWHM 11.35 samples) misses by 1.6 % (log-normal) and 7.8 % (normal). This is
    `test_req150_cramer_rao_bound`.

  Measured with `shift_scale` at 1 kHz, SNR 2, six runs per cell. Medians, with the
  [min, max] latency range at a jitter of 1 ms:

  | FWHM (samples) | Latency MAE (samples), 1 / 2 / 5 ms | Amplitude MAE, 1 / 2 / 5 ms | Raw `jitter_sd` / truth, 1 / 2 / 5 ms |
  |---|---|---|---|
  | 8 | 0.90 [0.70, 1.28] / 1.01 / 1.61 | 18.0 / 17.7 / 27.9 % | 1.52 / 1.20 / 0.99 |
  | 10 | 1.16 / 1.02 / 1.23 | 16.2 / 15.8 / 21.1 % | 1.96 / 1.28 / 0.99 |
  | 11 | 1.16 / 1.05 / 1.23 | 15.7 / 15.1 / 19.4 % | 1.91 / 1.27 / 0.99 |
  | 12 | 1.16 / 1.21 / 1.22 | 15.0 / 14.6 / 17.7 % | 1.90 / 1.27 / 0.99 |
  | 14 | 1.17 / 1.21 / 1.24 | 14.1 / 13.7 / 14.7 % | 2.00 / 1.26 / 1.01 |
  | 20 | 1.33 / 1.38 / 1.36 | 12.3 / 12.1 / 11.4 % | 2.15 / 1.37 / 1.02 |

  The jitter part also fails at 1 ms: 1 ms is one sample here, and the per-trial latency
  error is larger. The test as written is `test_req150_as_written` (strict xfail, FWHM 11
  samples); the SNR-3 test is kept.
- REQ-151 as written (F37), in every run. With 40 trials per run it fails in a few runs
  (seed 104: 0.768; 7 of 200 runs below 0.8, smallest 0.756), because the SD of 40 shifts
  varies by about 11 % from run to run. This is the strict xfail
  `test_req151_as_written_40_trials`. With 200 trials per run it holds in every run under
  both null rules (`test_req151_as_written`, 20 runs, seeds 2000–2019, not used for any
  choice).
- REQ-155 type-I error (F38). Tested as written: <= 0.06 with no added tolerance, on 1000
  runs of 100 trials (binomial SD 0.007). Measured rate: 0.051. The runs are the channels
  of ten TrialSets (one fit_trials call each) and share the channel model of one pilot run.
  The power part keeps its own test.
- REQ-158 (F17).
  - The `trend` covariate now reports a t-test p-value of the slope and the number of
    trials.
  - `test_req158_direction_on_simulated_1hz_trains` uses simulated 1 Hz trains of 30 pulses
    with a declining N1 (`simulate_ccep`, gain 1 - 0.01 x index, seeds 10–12). The test
    asserts that the slope is negative at >= 80 % of the significant sites (at least 10);
    it passes.
  - REQ-158 as written cannot be run: spec §7 has no dataset of repeated single pulses at
    1 Hz with an N1.
    - The §7.3 CCEP datasets (ds004080, ds003708, ds004977) use about 0.2 Hz.
    - Localize-MI (1 Hz; Mikulan et al. 2020, Sci Data, doi:10.1038/s41597-020-0467-x)
      keeps epochs from -250 to +10 ms only, so it has no N1.
    - The 1 Hz study cited for train effects (Cornblath et al. 2023, Epilepsia,
      doi:10.1111/epi.17528; 30 pulses at 1 Hz) reports infrequent trends of both signs
      (significant N1 trends in 0.84 % of CCEPs per patient on average), not a decline.
    - The hosts of Localize-MI (GIN, EBRAINS) and of that paper's data were not reachable
      from the build machine.
  - On the six ds004080 runs (10 pulses of one pair, about 0.2 Hz, trial order), 4 of 68
    sites have a significant slope (the rate expected by chance), and 1 of the 4 declines.
  - `test_req158_as_written` is a strict xfail (not run). An owner decision is needed: a
    1 Hz dataset with an N1 and a published direction.

### Architecture, packaging and test infrastructure (audit F3, F4, F43, F45–F49, F60–F63, F66, F67)

- Inherited code removed (§4.4, §4.5, §12 rule 1). The old engine and its modules are
  deleted: `objs`, `param`, `plts`, `analysis`, `core`, `bands`, `utils/{data, params, io,
  reports, download, debug}` and `data/{data, conversions}`, with their tests. The package
  no longer exports `evoparam` (the class), `evoparamGroup`, `fit_evoparam_3d` or `Bands`.
  `core/modutils.py` is kept as `utils/_modutils.py`, and the signal plots are ported to
  `evoparam.plots` (`plot_signals`, ...). `ComponentWindows` is now in
  `evoparam.models.windows`. LICENSE and NOTICE keep the attribution. `import evoparam`
  loads only numpy and scipy: before this change, scikit-learn and pandas were loaded
  through `NotFittedError`, which is now built lazily.
- Import graph (REQ-296, decisions D3–D5). The test checks the resolved table of
  `docs/spec/DECISIONS.md` with no transitional edges. It finds cycles and top-level
  optional imports, runs a mutation check, and checks that the light core does not pull
  in the removed modules. `docs/spec/DECISIONS.md` records the D1–D16 resolutions and
  the import table.
- Stimulation site (F60, F61). The reviewed rule of §5.2 method 5 is available as
  `identify_stim_site(..., method="spec")` (and `detect_pulses(site_method="spec")`).
  `SPEC.md` §4.2 and §5.2 are back to the reviewed text. The default ("evidence") is kept
  as a measured improvement. Per pulse it scores 0.980 against 0.61 for the reviewed rule,
  and 0.995 against 0.66 after the vote. On the 12 held-out runs it scores 0.979 / 1.000.
  The numbers are in DECISIONS.md.
- Extras (F4). `all` is the union of every extra except `gpu`, `bayes` and `docs` (§9.2).
  It now includes `mne`, `bids`, `nwb`, `neo`, `edf`, `fast`, `stats`, `validate`,
  `stream`, `tests` and the others. The `pynwb` floor is the spec's 2.6. `tomli` is added
  for Python 3.10. `test_dependencies.py` checks that every package the tests skip
  without is in `all` (or in `bayes`, which the local runner installs).
- Local runner (F3, D16). `tools/run_all_tests.py` runs the whole suite on a chosen
  Python, with optional venv creation, with the network, slow and performance tests
  switched on. It writes a JUnit report, `summary.md` and `summary.json`, which name the
  tests skipped for a missing package. The workflow keeps its matrix, runs the REQ-113 and
  REQ-114 tests in every build (REQ-017), and defines a weekly or on-demand
  `network-slow-perf` job that runs the same thing.
- BIDS validator (F62, REQ-315). The reference invocation as written (`npx bids-validator@2
  --datasetTypes derivative`) is a network test. It is an expected failure while npm has
  no 2.x release: on 2026-10-09 the latest was 1.15.0, giving ETARGET. The REQ-342 checks
  pass `--datasetTypes derivative` for derivatives.
- Byte-identical outputs (F43, REQ-345). When `SOURCE_DATE_EPOCH` is set, creation times
  are that instant and step durations are 0. The CLI-versus-API tests now compare every
  file byte for byte, with no volatile fields removed.
- Settings record (F45, REQ-331). The preset is recorded as `{"name", "version"}` and a
  wrong version raises. `_overrides` lists only the keys the user set. A seed is drawn
  and recorded when `random_state` is None, so `fit(data, **result.settings)` gives the
  same result.
- Component table (F46, REQ-332, REQ-006). Each reported latency, amplitude, width,
  onset, offset and area has `_ci_low` and `_ci_high` columns in the same row. The
  documented column list is `docs/reference/tables.rst`, and `test_tidy.py` checks the
  tables against it.
- GroundTruth (F63). `GroundTruth.save` and `GroundTruth.load` use the `.evoparam`
  container (`register_type` adds types defined outside `evoparam.data`).
- Warnings (REQ-005). `evoparam.errors.warn` warns and logs. Every warning of the
  package goes through it. The call in `io/nwb.py` was converted when the NWB branch was
  merged (its strict xfail is removed), with the warnings that `evoparam.stream`,
  `io/bids.py`, `io/derivatives.py` and `reproduce` brought. The `evoparam` logger has a
  `NullHandler` only.
- `evoparam.fit` runs the existence test by default (REQ-002). On an Evoked it uses the
  weaker null and warns. If the test cannot run, the columns are NaN and a warning says
  so. `existence=False` skips it. A preset without a baseline uses the pre-stimulus
  samples. A baseline with fewer than 2 samples now raises `DataError` instead of a numpy
  `ValueError`. `evoparam.read` is exported.
- Existence intervals (REQ-006, REQ-007). The bootstrap over trials gives intervals for
  the onset and the offset as well as the duration. `response_duration` reports them
  with the p-value. `test_response` records a drawn seed when `random_state` is None.
  `reproduce` regenerates an `ArtifactResult`.
- Product-level tests of §3.1 (F47), each as written:
  - `test_one_model.py` (REQ-001, REQ-002): eleven presets on one simulated waveform
    each. All give status ok, with the same types and columns. The five input kinds
    pass. TEP is simulated at 1 kHz, because a 5 kHz TEP fit took 68 s against 1.6 s.
  - `test_polarity.py` (REQ-003): on sign-flipped data the largest difference is
    0.0 (CCEP and ERP) for fits, trial fits and the existence test. It also has a static
    sign-rule scan.
  - `test_no_deletion.py` (REQ-004): an AST scan of `evoparam/models` with seven
    mutation forms. `min_amplitude` (default 0) is the only size rule.
  - `test_status.py` (REQ-005).
  - `test_uncertainty_columns.py` (REQ-006).
  - `test_reproduce.py` (REQ-007): GroupFit, ResponseTest, ArtifactResult and Evoked
    are identical when regenerated.
  - `test_methods_text.py` (REQ-008).
  - `test_units.py` (REQ-009).
  - `test_core_only.py` (REQ-010).
  - `test_dependencies.py` (REQ-011: GPL deny list, licence scan).
  - `test_vocabulary.py` (REQ-012, with the word list in CONTRIBUTING.md).
  - `test_determinism.py` (REQ-013, including `n_jobs=2` against 1).
  - `test_research_use.py` (REQ-015).
  - `test_no_network.py` (REQ-016): 132 modules and the CLI and API chain ran with
    sockets refused, with 0 connection attempts.
  - `test_release_gate.py`: REQ-017; REQ-018 with `tools/check_traceability.py` (49
    addressed gaps traced; 100 of the 143 test files named in §14 exist); and REQ-019
    with `RELEASING.md`, the `Private :: Do Not Upload` classifier and a gated
    `make publish`.
  - REQ-014 is REQ-271 (`presets/test_presets.py`).
- Still not met as written (documented xfails, measured):
  - REQ-010: `detect_events`, `remove_artifact`, `fit` (the GroupFit index), `fit_trials`
    and saving a GroupFit need pandas.
  - REQ-006: the existence onset of an Evoked has no interval.
  - REQ-007: `fit_trials` records no provenance step, and `tf_fit` neither records a seed
    nor replays identically.
  - REQ-009: fit tables give the amplitude unit as "data".
  - REQ-015: there was no stream module yet on this branch. Since the merge of S18
    (`evoparam.stream`, section "Build step S18" below), `test_req015_stream_outputs`
    checks its statement.
  - REQ-315: npm has no `bids-validator@2`.

### Specification gaps: artifact, prep and events

- REQ-060 (polynomial). The published method fits the polynomial over the whole sweep and
  subtracts it everywhere, which moves the baseline: -9.1 % of the baseline level at
  2048 Hz (-18.3 % at 1 kHz, -4.0 % at 4800 Hz, -0.75 % at 25 kHz), above the 2 % of
  REQ-060. `polynomial` now has `poly_extent="local"` (default in `remove_artifact`): the
  correction is applied from the window start to the end of the artifact (first run of
  samples where the fit is back within 3 robust SD of its baseline level, after the window
  and the last railed sample), with half-cosine ramps; the baseline change is 0.0 %, and all
  methods now meet 2 % with no exception. On the overlay benchmark the response after the
  artifact is kept: RRMSE on 8-50 ms, global against local, 0.994 / 0.054 (1 kHz, onset
  12 ms), 0.993 / 0.224 (2048 Hz), 0.992 / 0.587 (4800 Hz); 0.030, 0.019, 0.013 for an
  onset at 20 ms. The published method stays as `poly_extent="global"` (REQ-065 runs it as
  written; `test_req060_polynomial_global_extent_as_published` is a strict xfail with its
  measured values); with the local extent the REQ-065 recording is also recovered, with a
  smaller error against the truth (9.34 against 9.74).
- REQ-061, second part (`tests/artifact/test_fitted_bias.py`). Template and learned
  cleaning, onsets 8, 12 and 20 ms, fit from b: median latency error at most 0.013 ms and
  amplitude error at most 0.10 % (criteria 0.5 ms and 5 %); a component found without the
  artifact but lost after cleaning counts as an infinite error. Plain blanking: 153 %, lost,
  lost, 42 % and 15 % for onsets 6 to 10 ms; at 10 ms below the >= 20 % of the
  specification (strict xfail).
- REQ-064 on real artifact templates (`tests/artifact/test_subspace_real.py`, network):
  TMS-EEG ds002094 (20 subjects) and intracranial SPES ds004080 (23 runs at 2048 Hz).
  SIR improvement: TMS median 1.6 dB (0.2 to 4.1), SPES 9.3 dB (5.0 to 18.4), not 30 dB
  (strict xfails). Cause: the specification's rule for the number of directions (95 % of
  the whitened energy, at most 6) keeps one; on ds002094 sub-283, 6, 10 and 20 directions
  give 17.6, 26.5 and 57.1 dB.
- REQ-081, simulation part: references at 0 to 90 % (0, 10, ..., 80, 84, 90 %), 10 seeds:
  median false negatives and false positives 0 at every level (at most 2 false positives on
  a single seed); the test asserts the specification's bounds as written.
- `reference_sensitivity` (§5.4) runs the existence test by default (`detector=
  "existence"`); registration by the package root, because `prep` may not import
  `existence` (docs/spec/DECISIONS.md). The interim detector stays as
  `detector="threshold"`.
- Specified methods that were missing: `TFFit.to_mne`, `HierarchicalFit.to_arviz`
  (posterior draws of the Bayesian engine, or draws of the normal approximation for the
  empirical-Bayes engine, said in `attrs`), `TrialSet.to_xarray` (extra `xarray`),
  `list_shapes`, `list_backgrounds` (`tests/data/test_exports.py`).
- `detect_pulses`, method 1 (Q3). The specification's procedure (median over all channels,
  first crossing, template pass with rho_min) is now `procedure="spec"`. Measured on the 25
  ds004080 runs: recall 0.445 and precision 1.000 in 57 s, onsets about 1 sample early,
  against 0.995, 0.990 in 2.4 s and 0.01 sample (held-out runs) for the default, which is
  kept. Threshold pass alone: 0.930 / 0.765 (specification) against 1.000 / 0.760.
- Samples corrected by subtraction (Q4). The specification masks them; they are kept as
  measurements by default. With a fit starting inside the window (onset 8 ms), median
  latency error 0.09 ms against 0.64 ms with them masked, amplitude 1.2 % against 17.8 %
  (onset 12 ms: 0.010 ms against 1.20 ms); masked is better only at 10 ms (0.43 against
  1.83 ms). Identical when the fit starts at b (the ccep preset). `mask_subtracted=True`
  gives the specification's behaviour.
- `tail` (Q5): the default now subtracts r(t), as written (`tail_part="full"`). The former
  default, r(t) - r(t_end), is kept as `tail_part="joined"`: it is better for a response
  inside the tail window (onset 20 ms: amplitude error 8.7 % against 48.5 %, latency
  1.24 ms against 8.35 ms) and worse after it (onset 50 ms: 9.6 % against 0 %; residual
  2.20 against 1.42).
- REQ-044 on the Mayo protocol (`tests/events/test_trains.py`, network, no environment
  variable besides EVOPARAM_NETWORK_TESTS): the events.tsv of 20 Mayo runs (ds003708,
  ds004457, ds004696, ds004977 at its uncensored site); median pulses per site between
  10 and 12 and intervals of 2.5 to 6 s on every run but ds004696 sub-07 (median 13,
  strict xfail).
- REQ-042 on the Mayo sets: agreement 0.67 -> 0.99 (160 of 162 detected pulses; 0.99 on
  the 2048 Hz runs, 1.0 on ds004977), accuracy 1.0 for `site_confidence >= 0.5` on 41 % of
  the pulses. Two kinds of evidence were added to `identify_stim_site` (nothing removed):
  a channel that reaches, within 5 ms of the onset, the largest absolute value of the
  recording (the amplifier's range, 34.8 mV on these sets), counted as held, when at least
  two channels reach it; and a silent pair (ds004977: the stimulating contacts are switched
  to the stimulator and carry 3 to 5 uV of noise against 260 to 340 uV), next to the
  largest artifact. ds004080 unchanged (0.995, accuracy 0.994 on 80 %). Held-out check on
  101 pulses at sites the test does not use: 0.93 (0.49 before). Reference pulses in
  censored (all-zero) data are not counted (none in the segments used).
- D9: `detect_pulses` reports the physical onset; the constant trigger delay of ds004080
  (0.97 ms) is estimated on the even runs and checked on the odd ones (within 1 sample);
  the reverse fold gives the same delay (0.972 ms; median error 0.02 sample).

### Statistics and time-frequency against the reviewed spec (audit F13-F15, F51-F53, F68, F70, D11, D12)

- REQ-234 (F14), `trials_needed(extrapolation=...)`: `"sqrt_n"` is the 1/sqrt(n) law of
  the spec (§5.11 method 5), kept; `"power"` fits `sd(m) = a m^b` on the resampled counts
  `m >= N/4` (log-log least squares, `b` clipped to [-1/2, 0]: a latency's SD is bounded by
  the window while the average is noisy and approaches 1/sqrt(n) from below); `"auto"`
  (default) uses `"power"` for `peak_latency` and `fractional_area_latency` and
  `"sqrt_n"` for every other score and for callables. The choice was made on development
  seeds (7 to 11; pink, white and AR(1) noise; per-trial SNR 0.5, 1 and 2; 30 channels of
  20 trials, SD at 80 trials over 300 blocks), not on the test's seed 41. Median ratio
  predicted / true SD at 80 trials from 20, development seeds: peak latency 0.55-0.76
  (1/sqrt(n)) against 0.84-1.18 (power); fractional area latency 0.75-0.89 against
  0.83-1.08; fractional peak latency 0.75-1.34 against 1.11-1.59 (power worse: kept on
  1/sqrt(n)); peak amplitude 0.93-1.08 against 0.96-1.22 (kept on 1/sqrt(n)). Also tried
  and rejected: bootstrap at the target count (fractional area latency 0.79-0.87), a
  Wiener-filtered centre (latencies 1.34-1.61), a centre-noise extrapolation, and a power
  law on the effective noise level of bootstrap averages (fractional area latency 0.93-1.03 at SNR 1
  but 1.59 at SNR 0.5, peak latency up to 1.78). Test seed 41 (REQ-234 as written, median
  within 15 %): mean amplitude 1.06 and peak amplitude 0.95 (60 channels, unchanged), peak
  latency 0.937 (met; 0.701 with 1/sqrt(n)), fractional peak latency 1.016 (met), area
  1.003 (met), fractional area latency 0.795 (not met, xfail with the value; 0.773 with
  1/sqrt(n)). Cause measured: with the true average as the centre the bootstrap at 80
  trials gives 0.95; around the noisy 20-trial average the rectified noise spreads the area
  over the window and the 50 % point moves less (0.84-0.86).
- REQ-231 (F13): the Kiesel et al. (2008) worked example stays an xfail. On 2026-10-09 the
  network policy of the build environment refused the author's copy (University of
  Freiburg), the publisher, doi.org, OpenAlex, PubMed Central and Europe PMC.
- REQ-233, ERP CORE part (F15, F70, S14 done-when): still not run. The licence
  (CC-BY-4.0) is in the registry; the OSF hosts (osf.io, files.osf.io, api.osf.io) are
  refused by the network policy of the build environment (HTTP 403 at the proxy on
  2026-10-09), so the N170 epochs could not be downloaded. The test is unchanged apart from
  its docstring (licence recorded, OSF node pfde9).
- Time-frequency, correction across channels (F52): `channel_correction` takes the
  corrections of §5.6, `"fdr_bh"` (default, was `"none"`), `"holm"`, `"bonferroni"`,
  `"none"`. Each map row (channel x condition) is a family of pixel tests whose p-value is
  its smallest adjusted pixel p-value (Simes for `fdr`, the permutation family-wise value
  for `cluster` and `tfce`); `fdr_bh` selects rows by Benjamini-Hochberg and tests the
  pixels of the R selected rows of m at `R alpha / m` (Benjamini and Bogomolov 2014);
  `holm` gives each rejected row its Holm level; `bonferroni` is the former rule (alpha /
  rows). The rows' p-values and levels are in `settings["row_p_value"]` and
  `settings["row_alpha"]`. The rows' levels need every row first: the split halves of the
  zones are computed in a second pass over the rows with significant pixels (one more
  transform of those rows). Null AR(1) channels (8 channels, 30 trials, 20 frequencies):
  runs with any significant pixel, test seeds (30 runs), `fdr_bh` 3 (0.10), `holm` 3,
  `bonferroni` 3, `none` 12 (0.40); 200 other runs (seeds 4000-4199), `fdr_bh` and
  `bonferroni` 7 (0.035). Under the complete null this is the false-discovery rate over
  channels, target 0.05. REQ-218 benchmark (morse, `fdr`), run alternately on the shared
  machine (load average 4 to 9 on 4 CPUs): previous version 162 s and 194 s, this version
  151 s and 223 s; no measurable cost, and the 60 s target is met by neither under that
  load (not measured on an idle machine).
- Time-frequency, CP rank (F51): chosen on split halves as §5.10 says
  (`cp_stability="split_half"`, default). `tf_fit` keeps, for each row with significant
  pixels, the maps of 20 random halves of its trials (float32, cropped to the box of the
  significant pixels: `maps["zone_halves_rows"]`); the frequency x time x row tensors of
  the two halves are factorised and their matched parts compared (Tucker congruence
  >= 0.8). Without those maps (a TFFit saved before), halves of the rows; with fewer than
  two rows, the trial halves of the group map (as for NMF). `cp_stability="restarts"`
  (the former rule) and `"rows"` stay in `tf_zones`. Measured on the same TFFits
  (6 channels, 30 trials, AR(1) noise, 5 runs per case, 20 frequencies; right = as many
  positive zones as bursts, each within 10 % in frequency and 20 ms in time), right
  answers for split halves of the trials / halves of the rows / restarts: one burst 5/5,
  5/5, 5/5; two bursts in every channel 3/5, 4/5, 4/5; two bursts in separate channels
  3/5, 0/5, 2/5; null 5/5, 5/5, 5/5; total 16/20, 14/20, 16/20. Mean time per call (shared
  machine): 87-126 s, 38-222 s, 147-163 s. The former rule is not measurably better, so
  the spec's rule is the default.
- Published zones (F53, F68, S13 done-when): the article [LIT-tf-4] was read in full
  (Brinyark et al. 2026, Clin Neurophysiol 186:2111855, doi:10.1016/j.clinph.2026.2111855,
  CC BY 4.0; LITERATURE.md entry now VERIFIED). The registry holds its eleven zones: the
  verified delayed high-gamma zone (alias `brinyark2026_GhN2`) and, as `unverified` with
  the page reference (§2.4 and Fig. 2, p. 3) as §5.10 requires, TN1, TN2, AN1, AN2, BN1,
  BN2, GlN1, GlN2, GhN1 (bands theta 4-8, alpha 8-13, beta 13-30, low gamma 30-50, high
  gamma 50-250 Hz x N1 15-50 ms, N2 50-500 ms) and WR (4-250 Hz x 15-500 ms). Boundary
  convention (not stated in the article; evoparam's reading): 50 ms belongs to N2 and each
  band edge to the higher band (half-open intervals in `intervals`), so that the ten zones
  tile WR. The article's method is in the registry and set by these zones: Morse wavelets
  (gamma 3, time-bandwidth 30), new statistic `stats="baseline_normal"` (trial-average dB
  power of each pixel against a normal fitted to the -450 to -150 ms baseline of its
  frequency, two-sided, Benjamini-Hochberg over the pixels of the 15 to 900 ms window), no
  correction across channels; the zone metric `fraction_significant` is the published
  significance fraction (significant pixels / pixels of the zone).
  `volume_conduction_weight(distance_mm)` gives the weights of §2.5 (0 below 10 mm, 1
  above 20 mm, `(d^2 - 100) / 300` between). The delayed zone keeps the settings of the
  spec (Hays et al. 2024 transform and baseline) and gains `channel_correction="none"`
  (its behaviour before the new default; the REQ-216 numbers are unchanged).
- REQ-213, strict reading (D11): new xfail test, a burst adding 20 % to the local 80 Hz
  power (amplitude 1.32) is recovered in 0 of 20 runs (target 18 of 20); the reading
  adopted (20 % of the signal power) stays met (20 of 20).
- §4.3.4 and REQ-211 (D12): verified; both the power maps (`total`, `evoked_footprint`,
  `residual`, `cross`) and the baseline-normalised maps (`total_norm`, `footprint_norm`,
  `residual_norm`) are given; a new test checks the normalised maps against
  `10 log10(map / baseline mean of the total)`.

### Build step S18: online mode (`evoparam.stream`)

For research use only (not a medical device). The free package holds the API and the
reference minimal estimator of spec §13.1; no device driver, controller logic or safety
interlock. The tests follow the repository layout used by every step (`evoparam/tests/`):
the spec's `tests/test_stream.py` and `tests/test_stream_lsl.py` are
`evoparam/tests/stream/test_stream.py` and `evoparam/tests/stream/test_stream_lsl.py`.
The reference machine and the benchmark numbers are in `benchmarks/README.md`.

- `evoparam.stream.StreamFitter` with the signature of spec §10.6 (`push`, `push_pulse`,
  `reset`, `latency_stats` with `p50`, `p95`, `p99`, `max`), `PulseEstimate` with the fields
  of §10.6, `replay(trialset_or_raw, fitter, chunk_size=32)`, `LSLInlet(name=None,
  stype="EEG", *, max_buflen=360)` and `LSLOutlet` (extra `stream`: `pylsl>=1.16`).
  `StreamFitter.existence(final=False)` returns the running existence test per channel.
  Parameters the specification needs but does not name: `saturation` (the level of the
  `"saturated"` status), `existence_alpha` (0.05), `existence_max_pulses` (100, the
  information horizon of the alpha-spending rule), `existence_looks` (`"pulse"`: a look
  after every pulse; None: only through `existence()`), `random_state`. Fixed: the 20 ms
  budget of the `"late"` status (REQ-324) and 1 000 sign flips (the default of
  `test_response`). `push` takes `[n_samples, n_channels]` (the layout of LSL chunks).
- Beyond the specification, for a measured gain in accuracy: `mode` of `StreamFitter` and
  `FixedShapeModel`. `"common"` is the method of §10.6 (one shift and one scale for the
  whole template); `"scale"` keeps one shift and gives each component its own scale; the
  default `"shift_scale"` gives each component its own shift and scale, within `max_shift`
  and half the distance to the nearest other component, as `fit_trials(mode="shift_scale")`.
  With a single component the three modes are the same model and give bit-identical
  estimates (tested). Measured on 200 simulated pulses with two components (N1 -40 uV at
  30 ms, N2 +30 uV at 120 ms, jitter SD 1.5 ms, amplitude CV 20 %; one jitter for both
  components / one per component), share of pulses within 1 % and one sample of
  `fit_trials` on both components: `"common"` 1.5 % / 0.5 % (largest differences 92 % and
  8.8 samples / 117 % and 12.7 samples), `"scale"` 41 % / 17 % (1.4 % and 9.0 samples /
  2.9 % and 13.3 samples), `"shift_scale"` 100 % / 100 % (largest differences 2.3e-4
  relative and 0.24 sample / 1.0e-4 and 0.31 sample). Latency (benchmark below, two
  components): push p99 10.7 ms (`"common"`), 10.3 ms (`"scale"`) and 11.5 ms
  (`"shift_scale"`), p50 5.7, 6.3 and 7.5 ms, all within the 20 ms budget.
  `"shift_scale"` is therefore the default; `mode="common"` keeps the method of §10.6,
  whose two-component agreement stays a documented strict xfail.
- REQ-349: the stream module computes no number of its own; the estimators are public
  functions of the analysis modules, and `evoparam/tests/reports/test_layering.py` now scans
  `evoparam/stream` too:
  - `evoparam.trials.FixedShapeModel`: the fixed-shape method of §10.6. Per pulse and
    channel, a common shift on a one-sample grid over `+/- max_shift`, refined by parabolic
    interpolation of the explained sum of squares, and a linear scale by closed-form least
    squares (the template evaluated at the refined shift), with the linear terms of the
    channel background re-estimated (time constants frozen). The normal matrices of every
    grid shift are inverted once. In the `"shift_scale"` mode the per-component shifts
    start from the best common shift and move by coordinate ascent on the grid (one
    component at a time, the others fixed, its gain in explained sum of squares by
    partitioned least squares from precomputed products of every pair of shifted
    components) until no shift changes, then each is refined by parabolic interpolation;
    the scales are then solved jointly at the refined shifts. The artifact window is
    replaced by the core `interpolate` method of §5.3 (linear, 3 anchor samples on each
    side) before the fit and the projections; the specification's "one-sided" interpolate
    is not a method of §5.3, so the §5.3 method is used (both sides are available once a
    window is complete).
  - `evoparam.existence.OnlineExistence`, `SequentialSignFlip`, `sign_flip_test`,
    `alpha_spending`. Per pulse, the projection of the window on the unit-norm unshifted
    template; `existence_z` = (projection - mean of the baseline projections so far) / their
    SD (the baseline is the segment of window length that ends before the window or the
    artifact window; NaN until two baselines). Running test: one-sided sign-flip test of the
    one-sample t of (window projection - baseline projection), with the Lan-DeMets
    O'Brien-Fleming spending function applied to the permutation trajectories themselves
    (the observed values and the sign-flipped copies are exchangeable; at information
    fraction t at most floor(alpha(t) (P + 1)) trajectories may have crossed, the largest
    live ones cross): the error over any number of looks is at most
    floor(alpha (P + 1)) / (P + 1) <= alpha.
  - `evoparam.events.OnlinePulseDetector`: the threshold step of `detect_pulses`, causal
    (same score, robust z from the first second, onset = first crossing, `min_interval`
    0.5 s), independent of the chunk sizes.
- REQ-324 / REQ-347, agreement: on 200 simulated pulses (two channels, single-component
  templates of the ccep preset), the streamed amplitude agrees with
  `fit_trials(mode="shift_scale", max_shift=0.005)` within 6.6e-5 relative and the latency
  within 0.0066 sample (limits 1 % and one sample). Chunk sizes 1, 32 and 1 024 give
  bit-identical estimates. The running test evaluated once at the end gives the decisions
  and p-values of `sign_flip_test` on the same per-pulse values (8 channels, 4 with a
  response: 4 found, 0 false). With a two-component template the default mode agrees
  within 2.3e-4 relative and 0.31 sample on 200 pulses (both jitter cases, test passes);
  the specification's single shift and scale (`mode="common"`) stays a documented strict
  xfail with the numbers above.
- REQ-324 / REQ-347, latency (`benchmarks/stream_latency.py`, 1 000 simulated pulses, 64
  channels at 2 048 Hz, a 0.5 s window from 0 to 0.5 s with the artifact window
  interpolated, `max_shift` 5 ms, threshold detector, running test with a look after every
  pulse, `OMP_NUM_THREADS=1`; the reference machine of `benchmarks/README.md`, other jobs
  running): wall time of the `push` call that delivers the last sample of a window, one
  component: p50 4.8 ms, p95 6.6 ms, p99 8.6 ms, max 11.3 ms with chunks of 32 samples
  (p99 10.4 ms with chunks of 1), compute time per pulse p99 8.1 ms; two components (N1 and
  N2 jittered independently), default mode: p50 7.5 ms, p95 10.0 ms, p99 11.5 ms, max
  17.9 ms. Budget: p99 < 20 ms. The timed test (one and two components) runs with
  `EVOPARAM_PERF_TESTS=1`; `test_latency_stats` always runs. Two speed-ups, numerically
  equivalent: each component is evaluated only over the samples where, at some shift
  within `max_shift`, it exceeds 1e-16 of its peak (largest change of any estimate
  2.9e-16 relative; time to evaluate the components of 64 channels 3.9 -> 2.4 ms per
  pulse with two components), and in the coordinate ascent a component is evaluated again
  only when another one has moved (4.4 -> 3.2 steps per pulse).
- REQ-348: over 50 looks on null data, false-positive rate 0.051 (4 000 channels of
  Student t(3) values, P = 500; the same looks without the spending rule: 0.219) and 0.038
  (16 null streams of 64 channels through `StreamFitter`); limit alpha + 0.01 = 0.06.
- REQ-325: `evoparam/tests/stream/test_stream_lsl.py` (marker `lsl`, skipped without pylsl
  or liblsl) sends a synthetic signal through a local outlet, reads 10 pulses with
  `LSLInlet` and `StreamFitter` and checks the outlet metadata `intended_use = "research
  only; not a medical device"` (passed with pylsl 1.16.2 and 1.18.5 on liblsl 1.16.2). The
  outlet has one float32 channel per `PulseEstimate` field (strings as indices into the
  JSON metadata) and is stamped with the pulse time. `evoparam info` prints `intended_use`; the tutorial
  `tutorials/92_online_mapping_with_lsl.py` (simulated stream) states it, checked by
  `evoparam/tests/test_docs.py`.

### Fitting engine as written, and the N1 case as written

Audit items F1, F2, F9, F24-F31, F50, F64, F65, Q1, Q2 and owner decisions D8, D14
(`etapes/ecarts-spec.md`). Every REQ test below follows the REQ text (data, counts,
tolerances); the ones not met stay as xfails that state the measured value.

- REQ-371 subject list (D14). `validation/ds004080_subjects.txt` lists six subjects: the
  first five qualifying subjects other than sub-ccepAgeUMCU01 (UMCU02 run-021804, UMCU03
  run-031411, UMCU04 run-031148, UMCU05 run-031133, UMCU06 run-030854), then
  sub-ccepAgeUMCU01 itself (run-061233, its first qualifying run; run-021448 is the
  example of spec §1). Before: five subjects, UMCU01 included, UMCU06 left out.
  `tools/ds004080_subjects.py` builds the list so, `test_subject_list.py` checks it and
  `test_engine_ccep.py` reads it (each run's first pair of >= 10 trials).
- `vanblooijs2023` (F65, D8). The registry default is again the row of spec §5.6 as
  written (the most negative sample of 9-100 ms beyond 3.4 SD, no SD floor, baseline
  unspecified). The published code (mnl_ccepAge: 50 uV SD floor, local minimum by
  peakfinder with selectivity 20 uV, first sample excluded, search to 500 ms, refusal above
  3000 uV, baseline (-2, -0.1) s) is the new registry rule `vanblooijs2023_code`.
- The two readings disagree with each other: kappa 0.35 over the 942 channels of the 12
  runs used here (229 detections with the registry row, 60 with the code). The registry
  row's extra detections are mostly slow drifts peaking at 70-100 ms (75) and small N1s of
  20-50 ms under the code's 170 uV floor (44). For any decision rule M, min(kappa(M, paper),
  kappa(M, code)) is at most 0.30 on the 46 PT contacts of the example run, 0.32 to 0.78 on
  each REQ-371 run and 0.60 pooled over them (exhaustive search over the 2 x 2 tables): D8
  ("REQ-113 under both readings") cannot be met by any detector.
- REQ-113 to the letter (F1, D8). Decision = the preset's own: a negative component with
  latency 9-50 ms (the component the ccep preset labels N1), no threshold borrowed from the
  rule; all trials of the pair, the preset pipeline. Kappa with this change (origin/main in
  brackets):
  - registry row: 46 PT contacts of the example 0.51 (0.35), 18 rule N1s missed;
    REQ-371 runs UMCU02 0.47 (0.42), UMCU03 0.54 (0.54), UMCU04 0.31 (0.34), UMCU05 0.43
    (0.40), UMCU06 0.31 (0.38), UMCU01 run-061233 0.40 (0.47); pooled 0.45 (0.47);
  - published code: PT contacts 0.17 (0.26), 1 rule N1 missed; REQ-371 runs 0.24 (0.20),
    0.21 (0.21), 0.27 (0.32), 0.45 (0.39), 0.36 (0.20), 0.46 (0.56); pooled 0.37 (0.33).
  REQ-113 is not met under either reading (`test_req113_as_written`, xfail for both).
  The decision the earlier wording used (the trough of the fitted curve, judged by the
  code's amplitude criterion; not the REQ) stays as a regression test: pooled kappa 0.89
  over the REQ-371 runs (0.93), each run >= 0.71, none missed on the example.
- REQ-114 to the letter (F2): the N1 component's latency within 2 ms of the rule's peak on
  23 % of the channels where both find an N1 with the registry row (75 channels, example
  and REQ-371 runs) and 32 % with the code (28); read as the trough of the fitted curve,
  41 % and 82 %. Not met (xfail); REQ-273 not met with them.
- REQ-135: kappa of the baseline permutation test with the registry row 0.687, with the code
  0.399 (xfail for both); with the registry row and the code's 50 uV floor 0.763, kept as a
  regression floor.
- REQ-104 (F24). With the candidates limited to the extrema above `detection_threshold`,
  as §5.5.3 step 3 says, the acceptance decisions for thresholds 1, 2 and 3 agreed on 82 %
  of the suite's waveforms (49 of 60): a threshold of 3 drops candidates the criterion
  accepts. New setting `threshold_role` (default `"order"`): every extremum is a
  candidate, the threshold only orders them (the "affects only the order" of step 3);
  `"select"` keeps step 3's reading. Agreement now 100 of 100 (met). Cost: about 1.7 times
  the fitting time of the ds004080 runs (interleaved timing, 2 runs x 2 rounds: 230 s
  against 410 s), and, on the N1 decision, kappa 0.41 instead of 0.55 on the two timed
  runs (more components accepted).
- REQ-105, REQ-106, REQ-110, REQ-111, REQ-116, REQ-117, REQ-119 (F25-F31): tests as
  written (spec §7.2 designs, 100 waveforms per condition, 1 000 with
  `EVOPARAM_SLOW_TESTS=1`). Met: REQ-104, REQ-106 for every shape fitted with its own shape
  and for the Gaussian with the default pair, REQ-110 (warning iff the latency is before
  t0; recall of edge components), REQ-111 at 1 FWHM, REQ-116 Jacobian coverage, REQ-117
  (independent ERPLAB-definition implementation; the ERP CORE part needs
  `EVOPARAM_ERPCORE`), REQ-119 with `n_jobs` 2 and 4 (byte-identical tables). Not met
  (xfail with the values): REQ-105 recall 0.885 (precision 1.00); REQ-106 with the default
  pair for skewed Gaussian, gamma and biexponential truths (latency MAE 0.52, 0.76 and
  0.60 sample); REQ-111 at 0.5 FWHM (one component in 100 of 100 runs, never flagged
  `merged`: the merge of step 5 runs only after a second component is accepted).
  REQ-116 bootstrap and profile coverage: slow test (200 runs), not run here.
- The comparisons on the 5 development runs of ds004080 (UMCU07, 19, 28, 40, 49; not the
  REQ-371 runs, which are only reported) were measured with a rule against cancelling
  pairs of components that is not part of this change (it is handed over to the work on
  component selection); every other comparison below is on this change's engine.
- Ordered latencies (F50). `latency_order="parameterised"` (now the default): latencies
  ordered inside every optimisation, mu_{k+1} = mu_k + exp(v_k) (spec §5.5.3 step 5),
  with the chain rule in the Jacobian. `"sorted"` (the earlier behaviour) was not better:
  the same results on the simulation suites, N1 kappa 0.41/0.31 (registry/code) against
  0.40/0.33 on 5 development runs, the same time on ds004080 (414 s against 416 s,
  interleaved) and on `fit_group` of the REQ-115 design (85 s against 83 s), 0.92 s against
  0.75 s per single waveform (5 waveforms).
- Noise model of step 1 (F64). `noise_model="neff"`: residuals scaled by sigma_n (no AR
  whitening) and n_eff = n (1 - rho) / (1 + rho) in the criterion and the covariance;
  named in the methods text. REQ-107 as written: <= 5 % false components on AR(1)
  rho = 0.9 noise with "neff" and with the default "ar", >= 30 % with "white" (met).
- Soft priors (Q1), measured against the spec procedure (no priors: background_prior = 0,
  amplitude_prior = 0, no edge limit, no background time-constant limit): kept.
  Layer-2 grid (40 cells of the weekly design, 5 replicates): F1 0.760 with against 0.743
  without (better in 7 cells, worse in 1), latency MAE 1.46 against 1.52 samples (better in
  26 of 38), mean amplitude bias -0.03 against -433 (diverging fits without them); the
  simulation suites of REQ-104 to REQ-112 unchanged; N1 kappa 0.40/0.33 against 0.37/0.32.
- Candidate search (Q2), measured against step 4 as written (`candidate_search="spec"`:
  candidates strongest first, each from n_starts starts, the first accepted kept): the
  earlier search (`"beam"`: the first three qualifying candidates compared, split
  proposals) is kept as the default. On CCEP-like simulations (40 waveforms: N1, a second
  component, a recovery tail) the step-4 search inflated an amplitude beyond twice the
  largest true one in 10 fits and lost the N1 in 8, the beam in 1 and 1; N1 kappa 0.33/0.27
  against 0.40/0.33 (5 development runs); about 9 times slower (layer-2 cells 125 s against
  14 s). On the layer-2 grid the step-4 search has the better F1 (0.776 against 0.760,
  better in 9 cells, worse in 3) and fewer false components (0.08 against 0.10 per
  waveform): the choice trades that for robustness and speed and is left to the owner.
- Speed (F9, REQ-115). The test now has the REQ's 2 048 samples (the earlier one fitted a
  window that the data did not reach and raised) and a `fit_group` part (100 channels x
  2 048 samples, `n_jobs=1`). Measured (4 shared cores, load about 1): median 0.92 s per
  waveform against 50 ms, `fit_group` 85 s against 10 s; with `threshold_role="select"`
  and `latency_order="sorted"` (origin/main's behaviour) 0.42 s and 54 s. Not met (xfail).
- Engine fix: `fit_waveform_core` had its docstring swallowing the wrapper body, so the
  evaluation counter was never reset and `n_evaluations` never reported.
- Kept green with these defaults: REQ-112 on the ds004080 baselines (>= 95 % silent),
  REQ-170 (`test_hier_eb.py`, fixture 474 s), the layer-2 runner tests, the preset
  simulations (REQ-276, REQ-277) and the scikit-learn checks.

### Integration of the N1 work, S16, S17 and the tep validation

- REQ-171: the test as written stays as a documented xfail (5 of 6 datasets, 17 of 20 with
  `EVOPARAM_SLOW_TESTS=1`). Proposed wording, written after these measures and pending a
  decision: over the datasets, the RMSE of the EB latencies is below that of the channel
  fits (1.47 ms against 1.59 ms over the 20 datasets); `test_req171_shrunken_latencies_are_better_on_average`
  checks it.
- `test_schema_version`: the frozen TrialSet is compared to the factory with a relative
  tolerance of 1e-9 on the data (the last bits, and the data hash with them, differ from
  one CPU to another), then exactly on everything else.
- `test_sim_truth`: ERP CORE (CC-BY-4.0) and ds002094 (CC0) now have a licence in the
  registry; the refusal of real noise without an allowed licence is checked on Localize-MI.

### N1 engine change: speed of the whitening, selection criterion, REQ-170 and REQ-113

- Speed. The exact start rows of `Whitener` were whitened by a LAPACK triangular solve at
  every residual and Jacobian evaluation; each call woke OpenBLAS's threads, whose spinning
  took 3 to 4 cores per process: the fits of one simulated subject used 43 s of CPU for
  10 s of wall time (10 s for 10 s before), and parallel fits stalled (the REQ-170
  reference fixture, 1 000 channels in 4 processes: 918 s instead of 135 s). The inverse
  Cholesky factor of the start rows is now computed once, by plain loops
  (`_inverse_cholesky`, p <= 10), and applied with `einsum`: 9.4 s of CPU for the same
  subject, 136 s for the reference fixture. The ds004080 N1 fits are 5 to 10 times faster
  (run-021448: 60 s instead of 552 s).
- Selection criterion. The criterion that decides the number of components (additions,
  splits, merges, the best rejected change) now counts the innovation rows only; the
  start rows still take part in every fit, and in the comparison of optima of one
  structure (`fit_crit`). Their evidence rests on the stationary spread of a short AR
  model at the window start, and a chance deviation of the first samples was "explained"
  by a recovery tail plus a broad component cancelling it: on the hierarchical
  simulations, a spurious N1 at 11 ms with a FWHM of 83 ms on a channel whose first
  sample was 2.7 SD above the baseline (the conditional likelihood found no component).
- REQ-170 (cause: the exact likelihood; least squares alone changes little): AUC of
  `P(response)` 0.9506 on e8db3db, 0.9396 with the exact likelihood (Yule-Walker or least
  squares alike), 0.9495 with least squares and the conditional likelihood, 0.9509 now;
  reliability slope 0.997, 1.037, 0.991, 0.990. Met again, with the margin it had.
- REQ-171 is not met (5 of 6 datasets in the default run, 17 of 20 with
  `EVOPARAM_SLOW_TESTS=1`; e8db3db: 6 of 6, 19 of 20). The cause is the improvement of the
  per-channel latencies the test compares with: their RMSE over the 20 datasets is 1.59 ms
  now against 1.94 ms on e8db3db, and the EB latencies are also better (1.47 ms against
  1.74 ms), but the margin between the two is smaller. On the failing dataset (seed 105)
  the per-channel RMSE is 0.98 ms and the EB 1.05 ms; recomputing the shrinkage with the
  true RMSE of the channel fits as standard error gives 0.99 ms: no choice of standard
  error makes the shrinkage win there. With the minimum of the Wald and trial standard
  errors (the fit of the average is at least as efficient as the mean of single-trial
  estimates) 18 of 20 datasets are won but seed 105 is still lost; Wald errors alone 15
  of 20, trial means 11 of 20. Not changed here; the requirement or its test needs a
  decision (see the report of this change).
- N1 results with this change (all trials, published rule read as published):
  run-021448 4 rule N1s, 4 model N1s, kappa 1.00 (PT03 is no longer a model N1), so 1.00
  on the 46 PT contacts; REQ-371 runs: UMCU01 run-061233 0.88 (1 missed), UMCU02 1.00,
  UMCU03 1.00, UMCU04 1.00, UMCU05 0.82 (1 missed; PB6 is now found); pooled 0.91 over the
  REQ-371 runs, 0.92 with the example. REQ-113 is met as written: `test_req113_as_written`
  is no longer an xfail. REQ-114: 4 of 4 on the example run (met), 17 of 19 over all runs
  (89 %; the two others are the rule's 9.3 ms peaks), still an xfail. REQ-112 on the
  ds004080 baselines: 97.5 % silent (unchanged). REQ-273 (the ccep preset meets REQ-113
  and REQ-114 on ds004080) is met, REQ-114 on run-021448 (89 % over all runs).
- Timings (4 cores, otherwise idle): `tests/hierarchical/test_hier_eb.py` about 13 min,
  as on e8db3db (REQ-170 fixture 133 s, REQ-171 108 s, REQ-172 234 s, REQ-175 230 s; on
  f6f90bf the fixture took 918 s and REQ-171 628 s); `tests/models/test_engine_ccep.py`
  with `EVOPARAM_NETWORK_TESTS=1` 8.5 min (fixture 445 s), against about 47 min before
  (six fits of 183 to 913 s).

### ERPs: targeted measurement (`presets.targeted`)

- Why. Split-half reliability on ERP CORE (40 participants, odd vs even trials, Spearman-Brown,
  parent waves at the published channels) showed that the model selection gives a value for
  few participants (BIC: 3 to 19 of 34 to 40) and that its latencies are less reliable than
  the classical ones (P3 0.28 against 0.85). The hierarchical model (EB) does not change this
  (P3 latency 0.26, amplitude 0.86). `selection="fixed"` with 2 to 4 components raises the
  coverage to 5 to 27 participants only (the labelled component is often outside the window).
- `presets.targeted(base, label, window, polarity, margin=0.05)`: a derived preset that fits
  one component of the expected polarity in the measurement window widened by `margin`, on
  a linear background (`selection="fixed"`, `n_components=1`); whether the component is
  there is the existence test's answer. Measured (targeted / classical; reliability of the
  amplitude, then of the latency; coverage): P3 0.89 / 0.80, 0.65 / 0.85, 34 of 34; MMN
  0.79 / 0.66, 0.57 / 0.12, 37 of 40; N170 0.95 / 0.97, 0.74 / 0.79, 34 of 39; N400 0.76 /
  0.85, 0.46 / 0.67, 39 of 40. The model amplitude is as reliable as or more reliable than
  the classical mean amplitude for three of the four components; the latencies are less
  reliable except for the MMN.

### Engine: noise from the plus-minus average (ERPs)

- Diagnosis. On ERP CORE (40 participants; N170, P3, MMN, N400), averages of the same trials
  with random signs (no response, the same noise) gave a component in 20 to 29 % of the
  fits with the `erp` preset (REQ-112 asks at most 5 % on noise). The false components
  appear exactly where the noise scale estimated on the baseline is too low (0.31 µV for a
  noise of 1.06 µV, 0.57 for 1.24): 200 ms of baseline, about 50 correlated samples, give a
  scale that varies by a factor 2 to 3 between participants.
- `FitSettings.noise_source` ("baseline" is spec §5.5.3 step 1, "plusminus", default
  "auto"). The plus-minus reference (Schimmel 1967) is the trials averaged with alternating
  signs over the whole epoch: it holds the noise of the average and none of the response, and
  it involves no random draw. "auto" uses it when the fit has at least 10 trials and the
  baseline holds fewer than 30 independent samples (n (1 - rho) / (1 + rho); the relative
  error of a noise scale is about 1 / sqrt(2 n)), the baseline otherwise: 200 ms of ERP
  baseline at 256 Hz gives about 9, 450 ms of CCEP baseline at 2048 Hz about 47. Each fit
  records the reference in `diagnostics["noise_reference"]`, the settings of a group in
  `noise_reference` ("baseline", "plusminus" or "mixed"), and the methods text names it.
  Plus-minus everywhere was tried first and rejected: it assumes the same response in every
  trial, and where the response varies (ERNA growing over a burst, trial variability of the
  hierarchical simulations) it keeps part of it as noise (REQ-276 0.90 -> 0.85, REQ-170
  AUC 0.95 -> 0.94).
- Measured on ERP CORE with "auto" and the trials (component found on the parent wave /
  a component on a random-sign average of the same trials): N170 38 % / 23 % -> 51 % / 13 %;
  P3 66 % / 29 % -> 66 % / 3 %; MMN 35 % / 20 % -> 48 % / 2 %; N400 28 % / 22 % -> 25 % /
  5 %. Simulated CCEPs and the ds004080 N1 tests (REQ-112, REQ-113, REQ-114) keep the
  baseline and are unchanged. With the plus-minus noise, lowering `delta` finds more
  components (N170 54 %, P3 74 % at delta 3) but brings the false ones to 10 to 15 %: the
  default stays 10. Still open: 13 % of the N170 random-sign averages get a component, above
  the 5 % of REQ-112.
- On the parent waves, half of the N170 "misses" are not fit failures: in 8 of 40
  participants the wave at PO8 never goes below zero in the N170 window (the N170 is a dip
  between P1 and P2, described by two positive components). ERP CORE measures it on the
  difference wave, where with the plus-minus noise (of both conditions) the false components
  fall from 12 to 28 % to 0 to 8 %, and the components found from 52 to 65 % to 40 to 65 %
  (N170 54 -> 46 %, P3 63 -> 53 %, MMN 52 -> 40 %, N400 65 -> 65 %): part of what was found
  before was noise.
- Layer 3 (`validation/erpcore.py`) passes the plus-minus reference of the waves (combined as
  the wave is) and applies the "auto" rule; `validation/layer3/erpcore_*.json` regenerated:
  difference waves, labelled and within 10 ms: LRP 90 %, ERN 85 %, N400 57.5 %, N170 45 %,
  MMN 37.5 %, N2pc 36 %, P3 30 %.
- The methods texts of the reference pipelines are regenerated for this sentence and the
  reseeding sentence (`tools/freeze_methods_text.py`).

### Engine: reseeding and unresolvable components (risk S1)

- Diagnosis. On a simulated cochlear ECAP the fit of contact E01 left a residual sum of
  squares of 3,144 where the simulated curve leaves 25: a search failure, not an
  ambiguity of the data. The greedy path adds the strongest component, then the next while
  refitting all of them; the first slides onto the second and the two end as a pair of
  opposite components that cancel, and the merge-and-compare of §5.5.3 step 5 keeps the
  pair because one merged component fits worse than the pair.
- Reseeding (`FitSettings.reseed`, default True). When the final state holds two
  neighbours of opposite sign closer than `overlap_c` times their summed widths, or a nearly
  singular design, the same number of components is refitted from starts on the extrema of
  the data (the residual of the background-only model), and the result is kept when it fits
  better (warning `reseeded`). `reseed="always"` tries it for every state. Measured
  (`score_fits` on 12 simulations, 3 seeds each): `ecap_ci` F1 0.71 -> 1.00 and amplitude
  error 21 % -> 5 %; `ecap` F1 0.77 -> 0.90 and 17 % -> 2 %; simulated ERP F1 0.83 -> 0.85;
  CCEP unchanged (0.74); time x1.8 on these simulations. On ERP CORE
  (layer 3, parent waves, 40 participants) the default changes nothing; `"always"` raises
  the N170 found from 37.5 % to 52.5 % of participants (within 10 ms: 25 % -> 40 %) at +43 %
  time on a 46-channel CCEP fit, so it is an option, recommended for ERPs.
- `Component.resolvable` (spec §5.5.3 step 6 and risk S1, "unresolvable rather than a wrong
  number"): False for the two members of a pair whose amplitudes or latencies correlate
  beyond 0.95, or of a cancelling pair; warning `unresolvable:<k>`; column `resolvable` in
  `to_dataframe()`, in `responses.tsv` (with `Levels`), in the NWB components table and in
  `ndx-evoparam`.
  Containers saved before keep loading (the field defaults to True).

### Build step S17: MNE, BIDS, NWB, Neo, scikit-learn, plugins and community files

Part 1 (readers and validation data):

- `read_nwb` / `write_nwb` (extra `nwb`): epochs of an `ElectricalSeries` around the
  events of an `EventsTable`, the `trials` table or `TimeIntervals`; events that fall in a
  gap of the timestamps are dropped (they gave identical epochs). `read_neo` (extra `neo`):
  epochs of a Neo `AnalogSignal` around an `Event` or `Epoch`, in µV through `quantities`.
- `TrialSet.from_mne` / `to_mne`, `Evoked.from_mne` / `to_mne`: channel types, bad
  channels, positions (mm <-> m, head frame), metadata and event names kept;
  `GroupFit.to_annotations`, `GroupFit.to_evoked_model`.
- `car_adaptive_select` follows the published CARLA code (covariance zeta curve, line
  notch, bootstrap-mean curve, sensitive optimum; REQ-372), checked on ds004977.
- Layer 3: Allen Neuropixels LFP (REQ-278) and ERP CORE (REQ-274): model 50 % area latency
  against the published measurement on 40 participants and seven components.

Part 2:

- scikit-learn (REQ-320, REQ-321). `ResponseModel`, `TrialFitter` (trial-by-trial shift,
  scale and latency of each component as features) and `ArtifactRemover` follow the
  estimator contract and pass every check of `check_estimator` (scikit-learn 1.6+): the
  list of excused checks, `evoparam/utils/_sklearn_checks.py`, is empty. Array inputs are
  checked the scikit-learn way (2-D or 3-D, dense, real, finite, at least 2 time points,
  the same number of time points as in `fit`). Without a fit window, the window is the data
  range; a range too short for a component (shorter than 3 x min_fwhm, or fewer than 5
  samples) gives skipped fits instead of an error. `ResponseModel.transform(X)`,
  `predict(X)` and `score(X)` fit X with the settings of the estimator (X=None: the data of
  `fit`); `predict` returns curves over the whole time axis (NaN outside the fit window);
  `n_iter_` is the largest number of residual evaluations of one waveform (recorded in
  `ResponseFit.diagnostics["n_evaluations"]`). `ArtifactRemover` returns X unchanged, with a
  warning, when its window does not fit inside the epoch. New extra `sklearn`; scikit-learn
  is in `tests` and `all`.
- Stimulation-metadata validator (REQ-314). `evoparam.io.validate_stim_metadata(bids_root,
  subject=None, strict=False)`: one row per run and field (the six stimulation fields,
  `ElectricalStimulation`, channel `reference`, `type`, coordinates) with the status
  `present`, `inferred` (unit guessed from the magnitude, site from `trial_type`, reference
  from the sidecar only), `missing` or `inconsistent` (partly stated, unreadable values,
  polarity without phase order, coordinates without units, `ElectricalStimulation` false
  with stimulation events); `format_stim_report` prints it. `evoparam validate-bids <root>
  [--sub S] [--strict]` prints the report and exits 5 with `--strict` when a field is
  missing.
- Plugins (REQ-328). The entry-point groups `evoparam.shapes`, `evoparam.presets` and
  `evoparam.readers` are loaded lazily on first access to `SHAPES`, `BACKGROUNDS`,
  `PRESETS` or `evoparam.io.read` (`evoparam/_plugins.py`, a module of the standard library
  and `errors` only, importable everywhere). Shapes pass the checks of `register_shape`,
  presets must be `Preset` instances, readers callables with `extensions`; a name taken by a
  built-in or by another plugin raises `SettingsError` naming both distributions;
  `EVOPARAM_DISABLE_PLUGINS=1` turns them off. Every plugin object a result used is in
  `provenance.environment["plugins"]`; `evoparam info` lists the plugins. Example plugin:
  `evoparam/tests/plugins/evoparam_dummy`; guide: `docs/contributing/add_shape.rst`.
- `evoparam.io.read` now dispatches `.nwb` files to `read_nwb` and Neo objects (or
  `format="neo"`) to `read_neo`.
- Community files (REQ-327): `CONTRIBUTING.md` (development install, tests, coding rules,
  plugins, Developer Certificate of Origin), `CODE_OF_CONDUCT.md` (Contributor Covenant
  2.1), `GOVERNANCE.md`, `SECURITY.md`, `SUPPORT.md`, issue forms under
  `.github/ISSUE_TEMPLATE/`; linked from the README. The DCO check as a required status on
  `main` is a repository setting, not set here.

Part 3 (audit of REQ-310 to REQ-329 against the specification, and what it found):

- MNE (REQ-310/311). `TrialSet.from_mne(stim_columns=...)` harmonises the stimulation
  columns of the metadata ("auto", or a `{column: target}` mapping); `to_mne` gives back the
  original metadata. Event times: `events[:, 0] / sfreq`, without adding `first_samp`.
  Deviation from §10.1, for correctness: MNE event samples already count from the start of
  the acquisition (they include `first_samp`); adding it again would shift every event and
  `to_mne` would not give the events back (tested: events identical after the round trip).
  `evoparam.fit` on an `mne.Evoked` is compared with the converted path in the tests. The
  harmonisation moved to `evoparam.data._harmonize` (re-exported by `evoparam.io`) so that
  `data` can use it.
- BIDS (REQ-313). `read_bids(stim_mapping=...)` passes a user mapping to the harmonisation
  (recorded in the provenance); `tests/io/test_bids_events.py` covers ds004080, ds003708
  and ds004977 shapes (published headers) and a minimal file. `read_bids` warns when epochs
  hold only zeros: ds004977 keeps the signal around one stimulation site per run and writes
  zeros elsewhere, which used to give silent all-zero epochs.
- BIDS derivatives (REQ-315/316). `responses.tsv` now holds the `classical_*` measures, the
  `evoparam_stim_*` columns and the standard `electrical_stimulation_*` columns of the source
  (from the fitted trials, `trials=`), the real `stim_site` (it held the condition label),
  and sidecars with `LongName`, `Units` and `Levels`; the README has the methods paragraph;
  `SourceDatasets` comes from the registry when the raw dataset gives no DOI; the datatype is
  taken from the source. Beyond the specification: each result is saved next to its table
  as an `.evoparam` container (`.evoparamset` for trial fits), so `read_bids_derivatives`
  gives back the exact objects (model curves included, not only "same tables and
  provenance") and detects a table edited after writing. Checked with the BIDS validator
  3.0.2 (`--datasetTypes derivative`, 0 errors, run locally) and `bidsschematools`;
  `docs/bids_stimulation_columns.rst` lists the proposed columns (the BEP issue is not
  opened: an action of the author).
- NWB (REQ-317/318). `write_nwb(extension=..., trials=..., model=...)`: `electrode`
  (`DynamicTableRegion` into the electrodes) and `trial` (into `trials`) columns, the
  missing columns (`stim_site`, `present`, `area`, `n_trials`, `ci_level`, `condition`),
  `evoparam_model` (fitted curves, V), `evoparam_results` (each result as a zipped
  container) and `read_nwb_results` (exact round trip, edited tables detected). The
  `ndx-evoparam` extension (namespace 0.1.0, `evoparam/io/ndx_evoparam`, YAML generated by
  `python -m evoparam.io.ndx_evoparam`) types the tables and adds `ResponseFitSettings`;
  files pass `nwbinspector` with and without it. Publishing it (PyPI, NDX catalog) is a
  release step. The StimTable event source is tested with a locally declared
  ndx-electrical-stim type. `read_nwb` fills one array in place: reading epochs covering
  10 % of a 1.07 GB file grows the peak memory by 0.248 GB (limit 0.25 GB; it was 0.47 GB);
  the epochs themselves are 0.214 GB and the rest is the HDF5 library's cache, so the margin
  is small on other machines (`tests/io/test_nwb_memory.py`, `EVOPARAM_SLOW_TESTS=1`).
- Containers. Columns holding pandas `NA` and nullable integers or booleans (`Int64`,
  `boolean`) are saved and read back exactly (they raised before).
- Command line (REQ-322/323). `fit`, `test` and `clean` accept one recording (`.vhdr`,
  `.edf`, `.bdf`, `.fif`, `.nwb`, `.npz`; events from `--events`, the BIDS events.tsv next to
  it or the EDF+ annotations); `--preset` defaults to `ccep` (iEEG) or `erp` (EEG);
  `simulate --seed`; `--config` reads TOML (with `[evoparam]` and per-command tables) as
  well as JSON; `clean` writes the cleaned epochs as FIF (or npz without `mne`); `info`
  lists the backends; a trial fit that did not converge gives exit code 3. Exit codes 3 and
  130 are tested. `benchmark` comes with step S19.
- Plugins (REQ-328). `reproduce` names a missing plugin (`missing_plugins`, with the `pip`
  command) and the error of a missing plugin preset says which package to install; each
  result records the plugins it used (not every plugin used in the process); built-in names
  are answered without loading the plugins, so a plugin may use the built-in presets while
  it is imported.
- Tutorials (REQ-326). `docs/tutorials`: one per preset (11, REQ-303), six integrations and
  `reporting_your_results`, each ending with the methods paragraph; `preset_ccep`,
  `preset_ccep_seeg`, `integration_bids` read ds003708 / ds004977 remotely, `preset_erp`
  reads ERP CORE through the new `fetch_erpcore` (SHA-256 of the registry, else of OSF).
  The registry records the SHA-256 of the files the tutorials download. `integration_streaming`
  comes with step S18. `tests/test_docs.py` checks them and runs them all
  (`EVOPARAM_SLOW_TESTS=1`, and `EVOPARAM_NETWORK_TESTS=1` for the data). A minimal Sphinx
  site (`docs/conf.py`, extra `docs`) builds without warning; the full site is step S20.
- `run_pipeline` takes the baseline of the artifact-quality score from the preset, or from
  the part of the epoch before the artifact: the ECAP presets (4 ms epochs) failed with the
  default -0.5 to -0.05 s. A fit window may end at the (excluded) end of the epoch.
- REQ-312 / REQ-329: `tools/check_no_inherited_code.py` (8-line windows against ERPparam at
  the NOTICE commit; no match in the S17 modules, 2,509 in the inherited `objs` and `core`)
  and `RELEASING.md` with the 1.0 checklist.

Found while writing the tutorials (the first is addressed below):

- The model selection accepts "cancelling pairs": two large components of opposite sign at
  almost the same latency, in place of the true components. Seen on clean simulated ECAPs
  (`ecap_ci`, contact E01: -320 and +218 µV for N1 -137 and P2 +111 µV), on a single
  Hanning bump (Neo tutorial), and in `fit_shared` on ds004977 (which also returned a
  component before the stimulus, outside the fit window).
- On ERP CORE (layer 3), the `erp` preset finds and labels the component in 28 to 68 % of
  the participants on the parent waves (N170 27.5 %, MMN 32.5 %, P3 67.5 %) and 41 to 70 % on
  the difference waves; ERN and LRP 78 to 95 %.
- `tep` on its own simulation: a residual of the decay artifact remains near 20 ms under the
  coil after interpolation and subspace projection, and N15/P30 are not separated from it.

### N1 case (REQ-113, REQ-114): diagnosis on ds004080 and fixes

- Diagnosis. The xfail test compared the model with a re-reading of the rule (most
  negative sample of 9–100 ms below 3.4 x the SD of (-0.5, -0.02) s), applied after the
  model's own preprocessing (template removal, `car_adaptive`), on the 8 trials of the cached
  40 s segment and on 120 contacts, `el###` channels marked bad included. The published
  pipeline (mnl_ccepAge, van Blooijs et al. 2023: `ccep01_averageCCEPsAndDetectN1s.m`,
  `ccep_averageConditions.m`, `ccep_detect_n1peak_ECoG.m`) does otherwise: monopolar raw
  data, no filter and no artifact removal, both polarities of the pair merged (10 trials
  for PT01-PT02), the median of (-1, -0.1) s removed from each trial, then the median of
  (-2, -0.1) s from the average; SD of (-2, -0.1) s with a 50 uV floor; local minima found
  by `peakfinder` (selectivity 20 uV) from 9 to 500 ms, the first sample excluded, kept up
  to the first sample after 100 ms, the one of largest magnitude chosen; refused above
  3000 uV; good ECoG channels of channels.tsv only (100 on this run). Read so, the rule
  finds 4 N1s on run-021448 (PT09, OC5, OC6, OC7), 1 of them on the 46 PT contacts; the old
  re-reading found 8, four of them the artifact residue at the window start (9.8 ms: PT03,
  PT10, PT11, OC8) and the rest drift near 100 ms. Disagreements between the model and the
  published rule fell in four classes: (1) the old reading's artifact and drift
  detections (the rule as read, not the model); (2) small real deflections (30–80 uV) that
  the model finds and the 170 uV floor of the rule ignores (the rule's design: 8 of 20
  engine-only N1s on six subjects lie between 3.4 SD and the floor, 5 below 3.4 SD);
  (3) N1 latencies of the fitted component several ms from the trough, the early component
  and the recovery tail trading amplitude (the engine's reading); (4) fits whose curve lay
  hundreds of uV from the data (R² < 0) and were still accepted (the engine).
- Engine, exact AR likelihood. `Whitener` takes the autocovariances of the baseline AR
  model (`ar_model(..., return_acov=True)`) and whitens the first `p` samples of the window
  by the Cholesky factor of their stationary covariance instead of dropping them. Near a
  unit root (sum of the AR coefficients 0.98–0.995 in intracranial averages at 2 kHz) the
  AR filter is almost a difference, and the conditional likelihood was blind to a misfit of
  the slow part of the waveform: the curve could float away from the data (class 4).
- Engine, noise model by least squares. `ar_model` fits the AR coefficients by least
  squares (covariance method; orders whose model is not stationary are skipped; order by
  AIC as before) instead of Yule-Walker, which is biased towards a faster decay near a unit
  root and so under-states the slow power of the noise. With Yule-Walker and the exact
  likelihood, correct fits read slow fluctuations of real baselines as components: REQ-112
  on the ds004080 baselines fell to 92.5 % silent (74 of 80; 95.0 % before, partly because
  the old fits failed: R² down to -2 on silent baselines). With least squares: 97.5 % (78
  of 80; the two others are the stimulated contacts PT01, flat, and PT02).
  The exact likelihood makes some fits slower (up to 10x on sub-ccepAgeUMCU19, 63 channels);
  REQ-115 was already unmet.
- `evoparam.models.fitted_peak(fit, window, polarity=...)`: the deepest interior trough
  (or highest interior peak) of the fitted curve in a window, NaN when the curve is
  monotone there. Where the tail and an early component overlap, their split is sloppy but
  the curve is well determined (class 3): with the Yule-Walker engine, the latency of the
  first N1 component was within 2 ms of the published peak on 27 % of the shared
  channels, the trough of the curve on 96 %.
- Registry rule `vanblooijs2023`: new fields `peak` (`"extremum"`, the default and the
  paper's description, unchanged; `"local"`, the published code to the letter, measured
  from the baseline median), `peak_selectivity_uv`, `peak_search_end`,
  `exclude_first_sample`, `saturation_uv`; new public helper
  `evoparam.existence.rules.selective_maxima` (re-implementation of N. C. Yoder's
  `peakfinder`, BSD; it matched a direct port on 20 000 random signals and on all 427
  channels below). With `peak="local"` and baseline (-2, -0.1) s the registry rule gives the
  published detections on all 427 channels. Kept from the published code: a positive-valued
  minimum is accepted on its magnitude unless all minima are positive. Notes: the code
  labels the onset sample t = 1/fs (latencies here are on the evoparam axis, 1/fs earlier);
  the registry SD stays the population SD (ddof 0; the code's ddof 1 changes no detection
  here). `"local"` is not the default: REQ-135 (permutation test vs vanblooijs2023, kappa
  0.763 >= 0.7, unchanged) falls to 0.40 with it, because the permutation test, like the
  paper's text, looks at the most extreme deflection.
- Model N1 for REQ-113/114: the trough of the fitted curve in 9–50 ms, at least as deep as
  the published amplitude criterion (3.4 x max(SD, 50 uV) of the published average), i.e.
  the rule's own criterion applied to the model's measurement. The engine runs the ccep
  preset pipeline (epochs (-1, 2) s, template removal (-2, 8) ms, `car_adaptive`, baseline
  (-0.5, -0.02) s); the rule runs the published pipeline on the same trials.
- Tests (`tests/models/test_engine_ccep.py`, network only): all trials of the pair (both
  polarities), read by byte ranges and cached as `.npz` under
  `data_folder()/n1-validation/ds004080`, on run-021448 (PT01-PT02, the example) and the
  runs of `validation/ds004080_subjects.txt` (REQ-371: UMCU01 run-061233, UMCU02
  run-021804, UMCU03 run-031411, UMCU04 run-031148, UMCU05 run-031133), each with the
  first stimulation pair of the run, in event order, with at least 10 trials (the list's
  own rule): PT01-PT02, FC01-FC02, C01-C02, TPa1-TPa2, P01-P02. These runs were not used to
  choose the method: the development subjects were UMCU07, 19, 28, 40 and 49 (the
  existence validation), reported below. Passing: `test_req113_model_finds_the_published_n1s`
  (at most one missed channel on the example; over the REQ-371 runs pooled kappa >= 0.8 and
  each run >= 0.60, the inter-rater kappa of human reviewers) and
  `test_req114_n1_latency_agrees_with_published_rule` (>= 90 % within 2 ms on the
  example). Documented xfails: `test_req113_as_written` (kappa >= 0.8 on the 46 PT
  contacts, and at most one missed channel and kappa >= 0.8 on each REQ-371 run) and
  `test_req114_n1_latency_agrees_on_all_runs`. New registry tests for `peak="local"` and
  `selective_maxima`.
- Results against the published rule read as published (all trials, good ECoG channels;
  kappa, missed channels, latency within 2 ms). Before: the S7 engine and reading (an N1
  component with latency in 9–50 ms). After: exact likelihood, least-squares noise model
  and the trough reading.

  Development subjects (the method was chosen on these):

  | subject (pair, channels) | rule N1 | kappa before | after | missed before | after | latency before | after |
  |---|---|---|---|---|---|---|---|
  | UMCU01 run-021448 (PT01-PT02, 100) | 4 | 0.34 | 0.88 | 0 | 0 | 2/4 | 4/4 |
  | the same, 46 PT contacts | 1 | 0.17 | 0.66 | 0 | 0 | 0/1 | 1/1 |
  | UMCU07 (sTa5-sTa6, 109) | 9 | 0.31 | 0.70 | 1 | 1 | 1/8 | 7/8 |
  | UMCU19 (PT01-PT02, 63) | 3 | 0.46 | 0.47 | 0 | 1 | 1/3 | 2/2 |
  | UMCU28 (F01-F02, 60) | 6 | 0.45 | 1.00 | 1 | 0 | 2/5 | 4/6 |
  | UMCU40 (FB1-FB2, 53) | 2 | 0.17 | 1.00 | 1 | 0 | 0/1 | 2/2 |
  | UMCU49 (HP01-HP02, 42) | 6 | 0.21 | 0.91 | 1 | 0 | 1/5 | 6/6 |
  | pooled (427) | 30 | 0.34 | 0.81 | 4 | 2 | 27 % | 89 % (28) |

  REQ-371 runs (confirmation, after the method was fixed):

  | subject, run (pair, channels) | rule N1 | model N1 | kappa | missed | latency |
  |---|---|---|---|---|---|
  | UMCU01 run-061233 (PT01-PT02, 100) | 5 | 4 | 0.88 | 1 | 3/4 |
  | UMCU02 run-021804 (FC01-FC02, 84) | 2 | 2 | 1.00 | 0 | 1/2 |
  | UMCU03 run-031411 (C01-C02, 70) | 3 | 3 | 1.00 | 0 | 3/3 |
  | UMCU04 run-031148 (TPa1-TPa2, 109) | 1 | 1 | 1.00 | 0 | 1/1 |
  | UMCU05 run-031133 (P01-P02, 91) | 6 | 5 | 0.71 | 2 | 4/4 |
  | pooled (454) | 17 | 15 | 0.87 | 3 | 12/14 |

  With the example run: pooled kappa 0.87 over 554 channels, latency 16 of 18.

  The old test's own measure (re-read rule on the 8-trial segment) was about 0.5. On the
  46 PT contacts the only disagreement is PT03, next to the stimulated pair: a monotone
  decaying tail for the rule, a 344 uV trough at 17 ms for the model. Missed channels:
  UMCU07 sOc7 (rule 70 ms, outside the model's 9–50 ms N1 label), UMCU19 sT7 (model trough
  169 uV against a 170 uV floor), UMCU01 run-061233 PT11 (the rule's peak is the second
  sample of its window, 9.3 ms at 2048 Hz: artifact residue, class 1), UMCU05 OB6 (no
  trough in the fitted curve) and PB6 (trough 146 uV, under the floor). The rule's
  9.3 ms peaks also make the two latency misses of the REQ-371 runs (PT03 and FC10, model
  trough at 17.5 and 33.3 ms); the development misses are 2.1–3.3 ms. UMCU19's low kappa
  comes from PT03-PT05, next to the stimulated pair, whose troughs of 2.7–3.3 mV the rule
  refuses (no local minimum, or above its 3000 uV saturation limit); the model's criterion
  uses the floor only. Intermediate steps on the development subjects, pooled kappa /
  latency: Yule-Walker engine with the trough reading 0.74 / 83 %; with the exact
  likelihood 0.79 / 96 % (REQ-112 then unmet, above); exact likelihood with the component
  reading 0.33 / 27 %.
- REQ-113 is still unmet as written (0.66 on the 46 PT contacts; UMCU05 misses two
  channels and has kappa 0.71). Its wording cannot be met by a better model alone: the
  rule's 170 uV floor and its rejection of responses whose trough is not a local minimum
  are a definition of "N1", not a measurement, its peak search takes artifact residue at
  9.3 ms at 2048 Hz, and on the example run the rule finds one N1 on the 46 PT contacts,
  so a single disagreement sets kappa at 0.66. "46 non-stimulated channels" are the PT
  grid; the published channel set is the good ECoG channels (100). Proposed wording (the
  spec text is not changed here): "on ds004080 sub-ccepAgeUMCU01 run-021448, stimulation
  pair PT01-PT02, all trials of the pair, the published rule applied as the published code
  applies it (monopolar average, median baselines, local minimum in 9–100 ms, 3.4 x SD with
  a 50 uV floor) on the good ECoG channels, and the model's N1 (the trough of the fitted
  curve in 9–50 ms) judged by the same amplitude criterion: the model misses at most one
  of the rule's channels; on the runs of the REQ-371 subjects (first pair of >= 10
  trials), Cohen's kappa is >= 0.8 pooled and >= 0.6 (the inter-rater agreement of human
  reviewers) on each run". Measured against it: met (0 missed on the example; pooled 0.87,
  each run >= 0.71). REQ-114 is met on run-021448 (4 of 4), not over all runs (89 %).
- REQ-273 (the ccep preset meets REQ-113 and REQ-114 on ds004080) follows: the test
  pipeline is the preset's (epochs, `template+tail:joint` artifact handling, `car_adaptive`,
  baseline, fit settings); REQ-114 met on the example run, REQ-113 not met as written (met
  under the proposed wording).

### Build step S12: shared shapes across channels, parameter maps and the volume-conduction screen

- New module `evoparam.spatial` (spec §5.9): `shared_shapes` with `convergent` (sites into
  one channel) and `divergent` (one site into the channels), returning a `ShapeBasis`;
  `as_template` (a basis curve as a `template` component shape); `fit_shared` (the
  engine's model with shape parameters shared across channels, optional per-channel
  latency offsets and a graph-Laplacian penalty over neighbouring contacts); `neighbours`;
  `parameter_map` and `plot_parameter_map`; `connectivity_matrix`, `connectivity_to_long`,
  `connectivity_from_long`, `connectivity_array`; `screen_volume_conduction`. The
  basis-profile-curve method is re-implemented from the published description (Miller,
  Müller and Hermes, 2021); the GPL code is neither imported nor copied. Only numpy, scipy
  and pandas are used (NMF, kernel PCA, k-means and the adjusted Rand index are written
  here; scikit-learn is not needed).
- `evoparam.models.references` re-exports `rereference` and `SCHEMES` of `evoparam.prep`,
  as `models.component_shapes` does for the shapes, so that `spatial` depends on `models`
  only (spec §4.2; new import-graph test).
- `evoparam.io._fetch`: a 404 whose body is the S3 error `NoSuchBucket` is retried (up to
  ten times, 0.5 s apart). Through the proxy of the development machine, S3 answered so to
  20 to 40 % of the requests for files that exist, which made every remote MEF read fail
  and could make an optional file (electrodes.tsv) look absent. Other 404s are still not
  retried.
- Choices and changes to the specification:
  - number of shapes: the largest `q <= max_shapes` whose clustering uses all `q`
    clusters, is stable (mean adjusted Rand index of the 20 restarts >= 0.8, as specified)
    and splits no cluster: for every pair of clusters, the mean t-value between them is
    below 0.75 times the geometric mean of the mean t-values within them. The published
    degeneracy index (overlap of the rows of H below 0.5) rejected the true `q = 3` when
    two shapes were partly correlated and the noise low; it is reported in the selection
    table (`h_overlap`). `"unstable_basis"` is warned when `q = 1` is returned although some
    `q >= 2` clustering splits no cluster but is unstable. A site joins the cluster of its
    largest NMF loading when that loading is at least half the equal-contribution loading
    `1/sqrt(n)` (the published tolerance); `kpca` clusters by k-means in the kernel PCA
    coordinates.
  - `ShapeBasis.weights` holds the mean projection `alpha` of each group on each curve
    (signed, data units times sqrt(samples)), not non-negative loadings; the NMF loading of
    each group is the column `nmf_loading` of `assignment` (with `group`, `shape_id`,
    `weight`, `r2`, `n_trials`, `used`).
  - `ShapeBasis.as_template(k)` is the function `evoparam.spatial.as_template(basis, k)`:
    the data containers may not import the shapes (spec §4.2). The ends of the curve are
    tapered (10 % cosine) so that the template starts and ends at rest.
  - `fit_shared`: `shared` names parameters or roles (`"latency"`, `"fwhm"`,
    `"asymmetry"`; `PARAMETER_ROLES`); the background time constants stay at the
    per-channel values; residuals are divided by each channel's noise SD; `group_fit=`
    reuses per-channel fits; `channels` defaults to the channels that are neither bad nor
    stimulating contacts. Penalty: `lambda` on a grid of multiples of the median curvature
    of the channels' fit in their offset (0 included); leave-one-channel-out predicts the
    left-out channel's offset from its smoothed neighbours through the penalty, with a weak
    pull (1e-3 of the median curvature) towards the shared latency, so that without
    coupling the prediction is the shared latency and not the neighbours' raw estimates
    (with them, the criterion chose a very large `lambda` when the latencies had no spatial
    structure and doubled the latency error); the chosen `lambda` is kept only when its
    held-out error is more than 1 % below that of `lambda = 0`. `smooth_lambda=` forces a
    value; `smooth=True` needs `free_latency=True`.
  - `parameter_map` takes `ch_info=` (a table, or the TrialSet / Evoked that carries it;
    a GroupFit has no channel table), `component=` (default: the component of largest
    absolute amplitude) and `basis=` (divergent shapes per site); `existence=` accepts a
    ResponseTest (`existence_p`, `existence_significant`) or any object with a
    `p_response` table (`p_response`; the hierarchical fits are build step S11). The map is
    a table plus an electrode figure (matplotlib); the glass brain is left to nilearn,
    which is not installed here. `connectivity_array` gives the (site x channel) array;
    `mne-connectivity` objects are not built (the extra is not installed).
  - `screen_volume_conduction(ts, group_fit=None, ...)`: the three flags are
    `no_reversal` (adjacent contacts exist and none correlates <= -0.5 with at least 20 %
    of the RMS), `decay` (zero-lag correlation with the waveform of the two contacts
    nearest to the site >= 0.9 after correction for the attenuation by the channel's noise,
    same sign, and amplitude within a factor 2 of the power law `|a| ~ r^-p`, `p > 0.5`,
    fitted over such copies; `decay_model="none"` keeps the copy test only) and `reference`
    (RMS below half, or correlation below 0.5, under a scheme of `reference_schemes`).
    Risk: two flags or more `high`, one `medium`, none `low`.
- Acceptance criteria met on simulations (tests in `tests/spatial/`): REQ-190 (10 of 10
  runs give `n_shapes = 3` with `nmf` and with `kpca`; matched curves correlate 0.971 to
  0.999 with the truth, median 0.991; site assignment exact), REQ-191 (latency RMSE over
  30 channels at SNR 2: 1.67, 9.67 and 1.86 ms independently, 0.62, 0.12 and 0.26 ms
  shared, on the channels where the independent fit found a component), REQ-192 (held-out
  error on fresh noise 0.9997 times that of `lambda = 0` with a drifting latency, 1.0 when
  the latencies have no spatial structure and the penalty is switched off), REQ-193 (`high`
  on 128 of 128 volume-conducted channels, `low` on 20 of 20 local dipolar contacts, five
  scenes), REQ-195.
- Real data (EVOPARAM_NETWORK_TESTS=1):
  - REQ-196 met on ds004080 sub-ccepAgeUMCU01 run 021448 (ECoG, 512 Hz; three sites of
    the six cached, PT01-PT02, PT03-PT02, PT03-PT04; 104 channels): one row per (site,
    channel), coordinates equal to `electrodes.tsv` (fsaverage, mm), no NaN amplitude
    where the fit status is `ok`. The fits take about 0.8 s each (8 minutes for six
    sites, hence three).
  - REQ-194 on ds003708 (basis-profile-curve paper; MEF 3.0 read remotely, one contact,
    459 good pulses of 48 sites): the paper's example contact is "a parahippocampal gyrus
    electrode", not named; `electrodes.tsv` labels LMS1 and LMS2 so. With the paper's window
    (50 ms to 2 s) and the monopolar recording, LMS2 gives **3 bases** as published (the
    same with `random_state` 0 to 4), 36 of the 46 sites assigned. LMS1 gives 6 bases with
    that window and 3 with the specification's default (9 to 500 ms); LMS2 gives 2 with the
    default window. The paper used the adjusted common average of the derivatives, not read
    here. The assignment part is **not verified** (test xfail): the figure's assignment is
    not published as numbers and the figure could not be read here; against a proxy built
    from the three regions the paper names (superior temporal gyrus, inferior temporal
    gyrus, fusiform gyrus; a site is scored when both contacts share the Destrieux label),
    4 of the 7 scored sites agree (0.57).

### Build step S14: statistics, intervals, comparisons, reliability and export to mixed models

- New module `evoparam.stats` (spec §5.11): `bootstrap` and `bootstrap_interval`
  (percentile and BCa, stratified by `by`), `jackknife_latency`, `jackknife_t`,
  `jackknife_f`, `compare_conditions` and `hedges_g`, `reliability`, `spearman_brown`,
  `dependability`, `trials_needed`, `variance_components` and `reml` (nested random
  intercepts by REML, with known sampling variances), `to_long`, `column_dictionary`,
  `mixed_formula`, `fit_mixed` and `group_summary`, and a vectorised `score_waveforms`
  (the classical scores of `classical_measures`, equal to them, on many waveforms at once).
  DataFrame outputs carry `attrs["provenance"]`, `attrs["units"]` and `attrs["settings"]`.
- Variance components for the hierarchical step (spec §5.8, empirical Bayes):
  `reml(y, groups, se=, X=, residual=)` returns the variance of each level, the residual
  (beyond the known sampling variances), the fixed effects and their covariance, the BLUPs
  with the standard errors of their prediction errors (including the uncertainty of the
  fixed effects), the shrunken unit estimates (`unit_mean`: mean design row times the fixed
  effects plus the effects on the unit's path, e.g. a channel inside its subject) with their
  standard errors, the shrinkage factors and the number of observations per unit. Each
  top-level block is inverted with the Woodbury identity; one level without sampling
  variances has a closed-form profile. Checked against dense computations and the inverse
  of Henderson's mixed-model equations to 1e-8. `variance_components` is the same fit on a
  DataFrame (`VarianceResult` with `table`, `fixed`, `random_effects` per level: `estimate`,
  `se`, `shrunken`, `shrunken_se`, `shrinkage`, `n_obs`).
- Choices and changes to the specification:
  - bootstrap intervals use the small-sample expansion of Hesterberg (2015) by default
    (`expand=True`: the normal quantile is replaced by `sqrt(n / (n - 1)) t_{n-1}`, with
    `n` the number of trials less the number of strata plus one): with 30 trials the
    textbook intervals cover 0.935 (normal) and 0.925 (log-normal, CV 0.5), below the 0.93
    of REQ-230; `expand=False` gives the textbook intervals;
  - `bootstrap` also takes a score name (`"mean_amplitude"`, `"peak_latency"`, ...) with
    `window` and the score options; resamples of a score are then computed for all
    resamples at once by matrix products. The BCa acceleration comes from the
    leave-one-trial-out jackknife;
  - `compare_conditions` also takes `by` (contrasts per channel, component, ...),
    `conditions` (the two levels to compare), `ci_level` and `n_boot` (percentile interval
    of the difference). Exchangeability is enforced for every level: the levels above the
    unit (trial < channel < subject) may not vary within a contrast, so trials of several
    subjects, or `paired=False` with units present in both conditions, raise
    `SettingsError`. Paired contrasts flip signs of the paired t (exactly enumerated when
    `2^n` is at most the number of permutations); unpaired contrasts permute labels of a
    Welch t;
  - `reliability` also takes `window` and `picks`; a callable scorer takes
    `(times, waveforms[n, n_times])` and returns one value per waveform. Dependability uses
    the REML variance components of trials within units, the D-study gives the trials
    needed for `target_phi`, and the standardised measurement error is the root mean square
    of the per-unit SMEs (bootstrap SME for non-linear scores);
  - `trials_needed`: the existence probability uses subsamples without replacement of the
    first `n` trials (it is not extrapolated beyond `N`, with a warning); the score SD
    uses bootstrap resamples of size `n`, extrapolated beyond `N` by `c / sqrt(n)` with `c`
    the mean of `sd(m) sqrt(m)` over `m >= N / 2`, times `sqrt(N / (N - 1))`;
  - `to_long` also accepts `ResponseFit`, `TrialFitSet` and plain DataFrames, with
    `metadata=` and `units=`; the JSON sidecar (`attrs["sidecar"]`, `column_dictionary`)
    follows the BIDS style (`Description`, `Units`, `Levels`). `mixed_formula` takes `se`
    (the `brms` measurement-error term `se(se, sigma = TRUE)`) and quotes names that are not identifiers (backticks in
    R, `Q("...")` for patsy). `fit_mixed` takes `nested` and `re_formula`;
  - `group_summary` also takes `ch_info` (regions), `n_boot`, `ci_level` and
    `random_state`;
  - the import graph allows `stats -> existence` (`trials_needed` runs the existence
    test); `stats` stays below `hierarchical`;
  - new extra `stats` (`statsmodels>=0.14`, `formulaic>=1.0`).
- Acceptance criteria met: REQ-230 (10 000 simulated channels of 30 trials: BCa 0.948 and
  percentile 0.949 on normal statistics, 0.940 and 0.938 on log-normal ones), REQ-231 for
  the Smulders et al. (2010) worked example (retrieved values and corrected F of 9.2, 9.8,
  4.9 reproduced) and for noise-free retrieval (centroid latency to 1e-9; the 50 %
  fractional area latency is not linear across participants and is retrieved within
  2.7 ms), REQ-232 (type-I error 0.052 trials, 0.049 channels, 0.043 and 0.049 subjects,
  paired and unpaired, 4 000 null simulations each; refusal of trials across subjects),
  REQ-233 against an independent implementation of the published formulas on simulated
  balanced and unbalanced data (within 0.001), REQ-234 for amplitude scores (median ratio
  of predicted to true SD at 80 trials 1.06 for the mean amplitude, 0.95 for the peak
  amplitude), REQ-235, REQ-236 for `lme4` and `brms`, REQ-237.
- Not met or not checked here:
  - REQ-231, Kiesel et al. (2008) example: the paper could not be read in the build
    environment, so its printed values are not frozen (xfail);
  - REQ-234 for latency scores (xfail): from 20 trials the `1/sqrt(n)` law underestimates
    the SD at 80 trials (median ratio 0.77 for the fractional area latency, 0.70 for the
    peak latency at a per-trial SNR of 1), because the latency of a noisy average improves
    faster than `1/sqrt(n)`;
  - REQ-233 on ERP CORE: the N170 test (`tests/stats/test_stats_erpcore.py`) reads a local
    copy given by `EVOPARAM_ERPCORE_N170` with `EVOPARAM_NETWORK_TESTS=1` and needs `mne`;
    ERP CORE has no licence record in the registry, so nothing is downloaded or
    redistributed. It was not run here (no copy, no `mne`);
  - REQ-236, `statsmodels` part: written, skipped when `statsmodels` is not installed (it
    was not installed in the build environment).
- Tests: `tests/stats/` (bootstrap, jackknife, compare, reliability, variance, export, ERP
  CORE).

### MEF 3.0 reader (the Mayo Clinic datasets)

- New module `evoparam.io.mef`: `read_mef(path, tmin=, tmax=, picks=, remote=, check_crc=)`
  and `read_mef_header`, in numpy and the standard library only, written from the public
  MEF 3.0 specification and the reference C library (meflib). It reads session folders
  (`.mefd`), their channels (`.timd`) and segments (`.segd`): universal headers, metadata
  (`.tmet`: sampling frequency, units conversion factor and units, start time), block
  indexes (`.tidx`) and data (`.tdat`), with range-encoded differences (RED) decoded for
  all the blocks at once (one symbol per numpy step; about 120 ns per symbol). Header,
  metadata, index and block CRCs are checked (`check_crc=False` skips them). Lossy blocks
  (scale factor, detrending) are restored as meflib does; `RED_NAN` gives NaN, the two
  infinity codes give -inf and +inf. Values are converted to µV from the conversion factor
  and the units of each channel (a channel whose unit is not a voltage is left unconverted,
  with a warning).
- Only the blocks under the requested window are read, located with `.tidx`. Remotely
  (an http(s) address, OpenNeuro by default) only the metadata and index files of the
  channels read and the bytes of those blocks are downloaded, by HTTP byte ranges, 16
  requests at a time. The channel list comes from the S3 listing of OpenNeuro, or from the
  BIDS `channels.tsv` beside the session (segments are then probed until one is missing).
- Time axis: sample 0 is the earliest block of the channels read. Within a run of blocks,
  samples follow the sample numbers of `.tidx`: the block time stamps of the Mayo
  recordings jitter and drift by up to 42 samples, while the `events.tsv` onsets are sample
  numbers divided by the sampling frequency. A new run starts at a block flagged as a
  discontinuity, or whose start time jumps forward by more than 1 s without the flag; the
  gap is NaN. Blocks whose index extrema are equal (the censored zeros of ds004977) are
  filled without decoding.
- Not supported: encrypted data or metadata (section 2, or encrypted blocks) raise
  `ReaderError` asking to decrypt the session with the MEF library first; an encrypted
  section 3 (subject identity and recording time offset only) is recorded in the header and
  otherwise ignored. Records (`.rdat`, annotations), video channels and big-endian files are
  not read.
- `read` dispatches `.mefd` (local or http(s)): the recording or a window of it, or epochs
  around `onsets`, with the NaN gaps marked in `masks`. `read_bids` reads MEF runs locally
  and remotely (a remote run is read from `.vhdr` when it exists, else from `.mefd`): only
  the metadata of the picked channels and the blocks under the epochs are read, and the
  gaps are masked. `provenance.source` records `format: "mef"`, the units and conversion
  factors, the start time, the number of blocks read, the bytes read and the sha256 of the
  metadata files and of the data bytes read (the same for a local and a remote read; not
  the digest of the whole files, as for BrainVision).
- Fix in `evoparam.io._fetch.RemoteFile`: a first range request that ends past the end of
  the file (size not known yet) was taken as an incomplete reply and retried until it
  failed.
- Real data (OpenNeuro, by byte ranges): ds003708 header of 89 channels in 8.6 s (21.5 MB,
  most of it the block indexes of the whole run, which are read in full), then 20 s of the
  89 channels in 11.8 s (25 MB in all, 3.7 MB of blocks); ds004977 header of 273 channels
  in 7.7 s (13 MB) and 2 s of the 273 channels in 16.8 s (24 MB, 10.5 MB of blocks: these
  recordings are stored almost uncompressed, 19 KB per channel and second). Every block CRC
  of the ds004977 channel checked in full matches, and the decoded extrema equal the index
  extrema. ds004977 stores nV with a conversion factor of 0.001 to µV; its runs are zeros
  outside the 54 s of one stimulation site.
- Mayo events (`tests/events/test_mayo_events.py`, 15 runs: ds003708, ds004457 sub-1 to
  sub-5, ds004696 sub-02, 03, 06, 07, 08 (sub-01, 04 and 05 repeat ds004457), ds004977 one
  run per subject; 20 s from the first pulse of three sites per run at 2048 Hz, one at
  4800 Hz; 172 pulses). `detect_pulses` with its defaults finds all 152 pulses of the
  2048 Hz runs, median onset error 1 sample (the `events.tsv` onsets are 0.49 ms later),
  but 29 false alarms (precision 0.84: transients of 0.1 to 3 mV against pulses of 35 to 45 mV in
  ds004457 sub-2 and ds004696 sub-07; REQ-040 is specified on ds004080 only). On ds004977
  it finds no pulse in 2 of the 4 runs (largest score 14.9 and 10.9 robust SD, threshold
  20; recall 0.50): the artifact is large only on a few contacts. With the sEEG channels
  and `threshold_sd=7`, all 20 are found, 2.1 ms after the `events.tsv` onsets.
- REQ-042 on the Mayo sets: **not met** (tests marked xfail). Agreement of
  `stim_site_guess` with `events.tsv` on 0.67 of the 162 detected pulses (0.72 on the
  2048 Hz runs, 0.63 per pulse before the vote; 0.77 with the `channels.tsv` table as
  `ch_info`). Confidence: accuracy 0.92 on the 22 % of pulses with `site_confidence >= 0.5`
  (0.95 on 25 % with `ch_info`). On ds004977 the agreement is 0: the stimulated contacts
  are nearly flat in the published files (SD 4 to 42 µV against about 460 µV for the other
  contacts), and the guess is the pair around them (RO2-RO5 for RO3-RO4, LA1-LA4 for
  LA2-LA3). ds003708 alone: 1.00 after the vote.
- REQ-081, ds004977 part: **met** (`tests/prep/test_reference_ds004977.py`). The 12 pulses
  of the uncensored site of one run per subject (sub-1 run-01, sub-2 run-12, sub-3 run-03,
  sub-4 run-02), sEEG channels, read with `read_bids(..., remote=True)`. Cross-channel R²
  (mean squared correlation of the trial average of a channel with those of the others,
  10 to 300 ms, bad channels and stimulated contacts left out): median 0.26 with `car`,
  0.18 with `car_adaptive` over 710 channels (one-sided Wilcoxon p = 1e-27); in each run
  p < 2e-5 (n* = 120, 168, 144 and 48 channels).
- Not done: REQ-372 (Jaccard index against the responsive channels of the published
  analysis) and the documentation gallery examples on the Mayo sets, now unblocked.
- Tests: a MEF 3.0 writer with RED compression in the test helpers (port of the meflib
  encoder: segments, gaps with and without the discontinuity flag, lossy blocks, NaN and
  infinity codes, encryption flags), 19 tests of round trips, windows, picks, segments,
  gaps, CRC errors, encryption, remote reads through the test HTTP server (with an
  emulated S3 listing) and `read_bids` (MEF equal to BrainVision, locally and remotely);
  network tests on ds003708 and ds004977 (`tests/io/test_mef_openneuro.py`).

### Stimulated site from held and leading contacts (REQ-042 met on ds004080)

- Diagnosis on ds004080 (25 runs, 199 pulses), where the step S4 rule agreed with
  `events.tsv` on 0.61 of the pulses (0.66 after the vote). The amplifier holds each
  stimulating contact, or leaves it at the rail (about ±3.2 mV), from the pulse to tens
  of milliseconds or seconds later, often until the next pulse, while the common
  reference spreads an artifact of 1 to 3 mV to every contact. So the artifact of a
  stimulating contact is capped by the rail, or is small when the contact is already
  held, and a quiet contact nearby (larger after division by its noise) or a contact
  that is not recording (a floating input with a large artifact) won. The stimulating
  contacts also move 1 to 3 ms before the onset of `detect_pulses` (the start of the
  artifact shared by most channels): the reference sample at -2 ms was already inside
  their artifact. Their first deflections are not always of opposite signs.
- `identify_stim_site` ranks the channels by evidence, then by artifact size. A channel
  that moves (artifact at least 5 % of the largest one) scores one point when it
  *leads* (before the onset it already moves by 10 noise SD and by 30 % of its
  artifact; the reference sample is now 5 ms before the onset, new `lead` argument)
  and one when it is *held* (from 5 to 100 ms after the onset, new `hold` argument, at
  least half of its steps are below a tenth of the channels' typical pre-pulse step).
  Without such evidence the rule is the largest artifact of the specification, now
  measured from -5 ms to +1 ms. A channel without noise of its own gets a tenth of the
  median noise (was a thousandth). `site_confidence` is the margin of the weaker contact
  of the pair over the third channel, on the evidence and then on the size
  (`(s_2 - s_3) / s_2` instead of `(a_1 - a_3) / a_1`, which stayed high when one
  contact dominated and the second was a toss-up), still halved by the polarity and
  electrode checks. For close pulses the lead window shrinks to a quarter of the
  interval (at least 2 ms) and the hold window ends before the next pulse. `vote_sites`
  is unchanged.
- ds004080 (25 runs of 21 patients, 199 pulses, from the data alone): agreement 0.980 per
  pulse, 0.995 after the vote (REQ-042: 0.95; the xfail is removed). The method was
  developed on the 13 even runs (103 pulses: 0.981 per pulse, 0.990 after the vote) and
  checked on the 12 odd runs, held out (96 pulses: 0.979, 1.000). Confidence: per pulse,
  accuracy 1.00 on the 81 % of pulses with `site_confidence >= 0.5` (0.78 on the even
  runs, 0.84 on the odd runs); after the vote, 0.994 on 80 %. The one error after the
  vote is the last pulse of a previous block (sTa6-sTa7), merged by the vote into the
  next block (sTa5-sTa6) because both share a contact. The four other per-pulse errors:
  two pulses where one stimulating contact shows no artifact at all (corrected by the
  vote) and two where an input that is not recording leads and holds like a stimulating
  contact (with the types of `channels.tsv` as `ch_info`, 0.990 per pulse). Halving or
  doubling any one threshold or window (the held fraction: 0.25 or 0.75) gives 0.94 to
  0.985 per pulse and 0.955 to 0.995 after the vote.
- Speed: `detect_pulses` with the site guess, 10 minutes of 100 channels at 2048 Hz, in
  0.79 s on the development machine (0.81 s before; `benchmarks/bench_events.py`).
- Tests: synthetic held and leading contacts, a saturated neighbour and a dead channel
  (`tests/events/test_stim_site.py`).

## 0.1.0 (in progress)

### Build step S16: reports, plots and the command line

- New module `evoparam.plots` (extra `plot`): `plot_fit` (data, model, components,
  background, residual drawn below the data, masked samples shaded, latency intervals as
  bands), `plot_group` (`grid`, `shaft`, `brain` with channel coordinates), `plot_trials`
  (trials sorted by latency and the latency histogram), `plot_existence` (statistic per
  channel and the response windows), `plot_tf` (total, footprint and residual maps with
  zones outlined and significant pixels contoured), `plot_shapes`, `plot_artifact`,
  `plot_compare`, `plot_hierarchical` (probability of a response and variance components).
  Every function returns the Figure, accepts `ax=`/`axes=`, never calls `show()`, uses the
  Okabe-Ito palette, and runs headless on every result type (REQ-344 met). The averages
  drawn by `plot_artifact` come from `TrialSet.average`, and the colour scales use the
  largest absolute value (no percentile computed in the plots). Deviation:
  `plot_existence` shows the statistic per channel; the statistic over time with its null
  band is not stored in `ResponseTest`.
- Methods text (`evoparam.reports.methods_text`, and `.methods_text()` of every result):
  - Built by `evoparam/data/_methods.py`, so that `ResponseFit.methods_text()` and
    `GroupFit.methods_text()` answer from the result alone (REQ-338: `provenance` and
    `settings` only; a saved result gives the same paragraph). It opens with the package
    version and the preset (name, version, validation note). It then gives one frozen
    template per recorded step: artifact (with the joint recovery tail, the template
    grouping "(those present)", the quality score), reference (it says when the adjusted
    common average fell back to the plain one), baseline, rejection, resampling,
    averaging, fit, existence test, trial fits, realignment, hierarchical model, variance
    components, time-frequency, shared shapes. Reading, simulation, selection and
    migration steps are not described.
  - It ends with the settings changed from the preset, the seed sentence (or the
    statement that no seed was set) with a pointer to `evoparam.reproduce`, the package
    citation and the citations of the methods used (Benjamini-Hochberg, rule sources). No
    path, date or duration enters the text. `reports.methods_text` also takes a
    `PipelineResult` or a list of results, and `cite=False` leaves the citations out.
  - The fit sentence says that the criterion is computed on residuals whitened by an
    autoregressive model of the baseline noise (since step S7). The preset fit records
    the names of the settings that differ from the preset (`settings["_overrides"]`).
  - `evoparam.existence.fit(existence=True)` records the test as a provenance step of the
    group and of every fit (new public `existence.attach_test`), and logs the paragraph
    at INFO (spec §5.13; `evoparam fit --verbose info` prints it).
  - REQ-340 met: `reports.run_pipeline(trials, preset)` runs the preset chain
    (`PipelineResult`), and `REFERENCE_PIPELINES` names the three reference pipelines
    (ERP CORE P3, ds004080 CCEP, simulated TEP), each with a simulated stand-in input.
    Their paragraphs are frozen in `evoparam/tests/reports/methods/*.txt`, with the
    package version written `{version}`; `tools/freeze_methods_text.py` regenerates them
    after a deliberate change of a template or preset (a preset change changes the frozen
    text). The text depends on the settings only. The network test runs the ds004080
    pipeline on the real recording (sub-ccepAgeUMCU01 run-021448, site PT01-PT02, 120
    channels) and gets the frozen words. The ERP CORE files are not on OpenNeuro and need
    MNE, so that pipeline is checked on its stand-in only.
  - The existence-test window of the pipeline starts after the samples masked in every
    trial: the tep preset's interpolation (artifact window to 12 ms plus its buffer)
    reaches into the fit window that starts at 12 ms, and the test refuses fully masked
    samples.
- `to_json(result)` / `from_json(...)`: the `.evoparam` container as one strict JSON
  document (structure, arrays with their dtype and shape, tables as CSV with their sidecar,
  provenance, settings). `validate_json` checks it against the shipped JSON Schema
  `evoparam/reports/schema/evoparam-result-1.json` (extra `validate`, jsonschema).
  REQ-341 met: the JSON validates and `from_json` reproduces `to_dataframe()` exactly.
- `qc(result, trials=, artifact=, trial_fits=)`: fits by status, warnings by type,
  ill-conditioned fits, masked fraction, artifact quality (from the new
  `ArtifactResult.quality_summary()`: median and 5th percentile), untrusted trials.
- `html(result, path)`: one self-contained file (figures as inline PNG, no network
  access), with summary table, quality, warnings, methods text and provenance; REQ-343 met
  (100 channels: well under 20 MB). Also `markdown(result)` and `pdf(result, path)` (text
  page, one page per condition, per-channel figures).
- `citation(bibtex=False)` reads `CITATION.cff`. The package holds a byte-identical copy,
  `evoparam/CITATION.cff`, rather than a symbolic link, which becomes a one-line text file
  on checkouts without link support; a test keeps the two identical (REQ-346 met).
- `reproduce(path, data=None, rtol=1e-9, atol=0, check_environment=True)` (also
  `evoparam.reproduce`):
  - It reloads a saved result (checksums verified, else reported) and replays the
    recorded steps from the recorded source: simulation, `read_bids`, `io.read`,
    container methods, artifact removal, reference, baseline, the group fit with the
    settings of `settings.json`, and the existence test. Every array, table and field of
    the new result is compared with the saved one.
  - The report gives the per-step status, the differing fields with their relative
    difference, the settings that differ between `settings.json` and the record, and the
    environment differences (with a warning).
  - It raises `EvoparamVersionError` when the recorded preset version no longer exists.
  - REQ-333 met: identical with `rtol=1e-9` on the simulated chain; a setting changed in
    `settings.json` is named on the fit step; `evoparam reproduce --json` returns the same
    report.
  - `reports.migrate(path)` rewrites a container under the current schema (written next
    to the old one, then swapped in).
- REQ-334 tested: `tests/data/test_schema_version.py` keeps frozen containers of schema
  major 1 (`tests/data/containers/`) that must load and equal their objects. It also
  checks a migration from an older major (recorded as `evoparam.data.migrate`) and that a
  newer major raises.
- `evoparam.io.write_bids_derivatives(results, bids_root, subject=..., ...)`: the
  derivative tree of spec §10.2. It writes `*_desc-evoparam_responses.tsv`,
  `_existence.tsv` and `_trials.tsv`, whose JSON sidecars give every column a description
  and a unit (`n/a` for labels and ratios), with the settings, preset and provenance. It
  also writes one `_artifact.json` per run (with every step when the preset chains
  several), `dataset_description.json` (`GeneratedBy` with the code URL and the
  provenance, REQ-336), `descriptions.tsv`, `README`, and a `.bidsignore` that declares
  the evoparam tables and saved containers, which have no BIDS suffix yet.
  - REQ-342 met: the derivatives and the simulated raw dataset pass the BIDS validator
    (`@bids/validator` 3.0.2) with no error. The test runs when `EVOPARAM_BIDS_VALIDATOR`
    names the validator command or `bids-validator` is on the PATH.
  - `evoparam.io.write_brainvision` writes BrainVision files.
- `evoparam.sim.bids.write_bids`: a simulated TrialSet as a BIDS dataset (trials end to end
  in one BrainVision recording, events, channels, sidecars with `iEEGReference`,
  electrodes and coordinate system for iEEG, participants, README) with the truth in
  `derivatives/truth`.
- The command line `evoparam` (also `python -m evoparam`; `argparse` only):
  - `fit`: the preset pipeline of `reports.run_pipeline`, the derivatives and the saved
    result.
  - `test`, `clean`, `report` (html/pdf/md/json), `simulate`.
  - `detect-events`: a BIDS events file, with `onset` and `duration` first.
  - `reproduce`, `migrate` (in place, or a copy with `--out`), `info`,
    `presets list|show`, `cite`, and `--version` with the research-use statement.
  - `benchmark` and `validate-bids` answer that they come with steps S19 and S17
    (REQ-314; exit code 1).
  - Common options: `--json` (one JSON summary on stdout), `--verbose`, `--n-jobs`,
    `--random-state` (default 0), `--config`. Exit codes 0, 1, 2, 3, 4, 5, 6, 130 as in
    §10.5.
  - REQ-345 met: every command runs through `subprocess` on a simulated BIDS dataset, and
    its files equal those of the Python API with the same seed. TSV files are compared
    byte for byte; JSON files are compared after removing the creation time and step
    durations. REQ-323 met: the exit codes are triggered and the `--json` summary
    parses. A corrupt `manifest.json` now raises `DataError`.
- Dependency table (§4.2, updated): the top layers are ordered
  `plots < reports < cli, benchmark`, without cycles. The spec had forbidden imports
  between them except `benchmark -> reports`, which would forbid figures in the HTML
  report and the `report` command. The import-graph test lists these edges.
  REQ-349 met: `tests/reports/test_layering.py` fails if `reports`, `plots` or `cli`
  import SciPy, call NumPy's statistical, linear-algebra, transform or fitting routines,
  or import a private module of an analysis package.
- `pyproject.toml`: console script `evoparam`. Extras: `validate` (jsonschema,
  bidsschematools, nwbinspector, as in §9.2), `cli`, `plot` (matplotlib >= 3.7), and
  jsonschema added to `all`. Package data: the JSON Schema, the citation copy, the frozen
  texts and containers.
- Not done: the stream module (`StreamFitter`, REQ-347, REQ-348: step S18), `write_nwb`
  and its provenance (REQ-336 for NWB, an expected failure until step S17),
  `validate-bids` (REQ-314, step S17), the benchmark (step S19).

### Build step S15, part 3: the `tep` preset on ds002094 (REQ-275)

- `evoparam/tests/validation/test_tep_ds002094.py` (EVOPARAM_NETWORK_TESTS=1): the 20
  subjects of OpenNeuro ds002094 1.0.0 (single TMS pulses over the right motor cortex, 600
  per subject, 30 EEG channels at 5 000 Hz). Each data file (0.6 to 2.4 GB) is downloaded,
  read with `read_bids`, interpolated over the artifact window at 5 000 Hz, resampled to
  1 000 Hz and deleted; the epochs (about 120 MB per subject) and the averages are cached in
  the evoparam data folder. The first run takes about 1 h 30 min (42 GB transferred), later
  runs a few minutes. Trials are removed as the dataset README asks (sub-767, sub-774,
  sub-780) and epochs that cross a recording pause (`New Segment`) are dropped.
- Method (written in the test before the pass rates were computed; criterion: components
  labelled N45 and N100 by the preset's fit on the channel of maximal response, the largest
  RMS of the average from 38 to 140 ms):
  - variant A, the preset's chain alone: interpolation (-2, 12) ms, subspace projection,
    average reference, baseline (-0.8, -0.1) s, mean of the trials; the channel is chosen
    among the 17 channels off the outer ring of the montage (without ICA, FP1/FP2, F7/F8,
    T7/T8, TP9/TP10, IZ, O1/O2 and P7/P8 carry the blink reflex and the cranial and neck
    muscle activity: on the development subject sub-283 the largest RMS was on FP1, FP2
    and T8);
  - variant B, A plus the ICA cleaning of the published TESA pipeline (Rogasch et al. 2017,
    [LIT-artifact-9]), re-implemented in the test from the paper with its default automatic
    thresholds: epoch demeaning, FastICA round 1 (TMS-evoked muscle: mean |amplitude| of
    the component average from 11 to 30 ms above 8 times its epoch mean), band-pass 1-100 Hz
    and 60 Hz notch, round 2 (TMS muscle, blink z > 2.5 at FP1/FP2, horizontal eye movement
    at F7/F8, electrode noise z > 4, persistent muscle: 30-100 Hz log-log slope above
    -0.31); the channel is chosen among all 30 channels (the literal rule).
- Result: **REQ-275 not met** (xfail with the measured values). Subjects with a labelled
  N45 / N100 / both, out of 20:
  - variant B, all channels (the criterion): 9 / 11 / 6 (30 %);
  - variant B, inner channels: 8 / 16 / 6;
  - variant A, inner channels: 11 / 13 / 9 (45 %), the best level reached;
  - variant A, all channels: 8 / 8 / 4.
  A floor test keeps these counts. The first run of variant B lacked the epoch demeaning of
  the published pipeline; on this DC-coupled recording its ICA then mixed offsets of
  thousands of microvolts into the data (a broken run, 5 of 20, not counted).
- Causes seen in the fits: after ICA the largest response is often on TP9, TP10 or IZ
  (neck muscle and electrode noise not removed by the published thresholds); the early
  negativity of this dataset often peaks between 55 and 75 ms, after the N45 window
  (38-55 ms) of the preset; and the fits often describe one deflection with two
  overlapping components of opposite sign (for example -24 and +52 uV at 77 and 78 ms),
  which take the label windows. The last point is an engine question (overlap of
  opposite-sign components), left to the robustness step S20; no engine or preset value
  was changed on these data.
- The methods text now carries the preset's caveat (`notes["methods"]`): for `tep`, that
  N100 and P180 may be contaminated by auditory and somatosensory responses to the coil
  click and the scalp sensation (REQ-275, second part, met; `presets.methods_note`). The
  `tep` preset's validation field states the result.
- Dataset registry 2026.10.08: ds002094 licence CC0 (its `dataset_description.json`), 5 000
  Hz, 20 subjects, 42.4 GB, events present, with its quirks (second run of three subjects,
  trials to remove, recording pauses). The real-noise licence test now uses Localize-MI as
  its example of a dataset without a recorded licence.

### Build step S15, part 2: validation layer 2 and the state of layer 3

- `validation/layer2.py` (spec §7.2): the layer-2 runner. Each cell simulates trials with the
  time scale, sampling rate and first shape of a modality's preset (erp, ccep, tep, erna,
  ecap, animal = rodent_ep), `K` components of a given width (samples) and separation (FWHM),
  SNR (peak / noise SD of the average), trial count, latency jitter (fraction of the FWHM)
  and noise family (white, AR(1) 0.5 and 0.9, pink 1 and 2), fits the average with the preset
  and scores it with `evoparam.sim.score_fits`; then the simulation-based REQ test files are
  run (with `EVOPARAM_SLOW_TESTS=1`) and reported passed or failed. Output:
  `validation/layer2/<version>/report.json` and `report.md`.
- Deviation from §7.2: the full factorial (276 480 cells, 1 000 replicates) does not fit in
  6 hours; the `weekly` design varies every factor one at a time around a base cell (SNR 5,
  two components 2 FWHM apart, width 12 samples, 20 trials, no jitter, white noise) for every
  modality, plus the SNR x separation map of F1 for erp and ccep: 204 cells x 50 replicates.
  The artifact factor is left to the artifact REQs (REQ-060 to REQ-069).
- REQ-370 met: the weekly design and the REQ tests take 28 min on the Mac Studio (10 cores;
  7 min of grid, 20 min of tests: 49 passed, 1 skipped, 0 failed). The weekly CI job is
  `.github/workflows/validation.yml` (Sundays, 6 h limit, report kept as an artifact).
  The report of this version is committed in `validation/layer2/0.1.0.dev0/`.
- What the grid shows (F1 of components; base cell unless stated):
  - erp, ccep, animal: F1 1.00 at SNR >= 5, about 0.98 at SNR 3, 0.5 to 0.66 at SNR 2,
    0 at SNR 1 (SNR of the average; the BIC rule with delta 10 is strict by design, REQ-112).
    Two components 0.3 FWHM apart are found as one below SNR 10 (F1 0 to 0.5).
  - ccep at high SNR: F1 0.84 at SNR 20 and 0.74 to 0.83 at 1 FWHM separation for SNR 10 to
    20 (an extra component; to examine with step S20, robustness).
  - Coloured noise lowers F1 to 0.74 to 0.80 at AR(1) 0.9 (erp, ccep, animal).
  - ecap: F1 0.92 at SNR 5 but an amplitude bias of +22 % and a width error of 53 %, and
    erna: width error 67 %: the widths of the biexponential and damped-oscillation shapes
    are not comparable with the simulated FWHM definition (to examine in S20).
  - erna: F1 falls with latency jitter (0.22 at a quarter FWHM): an oscillation averaged
    over jittered trials cancels.
  - tep: F1 0.55 with the shapes of the table: one `gabor` describes a pair of components
    of opposite sign (as N45 and P60), so the labelled sequence lost a component. Without
    gabor F1 1.00 (20 replicates). **Deviation**: the `tep` preset's shapes are now
    gaussian and skewed_gaussian (reason in its notes; `presets.derive` adds gabor back).
    After the change: F1 1.00 at SNR >= 5, 0.97 at SNR 3, 0.91 to 1.00 across noises.
- Layer 3, state:
  - REQ-275 (TEP on ds002094) **not met, documented**: first look on sub-728 (100 trials);
    the measurement on the 20 subjects, with and without ICA cleaning, is in part 3.
  - REQ-372 (ds004977, CARLA) **blocked**: the responsive channels of the published analysis
    are not in the dataset (its derivatives hold event annotations and FreeSurfer files) and
    not in the authors' repository (MATLAB code, GPL-3.0, results only as figures). A
    comparator needs their code run (MATLAB or Octave), not installed here.
  - REQ-274 (ERP CORE), REQ-278 (Allen Neuropixels, NWB reader of S17), REQ-373
    (Localize-MI) and REQ-374 (the layer-3 report format, with S16/S19 reports) remain open.

### Build step S15, part 1: presets

- New module `evoparam.presets` (spec §6): the eleven presets of §6.1 as JSON files
  (`evoparam/presets/*.json`, read with the standard library into the frozen `Preset`
  dataclass), `get`, `list`, `derive` (new frozen preset, the parent and the overrides
  recorded, the parent never changed), `register`, `to_json`/`from_json`, `to_yaml`/
  `from_yaml` (PyYAML), `fit_fields`, `methods_note`. Each preset has its citation keys,
  its version, its validation status and, per field, the reason of every value that is a
  choice of the specification. The module imports nothing from the package; every module may
  import it (import-graph test updated).
- One source of truth: `evoparam.models.FIT_PRESETS` is now built from the preset files, and
  `fit(..., preset=<name or Preset>)` reads them. `fit(..., fit_window=...)` is accepted as an
  override (the preset field name). The methods text names the preset, its version, its
  parent and overrides, and its validation status ("validated on simulations only" for
  `ecap`, `ecap_ci` and `erna`, REQ-279).
- Labels follow the table: `ccep` and `ccep_seeg` gain `delayed` (100 ms to 1 s, either
  sign); `sep` gains the tibial-nerve P37 and N45; the early window of `rodent_ep` and
  `primate_ep` is named `P1/N1` (was `early`).
- Deviation from the table: the `sep` background is `constant`, not `linear`. With a free
  linear drift, the N20 of simulated median-nerve SEPs (SNR 5, white noise) was described as a
  falling drift between two positive components and found within 0.5 ms in 9 of 20 runs; with
  a constant background in 20 of 20. A prior on the slope (drift of at most 3 baseline SD over
  the window) did not help (10 of 20) and was not kept. The reason is in the preset's notes.
- Engine changes for oscillating shapes: `FitSettings.freq_min`/`freq_max` reach the
  `damped_oscillation` and `gabor` shapes (the `erna` preset sets 200 to 450 Hz, spec §6.2);
  their initial frequency is the largest periodogram peak of the window within the range
  (it was the middle of the range, which left the fit in a wrong local minimum: no ERNA found
  in 7 of 10 simulated bursts at 4 096 Hz), and the damped oscillation starts a quarter period
  before the candidate peak with phase -pi/2.
- Results (`tests/presets/`):
  - REQ-270 met: a test parses the table of `docs/spec/SPEC.md` §6.1 and compares every
    field of every preset (the `sep` background deviation is listed and must be explained in
    the preset's notes).
  - REQ-271 met: every citation key resolves in `docs/spec/LITERATURE.md`; every choice has
    its reason.
  - REQ-272 met: `fit(..., preset="ccep", fit_window=...)` and `fit(..., preset=derive(...))`
    give identical components; `derive` leaves the parent unchanged.
  - REQ-276 met: `erna` on `simulate_erna` (channel at SNR 3; per-pulse epochs of the last
    five pulses of 20 bursts, whose amplitude is at least 95 % of its final value): frequency
    within 5 Hz and first-peak latency within 0.1 ms in 19 of 20 runs. With the simulator's
    default ERNA onset of 1 ms the first peak (about 1.6 ms) falls on the start of the fit
    window (1.5 ms) and the next peak was reported in 5 of 10 runs; the test uses a 2 ms onset
    (first peak at 2.3 to 3.2 ms).
  - REQ-277 met with white noise: N20 within 0.5 ms in 20 of 20 runs at SNR 5. With AR(1)
    noise at the same SNR, 16 of 20 (rho 0.5) and 8 of 20 (rho 0.8): the latency SD is about
    0.3 ms at rho 0.5 (documented limit, tested).
  - REQ-279 met; REQ-280 met: `docs/presets.md` is written by `tools/gen_presets_doc.py`, and
    a test fails when the committed page differs.
  - REQ-371 met: `tools/ds004080_subjects.py` applies the rule of §7.3 to OpenNeuro's
    listing and writes `validation/ds004080_subjects.txt` (dataset 1.2.4, SHA-256 of
    participants.tsv): sub-ccepAgeUMCU01 to sub-ccepAgeUMCU05 (sub-01 qualifies by the rule
    itself, with run-061233 at 2 048 Hz). The rule is tested offline on a fake listing.
- Not done in this part: REQ-273 (the N1 criteria, being worked on separately), REQ-274
  (ERP CORE: not redistributable, needs a local copy), REQ-275 (ds002094), REQ-278
  (Neuropixels LFP), the layer-2 grid (REQ-370) and the layer-3 comparisons REQ-372 to
  REQ-374.

### Build step S11: hierarchical model (trial -> channel -> subject)

- New module `evoparam.hierarchical` (spec §5.8): `fit_hierarchical`, `posterior_summary`,
  `compare_hierarchical`; imported by `import evoparam`. New extra `bayes` (numpyro, jax,
  PyMC, ArviZ); the default engine needs numpy and scipy only, and asking for a Bayesian
  engine without them raises an `ImportError` naming the extra (REQ-177).
- Inputs: one TrialSet per subject, or one TrialSet with a `subject` metadata column (one
  subject: the subject level is dropped, with a note). For each subject: the channel fits
  (§5.5, or `group_fits=`), then, per component, the trial fits (§5.7, shift and scale) with
  shifts up to 0.75 FWHM: with the default quarter FWHM of §5.7, trials of a 4 ms jitter
  reach the bound, are not trusted, and the jitter SD came out at half its value (2.2 ms
  for 4 ms); at 0.75 FWHM it is within 8 % at per-trial SNR 2 and 4. `n_jobs` runs the
  subjects in parallel processes (`-1`: one per core; scripts need the usual
  `if __name__ == "__main__":` guard of process pools on macOS and Windows).
- Empirical-Bayes engine (`engine="eb"`, default):
  - Existence: for every channel, the per-trial amplitude of the subject's template of the
    component (median shape of the subject's channels that carry it, or of all subjects
    when fewer than 3), least squares with an offset within +/- 1.5 FWHM; its channel mean
    and standard error follow a two-component mixture (no response: N(0, se²); response:
    N(m_s, tau² + se²)) fitted by EM with a per-subject mixing proportion and the subject
    means tied by a normal prior. `P(response)` is the posterior responsibility. The fitted
    amplitude is not used: a fit is only reported where a component was found, so it has
    no null distribution.
  - Shrinkage: REML on the nested model subject > channel with known standard errors
    (`evoparam.stats.variance.reml`, step S14). Channel statistics: the amplitude is the
    mean of the trusted trial amplitudes (latency jitter flattens the average: the channel
    SD of the amplitude came out 11 % low from the fits of the average); the latency is the
    fit's, with the standard error of the trusted trial latencies (the Wald errors of the
    fitted latency had z-scores of SD 1.2 to 1.5 on simulations, with some ten times too
    large, which shrank good channels away: EB beat the fits in 17 of 20 datasets with
    them, 19 of 20 without). Channels with fewer than 3 trusted trials get the subject value
    and the flag `prior_dominated`; a subject without usable channels gets the group value.
  - Jitter: REML on the trusted trial latencies with the latency noise of each channel
    (§5.7 null) as known error; the ex-Gaussian tau from the third moment.
  - Covariates: `formula="latency ~ age + state"` (subject-level metadata columns,
    constant within a subject; numeric or treatment-coded), in the REML design.
- Bayesian engines (`engine="numpyro"` or `"pymc"`): the same model in both, NUTS (4 chains,
  1 000 warm-up, 1 000 draws, target_accept 0.9), initialised at the EB variances.
  Deviation from §5.8 step 1: the likelihood is written on the statistics of the EB engine
  (channel estimates and their errors, trial latencies with their noise and an
  ex-Gaussian jitter, the existence mixture), not on every sample of every trial. Subject
  level non-centred, channel level centred (non-centred channels gave r_hat 1.02 with
  PyMC). Diagnostics: r_hat, ESS, divergences (warning above 1 %), posterior-predictive
  p-value of the spread of the channel latencies; `compare_hierarchical` gives PSIS-LOO
  on the channel latencies (ArviZ). `to_arviz()` is not implemented (the draws are not
  kept, only their summaries and the pointwise log-likelihood).
- Results (`tests/hierarchical/`; the reference runs of the Bayesian engines and the
  20-dataset versions of REQ-171, 172 and 175 run with `EVOPARAM_SLOW_TESTS=1`):
  - REQ-170 met: 20 subjects x 50 channels x 20 trials, 30 % responding, per-trial SNR 0.7:
    AUC 0.951 (at the limit), reliability-diagram slope 1.00.
  - REQ-171 met: per-trial SNR 1.5, channel latency RMSE of EB below that of the channel
    fits in 19 of 20 datasets.
  - REQ-172 met: EB between-channel amplitude SD and jitter SD within 20 % in 20 of 20
    datasets (ratios 0.85 to 0.97 and 0.92 to 1.00 on 12 printed ones; the amplitude SD is
    about 8 % low); NUTS intervals cover the channel latency SD and the jitter SD in at
    least 90 % of 10 datasets.
  - REQ-173 met on the reference dataset (10 subjects x 20 channels x 10 trials): numpyro
    r_hat 1.005, ESS 1 161, no divergence; PyMC r_hat 1.008, ESS 746, no divergence.
  - REQ-174 met: every posterior mean of the two engines within one posterior SD.
  - REQ-175 met: latency ~ age (0.2 ms/year), 20 subjects: the true slope is in the 95 %
    interval in at least 18 of 20 datasets.
  - REQ-176: EB 1.5 s per subject with `n_jobs=-1` on 10 cores (20 subjects x 50 channels
    in 30 s, of which 0.03 s per subject for the model itself); with `n_jobs=1` about 11 s
    per subject (9.4 s of channel fits of §5.5): **not met on one core**. NUTS: 48 s
    (numpyro) and 63 s (PyMC) per subject for 100 channels x 10 trials x 2 components
    (target 5 min).
  - REQ-177 met.
  - REQ-178 met in sign on ds004080 (20 subjects, one run and one site each, 7 or 8
    trials): N1 latency ~ age, -0.086 ms/year (95 % interval -0.29 to +0.12): shorter
    latencies with age, as the dataset's authors report; the interval includes zero.
- Not done: tutorials 40 to 42; covariates on the amplitude in the Bayesian engines (EB
  handles any of the three parameters); the spike-and-slab is on the template amplitude,
  separate from the fitted channel amplitude (spec: one amplitude).

### Build step S13: time-frequency parameterisation

- New module `evoparam.tf` (spec §5.10): `tf_fit`, `tf_zones`, `tf_compare`,
  `list_published_zones`, `methods_text`; `TFFit.methods_text()`. `evoparam.tf` and
  `evoparam.trials` are now imported by `import evoparam`.
- Transforms (`tf.transforms`): generalised Morse wavelets (gamma 3, beta = P²/gamma = 10 by
  default), complex Morlet, DPSS multitaper and STFT, all written as complex kernels applied
  in the frequency domain, so the coefficients are linear in the signal. Gain: a cosine of
  amplitude A gives a modulus A. Each kernel has a temporal support (the half-width holding
  99 % of its energy); a pixel is invalid when this support reaches an epoch edge (cone of
  influence) or a masked sample (the artifact mask widened at each frequency). Masked and
  missing samples are set to zero before the transform; invalid pixels are never filled and
  are left out of the statistics and the zones. Each frequency runs on the shortest padded
  grid that keeps its valid pixels free of wrap-around (4 times less work at high
  frequencies). Frequencies above sfreq/2.5 are dropped with a warning.
- Maps: `total`, `evoked_footprint`, `residual` and `cross` are trial-average **powers**,
  and `evoked_footprint + residual + cross == total` to 1e-9 relative for every pixel and
  every transform (REQ-211). Deviation from §4.3.4, which calls these maps
  baseline-normalised: the identity only holds for power; the normalised maps are
  `total_norm`, `residual_norm` and `footprint_norm` (the footprint on the total's
  baseline). The footprint is the transform of the channel's fitted model: components at all
  times, background inside the fit window with 5 ms cosine ramps (a cut background would put
  a step, broadband power, into both parts).
- Normalisation (`db`, `percent`, `zscore`, `logratio`, `ratio`) of each trial by its own
  baseline (`single_trial`, default), of the trial-average power (`evoked`), or of each
  trial by its full-epoch mean (`full_epoch`; with `zscore`: `SettingsError`). For
  `single_trial`, the reported map is the mode applied to the trial mean of the power
  ratios (Grandchamp and Delorme 2011): the mean of per-trial dB values is about -2.5 dB on
  noise (mean of a log below the log of the mean).
- Statistics: the tested quantity is not the normalised map but, per trial, the cube root
  of the power minus its mean over the baseline; one-sample t across trials; then
  Benjamini-Hochberg over the pixels of the window, cluster mass with sign flips, or TFCE
  (E 0.5, H 2, step 0.2 in t) with sign flips. Measured on null AR(1) trials: a t-test on
  dB values gave 20.5 % of maps with a false pixel under FDR, all positive (dB values are
  left-skewed); dividing by the trial's own baseline (a noisy divisor correlated with the
  subtracted baseline) gave 17 % family-wise errors with clusters (Jensen bias). With the
  cube root of the power (Wilson-Hilferty: nearly symmetric) and no division, the test does
  not depend on the mode or the level, and the errors are within the target (below).
  `channel_correction="bonferroni"` divides alpha by the number of map rows.
- Zones: the positive and the negative parts of the significant group map (mean over
  channels and conditions of the significant normalised pixels) are factorised separately by
  NMF (frequency profile x time course) or by a non-negative CP decomposition over
  (frequency x time x map row). Rank: the largest one whose parts are reproduced (mean
  matched Tucker congruence >= 0.8) between two halves of the trials, over 20 random
  splits (NMF), or across 20 restarts (CP: deviation, split halves of every row would need
  20 x 2 maps per row in memory); the search stops after two unstable ranks in a row (a
  rank-1 factor of two blobs of similar weight picks either one, so rank 1 can be unstable
  while rank 2 is not). Each significant pixel goes to the zone that explains most of it;
  touching zones with no dip between them (pass at least half of the smaller peak) are
  merged: NMF otherwise cut one burst into 2 or 3 stable pieces (recovery 0.90 instead of
  1.00 below). Zone table: bounds, time and log-frequency centroids and spreads, peak,
  mean, onset and offset at the centroid frequency, area (s x octave), share of the
  significant energy, stability, p-value (smallest trial-level p over rows, Bonferroni).
  Zone metrics per channel and condition (and per trial with `by_trial=True`, `trial` index
  joinable with `fit_trials`): signed mean of the normalised map, trial-level t and p, share
  of significant pixels, of significant increases and of significant decreases.
- Published zones (`tf/zones.json`, `zones="published:<name>"`, which also sets the
  transform and the baseline of the source): the delayed 50-500 ms x 50-250 Hz zone
  [LIT-tf-3] with the Morse transform and the -500 to -100 ms baseline of [LIT-tf-2]
  (bounds, transform and baseline marked verified as in the spec; dB mode and level marked
  unverified). The ten other zones of the eleven-zone scheme are **not entered**: their
  bounds are not in the spec or the literature file (the article [LIT-tf-4] is
  unverified), so the registry has one `unverified` entry without bounds instead of
  invented numbers; it raises `SettingsError` without `allow_unverified=True` (REQ-217) and
  again with it (no bounds). The bounds must be copied from the dissertation.
- `tf_compare(tf_a, tf_b)`: contrast with the map rows as units (sign flips of paired
  differences when the rows are the same, label permutation with Welch t otherwise),
  clusters with permutation p-values or FDR.
- Results (`tests/tf/`, `benchmarks/bench_tf.py`):
  - REQ-210 met: the Morse coefficient of a cosine equals the analytic value
    (half the wavelet's gain at that frequency) to 1e-6 relative at 4 signal frequencies x
    14 kernels; the chirp ridge follows the instantaneous frequency within 0.1 octave; the
    cone of influence is masked. No independent package (ssqueezepy) is installed: the
    reference is the closed form.
  - REQ-211 met: exact to 1e-9 for the four transforms.
  - REQ-212 met: sharp N1 (FWHM 8 ms, -60 µV, AR(1) noise of SD 10 µV, 30 trials, 40
    runs): significant high gamma in the total map in 100 % of runs, in the residual map in
    2.5 %.
  - REQ-213 met with an interpretation: "20 % power increase" is taken as a burst adding
    20 % to the signal power over its 100 ms (amplitude 10.3 µV for noise of SD 10 µV):
    frequency centroid within 10 % and time centroid within 20 ms in 20 of 20 runs (40 of 40
    in a longer check; median errors 4 % and 1 ms). Read as +20 % of the local 80 Hz power
    (0.8 dB, amplitude 1.3 µV), no method can find it pixel by pixel with 30 trials (pixel
    t about 0.8): the burst box has a significant pixel in at most 10 % of runs (test kept
    as a documented limit).
  - REQ-214 met: maps with any false pixel on null AR(1) trials: FDR 2.5 % (200 runs; the
    pixel-wise rate is 5e-5), cluster 3 % (100 runs, 200 permutations), TFCE 1 % (100 runs,
    200 permutations, step 0.2).
  - REQ-215 met: one trial with 100 times the noise power changes the zone value by a
    median 6 % (at most 8 %) of the change with `evoked` normalisation.
  - REQ-216 met on ds004080 (6 subjects, 7 or 8 trials, one site per subject, all
    recording channels): the published summary statistic, the share of significant
    increases in the delayed high-gamma zone, rises with the N1 amplitude across channels:
    Spearman rho 0.65, 0.35, 0.40, 0.61, 0.19, 0.63 (|rho| >= 0.3 in 5 of 6, all positive).
    The mean dB over the whole zone relates less consistently (rho -0.36, -0.23, 0.16,
    -0.38, 0.22, 0.42; mixed signs).
  - REQ-217 and REQ-219 met.
  - REQ-218 met: 100 channels x 50 trials x 2 048 samples x 40 frequencies, FDR and zones,
    in 33 to 43 s on one job (target 60 s).
- Not done: `TFFit.to_mne()` (step S17, ecosystem); tutorials 60 to 62; the ECoG-versus-sEEG
  comparison (needs an ECoG set in the registry and the modality covariate of S14). TFCE
  is slow (about 1.5 s per map row with 100 permutations).

### Build step S10: trial-by-trial parameters

- New module `evoparam.trials` (spec §5.7): `fit_trials` returns a `TrialFitSet` (one
  `TrialFit` per channel and condition, a tidy table with the trial metadata, skipped
  channels, save/load). Modes `shift_scale` (default: per-trial shift and amplitude of each
  component, variable projection), `free` (every parameter within +/-25 %), `template` (the
  component's segment of the channel average as template, Woody style) and `regularized`
  (ridge towards the channel values). Also `jitter`, `realign`, `dose_response` (Hill or
  linear, threshold current) and `prestimulus_dependence` (phase from three estimators with
  its circular SD, power, first-harmonic fit and permutation test); covariates `trend`,
  `dose`, `prestim`.
- Choices and changes to the specification:
  - the trial model always keeps every component of the channel (a complete model);
    `components=` only selects what is reported;
  - the r² of a trial (and so `trusted`) is computed around the reported components (from
    one width before their onset to one after their offset), not over the whole window,
    where ongoing activity dominates; a trial is untrusted when a *reported* component's
    shift is at its bound (a neighbour held by a tight bound no longer counts) — together
    these raised the trusted fraction on ds004080 from 0.14-0.43 to 0.86;
  - `jitter_sd_null` is estimated from five segments of each trial's baseline, with the
    same trust rule as the trials, and only the reported components moving;
  - realignment is refused unless the jitter exceeds both 1.2 x the noise-only jitter and the
    95 % quantile of the ratio of two noise SDs (F distribution);
  - `jitter()` adds `jitter_sd_excess = sqrt(jitter_sd^2 - jitter_sd_null^2)`, the best
    estimate of the true jitter when noise inflates the raw SD;
  - `regularized`: the penalty is chosen by split-sample prediction error (fit on even
    samples, score on odd), not by leave-one-trial-out, which would refit the channel model
    once per trial.
- Acceptance criteria met: REQ-150 at a per-trial SNR of 3 (latency error 0.94 sample,
  amplitude 9-10 %, excess jitter within 20 % at 2 and 5 ms; at SNR 2 the Cramer-Rao bound
  of a 20 ms component at 1 kHz is about 1.2 samples, so "<= 1 sample at SNR >= 2" cannot be
  reached; at 1 ms of jitter the error is 32 %, the jitter being as small as the noise's),
  REQ-151, REQ-152, REQ-153, REQ-154, REQ-155, REQ-157, and on ds004080 the trusted fraction
  of REQ-156 (median 0.86 over 87 N1 channels of six subjects).
- Not met: REQ-156 speed (0.3 to 4.5 s per subject; 2 s up to about 12 N1 channels; xfail);
  REQ-158 needs the 1 Hz repeated-pulse reference dataset of spec §7.

### Build step S8: is there a response?

- New module `evoparam.existence` (spec §5.6): `test_response` with four methods:
  `baseline_permutation` (default; pointwise t of the trials summarised by the largest |t|
  or the largest cluster mass, random sign flips of whole trials, every pattern when there
  are few trials; onset, offset and duration with a bootstrap interval; for an Evoked, a
  weaker baseline-window null with a warning), `projection` (cross-projection test and
  response duration, re-implemented from the published description of the canonical response
  parameterisation; the GPL code is not used), `model_comparison` (change of BIC with a
  parametric bootstrap) and `threshold` (the registry of published rules, `rules.json`:
  vanblooijs2023, hays2021, keller2014, silverstein2020, ftract, prime2020, custom; a field the
  paper does not state must be supplied, and the error names it and the paper).
  Corrections `fdr_bh`, `holm`, `bonferroni`, `none` across channels and stimulation sites;
  `response_duration`, `compare_detectors` (pairwise kappa, with the human reference 0.60),
  `train_discordance`, `list_rules`, `methods_text`, `detectors.<rule>(...)`.
- `evoparam.fit(..., existence=True)` (now `evoparam.existence.fit`) attaches the test to
  every channel's fit; `ResponseFit.to_dataframe()` adds `existence_p` and
  `existence_significant`, so a component on a channel without a significant response is
  shown, not hidden.
- Addition to the specification: permutation p-values use a generalised Pareto tail
  (Knijnenburg et al. 2009) when fewer than 10 null values reach the observed one. With 8
  trials there are 256 sign patterns, so the smallest plain p-value (1/256) never survives a
  correction over a hundred channels: on ds004080 the test detected nothing on three of six
  runs before this change.
- Acceptance criteria met: REQ-130 (false-positive rate within [0.8 alpha, 1.2 alpha] on 2000
  null channels, white and AR(1) 0.9 noise, both statistics), REQ-131 (power 0.9 or more;
  the advantage of the cluster statistic shows at SNR 0.7 with 10 trials; at the spec's SNR 1
  both statistics reach it), REQ-132 (duration within 10 %, onset within 2 ms), REQ-133,
  REQ-135 (agreement with vanblooijs2023 on six ds004080 subjects: kappa 0.76 pooled, 0.85
  median per run), REQ-136, REQ-137, REQ-138 (1.8 s for 100 channels x 50 trials x 1000
  permutations), REQ-139, REQ-140.
- Not done: REQ-134 (projection against the published numbers of ds003708) needs MEF3
  reading; the projection test is checked on simulations instead.

### Build step S7: the fitting engine

- New module `evoparam.models` (spec §5.5): `ResponseModel` (settings in `__init__`,
  `fit_waveform`, `fit_group`, `fit`, `predict`, `transform`, `score`, `get_params`,
  `set_params`), `evoparam.fit(data, preset=...)` (also `evoparam.fit`), `fit_response`,
  `classical_measures`, `label_components`, `compare_models`, `methods_text`,
  `evaluate_fit`, and the fit fields of the eleven presets of spec §6.1 (`FIT_PRESETS`; the
  full presets are build step S15).
- Engine: the model `y = background + sum_k A_k s_k(t; theta_k)`; amplitudes and background
  coefficients by variable projection, the other parameters by bounded least squares with
  analytic Jacobians; components added one at a time from the extrema of the residual (window
  edges included) and kept only when the criterion (BIC by default, AICc, cross-validation over
  trials, or a fixed number) improves by `delta` (10); merge-and-compare of overlapping
  neighbours; nothing deleted by a threshold, a width, an edge or an overlap rule. Results:
  features from the fitted curves, Wald and delta-method intervals (`"jacobian"`), bootstrap
  over trials, profile likelihood; covariance, condition number and sloppy pairs
  (`"ill_conditioned"`), parameters at a bound, goodness of fit, classical measures with their
  definitions, status `ok` / `no_response` (with the best rejected change) / `skipped` /
  `failed` (with a message), provenance. `ResponseFit` gains a `diagnostics` field.
- Changes to the specification, found on simulations and on ds004080:
  - noise: instead of deflating the sample count to `n(1 - rho)/(1 + rho)` (on ds004080, lag-1
    autocorrelation 0.98-0.99 leaves 1 or 2 effective samples, so not even a -500 uV N1 could
    be accepted), the residual is prewhitened with an autoregressive model of the baseline
    (Yule-Walker, order by AIC up to 10), and the criterion uses the number of whitened
    samples (`noise_model="ar"`; `"white"` shows the false components that whitening prevents);
  - three soft priors against degenerate decompositions (a negative N1 described as a negative
    offset plus positive components, or two huge components that cancel): the background
    constant is drawn to the baseline level with SD 3 baseline SD (`background_prior`); every
    other coefficient has SD 3 times the data range (`amplitude_prior`); a component's location
    stays within 5 % of the window length outside the window (`edge_extension`); and the CCEP
    presets cap the background time constant at 50 ms (`background_tau_max`);
  - the order of components is not a constraint inside the optimisation: they are sorted by
    latency after each step (the components are exchangeable in the model);
  - search: the first qualifying candidates are compared (three), and splitting a component
    into two narrower neighbours is proposed at each step; with a forced number of components
    (fixed, cv) merges are not tried (they undid the additions in an endless loop).
- `skew_mode` (shapes): safeguarded Newton instead of bisection, and a small cache: the same
  roots to 1e-15, about 15 times faster fits.
- Acceptance criteria met on simulations: REQ-103, REQ-104, REQ-105 (recall and precision
  0.99), REQ-106 (latency error under 0.5 sample, amplitude bias under 1 %), REQ-107 and
  REQ-112 (no component in 100 % of white and AR(1) 0.5 / 0.9 noise windows; without
  whitening at least 30 % false components), REQ-108, REQ-109, REQ-110, REQ-111 (two close
  peaks separated in 95 % of runs), REQ-116 (interval coverage), REQ-118, REQ-119, REQ-120,
  REQ-121, REQ-122 (cross-validation picks the true number in 10 of 10), REQ-123. Real
  baselines of ds004080 (REQ-112): no component in 95 % of 80 pre-stimulus windows of two
  runs (87 to 96 % per run over six runs).
- **Not met**: REQ-113 and REQ-114 (agreement with the published N1 rule on ds004080 sub-01):
  kappa 0.48 and latency within 2 ms on 14 % of the shared channels, on the 8-trial cached
  segment. Inspection channel by channel: several of the rule's detections are the artifact
  residue at 9.8 ms (the first sample of the window) or a slow drift at 100 ms; the model finds
  clear small N1s (-30 to -45 uV) under the rule's 50 uV floor; and it misses 4 of the rule's
  11 channels. The test is marked xfail; the floor test (at least 8 channels with an N1, where
  the inherited engine found 2) passes. REQ-115 (speed): about 0.2 s per waveform instead of
  50 ms (xfail).
- The example now re-references to the adjusted common average and fits with the `ccep`
  preset: an N1 with its uncertainty on 10 channels.
- The inherited engine (`objs`, used by `param`, `plts`, `analysis`) is kept until the plots
  and reports are rewritten (build step S16), instead of being deleted at the end of S7.

### Build step S6: light preparation

- New module `evoparam.prep` (spec §5.4):
  - `baseline_correct` (mean, median or linear trend; masked samples left out, error under
    50 % valid samples). With `event_times`, a warning when the baseline starts less than
    `delayed_margin` (1 s) after the previous pulse, with the shortest interval and a valid
    alternative: a shorter baseline when one exists, else `baseline_source="first_pulse"`
    (every pulse uses the baseline of the first pulse of its train).
  - `rereference`: `monopolar`, `bipolar` (consecutive contacts per electrode, gaps
    reported, names and order of `mne.set_bipolar_reference`), `car` (stimulating, bad,
    excluded and masked channels left out, per trial and sample), `car_adaptive` (adjusted
    common average by least anticorrelation, re-implemented from Huang et al. 2024; falls
    back to `car` under 8 channels), `laplacian` (neighbours within a radius). The reference
    is written to `ch_info["reference"]`; `car_adaptive` records `n_star`, `zeta`, the
    channels used and the bootstrap interval. `car_adaptive_select` is public.
  - `reference_sensitivity`: detections and latencies under several montages, Cohen's kappa
    and latency differences per pair (a bipolar `A1-A2` is compared with contact `A1`). Its
    default detector, `threshold_detector` (minimum below -3.4 baseline SD, SD floored at
    50 uV, the ER-detect default), is interim until the existence test of build step S8.
  - `reject_trials`: robust z-scores of peak-to-peak and RMS in the response and baseline
    windows, flat channels, masked fraction, quality; flagged trial-channels are masked, a
    trial flagged on half of its channels (or for quality) gets `rejected` and
    `reject_reason`; at most 30 % of the trials per channel, with a warning above. Rows are
    never deleted.
  - `resample` (polyphase, `event_times` exact, masks to the nearest sample, t = 0 kept on
    the grid), `average` (mean, 20 % trimmed, median, quality-weighted), `highpass`,
    `lowpass`, `notch` (zero-phase; warning on masked data).
- Adjusted common average, choice of n*: the global maximum of zeta, unless an earlier local
  maximum is followed by a significant fall (bootstrap over trials, to the next low point).
  Taking the first significant local maximum anywhere, as first written, chose a later peak
  after several responsive channels in 4 of 5 simulations at 50 % responsive channels.
- `contact_group` moved to `evoparam.utils._names` (shared by `events` and `prep`).
- Acceptance criteria: REQ-080 (1 Hz design warns and proposes a valid alternative, none at
  0.2 Hz); REQ-081 simulation part (50 channels, 12 trials: median false negatives 0 and
  false positives at most 1.5 from 0 to 90 % responsive channels, 10 simulations per level);
  REQ-082 (bipolar equal to MNE); REQ-083 (30 corrupted trials of 200 all flagged, no clean
  trial rejected); REQ-084 on ds004080 sub-01 (three schemes in 0.9 s; kappa 0.64 to 0.71
  between montages, latency differences up to 57 ms on the same channel); REQ-085 (latency
  within half a sample at 500, 512 and 2048 Hz, `event_times` unchanged).
- Not done: the ds004977 part of REQ-081 (MEF3 recordings cannot be read yet); a channel
  whose every trial is rejected cannot occur with the 30 % cap, so no automatic `bad` flag.

### Build step S9: simulators with a known truth, noise models and scorers

- New module `evoparam.sim` (spec §5.12), rewritten; it replaces the skeleton's `sim/*`.
  Generators, each returning `(TrialSet, GroundTruth)`, deterministic for a `seed`:
  - `simulate_erp`: scalp ERPs from the ERP presets `P1`, `N1`, `N170`, `MMN`, `N400`, `P3`
    and `ERN` (or `ComponentSpec` objects or dicts). It supports latency jitter (normal or
    ex-Gaussian, shared or per component), log-normal amplitude variability, and overlap
    with the previous and next events (role `overlap`, event -1 and +1).
  - `simulate_ccep`: sEEG shafts (3.5 mm) or an ECoG grid (10 mm) and a stimulation pair.
    Responders carry N1 (15-40 ms, FWHM 5-15 ms), N2 (80-250 ms, FWHM 40-150 ms) and,
    optionally, delayed responses (0.1-1 s). Other features:
    - the dipole artifact `∝ 1/r+ − 1/r−` × the rectangular pulse through a 4th-order
      Butterworth anti-aliasing filter, plus a double-exponential recovery tail;
    - saturation at the rails, with masks;
    - trains, recorded in `train_id` and `index_in_train`, with adaptation;
    - the stimulation metadata of Table 4.1.
  - `simulate_erna`: damped oscillations of 200-450 Hz after every pulse of a train, which
    build up over the train, plus the pulse-train artifact; at least 4096 Hz.
  - `simulate_tep`: the N15 to P180 components around the stimulated site and the
    auditory N100/P200 (role `confound`). The artifact has three parts: pulse, decay and
    muscle.
  - `simulate_ecap`: spinal or cochlear biexponential N1/P2 at 30 kHz. It supports the
    alternating-polarity and forward-masking (A/B/C/D) paradigms, with masker components
    (role `masker`) and the switching artifact.
  - `simulate_animal_ep`: a laminar probe with a depth-reversing profile and states
    (`awake` ×1, `nrem` ×2) as conditions. An electrical stimulus adds an artifact.
  - `inject`: adds simulated components to real recordings.
- Shared building blocks:
  - `make_noise`: white, AR(1), pink (`1/f^chi`), line (harmonics) and real noise, and
    their sums such as `"pink+line"`.
  - `shape_params`: places any shape by latency and FWHM.
  - `ComponentSpec`, the presets (`ERP_COMPONENTS`, `TEP_COMPONENTS`, `CCEP_N1`,
    `CCEP_N2`, `CCEP_DELAYED`) and the layouts.
- SNR is the peak amplitude over the per-trial baseline noise SD (t < 0), averaged over
  trials, on each channel. `snr` is a scalar, one value per component, or a dict keyed by
  label or channel. The SNR is reached on the channel of largest weight and scaled by the
  weight elsewhere. An SNR asked for a channel without a component raises `SettingsError`.
  `snr=None` keeps the amplitudes in µV.
- `GroundTruth` holds:
  - every component with its nominal parameters, features, channel, role, event and SNR;
  - per trial, its latency shift, amplitude scale and presence;
  - the conditions, the noise and artifact records, the seed and every setting;
  - optionally, the noise-free signal and the artifact.

  `to_dataframe("component" | "trial")` uses the column names and units of
  `GroupFit.to_dataframe()`, followed by `trial`/`n_trials`, `role`, `event`, `snr`,
  `shift`, `scale` and `param_<name>`. Other methods: `trial_params`, `waveform`, and
  `to_json` / `from_json` (strict JSON).
- Scorers:
  - `match_components`: Hungarian matching per waveform. The default tolerance is half the
    true FWHM; it also accepts `"fwhm/k"`, `"k*fwhm"`, seconds or None. Options are roles,
    window and polarity. The input can be a GroupFit, a ResponseFit or a table, at the
    component or trial level.
  - `score_fits`: recall, precision and F1; latency MAE and bias (s and samples);
    amplitude and FWHM relative errors; interval coverage; false components per waveform
    and the false rate on waveforms without a component (K = 0); optionally grouped
    (`by=`).
- Acceptance criteria met:
  - REQ-250: byte-identical reruns for every generator with an int seed, `None` or a
    Generator (the seed used is recorded). Fingerprints for seed 2024 are committed in
    `tests/sim/reference_sim.json` and compared at rtol 1e-9.
  - REQ-251: pink slope over 2-100 Hz, mean Welch spectrum of 100 realisations, for chi 0,
    0.5, 1, 1.5 and 2: -0.005, 0.498, 1.004, 1.512 and 2.024. AR(1) lag-1 for rho 0, 0.5,
    0.9 and -0.5: 0.000, 0.499, 0.898 and -0.500.
  - REQ-252: the measured SNR matches the request within 3e-5 relative for every generator
    at SNR 2 and 5, also with jitter, amplitude variability, every noise model and real
    noise.
  - REQ-253: `remove_artifact(method="template")` on the default `simulate_ccep` leaves a
    residual RMS in (-2, 8) ms of 1.2 % of the RMS before removal (seeds 0-2): 7.8 % on
    the non-stimulated contacts, and 3.9 % with saturation (clipped samples left out).
  - REQ-254: `sim` reaches none of `models`, `objs`, `param` or the legacy fitter, checked
    by a file-level transitive import test.
  - REQ-255: shared columns, units and dtypes for every generator and both levels.
  - REQ-256: every real-noise segment records the dataset, version, DOI, licence, subject,
    session, run, source trial and channels, and its offsets in the epoch and in the
    recording. Licences other than CC0, CC BY or CC BY-SA in the registry are refused.
- Choices where the specification leaves room:
  - `sim` imports `evoparam.io` (the licence registry); the import-graph rule now allows
    `sim → io`.
  - REQ-250 is met as byte-identical output between runs on one machine and agreement
    within 1e-9 across platforms (the committed fingerprints). No checksum is committed:
    the bytes differ in the last bits between CPUs (SIMD paths), even on Linux x86_64 with
    the same numpy and scipy, and between Linux and macOS.
  - Pink noise is shaped in the frequency domain with `scipy.fft` over a twice-longer
    segment and scaled to the exact SD. AR(1) starts from its stationary state.
  - Real noise is cut from a `TrialSet` given as `source`, by default its pre-stimulus part.
    Nothing is downloaded. The registry's ERP CORE and ds002094 entries have no licence
    recorded, so they are refused. Other rates are resampled with
    `scipy.signal.resample_poly` (the preprocessing step S6 does not exist yet), and the
    resampling is recorded.
  - `inject` accepts recordings that name no dataset (the user's own data), recorded with
    `dataset = None`. A named dataset must pass the licence rule.
  - The truth's `"component"` level is the expected component of the condition's average:
    latency moved by the mean shift, and amplitude × the mean scale, a trial without the
    component counting as 0. The smearing of the average by jitter is ignored. Trial-level
    rows are exact. Features are those of the whole curve, also when a large shift moves
    part of it outside the epoch.
  - Relative errors in the scorers divide by the true value, so a fitted magnitude larger
    than the truth gives a positive error whatever the polarity. `score_fits` scores every
    role within the epoch by default (`roles=None`).
  - The ERP preset ranges follow ERP CORE windows where ERP CORE defines the component.
    P1 and N1 are not ERP CORE components and use usual visual ranges. Widths and
    amplitudes are typical values.
  - The generators take more keyword arguments than §5.12 lists, among them `noise`,
    `snr`, `noise_sd`, `jitter`, `tmin`/`tmax` and the artifact sizes. `simulate_erna`
    refuses `sfreq < 4096`.
  - `GroundTruth` is saved as JSON (`to_json`), not as a `.evoparam` container.
  - Without pandas the generators still run (`metadata` and `ch_info` are None). The tables
    raise an ImportError that names the `data` extra. The ECAP paradigms need pandas.
- The skeleton's simulator (`sim.gen`, `sim.params`, `sim.utils`) moves unchanged to
  `evoparam.core.sim` for the inherited fitter until step S7; its tests move to
  `tests/core/test_legacy_sim_*.py`. It has no compatibility alias, because the package is
  unreleased and the name `simulate_erp` is reused.
- `neurodsp` is no longer a dependency: the `sim` extra is gone. The inherited noise is
  reproduced bit for bit with numpy, with the same draws from the global random state.

### Build step S5: stimulation-artifact removal with a quality score

- New module `evoparam.artifact` (spec §5.3), replacing the skeleton's `evoparam.clean`
  (removed; its interpolation is method 1 and lives in `evoparam.utils._interp`, shared with
  the legacy `utils.data.interpolate_signal`).
- `remove_artifact(ts, method=...)` returns an `ArtifactResult` (cleaned `TrialSet`, method,
  window, template or basis, per-trial quality and its components, every setting, provenance).
  Methods, each from the published equations:
  `interpolate` (linear, cubic, PCHIP); `template` (group median per site, current and
  polarity, least-squares scaled per trial and channel; `"local"` moving template of the
  nearest pulses; or a given array; masked pulses left out; warning when the template
  correlates with the response after the window; groups under 3 pulses raise, or fall back
  to interpolation); `tail` (two exponentials fitted after the window, `tau2 >= 3 tau1`,
  the decaying part subtracted; `"joint"` leaves it to the response model);
  `period_template` (fractional period by a three-stage grid search, then every sample minus
  the mean of the samples whole periods away; causal variant); `lowrank` (sequential
  regression across channels, pulses and trials); `subspace` (whitening with the baseline
  covariance, projection away from the artifact directions); `polynomial` (local cubic as a
  Savitzky-Golay filter, zeros while railed, one-sided fits after, masked until the residual
  looks like noise); `matching_pursuit` (line, response and artifact atoms; patent warning);
  `paradigm` (alternating polarity, forward masking; also `paradigm_subtract`); `learned`
  (PCA basis per group, ridge regression; basis returned); `none`.
- Masks: interpolated and zeroed samples are masked, with the existing (saturation) masks.
  Methods that subtract an estimate keep the corrected samples as measurements and do not
  mask them (a choice of this step: masking them would discard what template subtraction
  recovers).
- Quality per trial (`artifact_quality`): residual energy after the window, edge
  discontinuity, saturation fraction and template-fit r2; written to `metadata["quality"]`.
- Also: `estimate_artifact_template`, `estimate_period`, `apply_mask`, `compare_methods`
  (RRMSE and signal-to-interference ratio against a ground truth).
- Two additions to the published methods, found necessary by the tests: in `lowrank`, the
  across-channel principal-component scores are denoised across pulse segments before the
  regression (with 24 channels, raw scores inject the other channels' noise: silent-channel
  RMS +35 % instead of <= 10 %); in `tail`, the subtracted part is `r(t) - r(t_end)`, so the
  cleaned signal joins the untouched samples without a step.
- Acceptance criteria met on synthetic overlays (REQ-060 to REQ-067, REQ-069): locality and
  masks; >= 20 dB artifact reduction in the window for `template` and `learned`; period
  estimate SD below 1e-4 samples over 100 realisations and residual indistinguishable from
  the pre-stimulation noise (Wilcoxon p > 0.05) for a 130.2 Hz train with 10 and 50 Hz
  sinusoids; spike waveform correlation >= 0.95 and silent channels within 10 % for
  `lowrank`; >= 30 dB signal-to-interference improvement for `subspace`; recovery 2 ms after a
  railed pulse at 25 kHz for `polynomial`; quality AUC >= 0.9 for corrupted trials;
  alternating polarity exact to 1e-9 and the forward-masking identity; settings re-create the
  result to 1e-12.
- Not covered by this step: the latency and amplitude bias of *fitted* components after
  cleaning (REQ-061, second part) needs the fitting engine (build step S7); MEF3 data cannot
  be read yet.
- Speed (REQ-068, `benchmarks/bench_artifact.py`): 50 trials x 128 channels x 3 s at 2048 Hz
  in 0.5 to 1.3 s per method with the quality score, `subspace` 3.7 s and `lowrank` 8.6 s
  (limits 2 s and 20 s), on one core of the Mac Studio.
- The ds004080 example now subtracts a template; on that run, with a median reference and a
  window to 14 ms, the inherited fitter finds an N1 on 6 channels instead of 1. This is a
  diagnostic only: the N1 fix itself belongs to build steps S6 (referencing) and S7 (fitting
  engine).

### Build step S4: stimulation events from the raw signal

- New module `evoparam.events` (spec §5.2), replacing the skeleton's `evoparam.detect`
  (removed; `detect_stim_onsets` becomes `detect_pulses(..., method="threshold", refine=False)`).
- `detect_pulses` returns the event table of spec §4.3.4 (`onset_sample`, `onset_time`,
  `peak_abs`, `channel_argmax`, `saturated`, `n_saturated_channels`, `stim_site_guess`,
  `site_confidence`, `train_id`, `index_in_train`, `detector`, `threshold`), then `z_peak`,
  `rho`, `template_class`, `interval_flag` and, with channel names, `stim_site_pulse`.
  Steps: robust score (median over at most 16 evenly spaced channels of the absolute
  sample-to-sample change, robust z); one candidate per artifact (crossings closer than
  20 ms, or half of `min_interval`, are merged); onset at the first sample within 2 ms
  before the peak that reaches 25 % of it; strength gate (`min_relative`, 0.25 of the 90th
  percentile of the candidates' peaks); grey-zone candidates (0.4 to 1 times the gate)
  kept when their artifact repeats (`rho >= rho_min` with another candidate, ±0.5 ms
  tolerance); recovery of weaker pulses that match a confirmed template; refractory period
  in which the strongest pulse wins. `expected_rate` sets `min_interval` and the
  `interval_flag` column. `detect_events` accepts an array, a one-trial `TrialSet` or an
  MNE `Raw`; it is also `evoparam.detect_events`.
- `detect_saturation`: rail runs (2 samples with an estimated rail, 1 with a known rail) and
  stuck codes (`min_run` equal samples at least half-way to the rail), per channel and per
  pulse, plus `recovery` (5 ms); returns the mask and a table. An estimated rail must lie
  8 robust SD from the channel's median, so clean or quantised channels are never flagged.
- `identify_stim_site` (spec method: largest artifact, polarity and electrode checks,
  confidence `(a1 - a3) / a1`), with markers, triggers and auxiliary channels excluded by
  name or by channel type; `vote_sites` makes consecutive pulses agree on their site.
- `group_trains` (a new train at a site change, or at a gap longer than `max_gap`; the
  default gap is 3 times the expected or the median interval), `trains`,
  `events_to_metadata`, `match_onsets` (now with recall and precision) and
  `estimate_delay` (trigger-to-artifact delay of a system).
- Validation on OpenNeuro ds004080, 25 runs of 21 patients, 40 s each (200 pulses, 512 and
  2048 Hz): recall 0.995, precision 0.990 (REQ-040). The trigger in `events.tsv` precedes the
  first artifact sample by a constant 0.97 ms; after removing it (estimated on half of the
  runs) the median onset error is within one sample on the other runs. Stimulated site:
  agreement 0.66-0.68 with `events.tsv` (REQ-042 asks 0.95: **not met**, test marked xfail),
  but the confidence is calibrated (accuracy 1.00 on the 56 % of pulses with confidence
  >= 0.5). Cause: the amplifier holds the stimulating contacts flat or at a step, and the
  artifact reaches every contact at a similar size through the common reference.
- Mayo events (ds003708, ds004457, 6 runs): `group_trains` gives one train per block of one
  site, 10 pulses per train (median), about 0.2 Hz (REQ-044). The Mayo recordings (MEF3)
  cannot be read yet, so REQ-042 is not checked on them.
- Speed and memory (REQ-045): 10 minutes of 100 channels at 2048 Hz in 0.56 s (0.61 s with
  the site guess) on one core of the Mac Studio, 0.15 times the data size in extra memory
  (`benchmarks/bench_events.py`; opt-in test with `EVOPARAM_PERF_TESTS=1`).
- Not yet done: fractional-sample onsets for high-frequency trains need
  `artifact.estimate_period` (build step S5).

### Build step S3: component shapes and backgrounds

- `evoparam.shapes`: the eight component shapes of spec §5.5.1 (`gaussian`,
  `skewed_gaussian`, `exgaussian`, `gamma`, `biexponential`, `damped_oscillation`, `gabor`,
  `template`), unit amplitude (`max |s| = 1`), vectorised over parameter sets, with analytic
  Jacobians, default bounds from the window and the width/frequency settings, initial guesses
  and reported latencies. The Gaussian and skewed Gaussian of `core/funcs.py` are rewritten to
  this protocol; `core/funcs.py` itself is unchanged.
- `ComponentFeatures` and `compute_features`: latency, signed amplitude, polarity, FWHM,
  onset/offset (fractional peak, default 10 %), area, rise and decay times, symmetry,
  sharpness and all extrema, measured on the component curve on a grid ten times finer than
  `1/sfreq`; a latency outside the window is reported with the warning
  `"latency_outside_window"`.
- `evoparam.shapes.backgrounds`: the seven backgrounds of spec §5.5.2 (`none`, `constant`,
  `linear`, `exponential`, `double_exponential`, `pspline`, `artifact_basis`), with their
  linear basis and its Jacobian exposed for variable projection. The P-spline knot spacing
  must be at least `2 · max_fwhm` (`SettingsError` otherwise); its penalty can be chosen by
  generalised cross-validation on the baseline.
- `evoparam.shapes.registry`: `SHAPES`, `BACKGROUNDS`, `get_shape`, `get_background`, and
  `register_shape`, which checks a user shape (protocol, bounds, output shapes, unit amplitude,
  Jacobian against finite differences, initial guess, features) and raises `SettingsError`
  naming the failing check.
- Tests (REQ-100…REQ-102): every Jacobian against central finite differences on 100 random
  parameter sets inside the bounds; values against independent scalar references (in float64,
  and in mpmath at 1e-10 when mpmath is installed) and closed forms in `np.longdouble`; unit
  amplitude, latency formulas and features against closed forms.

### Build step S2: readers, remote partial reads, event harmonisation, dataset registry

Scope of this step: the readers of spec §5.1 that need no NWB or Neo support. `read_nwb`,
`read_neo`, `write_bids_derivatives` and `write_nwb` (REQ-024 and the NWB / Neo parts of
REQ-020) move to build step S17 (spec §10), with the rest of the NWB and BIDS-derivatives work.

- `evoparam.io.from_array` (2-D array → one trial, 3-D → trials) and `evoparam.io.read`, which
  dispatches on the type or extension of its input (array, MNE object, `TrialSet` / `Evoked`,
  `.vhdr`, `.edf`, BIDS folder, `.evoparam` container); NWB files and Neo objects raise
  `ReaderError` until S17.
- `evoparam.io.epoch` returns a `TrialSet`: windows `[tmin, tmax)` around the sample nearest each
  onset; trials outside the recording are dropped and `kept` is stored in the provenance step;
  overlapping epochs are kept and flagged in `metadata["overlaps_previous"]`.
- `evoparam.io.read_brainvision` (inherited return form `(data in µV, sfreq, ch_names, t0)`):
  INT_16, INT_32 and IEEE_FLOAT_32, multiplexed and vectorized, resolution per channel, every
  voltage unit converted to µV (a warning lists the converted and the non-voltage channels),
  duplicated channel names suffixed `-1`, `-2`. `read_brainvision_segment` keeps its behaviour.
- `evoparam.io.read_bids`: one run, or every matching run concatenated, from a local folder or
  remotely by HTTP byte ranges (OpenNeuro by default); `events="auto"`, `select` (dict or query),
  `picks` (names, indices or channel types); channel table from `channels.tsv` and
  `electrodes.tsv`; `provenance.source` records the data file, `bytes_read`, the byte `ranges`
  and the SHA-256 of the bytes read (the whole file for local reads), the BIDS entities and the
  dataset DOI. The ten-trial read of ds004080 (REQ-021) transfers about 9.5 MB.
- `evoparam.io.RemoteFile`: chunk cache (8 MiB chunks, 32 kept), retries after 2, 4, 8 and 16 s,
  and a whole-file download with a one-line warning when the server answers 200 instead of 206.
  `EVOPARAM_OFFLINE=1` refuses every connection; all network access is in `evoparam/io/_fetch.py`.
- `evoparam.io.harmonize_stim_events` (with `HarmonizationReport`) maps stimulation columns to
  the standard metadata columns with the unit rules of spec §10.2 ("4.0 mA", "1000 µs", "1ms",
  `n/a`, units from `events.json`, inferred units reported as `unit_inferred`, contact pairs in
  `trial_type`, polarity vocabulary); `validate_stimulation_events` (with `ValidationReport`)
  reports missing fields instead of raising; `select_stimulation_events` also accepts tables
  and a current.
- `fetch_openneuro_file` gains `sha256=`: a file that does not match is deleted and
  `ReaderError` is raised.
- Dataset registry `evoparam/io/registry.json` (schema `evoparam-registry/1`) with the 14 datasets
  of spec §5.1; `list_datasets` and `dataset_info`. Every value comes from the literature review
  (`docs/spec/LITERATURE.md`, datasets axis, and `docs/spec/GAPS.md`); values not stated there
  are null.
- `evoparam.io.read_mne` (extra `mne`) and `read_edf` / `read_events_edf` (extra `edf`) raise an
  `ImportError` naming the extra when the package is missing (REQ-025), through the new helper
  `evoparam.utils._optional.require`. Their tests are skipped when the packages are absent, and
  the two extras are not yet part of `all`.
- Tests in `evoparam/tests/io/`: mini BIDS dataset written by a fixture, local HTTP servers that
  honour or ignore `Range` (and one that fails before answering), BrainVision formats,
  harmonisation, registry. The read of the real ds004080 runs only with
  `EVOPARAM_NETWORK_TESTS=1` (pytest marker `network`).

### Build step S1: data model

- `evoparam.data.TrialSet` (trials × channels × time) and `evoparam.data.Evoked`
  (channels × time), with every invariant of spec §4.3 checked at creation: `select`, `crop`,
  `average` (mean or 20 % trimmed mean, grouped by metadata, rejected trials and masked
  samples left out, results independent of `chunk`), `convert_units`, `apply`,
  `as_evoked` / `as_trialset`, `copy`, `save` / `load` (with memory mapping).
- `evoparam.data.Provenance`: package version, source, ordered steps with settings,
  environment, random seed and data hash; every container method appends one step.
- Result dataclasses: `Component`, `GoodnessOfFit`, `ResponseFit`, `GroupFit`, `TrialFit`,
  `HierarchicalFit`, `ResponseTest`, `ArtifactResult`, `ShapeBasis`, `TFFit`; NaN-aware
  equality; tidy tables with units for `ResponseFit` and `GroupFit`.
- The `.evoparam` container (directory or zip): manifest with schema version and SHA-256 of
  every file, strict JSON (NaN written as null), arrays in npz/npy, tables in parquet or CSV
  with a sidecar of dtypes and units; a newer schema major raises `EvoparamVersionError`.
- `evoparam.errors`: the error hierarchy of spec §4.6. The inherited error classes now derive
  from `EvoparamError`.
- Test of the import graph against the dependency table of spec §4.2.

First skeleton of evoparam, derived from ERPparam (see NOTICE).

- Package layout in four blocks: `io`, `detect`, `clean`, `param`.
- `io`: BIDS events reader, BrainVision reader (local or remote, partial reads), epoching,
  OpenNeuro file helpers.
- `detect`: robust-threshold detection of stimulation onsets, and onset matching.
- `clean`: artifact interpolation for arrays of any shape.
- `param`: `fit_epochs`, which fits the peak model on the average response of each channel.
- Example on public data: `examples/ccep_pipeline.py` (OpenNeuro ds004080).
- Packaging moved to `pyproject.toml`; Python 3.10 or later.
- Fixed: plotting used a missing attribute; interpolation failed on 2d arrays and used a
  sample inside the interpolated window; tests depended on run order; a debug print.
- Removed: all material specific to power-spectrum fitting.
