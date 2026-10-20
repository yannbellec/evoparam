# Decisions on the reviewed specification

`SPEC.md` is the text the owner reviewed (branch `spec`, commit d8cccfd). It is a minimum and
is not edited. This file records:

1. the owner's resolution of each point where the letter of the spec is impossible or
   contradictory (D1–D16 of the audit of 2026-10-09);
2. the resolved import table of §4.2 (D3, D4, D5), which `evoparam/tests/test_import_graph.py`
   enforces and checks against this file;
3. the divergences kept because they measured better than the reviewed procedure. The
   reviewed procedure stays available as an option, except where an entry says otherwise;
4. readings of the spec that are needed to apply it, with the reason for each;
5. points where two requirements cannot both be read literally, and how each is resolved.

Measured values come from `CHANGELOG.md`, which holds the full record.

## 1. Owner decisions on D1–D16

| item | point of the spec | decision |
|---|---|---|
| D1 | REQ-250 (byte-identical across platforms) against REQ-295 (1e-9 across platforms) | Across platforms, results must agree within 1e-9 relative (REQ-295). On one machine they must be byte-identical (`tests/sim/test_sim_determinism.py`). The bytes differ in the last bits between CPUs (SIMD paths) even with the same numpy and scipy. |
| D2 | REQ-370: the full layer-2 factorial (276 480 cells × 1 000 replicates) in ≤ 6 h | This is impossible in 6 h (about 31 000 core-hours). Each week runs a fractional design, and the full factorial is covered over successive weeks. This is documented with the layer-2 runner. |
| D3 | §4.2 forbids imports between the top layers except `benchmark → reports`, while §5.13 needs reports with figures and a `report` command | The order `plots` < `reports` < `cli` is kept, with `benchmark` above `reports` and `plots`. This is required by §5.13. The table is in section 2. |
| D4 | §4.2 "`data` imports numpy only" against `DataError` (§4.6), methods text from the result (REQ-338) and plugin provenance (REQ-328) | The features win. `data` may import the leaves `errors` and `version` and the entry-point loader `_plugins`. The methods text is built in `evoparam/data/_methods.py`. `ShapeBasis.as_template` is the function `evoparam.spatial.as_template` (§5.9). |
| D5 | Other rows of §4.2 against features required elsewhere | The features win. The edges are `models → presets` (§6.3), `stats → existence` (§5.11), `sim → io` (REQ-256), `shapes`/`presets`/`io`/`data` → `_plugins` (REQ-328), and `hierarchical`/`sim` → `utils.require` only (§9.2, REQ-292). The table is in section 2. |
| D6 | REQ-274 names MMN, N2pc, LRP and ERN, but the `erp` preset has no windows for them | Add MMN, N2pc, LRP and ERN windows to the `erp` preset, using published windows with citations. |
| D7 | Merge-and-compare (§5.5.3 step 5) against `selection="fixed"` returning exactly K components (REQ-122) | Merges are not tried in `fixed` and `cv` mode. |
| D8 | "The published rule" for N1: the paper's text (extremum) or the published code (local minimum)? | The N1 REQs must pass under both readings, and both results are reported. Revised on 2026-10-13 after the exact joint bound (§5.8): the requirements are judged on the spec's registry row; the published code's reading stays measured and reported but does not gate. |
| D9 | REQ-040 onset error against `events.tsv`, whose triggers precede the artifact by a constant 0.97 ms | Onsets are physical onsets (the start of the artifact). The constant trigger delay is estimated on held-out runs, and this is documented. |
| D10 | REQ-372: the responsive channels of the published analysis exist only as figures | The authors' code, run on the data, is the comparator. |
| D11 | REQ-213: "20 % power increase" is ambiguous | The strict reading is tested too. If it cannot be reached, it is an xfail with the measured value. |
| D12 | §4.3.4 "baseline-normalised" maps against the exact identity of REQ-211 | Both maps are given: power maps for the exact identity and `*_norm` maps. |
| D13 | `ccep` label "delayed (0.100–1.0)" against fit window (0.009, 0.5) | The fit window is kept. "delayed" can be reached with an extended `fit_window`, and this is documented. |
| D14 | §7.3 "the first five subjects … plus `sub-ccepAgeUMCU01`" | Six subjects: the first five other than UMCU01, plus UMCU01. |
| D15 | REQ-194: the paper does not name its example channel | All channels are reported with the paper's window. |
| D16 | REQ-016/REQ-306 (no network by default) and REQ-017/§12 (green in CI) need a network job that the spec does not describe | A local runner, `tools/run_all_tests.py`, runs the whole suite with the network, slow and perf tests on a chosen Python. It writes a JUnit report and a summary. The GitHub workflow also defines the matching jobs (`network-slow-perf`, scheduled and manual). GitHub Actions is not used for now. |

## 2. Resolved import table (§4.2 with D3, D4, D5)

Two leaf modules, `evoparam.errors` (the hierarchy of §4.6) and `evoparam.version`, import
nothing from the package. Every module may import them. "may import" lists the other
modules a module may import. An edge that is not in this table makes
`tests/test_import_graph.py` fail, and the test exempts no module. The test also parses
`from evoparam import X` as an edge to `X`. It fails on cycles and on any top-level
import of an optional package (REQ-296 i–iii).

<!-- import-table -->

| module | may import | reason for edges beyond the reviewed table |
|---|---|---|
| `data` | `_plugins`, `utils` | D4: the plugins used are recorded in provenance (REQ-328); D5: `utils.require` only (§9.2, REQ-292) |
| `io` | `data`, `utils`, `_plugins` | D5: reader entry points (REQ-328) |
| `events`, `artifact`, `prep` | `data`, `utils`, `shapes` | as reviewed |
| `shapes` | `_plugins` | D5: shape entry points (REQ-328) |
| `models` | `data`, `shapes`, `prep`, `utils`, `presets` | D5: `fit(data, preset=...)` (§6.3) |
| `existence` | `data`, `models`, `utils` | as reviewed |
| `trials` | `data`, `models`, `existence`, `utils` | as reviewed |
| `hierarchical` | `data`, `models`, `trials`, `stats`, `utils` | D5: `utils.require` only (§9.2) |
| `spatial`, `tf` | `data`, `models`, `utils` | as reviewed |
| `stats` | `data`, `models`, `utils`, `existence` | D5: `trials_needed` resamples the existence test (§5.11) |
| `sim` | `data`, `shapes`, `artifact`, `utils`, `io` | D5: licence from the registry (REQ-256); `utils.require` only (§9.2) |
| `presets` | `_plugins`, `utils` | D5: preset entry points (REQ-328); `utils.require` only (§9.2, REQ-292) |
| `utils` | none (the two leaves only) | cross-cutting helpers of the §4.2 figure |
| `_plugins` | none (the two leaves only) | entry-point loader of §10.8 |
| `_extras`, `_deprecated` | none (the two leaves only) | data tables of §9.2 (extras, REQ-292) and §9.3 (deprecated names, REQ-293) |
| `reports`, `plots`, `cli`, `stream`, `benchmark` | any module above | D3: between them only `benchmark → reports`, `benchmark → plots`, `reports → plots`, `cli → reports`, `cli → plots` and (S19) `cli → benchmark`, because §8.4 and §10.5 make `evoparam benchmark` a command |
| `__init__` | any module above | the package root re-exports the public API; no module imports the root |
| `__main__` | `cli` | `python -m evoparam` |

The inherited modules (`objs`, `param`, `plts`, `analysis`, `core`, `bands`,
`utils/download.py`, `utils/debug.py` and the other inherited files of `utils/` and `data/`)
were removed at the end of S7, as §12 rule 1 and §4.5 require. No exemption for them is
left in the test.

## 3. Divergences kept, measured better than the reviewed procedure

### K6: stimulated-site identification (§5.2 method 5)

- **Reviewed rule** (still available as `identify_stim_site(..., method="spec")` and
  `detect_pulses(..., site_method="spec")`). The rule computes
  `a_c = max_{|t|≤1 ms} |x_c(t) − x_c(−2 ms)|` for each channel and ranks the channels.
  The guess is the two largest channels, ordered by channel index. The confidence is
  `(a_1 − a_3) / a_1`, halved when the first deflections are not of opposite signs or,
  when `ch_info.group` is given, when the two contacts are not on the same shaft.
- **Default** (`method="evidence"`). The artifact is measured from −5 ms to +1 ms and
  divided by the channel's pre-pulse noise. A channel that moves (at least 5 % of the
  largest artifact) gains one point of evidence when it *leads*: before the onset it
  already moves by ≥ 10 noise SD and by ≥ 30 % of its artifact. It gains another when it
  is *held*: from 5 to 100 ms after the onset, at least half of its steps are below a
  tenth of the typical pre-pulse step. Channels are ranked by evidence, then by size. The
  confidence is `(s_2 − s_3) / s_2`, computed on evidence when evidence separates the
  channels and on size otherwise. It is halved by the same polarity and shaft checks. For
  close pulses, the lead window shrinks to a quarter of the interval (at least 2 ms) and
  the hold window ends before the next pulse.
- **Measured** on ds004080 (25 runs of 21 patients, 199 pulses, from the data alone;
  REQ-042 asks for 0.95):

  | | reviewed rule | default |
  |---|---|---|
  | agreement with `events.tsv`, per pulse | 0.61 | 0.980 |
  | after `vote_sites` | 0.66 | 0.995 |
  | 13 even runs (development) | — | 0.981 / 0.990 |
  | 12 odd runs (held out) | — | 0.979 / 1.000 |

  Of the pulses with `site_confidence ≥ 0.5`, the default identifies 100 % correctly; these
  are 81 % of all pulses. Halving or doubling any one threshold or window gives 0.94 to
  0.985 per pulse and 0.955 to 0.995 after the vote. Speed is unchanged: 0.79 s against
  0.81 s for 10 minutes of 100 channels at 2048 Hz.
- **Why the reviewed rule fails on these data.** The amplifier holds the stimulating
  contacts, or leaves them at the rail, while the common reference spreads a 1–3 mV
  artifact to every contact. The stimulating contacts also move 1–3 ms before the
  shared artifact, so the −2 ms reference sample already lies inside their artifact.
- The text of §5.2 method 5 in `SPEC.md` is the reviewed text. The description above
  replaces the edit that had been made to the spec after review (audit F61).

### Engine defaults settled by measurement (§5.5.3 steps 1, 3, 4 and 5)

Measured on the merged engine of `integration-2` (2026-10-09). Choices were made on
development data: the five ds004080 development runs (UMCU07, 19, 28, 40, 49), simulation
seeds 0–2 of the ECAP simulators and the development seeds of the layer-2 grid (41 cells of
the weekly design, every fifth cell, 5 replicates). The values reported for REQs are on the
REQ data (the REQ-371 runs and the example run, held-out seeds). The full table is in
`CHANGELOG.md`, "Defaults settled by measurement".

- **`threshold_role` (step 3).** The reviewed reading (`"select"`: only the extrema above
  `detection_threshold` are candidates) fails REQ-104 (87 of 100 waveforms agree for
  thresholds 1, 2 and 3; REQ-104 asks 99). `"order"` (every extremum is a candidate, the
  threshold only orders them) meets it (100 of 100), so it stays the **engine default**.
  The `ccep` preset (v1.1) sets `"select"`: on the development runs the N1 decision agrees
  better with both readings of the published rule (kappa 0.40 and 0.39 against 0.36 and
  0.31) and the fits are 1.4 times faster; on the layer-2 grid the CCEP cells lose 0.02 of
  F1 on the development seeds and nothing on the held-out seeds. REQ-104 is defined on the
  `erp` preset, which keeps `"order"`.
- **`noise_model` (step 1).** `"neff"` (the reviewed step 1) stays available and meets
  REQ-107 as written. The default stays `"ar"` (AR whitening) for every data type: with
  `"neff"` the CCEP fits of ds004080 find almost no component (12 components over all
  the channels of 12 runs; N1 kappa about 0) and the layer-2 F1 drops (0.68 against 0.74 on the
  development seeds, 0.69 against 0.76 held out). The noise reference (`noise_source`) is a
  separate setting: `"auto"` (plus-minus average when the baseline holds fewer than 30
  independent samples, the baseline otherwise; measured on ERP CORE by the earlier work,
  CHANGELOG "Engine: noise from the plus-minus average") stays the default and
  `"baseline"` is the reviewed procedure.
- **Cancelling pairs (step 5, risk S1).** Two mechanisms were compared: refusing every
  model with two overlapping components of opposite signs (`reject_cancelling=True`) and
  refitting such a final state from the extrema of the data, marking the pair not
  resolvable when it stays (`reseed=True`, `Component.resolvable`). The **default is
  `reseed=True`, `reject_cancelling=False`**: on the ECAP simulations the refusal leaves
  the greedy path in a worse optimum (held-out seeds: amplitude error 14.6 % against 6.8 %
  cochlear, 10.8 % against 5.6 % spinal; 10 of 21 cochlear fits above five times the noise
  against 2), and on ds004977 ROC11–ROC12 the shared fit built on refused pairs drops three
  components and its residual cost is 1.8 times higher. On the layer-2 grid and the N1 case
  the two are within noise of each other. The ECAP presets (`ecap`, `ecap_ci`, v1.1) set
  `reseed="always"` (every final model refitted from the extrema): amplitude error 1.5 %
  against 8.3 % (cochlear, development seeds), 5.1 % against 6.8 % held out, at 1.7
  to 3.7 times the fitting time.
  `fit_shared` keeps its own rule (`fit_shared(..., reject_cancelling=True)`): the shared
  fit has no reseeding, and a pair present in the per-channel fits is removed by dropping
  the member whose removal fits best. Its latency bounds keep every channel's latency in the
  fit window (a bug fix, not a choice: on ds004977 all 11 channels had a component before
  the stimulation, at −9 ms, or −18 ms with free latencies, before it; none after).
- **Noise that does not fill its band (step 1).** When the noise has an empty band above
  1/k of the Nyquist frequency (k = 2 to 4: an upsampled recording, or an anti-alias or
  hardware low-pass well below the Nyquist frequency), the whitened residuals of step 1 are
  those of every k-th sample, whitened by the AR model of the noise at that rate
  (`engine.noise_stride`, `engine.StridedWhitener`). Whitening such a noise at the full rate
  amplifies the empty band by orders of magnitude: on ds004080 run 021448 upsampled from 512
  to 1024 Hz the `ccep` fit found an N1 on 54 of 100 channels instead of 14, and 12 with the
  stride. The band must be all but empty (below 1e-4 of the octave under it), and the
  samples kept are on the phase whose noise is highest at the top of its band
  (`engine.noise_phase`: the recorded samples of an upsampled recording, wherever the epoch
  starts). Steep low-passes well below the Nyquist frequency (order 4 at 30 or 40 Hz on
  256 Hz ERP simulations, 300 Hz on 2048 Hz CCEP ones) are strided too, with as good or
  better recall and fewer false positives (CHANGELOG table); gentle ones (order 2) are not.
  A noise that fills its band takes the reviewed procedure, bit for bit (CHANGELOG "Engine:
  noise that does not fill its band").
- **`candidate_search` (step 4).** `"beam"` stays the default (more robust on CCEP-like
  simulations, better N1 kappa, about 9 times faster; CHANGELOG "Fitting engine as written");
  the reviewed search is `candidate_search="spec"`.
- **Extrema tried before the search stops (step 4, REQ-290 P1; 2026-10-13).** The beam
  search tried up to 2 × `max_components` residual extrema at every step, so the last step
  of every fit, which adds nothing, cost up to 24 quick fits (a third of P1's evaluations).
  It now stops after the six strongest extrema when none of them improves the criterion by
  `delta` (`engine.CANDIDATE_SCAN`; a step with a qualifying extremum still scans until its
  beam is full). Six because the first qualifying extremum of a step was at most the sixth
  on the layer-2 development cells (the first in 546 of the 559 steps that added one). Measured on both seed
  sets of the layer-2 grid, the REQ-104–112 engine waveforms and the seven ds004080 runs
  of layer 3: recall, F1 on the engine waveforms and the N1 kappa unchanged, precision equal
  or higher, 3 of 936 simulated fits and 32 of 615 channels differ (CHANGELOG "Engine
  speed, part 4"). Three extrema were measured too and rejected: recall drops on `ecap`,
  `animal` and `erna` cells. So was polishing the background once at the end instead of
  after each accepted component (N1 kappa 0.386 → 0.347).
- **`latency_order` (step 5).** The reviewed ordered parameterisation
  (`"parameterised"`) stays the default; `"sorted"` is an option. On the ECAP simulations
  neither is better on both kinds and both seed sets.

### Calibrated acceptance, final shape choice, resolution check (§5.5.4, §5.5.3 steps 4 and 5)

Measured on development seeds (REQ-105 300 000 + ..., noise windows 500 000 + ..., REQ-106
400 600 + ..., REQ-111 700 000 + ..., ds004080 baselines of 23 runs other than the two REQ
runs), reported on the tests' seeds. Full tables: `CHANGELOG.md`, "Calibrated acceptance"
and "Calibrated acceptance: the change of the noise spectrum". The final shape choice
(`shape_choice="hqc"`, refined in models 1.5 into the default `"penalised"`, below) and
`resolution_check=True` are **package defaults** (no preset); the calibrated acceptance is
available on request (`delta="calibrated"`) and the default `delta` is the fixed 10 of
§5.5.4. Every preset sets the reviewed procedure (`delta=10`, `shape_choice="bic"`,
`resolution_check=False`), and with those three settings the engine gives the same fits
as before, value for value (123 fits of the engine, `erp`, `ccep` and `tep` compared
with qualite-zero; the reference fixture unchanged).

- **`delta="calibrated"`, `alpha=0.04` (§5.5.4), on request.** §5.5.4 fixes `delta = 10`
  and says it is "calibrated by simulation (REQ-112: pure-noise acceptance <= 5 % at the
  default `delta`)". A fixed `delta` cannot be calibrated for every noise: the fall of BIC
  that pure noise reaches depends on the window, the widths searched, the whitening filter
  and how well the noise spectrum is known. For about 2.5 % of noise windows with a
  component, white noise needs `delta` about -1, AR(1) rho = 0.5 about +1 and rho = 0.9
  about +2; at `delta` 10 every noise is near 0 % and REQ-105 fails (recall 0.883). With
  `delta="calibrated"` the threshold of each fit is derived from its own noise model
  (`models/_calibration.py`): the expected Euler characteristic of the whitened scan over
  latency and width (Siegmund and Worsley 1995) sets the deviance pure noise exceeds with
  probability `alpha`, widened for the sampling error of the AR spectrum and for the change
  of the spectrum over the baseline (AR models of the same order fitted on three segments
  of the baseline; the spread of their log spectra beyond what sampling explains, weighted
  by the component's spectrum). The criterion stays BIC (step 4 unchanged); only the size
  of the fall it needs is set per fit. Measured: REQ-105 recall 0.950, precision 0.986
  (test seeds, 100 per K; 0.9545 and 0.980 on 1 000), met; on the ds004080 baselines 92.5
  % silent on the two REQ runs (94.6 % on the development runs), short of the 95 % of
  REQ-112 (§4 "REQ-112 on real baselines"). It is therefore not the default. The
  calibration applies with `noise_model="ar"` and `selection="bic"`; otherwise the fixed
  10 is used.
- **`shape_choice="hqc"` (§5.5.3 step 4).** Step 4 compares the allowed shapes of a
  candidate on the quick fits of the forward search ("the best kept", by the criterion).
  With `"hqc"` (the package default of models 1.4; refined in 1.5 into the default
  `"penalised"`, below), every component of the final model, whatever its width, is refitted
  with every allowed shape (asymmetry started at 0 and +/- 2) and the lowest HQC (Hannan
  and Quinn 1979, `2 ln ln n` per parameter, about 3.7) is kept: BIC charges the asymmetry
  ln n (about 6.4), which a skewed component of SNR 5 to 20 rarely pays, and its Gaussian
  centre lies up to 1.55 samples from the mode. REQ-106 skewed truth with the default
  pair: latency MAE 0.52 (BIC on the quick fits) to 0.46 sample (test seeds, package
  defaults; 0.456 on 1 000), met. The first version of this choice (AIC, components of at
  least 8 samples only) skipped narrow components under a width threshold, which REQ-004
  forbids (§4 "REQ-004, a step skipped under a width"); `"aic"` stays available, on every
  component. `"bic"` is the reviewed comparison and every preset sets it (the ds004080 N1
  decision was validated with it).
- **`resolution_check=True` (§5.5.3 step 5, REQ-111).** Step 5 merges two overlapping
  components after a step accepted the second one; two equal Gaussians 0.5 FWHM apart
  differ from one Gaussian by a chi-square of 1.4 at SNR 10, so no step accepts the second
  and the warning `merged:<k>,<k+1>` of REQ-111 was never set. The check asks the other
  question for every final component: would a pair of equal Gaussians `overlap_c`
  (sigma_1 + sigma_2) apart, whose single-Gaussian fit is this component, have been told
  apart from it (its deviance against the best single Gaussian, in the fit's whitened
  metric, against the deviance a new component needs)? When not, the component is flagged
  `merged:k,k+1`. REQ-111 at 0.5 FWHM: one component with the warning in 100 of 100 runs
  (none before; 0.998 on 1 000; package defaults). The flag says what the data cannot
  exclude, not that two components were seen: it was set on every single Gaussian
  measured from SNR 5 to 80 (FWHM 10 and 40 ms); at SNR 200 it is set with the fixed
  `delta` 10 (the pair would need a deviance of 29, it has 22), not with the calibrated
  acceptance; at SNR 400 it is not set. Every preset sets `resolution_check=False`.

### Penalised estimate of a skewed component (§5.5.3 steps 4 and 6, REQ-106)

The package default `shape_choice="penalised"` (models 1.5; `models/_penalised.py`;
CHANGELOG "REQ-106 with the default pair: a penalised estimate of the asymmetry") replaces,
for a skewed component of the final model, the least-squares estimate of §5.5.3 step 6
with a penalised one. Every preset keeps `"bic"`, the reviewed procedure; `"hqc"` (the
default of models 1.4) and `"aic"` stay available.

- **What it does.** As with `"hqc"`, every final component is refitted with every allowed
  shape, whatever its width (REQ-004, §4), and the lowest HQC is kept. A skewed component
  (the others and the background held, every linear coefficient by variable projection)
  is then estimated in three steps:
  1. A Gaussian prior on its asymmetry `alpha` (mean 0, SD 2.5).
  2. Weights `sqrt(1 + 2 s(t)^2)` on the samples, `s` its unit curve.
  3. The samples below 0.3 of its peak further weighted by `min(1, 600 / E)`, `E` its
     whitened energy.

  The criteria, residuals and intervals are those of the unweighted data at that estimate.
- **Why.** The least-squares skewed Gaussian puts the mode of a gamma or biexponential
  deflection early (DECISIONS §5.4; noise-free bias up to -1.0 sample at 25 samples). Its
  mode moves about 0.04 FWHM per unit of `alpha`, so the noise of `alpha` spreads the
  latency (SD 0.8 to 1.0 sample). The prior acts on the spread. The weights act on the
  bias: a skewed Gaussian follows any unimodal peak near its top, and the tails of a
  strong component, where a wrong shape's misfit exceeds the noise, count less. A weak
  component keeps all its samples.
- **The criterion inside it.** The constants were chosen with the shapes compared by AIC
  (before the HQC default was merged). After the merge both criteria were measured on
  the development seeds (REQ-105 300 000 + ..., 1 000 per K; REQ-106 400 600 + ...,
  1 000 per truth). Both meet every REQ-106 row; HQC has the better REQ-105 and is kept:

  | | REQ-105 recall / precision | Gaussian | skewed | ex-Gaussian | gamma | biexponential |
  |---|---|---|---|---|---|---|
  | `"hqc"` (before) | 0.8865 / 0.9987 | 0.233 | 0.455 | 0.403 | 0.722 | 0.587 |
  | penalised, AIC | 0.8867 / 0.9989 | 0.251 | 0.423 | 0.388 | 0.454 | 0.455 |
  | penalised, HQC (kept) | 0.8870 / 0.9992 | 0.220 | 0.465 | 0.393 | 0.482 | 0.462 |

- **Tests' seeds, measured once at the end** (1 000 per condition; `"hqc"` / penalised):
  REQ-105 0.8873 / 0.9991 to 0.8875 / 0.9992 (100 per K: 0.8833 / 1.0000 both); REQ-106
  Gaussian 0.254 / 0.237, skewed 0.456 / 0.478, ex-Gaussian 0.401 / 0.403, gamma 0.706 /
  0.477, biexponential 0.586 / 0.463. The skewed row is worse by 0.010 (development) and
  0.022 (tests' seeds) and the ex-Gaussian row by 0.002 on the tests' seeds; both stay
  met, and the deviation is kept because it meets two rows that were not met. Truths
  fitted with their own shape are unchanged, value for value. REQ-112 on the ds004080
  baselines is unchanged (0.975 silent with `ccep`, 0.9625 with the package defaults).
  REQ-111 fits a single shape and is unchanged by construction, and the REQ-104 tests
  pass as before.
- **Presets.** The `ccep` preset with `"penalised"` (by AIC) instead of `"bic"` was
  measured on the REQ-113/114 runs from cache. The example run's kappa is unchanged under
  both readings of D8 (0.011 and 0.191). The other runs move both ways (registry reading
  +0.03 to +0.17 on four of six runs, code reading -0.04 to +0.02, the others unchanged).
  The REQ-114 close fraction goes from 0.404 to 0.371 (registry) and from 0.400 to 0.433
  (code). This is not a clear gain, so every preset keeps `"bic"`. The presets' fits and
  the reference fixture's results are unchanged.
- **The constants** (SD 2.5, peak weight 2, level 0.3, energy 600) were chosen on the
  development seeds only, from the sweep in the CHANGELOG. They are not fitted to the
  truth shapes' latency: no shift of the latency is applied, and the prior's mean is the
  symmetric shape.

### Cross-projection test: response duration and pairs tested (§5.6 item 2, REQ-134)

- **Reviewed formula** (still available as `test_response(..., method="projection",
  projection_profile="normalised")`): `tau_R = argmax_t mean P(t) / sqrt(t - t0)`. The
  projections are semi-normalised (`<V_k, V_l> / ||V_l||`), so they do not accumulate with
  the window: after the response the norm of the projected trial grows and the mean
  projection falls. The extra division moves the peak to the first milliseconds of the
  window.
- **Default** (`projection_profile="published"`): `tau_R = argmax_t mean P(t)`, the
  definition of the paper [LIT-ccep-15], which §5.6 says the implementation re-implements
  and REQ-134 says it must reproduce.
- **Measured** on the paper's public example data (the `sampledata` of the authors'
  repository, commit f6efb4e; window 15 ms to 1 s), against the durations the paper prints:

  | example | published | default | reviewed formula |
  |---|---|---|---|
  | Fig. 3 | 80 ms | 76.2 ms | 15.6 ms |
  | Fig. 9, all trials | 0.23 s | 232.9 ms | 17.1 ms |
  | Fig. 9, trial 6 removed | 0.28 s | 282.7 ms | 15.1 ms |

- **Pairs tested at `tau_R`.** "One half of the non-overlapping pairs" has two readings.
  The paper's set (`projection_pairs="half"`: one projection per unordered pair of trials,
  `K(K-1)/2` values) reproduces the paper's ten published decisions (Fig. 5 and Fig. 6, 10
  of 10), but its values share trials: on pure noise (white or AR(1) 0.9, 10 or 20 trials)
  it rejects at 0.33 to 0.44 at alpha = 0.05. The default stays the earlier reading
  (`"disjoint"`: each trial in one pair, `K // 2` independent values): 0.07 to 0.13 on the
  same noise, and 0.8 to 1.0 agreement with the published decisions over seeds 0 to 9
  (mean 0.89), because the pairing is random. The REQ-134 significance test runs with
  `projection_pairs="half"`.
- **ds003708.** REQ-134 names ds003708 as the public example dataset of the
  cross-projection paper. ds003708 is the dataset of the basis-profile-curve paper
  [LIT-ccep-16]; the cross-projection paper's example data are the files of its code
  repository, and no per-channel-pair duration or significance has been published for
  ds003708. REQ-134 on ds003708 is a strict expected failure stating this; the published
  values are tested on the paper's own example files
  (`tests/existence/test_projection_published.py`).

### `tep` preset 1.1: constant background and refusal of cancelling pairs (§6.1, REQ-275)

- **Reviewed values** (still available as
  `presets.derive("tep", background="double_exponential", fit_extras={})`): background
  `double_exponential`, cancelling pairs refitted, not refused (engine default).
- **Kept** (`tep` 1.1): background `constant` and `reject_cancelling=True`
  (`fit_extras`). A double exponential with coefficients of opposite signs is the
  biexponential component shape starting at the window's first sample, so the background
  can represent a component, which §5.5.2 rules out for `pspline` by its knot rule. On
  ds002094 it did: the background had an interior extremum in 30 of the 32 development
  fits, taking the N100 (sub-764) or the P30-N45-P60 complex (sub-749). The constant is the
  §5.5.2 background "when a baseline correction has been applied", which the `tep` chain
  does; the decay artifact is interpolated up to 12 ms and recovers within 5-10 ms with
  DC-coupled amplifiers ([LIT-artifact-9]). Refusing cancelling pairs closes the other way
  of describing one deflection with two opposing terms; alone, it changed little because
  the background's bump took the part of the second term.
- **Measured** on ds002094, chosen on 8 development subjects before the 12 others were
  fitted (subjects and the full table in `CHANGELOG.md`, "Preset tep 1.1"). Subjects with
  N45 and N100 both labelled, development / held-out / total, 1.0 → 1.1: with ICA, all
  channels (the REQ-275 reading) 3 / 2 / 5 → 6 / 2 / 8; with ICA, 17 inner channels
  3 / 2 / 5 → 6 / 6 / 12; preset chain, 17 inner channels 3 / 5 / 8 → 3 / 5 / 8; preset
  chain, all channels 1 / 2 / 3 → 2 / 2 / 4. Components not resolvable over the 32
  development fits: 56 → 19. REQ-275 stays an expected failure (8 of 20; 16 needed): on
  10 of 20 subjects the channel of maximal response after ICA is TP9, TP10, IZ or O1,
  where the N100 is positive under the average reference. The channel rule is the spec's
  and is not changed.
- **Cost on simulations** (layer 2, the 26 `tep` cells, 50 replicates): mean F1 0.947 →
  0.945, false components per waveform 0.046 → 0.058; worse under pink noise (1/f^2: F1
  0.99 → 0.91, 0.40 false components per waveform), which a constant background cannot
  follow, and for opposite components 0.3-0.5 FWHM apart; better with latency jitter and
  three components.
- **Strided whitening** (engine, not the preset). The ICA variant is low-passed at 100 Hz
  (1 000 Hz data), so its noise is strided (k = 3 on 19 of 20 fits): with 1.0 this lowers
  the subjects with both labels from 8 to 5 (all channels), with 1.1 it raises them from 6
  to 8. Without it the whitening gain is about 2e5 and the fits fill up to
  `max_components`. The filter cannot be told from the low-passed data on which the stride
  was validated, so the engine rule is kept and 1.1 was chosen with it (`CHANGELOG.md`).

### `sep` preset: constant background (§6.1, REQ-277)

- **Reviewed value** (still available as `presets.derive("sep", background="linear")`): the
  §6.1 table gives `sep` the `linear` background (§5.5.2: "slow drift inside the window").
- **Kept** (`sep` 1.0): `constant`, the §5.5.2 background "when a baseline correction has
  been applied". The reason is in the preset's `notes["background"]`.
- **Measured** on simulated median-nerve SEPs (SNR 5, white noise; CHANGELOG "Build step
  S15, part 1"). With a free linear drift the N20 was described as a falling drift between
  two positive components, and found within 0.5 ms in 9 of 20 runs; with a constant
  background in 20 of 20. A prior on the slope (a drift of at most 3 baseline SD over the
  window) did not help (10 of 20) and was not kept.
- Tests: `tests/presets/test_presets.py::test_req270_every_preset_matches_the_table` (the
  value is listed in `DEVIATIONS` and must be explained in the notes),
  `tests/presets/test_presets_sim.py::test_req277_sep_n20_latency` and
  `test_sep_with_coloured_noise_is_a_documented_limit` (with AR(1) noise at the same SNR:
  16 of 20 at rho 0.5, 8 of 20 at rho 0.8).

### `tep` preset: no `gabor` shape (§6.1, §6.2)

- **Reviewed value** (still available as
  `presets.derive("tep", shapes=("gaussian", "skewed_gaussian", "gabor"))`): the §6.1
  table gives `tep` the shapes gaussian, skewed_gaussian and gabor; §6.2 says "`gabor` is
  included because TEPs contain oscillatory bursts".
- **Kept** (since `tep` 1.0): gaussian and skewed_gaussian. One `gabor` describes a pair of
  components of opposite sign (as N45 and P60), so the labelled TEP sequence lost a
  component. The reason is in the preset's `notes["shapes"]`.
- **Measured** on the layer-2 grid (CHANGELOG "Build step S15, part 2"; base cell: two
  components 2 FWHM apart, SNR 5, 20 replicates): F1 0.55 with gabor, 1.00 without. After
  the change: F1 1.00 at SNR >= 5, 0.97 at SNR 3, 0.91 to 1.00 across the noises. The
  `tep` 1.1 measurements on ds002094 (above) were made without gabor.
- Test: `tests/presets/test_presets.py::test_req270_every_preset_matches_the_table` (listed
  in `DEVIATIONS`, explained in the notes). The F1 values come from the layer-2 runner
  (`validation/layer2.py`), not from a test assertion.

### Trial model of `fit_trials`: `trial_model="anchored"` (§5.7 method 1, REQ-156)

- **Reviewed reading** (still available as `fit_trials(..., trial_model="selected")`): the
  trial model of §5.7 method 1 holds the components that are fitted per trial, each with
  its own shift and scale; the code reads it as the components named by `components`.
- **Default** (`"anchored"`): every component of the channel fit is in the trial model,
  but only those of `components` move. The others keep their channel latency and shape,
  and only their amplitude is fitted; `max_shift` is clipped against moving neighbours
  only. `"all"` lets every component move. The three differ only when `components` names
  a subset (REQ-156 uses `components=["N1"]`); with the default (every component) they
  are the same model.
- **Why** (CHANGELOG "Trials: speed, REQ-150/151/155/156/158 as written, spec options",
  Q8). With every component moving, the channel fits of ds004080 often split an N1 into
  overlapping parts (`N1`, `N1_b`, …); the neighbour rule then clips the N1 `max_shift` to
  0.1–2 ms and the trials sit at that bound. Median trusted fraction on the significant N1
  channels, `"anchored"` / `"all"`: 0.9 / 0.7 on four held-out subjects (UMCU07, 08, 10,
  11; 25 channels; the choice was made there) and 0.9 / 0.7 on the six REQ-156 subjects
  (56 channels). `"anchored"` is 30–45 % faster than `"all"`, and the trials and
  `jitter_sd_null` then use the same procedure (the null fit already held the other
  components at their channel values).
- **Cost on simulations.** `simulate_ccep` with 10 trials, N1 latency and amplitude MAE,
  `"anchored"` / `"all"`: 0.415 / 0.411 ms and 8.48 / 8.23 % at 2 ms of jitter, 1.754 /
  1.748 ms and 18.75 / 18.18 % at 5 ms; `"selected"` 0.532 ms and 24.3 % at 2 ms. REQ-153:
  regularized latency MAE 1.828 ms (all), 1.858 ms (anchored), 1.902 ms (selected).
  REQ-150 at SNR 3 with an N2 added: `"all"` and `"anchored"` give the same numbers. The
  CHANGELOG flags this default for the owner as a trade-off (robustness and speed on real
  data against about 1–3 % of simulated MAE).
- Tests: `tests/trials/test_trials_ccep.py::test_req156_trusted_fraction` and
  `test_req156_speed` (network, the default), `tests/trials/test_trials_table.py`
  (`test_options_are_checked_and_recorded`,
  `test_selected_trial_model_keeps_only_the_chosen_components`,
  `test_vectorised_solver_matches_trial_by_trial_solver`, with every component moving and
  with the first one only), `tests/trials/test_trials_sim.py::test_req153_regularized_beats_free_at_low_snr`
  (`"all"` and `"selected"`).

### Null jitter of `fit_trials`: `null_rule="segments"` (§5.7 method 4, REQ-151)

- **Reviewed rule** (still available as `fit_trials(..., null_rule="spec")`):
  `jitter_sd_null` is the same procedure applied to the pre-stimulus baseline (one
  baseline segment per trial), and realignment is refused when
  `jitter_sd ≤ 1.2·jitter_sd_null`.
- **Default** (`"segments"`): `jitter_sd_null` is estimated from five baseline segments per
  trial, and realignment is refused unless `jitter_sd` exceeds both 1.2 ×
  `jitter_sd_null` and the 95 % quantile of the ratio of two noise SDs (F distribution).
- **Measured** on REQ-151 (CHANGELOG "Trials: speed, …, spec options", Q9). 200 jitter-free
  runs of 40 trials (seeds 100–299), segments / spec: ratio `jitter_sd_null / jitter_sd`
  below 0.8 in 7 / 15 runs; realignment refused in 99.0 % / 86.5 % of runs (REQ-151 asks
  for at least 95 %). With 200 trials per run (100 runs, seeds 1000–1099) both rules pass
  in every run.
- Tests: `tests/trials/test_trials_sim.py::test_req151_no_invented_jitter_and_realign_refused`
  (default), `test_req151_as_written` (both rules, 200 trials per run) and
  `test_req151_as_written_40_trials` (strict xfail with the measured values),
  `tests/trials/test_trials_table.py::test_options_are_checked_and_recorded`.

### Pulse detection: `procedure="evoparam"` (§5.2 methods 1 and 2)

- **Reviewed procedure** (still available as `detect_pulses(..., procedure="spec")`,
  `events/_spec.py`): the median of `|x_c[n] − x_c[n−1]|` over every picked channel, a
  robust z-score, the first crossing of `threshold_sd` after the refractory period, the
  onset at the largest change within ±2 ms minus one sample; then a template from the
  first `n_template` detections, scanned by normalised cross-correlation over the whole
  recording, which rejects threshold detections that do not match it and adds the pulses
  the threshold missed.
- **Default** (`"evoparam"`, the steps in the docstring of `detect_pulses`): the score's
  median over at most `score_channels` (16) evenly spaced channels; crossings closer than
  20 ms are one artifact, whose onset is the first sample within 2 ms before its peak that
  reaches 25 % of the peak; a strength gate relative to the 90th percentile of the
  candidates' peaks, with candidates between 0.4 and 1 times the gate kept only when they
  repeat (clusters by normalised cross-correlation); a recovery pass at `recovery_sd`
  (default `threshold_sd / 4`); within `min_interval` the strongest pulse is kept.
- **Measured** on the 25 ds004080 runs (CHANGELOG "Specification gaps: artifact, prep and
  events", Q3). Reviewed procedure: recall 0.445 and precision 1.000 in 57 s, onsets about
  1 sample early. Default: 0.995 and 0.990 in 2.4 s, onsets within 0.01 sample (held-out
  runs). Threshold pass alone: 0.930 / 0.765 (reviewed) against 1.000 / 0.760.
- Tests: `tests/events/test_threshold.py::test_spec_procedure_on_the_legacy_recording` and
  `test_spec_procedure_template_pass_recovers_and_rejects` (the reviewed procedure),
  `tests/events/test_ds004080_events.py::test_req040_recall_precision_and_onset` (the
  default) and `test_spec_procedure_is_less_accurate_than_the_default` (network).

### Soft priors of the fitting engine (§5.5.3 steps 2 and 4, §5.5.2)

- **Reviewed procedure.** §5.5.3 step 2 solves the linear coefficients by plain weighted
  least squares, step 4 optimises the non-linear parameters within the shapes' bounds, and
  §5.5.2 bounds the exponential background's time constant by `[0.5 ms, |W|]`.
- **Kept** (CHANGELOG "Build step S7", "Changes to the specification"), against
  degenerate decompositions (a negative N1 described as a negative offset plus positive
  components, or two huge components that cancel):
  - `background_prior=3.0`: a pseudo-observation draws the background constant to the
    baseline level with SD 3 baseline SD (only when the fit has a baseline);
  - `amplitude_prior=3.0`: every other linear coefficient has SD 3 times the range of the
    data in the window;
  - `edge_extension=0.05`: a component's location stays within 5 % of the window length
    outside the window;
  - `background_tau_max`: none by default; the `ccep` and `ccep_seeg` presets cap the
    background time constant at 50 ms (`fit_extras`), because a slower "tail" would only
    imitate an offset or a broad component (comment of `engine._background`).

  `background_prior=0` and `amplitude_prior=0` remove the prior rows
  (`models/_settings.py`, `engine._structure_maker`, `engine.prior_rows`).
- **Measured** against the procedure without them (no prior rows, no edge limit, no
  time-constant limit; CHANGELOG "Fitting engine as written, and the N1 case as written",
  Q1). Layer-2 grid (40 cells of the weekly design, 5 replicates): F1 0.760 with against
  0.743 without (better in 7 cells, worse in 1), latency MAE 1.46 against 1.52 samples
  (better in 26 of 38), mean amplitude bias -0.03 against -433 (diverging fits without
  them). The simulation suites of REQ-104 to REQ-112 are unchanged; N1 kappa 0.40/0.33
  against 0.37/0.32. No value was recorded for the 50 ms cap of the CCEP presets alone.
- Tests: no test compares fits with and without the priors. The REQ-104 to REQ-112 suites
  (`tests/models/test_engine_sim.py` and the other engine tests) run with them;
  `tests/models/test_fast_kernel.py` and `tests/models/test_fast_paths.py` check the
  compiled path against the NumPy path with and without prior rows.

### Permutation p-values: generalised Pareto tail (§5.6 method 1)

There is no option for the reviewed p-value; the plain value is returned only where the
tail is not used.

- **Reviewed rule.** "`p_value` is the fraction of null statistics ≥ the observed one, with
  the `+1` correction."
- **Kept** (`existence/permutation.py`, `tail_p_value`; both nulls of
  `baseline_permutation`: sign flips, and the baseline windows of an Evoked). When fewer
  than 10 null values reach the observed one and the null holds at least 50 values, a
  generalised Pareto distribution (location 0) is fitted to the exceedances of the null
  over its 90th percentile (at least 10 of them), and the p-value is the fraction of the
  null above that percentile times the Pareto survival function (Knijnenburg et al. 2009),
  never larger than `(1 + count) / (1 + n)`. Otherwise, or when the Pareto fit fails (a
  DEBUG record names the error), `(1 + count) / (1 + n)` is returned.
- **Why** (CHANGELOG "Build step S8", "Addition to the specification"). With 8 trials there
  are 256 sign patterns, so the smallest plain p-value (1/256) never survives a correction
  over a hundred channels: on ds004080 the test detected nothing on three of six runs
  before this change.
- Tests: no test calls `tail_p_value` directly.
  `tests/existence/test_existence_null.py::test_req130_type_one_error` (500 sign flips,
  rejection rate within [0.8 alpha, 1.2 alpha] at 0.05 and 0.01) covers it: a p-value of
  0.01 or less there needs fewer than 10 of the 500 null values to reach the statistic,
  the branch where the tail is fitted.

### Bootstrap intervals: expanded percentile, no BCa for a fit (§5.5.5, §5.11, REQ-116, REQ-230)

There is no option for a BCa interval of a fit.

- **Reviewed text.** §5.5.5, `uncertainty="bootstrap"`: "percentile and BCa intervals on
  every parameter and feature". §5.11 method 1 (`stats.bootstrap`): "percentile and
  bias-corrected accelerated (BCa) intervals".
- **Kept for a fit** (`models/api.py`, `_bootstrap`). One interval per parameter of each
  component (the shape's parameters and the amplitude `A`, since models 1.6) and per
  feature (latency, amplitude, FWHM, onset, offset, area): the expanded percentile interval
  of Hesterberg (2015), with the tails at `Phi(-sqrt(n / (n - 1)) t_{n-1})` instead of
  `alpha / 2` (`n` trials). The draws are kept in `Component.bootstrap` (§4, "§5.5.5, the
  bootstrap distribution"). The plain percentile interval of an average of `n` trials is too narrow by
  `sqrt((n - 1) / n)` and uses z instead of `t_{n-1}`. BCa was neither implemented nor
  measured for a fit; the CHANGELOG, the code and the git log give no other reason.
- **`stats.bootstrap`** has both intervals as §5.11 asks (`method="bca"` is its default),
  with the same small-sample expansion by default (`expand=True`; `expand=False` gives the
  textbook intervals).
- **Measured.** REQ-116, fit intervals, 200 runs each, plain percentile → expanded
  (CHANGELOG "Interval coverage", 2026-10-10): latency 0.900 → 0.935 and amplitude 0.945 →
  0.960 under white noise; 0.935 → 0.945 and 0.900 → 0.940 under AR(1) rho 0.9 (REQ-116
  asks 92–97 %). `stats.bootstrap` with 30 trials (CHANGELOG "Build step S14"): the
  textbook intervals cover 0.935 (normal) and 0.925 (log-normal, CV 0.5), below the 0.93
  of REQ-230; expanded, BCa 0.948 and percentile 0.949 on normal statistics, 0.940 and
  0.938 on log-normal ones.
- Tests: `tests/models/test_engine_sim.py::test_req116_bootstrap_profile_coverage_as_written`
  (slow), `tests/stats/test_stats_bootstrap.py::test_req230_coverage` (both methods; the
  textbook interval is checked to be narrower and to cover less).

## 4. Readings needed to apply the spec

- **§4.5, `core/info.py` and `core/items.py` ("merge into `utils/`").** These files only
  describe the attributes of the inherited fitting classes, which §4.4 and §4.5 remove.
  Nothing is left for them to describe, so they were removed with those classes.
  `core/modutils.py` is kept as is in `evoparam/utils/_modutils.py`. The error hierarchy
  is in `evoparam/errors.py` (§4.6).
- **§4.5, `plts/signals.py`, `plts/templates.py`, `plts/utils.py`, `plts/style.py`
  ("keep inside `evoparam.plots` with a lazy matplotlib import").** These are kept as
  `plot_signals`, `plot_signals_shading`, `plot_signals_yshade`, `plot_scatter_1`,
  `plot_scatter_2` and `plot_hist` in `evoparam/plots/_signals.py`. They follow the
  conventions of `evoparam.plots` (§5.13): each returns the `Figure`, takes `ax=`, uses
  the Okabe–Ito palette, and seeds the scatter jitter from `random_state` (§9.4). The
  inherited style decorators are not kept: they used `legend.legendHandles`, which was
  removed in matplotlib ≥ 3.9 (§4.5). `plts/fm.py` and `plts/peaks.py` are removed, as
  §4.5 requires.
- **§4.4, `Bands` → `ComponentWindows` in `evoparam.models.windows`.** This module is
  `evoparam/models/windows.py`, a read-only mapping `name → (start, end, polarity)`.
  `label_components` (§5.5.7, `evoparam.models.labels`) is also importable from
  `evoparam.models.windows` (§4.5).
- **§5.2 method 5, reviewed rule: the guess when the checks fail.** The confidence is
  halved when the polarity or shaft checks fail, which implies a guess exists. So the
  two top channels are always reported, and only the confidence changes. The channel
  selection (picks, and the markers and auxiliary channels excluded by name or type) is
  the same for both methods.
- **§9.2, `all` extra.** `all` is the union of every extra except `gpu`, `bayes` and
  `docs`, as written. The local runner installs `.[all,bayes]` so that the NUTS tests run
  (REQ-172, REQ-173, REQ-174, REQ-176).
- **REQ-345, byte-identical outputs.** Results record their creation time and the
  duration of each step. When `SOURCE_DATE_EPOCH` is set (the convention of
  reproducible builds), the creation time is that instant and durations are recorded as
  0. With the same seed, the JSON files written by the command line and by the Python
  API are then byte-identical.
- **REQ-002 and §5.6, the existence test of `evoparam.fit`.** REQ-002 asks for existence
  results "with no other argument required", and §5.6 describes `existence=True`. Both
  hold: the default (`existence=None`) runs the test (on an Evoked, with the weaker
  null of baseline windows, which warns), and when the test cannot run it returns the
  fit with a warning and NaN existence columns. `existence=True` raises instead, and
  `existence=False` skips the test. The test seed is `random_state`, or the seed drawn and
  recorded by the fit. When the fit settings have no baseline, the test uses
  (−0.5, −0.05) s if the data start before −0.5 s, else the pre-stimulus samples.
  `evoparam.read` is `evoparam.io.read`, as REQ-002 names it.
- **REQ-006, the existence table.** `response_duration` gives the onset, the offset and
  the duration, each with its interval (percentiles of the same bootstrap over trials
  that already gave the duration interval), and the decision with its p-value. An
  Evoked has no trials to resample. Its intervals come from a baseline bootstrap (200
  resamples by default, `n_boot`): each resample adds to the z-scored window of a channel
  a segment of the same length of its own z-scored baseline, at a random circular start,
  and the onset and offset are found again with the same critical value. The observed
  window already holds one noise realisation, so the noise is counted twice and the
  intervals are conservative (wider than a trial bootstrap would give). The resamples
  are drawn after the null, so the p-values and decisions are unchanged.
- **REQ-005, warnings logged.** `evoparam.errors.warn` issues the warning and logs it on
  the `evoparam` logger. Every warning of the package goes through it, except the single
  call in `io/nwb.py`: that file is being edited on another branch, so its test is a
  strict xfail. The package installs a `NullHandler` only (REQ-304), so a warning is not
  printed twice when the application has no logging set up.
- **REQ-007, seeds.** `test_response` and `tf_fit` draw and record a seed when
  `random_state` is None, as the fits do (REQ-331). `reproduce` keeps an `ArtifactResult`
  as the result of its last step. The step of `tf_fit` holds every keyword argument that
  changes the result by its name (the frequencies as the grid used, the channels by
  name), so that `reproduce` calls `tf_fit` again with them; a time-frequency fit made
  with a `group_fit` records that it used one (`footprint`) and is not replayed with it.
- **REQ-307 (ii), the unit of `test_response`, `remove_artifact` and `fit_trials`.**
  The unit is the channel (with its condition for `fit_trials`, and its group of trials
  for `test_response`). `fit_trials` fits each channel on its own, so a failing channel
  is caught where it fails. `test_response` and `remove_artifact` work on all the
  channels at once (vectorised): when that raises, the channels are run one at a time to
  find the ones that fail on their own, and only those are marked failed. If none fails
  alone, the exception is not a channel's failure and it propagates. `remove_artifact`
  does this for the methods that clean each channel on their own (`CHANNEL_WISE`);
  `lowrank`, `subspace` and `learned` work across channels and `paradigm` changes the
  trials, so their failures propagate. A failed channel keeps its data with the artifact
  window masked. `test_response` runs each channel from the random state of the group,
  so with a failed channel the bootstrap intervals of the others can differ from a run
  where none fails; the p-values of the sign-flip test do not. The errors of the data or
  of the settings of the call propagate: `DataError`, `SettingsError`, `ReaderError`,
  `EvoparamVersionError` (REQ-307 i), and also the documented `ArtifactError` and
  `EventError` and the `ImportError` of a missing optional package
  (`evoparam.utils._failures.PROPAGATE`). `message` holds `str(exc)` after the class
  name of the exception (`"LinAlgError: singular matrix"`), as `fit_group` already did.
  So it is never empty, even for an exception raised without text, and it says what
  failed. Results carry `status` and `message`: per `TrialFit`, and arrays shaped like
  the channels for `ResponseTest` and `ArtifactResult`. The failed channels of
  `model_comparison` are those whose model fit raises (`fit_waveform` with
  `on_error="raise"`).
- **REQ-009, the unit of the amplitudes of a fit.** A `ResponseFit` records the unit of
  the data it was fitted to (`units`, set by `fit_group` from the TrialSet or Evoked).
  The tables of `ResponseFit` and `GroupFit` give it in `attrs["units"]` for the
  amplitudes and their intervals, `area` (`<unit>·s`), `rmse`, `mae` and `baseline_sd`.
  `"data"` stays only for a waveform fitted from a bare array, which has no unit. A
  `GroupFit` whose fits come from data in different units raises `DataError`.
  `docs/reference/tables.rst` writes the unit of these columns as "data unit";
  `test_tidy.py` now reads "data unit" as the unit of the fitted data (`uV` for the
  simulated CCEP) instead of the placeholder `"data"`, so it checks the actual unit.
- **REQ-010, tables without pandas.** The core functions of REQ-291 return or hold tables
  (the event table of `detect_events`, the quality column that `remove_artifact` adds to
  the trial metadata, the index of a `GroupFit`, the table of `fit_trials`). With pandas
  installed (extra `data`) they stay pandas DataFrames, so every result with pandas is
  unchanged. Without pandas they are `evoparam.data.Table` objects: named NumPy columns
  of equal length with the subset of the DataFrame interface that the package uses
  (`columns`, `len`, `t["name"].to_numpy()`, boolean and list selection, `iloc`,
  `sort_values`, `reset_index`, `itertuples`, `to_dict`), and `to_pandas()` to convert.
  `evoparam.data._table.new_table` and `concat_tables` choose the type; `pd.concat` and
  `pd.DataFrame` are no longer called by the core. A container saved without pandas writes
  its tables as CSV (`table_format="csv"`, and `"auto"` without pyarrow) and reads them
  back as Tables; parquet tables need the extra `data` to write and to read. Operations that are pandas by nature
  (`TrialSet.query`, averaging by metadata columns, `GroupFit.to_dataframe`) keep requiring
  the extra `data` through `require`. `read_neo` builds its metadata the same way, so a
  missing neo names the extra `neo`. The 37 test modules that use pandas directly skip
  without it (`pytest.importorskip`), as the test modules of the other extras do, so that
  the core tests (`-m core`) run in an environment with NumPy and SciPy only; the
  core-only test now checks the content of the tables, not only that the steps run.
- **REQ-012, the internal style guide.** It is the *Vocabulary* section of
  `CONTRIBUTING.md`, with the table of prohibited words between markers.
  `test_vocabulary.py` reads the table from there.
- **REQ-019, the release gate.** `RELEASING.md` holds the gate. Until the HIIE decision is
  on record, the package carries the classifier `Private :: Do Not Upload`, which PyPI
  refuses, and `make publish` stops.
- **REQ-018.** `tools/check_traceability.py` checks GAPS.md against §14.1 and §14.2.
  `--files` also lists the test files named in §14 that do not exist yet.
- **§8.3 item 6, REQ-309 (b) and REQ-354, where the GPL adapters live.** REQ-309 (b)
  names a separate distribution `evoparam-adapters`; §8.3 and REQ-354 name a separate
  directory with its own licence notice. Both are met by `adapters/` at the root of the
  repository: GPL-3.0-or-later notice in `adapters/LICENSE`, an SPDX line in every
  script, excluded from the source and binary distributions (`MANIFEST.in`: `prune
  adapters`), and never imported by the package. The scripts use only NumPy and the task
  files, so they can move to their own repository unchanged. The ER-detect and CRP
  adapters are not verified against the tools, which are not installed here; their
  README says so.
- **§8.2, the hidden split.** 20 % of the cells of every group: 2 of 10 in `core`
  (12 of 60 cells) and 10 of 50 in `full` (60 of 300). One hidden seeds document serves
  both suites. It holds a random 256-bit nonce, so the hash cannot be inverted by trying
  seeds, and both suite files hold its SHA-256 (canonical JSON). The document is kept
  outside the repository. Methods see hidden cells only through the evaluator.
- **REQ-355 and REQ-356, the deposit.** The deposit and the publication of the
  commitment are public steps that wait for the owner's HIIE decision.
  `evoparam benchmark prepare-deposit` writes, locally only, what the deposit will hold:
  BIDS, NWB, inputs, truth, and a manifest with the generator version, the public seeds,
  the file hashes and the commitment. Both criteria stay strict xfails until the deposit
  is made.
- **§8.4, "signed JSON".** Read as content-addressed: `signature.digest` is the SHA-256
  of the document without its signature. It detects any hand edit, and the leaderboard
  refuses such a file. It is not a key-based signature: anyone can recompute it, so it
  does not authenticate the submitter. Pull-request review does that (§8.4).
- **REQ-352.** A result holds no clock time and no machine path, so the same inputs and
  outputs give the same bytes. It does record the package version, so byte identity
  holds within one environment, which is what the test checks.
- **§8.1, truth on real data.** No dataset of §8.2 ships human N1 ratings, so the T1/T2
  truth is the consensus of at least 2 of the 4 rules named in each index; contacts where
  exactly one rule detects a response are not scored. The T6 truth is the ERP CORE
  definition computed on the noise-free average (simulated) or on the average (real).
- **REQ-350, "the reference CI machine".** No CI machine is defined yet. The measured
  time is that of the shared 4-CPU container used for the step, with one process
  (`docs/benchmark/run-core.json`).
- **§8.3 item 1, `vanblooijs2023_code`.** The registry's rule taken from the published
  code, applied to the letter, is run as a seventh threshold baseline. Its 50 µV floor on
  the baseline SD puts the threshold at 170 µV, above every simulated N1 of `core`, so it
  detects nothing there. That is a property of the code as published, not a failure of
  the run.
- **REQ-231, "the corrected statistics of the reference examples in [LIT-erpmeas-2]".**
  Kiesel et al. (2008) was read on 2026-10-12 from the authors' copy at the University
  of Freiburg (the URL of `LITERATURE.md` followed by `/at_download/file`), pages
  250-264. It gives the corrections `t_c = t / (n - 1)` and `F_c = F / (n - 1)^2` only as
  Equation 1 and reports no t or F value: its tables are Monte Carlo summaries over data
  sets whose participant data are not published. Pages 265-274 could not be read from the
  build environment. No corrected statistic of that paper has been found to reproduce, so
  this part of REQ-231 is not marked met: it stays an expected failure
  (`tests/stats/test_stats_jackknife.py::test_req231_kiesel_worked_example`) whose reason
  lists what was read and what was refused, until a printed corrected statistic is found
  on the pages not read or the owner rules on it. The criterion is not changed. The
  corrections are checked on the worked example of [LIT-erpmeas-6] (Smulders 2010), and
  the one worked number of [LIT-erpmeas-2] (the interpolated criterion crossing of p. 254,
  64.46 ms) is checked by `test_req231_kiesel_interpolation_example`.
- **REQ-115 and REQ-295, the compiled fit of the `fast` extra.** With numba installed
  (§9.2, extra `fast`), the engine fits the structures made of Gaussian and
  skewed-Gaussian components, with an exponential background or one without non-linear
  parameter, in compiled code (`evoparam/models/_fast.py`). It is the same algorithm as
  the NumPy path, step by step, but not the same rounding: its exponential and error
  function are written so that the loops vectorise (within one ulp of NumPy's, and Cephes'
  erfc, which SciPy uses), a component is 0 beyond 12 widths from its centre (below 1e-31
  of its peak; on the side where a skewed Gaussian decays faster, beyond 12 / sqrt(1 +
  alpha^2) widths, where `erfc(u) <= exp(-u^2)` gives the same bound), the compiler fuses
  products and sums (FMA) and vectorises inner products, and the SVD of each trust-region
  step comes from the one-sided Jacobi method instead of LAPACK. Any other structure, a
  design that is not clearly of full rank, and a machine without numba take the NumPy
  path, which stays the reference. REQ-295 is read with the extra as part of the platform,
  as the GPU backends of P4 are: the two paths agree within the solver tolerance (1e-6
  relative on parameters and features, `tests/models/test_fast_kernel.py`), and the reference fixture within 1e-9 with either path
  (2.4e-12 and 3.5e-12 measured), so `tests/test_determinism.py` holds with and without
  numba. As with any change of rounding (CHANGELOG, "Engine speed"), a few ill-conditioned
  fits of real data with many components end on another optimum: 12 of 360 ds004080 fits,
  with no status or shape changed, one number of components changed (UMCU19 el088, 5 to 6,
  with a lower criterion; a channel that already moved under 1e-15 perturbations of the
  data), and the N1 agreement of REQ-114 equal or higher.
- **REQ-112 on real baselines, "at the default `delta`".** Read as both defaults a user
  meets: the `ccep` preset (the test's model before 2026-10-13) and the package defaults
  (`ResponseModel()` without a preset). Both must keep >= 95 % of the pre-stimulus windows
  of the two REQ runs (ds004080 UMCU01 run-021448, UMCU04 run-031148; 80 windows, the
  first 40 contacts of each) without a component; the test is parametrised over both.
  Choices are made on 920 windows of 23 other runs (development) and reported on these 80
  once. The calibrated acceptance, with the spread of the noise spectrum across baseline
  segments added and `alpha` 0.04 chosen on development runs (94.6 % silent there, 87.3 %
  without the segment term, 90.1 % at `delta` 10), kept 92.5 % of the test windows (74 of
  80) against 96.25 % (77 of 80) at `delta` 10: the package default is the fixed `delta`
  10 again (§3, §5.3), and REQ-105 is an expected failure as before the calibration.
- **REQ-004, a step skipped under a width.** REQ-004 says components are accepted or
  rejected only by the model-selection criterion, and its test forbids skipping a
  component under a comparison of its size or width with a threshold. Read as covering
  every step that may change a component, not only its deletion: the final shape choice
  (§3 "`shape_choice`") refits every component, and which shape a narrow component keeps
  is decided by the criterion. The criterion is HQC, not AIC: with AIC on every component,
  asymmetries the data do not determine moved the mode of narrow, low-SNR components out of
  the REQ-105 match (recall 0.8863 to 0.8852 on development seeds); with HQC the recall is
  0.8865 (CHANGELOG, "Final shape choice on every component").
- **REQ-003, the resolution check.** The check (§3) passes over a component that has
  another component within twice its FWHM, since `merged:k,k+1` would then name that
  neighbour. It first passed over same-sign neighbours only; a code path that depends on
  the signs of components, which REQ-003 forbids without `polarity`. Read as: the rule is
  the same for neighbours of either sign. REQ-111 is unchanged (its pairs have one sign).
- **A preset that asks for the calibrated acceptance.** `delta="calibrated"` is the
  sentinel, in a `Preset`, in `ResponseModel(...)` and in `FitSettings`; it is recorded
  as given, so a stored fit and a preset written to JSON say which rule they used. A number is the fixed fall of BIC. `delta=None` keeps its meaning
  of "not given" (the preset's value, else the fixed 10), so no stored setting changes its
  reading. Any other string is refused (`SettingsError`). Fits recorded under 1.2 with
  `delta=None` (the calibrated default of that version, not released) are described as
  calibrated by `methods_text()`.
- **§5.6 method 4, the `ftract` rule has no `k`.** The registry row gives the threshold as
  "`Z` between 3 and 7", a range, not one value. `evoparam/existence/rules.json` therefore
  has `"k": null`, lists `k` under `unspecified` (with `"verified": {"k": false}`) and
  hints at the published range and at `max_peak_delay=0.2` for the optional peak-delay
  cut. As for the rows whose paper does not state the baseline, running the rule without
  `k` raises `SettingsError` naming it (REQ-137). Window, baseline and band are the row's
  (0–800 ms, −400 to −10 ms, 1–45 Hz). The benchmark sets `k = 5`, "the middle of the
  published 3 to 7", without the peak-delay cut, and records it in the method's settings
  (`evoparam/benchmark/baselines.py`, §8.3 item 1). No measurement chose this value.
  Tests: `tests/existence/test_rules_registry.py::test_req137_unspecified_fields_raise`;
  the benchmark tests run the rule with `k = 5` (`tests/benchmark/test_suite_core.py`).
- **REQ-133 for the `ftract` rule.** "Every registry rule" includes `ftract`; its row has
  no `k`, so the test passes `k` explicitly, at 3, 5 and 7 (both ends and the middle of
  the published range). The statistic is taken after the 1–45 Hz band-pass, so the
  waveform of the other rules (one sample at the threshold) does not lie at the threshold
  after the filter. The test builds one that does: a 20 Hz Hann-tapered sinusoid in the
  baseline (inside the band; tapered so that it leaks below `|z| = 1e-3` into the window)
  and a 10 Hz burst in the window whose scale is solved (Brent) until the band-passed
  statistic equals `k ± 1e-6` to 1e-9. The row names the band, not the filter; the test
  uses the rule's zero-phase 4th-order Butterworth. The decision flips at `k ± 1e-6` for
  the three values (`test_req133_ftract_flips_at_threshold`).
- **§5.3 item 12, intervals of a component whose onset is masked.** "`Component.ci`
  inflated by the masked fraction of its support" is read as: the support is `[onset,
  offset]` of the component (where it reaches `onset_fraction`, 10 %, of its peak, §5.5.1),
  read on the samples of the epoch; the onset lies in a masked interval when the sample
  nearest to it is masked (a NaN sample counts as masked, §5.5.9); `f` is the fraction of
  masked samples of the support; every interval of the component (its parameters, `A` and
  its features, whatever the uncertainty method) keeps its centre and has its width
  multiplied by `1 + f` ("inflated by the fraction": increased by that fraction of
  itself), and its standard errors by the same factor, so that `se` and `ci` stay
  consistent. Other components keep their intervals. The point estimates do not change.
  The masked samples already have zero weight, so the Jacobian interval is already wider
  than without the mask; the factor is on top of it, for the disagreement of cleaning
  pipelines the spec gives as the reason. Recorded in `diagnostics["masked_support"]`
  (`component`, `masked_fraction`, `factor`) and the warning `masked_onset:<i>`.
  Example (`tests/models/test_varpro.py::test_masked_onset_widens_the_intervals`, a
  Gaussian at 100 ms, FWHM 40 ms, a 55–75 ms gap over its onset): `f` = 0.130, latency SE
  0.517 ms without the mask, 0.536 ms with it, 0.606 ms widened.
- **§5.5.9, decimated search of a long window.** The rule applies when the fit window is
  longer than 10 s *and* the sampling rate is 1 kHz or more, as written (a 50 s window at
  500 Hz is not decimated). The factor is the smallest integer `q` that leaves at most
  20 000 samples in the window; a 15 s window at 1 kHz has 15 000 and is not decimated.
  The decimated copy is made of block means of `q` samples (a moving-average low-pass
  before the decimation, so that the copy does not alias; masked samples are left out of
  the means, an all-masked block is masked; the last incomplete block is dropped; the
  standard errors and the plus-minus reference are decimated the same way). "The stepwise
  search" is read as the whole search (candidates, forward steps, merges, reseeding,
  final shape choice, resolution check), whose warnings are kept. "Refits at full
  resolution": the model found is refitted on the full-resolution window from the
  parameters found, with the full-resolution noise model (for `shape_choice="penalised"`,
  a skewed component is then re-estimated as in the final shape choice; with
  `selection="cv"` every model of the path is refitted and the choice made at full
  resolution); the criteria, covariance, intervals and features are those of the refit.
  The note is a key of the provenance step, `decimated_search` (`factor`, `n_samples` of
  the search, `n_samples_full`), and is repeated in `diagnostics` and `steps`. Measured on
  20 development waveforms (12 s at 2 kHz, three components of 0.4 to 1.5 s, seeds 100 to
  119): same number of components as the search at full resolution in 20 of 20, features
  within 8e-4 standard error of it (`tests/models/test_long_window.py`).
- **§5.5.5, the bootstrap distribution.** "Also returns the bootstrap distribution for
  plotting": §4.3.4 has no field for it, so `Component` gains one, `bootstrap` (None
  unless `uncertainty="bootstrap"`): one array of the successful draws per name, the
  shape's parameters, `A` and the six features with an interval. It is saved and loaded
  with the result (an optional field: the frozen containers of schema 1 load unchanged).
  "Intervals on every parameter" is read as every parameter of each component; the
  background has no interval at any uncertainty level, and gets none here. The matching
  of draws to components follows the engine's order (the components of a `ResponseFit`
  are sorted by latency, which can differ from the engine's order for overlapping
  skewed components); the profile intervals use the same matching.
- **D6, the four ERP CORE windows of the `erp` preset.** The table of §6.1 labels three of
  the seven components that REQ-274 names (P3, N170, N400) and gives no window for MMN,
  N2pc, LRP and ERN; §6.2 gives the source of the `erp` windows, ERP CORE
  [LIT-erpmeas-39], and D6 asks for published windows with citations. The windows added
  are ERP CORE's published measurement windows, the ones `validation/erpcore.py` and task
  T6 of the benchmark (`evoparam/benchmark/erpcore.json`) already use: MMN 125–225 ms,
  N2pc 200–275 ms, LRP −100–0 ms, ERN 0–100 ms, all negative. No value was chosen here.
  They come after the six labels of the table, which keep their windows and their
  precedence where windows overlap (a negative component at 150 ms is still N1). The
  label windows are names only (§5.5.7): in a stimulus-locked fit a negative component
  at 50–100 ms is named ERN and one at 200–225 ms MMN. REQ-270 compares the preset with
  the table: the new labels are a listed deviation explained in the preset's notes
  (`DEVIATIONS` in `tests/presets/test_presets.py`); `erp` 1.1 → 1.2.

## 5. Requirements that cannot both be read literally

### 5.1 `prep.reference_sensitivity` runs the existence test (§5.4) — import graph (§4.2)

- §5.4, method 3: `reference_sensitivity` "runs the existence test (§5.6) under each
  scheme". §4.2: `prep` may import `data`, `utils` and `shapes` only, and `models`
  imports `prep`, while `existence` imports `models`: an import `prep -> existence` would
  break the rule and close a cycle.
- Resolution (the feature wins, the graph stays acyclic): `evoparam.prep` keeps a registry
  of detectors, `prep.DETECTORS`, filled with `prep.register_detector(name, detector)`.
  `evoparam/__init__.py`, the package root (outside the per-module rules), registers
  `existence.existence_detector` as `"existence"` once both subpackages are imported, and
  `reference_sensitivity(..., detector="existence")` looks it up by name. The interim
  threshold detector stays available as `detector="threshold"`; a callable is also
  accepted.
- Resolved import table for this point: `prep -> {data, utils, shapes}` (unchanged);
  `existence -> {data, models, utils}` (unchanged); root `evoparam -> prep, existence`
  (registration only).
- Tests: `tests/prep/test_reference.py::test_reference_sensitivity_runs_the_existence_test`
  (the table equals the existence test's detections) and
  `tests/prep/test_reference_sensitivity.py` (REQ-084 on ds004080, `attrs["detector"] ==
  "existence_detector"`); `tests/test_import_graph.py` unchanged.

### 5.2 REQ-114: the published rule's peak is not always an N1 (§5.5.10, §5.5.7)

- REQ-114 compares "the model latency" with the published rule's peak on the channels where
  both find an N1. The rule's peak is the most negative sample of 9–100 ms (registry row)
  or the largest local minimum there (published code). On the REQ runs (the example run
  and the REQ-371 runs) that peak is an N1 trough of the average the rule reads (the
  deepest negative interior local minimum in 9–50 ms) on 39 of the 74 channels where the
  registry row and the preset both find an N1, and on 24 of 30 with the code. The other
  peaks are the first samples of the window on the recovery tail (18 and 3), slow
  deflections at 52–100 ms (16 and 2) and one more each (the deepest sample just after
  50 ms; a positive-valued minimum). A model whose N1 latency is a trough of the data is
  therefore bounded by 53 % (registry row) and 80 % (code). Any latency in 9–50 ms is
  bounded by 78 % (58 of 74) and 93 % (28 of 30); reaching the second bound would mean
  naming an N1 at the window start, where the average is on the recovery tail and has no
  trough. The criterion is unchanged: the tests stay expected failures with the measured
  values.
- "The model latency" is read two ways, as before: the latency of the component the
  preset names N1 (§5.5.7: the larger `|A|` of the negative components in 9–50 ms; the
  test as written) and the trough of the fitted curve (`fitted_peak`, K3). On the
  channels where the rule's peak is an N1 trough, the trough is within 2 ms on 24 of 24
  (code) and 31 of 39 (registry row), the component on 11 of 24 and 18 of 39. On real
  N1s the engine's fit puts its trough on the data's trough; the component's own mode
  misses because the N1 is fitted by overlapping components (several negative components
  in 9–50 ms, the one named N1 often 20–250 ms wide, about half in cancelling pairs, every
  fit `ill_conditioned`). The label rule of §5.5.7 is kept.
- With the registry row, 8 of the 39 N1 troughs are missed by the fitted trough. On 6 of
  them the trough of the preset's own input misses too: the rule reads the monopolar
  average, the `ccep` preset the adjusted common average, which moves shallow troughs
  (sub-ccepAgeUMCU03: the registry row's peak is at 32.7 ms on 35 of the 70 channels, the
  trough of the channels' median, a common-mode deflection the common average removes). Fitted on the
  rule's own average (no re-referencing; a diagnostic, not the preset), the input's trough
  is within 2 ms on 33 of 33 such channels and the fitted trough on 29 of 33. The other
  misses are flat-bottomed N1s 20–70 ms wide, whose deepest sample moves with the noise.
- Model changes tried for the component latency, chosen on the five development runs
  (UMCU07, 19, 28, 40, 49), none adopted (component / trough within 2 ms, registry row,
  then code; current 26 % / 54 % and 33 % / 81 %): `reject_cancelling=True` 26 / 51 and
  40 / 80; `reseed="always"` 29 / 57 and 36 / 82 (1.5 times slower); `min_fwhm` 2 ms
  26 / 50 and 37 / 79; `amplitude_prior` 1 30 / 54 and 36 / 86; `max_components` 8 29 / 49
  and 33 / 76; shapes plus `exgaussian` 22 / 53 and 32 / 84, plus `biexponential` 24 / 55
  and 30 / 80, plus `gamma` 29 / 47 and 41 / 77. Each moves one to three of 20 to 38
  channels, in both directions. The two that also raised the development kappa were
  checked on the REQ runs and lowered it there: shapes plus `gamma` (development 0.40 to
  0.48 and 0.39 to 0.44; REQ runs 0.46 to 0.44 and 0.40 to 0.34, code recall 0.88 to 0.74)
  and `amplitude_prior` 1 (development 0.40 to 0.42 and 0.39 to 0.40; REQ runs 0.46 to 0.44
  and 0.40 to 0.32, code recall 0.88 to 0.76). Neither is used.

- A third reading (2026-10-13): the rule's own peak operator applied to the fitted curve
  (components and background) instead of the noisy average, so that both sides measure
  the same quantity. On the REQ runs: 84 % (62 of 74, registry row) and 90.0 % (27 of 30,
  code). On the development runs: 66 % and 86 %. Every miss is either a peak the
  preset's own input does not have (the reference: 10 and 3) or a tie within noise
  (2 and 0). None is a misfit of the model. Fitting the recording reference (as the
  published pipeline reads it) removes the reference misses. It gives 91 % / 95 % on the
  development runs but 88 % / 88 % on the REQ runs, and lowers REQ-113 under the registry
  row, so it is not adopted (CHANGELOG, "N1 case: the two readings"). The test is
  `test_req114_rule_peak_on_the_fitted_curve` (xfail).
- A fourth reading does the comparison in one reference, without changing the preset. The
  adjusted common average subtracts one curve from every channel. Adding that curve back to
  each fitted curve gives the fit in the reference the rule reads, and the rule's operator
  then reads it. With this reading REQ-114 is met under both readings: 91 % (67 of 74) and
  90.0 % (27 of 30) on the REQ runs, 91 % and 95 % on the development runs, and the same on
  the NumPy path (91 % and 90.0 %). The margin is zero channels on the REQ runs. Test:
  `test_req114_rule_peak_on_the_fitted_curve_in_the_rules_reference`. The component
  reading (the test as written) stays an expected failure. REQ-273 is still not met,
  because it also needs REQ-113 (§5.8). The criterion is unchanged.
- The numbers above read the registry row in the code's frame. In the row's own frame
  (§5.8, 2026-10-13) the third reading gives 86 % (59 of 69) on the REQ runs and 69 % on
  the development runs. The fourth gives 93 % (64 of 69) on the REQ runs and 89 % (32 of
  36) on the development runs, so it is still met on the REQ runs, with a margin of one
  channel. The code's numbers are unchanged. The NumPy path was not re-measured in the
  new frame. With the row read as a negative peak (§5.8), the fourth reading gives 82 %
  (47 of 57) on the REQ runs and 84 % on the development runs, so it is no longer met
  with the registry row. 7 of the 10 misses are the rule's peaks at 87-100 ms: small
  local minima of the noisy average on a slope, where the fitted curve has no local
  minimum. The third reading gives 72 %.

### 5.3 REQ-105 (recall >= 0.95 from SNR 3) — REQ-112 (>= 95 % of pure-noise windows silent)

- REQ-105 asks a recall of 0.95 down to SNR 3 and a FWHM of 3 samples; REQ-112 asks no
  component in >= 95 % of pure-noise windows at the default `delta`. A Gaussian of FWHM 3
  samples at SNR 3 carries a chi-square of about 20; BIC with `delta` 10 asks about 29.
  With a fixed `delta` the two cannot both hold: recall and white-noise false components
  are 0.875 and 0 % at `delta` 10, 0.947 and 3 % at 0, 0.956 and 5.5 % at -2, and coloured
  noise needs a larger `delta` than white noise for the same rate (rho = 0.9 about 3 more).
  Even an oracle matched filter that knows the latency range and width finds 73 % of the
  FWHM-3, SNR-3 components at 5 % false alarms: the SNR-3 levels sit at the detection
  limit, and the recall of 0.95 is reachable only because the other levels are found.
- Resolution (2026-10-13, revised the same day): the calibrated acceptance
  (`delta="calibrated"`, §3) meets REQ-105 (recall 0.950, precision 0.986 on the test's
  seeds; 0.9545 and 0.980 on 1 000 per K) and the simulated part of REQ-112, but on the
  ds004080 baselines of the two REQ runs it keeps 92.5 % of the windows silent, where
  `delta` 10 keeps 96.25 % (§4 "REQ-112 on real baselines"). REQ-112 is kept: the package
  default is `delta` 10, and REQ-105 stays an expected failure on its recall (0.883 on the
  test's seeds, 0.887 on 1 000 per K; precision 1.00). A preset or a user who wants the
  recall asks for `delta="calibrated"`.

### 5.4 REQ-106 with the default pair — the latency of §5.5.1 and the pair of REQ-106

- REQ-106 asks a latency MAE <= 0.5 sample for gamma and biexponential truths fitted with
  the default `("gaussian", "skewed_gaussian")` pair; §5.5.1 defines the latency as the
  mode of the fitted component. Without noise, the best skewed Gaussian puts the mode 0.47
  (gamma, kappa 3) and 0.54 (biexponential, decay / rise 4) sample from the truth's, on
  average over the widths 3 to 25 samples (1.0 and 1.2 at 25 samples); noise adds to it.
  A latency defined from the fitted curve cannot be the mode of both a skewed-Gaussian
  truth and a gamma truth of the same fitted curve: the pair does not hold the gamma and
  biexponential shapes.
- Resolution: the latency definition and the pair stay as specified. Skewed truths are
  met with the final shape choice (§3; MAE 0.46 on the test's seeds with HQC, 0.45 with
  AIC from 8 samples, 0.52 before either); gamma and biexponential stayed expected failures
  until models 1.5, third pass below
  (0.721 and 0.599 on the test's seeds, 0.706 and 0.586 on 1 000; 0.76 and 0.60 before). Measured and not kept (development seeds): refitting the peak region
  only (above 0.3 or 0.5 of the peak; gamma 0.75 to 0.61, biexponential unchanged at 0.62,
  skewed and Gaussian truths worse), a parabola through the samples above 0.6 of the peak
  (biexponential 0.62 to 1.16), a shift of the fitted curve onto the peak (no gain), and a
  constant shift of the latency toward the later mode of these truths (rejected unmeasured:
  it would be tuned to the truth shapes of the test).
- Second pass (2026-10-14; development seeds 400 600 + ..., 300 waveforms per truth unless
  stated; the tests' seeds measured once at the end; CHANGELOG "REQ-106 with the default
  pair: second pass"). Readings checked, each kept as it was:
  - **Latency of the fitted component (§5.5.1).** The engine reports the analytic mode
    (`skew_mode`: the root of the derivative, to machine precision), which is the limit of
    the extremum on the grid ten times finer than `1/sfreq`. Reading the extremum on that
    grid literally changes the MAE by at most 0.003 sample on every row (gamma 0.757 to
    0.757, biexponential 0.628 to 0.628); it is not a sample-grid reading.
  - **Truth latency in the test.** `simulate_erp` places the truth by
    `shape_params(shape, latency, fwhm)`: the table's latency is the shape's §5.5.1 latency
    (`t_0 + (kappa - 1) theta`, the biexponential formula), the exact mode; the simulated
    samples equal the shape evaluated at the sample times. Reading the truth latency on the
    same tenfold grid as the fit: at most 0.003 sample of change (biexponential 0.626).
  - **Asymmetry bounds.** `alpha_max` is 10, the bound of §5.5.1, not narrower. At the
    least-squares optimum without noise the gamma fit has alpha 4.9 (inside the bounds);
    the biexponential fit sits on alpha = 10, and widening to 30 moves its mode further
    from the truth's (1.2 to 2.1 samples at 25 samples; with noise, biexponential MAE 0.628
    to 1.107, gamma unchanged at 0.757).
  - **Starting values.** On 36 noisy gamma waveforms of 12 and 25 samples, the engine's
    skewed fit and a fit from eight asymmetry starts (-2 to 10) reach the same least-squares
    cost (within 0.2 in 300) and the same mode (within 0.11 sample): the engine already
    returns the least-squares optimum, so no start or refinement can move it.
  - **Other settings measured** (gamma / biexponential MAE, 0.757 / 0.628 with the
    defaults): constant background 0.750 / 0.615; `noise_model="neff"` (§5.5.3 step 1)
    0.752 / 0.631; the final shape choice on every component whatever its width 0.706 /
    0.572 (Gaussian truth 0.251 to 0.271, ex-Gaussian 0.402 to 0.424: not kept); two
    components forced (`selection="fixed"`) 0.802 / 0.679; the skewed Gaussian alone
    0.682 / 0.583.
- What the measurements point to: with noise, the mean signed error of the skewed fit at
  each width equals its noise-free bias (gamma -0.10, -0.21, -0.52, -1.04 sample at 3, 6,
  12 and 25 samples against -0.14, -0.24, -0.49, -1.02 without noise), and the spread of
  its mode is large (SD 0.79 and 1.03 sample at 12 and 25 samples). Removing the whole
  bias of the skewed-only fits would leave 0.47 (gamma) and 0.28 (biexponential); with the
  pair's shape choice, 0.56 and 0.37. The limit is the least-squares estimate of §5.5.3
  steps 2 to 6 with a shape of the pair, not the reading of its latency. Not met on the
  tests' seeds (1 000 per condition): gamma 0.720, biexponential 0.637 (development seeds
  0.748 and 0.644); every other REQ-106 row is met and unchanged (CHANGELOG).
- To try next (each changes the estimator of §5.5.3 and needs the owner's decision before
  it can be a default): a bias-reduced estimate of the skewed component (a penalised
  likelihood of the Firth type, or a prior on the asymmetry from the hierarchical layer of
  §5.8), which acts on both the bias and the spread of the mode; a final refit weighted
  toward the peak, measured once (gamma 0.75 to 0.61, biexponential unchanged) and to be
  combined with the former; and a REQ-106 run per §7.2 cell (SNR x width), which locates
  the failure at 12 and 25 samples (gamma 0.75 and 1.28, biexponential 0.55 and 1.22 on
  the development seeds).
- Third pass (2026-10-14, with the owner's agreement to an estimator better than §5.5.3,
  provided no requirement gets worse). Resolved by the penalised estimate of a skewed
  component, now the package default (§3 "Penalised estimate of a skewed component";
  CHANGELOG "REQ-106 with the default pair: a penalised estimate of the asymmetry").
  The latency definition and the pair stay as specified.
  - Met on both seed sets. Development seeds: gamma 0.482, biexponential 0.462 (shapes
    compared by HQC, the kept criterion; by AIC 0.454 and 0.455, and 0.444 and 0.450 on a
    second set). Tests' seeds: gamma 0.477, biexponential 0.463. The expected failures are
    removed.
  - Tried first and not kept (development seeds, gamma / biexponential):
    - a Firth penalty, 0.626 / 0.551;
    - a heavy-tailed prior on `alpha`, 0.621 / 0.568;
    - the prior alone, 0.494 / 0.604;
    - the prior with a stronger peak weight, 0.487 / 0.497;
    - flank weights from the misfit of the shape, biexponential 0.61 to 1.9.

    Each reduced the bias or the spread, not both, or did so for one truth only.
  - Every other REQ-106 row stays met. Against `"hqc"`, the default before, the skewed row
    moves by +0.010 (development) and +0.022 (tests' seeds), the ex-Gaussian row by
    -0.010 and +0.002, the Gaussian row by -0.013 and -0.017 (§3).

### 5.5 REQ-111 at 0.5 FWHM — §5.5.3 steps 4 and 5

- REQ-111 asks one component flagged `merged:<k>,<k+1>` for two equal Gaussians 0.5 FWHM
  apart at SNR 10, and never two components unless flagged ill-conditioned. The pair
  differs from one Gaussian by a chi-square of 1.4 at SNR 10, far below what step 4 asks
  for a second component, so no step adds it and the merge of step 5 never runs.
- Resolution of the first clause: the resolution check (§3, `resolution_check=True`)
  sets `merged:k,k+1` on a component that such a pair would explain as well: one
  component with the warning in 100 of 100 runs at `delta` 10 (0.998 on 1 000; 0.96 and
  0.937 with the calibrated acceptance). It is set on most single components below SNR 80 too: a flag of what
  the data cannot exclude.
- The second clause conflicts with REQ-105. With the calibrated acceptance, 4 of the
  test's 100 runs (59 of 1 000) return the pair as one component plus a false component of
  about 1.7 noise SD elsewhere in the window: it admits a noise component in a few % of
  windows, which REQ-105 needs (§5.3). "Never" in 100 runs needs a false-component rate
  well below 1 %. A stricter threshold for the second and later components only was
  rejected: the REQ-105 waveforms with K = 2 and 3 need the same sensitivity at every step
  (estimated recall 0.92). With the package defaults (`delta` 10 and the resolution check,
  §5.3) both clauses hold on the test's seeds: one component with the warning in 100 of
  100 runs, no two-component fit (998 and 2 of 1 000); at 1.0 FWHM two components in 100
  of 100. REQ-111 is met; the conflict is carried by REQ-105 (§5.3).

### 5.6 REQ-171: how often a posterior mean beats the channel fits (§5.8)

- REQ-171 asks the EB channel latencies to have a lower RMSE than the channel fits on
  ≥ 90 % of simulated datasets at per-trial SNR ≤ 2. The test's design is 4 subjects × 20
  channels × 20 trials, 60 % responding, SNR 1.5: about 39 responding channels with a
  latency per dataset, a between-channel SD of 4 ms and channel errors of 1 to 2 ms.
- Measured bound. With the true subject latencies, the true between-channel SD and the
  standard errors of the engine, the posterior mean wins on 86 % of 200 development datasets
  (seeds 1000–1199); on datasets drawn from the Gaussian model itself (known prior, errors
  exactly normal with those standard errors) it wins on 84 %. The posterior mean minimises
  the expected squared error, not the chance of beating the raw estimates on a given
  dataset: that chance rises as the shrinkage falls (with the new standard errors divided
  by 2, 4 and 10, the EB wins on 93.5 %, 96.5 % and 97 % of the development datasets while
  its RMSE goes from 1.48 ms to 1.48, 1.65 and 1.77 ms, against 1.79 ms for the channel
  fits). Halved on purpose, the new standard errors keep the RMSE and reach 93.5 %, but
  only 5 of 6 and 17 of 20 on the test's seeds; they under-shrink the channels with small
  errors and are not the empirical-Bayes estimate, so they are not used (owner's call).
- What changed (the standard error of the channel latency): the noise of the average (Wald
  error of the fit) plus the mean of the trial jitter (jitter SD beyond noise, pooled by
  REML over the channels, over the square root of the number of trials), truncated to the
  label window. Before, the spread of the trusted trials in quadrature with the Wald error
  counted the noise twice (median |z| 0.54 against the normal 0.67), and the Wald error of
  poorly determined latencies (median 7.1 and 15.5 ms in the two largest bins, where the
  errors had an RMS of 4.2 and 9.7 ms) shrank them too far. Development datasets: wins
  64.5 % → 80 %, RMSE 1.56 → 1.48 ms (channel fits 1.80 ms), between-channel latency SD
  0.905 → 0.943 of the realised one (median). Test seeds: 3 → 4 of 6, 11 → 14 of 20.
- Measured and not used: REML on the channels with `P(response) ≥ 0.5` only (64 %, false
  positives are 7 channels in 200 datasets); the Wald error alone (78 %, 1.50 ms); the
  minimum of the Wald and trial errors (93.5 % but RMSE 1.62 ms, worse than before: it
  under-shrinks); standard errors scaled by a factor estimated by REML in each dataset
  (62 % to 70 %, too noisy with 40 channels); trial-bootstrap errors of the fit
  (`uncertainty="bootstrap"`, 100 resamples, 22 datasets: z-score SD 1.39 against 1.21,
  13 % beyond 1.96, 50 s per dataset); the mean of the trusted trial latencies combined
  with the fit (errors correlated 0.89, RMSE 1.774 against 1.781 ms at best).
- The criterion is unchanged; the test stays an expected failure with the measured values.

### 5.7 REQ-395 (no copyleft package declared as an extra) — §9.2 (`bids` declares `pybids>=0.16`)

- The §9.2 table gives the extra `bids` two packages: `mne-bids>=0.15` and `pybids>=0.16`.
  REQ-395 forbids declaring as an extra any package under GPL, LGPL or AGPL, and its
  acceptance scans the *resolved* dependency tree of every extras set. Every `pybids`
  release that §9.2 allows requires `num2words` (LGPL), and from 0.19 also `frozendict`
  (LGPL-3.0): checked on PyPI for each of the 15 releases from 0.16.0 to 0.22.0 (the one
  locked before). Both requirements cannot hold together.
- Kept: REQ-395, the legal rule that protects the licence model (§13.7; a single copyleft
  dependency reaches the paid modules). `pybids` is withheld from the extra `bids` and
  from `all`; `evoparam._extras.WITHHELD` records it with the reason, and
  `tests/test_dependencies.py` checks that a withheld package is named here. Nothing in the
  package imported `pybids` (the BIDS reader, the `events.tsv` harmonisation and the
  derivatives writer use `mne-bids` and the standard library), so no feature is lost.
  The lock files were regenerated with their own `uv pip compile` commands: the only
  changes are the removal of `pybids` and of the packages that only it required.
- The scan (`tools/licence_scan.py`, standard library only) follows the requirements of
  every installed distribution from the packages that the chosen extras declare and reads
  the licence expression, the short `License` field and the `License ::` classifiers. The
  CI job `licence-scan` installs each of the 22 extras in its own environment and scans it;
  the test runs the same scan on the test environment, and checks that the scan flags a
  copyleft licence. One distribution carries a GPL classifier that is not a licence of what
  is installed: `docutils` (required by Sphinx, extra `docs`) is public domain and
  BSD-2-Clause except `tools/editors/emacs/rst.el`, which is GPL-3.0 and is not in its
  wheel. The scan accepts it only after checking, on the installed file list, that this
  file is absent (`FILE_SCOPED`); it is printed as accepted, not skipped.
- The import-graph test (`tests/test_import_graph.py`) adds the licence allow-list:
  every third-party module that a source file imports must belong to a package whose
  `soup.json` licence is Apache-2.0, BSD-2-Clause, BSD-3-Clause, MIT, PSF-2.0 or MPL-2.0
  (MPL-2.0 is file-level weak copyleft, not GPL, LGPL or AGPL). The PR template carries
  the checklist item.

### 5.8 D8 (both readings of the published N1 rule) — REQ-113, REQ-114, REQ-135, REQ-273

- D8 asks REQ-113 (kappa >= 0.8 for the preset's N1 decision) and REQ-135 (kappa >= 0.7 for
  the default existence method) under both readings of the published rule: the registry
  row (the paper's text) and the published code. Each requirement has one decision, and
  it must agree with both readings. The two readings disagree with each other: kappa 0.06
  on the 46 PT contacts of the example run, 0.32 over the six REQ-371 runs.
- For any decision of the channels, the smaller of its two kappas is bounded. Kappa
  depends only on how many channels the decision marks in each cell of the readings'
  2 x 2 table, so the bound is found by searching every combination (exact;
  `joint_kappa_bound` in `tests/models/test_engine_ccep.py`). REQ-113: 0.274 on the PT
  contacts (0.8 asked), and on the REQ-371 runs 0.656, 0.380, 0.323, 0.783, 0.714 and
  0.557 (0.8 asked on 5 of 6), with the row read as the window's most negative sample in
  its own frame (below; in the code's frame for both, 0.295 and 0.656, 0.343, 0.323,
  0.783, 0.681, 0.563; with the row read as a negative peak, 0.378 and 0.738, 0.387,
  0.478, 0.783, 0.752, 0.557). REQ-135: 0.592 pooled over the REQ-371 runs (0.7 asked;
  0.640 with the negative peak), 0.698 on the development runs, 0.6996 on the segments
  the earlier test used. Under D8 no
  detector can meet either requirement on the spec's data. This is a property of the two
  comparators, not of the model. The tests `test_d8_req113_cannot_hold_under_both_readings`
  and `test_req135_cannot_hold_under_both_readings` keep it measured.
- Owner decision (2026-10-13, revising D8 for the N1 requirements). The reading the
  requirements are judged on is the one the spec itself defines: its §5.6 registry row
  (`|x| > 3.4·SD_baseline`, a negative peak, 9–100 ms; rule `vanblooijs2023`). REQ-113,
  REQ-114, REQ-135 and REQ-273 pass or fail on that reading. The published code's reading
  (`vanblooijs2023_code`) is still measured and reported by its own parametrised tests,
  which stay expected failures where it is not met, but it no longer gates. The reason is
  the joint bound above: no decision can reach kappa 0.8 against both readings on the PT
  contacts (0.274) or on 5 of the 6 REQ-371 runs (at most 0.783), nor 0.7 for REQ-135
  (0.592). The two readings disagree with each other (kappa 0.06 and 0.32).
- Each reading in its own frame (2026-10-13). The §5.6 row gives no baseline, and the
  registry names the ER-detect default (-1, -0.1) s. The row also gives its window on the
  data's time axis. Until this date the REQ-113/REQ-114 tests read the row with two
  details of the published code instead:
  - its SD baseline (-2, -0.1) s;
  - its time axis, which labels the stimulation sample t = 1/sfreq, so that 9 ms falls
    one sample earlier: 7.8 ms instead of 9.8 ms on the 512 Hz example run, inside the
    template's (-2, 8) ms.

  The row is now read on the data's axis with the registry's baseline, as the REQ-135
  test and the benchmark already did. The code's reading keeps its own baseline and axis
  (`FRAMES` in `tests/models/test_engine_ccep.py` and `validation/robustness.py`). The
  choice follows the spec text, not the results: REQ-113 on the PT contacts falls from
  0.46 to 0.24.
- The row's peak (2026-10-13). The row reads "`|x| > 3.4·SD_baseline` with a negative
  peak, ≥ 1 sample", in 9–100 ms. A peak is a local extremum, so the window's most
  negative sample is not a peak when it lies on a monotone slope at the window's edge
  (its first sample on the recovery tail, its last on a drift). The published code reads
  the word the same way: peakfinder's local minima, the first sample excluded. The row
  is therefore read as the deepest strict local minimum of the window, lower than both
  neighbours (read outside the window at its edges), below -3.4 SD. The registry option
  is `peak="strict"`. There is no SD floor and no selectivity, because the row states
  neither; they stay with the code's reading. This is the gating reading of REQ-113,
  REQ-114, REQ-135 and REQ-273 (`FRAMES["paper"]`). The window's most negative sample
  ("extremum", the reading until this date) is still measured.

  The registry's default for `vanblooijs2023` stays `peak="extremum"` for now, because
  changing it changes other committed results:
  - the benchmark's published results for the rule (`docs/benchmark/results/vanblooijs2023__T*.json`);
  - the T2 truth of `evoparam/benchmark/realdata`;
  - every `test_response(method="threshold", rule="vanblooijs2023")` (an algorithm
    version).

  The tests pass the option. Making it the registry default is the owner's call.
- The kappa of 1.00 recorded as REQ-113 met on 2026-10-08 was not REQ-113 as written:
  - the comparator was the code's reading (50 uV SD floor, local minimum), with 1 N1 on
    the 46 PT contacts;
  - the decision was the fitted trough judged by the code's amplitude criterion on the
    rule's own average, not the preset's N1.

  It is reproduced exactly today (1.00, 0.91 pooled) by
  `test_n1_trough_with_the_code_floor_regression`. CHANGELOG, "N1 case: the comparator
  trail".
- Values under each reading (registry row read as a negative peak / code; the row read
  as the window's most negative sample in brackets):
  - REQ-113: 0.01 (0.24) / 0.19 on the PT contacts, 0.44 (0.44) / 0.40 pooled. No
    measured variant exceeds 0.51 / 0.56. Ten of the row's 11 PT detections peak after
    50 ms, outside the N1 window REQ-113 names. A trough oracle on the row's own average
    meets REQ-113 on two of the six REQ-371 runs only, and gives 0.13 on the PT contacts.
  - REQ-135: 0.495 (0.586) / 0.496 with the default method. With the rule's per-channel
    decision (`correction="none"`), the registry row is met on the held-out runs: 0.710
    (0.807). The code reading falls to 0.321. The `ccep` preset keeps FDR across channels
    (§6).
  - REQ-114, read as the rule's peak on the fitted curve in the rule's reference: 82 %
    (93 %) / 90.0 % (§5.2). It is met with the code only.
- Under the registry row (after this decision). REQ-113: many detections of the rule are
  not N1s. Read as a negative peak, its average has no negative local minimum in 9-50 ms
  on 9 of the 11 PT contacts it detects in the example run, and on 0, 1, 1, 2, 5 and 12
  detected channels of the REQ-371 runs (extremum: 16 of 20; 0, 1, 5, 2, 6, 12). The
  condition "every channel the rule detects minus at most one" therefore fails for any
  N1 detector on the example run and on 3 of the 6 REQ-371 runs
  (`test_req113_registry_row_counts_channels_without_an_n1_trough`). A trough oracle on
  the rule's own average reaches kappa 0.132 on the PT contacts, 0.816 pooled (extremum
  0.056 and 0.715). REQ-135: with FDR kept, the best variant chosen on the development
  runs (the `ccep` preset's own chain before the test) gives 0.548 (0.623) on the
  REQ-371 runs; without the correction 0.710 (0.807).
  Measured variants and numbers: CHANGELOG, "N1 case: the two readings".
- REQ-135's subjects. Spec §7.3 lists REQ-135 among the criteria on "ds004080, ≥ 5
  subjects chosen by the rule below". The test therefore uses the REQ-371 list, with every
  trial of each run's first pair of at least 10 trials, as REQ-113 does. The five
  development subjects are used to choose settings.

### 5.9 REQ-274: the `erp` preset as written — the targeted measurement

REQ-274 reads "`erp` on ERP CORE ... recovers the component named by the paradigm ... as a
labelled component on the canonical channel for >= 90 % of participants, with model latency
within +/-10 ms of the fractional-area latency of the same average".

- **As written** (`validation/erpcore.py`, `reading="as_written"`): the `erp` preset itself
  (its own model selection, background and fit window), the paradigm's component as one
  labelled component (label window = measurement window +/- 30 ms), its own latency (the
  50 % area latency of that component's curve). Measured 2026-10-09: LRP 90 % (met), ERN
  85 %, N400 57.5 %, N170 45 %, MMN 37.5 %, N2pc 36 %, P3 30 %. Kept as its own tests
  (`test_req274_*_as_written`, expected failures with these values): not met.
- **Targeted measurement** (`reading="targeted"`, the default of the script): the `erp`
  preset derived by `presets.targeted` — three components of the component's polarity in
  the measurement window +/- 25 ms, no background; the component is the sum of its labelled
  parts. Measured 2026-10-13: ERN 95 %, LRP 95 %, MMN 100 %, N170 95 %, N2pc 92 %, N400
  95 %, P3 90 % (no margin). Two limits, stated with the numbers: the settings (three parts,
  25 ms, no background) were chosen on the same 40 participants (a development measure, to
  be confirmed on ERP data not used to choose them); and with three free components and no
  background in a window about as wide as the measurement window, the fitted sum follows
  the average closely, so the latency agreement says that the fit reproduces the wave, not
  that a single component was recovered. The targeted measurement is therefore reported as
  a reading of REQ-274, and the requirement stays marked not met as written.

## 6. Readings of build step S20 (§7.4, §9, §11.8, §13.7)

Each point below is a reading needed to apply a requirement of S20 literally, or a place
where the literal acceptance cannot hold. Where a criterion is not met, the test is a strict
expected failure whose reason gives the measured value; no criterion was weakened.

- **CITATION.cff version vs package version (REQ-391).** The package is `0.1.0.dev0`;
  `CITATION.cff` carries `version: 0.1.0`, the version the first release will have, because
  the Citation File Format describes a released work. `evoparam.cite()` prints the same
  version. `tests/test_citation.py` checks that `CITATION.cff` carries the version being
  prepared (the package version itself on a release); RELEASING.md bumps both together.
  No DOI is recorded before the owner's HIIE decision (REQ-019, REQ-398).
- **Ruff rule PLC0415 (REQ-305) vs lazy imports (REQ-292).** PLC0415 forbids imports
  outside the top level, which REQ-292 requires for every optional dependency. It is ignored
  in `pyproject.toml` (`[tool.ruff.lint] ignore`); every other rule of the spec's selection
  stays on.
- **jax in two extras (REQ-292 "exactly one extra").** `bayes` needs `jax` (CPU) for
  `numpyro`, and `gpu` needs `jax[cuda12]`. These are two different requirement strings for
  two installs. Read as: every distribution is declared once per *purpose*. The extras table
  (`evoparam/_extras.py`) lists `jax` under `bayes` and `jax[cuda12]` under `gpu`.
- **`grep errstate` acceptance (REQ-307 iv): met.** `grep -rn "errstate" evoparam/ | grep
  -v tests` reports nothing. The spec places the tests in a top-level `tests/` folder (§11);
  here they live in `evoparam/tests`, so the grep excludes that folder, and the test scans
  every text file of the package, as `grep -r` does, not only the Python files. The
  documented NaN and inf results (0/0, x/0, the logarithm of 0, the square root of a
  negative variance) are computed with `evoparam.utils._numeric` (`divide`, `sqrt`, `log`,
  `log10`), which computes the operation where it is defined and writes the IEEE value
  elsewhere, so no floating-point warning is raised and none is hidden. The helpers return
  the same bits as the plain NumPy operation (tested); the reference fixture and the frozen
  methods texts are unchanged. `hierarchical`, which may import `utils.require` only (§4.2),
  writes the same two lines inline. Two NumPy *warnings* (not floating-point errors) are
  filtered by message where a row has a single value: "Degrees of freedom <= 0" in
  `TrialSet.average` and "Mean of empty slice" in the z-map of `tf.stats`.
- **Network grep (REQ-306): met.** The grep of §11.5 matched 7 lines in 2 modules that are
  not network access ("HTTP range requests" in docstrings, `urllib.parse` for URL text).
  The docstrings now say "HTTP reads with a `Range` header", and `quote`/`unquote` are
  imported from `evoparam.io._fetch`, the one module allowed to touch the network, so the
  literal grep reports `io/_fetch.py` only. As for REQ-307 iv, the scan covers every text
  file of the package outside `evoparam/tests`.
- **Copyleft packages in the resolved tree (REQ-395): met by withholding `pybids`.** See
  §5.7.
- **Preset perturbation (§7.4 item 2).** "±50 %" is read as ×0.5 and ×1.5 of each
  *(choice)* field of `Preset.choices`, one at a time. Intervals are scaled at both ends,
  integers are rounded with a floor of 1, and flags are left alone. "Change of > 10 % in
  recall" is relative to the unperturbed recall; "> 2 ms median latency" is the change of the
  median absolute latency error, in ms. `artifact_window` is compared with its own
  unperturbed case: simulated stimulation with artifacts, since the clean simulation has no
  artifact to remove. `component_labels` is scored by label recall.
- **Sampling-rate recall (§7.4 item 3).** "N1 recall" is the share of the channels where the
  published rule (registry `vanblooijs2023`, both readings: paper and code) finds an N1 at
  the native rate, and where the `ccep` pipeline also finds one at the given rate. The
  criterion compares the recall at 1024 Hz with the recall at the native rate, in percentage
  points.
- **Layer-3 N1 file (REQ-376).** The gate needs a layer-3 N1 metric file. Layer 3 had none
  for ds004080 (the N1 agreement lived in tests). `validation/robustness.py` writes it
  (`validation/layer3/n1_ds004080.json`), and the gate reads it with the ERP CORE files.
  Without a release tag, the baseline is the committed files.
- **Six-step chain (REQ-294): met.** "The final result holds the six steps in order" is
  read as: the six named functions are a subsequence of the recorded steps, in order. Two
  steps that the user calls between them are recorded too, because REQ-294 asks for every
  public function that transforms data: `evoparam.io.epoch` (between `detect_events` and
  `remove_artifact`) and `TrialSet.average` (inside `evoparam.fit` on trials).
  `evoparam.fit` records the steps it runs, `ResponseModel.fit_group` then `test_response`,
  as before. To make the chain hold: reading a saved container appends a read step with
  `format: "evoparam"`; `detect_events` records its step in the record of the event table
  (`EventTable.attrs["provenance"]`), which `epoch` continues; `fit_trials` continues the
  record of the group fit when that record extends the trials' own, and every `TrialFit`
  carries it. `evoparam.reproduce` replays the whole chain (`detect_events`, `epoch`,
  `fit_trials` added to the replayed steps, REQ-333). `tests/data/factories.core_chain`
  runs the chain on the bundled simulated recording.
- **"Modified from" notice vs the vocabulary rule (REQ-396 vs REQ-012).** REQ-396 asks every
  inherited file to say what it was modified from. REQ-012 forbids naming the inherited
  project outside `NOTICE`. The three inherited files therefore say "Modified from the
  upstream project and commit named in NOTICE (Apache-2.0); changes: …", and `NOTICE` names
  them.
- **SPDX scope (REQ-396).** "Every source file" is read as every Python file outside
  `adapters/`; `tools/add_spdx_headers.py --check` enforces it in CI. Adapters are
  external tools run in a separate process, under their own licences. Data, documentation
  and configuration files are not source files.
- **Localize-MI (REQ-373).** The criterion needs the stimulations "that the dataset authors
  report as producing a scalp response" and sham or low-intensity trials. The data paper
  [LIT-datasets-24] reports no such list (its validation is the source localisation of the
  stimulation artifact), the BIDS metadata of a session give the stimulated pair and the
  intensity only, there is no sham condition (lowest intensity 0.1 mA), and the authors'
  conversion code crops the shared epochs at +10 ms after the pulse. The hosts of the data
  (G-Node GIN, EBRAINS) are refused by the network policy of the build machine. Both points
  are strict expected failures (`tests/validation/test_localize_mi.py`); the second reads
  the header of one epochs file on GIN when run with the network.
- **Layer-3 comparison records (REQ-374).** The records are generated from the layer-3
  result files by `validation/layer3_report.py` into `validation/layer3/comparisons.json`:
  for each comparison, the comparator's number, evoparam's number, the difference
  (evoparam minus comparator) with a 95 % paired percentile bootstrap interval over the
  participants or channels, the dataset id, version and DOI, and the comparator definition
  (registry rule id with the SHA-256 of its entry and overrides, or the definition written
  out with its SHA-256, or the commit of an external comparator). Kappa and recall stay in
  the result files (they are agreement measures, not two numbers to subtract). Versions not
  recorded when a result was produced are `null` with a note: the ERP CORE files read on
  2026-10-09 (OSF has no versions; their SHA-256 were not kept) and the commit of
  CARLA_JNM run on 2026-10-08. Both producers record them from now on; until a rerun, the
  version check is a strict expected failure.
- **Traceability counting (REQ-399).** A requirement counts as tested when a test file names
  it in a test name, a docstring or a comment. Other strings do not count, so that a list of
  untested requirements, or the reason of an expected failure, cannot mark them as tested.
- **P3 chunked fit (REQ-290).** §9.1 P3 reads "chunked", but `evoparam.fit` has no `chunk`
  argument. P3 is measured on the plain call and on the memory-mapped set. The full size
  (500 × 256 × 4 096 float64, 4.19 GB) was not run: the memory-mapped variant needs the
  4.19 GB file on a disk that has about 5 GB free, and the in-memory one needs the data
  and its peak (1.5× the data at the CI size since 2026-10-13, 3.3× before) in the 15 GB
  of RAM next to the other jobs of the machine.
- **P6 in a full environment.** P6 is measured with every extra installed. No optional
  module is loaded by `import evoparam` (`optional_modules_loaded` is empty in the results),
  so the time is that of the core.
- **Coverage (REQ-298).** The gates count lines and branches together (`branch = true`),
  which the 100 % target of `evoparam.shapes` needs; the 90 % and 80 % gates therefore ask
  more than the line coverage that REQ-298 names. Every module is measured
  (`[tool.coverage.run] omit` holds the tests only); before, it also left out `plots`,
  `cli`, `stream` and `benchmark`, so the 80 % gate had no data. "Core packages" is the
  list of REQ-298 plus the top-level modules they use (`errors`, `_extras`, `_plugins`,
  ...), in `[tool.coverage.report] include`. "io local readers" are the readers of local
  files (`readers`, `bids`, `brainvision`, `mef`, `nwb`, `neo`, `epochs`, ...);
  `brainvision` and `mef` also read remote files and count with the core (the stricter
  gate). "Remote I/O" is `io/_fetch.py`, `io/fetch.py` and `io/registry.py`, gated with
  the outer layers. The shapes target is reached by the tests of `evoparam/tests/shapes`
  alone (as `test_req298_shapes_coverage_is_100_percent` measures), without any
  `pragma: no cover`. Measured on the fast suite (no slow, perf or network tests, as the
  CI job), Python 3.13, 2026-10-13: core 90.5 % (lines and branches; lines alone 93.3 %),
  shapes 100 %, outer layers 82.0 %.
- **SOUP inventory roles (REQ-396).** `soup.json` lists every third-party package that the
  source imports or the extras declare. Packages imported only through another package's
  API (`packaging`, `tabulate`, `patsy`, `pytensor`) have the role `transitive`. These
  declared extras are not imported by the source: `bidsschematools`, `formulaic`,
  `mne-bids`, `ndx-events`, `nwbinspector`. They are kept, because §9.2 names them; the
  inventory marks them `optional`. `pybids` is withheld (§5.7).
- **Public steps (REQ-308 run, REQ-391 archive, REQ-392, REQ-393, REQ-394, REQ-398).**
  Everything that can be prepared locally is in the repository. The steps that publish
  (PyPI, Zenodo, a public leaderboard, a preprint, the freeze of the name) wait for the
  owner's HIIE decision. Their tests are strict expected failures with that reason.

- **REQ-315 (reference BIDS validator, 2026-10-13).** The requirement is "bids-validator
  ≥ 2.0 in derivative mode, errors not allowed"; its acceptance names the command
  `npx bids-validator@2`. npm has never had a 2.x release of `bids-validator` (latest
  1.15.0, measured 2026-10-09 and 2026-10-13); the BIDS maintainers publish 2.x and 3.x
  on JSR as `@bids/validator`. The test runs npx as soon as npm resolves a 2.x release and
  `deno run -A jsr:@bids/validator@3.0.2` until then: same validator line, same mode, same
  rule (zero errors). It passes on macOS with deno 2 (15 of 15 BIDS tests).

- **REQ-231, Kiesel et al. (2008) part (2026-10-13).** The requirement asks the corrected
  statistics of the "worked numerical examples" of [LIT-erpmeas-2] to the printed
  precision. Read in full (25 pages, the authors' copy archived on 2024-06-01), the paper
  prints the corrections only as Equation 1 and no corrected t or F; its one worked number
  is the interpolation example of p. 254. The test reproduces that number (64.46 ms) and
  Equation 1 on jackknifed scores (the corrected t and F equal those of the retrieved
  individual scores to 1e-9); the Smulders (2010) part reproduces the printed F values.

- **REQ-044, Mayo protocol (2026-10-13).** "Reproduces the train structure of the Mayo
  protocol (≈ 0.2 Hz, 10–12 pulses per site)": over all the runs of the four Mayo
  datasets the median is 10–12 (tested); one run, ds004696 sub-07, has 13 per site in its
  own events.tsv (378 pulses on 27 sites). For that run the structure reproduced is the
  run's (blocks, pulse counts, 0.2 Hz), not the protocol's range, which the data do not
  follow.

- **REQ-134, significance of the cross-projection test (2026-10-13).** §5.6 item 2 tests
  "the projection magnitudes of one half of the non-overlapping pairs" with a one-sample
  t-test. Read as the paper's set (one projection per unordered pair) it reproduces the
  published decisions but rejects pure noise at 0.33 to 0.44 for alpha = 0.05 (the values
  share trials); read as disjoint pairs it is nearly valid (0.145) but its decisions depend
  on the random pairing (0.89 agreement on average). The default is a sign-flip
  permutation test of the same projections (the null of trials of random sign, the peak of
  the profile recomputed for every pattern): published decisions 10 of 10 and a rejection
  rate of 0.05 on noise, measurably better than both readings; the two t-tests stay
  available (`projection_pairs`).

## 7. Static quality rules applied as written (REQ-305, REQ-301)

REQ-305 and REQ-301 are met as written. All the rule sets stay selected. The thresholds
are unchanged (line length 99, complexity 12, 8 arguments, 60 statements), and no package
file is excluded. The commands, on branch `qualite-zero`, against 833968f:

| tool | before | after |
|---|---|---|
| `ruff check evoparam benchmarks docs/tutorials examples` | 1 264 findings | 0 |
| `ruff format --check` (same paths) | clean (440 files) | clean (441 files after the merge) |
| `interrogate -v evoparam` | 68.3 % | 97.1 % |
| `mypy --strict evoparam` | 1 error (patsy) | 0 (172 files) |
| `pyright --verifytypes evoparam --ignoreexternal` | 96.2 % | 96.8 % |

The 1 264 findings before were:

| rule | package | tests | other paths |
|---|---|---|---|
| PLR2004 (magic values) | 211 | 834 | 1 |
| PLR0913 (more than 8 arguments) | 51 | 9 | 0 |
| C901 (complexity above 12) | 46 | 3 | 0 |
| PLR0912 (too many branches) | 44 | 1 | 0 |
| PLR0917 (more than 8 positional arguments) | 25 | 7 | 0 |
| PLR0915 (more than 60 statements) | 19 | 1 | 0 |
| PLR0911 (too many returns) | 12 | 0 | 0 |

A top-level `tests/` folder does not exist: the tests live in `evoparam/tests`, which
`ruff check evoparam` covers (see §6, `grep errstate`). The changes below leave the
behaviour the same, except for the signatures listed at the end of §7.2. Each refactored
function was compared with 833968f on reference cases: results, written files, and the
category, message and source file of each warning. Each magic-value change was checked
by comparing syntax trees, with the new constants replaced by their values. The test
files of each module were run.

### 7.1 Magic values (PLR2004)

Each compared literal is now a named module constant with a comment: what the value
is, and the requirement or quantity it comes from. Constants are private (`_NAME`) in
the package and plain (`NAME`) in the tests. After inlining the constants, the syntax
trees of the changed files equal those of 833968f. A value used by several modules
stays in each module, so that each module reads on its own.

### 7.2 Signatures with more than 8 parameters (PLR0913): the spec vs REQ-305

REQ-305 caps functions at 8 arguments. Yet §5 gives many public signatures more than 8
parameters, and the build added keyword-only options to some of them. A public signature
cannot change without changing the API, so these functions keep one per-line
`# noqa: PLR0913`. Each noqa names the spec section and the counts. Every private function
above the cap was refactored instead. Its parameters are grouped in a private dataclass
or NamedTuple, or it was split in two. The PLR0913 noqa went from 24 to 42. The 60
findings before are now 0: 19 public signatures gained a noqa, and the other 41 functions
were refactored (private functions, internal helpers and test helpers). The same
refactoring removed the noqa of `existence.permutation.permutation_test`.

Public signatures fixed by the spec. Counts exclude `self`, `*args` and `**kwargs`. "Added"
lists the keywords beyond the spec signature:

| function | spec | spec count | actual | added |
|---|---|---|---|---|
| `artifact.remove_artifact` | §5.3 | 28 | 40 | 12 method options |
| `events.detect_pulses` | §5.2 | 14 | 23 | `min_relative`, `recovery_sd`, `ch_names`, `ch_info`, `saturation`, `vote`, `score_channels`, `procedure`, `site_method` |
| `events.detect_saturation` | §5.2 | 7 | 10 | `t0`, `ch_names`, `picks` |
| `events.identify_stim_site` | §5.2 | 7 | 12 | `picks`, `exclude`, `lead`, `hold`, `method` |
| `existence.test_response` | §5.6 | 17 | 23 | `rule_overrides`, `allow_artifact_overlap`, `n_boot`, `projection_profile`, `projection_pairs`, `on_error` |
| `hierarchical.fit_hierarchical` | §5.8 | 14 | 17 | `preset`, `group_fits`, `subject_column` |
| `io.read_bids` | §5.1 | 14 | 15 | `stim_mapping` (§10.2) |
| `io.write_bids_derivatives` | §5.1 | 11 | 14 | `units`, `source_dataset`, `trials` |
| `io.epoch` | §5.1 | 10 | 10 | |
| `io.read_nwb` | §5.1 | 7 | 9 | `select`, `units` |
| `io.from_array` | §5.1 | 9 | 9 | |
| `models.ResponseModel.__init__` | §5.5.8 | 30 | 43 | 13 engine settings (`noise_model` … `threshold_role`) |
| `models.ResponseModel.fit_waveform` | §5.5.8 | 7 | 11 | `random_state`, `noise_ref`, `unit`, `on_error` |
| `models.fit` | §5.5.8 | 14 | 14 | |
| `models.classical_measures` | §5.5.8 | 12 | 12 | |
| `prep.reject_trials` | §5.4 | 9 | 10 | `all_channels_fraction` |
| `sim.simulate_erp` | §5.12 item 1 | 12 | 18 | `jitter`, `jitter_tau`, `noise_sd`, `noise_kw`, `ch_names`, `keep_arrays` |
| `sim.simulate_ccep` | §5.12 item 2 | 14 | 31 | 17 options (time span, layout, response probabilities, noise, variability, artifact shape) |
| `spatial.shared_shapes` | §5.9 | 11 | 11 | |
| `spatial.fit_shared` | §5.9 | 8 | 11 | `smooth_lambda`, `group_fit`, `reject_cancelling` |
| `stats.bootstrap` | §5.11 | 8 | 10 | `window`, `expand` |
| `stats.compare_conditions` | §5.11 | 8 | 12 | `by`, `conditions`, `ci_level`, `n_boot` |
| `stats.reliability` | §5.11 | 7 | 9 | `window`, `picks` |
| `stats.trials_needed` | §5.11 | 5 | 9 | `window`, `picks`, `existence_kwargs`, `extrapolation` |
| `stream.StreamFitter.__init__` | §10.6 | 11 | 17 | `saturation`, `existence_alpha`, `existence_max_pulses`, `existence_looks`, `random_state`, `mode` |
| `tf.tf_fit` | §5.10 | 25 | 27 | `stft_window`, `channel_correction` |
| `trials.fit_trials` | §5.7 | 16 | 21 | `preset`, `trial_model`, `null_rule`, `ridge_cv`, `on_error` |

In nine rows the spec signature itself has 8 parameters or fewer. Only the added keywords
put the function above the cap (`detect_saturation`, `identify_stim_site`, `read_nwb`,
`fit_waveform`, `fit_shared`, `bootstrap`, `reliability`, `trials_needed`, and
`reject_trials` by one). Moving those keywords into an options object would change a
released keyword API, so they are kept and listed here.

Public functions that the spec does not give a signature for. Some it names only, such as
the generators of §5.12, line "`sim.gen.simulate_erp`". These have keyword-only options
and the noqa reason "public API outside the spec signatures":
`benchmark.run_suite` (10), `io.read_bids_derivatives` (9), `plots.plot_group` (9),
`plots.plot_signals_yshade` (10), `reports.run_pipeline` (12), `sim.simulate_bursts` (11),
`sim.simulate_erna` (23), `sim.simulate_tep` (19), `sim.simulate_ecap` (25),
`sim.simulate_animal_ep` (22), `sim.inject` (11), `sim.make_noise` (16),
`sim.match_components` (9), `stats.score_waveforms` (9), `tf.make_kernels` (9).

Some functions with more than 8 parameters are listed in the `__all__` of their module.
Their subpackage does not re-export them, and neither the spec nor the documentation names
them; they serve the package internally. They now take their settings grouped:

- `existence.permutation.permutation_test` and `shuffle_test` take
  `options=PermutationOptions(...)` (permutations, statistic, cluster threshold,
  polarity, alpha, bootstrap draws, jobs).
- `hierarchical.bayes.sample` takes `nuts=NutsSettings(...)` (chains, warm-up, draws,
  target acceptance, seed).
- `sim.artifacts.pulse_shape` takes `amplifier=AntiAliasFilter(cutoff_fraction, order)` in
  place of `cutoff_fraction` and `filter_order`.
- `sim.generators.stimulation_artifact` takes a `StimulusPulse` and a `RecoveryTail`.
- In private modules: `sim._engine.assemble` takes `Design`, `NoiseSpec`, `Variability`
  and `Recording`; `models._trf.least_squares_trf` takes `TrfOptions`;
  `models._fast.optimise` takes grouped tuples.

The default values are the same, so the public entry points that call these functions
(`test_response`, `fit_hierarchical`, the generators) give the same results; §7 explains
how that was checked. One more change can be seen. The private
`io.readers._read_continuous` now reads the TrialSet labels (`metadata`, `ch_info`,
`species`, `modality`) into a private `_Labels` dataclass. If `read()` gets an unknown
keyword for a BrainVision, EDF or MEF file, it still raises `TypeError`, but the message
now names `_Labels.__init__()` instead of `_read_continuous()`.

### 7.3 PLR0917 (too many positional arguments)

PLR0917 belongs to the `PL` set that REQ-305 selects, so it stays on. Before, it fired 32
times: on 25 private functions of the package (the compiled kernels of `models._fast`,
the BIDS run readers, `spatial.shared`, ...) and on 7 test helpers. It never fired on a
public signature, because the spec makes the options keyword-only. Every finding was
refactored, not ignored:

- the numba kernels take grouped tuples (model, whitening, priors, work spaces);
- the other private functions take a NamedTuple or dataclass, or keyword-only parameters;
- the test helpers take a frozen dataclass of their defaults (§7.5).

### 7.4 PT028 in two package modules

PT028 (a pytest test function with default arguments) fired 27 times on
`existence.test_response` (§5.6) and `tf.stats.test_map`. Both are statistical tests
whose names the API fixes, not pytest tests, so the rule does not apply to them. Before,
there were 27 per-line noqa. Now two `per-file-ignores` entries (`PT028` only) in
`pyproject.toml` say so with the reason. Every other rule stays on in these files. This is
the only per-file rule exemption added.

### 7.5 Complexity, branches, statements and returns

Before, C901, PLR0912, PLR0915 and PLR0911 had 72 noqa and 126 open findings. Now they
have 0 noqa and 0 findings. Each long function was split into named private helpers,
with the order of operations and of random draws unchanged. A warning moved into a helper
gets a `stacklevel` one higher per level, so it still points at the user's call; the
reference cases compare the source file of each warning.

The test helpers that built simulated data from many keywords now take
`**keywords: Any`, read into a frozen dataclass of the same defaults. These are
`GroupDesign` (hierarchical), `OverlayDesign` (artifact), `RecordingDesign` (events),
`EpochsDesign` (prep), `TrialsDesign` (trials), `WaveformOptions` (models), and
`BrainVisionLayout` and `MefOptions` (io). Calls are unchanged, and each helper returns
the same arrays and files as before.

### 7.6 Other noqa kept, each with its reason

Each remaining `# noqa` names its reason on the line. There are 197, against 231 before.
Besides PLR0913 (§7.2), they cover:

- `ARG001`/`ARG002`: arguments of a fixed interface (the shape protocol, scikit-learn's
  `X, y`), or arguments read by the `@verbose` decorator.
- `N803`: `X` in the scikit-learn API (§10.4).
- `BLE001`: user code or optional steps whose failure is reported, never hidden.
- `E402`: sphinx-gallery sections, `pytest.importorskip`, the `sys.path` insert of
  `benchmarks/run_benchmarks.py`.
- `D205`/`D400`: sphinx-gallery titles in the tutorials and examples.
- `F401`: availability checks (`mne`, `tabulate`, `jsonschema`, scikit-learn ≥ 1.6).
- `A001`/`A002`: `presets.list` and `read(format=...)`, named by the spec, and the
  public keyword `map` of `to_mne`.
- `NPY201`: the `trapz` fallback for NumPy < 2.0.
- A few single cases, each explained on its line.

Two `E731` lambdas of `sim.generators._overlap_offsets` were removed (direct returns, same
draws).

### 7.7 Docstrings, typing and pre-commit

- `interrogate` (97.1 %) measures the package as `[tool.interrogate]` configured it
  before this work, with `evoparam/tests` excluded. Ruff's `D` rules are likewise ignored
  for `evoparam/tests/**`. Both settings predate this work, and the spec places the tests
  outside the package (`tests/`). They were kept, and the D rules apply to every other
  file. With the tests counted, interrogate would report 66.8 %.
- `mypy --strict`: `patsy` (extra `stats`) and `pytensor` (extra `bayes`) joined the
  `ignore_missing_imports` override. Both are optional packages without type information.
  No core module is listed.
- `.pre-commit-config.yaml` pins ruff, mypy and interrogate to the versions of the CI lock
  files. The mypy hook checks the whole package with the locked stubs. `end-of-file-fixer`
  leaves out the byte-exact format-1 containers (each checked by its sha256) and the
  generated validation reports.

### 7.8 `pre-commit run --all-files` and the CI lint job

REQ-305 asks for two more checks:

- `pre-commit run --all-files` must pass. It runs the ruff hooks on every file of the
  repository.
- The `lint` job of `ci.yml` runs `ruff check` on `validation` and `tools` as well.

So these checks cover more than the REQ-305 paths. Before, they failed on 275 findings in
`tools/`, `validation/`, `tutorials/` and `docs/conf.py`, and 24 of those files needed
reformatting. These files were fixed the same way as the package, with nothing excluded:

- docstrings were added or rewritten;
- named constants replace magic values;
- the layer-3 report rows take grouped values;
- `strict=True` is set on `zip` where the inputs come from one source;
- each remaining noqa has its reason.

The generated files they write are unchanged: `--check` passes for `gen_soup`,
`gen_glossary`, `gen_registry_doc`, `gen_references_bib`, `gen_api_index`,
`freeze_methods_text` and `validation/layer3_report.py`, and `check_traceability`,
`add_spdx_headers --check` and `layer3_gate` give the same results. The `--help` text of
`benchmark_suites` and `gen_soup` now shows the new one-line summary of the module.

The ruff-format hook also formats the Python blocks of Markdown files. It would reformat
`docs/spec/SPEC.md`, which is never edited. `SPEC.md` is therefore in ruff's
`extend-exclude` (`pyproject.toml`), and no other file is excluded. With this, `pre-commit
run --all-files` passes every hook, and leaves the tree unchanged: ruff, ruff-format, mypy,
interrogate, check-yaml and end-of-file-fixer. nbstripout has no notebook to check.

### 7.9 Merge of integration-2 (4f2c44d)

integration-2 moved on after 833968f (lazy package root, compiled kernel, P1 and P3 work
in the engine, the existence test and the containers, pulse detection, ERP CORE). Where
it changed a function that §7 had split, its change was carried into the split version:

- `events.detect_pulses`: the fallback to the 90th percentile score and the size gate are
  in `_evoparam_candidates` and `_strength_gate`. The deflection size is measured by
  `artifact_size`, which `_event_table` also uses.
- `existence.test_response`: the single copy of the tested trials (`_select`, `_span`) is
  in `_Tester.run`.
- `models._fast`: the lag groups of the whitening are split into `_lag_groups` and
  `_lags`. The reassociated `_finite` replaces the per-entry test of §7, so
  `_all_finite` is gone.
- `models.engine`: the changes are carried into the helpers of §7. These are the compiled
  fit before the objective, the memo of the background bounds, the shapes and the
  features, the array selection of `candidates` (`_select_list`, `_select_array`) and
  `CANDIDATE_SCAN`.

The tests and validation scripts are integration-2's. Their results equal integration-2's
to the bit:

- 128 reference cases match, including the compiled and NumPy fits.
- 68 cases of pulse detection and existence tests match; these cover the 90th-percentile
  fallback, the size gate and every pairing of the projection test.
- The fixture matches (largest difference 0).
- `freeze_methods_text`, `gen_api_index` and `gen_soup` pass `--check`.

The merged tree had 67 new ruff findings (PLR2004 31, docstrings 25, E501 3, C901 2, PT018
2, I001 2, SIM105 1, S110 1), 5 unformatted files and 2 mypy errors, all in code from
integration-2. They were fixed in the same way as in the rest of §7:

- Each magic value is now a named constant, or uses an existing one.
- The two composite assertions of `test_subspace_real` are split in two.
- `validation/robustness.py` `main` and `markdown` are split into named helpers.
- The warm-up of `bench_artifact` uses `contextlib.suppress`.
- The frozen-dataclass test of `engine._shape` reads `__dataclass_params__` once.

One noqa was added: the E402 of the detector registration at the package root.

## 8. Plain entry points for a first try (tutorial 01)

The owner asked (20/10/2026) that a first try of the package take a few lines, as in the
first tutorial of the spectral ancestor tool: one line to make an example, one to fit it and
see the result. Three additions do it without changing the engine or any default:

- `evoparam.sim.gen_response(components, background, noise)` makes one example response.
  Its components are given as `[latency (ms), amplitude (µV), width (ms)]`, the way a
  response is described in words; the times it returns are in seconds, the unit of the rest
  of the package.
- `evoparam.report(times, signal, preset="ccep")` (in `evoparam.reports`) fits one waveform
  with `ResponseModel(preset).fit_group`, draws `plots.plot_fit` on the fit window and
  returns the `ResponseFit`. It sits in `reports` because §4.2 does not let `models` or
  `data` import `plots`.
- `ResponseFit.results_text()` is a short plain-text summary (window, background, one line
  per component in ms and µV, R² and noise level). `str(fit)` and the notebook
  representations (`_repr_html_`, `_repr_pretty_`) show it. The ancestor tool prints its
  summary from `report`; here nothing prints, because REQ-304 allows `print` only in the
  command line: the summary appears when the fit is the last expression of a notebook cell,
  or with `print(fit)`.
- `plots.plot_response(times, signals, xlim=...)` draws raw curves (time in ms, stimulus
  line at 0), the counterpart of the ancestor's spectrum plot, and
  `plots.plot_annotated_fit(fit)` draws each number of the fit where it is read on the curve,
  the counterpart of its annotated model plot.
- The names a first try needs (`gen_response`, `report`, `plot_response`, `plot_fit`,
  `plot_annotated_fit`, `ResponseModel`) are served from the package root, so the tutorial
  imports them in one line. matplotlib stays optional and is imported only inside a plot
  call (`utils._optional.require`, extra `plot`), as the ancestor tool does; `import
  evoparam` does not load it. Because §5.13 has every plot function return its `Figure`, the
  tutorial assigns it (`fig = plot_response(...)`): a returned figure left as the last
  expression of a notebook cell is shown twice.
