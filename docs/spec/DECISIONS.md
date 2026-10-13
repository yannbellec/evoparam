# Decisions on the reviewed specification

`SPEC.md` is the text the owner reviewed (branch `spec`, commit d8cccfd). It is a minimum and
is not edited. This file records:

1. the owner's resolution of each point where the letter of the spec is impossible or
   contradictory (D1–D16 of the audit of 2026-10-09);
2. the resolved import table of §4.2 (D3, D4, D5), which `evoparam/tests/test_import_graph.py`
   enforces and checks against this file;
3. the divergences kept because they measured better than the reviewed procedure. The
   reviewed procedure stays available as an option;
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
| D8 | "The published rule" for N1: the paper's text (extremum) or the published code (local minimum)? | The N1 REQs must pass under both readings, and both results are reported. |
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
- **`latency_order` (step 5).** The reviewed ordered parameterisation
  (`"parameterised"`) stays the default; `"sorted"` is an option. On the ECAP simulations
  neither is better on both kinds and both seed sets.

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

### 5.3 REQ-105 (recall >= 0.95 from SNR 3) — REQ-112 (>= 95 % of pure-noise windows silent)

- REQ-105 asks a recall of 0.95 down to SNR 3 and a FWHM of 3 samples; REQ-112 asks no
  component in >= 95 % of pure-noise windows at the default `delta`. A Gaussian of FWHM 3
  samples at SNR 3 carries a chi-square of about 20; BIC with `delta` 10 asks about 29.
  On seeds other than the test's, the missed components (all at SNR 3) change BIC by -8 to
  +9 when refitted from their true values: the criterion decides, not the search. Recall
  and white-noise false components: `delta` 10: 0.875 and 0 %; 0: 0.947 and 3 %; -2: 0.956
  and 5.5 %. No `delta` meets both on white noise alone; coloured noise and the ds004080
  baselines leave less room.
- Resolution: REQ-112 wins (it is the calibration §5.5.4 gives for `delta`), `delta`
  stays 10 and REQ-105 stays an expected failure with its measured values. The test's
  design steps SNR and width together, so every K = 1 waveform at SNR 3 has a FWHM of 3
  samples (hence the lowest recall at K = 1); the test is kept as written. CHANGELOG,
  "Engine simulation requirements".

### 5.4 REQ-106 with the default pair — the latency of §5.5.1 and the pair of REQ-106

- REQ-106 asks a latency MAE <= 0.5 sample for gamma and biexponential truths fitted with
  the default `("gaussian", "skewed_gaussian")` pair; §5.5.1 defines the latency as the
  mode of the fitted component. Without noise, the best skewed Gaussian puts the mode 0.47
  (gamma, kappa 3) and 0.54 (biexponential, decay / rise 4) sample from the truth's, on
  average over the widths 3 to 25 samples; noise only adds to it (measured 0.76 and 0.60).
  For skewed-Gaussian truths the criterion keeps the Gaussian in most fits (the asymmetry
  does not pay its BIC penalty at SNR 5 to 20), whose centre is 0.19 to 1.55 samples from
  the mode (measured 0.52).
- Resolution: the latency definition and the pair stay as specified; the three cases stay
  expected failures. Refining every allowed shape before keeping the best (§5.5.3 step 4)
  lowers the MAE on other seeds but not to 0.5 on the test's (0.505, 0.750, 0.599) and
  lowers the ds004080 N1 kappa on the development runs (0.40 / 0.39 to 0.37 / 0.34) and the
  REQ runs (0.458 / 0.398 to 0.426 / 0.369): not used.

### 5.5 REQ-111 at 0.5 FWHM — §5.5.3 steps 4 and 5

- REQ-111 asks one component flagged `merged:<k>,<k+1>` for two equal Gaussians 0.5 FWHM
  apart at SNR 10. The warning comes from the merge-and-compare of step 5, which runs after
  a step has accepted a component. The pair differs from one Gaussian by a chi-square of
  1.4, far below what step 4 asks for a second component, so no step adds it; and step 5's
  overlap rule, `|mu_2 - mu_1| < overlap_c (sigma_1 + sigma_2)` with `overlap_c` 0.5, is
  not met by the true pair (20 ms against 17 ms).
- Resolution: steps 4 and 5 are kept as written (merge after an accepted step; the engine
  sets `merged:<k>,<k+1>` there). Merging the candidates a step rejects was measured and
  is not used: with the best rejected candidate the warning appears on 7 % of the pairs
  and 1 to 7 % of single Gaussians; with any rejected candidate on 49 % and 33 to 77 %;
  reading the overlap on FWHM instead of sigma, on 99 % and 91 to 100 %. A warning set as
  often on single components would not say that two were merged. REQ-111 at 0.5 FWHM stays
  an expected failure; its other clause (no unflagged pair) holds.

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
  (500 × 256 × 4 096 float64, 4.19 GB) was not run on the 15 GB reference machine: the
  in-memory peak measured at the CI size, 3.3× the data, would exceed its RAM.
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

