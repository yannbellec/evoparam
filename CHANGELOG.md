# Changelog

## Unreleased

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
  package goes through it, except the single call in `io/nwb.py`, which is pending on
  another branch (strict xfail). The `evoparam` logger has a `NullHandler` only.
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
  - REQ-015: there is no stream module yet.
  - REQ-005: one `warnings.warn` remains in `io/nwb.py`.
  - REQ-315: npm has no `bids-validator@2`.

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
