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
| `data` | `_plugins` | D4: the plugins used are recorded in provenance (REQ-328) |
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
| `presets` | `_plugins` | D5: preset entry points (REQ-328) |
| `utils` | none (the two leaves only) | cross-cutting helpers of the §4.2 figure |
| `_plugins` | none (the two leaves only) | entry-point loader of §10.8 |
| `reports`, `plots`, `cli`, `stream`, `benchmark` | any module above | D3: between them only `benchmark → reports`, `benchmark → plots`, `reports → plots`, `cli → reports` and `cli → plots` |
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
  Evoked has no trials to resample, so its onset has no interval. This is a documented
  xfail.
- **REQ-005, warnings logged.** `evoparam.errors.warn` issues the warning and logs it on
  the `evoparam` logger. Every warning of the package goes through it, except the single
  call in `io/nwb.py`: that file is being edited on another branch, so its test is a
  strict xfail. The package installs a `NullHandler` only (REQ-304), so a warning is not
  printed twice when the application has no logging set up.
- **REQ-007, seeds.** `test_response` draws and records a seed when `random_state` is
  None, as the fits do (REQ-331). `reproduce` keeps an `ArtifactResult` as the result of
  its last step.
- **REQ-012, the internal style guide.** It is the *Vocabulary* section of
  `CONTRIBUTING.md`, with the table of prohibited words between markers.
  `test_vocabulary.py` reads the table from there.
- **REQ-019, the release gate.** `RELEASING.md` holds the gate. Until the HIIE decision is
  on record, the package carries the classifier `Private :: Do Not Upload`, which PyPI
  refuses, and `make publish` stops.
- **REQ-018.** `tools/check_traceability.py` checks GAPS.md against §14.1 and §14.2.
  `--files` also lists the test files named in §14 that do not exist yet.

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
