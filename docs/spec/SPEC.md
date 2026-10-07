# evoparam — Software specification

*Parameterization of evoked responses across modalities and species*

| | |
|---|---|
| Author | Yann V. Bellec |
| Status | Complete specification, private (not for distribution before the institutional decision of §11.4) |
| Date | 2026-10-07 |
| Repository state | `main` at `ef4fad0` (skeleton 0.1, 199 tests); this document on branch `spec` |
| Working name | `evoparam` (final name: §16 Q1) |
| Companion documents | `RESUME-FR.md` (French summary), `LITERATURE.md` (sourced review, reference lists cited here as `[LIT-<axis>-<n>]`), `GAPS.md` (gap table) |

**How to read this document.** §1–§3 say what the product is, what it covers and what it must satisfy. §4 is the architecture and data model; §5 specifies every module with methods, equations, full Python signatures, edge cases, acceptance criteria, tests and tutorials. §6–§8 fix presets, the scientific validation and the public benchmark. §9–§10 cover engineering and the ecosystem. §11 and §13 cover citation, publication and commercial valorisation. §12 gives the build order, §14 the traceability matrix, §15 the risks and §16 the questions only the author can decide.

**Citation keys.** `[LIT-<axis>-<n>]` is reference `n` of the axis section of `LITERATURE.md` (axis prefixes: `fooof` for the review of the spectral ancestor tools, erptools, ccep, artifact, fitting, datasets, domains, tf, variability, naming, erpmeas, citation, needs, standards). `[CODE §x]` refers to the audit of the inherited code summarised in §4.5. References marked UNVERIFIED in `LITERATURE.md` are used only for context, never as the sole support of an acceptance threshold.

## Contents

1. Vision and positioning
2. Scope
3. Requirements
4. Architecture
5. Module specifications (5.1 io · 5.2 events · 5.3 artifact · 5.4 prep · 5.5 shapes and models · 5.6 existence · 5.7 trials · 5.8 hierarchical · 5.9 spatial · 5.10 tf · 5.11 stats · 5.12 sim · 5.13 reports, plots, CLI, online mode)
6. Presets
7. Scientific validation
8. Public benchmark
9. Engineering
10. Ecosystem and adoption
11. Citation and publication strategy
12. Build order
13. Valorisation
14. Traceability matrix
15. Risks and mitigations
16. Open questions for the author

## 1. Vision and positioning

### 1.1 The problem in one paragraph

An evoked response is what a brain (or a nerve) produces after a stimulus: a sound, an image, a magnetic pulse, an electrical pulse delivered through an implanted contact. It is measured everywhere, from scalp ERP research to epilepsy surgery, deep-brain-stimulation programming, spinal-cord stimulation and animal physiology, and it is still quantified with rules that differ from lab to lab. In the cortico-cortical evoked potential (CCEP) literature alone, half of 187 studies do not state how a response was detected and two thirds of the rest use a fixed amplitude threshold [LIT-ccep-11]; the thresholds range from 2.6 to 6 baseline standard deviations [LIT-ccep-13], [LIT-ccep-3], [LIT-ccep-5] and the windows start anywhere between 5 and 16 ms [LIT-ccep-22], [LIT-ccep-10]; two trained human raters agree on N1 presence with Cohen's κ ≈ 0.60 [LIT-ccep-8]. For scalp ERPs, the complete measurement canon (fractional-area latency, jackknife scoring, standardised measurement error) exists essentially only in MATLAB [LIT-erpmeas-45], [LIT-erptools-1], and the most-used Python library offers an extremum finder whose improvement request has been open since 2021 [LIT-needs-7] and an ERP measurement tool requested since 2020 [LIT-needs-2]. Single-trial estimation, artifact handling for electrical stimulation, response-existence testing and uncertainty on every number are absent from the Python ecosystem [LIT-needs-29], [LIT-needs-39], [LIT-needs-43].

The fork from which this project starts reproduces the problem: on the example CCEP run of the public dataset ds004080 (subject `ccepAgeUMCU01`, stimulation pair PT01–PT02), it finds the N1 on 2 of 46 channels, because three hard-coded rules delete it (an edge rule, a width rule and an overlap rule) and because its acceptance threshold passes 93 % of pure-noise windows, which forces aggressive pruning [CODE §0], [CODE §4]. §5.5 replaces that logic; REQ-113 makes the fix measurable.

### 1.2 Vision

**evoparam turns any evoked response, in any species and modality, into a short list of named components with latency, amplitude, width, shape, existence probability and uncertainty, from one call, with a methods paragraph that the user pastes into the paper.**

Five commitments follow from it and shape the whole specification:
1. **One model for every modality.** The same sum-of-components-plus-background model (§5.5) parameterises an ERP P3, a CCEP N1, a TEP N100, an ERNA oscillation and a spinal ECAP; presets (§6) change the numbers, not the code.
2. **Existence before description.** No component is reported without a calibrated test of whether a response exists at all (§5.6), and every published threshold rule can be reproduced exactly for comparison.
3. **Every number has an uncertainty and a provenance.** Intervals at four cost levels (§5.5.5), status codes instead of silent NaN (§4.6), a provenance record that regenerates the result (§9.12).
4. **Trials, channels and subjects are levels of one model**, not separate scripts (§5.7, §5.8).
5. **Light by default, deep on request.** `numpy` and `scipy` are the only required dependencies; Bayesian, MNE, NWB and time-frequency features are extras (§9.2).

### 1.3 The sentence users will cite

> Evoked responses were parameterized with evoparam (version X.Y.Z; Bellec et al., *methods paper*; software DOI 10.5281/zenodo.NNNNNNN) using the `ccep` preset: responses were tested against the pre-stimulus baseline by sign-flip permutation (1 000 permutations, FDR-corrected across channels), and each responsive channel's average was described as a sum of skewed-Gaussian components on an exponential background, the number of components being chosen by BIC on whitened residuals; intervals are 95 % Jacobian-based intervals corrected for residual autocorrelation.

This sentence is generated, with the user's actual settings, by `methods_text()` (§5.13, REQ-340) and printed at the end of every `evoparam.fit` call. The mechanism and the reasons it drives citations are in §11.1.

### 1.4 Positioning against existing tools

| tool | language, licence | what it does well | what it does not do (and evoparam does) | evidence |
|---|---|---|---|---|
| ERPLAB | MATLAB, GPL-3.0 | complete classical ERP measurement canon, SME, wide user base (~300 citations/year) | no model of components, no single-trial parameters, no uncertainty beyond SME, no intracranial/stimulation features, MATLAB licence needed | [LIT-erptools-1], [LIT-erptools-2], [LIT-erptools-9] |
| MNE-Python | Python, BSD-3 | data structures, I/O, preprocessing, cluster statistics; grand-average ERP measures since 1.13 | `get_peak` is an extremum finder; no component model, no single-trial estimation, stimulation artifacts only by window interpolation | [LIT-erptools-17], [LIT-erptools-20], [LIT-needs-4], [LIT-needs-7] |
| FieldTrip, EEGLAB, Brainstorm | MATLAB, GPL/BSD | general pipelines, statistics | ERP parameter extraction left to the user (forum questions) | [LIT-erptools-25], [LIT-erptools-13], [LIT-erptools-33] |
| ER-detect | Python, GPL-3.0 | reference N1 detection for BIDS CCEP data | detection only (N1, ECoG, ~0.2 Hz validated); no shapes, no uncertainty, no N2/late responses | [LIT-ccep-8], [LIT-ccep-28], [LIT-ccep-30] |
| Canonical response parameterisation / basis profile curves | MATLAB/Python scripts, GPL | data-driven response duration and shared shapes | one shape per site (rank one), no named components, scripts rather than a maintained package | [LIT-ccep-15], [LIT-ccep-16], [LIT-ccep-35] |
| TESA, TEPpy | MATLAB / Python, GPL | TMS-EEG cleaning and peak picking | fixed peak lists, no component model, no uncertainty | [LIT-domains-4], [LIT-fitting-11] |
| RIDE, eeg-ride | MATLAB / Python port | latency-variable component separation | no named parametric components, no existence test; port minimal (4 stars) | [LIT-erpmeas-11], [LIT-erpmeas-15] |
| Unfold, mTRF | MATLAB/Julia, Python | overlap deconvolution, regression ERPs | fixed response shape, no trial-level latency | [LIT-erpmeas-31], [LIT-erpmeas-35] |
| ERPparam (the fork origin) | Python, Apache-2.0 | first attempt at ERP parameterisation by Gaussian peaks | not released (no PyPI, no docs site, 4 stars), N1 deleted by heuristics, no existence test, no uncertainty, no stimulation features; its own tracker asks for classical ERP metrics | [LIT-fooof-39], [LIT-fooof-59], [LIT-fooof-60], [LIT-needs-28], [CODE §0] |
| FOOOF / specparam | Python, Apache-2.0 | parameterisation of power spectra; 2 533 citations, about 600 per year (OpenAlex, 2026-10-06) | spectra only — not a competitor in the time domain, but the citation model to match (named method, one call, methods template) | [LIT-fooof-3], [LIT-citation-1] |

**Differentiation in one line per competitor family.** Against MATLAB toolboxes: the same measures in Python plus models and uncertainty. Against MNE: evoparam consumes and returns MNE objects (§10) and adds what MNE's tracker has asked for since 2017–2021. Against detection-only CCEP tools: detection with calibrated error rates *and* parameterisation, across ECoG and sEEG, with published rules reproduced for comparison. Against the fork origin: a different fitting engine, an existence test, stimulation features, trials and hierarchy, and a release.

### 1.5 Success criteria for the product

Measured as in §11.7: adoption (downloads, dependants), citation curve (target at roughly one third of the spectral ancestor's per-year curve over the same years after release, §11.7), benchmark submissions by other groups (§8), and the share of the CCEP literature that states its detection method with a named, reproducible rule.
## 2. Scope

This specification describes one complete product. There are no "version 1 / version 2" tiers: everything in §2.1 is in scope and is built in the order of §12; everything in §2.3 is out of scope, each with the reason. Nothing is left as unspecified future work.

### 2.1 In scope

**Signals and recording types.**
| family | recordings | typical sampling | presets |
|---|---|---|---|
| scalp evoked potentials | EEG, MEG (sensor level), ERP paradigms, SEP, BCI P300 | 250 Hz – 5 kHz | `erp`, `sep`, `p300_bci` |
| electrical-stimulation evoked responses, intracranial | SPES/CCEP on ECoG and sEEG, delayed responses, spectral responses | 512 Hz – 10 kHz | `ccep`, `ccep_seeg` |
| non-invasive stimulation evoked responses | TMS-EEG (TEP) | 1 – 20 kHz | `tep` |
| neuromodulation evoked responses | DBS ERNA, spinal-cord-stimulation ECAP, cochlear-implant ECAP | 2 – 30 kHz | `erna`, `ecap`, `ecap_ci` |
| animal evoked field potentials | rodent and non-human primate LFP/ECoG, laminar probes, electrical, sensory or optogenetic stimuli | 1 – 30 kHz (LFP band) | `rodent_ep`, `primate_ep` |

**Species.** Human, mouse, rat, macaque; any other species through `species=` metadata and custom presets (no species-specific code exists).

**Functions** (one module each, §5): reading (BIDS, MNE, EDF, BrainVision, NWB, Neo, arrays; remote partial reads), stimulation-event detection from the signal, artifact removal (eleven methods with a per-trial quality score), light preparation (baseline, references including adjusted CAR, rejection flags), component model fitting with model selection and four uncertainty levels, classical ERP measures, existence testing with a registry of published rules, trial-by-trial parameters with covariates (train order, dose, pre-stimulus state), hierarchical trial→channel→subject models (empirical Bayes by default, full Bayesian as an extra), shared shapes across channels and sites with maps and a volume-conduction screen, time-frequency parameterisation with data-driven zones and a compatibility layer for published zones, statistics (bootstrap, jackknife, permutation contrasts, reliability, trial-count planning, mixed-model export), simulators with ground truth for every modality, reports (methods text, JSON, BIDS derivatives, HTML/PDF), plots, a command-line interface, an online mode for streamed epochs, and a public benchmark.

**Users.** Researchers (scalp ERP, intracranial, TMS, animal), clinical research teams (epilepsy surgery planning, DBS/SCS research), methods developers (benchmark), and downstream software (the paid modules of §13 and third-party packages through the public API).

### 2.2 Interfaces in scope

Python ≥ 3.10 API (§5), command line (§5.13, §10.5), file formats (§4.3.5, §9.12, §10), MNE objects in and out (§10), scikit-learn-compatible estimator conventions (§10), streaming input from Lab Streaming Layer through `pylsl` (§5.13, §10.6).

### 2.3 Explicit exclusions, with reasons

| excluded | reason | where the need is met instead |
|---|---|---|
| Graphical user interface | A GUI is a different product with different users and release cadence; it is the main paid module of the open-core model (§13.1) and would tie the free engine to a GUI toolkit. | a paid graphical module (§13.1, §13.6); `report.html` and `plots` for viewing results. |
| Closed-loop control of a stimulator | Acting on a patient or animal makes the software a medical device or a safety-critical controller (§13.3); the free package exposes the numbers only. | `evoparam.stream` publishes per-epoch parameters; control loops are built by the user or the paid real-time module. |
| Clinical decision outputs (e.g. "seizure onset zone here", "nerve at risk") | Clinical validity is not established by this specification's validation (§7.5) and such outputs fall under medical-device rules (§13.3). | Research reporting only; clinical studies can use the outputs as research variables. |
| Source localisation, forward models, connectivity graph metrics | Mature dedicated tools exist (MNE, Brainstorm, network packages); duplicating them adds no differentiation. | `connectivity_matrix` export (§5.9) and MNE objects out (§10). |
| Spike sorting, single-unit analysis, calcium imaging | Different signals and models; the package parameterises continuous field potentials. | — |
| General-purpose EEG/MEG preprocessing (ICA, filtering pipelines, bad-channel detection) | MNE and its ecosystem do this; evoparam does only the preparation that changes evoked parameters (baseline, reference, trial flags, stimulation artifacts). | MNE, autoreject, mne-bids-pipeline upstream of evoparam (§10). |
| Spectral parameterisation of resting-state power spectra | It is the subject of other packages and is not an evoked-response problem; keeping the vocabulary and code separate is a design rule (§4.1). | The time-frequency module parameterises *evoked* time-frequency responses only. |
| Classification/decoding pipelines (BCI accuracy, deep learning decoders) | Benchmarked by dedicated frameworks; evoparam provides the single-trial parameters such pipelines can use as features. | `to_long` export (§5.11); scikit-learn compatible estimators (§10). |
| A full mixed-effects modelling engine | Mature in `statsmodels`, `lme4`, `brms`; reimplementing would be costly and less trusted. | `to_long`, `mixed_formula`, `fit_mixed` (statsmodels) in §5.11; the hierarchical module covers the evoked-specific model. |
| Graph-based or deep-learning response detectors as defaults | Not calibrated, hard to explain in a methods section, require training data that do not exist for most modalities. | The benchmark (§8) accepts them as submissions, so they can be compared on equal terms. |
| Code copied from GPL or unlicensed implementations (ER-detect, ieegprep, canonical response parameterisation scripts, TEPpy, MPARRM code) | Licence incompatibility with Apache-2.0 distribution and the open-core model (§13.4). | Clean-room re-implementation from the published papers; GPL tools compared through process-isolated adapters (§8.3). |
| Real-time operating-system guarantees (hard deadlines) | Python cannot guarantee them; the online mode targets soft real time (99th-percentile latency budget, §10.6 REQ-324). | Paid real-time module (§13.1). |

### 2.4 Assumptions

- Data arrive epoched or with event times (or the events can be detected from the stimulation artifact, §5.2).
- Times are in seconds relative to stimulus onset; amplitudes carry units (default µV).
- The user is responsible for ethics approval and data licences; the dataset registry (§5.1) records each public dataset's licence and the benchmark respects it (§8.2).
## 3. Requirements

Every requirement has a stable identifier `REQ-nnn`, a testable acceptance criterion, and a source (literature key, code evidence, or a rule of this specification). Identifiers are grouped in blocks: 001–019 product-level rules; 020–039 input/output; 040–059 events; 060–079 artifact removal; 080–099 preparation; 100–129 shapes and fitting; 130–149 existence; 150–169 trials; 170–189 hierarchical; 190–209 spatial; 210–229 time-frequency; 230–249 statistics; 250–269 simulation; 270–289 presets; 290–309 engineering; 310–329 ecosystem; 330–349 reports, output schema, CLI and online mode; 350–369 benchmark; 370–389 validation; 390–399 publication, legal and governance. Gaps in the numbering are deliberate (room for insertions without renumbering). The full text of each requirement is in the section given in §3.2; §3.1 states the product-level rules in full; §14 traces every gap of GAPS.md to requirements, modules, tests and validation.

### 3.1 Product-level requirements (REQ-001…REQ-019)

- REQ-001 (one model): the same fitting engine and result types serve every preset of §6; a test fits one simulated waveform of each modality through `evoparam.fit` with only the preset name changed and checks that every result has the same type and columns. Source: §1.2.
- REQ-002 (one call): `evoparam.fit(data, preset=...)` on an `mne.Epochs`, an `mne.Evoked`, a `TrialSet`, a BIDS path read by `evoparam.read` or a NumPy array with `times` returns a `GroupFit` with existence results, components, uncertainties and a methods paragraph, with no other argument required. Source: citation lesson "one-object, one-call API with sensible defaults" [LIT-citation-32].
- REQ-003 (polarity-agnostic): no code path in fitting, existence or trial modules discards a candidate or component because of its sign unless the user sets `polarity`; a test on sign-flipped data returns mirrored results (latencies equal, amplitudes negated) to 1e-9. Source: §6 general rules, [LIT-ccep-8].
- REQ-004 (no deletion heuristics): components are accepted or rejected only by the model-selection criterion (§5.5.4); a grep-based test forbids threshold-based deletion of fitted components in `evoparam/models`. Source: [CODE §0], §5.5.
- REQ-005 (no silent failure): every public function either returns a result whose `status` explains a non-result or raises a documented exception; no NaN parameter appears with `status="ok"` (REQ-118); every warning is also logged. Source: [CODE §2.2], §4.6.
- REQ-006 (uncertainty everywhere): every reported latency, amplitude, width, onset, area and existence decision has an interval or a p-value in the same table row. Source: [LIT-fitting-1], [LIT-fitting-3], [LIT-needs-84].
- REQ-007 (provenance everywhere): every result can be regenerated from its saved provenance with `evoparam.reproduce` (REQ-333). Source: §9.12.
- REQ-008 (methods text): every result produces a methods paragraph naming the package version, preset, settings that differ from the preset, statistical choices and citations (REQ-340, REQ-390). Source: §11.1.
- REQ-009 (units and time): all times in seconds, all amplitudes with an explicit unit; mixing units raises `DataError`. Source: §4.1.
- REQ-010 (light core): `import evoparam` and the core functions of REQ-291 work with NumPy and SciPy only. Source: §9.2.
- REQ-011 (licence hygiene): no file of the package is copied or derived from GPL or unlicensed code; the licence scan and the import deny list run in CI (REQ-292, REQ-309, REQ-354). Source: §13.4.
- REQ-012 (no inherited spectral vocabulary): public names, docstrings and documentation contain none of the prohibited words of the internal style guide; a CI test greps the package and documentation. Source: §4.1.
- REQ-013 (determinism): identical inputs, settings and `random_state` give identical outputs (REQ-119, REQ-250, §9.4).
- REQ-014 (presets cite their sources): REQ-271.
- REQ-015 (research use statement): the package documentation, CLI `--version` output, HTML/PDF reports and stream outputs carry the statement "research use only; not a medical device" until a regulated product exists (§13.3).
- REQ-016 (no network by default): no function accesses the network unless it is a documented fetch/remote-read function called with remote options; no telemetry (§9.10).
- REQ-017 (the N1 fix is permanent): REQ-113 and REQ-114 run in every CI build from step S7 of §12 onwards.
- REQ-018 (every gap traced): every gap of GAPS.md marked "addressed" maps to at least one REQ, one module, one test file and one validation item in §14; a script checks the mapping.
- REQ-019 (no public step before the institutional decision): no public repository, package upload, preprint, abstract or talk before the HIIE decision on record (REQ-398, §11.4).

**Test map for §3.1.** `tests/test_one_model.py` (REQ-001, REQ-002), `tests/test_polarity.py` (REQ-003), `tests/test_no_deletion.py` (REQ-004), `tests/test_status.py` (REQ-005), `tests/test_uncertainty_columns.py` (REQ-006), `tests/test_reproduce.py` (REQ-007), `tests/test_methods_text.py` (REQ-008), `tests/test_units.py` (REQ-009), `tests/test_core_only.py` (REQ-010), `tests/test_dependencies.py` (REQ-011), `tests/test_vocabulary.py` (REQ-012), `tests/test_determinism.py` (REQ-013), `tests/test_presets.py` (REQ-014), `tests/test_research_use.py` (REQ-015), `tests/test_no_network.py` (REQ-016), `tests/test_engine_ccep.py` (REQ-017), `tools/check_traceability.py` (REQ-018), `RELEASING.md` gate checked by `tests/test_release_gate.py` (REQ-019).

### 3.2 Index of all requirements

The table below is generated from the section files; it lists each requirement's identifier, the section that holds its full text, and its first words.

| REQ | section | summary |
|---|---|---|
| REQ-001 | §3.1 | (one model): the same fitting engine and result types serve every preset of §6; a test … |
| REQ-002 | §3.1 | (one call): evoparam.fit(data, preset=...) on an mne.Epochs, an mne.Evoked, a TrialSet, a BIDS path read by … |
| REQ-003 | §3.1 | (polarity-agnostic): no code path in fitting, existence or trial modules discards a candidate or component because … |
| REQ-004 | §3.1 | (no deletion heuristics): components are accepted or rejected only by the model-selection criterion (§5.5.4); a grep-based … |
| REQ-005 | §3.1 | (no silent failure): every public function either returns a result whose status explains a non-result or … |
| REQ-006 | §3.1 | (uncertainty everywhere): every reported latency, amplitude, width, onset, area and existence decision has an interval or … |
| REQ-007 | §3.1 | (provenance everywhere): every result can be regenerated from its saved provenance with evoparam.reproduce (REQ-333). Source: §9.12. |
| REQ-008 | §3.1 | (methods text): every result produces a methods paragraph naming the package version, preset, settings that differ … |
| REQ-009 | §3.1 | (units and time): all times in seconds, all amplitudes with an explicit unit; mixing units raises … |
| REQ-010 | §3.1 | (light core): import evoparam and the core functions of REQ-291 work with NumPy and SciPy only. … |
| REQ-011 | §3.1 | (licence hygiene): no file of the package is copied or derived from GPL or unlicensed code; … |
| REQ-012 | §3.1 | (no inherited spectral vocabulary): public names, docstrings and documentation contain none of the prohibited words of … |
| REQ-013 | §3.1 | (determinism): identical inputs, settings and random_state give identical outputs (REQ-119, REQ-250, §9.4). |
| REQ-014 | §3.1 | (presets cite their sources): REQ-271. |
| REQ-015 | §3.1 | (research use statement): the package documentation, CLI --version output, HTML/PDF reports and stream outputs carry the … |
| REQ-016 | §3.1 | (no network by default): no function accesses the network unless it is a documented fetch/remote-read function … |
| REQ-017 | §3.1 | (the N1 fix is permanent): REQ-113 and REQ-114 run in every CI build from step S7 … |
| REQ-018 | §3.1 | (every gap traced): every gap of GAPS.md marked "addressed" maps to at least one REQ, one … |
| REQ-019 | §3.1 | (no public step before the institutional decision): no public repository, package upload, preprint, abstract or talk … |
| REQ-020 | §5.1 | from_array, read_mne, read_bids (local and remote), read_edf, read_nwb, read_neo each produce a TrialSet passing TrialSet.validate(); round … |
| REQ-021 | §5.1 | remote read_bids(..., remote=True) of ten trials (−0.5 s, 1.0 s) from ds004080 sub-ccepAgeUMCU01 run-021448 transfers less … |
| REQ-022 | §5.1 | harmonize_stim_events on the events.tsv of ds004080 and of the four Mayo datasets fills stim_site, stim_current_mA, stim_pulse_width_us … |
| REQ-023 | §5.1 | the dataset registry lists at least the 14 datasets above with license, size_gb, sfreq, events_status ∈ … |
| REQ-024 | §5.1 | write_bids_derivatives output passes bids-validator (derivatives mode) when the validator is installed in CI; write_nwb output re-reads … |
| REQ-025 | §5.1 | the package imports and from_array works with only numpy and scipy installed (readers for mne/nwb/neo/edf raise … |
| REQ-040 | §5.2 | on ds004080, across at least 5 subjects and 20 runs with events.tsv as reference, detect_pulses reaches … |
| REQ-041 | §5.2 | on synthetic recordings (100 channels, 2048 Hz, 200 pulses with artifact amplitude ≥ 10× baseline SD, … |
| REQ-042 | §5.2 | identify_stim_site agrees with events.tsv on ≥ 95 % of pulses on ds004080 and on the Mayo … |
| REQ-043 | §5.2 | detect_saturation finds 100 % of synthetic clipping runs ≥ 3 samples and ≥ 95 % of … |
| REQ-044 | §5.2 | group_trains reproduces the train structure of the Mayo protocol (≈ 0.2 Hz, 10–12 pulses per site) … |
| REQ-045 | §5.2 | runtime ≤ 1 s per 10 minutes of 100-channel data at 2048 Hz on one core; … |
| REQ-060 | §5.3 | for every method, replaced samples are exactly the masked samples and no sample outside the declared … |
| REQ-061 | §5.3 | template and learned reduce the artifact-window residual by ≥ 20 dB relative to interpolate leaving the … |
| REQ-062 | §5.3 | period_template on a synthetic 130 Hz train at 1 kHz sampling with a 10 Hz and … |
| REQ-063 | §5.3 | lowrank on a synthetic 24-channel, 20-pulse, 10-trial array recording recovers evoked spikes with waveform correlation ≥ … |
| REQ-064 | §5.3 | subspace reaches ≥ 30 dB signal-to-interference improvement on synthetic EEG with real artifact templates, matching the … |
| REQ-065 | §5.3 | polynomial recovers the signal within 2 ms after a 600 mV pulse at 25 kHz on … |
| REQ-066 | §5.3 | the quality score separates clean from corrupted synthetic trials (10 % of trials with residual artifact, … |
| REQ-067 | §5.3 | paradigm with alternating polarity cancels an artifact that exactly inverts to ≤ 1 % residual and … |
| REQ-068 | §5.3 | every method runs in ≤ 2 s for 50 trials × 128 channels × 3 s … |
| REQ-069 | §5.3 | results carry method, all parameters, window, the template or basis used, and the masks, so that … |
| REQ-080 | §5.4 | baseline check detects a baseline overlapping the previous pulse's delayed window in a synthetic 1 Hz … |
| REQ-081 | §5.4 | car_adaptive reproduces the published simulation outcome on a synthetic 50-channel, 12-trial dataset with 0–90 % responsive … |
| REQ-082 | §5.4 | bipolar naming and ordering match mne.set_bipolar_reference on the same montage; the reference is recorded in ch_info.reference … |
| REQ-083 | §5.4 | reject_trials flags ≥ 95 % of synthetic corrupted trials (amplitude 10× baseline, flat, or 30 % … |
| REQ-084 | §5.4 | reference_sensitivity produces the κ matrix and latency differences for three schemes on ds004080 sub-01 in < … |
| REQ-085 | §5.4 | resample preserves component latencies within 1/(2·sfreq_new) on simulated data and keeps event_times exact. |
| REQ-100 | §5.5.10 | the eight shapes of §5.5.1 and the seven backgrounds of §5.5.2 are implemented with the stated … |
| REQ-101 | §5.5.10 | every analytic Jacobian matches central finite differences to 1e-6 relative on 100 random parameter sets inside … |
| REQ-102 | §5.5.10 | register_shape accepts a user-defined shape implementing the protocol and rejects (with a SettingsError naming the failing … |
| REQ-103 | §5.5.10 | variable projection — for fixed Θ, the amplitudes and background coefficients returned equal the weighted least-squares … |
| REQ-104 | §5.5.10 | detection_threshold never changes which components are accepted: on the simulation suite of §5.12, acceptance decisions for … |
| REQ-105 | §5.5.10 | (simulation recall/precision): on simulate_erp waveforms with K ∈ {1,2,3} components, SNR ≥ 3 (peak amplitude over … |
| REQ-106 | §5.5.10 | (accuracy): at SNR ≥ 5, latency MAE ≤ 0.5 sample and amplitude relative bias ≤ 5 … |
| REQ-107 | §5.5.10 | (coloured noise): with AR(1) noise ρ = 0.9, the false-component rate on pure-noise windows is ≤ … |
| REQ-108 | §5.5.10 | (masks): a component whose true latency falls inside a masked 2 ms gap is still recovered … |
| REQ-109 | §5.5.10 | (DC invariance): adding any constant c to y changes no component feature by more than 1e-6 … |
| REQ-110 | §5.5.10 | (edge components): for true components with latency in [t_0, t_0 + FWHM] (partly cut by the … |
| REQ-111 | §5.5.10 | (resolution): two equal-amplitude Gaussians separated by 1.0·FWHM are recovered as two components in ≥ 90 % … |
| REQ-112 | §5.5.10 | (type-I of selection): pure-noise windows (white, AR(1) ρ ∈ {0.5, 0.9}, and real baseline segments of … |
| REQ-113 | §5.5.10 | (the N1 case): on ds004080 sub-ccepAgeUMCU01 run-021448, stimulation pair PT01-PT02, the default ccep preset (§6) detects … |
| REQ-114 | §5.5.10 | (N1 latency agreement): on the channels where both agree, the model latency and the published rule's … |
| REQ-115 | §5.5.10 | (speed): one waveform of 2 048 samples with max_components = 6, default shapes and uncertainty="jacobian" fits … |
| REQ-116 | §5.5.10 | (coverage): nominal 95 % intervals cover the true latency and amplitude in 90–97 % of simulations … |
| REQ-117 | §5.5.10 | (classical measures): classical_measures reproduces, to 1e-9, reference values computed by an independent implementation of the ERPLAB … |
| REQ-118 | §5.5.10 | (status semantics): no code path returns NaN parameters with status="ok"; every "failed" fit carries a non-empty … |
| REQ-119 | §5.5.10 | (reproducibility): two runs with the same random_state give byte-identical to_dataframe() output (also with n_jobs 1). |
| REQ-120 | §5.5.10 | (tidy output): GroupFit.to_dataframe() has one row per component, no nested objects, and GroupFit.save/load round-trips exactly. |
| REQ-121 | §5.5.10 | (conditioning report): ResponseFit.diagnostics contains cond, eigvals, corr and the list of flagged pairs; a two-component fit … |
| REQ-122 | §5.5.10 | (selection alternatives): selection="cv" on a TrialSet with 40 trials chooses the true K in ≥ 85 … |
| REQ-123 | §5.5.10 | (methods text): ResponseFit.methods_text() names the shapes, background, criterion, delta, n_eff correction, uncertainty method and the package … |
| REQ-130 | §5.6 | (type I): on pure-noise TrialSets (white, AR(1) ρ = 0.9, and ds004080 baseline segments re-epoched at … |
| REQ-131 | §5.6 | (power): with 10 trials and a Gaussian response of FWHM 10 ms at SNR = 2 … |
| REQ-132 | §5.6 | (duration): the estimated duration is within ±10 % of the simulated one (median) for SNR ≥ … |
| REQ-133 | §5.6 | (registry exactness): every registry rule, applied to a synthetic waveform built to lie exactly at its … |
| REQ-134 | §5.6 | (projection validation): method="projection" reproduces the published duration and significance for the public example dataset of the … |
| REQ-135 | §5.6 | (agreement with the field): on ds004080 (≥ 5 subjects), the default method agrees with vanblooijs2023 with … |
| REQ-136 | §5.6 | (corrections): with 1 000 null channels, fdr_bh at α = 0.05 yields ≤ 5 % false … |
| REQ-137 | §5.6 | (no silent defaults): running a registry rule with an unspecified field without supplying it raises; the … |
| REQ-138 | §5.6 | (speed): test_response with 1 000 permutations on 100 channels × 50 trials × 1 024 samples … |
| REQ-139 | §5.6 | (discordance): train_discordance returns 0 for simulated identical trains and ≈ 1 for trains alternating response/no response. |
| REQ-140 | §5.6 | (methods text): ResponseTest.methods_text() states the method, statistic, window, baseline, α, correction, permutations and seed. |
| REQ-150 | §5.7 | on simulated trials with known per-trial shifts (jitter_sd ∈ {1, 2, 5} ms) and amplitudes (CV … |
| REQ-151 | §5.7 | jitter_sd_null on jitter-free simulated trials is ≥ 0.8·jitter_sd (the method does not invent jitter), and realign … |
| REQ-152 | §5.7 | amplitude_jitter_corrected is within ±10 % of the simulated jitter-free amplitude for Gaussian components at jitter_sd ≤ … |
| REQ-153 | §5.7 | "regularized" has lower trial-level latency MAE than "free" at per-trial SNR ≤ 1 on ≥ 90 … |
| REQ-154 | §5.7 | dose_response recovers I_50 within ±15 % and h within ±30 % on simulated Hill curves with … |
| REQ-155 | §5.7 | prestimulus_dependence has type-I error ≤ 0.06 at α = 0.05 on trials with no phase dependence … |
| REQ-156 | §5.7 | on ds004080 (10 trials per site), fit_trials on all N1 channels of ≥ 5 subjects completes … |
| REQ-157 | §5.7 | the trial table has exactly n_trials × n_components × n_channels_ok rows, no NaN in converged/trusted, and … |
| REQ-158 | §5.7 | the index_in_train effect (amplitude slope) reproduces the direction reported for repeated single pulses at 1 Hz … |
| REQ-170 | §5.8 | on simulated three-level data (20 subjects × 50 channels × 20 trials, 30 % responding channels), … |
| REQ-171 | §5.8 | EB shrunken channel latencies have lower RMSE than the per-channel §5.5 estimates on ≥ 90 % … |
| REQ-172 | §5.8 | variance components (σ_α between channels, σ_δ jitter) are recovered within ±20 % by EB and within … |
| REQ-173 | §5.8 | NUTS runs show r_hat ≤ 1.01, ESS ≥ 400 per parameter and < 1 % divergences … |
| REQ-174 | §5.8 | the two Bayesian engines agree with each other on every posterior mean within one posterior SD … |
| REQ-175 | §5.8 | a formula term with a simulated true effect (latency ~ age, slope 0.2 ms/year) is recovered … |
| REQ-176 | §5.8 | time budgets: EB ≤ 10 s per subject; NUTS ≤ 5 min per subject (100 channels … |
| REQ-177 | §5.8 | with engine="eb", the package has no new dependency; importing evoparam.hierarchical without numpyro/pymc works and raises an … |
| REQ-178 | §5.8 | on ds004080 (≥ 5 subjects), the developmental effect of age on N1 latency reported by the … |
| REQ-190 | §5.9 | on simulated multi-site data with 3 true shapes and 20 sites, shared_shapes recovers n_shapes = 3 … |
| REQ-191 | §5.9 | fit_shared reduces the latency RMSE across 30 simulated channels sharing one shape by ≥ 30 % … |
| REQ-192 | §5.9 | the graph-Laplacian penalty with λ chosen by leave-one-channel-out never increases the held-out prediction error by more … |
| REQ-193 | §5.9 | screen_volume_conduction flags high on ≥ 90 % of simulated channels whose signal is a scaled copy … |
| REQ-194 | §5.9 | shared_shapes(by="stim_site") on the public example dataset of the basis-profile-curve paper (ds003708 [LIT-ccep-36]) yields the same number … |
| REQ-195 | §5.9 | connectivity_matrix round-trips to and from long form and matches GroupFit.to_dataframe() cell by cell. |
| REQ-196 | §5.9 | parameter_map on ds004080 (one subject) produces a table with one row per (stim_site, channel) pair, coordinates … |
| REQ-210 | §5.10 | the Morse transform with γ = 3, β = 10 matches an independent reference implementation (ssqueezepy … |
| REQ-211 | §5.10 | evoked_footprint + residual + cross == total to 1e-9 for every pixel (exactness of the decomposition). |
| REQ-212 | §5.10 | on simulated trials with a sharp N1 (FWHM 8 ms) and no induced activity, the residual … |
| REQ-213 | §5.10 | on simulated induced gamma bursts (80 Hz, 100 ms, 20 % power increase, 30 trials), zone … |
| REQ-214 | §5.10 | pixel-wise type-I error with stats="fdr" ≤ 0.06 on null simulations with AR(1) trials; "cluster" and "tfce" … |
| REQ-215 | §5.10 | baseline_level="single_trial" reduces the influence of one injected high-power trial on the average map by ≥ 50 … |
| REQ-216 | §5.10 | on ds004080 (≥ 5 subjects), the published zone delayed 50–500 ms × 50–250 Hz computed by … |
| REQ-217 | §5.10 | running an unverified published zone without allow_unverified=True raises SettingsError naming the zone and its source. |
| REQ-218 | §5.10 | speed: 100 channels × 50 trials × 2 048 samples × 40 frequencies with stats="fdr" in … |
| REQ-219 | §5.10 | TFFit.methods_text() names the transform and its parameters, the baseline window and mode, the statistic, α and … |
| REQ-230 | §5.11 | BCa and percentile intervals reach 93–97 % coverage on simulated normal and log-normal statistics with 30 … |
| REQ-231 | §5.11 | jackknife_latency reproduces the corrected statistics of the reference examples in [LIT-erpmeas-2] and [LIT-erpmeas-6] (worked numerical examples) … |
| REQ-232 | §5.11 | compare_conditions keeps the type-I error within [0.04, 0.06] at α = 0.05 on null simulations at … |
| REQ-233 | §5.11 | reliability reproduces split-half and dependability values computed by an independent implementation of the published formulas on … |
| REQ-234 | §5.11 | trials_needed predicts, from the first 20 trials of a simulated channel, the parameter SD at 80 … |
| REQ-235 | §5.11 | to_long output has no nested objects, one value per cell, and a column dictionary (JSON sidecar) … |
| REQ-236 | §5.11 | mixed_formula strings parse in statsmodels (tested) and match frozen templates for lme4 and brms. |
| REQ-237 | §5.11 | group_summary reports, for every group, the count of each status value, so that n_ok + n_no_response … |
| REQ-250 | §5.12 | every generator is deterministic for a given seed (byte-identical output) across platforms in CI. |
| REQ-251 | §5.12 | make_noise("pink", χ) produces a spectral exponent within ±0.1 of χ (log–log slope fitted over 2–100 Hz … |
| REQ-252 | §5.12 | the measured SNR of every generated component equals the requested SNR within ±5 %. |
| REQ-253 | §5.12 | simulate_ccep artifacts are removed by remove_artifact(method="template") to a residual RMS within the artifact window ≤ 10 … |
| REQ-254 | §5.12 | sim does not import evoparam.models or anything that imports it (import-graph test, §4.2). |
| REQ-255 | §5.12 | GroundTruth.to_dataframe() and GroupFit.to_dataframe() share column names and units for every common parameter. |
| REQ-256 | §5.12 | the "real" noise option records dataset id, version, subject, run and sample offsets of every injected … |
| REQ-270 | §6.4 | every preset in §6.1 exists, loads from its JSON file into a frozen dataclass, and every … |
| REQ-271 | §6.4 | every preset field that is not marked (choice) has at least one citation key in Preset.citation … |
| REQ-272 | §6.4 | derive never mutates the parent; fit(..., preset=p, overrides) and fit(..., preset=derive(p, overrides)) give identical results. |
| REQ-273 | §6.4 | ccep on ds004080 meets REQ-113 and REQ-114 (the N1 criteria). |
| REQ-274 | §6.4 | erp on ERP CORE (all 40 participants, seven components) recovers the component named by the paradigm … |
| REQ-275 | §6.4 | tep on the public single-pulse TMS-EEG dataset of the registry [LIT-datasets-17] produces labelled N45 and N100 … |
| REQ-276 | §6.4 | erna on simulate_erna recovers frequency within ±5 Hz and first-peak latency within ±0.1 ms at SNR … |
| REQ-277 | §6.4 | sep on simulated median-nerve SEPs recovers N20 latency within ±0.5 ms at SNR ≥ 5. |
| REQ-278 | §6.4 | rodent_ep on a public Neuropixels LFP session (§7) recovers the visual evoked complex on ≥ 90 … |
| REQ-279 | §6.4 | presets validated only on simulations (ecap, ecap_ci, erna) carry validation="simulation_only" in their metadata, which is printed … |
| REQ-280 | §6.4 | presets.list() output is reproduced in the documentation by a build step (no hand-maintained copy). |
| REQ-290 | §9.1 | Performance targets P1–P6. The package meets the six targets of the table above. Acceptance: benchmarks/run_benchmarks.py on … |
| REQ-291 | §9.2 | Light core. import evoparam and every function of the core chain (read of the npz format, … |
| REQ-292 | §9.2 | Optional extras and lazy imports. Every dependency other than numpy and scipy is declared in exactly … |
| REQ-293 | §9.3 | Deprecation shims. A renamed or removed public name keeps a shim for two minor versions that … |
| REQ-294 | §9.4 | Provenance on every result. Every public function that creates or transforms a container (TrialSet, Evoked, ArtifactResult, … |
| REQ-295 | §9.4 | random_state and cross-platform determinism. Every stochastic function (restarts of the optimiser, bootstrap, permutation tests, simulators, trial … |
| REQ-296 | §9.5 | Import-graph test. tests/test_import_graph.py parses every module of the package with ast, builds the directed graph of … |
| REQ-297 | §9.5 | CI matrix and jobs. GitHub Actions workflows run on every pull request and on main: lint … |
| REQ-298 | §9.5 | Coverage thresholds. Line coverage of the core packages (data, io local readers, events, artifact, prep, shapes, … |
| REQ-299 | §9.5 | Performance job. A workflow perf runs benchmarks/run_benchmarks.py --suite ci on one fixed runner image (ubuntu-22.04, pinned … |
| REQ-300 | §9.5 | Nightly, packaging and smoke test. (a) A scheduled workflow nightly runs the test suite against pre-release … |
| REQ-301 | §9.6 | Static typing of the public API. The distribution ships evoparam/py.typed (PEP 561). Every public function, method … |
| REQ-302 | §9.7 | Documentation build. Sphinx with pydata-sphinx-theme, numpydoc docstrings (every public object; sections Parameters / Returns / Raises … |
| REQ-303 | §9.7 | Documentation content. The site contains: (a) Getting started (install, 10-line example, what the numbers mean); (b) … |
| REQ-304 | §9.8 | Logging through the standard library. All diagnostics go through logging.getLogger("evoparam") and child loggers named after the … |
| REQ-305 | §9.9 | Static quality rules. ruff with rule sets E, W, F, I, N, UP, B, A, C4, … |
| REQ-306 | §9.10 | No network, no telemetry. (a) The only functions that open network connections are the explicit evoparam.io.fetch_ … |
| REQ-307 | §9.11 | No silent failure. A function never returns a result while hiding an exception. Rules: (i) programming … |
| REQ-308 | §9.3 | Release process and citation metadata. Each release is produced by the checklist RELEASING.md and yields a … |
| REQ-309 | §9.2 | GPL isolation, lock file and version floors. (a) No module of evoparam imports, vendors or optionally … |
| REQ-310 | §10.1 | MNE round trip. TrialSet.from_mne, TrialSet.to_mne, Evoked.from_mne, Evoked.to_mne exist with the signatures above and the conversion rules … |
| REQ-311 | §10.1 | Fitting MNE objects directly. evoparam.fit accepts mne.Epochs, mne.Evoked and lists of mne.Evoked (one condition each, comment … |
| REQ-312 | §10.1 | Upstream policy. Helpers proposed to MNE or MNE-BIDS (candidates: reading BIDS stimulation columns into Annotations extras, … |
| REQ-313 | §10.2 | Stimulation-event harmonisation. read_bids calls harmonize_stim_events on every _events.tsv, producing Table 4.1 columns with the unit rules … |
| REQ-314 | §10.2 | Stimulation-metadata validator. evoparam.io.bids.validate_stim_metadata(bids_root, , subject=None, strict=False) - pd.DataFrame returns one row per run × field with … |
| REQ-315 | §10.2 | BIDS derivatives writer. write_bids_derivatives produces the tree and files above for GroupFit, ResponseTest, TrialFit and ArtifactResult … |
| REQ-316 | §10.2 | Prefixed stimulation columns and BEP contribution. Non-standard stimulation fields appear only with the evoparam_ prefix and … |
| REQ-317 | §10.3 | NWB reader. read_nwb implements the event-source order and harmonisation above and returns a TrialSet equal to … |
| REQ-318 | §10.3 | NWB writer and ndx-evoparam. write_nwb adds the tables above; files pass nwbinspector with no CRITICAL or … |
| REQ-319 | §10.3 | Neo reader. evoparam.io.read_neo(source: "neo.Block str", , segment: int None = None, signal: int str = 0, … |
| REQ-320 | §10.4 | Estimator contract. ResponseModel, TrialFitter and ArtifactRemover follow the scikit-learn rules above: no logic in __init__, fit … |
| REQ-321 | §10.4 | check_estimator subset. With scikit-learn installed, parametrize_with_checks([ResponseModel(sfreq=1000., tmin=0.), TrialFitter(sfreq=1000., tmin=0.), ArtifactRemover(method="interpolate", sfreq=1000., tmin=0., window=(0., 0.002))], expected_failed_checks=EXPECTED) passes, … |
| REQ-322 | §10.5 | CLI commands. The thirteen sub-commands above exist with the stated behaviour, and evoparam fit on a … |
| REQ-323 | §10.5 | Exit codes and JSON summary. Exit codes follow the list above; with --json, stdout holds exactly … |
| REQ-324 | §10.6 | StreamFitter latency budget and agreement. At 2048 Hz, 64 channels, a 0.5 s response window and … |
| REQ-325 | §10.6 | LSL adapter and intended use. LSLInlet/LSLOutlet work with pylsl ≥ 1.16 (extra stream) and are tested … |
| REQ-326 | §10.7 | Tutorials and gallery datasets. The 17 tutorials of the table exist as sphinx-gallery scripts, execute in … |
| REQ-327 | §10.8 | Community files. The files, templates and policies above exist and are linked from the README and … |
| REQ-328 | §10.8 | Plugin entry points. The groups evoparam.shapes, evoparam.presets and evoparam.readers are loaded with the rules above; adding … |
| REQ-329 | §10.8 | Integration roadmap. Each target of the table has a tracking issue on the roadmap board with … |
| REQ-330 | §9.12 | The .evoparam container. save() of every result type writes a directory or zip <name.evoparam with manifest.json … |
| REQ-331 | §9.12 | Settings and preset recorded in every result. Every result carries settings (the full resolved dictionary: preset … |
| REQ-332 | §9.12 | Valid JSON and tidy tables. All JSON written by the package is RFC 8259 valid: NaN/±Inf … |
| REQ-333 | §9.12 | evoparam.reproduce. evoparam.reproduce(result_path, , data=None, rtol=1e-9, atol=0.0, check_environment=True) - ReproduceReport reloads a .evoparam container, re-runs the recorded … |
| REQ-334 | §9.12 | Schema versioning and migration. The schema string evoparam-result/<major.<minor follows: minor increments add optional fields and are … |
| REQ-335 | §9.12 | Environment capture. Provenance.environment records python, platform (platform.platform()), cpu (platform.processor() or /proc/cpuinfo model name), numpy, scipy, the … |
| REQ-336 | §9.12 | BIDS derivatives and NWB exports carry the same provenance. write_bids_derivatives (§5.13, §10.2) writes provenance.json content into … |
| REQ-337 | §9.12 | Data hash and source record. Provenance.data_hash is the sha256 of data.tobytes() concatenated with times.tobytes() and ch_names … |
| REQ-338 | §9.12 | Methods text is derived from provenance only. ResponseFit.methods_text() and GroupFit.methods_text() (specified in §5, evoparam.reports) read exclusively … |
| REQ-340 | §5.13 | methods_text output for the three reference pipelines (ERP CORE P3, ds004080 CCEP, simulated TEP) matches frozen … |
| REQ-341 | §5.13 | to_json output validates against the shipped JSON Schema (validation uses jsonschema from the validate extra and … |
| REQ-342 | §5.13 | BIDS derivatives pass bids-validator in derivatives mode (CI job with the validator installed); every TSV column … |
| REQ-343 | §5.13 | report.html renders without network access (all assets inline) and is ≤ 20 MB for 100 channels. |
| REQ-344 | §5.13 | every plots function runs headless (Agg backend) on every result type in CI and returns a … |
| REQ-345 | §5.13 | every CLI command has a test invoking it through subprocess on the cached sample data; outputs … |
| REQ-346 | §5.13 | evoparam cite prints the citation from CITATION.cff (single source of truth, §11). |
| REQ-347 | §5.13 | StreamFitter meets the latency budget and the offline agreement of REQ-324 (§10.6: < 20 ms at … |
| REQ-348 | §5.13 | the sequential existence statistic keeps the overall false-positive rate ≤ α (+0.01) across 50 looks on … |
| REQ-349 | §5.13 | no function in reports, plots, cli or stream computes a scientific quantity that is not available … |
| REQ-350 | §8.5 | evoparam benchmark run --suite core completes in ≤ 30 min on the reference CI machine and … |
| REQ-351 | §8.5 | every task's input and output schemas are versioned (benchmark-task/<id/1) and validated on read; scoring rejects outputs … |
| REQ-352 | §8.5 | scoring is deterministic and the signed JSON is reproducible byte for byte from the same inputs. |
| REQ-353 | §8.5 | all six published threshold rules and the cross-projection test are present as baselines and their T1/T2 … |
| REQ-354 | §8.5 | the adapter mechanism runs an external command-line tool on task files without importing it; the adapters … |
| REQ-355 | §8.5 | the simulated suites are deposited as NWB/BIDS files with a DOI before the benchmark paper is … |
| REQ-356 | §8.5 | the hidden split's seeds are committed as a SHA-256 hash at deposit time and revealed at … |
| REQ-357 | §8.5 | the leaderboard page is generated by a script from result JSONs only; no hand-edited numbers. |
| REQ-370 | §7.2 | the full layer-2 grid runs in the scheduled CI job (weekly) in ≤ 6 h on … |
| REQ-371 | §7.3 | the ds004080 subject list is generated by the rule above by a script in the repository … |
| REQ-372 | §7.3 | on ds004977, rereference(scheme="car_adaptive") excludes the responsive channels identified in the published analysis with a Jaccard index … |
| REQ-373 | §7.3 | on Localize-MI, the existence test on the scalp signal is significant for ≥ 80 % of … |
| REQ-374 | §7.3 | every layer-3 comparison reports the comparator's number, evoparam's number, the difference with its interval, and the … |
| REQ-375 | §7.4 | the five robustness analyses run in the release pipeline of every minor version and are attached … |
| REQ-376 | §7.4 | a regression in any layer-3 metric larger than its pre-registered tolerance (κ − 0.05, latency +1 … |
| REQ-390 | §11.8 | Every result object (ResponseFit, GroupFit, TrialFit, HierarchicalFit, ResponseTest, ArtifactResult, TFFit) produces methods_text() from its Provenance and … |
| REQ-391 | §11.8 | The repository root carries CITATION.cff; every release is archived on Zenodo with a versioned DOI under … |
| REQ-392 | §11.8 | The benchmark (§8) is public: tasks, metrics, simulated sets and labels are versioned and archived; the … |
| REQ-393 | §11.8 | A preprint precedes every journal submission; the preprint names the exact package tag and benchmark version … |
| REQ-394 | §11.8 | The public name is frozen at the first public release; README, docs home, paper titles and … |
| REQ-395 | §13.7 | No module of the free package imports, vendors or declares as an extra any package under … |
| REQ-396 | §13.7 | A provenance map is maintained and shipped: SPDX licence header in every source file; inherited files … |
| REQ-397 | §13.7 | A TRADEMARK.md states who owns the name (per HIIE decision), what forks may be called, how … |
| REQ-398 | §13.7 | The invention disclosure is filed and HIIE's written decision (patent or not; licence; trademark owner) is … |
| REQ-399 | §13.7 | A machine-readable SOUP inventory (soup.json: package, version range, licence, anomaly source URL, functions used) is generated … |
## 4. Architecture

### 4.1 Design principles

1. **One chain, separable steps.** The package implements the chain *event → artifact → response → reliable numbers*. Every step is a pure function or a scikit-learn-style estimator that accepts and returns the containers defined in §4.3, so each step can be used alone, replaced by a user's own code, or run on data that never passed through the earlier steps.
2. **Light core, optional extras.** The import `import evoparam` must succeed with only `numpy` and `scipy` installed (REQ-291). Everything else (`mne`, `pandas`, `matplotlib`, `pynwb`, `neo`, `pyedflib`, `pymc`/`numpyro`, `cupy`, `tqdm`, `click`) is an optional extra loaded lazily inside the function that needs it, with an error message naming the extra to install.
3. **Numbers carry their provenance.** Every result object stores the package version, the full settings, the data hash, the random seed and the list of processing steps already applied to the input (REQ-294, REQ-331). Saving a result and reloading it reproduces the same numbers and the same methods paragraph.
4. **Units are explicit.** Time is in seconds everywhere in the public API (the convention of MNE-Python). Amplitude units travel with the data (`units` attribute, default `'uV'`). Stimulation parameters are stored with their unit in the column name (`stim_current_mA`, `stim_pulse_width_us`).
5. **Polarity-agnostic by default.** Intracranial responses change sign with montage and depth [LIT-ccep-11], [LIT-ccep-8]. All detectors and shape fits operate on signed data but report `polarity` as a separate field, and every preset for intracranial data searches both signs.
6. **No inherited spectral vocabulary.** Class, function and parameter names describe evoked responses in the time domain. The ancestor tools are acknowledged only in `NOTICE`.

### 4.2 Modules and dependencies

```
                       numpy · scipy  (hard dependencies)
                               │
   ┌───────────────────────────┴────────────────────────────┐
   │                        evoparam.data                     │  containers: TrialSet, Evoked, Provenance
   └───┬───────────┬──────────┬──────────┬──────────┬────────┘
       │           │          │          │          │
   evoparam.io  evoparam.events  evoparam.artifact  evoparam.prep  evoparam.shapes
   (readers,     (pulse         (interpolate,      (baseline,     (component shapes,
    writers,      detection,     template,          reference,     backgrounds,
    remote)       saturation,    period_template,     rejection)     features)
                  stim site)     learned, QC)                           │
       │           │          │          │                             │
       └───────────┴──────────┴──────────┴──────────┐                  │
                                                     ▼                  ▼
                                               evoparam.models  ◄───────┘
                                               (ResponseModel, fitting engine,
                                                model selection, classical measures)
                                                     │
          ┌──────────────┬──────────────┬────────────┼─────────────┬──────────────┐
          ▼              ▼              ▼            ▼             ▼              ▼
   evoparam.existence  evoparam.trials  evoparam.spatial  evoparam.tf  evoparam.stats  evoparam.sim
   (response test,     (trial-by-trial, (shared shapes,   (time-freq.  (bootstrap,     (simulators,
    duration)           jitter,          maps,             zones,       comparisons,    ground truth)
                        realignment)     convergent)       ECoG/sEEG)   mixed export)
                             │
                             ▼
                      evoparam.hierarchical
                      (trial→channel→subject, Bayesian option)

   cross-cutting:  evoparam.presets · evoparam.reports · evoparam.plots · evoparam.cli
                   evoparam.stream (online mode) · evoparam.benchmark · evoparam.utils
```

Dependency rules (enforced by an import-graph test, REQ-296):

| module | may import | must not import |
|---|---|---|
| `data` | `numpy` only (+ `pandas` lazily for `metadata`) | any other evoparam module |
| `io` | `data`, `utils` | `models` and anything below it |
| `events`, `artifact`, `prep` | `data`, `utils`, `shapes` (artifact template fits) | `models`, `trials`, `spatial`, `tf` |
| `shapes` | `numpy`, `scipy` | any other evoparam module |
| `models` | `data`, `shapes`, `prep` (baseline helpers), `utils` | `existence`, `trials`, `spatial`, `tf`, `stats`, `hierarchical` |
| `existence` | `data`, `models`, `utils` | `trials`, `hierarchical` |
| `trials` | `data`, `models`, `existence`, `utils` | `hierarchical` |
| `hierarchical` | `data`, `models`, `trials`, `stats` | — |
| `spatial`, `tf`, `stats` | `data`, `models`, `utils` | `hierarchical` (stats is imported by hierarchical, not the reverse) |
| `sim` | `data`, `shapes`, `artifact` (artifact templates) | `models` (ground truth must not depend on the fitter) |
| `presets` | nothing but `dataclasses` | everything (presets are pure data) |
| `reports`, `plots`, `cli`, `stream`, `benchmark` | any module above | each other, except `benchmark → reports` |

Circular imports are forbidden. The test `tests/test_import_graph.py` parses every module's imports with `ast` and fails on any edge not listed above. Two leaf modules, `evoparam.errors` (the hierarchy of §4.6) and `evoparam.version`, import nothing from the package and may be imported by every module, `data` included. Until the inherited modules are rewritten (§4.5, §12), the test lists their remaining edges explicitly, each with the build step that removes it.

### 4.3 Data model

All containers are frozen-by-convention `dataclasses` with explicit `__eq__`, `copy()`, `to_dict()`/`from_dict()` and a `__repr__` that prints dimensions and units. Arrays are `numpy.ndarray` of dtype `float64` unless stated. Time axis is always the last axis.

#### 4.3.1 `TrialSet` — trials × channels × time

```python
@dataclass
class TrialSet:
    data: np.ndarray          # shape (n_trials, n_channels, n_times); float64; NaN allowed only inside artifact windows marked in `masks`
    times: np.ndarray         # shape (n_times,); seconds; strictly increasing; 0.0 is the event onset (sample closest to onset)
    sfreq: float              # Hz; abs(1/median(diff(times)) - sfreq) / sfreq < 1e-6
    ch_names: list[str]       # length n_channels; unique; non-empty
    units: str = "uV"         # one of {"uV", "mV", "V", "fT", "au"}; conversions via TrialSet.convert_units
    metadata: pd.DataFrame | None = None   # exactly n_trials rows; standard columns in Table 4.1; extra columns allowed
    ch_info: pd.DataFrame | None = None    # exactly n_channels rows; columns in Table 4.2
    masks: np.ndarray | None = None        # bool, shape (n_trials, n_channels, n_times); True = sample replaced/invalid (artifact window, clipping)
    event_times: np.ndarray | None = None  # shape (n_trials,); absolute onset time (s) in the source recording
    species: str = "human"    # free text, lower case; presets check it ("human", "macaque", "mouse", "rat", "pig", ...)
    modality: str | None = None   # "spes" | "erp" | "tep" | "erna" | "sep" | "ecap" | "vep" | "baep" | "mep" | "lfp_ep" | None
    provenance: Provenance = field(default_factory=Provenance)
```

Invariants (checked in `__post_init__`, raising `ValueError` with the offending field):
- `data.ndim == 3`; `data.shape[2] == times.size`; `data.shape[1] == len(ch_names)`.
- `times` strictly increasing; `times.size >= 2`.
- If `metadata` is given, `len(metadata) == n_trials`; index is `RangeIndex` 0..n_trials−1.
- If `masks` is given, same shape as `data`.
- `units` in the allowed set.

Standard metadata columns (Table 4.1). None is mandatory; when present they must have the stated dtype. Readers fill them from BIDS `events.tsv`, NWB trials tables or user dictionaries.

| column | dtype | meaning |
|---|---|---|
| `trial` | int64 | trial index in the source run (0-based) |
| `subject`, `session`, `run`, `task` | str | BIDS entities |
| `condition` | str | experimental condition label (e.g. "standard", "deviant", "3mA") |
| `stim_site` | str | stimulated contact pair, BIDS style "PT01-PT02"; "n/a" for sensory paradigms |
| `stim_current_mA` | float64 | current amplitude (mA) |
| `stim_pulse_width_us` | float64 | pulse width per phase (µs) |
| `stim_polarity` | str | "monophasic_cathodic", "monophasic_anodic", "biphasic_cathodic_first", "biphasic_anodic_first", "alternating" |
| `stim_frequency_hz` | float64 | inter-pulse frequency in a train (Hz); NaN for single pulses |
| `stim_n_pulses` | int64 | pulses in the train that this trial belongs to |
| `stim_charge_uC` | float64 | derived: current × pulse width (µC/phase) |
| `state` | str | "awake", "nrem", "rem", "propofol", "sevoflurane", "unknown" |
| `block` | int64 | block or run segment |
| `quality` | float64 | 0–1 artifact-quality score (set by `evoparam.artifact`) |
| `rejected` | bool | trial excluded by `evoparam.prep.reject_trials` |
| `reject_reason` | str | "amplitude", "flat", "saturation", "artifact_residual", "manual", "" |

Channel table columns (Table 4.2), filled from BIDS `channels.tsv`/`electrodes.tsv`, MNE `info`, or NWB electrodes:

| column | dtype | meaning |
|---|---|---|
| `name` | str | equals `ch_names[i]` |
| `type` | str | "ecog", "seeg", "dbs", "eeg", "meg", "ecap", "lfp", "micro", "emg", "misc" |
| `group` | str | electrode shaft or grid name |
| `x`, `y`, `z` | float64 | coordinates (mm) in `coord_space` |
| `coord_space` | str | e.g. "MNI152NLin2009aSym", "ACPC", "scanner", "" |
| `reference` | str | "" (unknown), "monopolar:<name>", "bipolar:<a>-<b>", "car", "car_adaptive", "laplacian" |
| `bad` | bool | channel flagged bad |
| `stim_contact` | bool | channel is a stimulating contact for at least one trial (so it is excluded from response fits by default) |
| `tissue` | str | "gray", "white", "csf", "unknown" (optional, from the dataset) |
| `region` | str | atlas label (optional) |

Methods of `TrialSet`:

```python
def select(self, trials=None, picks=None, tmin=None, tmax=None, query: str | None = None) -> "TrialSet"
def average(self, by: str | list[str] | None = None, robust: bool = False,
            chunk: int | Literal["auto"] | None = None) -> "Evoked"                               # mean (or 20 % trimmed mean when robust) over trials, grouped by metadata column(s); chunk = channels per chunk (rule of §9.1)
def convert_units(self, units: str) -> "TrialSet"
def crop(self, tmin: float, tmax: float) -> "TrialSet"
def apply(self, func, axis="trials", chunk: int | Literal["auto"] | None = None) -> "TrialSet"      # vectorised user transform with provenance record; applied chunk by chunk when chunk is set (§9.1)
def to_mne(self) -> "mne.EpochsArray"                                                             # requires extra "mne"
def to_xarray(self) -> "xarray.DataArray"                                                         # requires extra "xarray"; dims ("trial", "channel", "time")
def save(self, path, format="npz") -> None                                                        # "npz" (core) | "nwb" | "fif"; the core format stores `data` as an uncompressed array so that it can be memory-mapped
@classmethod
def load(cls, path, *, mmap: bool = False) -> "TrialSet"                                         # mmap=True memory-maps `data` read-only instead of reading it into memory (workload P3 of §9.1)
```

`average` returns an `Evoked` whose `n_trials` and `sem` (standard error of the mean over trials) are kept, because the fitting engine weights residuals by `sem` when it is available (REQ-103).

#### 4.3.2 `Evoked` — channels × time

```python
@dataclass
class Evoked:
    data: np.ndarray        # (n_channels, n_times)
    times: np.ndarray
    sfreq: float
    ch_names: list[str]
    units: str = "uV"
    n_trials: int | np.ndarray = 1        # scalar or per-channel array (after per-channel rejection)
    sem: np.ndarray | None = None         # (n_channels, n_times) standard error of the mean, same units
    masks: np.ndarray | None = None       # (n_channels, n_times); True = invalid
    metadata: pd.DataFrame | None = None  # one row: the grouping values used by TrialSet.average
    ch_info: pd.DataFrame | None = None
    species: str = "human"
    modality: str | None = None
    provenance: Provenance = field(default_factory=Provenance)
```

A `TrialSet` with one trial and an `Evoked` are interchangeable inputs to every fitting function (the function calls `as_evoked()` / `as_trialset()`).

#### 4.3.3 `Provenance`

```python
@dataclass
class Provenance:
    package_version: str                 # evoparam.__version__
    created: str                         # ISO-8601 UTC
    source: dict                         # {"path": ..., "format": "brainvision", "reader": "evoparam.io.read_brainvision", "sha256": ..., "bytes_read": ...}
    steps: list[dict]                    # ordered; each {"function": "evoparam.artifact.remove_artifact", "settings": {...}, "version": "...", "duration_s": 1.2}
    environment: dict                    # {"python": "3.12.3", "numpy": "2.1.0", "scipy": "1.14.1", "platform": "..."}
    random_state: int | None
    data_hash: str | None                # sha256 of data bytes + times at creation time
```

Every public function that transforms a container appends one `steps` entry (REQ-294). `Provenance.to_json()` is what the methods-text generator reads.

#### 4.3.4 Shapes and models

```python
class ComponentShape(Protocol):          # registry entries in evoparam.shapes.SHAPES
    name: str                            # "gaussian", "skewed_gaussian", "exgaussian", "gamma", "biexponential", "damped_oscillation", "gabor", "template" (§5.5.1)
    param_names: tuple[str, ...]         # e.g. ("latency", "amplitude", "width") ; all in SI (s, data units, s)
    n_params: int
    def evaluate(self, times: np.ndarray, params: np.ndarray) -> np.ndarray          # vectorised: params shape (..., n_params) → (..., n_times)
    def jacobian(self, times, params) -> np.ndarray                                  # analytic, shape (..., n_times, n_params)
    def default_bounds(self, times, data_range) -> tuple[np.ndarray, np.ndarray]
    def initial_guess(self, times, waveform, peak_index: int) -> np.ndarray
    def features(self, times, params) -> "ComponentFeatures"                         # latency, amplitude, fwhm, onset, offset, area, rise_time, decay_time, symmetry, sharpness

class Background(Protocol):              # registry evoparam.shapes.BACKGROUNDS
    name: str                            # "none", "constant", "linear", "exponential", "double_exponential", "pspline", "artifact_basis" (§5.5.2)
    ... same methods ...
```

```python
@dataclass
class Component:
    shape: str
    params: np.ndarray                   # (n_params,)
    param_names: tuple[str, ...]
    latency: float                       # s, time of the extremum of the fitted component
    amplitude: float                     # data units, signed value at the extremum
    polarity: int                        # +1 or -1
    fwhm: float                          # s
    onset: float                         # s, first time the component reaches 10 % of |amplitude| (fraction configurable)
    offset: float                        # s, last time it is above 10 %
    area: float                          # units·s, signed integral over [onset, offset]
    rise_time: float                     # s, onset → latency
    decay_time: float                    # s, latency → offset
    symmetry: float                      # (decay_time − rise_time) / (decay_time + rise_time); 0 for a symmetric shape
    sharpness: float                     # |s''(latency)| · fwhm², dimensionless curvature of the unit-amplitude shape at the peak (§5.5.1)
    label: str | None = None             # assigned by `evoparam.models.label_components`, e.g. "N1", "P300"
    se: dict[str, float] | None = None   # standard errors per parameter (from the Jacobian or bootstrap)
    ci: dict[str, tuple[float, float]] | None = None   # (low, high) at `ci_level`
    posterior: dict | None = None        # hierarchical fits only: mean, sd, hdi, ess, r_hat per parameter
```

```python
@dataclass
class GoodnessOfFit:
    r2: float                            # 1 − SS_res / SS_tot on the fit window (coefficient of determination, not Pearson²)
    adj_r2: float
    rmse: float                          # data units
    mae: float
    nrmse: float                         # rmse / (max y − min y) on the fit window (dimensionless)
    aic: float
    aicc: float                          # small-sample corrected AIC (§5.5.4)
    bic: float
    n_params: int
    n_samples: int
    n_eff: float                         # effective sample count after the AR(1) correction (§5.5.3)
    residual_lag1_autocorr: float        # diagnostic of model misspecification
    baseline_sd: float                   # noise level used
```

```python
@dataclass
class ResponseFit:                       # result of fitting ONE waveform (one channel, one condition)
    components: list[Component]          # sorted by latency
    background: dict                     # {"mode": "exponential", "params": ..., "param_names": ...}
    gof: GoodnessOfFit
    status: str                          # "ok" | "no_response" | "failed" | "skipped"
    message: str                         # reason when not "ok"
    window: tuple[float, float]          # fit window (s)
    baseline: tuple[float, float]        # baseline window (s)
    settings: dict                       # full resolved settings (preset + overrides)
    classical: dict                      # ERP-style measures computed on the data in the same window (see §5.5.6)
    model: np.ndarray | None             # fitted curve on `times` (optional, dropped when saving with `light=True`)
    residual: np.ndarray | None
    existence: "ResponseTest | None"     # attached when the existence test was run
    warnings: list[str]
    provenance: Provenance
    def to_dataframe(self) -> pd.DataFrame        # one row per component (tidy)
    def methods_text(self) -> str
```

```python
@dataclass
class GroupFit:                          # ResponseFit per channel (and per condition when the input was grouped)
    fits: list[ResponseFit]
    ch_names: list[str]
    conditions: list[str] | None
    index: pd.DataFrame                  # (channel, condition) → position in `fits`
    def to_dataframe(self, level="component") -> pd.DataFrame     # "component" | "fit" | "classical"
    def get(self, channel, condition=None) -> ResponseFit
    def summary(self) -> pd.DataFrame
    def save(self, path, light=True) -> None
    @classmethod
    def load(cls, path) -> "GroupFit"
```

```python
@dataclass
class TrialFit:                          # trial-level parameters for one channel (and condition)
    channel: str
    template: ResponseFit                # the channel-level model that seeded the trial fits
    shift: np.ndarray                    # (n_trials,) latency shift of the whole response (s)
    scale: np.ndarray                    # (n_trials,) amplitude scale of the whole response
    params: np.ndarray                   # (n_trials, n_components, n_params) free parameters per trial after warm start
    latency: np.ndarray                  # (n_trials, n_components) s
    amplitude: np.ndarray                # (n_trials, n_components) units
    fwhm: np.ndarray                     # (n_trials, n_components) s
    gof: np.ndarray                      # (n_trials,) r2
    converged: np.ndarray                # (n_trials,) bool
    jitter_sd: dict[str, float]          # per component label: SD of latency across trusted trials (s)
    jitter_sd_null: dict[str, float]     # per component label: jitter obtained by the same procedure on the pre-stimulus baseline (s)
    amplitude_jitter_corrected: dict[str, float]   # per component label: amplitude a jitter-free average would show (§5.7)
    trusted: np.ndarray                  # (n_trials,) bool: converged, r2 ≥ r2_min and shift not at a bound
    table: pd.DataFrame                  # tidy, one row per trial × component, with the TrialSet metadata columns
    covariates: pd.DataFrame | None      # built-in covariates ("trend", "dose", "prestim") when requested (§5.7)
    realigned: "TrialSet | None"         # trials shifted by −shift (optional)
    settings: dict
    provenance: Provenance
```

```python
@dataclass
class HierarchicalFit:
    levels: tuple[str, ...]              # ("trial", "channel", "subject") or a subset
    summary: pd.DataFrame                # one row per (level, unit, component, parameter): mean, sd, hdi_low, hdi_high, ess_bulk, r_hat
    level_tables: dict[str, pd.DataFrame]   # one tidy table per level
    p_response: pd.DataFrame             # posterior probability of a response per (channel, stimulation site, component)
    variance_components: pd.DataFrame    # variance at each level per parameter
    effects: pd.DataFrame | None         # formula terms with intervals when a design formula was given
    engine: str                          # "eb" (empirical Bayes) | "numpyro" | "pymc"
    diagnostics: dict                    # divergences, max r_hat, min ess, runtime
    settings: dict
    provenance: Provenance
```

```python
@dataclass
class ResponseTest:                      # output of evoparam.existence
    method: str                          # "baseline_permutation" | "projection" | "model_comparison" | "threshold"
    statistic: np.ndarray                # (n_channels,)
    p_value: np.ndarray                  # (n_channels,) after the chosen multiple-comparison correction
    p_value_raw: np.ndarray
    significant: np.ndarray              # bool (n_channels,)
    duration: np.ndarray                 # (n_channels,) s; response duration τ_R, NaN when not significant
    onset: np.ndarray                    # (n_channels,) s; first significant time, NaN when not significant
    offset: np.ndarray                   # (n_channels,) s; last significant time, NaN when not significant
    snr_db: np.ndarray                   # (n_channels,) 10·log10(signal power / noise power) in the response window
    rule: str | None                     # registry rule name when method == "threshold" (§5.6)
    alpha: float
    correction: str                      # "none" | "fdr_bh" | "bonferroni" | "holm"
    n_permutations: int
    window: tuple[float, float]
    settings: dict
    provenance: Provenance
```

```python
@dataclass
class ArtifactResult:
    cleaned: TrialSet
    method: str                          # "interpolate" | "template" | "tail" | "period_template" | "lowrank" | "subspace" | "polynomial" | "matching_pursuit" | "paradigm" | "learned" (§5.3)
    window: tuple[float, float]          # (s) replaced / modelled window relative to onset
    template: np.ndarray | None          # (n_channels, n_window_samples) when a template was estimated
    quality: np.ndarray                  # (n_trials,) 0–1 score; also written to cleaned.metadata["quality"]
    quality_components: pd.DataFrame     # (n_trials rows) residual_ratio, edge_discontinuity, saturation_fraction, template_fit_r2
    basis: np.ndarray | None             # (n_basis, n_window_samples) artifact basis learned by method "learned"; used by the "artifact_basis" background
    settings: dict
    provenance: Provenance
```

```python
# evoparam.events returns a pandas DataFrame named EventTable with fixed columns:
# onset_sample (int64), onset_time (float64, s), peak_abs (float64, units), channel_argmax (str),
# saturated (bool), n_saturated_channels (int64), stim_site_guess (str, "A-B" or ""), site_confidence (float64, 0–1),
# train_id (int64), index_in_train (int64), detector (str), threshold (float64)
```

```python
@dataclass
class ShapeBasis:                        # evoparam.spatial shared shapes
    curves: np.ndarray                   # (n_shapes, n_times); unit-norm canonical curves
    times: np.ndarray
    weights: np.ndarray                  # (n_groups, n_shapes) non-negative weights
    r2: np.ndarray                       # (n_groups,) variance explained per group
    assignment: pd.DataFrame             # one row per group (stimulation site or channel): shape_id (int, −1 = none), weight, r2
    stability: np.ndarray                # (n_shapes,) reproducibility across restarts
    n_shapes: int
    method: str                          # "nmf" | "kpca" (§5.9)
    settings: dict
    provenance: Provenance
```

```python
@dataclass
class TFFit:                             # evoparam.tf
    freqs: np.ndarray                    # (n_freqs,) Hz
    times: np.ndarray                    # (n_times_tf,) s
    maps: dict[str, np.ndarray]          # "total", "evoked_footprint", "residual", "cross"; each (n_channels, n_freqs, n_times_tf), baseline-normalised (dB or z)
    tmaps: np.ndarray                    # (n_channels, n_freqs, n_times_tf) test statistic against the baseline
    significant: np.ndarray              # bool, same shape as tmaps, after the cluster correction
    zones: pd.DataFrame                  # one row per discovered zone: zone_id, t_start, t_end, f_low, f_high, sign, peak_t, peak_f, peak_value, area, p_value
    zone_metrics: pd.DataFrame           # zone score per channel (and per trial with by_trial)
    method: str                          # zone method: "nmf" | "cp" (§5.10)
    settings: dict
    provenance: Provenance
```

```python
@dataclass(frozen=True)
class Preset:                            # evoparam.presets
    name: str                            # "erp", "ccep", "ccep_seeg", "sep", "tep", "erna", "ecap", "ecap_ci", "rodent_ep", "primate_ep", "p300_bci" (§6)
    modality: str
    species: tuple[str, ...]
    epoch: tuple[float, float]
    baseline: tuple[float, float]
    artifact_window: tuple[float, float] | None
    artifact_method: str | None
    reference: str | None
    fit_window: tuple[float, float]
    shapes: tuple[str, ...]
    background: str
    max_components: int
    min_fwhm: float
    max_fwhm: float
    polarity: str                        # "both" | "negative" | "positive"
    existence_method: str
    existence_alpha: float
    component_labels: dict[str, tuple[float, float]]    # label → latency window (s), e.g. {"N1": (0.010, 0.050), "N2": (0.050, 0.300)}
    trial_fit: bool
    citation: tuple[str, ...]            # literature keys justifying the values
    version: str                         # preset version, printed in the methods text (REQ-303)
    validation: str                      # how the values were validated, e.g. "simulation_only" (REQ-279)
```

#### 4.3.5 Serialisation

- Core format: a directory or `.zip` named `<name>.evoparam` containing `manifest.json` (schema version `"evoparam-result/1"`, package version, object type), `data.npz` (arrays), `tables/*.parquet` or `*.csv` (DataFrames; parquet when `pyarrow` is available, CSV otherwise), `provenance.json`, `settings.json`. NaN is written as `null` in JSON and as NaN in npz. Loading a file with a newer major schema raises `EvoparamVersionError`.
- BIDS derivatives (`evoparam.io.write_bids_derivatives`) in §5.1, §5.13 and §10.2.
- NWB (`evoparam.io.write_nwb`) in §5.1 and §10.3.

### 4.4 Naming conventions and renames

| inherited name | new name | reason |
|---|---|---|
| class `evoparam` (one waveform fitter + results) | `ResponseModel` (estimator) + `ResponseFit` (result) | a class named like the package shadows the module in `from evoparam import evoparam`; estimator and result are separated as in scikit-learn so results are plain data |
| class `evoparamGroup` | `GroupFit` (result) ; fitting via `ResponseModel.fit_group` / `evoparam.fit` | same split |
| `fit_evoparam_3d` | removed; `evoparam.fit` accepts `TrialSet` (3-D) directly | one entry point |
| `gaussian_params_` / `shape_params_` | `Component.params` + feature fields | names say what the numbers are |
| `peak_width_limits` | `min_fwhm`, `max_fwhm` (seconds) | one unit, two explicit bounds |
| `max_n_peaks` | `max_components` | the model fits components, not only peaks |
| `min_peak_height` | `min_amplitude` (data units) | |
| `peak_threshold` | `detection_threshold` (in baseline-SD units) — used only for initial guesses, never as the final acceptance rule (§5.5.3) | |
| `peak_mode` | `shapes` (tuple of allowed shape names) | several shapes may compete under model selection |
| `gauss_overlap_thresh` | `overlap_threshold` | shape-agnostic |
| `maxfev` | `max_iter` | scipy-agnostic name |
| `amplitude_fraction` | `onset_fraction` (fraction of amplitude defining onset/offset, default 0.10) | the inherited 0.5 half-magnitude rule is replaced; 0.5 remains available |
| `r_squared_` (Pearson²) | `gof.r2` (coefficient of determination) | the inherited value is a correlation, not an R² |
| `Bands` / `bands` | `ComponentWindows` in `evoparam.models.windows` | windows name components in time, not frequency bands |
| `sim.gen.simulate_erp` | `evoparam.sim.simulate_erp` (kept name), plus `simulate_ccep`, `simulate_erna`, `simulate_tep`, `simulate_animal_ep` | |
| `plts.*` | `evoparam.plots.*` | full word |
| `objs/`, `core/`, `utils/data.py` | folded into `models/`, `shapes/`, `prep/` per §4.5 | |

General rules:
- Modules: lower-case nouns (`events`, `artifact`, `trials`). Public functions: verb first (`read_bids`, `detect_pulses`, `remove_artifact`, `fit_response`, `test_response`, `fit_trials`, `simulate_ccep`, `write_bids_derivatives`). Classes: `CapWords` nouns. Results end in `Fit`, `Test`, `Result`, `Table` or `Basis`.
- Argument names follow MNE-Python and scikit-learn where the concept exists: `tmin`, `tmax`, `sfreq`, `picks`, `n_jobs`, `random_state`, `verbose`, `copy`.
- Every keyword argument that is a time is in seconds; every one that is a count ends in `n_` or `_n`; every one that is a fraction is in [0, 1] and ends in `_fraction`; every threshold states its unit in the docstring and, when the unit is not the data unit, in the name (`threshold_sd`).
- Booleans are `is_`/`has_`/`use_` only for attributes; for arguments plain adjectives (`robust=True`).
- Deprecations: a renamed public name keeps a shim for two minor versions that emits `DeprecationWarning` naming the replacement (REQ-293).

### 4.5 Disposition of the inherited code

The inherited tree was read in full (inventory of 2026-10-06; file and line references are to commit `ef4fad0`). Decision per module:

| inherited module | decision | reason (evidence) |
|---|---|---|
| `io/brainvision.py`, `io/bids.py`, `io/epochs.py`, `io/fetch.py` (skeleton 0.1) | **keep**, extend into `evoparam.io` | already tested on ds004080; partial remote reads work (48 s of a 2 GB file) |
| `detect/threshold.py` (skeleton) | **keep** as `evoparam.events.detect_pulses` (threshold detector), add the other detectors of §5.2 | 10/10 pulses, 0 false alarms, 2.0 ms error on the example run |
| `clean/interpolate.py` (skeleton) | **keep** as `evoparam.artifact.interpolate` | correct for any array shape; raises on empty window / insufficient buffer |
| `param/epochs.py::fit_epochs` (skeleton) | **rewrite** into `evoparam.fit` | it averages before fitting and silently swallows `FitError` (`verbose=False` default) so a channel with no fit is indistinguishable from a channel with no response |
| `objs/fit.py` (≈1 500 lines; fitter + results + plotting + I/O in one class) | **rewrite** as `models/engine.py`, `models/response.py`, `models/select.py`; keep only the vectorised Gaussian/skewed-Gaussian functions after review | (a) the stop rule compares the window maximum with `peak_threshold × std(baseline)`; pure noise passes this at 3 SD in 93 % of simulated windows, so up to `max_n_peaks` noise guesses are generated per polarity, `curve_fit` then fails and the error is swallowed → empty channel; (b) three rules delete an early CCEP N1: the edge rule in SD units at the window start (`fit.py:1206-1208`), the half-magnitude points that must exist in the raw signal inside the window (`fit.py:1061-1071`, `1268-1277`), and the N1-on-N2-flank rule; starting the fit at 5 ms instead of 10 ms rescued all synthetic cases, which shows the rules, not the data, are the cause; (c) heights are bounded by the signal min/max so a DC offset yields no components; (d) `r_squared_` is a Pearson correlation squared; (e) the class mixes settings, data, results, plotting and JSON I/O; (f) `verbose=False` hides every failure |
| `objs/group.py` | **rewrite** as `GroupFit` + `fit_group` | `average_fg` raises `NameError`; `combine_evoparams` reads `fg.signal` which does not exist; loading a group with a null fit fails; `get_params('peak_indices')` breaks on numpy ≥ 1.24 |
| `objs/utils.py` (`compare_info`, `average_fg`, `combine_evoparams`, `fit_evoparam_3d`) | **remove**; replace by `GroupFit.summary`, `TrialSet.average`, `evoparam.fit` | broken or redundant (see above) |
| `core/funcs.py` (shape functions) | **keep** as the seed of `evoparam.shapes` after adding analytic Jacobians and the four new shapes | functions are correct and vectorised |
| `core/corrections.py` (`correct_peaks_indices`, overlap/half-magnitude relocation) | **remove**; model selection replaces heuristics | asymmetric-overlap case returns NaN; the heuristics are the documented source of lost early components |
| `core/io.py` (JSON save/load) | **rewrite** as `data/serialize.py` (§4.3.5) | writes NaN literals into JSON (invalid JSON), no schema version |
| `core/reports.py`, `core/strings.py` | **rewrite** as `evoparam.reports` (methods text, PDF) | text and settings lists are spectral-era; the methods paragraph must be generated from `Provenance` |
| `core/info.py`, `core/items.py`, `core/modutils.py`, `core/errors.py` | **merge** into `utils/` (`modutils` and `errors` kept as is) | small utilities; `errors.py` defines the exception hierarchy reused in §5 |
| `data/data.py` (named tuples for settings/results), `data/conversions.py` | **rewrite** as the dataclasses of §4.3 | named tuples cannot carry uncertainty, provenance or warnings |
| `analysis/peaks.py` (window-based extraction) | **rewrite** as `models/windows.py::label_components` | the logic (highest component in a named window) is right; the API must operate on `Component` lists |
| `analysis/error.py` | **merge** into `GoodnessOfFit` | |
| `bands/` | **rename** `models/windows.py` (`ComponentWindows`) | time windows, not bands |
| `sim/gen.py`, `sim/params.py`, `sim/utils.py` | **rewrite** as `evoparam.sim` with the generators of §5.12; drop the `neurodsp` dependency (power-law noise is generated in-house with `scipy.signal`) | the package currently cannot be imported without `neurodsp` (`sim/gen.py:4`) |
| `plts/signals.py`, `plts/templates.py`, `plts/utils.py`, `plts/style.py` | **keep** inside `evoparam.plots` with a lazy `matplotlib` import | the package currently cannot be imported without `matplotlib` (`plts/style.py:6`) |
| `plts/fm.py`, `plts/peaks.py` | **remove**; replaced by `plots.plot_fit`, `plots.plot_components` | symmetry column bug; 0.1 Hz grid hard-coded; `legend.legendHandles` removed in matplotlib ≥ 3.9 |
| `utils/download.py`, `utils/debug.py` | **remove** | spectral-era data download; debug prints |
| `utils/data.py` (`interpolate_signal`, `trim_signal`, `compute_average`) | **merge** into `prep/` and `artifact/` | `interpolate_signal` already delegates to `clean.interpolate_artifact` |
| `tests/` (199 tests) | **keep** the io/detect/clean/param tests; rewrite the fitter tests with ground-truth assertions | current tests assert that code runs, not that the numbers are right (no accuracy or robustness test, no HTTP test, non-deterministic fixtures) |
| `NOTICE`, `LICENSE` | **keep unchanged** | Apache-2.0 obligations |

### 4.6 Error hierarchy

```python
class EvoparamError(Exception): ...
class DataError(EvoparamError): ...          # malformed container, unit mismatch, NaN outside masks
class ReaderError(EvoparamError): ...        # unreadable file, missing sidecar, remote range request refused
class EventError(EvoparamError): ...         # no pulses found, inconsistent train structure
class ArtifactError(EvoparamError): ...      # window outside epoch, template cannot be estimated
class FitError(EvoparamError): ...           # optimiser failure after all restarts (never silently caught; see REQ-118)
class NoResponseWarning(UserWarning): ...    # emitted, not raised, when the existence test is negative and a fit was still requested
class ConvergenceWarning(UserWarning): ...   # trial fits or MCMC not converged (r_hat > 1.01, ess < 400, divergences > 0)
class SettingsError(EvoparamError): ...      # invalid preset override (e.g. min_fwhm > max_fwhm)
class EvoparamVersionError(EvoparamError): ...
```

Rule: a function never returns a result and hides an exception. When a per-channel or per-trial failure is acceptable, the result carries `status="failed"` and `message`, and the function emits a `logging` record at level WARNING with the channel name (REQ-118, REQ-307).
## 5. Module specifications

Every module below follows the same template: purpose; methods with equations and references; public API (full Python signatures, types, defaults, errors); edge cases; measurable acceptance criteria; tests to write; tutorials. Requirement ids come from the blocks of §3. Citation keys `[LIT-<axis>-<n>]` resolve to the numbered reference lists of `LITERATURE.md`; `[CODE §x]` refers to the code inventory summarised in §4.5.

Conventions common to all modules:
- Time arguments are in seconds relative to the event onset; windows are `(tmin, tmax)` tuples, inclusive of `tmin`, exclusive of `tmax`.
- Every function accepting a `TrialSet` also accepts an `Evoked` (treated as one trial without `sem` unless given) and, when the extra `mne` is installed, `mne.Epochs`/`mne.Evoked` (converted through `evoparam.io.read_mne`).
- Every function returns a new object unless `inplace=True`; every returned object has a new `provenance.steps` entry.
- `n_jobs` uses `joblib` when installed, otherwise a thread pool; `random_state` accepts `int | numpy.random.Generator | None`.
- Functions that loop over channels (`fit`, `fit_trials`, `remove_artifact`, `test_response`, `TrialSet.average`, `TrialSet.apply`) accept `chunk: int | "auto" | None` (channels per chunk): `"auto"` applies the memory rule of §9.1, `None` processes all channels at once; results do not depend on the chunk size.
- `verbose` accepts `None` (inherit the package logger level), `bool`, or a logging level name.

### 5.1 `evoparam.io` — inputs, outputs, remote reads

**Purpose.** Build a `TrialSet` from any common container or file format, locally or remotely, with per-trial stimulation metadata harmonised into the standard columns of Table 4.1; write results back as portable files (`.evoparam`, BIDS derivatives, NWB).

**Methods.**
- Epoching: for event onsets `o_j` (s, absolute), window `(tmin, tmax)`, the trial `j` holds samples `n` with `round((o_j + tmin − t0)·f_s) ≤ n < round((o_j + tmax − t0)·f_s)`; `times = (n − n_onset)/f_s` so that the sample nearest the onset is `t = 0`. Trials whose window exceeds the recording are dropped and listed in `kept` (inherited `epoch_data`, [CODE §1.1]).
- Remote partial reads: HTTP `Range: bytes=a-b` requests on the binary data file, with the byte offsets computed from the header (`n_channels`, `dtype`, orientation) so that only the needed seconds are transferred; the inherited BrainVision reader already issues such `Range` requests [CODE §1.1] (untested over HTTP in the inherited suite [CODE §5]), and a 48 s read of a 2 GB ds004080 file worked in the skeleton's example run (§4.5). Fallback to a full download with a warning when the server answers `200` instead of `206`.
- Event harmonisation: a mapping table from source columns to the standard columns (BIDS-iEEG recommended names `electrical_stimulation_site`, `electrical_stimulation_current`, `electrical_stimulation_type`, plus the free-text `ElectricalStimulationParameters` sidecar field; the specification has no controlled vocabulary for pulse width, polarity or pulse count [LIT-ccep-29][LIT-ccep-66]) with unit parsing (`"4.0 mA"`, `"1000 µs"`, `"1ms"`), `n/a` → `NaN`, and a validation report listing missing fields per run [LIT-datasets-4].
- Dataset registry: a versioned JSON manifest shipped with the package (JSON because it is read by the standard library; no YAML dependency in the core) listing public datasets (id, title, modality, species, `sfreq`, `n_subjects`, size, licence, events status, loader entry point, known quirks), seeded with ds004080, ds003708, ds004457, ds004696, ds004977, ds006254, ds002094, ds005691, ds006519, ERP CORE, Localize-MI, bigP3BCI, Allen Visual Coding Neuropixels, F-TRACT (derivatives only) [LIT-datasets-1][LIT-datasets-8][LIT-datasets-9][LIT-datasets-13].

**Public API.**

```python
# evoparam/io/__init__.py
def read(source, *, format: str | None = None, **kwargs) -> TrialSet
    """Dispatch on type/extension: ndarray | mne objects | '.vhdr' | '.edf' | '.nwb' | BIDS directory | Neo objects."""

def from_array(data: np.ndarray, sfreq: float, *, times: np.ndarray | None = None,
               ch_names: list[str] | None = None, units: str = "uV",
               metadata: "pd.DataFrame | dict | None" = None, ch_info=None,
               species: str = "human", modality: str | None = None) -> TrialSet
    # data 2-D (n_channels, n_times) → one trial; 3-D (n_trials, n_channels, n_times). Raises DataError on shape mismatch.

def read_mne(obj: "mne.Epochs | mne.EpochsArray | mne.Evoked", *, units: str = "uV") -> TrialSet
    # function form of TrialSet.from_mne / Evoked.from_mne (§10.1, same conversion rules);
    # converts V → uV (×1e6) for EEG/ECoG/sEEG/DBS, T → fT for MEG magnetometers; keeps obj.metadata; ch_info from info['chs'] (type, coords m → mm, 'head' space).

def read_bids(root: "str | os.PathLike", subject: str, *, session: str | None = None, task: str | None = None,
              run: str | None = None, acquisition: str | None = None, datatype: str = "ieeg",
              epoch: tuple[float, float] = (-0.5, 1.0), events: "str | pd.DataFrame" = "auto",
              select: "dict | str | None" = None, picks=None, remote: "bool | str" = False,
              cache_dir: "str | None" = None, verbose=None) -> TrialSet
    # events="auto": read *_events.tsv, harmonise, keep trial_type == 'electrical_stimulation' rows (iEEG) or all rows (EEG/MEG);
    # select: pandas query string or dict applied to the harmonised events (e.g. {"stim_site": "PT01-PT02"});
    # remote=True or "openneuro": files resolved to https://s3.amazonaws.com/openneuro.org/<ds>/<path> and read by byte ranges.
    # Raises ReaderError (missing files), EventError (no matching events), DataError (channels.tsv/data mismatch).

def read_brainvision(vhdr: str, *, tmin: float | None = None, tmax: float | None = None, picks=None,
                     remote: bool | None = None) -> tuple[np.ndarray, float, list[str], float]
    # inherited; returns (data[n_ch, n_t] in uV, sfreq, ch_names, t0). remote=None → infer from 'http' prefix.

def read_edf(path: str, *, tmin=None, tmax=None, picks=None) -> tuple[np.ndarray, float, list[str], float]
    # extra "edf" (edfio); EDF+ annotations returned through read_events_edf(path).

def read_nwb(path: str, *, electrical_series: str | None = None, events: str | None = None,
             tmin: float = -0.5, tmax: float = 1.0, picks=None, stim_columns="auto") -> TrialSet
    # extra "nwb"; electrical_series=None → the only ElectricalSeries in acquisition, else error listing candidates;
    # event-source order and harmonisation in §10.3 (REQ-317).

def read_neo(source: "neo.Block | str", *, segment: int | None = None, signal: int | str = 0, events: str | None = None,
             tmin: float = -0.5, tmax: float = 1.0, io_class: str | None = None) -> TrialSet
    # extra "neo"; Block/Segment/AnalogSignal + Event objects; rules in §10.3 (REQ-319).

def epoch(data: np.ndarray, sfreq: float, onsets: np.ndarray, tmin: float, tmax: float, *, t0: float = 0.0,
          ch_names=None, units="uV", metadata=None, ch_info=None) -> TrialSet
    # inherited core; `kept` is stored in provenance.steps[-1]["kept"].

def harmonize_stim_events(events: "pd.DataFrame", sidecar: "dict | None" = None, *,
                          mapping: "dict[str, str] | None" = None) -> "tuple[pd.DataFrame, HarmonizationReport]"   # mapping and unit rules in §10.2
def validate_stimulation_events(events: "pd.DataFrame") -> "ValidationReport"   # one events table; dataclass: missing (list[str]), warnings, n_rows, n_sites, sites (list)
    # the dataset-level validator over a BIDS tree is io.bids.validate_stim_metadata (§10.2, REQ-314)
def select_stimulation_events(events, *, site=None, trial_type="electrical_stimulation", current=None) -> "pd.DataFrame"   # inherited

# remote
OPENNEURO_URL: str
def openneuro_url(dataset: str, path: str) -> str
def fetch_openneuro_file(dataset: str, path: str, folder: str = "data", *, sha256: str | None = None) -> str
class RemoteFile:   # context manager; .read(offset, nbytes); chunk cache (default 8 MiB, LRU 32 chunks); retries 4 with backoff 2,4,8,16 s
    def __init__(self, url: str, *, chunk_size: int = 8 << 20, timeout: float = 30.0): ...

# datasets registry
def list_datasets(*, modality=None, species=None, max_size_gb=None) -> "pd.DataFrame"
def dataset_info(dataset_id: str) -> dict

# writers
def write_bids_derivatives(results: "GroupFit | TrialFit | ResponseTest | ArtifactResult | list", bids_root, *,
                           subject: str, session=None, task=None, run=None, datatype="ieeg",
                           out: str | None = None, desc: str = "evoparam",
                           pipeline_name: str = "evoparam", overwrite: bool = False) -> list[str]   # layout and columns in §10.2 (REQ-315)
def write_nwb(obj: "TrialSet | GroupFit | TrialFit", path: str, *, mode: str = "a") -> None
```

**Edge cases.**
- Mixed channel units in one file (µV and mV): converted to `units` with a warning listing the channels.
- `sfreq` differing between runs of one subject: `read_bids` raises `DataError` unless `resample_to` is given (then `evoparam.prep.resample`).
- Events with `onset` outside the recording, or overlapping epochs (inter-event interval < `tmax − tmin`): kept, flagged `metadata["overlaps_previous"]=True`; the baseline-overlap check of §5.4 uses it.
- `events.tsv` without `electrical_stimulation_site` but with `trial_type` values like `"PT01-PT02"`: the harmoniser recognises the pair pattern `^[A-Za-z]+\d+-[A-Za-z]+\d+$` and fills `stim_site` with a warning.
- BrainVision `vectorized` orientation and `INT_16`/`IEEE_FLOAT_32` formats both supported; `resolution` per channel applied.
- Remote file where the server ignores `Range`: full download to `cache_dir` with a one-line warning including the size.
- NWB files with several `ElectricalSeries` or with `rate` versus `timestamps`: both handled; irregular timestamps raise `DataError`.
- Channel names duplicated in the file: suffixed `-1`, `-2` with a warning.

**Acceptance criteria.**
- REQ-020: `from_array`, `read_mne`, `read_bids` (local and remote), `read_edf`, `read_nwb`, `read_neo` each produce a `TrialSet` passing `TrialSet.validate()`; round trips `TrialSet → to_mne → read_mne` and `TrialSet → save → load` are equal to 1e-12 relative on `data` and exactly on metadata.
- REQ-021: remote `read_bids(..., remote=True)` of ten trials (−0.5 s, 1.0 s) from `ds004080 sub-ccepAgeUMCU01 run-021448` transfers less than 60 MB and completes in under 30 s on a 50 Mbit/s link; the transferred byte count is recorded in `provenance.source["bytes_read"]`.
- REQ-022: `harmonize_stim_events` on the `events.tsv` of ds004080 and of the four Mayo datasets fills `stim_site`, `stim_current_mA`, `stim_pulse_width_us` for 100 % of `electrical_stimulation` rows; `validate_stimulation_events` on ds006254 reports the missing fields instead of raising.
- REQ-023: the dataset registry lists at least the 14 datasets above with `license`, `size_gb`, `sfreq`, `events_status ∈ {present, absent, unknown}`; a CI test checks that every loader entry point resolves.
- REQ-024: `write_bids_derivatives` output passes `bids-validator` (derivatives mode) when the validator is installed in CI; `write_nwb` output re-reads with `pynwb` and passes `pynwb.validate`.
- REQ-025: the package imports and `from_array` works with only `numpy` and `scipy` installed (readers for mne/nwb/neo/edf raise `ImportError` naming the extra).

**Tests.** `tests/io/test_array.py` (shapes, units, invariants), `test_mne_roundtrip.py` (skipped without mne), `test_bids_local.py` (synthetic mini-BIDS dataset created by a fixture: 2 subjects, 3 channels, 20 events), `test_bids_remote.py` (local `http.server` subclass honouring `Range`; a second server ignoring it to test the fallback), `test_brainvision_formats.py` (int16/float32, multiplexed/vectorized), `test_events_harmonize.py` (unit strings, `n/a`, pair pattern, validator report), `test_nwb_roundtrip.py`, `test_neo.py`, `test_registry.py`, `test_writers.py` (bids-validator optional).

**Test map.** `tests/io/test_array.py` (REQ-020, REQ-025), `tests/io/test_mne_roundtrip.py` (REQ-020), `tests/io/test_nwb_roundtrip.py` (REQ-020), `tests/io/test_neo.py` (REQ-020), `tests/io/test_bids_remote.py` (REQ-021), `tests/io/test_events_harmonize.py` (REQ-022), `tests/io/test_registry.py` (REQ-023), `tests/io/test_writers.py` (REQ-024), `tests/test_core_only.py` (REQ-025).

**Tutorials.** "Loading data: arrays, MNE, BIDS (local and remote), NWB, Neo"; "Stimulation metadata: what the standard columns mean and how to fill them".

### 5.2 `evoparam.events` — stimulation events from the raw signal

**Purpose.** Recover pulse onsets when the events file is missing or wrong (ds006254 has 31 sEEG subjects and no confirmed `events.tsv` [LIT-datasets-5]), detect amplifier saturation, identify the stimulated contact pair from the artifact, and group pulses into trains.

**Methods.**
1. Robust difference detector (inherited, [CODE §1.1]). For channel `c` and sample `n`, `d_c[n] = |x_c[n] − x_c[n−1]|`; `d[n] = median_c d_c[n]` (over `picks`); `z[n] = (d[n] − median(d)) / (1.4826·MAD(d))`; an onset is the first `n` with `z[n] > θ` (`threshold_sd`, default 20) after a refractory period `min_interval`; the onset is refined to the sample of the maximum `d` within ±2 ms, minus one sample (the last pre-artifact sample), as in the skeleton. The median over channels suppresses channel-specific transients.
2. Template detector. From the first `K` (default 5) robust detections a template `T` (−1 ms to +3 ms around the onset, all channels, unit-normalised per channel) is formed; normalised cross-correlation `ρ[n] = ⟨T, X[n:n+L]⟩ / (‖T‖‖X[n:n+L]‖)` is scanned and onsets are peaks with `ρ > ρ_min` (default 0.8) respecting `min_interval` (matched-filter detection with missed-instance search, as used for sensing-enabled DBS devices [LIT-artifact-13]). Used by default as a second pass to recover pulses missed by the threshold and to reject detections that do not match the template (movement transients).
3. Rate-constrained detection. When `expected_rate` is given (e.g. 0.2, 1, 130 Hz), `min_interval = 0.5/expected_rate` by default and the inter-event intervals are checked; isolated intervals shorter than `0.5/expected_rate` or longer than `2/expected_rate` are reported in the table (`interval_flag`). For high-frequency trains the period estimate of §5.3 (`estimate_period`) refines the onsets to fractional samples.
4. Saturation. A channel is saturated at `n` when `|x_c[n]| ≥ rail_c` (default `rail_c = 0.999·max|x_c|` over the recording, or the ADC range when known from the header) or when `x_c` is constant over `≥ min_run` samples (default 3) inside ±50 ms of a pulse. The unrecoverable window of channel `c` is `[n_first, n_last + recovery·f_s]` (`recovery` default 5 ms, following the real-time local-polynomial practice of emitting zeros while railed and using a statistical test after rail exit [LIT-artifact-27]); it is written into `TrialSet.masks` and counted in the event table (`n_saturated_channels`). Saturation is a property of the channel at that pulse, never of the whole recording, because published methods handle saturation per channel but inconsistently (zeros while railed versus exclusion) [LIT-artifact-27], [LIT-artifact-6].
5. Stimulated-site identification. At each onset the per-channel artifact magnitude `a_c = max_{|t|≤1 ms} |x_c(t) − x_c(−2 ms)|` is ranked; for bipolar stimulation the two largest magnitudes are expected on the two stimulating contacts with opposite signs of the first deflection; `stim_site_guess = "A-B"` (ordered by channel index) when the two top channels have opposite polarity and, when `ch_info.group` is available, belong to the same shaft/grid; `site_confidence = (a_1 − a_3) / a_1` (margin to the third channel) multiplied by 1 if the polarity/shaft checks pass, 0.5 otherwise. Pulses are then clustered by site to produce the site table. Validated against `events.tsv` on datasets where it exists.
6. Train grouping. Sorted onsets are split into trains at gaps `> max_gap` (default `3/expected_rate` or 2 s); `train_id` and `index_in_train` are assigned; per-train site, pulse count and mean interval are summarised in `trains()`.
7. Matching to a reference: hits, misses, false alarms and signed errors for a tolerance (inherited `match_onsets`).

**Public API.**

```python
def detect_pulses(data: np.ndarray, sfreq: float, *, method: str = "threshold+template",
                  threshold_sd: float = 20.0, min_interval: float | None = 0.5, expected_rate: float | None = None,
                  t0: float = 0.0, picks=None, template: np.ndarray | None = None, rho_min: float = 0.8,
                  n_template: int = 5, refine: bool = True, return_scores: bool = False, verbose=None
                  ) -> "pd.DataFrame"   # EventTable columns of §4.3.4; with return_scores also returns (table, z, rho)
    # method ∈ {"threshold", "template", "threshold+template"}; raises EventError when no onset found (message gives max z and suggested threshold).

def detect_saturation(data: np.ndarray, sfreq: float, *, events: "pd.DataFrame | None" = None, rail: "float | np.ndarray | None" = None,
                      min_run: int = 3, recovery: float = 0.005, search: float = 0.05) -> tuple[np.ndarray, "pd.DataFrame"]
    # returns (mask[n_ch, n_t] bool, table with channel, start_time, end_time, n_samples, event_index)

def identify_stim_site(data: np.ndarray, sfreq: float, events: "pd.DataFrame", *, ch_names: list[str], ch_info=None,
                       window: float = 0.001, t0: float = 0.0) -> "pd.DataFrame"   # adds stim_site_guess, site_confidence, polarity_check
def group_trains(events: "pd.DataFrame", *, max_gap: float | None = None, expected_rate: float | None = None) -> "pd.DataFrame"
def trains(events: "pd.DataFrame") -> "pd.DataFrame"   # one row per train: train_id, site, n_pulses, onset, mean_interval, current (if known)
def match_onsets(detected: np.ndarray, reference: np.ndarray, tolerance: float = 0.01) -> dict   # inherited
def events_to_metadata(events: "pd.DataFrame", *, site_column: str = "stim_site_guess") -> "pd.DataFrame"   # standard columns for TrialSet.metadata
def detect_events(ts_or_raw, ...) -> "pd.DataFrame"   # convenience wrapper exposed at package level as evoparam.detect_events
```

**Edge cases.** No pulses (`EventError` with diagnostics); pulses at the first/last 2 ms of the recording (kept, `refine` restricted to available samples); 130 Hz DBS trains at 250 Hz sampling (threshold on `d` still works; `min_interval` from `expected_rate`; site identification disabled with a warning because the inter-pulse interval is shorter than `window`); channels with different gains (the per-channel median removes gain effects because `d` is a median across channels, but `identify_stim_site` normalises `a_c` by the channel's baseline MAD); line-noise spikes and movement transients (rejected by the template pass); polarity-alternating protocols (template built per polarity class, two templates, flagged in the table as `template_class`); recordings where the stimulating contacts are not recorded (site guess returns `""` with `site_confidence = 0`).

**Acceptance criteria.**
- REQ-040: on ds004080, across at least 5 subjects and 20 runs with `events.tsv` as reference, `detect_pulses` reaches recall ≥ 0.99, precision ≥ 0.99 and median absolute onset error ≤ 1 sample (0.49 ms at 2048 Hz); the skeleton's single-run result (10/10, 0 false alarms, 2.0 ms mean error) is the floor.
- REQ-041: on synthetic recordings (100 channels, 2048 Hz, 200 pulses with artifact amplitude ≥ 10× baseline SD, with added 50 Hz line noise and 20 movement transients of 5× SD), recall and precision ≥ 0.99 and no transient is detected as a pulse when the template pass is on.
- REQ-042: `identify_stim_site` agrees with `events.tsv` on ≥ 95 % of pulses on ds004080 and on the Mayo sets; the confidence is calibrated (accuracy ≥ 0.98 for `site_confidence ≥ 0.5`).
- REQ-043: `detect_saturation` finds 100 % of synthetic clipping runs ≥ 3 samples and ≥ 95 % of 2-sample runs, with ≤ 1 false positive per 10 minutes of clean data; the resulting mask reaches the fitter (§5.5) so that masked samples have zero weight.
- REQ-044: `group_trains` reproduces the train structure of the Mayo protocol (≈ 0.2 Hz, 10–12 pulses per site) and of a synthetic 1 Hz × 30 pulses × 20 sites design exactly.
- REQ-045: runtime ≤ 1 s per 10 minutes of 100-channel data at 2048 Hz on one core; memory ≤ 2× the data size.

**Tests.** `tests/events/test_threshold.py` (inherited), `test_template.py`, `test_rate.py`, `test_saturation.py`, `test_stim_site.py` (synthetic bipolar artifact with known pair; shaft constraint), `test_trains.py`, `test_match.py`, `test_ds004080_events.py` (marked `remote`, runs nightly).

**Test map.** `tests/events/test_ds004080_events.py` (REQ-040, REQ-042), `tests/events/test_template.py` (REQ-041), `tests/events/test_rate.py` (REQ-041), `tests/events/test_stim_site.py` (REQ-042), `tests/events/test_saturation.py` (REQ-043), `tests/events/test_trains.py` (REQ-044), `benchmarks/bench_events.py` (REQ-045).

**Tutorials.** "Recover stimulation events from the raw signal (ds006254)"; "Check your events file against the signal".

### 5.3 `evoparam.artifact` — stimulation-artifact removal with a quality score

**Purpose.** Suppress the stimulus artifact before quantification without destroying the early response, across SPES/CCEP, DBS, TMS, microstimulation, optogenetics and ECAP paradigms; mark replaced samples; score each trial.

**Methods** (each implemented from the published equations; no code is taken from GPL or unlicensed repositories, see §13). Notation: trial `k`, channel `c`, pulse onset at `t = 0`, artifact window `W_a = [a, b]`.
1. `interpolate` (inherited [CODE §1.1]). Replace `W_a` by linear, cubic or PCHIP interpolation between the means of `buffer` samples on each side; mask set. Equation for linear: `x̂(t) = x(a) + (x(b) − x(a))·(t − a)/(b − a)` [LIT-artifact-1]. Default for scalp TMS-EEG and the first pass of every other method.
2. `template`. Artifact template per group `g` (default group = `(stim_site, stim_current_mA, stim_polarity)`): `T_g(t) = median_{k∈g} x_k(t)` on `W_a` (median rather than mean to resist outliers); per-trial least-squares scaling `α_k = ⟨x_k, T_g⟩_{W_a} / ⟨T_g, T_g⟩_{W_a}`; `x̂_k = x_k − α_k T_g` on `W_a`. Option `template="local"`: `T` from the `n_neighbors` nearest pulses in time (moving template) to follow drift [LIT-artifact-2][LIT-artifact-13]. Response protection: `W_a` must end before the earliest response of interest (default `b = 8 ms` for SPES at ≥ 1 kHz); the method reports `template_response_correlation`, the correlation between `T_g` and the mean post-window response, and warns above 0.5.
3. `tail`. After blanking the pulse, the recovery tail is fitted on `[b, b + tail_length]` as `r(t) = c_0 + c_1 e^{−(t−b)/τ_1} + c_2 e^{−(t−b)/τ_2}` with `τ` bounds (default 0.5–50 ms, `τ_2 ≥ 3 τ_1`) and subtracted; or, with `tail="joint"`, handed to §5.5 as the `exponential` background so that tail and components are fitted together (the collinearity of an exponential tail with an early broad component over the first time constants is the reason for the joint option [LIT-fitting-15], [LIT-fitting-34]).
4. `period_template` (period-based reconstruction, from the open-access description [LIT-artifact-1]; patent application US 2022/0338786 A1, Brown University, pending at the time of reading [LIT-artifact-4] — see §13). Stimulation period `T` in fractional samples is estimated by a penalised grid search minimising the MSE of a fit that repeats with period `T`; for sample `t` the artifact estimate is the mean of samples `s` with `N_skip < |s − t| ≤ N_bins` and `(|s − t| mod T) ≤ D_period` or `≥ T − D_period`; `x̂_t = x_t − mean(B_t)`. Linear FIR; one-sided variant for `evoparam.stream`. Parameters `n_bins`, `n_skip`, `d_period` with the defaults and the selection GUI logic replaced by an automatic rule: `d_period = 0.01·T`, `n_skip = round(0.05·f_s)`, `n_bins` = the smallest value such that `|B_t| ≥ 50`.
5. `lowrank` (sequential low-rank regression across channels, pulses and trials, from the published method [LIT-artifact-6]). Data unfolded to `(R·T·P) × C`; the top `K_C` principal components computed from all channels except `c` and its neighbours are regressed out of channel `c`; then across pulses (`K_P`) and across trials (`K_R`); a final pass on a post-train window removes the slow transient. Defaults `(K_C, K_P, K_R) = (4, 2, 4)`; requires `≥ 8` channels, `≥ 4` pulses per train and `≥ 4` trials, otherwise `ArtifactError`. Warning: time-locked responses that are low-rank across trials can be attenuated; the method reports the fraction of response-window variance removed.
6. `subspace` (pre-whitening and null projection, from the published method [LIT-artifact-14]). With baseline covariance `Σ_B` (from `baseline`), `X_w = Σ_B^{−1/2}(X − μ)`, artifact directions `U_a` = top `k` left singular vectors of `X_w` restricted to `W_a` (`k` chosen so that the removed energy ≥ 95 % of the artifact-window energy, max 6), `X̂ = Σ_B^{1/2}(I − U_a U_aᵀ)X_w + μ`.
7. `polynomial` (local cubic fit and subtraction, from the published method [LIT-artifact-27]). At every `t`, a cubic polynomial is least-squares fitted on `[t − τ, t + τ]` and `p_t(t)` is subtracted; recursive normal-equation updates make it `O(1)` per sample; forward-centred windows while railed (`depegging`), with zeros emitted while saturated and a statistical test over `0.5 τ` deciding when the estimate is trustworthy. Default `τ = 3 ms`.
8. `matching_pursuit` (per-pulse sequential decomposition: line noise, evoked potential, artifact atoms; published for single-pulse SEEG [LIT-artifact-5]; Washington University pre-grant publication US 2023/0240618 A1 [LIT-artifact-25] — see §13). Specified only from the public abstract and claim summary: dictionary = {sinusoids at the line frequency and harmonics, Gabor atoms for the evoked potential, exponential-decay atoms for the artifact}; greedy residual decomposition; the artifact atoms are subtracted. Shipped as `method="matching_pursuit"` with a `warnings.warn` at first use pointing to the patent note; exact benchmark numbers of the original are not reproduced because the full text was not accessible [LIT-artifact-5].
9. `paradigm`. Alternating polarity: pairs of trials with opposite `stim_polarity` are averaged, `x̂ = (x_cathodic + x_anodic)/2`, under the assumption that the artifact inverts and the response does not [LIT-artifact-36]; forward masking (masker–probe): `x̂ = A − B + (C − D)` with the four conditions given in `metadata["paradigm_condition"]` [LIT-artifact-37]. Pairing is automatic by trial order within a site, or explicit through `metadata["pair_id"]`.
10. `learned`. For each group, an artifact basis `U_g` (PCA on the artifact windows of all pulses in the group, `k` such that explained variance ≥ 95 %, max 8) is regressed out of each pulse on `W_a` with ridge penalty `λ`; the basis is stored in the result for reuse (and for the joint model of §5.5, background `"artifact_basis"`).
11. Quality score per trial (always computed): `q_k = 1 − clip(mean(p_1, …, p_4), 0, 1)` with penalties `p_1 = min(RMS(x̂_k on [b, b + 10 ms]) / RMS(baseline_k), 1)·0.5` (residual energy), `p_2 = min(|Δx̂_k| at the window edges / (3·MAD_baseline), 1)` (edge discontinuity), `p_3 = saturation fraction in `[a, b + 20 ms]`, `p_4 = 1 − r²` of the template fit (0 for methods without a template); `q_k` is written to `metadata["quality"]` and the components to `ArtifactResult.quality_components` (this specification's choice: no surveyed artifact-removal tool scores trials or records replaced samples in a machine-readable form [LIT-artifact-63]). The downstream rejection step (§5.4) uses `quality_min`.
12. Masks. Every replaced or saturated sample is `True` in `cleaned.masks`; fitters give such samples zero weight and widen the reported uncertainty of any component whose onset lies inside a masked interval (`Component.ci` inflated by the masked fraction of its support, documented in §5.5), because early-latency windows are where cleaning pipelines disagree most [LIT-artifact-7].

**Public API.**

```python
def remove_artifact(ts: TrialSet, *, method: str = "template", window: tuple[float, float] = (-0.002, 0.008),
                    buffer: int = 3, interpolation: str = "linear",
                    template: "str | np.ndarray" = "global", group_by: tuple[str, ...] = ("stim_site", "stim_current_mA", "stim_polarity"),
                    n_neighbors: int = 10, scale: bool = True,
                    tail: str = "none", tail_length: float = 0.05, tau_bounds: tuple[float, float] = (0.0005, 0.05),
                    period: float | None = None, expected_rate: float | None = None, n_bins: int | None = None, n_skip: int | None = None, d_period: float | None = None,
                    ranks: tuple[int, int, int] = (4, 2, 4), post_window: float = 0.03,
                    baseline: tuple[float, float] = (-0.5, -0.05), n_components: int | None = None,
                    poly_halfwidth: float = 0.003,
                    scheme: str | None = None, pairing: str = "auto",
                    quality: bool = True, inplace: bool = False, random_state=None, verbose=None) -> ArtifactResult
    # method ∈ {"interpolate", "template", "tail", "period_template", "lowrank", "subspace", "polynomial", "matching_pursuit", "paradigm", "learned", "none"}
    # Raises ArtifactError: window outside the epoch; template with < 3 pulses in a group (falls back to interpolate with a warning when fallback=True);
    #   lowrank with insufficient channels/pulses/trials; period_template without period and expected_rate; paradigm without the needed metadata.

def estimate_artifact_template(ts: TrialSet, window=(-0.002, 0.008), *, group_by=(...), statistic: str = "median") -> dict[tuple, np.ndarray]
def estimate_period(x: np.ndarray, sfreq: float, f_guess: float, *, search: float = 0.02, n_stages: int = 3) -> float
def artifact_quality(ts: TrialSet, *, window, baseline, template=None) -> "pd.DataFrame"
def paradigm_subtract(ts: TrialSet, *, scheme: str, pairing: str = "auto") -> TrialSet
def apply_mask(ts: TrialSet, mask: np.ndarray) -> TrialSet
def compare_methods(ts: TrialSet, methods: list[dict], *, truth: TrialSet | None = None, window_eval=(0.008, 0.05)) -> "pd.DataFrame"
    # runs several settings and reports RRMSE/SIR against truth (synthetic) or residual/quality statistics (real data)
```

**Edge cases.** Artifact window longer than the inter-pulse interval (DBS trains): `template`/`interpolate` refuse with a hint to `period_template`; a response that starts inside `W_a` (ECAP < 2 ms, ERNA peak ≈ 4 ms [LIT-artifact-5][LIT-artifact-17]): the mask records the loss and `window` presets for these modalities are short (`(−0.0005, 0.0015)` for ECAP) with `polynomial` or `period_template` methods; trains with changing current (template groups by current, falls back to `local`); channel-specific saturation (mask per channel; the template of a saturated pulse is excluded); recordings at 250 Hz (Percept) where the artifact aliases: `period_template` with the fractional period, otherwise `ArtifactError` explaining that the artifact is not resolvable [LIT-artifact-13]; optogenetic light-locked artifacts (template on light onsets; no polarity check); polarity-alternating designs with unequal pair counts (unpaired trials kept, flagged `paired=False`).

**Acceptance criteria** (benchmark of §7.2 and of §8 task T4: measured artifact templates overlaid on artifact-free epochs that contain simulated responses with known parameters; sampling 1 kHz, 2048 Hz, 4800 Hz; response onsets 5, 8, 12, 20 ms; 10–50 trials).
- REQ-060: for every method, replaced samples are exactly the masked samples and no sample outside the declared window changes, except for `period_template`, `lowrank`, `subspace` and `polynomial` whose global effect on the baseline window is ≤ 2 % RMS.
- REQ-061: `template` and `learned` reduce the artifact-window residual by ≥ 20 dB relative to `interpolate` leaving the window untouched, for template SNR ≥ 10; latency bias of a component with onset ≥ `b` is ≤ 0.5 ms and amplitude bias ≤ 5 % after cleaning (versus ≥ 20 % amplitude bias for plain blanking when the onset is within 2 ms of `b`).
- REQ-062: `period_template` on a synthetic 130 Hz train at 1 kHz sampling with a 10 Hz and a 50 Hz sinusoid injected removes the artifact so that the residual error is statistically indistinguishable from the pre-stimulation noise (Wilcoxon p > 0.05, as in the original validation [LIT-artifact-1]); the fractional period estimate has SD ≤ 1e-4 samples across 100 realisations.
- REQ-063: `lowrank` on a synthetic 24-channel, 20-pulse, 10-trial array recording recovers evoked spikes with waveform correlation ≥ 0.95 to the clean waveform and silent-channel RMS within 10 % of pre-stimulation RMS [LIT-artifact-6].
- REQ-064: `subspace` reaches ≥ 30 dB signal-to-interference improvement on synthetic EEG with real artifact templates, matching the published range (32–34 dB) [LIT-artifact-14].
- REQ-065: `polynomial` recovers the signal within 2 ms after a 600 mV pulse at 25 kHz on synthetic MEA data and emits zeros while railed [LIT-artifact-27].
- REQ-066: the quality score separates clean from corrupted synthetic trials (10 % of trials with residual artifact, saturation, or wrong-site template) with AUC ≥ 0.9.
- REQ-067: `paradigm` with alternating polarity cancels an artifact that exactly inverts to ≤ 1 % residual and leaves a non-inverting response unchanged to 1e-9; forward masking reproduces the `A − B + C − D` identity on synthetic conditions.
- REQ-068: every method runs in ≤ 2 s for 50 trials × 128 channels × 3 s at 2048 Hz on one core, except `lowrank` and `subspace` (≤ 20 s).
- REQ-069: results carry `method`, all parameters, `window`, the template or basis used, and the masks, so that `evoparam.reproduce` re-creates the cleaned data to 1e-12.

**Tests.** One file per method under `tests/artifact/`, each with: synthetic-overlay accuracy, mask correctness, parameter validation errors, and provenance; `test_quality.py`; `test_paradigm.py`; `test_period.py` (fractional period recovery); `test_compare_methods.py`.

**Test map.** `tests/artifact/test_masks.py` (REQ-060), `tests/artifact/test_template.py` (REQ-061), `tests/artifact/test_learned.py` (REQ-061), `tests/artifact/test_period.py` (REQ-062), `tests/artifact/test_lowrank.py` (REQ-063), `tests/artifact/test_subspace.py` (REQ-064), `tests/artifact/test_polynomial.py` (REQ-065), `tests/artifact/test_quality.py` (REQ-066), `tests/artifact/test_paradigm.py` (REQ-067), `benchmarks/bench_artifact.py` (REQ-068), `tests/artifact/test_provenance.py` (REQ-069).

**Tutorials.** "Removing the stimulation artifact in CCEP data (ds004080)"; "Repeating artifacts in DBS recordings (simulated 130 Hz)"; "Artifact ground truth: build your own overlay benchmark from a saline recording".

### 5.4 `evoparam.prep` — light preparation

**Purpose.** The minimum preparation that changes evoked-response numbers and is therefore specified here (baseline, reference, trial rejection, resampling, averaging); everything else (filtering, ICA, source analysis) is delegated to MNE-Python.

**Methods.**
1. Baseline correction. `x̂_k(t) = x_k(t) − m_k`, `m_k = mean` (or `median`, or a linear trend removed) of `x_k` over `baseline`; applied per trial and channel. Baseline validity checks: (a) the baseline must not intersect any artifact mask; (b) when `event_times` and the previous pulse are known, the baseline must start at least `delayed_margin` (default 1.0 s) after the previous pulse, because delayed responses extend to 1 s [LIT-ccep-19] and the published CCSR baseline (−500 to −100 ms) starts only 500 ms after the previous pulse at 1 Hz stimulation, inside that window [LIT-tf-2]; violations produce a warning with the shortest inter-pulse interval and a suggested baseline.
2. Referencing. `bipolar`: for each `group` (shaft/grid row) in `ch_info`, consecutive contacts `A_i − A_{i+1}` named `"A1-A2"`; `car`: mean of all channels not `bad`, not `stim_contact`, not saturated at that pulse; `car_adaptive` (adjusted common average by least anticorrelation, re-implemented from the published method [LIT-ccep-18]): channels are ranked by increasing mean cross-trial covariance in the response window (default 10–300 ms); common averages `CA_n` of the `n = 2..N` least responsive channels are formed; for each `n`, `ζ_n` = mean over channels of the Fisher-z Pearson correlation between the original channel and its re-referenced version; the chosen `n*` is the global maximum of `−ζ_n` (most negative mean correlation) or the first local maximum that is significant under a bootstrap (`n_boot` default 100, 95 % CI); the result records `n*`, `ζ`, the excluded channels and the CI. `laplacian`: for grids with coordinates, the mean of the nearest neighbours within `radius`. `monopolar` keeps the recording reference and records it. Stimulating contacts are excluded from every common average (they are `stim_contact` per trial through `metadata["stim_site"]`).
3. Reference-sensitivity report. `reference_sensitivity(ts, schemes=("bipolar","car","car_adaptive"), detector=...)` runs the existence test (§5.6) under each scheme and reports pairwise Cohen's κ of the detections and the per-channel latency differences, because montage alone changed N1 latency from 47 to 71 ms and PPV/NPV in a paediatric cohort [LIT-ccep-27] and a contaminated reference produced spurious deflections on most channels [LIT-ccep-45], and the adjusted common average exists only in MATLAB [LIT-ccep-18].
4. Trial rejection. Per trial and channel, robust z-scores of peak-to-peak amplitude and RMS in the response window and in the baseline (`z = (v − median)/(1.4826·MAD)` over trials); `rejected = True` with `reject_reason` when `z > ptp_z` (default 5), when the channel is flat (`MAD < flat_threshold`), when `metadata["quality"] < quality_min` (default 0.5), or when the saturation mask covers > `max_masked_fraction` (default 0.2) of the response window; the fraction rejected per channel is capped at `max_fraction` (default 0.3; above it a warning says the channel, not the trials, is the problem). Trials are never deleted; `TrialSet.select(query="not rejected")` filters.
5. Resampling. `scipy.signal.resample_poly` with an anti-aliasing FIR; `times`, `event_times`, masks (nearest) and metadata are updated; the ratio is recorded.
6. Averaging. `mean`, `trimmed` (20 % trimmed mean), `median`, `weighted` (weights `= quality`); `sem` from the un-trimmed trials; `n_trials` per channel after rejection.
7. Filters (thin wrappers over `scipy.signal`, zero-phase `sosfiltfilt`): `highpass`, `lowpass`, `notch`; each warns when applied to data whose masks are not yet filled (ringing around blanked samples) and records the filter in provenance. Recommended order: artifact → filter → baseline → reference.

**Public API.**

```python
def baseline_correct(ts: TrialSet, baseline: tuple[float, float] = (-0.5, -0.05), *, mode: str = "mean", check: bool = True, delayed_margin: float = 1.0) -> TrialSet
def rereference(ts: TrialSet, scheme: str = "bipolar", *, exclude: "list[str] | None" = None, groups: "str | dict | None" = "group",
                response_window: tuple[float, float] = (0.01, 0.3), n_boot: int = 100, radius: float = 12.0, random_state=None) -> TrialSet
    # scheme ∈ {"monopolar", "bipolar", "car", "car_adaptive", "laplacian"}; result.ch_info["reference"] updated; raises DataError when groups are missing for bipolar.
def reference_sensitivity(ts: TrialSet, *, schemes=("bipolar", "car", "car_adaptive"), test_kwargs: dict | None = None) -> "pd.DataFrame"
def reject_trials(ts: TrialSet, *, response_window=(0.01, 0.5), baseline=(-0.5, -0.05), ptp_z: float = 5.0, rms_z: float = 5.0,
                  flat_threshold: float = 1e-3, quality_min: float = 0.5, max_masked_fraction: float = 0.2, max_fraction: float = 0.3) -> TrialSet
def resample(ts: TrialSet, sfreq: float) -> TrialSet
def average(ts: TrialSet, *, by=None, method: str = "mean", trim: float = 0.2, use_rejected: bool = False) -> Evoked
def highpass(ts, cutoff: float, order: int = 4) -> TrialSet ; def lowpass(ts, cutoff, order=4) -> TrialSet ; def notch(ts, freq: float = 60.0, q: float = 30.0, harmonics: int = 3) -> TrialSet
```

**Edge cases.** sEEG shafts with missing contact numbers (bipolar pairs only between consecutive existing contacts, gaps reported); grids without coordinates (`laplacian` raises `DataError`); `car_adaptive` with fewer than 8 channels (falls back to `car` with a warning, as the bootstrap is meaningless); baseline window partially masked (the mask is respected; if < 50 % of baseline samples remain, `DataError`); all trials of a channel rejected (channel flagged `bad`, fits skipped with `status="skipped"`); white-matter contacts (kept; `ch_info.tissue` is reported, never used to drop channels silently, because white-matter contacts also produce responses [LIT-tf-7]).

**Acceptance criteria.**
- REQ-080: baseline check detects a baseline overlapping the previous pulse's delayed window in a synthetic 1 Hz design and proposes a valid alternative; no false warning for 0.2 Hz designs.
- REQ-081: `car_adaptive` reproduces the published simulation outcome on a synthetic 50-channel, 12-trial dataset with 0–90 % responsive channels: median false negatives 0 up to 84 % responsive channels and median false positives ≤ 2.5 (tolerance ±1) [LIT-ccep-18]; on ds004977 it yields a lower cross-channel `R²` than `car` (one-sided Wilcoxon p < 0.01 over channels) [LIT-ccep-18].
- REQ-082: bipolar naming and ordering match `mne.set_bipolar_reference` on the same montage; the reference is recorded in `ch_info.reference` for every channel.
- REQ-083: `reject_trials` flags ≥ 95 % of synthetic corrupted trials (amplitude 10× baseline, flat, or 30 % masked) with ≤ 2 % false rejections, and never deletes rows.
- REQ-084: `reference_sensitivity` produces the κ matrix and latency differences for three schemes on ds004080 sub-01 in < 2 min.
- REQ-085: `resample` preserves component latencies within 1/(2·sfreq_new) on simulated data and keeps `event_times` exact.

**Tests.** `tests/prep/test_baseline.py`, `test_reference.py` (bipolar vs mne, car exclusions, adaptive simulation), `test_reject.py`, `test_resample.py`, `test_average.py`, `test_filters.py`.

**Test map.** `tests/prep/test_baseline.py` (REQ-080), `tests/prep/test_reference.py` (REQ-081, REQ-082), `tests/prep/test_reject.py` (REQ-083), `tests/prep/test_reference_sensitivity.py` (REQ-084), `tests/prep/test_resample.py` (REQ-085).

**Tutorials.** "Re-referencing intracranial data for evoked responses: bipolar, common average, adaptive"; "Checking that your results do not depend on the montage".
### 5.5 `evoparam.shapes` and `evoparam.models` — component shapes, backgrounds and the fitting engine

**Purpose.** Describe an evoked waveform `y(t)` on a window `W = [t_0, t_1]` as a sum of a small number of named components plus a slow background, with every number traceable to an equation, a goodness-of-fit record and an uncertainty. This is the module that replaces the inherited extremum-and-threshold loop [CODE §2.2], which found the CCEP N1 on 2 of 46 channels of the example run (§1, §4.5): the three rules that deleted it (`fit.py:1206-1208`, `1061-1071`, `1268-1277` [CODE §0]) are not carried over, and no heuristic deletes a component after it has been fitted. Components are kept or dropped only by model comparison (§5.5.4).

The model, common to every modality, is

```
y(t) = b(t; β_b) + Σ_{k=1..K} A_k · s_k(t; θ_k) + ε(t),      t ∈ W
```

where `s_k` is a unit-amplitude shape from the registry of §5.5.1 (`max_t |s_k| = 1`, so `A_k` is the signed amplitude in data units), `θ_k` are the shape's non-linear parameters (latency, width, asymmetry, frequency…), `b` is a background from §5.5.2, and `ε` is coloured noise whose scale and lag-1 autocorrelation are estimated on the baseline (§5.5.3, step 1). Polarity is a property of `A_k` (negative or positive), never a filter on candidates: the same model fits an N1 and a P1.

#### 5.5.1 Component shapes (`evoparam.shapes`)

Every shape is a frozen dataclass implementing the `ComponentShape` protocol of §4.3.4: `evaluate(t, θ)`, `jacobian(t, θ)` (analytic for every shape below; checked against finite differences in CI, REQ-101), `default_bounds(window, sfreq, data_scale)`, `initial_guess(candidate)`, `features(t, θ)`, plus `n_params`, `param_names`, `latency_param` (the name of the parameter the ordering constraint of §5.5.3 acts on). All shapes are normalised so that `max_t |s(t)| = 1` on the real line (or numerically on the evaluation grid when no closed form exists); the amplitude `A_k` is a separate linear parameter solved by variable projection (§5.5.3, step 2). Shapes that are causal (zero before an onset `t_0`) are listed as such; they are the right choice for responses that must start after the stimulus.

| key | equation (unit amplitude) | non-linear θ | default bounds | latency (reported) | causal | typical use | refs |
|---|---|---|---|---|---|---|---|
| `gaussian` | `s(t) = exp(−(t−μ)²/(2σ²))` | `μ, σ` | `μ ∈ [t_0 − σ_max/2, t_1 + σ_max/2]`; `σ ∈ [min_fwhm, max_fwhm]/2.3548` | `μ` | no | symmetric ERP components; first pass for any modality | [LIT-fitting-5], [LIT-fitting-14], [CODE §2.2] |
| `skewed_gaussian` | `s(t) = g(z)·(1 + erf(α z/√2)) / c(α)`, `z = (t−ξ)/ω`, `g(z) = exp(−z²/2)`, `c(α)` the numerical maximum of the numerator | `ξ, ω, α` | `ω` as `σ`; `α ∈ [−10, 10]` | mode of `s` (numerical, Brent on the fitted curve) | no | CCEP N1/N2, TEP N100, any fast-rise slow-decay deflection | [LIT-fitting-14] |
| `exgaussian` | `s(t) = h(t)/max h`, `h(t) = (1/2τ)·exp((σ²/(2τ²)) − (t−μ)/τ)·erfc((σ/τ − (t−μ)/σ)/√2)` | `μ, σ, τ` | `τ ∈ [1/sfreq, max_fwhm]` | mode of `s` (numerical) | no | right-skewed components, latency distributions (§5.7) | [LIT-fitting-14], [LIT-fitting-27] |
| `gamma` | `s(t) = ((t−t_0)/((κ−1)θ))^{κ−1}·exp(κ − 1 − (t−t_0)/θ)` for `t ≥ t_0`, 0 before (peak at `t_0 + (κ−1)θ`, value 1) | `t_0, κ, θ` | `κ ∈ [1.5, 20]`; `θ` such that FWHM in `[min_fwhm, max_fwhm]` | `t_0 + (κ−1)θ` | yes | animal field potentials, synaptic-like responses; `κ = 2` is the alpha function | textbook definition (this specification's choice) |
| `biexponential` | `s(t) = [exp(−(t−t_0)/τ_d) − exp(−(t−t_0)/τ_r)] / m` for `t ≥ t_0`, `m` the peak value | `t_0, τ_r, τ_d` | `τ_r ≥ 1/sfreq`; `τ_d ≥ 1.5 τ_r`; `τ_d ≤ |W|` | `t_0 + τ_r τ_d/(τ_d − τ_r) · ln(τ_d/τ_r)` | yes | ECAP, compound responses with sharp onset; artifact-like tails when used as a component on purpose | textbook definition (this specification's choice) |
| `damped_oscillation` | `s(t) = exp(−(t−t_0)/τ)·cos(2πf(t−t_0) + φ)` for `t ≥ t_0` | `t_0, τ, f, φ` | `f ∈ [f_min, f_max]` (preset; ERNA 200–450 Hz); `τ ∈ [f⁻¹, |W|]` | `t_0` (onset) plus `peak_times` (all local extrema) | yes | ERNA, ringing after DBS pulses, resonant responses | [LIT-fitting-4], [LIT-fitting-24], [LIT-fitting-26] |
| `gabor` | `s(t) = exp(−(t−t_0)²/(2σ²))·cos(2πf(t−t_0) + φ)` | `t_0, σ, f, φ` | as `gaussian` + `f` bounds | `t_0` (envelope centre) | no | oscillatory bursts in TEPs and auditory steady-state windows | [LIT-fitting-23] |
| `template` | `s(t) = T((t − δ)/γ) / max|T|`, `T` a `ShapeBasis` waveform (data-driven or user-supplied), shift `δ`, optional stretch `γ` | `δ` (+ `γ` if `stretch=True`) | `δ ∈ ±max_shift`; `γ ∈ [0.8, 1.25]` | extremum of the shifted template | as `T` | reuse of a shape learned by §5.9 (shared shapes) or a canonical template; the trial model of §5.7 | [LIT-fitting-1], [LIT-fitting-2], [LIT-erpmeas-7] |

Notes.
- The `refs` column cites the primary references of the shape findings of the fitting-methods review; rows marked "textbook definition" have no single published source and are this specification's choice; the shape equations are textbook definitions and carry no licence.
- `features(t, θ)` are always computed on the fitted component curve `A_k s_k(t; θ_k)` evaluated on a grid ten times finer than `1/sfreq`, never on the raw data: `latency` (time of the signed extremum of the component), `amplitude` (`A_k`, signed), `polarity` (`sign(A_k)`), `fwhm` (full width at half the extremum, measured on the component curve), `onset`/`offset` (first/last time at which `|A_k s_k| ≥ onset_fraction·|A_k|`, default `onset_fraction = 0.10`, the "fractional peak" definition of [LIT-erpmeas-3]), `area` (signed trapezoidal integral between onset and offset), `rise_time` (`latency − onset`), `decay_time` (`offset − latency`), `symmetry` (`(decay_time − rise_time)/(decay_time + rise_time)`, 0 for a symmetric shape), `sharpness` (`|s''(latency)|·fwhm²`, dimensionless curvature at the peak). A latency may fall outside `W` (a component truncated by the window edge); it is then reported as is and the component carries the warning `"latency_outside_window"` instead of being deleted (this is one of the three rules that lost the N1 in the inherited code [CODE §0]).
- Shapes are registered by name in `evoparam.shapes.registry`; `register_shape(cls)` adds a user shape (REQ-102) after checking the protocol and the Jacobian numerically.

#### 5.5.2 Backgrounds (`evoparam.shapes.backgrounds`)

| key | equation | parameters | linear? | when | refs |
|---|---|---|---|---|---|
| `none` | `b = 0` | — | — | baseline-corrected scalp ERP windows | |
| `constant` | `b = c_0` | `c_0` | yes | default when a baseline correction has been applied | |
| `linear` | `b = c_0 + c_1 (t − t_0)` | `c_0, c_1` | yes | slow drift inside the window | |
| `exponential` | `b = c_0 + c_1·exp(−(t − t_0)/τ)` | `c_0, c_1` (linear), `τ` (non-linear, bounds `[0.5 ms, |W|]`) | partly | the recovery tail after a stimulation pulse (default for `ccep*`, `erna`, `ecap` presets); the joint option of §5.3 method 3 | [LIT-fitting-15], [LIT-fitting-16], [LIT-artifact-1] |
| `double_exponential` | `b = c_0 + c_1 e^{−(t−t_0)/τ_1} + c_2 e^{−(t−t_0)/τ_2}`, `τ_2 ≥ 3 τ_1` | `c_0..c_2`; `τ_1, τ_2` | partly | electrode polarisation plus amplifier recovery | [LIT-artifact-1] |
| `pspline` | `b = Σ_j c_j B_j(t)` cubic B-splines with knot spacing `≥ 2·max_fwhm` and a second-difference penalty `λ` chosen by generalised cross-validation on the baseline segment | `c_j` | yes | long windows with structured slow activity (delayed responses, animal recordings) | standard P-spline (this specification's choice) |
| `artifact_basis` | `b = Σ_j c_j u_j(t)`, `u_j` the artifact basis learned by §5.3 method 10 (`ArtifactResult.basis`) | `c_j` | yes | when artifact removal and fitting must share the model so that the artifact is not counted as a component | §5.3 |

The knot-spacing rule of `pspline` is what prevents a flexible background from absorbing a component: with knots at least two maximal component widths apart, the spline cannot represent a feature narrower than `max_fwhm`. The rule is enforced at construction (`SettingsError` otherwise).

#### 5.5.3 The fitting engine (`evoparam.models.engine`)

Inputs: `times` (s), `y` (one waveform, data units), optional `sem` (per-sample standard error from §5.4 `average`), optional `mask` (True where a sample was replaced by §5.3, from `TrialSet.masks`), `window W`, `baseline B`, a `ResponseModel` settings object. Output: a `ResponseFit`. The ten steps below are executed in order; each writes a line into `ResponseFit.provenance.steps` with its intermediate numbers.

1. **Noise model from the baseline.** On `B` (after the baseline correction of §5.4): `σ_n = 1.4826 · median(|y_B − median(y_B)|)` (MAD, robust to a stray artifact), and `ρ = corr(y_B[1:], y_B[:-1])` (lag-1 autocorrelation of the baseline). The effective sample count of the window is `n_eff = n_W · (1 − ρ)/(1 + ρ)`, the standard AR(1) correction; it enters every information criterion and covariance below, because residuals of evoked waveforms are autocorrelated and criteria computed on raw samples over-select components [LIT-fitting-3]. If `B` is shorter than `20` samples, `σ_n` falls back to the MAD of the first derivative of `y_W` divided by `√2` and a `SettingsError` is logged as a warning (`"short_baseline"`). When `sem` is given, weights `w_i = 1/max(sem_i, 0.1·σ_n)` are used in every least-squares problem; otherwise `w_i = 1/σ_n`. Masked samples get `w_i = 0` (they neither attract nor repel the fit, REQ-108).

2. **Variable projection.** For fixed non-linear parameters `Θ = (θ_1..θ_K, τ_b)` the model is linear in `β = (A_1..A_K, c_0..)`: with the design matrix `M(Θ) = [s_1(t;θ_1) … s_K(t;θ_K) | basis of b]`, the weighted linear solution is `β̂(Θ) = argmin ‖W^{1/2}(y − M(Θ)β)‖²` by `scipy.linalg.lstsq` (QR, rank-revealing), and the residual handed to the optimiser is `r(Θ) = W^{1/2}(y − M(Θ)β̂(Θ))`. This is the Golub–Pereyra variable-projection scheme; it halves the number of non-linear parameters, removes the amplitude/background scale degeneracy, and makes the problem invariant to a DC offset by construction (REQ-109) (standard method; this specification's choice). The Jacobian of `r` with respect to `Θ` uses the Kaufman approximation `−W^{1/2}(I − M M⁺) ∂M/∂Θ β̂`, assembled from the shapes' analytic Jacobians.

3. **Candidate generation (initial guesses only).** The residual of the current model `r_K(t) = y − ŷ_K` is smoothed with a Gaussian kernel of standard deviation `min_fwhm/2.3548/2` (half the narrowest allowed width, so that noise narrower than any admissible component is attenuated while real components are not) and standardised by `σ_n`. All local extrema of `|r̃_K|` with `|r̃_K| ≥ detection_threshold` (default `2.0`, in units of `σ_n`) are candidates, **including extrema at the window edges** (a sample is an edge extremum when it is the largest in absolute value within `min_fwhm` of the edge; the N1 of a CCEP window that starts at 9–10 ms is frequently such an edge extremum, which rule `fit.py:1206-1208` of the inherited code discarded [CODE §0]). For each candidate: `μ_0` = extremum time, `A_0` = signed residual value, `σ_0` = half-width at half-maximum of `|r̃_K|` around the extremum, clipped to `[min_fwhm, max_fwhm]/2.3548`, with the fallback `σ_0 = min_fwhm/2.3548` when the half-maximum is not reached inside `W` (an edge component); shape-specific parameters start at neutral values (`α_0 = 0`, `κ_0 = 2`, `τ_r0 = σ_0/2`, `τ_d0 = 2σ_0`, `f_0` = centre of the frequency bounds, `φ_0 = 0`). Candidates are ordered by `|r̃_K|` descending. `detection_threshold` affects **only** the order and number of candidates tried, never the acceptance of a component (§4.4; REQ-104). When `polarity="negative"` or `"positive"` is requested (a deliberate restriction, not the default), extrema of the wrong sign are skipped at this step only.

4. **Forward stepwise fitting with bounded least squares.** Starting from `K = 0` (background only; `τ_b` initialised by a grid over its bounds), the engine adds the best candidate, re-optimises all non-linear parameters of the current model with `scipy.optimize.least_squares(method="trf", jac=analytic, bounds=…, loss=loss, f_scale=f_scale, max_nfev=max_iter)` from `n_starts` initialisations (the candidate itself, and `n_starts − 1` perturbations of `μ_0` by `±σ_0/2` and `σ_0` by `×{0.5, 2}`), keeps the best of the starts, and computes the selection criterion of §5.5.4. The new component is **accepted** if the criterion improves by at least `delta` (`ΔBIC ≤ −10` by default); otherwise the next candidate is tried (at most `max_candidates = 2·max_components` per step). The step ends when no candidate is accepted or `K = max_components`. Robust losses (`loss="soft_l1"`, `"huber"`, `"cauchy"`) are available for waveforms contaminated by residual artifact; the default is `"linear"` because the information criteria are derived for it, and a non-linear loss switches the criterion to the pseudo-likelihood of the loss with a logged warning [LIT-fitting-39], [LIT-fitting-17].

5. **Ordering and separation constraints instead of deletion.** Inside each optimisation the components are kept in ascending latency order by parameterising latencies as `μ_1 = u_1`, `μ_{k+1} = μ_k + exp(v_k)` (the inherited code sorted and deleted; here the order is part of the model). After each step, any pair with `|μ_{k+1} − μ_k| < overlap_c · (σ_k + σ_{k+1})` (default `overlap_c = 0.5`, i.e. closer than half their summed widths) triggers a **merge-and-compare**: the two components are replaced by one component initialised at their amplitude-weighted centroid, the merged model is refitted, and the model with the better criterion is kept. Nothing is deleted on the grounds of overlap alone, because overlapping components are a sloppy, not a wrong, part of the model [LIT-fitting-34]; the inherited `gauss_overlap_thresh` deletion (`fit.py:1268-1277` [CODE §0]) is removed.

6. **Final joint refit and diagnostics.** All `K` components and the background are refitted jointly from the stepwise solution; the Jacobian `J` at the solution gives `Cov(Θ̂) = s² (JᵀJ)⁻¹` with `s² = RSS_w/(n_eff − p)` (`p` = number of free non-linear plus linear parameters), the linear parameters' covariance by the variable-projection formula, the condition number `cond(JᵀJ)`, the eigen-spectrum, and the pairwise correlation matrix. `cond > 1e8` or any `|corr| > 0.95` sets the fit warning `"ill_conditioned"` and lists the parameter pairs (sloppy directions) [LIT-fitting-34], [LIT-fitting-35]. The fit is never rejected for ill-conditioning; the user is told which numbers not to trust.

7. **Features.** Every component's features are computed from its fitted curve as in §5.5.1; `Component.se` and `Component.ci` come from §5.5.5.

8. **Goodness of fit.** `GoodnessOfFit` (§4.3.4): `r2 = 1 − RSS/TSS` (coefficient of determination, on the unweighted residual, replacing the inherited Pearson-squared [CODE §2.2]), `adj_r2`, `rmse`, `mae`, `nrmse = rmse/(max y − min y)`, `aic`, `aicc`, `bic` (all on whitened residuals with `n_eff`, §5.5.4), `residual_lag1_autocorr` (on the window residual, to be compared with `ρ`), `baseline_sd = σ_n`, `n_params`, `n_samples`, `n_eff`.

9. **Classical measures in the same window** (§5.5.6), stored in `ResponseFit.classical`, so that a paper can report both the model-based and the conventional numbers from one call.

10. **Status.** `status="ok"` when at least one component is accepted; `"no_response"` when `K = 0` after trying every candidate (the background-only model wins) — this is a legitimate, informative result, logged at INFO and reported with the best rejected candidate's `ΔBIC` so that borderline cases can be inspected; `"failed"` only when the optimiser raised or returned non-finite values after all starts, logged at WARNING with the exception message in `ResponseFit.message` (the inherited code swallowed `FitError` into NaN parameters silently [CODE §2.2]). `"skipped"` is set by callers (bad channel, all samples masked).

Why this fixes the N1 case of §1: on the example run, N1 at 15–25 ms is (i) an edge extremum of a window starting at 9–10 ms (step 3 keeps it), (ii) often as narrow as 4–8 ms at 2048 Hz, which the width rule `fit.py:1061-1071` rejected and step 3 now clips to `min_fwhm` instead, (iii) followed within 20–40 ms by a broad N2 of the same sign, which the overlap rule merged away and step 5 now merges only if the merged model is *better*, and (iv) accepted or not by `ΔBIC`, not by a noise-calibrated threshold that passed 93 % of pure-noise windows at `3σ` [CODE §0]. REQ-113 and REQ-114 make this measurable.

#### 5.5.4 Model selection

For whitened residuals `r_w = r/σ_n` and `n_eff` from step 1:
`BIC_K = n_eff · ln(RSS_w/n_eff) + p_K · ln n_eff`, `AIC_K = n_eff · ln(RSS_w/n_eff) + 2 p_K`, `AICc_K = AIC_K + 2 p_K (p_K + 1)/(n_eff − p_K − 1)` (textbook definitions computed on whitened residuals, as recommended for autocorrelated noise [LIT-fitting-3]). The default rule is `selection="bic"` with `delta = 10` (a component must lower BIC by 10, "very strong" evidence on the usual scale), because (a) BIC is consistent for the nested sequence used here, (b) the raw-sample `n` overstates the evidence by a factor `(1+ρ)/(1−ρ)` (≈ 19 at `ρ = 0.9`, typical of 2 kHz intracranial data), which `n_eff` corrects, and (c) the boundary problem (a component whose amplitude is zero has unidentified `θ`) makes the chi-square asymptotics of a likelihood-ratio test invalid, so the threshold is calibrated by simulation (REQ-112: pure-noise acceptance ≤ 5 % at the default `delta`) rather than by theory (this specification's choice). Alternatives: `selection="aicc"`; `selection="cv"` (trial-level `k`-fold cross-validation of the prediction error when a `TrialSet` is available: fit on the average of the training folds, score on the held-out average; chosen number of components = minimum of the CV curve within one standard error); `selection="fixed"` with `n_components` (a fixed component list, for presets that must reproduce a published definition). The `delta` value and the criterion are stored in `ResponseFit.settings` and printed by `methods_text()`.

#### 5.5.5 Uncertainty at four cost levels

| `uncertainty=` | what it computes | cost | when |
|---|---|---|---|
| `"jacobian"` (default) | `se` from `Cov(Θ̂)` of step 6 with the `n_eff` correction; Wald intervals `θ̂ ± z_{1−α/2}·se`; feature intervals by the delta method through the numerical feature gradient; flagged unreliable when a parameter is within `1e-3` of a bound or when `"ill_conditioned"` is set | free | every fit |
| `"profile"` | profile-likelihood intervals: for each parameter, `θ_i` is fixed on a grid and the others re-optimised; the interval is `{θ_i : 2[ln L(Θ̂) − ln PL(θ_i)] ≤ χ²_{1,1−α}}` (3.84 at 95 %); flat profiles → `"structurally_unidentifiable"`, one-sided → `"practically_unidentifiable"` [LIT-fitting-35], [LIT-fitting-48] | `≈ 2 × 20 × K` refits | overlapping components, reviewers' requests |
| `"bootstrap"` | `B` resamples of trials with replacement (`B = 200` default), re-average, refit from the full-data solution (warm start), percentile and BCa intervals on every parameter and feature; also returns the bootstrap distribution for plotting; requires a `TrialSet` [LIT-fitting-28], [LIT-fitting-7] | `B` refits | papers, trial-count planning |
| `"posterior"` | full posterior from §5.8 (`evoparam.hierarchical`) for one channel with flat trial structure | minutes | explicit request |

All four return the same fields (`Component.se`, `Component.ci`, `ResponseFit.uncertainty_method`), so downstream code does not change when the level changes. Coverage is validated in §7 (REQ-116).

#### 5.5.6 Classical measures (`evoparam.models.classical`)

Implemented from the textbook definitions so that every published ERP number can be reproduced in Python and reported next to the model-based one [LIT-erpmeas-1], [LIT-erpmeas-45], [LIT-erpmeas-3]:
- `peak_amplitude` and `peak_latency`: local peak (the extremum that is larger than its `n_neighbors` samples on each side, default 3 samples ≈ ERPLAB's "local peak") or simple extremum; polarity `"negative" | "positive" | "absolute"`; peak location constrained to the window.
- `mean_amplitude` over the window; `area` with the four variants (`"total"`, `"positive"`, `"negative"`, `"rectified"`) and the baseline options `"zero"` (default) or `"fraction_of_peak"` (adaptive baseline at a fraction of the peak, Liesefeld's recommendation [LIT-erpmeas-3]).
- `fractional_area_latency` (default 50 %, the time at which the chosen area reaches the fraction) and `fractional_peak_latency` (default 50 %, the time before the peak at which the waveform first reaches the fraction of the peak; also used for onset at 10 %).
- `sme` (standardised measurement error) for every score: analytic `SD(score over trials)/√n` for linear scores (mean amplitude, area), bootstrap over trials (`B = 1000`) for non-linear scores (peak, latencies) [LIT-erpmeas-41].
- `jackknife` helpers for group latency contrasts: leave-one-participant-out grand averages, corrected statistics `t_c = t/(n−1)`, `F_c = F/(n−1)²` [LIT-erpmeas-2], [LIT-erpmeas-4], and the retrieval of individual latencies `o_i = n·J − (n−1)·j_i` [LIT-erpmeas-6]; implemented in §5.11 but exposed here on the `classical` dictionary.
- `region_contrast_latency` (latency as a region contrast, [LIT-erpmeas-47]) is included in the registry with its published definition.

All scores record the exact definition used (`classical["peak_latency"]["definition"] = "local_peak(n_neighbors=3), polarity=negative, window=[0.009, 0.1]"`) so that the inconsistency of defaults across tools (fractional amplitude 0.5 vs 0.3, area baseline zero vs fraction of peak) documented in [LIT-erpmeas-3], [LIT-erpmeas-45], [LIT-erpmeas-46] is resolved by making the choice explicit.

#### 5.5.7 Labelling (`evoparam.models.labels`)

`label_components(fit, windows: ComponentWindows, *, polarity_rule="sign", unassigned="C{k}")` assigns names (N1, N2, P1, N15, P30…) by latency window and polarity from a `ComponentWindows` table (§4.4; preset-specific tables in §6). When two components fall in one window, the larger `|A|` gets the label and the other gets `"<label>_b"` and the warning `"multiple_in_window"`; a component outside every window keeps a generic `C{k}` name. Labels are cosmetic metadata: nothing in fitting or selection depends on them.

#### 5.5.8 Public API

```python
class ResponseModel(BaseEstimatorMixin):
    """Estimator for evoked waveforms (scikit-learn contract of §10.4); stateless until fit().

    Keywords left at None take the preset's value, or the package default named in the comment when preset is None.
    """
    def __init__(self, preset: str | Preset | None = None, *,
                 shapes: str | Sequence[str] | None = None,                      # package default ("gaussian", "skewed_gaussian")
                 background: str | Background | None = None,                     # package default "exponential"
                 max_components: int | None = None,                              # package default 6
                 min_fwhm: float | None = None,                                  # s; package default 0.002
                 max_fwhm: float | None = None,                                  # s; package default 0.300
                 min_amplitude: float = 0.0,         # data units; 0 = no restriction
                 detection_threshold: float = 2.0,   # in baseline-SD units, candidates only
                 polarity: Literal["both", "negative", "positive"] | None = None,  # package default "both"
                 fit_window: tuple[float, float] | None = None,                  # s; required when there is no preset
                 baseline: tuple[float, float] | None = None,                    # s; None with no preset = no baseline window
                 sfreq: float | None = None,                                     # required when X is an ndarray
                 tmin: float | None = None,                                      # s; time of the first sample when X is an ndarray
                 selection: Literal["bic", "aicc", "cv", "fixed"] = "bic",
                 delta: float = 10.0,
                 n_components: int | None = None,    # for selection="fixed"
                 overlap_c: float = 0.5,
                 loss: Literal["linear", "soft_l1", "huber", "cauchy"] = "linear",
                 f_scale: float | None = None,
                 max_iter: int = 2000,
                 n_starts: int = 3,
                 onset_fraction: float = 0.10,
                 uncertainty: Literal["jacobian", "profile", "bootstrap", "posterior"] = "jacobian",
                 n_bootstrap: int = 200,
                 ci_level: float = 0.95,
                 classical: bool = True,
                 windows: ComponentWindows | None = None,
                 n_jobs: int = 1,
                 random_state: int | np.random.Generator | None = None,
                 verbose: bool | str | None = None) -> None: ...
    def fit(self, X, y=None) -> "ResponseModel": ...      # X: Evoked | TrialSet | mne object | ndarray (n_channels, n_times) with sfreq, tmin;
                                                          # learned attributes fit_ (GroupFit), components_, feature_names_out_, n_features_in_ (§10.4)
    def fit_waveform(self, times: np.ndarray, y: np.ndarray, *,
                     window: tuple[float, float] | None = None,      # default: fit_window
                     baseline: tuple[float, float] | None = None,    # default: the model's baseline
                     sem: np.ndarray | None = None,
                     mask: np.ndarray | None = None,
                     trials: np.ndarray | None = None) -> ResponseFit: ...   # one waveform; trials needed for bootstrap/cv
    def fit_group(self, ts: TrialSet | Evoked, *, window=None, baseline=None, picks=None,
                  n_jobs: int | None = None) -> GroupFit: ...
    def evaluate(self, times: np.ndarray, fit: ResponseFit, *,
                 components: Sequence[int] | None = None, background: bool = True) -> np.ndarray: ...   # model curve of one ResponseFit on any time grid
    def predict(self, X=None) -> np.ndarray: ...          # fitted curves of fit_, (n_channels, n_times) (§10.4)
    def transform(self, X) -> np.ndarray: ...             # feature matrix (n_channels, n_features) (§10.4)
    def score(self, X, y=None) -> float: ...              # mean r2 over channels
    def get_feature_names_out(self, input_features=None) -> np.ndarray: ...
    def get_params(self, deep: bool = True) -> dict: ...
    def set_params(self, **params) -> "ResponseModel": ...
    def __sklearn_tags__(self): ...      # non-standard X: documented, check_estimator subset only

def fit(data: TrialSet | Evoked | "mne.Epochs" | "mne.Evoked" | "list[mne.Evoked]" | np.ndarray,
        preset: str | Preset | None = None, *,
        window: tuple[float, float] | None = None,
        baseline: tuple[float, float] | None = None,
        times: np.ndarray | None = None,            # required when data is an ndarray
        sfreq: float | None = None,
        picks: Sequence[int | str] | None = None,
        by: str | list[str] | None = None,           # metadata column(s) defining the conditions averaged and fitted separately
        average: bool = True,
        trials: bool = False,                       # also run §5.7 after the channel fit
        existence: bool = True,                     # also run §5.6 and attach ResponseTest
        n_jobs: int = 1,
        random_state=None, verbose=None,
        **overrides) -> GroupFit: ...
    # raises SettingsError on unknown override names or inconsistent window/baseline;
    # DataError when window exceeds the data; never raises on a non-converging channel
    # (that channel's ResponseFit.status == "failed").

def fit_response(times, y, *, window, baseline, preset=None, **overrides) -> ResponseFit: ...
def label_components(fit: ResponseFit | GroupFit, windows: ComponentWindows | str, **kw) -> same type: ...
def classical_measures(times, y, *, window, polarity="negative", measures=("peak", "mean", "area", "fractional_area_latency"),
                       fraction=0.5, area="total", area_baseline="zero", n_neighbors=3,
                       trials: np.ndarray | None = None, n_bootstrap=1000, random_state=None) -> dict: ...
def compare_models(fits: Sequence[ResponseFit]) -> pd.DataFrame: ...    # ΔBIC/ΔAICc table across settings
def list_shapes() -> list[str]; def register_shape(cls: type[ComponentShape]) -> None
def list_backgrounds() -> list[str]
```

Return containers are those of §4.3.4. `GroupFit.to_dataframe()` yields one row per component with columns `channel, label, shape, latency, amplitude, polarity, fwhm, onset, offset, area, rise_time, decay_time, symmetry, sharpness, se_latency, se_amplitude, ci_latency_low, ci_latency_high, …, r2, bic, delta_bic, status, warnings` (tidy, one number per cell, REQ-120).

#### 5.5.9 Edge cases

- Window shorter than `3·min_fwhm` → `SettingsError`. Window containing fewer than `p + 5` samples for the attempted model → that component is not attempted (`"window_too_short_for_K"`).
- All samples masked, or more than `max_masked_fraction` (default 0.5) → `status="skipped"`.
- Flat waveform (`σ_n = 0`, e.g. a zeroed channel) → `status="skipped"`, message `"flat"`.
- NaN in `y` inside the window → treated as masked; NaN in the baseline → excluded from the noise estimate; a `DataError` if fewer than 20 valid baseline samples remain and no `sem` is provided.
- Saturated samples flagged by §5.2 are masked automatically (the `TrialSet.masks` propagate into `average`).
- A component converging to a bound (latency at `t_1 + σ/2`, width at `max_fwhm`) is kept and flagged `"at_bound:<param>"`.
- Two candidates at the same time with opposite signs cannot both be accepted (the second is the residual of the first); the engine handles this naturally because candidates are regenerated from the residual after each acceptance.
- Very low sampling rate (`sfreq < 2/min_fwhm`): `min_fwhm` is raised to `2/sfreq` with a warning.
- `polarity != "both"`: components of the other sign may still appear if the data require them for a good background (the restriction is on candidates); the user is warned once.
- Extremely long windows (> 10 s at ≥ 1 kHz): the engine decimates internally to ≤ 20 000 samples for the stepwise search and refits at full resolution (`"decimated_search"` noted in provenance).

#### 5.5.10 Acceptance criteria (REQ-100…REQ-129)

- REQ-100: the eight shapes of §5.5.1 and the seven backgrounds of §5.5.2 are implemented with the stated equations; a test evaluates each on a reference grid against values computed independently with `mpmath` at 1e-10 relative error.
- REQ-101: every analytic Jacobian matches central finite differences to 1e-6 relative on 100 random parameter sets inside the bounds, for every shape and background.
- REQ-102: `register_shape` accepts a user-defined shape implementing the protocol and rejects (with a `SettingsError` naming the failing check) one whose Jacobian is wrong or whose `evaluate` is not unit-normalised.
- REQ-103: variable projection — for fixed `Θ`, the amplitudes and background coefficients returned equal the weighted least-squares solution to 1e-10; masked samples (`w = 0`) have no influence (the fit on `y` with a mask equals the fit on `y` with the masked samples replaced by any values).
- REQ-104: `detection_threshold` never changes which components are *accepted*: on the simulation suite of §5.12, acceptance decisions for `detection_threshold ∈ {1.0, 2.0, 3.0}` agree on ≥ 99 % of waveforms (differences are allowed only through candidate ordering, which the test logs).
- REQ-105 (simulation recall/precision): on `simulate_erp` waveforms with `K ∈ {1,2,3}` components, `SNR ≥ 3` (peak amplitude over `σ_n`) and `FWHM ≥ 3` samples, component recall ≥ 0.95 and precision ≥ 0.95 (a recovered component matches a true one if `|Δlatency| ≤ FWHM/2`).
- REQ-106 (accuracy): at `SNR ≥ 5`, latency MAE ≤ 0.5 sample and amplitude relative bias ≤ 5 % (median over 1 000 waveforms per condition), for `gaussian`, `skewed_gaussian`, `exgaussian`, `gamma` and `biexponential` truth shapes fitted with their own shape and with the default `("gaussian", "skewed_gaussian")` pair.
- REQ-107 (coloured noise): with AR(1) noise `ρ = 0.9`, the false-component rate on pure-noise windows is ≤ 5 % at the default `delta`, versus ≥ 30 % when `n_eff` is replaced by `n` (the test documents why `n_eff` exists); the inherited reference point is 93 % of pure-noise windows passing at `3σ` [CODE §0].
- REQ-108 (masks): a component whose true latency falls inside a masked 2 ms gap is still recovered with recall ≥ 0.9 when at least 60 % of its FWHM lies outside the mask.
- REQ-109 (DC invariance): adding any constant `c` to `y` changes no component feature by more than 1e-6 relative.
- REQ-110 (edge components): for true components with latency in `[t_0, t_0 + FWHM]` (partly cut by the window start), recall ≥ 0.95 at `SNR ≥ 5`; the `"latency_outside_window"` warning is set when and only when the fitted latency is `< t_0`.
- REQ-111 (resolution): two equal-amplitude Gaussians separated by `1.0·FWHM` are recovered as two components in ≥ 90 % of runs at `SNR = 10`; at `0.5·FWHM` the engine returns one component with the warning `"merged:<k>,<k+1>"` in ≥ 90 % of runs and never two components with `|corr| ≤ 0.95` (i.e. when two are returned they are flagged ill-conditioned).
- REQ-112 (type-I of selection): pure-noise windows (white, AR(1) `ρ ∈ {0.5, 0.9}`, and real baseline segments of ds004080) produce `status="no_response"` in ≥ 95 % of cases at the default `delta` and `max_components`.
- REQ-113 (the N1 case): on ds004080 `sub-ccepAgeUMCU01 run-021448`, stimulation pair `PT01-PT02`, the default `ccep` preset (§6) detects an N1 (negative component with latency in 9–50 ms) on every channel on which the published rule (`3.4 × SD` of the baseline in `9–100 ms` [LIT-ccep-8], [LIT-ccep-7]) detects one, minus at most one channel, and the agreement across all 46 non-stimulated channels is Cohen's `κ ≥ 0.8`; the result is reproduced with `κ ≥ 0.8` on at least five subjects of ds004080 (REQ-371 lists them).
- REQ-114 (N1 latency agreement): on the channels where both agree, the model latency and the published rule's peak latency differ by ≤ 2 ms on ≥ 90 % of channels.
- REQ-115 (speed): one waveform of 2 048 samples with `max_components = 6`, default shapes and `uncertainty="jacobian"` fits in ≤ 50 ms median on the reference machine of §9.1 (targets P1 and REQ-290); `fit_group` on 100 channels × 2 048 samples with `n_jobs=1` completes in ≤ 10 s.
- REQ-116 (coverage): nominal 95 % intervals cover the true latency and amplitude in 90–97 % of simulations under white noise and in ≥ 85 % under AR(1) `ρ = 0.9` for `uncertainty="jacobian"`; `"bootstrap"` and `"profile"` reach 92–97 % in both.
- REQ-117 (classical measures): `classical_measures` reproduces, to 1e-9, reference values computed by an independent implementation of the ERPLAB definitions on ERP CORE N170/P3 averages [LIT-erpmeas-39] (definitions, not code, are shared); `sme` for mean amplitude equals `SD/√n` exactly.
- REQ-118 (status semantics): no code path returns NaN parameters with `status="ok"`; every `"failed"` fit carries a non-empty `message` and a WARNING log line; every `"no_response"` carries `best_rejected_delta`.
- REQ-119 (reproducibility): two runs with the same `random_state` give byte-identical `to_dataframe()` output (also with `n_jobs > 1`).
- REQ-120 (tidy output): `GroupFit.to_dataframe()` has one row per component, no nested objects, and `GroupFit.save/load` round-trips exactly.
- REQ-121 (conditioning report): `ResponseFit.diagnostics` contains `cond`, `eigvals`, `corr` and the list of flagged pairs; a two-component fit at `0.3·FWHM` separation is flagged `"ill_conditioned"` in 100 % of runs.
- REQ-122 (selection alternatives): `selection="cv"` on a `TrialSet` with 40 trials chooses the true `K` in ≥ 85 % of simulations at `SNR ≥ 5`; `selection="fixed"` with `n_components=K` always returns exactly `K` components (possibly with `A ≈ 0` and a warning).
- REQ-123 (methods text): `ResponseFit.methods_text()` names the shapes, background, criterion, `delta`, `n_eff` correction, uncertainty method and the package version in a paragraph that a test compares to a frozen template.

#### 5.5.11 Tests to write

`tests/test_shapes.py` (REQ-100…102), `tests/test_varpro.py` (103, 109, 108), `tests/test_engine_sim.py` (104…107, 110…112, 116, 121, 122; uses `evoparam.sim` with fixed seeds, 1 000 waveforms per condition, marked `slow` beyond 100), `tests/test_engine_ccep.py` (113, 114; downloads the ten trials of §5.1 REQ-021 with `remote=True`, cached in CI), `tests/test_classical.py` (117), `tests/test_status.py` (118), `tests/test_reproducibility.py` (119, 120), `tests/test_methods_text.py` (123), `benchmarks/bench_engine.py` (115, run by `asv`-style timing in CI with a tolerance band).

#### 5.5.12 Tutorials

`tutorials/10_fit_one_waveform.py` (an ERP CORE P3 average: default fit, labels, classical measures side by side), `tutorials/11_ccep_n1_n2.py` (the ds004080 run of §1: artifact removal → fit → N1/N2 table over 46 channels → agreement with the published rule), `tutorials/12_shapes_and_backgrounds.py` (same waveform under five shapes, `compare_models` table), `tutorials/13_uncertainty_levels.py` (jacobian vs bootstrap vs profile on one channel), `tutorials/14_custom_shape.py` (registering a shape).

### 5.6 `evoparam.existence` — is there a response, and how long does it last?

**Purpose.** Answer, before any shape is reported, whether the window contains a response at all, with a calibrated false-positive rate; estimate the response duration and onset; and reproduce, exactly, the published detection rules so that results can be compared with the literature. The motivation is the CCEP literature's state: 49.7 % of 187 studies did not state their detection method and 68.1 % of those that did used a fixed amplitude threshold [LIT-ccep-11]; thresholds vary from `2.6` to `6` baseline SD and N1 windows from `10–50` to `9–100` ms (and whole-response windows up to 800 ms) across the same decade of papers [LIT-ccep-3], [LIT-ccep-5], [LIT-ccep-6], [LIT-ccep-7], [LIT-ccep-13], [LIT-ccep-14]; human inter-rater agreement on N1 presence is `κ ≈ 0.60` [LIT-ccep-8]. None of the ERP scoring tools tests presence before scoring, so absent components yield noise-driven peaks [LIT-erpmeas-1], [LIT-erpmeas-44].

**Methods.**
1. `baseline_permutation` (default). Null hypothesis: the window is exchangeable with the baseline. With `n` trials `x_k(t)`, the statistic is the pointwise `t(t) = mean_k x_k(t) / (SD_k x_k(t)/√n)`, summarised either by `max_t |t(t)|` over `W` (`statistic="max_t"`, controls the family-wise error over time) or by the largest cluster mass of `|t| > t_crit` (`statistic="cluster"`, more powerful for sustained responses). The null distribution is obtained by `n_permutations` (default 1 000) random sign flips of the trial polarity (valid when the baseline has been subtracted and trials are exchangeable; for an `Evoked` without trials, a baseline-shuffle null is used: the statistic is recomputed on windows of length `|W|` drawn from the baseline of the same channel and of the other channels, with a warning that the null is weaker). `p_value` is the fraction of null statistics ≥ the observed one, with the `+1` correction. Duration `τ_R` and onset: the earliest and the latest time of the significant cluster(s); `onset` is the first significant sample, `offset` the last, `duration = offset − onset`, each with a bootstrap interval over trials. SNR in dB: `10·log10(mean_t ŷ(t)² / σ_n²)` on `W`.
2. `projection` (cross-projection test, re-implemented from the published description of the canonical response parameterisation [LIT-ccep-15], [LIT-fitting-1]; the reference implementation is GPL and is neither imported nor copied, §13). For trials `V_k(t)` on `[t_0, t]`, the cross-projection of trial `k` on trial `l` is `P_{kl}(t) = ⟨V_k, V_l⟩_{[t_0,t]} / ‖V_l‖_{[t_0,t]}`; the mean profile over all ordered pairs `P̄(t)` is computed on a grid of `t`; the response duration `τ_R = argmax_t P̄(t)/√(t − t_0)` (the normalised profile, so that longer windows are not favoured by accumulation alone); at `τ_R` the projection magnitudes of one half of the non-overlapping pairs are tested against zero by a one-sample `t`-test (`statistic`, `p_value`); the canonical shape `C(t)` is the first right singular vector of the trial matrix on `[t_0, τ_R]`, the per-trial scalings `α_k = ⟨V_k, C⟩`, the explained variance `R²`, the SNR `‖α‖²/(n·σ_ε²)`. The implementation is validated against the numbers published for the public example data (REQ-134), not against the reference code.
3. `model_comparison`. `ΔBIC` between the best model of §5.5 and the background-only model, converted to a `p_value` by parametric bootstrap under the background-only model (the boundary problem of §5.5.4 invalidates the chi-square approximation): `n_permutations` synthetic windows are generated from the fitted background plus AR(1) noise with the baseline `σ_n`, `ρ`, each refitted, and `p = P(ΔBIC_null ≤ ΔBIC_obs)`.
4. `threshold` — a registry of published rules, reproduced to the letter so that a paper can state "detected with rule X" and an evoparam user can compare. Each entry has `name, statistic, k, window, baseline, polarity, min_duration, source`:

| name | rule | window (ms) | baseline (ms) | source |
|---|---|---|---|---|
| `"vanblooijs2023"` | `\|x\| > 3.4·SD_baseline` with a negative peak, ≥ 1 sample | 9–100 | `unspecified` in the paper; the ER-detect default baseline search epoch is −1 000 to −100 [LIT-ccep-30] | [LIT-ccep-7], [LIT-ccep-8] |
| `"hays2021"` | `\|z\| > 6` on the N1 peak | 10–50 | `unspecified` (pre-stimulus SD) | [LIT-ccep-5] |
| `"keller2014"` | `z > 6` on the polarity-agnostic magnitude of the early response (A1) | < 50 | −500 to −5 | [LIT-ccep-3] |
| `"silverstein2020"` | `z > 5` on the maximum of a negative deflection (N1) | 10–50 | `unspecified` | [LIT-ccep-6] |
| `"ftract"` | the 800 ms post-stimulus average exceeds `Z` between 3 and 7 (z-score against the `[−400, −10]` ms baseline), after the F-TRACT 1–45 Hz band-pass; optional 200 ms peak-delay cut | 0–800 (peak delay ≤ 200 with the cut) | −400 to −10 | [LIT-ccep-14], [LIT-ccep-31] |
| `"prime2020"` | RMS over 10–100 ms with volume-conduction screening (peak-to-peak stimulation artifact in the first 10 ms, inverse-square distance law); detection threshold `unspecified` | 10–100 | `unspecified` | [LIT-ccep-47] |
| `"custom"` | user-supplied `k`, window, baseline | — | — | — |

The exact window and baseline values of each row are frozen in `evoparam/existence/rules.json` with the citation and a `verified: true/false` flag per field; where the primary paper does not state a field (several do not state the baseline), the field is marked `unspecified` and the user must supply it (the registry refuses to run an incomplete rule silently, REQ-137).

5. **Corrections.** Across channels and stimulation sites: `correction="fdr_bh"` (default), `"holm"`, `"bonferroni"`, `"none"`; both `p_value_raw` and `p_value` are returned.
6. **Train-to-train discordance.** When `metadata["train_id"]` exists, the test is run per train and the discordance `d = 1 − 2·|P(sig) − 0.5|` (0 when all trains agree, 1 when half agree) is reported; this is the quantity that explains the ≈ 20–40 % inter-train variability of the F-TRACT reproducibility study [LIT-ccep-14] and the trial-to-trial variability of [LIT-ccep-9].
7. **Polarity.** All statistics are two-sided by default (`polarity="both"`); `"negative"` and `"positive"` run one-sided tests on the signed mean.

**Public API.**

```python
def test_response(ts: TrialSet | Evoked, *,
                  method: Literal["baseline_permutation", "projection", "model_comparison", "threshold"] = "baseline_permutation",
                  window: tuple[float, float] = (0.009, 0.5),
                  baseline: tuple[float, float] = (-0.5, -0.05),
                  alpha: float = 0.05,
                  correction: Literal["fdr_bh", "holm", "bonferroni", "none"] = "fdr_bh",
                  n_permutations: int = 1000,
                  statistic: Literal["max_t", "cluster"] = "max_t",
                  cluster_threshold: float | None = None,       # default t_{0.975, n-1}
                  polarity: Literal["both", "negative", "positive"] = "both",
                  rule: str | dict | None = None,               # for method="threshold"
                  picks=None, by: Sequence[str] = ("stim_site",),
                  return_null: bool = False,
                  random_state=None, n_jobs: int = 1, verbose=None) -> ResponseTest: ...
    # ResponseTest fields: method, statistic, p_value_raw, p_value, significant (bool array channels×groups),
    # duration, onset, offset (s, with ci), snr_db, alpha, correction, n_permutations, rule, provenance.

def response_duration(ts, *, method="baseline_permutation" | "projection", **kw) -> pd.DataFrame: ...
def compare_detectors(ts, rules: Sequence[str] = ("baseline_permutation", "vanblooijs2023", "hays2021", "projection"),
                      **kw) -> pd.DataFrame: ...      # per-channel decisions + pairwise Cohen's κ
def list_rules() -> pd.DataFrame
def train_discordance(ts, test: ResponseTest) -> pd.DataFrame
class detectors:                                      # namespace of the registry rules, callable individually
    @staticmethod
    def vanblooijs2023(ts, **overrides) -> ResponseTest: ...
    ...
```

`evoparam.fit(..., existence=True)` runs `test_response` with the preset's method and attaches the result to `GroupFit.existence`; channels with `significant == False` still get a `ResponseFit` (status may be `"no_response"` or `"ok"`), and `GroupFit.to_dataframe()` carries both columns so that the user sees when the two disagree (a fitted component on a non-significant channel is reported with `existence_p`, never hidden).

**Edge cases.** Fewer than 5 trials → permutation methods refuse (`DataError`) and `threshold` is suggested; `n_permutations` larger than `2^n` enumerates all sign flips instead; a window overlapping the artifact window of §5.3 → `SettingsError` unless `allow_artifact_overlap=True`; a baseline shorter than the window with `statistic="cluster"` → warning; channels with all samples masked → `significant = NaN`.

**Acceptance criteria (REQ-130…REQ-149).**
- REQ-130 (type I): on pure-noise `TrialSet`s (white, AR(1) `ρ = 0.9`, and ds004080 baseline segments re-epoched at random times), `baseline_permutation` rejects at a rate within `[0.8α, 1.2α]` for `α ∈ {0.05, 0.01}` with 20 trials, for both statistics.
- REQ-131 (power): with 10 trials and a Gaussian response of FWHM 10 ms at `SNR = 2` (per-trial peak over trial noise SD), power ≥ 0.9 at `α = 0.05` for `statistic="max_t"`; with a sustained 100 ms response at `SNR = 1`, `"cluster"` has power ≥ 0.9 where `"max_t"` has < 0.7 (the test documents when to use which).
- REQ-132 (duration): the estimated `duration` is within ±10 % of the simulated one (median) for `SNR ≥ 3`; `onset` within 2 ms.
- REQ-133 (registry exactness): every registry rule, applied to a synthetic waveform built to lie exactly at its threshold, flips decision at `k ± 1e-6`; fields marked `unspecified` raise `SettingsError` when missing.
- REQ-134 (projection validation): `method="projection"` reproduces the published duration and significance for the public example dataset of the cross-projection paper (ds003708 [LIT-ccep-36]) within ±5 ms on duration and with identical significance decisions on ≥ 95 % of channel pairs.
- REQ-135 (agreement with the field): on ds004080 (≥ 5 subjects), the default method agrees with `vanblooijs2023` with `κ ≥ 0.7` on N1 presence over 9–100 ms, and `compare_detectors` reports the pairwise κ matrix with the human reference value 0.60 [LIT-ccep-8] quoted in the output's caption.
- REQ-136 (corrections): with 1 000 null channels, `fdr_bh` at `α = 0.05` yields ≤ 5 % false discoveries in expectation and `bonferroni` ≤ 1 family-wise error in ≥ 95 % of runs.
- REQ-137 (no silent defaults): running a registry rule with an `unspecified` field without supplying it raises; the error names the field and the paper.
- REQ-138 (speed): `test_response` with 1 000 permutations on 100 channels × 50 trials × 1 024 samples completes in ≤ 20 s with `n_jobs=1`.
- REQ-139 (discordance): `train_discordance` returns 0 for simulated identical trains and ≈ 1 for trains alternating response/no response.
- REQ-140 (methods text): `ResponseTest.methods_text()` states the method, statistic, window, baseline, `α`, correction, permutations and seed.

**Tests to write.** `tests/test_existence_null.py` (130, 136), `tests/test_existence_power.py` (131, 132), `tests/test_rules_registry.py` (133, 137), `tests/test_projection.py` (134, needs the ds003708 sample cached in CI), `tests/test_existence_ccep.py` (135, 139), `benchmarks/bench_existence.py` (138), `tests/test_existence_text.py` (140).

**Tutorials.** `tutorials/20_is_there_a_response.py` (one CCEP site, four detectors, κ matrix, why they disagree), `tutorials/21_response_duration.py` (permutation vs projection duration on ds004080 and on an ERP CORE dataset), `tutorials/22_reproduce_a_published_rule.py` (registry rule end to end with the methods sentence).
### 5.7 `evoparam.trials` — trial-by-trial parameters, jitter and realignment

**Purpose.** Estimate, for every trial, the latency, amplitude and (optionally) width of each component found on the channel average, with a warm start from the channel model, so that trial-level variability becomes data rather than noise: latency jitter, amplitude adaptation across a train, dose–response across stimulation currents, dependence on the pre-stimulus state. Nothing comparable exists in the Python ecosystem (single-trial estimation was asked about on the MNE-Python tracker in 2017 and the issue closed without an implementation, and a related trial-dependent baseline request of 2018 is still open [LIT-needs-29], [LIT-needs-30]); the MATLAB lineage (Woody filtering, maximum-likelihood single-trial estimation, residue iteration decomposition, multiple linear regression on wavelet-filtered trials, Kalman and Bayesian estimators) is fragmented, unmaintained or closed [LIT-erpmeas-7], [LIT-erpmeas-8], [LIT-erpmeas-10], [LIT-erpmeas-21], [LIT-erpmeas-24], [LIT-erpmeas-26], and the one recent Bayesian trial-level model needs about ten hours per subject and ships no code [LIT-erpmeas-27].

**Methods.**
1. **Trial model.** With the channel fit `ŷ(t) = b̂(t) + Σ_k Â_k s_k(t; θ̂_k)`, trial `i` is modelled as

   `x_i(t) = b_i(t) + Σ_k a_{ik} · s_k(t − δ_{ik}; θ̂_k ⊕ γ_{ik}) + ε_i(t)`

   with per-trial amplitude scale `a_{ik}` (linear), latency shift `δ_{ik}` (non-linear, bounded by `max_shift`, default `0.25·FWHM_k` for intracranial presets and `0.5·FWHM_k` for scalp presets), optional width stretch `γ_{ik} ∈ [0.8, 1.25]` (`stretch=True`), and a per-trial background `b_i` restricted to the linear terms of the channel background (constant and slope; the exponential time constant stays fixed at the channel value because it is not identifiable from one trial). The noise `ε_i` keeps the AR(1) model with per-trial `σ_i`, `ρ_i` from the trial's own baseline. This is the "scaled, shifted template" model shared by Woody filtering, maximum-likelihood estimation and the random-shift Gaussian-process model [LIT-erpmeas-7], [LIT-erpmeas-8], [LIT-fitting-3]; here the template is parametric, so the shift and the scale are regular parameters with analytic Jacobians.
2. **Estimation levels** (`mode=`):
   - `"shift_scale"` (default): only `a_{ik}` and `δ_{ik}` are free; one bounded least-squares problem per trial, warm-started at `(a = 1, δ = 0)`, with the same variable projection as §5.5.3 (amplitudes solved linearly for each trial shift). Deterministic, ≈ 1 ms per trial.
   - `"free"`: all non-linear parameters of each component are freed within `±25 %` of the channel values (bounds relative to the channel fit, so that a trial cannot wander to another component); warm-started from `"shift_scale"`.
   - `"template"`: the shape is the channel's fitted component curve used as a `template` shape (§5.5.1), the closest to classical template matching [LIT-erpmeas-7]; useful when a parametric shape fits the average poorly.
   - `"regularized"`: `"free"` with a ridge penalty pulling each trial's parameters toward the channel values, `λ` chosen by leave-one-trial-out prediction error; this is the empirical-Bayes shrinkage that §5.8 makes explicit.
3. **Convergence and trust.** A trial fit is `converged` when the optimiser terminates on the gradient or step criterion with finite parameters; a trial is `trusted` when, in addition, its per-trial `r2 ≥ r2_min` (default 0.2) and the shift is not at a bound. Untrusted trials keep their numbers (with the flag) so that the user, not the engine, decides exclusion; the trial table never has missing rows.
4. **Jitter estimation.** The latency jitter of component `k` is `jitter_sd_k = SD_i(δ_{ik})` over trusted trials, with its bootstrap interval; the attenuation of the average's peak amplitude by jitter is `Â_k / a_k^{jitter-free} ≈ σ_k/√(σ_k² + jitter_sd_k²)` for Gaussian shapes, so `TrialFit.amplitude_jitter_corrected` reports the amplitude a jitter-free average would have shown (the correction documented for residue iteration decomposition [LIT-erpmeas-13]). A realigned average (`realign=True`) shifts each trusted trial by `−δ_{ik}` for the component of interest and re-averages; it is returned as a new `Evoked` with provenance (the Woody-filter average [LIT-erpmeas-7]) and must be used with the caution that realignment on noise sharpens noise: the engine reports `jitter_sd` against the null jitter obtained by applying the same procedure to the pre-stimulus baseline (`jitter_sd_null`), and refuses to realign (warning) when `jitter_sd ≤ 1.2·jitter_sd_null`.
5. **Covariates.** `TrialFit.table` (one row per trial × component) carries the `TrialSet.metadata` columns so that trial parameters can be regressed on `stim_current_mA`, `index_in_train`, `state`, `condition` (§5.11). Three built-in covariates are computed on request:
   - `trend`: a linear (or exponential, `trend="exp"`) term in trial order within a train, `a_{ik} = a_k(1 + c_k·i)` or `a_k·exp(−i/τ_k)`, to quantify adaptation/facilitation across a train [LIT-variability-4], [LIT-variability-6];
   - `dose`: amplitude and latency versus `stim_current_mA` (and `stim_charge_uC` when available), fitted as a saturating function `a(I) = a_max·I^h/(I_50^h + I^h)` (Hill, `h` free in `[0.5, 6]`) or a straight line when fewer than four currents are available, with the threshold current `I_thr = I_50·(0.1/0.9)^{1/h}` reported; this is the quantity behind the dose–response and charge-normalisation findings of [LIT-variability-5], [LIT-variability-6], [LIT-variability-36], [LIT-variability-39];
   - `prestim`: pre-stimulus phase and power of a chosen band (default 8–13 Hz, alpha/mu) on `[−0.5, −0.01]` s per trial, computed with a causal filter and the Hilbert transform restricted to the pre-stimulus interval (so that the stimulus does not leak into the phase estimate), with the phase-estimation uncertainty reported as the circular SD across three estimators (Hilbert of the causal filter, autoregressive forward prediction, and the Hilbert transform on the mirrored segment), because real-time phase estimates have known errors [LIT-variability-18], [LIT-variability-20]; the dependence of `a_{ik}` on phase is fitted as `a = c_0 + c_1·cos(φ − φ_0)` (a von Mises first harmonic) with a permutation test, following the pre-stimulus phase and power literature [LIT-variability-16], [LIT-variability-17], [LIT-variability-19], [LIT-variability-23].
6. **Trial ordering effects in CCEP trains** (`index_in_train` from §5.2) and **state** (`metadata["state"]`, sleep/wake/anaesthesia) are the two covariates that the literature flags as the largest sources of non-random variability (sleep changes N1 and N2 amplitudes and adds delayed responses [LIT-variability-11], [LIT-variability-12], [LIT-variability-14]; propofol changes N2 [LIT-ccep-12]); they are therefore in the default `TrialFit.table`.

**Public API.**

```python
def fit_trials(ts: TrialSet, group_fit: GroupFit | None = None, *,
               mode: Literal["shift_scale", "free", "template", "regularized"] = "shift_scale",
               components: Sequence[str | int] | None = None,     # default: all accepted components
               max_shift: float | Literal["auto"] = "auto",
               stretch: bool = False,
               r2_min: float = 0.2,
               realign: bool = False, realign_on: str | int | None = None,
               covariates: Sequence[Literal["trend", "dose", "prestim"]] = (),
               prestim_band: tuple[float, float] = (8.0, 13.0),
               prestim_window: tuple[float, float] = (-0.5, -0.01),
               by: Sequence[str] = ("stim_site",),
               n_jobs: int = 1, random_state=None, verbose=None) -> TrialFit: ...
    # if group_fit is None, evoparam.fit(ts, ...) is run first with the TrialSet's preset.
def jitter(trial_fit: TrialFit, component: str | int) -> pd.DataFrame      # per channel: jitter_sd, ci, jitter_sd_null, amplitude_jitter_corrected
def realign(ts: TrialSet, trial_fit: TrialFit, component) -> Evoked
def dose_response(trial_fit: TrialFit, *, x: str = "stim_current_mA", model: Literal["hill", "linear", "auto"] = "auto") -> pd.DataFrame
def prestimulus_dependence(trial_fit: TrialFit, *, band=(8., 13.), n_permutations=1000, random_state=None) -> pd.DataFrame
```

`TrialFit` (§4.3.4) exposes `table` (tidy; columns `trial, channel, component, label, shift, scale, latency, amplitude, fwhm, r2, converged, trusted, + metadata columns`), `jitter_sd`, `jitter_sd_null`, `realigned`, `covariates`, `provenance`.

**Edge cases.** A channel with `status != "ok"` yields no trial rows but a `skipped_channels` list; trials that are fully masked in the component's support are `converged=False, trusted=False`; `max_shift` larger than half the distance to the neighbouring component is clipped with a warning (so a trial's N1 cannot be fitted on its N2); `"dose"` with a single current raises `SettingsError`; `prestim` with a baseline shorter than three cycles of the band's lowest frequency raises `SettingsError`.

**Acceptance criteria (REQ-150…REQ-169).**
- REQ-150: on simulated trials with known per-trial shifts (`jitter_sd ∈ {1, 2, 5}` ms) and amplitudes (CV 30 %), `"shift_scale"` recovers single-trial latency with MAE ≤ 1 sample and amplitude with relative MAE ≤ 15 % at per-trial `SNR ≥ 2`; `jitter_sd` is within ±20 % of truth.
- REQ-151: `jitter_sd_null` on jitter-free simulated trials is ≥ 0.8·`jitter_sd` (the method does not invent jitter), and `realign` refuses in ≥ 95 % of jitter-free runs.
- REQ-152: `amplitude_jitter_corrected` is within ±10 % of the simulated jitter-free amplitude for Gaussian components at `jitter_sd ≤ σ_k`.
- REQ-153: `"regularized"` has lower trial-level latency MAE than `"free"` at per-trial `SNR ≤ 1` on ≥ 90 % of simulated channels (the shrinkage helps when trials are noisy).
- REQ-154: `dose_response` recovers `I_50` within ±15 % and `h` within ±30 % on simulated Hill curves with five currents and 10 trials per current at `SNR = 3`.
- REQ-155: `prestimulus_dependence` has type-I error ≤ 0.06 at `α = 0.05` on trials with no phase dependence and power ≥ 0.8 for a 30 % modulation with 100 trials.
- REQ-156: on ds004080 (10 trials per site), `fit_trials` on all N1 channels of ≥ 5 subjects completes in ≤ 2 s per subject and the median trusted fraction is ≥ 0.8 on channels where the existence test is significant.
- REQ-157: the trial table has exactly `n_trials × n_components × n_channels_ok` rows, no NaN in `converged/trusted`, and round-trips through `save/load`.
- REQ-158: the `index_in_train` effect (amplitude slope) reproduces the direction reported for repeated single pulses at 1 Hz in the reference dataset used in §7 (declining N1 amplitude across the train) with a sign agreement on ≥ 80 % of significant sites.

**Tests.** `tests/test_trials_sim.py` (150–155), `tests/test_trials_ccep.py` (156, 158), `tests/test_trials_table.py` (157).

**Tutorials.** `tutorials/30_single_trials.py` (N1 per trial on one site, jitter, realigned average with the null check), `tutorials/31_dose_response.py` (a multi-current dataset of §5.1's registry), `tutorials/32_prestimulus_state.py` (alpha phase and amplitude of a scalp ERP from ERP CORE).

### 5.8 `evoparam.hierarchical` — trial → channel → subject with uncertainty

**Purpose.** Replace the two-step practice (fit the average, then do statistics on point estimates) by one model in which trials are nested in channels and channels in subjects, so that (a) the uncertainty of each level propagates to the next, (b) sparse channels borrow strength from the group, and (c) the probability that a response exists is a posterior quantity rather than a threshold. The multilevel literature on ERPs shows the gain in power and in calibration [LIT-fitting-6], [LIT-fitting-30], [LIT-erpmeas-28]; the recent trial-level Bayesian model demonstrates the quality but at a cost (≈ 10 h per subject, no public code) that this module must bring below one minute for the default path [LIT-erpmeas-27], [LIT-fitting-3].

**Methods.**
1. **Model.** For subject `s`, channel `c`, trial `i`, component `k`:

   `x_{sci}(t) = b_{sci}(t) + Σ_k a_{scik} · s_k(t − δ_{scik}; θ_{sck}) + ε_{sci}(t)`
   `a_{scik} ~ N(α_{sck}, σ²_{a,sk})`, `δ_{scik} ~ exGaussian(0, σ_{δ,sk}, τ_{δ,sk})` (right-skewed latency jitter, [LIT-fitting-27], [LIT-erpmeas-27])
   `α_{sck} ~ N(α_{sk}, σ²_{α,k})`, `θ_{sck} ~ N(θ_{sk}, Σ_{θ,k})` (channel shapes around the subject's), `α_{sk} ~ N(α_k, σ²_{α,group})` (subjects around the group)
   `ε_{sci} ~ AR(1)(σ_{sci}, ρ_{sc})`.

   Existence enters as a spike-and-slab on the channel amplitude: `α_{sck} = z_{sck}·α̃_{sck}`, `z_{sck} ~ Bernoulli(π_{sk})`, so that `P(z_{sck} = 1 | data)` is the posterior probability that channel `c` responds with component `k`.
2. **Default inference: empirical Bayes (`engine="eb"`).** Trial-level estimates from §5.7 (`"shift_scale"`) with their standard errors are the sufficient statistics; the channel and subject variances are estimated by restricted maximum likelihood on the Gaussian approximation (a standard two-level meta-analytic model solved by iterated generalised least squares); shrunken channel and subject estimates and their standard errors follow in closed form; the existence posterior uses the two-component mixture (response/no-response) on the shrunken channel amplitudes with the mixing proportion `π` estimated by EM. Deterministic, seconds per subject, no extra dependency.
3. **Full Bayesian option (`engine="numpyro"` or `"pymc"`, extras).** The model of step 1 in non-centred parameterisation, NUTS sampling (4 chains, 1 000 warm-up, 1 000 draws by default), `r_hat`, effective sample size, divergences and the posterior-predictive check reported in `HierarchicalFit.diagnostics` [LIT-fitting-38], [LIT-fitting-40], [LIT-fitting-44]; model comparison by leave-one-out cross-validation with Pareto-smoothed importance sampling [LIT-fitting-32]. Initialised at the empirical-Bayes solution (halves warm-up). The reference time budget is ≤ 5 min per subject for 100 channels × 10 trials × 2 components on the CI machine (REQ-176).
4. **Outputs.** Posterior (or EB) means, SDs and 95 % intervals for every parameter at every level; `P(response)` per channel; shrinkage factors; the between-channel and between-subject SDs of latency and amplitude as first-class results (they are the quantities of interest in developmental and state studies [LIT-ccep-7], [LIT-variability-11]).
5. **Covariates at any level**: a design formula (`patsy`-style, implemented with `formulaic` as an extra, or a plain design matrix) on `α_{sk}` or `θ_{sk}` (`~ age + state`), giving population-level effects with intervals; the EB engine fits them by weighted least squares with the shrunken estimates and their variances; the Bayesian engines include them in the model.

**Public API.**

```python
def fit_hierarchical(ts: TrialSet | Sequence[TrialSet], *,
                     levels: Sequence[str] = ("trial", "channel", "subject"),
                     components: Sequence[str] | None = None,
                     engine: Literal["eb", "numpyro", "pymc"] = "eb",
                     existence: bool = True,
                     formula: str | None = None,              # e.g. "amplitude ~ age + state"
                     latency_dist: Literal["exgaussian", "normal"] = "exgaussian",
                     n_chains: int = 4, n_warmup: int = 1000, n_draws: int = 1000,
                     target_accept: float = 0.9,
                     random_state=None, n_jobs: int = 1, verbose=None) -> HierarchicalFit: ...
def posterior_summary(fit: HierarchicalFit, level: str = "channel") -> pd.DataFrame
def compare_hierarchical(fits: Sequence[HierarchicalFit]) -> pd.DataFrame     # LOO table for Bayesian engines
```

`HierarchicalFit` (§4.3.4) carries `level_tables` (one tidy table per level), `p_response`, `variance_components`, `effects` (formula terms), `diagnostics`, `engine`, `provenance`; `to_arviz()` returns an `InferenceData` when `arviz` is installed.

**Edge cases.** A single subject → the subject level is dropped with a note; a channel with < 3 trusted trials gets the prior-only (fully shrunken) estimate and the flag `"prior_dominated"`; divergences > 1 % → warning with the suggestion to raise `target_accept`; `formula` referencing a missing metadata column → `SettingsError` naming it.

**Acceptance criteria (REQ-170…REQ-189).**
- REQ-170: on simulated three-level data (20 subjects × 50 channels × 20 trials, 30 % responding channels), the EB engine's `P(response)` has AUC ≥ 0.95 against the simulated truth and is calibrated (reliability-diagram slope within 0.8–1.2).
- REQ-171: EB shrunken channel latencies have lower RMSE than the per-channel §5.5 estimates on ≥ 90 % of simulated datasets at per-trial `SNR ≤ 2`.
- REQ-172: variance components (`σ_α` between channels, `σ_δ` jitter) are recovered within ±20 % by EB and within the 95 % posterior interval by NUTS in ≥ 90 % of simulated datasets.
- REQ-173: NUTS runs show `r_hat ≤ 1.01`, ESS ≥ 400 per parameter and < 1 % divergences on the simulated reference dataset with default settings.
- REQ-174: the two Bayesian engines agree with each other on every posterior mean within one posterior SD on the reference dataset.
- REQ-175: a formula term with a simulated true effect (latency ~ age, slope 0.2 ms/year) is recovered within its 95 % interval in ≥ 90 % of simulated datasets of 20 subjects.
- REQ-176: time budgets: EB ≤ 10 s per subject; NUTS ≤ 5 min per subject (100 channels × 10 trials × 2 components) on the CI machine.
- REQ-177: with `engine="eb"`, the package has no new dependency; importing `evoparam.hierarchical` without `numpyro`/`pymc` works and raises an informative `ImportError` only when a Bayesian engine is requested.
- REQ-178: on ds004080 (≥ 5 subjects), the developmental effect of age on N1 latency reported by the dataset's authors (decreasing latency with age across childhood and adolescence [LIT-ccep-7]) is recovered in sign by `formula="latency ~ age"` on the N1 component.

**Tests.** `tests/test_hier_eb.py` (170–172, 175, 176, 177), `tests/test_hier_bayes.py` (172–174, marked `slow`, runs weekly in CI), `tests/test_hier_ccep.py` (178).

**Tutorials.** `tutorials/40_hierarchical_eb.py`, `tutorials/41_hierarchical_bayes.py`, `tutorials/42_group_effects_formula.py`.

### 5.9 `evoparam.spatial` — shared shapes across channels, maps and the convergent paradigm

**Purpose.** Use the fact that one stimulation site drives many channels and one channel receives from many sites: find the small number of response shapes shared across channels (or across stimulation sites into one channel), quantify how each channel expresses them, produce brain maps of the parameters, and separate local from volume-conducted responses. The published data-driven approaches are the basis-profile-curve method (shapes shared across stimulation sites into one recording channel) [LIT-ccep-16] and its single-site, rank-one relative [LIT-ccep-15]; both are GPL and are re-implemented here from the papers (§13). Spatial screening of volume conduction and reference effects is required by the literature on sEEG CCEPs [LIT-ccep-47], [LIT-ccep-46], [LIT-ccep-18], [LIT-ccep-26].

**Methods.**
1. **Shared-shape decomposition (`shared_shapes`).** For a set of trials `V_{m,i}(t)` (`m` = group index: stimulation site into one channel, or channel for one site; `i` = trial), the cross-projection matrix `P_{(m,i),(m',i')} = ⟨V_{m,i}, V_{m',i'}⟩/‖V_{m',i'}‖` on `[t_0, τ]` is computed for all pairs with `m ≠ m'` or `i ≠ i'`; its significance structure (a `t`-statistic per group pair from the pair projections) is clustered by non-negative matrix factorisation of the positive part of the `t`-matrix [LIT-tf-45] with the number of clusters chosen by the stability of the factorisation across 20 random restarts (mean adjusted Rand index ≥ 0.8) and bounded by `max_shapes` (default 6); each cluster's basis curve is the first right singular vector of its trials; group-level weights `α_m` and explained variance `R²_m` follow by projection. Kernel PCA with a cosine kernel is the alternative clustering (`method="kpca"`). This reproduces the basis-profile-curve logic [LIT-ccep-16] from its published description; REQ-194 validates it against the published example dataset (ds003708 [LIT-ccep-36]).
2. **Multi-channel parametric model (`fit_shared`).** The parametric model of §5.5 with the shape parameters `θ_k` shared across a set of channels and only the amplitudes (and, optionally, latencies within `±max_shift`) free per channel: `y_c(t) = b_c(t) + Σ_k A_{ck} s_k(t − δ_{ck}; θ_k)`. Solved by alternating least squares (channel amplitudes by variable projection, shared `θ` by bounded least squares over the stacked residual), warm-started from the per-channel fits of §5.5. A graph-Laplacian smoothness penalty `λ Σ_{c~c'} w_{cc'} (δ_{ck} − δ_{c'k})²` over neighbouring contacts (neighbours by distance from `ch_info` `x, y, z`, default radius 10 mm) is optional (`smooth=True`), with `λ` chosen by leave-one-channel-out prediction. This is the model that makes "the same component with different latencies on neighbouring contacts" a testable statement.
3. **Maps (`parameter_map`).** For every stimulation site: per-channel tables of amplitude, latency, `P(response)` (from §5.6 or §5.8) and shape weights, joined with `ch_info` coordinates, returned as a tidy table and, with the `plot` extra, as a glass-brain or electrode-grid figure (MNI or subject space, `coord_space` honoured; no templates are shipped, the `nilearn` plotting backend is used when installed). The map carries the stimulation contacts, bad channels and masked channels as distinct markers.
4. **Local versus distant and volume conduction (`screen_volume_conduction`).** Three published screens are implemented as flags on each channel: (i) polarity reversal between adjacent contacts of the same shaft (a dipolar signature of a local source, [LIT-ccep-46]); (ii) amplitude decay with distance from the stimulation site following the published monotone decay used to flag "passive" spread [LIT-ccep-47], [LIT-ccep-50]; (iii) sensitivity of the component to the reference scheme (§5.4 `reference_sensitivity`). The combination is reported as `volume_conduction_risk ∈ {low, medium, high}` with the triggering screens listed; nothing is removed automatically.
5. **Convergent paradigm (`convergent`).** Many stimulation sites into one recording channel: the shared-shape decomposition of method 1 with groups = sites, giving the channel's "input repertoire": which sites evoke which shapes with which weights; and the reverse ("divergent": one site into all channels). Both views are the same computation with the grouping swapped, which is why they share one implementation.
6. **Connectivity export.** A `(site × channel)` matrix of any parameter (amplitude, latency, `P(response)`, shape weight) with the stimulation metadata, in long and wide form, for network analyses outside the package (graph measures are out of scope, §2); when the `mne` extra is installed, the matrix is also returned as an `mne_connectivity`-compatible array.

**Public API.**

```python
def shared_shapes(ts: TrialSet, *, by: Literal["stim_site", "channel"] = "stim_site",
                  channel: str | int | None = None, stim_site: str | None = None,
                  window: tuple[float, float] = (0.009, 0.5), max_shapes: int = 6,
                  method: Literal["nmf", "kpca"] = "nmf", n_restarts: int = 20,
                  stability_min: float = 0.8, random_state=None, n_jobs: int = 1) -> ShapeBasis: ...
def fit_shared(ts: TrialSet | Evoked, *, channels: Sequence[str] | None = None, preset=None,
               shared: Sequence[str] = ("latency", "fwhm", "asymmetry"), free_latency: bool = False,
               max_shift: float = 0.005, smooth: bool = False, radius_mm: float = 10.0,
               **model_kw) -> GroupFit: ...
def parameter_map(group_fit: GroupFit, *, parameter: str = "amplitude", stim_site: str | None = None,
                  existence: ResponseTest | HierarchicalFit | None = None) -> pd.DataFrame: ...
def screen_volume_conduction(ts: TrialSet, group_fit: GroupFit, *, reference_schemes=("bipolar", "car_adaptive"),
                             decay_model: Literal["published", "none"] = "published") -> pd.DataFrame: ...
def convergent(ts: TrialSet, channel: str, **kw) -> ShapeBasis; def divergent(ts: TrialSet, stim_site: str, **kw) -> ShapeBasis
def connectivity_matrix(group_fit: GroupFit, parameter: str = "amplitude", *, long: bool = False) -> pd.DataFrame
```

`ShapeBasis` (§4.3.4): `curves` (shapes × time), `times`, `weights` (group × shape), `r2` (per group), `assignment` (group → shape), `stability`, `n_shapes`, `method`, `provenance`; `as_template(k)` returns a `template` `ComponentShape` for §5.5 and §5.7.

**Edge cases.** Fewer than 3 groups → `DataError`; groups with fewer than 3 trials are projected but not used to define bases; unstable factorisations (stability < `stability_min` for every `n_shapes`) return `n_shapes = 1` with the warning `"unstable_basis"`; `fit_shared` with channels whose §5.5 fits disagree on `K` uses the modal `K` and flags the others; coordinates missing → maps return tables only, with a warning.

**Acceptance criteria (REQ-190…REQ-209).**
- REQ-190: on simulated multi-site data with 3 true shapes and 20 sites, `shared_shapes` recovers `n_shapes = 3` in ≥ 90 % of runs and the recovered curves correlate ≥ 0.95 with the true ones (after matching).
- REQ-191: `fit_shared` reduces the latency RMSE across 30 simulated channels sharing one shape by ≥ 30 % relative to independent fits at per-channel `SNR = 2`.
- REQ-192: the graph-Laplacian penalty with `λ` chosen by leave-one-channel-out never increases the held-out prediction error by more than 1 % relative to `λ = 0` (it can only help or be switched off).
- REQ-193: `screen_volume_conduction` flags `high` on ≥ 90 % of simulated channels whose signal is a scaled copy of the stimulation-site waveform with distance-decayed amplitude and no polarity reversal, and `low` on ≥ 90 % of simulated local dipolar sources.
- REQ-194: `shared_shapes(by="stim_site")` on the public example dataset of the basis-profile-curve paper (ds003708 [LIT-ccep-36]) yields the same number of bases as published for the example channel and an assignment of sites to bases agreeing with the published figure on ≥ 85 % of sites.
- REQ-195: `connectivity_matrix` round-trips to and from long form and matches `GroupFit.to_dataframe()` cell by cell.
- REQ-196: `parameter_map` on ds004080 (one subject) produces a table with one row per `(stim_site, channel)` pair, coordinates from the BIDS `electrodes.tsv`, and no NaN in `amplitude` for channels with `status == "ok"`.

**Tests.** `tests/test_spatial_sim.py` (190–193), `tests/test_spatial_bpc.py` (194, cached sample), `tests/test_spatial_tables.py` (195, 196).

**Tutorials.** `tutorials/50_shared_shapes.py`, `tutorials/51_maps_and_connectivity.py`, `tutorials/52_volume_conduction_screen.py`.

### 5.10 `evoparam.tf` — time-frequency parameterisation of evoked responses

**Purpose.** Parameterise the stimulation-evoked time-frequency response (the quantity behind "cortico-cortical spectral responses" and "stimulation-induced spectral responses" in the epilepsy literature [LIT-tf-1], [LIT-tf-2], [LIT-tf-3], [LIT-tf-5], [LIT-tf-28], [LIT-tf-30]) with explicit, reproducible choices for the transform, the baseline normalisation and the statistics; separate the time-frequency footprint of the fitted evoked components from the residual (induced or sustained) activity; find the responsive time-frequency regions from the data rather than from hand-drawn boxes; and keep a compatibility layer that reproduces published fixed-zone definitions exactly.

**Methods.**
1. **Transforms** (`transform=`): `"morse"` (generalised Morse wavelets with `γ = 3` and `β = P²/γ`, default `P² = 30` so `β = 10`, the parameters used in the stimulation-induced spectral response work [LIT-tf-2], [LIT-tf-24]), `"morlet"` (complex Morlet with `n_cycles` fixed or proportional to frequency), `"multitaper"` (DPSS, time-bandwidth product `NW`), `"stft"`. Frequencies on a log grid (default 2–250 Hz, 40 bins) or a user grid; the cone of influence is computed and masked for wavelets; the stimulation-artifact mask (§5.3) is propagated to the time-frequency plane as a time mask widened by the wavelet's temporal support at each frequency (so that artifact energy cannot leak into "high gamma" at the window start — the reason the published pipelines start their high-frequency windows after 10–50 ms [LIT-tf-2], [LIT-tf-28]).
2. **Normalisation** (`baseline_mode=`): `"db"` (`10·log10(P/P̄_B)`), `"percent"`, `"zscore"`, `"logratio"`, `"ratio"`, computed either on the average power (`"evoked"`) or per trial before averaging (`"single_trial"`, the normalisation that reduces sensitivity to noisy trials [LIT-tf-12], with the bias caveats of [LIT-tf-36]); `baseline` default `(−0.5, −0.1)` s [LIT-tf-2]; `"full_epoch"` normalises each trial by its own full-epoch mean (for paradigms without a clean baseline). The mode is recorded in provenance and in the methods text.
3. **Evoked footprint versus residual.** Three maps are produced: `total` (time-frequency of the trials), `evoked_footprint` (time-frequency of the §5.5 fitted model `ŷ(t)`, i.e. what the broadband components alone would produce), and `residual` (time-frequency of `x_i(t) − ŷ(t)` per trial, averaged). The residual map is where induced oscillations and sustained high-frequency changes live; the footprint map is what a sharp N1 produces in the gamma range by its mere sharpness, which is otherwise confounded with "induced high gamma" [LIT-tf-21], [LIT-tf-22], [LIT-tf-23]. The decomposition is exact (the transform is linear in the signal for the complex coefficients; power is not, and the cross term is reported as `cross`).
4. **Statistics.** Pixel-wise `t` against zero of the normalised map across trials, with (a) Benjamini–Hochberg false-discovery-rate control over pixels (the published pipeline's choice [LIT-tf-2]), (b) cluster-based permutation (sign flips, cluster mass, [LIT-tf-10]; interpretation caveats [LIT-tf-37]), or (c) threshold-free cluster enhancement with permutation (`E = 0.5, H = 2`, [LIT-tf-38], [LIT-tf-39]); `n_permutations` default 1 000; corrections across channels as in §5.6.
5. **Data-driven zones.** Instead of hand-defined boxes, the significant region of the group-average map is decomposed into a small number of time-frequency parts by (a) non-negative matrix factorisation of the positive and negative maps separately [LIT-tf-45], or (b) a rank-constrained canonical-polyadic tensor decomposition over `(time × frequency × channel)` or `(time × frequency × trial)` [LIT-tf-42], [LIT-tf-43], [LIT-tf-44], with the rank chosen by split-half stability (≥ 0.8 congruence over 20 restarts, bounded by `max_zones = 8`). Each zone is parameterised by its time centroid and extent, frequency centroid and extent (log scale), signed peak and mean change, onset and offset (first/last significant time at the frequency centroid), and the fraction of significant energy it explains. These are the quantities a paper reports; they are computed identically for ECoG and sEEG and carry the `modality` column so that the known differences between the two (reference, spatial extent, amplitude scale [LIT-tf-55], [LIT-tf-56], [LIT-tf-58], [LIT-tf-63]) can be modelled as covariates in §5.11 rather than hidden in different defaults.
6. **Compatibility layer (`zones="published:<name>"`).** The fixed zones of published work are shipped as a registry (`evoparam/tf/zones.json`) with their exact bounds and the citation, each field flagged `verified` or `unverified`; the one definition read in full and verified is the "delayed 50–500 ms × high gamma 50–250 Hz" zone of the dissertation-based parameter study [LIT-tf-3], [LIT-tf-4]; the other ten zones of that work are included as `unverified` with the page reference and must be confirmed by the user before publication (the registry refuses to run an unverified zone without `allow_unverified=True`, REQ-217). The metric per zone is the signed mean normalised power with its trial-level `t` and the fraction of significant pixels, reproducing the published summary statistic.
7. **Single-trial time-frequency parameters.** With `by_trial=True`, every zone's metric is computed per trial, giving a trial table joinable with §5.7's (same `trial` index), which is how the trial-to-trial variability of spectral responses and its relation to the N1 are studied [LIT-ccep-9].

**Public API.**

```python
def tf_fit(ts: TrialSet, *, transform: Literal["morse", "morlet", "multitaper", "stft"] = "morse",
           freqs: np.ndarray | tuple[float, float, int] = (2.0, 250.0, 40),
           gamma: float = 3.0, p2: float = 30.0, n_cycles: float | np.ndarray = 7.0, time_bandwidth: float = 4.0,
           window: tuple[float, float] = (0.0, 1.0), baseline: tuple[float, float] = (-0.5, -0.1),
           baseline_mode: Literal["db", "percent", "zscore", "logratio", "ratio"] = "db",
           baseline_level: Literal["evoked", "single_trial", "full_epoch"] = "single_trial",
           group_fit: GroupFit | None = None,                 # enables the footprint/residual split
           stats: Literal["fdr", "cluster", "tfce", "none"] = "fdr", alpha: float = 0.05, n_permutations: int = 1000,
           zones: Literal["auto", "none"] | str = "auto", max_zones: int = 8, zone_method: Literal["nmf", "cp"] = "nmf",
           allow_unverified: bool = False, by_trial: bool = False,
           picks=None, by: Sequence[str] = ("stim_site",), n_jobs: int = 1, random_state=None, verbose=None) -> TFFit: ...
def tf_zones(tf: TFFit, *, method="nmf", max_zones=8) -> pd.DataFrame
def list_published_zones() -> pd.DataFrame
def tf_compare(tf_a: TFFit, tf_b: TFFit, *, stats="cluster", n_permutations=1000) -> pd.DataFrame   # condition/modality contrast
```

`TFFit` (§4.3.4): `freqs`, `times`, `maps` (`total`, `evoked_footprint`, `residual`, `cross`; channels × freqs × times), `tmaps`, `significant`, `zones` (tidy table), `zone_metrics` (per channel and, with `by_trial`, per trial), `settings`, `provenance`; `to_mne()` returns `mne.time_frequency.AverageTFR` when available.

**Edge cases.** `window` starting before the artifact mask ends → the masked region is excluded from statistics and zones, never filled; frequencies above `sfreq/2.5` → dropped with a warning; fewer than 10 trials with `stats="cluster"` → warning (permutation resolution); `baseline_level="full_epoch"` with `baseline_mode="zscore"` → `SettingsError` (undefined); a `group_fit` on a different channel set → `SettingsError`.

**Acceptance criteria (REQ-210…REQ-229).**
- REQ-210: the Morse transform with `γ = 3, β = 10` matches an independent reference implementation (`ssqueezepy` or an analytic check on a chirp) to 1e-6 relative on the modulus; cone-of-influence masking is applied.
- REQ-211: `evoked_footprint + residual + cross == total` to 1e-9 for every pixel (exactness of the decomposition).
- REQ-212: on simulated trials with a sharp N1 (FWHM 8 ms) and no induced activity, the residual map shows no significant high-gamma (50–250 Hz) pixels in ≥ 95 % of runs, while the total map does in ≥ 80 % (the confound is demonstrated and removed).
- REQ-213: on simulated induced gamma bursts (80 Hz, 100 ms, 20 % power increase, 30 trials), zone recovery: frequency centroid within ±10 %, time centroid within ±20 ms, in ≥ 90 % of runs with `zone_method="nmf"`.
- REQ-214: pixel-wise type-I error with `stats="fdr"` ≤ 0.06 on null simulations with AR(1) trials; `"cluster"` and `"tfce"` family-wise error ≤ 0.06.
- REQ-215: `baseline_level="single_trial"` reduces the influence of one injected high-power trial on the average map by ≥ 50 % relative to `"evoked"` (measured as the change in the zone metric).
- REQ-216: on ds004080 (≥ 5 subjects), the published zone `delayed 50–500 ms × 50–250 Hz` computed by the compatibility layer with the published transform and baseline settings is monotonically related to the N1 amplitude across sites (Spearman `|ρ| ≥ 0.3` on at least half of the subjects) — a sanity relation reported in the stimulation-induced spectral response work [LIT-tf-2]; the test documents the sign.
- REQ-217: running an `unverified` published zone without `allow_unverified=True` raises `SettingsError` naming the zone and its source.
- REQ-218: speed: 100 channels × 50 trials × 2 048 samples × 40 frequencies with `stats="fdr"` in ≤ 60 s with `n_jobs=1`.
- REQ-219: `TFFit.methods_text()` names the transform and its parameters, the baseline window and mode, the statistic, `α` and the zone method.

**Tests.** `tests/test_tf_transform.py` (210, 211), `tests/test_tf_sim.py` (212–215), `tests/test_tf_ccep.py` (216), `tests/test_tf_zones_registry.py` (217), `benchmarks/bench_tf.py` (218), `tests/test_tf_text.py` (219).

**Tutorials.** `tutorials/60_spectral_response.py` (one site: total, footprint, residual; data-driven zones; published zone for comparison), `tutorials/61_ecog_vs_seeg.py` (the same pipeline on an ECoG and an sEEG dataset of the registry with `modality` as a covariate), `tutorials/62_single_trial_tf.py` (zone metric per trial joined with N1 per trial).
### 5.11 `evoparam.stats` — intervals, comparisons, reliability and export to mixed models

**Purpose.** Turn parameter tables into the numbers a paper reports: confidence intervals, condition and group contrasts, reliability, trial-count planning, and clean exports for mixed-effects modelling in other software. The ERP measurement literature specifies these procedures completely, but almost only in MATLAB [LIT-erpmeas-36], [LIT-erpmeas-37], [LIT-erpmeas-45]; Python has no reliability or trial-count tooling [LIT-erpmeas-37], [LIT-erpmeas-41].

**Methods.**
1. **Bootstrap over trials** (`bootstrap`): `B` resamples (default 1 000) of trials with replacement within each `by` group, any statistic computed by a callable (a `ResponseModel` refit, a classical measure, a zone metric); percentile and bias-corrected accelerated (BCa) intervals; stratification by `stim_site`/`condition` keeps group sizes fixed [LIT-fitting-28].
2. **Jackknife over participants** (`jackknife_latency`): leave-one-out grand averages, scoring on each, corrected statistics `t_c = t/(n−1)`, `F_c = F/(n−1)²` [LIT-erpmeas-2], [LIT-erpmeas-4], [LIT-erpmeas-5]; retrieval of individual estimates `o_i = n·J̄ − (n−1)·J_i` so that ordinary statistics can be run on them [LIT-erpmeas-6].
3. **Condition comparison** (`compare_conditions`): for any parameter, paired or unpaired contrasts by trial-label permutation (default 5 000 permutations) at the level the user names (`unit="trial" | "channel" | "subject"`); effect sizes (Hedges' `g`, latency difference in ms with interval); multiple-comparison corrections as in §5.6. Permutation at the correct exchangeability level is enforced: a contrast between subjects cannot permute trials (`SettingsError`).
4. **Reliability** (`reliability`): split-half (odd/even and 1 000 random splits) with Spearman–Brown correction; dependability `Φ` from variance components (trials within subjects, subjects) estimated by restricted maximum likelihood, and the decision study (D-study) curve "trials needed for `Φ ≥ 0.70`" [LIT-erpmeas-36], [LIT-erpmeas-38]; the standardised measurement error per score (§5.5.6) [LIT-erpmeas-41]. For intracranial data the "subject" facet is replaced by "stimulation train" so that the F-TRACT-style train-to-train reproducibility [LIT-ccep-14] is quantified on the same scale.
5. **Trial-count planning** (`trials_needed`): by resampling the user's own data at `n = 5, 10, …, N` trials, the probability that the existence test (§5.6) is significant and the SD of each parameter, extrapolated with the `1/√n` law beyond `N` and reported with the extrapolation flagged [LIT-erpmeas-43].
6. **Export to mixed models** (`to_long`): one row per trial × channel × component × parameter, with every metadata column, the parameter's standard error and the weights `1/se²`, in tidy long or wide format, plus a generated model formula string for `statsmodels`, `lme4` (R) and `brms` (R) [LIT-fitting-36], [LIT-fitting-37]. The package itself fits only the linear mixed model of `statsmodels` when installed (`fit_mixed`), because a full mixed-model engine is out of scope (§2).
7. **Group summaries** (`group_summary`): mean, median, SD, interquartile range and bootstrap interval of each parameter per channel group, region (`ch_info["region"]`), subject and condition; counts of `status` values so that non-responses are never silently averaged.

**Public API.**

```python
def bootstrap(ts: TrialSet, statistic: Callable[[TrialSet], float | np.ndarray | pd.Series], *,
              n_boot: int = 1000, by: Sequence[str] = ("stim_site",), ci_level: float = 0.95,
              method: Literal["percentile", "bca"] = "bca", random_state=None, n_jobs: int = 1) -> pd.DataFrame
def jackknife_latency(evokeds: Sequence[Evoked], *, scorer: Callable[[Evoked], float] | str = "fractional_area_latency",
                      retrieve: bool = True, **scorer_kw) -> pd.DataFrame
def compare_conditions(table: pd.DataFrame, *, parameter: str, condition: str, unit: str = "subject",
                       paired: bool = True, n_permutations: int = 5000, correction: str = "fdr_bh",
                       random_state=None) -> pd.DataFrame
def reliability(ts: TrialSet, *, scorer: str | Callable = "mean_amplitude", facets: Sequence[str] = ("trial", "subject"),
                method: Literal["split_half", "dependability", "both"] = "both", n_splits: int = 1000,
                target_phi: float = 0.70, random_state=None) -> pd.DataFrame
def trials_needed(ts: TrialSet, *, scorer="existence", steps: Sequence[int] | None = None, n_resamples: int = 200,
                  random_state=None) -> pd.DataFrame
def to_long(result: GroupFit | TrialFit | HierarchicalFit | TFFit, *, wide: bool = False) -> pd.DataFrame
def mixed_formula(table: pd.DataFrame, *, response: str, fixed: Sequence[str], groups: Sequence[str],
                  engine: Literal["statsmodels", "lme4", "brms"] = "lme4") -> str
def fit_mixed(table: pd.DataFrame, formula: str, groups: str) -> "statsmodels MixedLMResults"
def group_summary(table: pd.DataFrame, *, by: Sequence[str], parameters: Sequence[str] | None = None) -> pd.DataFrame
```

**Edge cases.** Fewer than 3 participants for the jackknife → `DataError`; BCa with fewer than 20 unique resample values → falls back to percentile with a warning; `reliability` with one trial per subject → `DataError`; `compare_conditions` on a table with `status != "ok"` rows → those rows are excluded and counted in the output's `n_excluded`.

**Acceptance criteria (REQ-230…REQ-249).**
- REQ-230: BCa and percentile intervals reach 93–97 % coverage on simulated normal and log-normal statistics with 30 trials.
- REQ-231: `jackknife_latency` reproduces the corrected statistics of the reference examples in [LIT-erpmeas-2] and [LIT-erpmeas-6] (worked numerical examples) to the printed precision, and retrieved individual latencies equal the direct per-subject latencies on noise-free simulations to 1e-9.
- REQ-232: `compare_conditions` keeps the type-I error within `[0.04, 0.06]` at `α = 0.05` on null simulations at each unit level, and refuses invalid exchangeability (`unit="trial"` across subjects).
- REQ-233: `reliability` reproduces split-half and dependability values computed by an independent implementation of the published formulas on an ERP CORE component (ERN or N170) [LIT-erpmeas-39] within ±0.01.
- REQ-234: `trials_needed` predicts, from the first 20 trials of a simulated channel, the parameter SD at 80 trials within ±15 %.
- REQ-235: `to_long` output has no nested objects, one value per cell, and a column dictionary (JSON sidecar) describing every column with its unit (REQ-342 reuses it for BIDS derivatives).
- REQ-236: `mixed_formula` strings parse in `statsmodels` (tested) and match frozen templates for `lme4` and `brms`.
- REQ-237: `group_summary` reports, for every group, the count of each `status` value, so that `n_ok + n_no_response + n_failed + n_skipped == n_channels`.

**Tests.** `tests/test_stats_bootstrap.py` (230), `tests/test_stats_jackknife.py` (231), `tests/test_stats_compare.py` (232), `tests/test_stats_reliability.py` (233, 234; ERP CORE subset cached), `tests/test_stats_export.py` (235–237).

**Tutorials.** `tutorials/70_condition_contrast.py`, `tutorials/71_reliability_and_trials_needed.py`, `tutorials/72_export_to_mixed_models.py`.

### 5.12 `evoparam.sim` — simulators with ground truth

**Purpose.** Provide the ground truth on which every acceptance criterion of §5.5–§5.11 and the benchmark of §8 are computed, for every supported modality, with noise that looks like real data. The simulator must not import the fitter (§4.2) so that the truth cannot be shaped by the estimator. The inherited simulator depended on a spectral-noise package and generated only Gaussian bumps [CODE §0], [CODE §1.5]; it is rewritten.

**Generators.** Each returns a `TrialSet` and a `GroundTruth` (a dataclass with the exact component list per trial and channel, the noise parameters, the artifact parameters, and the random seed).
1. `simulate_erp(n_trials=60, n_channels=1, sfreq=500., tmin=-0.2, tmax=0.8, components=[...], jitter_sd=0., amplitude_cv=0., noise="pink", snr=5., overlap=None, seed=None)`: sums of shapes from `evoparam.shapes` with per-trial latency jitter (normal or ex-Gaussian), amplitude variability (log-normal), optional overlap from a neighbouring event at a given stimulus-onset asynchrony (the case overlap-deconvolution tools address [LIT-erpmeas-31]); component presets for P1, N1, N170, P3, N400, MMN, ERN (latency/width ranges from ERP CORE [LIT-erpmeas-39]).
2. `simulate_ccep(n_trials=10, n_channels=46, sfreq=2048., stim_current_mA=4., pulse_width_us=200., biphasic=True, n1=(latency 15–40 ms, fwhm 5–15 ms), n2=(latency 80–250 ms, fwhm 40–150 ms), p_response=0.3, delayed=False, artifact=True, saturation=False, train_structure=None, seed=None)`: the artifact is a scaled biphasic pulse convolved with the amplifier's step response plus a double-exponential recovery tail whose amplitude decays with electrode distance (`ch_info` coordinates simulated on shafts or grids); responding channels and their parameters are drawn from distributions spanning the published ranges [LIT-ccep-1], [LIT-ccep-3], [LIT-ccep-7]; `delayed=True` adds delayed responses (100 ms–1 s, [LIT-ccep-19]); `saturation=True` clips the first samples at the rails; trains of 10 pulses at 0.2–1 Hz with an optional adaptation trend. This generator is what makes REQ-113-like criteria testable with a known truth.
3. `simulate_erna(n_pulses, sfreq=4096 or higher, f=(200, 450) Hz, tau, pulse_rate=130 Hz, artifact=True)`: damped oscillation after each DBS pulse plus an artifact repeating at the stimulation rate [LIT-domains-52], [LIT-domains-53], [LIT-fitting-4].
4. `simulate_tep(...)`: TEP components N15, P30, N45, P60, N100, P180 with the latency ranges of [LIT-domains-1], a TMS pulse artifact (large, 5–10 ms), a decay artifact, a muscle artifact (10–30 ms, high-frequency) and an auditory-evoked confound (N100/P200 from the click) [LIT-domains-12].
5. `simulate_ecap(...)`: spinal-cord or cochlear compound action potentials (`biexponential` N1–P2 complex at 0.3–1.5 ms) with alternating-polarity and forward-masking paradigms so that §5.3 method 9 can be evaluated [LIT-domains-37], [LIT-domains-61].
6. `simulate_animal_ep(species="mouse", ...)`: visual or electrical evoked LFP on a laminar probe with depth-dependent polarity reversal and state-dependent gain (up to ×3 between states [LIT-variability-50]), sampled at 1 250 Hz like the public Neuropixels LFP [LIT-datasets-50].
7. **Noise models** (`noise=`): `"white"`, `"ar1"` (`ρ`), `"pink"` (spectral exponent `χ ∈ [0, 2]`, generated by filtering white noise with `scipy.signal` — no external package), `"real"` (segments of real baselines injected under the simulated components: ds004080 pre-stimulus segments for intracranial data, ERP CORE pre-stimulus segments for scalp data, read through §5.1 with their licences recorded in the ground truth; this follows the recommendation to benchmark scoring methods on real-noise injection rather than on private simulations [LIT-erpmeas-2], [LIT-erpmeas-43], [LIT-erpmeas-44]), `"line"` (50/60 Hz and harmonics added to any of the others).
8. **Ground truth export**: `GroundTruth.to_dataframe()` in the same tidy schema as `GroupFit.to_dataframe()` so that scoring is a join.
9. **Scorers** (`evoparam.sim.score`): `match_components(truth, fit, tolerance="fwhm/2")` (Hungarian matching on latency), recall, precision, F1, latency/amplitude/width errors, interval coverage; used by tests and by §8.

**Public API.** The six generators above (signatures as listed, all keyword-only after the first, every one accepting `seed: int | np.random.Generator | None`), `GroundTruth`, `make_noise(n_times, sfreq, kind, **kw)`, `inject(ts_real: TrialSet, components, *, snr, channels=None, seed=None) -> (TrialSet, GroundTruth)`, `match_components`, `score_fits(truth, fits) -> pd.DataFrame`.

**Edge cases.** `snr` is defined everywhere as peak component amplitude over the per-trial noise SD measured on the simulated baseline; requesting an SNR for a channel without a component raises `SettingsError`; `"real"` noise with a sampling rate different from the target is resampled by §5.4 `resample` and the resampling recorded.

**Acceptance criteria (REQ-250…REQ-269).**
- REQ-250: every generator is deterministic for a given `seed` (byte-identical output) across platforms in CI.
- REQ-251: `make_noise("pink", χ)` produces a spectral exponent within ±0.1 of `χ` (log–log slope fitted over 2–100 Hz on 100 realisations); `"ar1"` has lag-1 autocorrelation within ±0.02 of `ρ`.
- REQ-252: the measured SNR of every generated component equals the requested SNR within ±5 %.
- REQ-253: `simulate_ccep` artifacts are removed by `remove_artifact(method="template")` to a residual RMS within the artifact window ≤ 10 % of the pre-removal RMS (sanity link between §5.3 and §5.12).
- REQ-254: `sim` does not import `evoparam.models` or anything that imports it (import-graph test, §4.2).
- REQ-255: `GroundTruth.to_dataframe()` and `GroupFit.to_dataframe()` share column names and units for every common parameter.
- REQ-256: the `"real"` noise option records dataset id, version, subject, run and sample offsets of every injected segment in the ground truth, and refuses datasets whose licence in the registry (§5.1) is not CC0, CC BY or CC BY-SA.

**Tests.** `tests/test_sim_determinism.py` (250), `tests/test_sim_noise.py` (251, 252), `tests/test_sim_artifact.py` (253), `tests/test_import_graph.py` (254), `tests/test_sim_truth.py` (255, 256).

**Tutorials.** `tutorials/80_simulate_and_score.py`, `tutorials/81_real_noise_injection.py`.

### 5.13 Reports, plots, command line and online mode (`evoparam.reports`, `evoparam.plots`, `evoparam.cli`, `evoparam.stream`)

**Purpose.** Make every result self-describing and citable, viewable in one figure, runnable without writing Python, and usable online during a recording.

**Reports (`evoparam.reports`).**
1. **Methods text.** `report.methods_text(result)` assembles one paragraph per processing step from `Provenance.steps` (events, artifact, preparation, fit, existence, trials, hierarchical, tf, stats), with every setting that changes a number, the package version, the random seed, and the citation of the package and of every published method used (e.g. the registry rule of §5.6, the transform of §5.10). The paragraph is deterministic (frozen templates; REQ-123, REQ-140, REQ-219) and is printed by default at the end of `evoparam.fit` when `verbose` is INFO, because citation follows from what users paste into their methods [LIT-citation-19], [LIT-citation-30] (§11).
2. **Machine-readable report.** `report.to_json(result)` writes the full result (parameters, uncertainties, status, warnings, settings, provenance) under the versioned schema `evoparam-result/1` (§4.3.5), validated by a JSON Schema shipped with the package.
3. **BIDS derivatives.** `io.write_bids_derivatives` (§5.1; tree, file names and columns in §10.2) writes `*_desc-evoparam_responses.tsv`, `*_desc-evoparam_existence.tsv`, `*_desc-evoparam_trials.tsv` and `*_desc-evoparam_artifact.json` with JSON sidecars describing each column (from §5.11 REQ-235; time-frequency zones are exported through `to_json` and `stats.to_long`) and a `dataset_description.json` with `GeneratedBy` (name, version, code URL when public) and `SourceDatasets`; there is no BIDS standard for evoked-response parameters yet (the electrophysiology-derivatives proposal is stalled [LIT-standards-14], [LIT-standards-16]), so the layout follows the general derivatives rules [LIT-standards-12] and is documented as a candidate for that proposal (§10).
4. **Human-readable report.** `report.html(result, path)` (single self-contained HTML file: summary table, per-channel figures, warnings, methods text, provenance) and `report.pdf(...)` when `matplotlib` is installed (one page per stimulation site); the PDF is the artefact a clinical research team would archive, and the stepping stone to the paid clinical reporting of §13 (which is not part of the free package).
5. **Quality summary.** `report.qc(result)`: counts of `status`, warnings by type, masked fraction, artifact quality scores, ill-conditioned fits, untrusted trials — the table a reviewer asks for.

**Plots (`evoparam.plots`, extra `plot`).** `plot_fit(fit)` (data, model, components, background, residual, masks shaded, intervals as bands), `plot_group(group_fit, kind="grid"|"shaft"|"brain")`, `plot_trials(trial_fit)` (trial image sorted by latency, jitter histogram), `plot_existence(test)` (statistic over time with the null band), `plot_tf(tf)` (total/footprint/residual with zones outlined), `plot_shapes(basis)`, `plot_artifact(artifact_result)` (before/after, template, quality), `plot_compare(fits)`. All functions return the `Figure`, accept `ax=`, never call `show()`, and use colour-blind-safe palettes.

**Command line (`evoparam.cli`, entry point `evoparam`).** The sub-commands, their options, the exit codes and the `--json` summary are specified in §10.5 (REQ-322, REQ-323); in summary:

```
evoparam fit <bids_root> --preset ccep --sub 01 [--ses ..] [--task ..] [--run ..] --out derivatives/   # read → events → artifact → fit → existence → derivatives
evoparam test <bids_root> --sub 01 --method baseline_permutation --out derivatives/                     # existence test only (§5.6)
evoparam report <derivatives or result.evoparam> --format html|pdf|md
evoparam simulate --modality ccep --n-trials 10 --seed 0 --out sim/
evoparam benchmark run --suite core|perf|accuracy|artifact [--out results/] ; evoparam benchmark score ...   # §8, §9.1
evoparam presets list | show <name>
evoparam validate-bids <bids_root> [--strict]                                                            # stimulation-metadata validator (§5.1, §10.2, REQ-314)
evoparam --version ; evoparam cite                                                                       # version with the research-use statement; citation and BibTeX
```

The other sub-commands (`detect-events`, `clean`, `reproduce`, `migrate`, `info`) are listed in §10.5. Every command writes the same outputs as the Python API (no CLI-only behaviour), returns the exit codes of §10.5, and accepts `--config` with the same keys as the Python keyword arguments.

**Online mode (`evoparam.stream`).** For intra-operative mapping and closed-loop studies. The class `StreamFitter`, its `PulseEstimate` output, the fixed-shape method, the LSL adapters (extra `stream`, `pylsl`) and the latency budget are specified in §10.6 (REQ-324, REQ-325); in summary, `StreamFitter(template, *, sfreq, ch_names, ...)` takes a channel template fitted offline (a `ResponseFit` or `GroupFit`), receives samples through `push` (from `LSLInlet` or any array source), and returns per pulse and channel the shift and scale of the frozen template (the `"shift_scale"` trial model of §5.7), the derived amplitude and latency, and a per-pulse existence score; the artifact window is handled by the one-sided `interpolate` method of §5.3. In addition, `StreamFitter` keeps a running existence statistic per channel: a sequential test (cumulative sign-flip permutation on the pulses received so far, with an alpha-spending rule so that repeated looks do not inflate the false-positive rate). No closed-loop control is shipped: the module exposes the numbers; acting on them is the user's (or the paid product's) responsibility (§13).

**Acceptance criteria (REQ-340…REQ-349; the container, settings and provenance requirements REQ-330…REQ-338 are in §9.12).**
- REQ-340: `methods_text` output for the three reference pipelines (ERP CORE P3, ds004080 CCEP, simulated TEP) matches frozen texts; every setting differing from the preset appears in the text.
- REQ-341: `to_json` output validates against the shipped JSON Schema (validation uses `jsonschema` from the `validate` extra and always runs in CI; without the extra, `to_json` writes but does not validate); `load` of the JSON reproduces `to_dataframe()` exactly.
- REQ-342: BIDS derivatives pass `bids-validator` in derivatives mode (CI job with the validator installed); every TSV column has a sidecar description and unit.
- REQ-343: `report.html` renders without network access (all assets inline) and is ≤ 20 MB for 100 channels.
- REQ-344: every `plots` function runs headless (Agg backend) on every result type in CI and returns a `Figure`.
- REQ-345: every CLI command has a test invoking it through `subprocess` on the cached sample data; outputs are byte-identical to the Python API with the same seed.
- REQ-346: `evoparam cite` prints the citation from `CITATION.cff` (single source of truth, §11).
- REQ-347: `StreamFitter` meets the latency budget and the offline agreement of REQ-324 (§10.6: < 20 ms at the 99th percentile for 64 channels at 2 048 Hz on one core of the reference machine of §9.1; amplitude and latency within 1 % relative and one sample of `fit_trials(mode="shift_scale")`), and the running existence statistic gives the same decision as the offline sign-flip test on the same pulses when it is evaluated once at the end of the stream.
- REQ-348: the sequential existence statistic keeps the overall false-positive rate ≤ `α` (+0.01) across 50 looks on null streams.
- REQ-349: no function in `reports`, `plots`, `cli` or `stream` computes a scientific quantity that is not available through the Python API of §5.1–§5.12 (reviewed by the import-graph rule and a test that greps for numerical code in these modules).

**Tests.** `tests/test_reports_text.py` (340), `tests/test_reports_json.py` (341), `tests/test_bids_derivatives.py` (342), `tests/test_reports_html.py` (343), `tests/test_plots.py` (344), `tests/test_cli.py` (345, 346), `tests/test_stream.py` (347, 348), `tests/test_layering.py` (349).

**Tutorials.** `tutorials/90_report_and_methods_text.py`, `tutorials/91_command_line.py`, `tutorials/92_online_mapping_with_lsl.py` (simulated stream).
## 6. Presets

A preset is a frozen `Preset` dataclass (§4.3.4) that fixes every setting a user would otherwise have to choose, for one recording modality. Presets are data, not code (`evoparam/presets/*.json`, read with the standard-library `json` module so that the core keeps only NumPy and SciPy as dependencies, loaded into dataclasses), so that a reviewer can read them and a paper can cite "preset `ccep`, evoparam x.y". Every value below is justified by a source or by a rule stated in this specification; values that are a design choice of this specification rather than a published convention are marked *(choice)* with the reason, and each such choice is a parameter of the benchmark of §8 so that it is tested, not assumed.

General rules applied to all presets.
- `min_fwhm` is the larger of the narrowest published component width for the modality and `3/sfreq_typical` (three samples), so that a component is never narrower than the sampling can resolve.
- `max_fwhm` is about half of the fit window, so that no single component can absorb the whole window.
- `polarity="both"` everywhere: the polarity of evoked components depends on the montage and on the electrode position relative to the source (sEEG polarity ambiguity is explicit in [LIT-ccep-8]; montage changes N1 latency from 47.4 ms to 70.7 ms in [LIT-ccep-27]), so the fitter never filters by sign; the labelling (§5.5.7) uses sign only to name.
- `existence_method="baseline_permutation"` with `existence_alpha=0.05` and FDR across channels everywhere, because it is the only method with a calibrated false-positive rate on every modality (§5.6, REQ-130); the published threshold rules of §5.6 are available on request and are reported side by side by `compare_detectors`.
- Every preset sets `selection="bic"`, `delta=10`, `uncertainty="jacobian"` (§5.5.4–5.5.5).

### 6.1 Preset table

| field | `erp` | `ccep` | `ccep_seeg` | `sep` | `tep` | `erna` | `ecap` (SCS) | `ecap_ci` | `rodent_ep` | `primate_ep` | `p300_bci` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| modality | scalp EEG/MEG | ECoG (SPES) | sEEG (SPES) | scalp/epidural SEP | TMS-EEG | DBS LFP | epidural SCS | cochlear implant | LFP/ECoG | LFP/ECoG | scalp EEG |
| species | human | human | human | human | human | human | human | human | mouse, rat | macaque | human |
| epoch (s) | (−0.2, 0.8) | (−1.0, 2.0) | (−1.0, 2.0) | (−0.05, 0.1) | (−1.0, 1.0) | (−0.002, 0.0077) per pulse | (−0.0005, 0.003) | (−0.0002, 0.002) | (−0.1, 0.5) | (−0.1, 0.5) | (−0.2, 0.8) |
| baseline (s) | (−0.2, 0) | (−0.5, −0.02) | (−0.5, −0.02) | (−0.05, −0.005) | (−0.8, −0.1) | per train: last 50 ms before the train | (−0.0005, −0.00005) | none (paradigm subtraction) | (−0.1, −0.005) | (−0.1, −0.005) | (−0.2, 0) |
| artifact window (s) | none | (−0.002, 0.008) | (−0.002, 0.008) | (−0.0005, 0.003) | (−0.002, 0.012) | (−0.0005, 0.0015) | (−0.00005, 0.0003) | (−0.00005, 0.0001) | stimulus-dependent; (−0.001, 0.002) for electrical | as rodent | none |
| artifact method | none | `template` + `tail="joint"` | `template` + `tail="joint"` | `interpolate` | `interpolate` then `subspace` | `period_template` | `paradigm` (alternating polarity) | `paradigm` (forward masking) | `template` | `template` | none |
| reference | as recorded | `car_adaptive` | `bipolar` | as recorded | average | bipolar (adjacent contacts) | as recorded | as recorded | as recorded | as recorded | as recorded |
| fit window (s) | (0.05, 0.7) | (0.009, 0.5) | (0.009, 0.5) | (0.010, 0.060) | (0.012, 0.3) | (0.0015, 0.0077) | (0.0003, 0.003) | (0.0001, 0.0015) | (0.005, 0.3) | (0.005, 0.3) | (0.2, 0.7) |
| shapes | gaussian, skewed_gaussian | skewed_gaussian, gaussian | skewed_gaussian, gaussian | gaussian, skewed_gaussian | gaussian, skewed_gaussian, gabor | damped_oscillation | biexponential, gaussian | biexponential, gaussian | gamma, skewed_gaussian | gamma, skewed_gaussian | gaussian, skewed_gaussian |
| background | constant | exponential | exponential | linear | double_exponential | exponential | exponential | linear | exponential | exponential | constant |
| max components | 6 | 6 | 6 | 5 | 8 | 2 | 3 | 2 | 5 | 5 | 4 |
| min FWHM (s) | 0.010 | 0.003 | 0.003 | 0.002 | 0.005 | 0.0005 | 0.0001 | 0.00005 | 0.003 | 0.003 | 0.020 |
| max FWHM (s) | 0.35 | 0.25 | 0.25 | 0.025 | 0.15 | 0.003 | 0.0015 | 0.0007 | 0.15 | 0.15 | 0.30 |
| component labels (s) | P1 (0.08–0.13), N1 (0.13–0.20), N170 (0.13–0.20), P2 (0.15–0.27), P3 (0.25–0.60), N400 (0.30–0.50) | N1 (0.009–0.050), N2 (0.050–0.300), delayed (0.100–1.0) | as `ccep` | N20 (0.017–0.024), P25 (0.022–0.030), N33 (0.028–0.040) for median nerve; P37 (0.033–0.045), N45 (0.040–0.055) for tibial | N15 (0.010–0.020), P30 (0.022–0.038), N45 (0.038–0.055), P60 (0.050–0.075), N100 (0.080–0.140), P180 (0.150–0.250) | ERNA (0.0015–0.0077) | N1 (0.0003–0.0012), P2 (0.0008–0.0025) | N1 (0.0001–0.0005), P2 (0.0004–0.0010) | P1/N1 by depth (0.005–0.060), late (0.060–0.300) | as rodent | P3 (0.25–0.60) |
| trial fit | no | yes (`shift_scale`) | yes | no | no | yes (per pulse, `free`) | yes (per pulse) | no | yes | yes | yes |
| key sources | [LIT-erpmeas-39], [LIT-erpmeas-1] | [LIT-ccep-7], [LIT-ccep-8], [LIT-ccep-1], [LIT-ccep-3], [LIT-ccep-5], [LIT-ccep-18] | [LIT-ccep-46], [LIT-ccep-47], [LIT-ccep-27], [LIT-ccep-25] | [LIT-domains-19], [LIT-domains-20] | [LIT-domains-1], [LIT-domains-3], [LIT-domains-12] | [LIT-domains-52], [LIT-domains-53], [LIT-domains-54] | [LIT-domains-37], [LIT-domains-38] | [LIT-domains-61], [LIT-domains-62] | [LIT-datasets-50], [LIT-variability-45], [LIT-variability-50] | [LIT-datasets-54], [LIT-datasets-55] | [LIT-domains-66], [LIT-domains-71] |

### 6.2 Justification, preset by preset

**`erp`.** Epoch and baseline are the ERP CORE conventions for its seven components [LIT-erpmeas-39]; component windows are the ranges used in that resource and in the measurement textbook [LIT-erpmeas-1]. The fit window starts at 50 ms because sensory components before 50 ms are not the target of this preset (the `sep` preset covers them). `min_fwhm = 10 ms`: the narrowest cortical scalp component in ERP CORE (the visual P1/N1 complex) is several tens of ms wide *(choice: one quarter of the narrowest component, to keep the search permissive)*. Background `constant`: scalp ERPs are baseline-corrected, so only a residual offset remains.

**`ccep`.** Epoch `[−1, 2] s` and baseline `[−0.5, −0.02] s` are the ER-detect defaults [LIT-ccep-30], themselves derived from the dataset paper [LIT-ccep-7]; the fit window starts at 9 ms as in the published N1 rule (`3.4 × SD` in 9–100 ms) [LIT-ccep-7] and the ER-detect response search epoch `[0.009, 0.09] s` [LIT-ccep-30], and ends at 0.5 s to include N2 (N2 windows end between 250 and 500 ms across studies [LIT-ccep-3], [LIT-ccep-4], [LIT-ccep-17]). The labels N1 (9–50 ms) and N2 (50–300 ms) are the most common definitions among the studies compared in the CCEP methods review [LIT-ccep-2] (Keller 10–60/60–250 ms [LIT-ccep-4], Hays 10–50 ms [LIT-ccep-5], Silverstein 10–50 ms [LIT-ccep-6], Cornblath 15–50/50–300 ms [LIT-ccep-9]); "delayed" is Valentín's 100 ms–1 s [LIT-ccep-19]. Artifact window `(−2, 8) ms` *(choice: ends 1 ms before the earliest fit sample, so the template never overlaps the response window; validated by REQ-061)*. `car_adaptive`: common average referencing biases CCEPs because responsive channels contaminate the average; the adjusted CAR removes them [LIT-ccep-18]. `min_fwhm = 3 ms`: the N1 at 2 048 Hz can be 4–8 ms wide (§5.5.3), and 3 ms is six samples at 2 kHz. Background `exponential` and `tail="joint"`: the recovery tail and an early broad component are collinear over the first time constants, so they are fitted together ([LIT-fitting-15], [LIT-fitting-34]; §5.3 method 3). Trial fit on by default because trial-to-trial variability is itself a finding in CCEP studies [LIT-ccep-9], [LIT-ccep-14].

**`ccep_seeg`.** As `ccep` with the bipolar reference, because sEEG monopolar references distort CCEP latency and amplitude and volume conduction is larger in depth recordings [LIT-ccep-46], [LIT-ccep-47], [LIT-ccep-27]; the international SEEG stimulation survey documents the variety of montages in use [LIT-ccep-25]. `reference_sensitivity` (§5.4) is run by default and its κ report attached.

**`sep`.** Latency windows for median nerve N20/P25/N33 and tibial P37/N45 are the clinical conventions of the ACNS evoked-potential guidelines and the SEP recommendations [LIT-domains-19], [LIT-domains-20]. The epoch is 100 ms because these are short-latency components. `interpolate` over the stimulus artifact (electrical nerve stimulus, a few ms). `max_fwhm = 25 ms` *(choice: about half of the 10–60 ms fit window)*.

**`tep`.** The six labels and their approximate latencies are the consensus TEP sequence for motor-cortex stimulation [LIT-domains-1]; late peaks (N100, P180) may be contaminated by auditory and somatosensory responses to the coil click and scalp sensation [LIT-domains-1], [LIT-domains-12], which is why the preset sets `trial_fit=False` and the methods text states the confound. The artifact window ends at 12 ms because the TMS pulse and decay artifacts dominate the first ~10 ms and the cleaning pipeline changes the result [LIT-domains-3]; `interpolate` then `subspace` reproduces the two-stage logic of the published pipelines without depending on a specific toolbox. `gabor` is included because TEPs contain oscillatory bursts (§5.5.1).

**`erna`.** ERNA is a damped 200–450 Hz oscillation following each DBS pulse; the published measures are the latency and amplitude of the first peak and the frequency from the inter-peak interval, or the largest spectral peak between 200 and 450 Hz [LIT-domains-53], recorded at 2 048–4 096 Hz [LIT-domains-53]. At 130 Hz stimulation the inter-pulse interval is 7.7 ms, hence the per-pulse epoch `(−2, 7.7) ms`; `damped_oscillation` with `f ∈ [200, 450] Hz` is the shape that encodes exactly these quantities [LIT-fitting-4]; `period_template` artifact removal because the artifact repeats exactly at the stimulation rate [LIT-artifact-1]. `max_components = 2` *(choice: the oscillation plus at most one slow evoked deflection)*.

**`ecap` (spinal cord stimulation).** The SCS ECAP is an N1–P2 complex recorded on non-stimulating contacts of the epidural lead, with amplitude as the dose measure used by the closed-loop systems [LIT-domains-37], [LIT-domains-38]. Window values are set for a sub-millisecond complex *(choice, pending public data: the literature search of this specification found no public SCS ECAP dataset, ECAP recordings being held in device-company systems; the preset is validated on simulations only, REQ-279)*. Alternating polarity artifact cancellation is the published standard paradigm [LIT-artifact-36].

**`ecap_ci`.** N1 at ~0.2–0.4 ms and P2 at ~0.6–0.8 ms after stimulus onset; forward-masking subtraction `A − B + C − D` or alternating polarity [LIT-domains-61]; automated N1/P2 identification is the vendor standard [LIT-domains-62]. The fit window starts at 0.1 ms.

**`rodent_ep` and `primate_ep`.** Sampling defaults follow the public Neuropixels LFP release at 1 250 Hz [LIT-datasets-50]; the `gamma` shape is the synaptic-like response shape for field potentials (§5.5.1); state-dependent gain changes up to about three-fold between waking states [LIT-variability-50], so `metadata["state"]` is a default covariate in the trial table. Laminar polarity reversal is expected, hence `polarity="both"` and the volume-conduction screen of §5.9.

**`p300_bci`.** The P3 window and sampling follow the large public P300 BCI corpora [LIT-domains-71] and the MOABB datasets [LIT-domains-66]; `min_fwhm = 20 ms` because the P3 is broad; trial fit on because single-trial latency jitter limits P300 classification [LIT-variability-30], [LIT-variability-31].

### 6.3 Preset API and rules

```python
evoparam.presets.get(name: str) -> Preset
evoparam.presets.list() -> pd.DataFrame          # name, modality, species, citation keys
evoparam.presets.derive(base: str | Preset, **overrides) -> Preset   # new frozen preset, records the parent
evoparam.presets.register(preset: Preset, *, overwrite: bool = False) -> None
evoparam.presets.to_json(preset) -> str ; evoparam.presets.from_json(text) -> Preset   # to_yaml/from_yaml also available when PyYAML is installed
```

`evoparam.fit(data, preset="ccep", fit_window=(0.012, 0.5))` overrides one field and records `preset="ccep"`, `overrides={"fit_window": ...}` in provenance and in the methods text. Unknown preset names and unknown override fields raise `SettingsError` listing the valid ones.

### 6.4 Acceptance criteria (REQ-270…REQ-289)

- REQ-270: every preset in §6.1 exists, loads from its JSON file into a frozen dataclass, and every field equals the table value (a test parses the table in this document's companion JSON and compares).
- REQ-271: every preset field that is not marked *(choice)* has at least one citation key in `Preset.citation` that resolves in LITERATURE.md.
- REQ-272: `derive` never mutates the parent; `fit(..., preset=p, **overrides)` and `fit(..., preset=derive(p, **overrides))` give identical results.
- REQ-273: `ccep` on ds004080 meets REQ-113 and REQ-114 (the N1 criteria).
- REQ-274: `erp` on ERP CORE (all 40 participants, seven components) recovers the component named by the paradigm (P3 in the oddball, N170 in the face task, N400, MMN, N2pc, LRP, ERN) as a labelled component on the canonical channel for ≥ 90 % of participants, with model latency within ±10 ms of the fractional-area latency of the same average.
- REQ-275: `tep` on the public single-pulse TMS-EEG dataset of the registry [LIT-datasets-17] produces labelled N45 and N100 on the channel of maximal response for ≥ 80 % of subjects, and the methods text names the sensory confound.
- REQ-276: `erna` on `simulate_erna` recovers frequency within ±5 Hz and first-peak latency within ±0.1 ms at SNR ≥ 3; on a public ERNA recording if one becomes available (none was found by the literature search of this specification), the same check is added to §7.
- REQ-277: `sep` on simulated median-nerve SEPs recovers N20 latency within ±0.5 ms at SNR ≥ 5.
- REQ-278: `rodent_ep` on a public Neuropixels LFP session (§7) recovers the visual evoked complex on ≥ 90 % of channels in visual cortex where the existence test is significant.
- REQ-279: presets validated only on simulations (`ecap`, `ecap_ci`, `erna`) carry `validation="simulation_only"` in their metadata, which is printed in the methods text until a public dataset validates them.
- REQ-280: `presets.list()` output is reproduced in the documentation by a build step (no hand-maintained copy).

**Tests.** `tests/test_presets.py` (REQ-270, REQ-271, REQ-272, REQ-280), `tests/test_engine_ccep.py` (REQ-273), `tests/validation/test_erpcore.py` (REQ-274), `tests/validation/test_tep_ds002094.py` (REQ-275), `tests/test_presets_sim.py` (REQ-276, REQ-277, REQ-279), `tests/validation/test_neuropixels_lfp.py` (REQ-278).
## 7. Scientific validation

Validation answers one question per claim: *does the number evoparam reports mean what the documentation says it means, on data where the answer is known or agreed?* It has four layers, run in this order because each one only makes sense once the previous one passes. Every layer produces a versioned report (`validation/<layer>/<version>/report.html` + JSON) generated by `evoparam benchmark run --suite validation`; the methods paper (§11) cites these reports by version DOI.

### 7.1 Layer 1 — Mathematical correctness (unit level)

Equations are implemented as written. Already specified per module: shape values against `mpmath` references (REQ-100), analytic Jacobians against finite differences (REQ-101), variable projection exactness (REQ-103), DC invariance (REQ-109), time-frequency decomposition exactness (REQ-211), deterministic simulators (REQ-250), measured SNR equals requested SNR (REQ-252). No layer-2 result is reported for a release in which a layer-1 test fails.

### 7.2 Layer 2 — Recovery on simulations with ground truth

All generators of §5.12, with three noise families (white, AR(1) `ρ ∈ {0.5, 0.9}`, pink `χ ∈ {1, 2}`) and real-noise injection (ds004080 and ERP CORE baselines). The design is a full factorial over:

| factor | levels |
|---|---|
| modality generator | erp, ccep, tep, erna, ecap, animal |
| SNR (peak / per-average noise SD) | 1, 2, 3, 5, 10, 20 |
| number of components | 0, 1, 2, 3 |
| component separation (in FWHM) | 0.3, 0.5, 1.0, 2.0 |
| width (samples) | 3, 6, 12, 25 |
| trial count | 5, 10, 20, 40, 100 |
| latency jitter SD (fraction of FWHM) | 0, 0.1, 0.25, 0.5 |
| artifact | none, removed by preset method, left in |

1 000 replicates per cell for the fast metrics, 200 for bootstrap and profile intervals, 50 for Bayesian fits; fixed seeds published with the report. Metrics: recall, precision and F1 of components (matched as in §5.12), latency MAE in samples, amplitude relative bias, width relative error, false-component rate on `K = 0`, interval coverage, existence-test type-I error and power, trial-level errors. The acceptance thresholds are the REQs of §5 (REQ-104…REQ-122, REQ-130…REQ-139, REQ-150…REQ-155, REQ-170…REQ-175, REQ-190…REQ-193, REQ-212…REQ-215, REQ-230…REQ-234). The report also maps *where each method fails* (heat maps of F1 over SNR × separation) because the limits are a result, not a footnote [LIT-fitting-34], [LIT-fitting-35].

- REQ-370: the full layer-2 grid runs in the scheduled CI job (weekly) in ≤ 6 h on the reference runner and every acceptance threshold of §5 that is defined on simulations is evaluated and reported pass/fail with its value.

### 7.3 Layer 3 — Agreement with published methods and human raters on real data

| dataset (registry id) | what is compared | reference | criterion |
|---|---|---|---|
| ds004080 CCEP, ≥ 5 subjects chosen by the rule below | N1 presence and latency vs the published `3.4 × SD`, 9–100 ms rule; N1 latency vs age | [LIT-ccep-7], [LIT-datasets-2] | REQ-113, REQ-114, REQ-135, REQ-178 |
| ds004080, the subset with two-observer visual N1 scoring if the dataset's derivatives contain it (to be checked; otherwise ds004080's propofol companion study [LIT-ccep-13]) | N1 presence vs human raters | [LIT-ccep-8], [LIT-ccep-13] | Cohen's κ evoparam–raters ≥ κ rater–rater (reported 0.60–0.89) |
| ds003708 (basis profile curves) | shared shapes and cross-projection duration | [LIT-datasets-5], [LIT-ccep-16] | REQ-134, REQ-194 |
| ds004977 (CARLA) | adjusted common average: responsive-channel exclusion | [LIT-datasets-11], [LIT-ccep-18] | REQ-372 |
| ERP CORE (40 participants, 7 components) | classical measures, model latency vs fractional-area latency, reliability, SME | [LIT-erpmeas-39], [LIT-datasets-30] | REQ-117, REQ-233, REQ-274 |
| ds002094 single-pulse TMS-EEG | TEP labels on the responsive channel | [LIT-datasets-17] | REQ-275 |
| Allen Visual Coding Neuropixels LFP (one session, NWB) | visual evoked complex on cortical channels, state dependence | [LIT-datasets-50], [LIT-datasets-51] | REQ-278 |
| Localize-MI (intracerebral stimulation with HD-EEG) | scalp responses to intracranial pulses: existence vs stimulation depth | [LIT-datasets-24], [LIT-datasets-25] | REQ-373 |

Subject selection for ds004080 is fixed before any result is seen, to avoid choosing the subjects where the method works: the first five subjects in alphabetical order of BIDS subject label that have at least one run with ≥ 10 trials per stimulation pair at 2 048 Hz, plus `sub-ccepAgeUMCU01` (the example of §1). The list is frozen in `validation/ds004080_subjects.txt` with the dataset version (1.2.4) and its SHA of `participants.tsv`.

- REQ-371: the ds004080 subject list is generated by the rule above by a script in the repository and committed before the first layer-3 run; changing it requires a changelog entry stating why.
- REQ-372: on ds004977, `rereference(scheme="car_adaptive")` excludes the responsive channels identified in the published analysis with a Jaccard index ≥ 0.8 per stimulation site.
- REQ-373: on Localize-MI, the existence test on the scalp signal is significant for ≥ 80 % of stimulations that the dataset authors report as producing a scalp response, and the false-positive rate on sham/low-intensity trials (where available) is ≤ α + 0.02.
- REQ-374: every layer-3 comparison reports the comparator's number, evoparam's number, the difference with its interval, and the exact versions of the data and of the comparator definition (registry rule id).

### 7.4 Layer 4 — Robustness and sensitivity

1. **Montage sensitivity**: every ds004080 and ds004977 result is recomputed under bipolar, CAR and adjusted CAR; κ of existence and latency differences are reported (motivated by the 47.4 ms vs 70.7 ms N1 latency difference between montages in [LIT-ccep-27]).
2. **Preset sensitivity**: each *(choice)* field of §6 is perturbed by ±50 % one at a time; any change of > 10 % in recall or > 2 ms median latency is reported as a sensitivity and listed in the preset documentation.
3. **Sampling rate**: ds004080 decimated to 1 024 and 512 Hz; N1 recall must stay within 5 percentage points down to 1 024 Hz.
4. **Trial count**: subsampling 5, 7, 10 trials per site; the change in existence decisions is reported per trial count (link with `trials_needed`, §5.11).
5. **Platform**: identical results (within 1e-9, REQ-295) on Linux, macOS and Windows for the deterministic paths; Bayesian paths within Monte Carlo error.

- REQ-375: the five robustness analyses run in the release pipeline of every minor version and are attached to the release notes.
- REQ-376: a regression in any layer-3 metric larger than its pre-registered tolerance (κ − 0.05, latency +1 ms median, recall − 3 points) blocks the release unless the changelog explains it.

**Tests.** `tests/validation/test_grid.py` (REQ-370, scheduled weekly), `tests/validation/test_subject_list.py` (REQ-371), `tests/validation/test_carla_ds004977.py` (REQ-372), `tests/validation/test_localize_mi.py` (REQ-373), `tests/validation/test_report_schema.py` (REQ-374), `tests/validation/test_robustness.py` (REQ-375, REQ-376).

### 7.5 What is *not* claimed

The validation does not establish clinical validity (e.g. that an N1 map localises a seizure onset zone, or that an ECAP amplitude predicts pain relief). Those are clinical studies (§13) and the package never states them. Agreement with published rules (layer 3) is agreement with a convention, not with a ground truth; this is why layer 2 exists and why both are reported.
## 8. Public benchmark (`evoparam.benchmark`)

**Why a benchmark is part of the product.** The CCEP field has no shared operational definition of a response (window starts from 5 to 16 ms, thresholds from 2.6 to 6 SD [LIT-ccep-10], [LIT-ccep-22], [LIT-ccep-13], [LIT-ccep-3]), half of the studies do not state their method [LIT-ccep-11], and no benchmark compares detection or scoring methods on common data; the ERP measurement literature built only private simulations [LIT-erpmeas-2], [LIT-erpmeas-43], [LIT-erpmeas-44]; the spectral ancestor of this tool still has an open benchmarking plan years after release [LIT-needs-91]. Benchmarks are used heavily and cited as a single target (the BCI benchmark harness has 1 061 GitHub stars [LIT-citation-32]); a benchmark with a leaderboard turns every competing method paper into a citation of the benchmark. It is therefore specified as a module, a dataset deposit, and a paper (§11).

### 8.1 Tasks

| id | task | input | output scored | metric(s) | ground truth |
|---|---|---|---|---|---|
| T1 | response detection | `TrialSet` per stimulation site (CCEP) | per channel: responsive yes/no, p-value | AUC, F1 at the method's own threshold, calibration of p-values | simulations; consensus of ≥ 2 published rules + human raters where available (ds004080) |
| T2 | early component latency and amplitude | as T1 | N1 latency, amplitude | MAE, bias, coverage of reported intervals | simulations; agreement with published rule on real data |
| T3 | component decomposition | evoked waveform | list of components (latency, amplitude, width) | matched F1, parameter errors, false-component rate | simulations only |
| T4 | artifact removal | raw epochs with stimulation artifact + known response | cleaned epochs | response RMSE in the first 50 ms, artifact residual RMS, preservation of N1 latency | simulations (§5.12) and real-noise injection |
| T5 | single-trial estimation | trials | per-trial latency and amplitude | MAE, correlation with truth, jitter SD error | simulations with jitter |
| T6 | ERP scoring | ERP CORE averages | classical measures | agreement with reference definitions; SME | ERP CORE [LIT-erpmeas-39] |
| T7 | time-frequency response | trials | responsive zones and metrics | zone recovery (centroid errors), type-I error | simulations with induced bursts |

### 8.2 Data

- **Simulated suites** (`core`, `full`): generated on the fly by `evoparam.sim` from a frozen JSON file (`benchmark/suites/*.json`) with fixed seeds; also deposited as files (NWB and BIDS) on Zenodo so that methods in other languages can be evaluated without the package.
- **Real-data suites**: ds004080 (CC0) subset of §7.3, ds003708, ds004977, ERP CORE (CC BY-SA), ds002094; the benchmark ships loaders and index files, not copies, except where the licence allows redistribution of derived epochs (CC0 data are redistributed as epoched NWB for convenience; CC BY-SA data are referenced only).
- **Hidden test split**: 20 % of the simulated cells use seeds not published until the benchmark paper's acceptance, to detect overfitting of submitted methods.

### 8.3 Baseline methods (re-implemented or wrapped, never copied from GPL code)

1. The published threshold rules of the §5.6 registry (vanblooijs2023, hays2021, keller2014, silverstein2020, ftract, prime2020).
2. Cross-projection response test (§5.6 method 2, re-implementation from the paper).
3. Classical measures: local peak in window, fractional area latency (§5.5.6).
4. Template matching / Woody-style single-trial alignment (§5.7 `"template"`).
5. evoparam defaults (`ccep`, `erp`, … presets).
6. External methods are run as *adapters* in separate processes (subprocess calling the external tool with its own licence) so that GPL tools can be compared without being linked: the adapter API is `Adapter.run(task_input_path) -> task_output_path` on files (§8.4). The adapters for ER-detect (GPL-3.0, Python) and for the canonical response parameterisation code (GPL) are provided as optional scripts in a separate directory with their own licence notice, not in the package.

### 8.4 Submission format and scoring

- Every task has a JSON Schema for inputs and outputs; outputs are TSV files with fixed columns (`site, channel, responsive, p_value, latency, amplitude, …`).
- `evoparam benchmark score --task T1 --predictions dir/ --suite core` returns the metrics table and a signed JSON (SHA-256 of inputs, outputs, suite version, package version).
- Leaderboard: a static page generated from the JSON results in the public repository (no server), one row per method × version, with the method's citation; entries are accepted by pull request with the JSON and the code or container that produced it.

### 8.5 Acceptance criteria (REQ-350…REQ-369)

- REQ-350: `evoparam benchmark run --suite core` completes in ≤ 30 min on the reference CI machine and covers T1–T7.
- REQ-351: every task's input and output schemas are versioned (`benchmark-task/<id>/1`) and validated on read; scoring rejects outputs that do not validate, naming the first violation.
- REQ-352: scoring is deterministic and the signed JSON is reproducible byte for byte from the same inputs.
- REQ-353: all six published threshold rules and the cross-projection test are present as baselines and their T1/T2 results on the simulated `core` suite are in the first public report.
- REQ-354: the adapter mechanism runs an external command-line tool on task files without importing it; the adapters for GPL tools live outside the Apache-licensed package tree (checked by a licence-scan test).
- REQ-355: the simulated suites are deposited as NWB/BIDS files with a DOI before the benchmark paper is submitted; the deposit includes the generator version and seeds of the public split.
- REQ-356: the hidden split's seeds are committed as a SHA-256 hash at deposit time and revealed at paper acceptance, so that the hidden split could not have been changed after seeing submissions.
- REQ-357: the leaderboard page is generated by a script from result JSONs only; no hand-edited numbers.

**Tests.** `tests/benchmark/test_suite_core.py` (REQ-350, REQ-353), `tests/benchmark/test_schemas.py` (REQ-351), `tests/benchmark/test_scoring.py` (REQ-352), `tests/benchmark/test_adapters.py` (REQ-354), `tests/benchmark/test_deposit.py` (REQ-355, REQ-356), `tests/benchmark/test_leaderboard.py` (REQ-357).
## 9. Engineering

Non-functional contract: speed, dependencies, tests, typing, documentation, versioning, reproducibility. Requirement ids REQ-290 … REQ-309 (engineering) and REQ-330 … REQ-338 (provenance, settings logging and output-schema versioning; the methods-text generator is specified with `evoparam.reports` in §5 and only referenced here).

### 9.1 Performance targets

The inherited fitter spends essentially all of its time in the SciPy optimiser and has never been benchmarked [LIT-fooof-49]; trial-level Bayesian waveform models in the literature cost about 10 h per subject [LIT-fitting-3]. The targets below are set so that a full CCEP session (hundreds of channels, tens of stimulation sites, 10 pulses per site at 2048 Hz [LIT-datasets-2]) fits on a laptop in minutes, not hours.

| id | workload | target | measured how |
|---|---|---|---|
| P1 | one averaged waveform, 2048 Hz, 1 s fit window (2048 samples), `max_components=6`, default preset `ccep`, 3 restarts | median wall time < 50 ms on one CPU core | 20 repetitions after one warm-up call; `n_jobs=1`; `OMP_NUM_THREADS=1` |
| P2 | trial-by-trial fits, 100 channels × 50 trials (5 000 fits), 2048 Hz, 1 s window, warm start from the channel template (§5 `fit_trials`) | < 60 s wall on 4 cores (`n_jobs=4`) | one run; includes process start-up of the worker pool |
| P3 | memory: `TrialSet` of 500 trials × 256 channels × 4096 samples float64 (4.19 GB) passed to `evoparam.fit(..., chunk="auto")` | peak RSS increase during the call < 2 × `data.nbytes` (< 8.4 GB); with a memory-mapped input (`TrialSet.load(path, mmap=True)`) peak RSS increase < 0.25 × `data.nbytes` + 512 MB | `resource.getrusage` / `psutil` sampled every 50 ms by the benchmark script |
| P4 | GPU backend (`backend="jax"` or `"cupy"`, float64 enabled) on the P1 and P2 workloads | every component parameter, `gof` field and classical measure equal to the CPU result within 1e-6 relative (`numpy.testing.assert_allclose(rtol=1e-6, atol=1e-12 × data scale)`) | the same fixture fitted with both backends |
| P5 | `evoparam.reproduce` (REQ-333) on a saved `GroupFit` of 256 channels | < 1.5 × the original fit wall time | benchmark script |
| P6 | import time `python -c "import evoparam"` with only numpy and scipy installed | < 300 ms (median of 10) | `python -X importtime` |

Chunking rule for P3: `evoparam.fit`, `fit_trials`, `remove_artifact` and `test_response` accept `chunk: int | "auto" | None` (channels per chunk). `"auto"` chooses the largest chunk such that `chunk × n_trials × n_times × 8 bytes × 3` (data, model, residual) stays below `max_memory_gb` (default 0.25 × available physical memory, read from `os.sysconf` when available, else 2 GB). Intermediate arrays are never allocated for the whole set at once; results are concatenated per chunk.

GPU rule for P4: the GPU path uses the same shape functions and the same optimiser logic (a batched Levenberg–Marquardt / trust-region solver written on top of `jax.numpy` or `cupy`, one problem per batch row); it is never a different algorithm. Float32 execution is permitted only when the caller passes `dtype="float32"` explicitly, and the result then carries `warnings=["float32 backend"]`. JAX [LIT-fitting-49] and CuPy [LIT-fitting-51] are both optional (`gpu` extra); the default backend is NumPy/SciPy [LIT-fitting-47].

Benchmark script and CI enforcement: the script `benchmarks/run_benchmarks.py` (thin wrapper around `evoparam.benchmark.timing.run_suite("perf")`, also reachable as `evoparam benchmark run --suite perf`) generates the fixtures deterministically with `evoparam.sim` (`random_state=0`), runs P1–P6, and writes `benchmarks/results/<platform>-<python>-<commit>.json` with fields `{id, median_s, p90_s, peak_rss_bytes, n_repeats, environment}`. The baseline for the fixed CI runner is committed as `benchmarks/baseline-ci.json`. The CI job `perf` (REQ-299) fails when any `median_s` exceeds 1.25 × its baseline or when a P3 memory figure exceeds its bound; it warns (annotation, no failure) between 1.10 × and 1.25 ×. The absolute targets in the table are checked on the reference machine named in `benchmarks/README.md` (one documented laptop-class CPU) at every minor release and recorded in the CHANGELOG.

> **REQ-290 — Performance targets P1–P6.** The package meets the six targets of the table above. *Acceptance:* `benchmarks/run_benchmarks.py` on the reference machine reports P1 median < 50 ms, P2 < 60 s, P3 peak RSS increase < 2 × data size (< 0.25 × + 512 MB memory-mapped), P4 CPU/GPU agreement within 1e-6 relative, P5 < 1.5 × original, P6 < 300 ms; the results file is attached to each minor release. *Source:* [LIT-fooof-49] (open benchmarking issue, fit time dominated by the optimiser), [LIT-fitting-3] (10 h per subject for a trial-level waveform model), [LIT-datasets-2] (session sizes), [LIT-datasets-3] (lazy, chunked readers for 0.6–269 GB datasets); quality rule from the project brief.

### 9.2 Dependency policy

The inherited skeleton declares `dependencies = ["numpy", "scipy>=1.8"]` and optional extras, but the package cannot be imported without `matplotlib` and `neurodsp` because two modules import them unconditionally [CODE §0] [CODE §1.7]. Two hard dependencies are one of the documented reasons the ancestor spectral tool was adopted [LIT-fooof-14] [LIT-fooof-15]; "fails with certain versions of dependencies" is one of its open issues [LIT-fooof-45]. The policy below keeps the core light and makes every extra explicit.

> **REQ-291 — Light core.** `import evoparam` and every function of the core chain (`read` of the npz format, `detect_events`, `remove_artifact` with the interpolation and template methods, `fit`, `test_response`, `fit_trials`, `simulate`, `TrialSet`/`Evoked`/`GroupFit` save and load in npz/CSV) work with only `numpy` and `scipy` installed. *Acceptance:* the CI job `core-only` (REQ-297) creates a virtual environment with `pip install evoparam` (no extras), runs `python -c "import evoparam; evoparam.selftest()"` and `pytest -m core`, and both pass; `evoparam.selftest()` fits one bundled synthetic CCEP and asserts the recovered N1 latency within 1 ms. *Source:* [CODE §0], [LIT-fooof-14], [LIT-fooof-15]; §4.1 principle 2.

Optional extras (declared in `pyproject.toml` `[project.optional-dependencies]`; `all` is the union of every extra except `gpu`, `bayes` and `docs`):

| extra | packages (floor) | what it unlocks |
|---|---|---|
| `mne` | `mne>=1.6` | `TrialSet.from_mne/to_mne`, `evoparam.fit(mne.Epochs | mne.Evoked)`, FIF read/write, montage-aware plots (§10.1) |
| `bids` | `mne-bids>=0.15`, `pybids>=0.16` | BIDS dataset discovery, `events.tsv` harmonisation, derivatives writer (§10.2) |
| `nwb` | `pynwb>=2.6`, `hdmf>=3.12`, `ndx-events>=0.2` | NWB read/write, `EventsTable` event source, `ndx-evoparam` extension (§10.3) |
| `neo` | `neo>=0.13` | `read_neo` for vendor formats through Neo (§10.3) |
| `edf` | `edfio>=0.4` (preferred) or `pyedflib>=0.1.36` | EDF/EDF+/BDF readers |
| `bayes` | `numpyro>=0.14`, `jax>=0.4.30`, `arviz>=0.18` (alternative backend `pymc>=5.10`) | hierarchical and full-posterior fits (§5 `hierarchical`) [LIT-fitting-38] [LIT-fitting-40] [LIT-fitting-44] |
| `gpu` | `jax[cuda12]>=0.4.30` or `cupy-cuda12x>=13` | batched GPU solver (P4) |
| `fast` | `numba>=0.59` | JIT kernels for shape evaluation and Jacobians on CPU [LIT-fitting-50] |
| `plot` | `matplotlib>=3.7` | `evoparam.plots` |
| `data` | `pandas>=2.0`, `pyarrow>=14` | DataFrame metadata, parquet tables, tidy exports |
| `xarray` | `xarray>=2023.1` | `TrialSet.to_xarray` (§4.3.1) |
| `stats` | `statsmodels>=0.14`, `formulaic>=1.0` | `stats.fit_mixed` (§5.11) and design formulas of the hierarchical model (§5.8) |
| `validate` | `bidsschematools>=0.7`, `nwbinspector>=0.5`, `jsonschema>=4.18` | JSON Schema validation of result files (REQ-341), schema checks of BIDS derivatives and NWB inspection of written files (REQ-315, REQ-316, REQ-318); the reference BIDS validator is the Node.js `bids-validator@2`, run with `npx` in CI and not a Python dependency |
| `cli` | none (stdlib `argparse`) | the `evoparam` console script is always installed; the extra exists only so that `pip install evoparam[cli]` is a valid spelling and installs `data` + `progress` |
| `progress` | `tqdm>=4.66` | progress bars when `verbose="progress"` |
| `parallel` | `joblib>=1.3` | process-based `n_jobs` (without it, a thread pool is used; §5 conventions) |
| `stream` | `pylsl>=1.16` | LSL input adapter for `evoparam.stream` (§10.6) |
| `docs` | `sphinx>=7`, `pydata-sphinx-theme>=0.15`, `sphinx-gallery>=0.15`, `numpydoc>=1.6`, `sphinx-copybutton`, `sphinxcontrib-bibtex` | documentation build (§9.7) |
| `tests` | `pytest>=8`, `pytest-cov>=5`, `pytest-timeout>=2.3`, `pytest-xdist>=3.5`, `hypothesis>=6.100`, `scikit-learn>=1.4` (for `check_estimator`), `mypy>=1.10`, `ruff>=0.5`, `interrogate>=1.7` | test and lint tool chain |
| `all` | union as stated above | everything a user can install without a GPU or a sampler |

Version floors and compatibility: `python>=3.10`, `numpy>=1.24,<3`, `scipy>=1.10`. Current releases of MNE-Python (1.13) and scikit-learn (1.9) require Python ≥ 3.11 [LIT-standards-34] [LIT-standards-51], so on Python 3.10 the `mne` and `tests` extras resolve to older releases that still satisfy the floors above (§16, Q12). The code is written against the NumPy 2.x API (`numpy.exceptions`, no `np.float_`, no `np.in1d`, copy semantics of `np.array(copy=False)` replaced by `np.asarray`); the CI matrix includes one job with the lowest floors (`numpy==1.24.*`, `scipy==1.10.*`) and one with the newest releases. Lazy imports go through one helper:

```python
# evoparam/utils/_optional.py
def require(module: str, *, extra: str, min_version: str | None = None) -> ModuleType:
    """Import an optional dependency or raise ImportError with the pip command to run.

    >>> mne = require("mne", extra="mne", min_version="1.6")
    ImportError: evoparam.io.read_fif needs the optional package 'mne' (>= 1.6).
    Install it with:  pip install "evoparam[mne]"
    """
```

`require` is the only place where optional packages are imported; a test greps the package for top-level `import mne`, `import pandas`, `import matplotlib`, … outside `require` and outside `TYPE_CHECKING` blocks and fails on any hit. The error message names the function that needs the package, the package, the floor and the exact `pip` command.

> **REQ-292 — Optional extras and lazy imports.** Every dependency other than `numpy` and `scipy` is declared in exactly one extra of the table above, is imported only through `evoparam.utils.require`, and a missing package raises `ImportError` whose message contains the extra name and the `pip install "evoparam[<extra>]"` command. *Acceptance:* `tests/test_dependencies.py` (i) parses `pyproject.toml` and asserts the extras table (names, floors) matches the one in this section (kept as `evoparam/_extras.py`), (ii) asserts by `ast` that no module imports an optional package at top level, (iii) runs each optional entry point in a subprocess with the package hidden (`sys.modules[name] = None`) and asserts the `ImportError` message matches the regex `pip install "evoparam\[(\w+)\]"`. *Source:* [CODE §0], [LIT-fooof-45] (version-dependent failures), quality rule from the project brief.

GPL isolation and pinning. The domain's CCEP tools are GPL-3.0 (`erdetect`, `ieegprep`, the MATLAB tool boxes) and so are `TEPpy` and `UltraNest` [LIT-naming-51] [LIT-fitting-11] [LIT-fitting-55]; the Apache Software Foundation's position is that GPL-3.0 software cannot be included in Apache-licensed projects because linking creates a derivative work [LIT-naming-14], whereas communication with a separate process at arm's length is not a combination [LIT-naming-57]. FieldTrip, Brainstorm and ERPLAB are MATLAB and GPL and are not import candidates in any case [LIT-erptools-28] [LIT-erptools-35] [LIT-erptools-6]. For reproducibility the package distinguishes what users get (loose floors) from what CI and papers run (locked versions).

> **REQ-309 — GPL isolation, lock file and version floors.** (a) No module of `evoparam` imports, vendors or optionally imports a GPL-licensed Python package; the deny list `erdetect`, `ieegprep`, `teppy`, `ultranest` is checked by `tests/test_dependencies.py::test_no_gpl_imports` against every `import` statement and every extra in `pyproject.toml`. (b) Adapters to GPL tools live in a separate distribution `evoparam-adapters` (its own repository and licence), call the tool through `subprocess` with files as the only exchange, and are never a dependency of `evoparam`. (c) `requirements-ci.lock` (generated by `uv pip compile pyproject.toml --all-extras` or `pip-compile`, one file per Python minor version in `ci/locks/`) pins every package used by the `tests` and `docs` jobs with hashes; the `tests` job installs from the lock, the `nightly` job (REQ-300) ignores it. (d) The sha256 of the lock file used for a fit is recorded in `Provenance.environment["lock_sha256"]` when the environment matches a lock, else `null`. *Acceptance:* (a) and (b) by the test named; (c) `ci/locks/py3.12.lock` exists, `pip install -r ci/locks/py3.12.lock --require-hashes` succeeds in CI, and a scheduled job refreshes the locks monthly by pull request; (d) `tests/test_provenance.py::test_lock_hash`. *Source:* [LIT-naming-14], [LIT-naming-57], [LIT-naming-51], [LIT-fitting-11], [LIT-fitting-55], [LIT-fooof-45] (pin dependencies to avoid version-dependent failures).

### 9.3 Versioning and deprecation

The ancestor tool's one breaking rename release was its main source of confusion before 2.0 and its MATLAB ports drifted away from it [LIT-fooof-18] [LIT-fooof-16]; a 30-month release-candidate series was the alternative chosen for the next major version [LIT-fooof-9] [LIT-fooof-20]. The package adopts semantic versioning with explicit 0.x rules.

- Version string: `evoparam.__version__`, single source in `pyproject.toml` (`[project] version`), read at runtime through `importlib.metadata`. Development builds carry `+g<sha>` (setuptools-scm or hatch-vcs).
- 0.x rules: minor versions (0.2, 0.3, …) may change the public API only through the deprecation path below; patch versions never change behaviour of a documented function except to fix a bug recorded in the CHANGELOG with the previous and the new value on the bundled fixture.
- 1.0 API freeze: the names listed in `docs/api/public.rst` (every object exported from `evoparam/__init__.py` and the module-level functions of §4.2) keep their signatures until 2.0; new keyword arguments must have defaults that reproduce the 1.x behaviour.
- Preset values are versioned separately (`Preset.version`, §6): changing a default window or threshold bumps the preset version and the CHANGELOG, never silently.
- `CHANGELOG.md` follows Keep a Changelog 1.1 (sections Added / Changed / Deprecated / Removed / Fixed / Security, an Unreleased section, one entry per pull request with the PR number).

> **REQ-293 — Deprecation shims.** A renamed or removed public name keeps a shim for two minor versions that emits `DeprecationWarning` (category `evoparam.utils.EvoparamDeprecationWarning`, a `DeprecationWarning` subclass) naming the replacement and the version of removal; the shim forwards all arguments; removal happens only in the minor version announced. *Acceptance:* `tests/test_deprecations.py` imports every name in `evoparam/_deprecated.py` (a registry `{old_name: (new_name, since, remove_in)}`), asserts the warning text contains both names and `remove_in`, asserts `remove_in − since == 2` minor versions, and fails if `evoparam.__version__ >= remove_in` while the shim still exists. *Source:* §4.4; [LIT-fooof-18], [LIT-fooof-16].

> **REQ-308 — Release process and citation metadata.** Each release is produced by the checklist `RELEASING.md` and yields a git tag `vX.Y.Z`, a GitHub release, a PyPI upload of sdist + wheel from the CI `release` workflow (trusted publishing, no stored token), a Zenodo DOI through the GitHub–Zenodo integration (concept DOI fixed in `CITATION.cff`, version DOI added to the release notes), and a CHANGELOG section. `CITATION.cff` (version, date-released, DOI, authors, preferred-citation) and `pyproject.toml` are kept in sync by a test. *Acceptance:* `tests/test_citation.py` parses `CITATION.cff` with `cffconvert`-free YAML loading and asserts `version == evoparam.__version__`, `date-released` is the tag date on a tagged commit, `doi` is a `10.5281/zenodo.` DOI, `authors[0].family-names == "Bellec"`; `RELEASING.md` contains the steps: freeze CHANGELOG, bump version, run `benchmarks/run_benchmarks.py` on the reference machine, build docs with `-W`, tag, verify PyPI smoke test (REQ-300), verify Zenodo DOI, announce. *Source:* [LIT-naming-1] (name and Zenodo DOI from the first release so that citations accrue to one identifier), [LIT-naming-24], [LIT-standards-52] [LIT-standards-55] (CITATION.cff and a DOI per GitHub release), [LIT-citation-29] (one concept DOI plus version DOIs, no citation fragmentation), quality rule from the project brief.

### 9.4 Reproducibility and determinism

The research evidence makes reproducibility the central non-functional requirement: 49.7 % of 187 CCEP studies did not state their detection method and 3 of 72 shared code [LIT-ccep-11]; only 56 % of clinical papers using the ancestor spectral tool reported its settings although a methods-text generator existed [LIT-fooof-4]; preprocessing pipelines alone change early TMS-evoked peak amplitudes with concordance as low as −0.03 [LIT-artifact-7]; none of the artifact-removal tools surveyed for this specification, MNE's blanking function included [LIT-artifact-63], writes which samples it replaced in a machine-readable sidecar (this specification's survey). Inherited tests seed `numpy.random` once per session and depend on the noise realisation [CODE §5].

> **REQ-294 — Provenance on every result.** Every public function that creates or transforms a container (`TrialSet`, `Evoked`, `ArtifactResult`, `ResponseFit`, `GroupFit`, `TrialFit`, `HierarchicalFit`, `ResponseTest`, `ShapeBasis`, `TFFit`) appends one entry to `provenance.steps` with `function` (fully qualified name), `settings` (every keyword argument after preset resolution, JSON-serialisable, arrays replaced by their sha256 and shape), `version` (`evoparam.__version__`), `preset` (`{"name": ..., "version": ...}` or `null`), `random_state`, `duration_s` and `n_jobs`; `Provenance.source`, `environment` and `data_hash` are filled at creation. *Acceptance:* `tests/test_provenance.py` runs the chain `read → detect_events → remove_artifact → fit → test_response → fit_trials` on the bundled fixture and asserts six steps in order, each JSON-serialisable (`json.dumps(..., allow_nan=False)`), each with the seven keys; `GroupFit.load(GroupFit.save(...))` returns an equal `Provenance`. *Source:* §4.1 principle 3, §4.3.3; [LIT-ccep-11], [LIT-fooof-4], [LIT-artifact-7], [LIT-artifact-63].

> **REQ-295 — `random_state` and cross-platform determinism.** Every stochastic function (restarts of the optimiser, bootstrap, permutation tests, simulators, trial subsampling, MCMC) takes `random_state: int | numpy.random.Generator | None` and derives all randomness from `numpy.random.default_rng(random_state)` (NumPy ≥ 1.17 generator; never the legacy global state); `None` draws a seed from the OS and records it in `provenance.random_state`. With the same `random_state`, results are identical across runs on one machine and agree across platforms (Linux, macOS, Windows; x86-64 and arm64) within 1e-9 relative for the CPU backend and 1e-6 for GPU backends. *Acceptance:* `tests/test_determinism.py` (i) calls each stochastic function twice with `random_state=0` and asserts bit-identical arrays; (ii) compares the fit of the bundled fixture with the reference values committed in `tests/reference/fit_fixture_ccep.json` (produced on the reference machine) with `rtol=1e-9` on CPU — the CI matrix (REQ-297) runs this test on all three operating systems; (iii) `grep -r "np.random.seed\|np.random.rand\|np.random.randn" evoparam/` returns nothing. *Source:* [CODE §5], [LIT-standards-68] (`default_rng` guidance), [LIT-standards-50] (`random_state` resolved in `fit`).

### 9.5 Continuous integration

The inherited workflow runs Python 3.10–3.13 on Ubuntu only, installs `[all]`, runs `pytest --cov` and doctests, and therefore never exercises the core-only install or the import graph [CODE §1.7] [CODE §5].

> **REQ-296 — Import-graph test.** `tests/test_import_graph.py` parses every module of the package with `ast`, builds the directed graph of intra-package imports and fails on (i) any edge not allowed by the table of §4.2, (ii) any cycle, (iii) any top-level import of an optional package (shared with REQ-292). *Acceptance:* the test exists, is in the `core` marker set, and a deliberately added forbidden import (`evoparam/shapes/__init__.py: import evoparam.models`) makes it fail in a mutation check run once per release. *Source:* §4.2.

> **REQ-297 — CI matrix and jobs.** GitHub Actions workflows run on every pull request and on `main`: *lint* (`ruff check`), *format* (`ruff format --check`), *typecheck* (`mypy --strict` on the public API set, REQ-301; `pyright` as a non-blocking job), *tests* on the matrix Python {3.10, 3.11, 3.12, 3.13} × {ubuntu-latest, macos-latest, windows-latest} installed from the lock file (REQ-309) with `[all]`, plus one *core-only* job (REQ-291) and one *lowest-floors* job, with coverage upload; *doctests* (`pytest --doctest-modules evoparam`); *docs* (REQ-302); *import-graph* (REQ-296); *perf* (REQ-299); *build* (sdist and wheel with `python -m build`, `twine check`, `check-wheel-contents`, `py.typed` present). Required to merge: lint, format, typecheck, tests (all matrix cells), doctests, docs, import-graph, build. *Acceptance:* `.github/workflows/ci.yml` defines these jobs; the branch protection rule on `main` lists them as required checks; a pull request with a failing cell cannot be merged; `pytest-timeout` caps every test at 120 s and the full matrix cell at 25 min. *Source:* [CODE §1.7], [CODE §5], [LIT-standards-77] (MNE matrix with minimum-version job), [LIT-standards-78] (runners free for public repositories).

> **REQ-298 — Coverage thresholds.** Line coverage of the core packages (`data`, `io` local readers, `events`, `artifact`, `prep`, `shapes`, `models`, `existence`, `trials`, `presets`, `utils`) is ≥ 90 % and coverage of `evoparam.shapes` is 100 % (lines and branches); `plots`, `cli`, `stream`, `benchmark` and remote I/O are ≥ 80 %. *Acceptance:* `pyproject.toml` `[tool.coverage.report] fail_under = 90`; a second invocation `coverage report --include="evoparam/shapes/*" --fail-under=100` in the tests job; both commands run in the job named `tests (3.12, ubuntu-latest)` and fail the job. *Source:* [CODE §5] (inherited tests assert that code runs, not that numbers are right), quality rule from the project brief.

> **REQ-299 — Performance job.** A workflow `perf` runs `benchmarks/run_benchmarks.py --suite ci` on one fixed runner image (`ubuntu-22.04`, pinned to a runner label, `OMP_NUM_THREADS=1`) on every pull request that touches `evoparam/shapes`, `evoparam/models`, `evoparam/trials`, `evoparam/artifact` or `benchmarks/`, and nightly on `main`; it compares against `benchmarks/baseline-ci.json` with a 25 % tolerance on median time and a 10 % tolerance on peak memory. *Acceptance:* a pull request that slows P1 by 2× fails the `perf` job with a message naming the workload and the ratio; a label `perf-baseline-update` on a merged pull request regenerates the baseline from three runs (median). *Source:* [LIT-fooof-49], quality rule from the project brief.

> **REQ-300 — Nightly, packaging and smoke test.** (a) A scheduled workflow `nightly` runs the test suite against pre-release wheels of numpy, scipy and mne (`pip install --pre --extra-index-url https://pypi.anaconda.org/scientific-python-nightly-wheels/simple numpy scipy` and `mne` main) and opens an issue labelled `upstream-breakage` on failure. (b) `build` produces a pure-Python wheel (`py3-none-any`) and an sdist; (c) after every release, and weekly, the job `smoke` creates a clean virtual environment, runs `pip install evoparam==<version>` from PyPI, then `python -c "import evoparam; evoparam.selftest()"` and `evoparam fit --help`. (d) No workflow uses a repository secret except `release` (trusted publishing needs none) and the Zenodo integration (configured on the Zenodo side); forks can run every job. *Acceptance:* the three workflow files exist; `grep -r "secrets\." .github/workflows/` matches only the `release` workflow's PyPI environment; the smoke job log of the latest release is linked from the release notes. *Source:* [LIT-fooof-45] (dependency-version failures), quality rule from the project brief.

### 9.6 Typing

> **REQ-301 — Static typing of the public API.** The distribution ships `evoparam/py.typed` (PEP 561). Every public function, method and dataclass field is fully annotated; arrays are typed with `numpy.typing.NDArray[np.float64]` (or `NDArray[np.bool_]` for masks); shapes and backgrounds are `typing.Protocol` classes (`ComponentShape`, `Background`, §4.3.4) so that user-defined shapes type-check without inheritance; optional-dependency types are imported under `if TYPE_CHECKING:` as strings. *Acceptance:* `mypy --strict evoparam` passes with `[[tool.mypy.overrides]] ignore_missing_imports = true` for optional packages only; `tests/test_typing.py` runs `mypy` on `tests/typing_examples/*.py` (one example per public entry point, including a user shape class that satisfies `ComponentShape`) and on a deliberately wrong example that must fail; `pyright --verifytypes evoparam` reports ≥ 95 % "known" symbols (non-blocking job). *Source:* quality rule from the project brief.

### 9.7 Documentation

The ancestor tool's documentation (sequential sphinx-gallery tutorials, an examples gallery including an MNE example, a FAQ answering terminology and applicability questions, a reporting page) is one of its documented success factors [LIT-fooof-21] [LIT-fooof-23] [LIT-fooof-25] [LIT-fooof-17]. Gallery examples must run in CI, so they use only datasets under 2 GB: ds003708 (620 MB) and ds004977 (1.5 GB) are the two CC0 CCEP datasets small enough [LIT-datasets-4] [LIT-datasets-5] [LIT-datasets-11], plus the bundled synthetic data of `evoparam.sim` and, for scalp ERP, one ERP CORE participant fetched on demand [LIT-datasets-31].

> **REQ-302 — Documentation build.** Sphinx with `pydata-sphinx-theme`, `numpydoc` docstrings (every public object; sections Parameters / Returns / Raises / Notes / References / Examples), `sphinx-gallery` for `examples/` and `tutorials/` (executed at build time against the CI fixtures: bundled synthetic data, ds003708 and ds004977 partial reads cached by `fetch_*` with checksum), `sphinx.ext.intersphinx` to numpy, scipy, pandas, mne, pynwb, `sphinxcontrib-bibtex` with `docs/references.bib` holding every literature entry (`[LIT-<axis>-<n>]` key) cited in the documentation. The build runs with `-W --keep-going` (warnings are errors) and `sphinx-build -b linkcheck` on every pull request; docs are published as `latest` (main) and `stable` (last tag) with a version switcher. *Acceptance:* the CI job `docs` builds with `-W` in < 20 min on the runner (gallery cached between runs with `sphinx-gallery`'s `run_stale_examples=False`), `linkcheck` reports 0 broken links (allow-list for DOIs behind paywalls), and the published site shows the `latest`/`stable` switcher. *Source:* [LIT-fooof-21], [LIT-fooof-23], [LIT-standards-73] (sphinx-gallery and pydata theme, MNE practice), [LIT-datasets-4], [LIT-datasets-5], [LIT-datasets-11].

> **REQ-303 — Documentation content.** The site contains: (a) *Getting started* (install, 10-line example, what the numbers mean); (b) one tutorial per preset (`erp`, `ccep`, `ccep_seeg`, `sep`, `tep`, `erna`, `ecap`, `ecap_ci`, `rodent_ep`, `primate_ep`, `p300_bci`) and one per integration (MNE, BIDS, NWB, Neo, scikit-learn, CLI, streaming; §10.7), each ending with the generated methods paragraph; (c) *Reporting and citing* — what to report (settings, preset version, package version, goodness of fit, existence test, artifact method, montage), how `ResponseFit.methods_text()` fills it, the citation (paper when available, Zenodo DOI always) and `CITATION.cff`; (d) a FAQ with at least the entries "which preset for my data", "why is polarity a separate field", "why is `r2` not the inherited correlation", "my N1 is missing", "GPU gives slightly different numbers", "how do I add a shape", "is this a medical device" (pointing to §13); (e) *API reference* generated by `autosummary`; (f) *Glossary* of every term in §4; (g) a *datasets registry* page generated from `evoparam/io/registry.json` (name, DOI, size, sampling rate, licence, events status, loader). *Acceptance:* `tests/test_docs.py` asserts the existence of the files `docs/tutorials/preset_<name>.py` for every preset in `evoparam.presets.PRESETS`, `docs/tutorials/integration_<name>.py` for the seven integrations, `docs/reporting.rst`, `docs/faq.rst` with the seven questions, and that every public symbol appears in the generated API index. *Source:* [LIT-fooof-17], [LIT-fooof-25], [LIT-fooof-4], [LIT-datasets-14] [LIT-datasets-4] (machine-readable dataset registry), quality rule from the project brief.

### 9.8 Logging

> **REQ-304 — Logging through the standard library.** All diagnostics go through `logging.getLogger("evoparam")` and child loggers named after the module (`evoparam.models`, …); the package installs a `NullHandler` only. The `verbose` argument of every public function maps to a level for the duration of the call: `None`/`False` → inherit, `True`/`"info"` → INFO, `"debug"` → DEBUG, `"warning"` → WARNING, `"error"` → ERROR, `"progress"` → INFO plus `tqdm` bars when installed; the mapping is implemented once in `evoparam.utils.verbose(level)` (context manager and decorator). `evoparam.set_log_level(level)` and `evoparam.set_log_file(path)` are the only global switches. No `print` call exists in the package outside `evoparam.cli` (which prints only its result). *Acceptance:* `tests/test_logging.py` uses `caplog` to assert that a failed channel emits exactly one WARNING record with the channel name (REQ-307), that `verbose="debug"` yields DEBUG records from `evoparam.models`, and that `grep -rn "print(" evoparam/ --include=*.py | grep -v "evoparam/cli/"` is empty. *Source:* §4.5 (debug prints and `verbose=False` hiding every failure in the inherited code), MNE convention for `verbose`.

### 9.9 Code quality

> **REQ-305 — Static quality rules.** `ruff` with rule sets `E, W, F, I, N, UP, B, A, C4, SIM, PT, RET, ARG, PL, NPY, D` (numpydoc convention), line length 99, `mccabe` max complexity 12 per function, `pylint` max 8 arguments and 60 statements per function (`PLR0913`, `PLR0915`); `ruff format`; docstring coverage measured by `interrogate --fail-under 95` on the package; `pre-commit` configuration with `ruff`, `ruff-format`, `mypy`, `interrogate`, `check-yaml`, `end-of-file-fixer`, `nbstripout`; a pull-request template (`.github/PULL_REQUEST_TEMPLATE.md`) with the checklist "tests added, CHANGELOG entry, docs updated, provenance step added for new public functions, no new hard dependency"; Conventional Commits are recommended but not enforced. *Acceptance:* `ruff check evoparam tests benchmarks docs/tutorials examples` and `ruff format --check` exit 0 in CI; `interrogate -v evoparam` reports ≥ 95 %; `.pre-commit-config.yaml` and the template exist; `pre-commit run --all-files` passes. *Source:* quality rule from the project brief.

### 9.10 Security and data policy

> **REQ-306 — No network, no telemetry.** (a) The only functions that open network connections are the explicit `evoparam.io.fetch_*` functions and `evoparam.io.read_*` when given an `http(s)://` URL; they log the URL at INFO, respect `EVOPARAM_OFFLINE=1` (raise `ReaderError` instead of connecting) and store downloads under `EVOPARAM_DATA` (default `~/.cache/evoparam`). (b) Every downloaded file is verified against the sha256 recorded in `evoparam/io/registry.json`; a mismatch deletes the file and raises `ReaderError`. (c) The package collects no usage data, sends nothing anywhere, and has no update check. (d) Serialisation never uses `pickle` for public formats (`npz` with `allow_pickle=False`, JSON, parquet/CSV); `GroupFit.load` of a file with pickled objects raises `DataError`. *Acceptance:* `tests/test_network.py` runs the whole `core` suite with a socket guard (`pytest-socket` or a monkey-patched `socket.socket` raising) and only tests marked `network` are allowed to connect; `tests/test_fetch.py` corrupts one byte of a cached fixture and asserts `ReaderError` and file removal; `grep -rn "pickle\|urllib\|requests\|httpx" evoparam/` matches only `evoparam/io/_fetch.py` and `np.load(..., allow_pickle=False)` lines. *Source:* quality rule from the project brief; [LIT-domains-82] [LIT-domains-84] (deterministic, auditable behaviour lowers the burden for groups that embed the package in regulated software, §13).

### 9.11 Error handling

The exception hierarchy is defined in §4.6. The inherited fitter swallowed optimiser failures so that an empty channel was indistinguishable from an absent response [CODE §4] §4.5.

> **REQ-307 — No silent failure.** A function never returns a result while hiding an exception. Rules: (i) programming and data errors (`DataError`, `SettingsError`, `ReaderError`, `EvoparamVersionError`) propagate; (ii) where a per-channel or per-trial failure is acceptable (`fit_group`, `fit_trials`, `test_response`, `remove_artifact` with `on_error="record"`), the unit's result carries `status="failed"` and `message=str(exc)` and exactly one WARNING record names the unit; `on_error="raise"` propagates the first `FitError`; (iii) a negative existence test followed by a requested fit emits `NoResponseWarning`, never silence; (iv) numerical warnings from numpy/scipy inside the optimiser are converted to `ConvergenceWarning` with context (channel, restart index), never suppressed with `np.errstate(all="ignore")` outside the three documented places in `evoparam/models/_engine.py`; (v) every `except` clause names an exception class (no bare `except:` and no `except Exception: pass`). *Acceptance:* `tests/test_errors.py` injects a shape whose `evaluate` raises on one channel and asserts `GroupFit.fits[i].status == "failed"`, `message` non-empty, one WARNING with the channel name, other channels `ok`; `ruff` rules `E722`, `S110`, `BLE001` are enabled; `grep -rn "errstate" evoparam/` matches at most three lines, all in `_engine.py`. *Source:* §4.6; [CODE §4], §4.5.

### 9.12 Output schema, settings logging and reproduction (reports block, REQ-330 … REQ-338)

The inherited JSON writer emits `NaN` tokens (not valid JSON), no schema version, no package version, no channel name and no units [CODE §2.5]; no general toolbox has a standard output schema for ERP measurements [LIT-erptools-9]. The methods-text generator (`ResponseFit.methods_text`, `GroupFit.methods_text`, PDF report) is specified with `evoparam.reports` in §5; the requirements below cover only what it reads: the container, its versioning, the settings record and the reproduction function.

> **REQ-330 — The `.evoparam` container.** `save()` of every result type writes a directory or zip `<name>.evoparam` with `manifest.json` (`schema: "evoparam-result/1"`, `object: "GroupFit"`, `package_version`, `created`, `files: {path: sha256}`), `data.npz` (arrays, `allow_pickle=False`), `tables/*.parquet` (or `*.csv` when `pyarrow` is absent, flagged in the manifest), `provenance.json`, `settings.json`; `load()` verifies the checksums. *Acceptance:* `tests/test_serialize.py` round-trips every result type of §4.3 (`obj == load(save(obj))` using the dataclass `__eq__`), checks `manifest.json` keys, and asserts that modifying one byte of `data.npz` makes `load` raise `DataError("checksum")`. *Source:* §4.3.5; [CODE §2.5]; [LIT-erptools-9].

> **REQ-331 — Settings and preset recorded in every result.** Every result carries `settings` (the full resolved dictionary: preset defaults overridden by call arguments, with a `_overrides` list naming the keys the user changed) and `provenance.steps[-1].preset = {"name", "version"}`; `settings` is JSON-serialisable and sufficient to call the same function again (`func(data, **settings)` reproduces the result). *Acceptance:* `tests/test_settings_log.py` fits with `preset="ccep", min_fwhm=0.003` and asserts `settings["min_fwhm"] == 0.003`, `settings["_overrides"] == ["min_fwhm"]`, `settings["preset"] == {"name": "ccep", "version": ...}`, and that `evoparam.fit(data, **{k: v for k, v in settings.items() if not k.startswith("_")})` gives an equal `GroupFit`. *Source:* §4.1 principle 3; [LIT-fooof-4] (56 % settings reporting), [LIT-fooof-47] [LIT-fooof-48] (units and defaults confusion).

> **REQ-332 — Valid JSON and tidy tables.** All JSON written by the package is RFC 8259 valid: `NaN`/`±Inf` are written as `null` (`json.dumps(..., allow_nan=False)` after a conversion pass) and read back as `NaN`; numbers keep full `float64` precision (`repr`); tables written by `to_dataframe()`/`save()` are tidy (one row per channel × condition × component; one column per quantity; units in the sidecar `tables/<name>.json` as `{"column": {"unit": "s" | "uV" | "uV·s" | ""}}`). *Acceptance:* `tests/test_serialize.py::test_json_valid` saves a `GroupFit` containing `NaN` and `Inf` values, re-parses every `.json` file with `json.loads` (strict, `parse_constant` raising) and asserts no exception; `tests/test_tidy.py` asserts the column set of `GroupFit.to_dataframe("component")` equals the documented list in `docs/reference/tables.rst` and that every column has a unit entry. *Source:* [CODE §2.5]; [LIT-erptools-9]; [LIT-erptools-33] (duplicated values in exports to make files openable).

> **REQ-333 — `evoparam.reproduce`.** `evoparam.reproduce(result_path, *, data=None, rtol=1e-9, atol=0.0, check_environment=True) -> ReproduceReport` reloads a `.evoparam` container, re-runs the recorded `provenance.steps` in order from the recorded source (`provenance.source["path"]`, or `data` when the source is not available) with the recorded settings, preset versions and `random_state`, compares every array and table of the new result with the saved one within tolerance, and returns a report (`identical: bool`, `max_rel_diff`, per-step status, environment differences). It raises `EvoparamVersionError` when a recorded step's preset version no longer exists, and warns when package or dependency versions differ. *Acceptance:* `tests/test_reproduce.py` runs the chain of REQ-294 on the bundled fixture, saves, calls `reproduce` and asserts `identical` with `rtol=1e-9`; changing one setting in `settings.json` makes `identical` false with the differing step named; the CLI `evoparam reproduce <path> --json` returns the same report. *Source:* §4.1 principle 3; [LIT-ccep-11]; [LIT-artifact-7]; [LIT-domains-82] [LIT-domains-84] (traceability for groups that embed the package in regulated software, §13).

> **REQ-334 — Schema versioning and migration.** The schema string `evoparam-result/<major>.<minor>` follows: minor increments add optional fields and are read by any reader of the same major; major increments change or remove fields. `load()` of a newer major raises `EvoparamVersionError` naming both versions; `load()` of an older major applies the migrations registered in `evoparam/data/_migrations.py` (`{(from, to): callable}`) and records them in `provenance.steps` as `{"function": "evoparam.data.migrate", ...}`; `evoparam migrate <path>` rewrites a container in place. *Acceptance:* `tests/test_schema_version.py` keeps one frozen container per released major under `tests/reference/containers/` and asserts each loads and equals the expected object; a container with `schema: "evoparam-result/99"` raises. *Source:* [CODE §2.5] (no schema version), quality rule from the project brief.

> **REQ-335 — Environment capture.** `Provenance.environment` records `python`, `platform` (`platform.platform()`), `cpu` (`platform.processor()` or `/proc/cpuinfo` model name), `numpy`, `scipy`, the versions of every installed optional package of the extras table (`null` when absent), `backend` (`"numpy" | "jax" | "cupy"`), `jax_enable_x64`, `n_threads` (`OMP_NUM_THREADS`/BLAS threads), the git commit of the package when installed from source, and `lock_sha256` (REQ-309). *Acceptance:* `tests/test_provenance.py::test_environment_keys` asserts the key set; `evoparam.reproduce` reports the differences between the recorded and current environment. *Source:* [LIT-fooof-45], quality rule from the project brief.

> **REQ-336 — BIDS derivatives and NWB exports carry the same provenance.** `write_bids_derivatives` (§5.13, §10.2) writes `provenance.json` content into the `*_desc-evoparam_responses.json` sidecar (keys `Settings`, `Provenance`, `PackageVersion`, `Preset`) and into `dataset_description.json` `GeneratedBy`; `write_nwb` (§5.1, §10.3) stores the same JSON string in the `processing/evoparam` module description and in the `ResponseFitSettings` container. *Acceptance:* `tests/test_bids_derivatives.py::test_provenance_sidecar` and `tests/test_nwb_io.py::test_provenance` reload the written files and assert `Provenance.from_dict(...) == groupfit.provenance`. *Source:* [LIT-artifact-63] [LIT-artifact-7] (no provenance sidecar standard for artifact removal), [LIT-standards-12] (`GeneratedBy`), [LIT-erptools-9].

> **REQ-337 — Data hash and source record.** `Provenance.data_hash` is the sha256 of `data.tobytes()` concatenated with `times.tobytes()` and `ch_names` joined by `\0`, computed at container creation; `Provenance.source` records `path`, `format`, `reader`, `sha256` of the bytes actually read (the full file for local reads; the byte ranges for partial remote reads with `bytes_read` and `ranges`), and the dataset DOI when the path matches the registry. *Acceptance:* `tests/test_provenance.py::test_data_hash` asserts a changed sample changes the hash and that two readers of the same bytes give the same `source.sha256`. *Source:* §4.3.3; [LIT-datasets-4] (sampling rate and channel counts live only in per-run sidecars; the record must be self-contained).

> **REQ-338 — Methods text is derived from provenance only.** `ResponseFit.methods_text()` and `GroupFit.methods_text()` (specified in §5, `evoparam.reports`) read exclusively `provenance`, `settings` and `gof`; a result loaded from disk produces the same paragraph as the live object. *Acceptance:* `tests/test_reports.py::test_methods_text_roundtrip` asserts `load(save(fit)).methods_text() == fit.methods_text()` and that the text contains the package version, preset name and version, fit window, artifact method, existence method and the reproducibility sentence (`random_state`). *Source:* [LIT-fooof-17], [LIT-fooof-4], [LIT-citation-19]; §5 (`evoparam.reports`).

### 9.13 Tests to write for this section

| file | asserts |
|---|---|
| `tests/test_dependencies.py` | extras table equals `evoparam/_extras.py`; no top-level optional import (`ast`); `require` error message regex; GPL deny list absent from imports and extras (REQ-292, REQ-309) |
| `tests/test_core_only.py` (marker `core`, run in the `core-only` job) | `import evoparam` and `selftest()` with numpy + scipy only; every core function listed in REQ-291 runs on the bundled fixture (REQ-291) |
| `tests/test_import_graph.py` | edges ⊆ §4.2 table; no cycles (REQ-296) |
| `tests/test_deprecations.py` | registry of shims, warning text, two-minor-version rule (REQ-293) |
| `tests/test_citation.py` | `CITATION.cff` ↔ `pyproject.toml` version, DOI prefix, author (REQ-308) |
| `tests/test_provenance.py` | steps appended by each public function, JSON-serialisable, environment keys, data hash, lock hash (REQ-294, REQ-335, REQ-337, REQ-309) |
| `tests/test_determinism.py` | bit-identical reruns; reference values within 1e-9 on all three OS; no legacy RNG (REQ-295) |
| `tests/test_typing.py` | `mypy --strict` on typing examples; wrong example fails (REQ-301) |
| `tests/test_docs.py` | tutorial files per preset and per integration; reporting page; FAQ entries; API index completeness (REQ-303) |
| `tests/test_logging.py` | one WARNING per failed unit; `verbose` mapping; no `print` outside `cli` (REQ-304) |
| `tests/test_errors.py` | failed unit recorded with `status`/`message`; other units unaffected; `on_error="raise"` propagates (REQ-307) |
| `tests/test_network.py`, `tests/test_fetch.py` | socket guard on the `core` suite; checksum mismatch → `ReaderError` + deletion (REQ-306) |
| `tests/test_serialize.py`, `tests/test_tidy.py` | container round trip, checksum, valid JSON with `null` for NaN, tidy columns and units (REQ-330, REQ-332) |
| `tests/test_settings_log.py` | resolved settings, overrides, refit equality (REQ-331) |
| `tests/test_reproduce.py` | `reproduce` identical within 1e-9; detects a changed setting; CLI parity (REQ-333) |
| `tests/test_schema_version.py` | frozen containers per major load; newer major raises; migrations recorded (REQ-334) |
| `tests/test_reports.py::test_methods_text_roundtrip` | methods text identical after reload and contains the required fields (REQ-338) |
| `benchmarks/run_benchmarks.py` + CI job `perf` | P1–P6 within targets on the reference machine; ≤ 1.25 × baseline in CI (REQ-290, REQ-299) |
| CI workflows (`ci.yml`, `nightly.yml`, `release.yml`, `smoke.yml`) | matrix, required checks, coverage gates, pre-release job, wheel/sdist, PyPI smoke install, no secrets outside `release` (REQ-297, REQ-298, REQ-300) |

**Test map (CI-level requirements).** `.github/workflows/ci.yml` (REQ-297, REQ-298, REQ-305), `.github/workflows/nightly.yml` (REQ-300), `.github/workflows/docs.yml` (REQ-302).
## 10. Ecosystem and adoption

Integration with MNE-Python, BIDS, NWB, Neo, scikit-learn and LSL, the CLI, the online mode and community processes. Requirement ids REQ-310 … REQ-329; containers and names from §4, serialisation and provenance from §9.12.

Evidence: the ancestor spectral tool's only integration was one MNE example and its MATLAB ports drifted [LIT-fooof-23] [LIT-fooof-16]; the most persistent request in the MNE tracker is an ERP measurement tool (issue open six years, 43 comments) [LIT-needs-2]; MNE's new `mne.stats.erp` measures work on averages only, with no model, no uncertainty and no existence test [LIT-needs-4]; stimulation metadata cannot round-trip through MNE-BIDS [LIT-needs-53]; the only BIDS-native CCEP tool is GPL-3.0 and detects without parameterizing [LIT-standards-11] [LIT-needs-61].

### 10.1 MNE-Python

MNE stores data in SI units (V) as float64 and represents continuous, epoched and averaged data as `Raw`, `Epochs` and `Evoked` [LIT-standards-32]. The accepted pattern for third-party packages is the one of autoreject: consume MNE objects, return MNE objects plus a result container, expose `fit/transform` [LIT-standards-36].

```python
# evoparam/data/_mne.py  (requires extra "mne")
@classmethod
def TrialSet.from_mne(cls, epochs: "mne.BaseEpochs", *, picks=None, units: str = "uV",
                      metadata: bool = True, stim_columns: str | dict | None = "auto") -> "TrialSet"
def TrialSet.to_mne(self, *, ch_types: str | list[str] | None = None) -> "mne.EpochsArray"
@classmethod
def Evoked.from_mne(cls, evoked: "mne.Evoked", *, picks=None, units: str = "uV") -> "Evoked"
def Evoked.to_mne(self) -> "mne.EvokedArray"

# evoparam.fit dispatches on type
def fit(data: "TrialSet | Evoked | mne.BaseEpochs | mne.Evoked | list[mne.Evoked] | np.ndarray",
        preset: str | Preset | None = None, *, times=None, sfreq=None, picks=None,
        by: str | list[str] | None = None, n_jobs: int = 1, random_state=None,
        verbose=None, **overrides) -> GroupFit                  # full keyword list in §5.5.8

# results back to MNE
def GroupFit.to_annotations(self, *, what: str = "components", orig_time=None) -> "mne.Annotations"
def GroupFit.to_evoked_model(self) -> "mne.EvokedArray"       # fitted curves as an Evoked, comment="evoparam model"
```

Conversion rules: `from_mne` multiplies by 1e6 for `units="uV"` (V → µV) and records the factor in provenance; `ch_info.type` comes from `info.get_channel_types()`; `metadata` is `epochs.metadata` (pandas) merged with `epochs.events[:, 2]` mapped through `event_id` into `condition`; `event_times` from `events[:, 0] / sfreq + first_samp / sfreq`; bad channels from `info["bads"]` into `ch_info.bad`; channel positions from `info["chs"][i]["loc"][:3]` × 1000 (m → mm) with `coord_space` from the montage frame. `to_mne` inverts every step; `TrialSet.from_mne(e).to_mne()` reproduces `e.get_data()` within 1e-12 relative. Results never go into `Evoked.comment` or `info["description"]` strings: the source of truth is the tidy table (`GroupFit.to_dataframe()`), and `to_annotations` exists only so components can be displayed on MNE plots (one annotation per component: onset = component `onset`, duration = `offset − onset`, description `"<label>|<channel>"`, `ch_names=(channel,)`).

> **REQ-310 — MNE round trip.** `TrialSet.from_mne`, `TrialSet.to_mne`, `Evoked.from_mne`, `Evoked.to_mne` exist with the signatures above and the conversion rules stated; units, bad channels, metadata, event times and channel positions survive the round trip. *Acceptance:* `tests/test_mne_io.py` builds `mne.EpochsArray` objects with `eeg`, `ecog`, `seeg` and `dbs` channels, metadata and one bad channel, and asserts `assert_allclose(TrialSet.from_mne(e).to_mne().get_data(), e.get_data(), rtol=1e-12)`, equal `info["bads"]`, equal metadata frames, positions within 1e-9 m. *Source:* [LIT-standards-32], [LIT-standards-36], [LIT-needs-2].

> **REQ-311 — Fitting MNE objects directly.** `evoparam.fit` accepts `mne.Epochs`, `mne.Evoked` and lists of `mne.Evoked` (one condition each, `comment` used as condition label) and returns a `GroupFit` whose `to_dataframe()` is identical to the result of converting first; `GroupFit.to_annotations()` and `to_evoked_model()` return MNE objects for display. The gallery example `examples/mne/plot_fit_mne_sample.py` fits the auditory N1/P2 of the MNE `sample` dataset and plots the model. *Acceptance:* `tests/test_mne_io.py::test_fit_dispatch` asserts equality of tables for both paths; the example runs in the docs job (REQ-302). *Source:* [LIT-needs-4], [LIT-needs-7] (`get_peak` returns extrema, not peaks), [LIT-fooof-23].

Upstream contributions. MNE accepts BSD-3-Clause code only; code upstreamed into MNE or MNE-BIDS must not be derived from the Apache-2.0 files inherited by this package (MNE policy marked unverified in the evidence; treated as binding) [LIT-naming-14] [LIT-naming-56].

> **REQ-312 — Upstream policy.** Helpers proposed to MNE or MNE-BIDS (candidates: reading BIDS stimulation columns into `Annotations` extras, a `fix_stim_artifact` variant for `Evoked` [LIT-needs-39], a fractional-measure parity test) are written new, in a separate repository `evoparam-upstream` licensed BSD-3-Clause, never by moving or adapting a file listed in `NOTICE`; each proposal links the evoparam feature it mirrors. *Acceptance:* `evoparam-upstream/LICENSE` is BSD-3-Clause; its CI runs `scripts/check_no_inherited_code.py`, which fails when any 8-line window of its sources matches a window of the inherited files listed in `NOTICE` (normalised whitespace); the PR checklist in that repository contains the statement "not derived from Apache-2.0 inherited files". *Source:* [LIT-naming-14], [LIT-naming-56], [LIT-naming-12] (BSD-3 for non-derived modules intended for upstreaming).

### 10.2 BIDS

BIDS describes electrical stimulation in iEEG through the sidecar fields `ElectricalStimulation` (boolean) and `ElectricalStimulationParameters` (free text) and three example `events.tsv` columns (`electrical_stimulation_type`, `_site`, `_current`), which are not in the schema's column registry: units, pulse width, polarity, pulse count, inter-pulse interval and stimulator have no standard column [LIT-standards-1] [LIT-standards-2] [LIT-standards-3]. Deposits encode stimulation differently across groups [LIT-datasets-2] [LIT-datasets-13]; reading arbitrary `events.tsv` columns is an open MNE-BIDS request [LIT-needs-54].

Harmonisation layer `evoparam.io.bids.harmonize_stim_events(events: pd.DataFrame, sidecar: dict | None, *, mapping: dict | None = None) -> tuple[pd.DataFrame, HarmonizationReport]` maps source columns to the metadata columns of Table 4.1:

| target (Table 4.1) | recognised source columns (first match wins; case-insensitive) | unit rule |
|---|---|---|
| `stim_site` | `electrical_stimulation_site`, `stim_site`, `site`, `electrodes_involved_onset`* | `"A-B"`; separators `-`, `_`, `,`, space normalised to `-`; *only when `trial_type` contains "stim" |
| `stim_current_mA` | `electrical_stimulation_current`, `stim_current`, `current` | from `events.json[col]["Units"]` when present (`A`, `mA`, `uA`); else values < 0.1 are read as A and converted (warning `unit_inferred`) |
| `stim_pulse_width_us` | `electrical_stimulation_pulsewidth`, `electrical_stimulation_pulse_width`, `pulse_width`, `pulsewidth` | `Units` when present (`s`, `ms`, `us`); else values < 0.01 read as s |
| `stim_polarity` | `electrical_stimulation_type`, `stim_type`, `polarity` | vocabulary map to the five values of Table 4.1 when the phase order is stated in the value or in `ElectricalStimulationParameters`; otherwise `"unknown"`, with the raw value kept in `evoparam_stim_type_raw` |
| `stim_frequency_hz`, `stim_n_pulses` | `electrical_stimulation_frequency`, `frequency`; `n_pulses`, `pulse_count` | Hz; integer |
| `condition` | `trial_type` (always copied verbatim to `trial_type` as well) | — |

User mappings (`mapping={"my_col": "stim_current_mA"}`) override the table; every applied mapping and inferred unit is listed in the report and appended to provenance.

> **REQ-313 — Stimulation-event harmonisation.** `read_bids` calls `harmonize_stim_events` on every `*_events.tsv`, producing Table 4.1 columns with the unit rules above, and keeps all original columns under their own names. *Acceptance:* `tests/test_bids_events.py` uses fixtures shaped like ds004080 (Utrecht columns), ds003708/ds004977 (Mayo style) and a minimal file with only `trial_type`; it asserts the resulting `stim_site`, `stim_current_mA` (8 mA from `0.008` with no units, warning `unit_inferred`), `stim_pulse_width_us` (1000 from `0.001`), and that a user mapping overrides the default. *Source:* [LIT-standards-1] [LIT-standards-3], [LIT-datasets-2] [LIT-datasets-5] [LIT-datasets-13] (events metadata inconsistent across deposits; a harmonisation layer is needed), [LIT-needs-53] [LIT-needs-54].

> **REQ-314 — Stimulation-metadata validator.** `evoparam.io.bids.validate_stim_metadata(bids_root, *, subject=None, strict=False) -> pd.DataFrame` returns one row per run × field with `field`, `status` (`present`, `inferred`, `missing`, `inconsistent`), `n_events`, `message`, for the fields `stim_site`, `stim_current_mA`, `stim_pulse_width_us`, `stim_polarity`, `stim_frequency_hz`, `stim_n_pulses`, `ElectricalStimulation`, and the channel-table fields `reference`, `type`, coordinates; with `strict=True` any `missing` raises `ReaderError`; the CLI `evoparam validate-bids` prints it and exits 5 on missing fields with `--strict`. *Acceptance:* `tests/test_bids_validator.py` asserts the expected statuses on the three fixtures of REQ-313 and that the text report lists the missing pulse width of the minimal fixture as `missing`. *Source:* [LIT-datasets-2] [LIT-datasets-13] (validator reporting missing stimulation fields), [LIT-ccep-11] (18.1 % of CCEP studies do not state the montage), [LIT-standards-2].

BIDS derivatives. Derivative datasets require `dataset_description.json` with `DatasetType: "derivative"`, a `GeneratedBy` array whose first `Name` is a substring of the pipeline directory, `SourceDatasets`, the `desc` entity for variants, and filenames that do not collide with raw files [LIT-standards-12] [LIT-standards-13]. No BIDS standard exists for epochs, averages or model-fit outputs in electrophysiology: BEP021 is "Proposed" and its implementation pull request was closed [LIT-standards-14] [LIT-standards-15] [LIT-standards-16]. `evoparam.io.write_bids_derivatives(result, bids_root, *, out=None, desc="evoparam", overwrite=False)` (full parameter list in §5.1) writes:

```
derivatives/evoparam-<version>/
├── dataset_description.json      # Name "evoparam", BIDSVersion, DatasetType "derivative",
│                                 # GeneratedBy [{Name: "evoparam", Version, CodeURL, Description}],
│                                 # SourceDatasets [{DOI, URL, Version}] (from the registry or the raw dataset_description)
├── descriptions.tsv              # desc_id, description (BEP021 draft convention)
├── README                        # generated: what each file holds, preset(s) used, methods paragraph
└── sub-<label>/[ses-<label>/]ieeg/          # or eeg/ for scalp data; datatype taken from the source
    ├── sub-<label>_[ses-<label>_]task-<label>_[run-<index>_]desc-evoparam_responses.tsv
    ├── …_desc-evoparam_responses.json       # settings, provenance, column descriptions, units
    ├── …_desc-evoparam_existence.tsv        # one row per channel × stimulation site (or condition)
    ├── …_desc-evoparam_existence.json
    ├── …_desc-evoparam_trials.tsv           # only when trial fits were run
    ├── …_desc-evoparam_trials.json
    └── …_desc-evoparam_artifact.json        # method, window, parameters, replaced-sample mask
```

Columns of `*_desc-evoparam_responses.tsv` (one row per channel × stimulation site or condition × component; `n/a` for missing values as BIDS requires): `channel`, `stim_site` (or `n/a`), `condition`, `component` (label, e.g. `N1`, or `c<k>`), `component_index`, `shape`, `status` (`ok`, `no_response`, `failed`, `skipped`), `present` (existence decision), `latency`, `latency_se`, `latency_ci_low`, `latency_ci_high`, `amplitude`, `amplitude_se`, `amplitude_ci_low`, `amplitude_ci_high`, `polarity`, `fwhm`, `onset`, `offset`, `area`, `rise_time`, `decay_time`, `symmetry`, `sharpness`, `r2`, `rmse`, `nrmse`, `n_trials`, `window_start`, `window_stop`, followed by the classical measures of §5.5.6 with the prefix `classical_` and the stimulation columns with the prefix `evoparam_` (below). Times in seconds, amplitudes in the unit recorded in the sidecar (`"uV"` default). The JSON sidecar describes every column (`LongName`, `Description`, `Units`, `Levels` for categorical columns) as BIDS tabular sidecars do, and adds `Settings`, `Preset` (`{"Name", "Version"}`), `PackageVersion` and `Provenance` (REQ-336).

`*_desc-evoparam_existence.tsv`: `channel`, `stim_site`, `condition`, `method`, `statistic`, `p_value`, `p_value_raw`, `significant`, `onset`, `offset`, `duration`, `snr_db`, `alpha`, `correction`, `n_permutations`, `n_trials`. `*_desc-evoparam_trials.tsv`: `trial` (index in the source run), `onset` (s in the source recording), `channel`, `stim_site`, `component`, `shift`, `scale`, `latency`, `amplitude`, `fwhm`, `r2`, `converged`, `rejected`, `reject_reason`, `quality`. `*_desc-evoparam_artifact.json`: `Method`, `Window` (s), `Parameters`, `PackageVersion`, `QualitySummary` (median and 5th percentile of the 0–1 quality score) and `ReplacedSamples` (list of `{"trial", "channel", "start", "stop"}` intervals in seconds relative to onset, run-length encoding of `TrialSet.masks`), which fills a gap: none of the artifact-removal tools surveyed for this specification, MNE's blanking function included [LIT-artifact-63], records the replaced samples (this specification's survey).

Stimulation columns without a BIDS definition. Until a BEP defines them, derived and non-standard stimulation fields are written with the `evoparam_` prefix and defined in the sidecar: `evoparam_stim_current_mA`, `evoparam_stim_pulse_width_us`, `evoparam_stim_polarity`, `evoparam_stim_n_pulses`, `evoparam_stim_frequency_hz`, `evoparam_stim_charge_uC`, `evoparam_stim_type_raw`. The standard example columns `electrical_stimulation_site`, `_current`, `_type` are copied unchanged when they exist in the source. The project offers the column set, with units and the polarity vocabulary of Table 4.1, to the BEP021 and BEP037 (non-invasive brain stimulation) groups [LIT-standards-14] as a worked proposal.

> **REQ-315 — BIDS derivatives writer.** `write_bids_derivatives` produces the tree and files above for `GroupFit`, `ResponseTest`, `TrialFit` and `ArtifactResult` inputs (any subset), passes the BIDS validator (`bids-validator` ≥ 2.0 in derivative mode, warnings allowed, errors not), and `read_bids_derivatives` reads them back into equal result objects. *Acceptance:* `tests/test_bids_derivatives.py` writes the derivatives of the bundled synthetic BIDS fixture, runs the validator through `npx bids-validator@2 --datasetTypes derivative` in the `docs` CI job (or the Python schema validator `bidsschematools` in the unit test), asserts zero errors, asserts the `GeneratedBy[0].Name` substring rule, the column list and that every column has a sidecar description with `Units` where applicable, and asserts `read_bids_derivatives(...) == groupfit` (same tables, same provenance). *Source:* [LIT-standards-12] [LIT-standards-13] [LIT-standards-15], [LIT-erptools-9] (no standard output schema for ERP measurements), [LIT-needs-89] (tidy, complete parameter tables).

> **REQ-316 — Prefixed stimulation columns and BEP contribution.** Non-standard stimulation fields appear only with the `evoparam_` prefix and a sidecar definition; standard example columns are never renamed; a document `docs/bids_stimulation_columns.rst` lists the proposed columns, units and vocabularies and links the BEP issue opened with them. *Acceptance:* `tests/test_bids_derivatives.py::test_stim_columns` asserts the prefix rule on every written column not in the BIDS schema registry (checked against `bidsschematools`), and the docs test asserts the page exists. *Source:* [LIT-standards-2] [LIT-standards-14], [LIT-needs-53].

### 10.3 NWB, Neo and DANDI

NWB stores acquired signals as `ElectricalSeries` under `/acquisition`, trials as `TimeIntervals`, stimuli under `/stimulus` and processed results in `ProcessingModule`s [LIT-standards-20]. `ndx-events` (`EventsTable`) is the approved event type, and the only stimulation extension (`ndx-electrical-stim`, `StimTable` with `bipolar_pair`, `frequency`, `amplitude`, `pulse_width`) is dormant and not on PyPI [LIT-standards-24] [LIT-standards-27]. Neo is a data-representation layer with a large IO set (Block/Segment/AnalogSignal/Event/Epoch on `quantities`) [LIT-standards-29].

Reading. `evoparam.io.read_nwb(path, *, electrical_series: str | None = None, events: str | None = None, tmin, tmax, picks=None, stim_columns="auto") -> TrialSet` epochs one `ElectricalSeries` (or any `TimeSeries` with `data` of shape `(n_times, n_channels)`) around event times taken, in order of preference, from: (1) an `ndx-events` `EventsTable` named by `events`; (2) the `trials` table (`start_time` plus an optional `stim_onset` column); (3) a `TimeIntervals` table under `/intervals`; (4) a `StimTable` of `ndx-electrical-stim` when importable; (5) onsets of an intracellular stimulus series (`icephys` `CurrentClampStimulusSeries`/`VoltageClampStimulusSeries` under `/stimulus/presentation`, onset = first sample above 10 % of the absolute maximum of each sweep) for slice and in-vitro preparations. Columns of the event table are harmonised with the rules of REQ-313 (NWB `amplitude` in A, `pulse_width` in s). Electrode metadata (`location`, `group`, `x/y/z`, `rel_x…`, `reference`) fill `ch_info`; data are read lazily through `hdmf` datasets, chunked by trials.

Writing. `evoparam.io.write_nwb(result, nwbfile_or_path, *, module="evoparam", extension: bool = False)` adds to `processing/evoparam`:

| object | NWB type | columns / content |
|---|---|---|
| `evoparam_components` | `DynamicTable` | `electrode` (`DynamicTableRegion` into `/general/extracellular_ephys/electrodes`), `stim_site`, `condition`, `component`, `shape`, `status`, `present`, `latency` (s), `latency_se`, `amplitude` (V, converted from the result unit), `amplitude_se`, `polarity`, `fwhm`, `onset`, `offset`, `area`, `r2`, `n_trials`, `ci_level` |
| `evoparam_existence` | `DynamicTable` | `electrode`, `stim_site`, `condition`, `method`, `statistic`, `p_value`, `significant`, `onset`, `offset`, `duration`, `snr_db` |
| `evoparam_trials` | `DynamicTable` | `trial` (`DynamicTableRegion` into `trials` when present), `electrode`, `component`, `shift`, `scale`, `latency`, `amplitude`, `converged` |
| `evoparam_model` | `ElectricalSeries` (optional) | fitted curves of `GroupFit.to_evoked_model()` |
| module `description` | string | JSON of `settings` + `provenance` (REQ-336) |

With `extension=True` the same content uses the namespace `ndx-evoparam` (published from the NWB extension template to PyPI and the NDX catalog): `EvokedResponseFitTable` (columns above plus `param_names`/`params` ragged columns), `ResponseTestTable`, `TrialFitTable`, `StimulationEventsTable` (subclass of `EventsTable` with `stim_site`, `stim_current` (A), `stim_pulse_width` (s), `stim_polarity`, `stim_n_pulses`, `stim_frequency` (Hz)), and `ResponseFitSettings` (`LabMetaData` with `settings_json`, `provenance_json`, `package_version`, `preset_name`, `preset_version`).

> **REQ-317 — NWB reader.** `read_nwb` implements the event-source order and harmonisation above and returns a `TrialSet` equal to the one built from the same arrays in memory. *Acceptance:* `tests/test_nwb_io.py` writes synthetic NWB files (pynwb, in a temporary directory) for each of the five event sources and asserts the epoch onsets within one sample, `stim_current_mA` from an `amplitude` of `0.006` A equal to 6.0, and, on a 1 GB synthetic `ElectricalSeries` file (generated once with a fixed seed and cached in CI), that reading epochs covering 10 % of the samples keeps peak RSS increase below 0.25 GB. *Source:* [LIT-standards-20] [LIT-standards-24] [LIT-standards-27], [LIT-datasets-50] (Allen NWB sensory-evoked LFP as the animal benchmark).

> **REQ-318 — NWB writer and `ndx-evoparam`.** `write_nwb` adds the tables above; files pass `nwbinspector` with no `CRITICAL` or `BEST_PRACTICE_VIOLATION` messages; `ndx-evoparam` 0.1 is released to PyPI and submitted to the NDX catalog before evoparam 1.0. *Acceptance:* `tests/test_nwb_io.py::test_write_inspect` runs `nwbinspector.inspect_nwbfile` on the written file; round trip `read_nwb_results(path) == groupfit`; the 1.0 release checklist contains the NDX catalog link. *Source:* [LIT-standards-20], [LIT-standards-28] (nwbinspector, neuroconv, dandi), [LIT-standards-27] (no maintained stimulation extension).

> **REQ-319 — Neo reader.** `evoparam.io.read_neo(source: "neo.Block | str", *, segment: int | None = None, signal: int | str = 0, events: str | None = None, tmin, tmax, io_class: str | None = None) -> TrialSet` accepts a `neo.Block` or a path opened with `neo.io.get_io` (or `io_class`), takes `AnalogSignal` channels (units via `quantities`, rescaled to `uV`), event times from a named `Event` (labels → `condition`) or from `Epoch` starts, and concatenates segments as runs (`metadata.run`). *Acceptance:* `tests/test_neo_io.py` builds a `Block` with two segments, one `AnalogSignal` in mV and one `Event` and asserts shapes, the 1000 × unit factor, event labels in `condition`, and `run` values. *Source:* [LIT-standards-29], [LIT-standards-31] (Elephant analyses on Neo; no evoked fitting there).

DANDI: no account is needed (`read_nwb` takes a local path or a URL under the `fetch_*` rules of REQ-306). After 1.0 the project contributes an example notebook to the DANDI example-notebooks collection (one sensory-evoked LFP dandiset fitted with the `rodent_ep` preset) and asks for a listing on the NWB tools page (REQ-329).

### 10.4 scikit-learn API

The scikit-learn contract: `__init__` stores keyword arguments only, `fit` returns `self` and stores learned state in `*_` attributes, `get_params`/`set_params` with nested `step__param` syntax, `random_state` resolved inside `fit`, transformers keep the number of samples, compliance checked with `check_estimator`/`parametrize_with_checks` [LIT-standards-50]. scikit-learn is not a dependency: the base classes are implemented in `evoparam.utils._estimator` (a `BaseEstimatorMixin` with `get_params`/`set_params`/`__repr__`/`__sklearn_is_fitted__`, and `__sklearn_tags__` returning tags only when scikit-learn is importable).

```python
class ResponseModel(BaseEstimatorMixin):
    def __init__(self, preset: str | Preset | None = None, *, shapes=None, background=None, max_components=None,
                 min_fwhm=None, max_fwhm=None, polarity=None, fit_window=None, baseline=None,
                 sfreq: float | None = None, tmin: float | None = None, ci_level: float = 0.95,
                 n_jobs: int = 1, random_state=None, verbose=None, **other_settings): ...   # abridged; full keyword list and defaults in §5.5.8
    def fit(self, X, y=None) -> "ResponseModel"      # X: Evoked | TrialSet | mne object | ndarray (n_channels, n_times) with sfreq, tmin
    def predict(self, X=None) -> np.ndarray           # fitted curves, (n_channels, n_times)
    def transform(self, X) -> np.ndarray              # feature matrix (n_channels, n_features): latency/amplitude/fwhm per component slot, NaN when absent
    def score(self, X, y=None) -> float               # mean r2 over channels
    # learned attributes: fit_ (GroupFit), components_ (pd.DataFrame), feature_names_out_, n_features_in_
    def get_feature_names_out(self, input_features=None) -> np.ndarray

class TrialFitter(BaseEstimatorMixin, TransformerMixin):   # X: TrialSet or (n_trials, n_channels, n_times)
    def fit(self, X, y=None) -> "TrialFitter"           # learns the channel templates (ResponseModel)
    def transform(self, X) -> np.ndarray                # (n_trials, n_channels * n_components * 3): shift, scale, latency per component
class ArtifactRemover(BaseEstimatorMixin, TransformerMixin):
    def fit(self, X, y=None) -> "ArtifactRemover"       # estimates templates when the method needs them
    def transform(self, X)                              # same type and shape as X; masks carried on TrialSet inputs
```

`TransformerMixin` is a 10-line local class (`fit_transform`, `set_output` no-op) so that these objects work in `sklearn.pipeline.Pipeline` (e.g. `make_pipeline(ArtifactRemover("template"), TrialFitter(preset="erp"), StandardScaler(), LogisticRegression())` on P300 data) without importing scikit-learn at module level.

> **REQ-320 — Estimator contract.** `ResponseModel`, `TrialFitter` and `ArtifactRemover` follow the scikit-learn rules above: no logic in `__init__`, `fit` returns `self`, learned attributes end in `_`, `get_params`/`set_params` round trip including nested pipeline parameters, `clone` gives an unfitted copy, `__sklearn_is_fitted__` reflects state, calling `transform` before `fit` raises `NotFittedError` (scikit-learn's when available, else `evoparam.utils.NotFittedError`, a subclass of `ValueError` and `AttributeError`), and two fits with the same `random_state` give identical `fit_`. *Acceptance:* `tests/test_sklearn_api.py` asserts each rule, including `Pipeline.set_params(trialfitter__max_components=3)` and `pickle` round trip. *Source:* [LIT-standards-50], [LIT-standards-36] (autoreject precedent).

> **REQ-321 — `check_estimator` subset.** With scikit-learn installed, `parametrize_with_checks([ResponseModel(sfreq=1000., tmin=0.), TrialFitter(sfreq=1000., tmin=0.), ArtifactRemover(method="interpolate", sfreq=1000., tmin=0., window=(0., 0.002))], expected_failed_checks=EXPECTED)` passes, where `EXPECTED` (in `evoparam/utils/_sklearn_checks.py`) lists by name each check skipped and the reason (checks that feed random tabular data with fewer than 32 columns or integer dtypes to a time-series model, and sample-weight checks); at most 15 checks may be listed, and `check_get_params_invariance`, `check_set_params`, `check_no_attributes_set_in_init`, `check_dont_overwrite_parameters`, `check_fit_idempotent`, `check_estimators_unfitted`, `check_parameters_default_constructible`, `check_estimators_overwrite_params`, `check_n_features_in` must not be listed. *Acceptance:* the test runs in the `tests` job with `[tests]` installed. *Source:* [LIT-standards-50].

### 10.5 Command-line interface

The console script `evoparam` (entry point `evoparam.cli:main`) uses the standard-library `argparse` only, so it is installed with the core (REQ-291). Every sub-command accepts `--json`, `--verbose {warning,info,debug}`, `--n-jobs`, `--random-state`, `--config <file.json|toml>` (command-line values win) and writes nothing outside `--out`.

| command | example | does |
|---|---|---|
| `fit` | `evoparam fit <bids_root> --preset ccep --sub 01 [--ses 1] [--task SPESclin] [--run …] --out derivatives/` | read → harmonise events → artifact → fit → existence (→ trials when the preset says so) → `write_bids_derivatives`; also accepts a single file (`.vhdr`, `.edf`, `.fif`, `.nwb`, `.npz`) |
| `detect-events` | `evoparam detect-events run.vhdr --channels "PT*" --out events.tsv` | `detect_events` on continuous data; writes a BIDS-style `events.tsv` + `events.json` with EventTable columns |
| `clean` | `evoparam clean <bids_root> --sub 01 --method template --window -0.002 0.010 --out derivatives/` | artifact removal only; writes cleaned epochs (FIF when `mne` is installed, else npz) and `*_desc-evoparam_artifact.json` |
| `report` | `evoparam report derivatives/evoparam-0.4.0 --format html|pdf|md` | methods paragraph, summary tables and figures (§5, `evoparam.reports`) |
| `test` | `evoparam test <bids_root> --sub 01 --method baseline_permutation --out derivatives/` | existence test only (§5.6); writes `*_desc-evoparam_existence.tsv` |
| `benchmark` | `evoparam benchmark run --suite core|perf|accuracy|artifact --out results/`; `evoparam benchmark score --task T1 --predictions dir/ --suite core` | runs §8 suites and the §9.1 performance suite; scores external predictions (§8) |
| `simulate` | `evoparam simulate --modality ccep --n-channels 64 --n-trials 50 --seed 0 --out sim/` | writes a synthetic BIDS dataset with ground truth (`*_desc-truth_responses.tsv`) |
| `validate-bids` | `evoparam validate-bids <bids_root> [--strict]` | REQ-314 |
| `reproduce` | `evoparam reproduce result.evoparam` | REQ-333 |
| `migrate` | `evoparam migrate result.evoparam` | REQ-334 |
| `info` | `evoparam info` | versions, installed extras, registered plugins, backends |
| `presets` | `evoparam presets list`, `evoparam presets show ccep` | lists the presets of §6 or prints one with its citations and validation status |
| `cite` | `evoparam cite [--bibtex]` | prints the package citation and DOIs from `CITATION.cff` (REQ-346, REQ-391) |

`evoparam --version` prints the version followed by the research-use statement (REQ-015).

Exit codes: `0` success; `1` runtime error (`EvoparamError` not covered below); `2` usage error (argparse); `3` partial success (at least one channel or trial with `status="failed"`); `4` missing optional dependency (message gives the `pip` command); `5` input validation error (`DataError`, `ReaderError`, validator failure with `--strict`); `6` reproduction mismatch (`reproduce` with differences); `130` interrupted.

> **REQ-322 — CLI commands.** The thirteen sub-commands above exist with the stated behaviour, and `evoparam fit` on a BIDS dataset produces the same derivatives as the Python call `write_bids_derivatives(fit(read_bids(...), preset=...))`. *Acceptance:* `tests/test_cli.py` runs each sub-command with `subprocess` on the bundled synthetic BIDS fixture (`evoparam simulate` output), asserts exit code 0, the presence of the expected files, and byte-identical TSV files between the CLI and the Python path with the same `--random-state`. *Source:* [LIT-standards-11] (the existing CCEP tool offers GUI/CLI/JSON configuration, under GPL), [LIT-needs-61], quality rule from the project brief.

> **REQ-323 — Exit codes and JSON summary.** Exit codes follow the list above; with `--json`, stdout holds exactly one JSON object `{"command", "version", "status", "exit_code", "outputs": [paths], "n_units": {"ok", "no_response", "failed", "skipped"}, "warnings": [...], "duration_s"}` valid under RFC 8259 and stderr holds the logs. *Acceptance:* `tests/test_cli.py::test_exit_codes` triggers each code (missing package via `sys.modules` hiding, a corrupt file, a forced channel failure, a `reproduce` mismatch) and parses stdout with `json.loads`. *Source:* quality rule from the project brief.

### 10.6 Streaming and online mode (`evoparam.stream`)

Closed-loop products already use evoked responses as control signals (ECAP-controlled spinal cord stimulation with more than 100 updates per second; 50 ECAP samples per second in another system) and ERNA is proposed as a DBS programming signal [LIT-domains-42] [LIT-domains-43] [LIT-domains-53]. A minimal real-time estimator (amplitude, latency, existence) with latency well under the inter-pulse interval, separate from the offline model, is the requirement this specification draws for an analysis package (this specification's choice, from the control rates above and [LIT-domains-38]). LSL is the standard research transport (`liblsl`, `pylsl`; `mne-lsl` bridges streams to MNE objects) [LIT-standards-47] [LIT-standards-48] [LIT-standards-43].

```python
class StreamFitter:
    def __init__(self, template: ResponseFit | GroupFit, *, sfreq: float, ch_names: list[str],
                 window: tuple[float, float] | None = None, artifact_window: tuple[float, float] | None = None,
                 detector: str | Callable = "threshold", detector_channel: str | None = None,
                 max_shift: float = 0.005, existence_threshold_sd: float = 3.0,
                 buffer_seconds: float = 5.0, backend: str = "numpy"): ...
    def push(self, samples: np.ndarray, timestamps: np.ndarray | None = None) -> list["PulseEstimate"]
    def push_pulse(self, onset_time: float) -> None        # external trigger instead of the detector
    def reset(self) -> None
    @property
    def latency_stats(self) -> dict                        # p50, p95, p99, max compute time per pulse (ms)

@dataclass(frozen=True)
class PulseEstimate:
    pulse_time: float          # s, stream clock
    channel: str
    component: str             # label from the template
    amplitude: float           # data units (template amplitude × fitted scale)
    latency: float             # s (template latency + fitted shift)
    scale: float
    shift: float               # s
    existence_z: float         # (projection of the window on the template − baseline mean) / baseline SD
    present: bool              # existence_z > existence_threshold_sd
    r2: float
    compute_ms: float
    status: str                # "ok" | "artifact" (window overlaps a masked sample) | "late" (budget exceeded) | "saturated"
```

Method (fixed shape): the template's components are frozen; per pulse and channel the only free parameters are a common shift (grid over `±max_shift` at one-sample resolution, refined by parabolic interpolation) and a linear scale (closed-form least squares at each shift), plus the template's background re-estimated linearly. This makes the work O(n_channels × n_shift × n_window) with preallocated buffers and no Python allocation in the hot path; the artifact window is interpolated with the core `interpolate` method before projection. `evoparam.stream.LSLInlet(name: str | None = None, stype: str = "EEG", *, max_buflen=360)` (extra `stream`, `pylsl`) wraps an inlet and feeds `StreamFitter.push`; `evoparam.stream.LSLOutlet` publishes `PulseEstimate` rows as an irregular-rate LSL stream of float32 channels with JSON stream metadata. `evoparam.stream.replay(trialset_or_raw, fitter, chunk_size=32)` replays recorded data for validation.

> **REQ-324 — StreamFitter latency budget and agreement.** At 2048 Hz, 64 channels, a 0.5 s response window and `max_shift=0.005` s, the time from the `push` call that delivers the last sample of a pulse's window to its return is < 20 ms at the 99th percentile on one CPU core (`OMP_NUM_THREADS=1`) of the reference machine (§9.1); on replayed data the streamed `amplitude` and `latency` agree with the offline `fit_trials` estimates obtained with `mode="shift_scale"` within 1 % relative and one sample. *Acceptance:* `benchmarks/stream_latency.py` (part of the `perf` suite, REQ-299) replays 1 000 simulated pulses through `StreamFitter` and reports p50/p95/p99 compute times; `tests/test_stream.py` asserts the agreement on 200 simulated pulses and that chunk sizes 1, 32 and 1 024 give identical estimates. *Source:* [LIT-domains-42] [LIT-domains-43] [LIT-domains-38]; the 20 ms budget is this specification's choice.

> **REQ-325 — LSL adapter and intended use.** `LSLInlet`/`LSLOutlet` work with `pylsl` ≥ 1.16 (extra `stream`) and are tested against a local outlet; every `PulseEstimate` stream carries the metadata `intended_use = "research only; not a medical device"`; the `evoparam.stream` module docstring, the streaming tutorial and the CLI `info` output state that the package is not a medical device and is not validated for clinical decisions (§13). *Acceptance:* `tests/test_stream_lsl.py` (marker `lsl`, skipped when `pylsl` or its library is missing) creates an outlet with a synthetic signal, reads 10 pulses through `LSLInlet` and `StreamFitter`, and checks the outlet metadata string; `tests/test_docs.py` greps the tutorial for the statement. *Source:* [LIT-standards-47] [LIT-standards-48], [LIT-domains-82] [LIT-domains-83] (software that analyses physiological signals for clinical use is a medical device).

### 10.7 Documentation, tutorials and gallery datasets

The tutorial list complements REQ-303 and is the adoption-facing contract (the ancestor tool's sequential tutorials and its MNE example are cited as success factors [LIT-fooof-21] [LIT-fooof-23]; domain communities voice needs on forums, not issue trackers, and asked for a TMS-EEG Python example for seven years [LIT-needs-43] [LIT-needs-57]).

| tutorial | data | shows |
|---|---|---|
| `preset_ccep` | ds003708 (620 MB, CC0) partial read | detection → artifact → N1/N2 fit → existence map |
| `preset_ccep_seeg` | ds004977 (1.5 GB, CC0) partial read | adjusted CAR, both polarities, shared shapes |
| `preset_erp` | ERP CORE, one participant (N170, P3, MMN) [LIT-datasets-29] | classical measures vs model, condition contrast |
| `preset_sep`, `preset_erna`, `preset_tep`, `preset_rodent_ep`, `preset_primate_ep`, `preset_ecap` | bundled simulations (`evoparam.sim`, < 5 MB) and, for `rodent_ep`, an Allen sensory-evoked LFP excerpt when reachable [LIT-datasets-50] | preset defaults and their citations |
| `integration_mne`, `integration_bids`, `integration_nwb`, `integration_neo`, `integration_sklearn`, `integration_cli`, `integration_streaming` | bundled simulations; ds003708 for BIDS | the APIs of §10.1–§10.6 |
| `reporting_your_results` | any result above | methods paragraph, tables, derivatives, citation, `reproduce` |

Gallery datasets (< 2 GB, open licence): ds003708, ds004977 [LIT-datasets-5] [LIT-datasets-11], one ERP CORE participant [LIT-datasets-29], bundled simulations; each listed with DOI, licence and sha256 in `evoparam/io/registry.json`.

> **REQ-326 — Tutorials and gallery datasets.** The 17 tutorials of the table exist as sphinx-gallery scripts, execute in the docs job on the listed data, end with the generated methods paragraph, and download only through `fetch_*` with checksums. *Acceptance:* `tests/test_docs.py` (REQ-303) plus a check that each tutorial file calls `methods_text()` and that every dataset it names is in the registry with a sha256. *Source:* [LIT-fooof-21] [LIT-fooof-23], [LIT-needs-43] [LIT-needs-57], [LIT-datasets-4] (only four sets under 2 GB suit CI).

### 10.8 Community, governance and extension

Files at the repository root: `CONTRIBUTING.md` (development install with `uv`/`pip -e .[all,tests,docs]`, test and docs commands, coding rules of §9.9, how to add a preset, shape, reader or dataset, DCO sign-off), `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1, enforcement contact of the project author), `GOVERNANCE.md`, `SECURITY.md`, `SUPPORT.md`, `.github/ISSUE_TEMPLATE/` (`bug.yml` with `evoparam info` output field, `feature.yml`, `new_preset.yml` asking for literature keys and a test dataset, `new_dataset.yml`, `config.yml` redirecting questions to Discussions).

Governance: during 0.x the project author (Yann V. Bellec) is the final decision maker (BDFL) and sole release manager; at 1.0 a steering group of three (the author plus two maintainers elected by maintainers with ≥ 10 merged pull requests in the previous 12 months, two-year terms) takes decisions by lazy consensus with a 7-day window and majority vote when consensus fails; the steering group owns releases, the roadmap and the code of conduct. Roadmap: a public GitHub Project board with columns `Next minor`, `Later`, `Needs data`, `Needs decision`, each item linked to a REQ id. Release cadence: one minor release about every three months during 0.x, patch releases as needed, at most one breaking change set per minor under REQ-293. Support channels: GitHub Discussions (questions, show-and-tell), the Neurostars tag `evoparam`, issue tracker for bugs only; median first response target 7 days (reported in the yearly status post). Contributions: inbound = outbound (Apache-2.0) with the Developer Certificate of Origin (`Signed-off-by` on every commit, checked by a DCO app on pull requests); no contributor licence agreement during 0.x; any change of this policy is a steering-group decision recorded in `GOVERNANCE.md` (the institutional questions on CLA versus DCO are listed in §13) [LIT-naming-12].

> **REQ-327 — Community files.** The files, templates and policies above exist and are linked from the README and the docs; JOSS requires CONTRIBUTING and a support/governance statement for single-author projects [LIT-standards-58]. *Acceptance:* `tests/test_repo_files.py` asserts the presence of each file, the DCO statement in `CONTRIBUTING.md`, the three-person steering-group rule in `GOVERNANCE.md`, and the issue-template names; the DCO check is a required status on `main`. *Source:* [LIT-standards-58], [LIT-fooof-11] (single-maintainer delays), [LIT-needs-57].

Plugins. Third-party packages extend the registries through `importlib.metadata` entry points:

```toml
# in a plugin's pyproject.toml
[project.entry-points."evoparam.shapes"]
gamma = "evoparam_gamma:GammaShape"            # object satisfying ComponentShape (or Background with kind="background")
[project.entry-points."evoparam.presets"]
mep_hand = "evoparam_mep:MEP_HAND"             # a Preset instance
[project.entry-points."evoparam.readers"]
tdt = "evoparam_tdt:read_tdt"                  # callable(path, **kw) -> TrialSet; declares `extensions = (".tev", ".tsq")`
```

Loading is lazy (first access to `SHAPES`, `PRESETS` or `read`), can be disabled with `EVOPARAM_DISABLE_PLUGINS=1`, validates each object (`isinstance(obj, ComponentShape)` runtime-checkable Protocol; `Preset` type; reader signature), refuses name collisions with built-ins or other plugins (`SettingsError` naming both distributions), and records every plugin used by a result in `provenance.environment["plugins"] = {name: "dist==version"}` so `reproduce` can report a missing plugin.

> **REQ-328 — Plugin entry points.** The groups `evoparam.shapes`, `evoparam.presets` and `evoparam.readers` are loaded with the rules above; adding a shape or preset requires no change to the package. *Acceptance:* `tests/test_plugins.py` installs a dummy distribution from `tests/plugins/evoparam_dummy` (editable install in a temporary venv, or a monkey-patched `entry_points`), asserts its shape fits a simulated waveform, its preset resolves, its reader opens a file, a colliding name raises `SettingsError`, and the plugin appears in provenance; `docs/contributing/add_shape.rst` walks through the same dummy plugin. *Source:* [LIT-fooof-19] (the ancestor's 2.0 opened its fit-function modes and fitting algorithms to custom extensions), [LIT-naming-12] (open core through entry points).

Integration targets after 1.0 (approached only once the API is frozen, REQ-293):

| target | integration |
|---|---|
| MNE-Python | gallery example in MNE's documentation using `evoparam.fit` on `Evoked`; parity tests between `evoparam` classical measures and `mne.stats.erp` (`compute_peak`, `compute_frac_area_latency`, `compute_sme`) [LIT-needs-4]; BSD-3 helpers of REQ-312 |
| MNE-BIDS | reading arbitrary `events.tsv` stimulation columns and the `evoparam_` column proposal (issues on stimulation markings and arbitrary columns) [LIT-needs-53] [LIT-needs-54] [LIT-standards-41] |
| SpikeInterface | a preprocessing wrapper exposing the `evoparam.artifact` `template` and `tail` (with `tail="joint"`) methods next to its blanking `remove_artifacts` step; evoked-LFP fits from recordings with stimulation triggers [LIT-artifact-62] |
| pynapple | converters `TrialSet.from_pynapple(tsdframe, ts_events, minmax)` / `to_pynapple`, i.e. peri-event `TsdFrame` → `TrialSet` |
| Elephant / Neo | `read_neo` (REQ-319) contributed as a documented recipe; evoked-LFP measures returned as Neo `Epoch` annotations [LIT-standards-31] |
| MOABB | a `TrialFitter`-based pipeline (component latency/amplitude features) registered for the P300 and ErrP paradigms [LIT-domains-66] |
| DANDI / NWB | `ndx-evoparam` in the NDX catalog, an example notebook on a DANDI dandiset, listing on the NWB tools page [LIT-standards-28] |
| conda-forge | `evoparam` feedstock created through `staged-recipes` at 1.0 [LIT-standards-72] |
| MATLAB users | a reader for the `.evoparam` container and BIDS derivatives in MATLAB (read side only, BSD-3, no GPL code), since about 80 % of 2025 citations of EEG tools go to MATLAB toolboxes [LIT-citation-33] |

> **REQ-329 — Integration roadmap.** Each target of the table has a tracking issue on the roadmap board with the integration described above and an owner; at the 1.0 release at least the MNE example, the DANDI/NWB listing and the conda-forge feedstock are done. *Acceptance:* the 1.0 release checklist (`RELEASING.md`) contains these three items with links; the roadmap board lists all nine targets. *Source:* [LIT-needs-2] [LIT-needs-4], [LIT-standards-28] [LIT-standards-72], [LIT-citation-33], [LIT-fooof-16] (ports that drift when not owned).

### 10.9 Tests to write for this section

| file | requirement(s) checked (assertions are those of each acceptance criterion) |
|---|---|
| `tests/test_mne_io.py` | REQ-310, REQ-311 |
| `evoparam-upstream`: `scripts/check_no_inherited_code.py` | REQ-312 |
| `tests/test_bids_events.py`, `tests/test_bids_validator.py` | REQ-313, REQ-314 |
| `tests/test_bids_derivatives.py` | REQ-315, REQ-316, REQ-336 |
| `tests/test_nwb_io.py`, `tests/test_neo_io.py` | REQ-317, REQ-318, REQ-319 |
| `tests/test_sklearn_api.py`, `tests/test_sklearn_checks.py` | REQ-320, REQ-321 |
| `tests/test_cli.py` | REQ-322, REQ-323 |
| `tests/test_stream.py`, `tests/test_stream_lsl.py`, `benchmarks/stream_latency.py` | REQ-324, REQ-325 |
| `tests/test_docs.py`, `tests/test_repo_files.py`, `tests/test_plugins.py` | REQ-326, REQ-327, REQ-328 |

**Test map (process requirements).** `tools/check_no_inherited_code.py` (REQ-312, run in the separate upstream repository), `tests/test_release_checklist.py` (REQ-329).
## 11. Citation and publication strategy

This section fixes how the package earns citations, which articles carry them, in which order the public steps happen relative to UAB's technology-transfer process, and how success is measured. Competitor and ancestor tools are named here because the strategy depends on them; the vocabulary of the spectral ancestor is not reused.

### 11.1 The citation mechanism

A method gets cited when the sentence that cites it is the easiest sentence to write. Five pieces of the package exist for that reason.

| mechanism | what the user does | what the package does | why it produces citations |
|---|---|---|---|
| the citation sentence | pastes `ResponseFit.methods_text()` into a Methods section | writes "Evoked responses were parameterized with evoparam vX.Y.Z (DOI 10.5281/zenodo.NNN; <methods paper>) using the `ccep` preset: fit window …, shapes …, background …, existence test …" from `Provenance` (§4.3.3), with the two references already formatted | the spectral ancestor's `methods_report_text()` is the documented driver of its 2,670 citations [LIT-fooof-1][LIT-fooof-17]; papers that paste generated text cite the generator [LIT-fooof-17][LIT-fooof-4] |
| `CITATION.cff` + Zenodo DOI | runs `evoparam.cite()` or reads the GitHub "Cite this repository" box | one concept DOI for the project and one versioned DOI per release, both listed in the methods text | a DOI accrues citations from the first 0.x release, before any paper exists; the ancestor's PyPI package predated its paper by three years [LIT-fooof-8] |
| the public benchmark (§8) | compares a new detector or fitter against the leaderboard | publishes tasks, labelled ground truth, metrics and a submission format | every later method paper must cite the benchmark to report a comparison; the ancestor's own benchmarking ticket has been open since 2023 and is deferred until after its 2.0 release [LIT-fooof-49] |
| presets (§6) | writes `evoparam.fit(trials, preset="ccep")` | resolves every setting from literature-justified defaults and records them | one line is enough to reproduce a figure in a tutorial, a poster or a reviewer's check; the ancestor's one-object, four-setting API is the core of its documented quick start [LIT-fooof-14][LIT-fooof-15] |
| name + phrase | searches "evoked response parameterization" | every title (README, docs home, paper titles, abstracts) pairs `evoparam` with the phrase "evoked response parameterization" | the "evo" prefix collides with evolutionary computation in search results [LIT-naming-12]; the phrase recovers searchability the way "spectral parameterization" does for the ancestor [LIT-naming-38] |

Two rules follow. (1) The methods paragraph is generated by default and stored inside every saved result (REQ-390), not produced by an optional call: only 56 % of clinical reports that used the ancestor documented their settings although the call existed [LIT-fooof-4]. (2) The package name never changes. The ancestor's rename produced a FieldTrip thread titled "The Technique Formerly Known As FOOOF" and ports whose maintenance status is unknown [LIT-fooof-16]; the cost of a rename is paid by every downstream citation.

### 11.2 Lessons from the citation curves

Numbers collected 2026-10-06 (OpenAlex unless stated).

| paper | type | total | per year | key dates |
|---|---|---|---|---|
| Donoghue et al. 2020, Nat Neurosci (spectral parameterization) [LIT-fooof-1][LIT-fooof-3] | method + package | 2,533 (Crossref 2,670; ~75,000 accesses) | 2018: 3 · 2019: 13 · 2020: 48 · 2021: 178 · 2022: 283 · 2023: 425 · 2024: 506 · 2025: 600 · 2026 to Oct 6: 472 | PyPI 0.1.0 2017-12-27; bioRxiv 2018; 1.0.0 2020-04-27; paper online 2020-11-23 [LIT-fooof-2][LIT-fooof-8] |
| Lopez-Calderon & Luck 2014, Front Hum Neurosci (ERPLAB) [LIT-erptools-1][LIT-erptools-9] | measurement toolbox | 2,753 (Crossref 2,231) | flat ~300/yr: 2021: 314 · 2022: 288 · 2023: 302 · 2024: 300 · 2025: 307 | >7,000 downloads before the paper |
| Gramfort et al. 2013, Front Neurosci (MNE-Python) [LIT-erptools-22] | software paper | 4,072 | 815 in 2025 | second paper 2014 NeuroImage: 2,220 total, 274 in 2025 [LIT-erptools-23] |
| Oostenveld et al. 2011 (FieldTrip) [LIT-erptools-30] | toolbox | 11,502 | 1,211 in 2025 | — |
| Tadel et al. 2011 (Brainstorm) [LIT-erptools-36] | toolbox | 3,980 | 409 in 2025 | — |
| Delorme & Makeig 2004 (EEGLAB) [LIT-erptools-15] | toolbox | 25,334 | 2,605 in 2025 | — |
| Miller et al. 2023, PLoS Comput Biol (CRP) [LIT-fooof-92] | method, MATLAB scripts | 37 | — | 2 patients; no package |

What the curves say. (a) Software papers are cited for a decade or more and the curves rise or stay flat; none decays. (b) A method paper tied to a pip-installable package with a problem framing (Donoghue 2020) reached 600/yr in its fifth year, more than ERPLAB's measurement toolbox (300/yr) and far more than a method published as scripts (CRP, 37 total in three years). (c) Two citable objects (MNE 2013 + 2014) both keep accruing; the software paper collects more than the detailed paper. (d) Adoption preceded the paper in every case: the ancestor had a 1.0 release, nine tutorials, a FAQ and a preprint before the journal article [LIT-fooof-8][LIT-fooof-21][LIT-fooof-25][LIT-fooof-2]; ERPLAB had 7,000 downloads.

| lesson (evidence) | concrete action in this project | spec section |
|---|---|---|
| PyPI release three years before the paper; 1.0 seven months before [LIT-fooof-8] | 0.x on PyPI the day HIIE clears release; 1.0 tagged before the methods preprint | §12, §11.4 |
| bioRxiv preprint cited for two years before the article [LIT-fooof-2] | preprint at each submission (REQ-393); preprint DOI listed in `CITATION.cff` as `preferred-citation` until the article exists | §11.4 |
| one object, four settings, one range [LIT-fooof-14][LIT-fooof-17] | `evoparam.fit(data, preset=...)`; `ResponseModel` with preset-resolved settings; `report()` one-liner that fits, plots and prints | §4.3, §6 |
| simulation validation with 1,000 cases per condition and deliberate model violations [LIT-fooof-1] | `evoparam.sim` ground truth per modality; validation suite with skewed, overlapping, jittered and non-stationary violations | §5 (`sim`), §7 |
| comparison with existing methods and nine human raters [LIT-fooof-1] | benchmark tasks against threshold detectors, CRP, classical measures and the ER-detect human annotations (kappa 0.60) [LIT-ccep-8] | §8 |
| methods-text generator and a "Reporting & Referencing" tutorial [LIT-fooof-17][LIT-fooof-21] | `methods_text()` on every result; generated by default; reporting tutorial is tutorial 09 | §5 (`reports`), REQ-390 |
| FAQ that states limits in plain language [LIT-fooof-25] | FAQ page written before the software paper; limits section of the methods paper mirrors it | §9 (docs), §16 |
| sibling packages and MATLAB ports put the tool in front of non-Python users [LIT-fooof-27][LIT-fooof-29][LIT-fooof-30] | MNE, BIDS, NWB first-class (§10); a thin MATLAB caller is documented, not a port (ports drift [LIT-fooof-16]) | §10 |
| rename cost adoption confusion [LIT-fooof-11][LIT-fooof-16] | name frozen at first release (REQ-394) | §11.1 |
| single maintainer, 30 months of release candidates [LIT-fooof-5][LIT-fooof-9][LIT-fooof-11] | two-maintainer rule before 1.0; release cadence in §9 | §9, §15 |
| ERPLAB: measurement vocabulary users already know [LIT-erptools-1] | classical measures emitted next to model parameters (`ResponseFit.classical`) | §5.5.6 |
| 67 % of clinical reports used the ancestor but 30 % reported goodness of fit [LIT-fooof-4] | goodness of fit, existence decision and warnings inside the generated paragraph, not optional | REQ-390 |

### 11.3 The three articles

Authorship principle for all three: the project author is first author; the Smith lab PI is senior author; collaborators are co-authors in proportion to a concrete contribution (labelled data, a dataset not otherwise public, a module, an analysis), following ICMJE criteria. Reuse of CC0 OpenNeuro data earns a citation of the data descriptor, not authorship; reuse of restricted or newly consented data (UAB, HUP cohorts) earns authorship for the clinicians who collected and labelled it.

#### 11.3.1 Software paper

| item | decision |
|---|---|
| venue | Journal of Open Source Software (JOSS); requires an OSI-approved licence, tests, documentation, contribution guidelines [LIT-naming-24]; fallback: SoftwareX; Neuroinformatics for a longer software description |
| preprint | bioRxiv (section Neuroscience) at submission; arXiv q-bio.NC if bioRxiv declines software-only text |
| title (working form) | "evoparam: evoked response parameterization in Python" |
| length | JOSS limit (~1,000 words, 2 figures); the preprint carries the full figure set below |
| content | statement of need (no released, documented, citable package parameterizes evoked waveforms [LIT-fooof-39][LIT-fooof-59][LIT-fooof-60]; MNE exposes one peak primitive [LIT-erptools-17]; ERPLAB is GPL MATLAB [LIT-erptools-1]); the chain event → artifact → response → numbers (§4.1); containers and estimator API; presets; methods text; I/O; a comparison table against ERPparam, CRP, ERPLAB, MNE; acknowledgement of the Apache-2.0 lineage per `NOTICE` |
| datasets | ds004080 one subject (CCEP, ECoG, 2048 Hz) [LIT-datasets-2]; ERP CORE P3b and N170 from the 40-participant release [LIT-datasets-29][LIT-datasets-30]; ds002094 one subject (TMS-EEG) [LIT-datasets-17]; Allen Neuropixels flash-evoked LFP one session (1250 Hz release) [LIT-datasets-50] |
| comparison methods | none beyond the feature table; the paper claims only that the software exists, installs and runs |
| claims and evidence | "installs with numpy and scipy only" → CI matrix (REQ-291); "one call per modality" → the four tutorials; "numbers carry provenance" → round-trip test of `methods_text()` after save/load (REQ-294); "same API across modalities" → the four-panel figure |
| figures | **F1** pipeline schematic: the six containers of §4.3 as boxes, arrows labelled with the public functions (`read`, `detect_events`, `remove_artifact`, `fit`, `test_response`, `fit_trials`, `report`). **F2** four panels, one per modality (ds004080 CCEP channel, ERP CORE P3b at Pz, ds002094 TEP at C3, Allen flash-evoked LFP in V1): raw average in grey, fitted model in colour, individual components as filled curves, background as dashed, existence p-value and R² in the panel corner; identical call shown under each panel with only `preset=` changing. Preprint only: **F3** the generated methods paragraph for F2a with each phrase colour-linked to the `Provenance` field it came from. **F4** time and memory: fit time per channel (log axis) versus number of samples at 1, 2, 4.8 and 30 kHz, for one channel, one subject (ds004080, 133 channels) and the trial-level fit (10 trials × 133 channels), on one laptop core; the ancestor's "10–20 ms per spectrum" [LIT-fooof-1] as a reference line. **F5** feature matrix (rows: event detection, artifact removal, background, existence test, trial fits, uncertainty, shared shapes, time-frequency, simulation, benchmark, methods text, MNE/BIDS/NWB; columns: evoparam, ERPparam, CRP, ERPLAB, MNE), filled from [LIT-fooof-74][LIT-fooof-92][LIT-fooof-93][LIT-fooof-94]. |
| code that must exist | 0.x on PyPI and conda-forge; docs site with ≥ 9 tutorials; test coverage at the REQ-298 thresholds (≥ 90 % core, 100 % `shapes`) with ground-truth assertions (§7); `CITATION.cff`, Zenodo DOI, `CONTRIBUTING.md`, code of conduct; `methods_text()`; the four presets used in F2 |

#### 11.3.2 Main methods paper

| item | decision |
|---|---|
| venue | Nature Methods (Article); fallbacks in order: Nature Biomedical Engineering, eLife, PLOS Computational Biology, Journal of Neuroscience Methods |
| preprint | bioRxiv when 1.0 is feature-complete, before journal submission (REQ-393) |
| title (working form) | "Evoked response parameterization: a component model with trial-level uncertainty for stimulation- and sensory-evoked potentials" |
| length | ~4,500 words main text, 6 main figures, Methods section ~3,000 words, Extended Data (≤ 10), Supplementary Note with the full benchmark tables |
| problem framing | there is no standard for measuring an evoked response: of 187 CCEP studies at full-text review, 49.7 % did not state a detection method and 9.1 % used visual inspection only; among the 72 that did, 68.1 % used a threshold with thresholds from 2.6 to 6 SD and N1 windows starting anywhere from 5 to 16 ms [LIT-ccep-11]; human inter-rater agreement on CCEP presence is kappa 0.60 on 59,178 responses [LIT-ccep-8]; TEP pipelines disagree on early peaks (concordance −0.03 to 0.59 under 100 ms) [LIT-artifact-7]; IONM guidelines reject fixed 50 %/10 % alarm thresholds for criteria that account for trial-to-trial variability [LIT-domains-25]; trial averages hide the variability that carries clinical information (single-trial CCEP variability separates epileptogenic tissue, AUC 0.71–0.74) [LIT-variability-1] |
| the model | components with explicit shapes over a slow background (§5.5); existence test with a permutation null from the baseline (§5, `existence`); trial-level fit with shared shape and per-trial latency shift and amplitude scale, pooled hierarchically across trials, channels and subjects (§5, `trials`, `hierarchical`); model selection on whitened residuals rather than thresholds (§5.5.3) |
| datasets | CCEP: ds004080 (74 subjects, ECoG, 2048 Hz) [LIT-datasets-2]; Mayo sEEG family ds003708, ds004457, ds004696, ds004977 (4800 Hz) [LIT-datasets-5][LIT-datasets-11]; ds006254 (31 subjects, sEEG) if events are recoverable [LIT-datasets-13]. ERP: ERP CORE, 40 participants, 7 components [LIT-datasets-29]. TMS-EEG: ds002094 plus the 24-participant figshare set [LIT-datasets-17]. Animal: Allen Neuropixels flash-evoked LFP [LIT-datasets-50]. Scalp responses with known onset: Localize-MI (7 participants, 61 sessions, 8 kHz hd-EEG during sEEG pulses) [LIT-datasets-24] |
| comparison methods | threshold detectors at 2.6, 3.4, 5 and 6 SD with each paper's window [LIT-ccep-5][LIT-ccep-7][LIT-ccep-13]; ER-detect defaults [LIT-datasets-62]; CRP (shape-free canonical response; re-implemented from the paper, validated against the authors' scripts) [LIT-fooof-92]; classical peak, mean, fractional-area and fractional-peak measures (ERPLAB definitions, re-implemented [LIT-erptools-1]); MNE `get_peak` [LIT-erptools-17]; a Gaussian-only fit without background (the ERPparam model as published in its repository [LIT-fooof-74]); RPAGP for trial-level latency [LIT-fitting-3]; human raters (ER-detect annotation set if released; otherwise two UAB raters on a 1,000-response subset) |
| figures | **F1** the problem: (a) histogram of N1 window starts and ends and of z-thresholds from the 72 described studies [LIT-ccep-11]; (b) one ds004080 channel with the detection decision of six published rules overlaid (present/absent per rule); (c) same channel, 10 single trials in grey, average in black, showing latency jitter that the average smears. **F2** the model: (a) schematic of components + background + artifact tail with parameter names; (b) a CCEP, a TEP, an ERP and an ERNA fit with components filled; (c) hierarchy diagram trial → channel → subject with the pooled quantities; (d) existence test: null distribution from baseline permutations and the observed statistic. **F3** simulation validation: median absolute error of latency (ms), amplitude (%) and width (ms) versus SNR (−10 to 20 dB) and trial count (5–100) for each method; columns for component overlap 0, 0.5 and 1 width; rows for model violations (skewed shapes, exponential artifact tail inside the window, slow drifting background, latency jitter 0–5 ms SD); coverage of 95 % intervals as a separate panel. **F4** detection benchmark: ROC and precision-recall for evoparam's existence test, each threshold rule, CRP's extraction significance and ER-detect against (a) simulated ground truth and (b) human labels, with the kappa 0.60 human ceiling as a line; confusion examples. **F5** trial-level and hierarchical results: (a) per-trial latency and amplitude posteriors for one connection across 10 trials; (b) jitter SD per connection across ds004080 versus distance; (c) shrinkage: trial estimates with and without pooling at 10, 30 and 100 trials; (d) state covariate (sleep stage or anaesthetic where labelled) effect on amplitude with credible intervals. **F6** across modalities and the reporting function: (a) ERP CORE: component latencies and amplitudes for the 7 components across 40 participants compared with the published values; (b) TEP: N15–P180 presence per subject with sensory-contamination flag; (c) Allen LFP: flash-evoked components across cortical depth; (d) the generated methods paragraph and the reporting checklist (window, threshold, montage, trials, charge, detector version) [LIT-ccep-11]. **Extended Data** reference-montage sensitivity (bipolar, CAR, adjusted CAR [LIT-ccep-18]) with N1 latency shift (47.4 vs 70.7 ms reported under montage change [LIT-ccep-27]); dose-response normalisation across current; runtime; failure modes. |
| claims and evidence | "the component model recovers latency within 1 ms and amplitude within 5 % at SNR ≥ 5 dB with 10 trials" → F3 simulations with ground truth; "the existence test matches human raters at the inter-rater ceiling" → F4 kappa ≥ 0.60 against labels [LIT-ccep-8], with sensitivity/specificity for each published rule on the same data; "trial-level estimates are calibrated" → F3 coverage panel (95 % intervals contain truth 93–97 %); "pooling improves single-trial estimates at low trial counts" → F5c error reduction versus independent fits at 10 trials; "the same model applies across modalities" → F6 with preset-only changes; "reporting is complete by construction" → F6d plus the checklist fields enumerated; "results are montage-sensitive and the tool reports it" → Extended Data |
| code that must exist | 1.0 feature-complete: `models`, `existence`, `trials`, `hierarchical` (empirical Bayes backend at minimum; MCMC backend for F5), `artifact` (interpolate, template, tail with `tail="joint"`), `sim` for all six modalities, `benchmark` with public leaderboard (REQ-392), `reports`; validation suite of §7 passing; every number in the figures regenerated by a `make figures` target from a tagged version |

#### 11.3.3 Benchmark paper (with the Smith lab)

| item | decision |
|---|---|
| venue | Scientific Data (Data Descriptor); fallbacks: NeuroImage (Methods), Clinical Neurophysiology (the lab's venue for its CCSR work [LIT-tf-2]) |
| title (working form) | "A public benchmark for evoked response parameterization: labelled ground truth, tasks and a leaderboard across CCEP, ERP, TMS-EEG and animal LFP" |
| length | ~4,000 words (Background, Methods, Data Records, Technical Validation, Usage Notes), 6 figures, data on Zenodo and OpenNeuro derivatives |
| content | task definitions (detection, latency, amplitude, width, trial-level jitter, duration, existence calibration); simulated sets with ground truth per modality; labelled real sets (expert labels on ds004080 and the Mayo sets; UAB and HUP cohorts under the lab's IRB); metrics and scoring code; leaderboard with submission format; the ECoG-versus-sEEG comparison and the time-frequency-zone comparison that the lab's own work requires [LIT-tf-3][LIT-tf-4][LIT-tf-11] |
| datasets | ds004080, ds003708, ds004457, ds004696, ds004977, ds006254 (all CC0) [LIT-datasets-2][LIT-datasets-5][LIT-datasets-11][LIT-datasets-13]; ERP CORE [LIT-datasets-29]; ds002094 [LIT-datasets-17]; Allen LFP [LIT-datasets-50]; the lab's ECoG and sEEG cohorts (UAB, HUP) as the only mixed-implant set with SOZ labels; simulated ERNA, ECAP and IONM sets because no public data exist [LIT-datasets-44][LIT-domains-53] |
| comparison methods | every entry of §11.3.2 plus the lab's hand-defined time-frequency zones [LIT-tf-3] versus data-driven zones (`evoparam.tf`) |
| claims and evidence | "labels are reliable" → two-rater kappa per set with adjudication; "tasks discriminate methods" → spread of scores across the baseline entries; "ECoG and sEEG need different defaults" → per-implant detection and latency distributions on the mixed cohort [LIT-tf-11]; "data-driven zones reproduce the published zone result and add effect size" → SOZ AUC of zone fractions versus data-driven zone loadings [LIT-tf-3] |
| figures | **F1** benchmark map: datasets × tasks × modalities grid with counts of labelled responses. **F2** labelling protocol and agreement: rater interface screenshot, kappa per dataset, examples of disagreement. **F3** baseline leaderboard: one bar group per task with every comparison method. **F4** ECoG vs sEEG: latency, amplitude, polarity-agreement and detection-rate distributions by implant type and montage. **F5** time-frequency zones: the 11 hand-defined zones [LIT-tf-4] next to data-driven zones on the same cohort, with SOZ AUC per zone. **F6** simulated-set fidelity: real versus simulated response statistics (latency, amplitude, jitter, noise spectrum) per modality. |
| authorship | project author first; the lab member who leads labelling second; clinicians who contributed the UAB and HUP data; collaborators who contributed labels on OpenNeuro sets (e.g. the ER-detect annotation set [LIT-ccep-8] if shared); Smith senior |
| code that must exist | `evoparam.benchmark` with versioned tasks, `evoparam.tf` with zone discovery, `evoparam.spatial`, the labelling tool (a minimal viewer writing BIDS-derivative annotations), leaderboard automation, Zenodo deposit of simulated sets and labels |

### 11.4 Sequence of public steps relative to the HIIE disclosure

Nothing is public before the HIIE decision: no public repository, no PyPI upload, no preprint, no abstract, no talk, no poster, no conversation with outside groups about the specific method. HIIE's own pages count "publication, presentation, abstract or conversation" as disclosure [LIT-naming-25]. The United States allows a one-year grace period for the inventor's own disclosure; almost every other jurisdiction applies absolute novelty, so a disclosure before filing forfeits foreign rights [LIT-naming-26][LIT-naming-30][LIT-naming-31]. The sequence is therefore a dependency chain, not a calendar:

```
M0  invention disclosure filed (provenance map, contributors, funding, data, proposed licence, trademark request)  — §13.5
      │ requires: private repository with SPDX headers and NOTICE (REQ-396); the candidate-novelty list of §13.4
M1  HIIE written decision: (a) patent or not (if yes: provisional on file), (b) licence approved (Apache-2.0 or BSD-3/Apache mix), (c) trademark owner
      │ gate for every public step (REQ-398)
M2  first public release 0.x: GitHub (evoparam-tools organisation), PyPI, Zenodo DOI — same day; CITATION.cff; docs site live
      │ requires: §7 validation suite green; four presets; methods_text()
M3  tutorials and docs complete (≥ 9 tutorials, FAQ, reporting page); conda-forge feedstock
M4  software paper submitted (JOSS) + preprint
      │ requires: M3; test coverage at the REQ-298 thresholds
M5  conference abstracts (SfN, OHBM, AES, IFCN/ICCN) — after M2, never before
M6  1.0 feature-complete (hierarchical, benchmark, reports) → methods paper preprint
      │ requires: benchmark leaderboard public (REQ-392); figures regenerated from a tag
M7  methods paper journal submission
M8  benchmark paper (after labels and the lab cohorts are cleared by IRB for release)
```

If HIIE files a provisional at M1, M2 must follow within the twelve months of the provisional's life (so that the non-provisional can claim it), and foreign filing decisions are taken before M2 because M2 is an absolute-novelty disclosure. If HIIE decides not to patent, the M2 disclosure becomes prior art against later claims on the same methods by others (novelty rule [LIT-naming-30]; this specification's choice of strategy).

### 11.5 Strategy towards the Voytek lab

Facts. The inherited code is a fork of ERPparam, itself a fork of the spectral ancestor (Apache-2.0) [LIT-fooof-58]. ERPparam has no PyPI release, no documentation site, no preprint, 4 stars, a `main` branch last committed 2025-10-06, and open pull requests dated 2026-01-13, 2026-01-14 and 2026-07-30 [LIT-fooof-39][LIT-fooof-59][LIT-fooof-60][LIT-fooof-61][LIT-fooof-64]. Its only dissemination is a 2023 SfN abstract with Allen Institute co-authors [LIT-fooof-69]. Its `CITATION.cff` still points to the 2020 spectral paper [LIT-fooof-73].

| option | gain | cost |
|---|---|---|
| independent, acknowledgement only (NOTICE + paper) | full control of design, licence, timing; no disclosure risk | no access to their simulation experience or the Allen data; possible perception of competition |
| co-authorship on the software paper | goodwill; their user base | JOSS authorship implies code contribution; blurs UAB ownership and the HIIE process |
| co-authorship on the core methods paper | prestige | dilutes the differentiation that this project needs; conflicts with the IP decision at M1 |
| joint benchmark comparison (ERPparam entry on the leaderboard, co-authored benchmark section) | the comparison becomes collaborative, not adversarial; their entry is fair because they run it | none beyond coordination; requires M2 to exist first |

Recommended option: independent until M2, then offer the joint benchmark. Reasons: differentiation must be visible before contact (background model, existence test, trial hierarchy, artifact handling, multi-modality presets, I/O — none of which ERPparam has [LIT-fooof-74][LIT-fooof-57]); contact before the HIIE decision would be a disclosure; and the benchmark is the one place where co-authorship costs nothing in ownership and gains credibility. Contact is made by the project author with the PI copied, after M2, with the 0.x release, the docs and the benchmark comparison already public. Never shared before M1: the repository, the model equations beyond what the ancestors already publish, the candidate-novelty list, the benchmark design, this specification.

Handling an ERPparam release. (a) If ERPparam is released or preprinted before M2: do not rename, do not change the model; include ERPparam as a comparison method on every task; cite it; state the lineage in NOTICE and in both papers; the differentiators remain the same. (b) If it is released after M2: invite the authors to submit to the leaderboard and offer the joint benchmark section. (c) If it stays dormant: the Apache-2.0 obligations (NOTICE, modified-file notices [LIT-naming-13]) and a one-sentence acknowledgement in each paper are the whole relationship.

### 11.6 Conferences, community and documentation

Conferences after M2 only: SfN (poster, Techniques theme), OHBM (software demonstration; MNE-adjacent audience), AES (CCEP and SOZ results with the lab), IFCN/ICCN (clinical neurophysiology; the IONM alarm-statistics work), IEEE NER (engineering audience; real-time module). Community channels: an MNE forum thread and an MNE example gallery entry (the ancestor's only ecosystem integration was an MNE example [LIT-fooof-23]); a BIDS-derivatives schema proposal for evoked-response parameters (§10); registration as an NWB/DANDI analysis tool [LIT-domains-76]; a pyOpenSci review after 1.0; Brainhack and Neuromatch tutorials.

Documentation is a citation driver because readers copy from it. The docs site mirrors the ancestor's layout that users already know: nine ordered tutorials (model, fitting one response, the algorithm, the result objects, background fitting, group fits, tuning and troubleshooting, further analysis, reporting and referencing) [LIT-fooof-21], an examples gallery grouped by modality, a "motivations" gallery, a FAQ and a glossary [LIT-fooof-25]. Every tutorial ends with the generated methods paragraph, so the citation sentence is the last thing read. The docs home title is "evoparam — evoked response parameterization".

### 11.7 Measuring success

Metrics, each with its source: PyPI downloads per month (pypistats), dependent packages and repositories (libraries.io), GitHub stars and forks, citations of the Zenodo concept DOI (DataCite) and of the papers (OpenAlex), benchmark submissions, median time to first response on issues.

Reference points: the ancestor has ~80,000 downloads/month across its two names, 450 stars, 116 forks and 17 dependent packages eight years after first release [LIT-fooof-5][LIT-fooof-7][LIT-fooof-12][LIT-fooof-13]; its paper drew 178 citations in year 1, 425 in year 3 and 600 in year 5 [LIT-fooof-3]; CRP, a method without a package, has 37 citations after three years [LIT-fooof-92]; ERPLAB holds 300/yr [LIT-erptools-9]. The evoked-response literature is large (21,579 ERP articles in one automated meta-analysis [LIT-fooof-103]) but the parameterization use case is new; targets are set at roughly one third of the ancestor's curve, well above the CRP floor.

| metric | 1 year after the methods paper | 3 years | 5 years |
|---|---|---|---|
| cumulative citations (paper + preprint + DOI) | ≥ 50 | ≥ 250 (≥ 100/yr) | ≥ 600 (≥ 180/yr) |
| PyPI downloads / month | ≥ 3,000 | ≥ 10,000 | ≥ 25,000 |
| GitHub stars | ≥ 100 | ≥ 250 | ≥ 500 |
| dependent packages (libraries.io) | ≥ 2 | ≥ 8 | ≥ 20 |
| benchmark submissions by outside groups | ≥ 2 | ≥ 6 | ≥ 15 |
| modalities with an outside-group publication using the package | 2 | 4 | 6 |

Falling below half of a target at any checkpoint triggers the review in §15 (adoption risks).

### 11.8 Requirements

| id | requirement | verification |
|---|---|---|
| REQ-390 | Every result object (`ResponseFit`, `GroupFit`, `TrialFit`, `HierarchicalFit`, `ResponseTest`, `ArtifactResult`, `TFFit`) produces `methods_text()` from its `Provenance` and settings, including package version, versioned DOI, the two references, all resolved settings, goodness of fit, the existence decision and active warnings; the text is stored in the saved result and in BIDS/NWB derivatives by default. | round-trip test: save → load → `methods_text()` byte-identical; snapshot test per preset |
| REQ-391 | The repository root carries `CITATION.cff`; every release is archived on Zenodo with a versioned DOI under one concept DOI; `evoparam.cite()` and `evoparam cite` print both DOIs and the preferred paper citation; the CFF `preferred-citation` is updated at each publication milestone. | CI validates CFF; release checklist item; test that `cite()` contains the version's DOI |
| REQ-392 | The benchmark (§8) is public: tasks, metrics, simulated sets and labels are versioned and archived; the leaderboard accepts outside submissions in a documented format; every leaderboard entry is reproducible with one CLI command from a tagged version. | `evoparam benchmark run --task <t> --method <m>` reproduces the published score within tolerance; submission CI |
| REQ-393 | A preprint precedes every journal submission; the preprint names the exact package tag and benchmark version used; that tag is immutable and archived; figures are regenerated from the tag by one command. | `make figures` on the tag reproduces every figure file hash |
| REQ-394 | The public name is frozen at the first public release; README, docs home, paper titles and abstracts pair the name with "evoked response parameterization"; renamed public symbols keep deprecation shims for two minor versions (REQ-293). | docs build check for the phrase; release checklist |

**Test map.** `tests/test_methods_text.py` (REQ-390), `tests/test_citation.py` (REQ-391), `tests/benchmark/test_leaderboard.py` (REQ-392), `tests/test_figures_from_tag.py` (REQ-393), `tests/test_deprecations.py` (REQ-394).
## 12. Build order

The product is built in the order of its dependencies, not in releases of decreasing ambition: every step below ends with its REQs green in CI, and no step starts before the steps it depends on are green. Steps on the same line of the dependency graph can be built in parallel. The skeleton already in the repository (readers, threshold onset detection, interpolation, epoch fitting with the inherited engine; 199 tests, CI green on Python 3.10–3.13) is step 0.

```
S0 skeleton (done) ─► S1 data model ─► S2 io ─────────────┐
                                  └─► S3 shapes ──────────┤
                                                          ▼
                       S4 events ─► S5 artifact ─► S6 prep ─► S7 fitting engine ─► S8 existence
                                                                     │                  │
                      S9 sim (needs S1, S3, S5; not S7) ─────────────┤                  │
                                                                     ▼                  ▼
                                   S10 trials ─► S11 hierarchical      S12 spatial   S13 tf   S14 stats
                                                                     ▼
                         S15 presets + validation layers 1–3 ─► S16 reports/plots/cli ─► S17 ecosystem (MNE, BIDS, NWB, sklearn)
                                                                     ▼
                                       S18 stream ─► S19 benchmark ─► S20 validation layer 4 + release engineering
```

| step | content | depends on | REQs that must be green | inherited code used | done when |
|---|---|---|---|---|---|
| S0 | skeleton | — | existing tests | all kept modules | done (main @ ef4fad0) |
| S1 | `TrialSet`, `Evoked`, `Provenance`, result dataclasses, error hierarchy, serialisation container | S0 | REQ-020 (round trips), REQ-330…REQ-332, REQ-296 | none (new) | containers round-trip; import-graph test active |
| S2 | readers, remote partial reads, event harmonisation, dataset registry | S1 | REQ-020…REQ-025 | `io/brainvision`, `io/bids`, `io/epochs`, `io/fetch` (kept, wrapped) | ds004080 remote read in CI |
| S3 | shapes and backgrounds with analytic Jacobians | S1 | REQ-100…REQ-102 | `core/funcs` Gaussian and skewed Gaussian (rewritten to the protocol) | all shapes pass mpmath and Jacobian tests |
| S4 | pulse detection, saturation, stimulation-site identification | S2 | REQ-040…REQ-045 | `detect/threshold` (kept, extended) | recall/precision ≥ 0.99 on ds004080 |
| S5 | artifact methods and quality score | S3, S4 | REQ-060…REQ-069 | `clean/interpolate` (kept as method 1) | eleven methods pass |
| S6 | baseline, references (incl. adjusted CAR), rejection flags, averaging | S5 | REQ-080…REQ-085 | none | reference-sensitivity report on ds004080 |
| S7 | fitting engine, model selection, uncertainty (jacobian, profile, bootstrap), classical measures, labels | S3, S6 | REQ-103…REQ-123 (REQ-113/114 need S8 for the comparison rule) | none (`objs/fit` replaced, §4.5) | N1 fixed on the example run (REQ-113) |
| S8 | existence tests, rule registry, detector comparison | S7 | REQ-130…REQ-140 | none | κ vs published rule on 5 subjects |
| S9 | simulators, noise, ground truth, scorers | S1, S3, S5 | REQ-250…REQ-256 | none (`sim/*` rewritten) | — (built in parallel with S4–S8; S7 tests need it) |
| S10 | trial-by-trial fitting, jitter, covariates | S7, S8 | REQ-150…REQ-158 | none | — |
| S11 | hierarchical (EB, then Bayesian extras) | S10, S14 (variance components) | REQ-170…REQ-178 | none | EB within budget; NUTS diagnostics |
| S12 | shared shapes, shared-parameter fits, maps, volume-conduction screen | S7, S8 | REQ-190…REQ-196 | none | ds003708 check |
| S13 | time-frequency | S7 (footprint), S8 | REQ-210…REQ-219 | none | exact decomposition; published-zone layer |
| S14 | statistics and reliability | S7 | REQ-230…REQ-237 | none | ERP CORE reliability check |
| S15 | presets; validation layers 1–3 | S7–S14 | REQ-270…REQ-280, REQ-370…REQ-374 | `bands` → `ComponentWindows` | validation report v1 |
| S16 | reports, plots, CLI | S15 | REQ-333…REQ-338, REQ-340…REQ-346, REQ-349 | `plts` → `plots` (rewritten), `core/reports`/`strings` (rewritten) | methods text frozen for reference pipelines |
| S17 | MNE, BIDS derivatives, NWB, scikit-learn conventions | S16 | REQ-310…REQ-329 | none | bids-validator and pynwb validate in CI |
| S18 | online mode | S10, S17 | REQ-347, REQ-348 and the stream REQs of §10 | none | latency budget met on reference machine |
| S19 | benchmark tasks, suites, adapters, leaderboard generator | S9, S15, S16 | REQ-350…REQ-357 | none | core suite runs in ≤ 30 min |
| S20 | robustness layer, release engineering, documentation complete | all | REQ-290…REQ-309, REQ-375, REQ-376, REQ-390…REQ-399 | — | first public release allowed only after the HIIE gate of §11.4 (M1) |

Rules.
1. The inherited fitting engine stays importable (as `evoparam._legacy`) until S7 is green, so that the skeleton's tests keep passing; it is deleted at the end of S7 with a changelog entry.
2. The N1 example (§1.1) is a test from S7 onwards: no later step may break REQ-113.
3. Public steps (repository, PyPI, preprint, abstracts) are not build steps: they follow the HIIE sequence of §11.4 and are independent of how far the build has progressed.
## 13. Valorisation (commercial pathways)

This section maps the commercial options that the open-source package keeps open, the regulatory line the package must not cross, the intellectual-property questions that only UAB's technology-transfer office (HIIE / UABRF) can settle and how paid modules built on the free engine relate to it. Nothing here is legal advice; every legal statement cites the public page it comes from and is marked UNVERIFIED where the source could not be opened.

### 13.1 Open-core architecture

| layer | free (permissive licence, this repository) | paid (commercial modules, separate repositories) |
|---|---|---|
| engine | every module of §4.2: `io`, `events`, `artifact`, `prep`, `shapes`, `models`, `existence`, `trials`, `hierarchical`, `spatial`, `tf`, `stats`, `sim` | — |
| entry points | presets (§6), CLI (`evoparam.cli`), Python API | graphical interface: viewer, labelling, batch runner, project files |
| I/O | BIDS, NWB, MNE, Neo, BrainVision, EDF, remote partial reads (§10) | vendor-format readers under NDA (IONM and EP platforms), DICOM waveform export |
| online mode | `evoparam.stream`: the API and a reference minimal estimator (fixed shape, amplitude and latency only) so the published closed-loop science is reproducible | real-time integration: device drivers, latency-guaranteed builds, controller logic, safety interlocks, hardware-in-the-loop tests |
| reports | methods text, JSON, BIDS derivatives, basic PDF (`evoparam.reports`) | clinical report templates, normative databases, signed PDF, audit trail, EHR export |
| IONM | the alarm statistic itself (sequential trial-level fit, calibrated probability of a decrement against a drifting baseline) lives in `stats`/`trials`, free | the operational product: live display, alarm management, technologist workflow, validated build |
| benchmark | tasks, labels, metrics, leaderboard (§8) | — (never paid) |
| cloud | — | hosted pipeline, multi-site aggregation, storage, access control |
| services | community support (issues, forum) | support contracts, training, analysis consulting, validated builds with an IEC 62304-style dossier |

Technical boundary. (1) Plugins register through the entry-point groups of §10.8 (`evoparam.shapes`, which also carries backgrounds, `evoparam.presets` and `evoparam.readers`; groups for artifact methods, detectors and report templates follow the same rules when those registries open); the core discovers them and never imports them by name. (2) The public API is frozen at 1.0 under semantic versioning with the deprecation rule of REQ-293; paid modules depend on the public API only. (3) The import-graph test (REQ-296) is extended with a rule that no module of the free package imports anything outside its declared extras, so a paid module can never become a hidden dependency. (4) The brand is protected by trademark, not by copyright: Apache-2.0 §6 grants no trademark rights beyond reasonable attribution, so a registered mark on the name is what controls what forks and products may call themselves [LIT-naming-13]; the policy is published as `TRADEMARK.md` (REQ-397).

Rule: nothing paid may be required for the published science. Every figure in the three articles of §11, every leaderboard entry and every tutorial runs from the free package alone (REQ-392, REQ-393). Paid modules add operation (interfaces, devices, compliance, hosting), never numbers.

### 13.2 Markets

| market | product | customer | size (source) | competitors | differentiator | realistic entry for a PhD-stage project |
|---|---|---|---|---|---|---|
| research: CCEP, ERP, TMS-EEG, animal electrophysiology | free package; support and training; analysis consulting; a research edition with a graphical interface (paid) | labs in the MNE ecosystem (3,536 stars) [LIT-naming-39]; CCEP groups (72 described studies, 3,424 patients) [LIT-ccep-11]; TMS-EEG labs (author estimate 150–300 papers/yr, UNVERIFIED) [LIT-domains-1] | not monetised directly: this is the citation engine of §11 | ERPLAB (GPL, MATLAB) [LIT-naming-45]; MNE `get_peak` [LIT-erptools-17]; FieldTrip, Brainstorm (GPL) [LIT-naming-41][LIT-naming-43]; TESA (MATLAB) [LIT-domains-4]; ER-detect (GPL-3) [LIT-naming-51]; CRP scripts [LIT-fooof-92]; the unreleased fork origin (§1) [LIT-fooof-59] | the only permissively licensed Python package covering event → artifact → response → uncertainty across modalities | now: free release; consulting on archived data; workshops |
| intraoperative neurophysiological monitoring (IONM) | variability-aware alarm statistic (SSEP/MEP amplitude and latency change with calibrated probability) as a review tool, later as an OEM component | IONM service companies and platform vendors: Natus, Cadwell, Nihon Kohden, Inomed, Medtronic NIM, NuVasive, Soterix, SafeOp/Alphatec [LIT-domains-24][LIT-domains-32] | US$3.65 bn (2025) → US$4.93 bn (2031), CAGR 5.14 %; SSEP 36.3 % of revenue (Mordor) [LIT-domains-34]; US$3.75 bn (2025) → US$6.34 bn (2034), CAGR 6.0 %; Medtronic 28 % share (Fortune) [LIT-domains-35] | vendor displays with human judgement; SafeOp EPAD K182542 (automated SSEP) [LIT-domains-26] | ISIN 2019 rejects fixed 50 %/10 % thresholds and asks for changes "clearly exceeding trial-to-trial variability" [LIT-domains-25]; no product computes that; simulation-validated because no public IONM recordings were found | research licence for evaluation on a vendor's archived recordings; consulting; OEM component under the vendor's 510(k) later |
| outpatient clinical EPs (SSEP, VEP, BAEP, MEP) | automated peak measurement with uncertainty and normative comparison | same vendors (Natus Nicolet, Cadwell Sierra, Nihon Kohden Neuropack, Inomed, Neurosoft, Medtronic Keypoint) [LIT-domains-23][LIT-domains-24] | no dedicated report opened; IONM is the tracked segment | vendor cursor placement | same statistic, lower urgency | after IONM; same partner |
| spinal cord stimulation ECAP | open reference implementation: N1–P2 amplitude, latency and conduction velocity, growth-function slope, threshold with confidence interval; preclinical closed-loop research | research groups with exported traces; preclinical labs; research arms of Saluda, Medtronic, Abbott, Boston Scientific, Nevro | closed-loop evidence: 79.1 % vs 53.7 % responders at 24 months in the Evoke trial [LIT-domains-38]; SCS market size not retrieved (UNVERIFIED) | Saluda Evoke firmware (PMA P190002, 2022-02-28) [LIT-domains-42]; Medtronic Inceptiv (FDA 2024-04-26, 50 samples/s) [LIT-domains-43] | open, comparable analyses; manufacturers will not run external code in firmware, but published comparisons against their algorithms are valuable | research licences; collaboration papers |
| DBS evoked resonant neural activity (ERNA) | damped-oscillation component model with trial-level uncertainty; simulation; targeting and programming support | DBS centres (about a dozen groups); research arms of Medtronic, Boston Scientific, Abbott (sensing leads) | 60–100 papers total since 2018, 15–30 per year (author estimate, UNVERIFIED) [LIT-domains-52][LIT-domains-53] | none shared; each group writes its own peak picker or spectral tracker [LIT-domains-53][LIT-domains-54] | first shared toolbox; stated clinical use (targeting, programming) [LIT-domains-54] | free + collaboration; research licence to industry |
| cochlear-implant ECAP | ECAP parameterization for exported research traces | audiology research groups | US$2.80 bn (2024) → US$4.73 bn (2030), CAGR 9.2 % [LIT-domains-65] | Cochlear NRT/AutoNRT, Advanced Bionics NRI, MED-EL ART, with manufacturer patents on threshold determination [LIT-domains-61][LIT-domains-62][LIT-domains-64] | open growth-function and threshold estimation with intervals | research only; low priority |
| TMS-EEG | TEP parameterization consuming TESA, ARTIST or SOUND outputs | TMS-EEG labs; stimulator and amplifier vendors (Magstim, MagVenture, Nexstim, Brainsway; Brain Products, Bittium, ANT Neuro, g.tec) | TMS systems about US$1.9 bn by 2030 (listing only, report not opened) [LIT-domains-13] | TESA (MATLAB, 28 stars) [LIT-domains-4]; no vendor product parameterizes TEPs | component model with existence test per nominal peak and a sensory-contamination flag [LIT-domains-1] | free; vendor partnership for mapping reports later |
| BCI | single-trial P300 and ErrP amplitude and latency posteriors as features and clinical descriptors | BCI research (MOABB, 1.0k stars) [LIT-domains-66]; g.tec, Brain Products, EMOTIV | non-invasive BCI US$397.59 M (2025) → US$773.82 M (2033), CAGR 8.73 %; healthcare 57.11 % [LIT-domains-75] | MOABB and pyRiemann classify rather than parameterize [LIT-domains-67] | trial-level parameters benchmarked on bigP3BCI (304 participants, CC BY 4.0) [LIT-domains-71] | MOABB pipeline entry; free |

Realistic path. The IONM and cochlear markets are large, but they are hardware-and-services markets in which analysis software is an OEM component inside a cleared device; a doctoral project cannot clear a device. The order that fits the project is therefore: (1) free engine and the citation strategy of §11; (2) research licences and consulting (vendor R&D evaluations on archived data; analysis contracts for labs; training) through UABRF; (3) a research edition with a graphical interface as the first paid product, sold to labs, with no clinical claim; (4) clinical modules only with a regulated partner that carries the device burden (§13.3).

### 13.3 Regulatory path

When the software is not a device. A package released for research, distributed with the statement "research use only, not intended for diagnosis, monitoring or treatment", that makes no clinical claim, is not a medical device; intended use is the hinge. FDA's Clinical Decision Support guidance (final, listed as issued January 2026) excludes some software functions from the device definition under section 520(o)(1)(E), but software that analyses physiological signals such as EEG or evoked potentials does not meet the first exclusion criterion and remains a device *if intended for clinical use* [LIT-domains-82].

When it becomes software as a medical device (SaMD). As soon as the intended use is to inform a diagnosis, a monitoring decision or a treatment in a patient:

- United States. Evoked-potential acquisition and analysis systems are class II under 21 CFR 882.1870 (evoked response electrical stimulator; product codes GWF, GWE and related) and clear by 510(k) with predicates: Nihon Kohden Neuromaster G1 MEE200, K142624, decided 2015-04-24 [LIT-domains-23]; Cadwell Guardian, K230415, decided 2023-08-17, predicates K162199 and K212166, covering SSEP, VEP, BAEP, MEP, EEG, ECoG and EMG [LIT-domains-24]; SafeOp EPAD, K182542, the one automated-SSEP alerting product found [LIT-domains-26]. No 510(k) or De Novo for stand-alone EP-analysis software was identified (absence not confirmed) [LIT-domains-82]. Implanted closed-loop stimulators are class III PMAs (Evoke P190002; Inceptiv) and out of scope [LIT-domains-42][LIT-domains-43].
- European Union. MDR Annex VIII Rule 11 classifies software that provides information used for diagnostic or therapeutic decisions as class IIa or higher; MDCG 2019-11 is the guidance (not opened) [LIT-domains-83].
- Process standard. IEC 62304 (medical device software life-cycle processes) with safety classes A, B and C [LIT-domains-84]. The standard was not opened for this specification; the summary below is from public summaries and must be checked against the standard before any regulated build.

| IEC 62304 class | harm if the software fails | what the class requires (in addition to the lower class) |
|---|---|---|
| A | no injury or damage to health possible | documented development plan; software requirements; system testing; release; configuration management; problem resolution; risk management per ISO 14971; a SOUP (software of unknown provenance) list with the version of each third-party item |
| B | non-serious injury possible | architectural design; integration and integration testing; unit verification; SOUP risk evaluation (anomaly lists, hardware/software requirements) |
| C | death or serious injury possible | detailed design; more rigorous unit verification; segregation justification between items of different classes |

For this package the SOUP items are `numpy`, `scipy` and every optional extra; a regulated product must list their versions, their published anomaly sources and the tests that cover the functions used.

Design choices in this specification that keep the path open (none of them turns the package into a device):

| regulated-build need | what the spec already provides |
|---|---|
| deterministic, versioned algorithms | every `Provenance.steps` entry carries the function's own version; `random_state` recorded; identical input and settings reproduce identical numbers (REQ-294) |
| requirements traceability | REQ ids → tests matrix in §14; CI fails on an untested requirement |
| validation dossier | §7 simulations with ground truth and model violations; §8 public benchmarks with fixed versions |
| no hidden retraining | no learned model is updated silently; `artifact_method="learned"` templates are stored, versioned and listed in `Provenance` |
| SOUP handling | REQ-399 inventory; light core (REQ-291) keeps the list short |
| configuration management | immutable git tags; Zenodo archive per release (REQ-391); result schema version (§4.3.5) |
| problem resolution | public issue tracker; changelog entries that name the affected algorithm version; REQ-108 (no silent failure) |

Statement for the README, the paper and every report: the open-source package is not a medical device and makes no clinical claim; any clinical use requires a separate, regulated product with its own quality system and clearance, in which the package may appear as a documented SOUP component.

### 13.4 Intellectual property

What may be protectable (HIIE decides; the inventor does not pre-filter [LIT-naming-25][LIT-naming-26]):

- Trademark on the name. `evoparam` is free on PyPI, conda-forge, `.com` and `.org`; the GitHub user handle is taken (organisation `evoparam-tools` instead) [LIT-naming-1][LIT-naming-7][LIT-naming-8][LIT-naming-55]. No USPTO or EUIPO search could be run; it must be done before any announcement (UNVERIFIED) [LIT-naming-59]. Near names are already in clinical and scientific use (evoShape skull implant, EvoFIT) [LIT-naming-10][LIT-naming-11].
- Possible method claims. Candidate elements that are not in the published ancestors or in the GPL CCEP tools; listed for the disclosure, not pre-judged: (1) a joint artifact-plus-response fit in which a pulse-shaped artifact basis with exponential tails and the response components are estimated in one model with identifiability constraints (artifact amplitude scales with current; response does not invert with polarity) — §5.3 `tail` with `tail="joint"` and the §5.5.2 background `artifact_basis`; (2) a hierarchical trial → channel → subject model that returns a per-trial existence posterior from the single-trial weight distribution — §5 `hierarchical`, `existence`; (3) data-driven time-frequency zones with per-trial loadings and signed metrics — §5 `tf`; (4) dose-response normalisation of amplitudes across stimulation protocols (charge-normalised, per-channel dose-response curve) — §5 `stats`; (5) a variability-adaptive sequential alarm statistic for IONM — `stats`; (6) mask-aware fitting with uncertainty inflation inside the blanked window — `models`. UAB's FAQ states that "some software" can be patented [LIT-naming-27]; post-Alice eligibility for signal-processing methods depends on claiming a concrete technical improvement (general knowledge, UNVERIFIED).
- Copyright in the new code, owned by UAB under its policy (below).

What is not protectable: re-implementations of published methods — CRP [LIT-fooof-92], the ER-detect rule [LIT-datasets-62], ERPLAB's measurement definitions [LIT-erptools-1], CARLA [LIT-ccep-18], the PARRM equations [LIT-artifact-1], matching-pursuit removal [LIT-artifact-5]. Copyright protects code, not methods; the reference MATLAB codes of PARRM and MPARRM carry no licence and so cannot be ported, which is why every such method is written clean-room from the paper and documented as such (REQ-396) [LIT-naming-36][LIT-naming-37][LIT-erptools-87]. Patents may still cover some of them (next paragraph).

Apache-2.0 §3 versus BSD-3-Clause. Apache-2.0 §3 makes every contributor, a university included, grant a perpetual royalty-free patent licence on claims "necessarily infringed" by its contribution; releasing an implementation of a patented method under Apache-2.0 therefore licenses that patent, for that implementation, to the world. BSD-3-Clause and MIT contain no express patent grant, so a university can release code permissively and keep method claims for other embodiments, at the cost of an implied-licence argument and some user distrust [LIT-naming-13]. Consequences for this repository: the inherited files stay Apache-2.0 whatever happens (§4 obligations: licence copy, modified-file notices, retained notices, NOTICE file) [LIT-naming-13]; the new modules go either Apache-2.0 (simplest; same as the ancestor and PyMC [LIT-naming-38][LIT-naming-49]) or BSD-3-Clause, declared as the SPDX expression `BSD-3-Clause AND Apache-2.0` under PEP 639 [LIT-naming-54], which also matches MNE-Python for any later upstreaming [LIT-naming-39][LIT-naming-56]. This is question 3 for HIIE and must be answered before the first public commit.

Freedom to operate. Two pre-grant publications matter:

| application | applicant, inventors | dates | first independent claim (as read) | relevance |
|---|---|---|---|---|
| US 2022/0338786 A1, application 17/722,903 (PARRM) [LIT-naming-33][LIT-artifact-4] | Brown University; Borton, Dastin-van Rijn, Harrison, Provenza, Chen, Kim, Darbon | priority 2021-04-16; filed 2022-04-18; published 2022-10-27; status "pending" when read; grant status after publication UNVERIFIED | a system that receives real-time DBS waveform data, determines the stimulation period relative to the sampling rate, applies kernel regression and linear filtering to identify the repeating artifact, subtracts it and outputs filtered data | the `period_template` artifact method (§5) for DBS-style trains may read on issued claims; single-pulse SPES/CCEP/TEP handling is outside the "stimulation period" framing, but that is a question for counsel |
| US 2023/0240618 A1, application 18/163,257 (matching-pursuit removal) [LIT-artifact-25] | Washington University; Brunner, Xie, Willie, Leuthardt | provisional 2022-02-01; filed 2023-02-01; published 2023-08-03; grant status UNVERIFIED | computing systems and methods for removing brain stimulation artifacts in neural signals by matching pursuit (claim text not read in full) | the joint artifact-plus-response fit must be designed and documented as a parametric least-squares model, not a dictionary matching pursuit, and reviewed for freedom to operate before release |

Actions: a freedom-to-operate opinion before the `period_template` and `matching_pursuit` artifact methods (§5.3) ship (questions 7 and 21); clean-room implementation from the open-access papers; a note on both applications in the documentation of the artifact module. The MIT-licensed PyPARRM port exists [LIT-artifact-3]; its existence settles the copyright question for its own code, not the patent question.

Grace period and absolute novelty. 35 U.S.C. 102(b)(1)(A): a disclosure by the inventor within one year before the US filing date is not prior art against the inventor's own application (statute not fetched; corroborated by UAB's guide) [LIT-naming-30][LIT-naming-26]. UAB's Inventors' Guide: "in almost all other countries, an inventor will lose any patent rights if the invention is publicly disclosed prior to filing" [LIT-naming-26]; the European Patent Convention applies absolute novelty (Art. 54) with exceptions (Art. 55) that do not cover an ordinary publication (not fetched) [LIT-naming-31]. Hence the gate of §11.4: nothing public before the HIIE decision.

Ownership, revenue and roles at UAB [LIT-naming-26][LIT-naming-27]. Under the UAB Patent Policy an invention that results from research under a UAB employee's direction, is related to the employee's field of work, or is developed with UAB resources "is the property of the University of Alabama at Birmingham, not of the individual inventor"; faculty, employees and students agree to disclose "as a condition of employment or enrollment"; the scope explicitly includes software [LIT-naming-25]. Revenue sharing, net of a 15 % UABRF administrative share and 100 % of legal expenses: inventors 75 % of the first $20,000, 50 % from $20,000 to $100,000, 30 % from $100,000 to $2 M, 30 % above $2 M; the remainder goes to department, school and UABRF (6.25 / 3.125 / 15.625 % in the first tier; 24.5 / 12.25 / 33.25 % in the top tier) [LIT-naming-27]. Startups: full-time faculty may be founder, chief scientific or technical officer, board member or consultant, but not president, CEO, CFO or COO [LIT-naming-26][LIT-naming-27]; the rule for a graduate student is not on the public pages (UNVERIFIED). The Faculty Handbook Patent Policy and the UAB Copyright Policy could not be retrieved [LIT-naming-28]. Federally funded work adds Bayh-Dole reporting through the institution (37 CFR 401; not fetched) [LIT-naming-32]. A UAB laboratory already publishes CCEP detection code without a licence (`UAB-NSPM-Lab/CCEP_N1_N2_Detection`) [LIT-naming-52]; folding it into this package under the approved licence gives HIIE a precedent and the lab a citable artefact.

### 13.5 Questions for HIIE

The fifteen questions of the licensing research, tightened, followed by six specific to this specification.

1. Does UAB claim ownership of this software under the Patent Policy, the Copyright Policy, or both, and who signs the open-source licence on UAB's behalf (UABRF or the department)?
2. Is there a standing approval path for releasing UAB-owned software under Apache-2.0 or BSD-3-Clause, and how long does it take?
3. Given that part of the code derives from Apache-2.0 files, does UABRF prefer Apache-2.0 for the whole project or a BSD-3-Clause/Apache-2.0 mix, and does it accept the Apache §3 patent grant on UAB-owned contributions?
4. Does UABRF want to evaluate patentability of any element in §13.4 before the first public release, and what is the earliest release date compatible with a provisional filing?
5. If a provisional is filed, which jurisdictions matter (US only versus PCT/EP), given absolute novelty abroad?
6. Will UABRF register a trademark on the chosen name, who owns it, and under what policy may forks and companies use it?
7. Does the pending Brown University application US 2022/0338786 A1 (PARRM) affect a Python re-implementation of period-based artifact removal, and should a freedom-to-operate opinion be obtained before adding it?
8. Are there sponsor obligations (NIH, Bayh-Dole, foundation grants) that require agency disclosure, government-use licences or specific sharing terms?
9. How are outside contributors handled: is a contributor licence agreement required, or is the Developer Certificate of Origin enough for UABRF?
10. Do student contributors own their copyright in contributions, and how is that documented?
11. If clinical users or companies request support, can the lab offer paid services or a commercial plugin licence, and would that go through UABRF licensing?
12. Does the conflict-of-interest policy restrict roles if a startup is formed around the tool (founder/CSO permitted; CEO not), and what approvals are needed?
13. Which licence terms would UABRF accept in a dual-licensing or open-core scheme, and what revenue split applies to software licence income versus patent royalties?
14. Are there approved templates for an inter-institutional agreement if co-authors at other universities contribute code?
15. Does UAB require a specific copyright line in source files (for example "Copyright (c) 2026 The UAB Research Foundation")?
16. Does UAB claim ownership of paid modules the student would develop on top of the free engine? Which conditions (own equipment, outside working hours, no UAB data, no UAB funds) keep them outside the policies, and can a written carve-out, option or licence be set up now rather than after the fact?
17. What rules apply to a graduate student, as opposed to full-time faculty, who wishes to hold an officer role in a startup built on UAB-owned software?
18. When UAB clinicians provide labelled recordings for the public benchmark (§8), who owns the labels, and under which licence (CC0, CC BY) may labels and derived simulated sets be published?
19. Does HIIE accept a staged process: one disclosure now for the engine with a permissive release, and later separate disclosures for the paid modules as they reach novelty?
20. Which data-protection and export-control reviews apply to a hosted pipeline that processes clinical recordings (HIPAA; GDPR for European users)?
21. Is a freedom-to-operate opinion on the Washington University application US 2023/0240618 A1 (matching-pursuit artifact removal) also needed before the joint artifact-plus-response module ships?

### 13.6 Positioning of paid modules

Any paid module is a product; the free package is its engine. The relationship is designed so that a product never conflicts with the engine's licence or with the science:

- Dependency, not fork. A paid module pins a released version of the engine and never carries a private copy; fixes go upstream first. A permissive licence allows proprietary linking: Apache-2.0 §4 requires only the licence copy and retained notices, BSD-3-Clause only notice retention; neither requires the product's source [LIT-naming-13]. A GPL-free core (REQ-395) is what makes this possible; a single GPL import would reach the product.
- Naming. A paid product carries its own name and is described as "built on evoparam" under the trademark policy (REQ-397); it is never called evoparam and never implies that the engine is paid.
- Boundary. Paid features are plugins and applications on the public API and entry points (§13.1); the engine's import-graph test proves that nothing in the free package depends on them.
- Science stays free. Every published number comes from the engine; paid modules add viewing, labelling, batch operation, device integration, compliance and hosting. A reviewer can reproduce any paper figure without the product.
- Validated builds. The product may ship "validated builds" of the engine (a frozen version with the SOUP list, the traceability matrix and the validation dossier of §13.3), which is a service on top of the free code, not a different code.
- Ownership. Whether UAB owns such a product, and under which licence the student may commercialise it, is question 16; revenue from a UAB-owned product flows through UABRF under the shares of §13.4. Pricing and sales are outside this specification.

### 13.7 Requirements

| id | requirement | verification |
|---|---|---|
| REQ-395 | No module of the free package imports, vendors or declares as an extra any package under GPL, LGPL or AGPL; GPL tools (FieldTrip, Brainstorm, ERPLAB, ER-detect, CCEP_Toolbox) are reached only at arm's length (subprocess or file exchange) from separately licensed adapter packages, never from this repository [LIT-naming-14][LIT-naming-57]. | CI licence scan of the resolved dependency tree for every extras set; import-graph test extended with a licence allow-list; PR checklist item |
| REQ-396 | A provenance map is maintained and shipped: SPDX licence header in every source file; inherited files marked "Modified from <upstream project and version> (Apache-2.0); changes: …" (the upstream projects are named in `NOTICE`, §11.5); a `NOTICE` file; `THIRD_PARTY.md` listing every third-party package with version and licence; `REIMPLEMENTED.md` listing every algorithm written from a paper, with the paper, the statement that no code was copied, and any known patent publication. | CI fails on a file without SPDX header or on an inherited file without the modification notice; release checklist |
| REQ-397 | A `TRADEMARK.md` states who owns the name (per HIIE decision), what forks may be called, how plugins and products may reference the name ("for evoparam", "built on evoparam") and that the licence grants no trademark rights (Apache-2.0 §6). | file present and linked from README; reviewed at each major release |
| REQ-398 | The invention disclosure is filed and HIIE's written decision (patent or not; licence; trademark owner) is on record before the first public commit, PyPI upload, preprint, abstract, talk or external demonstration; until then the repository is private and the release workflow refuses to publish without the recorded decision reference. | release workflow checks for `GOVERNANCE.md` containing the decision reference and date; §11.4 milestone M1 |
| REQ-399 | A machine-readable SOUP inventory (`soup.json`: package, version range, licence, anomaly source URL, functions used) is generated at each release; every algorithm carries its own version string recorded in `Provenance.steps`, bumped whenever a change alters numerical output; the REQ → test traceability matrix of §14 is regenerated at each release and has no untested requirement. | CI generates and diffs `soup.json`; numerical regression tests fail when outputs change without an algorithm version bump; traceability report in the release artefacts |

**Test map.** `tests/test_dependencies.py` (REQ-395), `tests/test_spdx_headers.py` (REQ-396), `tests/test_trademark_file.py` (REQ-397), `tests/test_release_gate.py` (REQ-398), `tests/test_soup_inventory.py` (REQ-399).
## 14. Traceability matrix

This section links every gap of `GAPS.md` to the requirements that address it, the module that implements them, the test files that check them and the validation layer of §7 that measures them (REQ-018). It is generated from the section files and `GAPS.md`: a requirement's module is the section that defines it, its tests are the test files named next to it in that section, and its validation layer follows §7 (§7.1 mathematical checks, §7.2 simulations with ground truth, §7.3 real data and published methods, §9.5 continuous integration for engineering and process requirements). A gap with status "Addressed" must have at least one requirement and one test file; the check passes for all 58 gaps. Gaps marked "Not addressed" are excluded on purpose, with reasons, in `GAPS.md` ("Gaps deliberately not addressed") and in §2.3.

### 14.1 Gap → requirement → module → test → validation

| gap | gap (short) | status | requirements | module(s) | tests | validation |
|---|---|---|---|---|---|---|
| GAP-01 | No released, documented, citable package parameterizes evoked waveforms | Addressed | REQ-001, REQ-002, REQ-308, REQ-391, REQ-393, REQ-394 | engineering, product, publication | `run_benchmarks.py`, `test_citation.py`, `test_deprecations.py`, `test_figures_from_tag.py`, `test_one_model.py` | §9.5 CI |
| GAP-02 | No time-domain model of components over a slow background; tools report geometric features of the average o… | Addressed | REQ-100, REQ-101, REQ-103 | shapes/models | `test_engine_sim.py`, `test_shapes.py`, `test_varpro.py` | §7.1 math |
| GAP-03 | The number of components is set by thresholds and deletion heuristics, not by model selection; no tool repo… | Addressed | REQ-004, REQ-104, REQ-107, REQ-111, REQ-112, REQ-121, REQ-122 | product, shapes/models | `test_engine_sim.py`, `test_no_deletion.py` | §7.2 simulations; §9.5 CI |
| GAP-04 | No parameter uncertainty on fitted or scored values | Addressed | REQ-006, REQ-116, REQ-230 | product, shapes/models, stats | `test_engine_sim.py`, `test_stats_bootstrap.py`, `test_uncertainty_columns.py` | §7.2 simulations; §9.5 CI |
| GAP-05 | Artifact tail, background and components are removed or fitted sequentially, so early responses are distorted | Addressed | REQ-061, REQ-100 | artifact, shapes/models | `test_learned.py`, `test_template.py`, `test_engine_sim.py`, `test_shapes.py` | §7.1 math; §7.2 simulations |
| GAP-06 | ERNA and other oscillatory evoked responses are quantified by RMS windows or spectral peaks, not by a dampe… | Addressed | REQ-100, REQ-276, REQ-279 | presets, shapes/models | `test_engine_sim.py`, `test_presets_sim.py`, `test_shapes.py` | §7.1 math; §7.3 presets |
| GAP-07 | No evoked-response fitter uses robust losses | Partly addressed | §5.5.3 | — | — | — |
| GAP-08 | Trial-level fitting at scale is slow; GPU least-squares stacks in Python are unmaintained or immature | Addressed | REQ-115, REQ-290, REQ-292 | engineering, shapes/models | `bench_engine.py`, `run_benchmarks.py`, `test_dependencies.py` | §7.2 simulations; §9.5 CI |
| GAP-09 | No permissively licensed Python implementation of the ERPLAB measurement vocabulary (local peak, four area … | Addressed | REQ-005, REQ-110, REQ-117, REQ-231 | product, shapes/models, stats | `test_classical.py`, `test_engine_sim.py`, `test_stats_jackknife.py`, `test_status.py` | §7.2 simulations; §7.3 real data; §9.5 CI |
| GAP-10 | Reliability and data-quality metrics (SME, dependability, trials needed) are absent from Python or tied to … | Addressed | REQ-117, REQ-230, REQ-233, REQ-234 | shapes/models, stats | `test_classical.py`, `test_stats_bootstrap.py`, `test_stats_reliability.py` | §7.2 simulations; §7.3 real data |
| GAP-11 | Single-trial amplitude and latency, latency jitter, jitter-corrected amplitude and latency distributions ar… | Addressed | REQ-150, REQ-151, REQ-152, REQ-153, REQ-157 | trials | `test_trials_sim.py`, `test_trials_table.py` | §7.2 simulations |
| GAP-12 | Overlap deconvolution and trial-specific latency are handled by disjoint tool families | Not addressed | none | — | — | — |
| GAP-13 | No agreed response-existence test; studies use undeclared or ad hoc thresholds, and train-to-train reproduc… | Addressed | REQ-130, REQ-131, REQ-132, REQ-133, REQ-134, REQ-135, REQ-137, REQ-139 | existence | `test_existence_ccep.py`, `test_existence_null.py`, `test_existence_power.py`, `test_projection.py`, `test_rules_registry.py` | §7.2 simulations; §7.3 real data |
| GAP-14 | Delayed responses (100 ms–1 s), the oldest SPES biomarker, have no automated single-trial detector | Partly addressed | REQ-216 | tf | `test_tf_ccep.py` | §7.3 real data |
| GAP-15 | No modality-agnostic, permissively licensed Python library for stimulation artifacts | Addressed | REQ-011, REQ-060, REQ-061, REQ-062, REQ-063, REQ-064, REQ-065, REQ-068 | artifact, product | `bench_artifact.py`, `test_learned.py`, `test_lowrank.py`, `test_masks.py`, `test_period.py`, `test_polynomial.py`, `test_subspace.py`, `test_template.py`, `test_dependencies.py` | §7.2 simulations; §9.5 CI |
| GAP-16 | Blanked or interpolated windows are treated as data, although the earliest features live there | Addressed | REQ-060, REQ-108, REQ-110 | artifact, shapes/models | `test_masks.py`, `test_engine_sim.py`, `test_varpro.py` | §7.2 simulations |
| GAP-17 | Alternating-polarity and forward-masking cancellation exist only in cochlear and spinal ECAP practice | Addressed | REQ-067 | artifact | `test_paradigm.py` | §7.2 simulations |
| GAP-18 | Modality-specific artifacts (implanted-device sampling, aliasing and ECG coupling; photoelectric artifacts … | Partly addressed | REQ-061, REQ-062 | artifact | `test_learned.py`, `test_period.py`, `test_template.py` | §7.2 simulations |
| GAP-19 | Amplifier saturation is handled inconsistently and is not propagated to later stages | Addressed | REQ-043, REQ-065, REQ-066 | artifact, events | `test_polynomial.py`, `test_quality.py`, `test_saturation.py` | §7.2 simulations |
| GAP-20 | Artifact-removal methods have no shared ground-truth benchmark | Addressed | REQ-253, REQ-350, REQ-351 | benchmark, sim | `test_schemas.py`, `test_suite_core.py`, `test_sim_artifact.py` | §7.2 simulations |
| GAP-21 | Re-referencing is not treated as part of response detection | Addressed | REQ-081, REQ-082, REQ-084, REQ-372 | prep, validation | `test_reference.py`, `test_reference_sensitivity.py`, `test_carla_ds004977.py` | §7.2 simulations; §7.3 real data |
| GAP-22 | Polarity and volume conduction: sEEG responses are sign-ambiguous and detectors look for negative peaks only | Addressed | REQ-003, REQ-193 | product, spatial | `test_polarity.py`, `test_spatial_sim.py` | §7.2 simulations; §9.5 CI |
| GAP-23 | Channels are fitted independently; no shared shapes across channels or sites with per-channel timing | Addressed | REQ-190, REQ-191, REQ-192, REQ-194 | spatial | `test_spatial_bpc.py`, `test_spatial_sim.py` | §7.2 simulations; §7.3 real data |
| GAP-24 | No hierarchical trial → channel → subject model for parametric components | Addressed | REQ-170, REQ-171, REQ-172, REQ-173, REQ-175, REQ-177 | hierarchical | `test_hier_bayes.py`, `test_hier_eb.py` | §7.2 simulations |
| GAP-25 | Stimulation-parameter (dose) dependence and position within a train are not modelled; BIDS has no controlle… | Addressed | REQ-022, REQ-044, REQ-154, REQ-158 | events, io, trials | `test_trains.py`, `test_events_harmonize.py`, `test_trials_ccep.py`, `test_trials_sim.py` | §7.2 simulations; §7.3 real data; §9.5 CI |
| GAP-26 | Brain state (pre-stimulus phase and power, sleep stage, anaesthetic, arousal) is not a covariate in evoked-… | Addressed | REQ-155, REQ-175 | hierarchical, trials | `test_hier_eb.py`, `test_trials_sim.py` | §7.2 simulations |
| GAP-27 | Phase-estimation uncertainty (about 45° in real time) is not propagated into phase–response analyses | Not addressed | none | — | — | — |
| GAP-28 | Time-frequency response metrics are computed on total power, confounding the evoked footprint with induced … | Addressed | REQ-211, REQ-212 | tf | `test_tf_sim.py`, `test_tf_transform.py` | §7.1 math; §7.2 simulations |
| GAP-29 | Hand-defined time-frequency zones are fixed across patients and sampling rates, and suppression is not mode… | Addressed | REQ-213, REQ-216, REQ-217 | tf | `test_tf_ccep.py`, `test_tf_sim.py`, `test_tf_zones_registry.py` | §7.2 simulations; §7.3 real data |
| GAP-30 | Time-frequency baselines ignore stimulation timing (single-trial division is skewed), and pixel-wise FDR is… | Addressed | REQ-080, REQ-214, REQ-215 | prep, tf | `test_baseline.py`, `test_tf_sim.py` | §7.2 simulations |
| GAP-31 | TMS-EEG evoked potentials are scored by window peak picking on pipelines that disagree | Addressed | REQ-275, REQ-326 | ecosystem, presets | `test_docs.py`, `test_plugins.py`, `test_repo_files.py`, `test_tep_ds002094.py` | §7.3 real data; §9.5 CI |
| GAP-32 | Spinal-cord and cochlear ECAP analysis is locked in proprietary firmware | Addressed (simulation-validated) | REQ-067, REQ-279 | artifact, presets | `test_paradigm.py`, `test_presets_sim.py` | §7.2 simulations; §7.3 presets |
| GAP-33 | BCI toolkits classify single-trial ERPs but do not parameterize them | Addressed | REQ-235, REQ-320 | ecosystem, stats | `test_sklearn_api.py`, `test_sklearn_checks.py`, `test_stats_export.py` | §7.2 simulations; §9.5 CI |
| GAP-34 | Animal evoked responses lack NWB-native fitting tools and a cross-species shape model | Addressed | REQ-278, REQ-317, REQ-318, REQ-319 | ecosystem, presets | `test_neo_io.py`, `test_nwb_io.py`, `test_neuropixels_lfp.py` | §7.3 real data; §9.5 CI |
| GAP-35 | No open, validated estimator for real-time evoked-parameter updates; intraoperative monitoring alarms use f… | Partly addressed | REQ-015, REQ-324, REQ-325, REQ-347, REQ-348 | ecosystem, product, reports/plots/cli/stream | `stream_latency.py`, `test_docs.py`, `test_research_use.py`, `test_stream.py`, `test_stream_lsl.py` | §9.5 CI |
| GAP-36 | Regulatory readiness is not designed into research evoked-potential software | Partly addressed | REQ-013, REQ-015, REQ-399 | legal/valorisation, product | `test_determinism.py`, `test_research_use.py`, `test_soup_inventory.py` | §9.5 CI |
| GAP-37 | Stimulation-event metadata are inconsistent across deposits and underspecified in BIDS; MNE-BIDS cannot rou… | Addressed | REQ-022, REQ-313, REQ-314, REQ-316 | ecosystem, io | `test_events_harmonize.py`, `test_bids_derivatives.py`, `test_bids_events.py`, `test_bids_validator.py` | §9.5 CI |
| GAP-38 | Dataset discovery is brittle, no machine-readable registry of evoked-response datasets exists, and datasets… | Addressed | REQ-021, REQ-023 | io | `test_bids_remote.py`, `test_registry.py`, `test_engine_ccep.py` | §9.5 CI |
| GAP-39 | No public data for DBS ERNA, spinal ECAP or intraoperative monitoring benchmarks | Partly addressed | REQ-250, REQ-279, REQ-355 | benchmark, presets, sim | `test_deposit.py`, `test_presets_sim.py`, `test_sim_determinism.py` | §7.1 math; §7.2 simulations; §7.3 presets |
| GAP-40 | No community benchmark with ground truth for detection, decomposition, single-trial and scoring tasks | Addressed | REQ-350, REQ-351, REQ-352, REQ-353, REQ-355, REQ-356, REQ-357, REQ-392 | benchmark, publication | `test_deposit.py`, `test_leaderboard.py`, `test_schemas.py`, `test_scoring.py`, `test_suite_core.py` | §7.2 simulations; §9.5 CI |
| GAP-41 | Simulators exist as private figures, not as reusable products | Addressed | REQ-250, REQ-251, REQ-252, REQ-255, REQ-256 | sim | `test_sim_determinism.py`, `test_sim_noise.py`, `test_sim_truth.py` | §7.1 math; §7.2 simulations |
| GAP-42 | Reporting is incomplete: settings and fit quality are rarely stated, and no EEG/MEG tool writes a methods p… | Addressed | REQ-008, REQ-123, REQ-140, REQ-219, REQ-338, REQ-340, REQ-390 | engineering, existence, product, publication, reports/plots/cli/stream, shapes/models, tf | `test_existence_text.py`, `test_methods_text.py`, `test_reports.py`, `test_reports_text.py`, `test_tf_text.py` | §7.2 simulations; §9.5 CI |
| GAP-43 | Provenance is not emitted per output by electrophysiology tools; no artifact tool records which samples wer… | Addressed | REQ-007, REQ-069, REQ-294, REQ-330, REQ-331, REQ-333, REQ-335, REQ-336, REQ-337 | artifact, engineering, product | `test_provenance.py`, `test_bids_derivatives.py`, `test_nwb_io.py`, `test_provenance.py`, `test_reproduce.py`, `test_serialize.py`, `test_settings_log.py`, `test_tidy.py` | §7.2 simulations; §9.5 CI |
| GAP-44 | No tidy, complete output schema for evoked measurements | Addressed | REQ-120, REQ-235, REQ-332, REQ-341 | engineering, reports/plots/cli/stream, shapes/models, stats | `test_reports_json.py`, `test_reproducibility.py`, `test_serialize.py`, `test_stats_export.py`, `test_tidy.py` | §7.2 simulations; §9.5 CI |
| GAP-45 | No standard place for fitted evoked parameters in BIDS or NWB | Addressed | REQ-024, REQ-315, REQ-316, REQ-318, REQ-342 | ecosystem, io, reports/plots/cli/stream | `test_writers.py`, `test_bids_derivatives.py`, `test_neo_io.py`, `test_nwb_io.py` | §9.5 CI |
| GAP-46 | Parameterization tools are not integrated with MNE objects | Addressed | REQ-002, REQ-310, REQ-311, REQ-312 | ecosystem, product | `test_mne_io.py`, `test_one_model.py`, `check_no_inherited_code.py` | §9.5 CI |
| GAP-47 | No scikit-learn-compatible estimator for evoked parameterization | Addressed | REQ-320, REQ-321 | ecosystem | `test_sklearn_api.py`, `test_sklearn_checks.py` | §9.5 CI |
| GAP-48 | Fits fail silently or return empty results indistinguishable from "no response" | Addressed | REQ-005, REQ-118, REQ-237, REQ-307 | engineering, product, shapes/models, stats | `test_errors.py`, `test_stats_export.py`, `test_status.py` | §7.2 simulations; §9.5 CI |
| GAP-49 | Units, scales and defaults confuse users; defaults lack provenance | Addressed | REQ-009, REQ-014, REQ-270, REQ-271, REQ-272 | presets, product | `test_presets.py`, `test_units.py` | §7.3 presets; §9.5 CI |
| GAP-50 | Inherited stop rule compares the maximum of about 1,000 noise samples with 2–3 SD, so pure noise yields pea… | Addressed | REQ-004, REQ-017, REQ-107, REQ-112, REQ-113, REQ-114 | product, shapes/models | `test_engine_ccep.py`, `test_engine_sim.py`, `test_no_deletion.py` | §7.2 simulations; §7.3 real data; §9.5 CI |
| GAP-51 | Edge rule and half-magnitude deletion remove a genuine early N1 (onset during the artifact window, or on th… | Addressed | REQ-004, REQ-017, REQ-110, REQ-113 | product, shapes/models | `test_engine_ccep.py`, `test_engine_sim.py`, `test_no_deletion.py` | §7.2 simulations; §7.3 real data; §9.5 CI |
| GAP-52 | Inherited code cannot host negative peaks under a DC offset, follows the raw value for sign, and reports tw… | Addressed | REQ-003, REQ-006, REQ-109, REQ-118 | product, shapes/models | `test_polarity.py`, `test_status.py`, `test_uncertainty_columns.py`, `test_varpro.py` | §7.1 math; §7.2 simulations; §9.5 CI |
| GAP-53 | Broken inherited functions (`average_fg`, `combine_evoparams`, `get_params('peak_indices')`, plotting under… | Addressed | REQ-010, REQ-025, REQ-291, REQ-292, REQ-296, REQ-300, REQ-332, REQ-334, REQ-341 | engineering, io, product, reports/plots/cli/stream | `nightly.yml`, `test_array.py`, `test_core_only.py`, `test_dependencies.py`, `test_import_graph.py`, `test_reports_json.py`, `test_schema_version.py`, `test_serialize.py`, `test_tidy.py` | §9.5 CI |
| GAP-54 | No accuracy or robustness tests of the fitting core; tests depend on one noise realisation | Addressed | REQ-013, REQ-105, REQ-106, REQ-119, REQ-295, REQ-298 | engineering, product, shapes/models | `ci.yml`, `test_determinism.py`, `test_engine_sim.py`, `test_reproducibility.py` | §7.2 simulations; §9.5 CI |
| GAP-55 | Citations fragment across renames and multiple papers; trademark conflicts exist for close names | Addressed | REQ-346, REQ-391, REQ-394, REQ-397 | legal/valorisation, publication, reports/plots/cli/stream | `test_citation.py`, `test_cli.py`, `test_deprecations.py`, `test_trademark_file.py` | §9.5 CI |
| GAP-56 | Institutional ownership and patent questions are unresolved before release | Addressed | REQ-019, REQ-398 | legal/valorisation, product | `test_release_gate.py` | §9.5 CI |
| GAP-57 | The CCEP software niche is GPL or unlicensed, and a period-based artifact method has a pending patent appli… | Addressed | REQ-011, REQ-309, REQ-354, REQ-395, REQ-396 | benchmark, engineering, legal/valorisation, product | `test_adapters.py`, `test_dependencies.py`, `test_provenance.py`, `test_spdx_headers.py` | §7.2 simulations; §9.5 CI |
| GAP-58 | Python-only tools forgo most of the citing population, which works in MATLAB, and the CCEP, TMS-EEG and DBS… | Partly addressed | REQ-303, REQ-322, REQ-326, REQ-329 | ecosystem, engineering | `test_cli.py`, `test_docs.py`, `test_plugins.py`, `test_release_checklist.py`, `test_repo_files.py` | §9.5 CI |

### 14.2 Requirement → module → test → validation

Every requirement of §3.2 has at least one test file; requirements defined on simulations are measured in §7.2, those defined on public datasets in §7.3.

| REQ | section | module | test files | validation |
|---|---|---|---|---|
| REQ-001 | §3.1 | product | `tests/test_one_model.py` | §9.5 CI |
| REQ-002 | §3.1 | product | `tests/test_one_model.py` | §9.5 CI |
| REQ-003 | §3.1 | product | `tests/test_polarity.py` | §9.5 CI |
| REQ-004 | §3.1 | product | `tests/test_no_deletion.py` | §9.5 CI |
| REQ-005 | §3.1 | product | `tests/test_status.py` | §9.5 CI |
| REQ-006 | §3.1 | product | `tests/test_uncertainty_columns.py` | §9.5 CI |
| REQ-007 | §3.1 | product | `tests/test_reproduce.py` | §9.5 CI |
| REQ-008 | §3.1 | product | `tests/test_methods_text.py` | §9.5 CI |
| REQ-009 | §3.1 | product | `tests/test_units.py` | §9.5 CI |
| REQ-010 | §3.1 | product | `tests/test_core_only.py` | §9.5 CI |
| REQ-011 | §3.1 | product | `tests/test_dependencies.py` | §9.5 CI |
| REQ-012 | §3.1 | product | `tests/test_vocabulary.py` | §9.5 CI |
| REQ-013 | §3.1 | product | `tests/test_determinism.py` | §9.5 CI |
| REQ-014 | §3.1 | product | `tests/test_presets.py` | §9.5 CI |
| REQ-015 | §3.1 | product | `tests/test_research_use.py` | §9.5 CI |
| REQ-016 | §3.1 | product | `tests/test_no_network.py` | §9.5 CI |
| REQ-017 | §3.1 | product | `tests/test_engine_ccep.py` | §9.5 CI |
| REQ-018 | §3.1 | product | `tools/check_traceability.py` | §9.5 CI |
| REQ-019 | §3.1 | product | `tests/test_release_gate.py` | §9.5 CI |
| REQ-020 | §5.1 | io | `tests/io/test_array.py`, `tests/io/test_mne_roundtrip.py`, `tests/io/test_neo.py`, `tests/io/test_nwb_roundtrip.py` | §9.5 CI |
| REQ-021 | §5.1 | io | `tests/io/test_bids_remote.py`, `tests/test_engine_ccep.py` | §9.5 CI |
| REQ-022 | §5.1 | io | `tests/io/test_events_harmonize.py` | §9.5 CI |
| REQ-023 | §5.1 | io | `tests/io/test_registry.py` | §9.5 CI |
| REQ-024 | §5.1 | io | `tests/io/test_writers.py` | §9.5 CI |
| REQ-025 | §5.1 | io | `tests/io/test_array.py`, `tests/test_core_only.py` | §9.5 CI |
| REQ-040 | §5.2 | events | `tests/events/test_ds004080_events.py` | §7.3 real data |
| REQ-041 | §5.2 | events | `tests/events/test_rate.py`, `tests/events/test_template.py` | §7.2 simulations |
| REQ-042 | §5.2 | events | `tests/events/test_ds004080_events.py`, `tests/events/test_stim_site.py` | §7.3 real data |
| REQ-043 | §5.2 | events | `tests/events/test_saturation.py` | §7.2 simulations |
| REQ-044 | §5.2 | events | `tests/events/test_trains.py` | §7.2 simulations |
| REQ-045 | §5.2 | events | `benchmarks/bench_events.py` | §7.2 simulations |
| REQ-060 | §5.3 | artifact | `tests/artifact/test_masks.py` | §7.2 simulations |
| REQ-061 | §5.3 | artifact | `tests/artifact/test_learned.py`, `tests/artifact/test_template.py` | §7.2 simulations |
| REQ-062 | §5.3 | artifact | `tests/artifact/test_period.py` | §7.2 simulations |
| REQ-063 | §5.3 | artifact | `tests/artifact/test_lowrank.py` | §7.2 simulations |
| REQ-064 | §5.3 | artifact | `tests/artifact/test_subspace.py` | §7.2 simulations |
| REQ-065 | §5.3 | artifact | `tests/artifact/test_polynomial.py` | §7.2 simulations |
| REQ-066 | §5.3 | artifact | `tests/artifact/test_quality.py` | §7.2 simulations |
| REQ-067 | §5.3 | artifact | `tests/artifact/test_paradigm.py` | §7.2 simulations |
| REQ-068 | §5.3 | artifact | `benchmarks/bench_artifact.py` | §7.2 simulations |
| REQ-069 | §5.3 | artifact | `tests/artifact/test_provenance.py` | §7.2 simulations |
| REQ-080 | §5.4 | prep | `tests/prep/test_baseline.py` | §7.2 simulations |
| REQ-081 | §5.4 | prep | `tests/prep/test_reference.py` | §7.2 simulations |
| REQ-082 | §5.4 | prep | `tests/prep/test_reference.py` | §7.2 simulations |
| REQ-083 | §5.4 | prep | `tests/prep/test_reject.py` | §7.2 simulations |
| REQ-084 | §5.4 | prep | `tests/prep/test_reference_sensitivity.py` | §7.3 real data |
| REQ-085 | §5.4 | prep | `tests/prep/test_resample.py` | §7.2 simulations |
| REQ-100 | §5.5.10 | shapes/models | `tests/test_engine_sim.py`, `tests/test_shapes.py` | §7.1 math |
| REQ-101 | §5.5.10 | shapes/models | `tests/test_shapes.py` | §7.1 math |
| REQ-102 | §5.5.10 | shapes/models | `tests/test_shapes.py` | §7.2 simulations |
| REQ-103 | §5.5.10 | shapes/models | `tests/test_varpro.py` | §7.1 math |
| REQ-104 | §5.5.10 | shapes/models | `tests/test_engine_sim.py` | §7.2 simulations |
| REQ-105 | §5.5.10 | shapes/models | `tests/test_engine_sim.py` | §7.2 simulations |
| REQ-106 | §5.5.10 | shapes/models | `tests/test_engine_sim.py` | §7.2 simulations |
| REQ-107 | §5.5.10 | shapes/models | `tests/test_engine_sim.py` | §7.2 simulations |
| REQ-108 | §5.5.10 | shapes/models | `tests/test_varpro.py` | §7.2 simulations |
| REQ-109 | §5.5.10 | shapes/models | `tests/test_varpro.py` | §7.1 math |
| REQ-110 | §5.5.10 | shapes/models | `tests/test_engine_sim.py` | §7.2 simulations |
| REQ-111 | §5.5.10 | shapes/models | `tests/test_engine_sim.py` | §7.2 simulations |
| REQ-112 | §5.5.10 | shapes/models | `tests/test_engine_sim.py` | §7.2 simulations |
| REQ-113 | §5.5.10 | shapes/models | `tests/test_engine_ccep.py` | §7.3 real data |
| REQ-114 | §5.5.10 | shapes/models | `tests/test_engine_ccep.py` | §7.3 real data |
| REQ-115 | §5.5.10 | shapes/models | `benchmarks/bench_engine.py` | §7.2 simulations |
| REQ-116 | §5.5.10 | shapes/models | `tests/test_engine_sim.py` | §7.2 simulations |
| REQ-117 | §5.5.10 | shapes/models | `tests/test_classical.py` | §7.3 real data |
| REQ-118 | §5.5.10 | shapes/models | `tests/test_status.py` | §7.2 simulations |
| REQ-119 | §5.5.10 | shapes/models | `tests/test_reproducibility.py` | §7.2 simulations |
| REQ-120 | §5.5.10 | shapes/models | `tests/test_reproducibility.py` | §7.2 simulations |
| REQ-121 | §5.5.10 | shapes/models | `tests/test_engine_sim.py` | §7.2 simulations |
| REQ-122 | §5.5.10 | shapes/models | `tests/test_engine_sim.py` | §7.2 simulations |
| REQ-123 | §5.5.10 | shapes/models | `tests/test_methods_text.py` | §7.2 simulations |
| REQ-130 | §5.6 | existence | `tests/test_existence_null.py` | §7.2 simulations |
| REQ-131 | §5.6 | existence | `tests/test_existence_power.py` | §7.2 simulations |
| REQ-132 | §5.6 | existence | `tests/test_existence_power.py` | §7.2 simulations |
| REQ-133 | §5.6 | existence | `tests/test_rules_registry.py` | §7.2 simulations |
| REQ-134 | §5.6 | existence | `tests/test_projection.py` | §7.3 real data |
| REQ-135 | §5.6 | existence | `tests/test_existence_ccep.py` | §7.3 real data |
| REQ-136 | §5.6 | existence | `tests/test_existence_null.py` | §7.2 simulations |
| REQ-137 | §5.6 | existence | `tests/test_rules_registry.py` | §7.2 simulations |
| REQ-138 | §5.6 | existence | `benchmarks/bench_existence.py` | §7.2 simulations |
| REQ-139 | §5.6 | existence | `tests/test_existence_ccep.py` | §7.2 simulations |
| REQ-140 | §5.6 | existence | `tests/test_existence_text.py` | §7.2 simulations |
| REQ-150 | §5.7 | trials | `tests/test_trials_sim.py` | §7.2 simulations |
| REQ-151 | §5.7 | trials | `tests/test_trials_sim.py` | §7.2 simulations |
| REQ-152 | §5.7 | trials | `tests/test_trials_sim.py` | §7.2 simulations |
| REQ-153 | §5.7 | trials | `tests/test_trials_sim.py` | §7.2 simulations |
| REQ-154 | §5.7 | trials | `tests/test_trials_sim.py` | §7.2 simulations |
| REQ-155 | §5.7 | trials | `tests/test_trials_sim.py` | §7.2 simulations |
| REQ-156 | §5.7 | trials | `tests/test_trials_ccep.py` | §7.3 real data |
| REQ-157 | §5.7 | trials | `tests/test_trials_table.py` | §7.2 simulations |
| REQ-158 | §5.7 | trials | `tests/test_trials_ccep.py` | §7.3 real data |
| REQ-170 | §5.8 | hierarchical | `tests/test_hier_eb.py` | §7.2 simulations |
| REQ-171 | §5.8 | hierarchical | `tests/test_hier_eb.py` | §7.2 simulations |
| REQ-172 | §5.8 | hierarchical | `tests/test_hier_bayes.py`, `tests/test_hier_eb.py` | §7.2 simulations |
| REQ-173 | §5.8 | hierarchical | `tests/test_hier_bayes.py` | §7.2 simulations |
| REQ-174 | §5.8 | hierarchical | `tests/test_hier_bayes.py` | §7.2 simulations |
| REQ-175 | §5.8 | hierarchical | `tests/test_hier_eb.py` | §7.2 simulations |
| REQ-176 | §5.8 | hierarchical | `tests/test_hier_eb.py` | §7.2 simulations |
| REQ-177 | §5.8 | hierarchical | `tests/test_hier_eb.py` | §7.2 simulations |
| REQ-178 | §5.8 | hierarchical | `tests/test_hier_ccep.py` | §7.3 real data |
| REQ-190 | §5.9 | spatial | `tests/test_spatial_sim.py` | §7.2 simulations |
| REQ-191 | §5.9 | spatial | `tests/test_spatial_sim.py` | §7.2 simulations |
| REQ-192 | §5.9 | spatial | `tests/test_spatial_sim.py` | §7.2 simulations |
| REQ-193 | §5.9 | spatial | `tests/test_spatial_sim.py` | §7.2 simulations |
| REQ-194 | §5.9 | spatial | `tests/test_spatial_bpc.py` | §7.3 real data |
| REQ-195 | §5.9 | spatial | `tests/test_spatial_tables.py` | §7.2 simulations |
| REQ-196 | §5.9 | spatial | `tests/test_spatial_tables.py` | §7.3 real data |
| REQ-210 | §5.10 | tf | `tests/test_tf_transform.py` | §7.2 simulations |
| REQ-211 | §5.10 | tf | `tests/test_tf_transform.py` | §7.1 math |
| REQ-212 | §5.10 | tf | `tests/test_tf_sim.py` | §7.2 simulations |
| REQ-213 | §5.10 | tf | `tests/test_tf_sim.py` | §7.2 simulations |
| REQ-214 | §5.10 | tf | `tests/test_tf_sim.py` | §7.2 simulations |
| REQ-215 | §5.10 | tf | `tests/test_tf_sim.py` | §7.2 simulations |
| REQ-216 | §5.10 | tf | `tests/test_tf_ccep.py` | §7.3 real data |
| REQ-217 | §5.10 | tf | `tests/test_tf_zones_registry.py` | §7.2 simulations |
| REQ-218 | §5.10 | tf | `benchmarks/bench_tf.py` | §7.2 simulations |
| REQ-219 | §5.10 | tf | `tests/test_tf_text.py` | §7.2 simulations |
| REQ-230 | §5.11 | stats | `tests/test_stats_bootstrap.py` | §7.2 simulations |
| REQ-231 | §5.11 | stats | `tests/test_stats_jackknife.py` | §7.2 simulations |
| REQ-232 | §5.11 | stats | `tests/test_stats_compare.py` | §7.2 simulations |
| REQ-233 | §5.11 | stats | `tests/test_stats_reliability.py` | §7.3 real data |
| REQ-234 | §5.11 | stats | `tests/test_stats_reliability.py` | §7.2 simulations |
| REQ-235 | §5.11 | stats | `tests/test_stats_export.py` | §7.2 simulations |
| REQ-236 | §5.11 | stats | `tests/test_stats_export.py` | §7.2 simulations |
| REQ-237 | §5.11 | stats | `tests/test_stats_export.py` | §7.2 simulations |
| REQ-250 | §5.12 | sim | `tests/test_sim_determinism.py` | §7.1 math |
| REQ-251 | §5.12 | sim | `tests/test_sim_noise.py` | §7.1 math |
| REQ-252 | §5.12 | sim | `tests/test_sim_noise.py` | §7.1 math |
| REQ-253 | §5.12 | sim | `tests/test_sim_artifact.py` | §7.2 simulations |
| REQ-254 | §5.12 | sim | `tests/test_import_graph.py` | §7.2 simulations |
| REQ-255 | §5.12 | sim | `tests/test_sim_truth.py` | §7.2 simulations |
| REQ-256 | §5.12 | sim | `tests/test_sim_truth.py` | §7.2 simulations |
| REQ-270 | §6.4 | presets | `tests/test_presets.py` | §7.3 presets |
| REQ-271 | §6.4 | presets | `tests/test_presets.py` | §7.3 presets |
| REQ-272 | §6.4 | presets | `tests/test_presets.py` | §7.3 presets |
| REQ-273 | §6.4 | presets | `tests/test_engine_ccep.py` | §7.3 presets |
| REQ-274 | §6.4 | presets | `tests/validation/test_erpcore.py` | §7.3 real data |
| REQ-275 | §6.4 | presets | `tests/validation/test_tep_ds002094.py` | §7.3 real data |
| REQ-276 | §6.4 | presets | `tests/test_presets_sim.py` | §7.3 presets |
| REQ-277 | §6.4 | presets | `tests/test_presets_sim.py` | §7.3 presets |
| REQ-278 | §6.4 | presets | `tests/validation/test_neuropixels_lfp.py` | §7.3 real data |
| REQ-279 | §6.4 | presets | `tests/test_presets_sim.py` | §7.3 presets |
| REQ-280 | §6.4 | presets | `tests/test_presets.py` | §7.3 presets |
| REQ-290 | §9.1 | engineering | `benchmarks/run_benchmarks.py` | §9.5 CI |
| REQ-291 | §9.2 | engineering | `tests/test_core_only.py` | §9.5 CI |
| REQ-292 | §9.2 | engineering | `tests/test_dependencies.py` | §9.5 CI |
| REQ-293 | §9.3 | engineering | `tests/test_deprecations.py` | §9.5 CI |
| REQ-294 | §9.4 | engineering | `tests/test_provenance.py` | §9.5 CI |
| REQ-295 | §9.4 | engineering | `tests/test_determinism.py` | §9.5 CI |
| REQ-296 | §9.5 | engineering | `tests/test_import_graph.py` | §9.5 CI |
| REQ-297 | §9.5 | engineering | `.github/workflows/ci.yml` | §9.5 CI |
| REQ-298 | §9.5 | engineering | `.github/workflows/ci.yml` | §9.5 CI |
| REQ-299 | §9.5 | engineering | `benchmarks/run_benchmarks.py`, `benchmarks/stream_latency.py` | §9.5 CI |
| REQ-300 | §9.5 | engineering | `.github/workflows/nightly.yml` | §9.5 CI |
| REQ-301 | §9.6 | engineering | `tests/test_typing.py` | §9.5 CI |
| REQ-302 | §9.7 | engineering | `.github/workflows/docs.yml` | §9.5 CI |
| REQ-303 | §9.7 | engineering | `tests/test_docs.py` | §9.5 CI |
| REQ-304 | §9.8 | engineering | `tests/test_logging.py` | §9.5 CI |
| REQ-305 | §9.9 | engineering | `.github/workflows/ci.yml` | §9.5 CI |
| REQ-306 | §9.10 | engineering | `tests/test_fetch.py`, `tests/test_network.py` | §9.5 CI |
| REQ-307 | §9.11 | engineering | `tests/test_errors.py` | §9.5 CI |
| REQ-308 | §9.3 | engineering | `benchmarks/run_benchmarks.py`, `tests/test_citation.py` | §9.5 CI |
| REQ-309 | §9.2 | engineering | `tests/test_dependencies.py`, `tests/test_provenance.py` | §9.5 CI |
| REQ-310 | §10.1 | ecosystem | `tests/test_mne_io.py` | §9.5 CI |
| REQ-311 | §10.1 | ecosystem | `tests/test_mne_io.py` | §9.5 CI |
| REQ-312 | §10.1 | ecosystem | `tools/check_no_inherited_code.py` | §9.5 CI |
| REQ-313 | §10.2 | ecosystem | `tests/test_bids_events.py`, `tests/test_bids_validator.py` | §9.5 CI |
| REQ-314 | §10.2 | ecosystem | `tests/test_bids_events.py`, `tests/test_bids_validator.py` | §9.5 CI |
| REQ-315 | §10.2 | ecosystem | `tests/test_bids_derivatives.py` | §9.5 CI |
| REQ-316 | §10.2 | ecosystem | `tests/test_bids_derivatives.py` | §9.5 CI |
| REQ-317 | §10.3 | ecosystem | `tests/test_neo_io.py`, `tests/test_nwb_io.py` | §9.5 CI |
| REQ-318 | §10.3 | ecosystem | `tests/test_neo_io.py`, `tests/test_nwb_io.py` | §9.5 CI |
| REQ-319 | §10.3 | ecosystem | `tests/test_neo_io.py`, `tests/test_nwb_io.py` | §9.5 CI |
| REQ-320 | §10.4 | ecosystem | `tests/test_sklearn_api.py`, `tests/test_sklearn_checks.py` | §9.5 CI |
| REQ-321 | §10.4 | ecosystem | `tests/test_sklearn_api.py`, `tests/test_sklearn_checks.py` | §9.5 CI |
| REQ-322 | §10.5 | ecosystem | `tests/test_cli.py` | §9.5 CI |
| REQ-323 | §10.5 | ecosystem | `tests/test_cli.py` | §9.5 CI |
| REQ-324 | §10.6 | ecosystem | `benchmarks/stream_latency.py`, `tests/test_stream.py`, `tests/test_stream_lsl.py` | §9.5 CI |
| REQ-325 | §10.6 | ecosystem | `benchmarks/stream_latency.py`, `tests/test_docs.py`, `tests/test_stream.py`, `tests/test_stream_lsl.py` | §9.5 CI |
| REQ-326 | §10.7 | ecosystem | `tests/test_docs.py`, `tests/test_plugins.py`, `tests/test_repo_files.py` | §9.5 CI |
| REQ-327 | §10.8 | ecosystem | `tests/test_docs.py`, `tests/test_plugins.py`, `tests/test_repo_files.py` | §9.5 CI |
| REQ-328 | §10.8 | ecosystem | `tests/test_docs.py`, `tests/test_plugins.py`, `tests/test_repo_files.py` | §9.5 CI |
| REQ-329 | §10.8 | ecosystem | `tests/test_release_checklist.py` | §9.5 CI |
| REQ-330 | §9.12 | engineering | `tests/test_serialize.py`, `tests/test_tidy.py` | §9.5 CI |
| REQ-331 | §9.12 | engineering | `tests/test_settings_log.py` | §9.5 CI |
| REQ-332 | §9.12 | engineering | `tests/test_serialize.py`, `tests/test_tidy.py` | §9.5 CI |
| REQ-333 | §9.12 | engineering | `tests/test_reproduce.py` | §9.5 CI |
| REQ-334 | §9.12 | engineering | `tests/test_schema_version.py` | §9.5 CI |
| REQ-335 | §9.12 | engineering | `tests/test_provenance.py` | §9.5 CI |
| REQ-336 | §9.12 | engineering | `tests/test_bids_derivatives.py`, `tests/test_nwb_io.py` | §9.5 CI |
| REQ-337 | §9.12 | engineering | `tests/test_provenance.py` | §9.5 CI |
| REQ-338 | §9.12 | engineering | `tests/test_reports.py` | §9.5 CI |
| REQ-340 | §5.13 | reports/plots/cli/stream | `tests/test_reports_text.py` | §9.5 CI |
| REQ-341 | §5.13 | reports/plots/cli/stream | `tests/test_reports_json.py` | §9.5 CI |
| REQ-342 | §5.13 | reports/plots/cli/stream | `tests/test_bids_derivatives.py` | §9.5 CI |
| REQ-343 | §5.13 | reports/plots/cli/stream | `tests/test_reports_html.py` | §9.5 CI |
| REQ-344 | §5.13 | reports/plots/cli/stream | `tests/test_plots.py` | §9.5 CI |
| REQ-345 | §5.13 | reports/plots/cli/stream | `tests/test_cli.py` | §9.5 CI |
| REQ-346 | §5.13 | reports/plots/cli/stream | `tests/test_cli.py` | §9.5 CI |
| REQ-347 | §5.13 | reports/plots/cli/stream | `tests/test_stream.py` | §9.5 CI |
| REQ-348 | §5.13 | reports/plots/cli/stream | `tests/test_stream.py` | §9.5 CI |
| REQ-349 | §5.13 | reports/plots/cli/stream | `tests/test_layering.py` | §9.5 CI |
| REQ-350 | §8.5 | benchmark | `tests/benchmark/test_suite_core.py` | §7.2 simulations |
| REQ-351 | §8.5 | benchmark | `tests/benchmark/test_schemas.py` | §7.2 simulations |
| REQ-352 | §8.5 | benchmark | `tests/benchmark/test_scoring.py` | §7.2 simulations |
| REQ-353 | §8.5 | benchmark | `tests/benchmark/test_suite_core.py` | §7.2 simulations |
| REQ-354 | §8.5 | benchmark | `tests/benchmark/test_adapters.py` | §7.2 simulations |
| REQ-355 | §8.5 | benchmark | `tests/benchmark/test_deposit.py` | §7.2 simulations |
| REQ-356 | §8.5 | benchmark | `tests/benchmark/test_deposit.py` | §7.2 simulations |
| REQ-357 | §8.5 | benchmark | `tests/benchmark/test_leaderboard.py` | §7.2 simulations |
| REQ-370 | §7.2 | validation | `tests/validation/test_grid.py` | §7.3–7.4 |
| REQ-371 | §7.3 | validation | `tests/validation/test_subject_list.py` | §7.3 real data |
| REQ-372 | §7.3 | validation | `tests/validation/test_carla_ds004977.py` | §7.3 real data |
| REQ-373 | §7.3 | validation | `tests/validation/test_localize_mi.py` | §7.3 real data |
| REQ-374 | §7.3 | validation | `tests/validation/test_report_schema.py` | §7.3 real data |
| REQ-375 | §7.4 | validation | `tests/validation/test_robustness.py` | §7.3–7.4 |
| REQ-376 | §7.4 | validation | `tests/validation/test_robustness.py` | §7.3–7.4 |
| REQ-390 | §11.8 | publication | `tests/test_methods_text.py` | §9.5 CI |
| REQ-391 | §11.8 | publication | `tests/test_citation.py` | §9.5 CI |
| REQ-392 | §11.8 | publication | `tests/benchmark/test_leaderboard.py` | §9.5 CI |
| REQ-393 | §11.8 | publication | `tests/test_figures_from_tag.py` | §9.5 CI |
| REQ-394 | §11.8 | publication | `tests/test_deprecations.py` | §9.5 CI |
| REQ-395 | §13.7 | legal/valorisation | `tests/test_dependencies.py` | §9.5 CI |
| REQ-396 | §13.7 | legal/valorisation | `tests/test_spdx_headers.py` | §9.5 CI |
| REQ-397 | §13.7 | legal/valorisation | `tests/test_trademark_file.py` | §9.5 CI |
| REQ-398 | §13.7 | legal/valorisation | `tests/test_release_gate.py` | §9.5 CI |
| REQ-399 | §13.7 | legal/valorisation | `tests/test_soup_inventory.py` | §9.5 CI |
## 15. Risks and mitigations

Likelihood (L) and impact (I) are rated L/M/H. "Early warning" is the observable signal that triggers the mitigation; the last column names the section of this specification that implements it. Evidence is cited with the research keys; competitor tools are named because the risks depend on them.

### 15.1 Competition

| id | risk | L | I | early warning | mitigation | spec |
|---|---|---|---|---|---|---|
| C1 | ERPparam is released on PyPI or published with a preprint. Its `main` branch has been dormant since 2025-10-06, but pull requests are dated 2026-01-13, 2026-01-14 and 2026-07-30 and a 2023 SfN abstract exists [LIT-fooof-61][LIT-fooof-64][LIT-fooof-69]. | M | M | a PyPI project `erpparam`; a bioRxiv preprint; a Voytek-lab talk or abstract listing it | differentiation that ERPparam does not have (slow background, existence test, trial hierarchy, artifact handling, multi-modality presets, BIDS/NWB) [LIT-fooof-74][LIT-fooof-57]; include it as a benchmark comparison; acknowledge the lineage in NOTICE and papers; do not rename; offer the joint benchmark | §1, §5, §8, §11.5 |
| C2 | A Voytek-lab time-domain extension of its spectral tool reaches evoked responses: specparam 2.0 carries event-locked time models and a 2025 SPR abstract addresses event-related oscillations [LIT-fooof-19][LIT-fooof-69]. | M | M | specparam 2.0 final release notes; abstracts mentioning evoked or event-related parameterization | the time-domain component model and the existence/trial layers are a different object; the `tf` module reuses rather than competes (spectra of residuals after the component fit); cite and interoperate | §5 (`tf`), §10 |
| C3 | MNE-Python adds classical measures natively (fractional latencies, local peaks, area) [LIT-erptools-17][LIT-erptools-19]. | M | L–M | a pull request touching `mne.stats.erp` or `Evoked.get_peak` | contribute the BSD-3-licensed classical-measure code upstream and keep the model, existence, trial and hierarchical layers here; MNE is a channel, not a competitor | §5.5.6, §10 |
| C4 | ER-detect (already Python, GPL-3) or CRP (a Python module exists in BEST) is packaged and published as the Python reference [LIT-naming-51][LIT-artifact-46][LIT-fooof-92]. | M | M | a JOSS submission or PyPI release; citation growth of the ER-detect paper | both are single-modality detectors without shape timing or uncertainty [LIT-fooof-92]; both are comparison methods on the leaderboard; the permissive licence lets MNE-based pipelines import this package and not a GPL one | §8, §11.3.2 |

The ERPparam case deserves one paragraph. The inherited code is a fork of it, so an ERPparam release would not make this project derivative in any new sense; it would make the lineage visible. The response is the same in every scenario (§11.5): keep the name, keep the model, add the comparison, cite, and let the benchmark show the difference. The worst case is not competition but ambiguity, which is why REQ-396 documents what was inherited and what was rewritten (§4.5).

### 15.2 Legal

| id | risk | L | I | early warning | mitigation | spec |
|---|---|---|---|---|---|---|
| L1 | GPL contamination: a contributor imports ER-detect, FieldTrip-derived code or another GPL tool into the core; the Apache Software Foundation holds that GPLv3 code cannot be included in Apache projects and GPLv2 is incompatible with Apache-2.0 [LIT-naming-14]. | M | H | a pull request adding a dependency; a licence scanner alert; `pip` resolving a GPL package through an extra | REQ-395 licence scan in CI; GPL tools only at arm's length from separate adapter packages [LIT-naming-57]; PR template question | §13.7 |
| L2 | The two pre-grant applications issue with claims that read on the `period_template` or `matching_pursuit` artifact methods or on the joint artifact-plus-response fit: PARRM (Brown, US 2022/0338786 A1) [LIT-naming-33][LIT-artifact-4] and matching-pursuit removal (Washington University, US 2023/0240618 A1) [LIT-artifact-25]. | L–M | M–H | a grant notice on either application; a letter | freedom-to-operate opinion before either method ships (HIIE questions 7 and 21); clean-room implementation from the papers [LIT-naming-36][LIT-naming-37]; the joint fit documented as parametric least squares, not matching pursuit; the `period_template` method optional and removable without breaking the API | §13.4, §5 (`artifact`) |
| L3 | Apache-2.0 §3 grants a patent licence on UAB contributions that HIIE later wants to keep exclusive [LIT-naming-13]. | M | M | HIIE raises exclusivity after a public release | decide the licence (Apache-2.0 versus `BSD-3-Clause AND Apache-2.0`) at milestone M1, before any release; inherited files stay Apache-2.0 regardless | §11.4, §13.4 |
| L4 | Public disclosure before the HIIE decision (a poster deadline, a repository toggled public, a demonstration to a visiting group); HIIE counts "publication, presentation, abstract or conversation" as disclosure, and absolute novelty applies outside the US [LIT-naming-25][LIT-naming-26][LIT-naming-31]. | M | H | an abstract submission form open before M1; a request to show the tool outside the lab | REQ-398 gate in the release workflow; private repository; disclosure filed early (M0) so that the waiting time is short; a one-page "what may be said" note for lab members | §11.4, §13.7 |
| L5 | Unlicensed code is copied: the PARRM and MPARRM MATLAB references and a UAB CCEP repository carry no licence (all rights reserved by default) [LIT-naming-36][LIT-naming-37][LIT-naming-52]. | L | M | a file with MATLAB-translated structure and no paper reference | REQ-396 provenance map with `REIMPLEMENTED.md`; code review asks for the equation source | §13.7 |

### 15.3 Scientific

| id | risk | L | I | early warning | mitigation | spec |
|---|---|---|---|---|---|---|
| S1 | Overlapping components are not identifiable: sum-of-kernel models are "sloppy" [LIT-fitting-34]; profile likelihood separates structural, practical and identifiable cases [LIT-fitting-35]; the spectral ancestor cannot separate strongly overlapping peaks [LIT-fooof-50]. | H | H | parameter correlations above 0.95; flat profile-likelihood scans; simulation MAE rising sharply when separation falls below one FWHM | identifiability diagnostics as a first-class output (FIM spectrum, correlations, profile scans); minimum-separation and ordering constraints; model selection instead of thresholds; `status="unresolvable"` rather than a wrong number; resolvability curves in §7 | §5.5, §7 |
| S2 | Trial-level fits fail at low SNR: the ancestor advises segments of at least 500 ms for per-trial fits [LIT-fooof-25]; a full Bayesian waveform model costs about 10 h per subject [LIT-fitting-3]; threshold-defined CCEP responses reproduce near zero between trains at Z = 3 [LIT-variability-7]. | H | M–H | interval coverage below 90 % in §7; `ConvergenceWarning` rate above 5 % on real data; trial fits slower than the budget in §9 | trial-level degrees of freedom restricted to a latency shift and an amplitude scale of the channel template, with free per-component parameters only above an SNR gate; hierarchical shrinkage; empirical-Bayes backend by default and MCMC on request; coverage tests | §5 (`trials`, `hierarchical`), §7 |
| S3 | Reference-montage dependence: N1 latency 47.4 ms (bipolar) versus 70.7 ms (referential) in one pediatric series [LIT-ccep-27]; 18.1 % of CCEP studies do not state the montage [LIT-ccep-11]; adjusted common average (CARLA) exists only in MATLAB [LIT-ccep-18]. | H | M | benchmark scores that change by more than the inter-rater spread when the montage changes | montage as mandatory metadata (`ch_info.reference`); bipolar, CAR and adaptive CAR computed in parallel; a montage-sensitivity index in every `GroupFit.summary`; Python adjusted-CAR in `prep` | §5 (`prep`), §8 |
| S4 | No public ERNA, SCS-ECAP, CI-ECAP or IONM data: the 2025 pallidal ERNA study withholds data; none was found for the others [LIT-datasets-44][LIT-domains-53]. | H | M | the ERNA, ECAP and IONM presets have only simulated validation at 1.0 | simulators calibrated to published waveform statistics (ERNA 200–450 Hz decays [LIT-domains-53]); presets flagged "simulation-validated only" in `Preset.citation` and in the methods text; NWB/BIDS export template to lower the deposit barrier; outreach to DBS and SCS groups | §5 (`sim`), §6, §8 |
| S5 | The human ceiling: inter-rater agreement for CCEP presence is kappa 0.60 (sensitivity 68 %, specificity 96 %) on 59,178 responses; the best automated rule reaches 0.59 [LIT-ccep-8]. No detector can be shown better than the labels. | H | M | benchmark kappa against labels saturating near 0.6; reviewers asking what "truth" is | exact truth from simulations for accuracy claims; adjudicated two-rater labels with per-rater agreement reported; calibration (does the existence probability match the rater agreement rate?) rather than accuracy as the claim on real data | §8, §11.3.2 |
| S6 | Coloured noise invalidates AIC/BIC: evoked-potential residuals are strongly autocorrelated, so the effective sample size is far below the number of samples and information criteria over-select components unless the noise model is AR(p) or residuals are whitened [LIT-fitting-3]. | H | M–H | `GoodnessOfFit.residual_lag1_autocorr` above 0.5 on fits that pass selection; component counts rising with sampling rate on the same data | AICc/BIC on whitened residuals; AR noise model option; parametric-bootstrap calibration of the K versus K+1 statistic; the lag-1 autocorrelation is stored in every fit (§4.3.4) | §5.5.3, §7 |
| S7 | Model misspecification: sEEG responses are "polymorphic" with variable polarity, only 48.5 % of one series matched a canonical morphology [LIT-ccep-11][LIT-ccep-23]; background families may be wrong, and detrending without justification is warned against [LIT-fooof-54]. | M–H | H | systematic residual structure in a modality; `nrmse` distributions bimodal; preset defaults overridden in most user issues | extensible shape and background registries (`spline` background); misspecification diagnostics reported; the shape-free CRP available as fallback and comparison; validation with deliberate violations; FAQ states the limits | §5.5, §7, §16 |
| S8 | Early components overlap the artifact window: TEP peaks under 100 ms have pipeline concordance −0.03 to 0.59 [LIT-artifact-7]; CCEP N1 begins near 10 ms; ERNA is read 4–20 ms after a pulse. | H | H | fits whose first component sits at the mask edge; amplitude bias versus mask length in §7 | mask-aware fitting with uncertainty inflation near the pulse; joint artifact-plus-response fit; per-sample validity masks propagated to the existence test | §5 (`artifact`, `models`) |

Two of these dominate. S1 and S6 together mean that "how many components" is the question on which every downstream number depends; the specification answers it with diagnostics and calibrated model selection rather than thresholds, and the methods paper must show the failure region explicitly (§11.3.2, F3). S5 means that the benchmark's real-data claims are claims about calibration, not accuracy; the wording of every paper follows that.

### 15.4 Adoption

| id | risk | L | I | early warning | mitigation | spec |
|---|---|---|---|---|---|---|
| A1 | Searchability: "evo" is the house prefix of evolutionary computation (EvoStar tracks; evotorch, evojax, evokit on PyPI/conda-forge) [LIT-naming-12][LIT-naming-4][LIT-naming-2]. | H | L–M | search results for the name dominated by evolutionary tools after release | every title pairs the name with "evoked response parameterization" (REQ-394); docs home and README optimised for the phrase; fallback name `evoresponse` reserved (scored 20/25 against 22/25) [LIT-naming-1][LIT-naming-5] | §11.1, §11.8 |
| A2 | MATLAB-centric communities: the CCEP, ERP and TMS-EEG tool chains are GPL MATLAB (FieldTrip 989 stars, Brainstorm 488, ERPLAB 310, CCEP_Toolbox, TESA) [LIT-naming-41][LIT-naming-43][LIT-naming-45][LIT-naming-52][LIT-domains-4]; MATLAB ports of the ancestor drifted and lost maintenance [LIT-fooof-16][LIT-fooof-27]. | H | M | feature requests for a MATLAB version; EEGLAB-plugin requests | BIDS derivatives as the exchange format; the CLI for non-Python users; a documented MATLAB calling example (`pyrun`) rather than a port; ERPLAB-compatible measure names and wide/long tables | §10, §5.5.6 |
| A3 | Single-maintainer risk: the ancestor's 2.0 spent 30 months in release candidates with delays attributed to a job transition [LIT-fooof-9][LIT-fooof-11]; ERPparam went dormant [LIT-fooof-61]. | H | H | issue response time above two weeks; no second person with release rights | a second maintainer with release rights before 1.0; `CONTRIBUTING.md` and a release runbook; lab succession plan with the PI; small, bounded modules that students can own | §9 |
| A4 | Documentation debt: ERPparam still ships the ancestor's spectral tutorials, example spectra and docs tree three years after the fork [LIT-fooof-85][LIT-fooof-86]. | M | M | tutorials failing to execute in CI; docs pages older than the API they describe | tutorials are executed in CI and gate releases; docstring coverage check; the nine-tutorial structure fixed in §11.6 | §9 |
| A5 | Breaking API changes: the ancestor's 1.0 renamed its core terms and its 2.0 is a declared breaking change with ports not yet audited [LIT-fooof-18][LIT-fooof-11]. | M | M | more than one deprecation per minor release; downstream issues after a release | API freeze at 1.0 under semantic versioning; REQ-293 shims for two minor versions; result schema version with a loader that refuses newer majors (§4.3.5) | §4.4, §9 |
| A6 | Reporting stays incomplete despite the generator: 56 % of clinical reports using the ancestor documented settings and 30 % a goodness-of-fit check [LIT-fooof-4]. | H | M | published papers citing the package without the generated paragraph | the paragraph is generated by default and stored in every result and derivative (REQ-390); a reporting checklist in the PDF report; tutorial 09 | §5 (`reports`), §11.8 |

### 15.5 Engineering

| id | risk | L | I | early warning | mitigation | spec |
|---|---|---|---|---|---|---|
| E1 | Performance at scale: CCEP datasets hold thousands of trials and hundreds of channels at 2–4.8 kHz [LIT-datasets-2][LIT-datasets-11]; the ancestor's fit time is "basically all spent in curve_fit" [LIT-fooof-49]; ERPparam's skewed fits are slow [LIT-fooof-90]; a Gaussian-process trial model takes 10 h per subject [LIT-fitting-3]. | M | M | §9 performance tests above budget; users reporting hour-long group fits | analytic Jacobians in every `ComponentShape`; variable projection for linear parameters; batched vectorised least squares; warm starts across channels and trials; `n_jobs`; optional numba/JAX; performance regression tests | §4.3.4, §9 |
| E2 | Dependency drift: the ancestor reports fits "failing with certain versions of dependencies" [LIT-fooof-45]; the inherited code broke on numpy ≥ 1.24 and matplotlib ≥ 3.9 (§4.5). | H | M | CI failures on the minimum- or maximum-version matrix | light core (numpy, scipy only; REQ-291); extras lazily imported; CI on oldest and newest supported versions; lock file for reproducible docs; REQ-399 SOUP inventory | §4.1, §9, §13.7 |
| E3 | GPU optionality creates two code paths with different numbers. | M | L | CPU/GPU parity tests outside tolerance | GPU strictly optional; parity tests at 1e-6 relative; results record the backend in `Provenance.environment` | §9 |
| E4 | Numerical irreproducibility across platforms (BLAS, random seeds, optimiser restarts). | M | M | cross-platform CI diffs beyond tolerance | `random_state` everywhere; tolerance-based regression tests; environment recorded in `Provenance`; deterministic restart schedule | §4.3.3, §9 |

### 15.6 Data

| id | risk | L | I | early warning | mitigation | spec |
|---|---|---|---|---|---|---|
| D1 | OpenNeuro availability and discovery are brittle: one dataset's mirror is empty, the catalogue lags new datasets, the API is not statically readable [LIT-datasets-14][LIT-datasets-4]. | M | M | a tutorial download failing; a dataset version withdrawn | a versioned dataset registry (manifest with DOI, version, sampling rate, size, licence, events status, loader); CI fixtures copied to Zenodo from the four sub-2 GB sets; readers pinned to dataset versions | §5 (`io`), §8 |
| D2 | Size: ds004080 is 269.4 GB; sampling rates span 1024 Hz to 30 kHz [LIT-datasets-2][LIT-datasets-50]. | H | M | benchmark runs needing more than a workstation | partial remote reads (48 s for the example run of a 2 GB file, §4.5); the NEMAR Zarr copy [LIT-datasets-3]; chunked readers; sub-2 GB fixtures for CI | §5 (`io`), §9 |
| D3 | Metadata inconsistencies: stimulation event columns differ across groups; one 31-subject set has a minimal description and unknown events; 18.1 % of studies omit the montage [LIT-datasets-13][LIT-datasets-2][LIT-ccep-11]. | H | M | reader warnings about missing stimulation fields on a new dataset | an events-harmonisation layer mapping every known deposit to the canonical table of §4.3.1 with a validator that reports missing fields; montage and contact geometry mandatory in `ch_info` | §5 (`io`, `events`) |
| D4 | The lab's ECoG and sEEG cohorts cannot be released for the benchmark paper (IRB, data-use terms). | M | M | IRB amendment refused or delayed | publish derived labels and simulated surrogates only; a data-use-agreement path for the raw cohort; the public tasks stand on CC0 sets | §8, §11.3.3 |

### 15.7 Project

| id | risk | L | I | early warning | mitigation | spec |
|---|---|---|---|---|---|---|
| P1 | Scope against a PhD timeline: the specification covers eight modalities, a hierarchical model and a benchmark. | H | H | milestone M2 (first release) slipping past the build-order estimate; modules started before the CCEP chain is validated | the build order of §12 (CCEP chain first: `io` → `events` → `artifact` → `models` → `existence` → `reports`); each paper gated by a code checklist (§11.3); a cut list that drops `tf` zones, MCMC backend and non-CCEP presets to "after 1.0" if needed | §12 |
| P2 | One developer: the same person writes code, labels data, runs the benchmark and writes the papers. | H | H | commit activity concentrated in one account; no outside pull request after M2 | bounded modules for rotation students (readers, presets, a shape); a lab co-maintainer; the two-maintainer rule before 1.0 (A3); community issues triaged weekly | §9 |
| P3 | Labelling and compute cost for the benchmark. | M | M | fewer than two raters available; label throughput below plan | a minimal labelling viewer that writes BIDS-derivative annotations; reuse of the ER-detect annotation set if its authors share it [LIT-ccep-8]; software-supplement and open-science grant applications | §8 |
| P4 | HIIE processing delays the first public release and with it every downstream milestone (§11.4). | M | M | no written decision within the office's stated time after M0 | file the disclosure as soon as the novelty list of §13.4 is stable, not when the code is finished; keep building privately; prepare all release artefacts (docs, CFF, Zenodo metadata) in parallel so that M2 follows M1 within days | §11.4, §13.5 |

### 15.8 Review cadence

The table is reviewed at each milestone of §11.4 and whenever a success metric of §11.7 falls below half of its target. A risk whose early warning has fired is listed in the release notes of the next version with the mitigation taken.
## 16. Open questions for the author

Only questions that the author can decide alone are listed here; questions that belong to the university (ownership, licence approval, patents, trademark, startup roles) are in §13.5 and are not repeated. Each question has a recommendation; the specification is written under the recommended answer, so that work can proceed until the author decides otherwise.

| # | question | options | recommendation | what changes if the other option is chosen |
|---|---|---|---|---|
| Q1 | Final package name | `evoparam`; `evoresponse` (fallback); another name from the naming study | `evoparam` (best score in the naming study: free on PyPI, conda-forge and the main domains; GitHub organisation `evoparam-tools` because the user name `evoparam` is taken) [LIT-naming-1], [LIT-naming-4], [LIT-naming-7], [LIT-naming-8], [LIT-naming-55] | a global rename before the first public commit; nothing after M2 |
| Q2 | Licence to propose to HIIE | Apache-2.0 for the whole project; Apache-2.0 for inherited files and BSD-3 for new files | Apache-2.0 for the whole project (one licence, `NOTICE` already in place, patent grant protects users); MNE contributions written separately under BSD-3 (§10.1) | per-file SPDX headers and a mixed-licence `LICENSE` directory; contributor guide explains both |
| Q3 | Default shapes of the `ccep` preset | skewed Gaussian + Gaussian; ex-Gaussian + Gaussian; gamma + Gaussian | skewed Gaussian + Gaussian until the layer-2 validation (§7.2) compares them on CCEP simulations and ds004080; the winner becomes the default before M2 | preset values and the methods text; no API change |
| Q4 | Default selection threshold | `ΔBIC ≤ −10`; `ΔBIC ≤ −6`; AICc | `−10`, then calibrated so that REQ-112 holds (≤ 5 % false components on noise); report the calibration in the methods paper | sensitivity and false-component rate trade-off (REQ-105 vs REQ-112) |
| Q5 | Whether to implement the two artifact methods covered by pending patent applications before the freedom-to-operate opinion | implement now behind an opt-in flag in a private branch; wait | implement in a private branch now (research use is not distribution), merge into the public package only after HIIE's freedom-to-operate answer (§13.4, §13.5 Q7 and Q21) | if waiting: §5.3 methods 4 and 8 are delivered later; presets `erna` use `template` instead |
| Q6 | First Bayesian engine | NumPyro; PyMC | NumPyro (JAX speed, vectorised chains, smaller dependency footprint); PyMC second through the same model definition | REQ-176 time budget is re-measured |
| Q7 | Contact with the Voytek lab | before first release; after first public release; never | after the first public release, offering a joint benchmark rather than co-authorship of the core method (§11.5) | earlier contact risks disclosure before the HIIE decision and dilutes differentiation |
| Q8 | Real clinical data in the benchmark | public data only; add labelled UAB recordings | public data only for the first public release; add UAB recordings only after IRB and HIIE clearance of labels (§13.5 Q18) | benchmark paper timeline (§11.3.3) |
| Q9 | Content of the first public release (M2) | everything of S0–S15; the full build S0–S20 | S0–S16 (engine, existence, trials, spatial, tf, stats, presets, validation, reports, CLI); hierarchical Bayesian engines, stream and benchmark follow in minor releases — this is a release cadence, not a scope reduction (§2) | M2 moves later; the JOSS six-month public-history clock starts later [LIT-standards-58] |
| Q10 | Governance during 0.x | author as sole decision maker; small steering group from the start | sole decision maker during 0.x, steering group of three at 1.0 (§10.8) | contributor guide and release process |
| Q11 | Where the free online mode stops | free soft-real-time `StreamFitter`; no online mode in the free package | free soft-real-time online mode (research use), paid hard-real-time and device integration (§13.1) | §5.13 and §10.6 move to the paid module; adoption by intra-operative research groups is lower |
| Q12 | Minimum Python version | 3.10; 3.11 | 3.10 (CI already covers 3.10–3.13); current MNE-Python and scikit-learn releases require 3.11 [LIT-standards-34] [LIT-standards-51] and MNE's own CI tests 3.11–3.14 [LIT-standards-77], so 3.10 support relies on older releases of the `mne` and `tests` extras and should be dropped when they fall behind the floors of §9.2 | typing syntax and CI matrix |
| Q13 | Display unit for amplitudes | µV everywhere; native unit of the recording | µV by default with the unit carried in every container and table (§4.3), so ECAP and scalp data can keep their native scale on request | reports and plots labels only |
| Q14 | Authorship rule for the three papers | author first and corresponding on all three; shared first authorship on the benchmark paper | first and corresponding author on the software and methods papers; benchmark paper first author, with data-providing labs as co-authors (§11.3) | contributor agreements and the benchmark consortium invitation |
