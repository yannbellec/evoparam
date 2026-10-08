# Changelog

## Unreleased

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
