# Changelog

## Unreleased

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
