# Changelog

## 0.1.0 (in progress)

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
