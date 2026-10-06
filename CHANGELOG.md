# Changelog

## 0.1.0 (in progress)

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
