# Changelog

## 0.1.0 (in progress)

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
