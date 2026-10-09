"""Data containers: :class:`TrialSet` (trials × channels × time) and :class:`Evoked`.

Containers are dataclasses that are frozen by convention: every method returns
a new object and appends one step to its provenance. Time is in seconds,
relative to the event onset, and is always the last axis. Amplitude units
travel with the data.
"""

import os
import time as _time
import warnings
from dataclasses import dataclass, field, fields, replace

import numpy as np

from evoparam.errors import DataError
from evoparam.data._compare import DeepEqualMixin
from evoparam.data.provenance import Provenance, compute_data_hash

__all__ = ["TrialSet", "Evoked", "UNITS", "MODALITIES", "METADATA_COLUMNS", "CH_INFO_COLUMNS",
           "auto_chunk"]

# Allowed amplitude units and their factor to volts (None: not a voltage).
UNITS = {"uV": 1e-6, "mV": 1e-3, "V": 1.0, "fT": None, "au": None}

MODALITIES = ("spes", "erp", "tep", "erna", "sep", "ecap", "vep", "baep", "mep", "lfp_ep")

# Standard metadata columns (spec Table 4.1) and the dtype kind they must have.
# Kinds: "i" integer, "f" float, "b" bool, "s" string.
METADATA_COLUMNS = {
    "trial": "i", "subject": "s", "session": "s", "run": "s", "task": "s", "condition": "s",
    "stim_site": "s", "stim_current_mA": "f", "stim_pulse_width_us": "f", "stim_polarity": "s",
    "stim_frequency_hz": "f", "stim_n_pulses": "i", "stim_charge_uC": "f", "state": "s",
    "block": "i", "quality": "f", "rejected": "b", "reject_reason": "s",
}

# Standard channel-table columns (spec Table 4.2).
CH_INFO_COLUMNS = {
    "name": "s", "type": "s", "group": "s", "x": "f", "y": "f", "z": "f", "coord_space": "s",
    "reference": "s", "bad": "b", "stim_contact": "b", "tissue": "s", "region": "s",
}

TRIMMED_PROPORTION = 0.2


###################################################################################################
## Helpers

def _is_dataframe(obj):
    return type(obj).__name__ == "DataFrame" and type(obj).__module__.startswith("pandas")


def _dtype_kind(series):
    kind = series.dtype.kind
    if kind in "iu":
        return "i"
    if kind == "f":
        return "f"
    if kind == "b":
        return "b"
    if kind in "OUT" or str(series.dtype) in ("str", "string"):
        return "s"
    return kind


def _check_table(table, name, n_rows, columns):
    if not _is_dataframe(table):
        raise DataError(f"{name}: expected a pandas DataFrame, got {type(table).__name__}.")
    if len(table) != n_rows:
        raise DataError(f"{name}: expected {n_rows} rows, got {len(table)}.")
    index = table.index
    if not (type(index).__name__ == "RangeIndex" and index.start == 0 and index.step == 1):
        raise DataError(f"{name}: the index must be a RangeIndex 0..{n_rows - 1}.")
    for column, kind in columns.items():
        if column in table.columns:
            found = _dtype_kind(table[column])
            if found != kind and not (kind == "f" and found == "i"):
                raise DataError(f"{name}: column '{column}' must have dtype kind '{kind}', "
                                f"got '{table[column].dtype}'.")


def _check_axis(times, sfreq, n_times):
    if times.ndim != 1 or times.size < 2:
        raise DataError("times: must be a 1-D array with at least 2 samples.")
    if times.size != n_times:
        raise DataError(f"times: {times.size} samples but data has {n_times} on its last axis.")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise DataError("times: must be finite and strictly increasing.")
    if not (np.isfinite(sfreq) and sfreq > 0):
        raise DataError(f"sfreq: must be a positive number, got {sfreq}.")
    implied = 1.0 / np.median(np.diff(times))
    if abs(implied - sfreq) / sfreq >= 1e-6:
        raise DataError(f"sfreq: {sfreq} Hz does not match the time axis ({implied:.6f} Hz).")


def _check_names(ch_names, n_channels):
    if len(ch_names) != n_channels:
        raise DataError(f"ch_names: {len(ch_names)} names for {n_channels} channels.")
    if any(not isinstance(name, str) or not name for name in ch_names):
        raise DataError("ch_names: every name must be a non-empty string.")
    if len(set(ch_names)) != len(ch_names):
        duplicated = sorted({name for name in ch_names if ch_names.count(name) > 1})
        raise DataError(f"ch_names: names must be unique; duplicated: {duplicated}.")


def _check_common(obj):
    if obj.units not in UNITS:
        raise DataError(f"units: '{obj.units}' is not one of {sorted(UNITS)}.")
    if obj.modality is not None and obj.modality not in MODALITIES:
        raise DataError(f"modality: '{obj.modality}' is not one of {MODALITIES} or None.")
    if not isinstance(obj.species, str) or not obj.species or obj.species != obj.species.lower():
        raise DataError(f"species: must be a non-empty lower-case string, got {obj.species!r}.")
    if not isinstance(obj.provenance, Provenance):
        raise DataError("provenance: must be a Provenance instance.")


def _check_nan(data, masks):
    """NaN is allowed only where ``masks`` is True."""

    for index in range(data.shape[0]):
        bad = np.isnan(data[index])
        if masks is not None:
            bad &= ~masks[index]
        if bad.any():
            raise DataError("data: NaN found outside the samples marked in masks.")


def _time_slice(times, sfreq, tmin, tmax):
    """Samples with tmin <= t < tmax, as a slice (windows include tmin, exclude tmax)."""

    tol = 1e-3 / sfreq
    keep = np.ones(times.size, dtype=bool)
    if tmin is not None:
        keep &= times >= tmin - tol
    if tmax is not None:
        keep &= times < tmax - tol
    index = np.flatnonzero(keep)
    if index.size < 2:
        raise DataError(f"Time window [{tmin}, {tmax}) keeps fewer than 2 samples.")
    return slice(int(index[0]), int(index[-1]) + 1)


def _pick_index(ch_names, picks):
    n_channels = len(ch_names)
    if picks is None:
        return np.arange(n_channels)
    if isinstance(picks, (str, int, np.integer)):
        picks = [picks]
    picks = list(picks) if not isinstance(picks, np.ndarray) else picks
    if isinstance(picks, np.ndarray) and picks.dtype == bool:
        if picks.size != n_channels:
            raise DataError(f"picks: boolean mask of length {picks.size} for {n_channels} channels.")
        return np.flatnonzero(picks)
    index = []
    lookup = {name: position for position, name in enumerate(ch_names)}
    for pick in picks:
        if isinstance(pick, str):
            if pick not in lookup:
                raise DataError(f"picks: unknown channel '{pick}'.")
            index.append(lookup[pick])
        elif isinstance(pick, (int, np.integer)):
            if not -n_channels <= pick < n_channels:
                raise DataError(f"picks: channel index {pick} out of range.")
            index.append(int(pick) % n_channels)
        else:
            raise DataError(f"picks: cannot use {pick!r} to select a channel.")
    if not index:
        raise DataError("picks: the selection is empty.")
    return np.asarray(index, dtype=int)


def auto_chunk(n_trials, n_times, n_channels, max_memory_gb=None):
    """Channels per chunk under the memory rule of the specification (§9.1).

    The largest chunk such that ``chunk × n_trials × n_times × 8 bytes × 3``
    (data, model, residual) stays below ``max_memory_gb``. The default budget is
    a quarter of the physical memory when the system reports it, otherwise 2 GB.
    """

    if max_memory_gb is None:
        try:
            physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            budget = 0.25 * physical
        except (AttributeError, ValueError, OSError):
            budget = 2 * 1024 ** 3
    else:
        budget = max_memory_gb * 1024 ** 3
    per_channel = max(1, n_trials) * max(1, n_times) * 8 * 3
    return int(min(n_channels, max(1, budget // per_channel)))


def _chunks(n_channels, chunk, n_trials, n_times):
    if chunk is None:
        size = n_channels
    elif chunk == "auto":
        size = auto_chunk(n_trials, n_times, n_channels)
    elif isinstance(chunk, (int, np.integer)) and chunk >= 1:
        size = int(chunk)
    else:
        raise DataError(f"chunk: expected a positive integer, 'auto' or None, got {chunk!r}.")
    return [slice(start, min(start + size, n_channels)) for start in range(0, n_channels, size)]


def _take0(values, index):
    """values[index[...], ...] along axis 0, with index shaped like values[0]."""

    return np.take_along_axis(values, index[np.newaxis], axis=0)[0]


def _average_block(block, mask_block, robust):
    """Mean (or 20 % trimmed mean), standard error and count over axis 0."""

    block = np.array(block, dtype=float, copy=True)
    if mask_block is not None:
        block[mask_block] = np.nan
    count = np.sum(~np.isnan(block), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if not robust:
            mean = np.nansum(block, axis=0) / count
            spread = np.nanstd(block, axis=0, ddof=1)
            sem = spread / np.sqrt(count)
        else:
            ordered = np.sort(block, axis=0)                       # NaN sorted last
            trim = np.floor(TRIMMED_PROPORTION * count).astype(int)
            low, high = trim, count - trim                         # kept: ordered[low:high]
            cumulative = np.concatenate([np.zeros((1,) + block.shape[1:]),
                                         np.cumsum(np.nan_to_num(ordered), axis=0)], axis=0)
            total = _take0(cumulative, high) - _take0(cumulative, low)
            mean = total / (high - low)
            low_value = _take0(ordered, np.clip(low, 0, block.shape[0] - 1))
            high_value = _take0(ordered, np.clip(high - 1, 0, block.shape[0] - 1))
            winsorised = np.clip(block, low_value, high_value)
            spread = np.nanstd(winsorised, axis=0, ddof=1)
            sem = spread / ((1 - 2 * TRIMMED_PROPORTION) * np.sqrt(count))
    mean[count == 0] = np.nan
    sem[count < 2] = np.nan
    return mean, sem, count


###################################################################################################
## Shared behaviour

class _ContainerMixin(DeepEqualMixin):
    """copy, to_dict / from_dict, save / load and the new-object helper."""

    def copy(self):
        """Return a deep copy (arrays, tables and provenance are copied)."""

        return type(self).from_dict(self.to_dict())

    def to_dict(self):
        """Return the fields as a dictionary of copies."""

        out = {}
        for fld in fields(self):
            value = getattr(self, fld.name)
            if isinstance(value, np.ndarray):
                value = np.array(value, copy=True)
            elif _is_dataframe(value):
                value = value.copy(deep=True)
            elif isinstance(value, Provenance):
                value = value.copy()
            elif isinstance(value, list):
                value = list(value)
            out[fld.name] = value
        return out

    @classmethod
    def from_dict(cls, dct):
        """Build from :meth:`to_dict` output."""

        return cls(**dct)

    def _new(self, function, settings, start, data_changed=True, **changes):
        provenance = self.provenance.with_step(function, settings,
                                               duration_s=_time.perf_counter() - start)
        if data_changed:
            provenance.data_hash = None
        return replace(self, provenance=provenance, **changes)

    def save(self, path, overwrite=False):
        """Save to a ``.evoparam`` container (a directory, or a zip when ``path`` ends in .zip).

        The main ``data`` array is stored uncompressed in ``data.npy`` so that
        :meth:`load` can memory-map it.
        """

        from evoparam.data.serialize import save_object
        return save_object(self, path, overwrite=overwrite)

    @classmethod
    def load(cls, path, *, mmap=False, verify=True):
        """Load from a ``.evoparam`` container.

        Parameters
        ----------
        path : str or path-like
            Container directory or zip.
        mmap : bool
            Memory-map ``data`` read-only instead of reading it into memory
            (directory containers only).
        verify : bool
            Check the SHA-256 of every file against the manifest.
        """

        from evoparam.data.serialize import load_object
        obj = load_object(path, mmap=mmap, verify=verify)
        if not isinstance(obj, cls):
            raise DataError(f"{path} holds a {type(obj).__name__}, not a {cls.__name__}.")
        return obj

    @property
    def n_channels(self):
        return len(self.ch_names)

    @property
    def n_times(self):
        return int(self.times.size)

    @property
    def tmin(self):
        return float(self.times[0])

    @property
    def tmax(self):
        return float(self.times[-1])

    def convert_units(self, units):
        """Return a copy in another voltage unit ("uV", "mV" or "V")."""

        start = _time.perf_counter()
        if units not in UNITS:
            raise DataError(f"units: '{units}' is not one of {sorted(UNITS)}.")
        if units == self.units:
            return self._new(f"evoparam.data.{type(self).__name__}.convert_units",
                             {"units": units}, start, data_changed=False)
        if UNITS[units] is None or UNITS[self.units] is None:
            raise DataError(f"Cannot convert '{self.units}' to '{units}'.")
        factor = UNITS[self.units] / UNITS[units]
        changes = {"data": np.asarray(self.data) * factor, "units": units}
        if getattr(self, "sem", None) is not None:
            changes["sem"] = self.sem * factor
        return self._new(f"evoparam.data.{type(self).__name__}.convert_units",
                         {"units": units}, start, **changes)


###################################################################################################
## TrialSet

@dataclass(eq=False, repr=False)
class TrialSet(_ContainerMixin):
    """Epoched recordings: trials × channels × time.

    Parameters
    ----------
    data : ndarray, shape (n_trials, n_channels, n_times)
        Float64 samples. NaN is allowed only where ``masks`` is True.
    times : ndarray, shape (n_times,)
        Seconds, strictly increasing; 0.0 is the event onset.
    sfreq : float
        Sampling frequency (Hz); must match the time axis to 1e-6 relative.
    ch_names : list of str
        Unique, non-empty channel names.
    units : {"uV", "mV", "V", "fT", "au"}
        Amplitude unit of ``data``.
    metadata : pandas.DataFrame, optional
        One row per trial (RangeIndex). Standard columns: :data:`METADATA_COLUMNS`.
    ch_info : pandas.DataFrame, optional
        One row per channel (RangeIndex). Standard columns: :data:`CH_INFO_COLUMNS`.
    masks : ndarray of bool, shape like ``data``, optional
        True marks a replaced or invalid sample (artifact window, clipping).
    event_times : ndarray, shape (n_trials,), optional
        Absolute onset time (s) of each trial in the source recording.
    species : str
        Lower-case species name ("human", "macaque", "mouse", "rat", ...).
    modality : str, optional
        One of :data:`MODALITIES`.
    provenance : Provenance
        Origin and processing history.
    """

    data: np.ndarray
    times: np.ndarray
    sfreq: float
    ch_names: list
    units: str = "uV"
    metadata: object = None
    ch_info: object = None
    masks: np.ndarray | None = None
    event_times: np.ndarray | None = None
    species: str = "human"
    modality: str | None = None
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self):
        if not (isinstance(self.data, np.ndarray) and self.data.dtype == np.float64):
            self.data = np.asarray(self.data, dtype=np.float64)
        self.times = np.asarray(self.times, dtype=np.float64)
        self.sfreq = float(self.sfreq)
        self.ch_names = [str(name) for name in self.ch_names]
        if self.masks is not None:
            self.masks = np.asarray(self.masks, dtype=bool)
        if self.event_times is not None:
            self.event_times = np.asarray(self.event_times, dtype=np.float64)
        self.validate()
        if self.provenance.data_hash is None:
            self.provenance.data_hash = compute_data_hash(self.data, self.times, self.ch_names)

    def validate(self):
        """Check every invariant; raise :class:`~evoparam.errors.DataError` naming the field."""

        if self.data.ndim != 3:
            raise DataError(f"data: expected 3 dimensions (trials, channels, times), "
                            f"got shape {self.data.shape}.")
        n_trials, n_channels, n_times = self.data.shape
        if n_trials < 1:
            raise DataError("data: at least one trial is required.")
        _check_axis(self.times, self.sfreq, n_times)
        _check_names(self.ch_names, n_channels)
        _check_common(self)
        if self.masks is not None and self.masks.shape != self.data.shape:
            raise DataError(f"masks: shape {self.masks.shape} differs from data {self.data.shape}.")
        if self.event_times is not None and self.event_times.shape != (n_trials,):
            raise DataError(f"event_times: expected shape ({n_trials},), got {self.event_times.shape}.")
        if self.metadata is not None:
            _check_table(self.metadata, "metadata", n_trials, METADATA_COLUMNS)
        if self.ch_info is not None:
            _check_table(self.ch_info, "ch_info", n_channels, CH_INFO_COLUMNS)
            if "name" in self.ch_info.columns and list(self.ch_info["name"]) != self.ch_names:
                raise DataError("ch_info: column 'name' must equal ch_names.")
        _check_nan(self.data, self.masks)
        return self

    @property
    def n_trials(self):
        return int(self.data.shape[0])

    def __repr__(self):
        modality = f", {self.modality}" if self.modality else ""
        return (f"<TrialSet | {self.n_trials} trials × {self.n_channels} channels × "
                f"{self.n_times} times, {self.tmin:.3f} – {self.tmax:.3f} s, "
                f"{self.sfreq:g} Hz, {self.units}, {self.species}{modality}>")

    def _trial_index(self, trials, query):
        n_trials = self.n_trials
        if trials is None:
            index = np.arange(n_trials)
        elif isinstance(trials, slice):
            index = np.arange(n_trials)[trials]
        else:
            index = np.atleast_1d(np.asarray(trials))
            if index.dtype == bool:
                if index.size != n_trials:
                    raise DataError(f"trials: boolean mask of length {index.size} "
                                    f"for {n_trials} trials.")
                index = np.flatnonzero(index)
            elif index.dtype.kind not in "iu":
                raise DataError("trials: expected integers, a boolean mask or a slice.")
            elif np.any((index < -n_trials) | (index >= n_trials)):
                raise DataError("trials: index out of range.")
            index = index.astype(int) % n_trials
        if query is not None:
            if self.metadata is None:
                raise DataError("query: this TrialSet has no metadata.")
            matching = set(self.metadata.query(query).index.tolist())
            index = np.asarray([i for i in index if i in matching], dtype=int)
        if index.size == 0:
            raise DataError("The trial selection is empty.")
        return index

    def select(self, trials=None, picks=None, tmin=None, tmax=None, query=None):
        """Select trials, channels and a time window.

        Parameters
        ----------
        trials : int, sequence of int, slice or boolean mask, optional
            Trials to keep, in the given order.
        picks : str, int or sequence of str/int, or boolean mask, optional
            Channels to keep, by name or index, in the given order.
        tmin, tmax : float, optional
            Time window in seconds; ``tmin`` included, ``tmax`` excluded.
        query : str, optional
            pandas ``DataFrame.query`` expression on ``metadata``; trials that do
            not match are dropped.
        """

        start = _time.perf_counter()
        t_index = self._trial_index(trials, query)
        c_index = _pick_index(self.ch_names, picks)
        s_slice = _time_slice(self.times, self.sfreq, tmin, tmax)
        changes = {
            "data": np.asarray(self.data[t_index][:, c_index, s_slice], dtype=np.float64),
            "times": self.times[s_slice].copy(),
            "ch_names": [self.ch_names[i] for i in c_index],
            "masks": None if self.masks is None else self.masks[t_index][:, c_index, s_slice],
            "event_times": None if self.event_times is None else self.event_times[t_index],
            "metadata": None if self.metadata is None
            else self.metadata.iloc[t_index].reset_index(drop=True),
            "ch_info": None if self.ch_info is None
            else self.ch_info.iloc[c_index].reset_index(drop=True),
        }
        settings = {"trials": t_index.tolist(), "picks": changes["ch_names"],
                    "tmin": tmin, "tmax": tmax, "query": query}
        return self._new("evoparam.data.TrialSet.select", settings, start, **changes)

    def crop(self, tmin=None, tmax=None):
        """Keep samples with ``tmin <= t < tmax`` (seconds)."""

        start = _time.perf_counter()
        s_slice = _time_slice(self.times, self.sfreq, tmin, tmax)
        changes = {"data": np.asarray(self.data[:, :, s_slice], dtype=np.float64),
                   "times": self.times[s_slice].copy(),
                   "masks": None if self.masks is None else self.masks[:, :, s_slice]}
        return self._new("evoparam.data.TrialSet.crop", {"tmin": tmin, "tmax": tmax}, start,
                         **changes)

    @classmethod
    def from_mne(cls, epochs, *, picks=None, units="uV", metadata=True, stim_columns="auto"):
        """TrialSet from ``mne.Epochs``: V -> ``units``, channel types, bad channels,
        positions (mm), metadata with the event names in ``condition``, event times
        (spec §10.1, extra ``mne``)."""

        from ._mne import trialset_from_mne

        return trialset_from_mne(cls, epochs, picks=picks, units=units, metadata=metadata,
                                 stim_columns=stim_columns)

    def to_mne(self, *, ch_types=None):
        """``mne.EpochsArray`` holding the same data, channels, metadata and events."""

        from ._mne import trialset_to_mne

        return trialset_to_mne(self, ch_types=ch_types)

    def to_xarray(self):
        """``xarray.DataArray`` of the data, dims ``("trial", "channel", "time")`` (extra
        ``xarray``).

        Coordinates: ``trial`` (0..n-1), ``channel`` (names), ``time`` (s); every metadata
        column of plain values is a coordinate along ``trial`` and every ``ch_info``
        column along ``channel``; ``masks`` (when set) is the boolean coordinate ``mask``
        on all three dims. Attributes: ``sfreq``, ``units``, ``species``, ``modality``.
        """

        try:
            import xarray as xr
        except ImportError as err:
            raise ImportError("TrialSet.to_xarray needs the extra 'xarray': pip install "
                              "'evoparam[xarray]'.") from err
        coords = {"trial": np.arange(self.n_trials), "channel": list(self.ch_names),
                  "time": np.asarray(self.times, dtype=float)}
        for table, dim in ((self.metadata, "trial"), (self.ch_info, "channel")):
            if table is None:
                continue
            for column in table.columns:
                if column in coords or column in ("trial", "channel", "time", "mask"):
                    continue
                values = table[column].to_numpy()
                if values.dtype == object and not all(
                        isinstance(v, (str, int, float, bool, np.generic)) or v is None
                        for v in values):
                    continue
                coords[str(column)] = (dim, values)
        if self.masks is not None:
            coords["mask"] = (("trial", "channel", "time"), np.asarray(self.masks, dtype=bool))
        attrs = {"sfreq": float(self.sfreq), "units": self.units, "species": self.species,
                 "modality": self.modality if self.modality is not None else ""}
        return xr.DataArray(np.asarray(self.data), dims=("trial", "channel", "time"),
                            coords=coords, attrs=attrs, name="data")

    def average(self, by=None, robust=False, chunk=None):
        """Average over trials.

        Trials whose ``metadata["rejected"]`` is True are left out. Masked
        samples are left out sample by sample; a sample with no valid trial is
        NaN and masked in the result.

        Parameters
        ----------
        by : str or list of str, optional
            Metadata column(s) to group by. With ``by``, one :class:`Evoked` per
            group is returned, in sorted group order, each with a one-row
            ``metadata`` holding its group values.
        robust : bool
            Use the 20 % trimmed mean instead of the mean; its standard error is
            the winsorised standard deviation divided by (1 − 2 × 0.2) √n.
        chunk : int, "auto" or None
            Channels processed at once (results do not depend on it).

        Returns
        -------
        Evoked or list of Evoked
        """

        start = _time.perf_counter()
        keep = np.arange(self.n_trials)
        n_rejected = 0
        if self.metadata is not None and "rejected" in self.metadata.columns:
            rejected = self.metadata["rejected"].to_numpy(dtype=bool)
            keep = np.flatnonzero(~rejected)
            n_rejected = int(rejected.sum())
            if keep.size == 0:
                raise DataError("average: every trial is marked rejected.")
        settings = {"by": by, "robust": bool(robust), "n_rejected": n_rejected}
        if by is None:
            return self._average_trials(keep, None, settings, start, chunk)
        if self.metadata is None:
            raise DataError("average: 'by' needs metadata.")
        columns = [by] if isinstance(by, str) else list(by)
        missing = [column for column in columns if column not in self.metadata.columns]
        if missing:
            raise DataError(f"average: unknown metadata column(s) {missing}.")
        subset = self.metadata.iloc[keep]
        out = []
        for key, group in subset.groupby(columns, sort=True, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            row = {column: [value] for column, value in zip(columns, key)}
            out.append(self._average_trials(group.index.to_numpy(), row, settings, start, chunk))
        return out

    def _average_trials(self, index, group_row, settings, start, chunk):
        n_times = self.n_times
        mean = np.empty((self.n_channels, n_times))
        sem = np.empty((self.n_channels, n_times))
        count = np.empty((self.n_channels, n_times), dtype=int)
        for part in _chunks(self.n_channels, chunk, index.size, n_times):
            block = self.data[index][:, part] if index.size != self.n_trials else self.data[:, part]
            mask_block = None
            if self.masks is not None:
                mask_block = self.masks[index][:, part]
            mean[part], sem[part], count[part] = _average_block(block, mask_block, settings["robust"])
        masks = count == 0
        metadata = None
        if group_row is not None:
            import pandas as pd
            metadata = pd.DataFrame(group_row)
        step_settings = dict(settings, group=None if group_row is None
                             else {k: v[0] for k, v in group_row.items()},
                             n_trials=int(index.size))
        provenance = self.provenance.with_step("evoparam.data.TrialSet.average", step_settings,
                                               duration_s=_time.perf_counter() - start)
        provenance.data_hash = None
        return Evoked(data=mean, times=self.times.copy(), sfreq=self.sfreq,
                      ch_names=list(self.ch_names), units=self.units, n_trials=int(index.size),
                      sem=sem, masks=masks if masks.any() else None, metadata=metadata,
                      ch_info=None if self.ch_info is None else self.ch_info.copy(deep=True),
                      species=self.species, modality=self.modality, provenance=provenance)

    def apply(self, func, axis="trials", chunk=None):
        """Apply a vectorised transform and record it in the provenance.

        Parameters
        ----------
        func : callable
            With ``axis="trials"``, receives an array (n_trials, n_chunk_channels,
            n_times) and returns an array of the same shape. With
            ``axis="times"``, receives a 2-D array (n_trials × n_chunk_channels,
            n_times) and returns the same shape (for example a filter).
        axis : {"trials", "times"}
        chunk : int, "auto" or None
            Channels passed to ``func`` at once.
        """

        start = _time.perf_counter()
        if axis not in ("trials", "times"):
            raise DataError(f"apply: axis must be 'trials' or 'times', got {axis!r}.")
        out = np.empty(self.data.shape)
        for part in _chunks(self.n_channels, chunk, self.n_trials, self.n_times):
            block = np.array(self.data[:, part], dtype=np.float64)
            if axis == "trials":
                result = np.asarray(func(block), dtype=np.float64)
            else:
                flat = block.reshape(-1, self.n_times)
                result = np.asarray(func(flat), dtype=np.float64)
                if result.shape != flat.shape:
                    raise DataError(f"apply: func returned shape {result.shape}, "
                                    f"expected {flat.shape}.")
                result = result.reshape(block.shape)
            if result.shape != block.shape:
                raise DataError(f"apply: func returned shape {result.shape}, expected {block.shape}.")
            out[:, part] = result
        name = getattr(func, "__qualname__", repr(func))
        module = getattr(func, "__module__", None)
        settings = {"func": f"{module}.{name}" if module else name, "axis": axis}
        return self._new("evoparam.data.TrialSet.apply", settings, start, data=out)

    def as_trialset(self):
        """Return self (for functions that accept both containers)."""

        return self

    def as_evoked(self):
        """One-trial TrialSet → Evoked without SEM; several trials → :meth:`average`."""

        if self.n_trials > 1:
            return self.average()
        start = _time.perf_counter()
        provenance = self.provenance.with_step("evoparam.data.TrialSet.as_evoked", {},
                                               duration_s=_time.perf_counter() - start)
        return Evoked(data=np.array(self.data[0]), times=self.times.copy(), sfreq=self.sfreq,
                      ch_names=list(self.ch_names), units=self.units, n_trials=1, sem=None,
                      masks=None if self.masks is None else self.masks[0].copy(),
                      metadata=None, ch_info=None if self.ch_info is None
                      else self.ch_info.copy(deep=True),
                      species=self.species, modality=self.modality, provenance=provenance)


###################################################################################################
## Evoked

@dataclass(eq=False, repr=False)
class Evoked(_ContainerMixin):
    """Averaged response: channels × time.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    times, sfreq, ch_names, units, ch_info, species, modality, provenance
        As in :class:`TrialSet`.
    n_trials : int or ndarray of int, shape (n_channels,)
        Trials in the average (scalar, or per channel after per-channel rejection).
    sem : ndarray, shape (n_channels, n_times), optional
        Standard error of the mean, same units as ``data``.
    masks : ndarray of bool, shape (n_channels, n_times), optional
        True marks an invalid sample.
    metadata : pandas.DataFrame, optional
        At most one row: the grouping values used by :meth:`TrialSet.average`.
    """

    data: np.ndarray
    times: np.ndarray
    sfreq: float
    ch_names: list
    units: str = "uV"
    n_trials: object = 1
    sem: np.ndarray | None = None
    masks: np.ndarray | None = None
    metadata: object = None
    ch_info: object = None
    species: str = "human"
    modality: str | None = None
    provenance: Provenance = field(default_factory=Provenance)


    @classmethod
    def from_mne(cls, evoked, *, picks=None, units="uV"):
        """Evoked from ``mne.Evoked`` (spec §10.1, extra ``mne``)."""

        from ._mne import evoked_from_mne

        return evoked_from_mne(cls, evoked, picks=picks, units=units)

    def to_mne(self, *, ch_types=None, comment=""):
        """``mne.EvokedArray`` holding the same data and channels."""

        from ._mne import evoked_to_mne

        return evoked_to_mne(self, ch_types=ch_types, comment=comment)
    def __post_init__(self):
        if not (isinstance(self.data, np.ndarray) and self.data.dtype == np.float64):
            self.data = np.asarray(self.data, dtype=np.float64)
        self.times = np.asarray(self.times, dtype=np.float64)
        self.sfreq = float(self.sfreq)
        self.ch_names = [str(name) for name in self.ch_names]
        if self.sem is not None:
            self.sem = np.asarray(self.sem, dtype=np.float64)
        if self.masks is not None:
            self.masks = np.asarray(self.masks, dtype=bool)
        if isinstance(self.n_trials, (np.ndarray, list, tuple)):
            self.n_trials = np.asarray(self.n_trials, dtype=np.int64)
        elif isinstance(self.n_trials, (int, np.integer)) and not isinstance(self.n_trials, bool):
            self.n_trials = int(self.n_trials)
        self.validate()
        if self.provenance.data_hash is None:
            self.provenance.data_hash = compute_data_hash(self.data, self.times, self.ch_names)

    def validate(self):
        """Check every invariant; raise :class:`~evoparam.errors.DataError` naming the field."""

        if self.data.ndim != 2:
            raise DataError(f"data: expected 2 dimensions (channels, times), got shape "
                            f"{self.data.shape}.")
        n_channels, n_times = self.data.shape
        _check_axis(self.times, self.sfreq, n_times)
        _check_names(self.ch_names, n_channels)
        _check_common(self)
        if isinstance(self.n_trials, np.ndarray):
            if self.n_trials.shape != (n_channels,) or np.any(self.n_trials < 1):
                raise DataError("n_trials: per-channel counts must have shape (n_channels,) "
                                "and be >= 1.")
        elif not isinstance(self.n_trials, int) or self.n_trials < 1:
            raise DataError(f"n_trials: must be a positive integer, got {self.n_trials!r}.")
        if self.sem is not None and self.sem.shape != self.data.shape:
            raise DataError(f"sem: shape {self.sem.shape} differs from data {self.data.shape}.")
        if self.masks is not None and self.masks.shape != self.data.shape:
            raise DataError(f"masks: shape {self.masks.shape} differs from data {self.data.shape}.")
        if self.metadata is not None:
            if not _is_dataframe(self.metadata) or len(self.metadata) > 1:
                raise DataError("metadata: must be a DataFrame with at most one row.")
        if self.ch_info is not None:
            _check_table(self.ch_info, "ch_info", n_channels, CH_INFO_COLUMNS)
            if "name" in self.ch_info.columns and list(self.ch_info["name"]) != self.ch_names:
                raise DataError("ch_info: column 'name' must equal ch_names.")
        _check_nan(self.data[np.newaxis],
                   None if self.masks is None else self.masks[np.newaxis])
        return self

    def __repr__(self):
        modality = f", {self.modality}" if self.modality else ""
        n_trials = self.n_trials if isinstance(self.n_trials, int) else \
            f"{int(self.n_trials.min())}–{int(self.n_trials.max())}"
        return (f"<Evoked | {self.n_channels} channels × {self.n_times} times, "
                f"{self.tmin:.3f} – {self.tmax:.3f} s, {self.sfreq:g} Hz, {self.units}, "
                f"n_trials={n_trials}, {self.species}{modality}>")

    def select(self, picks=None, tmin=None, tmax=None):
        """Select channels (by name or index) and a time window (``tmin`` ≤ t < ``tmax``)."""

        start = _time.perf_counter()
        c_index = _pick_index(self.ch_names, picks)
        s_slice = _time_slice(self.times, self.sfreq, tmin, tmax)
        changes = {
            "data": self.data[c_index, s_slice].copy(),
            "times": self.times[s_slice].copy(),
            "ch_names": [self.ch_names[i] for i in c_index],
            "n_trials": self.n_trials[c_index] if isinstance(self.n_trials, np.ndarray)
            else self.n_trials,
            "sem": None if self.sem is None else self.sem[c_index, s_slice].copy(),
            "masks": None if self.masks is None else self.masks[c_index, s_slice].copy(),
            "ch_info": None if self.ch_info is None
            else self.ch_info.iloc[c_index].reset_index(drop=True),
        }
        settings = {"picks": changes["ch_names"], "tmin": tmin, "tmax": tmax}
        return self._new("evoparam.data.Evoked.select", settings, start, **changes)

    def crop(self, tmin=None, tmax=None):
        """Keep samples with ``tmin <= t < tmax`` (seconds)."""

        return self.select(tmin=tmin, tmax=tmax)

    def as_evoked(self):
        """Return self (for functions that accept both containers)."""

        return self

    def as_trialset(self):
        """Return a one-trial :class:`TrialSet` (the SEM is not carried over)."""

        start = _time.perf_counter()
        provenance = self.provenance.with_step("evoparam.data.Evoked.as_trialset", {},
                                               duration_s=_time.perf_counter() - start)
        return TrialSet(data=self.data[np.newaxis].copy(), times=self.times.copy(),
                        sfreq=self.sfreq, ch_names=list(self.ch_names), units=self.units,
                        masks=None if self.masks is None else self.masks[np.newaxis].copy(),
                        ch_info=None if self.ch_info is None else self.ch_info.copy(deep=True),
                        species=self.species, modality=self.modality, provenance=provenance)
