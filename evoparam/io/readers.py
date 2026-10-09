"""Build a TrialSet from arrays, files and objects of other packages.

:func:`from_array` and :func:`read` need numpy only (pandas when tables are
given); so do the BrainVision and MEF 3.0 readers. :func:`read_mne` needs the extra ``mne`` and :func:`read_edf` the extra
``edf``; without them they raise an ImportError that names the extra.
"""

import os
import time as _time
from dataclasses import replace

import numpy as np

import evoparam._plugins as _plugins
from evoparam.data import Evoked, Provenance, TrialSet, load_object
from evoparam.errors import DataError, ReaderError, SettingsError, warn as _warn
from evoparam.io._fetch import is_url, sha256_file
from evoparam.utils._optional import require

__all__ = ["from_array", "read", "read_mne", "read_edf", "read_events_edf", "READERS"]

# formats of read(format=...) and their extensions; a plugin reader may use neither
BUILTIN_FORMATS = ("array", "mne", "brainvision", "mef", "edf", "bids", "evoparam", "nwb",
                   "neo")
BUILTIN_EXTENSIONS = (".vhdr", ".edf", ".bdf", ".nwb", ".mefd", ".evoparam", ".zip")


def _plugin_reader(name, obj, origin):
    """Check a reader of the entry-point group ``evoparam.readers`` (spec §10.8):
    ``callable(path, **kw) -> TrialSet`` with ``extensions = (".ext", ...)``."""

    import inspect

    if not callable(obj):
        raise SettingsError(f"plugin reader {name!r} of {origin} is not callable.")
    extensions = getattr(obj, "extensions", None)
    if isinstance(extensions, str) or not isinstance(extensions, (tuple, list)) or not \
            extensions or not all(isinstance(e, str) and e.startswith(".") for e in extensions):
        raise SettingsError(f"plugin reader {name!r} of {origin} must declare extensions = "
                            f"('.ext', ...).")
    taken = sorted({e.lower() for e in extensions} & set(BUILTIN_EXTENSIONS))
    if taken:
        raise SettingsError(f"plugin reader {name!r} of {origin} claims the built-in "
                            f"extension(s) {taken}.")
    try:
        params = list(inspect.signature(obj).parameters.values())
    except (TypeError, ValueError):
        params = None
    if params is not None and not (params and params[0].kind in (
            params[0].POSITIONAL_ONLY, params[0].POSITIONAL_OR_KEYWORD, params[0].VAR_POSITIONAL)):
        raise SettingsError(f"plugin reader {name!r} of {origin} must take the path as its "
                            f"first positional argument.")
    return obj


# plugin readers by name, loaded on the first call of read (no built-in entries: the
# built-in formats are dispatched by read itself)
READERS = _plugins.LazyRegistry("evoparam.readers", {}, _plugin_reader,
                                builtin=BUILTIN_FORMATS)


def _plugin_for(path, format):
    """(name, reader) of the plugin reader for ``format`` or the extension of ``path``;
    None when no plugin claims it. Two plugins claiming one extension raise SettingsError."""

    if format is not None:
        return (format, READERS[format]) if format in READERS else None
    lower = path.lower().rstrip("/")
    claims = [(name, reader) for name, reader in READERS.items()
              if any(lower.endswith(e.lower()) for e in reader.extensions)]
    if len(claims) > 1:
        origins = READERS.plugins()
        raise SettingsError("plugin readers " + " and ".join(
            f"{name!r} ({origins[name]})" for name, _ in claims) + f" both claim {path!r}.")
    if claims:
        name = claims[0][0]
        return name, READERS[name]          # marks the plugin as used
    return None


def _read_plugin(name, reader, path, kwargs):
    out = reader(path, **kwargs)
    if not isinstance(out, TrialSet):
        raise ReaderError(f"plugin reader {name!r} returned a {type(out).__name__}, not a "
                          f"TrialSet.")
    origin = READERS.plugins()[name]
    source = dict(out.provenance.source)
    source.setdefault("format", name)
    source["plugin"] = {name: origin}
    provenance = Provenance(source=source, steps=list(out.provenance.steps)).with_step(
        "evoparam.io.read", {"format": name, "plugin": origin})
    environment = dict(provenance.environment, plugins={name: origin})
    return replace(out, provenance=replace(provenance, environment=environment))

# MNE channel types measured in volts, and their type in the channel table (Table 4.2).
_MNE_VOLTS = {"eeg": "eeg", "ecog": "ecog", "seeg": "seeg", "dbs": "dbs", "emg": "emg",
              "eog": "misc", "ecg": "misc"}
_MNE_HEAD_FRAME = 4                       # FIFFV_COORD_HEAD

# Physical dimensions of EDF signals -> factor to µV.
_EDF_TO_UV = {"uv": 1.0, "µv": 1.0, "μv": 1.0, "mv": 1e3, "v": 1e6, "nv": 1e-3}


def _default_names(n_channels):
    width = max(3, len(str(n_channels)))
    return ["CH{:0{}d}".format(ind + 1, width) for ind in range(n_channels)]


def _table(value, name):
    if value is None:
        return None
    pd = require("pandas", extra="data", caller=f"evoparam.io.from_array ({name})")
    if isinstance(value, dict):
        value = pd.DataFrame(value)
    if not isinstance(value, pd.DataFrame):
        raise DataError(f"{name}: expected a pandas DataFrame or a dict of columns.")
    return value.reset_index(drop=True)


def from_array(data, sfreq, *, times=None, ch_names=None, units="uV", metadata=None,
               ch_info=None, species="human", modality=None):
    """Build a TrialSet from an array.

    Parameters
    ----------
    data : ndarray
        2-D (n_channels, n_times): one trial; 3-D (n_trials, n_channels, n_times).
    sfreq : float
        Sampling frequency, in Hz.
    times : ndarray, optional
        Seconds relative to the event, one per sample; default ``arange(n_times) / sfreq``.
    ch_names : list of str, optional
        Channel names; default ``CH001``, ``CH002``, ...
    units : str
        Amplitude unit of ``data`` ("uV", "mV", "V", "fT" or "au").
    metadata : pandas.DataFrame or dict, optional
        One row per trial (pandas needed).
    ch_info : pandas.DataFrame or dict, optional
        One row per channel (pandas needed).
    species : str
        Lower-case species name.
    modality : str, optional
        One of the modalities of the data model ("spes", "erp", ...).

    Returns
    -------
    trials : TrialSet

    Raises
    ------
    DataError
        If the shapes of ``data``, ``times``, ``ch_names`` or the tables do not match,
        or any other invariant of TrialSet fails.
    """

    start = _time.perf_counter()
    data = np.asarray(data)
    if data.dtype.kind not in "biuf":
        raise DataError(f"data: expected a numeric array, got dtype {data.dtype}.")
    if data.ndim == 2:
        data = data[np.newaxis]
    elif data.ndim != 3:
        raise DataError(f"data: expected 2 dimensions (channels, times) or 3 (trials, channels, "
                        f"times), got shape {data.shape}.")
    n_trials, n_channels, n_times = data.shape
    sfreq = float(sfreq)
    if not (np.isfinite(sfreq) and sfreq > 0):
        raise DataError(f"sfreq: must be a positive number, got {sfreq}.")
    times = np.arange(n_times) / sfreq if times is None else np.asarray(times, dtype=float)
    if times.shape != (n_times,):
        raise DataError(f"times: {times.size} values for {n_times} samples.")
    ch_names = _default_names(n_channels) if ch_names is None else [str(n) for n in ch_names]
    if len(ch_names) != n_channels:
        raise DataError(f"ch_names: {len(ch_names)} names for {n_channels} channels.")
    settings = {"sfreq": sfreq, "shape": [n_trials, n_channels, n_times], "units": units}
    provenance = Provenance(source={"path": None, "format": "array",
                                    "reader": "evoparam.io.from_array"})
    provenance = provenance.with_step("evoparam.io.from_array", settings,
                                      duration_s=_time.perf_counter() - start)
    return TrialSet(data=np.asarray(data, dtype=np.float64), times=times, sfreq=sfreq,
                    ch_names=ch_names, units=units, metadata=_table(metadata, "metadata"),
                    ch_info=_table(ch_info, "ch_info"), species=species, modality=modality,
                    provenance=provenance)


###################################################################################################
## Dispatch

def _guess_format(path):
    lower = path.lower().rstrip("/")
    if lower.endswith(".vhdr"):
        return "brainvision"
    if lower.endswith((".edf", ".bdf")):
        return "edf"
    if lower.endswith(".nwb"):
        return "nwb"
    if lower.endswith(".mefd"):
        return "mef"
    if lower.endswith((".evoparam", ".evoparam.zip")):
        return "evoparam"
    if not is_url(path) and os.path.isdir(path):
        if os.path.isfile(os.path.join(path, "manifest.json")):
            return "evoparam"
        if os.path.isfile(os.path.join(path, "dataset_description.json")) or \
                any(name.startswith("sub-") for name in os.listdir(path)):
            return "bids"
    if lower.endswith(".zip"):
        return "evoparam"
    raise ReaderError(f"Cannot tell the format of {path!r}; give format=.")


def _read_continuous(path, fmt, *, onsets=None, epoch=(-0.5, 1.0), tmin=None, tmax=None,
                     picks=None, remote=None, metadata=None, ch_info=None, species="human",
                     modality=None):
    """A recording file: epochs around ``onsets``, or the recording as one trial."""

    from evoparam.io.brainvision import _read_with_source
    from evoparam.io.epochs import epoch as cut, epoch_windows

    start = _time.perf_counter()
    if onsets is not None:
        onsets = np.atleast_1d(np.asarray(onsets, dtype=float))
        # Read only the span that holds the epochs (one sample of margin on each side)
        tmin = max(float(onsets.min()) + epoch[0] - 1.0, 0.0)
        tmax = float(onsets.max()) + epoch[1] + 1.0
    if fmt == "brainvision":
        data, sfreq, names, t0, source = _read_with_source(path, tmin, tmax, picks, remote)
    elif fmt == "mef":
        from evoparam.io.mef import _read_with_source as _read_mef
        data, sfreq, names, t0, source = _read_mef(path, tmin, tmax, picks, remote)
    else:
        data, sfreq, names, t0 = read_edf(path, tmin=tmin, tmax=tmax, picks=picks)
        source = {"path": path, "format": "edf", "sha256": sha256_file(path)}
    source["reader"] = "evoparam.io.read"
    # Samples without data (gaps of a MEF recording) are NaN and masked
    gaps = np.isnan(data) if fmt == "mef" else np.zeros(data.shape, dtype=bool)
    if onsets is not None:
        trials = cut(np.where(gaps, 0.0, data) if gaps.any() else data, sfreq, onsets, epoch[0],
                     epoch[1], t0=t0, ch_names=names, metadata=metadata, ch_info=ch_info)
        step = trials.provenance.steps[-1]
        provenance = Provenance(source=source).with_step(
            "evoparam.io.read", {"format": fmt, "epoch": list(epoch), "picks": picks},
            duration_s=_time.perf_counter() - start, kept=step["kept"],
            overlaps_previous=step["overlaps_previous"])
        changes = {}
        if gaps.any():
            times, starts, kept = epoch_windows(onsets, sfreq, epoch[0], epoch[1],
                                                data.shape[1], t0=t0)
            index = starts[kept][:, None] + np.arange(times.size)[None, :]
            masks = np.moveaxis(gaps[:, index], 0, 1)
            changes = {"data": np.where(masks, np.nan, trials.data), "masks": masks}
        return replace(trials, provenance=provenance, species=species, modality=modality,
                       **changes)
    provenance = Provenance(source=source).with_step(
        "evoparam.io.read", {"format": fmt, "tmin": tmin, "tmax": tmax, "picks": picks},
        duration_s=_time.perf_counter() - start)
    return TrialSet(data=data[np.newaxis], times=t0 + np.arange(data.shape[1]) / sfreq,
                    sfreq=sfreq, ch_names=names, units="uV", metadata=_table(metadata, "metadata"),
                    ch_info=_table(ch_info, "ch_info"),
                    masks=gaps[np.newaxis] if gaps.any() else None, species=species,
                    modality=modality, provenance=provenance)


def read(source, *, format=None, **kwargs):
    """Read data from any supported source into a TrialSet.

    Dispatch on the type or the extension of ``source``:

    - ``numpy.ndarray``: :func:`from_array` (``sfreq`` required);
    - MNE ``Epochs`` / ``Evoked``: :func:`read_mne` (extra ``mne``);
    - ``TrialSet`` (returned unchanged) or ``Evoked`` (one trial);
    - ``.vhdr`` (local or http(s)), ``.mefd`` (MEF 3.0 session folder, local or http(s);
      see :func:`read_mef`) and ``.edf`` / ``.bdf`` (extra ``edf``): with ``onsets`` (s)
      and ``epoch=(tmin, tmax)``, epochs around the onsets (see :func:`epoch`); otherwise
      the recording, or ``tmin`` … ``tmax`` of it, as one trial whose ``times`` are
      seconds from the start of the recording. Samples without data (gaps of a MEF
      recording) are NaN and marked in ``masks``;
    - a BIDS dataset folder: :func:`read_bids` (``subject`` required);
    - a ``.evoparam`` container (folder or zip): the saved TrialSet;
    - ``.nwb``: :func:`read_nwb` (extra ``nwb``); a Neo ``Block`` or a file with
      ``format="neo"``: :func:`read_neo` (extra ``neo``);
    - a file whose extension a plugin reader declares (entry-point group
      ``evoparam.readers``, spec §10.8), or ``format=<plugin name>``.

    Parameters
    ----------
    source : ndarray, object, str or path-like
    format : str, optional
        Force the format: "array", "mne", "brainvision", "mef", "edf", "bids",
        "evoparam", "nwb", "neo" or the name of a plugin reader.
    **kwargs
        Passed to the reader.

    Returns
    -------
    trials : TrialSet

    Raises
    ------
    ReaderError
        Unknown or unsupported format.
    """

    if isinstance(source, TrialSet):
        return source
    if isinstance(source, Evoked):
        return source.as_trialset()
    if format == "array" or isinstance(source, np.ndarray):
        return from_array(source, **kwargs)
    module = type(source).__module__ or ""
    if format == "mne" or module.split(".")[0] == "mne":
        return read_mne(source, **kwargs)
    if format == "neo" or module.split(".")[0] == "neo":
        from evoparam.io.neo import read_neo
        return read_neo(source, **kwargs)
    if not isinstance(source, (str, os.PathLike)):
        raise ReaderError(f"Cannot read an object of type {type(source).__name__}.")
    path = os.fspath(source)
    if format is None or format not in BUILTIN_FORMATS:
        plugin = _plugin_for(path, format)
        if plugin is not None:
            return _read_plugin(*plugin, path, kwargs)
        if format is not None:
            raise ReaderError(f"Unknown format {format!r}; built-in: {BUILTIN_FORMATS}, "
                              f"plugins: {sorted(READERS)}.")
    fmt = format or _guess_format(path)
    if fmt in ("brainvision", "edf", "mef"):
        return _read_continuous(path, fmt, **kwargs)
    if fmt == "bids":
        from evoparam.io.bids import read_bids
        return read_bids(path, **kwargs)
    if fmt == "evoparam":
        obj = load_object(path, **kwargs)
        if isinstance(obj, Evoked):
            return obj.as_trialset()
        if not isinstance(obj, TrialSet):
            raise ReaderError(f"{path} holds a {type(obj).__name__}, not a TrialSet.")
        return obj
    if fmt == "nwb":
        from evoparam.io.nwb import read_nwb
        return read_nwb(path, **kwargs)
    raise ReaderError(f"Unknown format {fmt!r}.")


###################################################################################################
## MNE-Python

def read_mne(obj, *, units="uV"):
    """Convert an MNE-Python ``Epochs`` or ``Evoked`` object to a TrialSet (extra ``mne``).

    Voltages (EEG, ECoG, sEEG, DBS, EMG) are converted from V to ``units``; MEG
    magnetometers from T to fT (``units="fT"``). ``obj.metadata`` is kept. The channel
    table holds the type, the bad flag and the coordinates in mm when the channel
    positions are in the head frame.

    Parameters
    ----------
    obj : mne.Epochs, mne.EpochsArray or mne.Evoked
    units : str
        Amplitude unit of the result.

    Returns
    -------
    trials : TrialSet
        An ``Evoked`` gives one trial.

    Raises
    ------
    ImportError
        If mne (or pandas) is not installed; the message names the extra.
    DataError
        If the channels mix units (for example EEG with MEG, or gradiometers).
    """

    mne = require("mne", extra="mne", min_version="1.6", caller="evoparam.io.read_mne")
    pd = require("pandas", extra="data", caller="evoparam.io.read_mne")
    start = _time.perf_counter()
    if isinstance(obj, mne.Evoked):
        data, metadata, event_times = obj.data[np.newaxis], None, None
    elif isinstance(obj, mne.BaseEpochs):
        data = obj.get_data(copy=True)
        metadata = None if obj.metadata is None else obj.metadata.reset_index(drop=True)
        event_times = obj.events[:, 0] / obj.info["sfreq"]
    else:
        raise DataError(f"read_mne: expected mne.Epochs or mne.Evoked, got {type(obj).__name__}.")

    types = obj.get_channel_types()
    names = list(obj.ch_names)
    if all(kind in _MNE_VOLTS for kind in types):
        factor = {"uV": 1e6, "mV": 1e3, "V": 1.0}.get(units)
        if factor is None:
            raise DataError(f"read_mne: voltage channels cannot be given in {units!r}.")
    elif all(kind == "mag" for kind in types):
        if units != "fT":
            raise DataError("read_mne: magnetometers are converted to fT; use units='fT'.")
        factor = 1e15
    else:
        raise DataError(f"read_mne: channel types {sorted(set(types))} do not share one unit; "
                        "pick channels of one kind first (obj.copy().pick(...)).")

    coords = np.array([ch["loc"][:3] for ch in obj.info["chs"]], dtype=float) * 1e3
    frames = {ch.get("coord_frame") for ch in obj.info["chs"]}
    head = frames == {_MNE_HEAD_FRAME}
    ch_info = pd.DataFrame({
        "name": names,
        "type": [_MNE_VOLTS.get(kind, "meg" if kind in ("mag", "grad") else "misc")
                 for kind in types],
        "x": coords[:, 0] if head else np.nan, "y": coords[:, 1] if head else np.nan,
        "z": coords[:, 2] if head else np.nan, "coord_space": "head" if head else "",
        "bad": [name in obj.info["bads"] for name in names],
    })
    source = {"path": None, "format": "mne", "reader": "evoparam.io.read_mne",
              "object": type(obj).__name__}
    provenance = Provenance(source=source).with_step(
        "evoparam.io.read_mne", {"units": units}, duration_s=_time.perf_counter() - start)
    return TrialSet(data=np.asarray(data, dtype=np.float64) * factor, times=np.array(obj.times),
                    sfreq=obj.info["sfreq"], ch_names=names, units=units, metadata=metadata,
                    ch_info=ch_info, event_times=event_times, provenance=provenance)


###################################################################################################
## EDF

def read_edf(path, *, tmin=None, tmax=None, picks=None):
    """Read an EDF, EDF+ or BDF recording, or a time window of it (extra ``edf``).

    Parameters
    ----------
    path : str
        Path of the file.
    tmin, tmax : float, optional
        Window in seconds from the start of the recording; ``tmin`` included,
        ``tmax`` excluded.
    picks : list of str or int, optional
        Channels by name or index. The picked signals must share one sampling frequency.

    Returns
    -------
    data : 2d array
        Signal values in µV (voltage signals; others unconverted), [n_channels, n_times].
    sfreq : float
        Sampling frequency, in Hz.
    ch_names : list of str
    t0 : float
        Time of the first returned sample, in seconds from the start of the recording.

    Raises
    ------
    ImportError
        If edfio is not installed; the message names the extra.
    DataError
        If the picked signals have different sampling frequencies.
    """

    from evoparam.io.brainvision import _pick, _sample

    edfio = require("edfio", extra="edf", min_version="0.4", caller="evoparam.io.read_edf")
    edf = edfio.read_edf(path)
    signals = list(edf.signals)
    names = [signal.label for signal in signals]
    index = _pick(names, picks)
    rates = {float(signals[ind].sampling_frequency) for ind in index}
    if len(rates) != 1:
        raise DataError(f"read_edf: the picked signals have different sampling frequencies "
                        f"{sorted(rates)}; pick signals of one frequency.")
    sfreq = rates.pop()
    first = 0 if tmin is None else max(_sample(tmin, sfreq), 0)
    rows, other = [], []
    for ind in index:
        signal = signals[ind]
        stop = len(signal.data) if tmax is None else min(_sample(tmax, sfreq), len(signal.data))
        factor = _EDF_TO_UV.get(str(signal.physical_dimension).strip().lower())
        if factor is None:
            factor = 1.0
            other.append(signal.label)
        rows.append(np.asarray(signal.data[first:stop], dtype=np.float64) * factor)
    if other:
        _warn("Signals whose unit is not a voltage were left unconverted: "
              + ", ".join(other) + ".", UserWarning, stacklevel=2)
    if not rows or rows[0].size == 0:
        raise ReaderError(f"{path}: the requested time window is outside the recording.")
    return np.vstack(rows), sfreq, [names[ind] for ind in index], first / sfreq


def read_events_edf(path):
    """Annotations of an EDF+ file (extra ``edf``).

    Returns
    -------
    events : list of dict
        One dict per annotation with keys 'onset' (s), 'duration' (s, None when not
        given) and 'description'.
    """

    edfio = require("edfio", extra="edf", min_version="0.4", caller="evoparam.io.read_events_edf")
    edf = edfio.read_edf(path)
    return [{"onset": float(item.onset),
             "duration": None if item.duration is None else float(item.duration),
             "description": item.text} for item in edf.annotations]
