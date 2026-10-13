# SPDX-License-Identifier: Apache-2.0
"""Cut a continuous recording into epochs around events."""

import time as _time

import numpy as np

from evoparam._typing import Any, ArrayLike, NDArray
from evoparam.data import Provenance, TrialSet
from evoparam.errors import DataError, EventError
from evoparam.utils._optional import require

__all__ = ["epoch", "epoch_data"]

###################################################################################################
###################################################################################################


def epoch_data(
    data: ArrayLike,
    sfreq: float,
    onsets: ArrayLike,
    tmin: float,
    tmax: float,
    t0: float | None = 0.0,
) -> tuple[Any, NDArray[Any], Any]:
    """Cut epochs from continuous data, around event onsets.

    Parameters
    ----------
    data : 1d or 2d array
        Continuous data, as [n_times] or [n_channels, n_times].
    sfreq : float
        Sampling frequency, in Hz.
    onsets : 1d array
        Event onsets, in seconds, on the same clock as `t0`.
    tmin, tmax : float
        Epoch window relative to each onset, in seconds (for example -0.5 and 1.0).
    t0 : float, optional, default: 0
        Time of the first sample of `data`, in seconds.

    Returns
    -------
    times : 1d array
        Time of each epoch sample relative to the onset, in seconds.
    epochs : 2d or 3d array
        Epochs, as [n_events, n_times] or [n_events, n_channels, n_times].
    kept : 1d array of bool
        Which onsets gave an epoch; epochs that do not fit inside `data` are dropped.
    """
    data = np.asarray(data)
    first, last = int(np.round(tmin * sfreq)), int(np.round(tmax * sfreq))
    times = np.arange(first, last + 1) / sfreq

    centers = np.round((np.asarray(onsets, dtype=float) - t0) * sfreq).astype(int)
    kept = (centers + first >= 0) & (centers + last < data.shape[-1])

    inds = centers[kept][:, None] + np.arange(first, last + 1)[None, :]
    epochs = data[..., inds]
    if data.ndim == 2:
        epochs = np.moveaxis(epochs, 0, 1)

    return times, epochs, kept


def epoch_windows(
    onsets: Any, sfreq: float, tmin: float, tmax: float, n_samples: int, t0: float = 0.0
) -> tuple[Any, Any, Any]:
    """Sample windows of the epochs ``[tmin, tmax)`` around onsets.

    The onset sample is the sample nearest each onset; the window holds the samples
    from ``round(tmin * sfreq)`` (included) to ``round(tmax * sfreq)`` (excluded)
    relative to it, so every epoch has the same length and ``times`` holds 0.

    Returns
    -------
    times : 1d array
        Seconds relative to the onset.
    starts : 1d array of int
        First sample of each window, for every onset.
    kept : 1d array of bool
        Which windows lie inside the recording of ``n_samples`` samples.
    """
    if not tmax > tmin:
        raise DataError(f"epoch: tmax ({tmax}) must be greater than tmin ({tmin}).")
    first, stop = int(np.round(tmin * sfreq)), int(np.round(tmax * sfreq))
    if stop - first < 2:
        raise DataError(f"epoch: the window ({tmin}, {tmax}) holds fewer than 2 samples.")
    onsets = np.atleast_1d(np.asarray(onsets, dtype=float))
    if onsets.ndim != 1 or not np.all(np.isfinite(onsets)):
        raise EventError("epoch: onsets must be a 1d array of finite times (s).")
    centers = np.round((onsets - t0) * sfreq).astype(np.int64)
    starts = centers + first
    kept = (starts >= 0) & (centers + stop <= n_samples)
    return np.arange(first, stop) / sfreq, starts, kept


def overlap_flags(onsets: Any, length: Any) -> NDArray[Any]:
    """Return True for each event closer than ``length`` seconds to the previous event in time."""
    onsets = np.asarray(onsets, dtype=float)
    order = np.argsort(onsets, kind="stable")
    flags = np.zeros(onsets.size, dtype=bool)
    if onsets.size > 1:
        flags[order[1:]] = np.diff(onsets[order]) < length
    return flags


def _as_table(table: Any, n_rows: Any, name: Any) -> Any:
    """DataFrame (RangeIndex) from a DataFrame or a dict of columns."""
    require("pandas", extra="data", caller="evoparam.io.epochs._as_table")
    import pandas as pd

    if isinstance(table, dict):
        table = pd.DataFrame(table)
    if not (type(table).__name__ == "DataFrame"):
        raise DataError(f"{name}: expected a pandas DataFrame or a dict of columns.")
    if len(table) != n_rows:
        raise DataError(f"{name}: expected {n_rows} rows, got {len(table)}.")
    return table.reset_index(drop=True)


def _pandas() -> Any:
    try:
        require("pandas", extra="data", caller="evoparam.io.epochs._pandas")
        import pandas
    except ImportError:
        return None
    return pandas


def epoch(  # noqa: PLR0913  (public API, signature fixed by spec §5.1)
    data: ArrayLike,
    sfreq: float,
    onsets: Any,
    tmin: float,
    tmax: float,
    *,
    t0: float = 0.0,
    ch_names: list[str] | None = None,
    units: str = "uV",
    metadata: Any = None,
    ch_info: Any = None,
) -> "TrialSet":
    """Cut a continuous recording into a TrialSet of epochs around events.

    Parameters
    ----------
    data : 2d array
        Continuous data, as [n_channels, n_times] (a 1d array is one channel).
    sfreq : float
        Sampling frequency, in Hz.
    onsets : 1d array or pandas.DataFrame
        Event onsets, in seconds, on the same clock as ``t0``; or the event table of
        :func:`evoparam.events.detect_pulses` (its ``onset_time`` column), whose detection
        step is then recorded in the provenance before the epoching (so that the methods
        text describes it).
    tmin, tmax : float
        Epoch window relative to each onset, in seconds; ``tmin`` included,
        ``tmax`` excluded.
    t0 : float
        Time of the first sample of ``data``, in seconds.
    ch_names : list of str, optional
        Channel names; default ``CH001``, ``CH002``, ...
    units : str
        Amplitude unit of ``data``.
    metadata : pandas.DataFrame or dict, optional
        One row per onset (before dropping); the rows of the dropped onsets are
        dropped too.
    ch_info : pandas.DataFrame or dict, optional
        One row per channel.

    Returns
    -------
    trials : TrialSet
        One trial per kept onset. ``event_times`` holds the kept onsets. The last
        provenance step records ``kept`` (one bool per onset: False when the window
        exceeds the recording) and ``overlaps_previous``. With pandas installed,
        ``metadata`` holds a ``trial`` column (index of the onset) when none is given,
        and a bool column ``overlaps_previous``, True when the epoch starts less than
        ``tmax - tmin`` after the previous event (such trials are kept).

    Raises
    ------
    DataError
        Inconsistent shapes or window.
    EventError
        No epoch fits inside the recording.
    """
    start = _time.perf_counter()
    prior = []
    prior_source = None
    if type(onsets).__name__ == "DataFrame":
        record = onsets.attrs.get("provenance") or {}
        prior = (
            list(record.get("steps", []))
            if isinstance(record, dict)
            else list(getattr(record, "steps", []))
        )
        # the source of the recording the events were detected in (REQ-294: the trials
        # continue its record)
        prior_source = (
            record.get("source") if isinstance(record, dict) else getattr(record, "source", None)
        )
        if "onset_time" not in onsets.columns:
            raise DataError("epoch: an event table needs the column onset_time.")
        onsets = onsets["onset_time"].to_numpy(float)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[np.newaxis]
    if data.ndim != 2:
        raise DataError(f"epoch: data must be 2d (n_channels, n_times), got shape {data.shape}.")
    onsets = np.atleast_1d(np.asarray(onsets, dtype=float))
    times, starts, kept = epoch_windows(onsets, sfreq, tmin, tmax, data.shape[1], t0=t0)
    if not kept.any():
        raise EventError(f"epoch: no epoch of ({tmin}, {tmax}) s fits inside the recording.")

    index = starts[kept][:, None] + np.arange(times.size)[None, :]
    trials = np.moveaxis(data[:, index], 0, 1)
    overlaps = overlap_flags(onsets[kept], tmax - tmin)

    ch_names = _default_names(data.shape[0]) if ch_names is None else list(ch_names)
    pd = _pandas()
    table = None
    if metadata is not None:
        table = _as_table(metadata, onsets.size, "metadata").iloc[np.flatnonzero(kept)]
        table = table.reset_index(drop=True)
    elif pd is not None:
        table = pd.DataFrame({"trial": np.flatnonzero(kept).astype(np.int64)})
    if table is not None:
        table["overlaps_previous"] = overlaps
    if ch_info is not None:
        ch_info = _as_table(ch_info, len(ch_names), "ch_info")

    settings = {
        "sfreq": float(sfreq),
        "tmin": float(tmin),
        "tmax": float(tmax),
        "t0": float(t0),
        "n_onsets": int(onsets.size),
        "units": units,
    }
    source = {"path": None, "format": "array", "reader": "evoparam.io.epoch"}
    if prior_source and prior_source.get("reader") not in (
        None,
        "evoparam.events.detect_pulses",
        "evoparam.events.detect_events",
    ):
        source = dict(prior_source)
    provenance = Provenance(source=source, steps=prior)
    provenance = provenance.with_step(
        "evoparam.io.epoch",
        settings,
        duration_s=_time.perf_counter() - start,
        kept=kept.tolist(),
        overlaps_previous=overlaps.tolist(),
    )
    return TrialSet(
        data=trials,
        times=times,
        sfreq=sfreq,
        ch_names=ch_names,
        units=units,
        metadata=table,
        ch_info=ch_info,
        event_times=onsets[kept],
        provenance=provenance,
    )


def _default_names(n_channels: int) -> list[str]:
    width = max(3, len(str(n_channels)))
    return ["CH{:0{}d}".format(ind + 1, width) for ind in range(n_channels)]
