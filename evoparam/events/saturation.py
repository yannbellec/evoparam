# SPDX-License-Identifier: Apache-2.0
"""Amplifier saturation: find the samples where a channel is stuck at its rail."""

import dataclasses

import numpy as np

from evoparam._typing import Any, ArrayLike, FloatArray, NDArray
from evoparam.data._table import new_table
from evoparam.errors import DataError

from ._score import as_continuous, pick_index

__all__ = ["detect_saturation"]

# Fraction of the rail above which a sample counts as railed when the rail is estimated.
RAIL_FRACTION = 0.999

# A flat run counts as saturation only if it sits at least this far towards the rail,
# so that quantised low-amplitude noise (repeated integer codes) is never flagged.
FLAT_MIN_FRACTION = 0.5

# An estimated rail must lie at least this many robust SD from the channel's median:
# a clean channel's largest value is only a few SD, and on quantised data it repeats.
RAIL_MIN_SD = 8.0

# Step of the subsample used to estimate a channel's spread.
_SPREAD_STEP = 16


def detect_saturation(  # noqa: PLR0913  (public keywords; DECISIONS.md §7.2)
    data: FloatArray,
    sfreq: float,
    *,
    events: Any = None,
    rail: Any = None,
    min_run: int = 3,
    recovery: float = 0.005,
    search: float = 0.05,
    t0: float = 0.0,
    ch_names: list[str] | None = None,
    picks: ArrayLike | None = None,
) -> tuple[NDArray[Any], Any]:
    """Find amplifier saturation, channel by channel and pulse by pulse.

    A channel is saturated over a run of samples when either:

    - its absolute value stays at the rail (``|x| >= rail``) for at least 2 samples, or
      for 1 sample when ``rail`` is given (the true ADC range is known); or
    - it stays exactly constant for at least ``min_run`` samples, at a value at least
      half-way to the rail (an ADC stuck at a code).

    The unrecoverable window of a run is ``[first sample, last sample + recovery]``:
    after leaving the rail the amplifier needs some time to recover. These windows
    form the returned mask; later stages give masked samples zero weight.

    Parameters
    ----------
    data : 1d or 2d array
        Continuous data, as [n_times] or [n_channels, n_times].
    sfreq : float
        Sampling frequency, in Hz.
    events : pandas.DataFrame, optional
        Event table with an ``onset_sample`` column (as returned by
        :func:`detect_pulses`). When given, only the samples within ``search`` seconds
        of a pulse are examined, and each run is linked to its pulse. When None, the
        whole recording is examined.
    rail : float or 1d array, optional
        Saturation level per channel, in the units of ``data`` (for example the ADC
        range read from the file header). Default: 0.999 times the largest absolute
        value of each channel over the examined samples, when that value lies at least
        8 robust SD from the channel's median (otherwise the channel is not examined: a
        clean channel never reaches a rail).
    min_run : int, default: 3
        Minimum length, in samples, of a flat run.
    recovery : float, default: 0.005
        Recovery time added after each run, in seconds.
    search : float, default: 0.05
        Half-width, in seconds, of the window around each pulse that is examined.
    t0 : float, default: 0
        Time of the first sample, in seconds.
    ch_names : list of str, optional
        Channel names, used in the table. Default: channel indices as strings.
    picks : array-like, optional
        Channels to examine (indices, names or a boolean mask). Others are never flagged.

    Returns
    -------
    mask : 2d array of bool, [n_channels, n_times]
        True on the unrecoverable samples.
    table : pandas.DataFrame (a :class:`~evoparam.data.Table` without pandas)
        One row per saturated run: ``channel``, ``start_time``, ``end_time`` (s, end of
        the recovery window), ``n_samples`` (railed samples), ``event_index`` (row of
        ``events`` the run belongs to, -1 when ``events`` is None).

    Raises
    ------
    DataError
        Invalid data, ``rail`` or ``min_run``.
    """
    data = as_continuous(data)
    n_channels, n_times = data.shape
    if min_run < _MIN_RAIL_RUN:
        raise DataError("min_run must be at least 2.")
    names = list(ch_names) if ch_names is not None else [str(ind) for ind in range(n_channels)]
    if len(names) != n_channels:
        raise DataError("ch_names must have one name per channel.")
    index = pick_index(n_channels, picks, names)
    positions, valid = _examined(events, search, sfreq, n_times)
    known = rail is not None
    rails = (
        np.broadcast_to(np.abs(np.asarray(rail, dtype=float)), (n_channels,)) if known else None
    )
    scan = _Scan(
        positions=positions,
        names=names,
        sfreq=sfreq,
        t0=t0,
        extra=int(round(recovery * sfreq)),
        min_run=min_run,
        known=known,
        per_event=events is not None,
    )
    mask = np.zeros((n_channels, n_times), dtype=bool)
    rows = []
    for ch in index:
        trace = data[ch]
        segment = trace[positions].astype(float)
        segment[~valid] = np.nan
        level = rails[ch] if rails is not None else _estimated_rail(trace, segment)
        if level is None or not 0 < level < np.inf:
            continue
        rows += _channel_runs(ch, segment, level, scan, mask)
    return mask, _run_table(rows)


# Runs at the rail count from 2 samples when the rail is estimated (and flat runs from 2).
_MIN_RAIL_RUN = 2
_MAD_TO_SD = 1.4826


@dataclasses.dataclass(frozen=True)
class _Scan:
    """What the channels of one saturation scan share."""

    positions: NDArray[Any]
    names: list[str]
    sfreq: float
    t0: float
    extra: int
    min_run: int
    known: bool
    per_event: bool


def _examined(events: Any, search: float, sfreq: float, n_times: int) -> tuple[Any, Any]:
    """Return the samples examined (one row per pulse, or the recording) and their validity."""
    if events is not None:
        onsets = np.asarray(events["onset_sample"], dtype=int)
        half = int(round(search * sfreq))
        positions = onsets[:, None] + np.arange(-half, half + 1)[None]
    else:
        positions = np.arange(n_times)[None]
    valid = (positions >= 0) & (positions < n_times)
    return np.clip(positions, 0, n_times - 1), valid


def _estimated_rail(trace: Any, segment: Any) -> Any:
    """Return the rail of a channel from its examined samples, or None when it has none."""
    level = RAIL_FRACTION * np.nanmax(np.abs(segment))
    # The spread is estimated on every 16th sample of the recording: accurate to
    # about 1 % with tens of thousands of samples, and 16 times faster
    sample = trace[::_SPREAD_STEP]
    sample = sample[~np.isnan(sample)]
    center = np.median(sample) if len(sample) else 0.0
    spread = _MAD_TO_SD * np.median(np.abs(sample - center)) if len(sample) else 0.0
    # A channel whose largest value is not an outlier has no estimable rail
    if level - abs(center) < RAIL_MIN_SD * spread:
        return None
    return level


def _channel_runs(ch: int, segment: Any, level: Any, scan: _Scan, mask: Any) -> list[Any]:
    """Return the saturated runs of channel ``ch`` and mark them in ``mask``."""
    railed = np.abs(segment) >= level
    # a zero difference of two samples (equal and finite: inf - inf is not computed)
    flat = (segment[:, 1:] == segment[:, :-1]) & np.isfinite(segment[:, 1:])
    flat = np.c_[np.zeros((len(segment), 1), bool), flat]
    flat &= np.abs(np.nan_to_num(segment)) >= FLAT_MIN_FRACTION * level
    # A flat run of k equal samples has k - 1 zero differences: grow it by one sample.
    flat |= np.c_[flat[:, 1:], np.zeros((len(segment), 1), bool)]
    rows = []
    for kind, values, min_len in (
        ("rail", railed, 1 if scan.known else _MIN_RAIL_RUN),
        ("flat", flat, scan.min_run),
    ):
        if values.any():
            rows += _runs(ch, kind, (values, min_len), scan, mask)
    return rows


def _runs(ch: int, kind: str, runs: Any, scan: _Scan, mask: Any) -> list[Any]:
    """Return the runs of ``runs = (values, min_len)`` not yet masked, and mask them."""
    values, min_len = runs
    n_times = mask.shape[1]
    positions, sfreq, t0 = scan.positions, scan.sfreq, scan.t0
    edges = np.diff(np.pad(values.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    starts, stops = np.nonzero(edges == 1), np.nonzero(edges == -1)
    rows = []
    for row, first, last in zip(starts[0], starts[1], stops[1], strict=False):
        if last - first < min_len:
            continue
        a_, b_ = positions[row, first], positions[row, last - 1] + 1
        if mask[ch, a_:b_].all():
            continue
        end = min(n_times, b_ + scan.extra)
        mask[ch, a_:end] = True
        rows.append(
            {
                "channel": scan.names[ch],
                "start_time": t0 + a_ / sfreq,
                "end_time": t0 + end / sfreq,
                "n_samples": int(b_ - a_),
                "event_index": int(row) if scan.per_event else -1,
                "kind": kind,
            }
        )
    return rows


def _run_table(rows: list[Any]) -> Any:
    """Return the table of the runs, sorted by start time and channel, duplicates dropped."""
    # sorted by start time and channel, the first of the runs with the same channel and
    # start kept (a table without pandas, REQ-010: the same rows and dtypes)
    rows.sort(key=lambda run: (run["start_time"], run["channel"]))
    seen: set[tuple[Any, Any]] = set()
    kept = []
    for run in rows:
        key = (run["channel"], run["start_time"])
        if key not in seen:
            seen.add(key)
            kept.append(run)
    dtypes: dict[str, Any] = {
        "channel": object,
        "start_time": float,
        "end_time": float,
        "n_samples": "int64",
        "event_index": "int64",
        "kind": object,
    }
    return new_table(
        {name: np.array([run[name] for run in kept], dtype=kind) for name, kind in dtypes.items()}
    )
