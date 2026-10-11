# SPDX-License-Identifier: Apache-2.0
"""Amplifier saturation: find the samples where a channel is stuck at its rail."""

import numpy as np

from evoparam.errors import DataError
from evoparam.utils._optional import require

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


def detect_saturation(data, sfreq, *, events=None, rail=None, min_run=3, recovery=0.005,
                      search=0.05, t0=0.0, ch_names=None, picks=None):
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
    table : pandas.DataFrame
        One row per saturated run: ``channel``, ``start_time``, ``end_time`` (s, end of
        the recovery window), ``n_samples`` (railed samples), ``event_index`` (row of
        ``events`` the run belongs to, -1 when ``events`` is None).

    Raises
    ------
    DataError
        Invalid data, ``rail`` or ``min_run``.
    """

    pd = require("pandas", extra="data", caller="evoparam.events.detect_saturation")
    data = as_continuous(data)
    n_channels, n_times = data.shape
    if min_run < 2:
        raise DataError("min_run must be at least 2.")
    names = list(ch_names) if ch_names is not None else [str(ind) for ind in range(n_channels)]
    if len(names) != n_channels:
        raise DataError("ch_names must have one name per channel.")
    index = pick_index(n_channels, picks, names)

    # Samples examined: windows around the pulses (one row per pulse), or the recording
    if events is not None:
        onsets = np.asarray(events["onset_sample"], dtype=int)
        half = int(round(search * sfreq))
        positions = onsets[:, None] + np.arange(-half, half + 1)[None]
    else:
        positions = np.arange(n_times)[None]
    valid = (positions >= 0) & (positions < n_times)
    positions = np.clip(positions, 0, n_times - 1)

    known = rail is not None
    if known:
        rails = np.broadcast_to(np.abs(np.asarray(rail, dtype=float)), (n_channels,))

    extra = int(round(recovery * sfreq))
    mask = np.zeros((n_channels, n_times), dtype=bool)
    rows = []
    for ch in index:
        trace = data[ch]
        segment = trace[positions].astype(float)
        segment[~valid] = np.nan
        if known:
            level = rails[ch]
        else:
            level = RAIL_FRACTION * np.nanmax(np.abs(segment))
            # The spread is estimated on every 16th sample of the recording: accurate to
            # about 1 % with tens of thousands of samples, and 16 times faster
            sample = trace[::_SPREAD_STEP]
            sample = sample[~np.isnan(sample)]
            center = np.median(sample) if len(sample) else 0.0
            spread = 1.4826 * np.median(np.abs(sample - center)) if len(sample) else 0.0
            # A channel whose largest value is not an outlier has no estimable rail
            if level - abs(center) < RAIL_MIN_SD * spread:
                continue
        if not 0 < level < np.inf:
            continue
        with np.errstate(invalid="ignore"):
            railed = np.abs(segment) >= level
            flat = np.diff(segment, axis=1) == 0
        flat = np.c_[np.zeros((len(segment), 1), bool), flat]
        flat &= np.abs(np.nan_to_num(segment)) >= FLAT_MIN_FRACTION * level
        # A flat run of k equal samples has k - 1 zero differences: grow it by one sample.
        flat |= np.c_[flat[:, 1:], np.zeros((len(segment), 1), bool)]
        for kind, values, min_len in (("rail", railed, 1 if known else 2),
                                      ("flat", flat, min_run)):
            if not values.any():
                continue
            edges = np.diff(np.pad(values.astype(np.int8), ((0, 0), (1, 1))), axis=1)
            starts, stops = np.nonzero(edges == 1), np.nonzero(edges == -1)
            for row, first, last in zip(starts[0], starts[1], stops[1]):
                if last - first < min_len:
                    continue
                a_, b_ = positions[row, first], positions[row, last - 1] + 1
                if mask[ch, a_:b_].all():
                    continue
                end = min(n_times, b_ + extra)
                mask[ch, a_:end] = True
                rows.append({"channel": names[ch], "start_time": t0 + a_ / sfreq,
                             "end_time": t0 + end / sfreq, "n_samples": int(b_ - a_),
                             "event_index": int(row) if events is not None else -1,
                             "kind": kind})

    columns = ["channel", "start_time", "end_time", "n_samples", "event_index", "kind"]
    table = pd.DataFrame(rows, columns=columns)
    if len(table):
        table = (table.sort_values(["start_time", "channel"])
                 .drop_duplicates(["channel", "start_time"]).reset_index(drop=True))
    table = table.astype({"n_samples": "int64", "event_index": "int64"})
    return mask, table
