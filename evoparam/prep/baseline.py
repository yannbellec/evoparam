# SPDX-License-Identifier: Apache-2.0
"""Baseline correction, with checks against artifact masks and delayed responses."""

import time as _time
import warnings

import numpy as np

from evoparam.errors import DataError, SettingsError
from evoparam.errors import warn as _warn

from ._common import check_trialset, masked_values, new, window_mask

__all__ = ["baseline_correct", "train_baseline", "previous_pulse_gaps", "train_starts"]

MODES = ("mean", "median", "linear")
SOURCES = ("own", "first_pulse")

# Fewest valid (unmasked) baseline samples, as a fraction of the baseline window.
MIN_VALID_FRACTION = 0.5


def previous_pulse_gaps(event_times):
    """Time from the previous pulse to each pulse (inf for the first one)."""
    times = np.asarray(event_times, dtype=float)
    order = np.argsort(times, kind="stable")
    gaps = np.full(len(times), np.inf)
    gaps[order[1:]] = np.diff(times[order])
    return gaps


def train_starts(event_times, min_gap):
    """For each pulse, the index of the first pulse of its train (pulses closer than
    ``min_gap`` seconds belong to the same train).
    """
    times = np.asarray(event_times, dtype=float)
    order = np.argsort(times, kind="stable")
    first = np.empty(len(times), dtype=int)
    current = order[0] if len(order) else 0
    for pos, ind in enumerate(order):
        if pos and times[ind] - times[order[pos - 1]] >= min_gap:
            current = ind
        first[ind] = current
    return first


def _check_delayed(ts, baseline, delayed_margin, source):
    """Warn when a baseline starts inside the previous pulse's delayed-response window."""
    if ts.event_times is None or source == "first_pulse":
        return
    gaps = previous_pulse_gaps(ts.event_times)
    # The baseline starts |baseline[0]| s before the pulse, i.e. gap + baseline[0] s after
    # the previous pulse; it must be at least delayed_margin
    after_previous = gaps + baseline[0]
    bad = after_previous < delayed_margin
    if not bad.any():
        return
    shortest = float(np.min(gaps))
    latest_start = -(shortest - delayed_margin)
    if latest_start < baseline[1]:
        suggestion = f"baseline=({latest_start:.3f}, {baseline[1]:.3f})"
    else:
        suggestion = (
            "baseline_source='first_pulse' (each pulse uses the baseline of the "
            "first pulse of its train), or a smaller delayed_margin if delayed "
            "responses are not expected"
        )
    _warn(
        f"baseline {tuple(baseline)} s starts within {delayed_margin} s of the previous pulse "
        f"for {int(bad.sum())} of {len(bad)} trials (shortest interval between pulses "
        f"{shortest:.3f} s): delayed responses of the previous pulse may bias it. "
        f"Suggested: {suggestion}.",
        UserWarning,
        stacklevel=3,
    )


def baseline_correct(
    ts,
    baseline=(-0.5, -0.05),
    *,
    mode="mean",
    check=True,
    delayed_margin=1.0,
    baseline_source="own",
):
    """Subtract each trial's and channel's baseline.

    ``x(t) - m`` with ``m`` the mean (or median) of the baseline window, or with a straight
    line fitted on the baseline removed (``mode="linear"``). Masked samples are left out of
    ``m``; when fewer than half of the baseline samples remain, :class:`DataError`.

    With ``check`` and known ``event_times``, a warning is given when the baseline starts
    less than ``delayed_margin`` seconds after the previous pulse: delayed responses last up
    to about 1 s. The warning gives the shortest interval and a valid alternative.

    Parameters
    ----------
    ts : TrialSet
    baseline : (float, float)
        Window, in seconds.
    mode : {"mean", "median", "linear"}
    check : bool
    delayed_margin : float
        Seconds after a pulse during which its delayed responses may last.
    baseline_source : {"own", "first_pulse"}
        ``"first_pulse"``: every pulse uses the baseline of the first pulse of its train
        (pulses closer than ``delayed_margin + |baseline[0]|`` form a train), which lies
        before the train; needs ``event_times``.

    Returns
    -------
    TrialSet
    """
    start = _time.perf_counter()
    check_trialset(ts)
    if mode not in MODES:
        raise SettingsError(f"mode must be one of {MODES}, not {mode!r}.")
    if baseline_source not in SOURCES:
        raise SettingsError(f"baseline_source must be one of {SOURCES}.")
    inside = window_mask(ts.times, baseline, "baseline")
    if check:
        _check_delayed(ts, baseline, delayed_margin, baseline_source)

    values = masked_values(ts)[:, :, inside]
    valid = np.sum(~np.isnan(values), axis=-1)
    if np.any(valid < MIN_VALID_FRACTION * inside.sum()):
        raise DataError(
            f"baseline {tuple(baseline)} s: fewer than half of its samples are "
            "unmasked for some trial and channel; move or widen the baseline."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if mode == "linear":
            t = ts.times[inside] - ts.times[inside].mean()
            w = ~np.isnan(values)
            v = np.nan_to_num(values)
            n = w.sum(-1)
            st, sv = (w * t).sum(-1), v.sum(-1)
            stt, stv = (w * t * t).sum(-1), (v * t).sum(-1)
            slope = (n * stv - st * sv) / (n * stt - st**2)
            offset = (sv - slope * st) / n
            trend = offset[..., None] + slope[..., None] * (ts.times - ts.times[inside].mean())
        else:
            stat = np.nanmean if mode == "mean" else np.nanmedian
            level = stat(values, axis=-1)
            trend = level[..., None]

    if baseline_source == "first_pulse":
        if ts.event_times is None:
            raise DataError("baseline_source='first_pulse' needs event_times.")
        first = train_starts(ts.event_times, delayed_margin + abs(baseline[0]))
        if mode == "linear":
            raise SettingsError("baseline_source='first_pulse' needs mode 'mean' or 'median'.")
        trend = trend[first]

    data = ts.data - trend
    return new(
        ts,
        "evoparam.prep.baseline_correct",
        {
            "baseline": list(baseline),
            "mode": mode,
            "check": check,
            "delayed_margin": delayed_margin,
            "baseline_source": baseline_source,
        },
        start,
        data=data,
    )


def _continuous(raw, sfreq, t0, ts):
    """(data [channels of ts, n_times], sfreq, t0) of a continuous recording given as an
    array or as a TrialSet holding one continuous trial (``evoparam.io.read`` without
    onsets).
    """
    from evoparam.data import TrialSet

    if isinstance(raw, TrialSet):
        if raw.n_trials != 1:
            raise DataError("train_baseline: raw must hold one continuous trial.")
        names = list(raw.ch_names)
        missing = [n for n in ts.ch_names if n not in names]
        if missing:
            raise DataError(f"train_baseline: raw has no channel {missing[:3]}.")
        data = raw.data[0][[names.index(n) for n in ts.ch_names]]
        return data, float(raw.sfreq), float(raw.times[0])
    data = np.atleast_2d(np.asarray(raw, dtype=float))
    if data.shape[0] != ts.n_channels:
        raise DataError(
            f"train_baseline: raw has {data.shape[0]} channels, the trials {ts.n_channels}."
        )
    if sfreq is None:
        raise DataError("train_baseline: sfreq is required with an array raw.")
    return data, float(sfreq), float(t0)


def train_baseline(
    ts, length=0.05, *, raw=None, sfreq=None, t0=0.0, train_column="train_id", mode="mean"
):
    """Baseline per stimulation train: per channel, the mean (or median) of the last
    ``length`` seconds before each train is subtracted from every trial (pulse) of that
    train (spec §6.1, preset ``erna``: "per train: last 50 ms before the train").

    Trains: ``metadata[train_column]`` when present (``evoparam.events.group_trains``);
    else, with ``event_times``, a pulse starts a train when no other pulse came in the
    ``length`` seconds before it (so the baseline of a train never holds a pulse); else
    each trial is one train whose epoch is locked to the train onset (a burst per trial).
    The onset of a train is the earliest event time of its trials.

    The baseline samples come from the continuous recording ``raw`` (an array [n_channels,
    n_times] with ``sfreq`` and ``t0``, the time of its first sample, on the clock of
    ``event_times``; or a TrialSet with one continuous trial), or, without it, from the
    epoch of the first trial of each train, which must then start at least ``length``
    seconds before its onset. Masked samples are left out.

    Raises
    ------
    DataError
        The baseline samples are not available (epochs too short and no ``raw``; a train
        onset closer than ``length`` to the start of ``raw``), or fewer than half of them
        are unmasked.
    """
    start = _time.perf_counter()
    check_trialset(ts)
    if mode not in ("mean", "median"):
        raise SettingsError("train_baseline: mode must be 'mean' or 'median'.")
    if not length > 0:
        raise SettingsError("train_baseline: length must be positive.")
    meta = ts.metadata
    event_times = None if ts.event_times is None else np.asarray(ts.event_times, float)
    if meta is not None and train_column in meta.columns:
        keys = meta[train_column].astype(str).to_numpy()
        rule = train_column
    elif event_times is not None:
        first = train_starts(event_times, length)
        keys = first.astype(str)
        rule = "gap"
    else:
        keys = np.arange(ts.n_trials).astype(str)
        rule = "trial"
    stat = np.nanmean if mode == "mean" else np.nanmedian
    levels = np.empty((ts.n_trials, ts.n_channels))
    source = "continuous" if raw is not None else "epochs"
    if raw is not None:
        if event_times is None:
            raise DataError("train_baseline: raw needs the trials' event_times.")
        data, rate, first_time = _continuous(raw, sfreq, t0, ts)
    values = masked_values(ts)
    n_wanted = int(round(length * (rate if raw is not None else ts.sfreq)))
    for key in dict.fromkeys(keys):
        rows = np.flatnonzero(keys == key)
        if event_times is not None:
            lead = rows[np.argmin(event_times[rows])]
        else:
            lead = rows[0]
        if raw is not None:
            onset = event_times[lead]
            stop = int(round((onset - first_time) * rate))
            begin = stop - n_wanted
            if begin < 0 or stop > data.shape[1]:
                raise DataError(
                    f"train_baseline: the {length * 1000:g} ms before the train "
                    f"at {onset:.3f} s are outside raw."
                )
            segment = data[:, begin:stop]
        else:
            inside = (ts.times >= -length - 0.5 / ts.sfreq) & (ts.times < 0)
            if ts.times[0] > -length + 0.5 / ts.sfreq:
                raise DataError(
                    f"train_baseline: the epochs start at {ts.times[0]:.4f} s; "
                    f"the last {length * 1000:g} ms before the train need epochs "
                    f"from -{length:g} s, or the continuous recording (raw=)."
                )
            segment = values[lead][:, inside]
        valid = np.sum(np.isfinite(segment), axis=1)
        if np.any(valid < MIN_VALID_FRACTION * max(segment.shape[1], 1)):
            raise DataError(
                f"train_baseline: fewer than half of the baseline samples of "
                f"train {key} are unmasked."
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            levels[rows] = stat(segment, axis=1)[None]
    data_out = ts.data - levels[..., None]
    return new(
        ts,
        "evoparam.prep.train_baseline",
        {
            "length": float(length),
            "mode": mode,
            "trains": rule,
            "source": source,
            "n_trains": int(len(set(keys))),
        },
        start,
        data=data_out,
    )
