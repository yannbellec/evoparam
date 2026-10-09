"""Baseline correction, with checks against artifact masks and delayed responses."""

import time as _time
import warnings

import numpy as np

from evoparam.errors import DataError, SettingsError, warn as _warn

from ._common import check_trialset, masked_values, new, window_mask

__all__ = ["baseline_correct", "previous_pulse_gaps", "train_starts"]

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
    ``min_gap`` seconds belong to the same train)."""

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
        suggestion = (f"baseline=({latest_start:.3f}, {baseline[1]:.3f})")
    else:
        suggestion = ("baseline_source='first_pulse' (each pulse uses the baseline of the "
                      "first pulse of its train), or a smaller delayed_margin if delayed "
                      "responses are not expected")
    _warn(
        f"baseline {tuple(baseline)} s starts within {delayed_margin} s of the previous pulse "
        f"for {int(bad.sum())} of {len(bad)} trials (shortest interval between pulses "
        f"{shortest:.3f} s): delayed responses of the previous pulse may bias it. "
        f"Suggested: {suggestion}.", UserWarning, stacklevel=3)


def baseline_correct(ts, baseline=(-0.5, -0.05), *, mode="mean", check=True,
                     delayed_margin=1.0, baseline_source="own"):
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
        raise DataError(f"baseline {tuple(baseline)} s: fewer than half of its samples are "
                        "unmasked for some trial and channel; move or widen the baseline.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if mode == "linear":
            t = ts.times[inside] - ts.times[inside].mean()
            w = ~np.isnan(values)
            v = np.nan_to_num(values)
            n = w.sum(-1)
            st, sv = (w * t).sum(-1), v.sum(-1)
            stt, stv = (w * t * t).sum(-1), (v * t).sum(-1)
            slope = (n * stv - st * sv) / (n * stt - st ** 2)
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
    return new(ts, "evoparam.prep.baseline_correct",
               {"baseline": list(baseline), "mode": mode, "check": check,
                "delayed_margin": delayed_margin, "baseline_source": baseline_source},
               start, data=data)
