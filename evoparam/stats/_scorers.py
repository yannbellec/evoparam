"""Classical scores of many waveforms at once, with the definitions of
:func:`evoparam.models.classical_measures`.

The statistics of this module score thousands of waveforms (bootstrap resamples,
split halves, subsamples), so the scores are computed on arrays ``[..., n_times]``
instead of one waveform at a time. Each score follows the definition of
``classical_measures`` exactly (tested against it): window inclusive at both ends, local
peak larger than ``n_neighbors`` samples on each side and than their means (simple
extremum when there is none), trapezoidal areas.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from evoparam.errors import SettingsError
from evoparam.models.classical import AREAS, POLARITIES

__all__ = ["SCORES", "score_waveforms", "resolve_scorer", "LINEAR_SCORES", "TIME_SCORES"]

SCORES = ("mean_amplitude", "peak_amplitude", "peak_latency", "area",
          "fractional_area_latency", "fractional_peak_latency")

# Scores that are linear in the waveform: the score of an average is the average of the
# scores, so they can be computed once per trial.
LINEAR_SCORES = ("mean_amplitude",)

# Scores in seconds (the others are in the data unit, or data unit x s for the area).
TIME_SCORES = ("peak_latency", "fractional_area_latency", "fractional_peak_latency")

_TRAPEZOID = getattr(np, "trapezoid", None) or np.trapz


def _signed(y, polarity):
    if polarity == "negative":
        return -y
    if polarity == "positive":
        return y
    return np.abs(y)


def _local_peak(s, n_neighbors):
    """Index of the ERPLAB local peak of each row of ``s`` (the signed waveforms)."""

    n = s.shape[-1]
    if n < 2 * n_neighbors + 1:
        return np.argmax(s, axis=-1)
    win = sliding_window_view(s, 2 * n_neighbors + 1, axis=-1)
    centre = win[..., n_neighbors]
    left, right = win[..., :n_neighbors], win[..., n_neighbors + 1:]
    ok = (centre > left.max(axis=-1)) & (centre > right.max(axis=-1)) \
        & (centre > left.mean(axis=-1)) & (centre > right.mean(axis=-1))
    candidate = np.where(ok, centre, -np.inf)
    best = np.argmax(candidate, axis=-1) + n_neighbors
    return np.where(ok.any(axis=-1), best, np.argmax(s, axis=-1))


def _take(values, index):
    return np.take_along_axis(values, index[..., None], axis=-1)[..., 0]


def _baseline(y, peak, area_baseline, fraction):
    if area_baseline == "zero":
        return np.zeros(y.shape[:-1])
    if area_baseline == "fraction_of_peak":
        return fraction * _take(y, peak)
    raise SettingsError("area_baseline must be 'zero' or 'fraction_of_peak'.")


def _area_values(y, base, area):
    values = y - base[..., None]
    if area == "positive":
        return np.clip(values, 0, None)
    if area == "negative":
        return np.clip(values, None, 0)
    if area == "rectified":
        return np.abs(values)
    return values


def _fractional_area_latency(t, y, base, area, fraction):
    values = y - base[..., None]
    if area == "positive":
        values = np.clip(values, 0, None)
    elif area == "negative":
        values = -np.clip(values, None, 0)
    else:
        values = np.abs(values)
    steps = 0.5 * (values[..., 1:] + values[..., :-1]) * np.diff(t)
    cum = np.concatenate([np.zeros(values.shape[:-1] + (1,)), np.cumsum(steps, axis=-1)],
                         axis=-1)
    total = cum[..., -1]
    target = fraction * total
    i = np.argmax(cum >= target[..., None], axis=-1)
    i = np.clip(i, 1, len(t) - 1)
    hi, lo = _take(cum, i), _take(cum, i - 1)
    gap = hi - lo
    frac = np.divide(target - lo, gap, out=np.zeros_like(gap), where=gap > 0)
    out = t[i - 1] + frac * (t[i] - t[i - 1])
    return np.where(total > 0, out, np.nan)


def _fractional_peak_latency(t, s, peak, fraction):
    level = fraction * _take(s, peak)
    index = np.arange(s.shape[-1])
    below = (s < level[..., None]) & (index <= peak[..., None])
    has = below.any(axis=-1)
    j = s.shape[-1] - 1 - np.argmax(below[..., ::-1], axis=-1)
    j1 = np.minimum(j + 1, s.shape[-1] - 1)
    sj, sj1 = _take(s, j), _take(s, j1)
    gap = sj1 - sj
    frac = np.divide(level - sj, gap, out=np.zeros_like(gap), where=gap != 0)
    out = t[j] + frac * (t[j1] - t[j])
    out = np.where(j + 1 > peak, t[peak], out)
    return np.where(has, out, t[0])


def score_waveforms(times, waveforms, score, *, window, polarity="negative", fraction=0.5,
                    area="total", area_baseline="zero", n_neighbors=3):
    """One classical score of every waveform of an array ``[..., n_times]``.

    Parameters
    ----------
    times : ndarray, shape (n_times,)
        Time axis (s).
    waveforms : ndarray, shape (..., n_times)
        Waveforms on that axis.
    score : str
        One of :data:`SCORES`.
    window : (float, float)
        Scoring window (s), inclusive at both ends.
    polarity, fraction, area, area_baseline, n_neighbors
        As in :func:`evoparam.models.classical_measures`.

    Returns
    -------
    ndarray, shape (...)
        The score of each waveform (NaN when the window holds no sample, or for a
        fractional area latency of a zero area).

    Examples
    --------
    >>> t = np.linspace(0, 0.2, 201)
    >>> y = -np.exp(-0.5 * ((t - 0.1) / 0.01) ** 2)
    >>> round(float(score_waveforms(t, y, "peak_latency", window=(0.05, 0.15))), 3)
    0.1
    """

    if score not in SCORES:
        raise SettingsError(f"score must be one of {SCORES}, not {score!r}.")
    if polarity not in POLARITIES:
        raise SettingsError(f"polarity must be one of {POLARITIES}.")
    if area not in AREAS:
        raise SettingsError(f"area must be one of {AREAS}.")
    times = np.asarray(times, float)
    y = np.asarray(waveforms, float)
    inside = (times >= window[0]) & (times <= window[1])
    if not inside.any():
        raise SettingsError(f"the window {tuple(window)} holds no sample of the time axis.")
    t, y = times[inside], y[..., inside]
    if score == "mean_amplitude":
        return np.mean(y, axis=-1)
    s = _signed(y, polarity)
    peak = _local_peak(s, n_neighbors)
    if score == "peak_amplitude":
        return _take(y, peak)
    if score == "peak_latency":
        return t[peak]
    base = _baseline(y, peak, area_baseline, fraction)
    if score == "area":
        return _TRAPEZOID(_area_values(y, base, area), t, axis=-1)
    if score == "fractional_area_latency":
        return _fractional_area_latency(t, y, base, area, fraction)
    return _fractional_peak_latency(t, s, peak, fraction)


def default_window(times):
    """The post-stimulus part of the epoch, used when no window is given."""

    times = np.asarray(times, float)
    return (0.0, float(times[-1])) if times[-1] > 0 else (float(times[0]), float(times[-1]))


def resolve_scorer(scorer, times, window=None, **scorer_kw):
    """A function ``waveforms [..., n_times] -> scores [...]``, its name and whether it is linear.

    ``scorer`` is a name of :data:`SCORES` (scored with ``window`` and ``scorer_kw``) or a
    callable ``scorer(times, waveforms) -> scores`` that takes an array
    ``[n_waveforms, n_times]`` and returns ``[n_waveforms]`` values.
    """

    if window is None:
        window = default_window(times)
    window = (float(window[0]), float(window[1]))
    if isinstance(scorer, str):
        if scorer not in SCORES:
            raise SettingsError(f"scorer must be a callable or one of {SCORES}, "
                                f"not {scorer!r}.")

        def func(waveforms):
            return score_waveforms(times, waveforms, scorer, window=window, **scorer_kw)

        return func, scorer, scorer in LINEAR_SCORES, window
    if not callable(scorer):
        raise SettingsError("scorer must be a score name or a callable.")

    def call(waveforms):
        waveforms = np.asarray(waveforms, float)
        flat = waveforms.reshape(-1, waveforms.shape[-1])
        out = np.asarray(scorer(times, flat, **scorer_kw), float).reshape(-1)
        if out.size != flat.shape[0]:
            raise SettingsError(f"the scorer returned {out.size} values for {flat.shape[0]} "
                                "waveforms; it must return one value per row.")
        return out.reshape(waveforms.shape[:-1])

    return call, getattr(scorer, "__name__", "custom"), False, window
