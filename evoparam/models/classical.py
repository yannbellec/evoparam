"""Classical ERP measures from their textbook definitions (spec §5.5.6).

Every score records the exact definition used, so that the differing defaults of ERP
tools (fraction 0.5 or 0.3, area baseline zero or a fraction of the peak) are explicit.
"""

import numpy as np

from evoparam.errors import SettingsError

__all__ = ["classical_measures", "MEASURES"]

MEASURES = ("peak", "mean", "area", "fractional_area_latency", "fractional_peak_latency")
POLARITIES = ("negative", "positive", "absolute")
AREAS = ("total", "positive", "negative", "rectified")


def _signed(y, polarity):
    if polarity == "negative":
        return -y
    if polarity == "positive":
        return y
    return np.abs(y)


def _local_peak(t, y, polarity, n_neighbors):
    """ERPLAB-style local peak: larger than ``n_neighbors`` samples on each side (and than
    their mean); falls back to the simple extremum when no sample qualifies."""

    s = np.asarray(_signed(y, polarity), float)
    n = len(s)
    best = None
    if n_neighbors >= 1 and n > 2 * n_neighbors:
        # all samples at once (the loop over samples took a sixth of the time of a fit):
        # windows[j] holds s[j : j + n_neighbors], left of i is j = i - n_neighbors and
        # right of i is j = i + 1
        windows = np.lib.stride_tricks.sliding_window_view(s, n_neighbors)
        centre = s[n_neighbors:n - n_neighbors]
        left = windows[:n - 2 * n_neighbors]
        right = windows[n_neighbors + 1:n - n_neighbors + 1]
        ok = (centre > left.max(axis=1)) & (centre > right.max(axis=1)) \
            & (centre > left.mean(axis=1)) & (centre > right.mean(axis=1))
        if ok.any():
            # the first of the largest qualifying samples, as the loop kept
            best = n_neighbors + int(np.argmax(np.where(ok, centre, -np.inf)))
    elif n_neighbors < 1:
        for i in range(n_neighbors, n - n_neighbors):
            left, right = s[i - n_neighbors:i], s[i + 1:i + 1 + n_neighbors]
            if s[i] > left.max() and s[i] > right.max() and s[i] > left.mean() \
                    and s[i] > right.mean():
                if best is None or s[i] > s[best]:
                    best = i
    local = best is not None
    if best is None:
        best = int(np.argmax(s))
    return best, local


def _area(t, y, kind, baseline):
    values = y - baseline
    if kind == "positive":
        values = np.clip(values, 0, None)
    elif kind == "negative":
        values = np.clip(values, None, 0)
    elif kind == "rectified":
        values = np.abs(values)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(values, t))


def _fractional_area_latency(t, y, fraction, kind, baseline):
    values = y - baseline
    if kind == "positive":
        values = np.clip(values, 0, None)
    elif kind == "negative":
        values = -np.clip(values, None, 0)
    else:
        values = np.abs(values)
    cum = np.r_[0, np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(t))]
    if cum[-1] <= 0:
        return np.nan
    target = fraction * cum[-1]
    i = int(np.searchsorted(cum, target))
    i = min(max(i, 1), len(t) - 1)
    frac = (target - cum[i - 1]) / (cum[i] - cum[i - 1]) if cum[i] > cum[i - 1] else 0.0
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def _fractional_peak_latency(t, y, peak, fraction, polarity):
    s = _signed(y, polarity)
    level = fraction * s[peak]
    below = np.flatnonzero(s[:peak + 1] < level)
    if below.size == 0:
        return float(t[0])
    j = below[-1]
    if j + 1 > peak:
        return float(t[peak])
    frac = (level - s[j]) / (s[j + 1] - s[j]) if s[j + 1] != s[j] else 0.0
    return float(t[j] + frac * (t[j + 1] - t[j]))


def _scores(t, y, *, polarity, measures, fraction, area, area_baseline, n_neighbors):
    out = {}
    peak, local = _local_peak(t, y, polarity, n_neighbors)
    if area_baseline == "zero":
        base = 0.0
    elif area_baseline == "fraction_of_peak":
        base = fraction * y[peak]
    else:
        raise SettingsError("area_baseline must be 'zero' or 'fraction_of_peak'.")
    if "peak" in measures:
        out["peak_amplitude"] = float(y[peak])
        out["peak_latency"] = float(t[peak])
    if "mean" in measures:
        out["mean_amplitude"] = float(np.mean(y))
    if "area" in measures:
        out["area"] = _area(t, y, area, base)
    if "fractional_area_latency" in measures:
        out["fractional_area_latency"] = _fractional_area_latency(t, y, fraction, area, base)
    if "fractional_peak_latency" in measures:
        out["fractional_peak_latency"] = _fractional_peak_latency(t, y, peak, fraction, polarity)
    return out, local


def classical_measures(times, y, *, window, polarity="negative",
                       measures=("peak", "mean", "area", "fractional_area_latency"),
                       fraction=0.5, area="total", area_baseline="zero", n_neighbors=3,
                       trials=None, n_bootstrap=1000, random_state=None):
    """Conventional scores of one waveform in a window, with their definitions.

    - ``peak_amplitude`` / ``peak_latency``: local peak (larger than ``n_neighbors``
      samples on each side and than their means, as in ERPLAB), or the simple extremum
      when there is none; ``polarity`` negative, positive or absolute.
    - ``mean_amplitude``.
    - ``area``: ``"total"`` (signed), ``"positive"``, ``"negative"`` or ``"rectified"``,
      above a baseline of zero or of ``fraction`` times the peak.
    - ``fractional_area_latency``: time at which the area reaches ``fraction`` of itself.
    - ``fractional_peak_latency``: time before the peak at which the waveform first reaches
      ``fraction`` of the peak (onset with ``fraction=0.1``).

    With ``trials`` (array [n_trials, n_times] on the same time axis), the standardised
    measurement error ``sme`` of each score is added: ``SD over trials / sqrt(n)`` for the
    mean amplitude (analytic), a bootstrap over trials for the others.

    Returns
    -------
    dict
        Score name -> value, plus ``"definition"`` (dict score -> text) and, with
        ``trials``, ``"sme"`` (dict score -> value).
    """

    for value, allowed, name in ((polarity, POLARITIES, "polarity"), (area, AREAS, "area")):
        if value not in allowed:
            raise SettingsError(f"{name} must be one of {allowed}.")
    unknown = set(measures) - set(MEASURES)
    if unknown:
        raise SettingsError(f"unknown measures {sorted(unknown)}; known: {MEASURES}.")
    times = np.asarray(times, float)
    inside = (times >= window[0]) & (times <= window[1])
    t, w = times[inside], np.asarray(y, float)[inside]
    kwargs = dict(polarity=polarity, measures=measures, fraction=fraction, area=area,
                  area_baseline=area_baseline, n_neighbors=n_neighbors)
    scores, local = _scores(t, w, **kwargs)
    win = f"window=[{window[0]:g}, {window[1]:g}]"
    peak_def = (f"local_peak(n_neighbors={n_neighbors}), polarity={polarity}, {win}"
                + ("" if local else ", fallback=simple_extremum"))
    definition = {"peak_amplitude": peak_def, "peak_latency": peak_def,
                  "mean_amplitude": f"mean, {win}",
                  "area": f"area={area}, baseline={area_baseline}, {win}",
                  "fractional_area_latency": f"fractional_area(fraction={fraction}), area={area},"
                                             f" baseline={area_baseline}, {win}",
                  "fractional_peak_latency": f"fractional_peak(fraction={fraction}), "
                                             f"polarity={polarity}, {win}"}
    scores["definition"] = {k: v for k, v in definition.items() if k in scores}

    if trials is not None:
        trials = np.asarray(trials, float)
        n = len(trials)
        sme = {}
        if "mean_amplitude" in scores:
            sme["mean_amplitude"] = float(np.std(trials[:, inside].mean(axis=1), ddof=1)
                                          / np.sqrt(n))
        rng = np.random.default_rng(random_state)
        names = [k for k in scores if k not in ("definition", "mean_amplitude")]
        if names:
            boots = {k: [] for k in names}
            for _ in range(n_bootstrap):
                pick = rng.integers(0, n, n)
                values, _ = _scores(t, trials[pick][:, inside].mean(axis=0), **kwargs)
                for k in names:
                    boots[k].append(values[k])
            for k in names:
                sme[k] = float(np.nanstd(boots[k], ddof=1))
        scores["sme"] = sme
    return scores
