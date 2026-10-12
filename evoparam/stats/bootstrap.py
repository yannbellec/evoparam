# SPDX-License-Identifier: Apache-2.0
"""Bootstrap over trials: percentile and bias-corrected accelerated (BCa) intervals (spec §5.11).

Trials are resampled with replacement within each stratum (``by``: stimulation site,
condition), so the size of every stratum is the same in every resample. The statistic is
any callable of a :class:`~evoparam.data.TrialSet` (a model refit, a zone metric), or the
name of a classical score, which is then computed on the average of the resampled trials
for every channel at once (all resamples in a few matrix products).

The BCa interval (Efron 1987) moves the percentiles by the bias correction
``z0 = Phi^-1(share of resamples below the estimate)`` and the acceleration ``a``, the
skewness of the leave-one-trial-out (jackknife) values; with ``a = z0 = 0`` it is the
percentile interval.
"""

import time as _time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

from evoparam.data import Evoked, TrialSet
from evoparam.errors import DataError, SettingsError
from evoparam.errors import warn as _warn
from evoparam.utils._optional import require

from ._scorers import TIME_SCORES, default_window, resolve_scorer

__all__ = ["bootstrap", "bootstrap_interval", "BOOTSTRAP_COLUMNS"]

METHODS = ("percentile", "bca")

# Fewer distinct resample values than this: the BCa corrections are unreliable.
MIN_UNIQUE = 20

# Resamples averaged at once by the score path (bounds memory: chunk x trials).
_CHUNK = 256

BOOTSTRAP_COLUMNS = {
    "channel": "",
    "statistic": "",
    "estimate": "stat",
    "bias": "stat",
    "se": "stat",
    "ci_low": "stat",
    "ci_high": "stat",
    "ci_level": "",
    "method": "",
    "n_boot": "",
    "n_trials": "",
    "z0": "",
    "acceleration": "",
}


###################################################################################################
## Intervals


def _quantile_z(ci_level, expand_n):
    """Lower and upper standard-normal quantiles of the interval, expanded for small samples."""
    alpha = (1 - ci_level) / 2
    if expand_n is None:
        return norm.ppf([alpha, 1 - alpha])
    n = float(expand_n)
    if n < 2:
        raise SettingsError("expand_n must be at least 2.")
    # Hesterberg (2015): the percentile interval of a mean is too narrow by
    # sqrt((n - 1) / n) and uses z instead of t; use sqrt(n / (n - 1)) * t_{n-1} instead.
    q = np.sqrt(n / (n - 1)) * student_t.ppf(1 - alpha, n - 1)
    return np.array([-q, q])


def bootstrap_interval(
    estimate, boot, *, jackknife=None, ci_level=0.95, method="bca", expand_n=None
):
    """Percentile or BCa interval of one statistic from its bootstrap values.

    Parameters
    ----------
    estimate : float
        Statistic on the original data.
    boot : array-like
        Statistic on each resample (non-finite values are left out).
    jackknife : array-like, optional
        Statistic with each resampling unit left out in turn (for the acceleration;
        without it the acceleration is 0, the bias-corrected interval).
    ci_level : float
        Coverage of the interval.
    method : {"bca", "percentile"}
    expand_n : int, optional
        Sample size for the small-sample expansion (Hesterberg 2015): the normal quantile
        ``z`` of the interval is replaced by ``sqrt(n / (n - 1)) * t_{n-1}``. With
        ``n = 30`` and 95 %, the percentiles 1.9 % and 98.1 % are taken instead of 2.5 %
        and 97.5 %. None: the textbook intervals.

    Returns
    -------
    dict
        ``low``, ``high``, ``method`` (the one used: ``"percentile"`` when BCa had to fall
        back), ``z0``, ``acceleration``, ``reason`` (why it fell back, else "").

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=40)
    >>> boot = [rng.choice(x, 40).mean() for _ in range(2000)]
    >>> out = bootstrap_interval(x.mean(), boot, method="percentile")
    >>> bool(out["low"] < x.mean() < out["high"])
    True
    """
    jack = None if jackknife is None else np.asarray(jackknife, float).reshape(-1, 1)
    out = _intervals(
        np.atleast_1d(np.asarray(estimate, float)),
        np.asarray(boot, float).reshape(-1, 1),
        jack,
        ci_level,
        method,
        expand_n,
    )
    first = {key: value[0] for key, value in out.items()}
    return {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in first.items()
    }


def _sorted_quantile(sorted_boot, n_finite, levels):
    """Return linear-interpolation quantiles (as numpy's default) of each column.

    The array is column-sorted and its first ``n_finite`` entries are finite; ``levels`` [n_levels,
    n_columns].
    """
    pos = levels * (n_finite - 1)
    lo = np.clip(np.floor(pos).astype(np.int64), 0, None)
    hi = np.minimum(lo + 1, np.maximum(n_finite - 1, 0))
    frac = pos - lo
    cols = np.arange(sorted_boot.shape[1])
    a, b = sorted_boot[lo, cols], sorted_boot[hi, cols]
    return a + frac * (b - a)


def _intervals(estimate, boot, jack, ci_level, method, expand_n):
    """Percentile or BCa intervals of every column of ``boot`` [n_boot, k] at once."""
    if method not in METHODS:
        raise SettingsError(f"method must be one of {METHODS}, not {method!r}.")
    if not 0 < ci_level < 1:
        raise SettingsError("ci_level must be in (0, 1).")
    k = boot.shape[1]
    z = _quantile_z(ci_level, expand_n)
    finite = np.isfinite(boot)
    n_finite = finite.sum(axis=0)
    ordered = np.sort(np.where(finite, boot, np.inf), axis=0)
    usable = (n_finite >= 2) & np.isfinite(estimate)
    levels = np.repeat(norm.cdf(z)[:, None], k, axis=1)
    used = np.array([method] * k, dtype=object)
    reason = np.array([""] * k, dtype=object)
    z0 = np.full(k, np.nan)
    acc = np.full(k, np.nan)
    reason[~usable] = "fewer than 2 finite resample values"
    if method == "bca":
        steps = np.diff(ordered, axis=0) != 0
        index = np.arange(1, ordered.shape[0])[:, None]
        n_unique = 1 + np.sum(steps & (index < n_finite), axis=0)
        few = usable & (n_unique < MIN_UNIQUE)
        reason[few] = f"fewer than {MIN_UNIQUE} distinct resample values"
        below = np.sum(finite & (boot < estimate), axis=0)
        equal = np.sum(finite & (boot == estimate), axis=0)
        share = np.divide(
            below + 0.5 * equal, n_finite, out=np.full(k, np.nan), where=n_finite > 0
        )
        z0_all = norm.ppf(share)
        side = usable & ~few & ~np.isfinite(z0_all)
        reason[side] = "every resample value lies on one side of the estimate"
        ok = usable & ~few & ~side
        used[usable & ~ok] = "percentile"
        a = np.zeros(k)
        if jack is not None:
            jfin = np.isfinite(jack)
            count = np.maximum(jfin.sum(axis=0), 1)
            mean = np.where(jfin, jack, 0.0).sum(axis=0) / count
            d = np.where(jfin, mean - jack, 0.0)
            denom = 6.0 * np.sum(d**2, axis=0) ** 1.5
            a = np.divide(np.sum(d**3, axis=0), denom, out=np.zeros(k), where=denom > 0)
        zz = z0_all[None, :] + z[:, None]
        bca_levels = norm.cdf(z0_all[None, :] + zz / (1 - a[None, :] * zz))
        levels = np.where(ok[None, :], bca_levels, levels)
        z0[ok], acc[ok] = z0_all[ok], a[ok]
    bounds = _sorted_quantile(ordered, n_finite, levels)
    bounds[:, ~usable] = np.nan
    return {
        "low": bounds[0],
        "high": bounds[1],
        "method": used,
        "z0": z0,
        "acceleration": acc,
        "reason": reason,
    }


###################################################################################################
## Resampling


def _strata(ts, by):
    """Row indices of each stratum (one stratum when no ``by`` column is in the metadata)."""
    meta = ts.metadata
    columns = [c for c in (by or ()) if meta is not None and c in meta.columns]
    if not columns:
        return [np.arange(ts.n_trials)]
    keys = meta[columns].astype(str).agg("/".join, axis=1).to_numpy()
    return [np.flatnonzero(keys == key) for key in dict.fromkeys(keys)]


def _indices(strata, n_boot, rng):
    """Resampled trial indices [n_boot, n_trials], drawn within each stratum."""
    parts = [rows[rng.integers(0, len(rows), (n_boot, len(rows)))] for rows in strata]
    return np.concatenate(parts, axis=1)


def _subset(ts, rows):
    """Return the trials ``rows`` of a TrialSet (same provenance: a resample is not a new step)."""
    meta = None if ts.metadata is None else ts.metadata.iloc[rows].reset_index(drop=True)
    return replace(
        ts,
        data=ts.data[rows],
        metadata=meta,
        masks=None if ts.masks is None else ts.masks[rows],
        event_times=None if ts.event_times is None else ts.event_times[rows],
    )


def _as_vector(value):
    """Statistic value -> (labels, values)."""
    if hasattr(value, "index") and hasattr(value, "to_numpy"):  # pandas Series
        return [str(k) for k in value.index], np.asarray(value.to_numpy(), float).ravel()
    arr = np.asarray(value, float)
    if arr.ndim == 0:
        return ["value"], arr.reshape(1)
    arr = arr.ravel()
    return [str(i) for i in range(arr.size)], arr


def _score_path(ts, func, strata, n_boot, rng, inside):
    """Scores of the average of every resample, every channel.

    [n_boot, n_channels] (``func`` scores waveforms cropped to the samples ``inside`` the window).
    """
    x = ts.data[:, :, inside]
    if ts.masks is not None:
        x = np.where(ts.masks[:, :, inside], np.nan, x)
    n, n_ch, n_t = x.shape
    flat = x.reshape(n, -1)
    finite = np.isfinite(flat)
    values = np.where(finite, flat, 0.0)
    count = finite.astype(float)

    def average(weights):
        num, den = weights @ values, weights @ count
        avg = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)
        return avg.reshape(len(weights), n_ch, n_t)

    estimate = func(average(np.ones((1, n)))[0])
    index = _indices(strata, n_boot, rng)
    boot = np.empty((n_boot, n_ch))
    for start in range(0, n_boot, _CHUNK):
        rows = index[start : start + _CHUNK]
        weights = np.zeros((len(rows), n))
        np.add.at(weights, (np.repeat(np.arange(len(rows)), n), rows.ravel()), 1.0)
        boot[start : start + _CHUNK] = func(average(weights))
    jack = np.empty((n, n_ch))
    for start in range(0, n, _CHUNK):
        stop = min(start + _CHUNK, n)
        weights = np.ones((stop - start, n))
        weights[np.arange(stop - start), np.arange(start, stop)] = 0.0
        jack[start:stop] = func(average(weights))
    return np.atleast_1d(estimate), boot, jack


def _callable_path(ts, statistic, strata, n_boot, rng, n_jobs):
    labels, estimate = _as_vector(statistic(ts))
    index = _indices(strata, n_boot, rng)

    def one(rows):
        lab, val = _as_vector(statistic(_subset(ts, rows)))
        if val.size != estimate.size:
            raise DataError(
                f"bootstrap: the statistic returned {val.size} values on a "
                f"resample and {estimate.size} on the data."
            )
        return val

    loo = [np.delete(np.arange(ts.n_trials), i) for i in range(ts.n_trials)]
    if n_jobs and n_jobs != 1:
        with ThreadPoolExecutor(max_workers=None if n_jobs < 0 else n_jobs) as pool:
            boot = np.array(list(pool.map(one, index)))
            jack = np.array(list(pool.map(one, loo)))
    else:
        boot = np.array([one(rows) for rows in index])
        jack = np.array([one(rows) for rows in loo])
    return labels, estimate, boot.reshape(n_boot, -1), jack.reshape(len(loo), -1)


###################################################################################################
## Public function


def bootstrap(
    ts,
    statistic,
    *,
    n_boot=1000,
    by=("stim_site",),
    ci_level=0.95,
    method="bca",
    random_state=None,
    n_jobs=1,
    window=None,
    expand=True,
    **scorer_kw,
):
    """Bootstrap interval of any statistic of the trials.

    Parameters
    ----------
    ts : TrialSet
        The trials.
    statistic : callable or str
        ``statistic(trialset) -> float | ndarray | Series`` computed on each resample (a
        model refit, a zone metric, ...), or the name of a classical score
        (``"mean_amplitude"``, ``"peak_amplitude"``, ``"peak_latency"``, ``"area"``,
        ``"fractional_area_latency"``, ``"fractional_peak_latency"``) computed on the
        average of the resampled trials of every channel, in ``window`` with the options
        ``scorer_kw`` of :func:`evoparam.models.classical_measures` (``polarity``,
        ``fraction``, ...).
    n_boot : int
        Number of resamples.
    by : sequence of str
        Metadata columns whose groups are resampled separately (their sizes stay fixed);
        columns absent from the metadata are ignored.
    ci_level : float
        Coverage of the interval.
    method : {"bca", "percentile"}
        BCa falls back to percentile, with a warning, when fewer than 20 distinct
        resample values exist.
    random_state : int, Generator or None
    n_jobs : int
        Threads for a callable statistic (-1: all cores).
    window : (float, float), optional
        Scoring window (s) of a named score; default the post-stimulus epoch.
    expand : bool
        Small-sample expansion of the interval (Hesterberg 2015; see
        :func:`bootstrap_interval`), with ``n`` the number of trials minus the number of
        strata plus one. The textbook intervals are too narrow with few trials: with 30
        trials, 0.936 coverage of a normal mean and 0.925 of a log-normal one (CV 0.5)
        at 95 %; expanded, 0.95 and 0.94. False gives the textbook intervals.

    Returns
    -------
    DataFrame
        One row per value of the statistic (per channel for a named score): ``channel``,
        ``statistic``, ``estimate``, ``bias`` (mean of the resamples minus the estimate),
        ``se`` (SD of the resamples), ``ci_low``, ``ci_high``, ``ci_level``, ``method``
        (the one used), ``n_boot`` (finite resamples), ``n_trials``, ``z0``,
        ``acceleration``. ``attrs`` holds ``units``, ``settings`` and ``provenance``.

    Examples
    --------
    >>> from evoparam.sim import simulate_erp
    >>> ts, _ = simulate_erp(30, components=["N1"], snr=3., noise="white", seed=0)
    >>> table = bootstrap(ts, "mean_amplitude", window=(0.08, 0.12), n_boot=200,
    ...                   random_state=0)
    >>> table[["channel", "statistic", "method"]].values.tolist()
    [['Cz', 'mean_amplitude', 'bca']]
    """
    start = _time.perf_counter()
    pd = require("pandas", extra="data", caller="evoparam.stats.bootstrap")
    if isinstance(ts, Evoked):
        raise DataError("bootstrap resamples trials: give a TrialSet, not an Evoked.")
    if not isinstance(ts, TrialSet):
        raise DataError("bootstrap expects a TrialSet.")
    if method not in METHODS:
        raise SettingsError(f"method must be one of {METHODS}, not {method!r}.")
    n_boot = int(n_boot)
    if n_boot < 2:
        raise SettingsError("n_boot must be at least 2.")
    if ts.n_trials < 2:
        raise DataError("bootstrap needs at least 2 trials.")
    rng = np.random.default_rng(random_state)
    strata = _strata(ts, by)

    unit = "stat"
    if isinstance(statistic, str):
        window = default_window(ts.times) if window is None else window
        inside = (ts.times >= window[0]) & (ts.times <= window[1])
        func, name, _, window = resolve_scorer(statistic, ts.times[inside], window, **scorer_kw)
        estimate, boot, jack = _score_path(ts, func, strata, n_boot, rng, inside)
        channels, labels = list(ts.ch_names), [name] * ts.n_channels
        unit = "s" if name in TIME_SCORES else ("data·s" if name == "area" else ts.units)
    elif callable(statistic):
        labels, estimate, boot, jack = _callable_path(ts, statistic, strata, n_boot, rng, n_jobs)
        channels = [""] * len(labels)
    else:
        raise SettingsError("statistic must be a callable or the name of a classical score.")

    # small-sample expansion: n trials, minus one degree of freedom per extra stratum
    expand_n = max(ts.n_trials - len(strata) + 1, 2) if expand else None
    ci = _intervals(estimate, boot, jack, ci_level, method, expand_n)
    finite = np.isfinite(boot)
    n_finite = finite.sum(axis=0)
    total = np.where(finite, boot, 0.0).sum(axis=0)
    mean = np.divide(total, n_finite, out=np.full(len(estimate), np.nan), where=n_finite > 0)
    dev = np.where(finite, boot - mean, 0.0)
    sd = np.sqrt(
        np.divide(
            np.sum(dev**2, axis=0),
            n_finite - 1,
            out=np.full(len(estimate), np.nan),
            where=n_finite > 1,
        )
    )
    fallbacks = [
        f"{c or lab}: {why}"
        for c, lab, why, used in zip(channels, labels, ci["reason"], ci["method"], strict=False)
        if why and method == "bca" and used == "percentile"
    ]
    if fallbacks:
        _warn(
            "bootstrap: BCa fell back to the percentile interval for "
            + "; ".join(fallbacks[:5])
            + (" ..." if len(fallbacks) > 5 else ""),
            UserWarning,
            stacklevel=2,
        )
    rows = {
        "channel": channels,
        "statistic": labels,
        "estimate": estimate,
        "bias": mean - estimate,
        "se": sd,
        "ci_low": ci["low"],
        "ci_high": ci["high"],
        "ci_level": float(ci_level),
        "method": list(ci["method"]),
        "n_boot": n_finite.astype(np.int64),
        "n_trials": int(ts.n_trials),
        "z0": ci["z0"],
        "acceleration": ci["acceleration"],
    }
    table = pd.DataFrame(rows, columns=list(BOOTSTRAP_COLUMNS))
    settings = {
        "statistic": statistic
        if isinstance(statistic, str)
        else getattr(statistic, "__name__", "callable"),
        "n_boot": n_boot,
        "by": list(by or ()),
        "ci_level": ci_level,
        "method": method,
        "random_state": random_state
        if isinstance(random_state, (int, type(None)))
        else "generator",
        "window": None if window is None else list(window),
        "n_strata": len(strata),
        "expand": bool(expand),
        **scorer_kw,
    }
    table.attrs["units"] = {k: (unit if v == "stat" else v) for k, v in BOOTSTRAP_COLUMNS.items()}
    table.attrs["settings"] = settings
    table.attrs["provenance"] = ts.provenance.with_step(
        "evoparam.stats.bootstrap", settings, duration_s=_time.perf_counter() - start
    ).to_dict()
    return table
