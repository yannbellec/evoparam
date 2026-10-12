# SPDX-License-Identifier: Apache-2.0
"""Baseline-permutation test: is the window different from zero (the baseline level).

Sign flips of whole trials keep each trial's time course and the correlation between
channels; under the null hypothesis (no evoked response, baseline subtracted) every trial is
as likely with either sign.
"""

import warnings

import numpy as np
from scipy.stats import t as student_t

from evoparam.errors import DataError
from evoparam.errors import warn as _warn

__all__ = [
    "sign_flips",
    "permutation_test",
    "shuffle_test",
    "max_cluster_mass",
    "tail_p_value",
    "null_band",
]

MIN_TRIALS = 5

# Sign-flip patterns processed at once (bounds memory: chunk x channels x times).
_CHUNK = 64

# Tail approximation: used when fewer than this many null values reach the observed one,
# fitted on the exceedances over this quantile of the null.
TAIL_MIN_EXCEED = 10
TAIL_QUANTILE = 0.9


def tail_p_value(null, observed):
    """Return the permutation p-value, with a generalised Pareto tail when needed.

    The tail is used when the observed value lies in the far tail of the null (Knijnenburg et al.
    2009, Bioinformatics).

    With few trials there are few sign patterns (256 for 8 trials), so the smallest
    attainable p-value, 1/256, can never survive a correction across a hundred channels.
    When fewer than 10 null values reach the observed one, a generalised Pareto
    distribution is fitted to the exceedances of the null over its 90th percentile and
    gives the tail probability; otherwise the usual ``(1 + count) / (1 + n)`` is returned.
    """
    from scipy.stats import genpareto

    null = np.asarray(null, float)
    count = int(np.sum(null >= observed - 1e-12))
    plain = (1 + count) / (1 + len(null))
    if count >= TAIL_MIN_EXCEED or len(null) < 50:
        return plain
    u = np.quantile(null, TAIL_QUANTILE)
    excess = null[null > u] - u
    if len(excess) < 10 or np.ptp(excess) <= 0:
        return plain
    try:
        shape, _, scale = genpareto.fit(excess, floc=0.0)
    except Exception:  # the fit did not converge: keep the plain estimate
        return plain
    tail = np.mean(null > u) * genpareto.sf(observed - u, shape, loc=0.0, scale=scale)
    return float(np.clip(tail, 1e-300, plain))


def sign_flips(n_trials, n_permutations, rng):
    """Random sign patterns (P x n), or every pattern when there are at most n_permutations."""
    if 2**n_trials <= n_permutations:
        grid = (np.arange(2**n_trials)[:, None] >> np.arange(n_trials)) & 1
        return 1.0 - 2.0 * grid, True
    return rng.choice([-1.0, 1.0], size=(n_permutations, n_trials)), False


def max_cluster_mass(t, threshold, polarity="both"):
    """Largest sum of |t| over a run of samples beyond ``threshold`` (same sign), per row."""

    def runs(v):
        csum = np.cumsum(v, axis=-1)
        reset = np.maximum.accumulate(np.where(v == 0, csum, 0.0), axis=-1)
        return np.max(csum - reset, axis=-1)

    pos = runs(np.where(t > threshold, t, 0.0))
    neg = runs(np.where(t < -threshold, -t, 0.0))
    if polarity == "positive":
        return pos
    if polarity == "negative":
        return neg
    return np.maximum(pos, neg)


def _summary(t, statistic, threshold, polarity):
    if statistic == "cluster":
        return max_cluster_mass(t, threshold, polarity)
    if polarity == "positive":
        return np.max(t, axis=-1)
    if polarity == "negative":
        return np.max(-t, axis=-1)
    return np.max(np.abs(t), axis=-1)


def _t_series(sum_x, sum_x2, n):
    mean = sum_x / n
    var = np.clip((sum_x2 - n * mean**2) / (n - 1), 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = mean / np.sqrt(var / n)
    return np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def _extent(t, critical, statistic, threshold, polarity, times):
    """Onset and offset (s) of the significant samples or clusters of each channel."""
    n_ch = t.shape[0]
    onset, offset = np.full(n_ch, np.nan), np.full(n_ch, np.nan)
    signed = t if polarity != "negative" else -t
    for c in range(n_ch):
        if not np.isfinite(critical[c]):
            continue
        if statistic == "max_t":
            values = np.abs(t[c]) if polarity == "both" else signed[c]
            hits = np.flatnonzero(values > critical[c])
        else:
            hits = []
            for sign in (1, -1) if polarity == "both" else (1,):
                v = np.where(sign * signed[c] > threshold, sign * signed[c], 0.0)
                edges = np.diff(np.r_[0, (v > 0).astype(int), 0])
                for a, b in zip(
                    np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=False
                ):
                    if v[a:b].sum() > critical[c]:
                        hits.extend(range(a, b))
            hits = np.array(sorted(hits))
        if len(hits):
            onset[c], offset[c] = times[hits[0]], times[hits[-1]]
    return onset, offset


def permutation_test(
    data,
    times,
    window,
    baseline,
    *,
    n_permutations=1000,
    statistic="max_t",
    cluster_threshold=None,
    polarity="both",
    alpha=0.05,
    rng=None,
    n_boot=200,
    masks=None,
):
    """Sign-flip permutation test of every channel.

    Parameters
    ----------
    data : array [n_trials, n_channels, n_times]
        Trials with the baseline subtracted.
    times : 1d array
    window, baseline : (float, float)
    statistic : {"max_t", "cluster"}
        ``max_t``: the largest |t| over the window (controls the family-wise error over
        time). ``cluster``: the largest cluster mass of |t| beyond ``cluster_threshold``
        (default the two-sided 97.5 % quantile of Student's t with n - 1 dof): more
        powerful for sustained responses.

    Returns
    -------
    dict of arrays [n_channels]: statistic, p_value, onset, offset, duration, snr_db,
    duration_ci, onset_ci and offset_ci (n_channels x 2, percentile intervals of the same
    bootstrap over trials), critical, and ``exact`` (all sign patterns enumerated).
    """
    rng = np.random.default_rng(rng)
    n, n_ch, _ = data.shape
    if n < MIN_TRIALS:
        raise DataError(
            f"The permutation test needs at least {MIN_TRIALS} trials, got {n}; "
            "use method='threshold' with a published rule instead."
        )
    win = (times >= window[0]) & (times <= window[1])
    base = (times >= baseline[0]) & (times <= baseline[1])
    x = data[:, :, win]
    usable = np.ones(x.shape[1:], bool)
    if masks is not None:
        usable = ~masks[:, :, win].any(axis=0)
    x = np.where(usable[None], np.nan_to_num(x), 0.0)
    threshold = (
        student_t.ppf(0.975, n - 1) if cluster_threshold is None else float(cluster_threshold)
    )

    sum_x, sum_x2 = x.sum(axis=0), (x**2).sum(axis=0)
    t_obs = _t_series(sum_x, sum_x2, n)
    t_obs = np.where(usable, t_obs, 0.0)
    observed = _summary(t_obs, statistic, threshold, polarity)

    flips, exact = sign_flips(n, n_permutations, rng)
    null = np.empty((len(flips), n_ch))
    flat = x.reshape(n, -1)
    for start in range(0, len(flips), _CHUNK):
        chunk = flips[start : start + _CHUNK]
        sums = (chunk @ flat).reshape(len(chunk), n_ch, -1)
        t_null = _t_series(sums, sum_x2[None], n) * usable[None]
        null[start : start + len(chunk)] = _summary(t_null, statistic, threshold, polarity)
    p = np.array([tail_p_value(null[:, c], observed[c]) for c in range(n_ch)])
    critical = np.quantile(null, 1 - alpha, axis=0)
    onset, offset = _extent(t_obs, critical, statistic, threshold, polarity, times[win])

    # Duration interval: bootstrap over trials with the same critical value (the same
    # resamples give the intervals of the onset and the offset, REQ-006)
    onsets, offsets = np.full((n_boot, n_ch), np.nan), np.full((n_boot, n_ch), np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, n, n)
        xb = x[pick]
        tb = _t_series(xb.sum(axis=0), (xb**2).sum(axis=0), n) * usable
        onsets[b], offsets[b] = _extent(tb, critical, statistic, threshold, polarity, times[win])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        def interval(v):
            return (
                np.nanpercentile(v, [2.5, 97.5], axis=0).T
                if n_boot
                else np.full((n_ch, 2), np.nan)
            )

        ci, onset_ci, offset_ci = (interval(offsets - onsets), interval(onsets), interval(offsets))

    avg = data.mean(axis=0)
    noise = np.var(avg[:, base], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = 10 * np.log10(np.mean(avg[:, win] ** 2, axis=1) / noise)
    all_masked = ~usable.any(axis=1)
    p = np.where(all_masked, np.nan, p)
    # The statistic over time and its null band (spec §5.13, plot_existence): the pointwise t
    # of the trials and the level it must cross, the family-wise critical value of the
    # null maximum |t| (max_t) or the cluster-forming threshold (cluster)
    level = critical if statistic == "max_t" else np.full(n_ch, threshold)
    return {
        "statistic": observed,
        "p_value": p,
        "onset": onset,
        "offset": offset,
        "duration": offset - onset,
        "snr_db": snr,
        "duration_ci": ci,
        "onset_ci": onset_ci,
        "offset_ci": offset_ci,
        "critical": critical,
        "exact": exact,
        "threshold": threshold,
        "curve": np.where(usable, t_obs, np.nan),
        "curve_times": times[win],
        "null_band": null_band(level, polarity, int(win.sum())),
    }


def null_band(level, polarity, n_times):
    """Return the null band [n_channels, n_times, 2] (low, high) of a statistic over time.

    From one level per channel: (-level, level) for a two-sided test, NaN on the side a one-sided
    test does not look at.
    """
    level = np.asarray(level, float).reshape(-1)
    low = np.where(polarity in ("both", "negative"), -level, np.nan)
    high = np.where(polarity in ("both", "positive"), level, np.nan)
    band = np.stack([low, high], axis=-1)[:, None, :]
    return np.repeat(band, n_times, axis=1)


def shuffle_test(
    avg, times, window, baseline, *, n_permutations=1000, polarity="both", alpha=0.05, rng=None
):
    """Test for an average without trials.

    The largest |z| of the window against windows of the same length drawn from the baselines of
    every channel (a weaker null).
    """
    rng = np.random.default_rng(rng)
    _warn(
        "No trials: the null distribution comes from baseline windows of the "
        "average (weaker than a trial permutation).",
        UserWarning,
        stacklevel=3,
    )
    win = (times >= window[0]) & (times <= window[1])
    base = (times >= baseline[0]) & (times <= baseline[1])
    if base.sum() < 2:
        raise DataError(
            f"The baseline {tuple(baseline)} has {int(base.sum())} sample(s) in "
            f"the data ({times[0]:.4f} to {times[-1]:.4f} s); the test needs at "
            "least 2."
        )
    mu = avg[:, base].mean(axis=1, keepdims=True)
    sd = avg[:, base].std(axis=1, keepdims=True)
    z = (avg - mu) / np.where(sd > 0, sd, np.nan)

    def stat(seg):
        return (
            np.nanmax(np.abs(seg), axis=-1)
            if polarity == "both"
            else np.nanmax(seg if polarity == "positive" else -seg, axis=-1)
        )

    observed = stat(z[:, win])
    n_win, zb = int(win.sum()), z[:, base]
    if zb.shape[1] < n_win:
        _warn(
            "The baseline is shorter than the window: null windows wrap around.",
            UserWarning,
            stacklevel=3,
        )
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        ch = rng.integers(0, len(z))
        start = rng.integers(0, zb.shape[1])
        idx = (start + np.arange(n_win)) % zb.shape[1]
        null[i] = stat(zb[ch, idx][None])[0]
    p = np.array([tail_p_value(null, value) for value in observed])
    critical = np.quantile(null, 1 - alpha)
    onset, offset = np.full(len(z), np.nan), np.full(len(z), np.nan)
    for c in range(len(z)):
        values = (
            np.abs(z[c, win])
            if polarity == "both"
            else z[c, win] * (1 if polarity == "positive" else -1)
        )
        hits = np.flatnonzero(values > critical)
        if len(hits):
            onset[c], offset[c] = times[win][hits[0]], times[win][hits[-1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = 10 * np.log10(np.mean(avg[:, win] ** 2, axis=1) / np.var(avg[:, base], axis=1))
    return {
        "statistic": observed,
        "p_value": p,
        "onset": onset,
        "offset": offset,
        "duration": offset - onset,
        "snr_db": snr,
        "duration_ci": np.full((len(z), 2), np.nan),
        "critical": np.full(len(z), critical),
        "exact": False,
        "threshold": np.nan,
        "curve": z[:, win],
        "curve_times": times[win],
        "null_band": null_band(np.full(len(z), critical), polarity, int(win.sum())),
    }
