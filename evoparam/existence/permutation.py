# SPDX-License-Identifier: Apache-2.0
"""Baseline-permutation test: is the window different from zero (the baseline level).

Sign flips of whole trials keep each trial's time course and the correlation between
channels; under the null hypothesis (no evoked response, baseline subtracted) every trial is
as likely with either sign.
"""

import logging
import warnings

import numpy as np
from scipy.stats import t as student_t

from evoparam._typing import Any, FloatArray, NDArray
from evoparam.errors import DataError
from evoparam.errors import warn as _warn
from evoparam.utils._numeric import divide, log10
from evoparam.utils._parallel import n_workers, parallel_map

logger = logging.getLogger(__name__)

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
# Bytes of the t series of the null computed at once (a block of channels of one chunk).
_NULL_BLOCK_BYTES = 4 * 2**20

# Tail approximation: used when fewer than this many null values reach the observed one,
# fitted on the exceedances over this quantile of the null.
TAIL_MIN_EXCEED = 10
TAIL_QUANTILE = 0.9


def tail_p_value(null: Any, observed: Any) -> float:
    """Return the permutation p-value, with a generalised Pareto tail when needed.

    The tail is used when the observed value lies in the far tail of the null (Knijnenburg et al.
    2009, Bioinformatics).

    With few trials there are few sign patterns (256 for 8 trials), so the smallest
    attainable p-value, 1/256, can never survive a correction across a hundred channels.
    When fewer than 10 null values reach the observed one, a generalised Pareto
    distribution is fitted to the exceedances of the null over its 90th percentile and
    gives the tail probability; otherwise the usual ``(1 + count) / (1 + n)`` is returned.
    When the Pareto fit fails, the usual value is returned and a DEBUG record of the
    ``evoparam.existence.permutation`` logger names the error.
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
    except (RuntimeError, ValueError, ArithmeticError) as err:
        # the fit did not converge (scipy's FitError is a RuntimeError): the plain
        # estimate is kept, and the failure is logged (REQ-307: never without a trace)
        logger.debug(
            "tail_p_value: the generalised Pareto fit failed (%s: %s); the plain "
            "permutation p-value %.4g is kept.",
            type(err).__name__,
            err,
            plain,
        )
        return plain
    tail = np.mean(null > u) * genpareto.sf(observed - u, shape, loc=0.0, scale=scale)
    return float(np.clip(tail, 1e-300, plain))


def sign_flips(n_trials: int, n_permutations: Any, rng: np.random.Generator) -> Any:
    """Random sign patterns (P x n), or every pattern when there are at most n_permutations."""
    if 2**n_trials <= n_permutations:
        grid = (np.arange(2**n_trials)[:, None] >> np.arange(n_trials)) & 1
        return 1.0 - 2.0 * grid, True
    return rng.choice([-1.0, 1.0], size=(n_permutations, n_trials)), False


def max_cluster_mass(t: Any, threshold: Any, polarity: str = "both") -> Any:
    """Largest sum of |t| over a run of samples beyond ``threshold`` (same sign), per row."""

    def runs(v: Any) -> Any:
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


def _summary(t: Any, statistic: Any, threshold: Any, polarity: Any) -> Any:
    if statistic == "cluster":
        return max_cluster_mass(t, threshold, polarity)
    if polarity == "positive":
        return np.max(t, axis=-1)
    if polarity == "negative":
        return np.max(-t, axis=-1)
    return np.max(np.abs(t), axis=-1)


def _t_series(sum_x: Any, sum_x2: Any, n: Any) -> Any:
    mean = sum_x / n
    var = np.clip((sum_x2 - n * mean**2) / (n - 1), 0, None)
    t = divide(mean, np.sqrt(divide(var, n)))
    return np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def _extent(
    t: Any, critical: Any, statistic: Any, threshold: Any, polarity: Any, times: Any
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Onset and offset (s) of the significant samples or clusters of each channel."""
    n_ch = t.shape[0]
    onset, offset = np.full(n_ch, np.nan), np.full(n_ch, np.nan)
    signed = t if polarity != "negative" else -t
    for c in range(n_ch):
        if not np.isfinite(critical[c]):
            continue
        if statistic == "max_t":
            values = np.abs(t[c]) if polarity == "both" else signed[c]
            hits: Any
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


def _channel_block(block: tuple[Any, ...]) -> tuple[FloatArray, FloatArray, FloatArray]:
    """p-values and bootstrap onsets and offsets of a block of channels (one unit of n_jobs)."""
    x, usable, null, observed, critical, picks, statistic, threshold, polarity, tw, n = block
    p = np.array([tail_p_value(null[:, c], observed[c]) for c in range(len(observed))])
    onsets = np.full((len(picks), len(observed)), np.nan)
    offsets = np.full((len(picks), len(observed)), np.nan)
    # the resampled t series of a few channels at a time (each value is computed per channel
    # and sample, so the blocks change no result; the memory of P3, spec §9.1)
    step = max(1, _NULL_BLOCK_BYTES // max(1, 8 * x.shape[0] * x.shape[2]))
    tb = np.empty(x.shape[1:])
    for b, pick in enumerate(picks):
        for first in range(0, x.shape[1], step):
            c = slice(first, first + step)
            xb = x[:, c][pick]
            tb[c] = _t_series(xb.sum(axis=0), (xb**2).sum(axis=0), n) * usable[c]
        onsets[b], offsets[b] = _extent(tb, critical, statistic, threshold, polarity, tw)
    return p, onsets, offsets


def _null_maxima(
    x: Any, flips: Any, sum_x2: Any, usable: Any, statistic: Any, threshold: Any, polarity: Any
) -> FloatArray:
    """Summary of the t series of every sign pattern of ``flips`` [n_patterns, n_channels].

    The sums of a chunk of patterns go to one buffer and their t series are computed a few
    channels at a time: every value is computed per pattern, channel and sample, so the
    blocks change no result, and the temporaries stay small (the memory of P3, spec §9.1).
    """
    n, n_ch = x.shape[:2]
    null = np.empty((len(flips), n_ch))
    flat = x.reshape(n, -1)
    buffer = np.empty((min(_CHUNK, len(flips)), flat.shape[1]))
    step = max(1, _NULL_BLOCK_BYTES // max(1, 8 * buffer.shape[0] * x.shape[2]))
    for start in range(0, len(flips), _CHUNK):
        chunk = flips[start : start + _CHUNK]
        sums = np.matmul(chunk, flat, out=buffer[: len(chunk)]).reshape(len(chunk), n_ch, -1)
        for first in range(0, n_ch, step):
            c = slice(first, first + step)
            t_null = _t_series(sums[:, c], sum_x2[None, c], n) * usable[None, c]
            null[start : start + len(chunk), c] = _summary(t_null, statistic, threshold, polarity)
    return null


def permutation_test(
    data: FloatArray,
    times: FloatArray,
    window: tuple[float, float],
    baseline: tuple[float, float],
    *,
    n_permutations: int = 1000,
    statistic: str = "max_t",
    cluster_threshold: Any = None,
    polarity: str = "both",
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    n_boot: int = 200,
    masks: Any = None,
    n_jobs: int | None = 1,
) -> dict[str, Any]:
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
    n_jobs : int
        Workers for the p-values and the bootstrap, per block of channels (the random
        numbers are drawn before, so the result does not depend on ``n_jobs``).

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
    # the samples of the window (a copy, cleaned in place), 0 where masked
    x = np.nan_to_num(data[:, :, win], copy=False)
    usable = np.ones(x.shape[1:], bool)
    if masks is not None:
        usable = ~masks[:, :, win].any(axis=0)
        x = np.where(usable[None], x, 0.0)
    threshold = (
        student_t.ppf(0.975, n - 1) if cluster_threshold is None else float(cluster_threshold)
    )

    sum_x, sum_x2 = x.sum(axis=0), (x**2).sum(axis=0)
    t_obs = _t_series(sum_x, sum_x2, n)
    t_obs = np.where(usable, t_obs, 0.0)
    observed = _summary(t_obs, statistic, threshold, polarity)

    flips, exact = sign_flips(n, n_permutations, rng)
    null = _null_maxima(x, flips, sum_x2, usable, statistic, threshold, polarity)
    critical = np.quantile(null, 1 - alpha, axis=0)
    onset, offset = _extent(t_obs, critical, statistic, threshold, polarity, times[win])

    # Duration interval: bootstrap over trials with the same critical value (the same
    # resamples give the intervals of the onset and the offset, REQ-006). The resamples
    # are drawn here, in the order of the serial loop, so that the result does not depend
    # on n_jobs; the p-values and the bootstrap run per block of channels.
    picks = [rng.integers(0, n, n) for _ in range(n_boot)]
    common = (picks, statistic, threshold, polarity, times[win], n)
    workers = n_workers(n_jobs, n_ch)
    columns: list[Any]
    if workers == 1:
        blocks = [(x, usable, null, observed, critical, *common)]
        columns = [slice(None)]
    else:
        columns = list(np.array_split(np.arange(n_ch), workers))
        blocks = [
            (x[:, c], usable[c], null[:, c], observed[c], critical[c], *common) for c in columns
        ]
    p = np.empty(n_ch)
    onsets, offsets = np.full((n_boot, n_ch), np.nan), np.full((n_boot, n_ch), np.nan)
    for c, (p_c, on_c, off_c) in zip(
        columns,
        parallel_map(_channel_block, blocks, workers, desc="test_response"),
        strict=True,
    ):
        p[c], onsets[:, c], offsets[:, c] = p_c, on_c, off_c
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        def interval(v: Any) -> Any:
            return (
                np.nanpercentile(v, [2.5, 97.5], axis=0).T
                if n_boot
                else np.full((n_ch, 2), np.nan)
            )

        ci, onset_ci, offset_ci = (interval(offsets - onsets), interval(onsets), interval(offsets))

    avg = data.mean(axis=0)
    noise = np.var(avg[:, base], axis=1)
    snr = 10 * log10(divide(np.mean(avg[:, win] ** 2, axis=1), noise))
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


def null_band(level: Any, polarity: Any, n_times: int) -> NDArray[Any]:
    """Return the null band [n_channels, n_times, 2] (low, high) of a statistic over time.

    From one level per channel: (-level, level) for a two-sided test, NaN on the side a one-sided
    test does not look at.
    """
    level = np.asarray(level, float).reshape(-1)
    low = np.where(polarity in ("both", "negative"), -level, np.nan)
    high = np.where(polarity in ("both", "positive"), level, np.nan)
    band = np.stack([low, high], axis=-1)[:, None, :]
    return np.repeat(band, n_times, axis=1)


def _p_values(block: tuple[FloatArray, FloatArray]) -> FloatArray:
    """p-values of a block of observed values against one null (one unit of n_jobs)."""
    null, observed = block
    return np.array([tail_p_value(null, value) for value in observed], dtype=float)


def _baseline_bootstrap(
    zw: Any,
    zb: Any,
    critical: float,
    polarity: str,
    tw: Any,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    """Bootstrap onsets and offsets [n_boot, n_channels] of :func:`shuffle_test`.

    The z-scored window ``zw`` of each channel plus a circular segment of its baseline
    ``zb``; NaN where a resample never crosses ``critical``.
    """
    n_ch, n_win = zw.shape
    onsets, offsets = np.full((n_boot, n_ch), np.nan), np.full((n_boot, n_ch), np.nan)
    if not n_boot or zb.shape[1] == 0:
        return onsets, offsets
    steps = np.arange(n_win)
    for c in range(n_ch):
        starts = rng.integers(0, zb.shape[1], n_boot)
        noisy = zw[c][None] + zb[c][(starts[:, None] + steps[None]) % zb.shape[1]]
        values = (
            np.abs(noisy) if polarity == "both" else noisy * (1 if polarity == "positive" else -1)
        )
        above = values > critical
        crossed = above.any(axis=1)
        first = np.argmax(above, axis=1)
        last = n_win - 1 - np.argmax(above[:, ::-1], axis=1)
        onsets[crossed, c] = tw[first[crossed]]
        offsets[crossed, c] = tw[last[crossed]]
    return onsets, offsets


def shuffle_test(  # noqa: PLR0913  (signature shared by the existence tests, with n_jobs)
    avg: Any,
    times: Any,
    window: Any,
    baseline: Any,
    *,
    n_permutations: int = 1000,
    polarity: str = "both",
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    n_jobs: int | None = 1,
    n_boot: int = 200,
) -> dict[str, Any]:
    """Test for an average without trials.

    The largest |z| of the window against windows of the same length drawn from the baselines of
    every channel (a weaker null). ``n_jobs``: workers for the p-values, per block of
    channels.

    Intervals of the onset, offset and duration (REQ-006): with no trials to resample, a
    baseline bootstrap. Each of ``n_boot`` resamples adds to the z-scored window of a
    channel a segment of the same length of its own z-scored baseline (a random circular
    start), and the onset and offset are found again with the same critical value; the
    intervals are the 2.5 and 97.5 percentiles. The observed window already holds one
    noise realisation, so the intervals count the noise twice: they are conservative.
    The resamples are drawn after the null, so the p-values do not change with them.
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

    def stat(seg: Any) -> Any:
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
    workers = n_workers(n_jobs, len(observed))
    blocks = [(null, part) for part in np.array_split(observed, workers)]
    p = np.concatenate(parallel_map(_p_values, blocks, workers, desc="test_response"))
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
    onsets, offsets = _baseline_bootstrap(
        z[:, win], zb, critical, polarity, times[win], n_boot, rng
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a channel never above the level

        def interval(v: Any) -> Any:
            return (
                np.nanpercentile(v, [2.5, 97.5], axis=0).T
                if n_boot
                else np.full((len(z), 2), np.nan)
            )

        duration_ci, onset_ci, offset_ci = (
            interval(offsets - onsets),
            interval(onsets),
            interval(offsets),
        )
    snr = 10 * log10(divide(np.mean(avg[:, win] ** 2, axis=1), np.var(avg[:, base], axis=1)))
    return {
        "statistic": observed,
        "p_value": p,
        "onset": onset,
        "offset": offset,
        "duration": offset - onset,
        "snr_db": snr,
        "duration_ci": duration_ci,
        "onset_ci": onset_ci,
        "offset_ci": offset_ci,
        "critical": np.full(len(z), critical),
        "exact": False,
        "threshold": np.nan,
        "curve": z[:, win],
        "curve_times": times[win],
        "null_band": null_band(np.full(len(z), critical), polarity, int(win.sum())),
    }
