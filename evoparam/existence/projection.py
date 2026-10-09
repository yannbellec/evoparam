"""Cross-projection test and response duration.

Re-implemented from the published description of the canonical response parameterisation
(Miller et al., 2023, PLOS Computational Biology); the reference implementation is GPL and
is neither imported nor copied. The projection of trial k on trial l over [t0, t] grows with
t while the shared (evoked) part of the trials lasts, and only by chance after; the
normalised mean projection is largest at the end of the response.
"""

import numpy as np
from scipy.stats import ttest_1samp

from evoparam.errors import DataError

__all__ = ["projection_test"]


def _one_channel(v, times, t0_index, rng):
    """v: trials x times (window from t0). Returns duration end index, statistic, p, extra."""

    n, n_t = v.shape
    # Cumulative Gram matrices over the prefix [t0, t]
    outer = np.einsum("kt,lt->tkl", v, v)
    gram = np.cumsum(outer, axis=0)                               # times x n x n
    norms = np.sqrt(np.clip(np.einsum("tkk->tk", gram), 1e-300, None))
    proj = gram / norms[:, None, :]                               # P_kl = <V_k, V_l> / ||V_l||
    off = ~np.eye(n, dtype=bool)
    mean_proj = proj[:, off].mean(axis=1)
    duration = np.arange(1, n_t + 1)
    profile = mean_proj / np.sqrt(duration)
    end = int(np.argmax(profile))
    # Significance at the end: half of the non-overlapping pairs (each trial used once)
    order = rng.permutation(n)
    pairs = [(order[i], order[i + 1]) for i in range(0, n - 1, 2)]
    values = np.array([proj[end, k, l] for k, l in pairs])
    stat, p = ttest_1samp(values, 0.0, alternative="greater")
    # Canonical shape and per-trial scalings
    seg = v[:, :end + 1]
    _, s, vt = np.linalg.svd(seg, full_matrices=False)
    shape = vt[0] * np.sign(vt[0][np.argmax(np.abs(vt[0]))])
    alpha = seg @ shape
    resid = seg - np.outer(alpha, shape)
    r2 = 1 - np.sum(resid ** 2) / np.sum(seg ** 2) if np.sum(seg ** 2) > 0 else np.nan
    noise = np.sum(resid ** 2) / max(resid.size - n, 1)
    snr = np.sum(alpha ** 2) / (n * noise) if noise > 0 else np.nan
    return end, float(stat), float(p), {"profile": profile, "shape": shape, "alpha": alpha,
                                        "r2": float(r2), "snr": float(snr)}


def projection_test(data, times, window, baseline, *, rng=None):
    """Cross-projection test of every channel.

    Returns a dict of arrays [n_channels]: statistic (one-sample t of the projections at the
    response end), p_value (one-sided), onset (window start), offset (response end tau_R),
    duration, snr_db (of the canonical-shape scalings), plus ``extra`` (per channel: profile,
    canonical shape, alphas, R², SNR), and ``curve`` [n_channels, n_times] (the normalised
    profile), ``curve_times`` and ``null_band`` (NaN: no band over time for this method).
    """

    rng = np.random.default_rng(rng)
    n, n_ch, _ = data.shape
    if n < 5:
        raise DataError("The projection test needs at least 5 trials.")
    win = (times >= window[0]) & (times <= window[1])
    tw = times[win]
    out = {k: np.full(n_ch, np.nan) for k in ("statistic", "p_value", "onset", "offset",
                                               "duration", "snr_db")}
    extra = []
    for c in range(n_ch):
        v = np.nan_to_num(data[:, c, win])
        end, stat, p, info = _one_channel(v, tw, 0, rng)
        out["statistic"][c], out["p_value"][c] = stat, p
        out["onset"][c], out["offset"][c] = tw[0], tw[end]
        out["duration"][c] = tw[end] - tw[0]
        out["snr_db"][c] = 10 * np.log10(info["snr"]) if info["snr"] > 0 else np.nan
        extra.append(info)
    out["duration_ci"] = np.full((n_ch, 2), np.nan)
    out["extra"] = extra
    # The statistic over time: the normalised mean projection profile, whose maximum is the
    # response end; the method has no null band over time (its test is at the end only)
    out["curve"] = np.array([info["profile"] for info in extra])
    out["curve_times"] = tw
    out["null_band"] = np.full(out["curve"].shape + (2,), np.nan)
    return out
