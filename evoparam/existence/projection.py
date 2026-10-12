# SPDX-License-Identifier: Apache-2.0
"""Cross-projection test and response duration.

Re-implemented from the published description of the canonical response parameterisation
(Miller et al., 2023, PLOS Computational Biology); the reference implementation is GPL and
is neither imported nor copied. The projection of trial k on trial l over [t0, t] grows with
t while the shared (evoked) part of the trials lasts, and only by chance after; the mean
projection is largest at the end of the response.

Two readings of the response duration are available (``profile``):

- ``"published"`` (default): ``tau_R = argmax_t mean P(t)``, the peak of the mean
  semi-normalised projection, as the paper defines it. This reading reproduces the
  durations the paper publishes for its example data (REQ-134; measured in CHANGELOG.md,
  "Cross-projection against the published values").
- ``"normalised"``: ``tau_R = argmax_t mean P(t) / sqrt(t - t0)``, the formula of §5.6
  item 2 as reviewed. A semi-normalised projection does not accumulate with the window
  (after the response, the norm of the projected trial grows and the projection falls), so
  this extra division moves the peak to the first milliseconds of the window: 16 ms
  instead of the published 80 ms on the example of the paper's Fig. 3.

The significance at ``tau_R`` is a one-sided one-sample t-test of projections against zero,
on one of two sets of pairs (``pairs``):

- ``"disjoint"`` (default): each trial in one pair only (``K // 2`` pairs, drawn at random),
  so that the projections tested are independent under the null hypothesis. On pure noise
  (white or AR(1) 0.9, 10 or 20 trials) the rejection rate at alpha = 0.05 is 0.07 to 0.13,
  above alpha only by the choice of ``tau_R`` at the peak of the profile.
- ``"half"``: one projection per unordered pair of trials (one half of the ``K(K-1)``
  cross-projections, the two projections of a pair never both, each trial the normalised
  one in about half of its pairs), the set of the paper. It reproduces the paper's
  significance decisions (REQ-134), but its ``K(K-1)/2`` values share trials and are not
  independent: on the same pure noise it rejects at 0.33 to 0.44 at alpha = 0.05 (0.24 to
  0.35 with the ``"normalised"`` profile).
"""

import numpy as np
from scipy.stats import ttest_1samp

from evoparam.errors import DataError, SettingsError

__all__ = ["projection_test", "PROFILES", "PAIRS"]

PROFILES = ("published", "normalised")
PAIRS = ("disjoint", "half")


def _half_pairs(n):
    """One ordered pair per unordered pair {k, l}.

    (k, l) when k + l is odd, else (l, k), so that each trial is the normalised one in about half
    of its pairs.
    """
    first, second = np.triu_indices(n, 1)
    swap = (first + second) % 2 == 0
    return np.where(swap, second, first), np.where(swap, first, second)


def _one_channel(v, times, t0_index, rng, profile="published", pairs="disjoint"):
    """v: trials x times (window from t0). Returns duration end index, statistic, p, extra."""
    n, n_t = v.shape
    # Cumulative Gram matrices over the prefix [t0, t]
    outer = np.einsum("kt,lt->tkl", v, v)
    gram = np.cumsum(outer, axis=0)  # times x n x n
    norms = np.sqrt(np.clip(np.einsum("tkk->tk", gram), 1e-300, None))
    proj = gram / norms[:, None, :]  # P_kl = <V_k, V_l> / ||V_l||
    off = ~np.eye(n, dtype=bool)
    mean_proj = proj[:, off].mean(axis=1)
    if profile == "normalised":
        curve = mean_proj / np.sqrt(np.arange(1, n_t + 1))
    else:
        curve = mean_proj
    end = int(np.argmax(curve))
    # Significance at the end: each trial in one pair, or one projection per unordered pair
    if pairs == "half":
        rows, cols = _half_pairs(n)
    else:
        order = rng.permutation(n)
        rows, cols = order[0 : n - 1 : 2], order[1:n:2]
    values = proj[end, rows, cols]
    stat, p = ttest_1samp(values, 0.0, alternative="greater")
    # Canonical shape and per-trial scalings
    seg = v[:, : end + 1]
    _, s, vt = np.linalg.svd(seg, full_matrices=False)
    shape = vt[0] * np.sign(vt[0][np.argmax(np.abs(vt[0]))])
    alpha = seg @ shape
    resid = seg - np.outer(alpha, shape)
    r2 = 1 - np.sum(resid**2) / np.sum(seg**2) if np.sum(seg**2) > 0 else np.nan
    noise = np.sum(resid**2) / max(resid.size - n, 1)
    snr = np.sum(alpha**2) / (n * noise) if noise > 0 else np.nan
    return (
        end,
        float(stat),
        float(p),
        {"profile": curve, "shape": shape, "alpha": alpha, "r2": float(r2), "snr": float(snr)},
    )


def projection_test(
    data, times, window, baseline, *, rng=None, profile="published", pairs="disjoint"
):
    """Cross-projection test of every channel.

    ``profile`` chooses the reading of the response duration (module docstring):
    ``"published"`` (the paper's peak of the mean projection, default) or ``"normalised"``
    (the peak of the mean projection divided by the square root of the window length).
    ``pairs`` chooses the projections tested at the response end: ``"disjoint"`` (each
    trial in one pair, default) or ``"half"`` (one per unordered pair, as in the paper).

    Returns a dict of arrays [n_channels]: statistic (one-sample t of the projections at the
    response end), p_value (one-sided), onset (window start), offset (response end tau_R),
    duration, snr_db (of the canonical-shape scalings), plus ``extra`` (per channel: profile,
    canonical shape, alphas, R², SNR), and ``curve`` [n_channels, n_times] (the profile
    whose peak is the response end), ``curve_times`` and ``null_band`` (NaN: no band over
    time for this method).
    """
    if profile not in PROFILES:
        raise SettingsError(f"profile must be one of {PROFILES}, not {profile!r}.")
    if pairs not in PAIRS:
        raise SettingsError(f"pairs must be one of {PAIRS}, not {pairs!r}.")
    rng = np.random.default_rng(rng)
    n, n_ch, _ = data.shape
    if n < 5:
        raise DataError("The projection test needs at least 5 trials.")
    win = (times >= window[0]) & (times <= window[1])
    tw = times[win]
    out = {
        k: np.full(n_ch, np.nan)
        for k in ("statistic", "p_value", "onset", "offset", "duration", "snr_db")
    }
    extra = []
    for c in range(n_ch):
        v = np.nan_to_num(data[:, c, win])
        end, stat, p, info = _one_channel(v, tw, 0, rng, profile, pairs)
        out["statistic"][c], out["p_value"][c] = stat, p
        out["onset"][c], out["offset"][c] = tw[0], tw[end]
        out["duration"][c] = tw[end] - tw[0]
        out["snr_db"][c] = 10 * np.log10(info["snr"]) if info["snr"] > 0 else np.nan
        extra.append(info)
    out["duration_ci"] = np.full((n_ch, 2), np.nan)
    out["extra"] = extra
    # The statistic over time: the mean projection profile, whose maximum is the response
    # end; the method has no null band over time (its test is at the end only)
    out["curve"] = np.array([info["profile"] for info in extra])
    out["curve_times"] = tw
    out["null_band"] = np.full(out["curve"].shape + (2,), np.nan)
    return out
