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

The significance comes from one of three sets of projections (``pairs``):

- ``"signflip"`` (default): every cross-projection, its null distribution from sign flips
  of whole trials. Under the null hypothesis each trial is as likely with either sign, and
  flipping trial k changes the sign of every projection that involves it, so the profile of
  the mean projection is recomputed for every sign pattern (all of them up to
  ``MAX_FLIPS``, a random draw beyond) and the observed peak is compared with the peaks of
  the flipped profiles: the choice of ``tau_R`` at the peak is part of the null. On pure
  white noise (10 trials, 200 channels) it rejects at 0.05 for alpha = 0.05, and it gives
  the published decision on the 10 example channels of the paper (REQ-134). The statistic
  is the peak of the profile.
- ``"disjoint"``: a one-sided one-sample t-test of projections at ``tau_R``, each trial in
  one pair only (``K // 2`` pairs, drawn at random),
  so that the projections tested are independent under the null hypothesis. On pure noise
  (white or AR(1) 0.9, 10 or 20 trials) the rejection rate at alpha = 0.05 is 0.07 to 0.13,
  above alpha only by the choice of ``tau_R`` at the peak of the profile.
- ``"half"``: the same t-test on one projection per unordered pair of trials (one half of the ``K(K-1)``
  cross-projections, the two projections of a pair never both, each trial the normalised
  one in about half of its pairs), the set of the paper. It reproduces the paper's
  significance decisions (REQ-134), but its ``K(K-1)/2`` values share trials and are not
  independent: on the same pure noise it rejects at 0.33 to 0.44 at alpha = 0.05 (0.24 to
  0.35 with the ``"normalised"`` profile).
"""

import numpy as np
from scipy.stats import ttest_1samp

from evoparam._typing import Any, NDArray
from evoparam.errors import DataError, SettingsError
from evoparam.utils._parallel import parallel_map

__all__ = ["projection_test", "PROFILES", "PAIRS"]

PROFILES = ("published", "normalised")
PAIRS = ("signflip", "disjoint", "half")
# Sign patterns of the "signflip" set: every pattern up to this many, a random draw beyond
MAX_FLIPS = 4096


def _half_pairs(n: Any) -> tuple[NDArray[Any], NDArray[Any]]:
    """One ordered pair per unordered pair {k, l}.

    (k, l) when k + l is odd, else (l, k), so that each trial is the normalised one in about half
    of its pairs.
    """
    first, second = np.triu_indices(n, 1)
    swap = (first + second) % 2 == 0
    return np.where(swap, second, first), np.where(swap, first, second)


def _sign_patterns(n: int, rng: np.random.Generator) -> NDArray[Any] | None:
    """Random sign patterns (``MAX_FLIPS`` x n, the first all +1), or None when every
    pattern is enumerated."""

    if 2**n <= MAX_FLIPS:
        return None
    patterns = rng.choice([-1.0, 1.0], size=(MAX_FLIPS, n))
    patterns[0] = 1.0
    return patterns


def _signflip_p(proj: Any, off: Any, observed: float, profile: str, patterns: Any) -> float:
    """p-value of the peak of the mean-projection profile against its sign-flip null."""

    n_t, n, _ = proj.shape
    if patterns is None:
        grid = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
        patterns = 1.0 - 2.0 * grid
    p_off = np.where(off[None], proj, 0.0)
    scale = 1.0 / (n * (n - 1))
    if profile == "normalised":
        scale = scale / np.sqrt(np.arange(1, n_t + 1))
    else:
        scale = np.full(n_t, scale)
    peaks = np.full(len(patterns), -np.inf)
    for t in range(n_t):
        value = np.einsum("bk,bk->b", patterns @ p_off[t], patterns) * scale[t]
        np.maximum(peaks, value, out=peaks)
    count = np.sum(peaks >= observed - 1e-12 * max(1.0, abs(observed)))
    return float(count / len(patterns))


def _one_channel(
    v: Any,
    _times: Any,
    _t0_index: Any,
    rng: np.random.Generator | None,
    profile: str = "published",
    pairs: str = "disjoint",
    order: Any = None,
) -> tuple[int, float, float, dict[str, Any]]:
    """v: trials x times (window from t0). Returns duration end index, statistic, p, extra.

    ``order``: the permutation of the trials that forms the disjoint pairs (drawn from
    ``rng`` when None).
    """
    n, n_t = v.shape
    # Cumulative Gram matrices over the prefix [t0, t]
    outer = np.einsum("kt,lt->tkl", v, v)
    gram = np.cumsum(outer, axis=0)  # times x n x n
    norms = np.sqrt(np.clip(np.einsum("tkk->tk", gram), 1e-300, None))
    proj = gram / norms[:, None, :]  # P_kl = <V_k, V_l> / ||V_l||
    off = ~np.eye(n, dtype=bool)
    mean_proj = proj[:, off].mean(axis=1)
    curve = mean_proj / np.sqrt(np.arange(1, n_t + 1)) if profile == "normalised" else mean_proj
    end = int(np.argmax(curve))
    if pairs == "signflip":
        stat, p = float(curve[end]), _signflip_p(proj, off, curve[end], profile, order)
    else:
        # Significance at the end: each trial in one pair, or one projection per unordered pair
        if pairs == "half":
            rows, cols = _half_pairs(n)
        else:
            if order is None:
                order = np.random.default_rng(rng).permutation(n)
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


def _channel_unit(unit: tuple[Any, ...]) -> tuple[int, float, float, dict[str, Any]]:
    """Test one channel (one unit of n_jobs)."""
    v, tw, profile, pairs, order = unit
    return _one_channel(v, tw, 0, None, profile, pairs, order=order)


def projection_test(
    data: Any,
    times: Any,
    window: Any,
    baseline: Any,  # noqa: ARG001  (signature shared by the existence tests; this one uses no baseline)
    *,
    rng: np.random.Generator | None = None,
    profile: str = "published",
    pairs: str = "signflip",
    n_jobs: int | None = 1,
) -> dict[str, NDArray[Any]]:
    """Cross-projection test of every channel.

    ``profile`` chooses the reading of the response duration (module docstring):
    ``"published"`` (the paper's peak of the mean projection, default) or ``"normalised"``
    (the peak of the mean projection divided by the square root of the window length).
    ``pairs`` chooses the test (module docstring): ``"signflip"`` (every projection,
    sign-flip null of the profile peak, default), ``"disjoint"`` (each trial in one pair) or
    ``"half"`` (one per unordered pair, as in the paper).
    ``n_jobs``: workers, one channel per unit (the pairs are drawn before, in channel order,
    so the result does not depend on ``n_jobs``).

    Returns a dict of arrays [n_channels]: statistic (the profile peak with ``"signflip"``,
    else the one-sample t of the projections at the response end), p_value (one-sided), onset (window start), offset (response end tau_R),
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
    out: dict[str, Any] = {
        k: np.full(n_ch, np.nan)
        for k in ("statistic", "p_value", "onset", "offset", "duration", "snr_db")
    }
    extra = []
    # the random pairs of every channel, drawn in channel order (independent of n_jobs)
    orders = [
        None
        if pairs == "half"
        else _sign_patterns(n, rng)
        if pairs == "signflip"
        else rng.permutation(n)
        for _ in range(n_ch)
    ]
    units = [(np.nan_to_num(data[:, c, win]), tw, profile, pairs, orders[c]) for c in range(n_ch)]
    for c, (end, stat, p, info) in enumerate(
        parallel_map(_channel_unit, units, n_jobs, desc="test_response")
    ):
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
