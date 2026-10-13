# SPDX-License-Identifier: Apache-2.0
"""Recovery tail after the pulse: a two-exponential fit, subtracted."""

import numpy as np

from evoparam._typing import Any, ArrayLike, FloatArray, NDArray
from evoparam.errors import ArtifactError, SettingsError

__all__ = []

# Number of time constants tried between the bounds (log-spaced).
N_TAUS = 16

# The slower exponential is at least this many times slower than the faster one.
TAU_RATIO = 3.0


def fit_tail(
    times: FloatArray,
    data: ArrayLike,
    start: int,
    length: float,
    tau_bounds: tuple[float, float] = (0.0005, 0.05),
    part: str = "full",
) -> tuple[Any, NDArray[Any], dict[str, Any]]:
    """Fit ``r(t) = c0 + c1 exp(-(t - b) / tau1) + c2 exp(-(t - b) / tau2)`` on the tail.

    The time constants are chosen on a log grid (``tau2 >= 3 tau1``) and the coefficients
    by least squares, for every series at once (the design matrix is shared).

    Parameters
    ----------
    times : 1d array
    data : array, time last
    start : int
        First sample of the tail (the sample after the artifact window, ``b``).
    length : float
        Tail duration, in seconds.
    tau_bounds : (float, float)
        Smallest and largest time constant, in seconds.
    part : {"full", "joined"}
        Part of the fit returned for subtraction. ``"full"`` (default): ``r(t)`` itself, as
        the specification writes it (spec §5.3, method 3). ``"joined"``: ``r(t) - r(t_end)``,
        zero at the end of the tail, so the level the signal settles to (``c0`` plus
        whatever has not decayed) is kept and the cleaned signal joins the untouched samples
        after the tail without a step.

    Returns
    -------
    tail : array, like ``data`` restricted to the tail samples
        The part of the fit to subtract (``part``).
    inds : 1d array
        Indices of the tail samples.
    params : dict
        Arrays ``tau1``, ``tau2``, ``c0``, ``c1``, ``c2`` and ``rss`` (shape of the series).
    """
    lo, hi = tau_bounds
    if not 0 < lo < hi:
        raise SettingsError("tau_bounds must be (low, high) with 0 < low < high.")
    t0 = times[start]
    inds = np.flatnonzero((times >= t0) & (times <= t0 + length))
    if len(inds) < 8:
        raise ArtifactError(
            "The tail holds fewer than 8 samples; lengthen tail_length or the epoch."
        )
    rel = times[inds] - t0
    series = np.asarray(data)[..., inds]
    shape = series.shape[:-1]
    flat = series.reshape(-1, len(inds))

    taus: Any = np.geomspace(lo, hi, N_TAUS)  # tau1, tau2: grid values, then per-series arrays
    best_rss = np.full(len(flat), np.inf)
    best = np.zeros((len(flat), 5))
    for _i, tau1 in enumerate(taus):
        for tau2 in taus[taus >= TAU_RATIO * tau1]:
            design = np.c_[np.ones_like(rel), np.exp(-rel / tau1), np.exp(-rel / tau2)]
            pinv = np.linalg.pinv(design)
            coefs = flat @ pinv.T
            rss = np.sum((flat - coefs @ design.T) ** 2, axis=1)
            better = rss < best_rss
            best_rss[better] = rss[better]
            best[better] = np.c_[
                np.full(better.sum(), tau1), np.full(better.sum(), tau2), coefs[better]
            ]
    tau1, tau2 = best[:, 0], best[:, 1]

    def decay(r: Any) -> Any:
        return best[:, 3:4] * np.exp(-r / tau1[:, None]) + best[:, 4:5] * np.exp(
            -r / tau2[:, None]
        )

    if part == "full":
        tail = best[:, 2:3] + decay(rel[None])
    elif part == "joined":
        tail = decay(rel[None]) - decay(rel[-1:][None])
    else:
        raise SettingsError("part must be 'joined' or 'full'.")
    params = {
        "tau1": tau1.reshape(shape),
        "tau2": tau2.reshape(shape),
        "c0": best[:, 2].reshape(shape),
        "c1": best[:, 3].reshape(shape),
        "c2": best[:, 4].reshape(shape),
        "rss": best_rss.reshape(shape),
    }
    return tail.reshape(shape + (len(inds),)), inds, params
