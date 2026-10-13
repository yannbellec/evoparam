# SPDX-License-Identifier: Apache-2.0
"""Period-based reconstruction of a repeating artifact (high-frequency trains, DBS).

Implemented from the open-access description of period-based artifact reconstruction
and removal (Dastin-van Rijn et al., 2021, Cell Reports Methods). A patent application
on the method (US 2022/0338786 A1, Brown University) was pending when the specification
was written; see the specification, section 13.
"""

from typing import cast

import numpy as np

from evoparam._typing import Any, ArrayLike, FloatArray, IntArray, NDArray
from evoparam.errors import ArtifactError, SettingsError

__all__ = ["estimate_period", "period_lags", "period_filter"]

# Largest search half-width, as a fraction of the guessed period.
_MAX_SEARCH = 0.5

# Shortest period, in samples, that can be estimated (two samples per cycle).
_MIN_PERIOD_SAMPLES = 2

# Candidates per stage of the period search.
N_CANDIDATES = 401

# Half-width of the phase neighbourhood used to predict a sample, as a fraction of the
# number of samples (at least 2 neighbours on each side).
NEIGHBOUR_FRACTION = 0.002

# Fewest samples averaged into each estimate when n_bins is chosen automatically.
MIN_BIN_SAMPLES = 50


def _periodic_mse(x: Any, period: Any, half: Any) -> float:
    """Mean squared error of predicting each sample from its phase neighbours.

    Samples are sorted by phase ``(n mod period) / period``; each one is predicted by the
    mean of the ``half`` samples before and after it in phase order (circularly), itself
    excluded. A wrong period scatters the phases and raises the error.
    """
    n = len(x)
    phase = np.mod(np.arange(n), period) / period
    values = x[np.argsort(phase, kind="stable")]
    padded = np.r_[values[-half:], values, values[:half]]
    csum = np.r_[0.0, np.cumsum(padded)]
    window = csum[2 * half + 1 :] - csum[: -(2 * half + 1)]
    pred = (window - values) / (2 * half)
    return float(np.mean((values - pred) ** 2))


def estimate_period(
    x: ArrayLike, sfreq: float, f_guess: float, *, search: float = 0.02, n_stages: int = 3
) -> float:
    """Estimate the period of a repeating artifact, in fractional samples.

    A grid search minimises the error of a fit that repeats with the period (each sample
    predicted by its neighbours in phase); each stage refines around the previous best.

    Parameters
    ----------
    x : 1d array
        Signal that contains the artifact (a few seconds of stimulation).
    sfreq : float
        Sampling frequency, in Hz.
    f_guess : float
        Nominal stimulation frequency, in Hz.
    search : float, default: 0.02
        Relative half-width of the first search (0.02 = +/-2 % around ``sfreq / f_guess``).
    n_stages : int, default: 3
        Refinement stages; each narrows the grid around the best candidate.

    Returns
    -------
    period : float
        Period in samples (``sfreq / period`` is the stimulation frequency in Hz).
    """
    x = cast(FloatArray, np.asarray(x, dtype=float))
    if x.ndim != 1:
        raise SettingsError("estimate_period expects a 1d signal.")
    if not (sfreq > 0 and f_guess > 0 and 0 < search < _MAX_SEARCH):
        raise SettingsError("sfreq and f_guess must be positive, search in (0, 0.5).")
    guess = sfreq / f_guess
    if guess < _MIN_PERIOD_SAMPLES:
        raise ArtifactError(
            f"A {f_guess} Hz train is not resolvable at {sfreq} Hz "
            "(fewer than 2 samples per period)."
        )
    if len(x) < 10 * guess:
        raise ArtifactError("estimate_period needs at least 10 periods of signal.")
    x = x - np.mean(x)
    half = max(2, int(round(NEIGHBOUR_FRACTION * len(x))))
    lo, hi = guess * (1 - search), guess * (1 + search)
    best = guess
    for _ in range(n_stages):
        grid = np.linspace(lo, hi, N_CANDIDATES)
        errors = np.array([_periodic_mse(x, period, half) for period in grid])
        ind = int(np.argmin(errors))
        best = grid[ind]
        step = grid[1] - grid[0]
        lo, hi = best - 2 * step, best + 2 * step
    return float(best)


def period_lags(
    period: Any, n_times: int, *, n_bins: Any = None, n_skip: int = 0, d_period: Any = None
) -> tuple[NDArray[Any], int]:
    """Lags (in samples) whose phase matches the period.

    A lag ``L`` with ``n_skip < L <= n_bins`` is kept when ``L mod period`` is within
    ``d_period`` of 0 or of ``period``. When ``n_bins`` is None it is the smallest value
    that gives at least 50 samples per estimate (25 lags on each side).

    Returns
    -------
    lags : 1d array of int
    n_bins : int
    """
    if d_period is None:
        d_period = 0.01 * period
    limit = n_times - 1 if n_bins is None else int(n_bins)
    candidates = np.arange(int(n_skip) + 1, limit + 1)
    rem = np.mod(candidates, period)
    lags = candidates[(rem <= d_period) | (rem >= period - d_period)]
    if n_bins is None:
        needed = int(np.ceil(MIN_BIN_SAMPLES / 2))
        if len(lags) < needed:
            raise ArtifactError(
                f"Only {2 * len(lags)} matching samples within the epoch for period "
                f"{period:.4f}; use longer epochs or a larger d_period."
            )
        lags = lags[:needed]
        n_bins = int(lags[-1])
    if len(lags) == 0:
        raise ArtifactError("No lag matches the period; increase n_bins or d_period.")
    return lags, int(n_bins)


def period_filter(data: ArrayLike, lags: IntArray, causal: bool = False) -> tuple[Any, Any]:
    """Subtract, from every sample, the mean of the samples at the matching lags.

    ``x_hat[t] = x[t] - mean({x[t - L], x[t + L] : L in lags, inside the series})``. This is
    a linear FIR filter; with ``causal`` only past samples (``t - L``) are used, for
    online use.

    Parameters
    ----------
    data : array, time last
    lags : 1d array of int
    causal : bool

    Returns
    -------
    cleaned, estimate : arrays like ``data``
    """
    from scipy.signal import fftconvolve

    data = np.asarray(data, dtype=float)
    n_times = data.shape[-1]
    reach = int(np.max(lags))
    kernel = np.zeros(2 * reach + 1)
    kernel[reach + lags] = 1.0  # past samples x[t - L] (convolution flips)
    if not causal:
        kernel[reach - lags] = 1.0  # future samples x[t + L]
    flat = data.reshape(-1, n_times)
    sums = fftconvolve(flat, kernel[None], mode="same", axes=-1)
    counts = fftconvolve(np.ones((1, n_times)), kernel[None], mode="same", axes=-1)
    counts = np.round(counts)
    estimate = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    estimate = estimate.reshape(data.shape)
    return data - estimate, estimate
