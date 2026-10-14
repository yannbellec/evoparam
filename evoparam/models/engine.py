# SPDX-License-Identifier: Apache-2.0
"""The fitting engine (spec §5.5.3): background plus components, chosen by model comparison.

``y(t) = b(t) + sum_k A_k s_k(t; theta_k) + noise`` on the fit window. The amplitudes and
the background's linear coefficients are solved exactly for given non-linear parameters
(variable projection); the non-linear parameters are optimised by bounded least squares.
Components are added one at a time and kept only when the information criterion improves
by ``delta``; nothing is deleted by a threshold, a width rule, an edge rule or an overlap
rule.
"""

import dataclasses
import functools
import logging
import math
import threading
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.linalg.lapack import dgeqrf as _dgeqrf
from scipy.linalg.lapack import dormqr as _dormqr
from scipy.linalg.lapack import dtrtri as _dtrtri
from scipy.linalg.lapack import dtrtrs as _dtrtrs
from scipy.optimize import least_squares

from evoparam._typing import Any, FloatArray, NDArray
from evoparam.errors import DataError, SettingsError
from evoparam.models import _columns, _fast
from evoparam.models._calibration import Calibration
from evoparam.models._settings import CALIBRATED, FIXED_DELTA
from evoparam.models._trf import TrfOptions, least_squares_trf
from evoparam.shapes import (
    FWHM_PER_SIGMA,
    SHAPES,
    ExponentialBackground,
    Gaussian,
    SkewedGaussian,
    compute_features,
    get_background,
    get_shape,
)

if TYPE_CHECKING:
    from evoparam.shapes._base import ComponentShape

__all__ = ["fit_waveform_core", "noise_model", "criterion"]

logger: logging.Logger = logging.getLogger("evoparam")

MAD_TO_SD = 1.4826
MIN_BASELINE = 20
COND_LIMIT = 1e8
CORR_LIMIT = 0.95
BOUND_TOL = 1e-3

# Optimiser of the non-linear parameters: "trf" (models/_trf.py), "trf-full" (the same
# without the QR reduction: SciPy's operations), "scipy" (scipy.optimize.least_squares)
_OPTIMISER = "trf"

# The compiled fit of models/_fast.py (the ``fast`` extra, numba) for the structures it
# covers, with "trf": "auto" uses it when numba is installed, "off" never
_FAST = "auto"

# Qualifying candidates compared before one is accepted at each step.
CANDIDATE_BEAM = 3
# Residual extrema tried (strongest first) before the search of a further component stops
# when none of them improves the criterion by delta (the weaker ones are not tried). The
# first qualifying extremum of a step was at most the sixth on the layer-2 development
# cells; the step that adds nothing (the last one of every fit) tried up to
# 2 x max_components extrema.
CANDIDATE_SCAN = 6

# Two components make a pair (an ordering chain, cancelling or overlapping neighbours).
_PAIR = 2
# The fewest baseline samples a lag-1 autocorrelation is estimated from.
_MIN_RHO_SAMPLES = 3
# Above this estimate of the condition number the QR route gives way to the SVD solve.
_QR_COND_LIMIT = 1e11
# A smoothing kernel narrower than half a sample leaves the residual as it is.
_MIN_SMOOTH_SAMPLES = 0.5
# A fit window spans at least this many ``min_fwhm``.
MIN_WINDOW_FWHMS = 3
# The fewest valid samples of a fit window.
MIN_VALID_SAMPLES = 5
# Floor of the standard errors given by the caller, relative to the noise scale.
_SEM_FLOOR = 0.1

# What a shape's initial_guess raises when it cannot start at a candidate extremum (bounds
# that do not fit the window, an extremum at the edge, a non-finite width): that shape is
# not tried there, with a DEBUG record (REQ-307: never without a trace). Any other
# exception is a programming error and propagates (to the channel's status "failed").
GUESS_ERRORS = (SettingsError, DataError, ValueError, ArithmeticError, IndexError)


def _guess_failed(shape: Any, tw: Any, peaks: Any, err: BaseException) -> None:
    """Log at DEBUG level that ``shape`` cannot start at the ``peaks`` (REQ-307)."""
    logger.debug(
        "shape %s cannot start at %s s (%s: %s); not tried there.",
        getattr(shape, "name", type(shape).__name__),
        ", ".join(f"{float(tw[int(p)]):.4f}" for p in peaks if 0 <= int(p) < len(tw)),
        type(err).__name__,
        err,
    )


###################################################################################################
## Noise and criteria

# Largest autoregressive order of the noise model, and baseline samples needed per order.
MAX_AR_ORDER = 10
SAMPLES_PER_ORDER = 20


def ar_model(z: Any, return_acov: bool = False) -> Any:
    """Autoregressive model of a standardised baseline (least squares, order by AIC).

    Returns (filter coefficients ``a`` = [1, -phi_1, ..., -phi_p], innovation SD), and with
    ``return_acov`` the autocovariances of lags ``0 .. p - 1`` implied by the fitted model.

    Least squares (the covariance method), not Yule-Walker: intracranial averages sit near a
    unit root (sum of the coefficients 0.98 to 0.995 at 2 kHz), where Yule-Walker is biased
    towards a faster decay; the model then under-states the slow power of the noise, and a
    correct fit reads slow fluctuations of a real baseline as components (REQ-112 on
    ds004080: 92.5 % silent with Yule-Walker, 97.5 % with least squares). Orders whose
    fitted model is not stationary are skipped.
    """
    z = np.asarray(z, float) - np.mean(z)
    n = len(z)
    p_max = int(min(MAX_AR_ORDER, n // SAMPLES_PER_ORDER))
    target = z[p_max:]
    var0 = float(np.mean(target**2))
    if not var0 > 0:
        return (np.array([1.0]), 1.0, np.zeros(0)) if return_acov else (np.array([1.0]), 1.0)
    fits = [(len(target) * np.log(var0), 0, np.zeros(0), var0)]
    lagged_all = np.column_stack([z[p_max - k : n - k] for k in range(1, p_max + 1)])
    # the nested least-squares problems from one QR factorisation (order p: the first p
    # columns, R[:p, :p] phi = (Q^T target)[:p]); lstsq where R is numerically singular
    q_all, r_all = np.linalg.qr(lagged_all)
    qty = q_all.T @ target
    r_diag = np.abs(np.diag(r_all))
    singular = r_diag <= np.finfo(float).eps * max(lagged_all.shape) * np.max(r_diag, initial=0.0)
    for p in range(1, p_max + 1):
        lagged = lagged_all[:, :p]
        if np.any(singular[:p]):
            phi = np.linalg.lstsq(lagged, target, rcond=None)[0]
        else:
            phi = np.linalg.solve(r_all[:p, :p], qty[:p])
        var = float(np.mean((target - lagged @ phi) ** 2))
        if var > 0:
            fits.append((len(target) * np.log(var) + 2 * p, p, phi, var))
    # the lowest AIC among the stationary orders (the lowest such order on a tie): the
    # roots are computed in order of AIC until a stationary model is found
    best = fits[0]
    for aic, p, phi, var in sorted(fits[1:], key=lambda row: (row[0], row[1])):
        if not aic < best[0]:
            break
        if np.max(np.abs(np.roots(np.r_[1.0, -phi]))) < 1.0:
            best = (aic, p, phi, var)
            break
    _, _, phi, var = best
    a = np.r_[1.0, -phi]
    if not return_acov:
        return a, float(np.sqrt(var))
    return a, float(np.sqrt(var)), _ar_autocovariance(phi, var)[: len(phi)]


def _ar_autocovariance(phi: Any, var: Any) -> NDArray[np.floating[Any]]:
    """Return the autocovariances of lags 0 .. p of a stationary AR(p) model.

    The Yule-Walker equations are solved for the autocovariances.
    """
    p = len(phi)
    lhs = np.eye(p + 1)
    for k in range(p + 1):
        for j in range(1, p + 1):
            lhs[k, abs(k - j)] -= phi[j - 1]
    rhs = np.zeros(p + 1)
    rhs[0] = var
    return np.linalg.solve(lhs, rhs)


class Whitener:
    """Linear map that turns the model residual into (approximately) white noise.

    ``white(x) = AR-filter(x / scale) / innovation_sd``, with the autoregressive filter of
    the baseline noise. Rows that depend on a masked sample are set to zero: they carry no
    evidence. The first ``p`` rows depend on samples before the window: without ``acov``
    they are set to zero too (conditional likelihood); with ``acov`` (the autocovariances
    of lags ``0 .. p - 1`` of the stationary noise) they are whitened by the Cholesky factor
    of its ``p x p`` covariance (exact likelihood). The exact form keeps the information
    about the level of the waveform: near a unit root (lag-1 autocorrelation 0.99 and
    more, as in intracranial averages) the AR filter is almost a difference and the
    conditional likelihood is blind to a misfit of the slow part of the waveform, so that
    a curve hundreds of microvolts away from the data could be preferred.
    """

    def __init__(self, scale: Any, a: Any, innovation: Any, valid: Any, acov: Any = None) -> None:
        """Set the rows that carry evidence and the start rows of the exact likelihood."""
        self.scale: NDArray[Any] = np.asarray(scale, float)
        self.a: NDArray[Any] = np.asarray(a, float)
        self.innovation: float = float(innovation)
        p = len(self.a) - 1
        ok = np.asarray(valid, bool).copy()
        self.start: Any = None
        if p:
            bad = ~ok
            spread = np.convolve(bad.astype(float), np.ones(p + 1))[: len(ok)] > 0
            first_valid = bool(np.all(ok[:p])) and len(ok) > p
            ok = ~spread
            ok[:p] = False
            if acov is not None and len(acov) == p and first_valid:
                self.start = _inverse_cholesky(np.asarray(acov, float))
                ok[:p] = self.start is not None
        self.p = p
        self.skip = p  # rows before this one hold the start rows (if any)
        self.rows = ok
        self._row_weight = ok.astype(float)
        self.n: int = int(ok.sum())

    def __call__(self, x: Any) -> NDArray[Any]:
        """Return the whitened ``x`` (a vector, or a matrix with time along the first axis)."""
        x = np.asarray(x, float)
        raw = x / (self.scale if x.ndim == 1 else self.scale[:, None])
        u = raw
        if len(self.a) > 1:
            # the FIR filter u[n] = sum_k a[k] raw[n - k] as shifted sums: scipy's lfilter
            # with a denominator [1] convolves column by column (apply_along_axis), which
            # took an eighth of the time of a fit on intracranial recordings
            u = self.a[0] * raw
            for k in range(1, len(self.a)):
                if k < len(raw):
                    u[k:] += self.a[k] * raw[:-k]
        u = u / self.innovation
        if self.start is not None:
            u = np.array(u, copy=True)
            # einsum without BLAS: a LAPACK triangular solve here, called for every residual
            # and Jacobian, woke OpenBLAS's threads, whose spinning took 3 to 4 cores per
            # process (and stalled parallel fits)
            u[: self.p] = np.einsum("ij,j...->i...", self.start, raw[: self.p])
        if x.ndim == 1:
            return np.where(self.rows, u, 0.0)
        return np.where(self.rows[:, None], u, 0.0)

    def into(self, x: Any, out: Any) -> Any:
        """Apply ``self(x)`` to a matrix ``x`` (n x k, time along the first axis), into ``out``.

        ``out`` may be ``x``: the same operations, without the temporary arrays. Masked rows are
        multiplied by 0 (``self`` selects 0 for them).
        """
        raw = x / self.scale[:, None]
        if len(self.a) > 1:
            np.multiply(raw, self.a[0], out=out)
            for k in range(1, len(self.a)):
                if k < len(raw):
                    out[k:] += self.a[k] * raw[:-k]
        else:
            out[...] = raw
        out /= self.innovation
        if self.start is not None:
            out[: self.p] = np.einsum("ij,j...->i...", self.start, raw[: self.p])
        out *= self._row_weight[:, None]
        return out


def _inverse_cholesky(acov: Any) -> Any:
    """Return the inverse of the lower Cholesky factor of the Toeplitz matrix of ``acov``.

    For p <= 10, by plain loops (no LAPACK call, see Whitener); None when it is not positive
    definite.
    """
    p = len(acov)
    cov = acov[np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])]
    chol = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1):
            rest = cov[i, j] - np.dot(chol[i, :j], chol[j, :j])
            if i == j:
                if not rest > 0:
                    return None
                chol[i, i] = np.sqrt(rest)
            else:
                chol[i, j] = rest / chol[j, j]
    inv = np.zeros((p, p))
    for i in range(p):
        inv[i, i] = 1.0 / chol[i, i]
        for j in range(i):
            inv[i, j] = -np.dot(chol[i, j:i], inv[j:i, j]) / chol[i, i]
    return inv


# A band above 1 / k of the Nyquist frequency is empty when the median periodogram level of
# the half-octave above it is below EMPTY_BAND times that of the octave below it; strides
# tried, largest first, and periodogram points needed in each band
EMPTY_BAND = 1e-4
MAX_STRIDE = 4
MIN_BAND_POINTS = 8


def noise_stride(z: Any) -> int:
    """Return the largest stride ``k`` (2 .. ``MAX_STRIDE``) at which the noise band is empty.

    The noise ``z`` has an empty band just above ``1 / k`` of the Nyquist frequency; 1 when it
    fills its band.

    The test compares the median level of the Hann-tapered periodogram of the detrended
    noise in ``(1/k, 1.5/k]`` and in ``(0.5/k, 1/k]`` (fractions of the Nyquist frequency).
    Medians make it robust to lines and to the image left near the Nyquist frequency by
    an upsampling filter. ``EMPTY_BAND`` (-4 decades) asks for a band that is all but empty,
    as interpolation leaves it (CHANGELOG): the ratio is -1.7 decades or more on the
    ds004080 baselines at their recorded rates and decimated to 1024 and 512 Hz, -1.5 or
    more on unfiltered simulations, -5.4 or less on the example run (recorded at 512 Hz)
    upsampled to 1024 Hz. A zero-phase low-pass of order 4 at 30 or 40 Hz on 256 Hz data
    gives -4.5 to -6.8 (strided, which improved those fits); one of order 2 stays above -4
    at 256 and 1024 Hz (stride 1: the previous path).
    """
    z = np.asarray(z, float)
    t = np.arange(len(z), dtype=float)
    z = (z - np.polyval(np.polyfit(t, z, 1), t)) * np.hanning(len(z))
    power = np.abs(np.fft.rfft(z)) ** 2
    f = np.arange(len(power)) / max(len(power) - 1, 1)
    for k in range(MAX_STRIDE, 1, -1):
        above = power[(f > 1.0 / k) & (f <= 1.5 / k)]
        below = power[(f > 0.5 / k) & (f <= 1.0 / k)]
        if min(len(above), len(below)) >= MIN_BAND_POINTS and np.median(
            above
        ) < EMPTY_BAND * np.median(below):
            return k
    return 1


class StridedWhitener(Whitener):
    """Whitening of a noise that fills only the lowest ``1 / k`` of its band.

    ``k`` is ``stride``, from ``noise_stride``: an upsampled recording, or one whose anti-alias or
    hardware low-pass is well below the Nyquist frequency.

    Only every k-th row carries evidence, the rows of the samples ``phase``, ``phase + k``,
    ... of the window, and they are whitened by the AR model of the noise sampled at the
    same rate (``a`` acts at lags ``0, k, 2k, ...``); the other rows are set to zero. Such
    samples hold all that the band of the noise holds, and the fit is judged as it would be
    on a recording at the lower rate. The ordinary whitening of such a noise amplifies its
    empty band by orders of magnitude: misfits there (the interpolation of the model, the
    ringing of an upsampling filter next to a stimulation artifact) count as strong
    evidence, and the fit keeps narrow components, often in cancelling pairs (on ds004080
    upsampled from 512 to 1024 Hz: an N1 on 54 channels instead of 14, CHANGELOG). Rows that
    depend on a masked sample are zero; the first ``p`` kept rows are the start rows of the
    exact likelihood (``acov``, as in ``Whitener``).
    """

    def __init__(
        self,
        scale: Any,
        a: Any,
        innovation: Any,
        valid: Any,
        stride: Any,
        phase: int = 0,
        acov: Any = None,
    ) -> None:
        """Keep the rows ``phase, phase + stride, ...`` that carry evidence and the start rows."""
        self.scale: NDArray[Any] = np.asarray(scale, float)
        self.a: NDArray[Any] = np.asarray(a, float)
        self.innovation: float = float(innovation)
        self.stride: int
        k = self.stride = int(stride)
        valid = np.asarray(valid, bool)
        n = len(valid)
        p = len(self.a) - 1
        spread = ~valid
        for j in range(1, p + 1):
            if j * k < n:
                spread[j * k :] |= ~valid[: -j * k]
        kept = np.zeros(n, bool)
        kept[phase::k] = True
        ok = kept & valid & ~spread
        self.start: Any = None
        self.start_rows: NDArray[Any] = np.arange(phase, phase + p * k, k)
        if p:
            first = self.start_rows[self.start_rows < n]
            ok[first] = False
            if acov is not None and len(acov) == p and len(first) == p and valid[first].all():
                self.start = _inverse_cholesky(np.asarray(acov, float))
                ok[first] = self.start is not None
        self.p = p
        self.skip: int = int(self.start_rows[-1]) + 1 if p else 0
        self.rows = ok
        self._row_weight = ok.astype(float)
        self.n: int = int(ok.sum())

    def __call__(self, x: Any) -> NDArray[Any]:
        """Return the whitened ``x`` on the kept rows, 0 elsewhere."""
        x = np.asarray(x, float)
        raw = x / (self.scale if x.ndim == 1 else self.scale[:, None])
        u = self.a[0] * raw
        for j in range(1, len(self.a)):
            lag = j * self.stride
            if lag < len(raw):
                u[lag:] += self.a[j] * raw[:-lag]
        u = u / self.innovation
        if self.start is not None:
            u[self.start_rows] = np.einsum("ij,j...->i...", self.start, raw[self.start_rows])
        if x.ndim == 1:
            return np.where(self.rows, u, 0.0)
        return np.where(self.rows[:, None], u, 0.0)

    def into(self, x: Any, out: Any) -> Any:
        """Apply ``self(x)`` into ``out``; returns ``out``."""
        out[...] = self(x)
        return out


MIN_PLUS_MINUS = 10


def plus_minus(trials: Any) -> Any:
    """Plus-minus reference (Schimmel 1967).

    The trials averaged with alternating signs (+, -, +, ... in their order; an odd last trial is
    left out). It holds the noise of the average over the whole epoch and none of the response.
    None with fewer than ``MIN_PLUS_MINUS`` trials.
    """
    trials = np.asarray(trials, float)
    n = (len(trials) // 2) * 2
    if n < MIN_PLUS_MINUS:
        return None
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    return np.nanmean(signs[:, None] * trials[:n], axis=0)


# below this many independent baseline samples the noise scale is too uncertain (its
# relative standard error is about 1 / sqrt(2 n)), and "auto" uses the plus-minus reference
MIN_EFFECTIVE_BASELINE = 30


def use_plus_minus(settings: Any, times: Any, y: Any, mask: Any = None) -> bool:
    """Whether the plus-minus reference replaces the baseline for the noise model."""
    source = settings.noise_source
    if source == "plusminus":
        return True
    if source != "auto" or settings.baseline is None:
        return False
    times = np.asarray(times, float)
    base = (times >= settings.baseline[0]) & (times <= settings.baseline[1])
    if mask is not None:
        base &= ~np.asarray(mask, bool)
    yb = np.asarray(y, float)[base]
    yb = yb[np.isfinite(yb)]
    if yb.size < _MIN_RHO_SAMPLES:
        return True
    rho = float(np.corrcoef(yb[1:], yb[:-1])[0, 1]) if np.std(yb) > 0 else 0.0
    rho = min(max(rho, 0.0), 0.999)
    return cast("bool", yb.size * (1 - rho) / (1 + rho) < MIN_EFFECTIVE_BASELINE)


def noise_model(times: Any, y: Any, baseline: Any, window_mask: Any, mask: Any) -> Any:
    """Noise scale (MAD) and lag-1 autocorrelation from the baseline.

    Returns (sigma, rho, warning or None).
    """
    if baseline is not None:
        inside = (times >= baseline[0]) & (times <= baseline[1]) & ~mask & np.isfinite(y)
    else:
        inside = np.zeros_like(window_mask)
    if inside.sum() >= MIN_BASELINE:
        yb = y[inside]
        sigma = MAD_TO_SD * np.median(np.abs(yb - np.median(yb)))
        centred = yb - yb.mean()
        denom = np.sum(centred**2)
        rho = float(np.sum(centred[1:] * centred[:-1]) / denom) if denom > 0 else 0.0
        return float(sigma), float(np.clip(rho, 0.0, 0.999)), None
    yw = y[window_mask & ~mask & np.isfinite(y)]
    d = np.diff(yw)
    sigma = MAD_TO_SD * np.median(np.abs(d - np.median(d))) / np.sqrt(2) if len(d) else 0.0
    return float(sigma), 0.0, "short_baseline"


def criterion(rss_w: Any, n_eff: Any, p: Any, kind: Any) -> Any:
    """BIC, AIC, AICc or HQC on whitened residuals with the effective sample count.

    HQC is the criterion of Hannan and Quinn (1979), ``2 ln ln n`` per parameter (between the
    2 of AIC and the ``ln n`` of BIC).
    """
    n_eff = max(float(n_eff), p + 2.0)
    base = n_eff * np.log(max(rss_w, 1e-300) / n_eff)
    if kind == "bic":
        return base + p * np.log(n_eff)
    if kind == "hqc":
        return base + 2 * p * np.log(np.log(n_eff))
    aic = base + 2 * p
    if kind == "aic":
        return aic
    return aic + 2 * p * (p + 1) / max(n_eff - p - 1, 1e-9)


###################################################################################################
## Model structure


@dataclasses.dataclass
class _Structure:
    """Shapes of the current components and the background; parameter layout."""

    shapes: list[Any]
    background: Any  # a BackgroundBase (built-in) or a plugin's Background
    tw: NDArray[Any]
    white: Whitener  # or a StridedWhitener
    y: NDArray[Any]
    prior: float | None = None  # SD (data units) of the prior on the background constant
    scale: float | None = None  # SD (data units) of the prior on every other coefficient
    edge: float | None = None  # how far (s) a component's location may lie outside the window
    level: float = 0.0  # prior mean of the background constant (the baseline level)
    ordered: bool = False  # latencies ordered by parameterisation (spec §5.5.3 step 5)
    # bounds of each shape on this window (they depend on tw and edge only), shared by the
    # structures of one fit
    memo: dict[Any, Any] | None = dataclasses.field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Set the parameter layout and the caches of the structure."""
        self.sizes = [s.n_params for s in self.shapes]
        self.offsets = np.r_[0, np.cumsum(self.sizes)].astype(int)
        self.n_bg = getattr(self.background, "n_nonlinear", 0)
        self.n_lin_bg = getattr(self.background, "n_linear", 0)
        self.n_theta = int(self.offsets[-1]) + self.n_bg
        self._aug: tuple[Any, NDArray[Any] | None] | None = None
        # what this structure computes once (its bounds, the limits of the starting values,
        # the arguments of the compiled fit)
        self._own: dict[str, Any] = {}
        names = tuple(getattr(self.background, "linear_names", ()))
        self.prior_col = (
            len(self.shapes) + names.index("c0") if self.prior and "c0" in names else None
        )
        # positions in theta of the latency parameters, in component order (the chain of
        # the ordered parameterisation)
        self.chain = (
            [
                int(self.offsets[k]) + s.param_names.index(s.latency_param)
                for k, s in enumerate(self.shapes)
                if s.latency_param in s.param_names
            ]
            if self.ordered
            else []
        )

    def bounds(self) -> Any:
        """Return the bounds of the non-linear parameters (memoised over the fit)."""
        if self.memo is None:
            return self._bounds()
        known = self._own.get("bounds")
        if known is None:
            key = ("bounds", tuple(self.shapes), self.background, self.edge)
            if key not in self.memo:
                self.memo[key] = self._bounds()
            known = self._own["bounds"] = self.memo[key]
        lo, hi = known
        return lo.copy(), hi.copy()

    def start_limits(self) -> Any:
        """Bounds, and the interval inside them where a start is clipped, memoised."""
        known = self._own.get("start_limits")
        if known is None:
            lower, upper = self.bounds()
            span = upper - lower
            chain = getattr(self, "chain", [])
            omap = _OrderedMap(chain, lower, upper) if len(chain) >= _PAIR else None
            known = (lower, upper, lower + 1e-9 * span, upper - 1e-9 * span, omap)
            if omap is not None:
                width = omap.upper - omap.lower
                known += (omap.lower + 1e-9 * width, omap.upper - 1e-9 * width)
            if self.memo is not None:
                self._own["start_limits"] = known
        return known

    def _bounds(self) -> Any:
        """Return the bounds of the components' and the background's parameters."""
        lows, highs = [], []
        for shape in self.shapes:
            # the bounds of a shape on this window, the same in every structure of the fit
            key = ("shape_bounds", shape)
            known = self.memo.get(key) if self.memo is not None else None
            if known is None:
                known = self._shape_bounds(shape)
                if self.memo is not None:
                    self.memo[key] = known
            lows.append(known[0])
            highs.append(known[1])
        if self.n_bg:
            lo, hi = self._background_bounds()
            lows.append(np.asarray(lo)[self.n_lin_bg :])
            highs.append(np.asarray(hi)[self.n_lin_bg :])
        if not lows:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(lows).astype(float), np.concatenate(highs).astype(float)

    def _background_bounds(self) -> tuple[Any, Any]:
        """Return the background's default bounds, the same for every structure of the fit."""
        known = self.memo.get("background_bounds") if self.memo is not None else None
        if known is None or known[0] is not self.tw or known[1] != self.background:
            lo, hi = self.background.default_bounds(self.tw)
            known = (self.tw, self.background, lo, hi)
            if self.memo is not None:
                self.memo["background_bounds"] = known
        return known[2], known[3]

    def _shape_bounds(self, shape: Any) -> tuple[NDArray[Any], NDArray[Any]]:
        """Return the bounds of a shape on this window, the latency within ``edge`` of it."""
        lo, hi = (np.array(v, float) for v in shape.default_bounds(self.tw))
        if self.edge is not None and shape.latency_param in shape.param_names:
            # A component may be cut by the window edge, but is not centred far outside
            # it (there it would only imitate the background)
            i = shape.param_names.index(shape.latency_param)
            new_lo = max(lo[i], self.tw[0] - self.edge)
            new_hi = min(hi[i], self.tw[-1] + self.edge)
            if new_lo < new_hi:
                lo[i], hi[i] = new_lo, new_hi
        return lo, hi

    def comp_params(self, theta: Any, k: Any) -> Any:
        """Return the parameters of component ``k`` in ``theta``."""
        return theta[self.offsets[k] : self.offsets[k + 1]]

    def bg_params(self, theta: Any) -> Any:
        """Return the non-linear parameters of the background in ``theta``."""
        return theta[self.offsets[-1] :]

    def design(self, theta: Any) -> Any:
        """Return the unwhitened design matrix (one column per linear coefficient)."""
        cols = [s.evaluate(self.tw, self.comp_params(theta, k)) for k, s in enumerate(self.shapes)]
        if self.n_lin_bg:
            basis = self.background.basis(self.tw, self.bg_params(theta) if self.n_bg else None)
            cols.extend(list(np.atleast_2d(basis)))
        if not cols:
            return np.zeros((len(self.tw), 0))
        return np.column_stack(cols)

    def solve(self, theta: Any) -> Any:
        """Weighted linear solution; returns (beta, residual_w, design, Q)."""
        m = self.design(theta)
        a = self.white(m) if m.shape[1] else np.zeros((len(self.y), 0))
        b = self.white(self.y)
        rows = self.prior_rows(a.shape[1])
        if rows is not None:
            a = np.vstack([a, rows])
            target = np.zeros(len(rows))
            if self.prior_col is not None:
                # the constant is drawn towards the baseline level (not 0), so that adding
                # a constant to the data changes nothing but c0
                position = int(np.sum(np.diag(self.prior_rows_mask(a.shape[1]))[: self.prior_col]))
                target[position] = self.level / cast(float, self.prior)
            b = np.r_[b, target]
        if a.shape[1] == 0:
            return np.zeros(0), b, m, np.zeros((len(b), 0))
        u, s, vt = np.linalg.svd(a, full_matrices=False)
        keep = s > s[0] * 1e-12 if s.size and s[0] > 0 else np.zeros_like(s, bool)
        u, s, vt = u[:, keep], s[keep], vt[keep]
        beta = vt.T @ ((u.T @ b) / s)
        return beta, b - a @ beta, m, u

    def _target(self) -> tuple[Any, NDArray[Any] | None]:
        """Return the whitened data followed by the targets of the prior rows, and the prior rows.

        None without prior rows; computed once per structure.
        """
        if self._aug is None:
            n_cols = len(self.shapes) + self.n_lin_bg
            # the same for every structure of a fit (one memo per fit) that holds these data
            # (the bootstrap and the cross-validation refit copies with other data)
            known = self.memo.get("white_y") if self.memo is not None else None
            if known is not None and known[0] is self.y:
                b = known[1]
            else:
                b = self.white(self.y)
                if self.memo is not None:
                    self.memo["white_y"] = (self.y, b)
            rows = self.prior_rows(n_cols)
            if rows is not None:
                target = np.zeros(len(rows))
                if self.prior_col is not None:
                    position = int(np.sum(np.diag(self.prior_rows_mask(n_cols))[: self.prior_col]))
                    target[position] = self.level / cast(float, self.prior)
                b = np.r_[b, target]
            self._aug = (b, rows)
        return self._aug

    def reduced(self, theta: Any) -> Any:
        """Return the projected residual of :meth:`solve` in the basis of the QR factorisation.

        The QR factorisation of the whitened design, and what :meth:`reduced_jacobian` and
        :meth:`reduced_solution` need.

        With ``A = Q R`` (the whitened design and the prior rows, Householder QR) the
        residual is ``(I - Q1 Q1^T) b = Q2 Q2^T b``, returned as ``z = Q2^T b``: the same
        norm, and with the Jacobian expressed in the same basis the same inner products,
        which are all the optimiser uses. A design that is not finite or not clearly of
        full rank (Frobenius condition number of ``R``, an upper bound of the 2-norm one,
        not below 1e11) takes the SVD route of :meth:`solve`, which truncates it.
        """
        b, rows = self._target()
        n = len(self.tw)
        k_max = len(self.shapes)
        c = k_max + self.n_lin_bg
        r = 0 if rows is None else len(rows)
        if c == 0:
            return self._full(theta)
        # time along the first axis: the whitening filter then runs over contiguous blocks
        a = np.empty((n + r, c))
        aux = []
        for k, shape in enumerate(self.shapes):
            a[:n, k], extra = _columns.component(shape, self.tw, self.comp_params(theta, k))
            aux.append(extra)
        if self.n_lin_bg:
            aux.append(
                _columns.background(
                    self.background, self.tw, self.bg_params(theta), a[:n, k_max:].T
                )
            )
        self.white.into(a[:n], a[:n])
        if r:
            a[n:] = rows
        if not np.isfinite(a).all():
            return self._full(theta)
        qr, tau, _, info = _dgeqrf(a)
        upper = qr[:c] * _upper_mask(c)
        inverse, info_inv = _dtrtri(upper)
        if (
            info
            or info_inv
            or not np.isfinite(inverse).all()
            or not np.sqrt((upper**2).sum() * (inverse**2).sum()) < _QR_COND_LIMIT
        ):
            return self._full(theta)
        qtb, _, _ = _dormqr("L", "T", qr, tau, b[:, None], 64)
        qtb = qtb[:, 0]
        beta, _ = _dtrtrs(qr[:c, :c], qtb[:c])
        return qtb[c:], ("qr", qr, tau, beta, a, aux)

    def _full(self, theta: Any) -> tuple[Any, tuple[str, NDArray[Any], NDArray[Any], Any]]:
        """Return the residual and the context of :meth:`solve` (the SVD route)."""
        beta, rw, _, q = self.solve(theta)
        return rw, ("svd", beta, q, rw)

    def reduced_solution(self, _theta: Any, ctx: Any) -> tuple[Any, Any]:
        """(beta, residual_w) of :meth:`solve` from the ``ctx`` of :meth:`reduced`."""
        if ctx[0] == "svd":
            return ctx[1], ctx[3]
        beta, a = ctx[3], ctx[4]
        return beta, self._target()[0] - a @ beta

    def reduced_jacobian(self, theta: Any, ctx: Any) -> Any:
        """:meth:`kaufman_jacobian` in the basis of :meth:`reduced` (the same ``ctx``)."""
        if ctx[0] == "svd":
            return self.kaufman_jacobian(theta, ctx[1], ctx[2])
        _, qr, tau, beta, _, aux = ctx
        n = len(self.tw)
        c = len(self.shapes) + self.n_lin_bg
        dm = np.zeros((qr.shape[0], self.n_theta))
        for k, shape in enumerate(self.shapes):
            dm[:n, self.offsets[k] : self.offsets[k + 1]] = beta[
                k
            ] * _columns.component_derivatives(shape, self.tw, self.comp_params(theta, k), aux[k])
        if self.n_bg:
            dm[:n, self.offsets[-1] :] = _columns.background_derivatives(
                self.background, self.tw, self.bg_params(theta), beta[len(self.shapes) :], aux[-1]
            )
        self.white.into(dm[:n], dm[:n])
        qtd, _, _ = _dormqr("L", "T", qr, tau, dm, 64 * max(self.n_theta, 1), overwrite_c=1)
        return -qtd[c:]

    def prior_rows(self, n_cols: Any) -> Any:
        """Pseudo-observations (coefficient = 0) that keep the linear coefficients sane.

        - The background constant ``c0`` has SD ``prior`` (a few baseline SD): after a
          baseline correction it must stay near zero, so that a negative component is not
          described as a negative offset plus positive components.
        - Every other coefficient (amplitudes, tail amplitudes) has SD ``scale`` (three
          times the range of the data in the window): a real component is never much
          larger than the data, while two huge components of opposite signs that cancel
          (a degenerate solution) are.
        """
        if not n_cols or (self.prior_col is None and not self.scale):
            return None
        diag = np.full(n_cols, 1.0 / self.scale if self.scale else 0.0)
        if self.prior_col is not None:
            diag[self.prior_col] = 1.0 / cast(float, self.prior)
        keep = diag > 0
        return np.diag(diag)[keep]

    def prior_rows_mask(self, n_cols: Any) -> NDArray[Any]:
        """Return the diagonal mask of the columns that carry a prior row."""
        diag = np.full(n_cols, bool(self.scale))
        if self.prior_col is not None:
            diag[self.prior_col] = True
        return np.diag(diag)

    def kaufman_jacobian(self, theta: Any, beta: Any, q: Any) -> Any:
        """Jacobian of the projected residual with respect to the non-linear parameters."""
        cols = []
        for k, shape in enumerate(self.shapes):
            jac = shape.jacobian(self.tw, self.comp_params(theta, k))  # n x p_k
            cols.append(beta[k] * jac)
        if self.n_bg:
            dj = self.background.basis_jacobian(self.tw, self.bg_params(theta))  # lin x n x nl
            coefs = beta[len(self.shapes) :]
            cols.append(np.einsum("l,lnj->nj", coefs, dj))
        if not cols:
            return np.zeros((len(self.tw), 0))
        dm = self.white(np.column_stack(cols))
        rows = self.prior_rows(len(beta))
        if rows is not None:
            dm = np.vstack([dm, np.zeros((len(rows), dm.shape[1]))])
        return -(dm - q @ (q.T @ dm))

    def full_jacobian(self, theta: Any, beta: Any) -> Any:
        """Whitened Jacobian of the model with respect to (theta, beta)."""
        nl = []
        for k, shape in enumerate(self.shapes):
            nl.append(beta[k] * shape.jacobian(self.tw, self.comp_params(theta, k)))
        if self.n_bg:
            dj = self.background.basis_jacobian(self.tw, self.bg_params(theta))
            nl.append(np.einsum("l,lnj->nj", beta[len(self.shapes) :], dj))
        parts = nl + [self.design(theta)]
        return self.white(np.column_stack(parts))


@functools.lru_cache(maxsize=64)
def _upper_mask(n: Any) -> Any:
    """``np.triu(np.ones((n, n)))``: the upper triangle of an ``n x n`` matrix, by product."""
    return np.triu(np.ones((n, n)))


class _OrderedMap:
    """Ordered parameterisation of the latencies (spec §5.5.3 step 5).

    The optimiser sees ``u = mu_1`` and ``v_k = ln(mu_{k+1} - mu_k)``, so that the components stay
    in ascending latency order inside every optimisation. Other parameters are unchanged.
    """

    def __init__(self, chain: Any, lower: Any, upper: Any) -> None:
        """Set the bounds of the log gaps between consecutive latencies of ``chain``."""
        self.idx = np.asarray(chain, int)
        lat_lo, lat_hi = lower[self.idx], upper[self.idx]
        span = float(np.max(lat_hi) - np.min(lat_lo))
        self.gap = 1e-9 * span if span > 0 else 1e-12
        self.lower, self.upper = lower.copy(), upper.copy()
        self.lower[self.idx[1:]] = np.log(self.gap)
        self.upper[self.idx[1:]] = np.log(max(span, 2 * self.gap))

    def to_phi(self, theta: Any) -> NDArray[Any]:
        """Return the parameters with the latencies after the first as log gaps."""
        phi = np.array(theta, float)
        lat = phi[self.idx]
        phi[self.idx[1:]] = np.log(np.maximum(np.diff(lat), self.gap))
        return phi

    def to_theta(self, phi: Any) -> NDArray[Any]:
        """Return the parameters with the log gaps turned back into latencies."""
        theta = np.array(phi, float)
        offsets = np.empty(len(self.idx))
        offsets[0] = 0.0
        np.cumsum(np.exp(theta[self.idx[1:]]), out=offsets[1:])
        theta[self.idx] = theta[self.idx[0]] + offsets
        return theta

    def jacobian(self, jac_theta: Any, phi: Any) -> NDArray[Any]:
        """D residual / d phi from d residual / d theta (chain rule)."""
        out = np.array(jac_theta, float)
        cols = jac_theta[:, self.idx]
        tail = np.cumsum(cols[:, ::-1], axis=1)[:, ::-1]  # sum over j >= i
        out[:, self.idx[0]] = tail[:, 0]
        out[:, self.idx[1:]] = tail[:, 1:] * np.exp(phi[self.idx[1:]])
        return out


class _Objective:
    """Residual and Jacobian of the projected problem, with the last solve cached.

    ``lean``: the residual in the basis of the QR factorisation (``_Structure.reduced``) for the
    lean optimiser; otherwise the whitened residual (a robust loss acts on each residual).
    ``omap``: the ordered parameterisation, or None.
    """

    def __init__(self, struct: Any, lean: bool, omap: Any) -> None:
        """Keep the structure, the kind of residual and the ordered map."""
        self.struct = struct
        self.lean = lean
        self.omap = omap
        self._cache: dict[Any, Any] = {}

    def solved(self, theta: Any) -> Any:
        """Return the linear solve at ``theta`` (cached for the last ``theta``)."""
        key = theta.tobytes()
        if key not in self._cache:
            self._cache.clear()
            self._cache[key] = (
                self.struct.reduced(theta) if self.lean else self.struct.solve(theta)
            )
        return self._cache[key]

    def jac_theta(self, theta: Any) -> Any:
        """Return the Jacobian of the residual with respect to ``theta``."""
        if self.lean:
            return self.struct.reduced_jacobian(theta, self.solved(theta)[1])
        beta, _, _, q = self.solved(theta)
        return self.struct.kaufman_jacobian(theta, beta, q)

    def fun(self, x: Any) -> Any:
        """Return the residual at the optimiser's variables ``x``."""
        theta = x if self.omap is None else self.omap.to_theta(x)
        return self.solved(theta)[0 if self.lean else 1]

    def jac(self, x: Any) -> Any:
        """Return the Jacobian at the optimiser's variables ``x``."""
        if self.omap is None:
            return self.jac_theta(x)
        return self.omap.jacobian(self.jac_theta(self.omap.to_theta(x)), x)


def _optimise(struct: Any, theta0: Any, settings: Any, f_scale: Any, tol: float = 1e-8) -> Any:
    """Bounded least squares on the projected residual; returns (theta, beta, rw, ok, msg)."""
    if struct.n_theta == 0:
        beta, rw, _, _ = struct.solve(theta0)
        return theta0, beta, rw, True, ""
    limits = struct.start_limits()
    lower, upper, omap = limits[0], limits[1], limits[4]
    # np.clip, without its overhead
    theta0 = np.minimum(np.maximum(theta0, limits[2]), limits[3])
    lean = settings.loss == "linear" and _OPTIMISER != "scipy"
    if omap is None:
        x0, lo, hi = theta0, lower, upper
    else:
        lo, hi = omap.lower, omap.upper
        x0 = np.minimum(np.maximum(omap.to_phi(theta0), limits[5]), limits[6])
    if lean and _OPTIMISER == "trf":
        # the compiled fit first: it needs none of the objective below
        done = _optimise_compiled(struct, x0, lo, hi, omap, settings, tol)
        if done is not None:
            return (theta0 if done[0] is None else done[0], *done[1:])
    objective = _Objective(struct, lean, omap)
    try:
        xr = _least_squares(objective, x0, (lo, hi), settings, f_scale, tol)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return theta0, None, None, False, f"{type(exc).__name__}: {exc}"
    x = xr if omap is None else omap.to_theta(xr)
    if lean:
        beta, rw = struct.reduced_solution(x, objective.solved(x)[1])
    else:
        beta, rw, _, _ = struct.solve(x)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(beta))):
        return x, None, None, False, "non-finite parameters"
    return x, beta, rw, True, ""


def _least_squares(
    objective: _Objective, x0: Any, bounds: Any, settings: Any, f_scale: Any, tol: float
) -> Any:
    """Run the optimiser of ``_OPTIMISER`` from ``x0``; returns the solution."""
    lo, hi = bounds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if objective.lean:
            # the same algorithm as least_squares(method="trf"), lighter (models/_trf.py)
            options = TrfOptions(
                ftol=tol,
                xtol=tol,
                gtol=tol,
                max_nfev=settings.max_iter,
                reduced=_OPTIMISER != "trf-full",
                m=len(objective.struct._target()[0]),
                final_jacobian=False,
            )
            xr, _, nfev, _ = least_squares_trf(objective.fun, objective.jac, x0, lo, hi, options)
        else:
            res = least_squares(
                objective.fun,
                x0,
                jac=objective.jac,
                bounds=(lo, hi),
                method="trf",
                loss=settings.loss,
                f_scale=f_scale or 1.0,
                max_nfev=settings.max_iter,
                x_scale="jac",
                ftol=tol,
                xtol=tol,
                gtol=tol,
            )
            xr, nfev = res.x, res.nfev
    _COUNTER.n = getattr(_COUNTER, "n", 0) + int(nfev)
    return xr


def _compiled_args(struct: Any) -> tuple[Any, ...] | None:
    """Return what :func:`_fast.optimise` needs to know of a structure, or None.

    Covered: Gaussian and skewed-Gaussian components, the exponential background or one
    without non-linear parameters (its basis is then fixed), the whitening of the engine.
    None for any other structure. Memoised on the structure.
    """
    if "compiled" in struct._own:
        return cast("tuple[Any, ...] | None", struct._own["compiled"])
    out = _compiled_spec(struct)
    if struct.memo is not None:
        struct._own["compiled"] = out
    return out


def _compiled_spec(struct: Any) -> tuple[Any, ...] | None:
    """Return the arguments of :func:`_fast.optimise` for a structure, or None (uncached)."""
    kinds = []
    for shape in struct.shapes:
        if type(shape) is Gaussian:
            kinds.append(_fast.GAUSSIAN)
        elif type(shape) is SkewedGaussian:
            kinds.append(_fast.SKEWED_GAUSSIAN)
        else:
            return None
    # the time axis, background and whitening are those of every structure of the fit
    key = ("compiled", struct.n_lin_bg)
    shared = struct.memo.get(key) if struct.memo is not None else None
    if shared is None:
        shared = _compiled_shared(struct)
        if struct.memo is not None:
            struct.memo[key] = shared
    if shared is False:
        return None
    n_cols = len(struct.shapes) + struct.n_lin_bg
    diag = np.zeros(n_cols)
    if n_cols and (struct.prior_col is not None or struct.scale):
        diag[:] = 1.0 / struct.scale if struct.scale else 0.0
        if struct.prior_col is not None:
            diag[struct.prior_col] = 1.0 / cast(float, struct.prior)
    prior_cols = np.flatnonzero(diag > 0).astype(np.int64)
    target = np.ascontiguousarray(struct._target()[0], dtype=float)
    if len(target) != len(struct.tw) + len(prior_cols):
        return None
    tw, bg_kind, lag, fixed, inv_scale, a, post, stride, start, start_rows = shared
    model = (
        tw,
        np.asarray(kinds, dtype=np.int64),
        np.asarray(struct.offsets, dtype=np.int64),
        bg_kind,
        lag,
        fixed,
    )
    whitening = (inv_scale, a, post, stride, start, start_rows)
    return model, whitening, (prior_cols, np.ascontiguousarray(diag[prior_cols]), target)


def _compiled_shared(struct: Any) -> Any:
    """Return the arguments of :func:`_fast.optimise` common to the structures of a fit.

    False when the kernel does not cover the whitening or the background.
    """
    tw = np.ascontiguousarray(struct.tw, dtype=float)
    background = struct.background
    lag = np.zeros(len(tw))
    if type(background) is ExponentialBackground:
        bg_kind = _fast.BG_EXPONENTIAL
        lag = np.ascontiguousarray(tw - background._t_ref(tw))
        fixed = np.zeros((0, len(tw)))
    elif struct.n_bg == 0:
        bg_kind = _fast.BG_FIXED
        fixed = (
            np.ascontiguousarray(np.atleast_2d(background.basis(tw, None)), dtype=float)
            if struct.n_lin_bg
            else np.zeros((0, len(tw)))
        )
        if fixed.shape != (struct.n_lin_bg, len(tw)):
            return False
    else:
        return False
    white = struct.white
    p = len(white.a) - 1
    if type(white) is Whitener:
        stride, start_rows = 1, np.arange(p, dtype=np.int64)
    elif type(white) is StridedWhitener:
        stride, start_rows = white.stride, np.asarray(white.start_rows, dtype=np.int64)
    else:
        return False
    start = np.zeros((0, 0)) if white.start is None else np.ascontiguousarray(white.start)
    return (
        tw,
        bg_kind,
        lag,
        fixed,
        1.0 / np.broadcast_to(np.asarray(white.scale, dtype=float), tw.shape),
        np.ascontiguousarray(white.a, dtype=float),
        white._row_weight / white.innovation,
        int(stride),
        start,
        start_rows,
    )


def _optimise_compiled(
    struct: Any, x0: Any, lo: Any, hi: Any, omap: Any, settings: Any, tol: float
) -> Any:
    """:func:`_optimise` by the compiled kernel; None when it does not apply."""
    kernel = _fast.kernel() if _FAST != "off" else None
    spec = _compiled_args(struct) if kernel is not None else None
    if kernel is None or spec is None:
        return None
    chain = _NO_CHAIN if omap is None else np.asarray(omap.idx, dtype=np.int64)
    start = (
        np.ascontiguousarray(x0, dtype=float),
        np.ascontiguousarray(lo, dtype=float),
        np.ascontiguousarray(hi, dtype=float),
    )
    status, theta, nfev, beta, rw = kernel(start, chain, spec, float(tol), int(settings.max_iter))
    if status == _fast.FALLBACK:
        return None  # the SVD route of the NumPy path
    _COUNTER.n = getattr(_COUNTER, "n", 0) + int(nfev)
    return _compiled_outcome(status, theta, beta, rw)


def _compiled_outcome(status: Any, theta: Any, beta: Any, rw: Any) -> Any:
    """Return ``(theta, beta, rw, ok, msg)`` from the status of the compiled kernel."""
    if status == _fast.VALUE_ERROR:
        return None, None, None, False, "ValueError: infeasible start or step"
    if status == _fast.LINALG_ERROR:
        return None, None, None, False, "LinAlgError: SVD did not converge"
    if not (_all_finite(theta) and _all_finite(beta)):
        return theta, None, None, False, "non-finite parameters"
    return theta, beta, rw, True, ""


_NO_CHAIN = np.zeros(0, dtype=np.int64)


def _all_finite(x: Any) -> bool:
    """Return ``np.all(np.isfinite(x))``.

    A finite sum is enough (NaN and infinities propagate); the entries are checked one by one
    only when the sum is not finite (also on an overflow).
    """
    return math.isfinite(x.sum()) or bool(np.isfinite(x).all())


###################################################################################################
## Candidates


def _smooth(values: Any, sfreq: float, sigma_s: Any) -> Any:
    """Return ``values`` smoothed by a Gaussian kernel of ``sigma_s`` seconds."""
    width = sigma_s * sfreq
    if width < _MIN_SMOOTH_SAMPLES:
        return values
    half = int(np.ceil(4 * width))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / width) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, half, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def candidates(
    _tw: Any, residual: Any, valid: Any, sigma: Any, sfreq: float, settings: Any
) -> tuple[list[Any], Any]:
    """Extrema of the smoothed, standardised residual (edges included), strongest first."""
    r = np.where(valid, residual, 0.0)
    rs = _smooth(r, sfreq, settings.min_fwhm / FWHM_PER_SIGMA / 2) / sigma
    mag = np.abs(rs)
    n = len(mag)
    inner = np.flatnonzero((mag[1:-1] >= mag[:-2]) & (mag[1:-1] > mag[2:])) + 1
    edge = max(1, int(round(settings.min_fwhm * sfreq)))
    picks: list[Any] = list(inner)
    if mag[0] >= mag[: edge + 1].max():
        picks.append(0)
    if mag[-1] >= mag[-edge - 1 :].max():
        picks.append(n - 1)
    if n < _FEW_SAMPLES or not np.isfinite(mag).all():
        picks = _select_list(picks, valid, rs, mag, settings.polarity)
    elif picks:
        picks = _select_array(picks, valid, rs, mag, settings.polarity)
    threshold = settings.detection_threshold
    levels = mag[np.asarray(picks, dtype=np.int64)] if picks else np.zeros(0)
    above = [p for p, ok in zip(picks, (levels >= threshold).tolist(), strict=True) if ok]
    if settings.threshold_role == "order":
        # the threshold orders the candidates (those above it first) but does not remove
        # any: acceptance is decided by the criterion alone (REQ-104)
        below = [p for p, ok in zip(picks, (levels < threshold).tolist(), strict=True) if ok]
        return above + below, rs * sigma
    return above, rs * sigma


_FEW_SAMPLES = 3  # candidates selects on Python lists below this many samples


def _select_list(picks: list[Any], valid: Any, rs: Any, mag: Any, polarity: str) -> list[Any]:
    """Return the valid picks of the polarity, strongest first (stable), on Python lists."""
    picks = [p for p in dict.fromkeys(picks) if valid[p]]
    if polarity == "negative":
        picks = [p for p in picks if rs[p] < 0]
    elif polarity == "positive":
        picks = [p for p in picks if rs[p] > 0]
    picks.sort(key=lambda p: -mag[p])
    return picks


def _select_array(picks: list[Any], valid: Any, rs: Any, mag: Any, polarity: str) -> list[Any]:
    """:func:`_select_list` with arrays: the same selection and stable order.

    The inner extrema lie in 1 .. n - 2, so the edges add no duplicate; the picks keep
    their types.
    """
    index = np.asarray(picks, dtype=np.int64)
    keep = np.asarray(valid, bool)[index]
    if polarity == "negative":
        keep &= rs[index] < 0
    elif polarity == "positive":
        keep &= rs[index] > 0
    chosen = np.flatnonzero(keep)
    order = chosen[np.argsort(-mag[index[chosen]], kind="stable")]
    return [picks[i] for i in order]


###################################################################################################
## One fit state


@dataclasses.dataclass
class _State:
    """One fit: shapes, parameters, amplitudes, whitened residual and criteria."""

    shapes: list[Any]
    theta: NDArray[Any]
    beta: NDArray[Any]
    rw: NDArray[Any]
    crit: float  # selection criterion: innovation rows only (decides K)
    fit_crit: float | None = None  # the same on every row: compares optima of one structure
    # the same with the penalty of ``shape_choice`` ("hqc": 2 ln ln n per parameter, "aic":
    # 2): compares the shapes of one component in the final shape choice
    shape_crit: float | None = None


def _shape(name: Any, settings: Any) -> "ComponentShape":
    """Return the shape ``name`` with the width and frequency limits of ``settings``."""
    values = (
        ("min_fwhm", settings.min_fwhm),
        ("max_fwhm", settings.max_fwhm),
        ("freq_min", settings.freq_min),
        ("freq_max", settings.freq_max),
    )
    # a shape class registered under the name builds an immutable shape (a frozen
    # dataclass) from the same settings every time: the forward search asks for it once per
    # candidate and shape, and gets the one already built
    entry = SHAPES.get(name) if name in SHAPES else None
    params = (
        getattr(entry, "__dataclass_params__", None)
        if isinstance(entry, type) and dataclasses.is_dataclass(entry)
        else None
    )
    frozen = params is not None and params.frozen
    key = (name, entry, values) if frozen else None
    if key is not None:
        known = _SHAPES_BUILT.get(key)
        if known is not None:
            return known
    fields = {f.name for f in dataclasses.fields(cast(Any, type(get_shape(name))))}
    kwargs = {k: v for k, v in values if k in fields and v is not None}
    shape = get_shape(name, **kwargs)
    if key is not None:
        if len(_SHAPES_BUILT) >= _SHAPES_BUILT_MAX:
            _SHAPES_BUILT.clear()
        _SHAPES_BUILT[key] = shape
    return shape


_SHAPES_BUILT: dict[Any, "ComponentShape"] = {}
_SHAPES_BUILT_MAX = 256  # the cache is emptied when it holds this many shapes


def _param_latencies(struct: Any, theta: Any) -> FloatArray:
    """Latency parameter of each component (the parameter the ordering acts on)."""
    out = []
    for k, shape in enumerate(struct.shapes):
        params = struct.comp_params(theta, k)
        out.append(
            float(params[shape.param_names.index(shape.latency_param)])
            if shape.latency_param in shape.param_names
            else _latency_of(shape, params)
        )
    return np.asarray(out)


def _cancelling_pair(struct: Any, theta: Any, beta: Any, settings: Any) -> bool:
    """Return True when two components of opposite signs overlap.

    Their latencies are closer than ``overlap_c * (sigma_a + sigma_b)``, the overlap of spec §5.5.3
    step 5 (sigma = FWHM / 2.3548 of each fitted component).

    Such a pair is not identifiable: two large components of opposite signs at almost the
    same latency describe one deflection (or a biphasic complex) as their difference, with
    the same fit as the right components and arbitrary amplitudes (on clean ECAP
    simulations -320 and +218 uV at 0.587 and 0.601 ms for a true N1 of -137 uV at 0.489 ms
    and P2 of +111 uV at 0.752 ms). The engine does not accept a model holding one; the
    search goes on with the other starts and candidates.
    """
    k_max = len(struct.shapes)
    if k_max < _PAIR:
        return False
    amps = beta[:k_max]
    if np.all(amps >= 0) or np.all(amps <= 0):
        return False
    lat, sig = [], []
    for k in range(k_max):
        feats = _component_features(struct, theta, k, amps[k])
        lat.append(feats.latency)
        sig.append(feats.fwhm / FWHM_PER_SIGMA if np.isfinite(feats.fwhm) else 0.0)
    return bool(cancelling_pairs(lat, sig, amps, settings.overlap_c))


def _component_features(struct: Any, theta: Any, k: int, amplitude: Any) -> Any:
    """``compute_features`` of component ``k`` with its amplitude, at the window's sampling rate.

    The same component is measured several times during a fit (the cancelling-pair test, the
    merge pass, the reseeding test): the features are kept in the memo of the fit, by shape,
    parameters and amplitude, for the time axis they were measured on.
    """
    shape = struct.shapes[k]
    params = struct.comp_params(theta, k)
    memo = struct.memo
    known: dict[Any, Any] | None = None
    if memo is not None:
        entry = memo.get("features")
        if entry is None or entry[0] is not struct.tw:
            entry = memo["features"] = (struct.tw, {}, 1.0 / np.median(np.diff(struct.tw)))
        known, sfreq = entry[1], entry[2]
        key = (shape, params.tobytes(), float(amplitude))
        feats = known.get(key)
        if feats is not None:
            return feats
    else:
        sfreq = 1.0 / np.median(np.diff(struct.tw))
    feats = compute_features(shape, struct.tw, params, amplitude, sfreq=sfreq)
    if known is not None:
        known[key] = feats
    return feats


def cancelling_pairs(latencies: Any, sigmas: Any, amplitudes: Any, overlap_c: Any) -> list[Any]:
    """Return the index pairs ``(a, b)`` of the cancelling components.

    Components of opposite signs whose latencies are closer than
    ``overlap_c * (sigma_a + sigma_b)`` (the identifiability rule of the engine, also used by
    :func:`evoparam.spatial.fit_shared`). A zero amplitude has no sign.
    """
    out = []
    for a in range(len(latencies)):
        for b in range(a + 1, len(latencies)):
            if amplitudes[a] * amplitudes[b] < 0 and abs(
                latencies[a] - latencies[b]
            ) < overlap_c * (sigmas[a] + sigmas[b]):
                out.append((a, b))
    return out


def _reorder(
    struct_maker: Any, shapes: Any, theta: Any, order: Any
) -> tuple[list[Any], NDArray[Any]]:
    """Return the shapes and parameters in the order ``order``."""
    struct = struct_maker(shapes)
    shapes = [shapes[i] for i in order]
    theta = (
        np.concatenate([struct.comp_params(theta, i) for i in order] + [struct.bg_params(theta)])
        if len(order)
        else theta
    )
    return shapes, theta


def _evaluate_state(
    struct_maker: Any,
    shapes: Any,
    theta0: Any,
    settings: Any,
    n_eff: Any,
    f_scale: Any,
    kind: Any,
    tol: float = 1e-8,
) -> Any:
    """Fit ``shapes`` from ``theta0``; returns ``(state or None, message)``.

    None with the message when the optimiser fails or the fit holds a cancelling pair (with
    ``reject_cancelling``).
    """
    # n_eff is the number of whitened (independent) samples used by the criterion
    struct = struct_maker(shapes)
    if struct.ordered and len(shapes) >= _PAIR:
        # the start must respect the order of the parameterisation
        order = np.argsort(_param_latencies(struct, theta0), kind="stable")
        if np.any(order != np.arange(len(order))):
            shapes, theta0 = _reorder(struct_maker, list(shapes), np.asarray(theta0, float), order)
            struct = struct_maker(shapes)
    theta, beta, rw, ok, msg = _optimise(struct, theta0, settings, f_scale, tol)
    if not ok:
        return None, msg
    if settings.reject_cancelling and _cancelling_pair(struct, theta, beta, settings):
        return None, "cancelling_pair"
    p = struct.n_theta + len(beta)
    data_rw = rw[: len(struct.y)]
    fit_crit = criterion(float(np.sum(data_rw**2)), n_eff, p, kind)
    # The start rows of the exact likelihood (Whitener) tie the curve to the data at the
    # window start, so they take part in every fit; but the evidence for a component (the
    # change of K, merges) counts the innovation rows only. The start rows judge the first
    # samples against the stationary spread of the noise, and a chance deviation there was
    # otherwise "explained" by a recovery tail plus a broad component cancelling it (on
    # simulations: a spurious N1 at the window start; REQ-170, REQ-171).
    exact = getattr(struct.white, "start", None) is not None
    skip = struct.white.p if exact else 0
    crit = (
        criterion(float(np.sum(data_rw[struct.white.skip :] ** 2)), n_eff - skip, p, kind)
        if skip
        else fit_crit
    )
    shape_crit = criterion(float(np.sum(data_rw**2)), n_eff, p, _shape_kind(settings))
    return _State(list(shapes), theta, beta, rw, crit, fit_crit, shape_crit), ""


def _polish(state: Any, search: "_Search") -> Any:
    """Re-optimise a state from several starting values of the background's time constants.

    A background fitted before a component existed may hold that component (a slow exponential that
    mimics a broad N2), a local optimum the optimiser would keep.
    """
    struct = search.make(state.shapes)
    if not struct.n_bg:
        return state
    lo, hi = struct.bounds()
    lo, hi = lo[struct.offsets[-1] :], hi[struct.offsets[-1] :]
    best = state
    for value in np.geomspace(np.maximum(lo, 1e-6), hi, 4):
        theta0 = np.concatenate((state.theta[: struct.offsets[-1]], value))
        cand, _ = search.evaluate(state.shapes, theta0)
        if cand is not None and cand.fit_crit < best.fit_crit:
            best = cand
    return best


def _latency_of(shape: Any, params: Any) -> float:
    """Return the latency of a shape with parameters ``params``."""
    lat = getattr(shape, "latency", None)
    return float(lat(params)) if lat is not None else float(params[0])


def _starts(shape: Any, guess: Any, n_starts: Any, tw: Any) -> list[NDArray[Any]]:
    """Return the guess and perturbations of its latency parameter by +/- a quarter width."""
    out = [np.asarray(guess, float)]
    if n_starts <= 1:
        return out
    idx = (
        shape.param_names.index(shape.latency_param)
        if shape.latency_param in shape.param_names
        else 0
    )
    lo, hi = shape.default_bounds(tw)
    step = max((tw[-1] - tw[0]) / 50, 2 * (tw[1] - tw[0]))
    for sign in (-1, 1):
        g = np.array(guess, float)
        g[idx] = np.clip(g[idx] + sign * step, lo[idx], hi[idx])
        out.append(g)
    return out[:n_starts]


# residual evaluations of the least-squares solves of the waveform being fitted (one
# waveform is fitted in one thread)
_COUNTER = threading.local()


def noise_phase(z: Any, index: Any, k: Any) -> int:
    """Return the phase (0 .. k - 1) of the strided samples that hold the noise ``z`` best.

    The samples ``index % k == phase`` of the noise sampled at ``index``: the phase whose strided
    noise has the largest level at the top of its band (median Hann-tapered periodogram of the
    detrended noise over its top 5 %, and at least ``MIN_BAND_POINTS`` points). 0 when a phase has
    fewer than ``4 * MIN_BAND_POINTS`` samples.

    An upsampled recording keeps its own samples on one phase (a windowed-sinc interpolator
    such as ``prep.resample`` passes them through). The others are interpolated, and the
    transition band of a finite interpolation filter, centred on the old Nyquist frequency,
    attenuates them there; on ds004080 run 021448 they also carry the filter's ringing next
    to the stimulation artifact (fitting them gives an N1 on 43 channels, the recorded
    samples 12; CHANGELOG). On that run upsampled to 1024 Hz and cropped by 0 or 1 sample,
    it picks the recorded samples on all 100 channels, by 0.6 decades of level or more. On
    upsampled noise (``resample_poly``, 300 samples or more per phase, k = 2 to 4) it never
    missed; with 60 per phase it missed 1 in 80 at k = 2 and 1 in 4 at k = 4.

    Limits. An ideal (FFT) interpolation leaves no phase special, and none is picked better
    than by chance; it also leaves no ringing that only some phases carry. Neither is a
    recording low-passed well below its Nyquist frequency: its phases differ by chance and so
    do its fits (on low-passed simulations, fits on the phase picked are neither better nor
    worse than on a fixed phase; CHANGELOG).
    """
    z = np.asarray(z, float)
    index = np.asarray(index)
    levels = []
    for phase in range(k):
        x = z[index % k == phase]
        if len(x) < 4 * MIN_BAND_POINTS:
            return 0
        t = np.arange(len(x), dtype=float)
        x = (x - np.polyval(np.polyfit(t, x, 1), t)) * np.hanning(len(x))
        power = np.abs(np.fft.rfft(x)) ** 2
        levels.append(np.median(power[-max(MIN_BAND_POINTS, len(power) // 20) :]))
    return int(np.argmax(levels))


def _noise_stride(times: Any, y: Any, settings: Any, mask: Any, noise_ref: Any) -> Any:
    """Return ``(noise_stride, noise_phase)`` of the noise the AR model is fitted to.

    That noise is the plus-minus reference or the baseline; ``(1, 0)`` for the other noise models
    or without enough noise samples.
    """
    if settings.noise_model != "ar":
        return 1, 0
    if noise_ref is not None:
        ref = np.asarray(noise_ref, float)
        keep = np.isfinite(ref) & ~mask
        z = ref[keep]
    elif settings.baseline is not None:
        keep = (
            (times >= settings.baseline[0])
            & (times <= settings.baseline[1])
            & ~mask
            & np.isfinite(y)
        )
        z = y[keep]
    else:
        return 1, 0
    stride = noise_stride(z) if len(z) >= MIN_BASELINE else 1
    if stride == 1:
        return 1, 0
    return stride, noise_phase(z, np.flatnonzero(keep), stride)


def fit_waveform_core(
    times: Any,
    y: Any,
    *,
    settings: Any,
    sfreq: float,
    mask: Any = None,
    sem: Any = None,
    select_k: Any = None,
    noise_ref: Any = None,
) -> Any:
    """Fit one waveform. Returns a dict with everything the result object needs.

    Keys: ``status``, ``message``, ``warnings``, ``shapes`` (objects), ``theta``, ``beta``,
    ``struct`` (final structure), ``sigma``, ``rho``, ``n_eff``, ``window_mask``, ``model``,
    ``residual``, ``crit``, ``best_rejected_delta``, ``steps`` (provenance lines),
    ``n_evaluations`` (residual evaluations of the least-squares solves), ``noise_reference``.
    """
    _COUNTER.n = 0
    out = _fit_waveform_core(
        times,
        y,
        settings=settings,
        sfreq=sfreq,
        mask=mask,
        sem=sem,
        select_k=select_k,
        noise_ref=noise_ref,
    )
    out["n_evaluations"] = int(_COUNTER.n)
    return out


def _fit_waveform_core(
    times: Any,
    y: Any,
    *,
    settings: Any,
    sfreq: float,
    mask: Any = None,
    sem: Any = None,
    select_k: Any = None,
    noise_ref: Any = None,
) -> Any:
    """Fit one waveform (see :func:`fit_waveform_core`), without the evaluation count."""
    times = np.asarray(times, float)
    y = np.asarray(y, float)
    mask = np.zeros(len(y), bool) if mask is None else np.asarray(mask, bool).copy()
    mask |= ~np.isfinite(y)
    if settings.fit_window is None:
        raise SettingsError("fit_window is required (no preset gives one).")
    out: dict[str, Any] = {"warnings": [], "steps": [], "best_rejected_delta": np.nan}
    settings, band = _noise_band(times, y, settings, sfreq, mask, noise_ref, out)
    win = settings.fit_window
    if win[1] - win[0] < MIN_WINDOW_FWHMS * settings.min_fwhm:
        raise SettingsError(f"fit_window {win} is shorter than 3 x min_fwhm.")
    window = _Window.of(times, y, mask, win)
    out["window_mask"] = window.wmask
    sigma, rho = _noise_scale(window, settings, noise_ref, out)
    if (
        window.valid.mean() < 1 - settings.max_masked_fraction
        or window.valid.sum() < MIN_VALID_SAMPLES
    ):
        out.update(status="skipped", message="masked: too few valid samples in the window")
        return out
    if not sigma > 0:
        out.update(status="skipped", message="flat")
        return out
    white, ar = _whitener(window, settings, _Noise(sigma, sem, noise_ref, band))
    n_valid = int(window.valid.sum())
    n_eff = n_valid * (1 - rho) / (1 + rho)
    n_white: float = white.n
    if settings.noise_model == "neff":
        # spec §5.5.3 step 1 to the letter: residuals scaled by sigma_n, not whitened, and
        # the AR(1) effective count n_eff = n_W (1 - rho) / (1 + rho) in every criterion
        # and in the covariance
        n_white = n_eff
    out.update(n_eff=n_eff, n_white=n_white, ar_order=white.p)
    search = _Search(
        make=_structure_maker(window, white, settings, sigma),
        settings=settings,
        n_eff=n_white,
        f_scale=settings.f_scale,
        kind="aicc" if settings.selection == "aicc" else "bic",
        window=window,
        sigma=sigma,
        sfreq=sfreq,
    )
    current = _background_only(search)
    if current is None:
        out.update(status="failed", message="the background-only model could not be fitted")
        logger.warning("fit failed: %s", out["message"])
        return out
    out["steps"].append({"K": 0, "criterion": current.crit})
    out["path"] = [current]
    forced = settings.selection in ("fixed", "cv")
    _calibrate(search, current, white, ar, band, forced, out)
    current, best_rejected = _forward_search(current, search, forced, out)
    if current.shapes and not forced and settings.reseed:
        current = _reseed_step(current, search, out)
    if current.shapes and not forced:
        current = _final_checks(current, search, white, out)
    out["best_rejected_delta"] = float(best_rejected) if np.isfinite(best_rejected) else np.nan
    out["make"] = search.make
    if settings.selection == "cv" and select_k is not None:
        current = out["path"][min(select_k(out["path"], search.make), len(out["path"]) - 1)]
    return _finish(out, current, search, y[window.wmask])


@dataclasses.dataclass(frozen=True)
class _Band:
    """The rate the noise occupies: every ``stride``-th sample, on the samples of ``phase``."""

    stride: int
    phase: int


@dataclasses.dataclass(frozen=True)
class _Window:
    """The epoch, its mask and the samples of the fit window."""

    times: NDArray[Any]
    y: NDArray[Any]
    mask: NDArray[Any]
    wmask: NDArray[Any]
    tw: NDArray[Any]
    yw: NDArray[Any]
    valid: NDArray[Any]

    @classmethod
    def of(cls, times: Any, y: Any, mask: Any, win: Any) -> "_Window":
        """Cut the fit window ``win`` out of the epoch; masked samples are set to 0."""
        wmask = (times >= win[0]) & (times <= win[1])
        return cls(
            times=times,
            y=y,
            mask=mask,
            wmask=wmask,
            tw=times[wmask],
            yw=np.where(mask[wmask], 0.0, y[wmask]),
            valid=~mask[wmask],
        )

    def baseline(self, settings: Any) -> Any:
        """Unmasked samples of the epoch inside the baseline of ``settings``."""
        return (
            (self.times >= settings.baseline[0])
            & (self.times <= settings.baseline[1])
            & ~self.mask
        )


@dataclasses.dataclass(frozen=True)
class _Noise:
    """The noise scale, the standard errors given by the caller and the noise reference."""

    sigma: float
    sem: Any
    noise_ref: Any
    band: _Band


@dataclasses.dataclass
class _Search:
    """What every step of the forward search shares: structures, criterion and window."""

    make: Callable[[Any], Any]
    settings: Any
    n_eff: float
    f_scale: Any
    kind: str
    window: _Window
    sigma: float
    sfreq: float
    # the calibrated acceptance of this fit (None: the fixed fall ``fixed_delta``)
    cal: Any = None
    fixed_delta: float = FIXED_DELTA

    @property
    def tw(self) -> NDArray[Any]:
        """Times of the fit window."""
        return self.window.tw

    def evaluate(self, shapes: Any, theta0: Any, tol: float = 1e-8) -> Any:
        """Fit ``shapes`` from ``theta0``: :func:`_evaluate_state` with the shared settings."""
        return _evaluate_state(
            self.make, shapes, theta0, self.settings, self.n_eff, self.f_scale, self.kind, tol
        )

    def need(self, state: Any, ref: Any) -> float:
        """Fall of the criterion needed to accept ``state`` (one component more than ``ref``)."""
        if self.cal is None:
            return self.fixed_delta
        return _need(self.cal, state, ref, self.make)

    def residual_of(self, state: Any) -> Any:
        """Residual of ``state`` on the (masked) samples of the fit window."""
        return self.window.yw - self.make(state.shapes).design(state.theta) @ state.beta

    def candidates(self, state: Any, settings: Any) -> tuple[list[Any], Any]:
        """Candidate extrema of the residual of ``state`` (see :func:`candidates`)."""
        return candidates(
            self.tw, self.residual_of(state), self.window.valid, self.sigma, self.sfreq, settings
        )


def _noise_band(
    times: Any, y: Any, settings: Any, sfreq: float, mask: Any, noise_ref: Any, out: Any
) -> tuple[Any, _Band]:
    """Return the settings (``min_fwhm`` raised to what the noise band carries) and the band."""
    stride, phase = _noise_stride(times, y, settings, mask, noise_ref)
    out["noise_stride"], out["noise_phase"] = stride, phase
    if stride > 1:
        out["warnings"].append(f"band_limited_noise:{stride}")
    min_fwhm = settings.min_fwhm
    if sfreq / stride < 2.0 / min_fwhm:
        # two samples at the rate the noise occupies (the recorded rate when stride is 1):
        # the narrowest shape that band carries, and the evidence lies on those samples only
        min_fwhm = 2.0 * stride / sfreq
        out["warnings"].append("min_fwhm_raised")
        settings = dataclasses.replace(
            settings, min_fwhm=min_fwhm, max_fwhm=max(settings.max_fwhm, 2 * min_fwhm)
        )
    return settings, _Band(stride, phase)


def _noise_scale(window: _Window, settings: Any, noise_ref: Any, out: Any) -> tuple[Any, Any]:
    """Return the noise scale ``sigma`` and lag-1 autocorrelation ``rho``."""
    out["noise_reference"] = "plusminus" if noise_ref is not None else "baseline"
    if noise_ref is not None:
        # the plus-minus reference: the noise of the average over the whole epoch (trials of
        # alternating sign averaged), longer than any baseline
        ref = np.asarray(noise_ref, float)
        good = np.isfinite(ref) & ~window.mask
        sigma = MAD_TO_SD * float(np.median(np.abs(ref[good] - np.median(ref[good]))))
        rho = float(np.corrcoef(ref[good][1:], ref[good][:-1])[0, 1])
        warn = None
    else:
        sigma, rho, warn = noise_model(
            window.times, window.y, settings.baseline, window.wmask, window.mask
        )
    if warn:
        out["warnings"].append(warn)
    out.update(sigma=sigma, rho=rho)
    return sigma, rho


@dataclasses.dataclass(frozen=True)
class _ArNoise:
    """AR model of the noise and the standardised samples it was estimated on (or None)."""

    a: NDArray[Any]
    innovation: float
    acov: Any
    z: Any


def _ar_coefficients(window: _Window, settings: Any, noise: _Noise, grid: Any) -> _ArNoise:
    """AR model of the noise on the samples of ``grid``."""
    z = None
    if settings.noise_model == "ar" and noise.noise_ref is not None:
        ref = np.asarray(noise.noise_ref, float)
        good = np.isfinite(ref) & ~window.mask
        z = ref[good & grid] / noise.sigma
    elif settings.noise_model == "ar" and settings.baseline is not None:
        base = window.baseline(settings)
        if base.sum() >= MIN_BASELINE:
            z = window.y[base & grid] / noise.sigma
    if z is None:
        return _ArNoise(np.array([1.0]), 1.0, None, None)
    a, innovation, acov = ar_model(z, return_acov=True)
    return _ArNoise(a, innovation, acov, z)


def _whitener(window: _Window, settings: Any, noise: _Noise) -> tuple[Whitener, _ArNoise]:
    """Return the whitening of the fit window by the noise model of ``settings``."""
    sigma, band = noise.sigma, noise.band
    scale: Any = np.full(len(window.tw), sigma)
    if noise.sem is not None:
        s_ = np.asarray(noise.sem, float)[window.wmask]
        scale = np.maximum(np.nan_to_num(s_, nan=sigma), _SEM_FLOOR * sigma)
    # the noise of a band-limited recording is modelled at the rate it occupies: on every
    # stride-th sample, on the phase of the epoch that holds the noise best (noise_phase:
    # an upsampled recording keeps its own samples there)
    grid = np.arange(len(window.y)) % band.stride == band.phase
    ar = _ar_coefficients(window, settings, noise, grid)
    a_coefs, innovation, acov = ar.a, ar.innovation, ar.acov
    if noise.sem is not None:
        innovation = 1.0 if len(a_coefs) == 1 else innovation
    ar = dataclasses.replace(ar, innovation=innovation)
    if band.stride > 1:
        strided = StridedWhitener(
            scale,
            a_coefs,
            innovation,
            window.valid,
            band.stride,
            phase=int(np.argmax(grid[window.wmask])),
            acov=acov,
        )
        return strided, ar
    return Whitener(scale, a_coefs, innovation, window.valid, acov=acov), ar


def _background(settings: Any) -> Any:
    """Return the background of ``settings``, with ``background_tau_max`` when it takes one."""
    bg_class = type(get_background(settings.background))
    # the settings a background takes (a plugin background need not be a dataclass)
    bg_fields: set[str] = (
        {f.name for f in dataclasses.fields(cast(Any, bg_class))}
        if dataclasses.is_dataclass(bg_class)
        else set()
    )
    bg_kwargs = {}
    if settings.background_tau_max is not None and "tau_max" in bg_fields:
        # Recovery tails decay within tens of ms; a slower "tail" would only imitate an
        # offset or a broad component
        bg_kwargs["tau_max"] = settings.background_tau_max
    return get_background(settings.background, **bg_kwargs)


def _structure_maker(
    window: _Window, white: Whitener, settings: Any, sigma: float
) -> Callable[[Any], Any]:
    """Return ``make(shapes)``: the structure of a sequence of shapes on this fit window."""
    background = _background(settings)
    tw, yw, valid = window.tw, window.yw, window.valid
    prior = (
        settings.background_prior * sigma
        if settings.background_prior and settings.baseline is not None
        else None
    )
    data_range = float(np.ptp(yw[valid])) if valid.any() else 0.0
    scale = settings.amplitude_prior * max(data_range, sigma) if settings.amplitude_prior else None
    edge = (
        settings.edge_extension * (tw[-1] - tw[0]) if settings.edge_extension is not None else None
    )
    level = 0.0
    if settings.baseline is not None:
        base = window.baseline(settings)
        if base.any():
            level = float(np.median(window.y[base]))
    ordered = settings.latency_order == "parameterised"
    memo: dict[Any, Any]
    memo = {}
    structures: dict[tuple[Any, ...], _Structure] = {}

    def make(shapes: Any) -> Any:
        """Return the structure of ``shapes`` (one per sequence of shapes)."""
        # one structure per sequence of shapes and fit (it caches its whitened target)
        key = tuple(shapes)
        struct = structures.get(key)
        if struct is None:
            struct = structures[key] = _Structure(
                list(shapes), background, tw, white, yw, prior, scale, edge, level, ordered, memo
            )
        return struct

    return make


def _background_only(search: _Search) -> Any:
    """K = 0: the background alone, its time constants started from a small grid."""
    base_struct = search.make([])
    lo, hi = base_struct.bounds()
    best = None
    grids: list[Any] = (
        [np.geomspace(max(low, 1e-6), h, 5) for low, h in zip(lo, hi, strict=False)]
        if base_struct.n_theta
        else [[]]
    )
    for theta0 in (
        (np.array(g) for g in np.array(np.meshgrid(*grids)).reshape(len(grids), -1).T)
        if base_struct.n_theta
        else [np.zeros(0)]
    ):
        state: Any
        state, _ = search.evaluate([], theta0)
        if state is not None and (best is None or state.fit_crit < best.fit_crit):
            best = state
    return best


def _forward_search(current: Any, search: _Search, forced: bool, out: Any) -> tuple[Any, Any]:
    """Add components one at a time (spec §5.5.3 step 4); returns the state and best rejection."""
    settings = search.settings
    target = settings.n_components if settings.selection == "fixed" else settings.max_components
    best_rejected = np.inf
    # At most a few rounds per component: merges may undo an addition
    for _round in range(3 * max(target, 1) + 3):
        if len(current.shapes) >= target:
            break
        accepted, rejected = _search_round(current, search, forced)
        best_rejected = min(best_rejected, rejected)
        if accepted is None:
            if settings.selection == "fixed":
                out["warnings"].append("fewer_candidates_than_n_components")
            break
        current = _sort_state(accepted, search.make)
        if not forced:
            # With a forced number of components (fixed, cv) merging would undo the addition
            current = _merge_pass(current, search, out)
        out["steps"].append({"K": len(current.shapes), "criterion": current.crit})
        out["path"].append(current)
    return current, best_rejected


def _search_round(current: Any, search: _Search, forced: bool) -> tuple[Any, Any]:
    """One step of the search: (accepted state or None, best rejected change of the criterion)."""
    settings = search.settings
    cand_settings = dataclasses.replace(settings, detection_threshold=0.0) if forced else settings
    picks, smooth = search.candidates(current, cand_settings)
    if settings.candidate_search == "spec":
        return _spec_step(current, picks, smooth, search, forced)
    ranked, rejected = _quick_candidates(current, picks, smooth, search, forced)
    # Split proposals: one component replaced by two narrower neighbours of the same sign
    # (a single broad component often fits two close peaks first)
    for split in _split_proposals(current, search):
        if forced or split.crit - current.crit <= -search.need(split, current):
            ranked.append((split, None, None))
    accepted, refused = _accept_ranked(current, ranked, search, forced)
    return accepted, min(rejected, refused)


def _with_component(current: Any, search: _Search, params: Any) -> Any:
    """Return the parameters of ``current`` with a component of parameters ``params`` appended."""
    split = search.make(current.shapes).offsets[-1]
    return np.concatenate((current.theta[:split], np.atleast_1d(params), current.theta[split:]))


def _quick_candidates(
    current: Any, picks: Any, smooth: Any, search: _Search, forced: bool
) -> "tuple[list[tuple[Any, ComponentShape | None, Any]], Any]":
    """Rank the candidates by a quick fit: the first ``CANDIDATE_BEAM`` that qualify."""
    settings = search.settings
    rejected = np.inf
    ranked: list[tuple[Any, ComponentShape | None, Any]] = []
    for scanned, peak in enumerate(picks[: 2 * max(settings.max_components, 1)]):
        if not ranked and scanned >= CANDIDATE_SCAN:
            # none of the strongest residual extrema improves the criterion: the
            # weaker ones are not tried
            break
        quick = _quick_fit(current, peak, smooth, search)
        if quick is None:
            continue
        gain = quick[0].crit - current.crit
        if forced or gain <= -search.need(quick[0], current):
            ranked.append(quick)
            # Compare the first CANDIDATE_BEAM qualifying candidates: the strongest
            # residual extremum is not always the right next component (two neighbours
            # of one sign leave a residual whose largest extremum lies between them,
            # with the opposite sign)
            if len(ranked) >= CANDIDATE_BEAM:
                break
        else:
            rejected = min(rejected, gain)
    return ranked, rejected


def _quick_fit(current: Any, peak: Any, smooth: Any, search: _Search) -> Any:
    """Every shape at ``peak`` from its initial guess, loose tolerance: the best, or None."""
    tw = search.tw
    quick = None
    for name in search.settings.shapes:
        shape: Any
        shape = _shape(name, search.settings)
        try:
            guess: Any
            guess = shape.initial_guess(tw, smooth, peak)
        except GUESS_ERRORS as err:  # a shape that cannot start here
            _guess_failed(shape, tw, (peak,), err)
            continue
        theta0 = _with_component(current, search, guess)
        state, _ = search.evaluate(current.shapes + [shape], theta0, tol=1e-4)
        if state is not None and (quick is None or state.fit_crit < quick[0].fit_crit):
            quick = (state, shape, guess)
    return quick


def _accept_ranked(current: Any, ranked: Any, search: _Search, forced: bool) -> tuple[Any, Any]:
    """Refine the ranked candidates in turn; the first that still qualifies is accepted."""
    settings = search.settings
    rejected = np.inf
    for state, shape, guess in sorted(ranked, key=lambda item: item[0].fit_crit):
        if shape is None:  # a split: already optimised
            best = _polish(state, search)
            gain = best.crit - current.crit
            if forced or gain <= -search.need(best, current):
                return best, rejected
            rejected = min(rejected, gain)
            continue
        best = _polish(_refine(current, state, shape, guess, search), search)
        gain = best.crit - current.crit
        # min_amplitude (§5.5.10, 0 = no limit): the only size rule, a user option
        if (
            settings.min_amplitude
            and np.min(np.abs(best.beta[: len(best.shapes)])) < settings.min_amplitude
        ):
            rejected = min(rejected, gain)
            continue
        if forced or gain <= -search.need(best, current):
            return best, rejected
        rejected = min(rejected, gain)
    return None, rejected


def _refine(current: Any, state: Any, shape: Any, guess: Any, search: _Search) -> Any:
    """Refit a qualifying candidate from perturbed starts; the best fit."""
    best = state
    for start in _starts(shape, guess, search.settings.n_starts, search.tw)[1:]:
        theta0 = _with_component(current, search, start)
        cand, _ = search.evaluate(current.shapes + [shape], theta0, tol=1e-5)
        if cand is not None and cand.fit_crit < best.fit_crit:
            best = cand
    return best


def _reseed_step(current: Any, search: _Search, out: Any) -> Any:
    """Refit a suspicious final state from the extrema of the data (see :func:`_reseed`)."""
    reseeded = _reseed(current, out["path"][0], search)
    if reseeded is None:
        return current
    out["warnings"].append("reseeded")
    current = _merge_pass(reseeded, search, out)
    # the greedy path needed its components to build the pair; after the refit,
    # each must still earn its place by the rule of the forward search
    current = _prune(current, search, out)
    out["path"][-1] = current
    out["steps"][-1] = {"K": len(current.shapes), "criterion": current.crit}
    return current


def _finish(out: Any, current: Any, search: _Search, y_window: Any) -> Any:
    """Record the selected state, its model and residual, and the status."""
    struct = search.make(current.shapes)
    out.update(
        struct=struct,
        shapes=current.shapes,
        theta=current.theta,
        beta=current.beta,
        crit=current.crit,
        kind=search.kind,
    )
    model_w = struct.design(current.theta) @ current.beta
    out["model"] = model_w
    out["residual"] = y_window - model_w
    if current.shapes:
        out.update(status="ok", message="")
    else:
        out.update(
            status="no_response",
            message="the background-only model is preferred "
            f"(best rejected change {out['best_rejected_delta']:.2f})",
        )
        logger.info("no response: %s", out["message"])
    return out


# Width parameters that a split halves, by name.
_WIDTH_PARAMS = ("sigma", "omega")


def _spec_starts(shape: Any, guess: Any, n_starts: Any, tw: Any) -> list[NDArray[Any]]:
    """Return the starting values of spec §5.5.3 step 4.

    The candidate itself and ``n_starts - 1`` perturbations, in this order: latency
    ``-/+ sigma_0/2``, width ``x 0.5`` and ``x 2``.
    """
    guess = np.asarray(guess, float)
    names = shape.param_names
    lo, hi = (np.asarray(v, float) for v in shape.default_bounds(tw))
    i = names.index(shape.latency_param) if shape.latency_param in names else None
    j = next((names.index(n) for n in _WIDTH_PARAMS if n in names), None)
    moves = []
    if i is not None and j is not None:
        moves += [(i, guess[i] - guess[j] / 2), (i, guess[i] + guess[j] / 2)]
    if j is not None:
        moves += [(j, guess[j] * 0.5), (j, guess[j] * 2.0)]
    out = [guess]
    for idx, value in moves[: max(n_starts - 1, 0)]:
        g = guess.copy()
        g[idx] = np.clip(value, lo[idx], hi[idx])
        out.append(g)
    return out


def _spec_step(
    current: Any, picks: Any, smooth: Any, search: _Search, forced: Any
) -> tuple[Any, Any]:
    """One step of the forward search of spec §5.5.3 step 4.

    The candidates in order of ``|r|``, each fitted with every allowed shape from ``n_starts``
    starts (all parameters re-optimised), the best kept; the first that improves the criterion by
    ``delta`` is accepted, otherwise the next is tried (at most ``2 * max_components``).

    Returns (accepted state or None, best rejected change of the criterion).
    """
    settings = search.settings
    rejected = np.inf
    for peak in picks[: 2 * max(settings.max_components, 1)]:
        best = _spec_candidate(current, peak, smooth, search)
        if best is None:
            continue
        gain = best.crit - current.crit
        if (
            settings.min_amplitude
            and np.min(np.abs(best.beta[: len(best.shapes)])) < settings.min_amplitude
        ):
            rejected = min(rejected, gain)
            continue
        if forced or gain <= -search.need(best, current):
            return best, rejected
        rejected = min(rejected, gain)
    return None, rejected


def _spec_candidate(current: Any, peak: Any, smooth: Any, search: _Search) -> Any:
    """Fit a component at ``peak``: every shape, every start of spec §5.5.3 step 4; the best."""
    settings, tw = search.settings, search.tw
    base = search.make(current.shapes)
    head, tail = current.theta[: base.offsets[-1]], current.theta[base.offsets[-1] :]
    best = None
    for name in settings.shapes:
        shape = _shape(name, settings)
        try:
            guess = shape.initial_guess(tw, smooth, peak)
        except GUESS_ERRORS as err:  # a shape that cannot start here
            _guess_failed(shape, tw, (peak,), err)
            continue
        for start in _spec_starts(shape, guess, settings.n_starts, tw):
            state, _ = search.evaluate(
                current.shapes + [shape], np.concatenate((head, start, tail))
            )
            if state is not None and (best is None or state.fit_crit < best.fit_crit):
                best = state
    return best


def _split_proposals(state: Any, search: _Search, n_split: int = 2) -> list[Any]:
    """Each of the largest components replaced by two of its shape at +/- one width."""
    out = []
    make = search.make
    struct = make(state.shapes)
    order = np.argsort(-np.abs(state.beta[: len(state.shapes)]))[:n_split]
    for k in order:
        shape = state.shapes[k]
        names = shape.param_names
        width = next((n for n in _WIDTH_PARAMS if n in names), None)
        if width is None or shape.latency_param not in names:
            continue
        params = struct.comp_params(state.theta, k)
        i, j = names.index(shape.latency_param), names.index(width)
        lo, hi = shape.default_bounds(struct.tw)
        pieces = []
        for sign in (-1, 1):
            p = params.copy()
            p[i] = np.clip(p[i] + sign * params[j], lo[i], hi[i])
            p[j] = np.clip(params[j] / np.sqrt(2), lo[j], hi[j])
            pieces.append(p)
        shapes = state.shapes[:k] + [shape, shape] + state.shapes[k + 1 :]
        theta0 = np.concatenate(
            [struct.comp_params(state.theta, m) for m in range(k)]
            + pieces
            + [struct.comp_params(state.theta, m) for m in range(k + 1, len(state.shapes))]
            + [struct.bg_params(state.theta)]
        )
        cand, _ = search.evaluate(shapes, theta0, tol=1e-5)
        if cand is not None:
            out.append(_sort_state(cand, make))
    return out


# Reseeding: the greedy path can end in a local optimum where two components of opposite sign
# overlap and cancel (a "cancelling pair"): adding the second component lets the first slide
# onto it, and the merge of the pair is worse than the pair, so it stays although placing
# the components on the extrema of the data fits far better. Such a state is refitted from
# starts on the extrema of the data (the residual of the background-only model).
RESEED_EXTREMA = 6
RESEED_COND = 1e5


def _suspicious(state: Any, make: Any, settings: Any) -> bool:
    """Tell whether a state is suspicious.

    A pair of neighbouring components of opposite sign closer than ``overlap_c`` times their summed
    widths, or a nearly singular design (one amplitude cancelling another).
    """
    struct = make(state.shapes)
    info = []
    for k in range(len(state.shapes)):
        feats = _component_features(struct, state.theta, k, state.beta[k])
        info.append((feats.latency, (feats.fwhm or 0) / FWHM_PER_SIGMA, np.sign(state.beta[k])))
    for (la, sa, pa), (lb, sb, pb) in zip(info, info[1:], strict=False):
        if pa != pb and abs(lb - la) < settings.overlap_c * (sa + sb):
            return True
    if len(state.shapes) >= _PAIR:
        design = struct.design(state.theta)[:, : len(state.shapes)]
        norms = np.linalg.norm(design, axis=0)
        if np.all(norms > 0):
            singular = np.linalg.svd(design / norms, compute_uv=False)
            if singular[-1] <= 0 or singular[0] / singular[-1] > RESEED_COND**0.5:
                return True
    return False


def _reseed(state: Any, base: Any, search: _Search) -> Any:
    """Return a better state with the same number of components, or None.

    The state is started on the extrema of the data. Tried only when ``state`` is suspicious (see
    :func:`_suspicious`).
    """
    import itertools

    settings, make, tw = search.settings, search.make, search.tw
    if settings.reseed != "always" and not _suspicious(state, make, settings):
        return None
    k = len(state.shapes)
    loose = dataclasses.replace(settings, detection_threshold=0.0)
    picks, smooth = search.candidates(base, loose)
    picks = sorted(picks[: max(RESEED_EXTREMA, k)])
    if len(picks) < k:
        return None
    names = list(dict.fromkeys(s.name for s in state.shapes)) or list(settings.shapes)
    best = None
    bg = make(state.shapes).bg_params(state.theta)
    for chosen in itertools.combinations(picks, k):
        for name in names:
            shape = _shape(name, settings)
            try:
                guesses = [shape.initial_guess(tw, smooth, p) for p in chosen]
            except GUESS_ERRORS as err:  # a shape that cannot start at one of the extrema
                _guess_failed(shape, tw, chosen, err)
                continue
            theta0 = np.concatenate(guesses + [bg])
            cand, _ = search.evaluate([shape] * k, theta0, tol=1e-5)
            if cand is not None and (best is None or cand.fit_crit < best.fit_crit):
                best = cand
    if best is None:
        return None
    best = _polish(_sort_state(best, make), search)
    return best if best.fit_crit < state.fit_crit and best.crit <= state.crit else None


def _sort_state(state: Any, make: Any) -> Any:
    """Components in ascending latency.

    With the ordered parameterisation the order is that of the latency parameters (part of the
    model); otherwise that of the features (sorted after each step, the order carrying no meaning
    in the model).
    """
    struct = make(state.shapes)
    lats: Any
    if struct.ordered:
        lats = _param_latencies(struct, state.theta)
    else:
        lats = [
            _latency_of(s, struct.comp_params(state.theta, k)) for k, s in enumerate(state.shapes)
        ]
    order = np.argsort(lats, kind="stable")
    shapes = [state.shapes[i] for i in order]
    theta = (
        np.concatenate(
            [struct.comp_params(state.theta, i) for i in order] + [struct.bg_params(state.theta)]
        )
        if state.shapes
        else state.theta
    )
    beta = np.concatenate((state.beta[order], state.beta[len(state.shapes) :]))
    return _State(shapes, theta, beta, state.rw, state.crit, state.fit_crit, state.shape_crit)


def _prune(state: Any, search: _Search, out: Any) -> Any:
    """Drop the components of a reseeded state that no longer lower the criterion by ``delta``.

    This is the acceptance rule of the forward search, §5.5.3 step 4: the state without each
    component is refitted from the current values, and the best removal is applied while its
    criterion is less than ``delta`` above the state's.
    """
    while len(state.shapes) >= _PAIR:
        struct = search.make(state.shapes)
        best = None
        for k in range(len(state.shapes)):
            shapes = [s for i, s in enumerate(state.shapes) if i != k]
            theta0 = np.concatenate(
                [struct.comp_params(state.theta, i) for i in range(len(state.shapes)) if i != k]
                + [struct.bg_params(state.theta)]
            )
            cand, _ = search.evaluate(shapes, theta0)
            if cand is not None and (best is None or cand.crit < best[0].crit):
                best = (cand, k)
        if best is None:
            break
        needed = (
            search.fixed_delta
            if search.cal is None
            else search.cal.required(struct.white(struct.design(state.theta)), best[1])
        )
        if not best[0].crit - state.crit < needed:
            break
        out["warnings"].append(f"pruned:{best[1]}")
        state = _sort_state(best[0], search.make)
    return state


def _merge_pass(state: Any, search: _Search, out: Any) -> Any:
    """Replace two overlapping neighbours by one when the merged model is better."""
    changed = True
    while changed and len(state.shapes) >= _PAIR:
        merged = _merge_once(state, search)
        changed = merged is not None
        if merged is not None:
            out["warnings"].append(f"merged:{merged[1]},{merged[1] + 1}")
            state = _sort_state(merged[0], search.make)
    return state


def _merge_once(state: Any, search: _Search) -> Any:
    """Find the first merge of two overlapping neighbours that does not raise the criterion.

    Returns ``(merged state, index of the first of the pair)`` or None.
    """
    struct = search.make(state.shapes)
    info = []
    for k in range(len(state.shapes)):
        feats = _component_features(struct, state.theta, k, state.beta[k])
        info.append((feats.latency, (feats.fwhm or 0) / FWHM_PER_SIGMA, abs(state.beta[k])))
    for k in range(len(state.shapes) - 1):
        (la, sa, aa), (lb, sb, ab) = info[k], info[k + 1]
        if not abs(lb - la) < search.settings.overlap_c * (sa + sb):
            continue
        keep = k if aa >= ab else k + 1
        shapes = [s for i, s in enumerate(state.shapes) if i not in (k, k + 1)]
        shapes.insert(k, state.shapes[keep])
        params = struct.comp_params(state.theta, keep).copy()
        shape = state.shapes[keep]
        if shape.latency_param in shape.param_names:
            idx = shape.param_names.index(shape.latency_param)
            params[idx] += (la * aa + lb * ab) / (aa + ab) - _latency_of(shape, params)
        theta0 = np.concatenate(
            [struct.comp_params(state.theta, i) for i in range(len(state.shapes)) if i < k]
            + [params]
            + [struct.comp_params(state.theta, i) for i in range(len(state.shapes)) if i > k + 1]
            + [struct.bg_params(state.theta)]
        )
        merged, _ = search.evaluate(shapes, theta0)
        if merged is not None and merged.crit <= state.crit:
            return merged, k
    return None


###################################################################################################
## Calibrated acceptance, final shape choice and resolution check (CHANGELOG "Calibrated
## acceptance"; models/_calibration.py)

# parameters of a Gaussian component (location, width, amplitude)
P_SCAN_GAUSS = 3
# values of ``shape_choice`` that refit every final component with every allowed shape
FINAL_SHAPE_CHOICES = ("hqc", "aic")
# starting asymmetries of a skewed shape in the final refinement (besides its own start)
SKEW_STARTS = (2.0, -2.0)
# shapes with more free parameters than this are not searched over their corners (2^p fits)
MAX_CORNER_PARAMS = 4
# a component with another component (of either sign) within this many FWHM is not checked
NEIGHBOUR_FWHMS = 2.0
# the narrowest component the noise band carries, in samples at the rate it occupies
MIN_BAND_SAMPLES = 2.0


def _calibrated(settings: Any, forced: bool) -> bool:
    """Whether the acceptance of this fit is calibrated (``delta="calibrated"``)."""
    return (
        isinstance(settings.delta, str)
        and settings.noise_model == "ar"
        and settings.selection == "bic"
        and not forced
    )


def _fixed_delta(settings: Any) -> float:
    """Return the fixed fall of the criterion: ``delta``, or FIXED_DELTA when calibrated."""
    return FIXED_DELTA if isinstance(settings.delta, str) else float(settings.delta)


def _calibrate(  # noqa: PLR0913, PLR0917  (the noise model of the fit, from the core)
    search: _Search,
    current: Any,
    white: Any,
    ar: _ArNoise,
    band: _Band,
    forced: bool,
    out: Any,
) -> None:
    """Set the acceptance of ``search`` (calibrated or fixed) and record it in ``out``."""
    settings = search.settings
    search.fixed_delta = _fixed_delta(settings)
    if not (settings.delta == CALIBRATED and _calibrated(settings, forced)):
        out["acceptance"] = {"rule": "fixed", "delta": search.fixed_delta}
        return
    bg_design = search.make([]).design(current.theta)
    search.cal = Calibration(
        settings.alpha,
        white,
        search.tw,
        white(bg_design) if bg_design.shape[1] else None,
        _reachable_fwhm(settings, search.tw, search.sfreq, band.stride),
        settings.max_fwhm,
        _n_crit(search, white),
        ar.a,
        ar.innovation,
        ar.acov,
        0 if ar.z is None else len(ar.z),
        baseline=ar.z,
    )
    out["acceptance"] = {
        "rule": "calibrated",
        "alpha": float(settings.alpha),
        "scan_level": float(search.cal.c0),
        "delta": float(search.cal.delta0),
    }


def _n_crit(search: _Search, white: Any) -> float:
    """Return the samples the criterion counts (the innovation rows of the exact likelihood)."""
    return float(search.n_eff - (white.p if getattr(white, "start", None) is not None else 0))


def _final_checks(current: Any, search: _Search, white: Any, out: Any) -> Any:
    """Apply the final shape choice (``shape_choice`` "hqc" or "aic") and the resolution check."""
    settings = search.settings
    if len(settings.shapes) > 1 and settings.shape_choice in FINAL_SHAPE_CHOICES:
        current = _refine_shapes(current, search)
        out["path"][-1] = current
        out["steps"][-1] = {"K": len(current.shapes), "criterion": current.crit}
    if settings.resolution_check:
        out["warnings"] += _resolution_warnings(
            current, search, _n_crit(search, white), out["warnings"]
        )
    return current


def _refine_shapes(state: Any, search: _Search) -> Any:
    """Refit each component of the final state with every allowed shape; keep the best.

    Every component is refitted, whatever its width or amplitude. The shapes are compared by
    ``_shape_key`` (the criterion of ``shape_choice``), from starts on the component's own
    curve (its extremum and width), with the asymmetry of a skewed shape started at 0 and
    at +/- 2: the quick, single-start fits of the forward search often leave a skewed shape
    at its symmetric start. The default criterion, HQC (``2 ln ln n`` per parameter, about
    3.7 here), lets an asymmetry that the data do not determine leave the shape of the
    search unchanged: with AIC (2 per parameter) on every component, a narrow component at
    low SNR took a spurious asymmetry and moved its mode by 1 to 2 samples (REQ-105 recall
    0.8863 to 0.8852, Gaussian truths of REQ-106 0.250 to 0.285 sample on development
    seeds; with HQC 0.8865 and 0.233; CHANGELOG "Final shape choice on every component").
    """
    for k in range(len(state.shapes)):
        struct = search.make(state.shapes)
        best = state
        for start, shape in _shape_starts(state, k, search):
            shapes = state.shapes[:k] + [shape] + state.shapes[k + 1 :]
            theta0 = np.concatenate(
                [
                    start if i == k else struct.comp_params(state.theta, i)
                    for i in range(len(state.shapes))
                ]
                + [struct.bg_params(state.theta)]
            )
            cand, _ = search.evaluate(shapes, theta0)
            if cand is not None and _shape_key(cand, search.settings) < _shape_key(
                best, search.settings
            ):
                best = cand
        state = _sort_state(best, search.make)
    return state


def _shape_starts(state: Any, k: int, search: _Search) -> list[tuple[NDArray[Any], Any]]:
    """Return the starts ``(parameters, shape)`` of component ``k`` with every allowed shape."""
    tw, settings = search.tw, search.settings
    struct = search.make(state.shapes)
    own = state.shapes[k]
    params = struct.comp_params(state.theta, k)
    curve = state.beta[k] * own.evaluate(tw, params)
    peak = int(np.argmax(np.abs(curve)))
    out = []
    for name in settings.shapes:
        shape = _shape(name, settings)
        try:
            guess = np.asarray(shape.initial_guess(tw, curve, peak), float)
        except GUESS_ERRORS as err:
            _guess_failed(shape, tw, (peak,), err)
            continue
        first = guess if shape.name != own.name else np.asarray(params, float)
        out.append((first, shape))
        if "alpha" in shape.param_names:
            j = shape.param_names.index("alpha")
            lo, hi = shape.default_bounds(tw)
            for value in SKEW_STARTS:
                g = first.copy()
                g[j] = np.clip(value, lo[j], hi[j])
                out.append((g, shape))
    return out


def _reachable_fwhm(settings: Any, tw: Any, sfreq: float, stride: int) -> float:
    """Narrowest FWHM (s) any allowed shape can take within its bounds, not below 2 samples.

    The bounds of a shape constrain its width parameters, not its FWHM: a skewed Gaussian
    at the largest asymmetry is about half as wide as a Gaussian with the same scale, so
    with ``min_fwhm`` well above the sampling limit the search reaches narrower components
    than ``min_fwhm`` (on white noise with ``min_fwhm`` 8 ms at 1 kHz, 17 of 59 false
    components were skewed Gaussians 4 to 8 ms wide). The scan level of the calibrated
    acceptance covers the widths the search can reach: the FWHM of each shape at the
    corners of its bounds (latency at the window centre), and at least two samples at the
    rate the noise occupies (the engine's own floor for ``min_fwhm``).
    """
    narrowest = float(settings.min_fwhm)
    for name in settings.shapes:
        narrowest = min(narrowest, _corner_fwhm(_shape(name, settings), tw, sfreq))
    return max(narrowest, MIN_BAND_SAMPLES * stride / sfreq)


def _corner_fwhm(shape: Any, tw: Any, sfreq: float) -> float:
    """Narrowest FWHM of ``shape`` at the corners of its bounds (inf when not searched)."""
    import itertools

    try:
        lo, hi = (np.asarray(v, float) for v in shape.default_bounds(tw))
    except GUESS_ERRORS:
        return np.inf
    names = shape.param_names
    free = [i for i, n in enumerate(names) if n != shape.latency_param]
    if len(free) > MAX_CORNER_PARAMS:
        return np.inf
    narrowest = np.inf
    for corner in itertools.product(*[(lo[i], hi[i]) for i in free]):
        params = 0.5 * (lo + hi)
        if shape.latency_param in names:
            params[names.index(shape.latency_param)] = 0.5 * (tw[0] + tw[-1])
        params[free] = corner
        try:
            fwhm = compute_features(shape, tw, params, 1.0, sfreq=sfreq).fwhm
        except GUESS_ERRORS:
            continue
        if fwhm is not None and np.isfinite(fwhm) and fwhm > 0:
            narrowest = min(narrowest, float(fwhm))
    return float(narrowest)


@functools.lru_cache(maxsize=8)
def _pair_probe(ratio: float) -> tuple[float, float]:
    """FWHM and peak of the best single Gaussian fitted to two equal unit Gaussians.

    The two have FWHM 1 and lie ``ratio`` FWHM apart. Returns (FWHM of the single,
    amplitude of the single), noise-free least squares on a fine grid.
    """
    t = np.linspace(-6.0, 6.0, 4001)
    sig = 1.0 / FWHM_PER_SIGMA
    pair = np.exp(-0.5 * ((t - ratio / 2) / sig) ** 2) + np.exp(
        -0.5 * ((t + ratio / 2) / sig) ** 2
    )

    def misfit(q: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(q[0] * np.exp(-0.5 * (t / q[1]) ** 2) - pair, float)

    res = least_squares(
        misfit,
        [float(pair.max()), sig * (1 + ratio / 2)],
        bounds=([0.0, 1e-3], [10.0, 10.0]),
    )
    return float(res.x[1] * FWHM_PER_SIGMA), float(res.x[0])


def _resolution_warnings(state: Any, search: _Search, n_crit: float, existing: Any) -> list[str]:
    """``merged:<k>,<k+1>`` for each component that a pair of overlapping ones would explain.

    Spec §5.5.3 step 5 merges two neighbours closer than ``overlap_c`` times their summed
    widths when the merged model is no worse, and REQ-111 asks that two equal Gaussians
    0.5 FWHM apart come out as one component carrying ``merged``. Such a pair differs from
    one wider Gaussian by a chi-square of 1.4 at SNR 10 (CHANGELOG): the forward search
    never accepts its second member, so the merge of step 5 never runs. This check makes
    the comparison from the other side: for each component, the pair of equal Gaussians
    ``overlap_c`` times their summed widths apart (step 5's overlap) whose single-Gaussian
    fit is the component (same latency, FWHM and amplitude) is built, its best
    single-Gaussian fit is found in this fit's whitened metric (the other components and
    the background projected out), and the deviance between the two, ``lambda``, is
    compared with the deviance a new component needs (the calibrated scan level, or
    ``3 ln n + delta``). When ``lambda`` is smaller, the data could not have separated
    such a pair: the component is reported as their merger, ``merged:k,k+1``. The flag
    says what the data cannot exclude (it is set on most single components below the SNR
    at which such a pair would be resolved, CHANGELOG "Calibrated acceptance"), not that two
    components were seen. A component with another component within twice its FWHM, of
    either sign, is not flagged: the indices ``k, k+1`` would name that neighbour, and the
    data near it already hold two components. The rule does not depend on the signs of the
    components (REQ-003).
    """
    struct = search.make(state.shapes)
    sfreq = 1.0 / float(np.median(np.diff(search.tw)))
    info = []
    for k, shape in enumerate(state.shapes):
        params = struct.comp_params(state.theta, k)
        feats = compute_features(shape, search.tw, params, state.beta[k], sfreq=sfreq)
        info.append((float(feats.latency), float(feats.fwhm or 0.0), float(state.beta[k])))
    design_w = struct.white(struct.design(state.theta))
    out = []
    for k, (lat, fwhm, _) in enumerate(info):
        if not (np.isfinite(lat) and fwhm > 0) or any(
            w.startswith(f"merged:{k},") for w in existing
        ):
            continue
        if any(
            j != k and abs(la - lat) < NEIGHBOUR_FWHMS * fwhm for j, (la, _, _) in enumerate(info)
        ):
            continue
        lam, direction = _pair_deviance(struct, np.delete(design_w, k, axis=1), info[k], search)
        if search.cal is not None:
            needed = search.cal.deviance_needed(direction)
        else:
            needed = P_SCAN_GAUSS * float(np.log(max(n_crit, 5.0))) + search.fixed_delta
        if lam < needed:
            out.append(f"merged:{k},{k + 1}")
    return out


def _pair_deviance(
    struct: Any, others: Any, component: tuple[float, float, float], search: _Search
) -> tuple[float, NDArray[Any]]:
    """Deviance between the pair behind ``component`` and its best single Gaussian.

    ``component`` is (latency, FWHM, amplitude); ``others`` the whitened columns projected
    out. Returns the deviance and the whitened residual direction.
    """
    lat, fwhm, amp = component
    tw = search.tw
    q_o = np.linalg.qr(others)[0] if others.shape[1] else None

    def project(x: Any) -> Any:
        return x - q_o @ (q_o.T @ x) if q_o is not None else x

    ratio = search.settings.overlap_c * 2.0 / FWHM_PER_SIGMA
    span_f, span_a = _pair_probe(ratio)
    sig_each = fwhm / span_f / FWHM_PER_SIGMA
    d = ratio * fwhm / span_f
    pair = (amp / span_a) * (
        np.exp(-0.5 * ((tw - lat + d / 2) / sig_each) ** 2)
        + np.exp(-0.5 * ((tw - lat - d / 2) / sig_each) ** 2)
    )
    pair_w = project(struct.white(pair))
    sig0 = fwhm / FWHM_PER_SIGMA

    def resid(q: Any) -> Any:
        return pair_w - project(struct.white(q[0] * np.exp(-0.5 * ((tw - q[1]) / q[2]) ** 2)))

    fit = least_squares(
        resid, [amp, lat, sig0], x_scale=[max(abs(amp), 1e-12), sig0, sig0], max_nfev=200
    )
    return float(np.sum(fit.fun**2)), np.asarray(fit.fun, float)


def _shape_kind(settings: Any) -> str:
    """Return the criterion the final shape choice compares on ("hqc" or "aic")."""
    return "aic" if getattr(settings, "shape_choice", None) == "aic" else "hqc"


def _shape_key(state: Any, settings: Any) -> float:
    """Return the value the shapes of one candidate are compared on (``shape_choice``)."""
    if settings.shape_choice in FINAL_SHAPE_CHOICES and state.shape_crit is not None:
        return float(state.shape_crit)
    return float(state.fit_crit)


def _added_index(state: Any, ref: Any, make: Any) -> int:
    """Index of the component of ``state`` that ``ref`` (one component fewer) lacks."""
    if not ref.shapes:
        return 0
    s_struct, r_struct = make(state.shapes), make(ref.shapes)
    lat = [
        _latency_of(sh, s_struct.comp_params(state.theta, i)) for i, sh in enumerate(state.shapes)
    ]
    ref_lat = np.sort(
        [_latency_of(sh, r_struct.comp_params(ref.theta, i)) for i, sh in enumerate(ref.shapes)]
    )
    costs = []
    for j in range(len(state.shapes)):
        rest = np.sort([v for i, v in enumerate(lat) if i != j])
        costs.append(
            float(np.sum(np.abs(rest - ref_lat))) if len(rest) == len(ref_lat) else np.inf
        )
    return int(np.argmin(costs))


def _need(cal: Any, state: Any, ref: Any, make: Any) -> float:
    """Calibrated fall of the criterion needed to accept ``state`` over ``ref``."""
    struct = make(state.shapes)
    k = _added_index(state, ref, make) if len(state.shapes) == len(ref.shapes) + 1 else None
    design_w = struct.white(struct.design(state.theta))
    if k is None:
        # not one added component (a split: one replaced by two): the stricter of the parts
        return float(max(cal.required(design_w, j) for j in range(len(state.shapes))))
    return float(cal.required(design_w, k))
