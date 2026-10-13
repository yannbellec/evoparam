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
from evoparam.models._trf import least_squares_trf
from evoparam.shapes import (
    FWHM_PER_SIGMA,
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

# What a shape's initial_guess raises when it cannot start at a candidate extremum (bounds
# that do not fit the window, an extremum at the edge, a non-finite width): that shape is
# not tried there, with a DEBUG record (REQ-307: never without a trace). Any other
# exception is a programming error and propagates (to the channel's status "failed").
GUESS_ERRORS = (SettingsError, DataError, ValueError, ArithmeticError, IndexError)


def _guess_failed(shape: Any, tw: Any, peaks: Any, err: BaseException) -> None:
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
    with np.errstate(invalid="ignore"):
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
    if yb.size < 3:
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
    """BIC, AIC or AICc on whitened residuals with the effective sample count."""
    n_eff = max(float(n_eff), p + 2.0)
    base = n_eff * np.log(max(rss_w, 1e-300) / n_eff)
    if kind == "bic":
        return base + p * np.log(n_eff)
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
            omap = _OrderedMap(chain, lower, upper) if len(chain) >= 2 else None
            known = (lower, upper, lower + 1e-9 * span, upper - 1e-9 * span, omap)
            if omap is not None:
                width = omap.upper - omap.lower
                known += (omap.lower + 1e-9 * width, omap.upper - 1e-9 * width)
            if self.memo is not None:
                self._own["start_limits"] = known
        return known

    def _bounds(self) -> Any:
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
            lo, hi = self.background.default_bounds(self.tw)
            lows.append(np.asarray(lo)[self.n_lin_bg :])
            highs.append(np.asarray(hi)[self.n_lin_bg :])
        if not lows:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(lows).astype(float), np.concatenate(highs).astype(float)

    def _shape_bounds(self, shape: Any) -> tuple[NDArray[Any], NDArray[Any]]:
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
        return theta[self.offsets[k] : self.offsets[k + 1]]

    def bg_params(self, theta: Any) -> Any:
        return theta[self.offsets[-1] :]

    def design(self, theta: Any) -> Any:
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
            or not np.sqrt((upper**2).sum() * (inverse**2).sum()) < 1e11
        ):
            return self._full(theta)
        qtb, _, _ = _dormqr("L", "T", qr, tau, b[:, None], 64)
        qtb = qtb[:, 0]
        beta, _ = _dtrtrs(qr[:c, :c], qtb[:c])
        return qtb[c:], ("qr", qr, tau, beta, a, aux)

    def _full(self, theta: Any) -> tuple[Any, tuple[str, NDArray[Any], NDArray[Any], Any]]:
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
        self.idx = np.asarray(chain, int)
        lat_lo, lat_hi = lower[self.idx], upper[self.idx]
        span = float(np.max(lat_hi) - np.min(lat_lo))
        self.gap = 1e-9 * span if span > 0 else 1e-12
        self.lower, self.upper = lower.copy(), upper.copy()
        self.lower[self.idx[1:]] = np.log(self.gap)
        self.upper[self.idx[1:]] = np.log(max(span, 2 * self.gap))

    def to_phi(self, theta: Any) -> NDArray[Any]:
        phi = np.array(theta, float)
        lat = phi[self.idx]
        phi[self.idx[1:]] = np.log(np.maximum(np.diff(lat), self.gap))
        return phi

    def to_theta(self, phi: Any) -> NDArray[Any]:
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


def _optimise(struct: Any, theta0: Any, settings: Any, f_scale: Any, tol: float = 1e-8) -> Any:
    """Bounded least squares on the projected residual; returns (theta, beta, rw, ok, msg)."""
    if struct.n_theta == 0:
        beta, rw, _, _ = struct.solve(theta0)
        return theta0, beta, rw, True, ""
    limits = struct.start_limits()
    lower, upper, omap = limits[0], limits[1], limits[4]
    # np.clip, without its overhead
    theta0 = np.minimum(np.maximum(theta0, limits[2]), limits[3])
    cache: dict[Any, Any]
    cache = {}
    lean = settings.loss == "linear" and _OPTIMISER != "scipy"

    def solved(theta: Any) -> Any:
        key = theta.tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = struct.reduced(theta) if lean else struct.solve(theta)
        return cache[key]

    def jac_theta(theta: Any) -> Any:
        if lean:
            return struct.reduced_jacobian(theta, solved(theta)[1])
        beta, _, _, q = solved(theta)
        return struct.kaufman_jacobian(theta, beta, q)

    # the lean optimiser works on the residual in the basis of the QR factorisation
    # (Structure.reduced); a robust loss acts on each residual and needs the residuals
    fun: Callable[[Any], Any]
    jac: Callable[[Any], Any]
    if omap is None:
        x0, lo, hi = theta0, lower, upper

        def fun(theta: Any) -> Any:
            return solved(theta)[0 if lean else 1]

        jac = jac_theta
    else:
        lo, hi = omap.lower, omap.upper
        x0 = np.minimum(np.maximum(omap.to_phi(theta0), limits[5]), limits[6])

        def fun(phi: Any) -> Any:
            return solved(omap.to_theta(phi))[0 if lean else 1]

        def jac(phi: Any) -> Any:
            return omap.jacobian(jac_theta(omap.to_theta(phi)), phi)

    if lean and _OPTIMISER == "trf":
        done = _optimise_compiled(struct, x0, lo, hi, omap, settings, tol)
        if done is not None:
            return (theta0 if done[0] is None else done[0], *done[1:])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if lean:
                # the same algorithm as least_squares(method="trf"), lighter (models/_trf.py)
                xr, _, nfev, _ = least_squares_trf(
                    fun,
                    jac,
                    x0,
                    lo,
                    hi,
                    ftol=tol,
                    xtol=tol,
                    gtol=tol,
                    max_nfev=settings.max_iter,
                    reduced=_OPTIMISER != "trf-full",
                    m=len(struct._target()[0]),
                    final_jacobian=False,
                )
            else:
                res = least_squares(
                    fun,
                    x0,
                    jac=jac,
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
    except (ValueError, np.linalg.LinAlgError) as exc:
        return theta0, None, None, False, f"{type(exc).__name__}: {exc}"
    x = xr if omap is None else omap.to_theta(xr)
    if lean:
        beta, rw = struct.reduced_solution(x, solved(x)[1])
    else:
        beta, rw, _, _ = struct.solve(x)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(beta))):
        return x, None, None, False, "non-finite parameters"
    return x, beta, rw, True, ""


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
    return (
        tw,
        np.asarray(kinds, dtype=np.int64),
        np.asarray(struct.offsets, dtype=np.int64),
        bg_kind,
        lag,
        fixed,
        inv_scale,
        a,
        post,
        stride,
        start,
        start_rows,
        prior_cols,
        np.ascontiguousarray(diag[prior_cols]),
        target,
    )


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
    if _FAST == "off":
        return None
    kernel = _fast.kernel()
    if kernel is None:
        return None
    spec = _compiled_args(struct)
    if spec is None:
        return None
    chain = np.asarray(omap.idx if omap is not None else [], dtype=np.int64)
    status, theta, nfev, beta, rw = kernel(
        np.ascontiguousarray(x0, dtype=float),
        np.ascontiguousarray(lo, dtype=float),
        np.ascontiguousarray(hi, dtype=float),
        chain,
        *spec,
        float(tol),
        int(settings.max_iter),
    )
    if status == _fast.FALLBACK:
        return None  # the SVD route of the NumPy path
    _COUNTER.n = getattr(_COUNTER, "n", 0) + int(nfev)
    if status == _fast.VALUE_ERROR:
        return None, None, None, False, "ValueError: infeasible start or step"
    if status == _fast.LINALG_ERROR:
        return None, None, None, False, "LinAlgError: SVD did not converge"
    if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(beta))):
        return theta, None, None, False, "non-finite parameters"
    return theta, beta, rw, True, ""


###################################################################################################
## Candidates


def _smooth(values: Any, sfreq: float, sigma_s: Any) -> Any:
    width = sigma_s * sfreq
    if width < 0.5:
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
    picks = [p for p in dict.fromkeys(picks) if valid[p]]
    if settings.polarity == "negative":
        picks = [p for p in picks if rs[p] < 0]
    elif settings.polarity == "positive":
        picks = [p for p in picks if rs[p] > 0]
    picks.sort(key=lambda p: -mag[p])
    above = [p for p in picks if mag[p] >= settings.detection_threshold]
    if settings.threshold_role == "order":
        # the threshold orders the candidates (those above it first) but does not remove
        # any: acceptance is decided by the criterion alone (REQ-104)
        return above + [p for p in picks if mag[p] < settings.detection_threshold], rs * sigma
    return above, rs * sigma


###################################################################################################
## One fit state


@dataclasses.dataclass
class _State:
    shapes: list[Any]
    theta: NDArray[Any]
    beta: NDArray[Any]
    rw: NDArray[Any]
    crit: float  # selection criterion: innovation rows only (decides K)
    fit_crit: float | None = None  # the same on every row: compares optima of one structure


def _shape(name: Any, settings: Any) -> "ComponentShape":
    fields = {f.name for f in dataclasses.fields(cast(Any, type(get_shape(name))))}
    kwargs = {
        k: v
        for k, v in (
            ("min_fwhm", settings.min_fwhm),
            ("max_fwhm", settings.max_fwhm),
            ("freq_min", settings.freq_min),
            ("freq_max", settings.freq_max),
        )
        if k in fields and v is not None
    }
    return get_shape(name, **kwargs)


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
    if k_max < 2:
        return False
    amps = beta[:k_max]
    if np.all(amps >= 0) or np.all(amps <= 0):
        return False
    sfreq = 1.0 / np.median(np.diff(struct.tw))
    lat, sig = [], []
    for k, shape in enumerate(struct.shapes):
        feats = compute_features(
            shape, struct.tw, struct.comp_params(theta, k), amps[k], sfreq=sfreq
        )
        lat.append(feats.latency)
        sig.append(feats.fwhm / FWHM_PER_SIGMA if np.isfinite(feats.fwhm) else 0.0)
    return bool(cancelling_pairs(lat, sig, amps, settings.overlap_c))


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
    # n_eff is the number of whitened (independent) samples used by the criterion
    struct = struct_maker(shapes)
    if struct.ordered and len(shapes) >= 2:
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
    return _State(list(shapes), theta, beta, rw, crit, fit_crit), ""


def _polish(state: Any, make: Any, settings: Any, n_eff: Any, f_scale: Any, kind: Any) -> Any:
    """Re-optimise a state from several starting values of the background's time constants.

    A background fitted before a component existed may hold that component (a slow exponential that
    mimics a broad N2), a local optimum the optimiser would keep.
    """
    struct = make(state.shapes)
    if not struct.n_bg:
        return state
    lo, hi = struct.bounds()
    lo, hi = lo[struct.offsets[-1] :], hi[struct.offsets[-1] :]
    best = state
    for value in np.geomspace(np.maximum(lo, 1e-6), hi, 4):
        theta0 = np.r_[state.theta[: struct.offsets[-1]], value]
        cand, _ = _evaluate_state(make, state.shapes, theta0, settings, n_eff, f_scale, kind)
        if cand is not None and cand.fit_crit < best.fit_crit:
            best = cand
    return best


def _latency_of(shape: Any, params: Any) -> float:
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

    times = np.asarray(times, float)
    y = np.asarray(y, float)
    mask = np.zeros(len(y), bool) if mask is None else np.asarray(mask, bool).copy()
    mask |= ~np.isfinite(y)
    win = settings.fit_window
    if win is None:
        raise SettingsError("fit_window is required (no preset gives one).")
    out: dict[str, Any] = {"warnings": [], "steps": [], "best_rejected_delta": np.nan}
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
    if win[1] - win[0] < 3 * settings.min_fwhm:
        raise SettingsError(f"fit_window {win} is shorter than 3 x min_fwhm.")
    wmask = (times >= win[0]) & (times <= win[1])
    out["window_mask"] = wmask
    tw = times[wmask]
    yw = np.where(mask[wmask], 0.0, y[wmask])
    valid = ~mask[wmask]

    out["noise_reference"] = "plusminus" if noise_ref is not None else "baseline"
    if noise_ref is not None:
        # the plus-minus reference: the noise of the average over the whole epoch (trials of
        # alternating sign averaged), longer than any baseline
        ref = np.asarray(noise_ref, float)
        good = np.isfinite(ref) & ~mask
        sigma = 1.4826 * float(np.median(np.abs(ref[good] - np.median(ref[good]))))
        rho = float(np.corrcoef(ref[good][1:], ref[good][:-1])[0, 1])
        warn = None
    else:
        sigma, rho, warn = noise_model(times, y, settings.baseline, wmask, mask)
    if warn:
        out["warnings"].append(warn)
    out.update(sigma=sigma, rho=rho)
    if valid.mean() < 1 - settings.max_masked_fraction or valid.sum() < 5:
        out.update(status="skipped", message="masked: too few valid samples in the window")
        return out
    if not sigma > 0:
        out.update(status="skipped", message="flat")
        return out

    scale: Any = np.full(len(tw), sigma)
    if sem is not None:
        s_ = np.asarray(sem, float)[wmask]
        scale = np.maximum(np.nan_to_num(s_, nan=sigma), 0.1 * sigma)
    a_coefs, innovation, acov = np.array([1.0]), 1.0, None
    # the noise of a band-limited recording is modelled at the rate it occupies: on every
    # stride-th sample, on the phase of the epoch that holds the noise best (noise_phase:
    # an upsampled recording keeps its own samples there)
    grid = np.arange(len(y)) % stride == phase
    if settings.noise_model == "ar" and noise_ref is not None:
        a_coefs, innovation, acov = ar_model(ref[good & grid] / sigma, return_acov=True)
    elif settings.noise_model == "ar" and settings.baseline is not None:
        base = (times >= settings.baseline[0]) & (times <= settings.baseline[1]) & ~mask
        if base.sum() >= MIN_BASELINE:
            a_coefs, innovation, acov = ar_model(y[base & grid] / sigma, return_acov=True)
    if sem is not None:
        innovation = 1.0 if len(a_coefs) == 1 else innovation
    white: Whitener
    if stride > 1:
        white = StridedWhitener(
            scale, a_coefs, innovation, valid, stride, phase=int(np.argmax(grid[wmask])), acov=acov
        )
    else:
        white = Whitener(scale, a_coefs, innovation, valid, acov=acov)
    n_valid = int(valid.sum())
    n_eff = n_valid * (1 - rho) / (1 + rho)
    n_white: float = white.n
    if settings.noise_model == "neff":
        # spec §5.5.3 step 1 to the letter: residuals scaled by sigma_n, not whitened, and
        # the AR(1) effective count n_eff = n_W (1 - rho) / (1 + rho) in every criterion
        # and in the covariance
        n_white = n_eff
    out.update(n_eff=n_eff, n_white=n_white, ar_order=len(a_coefs) - 1)
    kind = "aicc" if settings.selection == "aicc" else "bic"
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
    background = get_background(settings.background, **bg_kwargs)
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
        base = (times >= settings.baseline[0]) & (times <= settings.baseline[1]) & ~mask
        if base.any():
            level = float(np.median(y[base]))
    ordered = settings.latency_order == "parameterised"
    memo: dict[Any, Any]
    memo = {}
    structures: dict[tuple[Any, ...], _Structure] = {}

    def make(shapes: Any) -> Any:
        # one structure per sequence of shapes and fit (it caches its whitened target)
        key = tuple(shapes)
        struct = structures.get(key)
        if struct is None:
            struct = structures[key] = _Structure(
                list(shapes), background, tw, white, yw, prior, scale, edge, level, ordered, memo
            )
        return struct

    f_scale = settings.f_scale

    # K = 0: background only (its time constants from a small grid)
    base_struct = make([])
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
        state, _ = _evaluate_state(make, [], theta0, settings, n_white, f_scale, kind)
        if state is not None and (best is None or state.fit_crit < best.fit_crit):
            best = state
    if best is None:
        out.update(status="failed", message="the background-only model could not be fitted")
        logger.warning("fit failed: %s", out["message"])
        return out
    current = best
    out["steps"].append({"K": 0, "criterion": current.crit})
    out["path"] = [current]

    def residual_of(state: Any) -> Any:
        return yw - make(state.shapes).design(state.theta) @ state.beta

    target = settings.n_components if settings.selection == "fixed" else settings.max_components
    forced = settings.selection in ("fixed", "cv")
    best_rejected = np.inf
    # At most a few rounds per component: merges may undo an addition
    for _round in range(3 * max(target, 1) + 3):
        if len(current.shapes) >= target:
            break
        cand_settings = (
            dataclasses.replace(settings, detection_threshold=0.0) if forced else settings
        )
        picks, smooth = candidates(tw, residual_of(current), valid, sigma, sfreq, cand_settings)
        accepted = None
        ranked: list[tuple[Any, ComponentShape | None, Any]] = []
        if settings.candidate_search == "spec":
            accepted, rejected = _spec_step(
                current, picks, smooth, make, settings, n_white, f_scale, kind, tw, forced
            )
            best_rejected = min(best_rejected, rejected)
            picks = []  # the beam search below has nothing to do
        for peak in picks[: 2 * max(settings.max_components, 1)]:
            # Quick evaluation: every shape from its initial guess, loose tolerance
            quick = None
            for name in settings.shapes:
                shape: Any
                shape = _shape(name, settings)
                try:
                    guess: Any
                    guess = shape.initial_guess(tw, smooth, peak)
                except GUESS_ERRORS as err:  # a shape that cannot start here
                    _guess_failed(shape, tw, (peak,), err)
                    continue
                theta0 = np.r_[
                    current.theta[: make(current.shapes).offsets[-1]],
                    guess,
                    current.theta[make(current.shapes).offsets[-1] :],
                ]
                state, _ = _evaluate_state(
                    make,
                    current.shapes + [shape],
                    theta0,
                    settings,
                    n_white,
                    f_scale,
                    kind,
                    tol=1e-4,
                )
                if state is not None and (quick is None or state.fit_crit < quick[0].fit_crit):
                    quick = (state, shape, guess)
            if quick is None:
                continue
            gain = quick[0].crit - current.crit
            if forced or gain <= -settings.delta:
                ranked.append(quick)
                # Compare the first CANDIDATE_BEAM qualifying candidates: the strongest
                # residual extremum is not always the right next component (two neighbours
                # of one sign leave a residual whose largest extremum lies between them,
                # with the opposite sign)
                if len(ranked) >= CANDIDATE_BEAM:
                    break
            else:
                best_rejected = min(best_rejected, gain)
        # Split proposals: one component replaced by two narrower neighbours of the same sign
        # (a single broad component often fits two close peaks first)
        splits = (
            []
            if settings.candidate_search == "spec"
            else _split_proposals(current, make, settings, n_white, f_scale, kind)
        )
        for split in splits:
            if forced or split.crit - current.crit <= -settings.delta:
                ranked.append((split, None, None))
        for state, shape, guess in sorted(ranked, key=lambda item: item[0].fit_crit):
            if shape is None:  # a split: already optimised
                best = _polish(state, make, settings, n_white, f_scale, kind)
                gain = best.crit - current.crit
                if forced or gain <= -settings.delta:
                    accepted = best
                    break
                best_rejected = min(best_rejected, gain)
                continue
            # Refine the best qualifying candidate: perturbed starts and background starts
            best = state
            for start in _starts(shape, guess, settings.n_starts, tw)[1:]:
                theta0 = np.r_[
                    current.theta[: make(current.shapes).offsets[-1]],
                    start,
                    current.theta[make(current.shapes).offsets[-1] :],
                ]
                cand, _ = _evaluate_state(
                    make,
                    current.shapes + [shape],
                    theta0,
                    settings,
                    n_white,
                    f_scale,
                    kind,
                    tol=1e-5,
                )
                if cand is not None and cand.fit_crit < best.fit_crit:
                    best = cand
            best = _polish(best, make, settings, n_white, f_scale, kind)
            gain = best.crit - current.crit
            # min_amplitude (§5.5.10, 0 = no limit): the only size rule, a user option
            if (
                settings.min_amplitude
                and np.min(np.abs(best.beta[: len(best.shapes)])) < settings.min_amplitude
            ):
                best_rejected = min(best_rejected, gain)
                continue
            if forced or gain <= -settings.delta:
                accepted = best
                break
            best_rejected = min(best_rejected, gain)
        if accepted is None:
            if settings.selection == "fixed":
                out["warnings"].append("fewer_candidates_than_n_components")
            break
        current = _sort_state(accepted, make)
        if not forced:
            # With a forced number of components (fixed, cv) merging would undo the addition
            current = _merge_pass(current, make, settings, n_white, f_scale, kind, out)
        out["steps"].append({"K": len(current.shapes), "criterion": current.crit})
        out["path"].append(current)

    if current.shapes and not forced and settings.reseed:
        reseeded = _reseed(
            current,
            out["path"][0],
            make,
            settings,
            n_white,
            f_scale,
            kind,
            tw,
            valid,
            sigma,
            sfreq,
            residual_of,
        )
        if reseeded is not None:
            out["warnings"].append("reseeded")
            current = _merge_pass(reseeded, make, settings, n_white, f_scale, kind, out)
            # the greedy path needed its components to build the pair; after the refit,
            # each must still earn its place by the rule of the forward search
            current = _prune(current, make, settings, n_white, f_scale, kind, out)
            out["path"][-1] = current
            out["steps"][-1] = {"K": len(current.shapes), "criterion": current.crit}

    out["best_rejected_delta"] = float(best_rejected) if np.isfinite(best_rejected) else np.nan
    out["make"] = make
    if settings.selection == "cv" and select_k is not None:
        current = out["path"][min(select_k(out["path"], make), len(out["path"]) - 1)]
    struct = make(current.shapes)
    out.update(
        struct=struct,
        shapes=current.shapes,
        theta=current.theta,
        beta=current.beta,
        crit=current.crit,
        kind=kind,
    )
    model_w = struct.design(current.theta) @ current.beta
    out["model"] = model_w
    out["residual"] = y[wmask] - model_w
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
    current: Any,
    picks: Any,
    smooth: Any,
    make: Any,
    settings: Any,
    n_eff: Any,
    f_scale: Any,
    kind: Any,
    tw: Any,
    forced: Any,
) -> Any:
    """One step of the forward search of spec §5.5.3 step 4.

    The candidates in order of ``|r|``, each fitted with every allowed shape from ``n_starts``
    starts (all parameters re-optimised), the best kept; the first that improves the criterion by
    ``delta`` is accepted, otherwise the next is tried (at most ``2 * max_components``).

    Returns (accepted state or None, best rejected change of the criterion).
    """
    rejected = np.inf
    base = make(current.shapes)
    head, tail = current.theta[: base.offsets[-1]], current.theta[base.offsets[-1] :]
    for peak in picks[: 2 * max(settings.max_components, 1)]:
        best = None
        for name in settings.shapes:
            shape = _shape(name, settings)
            try:
                guess = shape.initial_guess(tw, smooth, peak)
            except GUESS_ERRORS as err:  # a shape that cannot start here
                _guess_failed(shape, tw, (peak,), err)
                continue
            for start in _spec_starts(shape, guess, settings.n_starts, tw):
                state, _ = _evaluate_state(
                    make,
                    current.shapes + [shape],
                    np.r_[head, start, tail],
                    settings,
                    n_eff,
                    f_scale,
                    kind,
                )
                if state is not None and (best is None or state.fit_crit < best.fit_crit):
                    best = state
        if best is None:
            continue
        gain = best.crit - current.crit
        if (
            settings.min_amplitude
            and np.min(np.abs(best.beta[: len(best.shapes)])) < settings.min_amplitude
        ):
            rejected = min(rejected, gain)
            continue
        if forced or gain <= -settings.delta:
            return best, rejected
        rejected = min(rejected, gain)
    return None, rejected


def _split_proposals(
    state: Any, make: Any, settings: Any, n_eff: Any, f_scale: Any, kind: Any, n_split: int = 2
) -> list[Any]:
    """Each of the largest components replaced by two of its shape at +/- one width."""
    out = []
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
        cand, _ = _evaluate_state(make, shapes, theta0, settings, n_eff, f_scale, kind, tol=1e-5)
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
    for k, shape in enumerate(state.shapes):
        params = struct.comp_params(state.theta, k)
        feats = compute_features(
            shape, struct.tw, params, state.beta[k], sfreq=1 / np.median(np.diff(struct.tw))
        )
        info.append((feats.latency, (feats.fwhm or 0) / FWHM_PER_SIGMA, np.sign(state.beta[k])))
    for (la, sa, pa), (lb, sb, pb) in zip(info, info[1:], strict=False):
        if pa != pb and abs(lb - la) < settings.overlap_c * (sa + sb):
            return True
    if len(state.shapes) >= 2:
        design = struct.design(state.theta)[:, : len(state.shapes)]
        norms = np.linalg.norm(design, axis=0)
        if np.all(norms > 0):
            singular = np.linalg.svd(design / norms, compute_uv=False)
            if singular[-1] <= 0 or singular[0] / singular[-1] > RESEED_COND**0.5:
                return True
    return False


def _reseed(
    state: Any,
    base: Any,
    make: Any,
    settings: Any,
    n_eff: Any,
    f_scale: Any,
    kind: Any,
    tw: Any,
    valid: Any,
    sigma: Any,
    sfreq: float,
    residual_of: Any,
) -> Any:
    """Return a better state with the same number of components, or None.

    The state is started on the extrema of the data. Tried only when ``state`` is suspicious (see
    :func:`_suspicious`).
    """
    import itertools

    if settings.reseed != "always" and not _suspicious(state, make, settings):
        return None
    k = len(state.shapes)
    loose = dataclasses.replace(settings, detection_threshold=0.0)
    picks, smooth = candidates(tw, residual_of(base), valid, sigma, sfreq, loose)
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
            cand, _ = _evaluate_state(
                make, [shape] * k, theta0, settings, n_eff, f_scale, kind, tol=1e-5
            )
            if cand is not None and (best is None or cand.fit_crit < best.fit_crit):
                best = cand
    if best is None:
        return None
    best = _polish(_sort_state(best, make), make, settings, n_eff, f_scale, kind)
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
    beta = np.r_[state.beta[order], state.beta[len(state.shapes) :]]
    return _State(shapes, theta, beta, state.rw, state.crit, state.fit_crit)


def _prune(
    state: Any, make: Any, settings: Any, n_eff: Any, f_scale: Any, kind: Any, out: Any
) -> Any:
    """Drop the components of a reseeded state that no longer lower the criterion by ``delta``.

    This is the acceptance rule of the forward search, §5.5.3 step 4: the state without each
    component is refitted from the current values, and the best removal is applied while its
    criterion is less than ``delta`` above the state's.
    """
    while len(state.shapes) >= 2:
        struct = make(state.shapes)
        best = None
        for k in range(len(state.shapes)):
            shapes = [s for i, s in enumerate(state.shapes) if i != k]
            theta0 = np.concatenate(
                [struct.comp_params(state.theta, i) for i in range(len(state.shapes)) if i != k]
                + [struct.bg_params(state.theta)]
            )
            cand, _ = _evaluate_state(make, shapes, theta0, settings, n_eff, f_scale, kind)
            if cand is not None and (best is None or cand.crit < best[0].crit):
                best = (cand, k)
        if best is None or not best[0].crit - state.crit < settings.delta:
            break
        out["warnings"].append(f"pruned:{best[1]}")
        state = _sort_state(best[0], make)
    return state


def _merge_pass(
    state: Any, make: Any, settings: Any, n_eff: Any, f_scale: Any, kind: Any, out: Any
) -> Any:
    """Replace two overlapping neighbours by one when the merged model is better."""
    changed = True
    while changed and len(state.shapes) >= 2:
        changed = False
        struct = make(state.shapes)
        info = []
        for k, shape in enumerate(state.shapes):
            params = struct.comp_params(state.theta, k)
            feats = compute_features(
                shape, struct.tw, params, state.beta[k], sfreq=1 / np.median(np.diff(struct.tw))
            )
            info.append((feats.latency, (feats.fwhm or 0) / FWHM_PER_SIGMA, abs(state.beta[k])))
        for k in range(len(state.shapes) - 1):
            (la, sa, aa), (lb, sb, ab) = info[k], info[k + 1]
            if not abs(lb - la) < settings.overlap_c * (sa + sb):
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
                + [
                    struct.comp_params(state.theta, i)
                    for i in range(len(state.shapes))
                    if i > k + 1
                ]
                + [struct.bg_params(state.theta)]
            )
            merged, _ = _evaluate_state(make, shapes, theta0, settings, n_eff, f_scale, kind)
            if merged is not None and merged.crit <= state.crit:
                out["warnings"].append(f"merged:{k},{k + 1}")
                state = _sort_state(merged, make)
                changed = True
                break
    return state
