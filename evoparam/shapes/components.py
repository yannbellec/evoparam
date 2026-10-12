# SPDX-License-Identifier: Apache-2.0
# Modified from the upstream project and commit named in NOTICE (Apache-2.0); changes: Gaussian and
#   skewed-Gaussian equations of core/funcs.py rewritten with unit amplitude, vectorised over
#   parameter sets, with analytic Jacobians; the other shapes are new
"""Component shapes with analytic Jacobians (spec §5.5.1).

Every shape is a frozen dataclass whose fields are settings (width limits,
frequency limits); the parameters themselves are passed to the methods as an
array of shape ``(..., n_params)``. Shapes have unit amplitude
(``max_t |s(t)| = 1`` on the real line); the signed amplitude ``A`` is a
separate linear parameter. Parameters are in SI units (s, Hz, rad).

The Gaussian and skewed-Gaussian equations come from the inherited shape functions
(spec §4.5); here they are rewritten with a fixed unit amplitude,
vectorised over parameter sets and given analytic Jacobians.
"""

import math
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import erfc, erfcx, lambertw

from evoparam.errors import DataError, SettingsError
from evoparam.shapes._base import (
    FWHM_PER_SIGMA,
    ShapeBase,
    as_times,
    check_bounds,
    check_waveform,
    clip_to_bounds,
    peak_fwhm,
    window_info,
)

__all__ = [
    "Gaussian",
    "SkewedGaussian",
    "ExGaussian",
    "Gamma",
    "Biexponential",
    "DampedOscillation",
    "Gabor",
    "Template",
    "gamma_fwhm_factor",
]

_SQRT2 = np.sqrt(2.0)
_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
_TWO_PI = 2.0 * np.pi
_BISECTIONS = 200


def _stack(columns):
    return np.stack(np.broadcast_arrays(*columns), axis=-1)


def _bisect(past_root, low, high):
    """Vectorised bisection: ``past_root(x)`` is True to the right of the root.

    Stops when every interval is down to a few ulps of ``max(|x|, 1)``.
    """
    low, high = np.broadcast_arrays(
        np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    )
    for _ in range(_BISECTIONS):
        scale = np.maximum(np.maximum(np.abs(low), np.abs(high)), 1.0)
        if np.all(high - low <= 4.0 * np.finfo(np.float64).eps * scale):
            break
        mid = 0.5 * (low + high)
        right = past_root(mid)
        low, high = np.where(right, low, mid), np.where(right, mid, high)
    return 0.5 * (low + high)


# exp(-z^2 / 2) is exactly 0 in double precision for |z| > 38.61 (below the smallest
# subnormal number): beyond this bound the Gaussian factors are not computed
SUPPORT_Z = 38.7


def gaussian_factor(z, value):
    """Compute ``value(z)``, a product with ``exp(-z^2/2)`` as a factor, where it is not 0.

    The factor may also be a narrower Gaussian. The product is computed only where
    ``|z| < SUPPORT_Z``: elsewhere it is exactly 0, which is returned without evaluating the
    exponential and the other factors (the same array).
    """
    outside = np.abs(z) >= SUPPORT_Z
    n_out = np.count_nonzero(outside)
    if 4 * n_out < z.size:
        return value(z)
    out = np.zeros(z.shape)
    if n_out < z.size:
        inside = ~outside
        out[inside] = value(z[inside])
    return out


###################################################################################################
## Shapes with width settings


@dataclass(frozen=True)
class _WidthShape(ShapeBase):
    """Shape whose bounds depend on the admissible widths ``[min_fwhm, max_fwhm]`` (s).

    When a limit is None it is set from the window: ``min_fwhm = 2/sfreq`` and
    ``max_fwhm = |W|/2``.
    """

    min_fwhm: float | None = None
    max_fwhm: float | None = None

    def __post_init__(self):
        for name in ("min_fwhm", "max_fwhm"):
            value = getattr(self, name)
            if value is not None and not (np.isfinite(value) and value > 0):
                raise SettingsError(f"{self.name}: {name} must be a positive number, got {value}.")
        if (
            self.min_fwhm is not None
            and self.max_fwhm is not None
            and self.min_fwhm >= self.max_fwhm
        ):
            raise SettingsError(
                f"{self.name}: min_fwhm ({self.min_fwhm}) must be smaller "
                f"than max_fwhm ({self.max_fwhm})."
            )

    def _window(self, times):
        """Window start, end, length, sfreq, min_fwhm and max_fwhm."""
        start, end, length, sfreq = window_info(times)
        low = 2.0 / sfreq if self.min_fwhm is None else float(self.min_fwhm)
        high = length / 2.0 if self.max_fwhm is None else float(self.max_fwhm)
        if not low < high:
            raise SettingsError(
                f"{self.name}: min_fwhm ({low}) must be smaller than "
                f"max_fwhm ({high}) for this window."
            )
        return start, end, length, sfreq, low, high

    def _center_width(self, times, waveform, peak_index):
        """Candidate centre and Gaussian-equivalent sigma (spec §5.5.3, step 3)."""
        times, waveform = check_waveform(times, waveform, peak_index)
        _, _, _, _, low, high = self._window(times)
        fwhm = peak_fwhm(times, waveform, int(peak_index))
        fwhm = low if fwhm is None else float(np.clip(fwhm, low, high))
        return float(times[int(peak_index)]), fwhm / FWHM_PER_SIGMA


@dataclass(frozen=True)
class _OscillationShape(_WidthShape):
    """Width shape with a frequency range ``[freq_min, freq_max]`` (Hz).

    Defaults: two cycles per window to ``sfreq/4``.
    """

    freq_min: float | None = None
    freq_max: float | None = None

    def __post_init__(self):
        super().__post_init__()
        for name in ("freq_min", "freq_max"):
            value = getattr(self, name)
            if value is not None and not (np.isfinite(value) and value > 0):
                raise SettingsError(f"{self.name}: {name} must be a positive number, got {value}.")
        if (
            self.freq_min is not None
            and self.freq_max is not None
            and self.freq_min >= self.freq_max
        ):
            raise SettingsError(
                f"{self.name}: freq_min ({self.freq_min}) must be smaller "
                f"than freq_max ({self.freq_max})."
            )

    def _freq_guess(self, times, waveform, low, high):
        """Return the frequency of the largest periodogram peak of the waveform in [low, high].

        Zero-padded, Hann-tapered periodogram; the middle of the range when there is no peak.
        """
        times, waveform = np.asarray(times, float), np.asarray(waveform, float)
        if len(times) < 4 or not np.all(np.isfinite(waveform)):
            return 0.5 * (low + high)
        dt = float(np.median(np.diff(times)))
        x = (waveform - waveform.mean()) * np.hanning(len(waveform))
        n_fft = max(4096, 8 * len(x))
        power = np.abs(np.fft.rfft(x, n_fft)) ** 2
        freqs = np.fft.rfftfreq(n_fft, dt)
        inside = (freqs >= low) & (freqs <= high)
        if not inside.any() or not np.any(power[inside] > 0):
            return 0.5 * (low + high)
        return float(freqs[inside][np.argmax(power[inside])])

    def _freqs(self, length, sfreq):
        low = 2.0 / length if self.freq_min is None else float(self.freq_min)
        high = sfreq / 4.0 if self.freq_max is None else float(self.freq_max)
        if not low < high:
            raise SettingsError(
                f"{self.name}: freq_min ({low}) must be smaller than "
                f"freq_max ({high}) for this window."
            )
        return low, high


###################################################################################################
## Gaussian


@dataclass(frozen=True)
class Gaussian(_WidthShape):
    """Gaussian ``s(t) = exp(-(t - mu)² / (2 sigma²))``.

    Parameters ``(mu, sigma)`` in s. Latency ``mu``; FWHM ``2.3548 sigma``.
    Bounds: ``mu`` in ``[t_0 - sigma_max/2, t_1 + sigma_max/2]``,
    ``sigma`` in ``[min_fwhm, max_fwhm] / 2.3548``.
    """

    name: ClassVar[str] = "gaussian"
    param_names: ClassVar[tuple] = ("mu", "sigma")
    latency_param: ClassVar[str] = "mu"

    def evaluate(self, times, params):
        """Unit-amplitude shape, ``(..., n_params) -> (..., n_times)``."""
        params = self._params(params)
        if params.ndim == 1:
            z = (as_times(times) - params[0]) / params[1]
            return gaussian_factor(z, lambda x: np.exp(-0.5 * x**2))
        mu, sigma = self._columns(params)
        return np.exp(-0.5 * ((as_times(times) - mu) / sigma) ** 2)

    def jacobian(self, times, params):
        """Analytic derivatives, ``(..., n_params) -> (..., n_times, n_params)``."""
        mu, sigma = self._columns(params)
        z = (as_times(times) - mu) / sigma
        s = np.exp(-0.5 * z**2)
        return _stack([s * z / sigma, s * z**2 / sigma])

    def latency(self, params):
        """Return the latency (s) of the component, ``(..., n_params) -> (...)``."""
        return self._params(params)[..., 0]

    extremum = latency

    def default_bounds(self, times, data_range=None):  # noqa: ARG002  (shape protocol: signature shared by every shape)
        """Lower and upper bounds of the parameters for a window."""
        start, end, _, _, low, high = self._window(times)
        sigma_max = high / FWHM_PER_SIGMA
        return check_bounds(
            self.name,
            [start - sigma_max / 2, low / FWHM_PER_SIGMA],
            [end + sigma_max / 2, sigma_max],
        )

    def initial_guess(self, times, waveform, peak_index):
        """Return the starting parameters for a candidate extremum of ``waveform``."""
        mu, sigma = self._center_width(times, waveform, peak_index)
        return clip_to_bounds([mu, sigma], self.default_bounds(times))


###################################################################################################
## Skewed Gaussian


def _skew_root(a):
    """Root of ``f(z) = z erfc(-a z/√2) - √(2/π) a exp(-a² z²/2)`` on ``[0, 2]``.

    Safeguarded Newton (``f`` is increasing on the bracket): each step is kept inside the
    current bracket, falling back to bisection otherwise; converges to a few ulps in about
    six iterations instead of the ~50 of plain bisection.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.size == 1:
        # one skewness (the usual case inside a fit): the same iteration on Python floats,
        # about ten times faster than on one-element arrays
        return np.full(a.shape, _skew_root_scalar(float(a.reshape(-1)[0])))
    low, high = np.zeros_like(a), np.full_like(a, 2.0)
    z = np.clip(0.6 * a / np.sqrt(1.0 + a**2), 0.0, 2.0)
    for _ in range(60):
        g = np.exp(-0.5 * (a * z) ** 2)
        f = z * erfc(-a * z / _SQRT2) - _SQRT_2_OVER_PI * a * g
        low = np.where(f <= 0, z, low)
        high = np.where(f >= 0, z, high)
        slope = erfc(-a * z / _SQRT2) + _SQRT_2_OVER_PI * a * z * g * (1.0 + a**2)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(slope > 0, f / slope, np.inf)
        new = z - step
        outside = ~((new >= low) & (new <= high))
        new = np.where(outside, 0.5 * (low + high), new)
        done = np.abs(new - z) <= 4.0 * np.finfo(np.float64).eps * np.maximum(np.abs(z), 1.0)
        z = new
        if np.all(done | (f == 0)):
            break
    return z


def _skew_root_scalar(a):
    """:func:`_skew_root` for one value of ``a``."""
    if not math.isfinite(a):
        return float(_skew_root(np.array([a]))[0])
    low, high = 0.0, 2.0
    z = min(max(0.6 * a / math.sqrt(1.0 + a * a), 0.0), 2.0)
    eps = 4.0 * np.finfo(np.float64).eps
    for _ in range(60):
        g = math.exp(-0.5 * (a * z) ** 2)
        e = math.erfc(-a * z / _SQRT2)
        f = z * e - _SQRT_2_OVER_PI * a * g
        if f <= 0:
            low = z
        if f >= 0:
            high = z
        slope = e + _SQRT_2_OVER_PI * a * z * g * (1.0 + a * a)
        new = z - f / slope if slope > 0 else -math.inf
        if not low <= new <= high:
            new = 0.5 * (low + high)
        done = abs(new - z) <= eps * max(abs(z), 1.0)
        z = new
        if done or f == 0:
            break
    return z


_MODE_CACHE = {}
_MODE_CACHE_SIZE = 256


def skew_mode(alpha):
    """Mode ``z*`` of ``g(z)(1 + erf(alpha z/√2))``, its value ``c`` and ``dc/dalpha``.

    The mode solves ``z erfc(-alpha z/√2) = √(2/π) alpha exp(-alpha² z²/2)``
    (unique, the numerator being log-concave); it is found by bisection on
    ``[0, 2]`` for ``|alpha|`` and mirrored for negative ``alpha``. By the
    envelope theorem ``dc/dalpha = √(2/π) z* exp(-z*²(1 + alpha²)/2)``.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.size == 1:
        # one skewness (the usual case inside a fit): the same operations on scalars, which
        # give the same values as on one-element arrays without their overhead
        shape = alpha.shape
        value = alpha.reshape(-1)[0]
        a = abs(value)
        key = ((1,), np.array([a]).tobytes())
        z = _MODE_CACHE.get(key)
        if z is None:
            z = _skew_root(np.array([a]))
            if len(_MODE_CACHE) >= _MODE_CACHE_SIZE:
                _MODE_CACHE.clear()
            _MODE_CACHE[key] = z
        z = z[0]
        c = np.exp(-0.5 * (z * z)) * erfc(-a * z / _SQRT2)
        z_signed = np.sign(value) * z
        dc = _SQRT_2_OVER_PI * z_signed * np.exp(-0.5 * (z * z) * (1.0 + value * value))
        return (np.full(shape, z_signed), np.full(shape, c), np.full(shape, dc))
    a = np.abs(alpha)
    # The same alpha is evaluated many times during one optimisation (model, Jacobian,
    # features): small arrays are cached by value
    key = (a.shape, a.tobytes()) if a.size <= 8 else None
    z = _MODE_CACHE.get(key) if key is not None else None
    if z is None:
        z = _skew_root(a)
        if key is not None:
            if len(_MODE_CACHE) >= _MODE_CACHE_SIZE:
                _MODE_CACHE.clear()
            _MODE_CACHE[key] = z
    c = np.exp(-0.5 * z**2) * erfc(-a * z / _SQRT2)
    z_signed = np.sign(alpha) * z
    dc = _SQRT_2_OVER_PI * z_signed * np.exp(-0.5 * z**2 * (1.0 + alpha**2))
    return z_signed, c, dc


@dataclass(frozen=True)
class SkewedGaussian(_WidthShape):
    """Skewed Gaussian ``s(t) = g(z)(1 + erf(alpha z/√2)) / c(alpha)``, ``z = (t - xi)/omega``.

    ``g(z) = exp(-z²/2)`` and ``c(alpha)`` is the maximum of the numerator, so
    that ``max s = 1``. Parameters ``(xi, omega, alpha)``: location (s), scale
    (s), skewness. Latency: the mode ``xi + omega z*(alpha)``. Bounds: ``xi`` and
    ``omega`` as the Gaussian ``mu`` and ``sigma``; ``alpha`` in
    ``[-alpha_max, alpha_max]`` (default 10).
    """

    alpha_max: float = 10.0
    name: ClassVar[str] = "skewed_gaussian"
    param_names: ClassVar[tuple] = ("xi", "omega", "alpha")
    latency_param: ClassVar[str] = "xi"

    def __post_init__(self):
        """Check and normalise the fields."""
        super().__post_init__()
        if not (np.isfinite(self.alpha_max) and self.alpha_max > 0):
            raise SettingsError(f"{self.name}: alpha_max must be positive, got {self.alpha_max}.")

    def _parts(self, times, params):
        xi, omega, alpha = self._columns(params)
        _, c, dc = skew_mode(alpha)
        z = (as_times(times) - xi) / omega
        if z.ndim == 1:
            numerator = gaussian_factor(
                z, lambda x: np.exp(-0.5 * x**2) * erfc(-alpha * x / _SQRT2)
            )
        else:
            numerator = np.exp(-0.5 * z**2) * erfc(-alpha * z / _SQRT2)
        return z, omega, alpha, c, dc, numerator

    def evaluate(self, times, params):
        """Unit-amplitude shape, ``(..., n_params) -> (..., n_times)``."""
        _, _, _, c, _, numerator = self._parts(times, params)
        return numerator / c

    def jacobian(self, times, params):
        """Analytic derivatives, ``(..., n_params) -> (..., n_times, n_params)``."""
        z, omega, alpha, c, dc, numerator = self._parts(times, params)
        gauss = _SQRT_2_OVER_PI * np.exp(-0.5 * z**2 * (1.0 + alpha**2))
        d_z = -z * numerator + alpha * gauss
        d_alpha = z * gauss
        return _stack(
            [-d_z / (omega * c), -d_z * z / (omega * c), d_alpha / c - numerator * dc / c**2]
        )

    def latency(self, params):
        """Return the latency (s) of the component, ``(..., n_params) -> (...)``."""
        params = self._params(params)
        z_mode, _, _ = skew_mode(params[..., 2])
        return params[..., 0] + params[..., 1] * z_mode

    extremum = latency

    def default_bounds(self, times, data_range=None):  # noqa: ARG002  (shape protocol: signature shared by every shape)
        """Lower and upper bounds of the parameters for a window."""
        start, end, _, _, low, high = self._window(times)
        sigma_max = high / FWHM_PER_SIGMA
        return check_bounds(
            self.name,
            [start - sigma_max / 2, low / FWHM_PER_SIGMA, -self.alpha_max],
            [end + sigma_max / 2, sigma_max, self.alpha_max],
        )

    def initial_guess(self, times, waveform, peak_index):
        """Return the starting parameters for a candidate extremum of ``waveform``."""
        mu, sigma = self._center_width(times, waveform, peak_index)
        return clip_to_bounds([mu, sigma, 0.0], self.default_bounds(times))


###################################################################################################
## Ex-Gaussian


def exgauss_core(z, lam):
    """``f(z, lam) = exp(lam²/2 - lam z) erfc((lam - z)/√2)``, without overflow.

    ``f`` is the ex-Gaussian density up to a constant, with ``z = (t - mu)/sigma``
    and ``lam = sigma/tau``. For ``x = (lam - z)/√2 >= 0`` it is computed as
    ``exp(-z²/2) erfcx(x)``.
    """
    x = (lam - z) / _SQRT2
    right = x >= 0
    scaled = np.exp(-0.5 * z**2) * erfcx(np.where(right, x, 0.0))
    direct = np.exp(np.minimum(lam * (0.5 * lam - z), 0.0)) * erfc(np.where(right, 0.0, x))
    return np.where(right, scaled, direct)


def exgauss_mode(lam):
    """Mode ``z*`` of ``f(., lam)`` and its value ``Q = √(2/π) exp(-z*²/2) / lam``.

    At the mode ``erfcx(x*) = √(2/π)/lam`` with ``x* = (lam - z*)/√2``; ``erfcx``
    is decreasing, so ``x*`` is found by bisection. ``dQ/dlam = -z* Q``.
    """
    lam = np.asarray(lam, dtype=np.float64)
    target = _SQRT_2_OVER_PI / lam
    # erfcx(x) < 1/(x √π) for x > 0 and erfcx(x) > exp(x²) for x < 0 bracket the root.
    high = 1.0 / (target * np.sqrt(np.pi))
    low = -np.sqrt(np.maximum(np.log(target), 0.0)) - 0.5
    x_mode = _bisect(lambda mid: erfcx(mid) <= target, low, high)
    z_mode = lam - _SQRT2 * x_mode
    return z_mode, _SQRT_2_OVER_PI * np.exp(-0.5 * z_mode**2) / lam


@dataclass(frozen=True)
class ExGaussian(_WidthShape):
    """Ex-Gaussian ``s(t) = h(t) / max h`` (exponentially modified Gaussian).

    ``h(t) = (1/2tau) exp(sigma²/(2 tau²) - (t - mu)/tau)
    erfc((sigma/tau - (t - mu)/sigma)/√2)``.
    Parameters ``(mu, sigma, tau)`` in s. Latency: the mode ``mu + sigma z*``.
    Bounds: ``mu`` and ``sigma`` as the Gaussian; ``tau`` in ``[1/sfreq, max_fwhm]``.
    """

    name: ClassVar[str] = "exgaussian"
    param_names: ClassVar[tuple] = ("mu", "sigma", "tau")
    latency_param: ClassVar[str] = "mu"

    def _parts(self, times, params):
        mu, sigma, tau = self._columns(params)
        lam = sigma / tau
        z = (as_times(times) - mu) / sigma
        z_mode, q = exgauss_mode(lam)
        return z, sigma, tau, lam, z_mode, q, exgauss_core(z, lam)

    def evaluate(self, times, params):
        """Unit-amplitude shape, ``(..., n_params) -> (..., n_times)``."""
        *_, q, f = self._parts(times, params)
        return f / q

    def jacobian(self, times, params):
        """Analytic derivatives, ``(..., n_params) -> (..., n_times, n_params)``."""
        z, sigma, tau, lam, z_mode, q, f = self._parts(times, params)
        gauss = _SQRT_2_OVER_PI * np.exp(-0.5 * z**2)
        s = f / q
        d_z = (gauss - lam * f) / q
        d_lam = ((lam - z) * f - gauss) / q + s * z_mode
        return _stack([-d_z / sigma, -d_z * z / sigma + d_lam / tau, -d_lam * lam / tau])

    def latency(self, params):
        """Return the latency (s) of the component, ``(..., n_params) -> (...)``."""
        params = self._params(params)
        z_mode, _ = exgauss_mode(params[..., 1] / params[..., 2])
        return params[..., 0] + params[..., 1] * z_mode

    extremum = latency

    def default_bounds(self, times, data_range=None):  # noqa: ARG002  (shape protocol: signature shared by every shape)
        """Lower and upper bounds of the parameters for a window."""
        start, end, _, sfreq, low, high = self._window(times)
        sigma_max = high / FWHM_PER_SIGMA
        return check_bounds(
            self.name,
            [start - sigma_max / 2, low / FWHM_PER_SIGMA, 1.0 / sfreq],
            [end + sigma_max / 2, sigma_max, high],
        )

    def initial_guess(self, times, waveform, peak_index):
        """Return the starting parameters for a candidate extremum of ``waveform``."""
        peak, sigma = self._center_width(times, waveform, peak_index)
        tau = sigma
        z_mode, _ = exgauss_mode(np.float64(1.0))
        return clip_to_bounds(
            [peak - sigma * float(z_mode), sigma, tau], self.default_bounds(times)
        )


###################################################################################################
## Gamma


def gamma_fwhm_factor(kappa):
    """FWHM of the gamma shape in units of ``theta``, from the Lambert W function.

    With ``a = kappa - 1`` the half-maximum points ``a theta y`` solve
    ``y exp(-y) = exp(-1) 2^(-1/a)``, i.e. ``y = -W(-exp(-1) 2^(-1/a))`` on the
    two real branches.
    """
    a = np.asarray(kappa, dtype=np.float64) - 1.0
    arg = -np.exp(-1.0) * 2.0 ** (-1.0 / a)
    return a * (lambertw(arg, 0).real - lambertw(arg, -1).real)


@dataclass(frozen=True)
class Gamma(_WidthShape):
    """Gamma shape, causal: zero before ``t0``.

    ``s(t) = ((t - t0)/((kappa - 1) theta))^(kappa - 1) exp(kappa - 1 - (t - t0)/theta)``
    for ``t >= t0``; peak 1 at ``t0 + (kappa - 1) theta``. Parameters
    ``(t0, kappa, theta)``: onset (s), shape, scale (s); ``kappa = 2`` is the
    alpha function. Bounds: ``t0`` in ``[t_0 - max_fwhm, t_1]``, ``kappa`` in
    ``[kappa_min, kappa_max]`` (default 1.5 to 20) and ``theta`` in the range
    that lets the FWHM reach ``[min_fwhm, max_fwhm]`` for every ``kappa``.
    """

    kappa_min: float = 1.5
    kappa_max: float = 20.0
    name: ClassVar[str] = "gamma"
    param_names: ClassVar[tuple] = ("t0", "kappa", "theta")
    latency_param: ClassVar[str] = "t0"
    onset_param: ClassVar[str] = "t0"

    def __post_init__(self):
        """Check and normalise the fields."""
        super().__post_init__()
        if not 1.0 < self.kappa_min < self.kappa_max:
            raise SettingsError(
                f"{self.name}: need 1 < kappa_min < kappa_max, got "
                f"{self.kappa_min}, {self.kappa_max}."
            )

    def _parts(self, times, params):
        t0, kappa, theta = self._columns(params)
        a = kappa - 1.0
        u = as_times(times) - t0
        after = u > 0
        u = np.where(after, u, 1.0)
        log_ratio = np.log(u / (a * theta))
        s = np.where(after, np.exp(a * log_ratio + a - u / theta), 0.0)
        return u, a, theta, log_ratio, s

    def evaluate(self, times, params):
        """Unit-amplitude shape, ``(..., n_params) -> (..., n_times)``."""
        return self._parts(times, params)[-1]

    def jacobian(self, times, params):
        """Analytic derivatives, ``(..., n_params) -> (..., n_times, n_params)``."""
        u, a, theta, log_ratio, s = self._parts(times, params)
        return _stack([s * (1.0 / theta - a / u), s * log_ratio, s * (u / theta - a) / theta])

    def latency(self, params):
        """Return the latency (s) of the component, ``(..., n_params) -> (...)``."""
        params = self._params(params)
        return params[..., 0] + (params[..., 1] - 1.0) * params[..., 2]

    extremum = latency

    def default_bounds(self, times, data_range=None):  # noqa: ARG002  (shape protocol: signature shared by every shape)
        """Lower and upper bounds of the parameters for a window."""
        start, end, _, _, low, high = self._window(times)
        theta_low = low / gamma_fwhm_factor(self.kappa_max)
        theta_high = high / gamma_fwhm_factor(self.kappa_min)
        return check_bounds(
            self.name, [start - high, self.kappa_min, theta_low], [end, self.kappa_max, theta_high]
        )

    def initial_guess(self, times, waveform, peak_index):
        """Return the starting parameters for a candidate extremum of ``waveform``."""
        peak, sigma = self._center_width(times, waveform, peak_index)
        kappa = 2.0
        theta = sigma * FWHM_PER_SIGMA / float(gamma_fwhm_factor(kappa))
        return clip_to_bounds(
            [peak - (kappa - 1.0) * theta, kappa, theta], self.default_bounds(times)
        )


###################################################################################################
## Biexponential


def biexp_peak(tau_rise, tau_decay):
    """Return the peak delay and the peak value of a biexponential.

    Peak delay ``u* = tau_r tau_d/(tau_d - tau_r) ln(tau_d/tau_r)`` and peak value
    ``m = r^(r/(1-r)) (1 - r)``, ``r = tau_r/tau_d``.
    """
    ratio = tau_rise / tau_decay
    delay = tau_rise * tau_decay / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
    value = np.exp(ratio / (1.0 - ratio) * np.log(ratio)) * (1.0 - ratio)
    return delay, value


@dataclass(frozen=True)
class Biexponential(_WidthShape):
    """Biexponential, causal: ``s(t) = [exp(-(t-t0)/tau_d) - exp(-(t-t0)/tau_r)] / m``.

    ``m`` is the peak value, reached at ``t0 + tau_r tau_d/(tau_d - tau_r) ln(tau_d/tau_r)``
    (the reported latency). Parameters ``(t0, tau_rise, tau_decay)`` in s.
    Box bounds: ``t0`` in ``[t_0 - max_fwhm, t_1]``, ``tau_rise`` in
    ``[1/sfreq, |W|/1.5]``, ``tau_decay`` in ``[1.5/sfreq, |W|]``. The constraint
    ``tau_decay >= 1.5 tau_rise`` of the specification is not a box constraint;
    it is listed in ``constraints`` for the fitting engine.
    """

    name: ClassVar[str] = "biexponential"
    param_names: ClassVar[tuple] = ("t0", "tau_rise", "tau_decay")
    latency_param: ClassVar[str] = "t0"
    onset_param: ClassVar[str] = "t0"
    constraints: ClassVar[tuple] = ("tau_decay >= 1.5 * tau_rise",)

    def _parts(self, times, params):
        t0, tau_r, tau_d = self._columns(params)
        u = as_times(times) - t0
        after = u > 0
        u = np.where(after, u, 0.0)
        decay, rise = np.exp(-u / tau_d), np.exp(-u / tau_r)
        numerator = np.where(after, decay - rise, 0.0)
        delay, peak = biexp_peak(tau_r, tau_d)
        return u, after, tau_r, tau_d, decay, rise, numerator, delay, peak

    def evaluate(self, times, params):
        """Unit-amplitude shape, ``(..., n_params) -> (..., n_times)``."""
        parts = self._parts(times, params)
        return parts[6] / parts[8]

    def jacobian(self, times, params):
        """Analytic derivatives, ``(..., n_params) -> (..., n_times, n_params)``."""
        u, after, tau_r, tau_d, decay, rise, numerator, delay, peak = self._parts(times, params)
        # Envelope theorem: the peak value moves with the partial derivatives at the peak.
        d_peak_r = -np.exp(-delay / tau_r) * delay / tau_r**2
        d_peak_d = np.exp(-delay / tau_d) * delay / tau_d**2
        d_t0 = np.where(after, decay / tau_d - rise / tau_r, 0.0)
        d_r = np.where(after, -rise * u / tau_r**2, 0.0)
        d_d = np.where(after, decay * u / tau_d**2, 0.0)
        return _stack(
            [
                d_t0 / peak,
                d_r / peak - numerator * d_peak_r / peak**2,
                d_d / peak - numerator * d_peak_d / peak**2,
            ]
        )

    def latency(self, params):
        """Return the latency (s) of the component, ``(..., n_params) -> (...)``."""
        params = self._params(params)
        return params[..., 0] + biexp_peak(params[..., 1], params[..., 2])[0]

    extremum = latency

    def default_bounds(self, times, data_range=None):  # noqa: ARG002  (shape protocol: signature shared by every shape)
        """Lower and upper bounds of the parameters for a window."""
        start, end, length, sfreq, _, high = self._window(times)
        return check_bounds(
            self.name, [start - high, 1.0 / sfreq, 1.5 / sfreq], [end, length / 1.5, length]
        )

    def initial_guess(self, times, waveform, peak_index):
        """Return the starting parameters for a candidate extremum of ``waveform``."""
        peak, sigma = self._center_width(times, waveform, peak_index)
        tau_r, tau_d = sigma / 2.0, 2.0 * sigma
        delay, _ = biexp_peak(tau_r, tau_d)
        return clip_to_bounds([peak - delay, tau_r, tau_d], self.default_bounds(times))


###################################################################################################
## Damped oscillation


def damped_peak(tau, freq, phase):
    """Return the delay ``u`` (from ``t0``) and value ``m`` of the largest damped cosine.

    The largest ``|exp(-u/tau) cos(2π f u + phase)|``.

    The envelope decreases, so the maximum is either at the onset (``|cos phase|``)
    or at the first stationary point ``u1 > 0``, where
    ``tan(2π f u1 + phase) = -1/(2π f tau)``.
    """
    omega = _TWO_PI * freq
    psi = -np.arctan(1.0 / (omega * tau))
    n = np.ceil((phase - psi) / np.pi)
    first = (psi + n * np.pi - phase) / omega
    at_first = np.exp(-first / tau) * np.cos(psi)
    at_onset = np.abs(np.cos(phase))
    onset_wins = at_onset >= at_first
    return np.where(onset_wins, 0.0, first), np.where(onset_wins, at_onset, at_first)


@dataclass(frozen=True)
class DampedOscillation(_OscillationShape):
    """Damped oscillation, causal: ``s(t) = exp(-(t-t0)/tau) cos(2π f (t-t0) + phi) / m``.

    ``m`` is the largest absolute value of the numerator, so that ``max |s| = 1``.
    Parameters ``(t0, tau, freq, phase)``: onset (s), decay (s), frequency (Hz),
    phase (rad). The reported latency is the onset ``t0``; ``features`` also
    lists every local extremum in ``peak_times``. Box bounds: ``t0`` in
    ``[t_0 - max_fwhm, t_1]``, ``tau`` in ``[1/freq_max, |W|]``, ``freq`` in
    ``[freq_min, freq_max]``, ``phase`` in ``[-π, π]``.
    """

    name: ClassVar[str] = "damped_oscillation"
    param_names: ClassVar[tuple] = ("t0", "tau", "freq", "phase")
    latency_param: ClassVar[str] = "t0"
    onset_param: ClassVar[str] = "t0"

    def _parts(self, times, params):
        t0, tau, freq, phase = self._columns(params)
        u = as_times(times) - t0
        after = u >= 0
        u = np.where(after, u, 0.0)
        envelope = np.where(after, np.exp(-u / tau), 0.0)
        angle = _TWO_PI * freq * u + phase
        raw = envelope * np.cos(angle)
        return u, tau, freq, phase, envelope, angle, raw

    def evaluate(self, times, params):
        """Unit-amplitude shape, ``(..., n_params) -> (..., n_times)``."""
        *_, tau, freq, phase = self._columns(params)
        raw = self._parts(times, params)[-1]
        return raw / damped_peak(tau, freq, phase)[1]

    def jacobian(self, times, params):
        """Analytic derivatives, ``(..., n_params) -> (..., n_times, n_params)``."""
        u, tau, freq, phase, envelope, angle, raw = self._parts(times, params)
        omega = _TWO_PI * freq
        sine = envelope * np.sin(angle)
        d_raw = [raw / tau + omega * sine, raw * u / tau**2, -sine * _TWO_PI * u, -sine]
        # Envelope theorem on the normaliser, at the delay of the largest extremum.
        delay, peak = damped_peak(tau, freq, phase)
        peak_angle = omega * delay + phase
        e_peak = np.exp(-delay / tau)
        sign = np.sign(np.cos(peak_angle))
        d_peak = [
            0.0,
            sign * e_peak * np.cos(peak_angle) * delay / tau**2,
            -sign * e_peak * np.sin(peak_angle) * _TWO_PI * delay,
            -sign * e_peak * np.sin(peak_angle),
        ]
        return _stack(
            [d / peak - raw * dp / peak**2 for d, dp in zip(d_raw, d_peak, strict=False)]
        )

    def latency(self, params):
        """Return the latency (s) of the component, ``(..., n_params) -> (...)``."""
        return self._params(params)[..., 0]

    def extremum(self, params):
        """Return the time (s) of the largest extremum, ``(..., n_params) -> (...)``."""
        params = self._params(params)
        delay, _ = damped_peak(params[..., 1], params[..., 2], params[..., 3])
        return params[..., 0] + delay

    def default_bounds(self, times, data_range=None):  # noqa: ARG002  (shape protocol: signature shared by every shape)
        """Lower and upper bounds of the parameters for a window."""
        start, end, length, sfreq, _, high = self._window(times)
        f_low, f_high = self._freqs(length, sfreq)
        return check_bounds(
            self.name, [start - high, 1.0 / f_high, f_low, -np.pi], [end, length, f_high, np.pi]
        )

    def initial_guess(self, times, waveform, peak_index):
        """Return the starting parameters for a candidate extremum of ``waveform``."""
        # Frequency from the periodogram of the window (a fixed mid-range start left the
        # fit in a wrong local minimum at a few samples per cycle); onset a quarter period
        # before the candidate peak, phase -pi/2 so that the first extremum is at the peak.
        peak, sigma = self._center_width(times, waveform, peak_index)
        lower, upper = self.default_bounds(times)
        freq = self._freq_guess(times, waveform, lower[2], upper[2])
        return clip_to_bounds(
            [peak - 0.25 / freq, 2.0 * sigma, freq, -0.5 * np.pi], (lower, upper)
        )


###################################################################################################
## Gabor


def gabor_peak(sigma, freq, phase, n_grid=257, n_newton=8):
    """Return the offset ``u*`` (from ``t0``) and value ``m`` of the largest Gabor value.

    The largest ``|exp(-u²/(2 sigma²)) cos(2π f u + phase)|``.

    The maximum lies within ``|u| <= 1/(4f)`` (a crest of the cosine is that
    close to the envelope centre), and in practice within ``8 sigma``; it is
    located on a grid and refined by Newton steps on the derivative.
    """
    sigma, freq, phase = np.broadcast_arrays(
        *(np.asarray(x, dtype=np.float64) for x in (sigma, freq, phase))
    )
    omega = _TWO_PI * freq
    reach = np.minimum(np.pi / (2.0 * omega), 8.0 * sigma)
    nodes = reach[..., None] * np.linspace(-1.0, 1.0, n_grid)
    values = np.exp(-0.5 * (nodes / sigma[..., None]) ** 2) * np.cos(
        omega[..., None] * nodes + phase[..., None]
    )
    best = np.take_along_axis(nodes, np.argmax(np.abs(values), axis=-1)[..., None], -1)[..., 0]
    step = 2.0 * reach / (n_grid - 1)
    u = best
    for _ in range(n_newton):
        env = np.exp(-0.5 * (u / sigma) ** 2)
        cos, sin = np.cos(omega * u + phase), np.sin(omega * u + phase)
        first = env * (-(u / sigma**2) * cos - omega * sin)
        second = env * (
            (u**2 / sigma**4 - 1.0 / sigma**2) * cos
            + 2.0 * (u / sigma**2) * omega * sin
            - omega**2 * cos
        )
        update = np.divide(first, second, out=np.zeros_like(first), where=second != 0)
        u = np.clip(u - update, best - step, best + step)
    value = np.exp(-0.5 * (u / sigma) ** 2) * np.cos(omega * u + phase)
    return u, np.abs(value)


@dataclass(frozen=True)
class Gabor(_OscillationShape):
    """Gabor ``s(t) = exp(-(t-t0)²/(2 sigma²)) cos(2π f (t-t0) + phi) / m``.

    ``m`` is the largest absolute value of the numerator, so that ``max |s| = 1``.
    Parameters ``(t0, sigma, freq, phase)``: envelope centre (s), envelope width
    (s), frequency (Hz), phase (rad). The reported latency is ``t0``. Bounds:
    ``t0`` and ``sigma`` as the Gaussian ``mu`` and ``sigma``, ``freq`` in
    ``[freq_min, freq_max]``, ``phase`` in ``[-π, π]``.
    """

    name: ClassVar[str] = "gabor"
    param_names: ClassVar[tuple] = ("t0", "sigma", "freq", "phase")
    latency_param: ClassVar[str] = "t0"

    def _parts(self, times, params):
        t0, sigma, freq, phase = self._columns(params)
        u = as_times(times) - t0
        envelope = np.exp(-0.5 * (u / sigma) ** 2)
        angle = _TWO_PI * freq * u + phase
        return u, sigma, freq, phase, envelope, angle, envelope * np.cos(angle)

    def evaluate(self, times, params):
        """Unit-amplitude shape, ``(..., n_params) -> (..., n_times)``."""
        _, sigma, freq, phase, _, _, raw = self._parts(times, params)
        return raw / gabor_peak(sigma, freq, phase)[1]

    def jacobian(self, times, params):
        """Analytic derivatives, ``(..., n_params) -> (..., n_times, n_params)``."""
        u, sigma, freq, phase, envelope, angle, raw = self._parts(times, params)
        omega = _TWO_PI * freq
        sine = envelope * np.sin(angle)
        d_raw = [
            u / sigma**2 * raw + omega * sine,
            raw * u**2 / sigma**3,
            -sine * _TWO_PI * u,
            -sine,
        ]
        # Envelope theorem on the normaliser, at the offset of the largest extremum.
        offset, peak = gabor_peak(sigma, freq, phase)
        e_peak = np.exp(-0.5 * (offset / sigma) ** 2)
        peak_angle = omega * offset + phase
        sign = np.sign(np.cos(peak_angle))
        d_peak = [
            0.0,
            peak * offset**2 / sigma**3,
            -sign * e_peak * np.sin(peak_angle) * _TWO_PI * offset,
            -sign * e_peak * np.sin(peak_angle),
        ]
        return _stack(
            [d / peak - raw * dp / peak**2 for d, dp in zip(d_raw, d_peak, strict=False)]
        )

    def latency(self, params):
        """Return the latency (s) of the component, ``(..., n_params) -> (...)``."""
        return self._params(params)[..., 0]

    def extremum(self, params):
        """Return the time (s) of the largest extremum, ``(..., n_params) -> (...)``."""
        params = self._params(params)
        offset, _ = gabor_peak(params[..., 1], params[..., 2], params[..., 3])
        return params[..., 0] + offset

    def default_bounds(self, times, data_range=None):  # noqa: ARG002  (shape protocol: signature shared by every shape)
        """Lower and upper bounds of the parameters for a window."""
        start, end, length, sfreq, low, high = self._window(times)
        f_low, f_high = self._freqs(length, sfreq)
        sigma_max = high / FWHM_PER_SIGMA
        return check_bounds(
            self.name,
            [start - sigma_max / 2, low / FWHM_PER_SIGMA, f_low, -np.pi],
            [end + sigma_max / 2, sigma_max, f_high, np.pi],
        )

    def initial_guess(self, times, waveform, peak_index):
        """Return the starting parameters for a candidate extremum of ``waveform``."""
        peak, sigma = self._center_width(times, waveform, peak_index)
        lower, upper = self.default_bounds(times)
        freq = self._freq_guess(times, waveform, lower[2], upper[2])
        return clip_to_bounds([peak, sigma, freq, 0.0], (lower, upper))


###################################################################################################
## Template


@dataclass(frozen=True, eq=False)
class Template(ShapeBase):
    """Template ``s(t) = T((t - shift)/stretch) / max|T|``.

    ``T`` is a waveform sampled at ``template_times`` (a ``ShapeBasis`` curve or
    a canonical template), interpolated by a clamped cubic spline (zero slope at
    both ends) and zero outside its support; it should therefore start and end
    at rest. Parameters ``(shift,)``, or ``(shift, stretch)`` when
    ``stretch=True``. The reported latency is the extremum of the shifted
    template, ``shift + stretch u*`` with ``u*`` the extremum of ``T``. Bounds:
    ``shift`` in ``±max_shift`` (default ``|W|/4``), ``stretch`` in
    ``[stretch_min, stretch_max]`` (default 0.8 to 1.25).
    """

    template_times: np.ndarray
    template_values: np.ndarray
    stretch: bool = False
    max_shift: float | None = None
    stretch_min: float = 0.8
    stretch_max: float = 1.25
    _spline: object = field(init=False, repr=False)
    _slope: object = field(init=False, repr=False)
    _norm: float = field(init=False, repr=False)
    _peak: float = field(init=False, repr=False)

    name: ClassVar[str] = "template"
    latency_param: ClassVar[str] = "shift"

    def __post_init__(self):
        """Check and normalise the fields."""
        times = np.asarray(self.template_times, dtype=np.float64)
        values = np.asarray(self.template_values, dtype=np.float64)
        if times.ndim != 1 or times.size < 4 or values.shape != times.shape:
            raise DataError(
                "template: template_times and template_values must be 1-D arrays "
                "of the same length (at least 4 samples)."
            )
        if not (np.all(np.isfinite(times)) and np.all(np.isfinite(values))):
            raise DataError("template: template_times and template_values must be finite.")
        if np.any(np.diff(times) <= 0):
            raise DataError("template: template_times must be strictly increasing.")
        if not np.any(values != 0):
            raise DataError("template: template_values are all zero.")
        if self.max_shift is not None and not self.max_shift > 0:
            raise SettingsError(f"template: max_shift must be positive, got {self.max_shift}.")
        if not 0 < self.stretch_min < 1.0 < self.stretch_max:
            raise SettingsError(
                "template: need 0 < stretch_min < 1 < stretch_max, got "
                f"{self.stretch_min}, {self.stretch_max}."
            )
        spline = CubicSpline(times, values, bc_type="clamped")
        slope = spline.derivative()
        roots = slope.roots(extrapolate=False)
        candidates = np.concatenate([roots[np.isfinite(roots)], times[[0, -1]], times])
        found = spline(candidates)
        best = int(np.argmax(np.abs(found)))
        for name, value in (
            ("template_times", times),
            ("template_values", values),
            ("_spline", spline),
            ("_slope", slope),
            ("_norm", float(abs(found[best]))),
            ("_peak", float(candidates[best])),
        ):
            object.__setattr__(self, name, value)

    @property
    def param_names(self):
        """Names of the parameters."""
        return ("shift", "stretch") if self.stretch else ("shift",)

    @property
    def norm(self):
        """``max |T|`` of the interpolated template."""
        return self._norm

    def _parts(self, times, params):
        columns = self._columns(params)
        shift = columns[0]
        stretch = columns[1] if self.stretch else 1.0
        v = (as_times(times) - shift) / stretch
        inside = (v >= self.template_times[0]) & (v <= self.template_times[-1])
        v_in = np.clip(v, self.template_times[0], self.template_times[-1])
        return v, stretch, inside, v_in

    def evaluate(self, times, params):
        """Unit-amplitude shape, ``(..., n_params) -> (..., n_times)``."""
        _, _, inside, v_in = self._parts(times, params)
        return np.where(inside, self._spline(v_in), 0.0) / self._norm

    def jacobian(self, times, params):
        """Analytic derivatives, ``(..., n_params) -> (..., n_times, n_params)``."""
        v, stretch, inside, v_in = self._parts(times, params)
        slope = np.where(inside, self._slope(v_in), 0.0) / (stretch * self._norm)
        columns = [-slope, -slope * v] if self.stretch else [-slope]
        return _stack(columns)

    def latency(self, params):
        """Return the latency (s) of the component, ``(..., n_params) -> (...)``."""
        params = self._params(params)
        stretch = params[..., 1] if self.stretch else 1.0
        return params[..., 0] + stretch * self._peak

    extremum = latency

    def default_bounds(self, times, data_range=None):  # noqa: ARG002  (shape protocol: signature shared by every shape)
        """Lower and upper bounds of the parameters for a window."""
        _, _, length, _ = window_info(times)
        max_shift = length / 4.0 if self.max_shift is None else float(self.max_shift)
        lower, upper = [-max_shift], [max_shift]
        if self.stretch:
            lower.append(self.stretch_min)
            upper.append(self.stretch_max)
        return check_bounds(self.name, lower, upper)

    def initial_guess(self, times, waveform, peak_index):
        """Return the starting parameters for a candidate extremum of ``waveform``."""
        times, _ = check_waveform(times, waveform, peak_index)
        guess = [times[int(peak_index)] - self._peak] + ([1.0] if self.stretch else [])
        return clip_to_bounds(guess, self.default_bounds(times))
