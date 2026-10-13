# SPDX-License-Identifier: Apache-2.0
"""Independent scalar references for the shape and background equations.

Each function takes a backend ``B`` providing ``mpf``, ``exp``, ``log``,
``erf``, ``erfc``, ``cos``, ``sin``, ``sqrt``, ``pi`` and ``bisections``; with
:data:`MATH` it runs in float64, with :func:`mp_backend` in mpmath at high
precision (REQ-100). The references evaluate the equations of spec §5.5.1 and
§5.5.2 directly, one sample at a time, and find every normaliser (maximum of
the un-normalised curve) by bisection on its derivative, without using the
closed forms or the vectorised code of ``evoparam.shapes``.
"""

import math
from types import SimpleNamespace
from typing import Any

MATH = SimpleNamespace(
    mpf=float,
    exp=math.exp,
    log=math.log,
    erf=math.erf,
    erfc=math.erfc,
    cos=math.cos,
    sin=math.sin,
    sqrt=math.sqrt,
    pi=math.pi,
    bisections=200,
)


def mp_backend(mpmath: Any, dps: Any = 40) -> Any:
    """Backend on mpmath with ``dps`` decimal digits."""

    mpmath.mp.dps = dps
    mp = mpmath.mp
    return SimpleNamespace(
        mpf=mp.mpf,
        exp=mp.exp,
        log=mp.log,
        erf=mp.erf,
        erfc=mp.erfc,
        cos=mp.cos,
        sin=mp.sin,
        sqrt=mp.sqrt,
        pi=mp.pi,
        bisections=4 * dps,
    )


def _bisect(bk: Any, positive_left: Any, low: Any, high: Any) -> Any:
    """Root of a function that is positive left of the root on ``[low, high]``."""

    for _ in range(bk.bisections):
        mid = (low + high) / 2
        if positive_left(mid):
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _max_abs(bk: Any, curve: Any, slope: Any, low: Any, high: Any, n_scan: Any) -> Any:
    """Largest ``|curve|`` on ``[low, high]``: scan, then bisection on the slope."""

    step = (high - low) / n_scan
    nodes = [low + i * step for i in range(n_scan + 1)]
    values = [abs(curve(x)) for x in nodes]
    best = max(range(n_scan + 1), key=lambda i: values[i])
    candidates = [values[0], values[-1], values[best]]
    if 0 < best < n_scan:
        sign = 1 if curve(nodes[best]) > 0 else -1
        x = _bisect(bk, lambda u: sign * slope(u) > 0, nodes[best - 1], nodes[best + 1])
        candidates.append(abs(curve(x)))
    return max(candidates)


def gaussian(bk: Any, mu: Any, sigma: Any) -> Any:
    return lambda t: bk.exp(-(((t - mu) / sigma) ** 2) / 2)


def skewed_gaussian(bk: Any, xi: Any, omega: Any, alpha: Any) -> Any:
    def numerator(z):
        return bk.exp(-(z**2) / 2) * (1 + bk.erf(alpha * z / bk.sqrt(2)))

    def slope(z):
        return -z * numerator(z) + bk.sqrt(2 / bk.pi) * alpha * bk.exp(
            -(z**2) * (1 + alpha**2) / 2
        )

    low, high = (bk.mpf(0), bk.mpf(2)) if alpha >= 0 else (bk.mpf(-2), bk.mpf(0))
    norm = numerator(_bisect(bk, lambda z: slope(z) > 0, low, high))
    return lambda t: numerator((t - xi) / omega) / norm


def exgaussian(bk: Any, mu: Any, sigma: Any, tau: Any) -> Any:
    def h(x):
        return (
            (1 / (2 * tau))
            * bk.exp(sigma**2 / (2 * tau**2) - (x - mu) / tau)
            * bk.erfc((sigma / tau - (x - mu) / sigma) / bk.sqrt(2))
        )

    lam = sigma / tau

    def positive_left(z):
        x = (lam - z) / bk.sqrt(2)
        return bk.sqrt(2 / bk.pi) * bk.exp(-(x**2)) - lam * bk.erfc(x) > 0

    norm = h(mu + sigma * _bisect(bk, positive_left, bk.mpf(-1), lam + 10))
    return lambda t: h(t) / norm


def gamma(bk: Any, t0: Any, kappa: Any, theta: Any) -> Any:
    def curve(t):
        if t <= t0:
            return bk.mpf(0)
        return ((t - t0) / ((kappa - 1) * theta)) ** (kappa - 1) * bk.exp(
            kappa - 1 - (t - t0) / theta
        )

    return curve


def biexponential(bk: Any, t0: Any, tau_rise: Any, tau_decay: Any) -> Any:
    def numerator(u):
        return bk.exp(-u / tau_decay) - bk.exp(-u / tau_rise)

    def slope(u):
        return -bk.exp(-u / tau_decay) / tau_decay + bk.exp(-u / tau_rise) / tau_rise

    norm = numerator(_bisect(bk, lambda u: slope(u) > 0, bk.mpf(0), 50 * tau_decay))
    return lambda t: numerator(t - t0) / norm if t > t0 else bk.mpf(0)


def damped_oscillation(bk: Any, t0: Any, tau: Any, freq: Any, phase: Any) -> Any:
    omega = 2 * bk.pi * freq

    def curve(u):
        return bk.exp(-u / tau) * bk.cos(omega * u + phase)

    def slope(u):
        angle = omega * u + phase
        return -bk.exp(-u / tau) * (bk.cos(angle) / tau + omega * bk.sin(angle))

    # The envelope decreases: the largest extremum is within one period of the onset.
    norm = _max_abs(bk, curve, slope, bk.mpf(0), 1 / freq, 400)
    return lambda t: curve(t - t0) / norm if t >= t0 else bk.mpf(0)


def gabor(bk: Any, t0: Any, sigma: Any, freq: Any, phase: Any) -> Any:
    omega = 2 * bk.pi * freq

    def curve(u):
        return bk.exp(-(u**2) / (2 * sigma**2)) * bk.cos(omega * u + phase)

    def slope(u):
        return bk.exp(-(u**2) / (2 * sigma**2)) * (
            -(u / sigma**2) * bk.cos(omega * u + phase) - omega * bk.sin(omega * u + phase)
        )

    # A crest of the cosine lies within a quarter period of the centre.
    reach = 1 / (4 * freq) + sigma / 100
    norm = _max_abs(bk, curve, slope, -reach, reach, 2000)
    return lambda t: curve(t - t0) / norm


def bspline(bk: Any, knots: Any, j: Any, degree: Any, t: Any) -> Any:
    """B-spline ``B_j`` of the given degree (Cox-de Boor recursion), right-open intervals."""

    if degree == 0:
        return bk.mpf(1) if knots[j] <= t < knots[j + 1] else bk.mpf(0)
    left = (t - knots[j]) / (knots[j + degree] - knots[j]) * bspline(bk, knots, j, degree - 1, t)
    right = (
        (knots[j + degree + 1] - t)
        / (knots[j + degree + 1] - knots[j + 1])
        * bspline(bk, knots, j + 1, degree - 1, t)
    )
    return left + right


SHAPES = {
    "gaussian": gaussian,
    "skewed_gaussian": skewed_gaussian,
    "exgaussian": exgaussian,
    "gamma": gamma,
    "biexponential": biexponential,
    "damped_oscillation": damped_oscillation,
    "gabor": gabor,
}


def reference_curve(bk: Any, name: Any, times: Any, params: Any) -> Any:
    """Reference values of shape ``name`` at ``times`` (floats), returned as floats."""

    curve = SHAPES[name](bk, *[bk.mpf(float(p)) for p in params])
    return [float(curve(bk.mpf(float(t)))) for t in times]
