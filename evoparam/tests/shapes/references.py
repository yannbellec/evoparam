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


def mp_backend(mpmath, dps=40):
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


def _bisect(B, positive_left, low, high):
    """Root of a function that is positive left of the root on ``[low, high]``."""

    for _ in range(B.bisections):
        mid = (low + high) / 2
        if positive_left(mid):
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _max_abs(B, curve, slope, low, high, n_scan):
    """Largest ``|curve|`` on ``[low, high]``: scan, then bisection on the slope."""

    step = (high - low) / n_scan
    nodes = [low + i * step for i in range(n_scan + 1)]
    values = [abs(curve(x)) for x in nodes]
    best = max(range(n_scan + 1), key=lambda i: values[i])
    candidates = [values[0], values[-1], values[best]]
    if 0 < best < n_scan:
        sign = 1 if curve(nodes[best]) > 0 else -1
        x = _bisect(B, lambda u: sign * slope(u) > 0, nodes[best - 1], nodes[best + 1])
        candidates.append(abs(curve(x)))
    return max(candidates)


def gaussian(B, mu, sigma):
    return lambda t: B.exp(-(((t - mu) / sigma) ** 2) / 2)


def skewed_gaussian(B, xi, omega, alpha):
    def numerator(z):
        return B.exp(-(z**2) / 2) * (1 + B.erf(alpha * z / B.sqrt(2)))

    def slope(z):
        return -z * numerator(z) + B.sqrt(2 / B.pi) * alpha * B.exp(-(z**2) * (1 + alpha**2) / 2)

    low, high = (B.mpf(0), B.mpf(2)) if alpha >= 0 else (B.mpf(-2), B.mpf(0))
    norm = numerator(_bisect(B, lambda z: slope(z) > 0, low, high))
    return lambda t: numerator((t - xi) / omega) / norm


def exgaussian(B, mu, sigma, tau):
    def h(x):
        return (
            (1 / (2 * tau))
            * B.exp(sigma**2 / (2 * tau**2) - (x - mu) / tau)
            * B.erfc((sigma / tau - (x - mu) / sigma) / B.sqrt(2))
        )

    lam = sigma / tau

    def positive_left(z):
        x = (lam - z) / B.sqrt(2)
        return B.sqrt(2 / B.pi) * B.exp(-(x**2)) - lam * B.erfc(x) > 0

    norm = h(mu + sigma * _bisect(B, positive_left, B.mpf(-1), lam + 10))
    return lambda t: h(t) / norm


def gamma(B, t0, kappa, theta):
    def curve(t):
        if t <= t0:
            return B.mpf(0)
        return ((t - t0) / ((kappa - 1) * theta)) ** (kappa - 1) * B.exp(
            kappa - 1 - (t - t0) / theta
        )

    return curve


def biexponential(B, t0, tau_rise, tau_decay):
    def numerator(u):
        return B.exp(-u / tau_decay) - B.exp(-u / tau_rise)

    def slope(u):
        return -B.exp(-u / tau_decay) / tau_decay + B.exp(-u / tau_rise) / tau_rise

    norm = numerator(_bisect(B, lambda u: slope(u) > 0, B.mpf(0), 50 * tau_decay))
    return lambda t: numerator(t - t0) / norm if t > t0 else B.mpf(0)


def damped_oscillation(B, t0, tau, freq, phase):
    omega = 2 * B.pi * freq

    def curve(u):
        return B.exp(-u / tau) * B.cos(omega * u + phase)

    def slope(u):
        angle = omega * u + phase
        return -B.exp(-u / tau) * (B.cos(angle) / tau + omega * B.sin(angle))

    # The envelope decreases: the largest extremum is within one period of the onset.
    norm = _max_abs(B, curve, slope, B.mpf(0), 1 / freq, 400)
    return lambda t: curve(t - t0) / norm if t >= t0 else B.mpf(0)


def gabor(B, t0, sigma, freq, phase):
    omega = 2 * B.pi * freq

    def curve(u):
        return B.exp(-(u**2) / (2 * sigma**2)) * B.cos(omega * u + phase)

    def slope(u):
        return B.exp(-(u**2) / (2 * sigma**2)) * (
            -(u / sigma**2) * B.cos(omega * u + phase) - omega * B.sin(omega * u + phase)
        )

    # A crest of the cosine lies within a quarter period of the centre.
    reach = 1 / (4 * freq) + sigma / 100
    norm = _max_abs(B, curve, slope, -reach, reach, 2000)
    return lambda t: curve(t - t0) / norm


def bspline(B, knots, j, degree, t):
    """B-spline ``B_j`` of the given degree (Cox-de Boor recursion), right-open intervals."""

    if degree == 0:
        return B.mpf(1) if knots[j] <= t < knots[j + 1] else B.mpf(0)
    left = (t - knots[j]) / (knots[j + degree] - knots[j]) * bspline(B, knots, j, degree - 1, t)
    right = (
        (knots[j + degree + 1] - t)
        / (knots[j + degree + 1] - knots[j + 1])
        * bspline(B, knots, j + 1, degree - 1, t)
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


def reference_curve(B, name, times, params):
    """Reference values of shape ``name`` at ``times`` (floats), returned as floats."""

    curve = SHAPES[name](B, *[B.mpf(float(p)) for p in params])
    return [float(curve(B.mpf(float(t)))) for t in times]
