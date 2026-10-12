# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# Derived from SciPy (scipy/optimize/_lsq/trf.py and common.py, BSD-3-Clause, named in NOTICE);
#   changes: only the bounded, dense, linear-loss path with x_scale="jac" and the exact
#   trust-region solver is kept; the Jacobian can be reduced by a QR factorisation once per
#   evaluation, so that each iteration works on n x n matrices instead of m x n ones
"""Trust-region reflective least squares for the small bounded problems of the engine.

The algorithm of :func:`scipy.optimize.least_squares` with ``method="trf"``, bounds, a dense
Jacobian, ``loss="linear"``, ``x_scale="jac"`` and ``tr_solver="exact"`` (Branch, Coleman
and Li 1999; the trust-region subproblem as in Moré 1977, from one SVD): the same steps,
the same radius updates, the same termination tests. Two things differ:

- with ``reduced=True`` (the default) the Jacobian ``J`` (m x n: m residuals, a thousand or
  more here, and n parameters, a few) is factorised once per evaluation, ``J = Q R``. Every
  quantity of the iteration is an inner product of columns of ``J`` or of ``J^T f``, so it
  is computed from ``R`` and ``Q^T f``; SciPy takes the SVD of the (m + n) x n augmented
  matrix at every iteration. ``reduced=False`` keeps SciPy's operations, in SciPy's order;
- none of the general wrappers of ``least_squares`` (function objects, input checks, loss
  functions, callbacks), whose overhead was a quarter of the time of a fit.

The iterates agree with SciPy's: exactly with ``reduced=False``, to rounding with
``reduced=True`` (evoparam/tests/models/test_trf.py).
"""

import functools
import math
from collections.abc import Callable
from typing import cast

import numpy as np
from scipy.linalg.lapack import dgeqrf, dgesdd, dgesdd_lwork, dormqr

from evoparam._typing import ArrayLike, BoolArray, FloatArray, IntArray

__all__ = ["least_squares_trf"]

EPS = np.finfo(float).eps


def least_squares_trf(
    fun: Callable[[FloatArray], ArrayLike],
    jac: Callable[[FloatArray], ArrayLike],
    x0: ArrayLike,
    lb: ArrayLike,
    ub: ArrayLike,
    *,
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    max_nfev: int | None = None,
    reduced: bool = True,
    m: int | None = None,
) -> tuple[FloatArray, float, int, int]:
    """Minimise ``0.5 * ||fun(x)||^2`` subject to ``lb <= x <= ub``.

    ``jac(x)`` is called only at a point where ``fun`` was just called (the engine keeps
    the linear solve of that point). Returns ``(x, cost, nfev, status)``, the status codes
    of SciPy (0: max_nfev reached, 1: gtol, 2: ftol, 3: xtol, 4: ftol and xtol). Raises
    ValueError, as ``least_squares`` does, for an infeasible start or non-finite initial
    residuals, and LinAlgError when an SVD does not converge.

    ``m``: the number of residuals of the problem, when ``fun`` and ``jac`` return them in
    a reduced orthonormal basis (fewer rows, the same inner products); it sets the rank
    threshold of the trust-region step as in SciPy. By default the rows of ``jac(x0)``.
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    if np.any(lb >= ub):
        raise ValueError("Each lower bound must be strictly less than each upper bound.")
    if not np.all((x0 >= lb) & (x0 <= ub)):
        raise ValueError("Initial guess is outside of provided bounds")
    x = _strictly_feasible(x0, lb, ub, 1e-10)
    f = np.atleast_1d(fun(x))
    jac_x = np.atleast_2d(jac(x))
    if not np.all(np.isfinite(f)):
        raise ValueError("Residuals are not finite in the initial point.")
    m = jac_x.shape[0] if m is None else int(m)
    n = jac_x.shape[1]
    if max_nfev is None:
        max_nfev = n * 100
    nfev = 1
    cost = 0.5 * np.dot(f, f)
    finite_lb, finite_ub = np.isfinite(lb), np.isfinite(ub)

    jr, jf, g = _reduce(jac_x, f, reduced)
    scale_inv = (jr**2).sum(axis=0) ** 0.5
    scale_inv[scale_inv == 0] = 1
    scale = 1 / scale_inv

    v, dv = _cl_scaling(x, g, lb, ub, finite_lb, finite_ub)
    v = np.where(dv != 0, v * scale_inv, v)
    delta = _norm(x * scale_inv / v**0.5)
    if delta == 0:
        delta = 1.0

    rows = jr.shape[0]
    augmented = np.empty((rows + n, n), order="F")
    lwork = int(dgesdd_lwork(rows + n, n, compute_uv=1, full_matrices=0)[0])
    f_augmented = np.zeros(rows + n)
    alpha = 0.0
    status = None
    while True:
        v, dv = _cl_scaling(x, g, lb, ub, finite_lb, finite_ub)
        g_norm = np.abs(g * v).max() if n else 0.0
        if g_norm < gtol:
            status = 1
        if status is not None or nfev == max_nfev:
            break
        v = np.where(dv != 0, v * scale_inv, v)
        d = v**0.5 * scale
        diag_h = g * dv * scale
        g_h = d * g

        f_augmented[:rows] = jf
        augmented[:rows] = jr * d
        j_h = augmented[:rows]
        augmented[rows:] = np.diag(diag_h**0.5)
        u, s, vt, info = dgesdd(augmented, compute_uv=1, full_matrices=0, lwork=lwork)
        if info > 0:
            raise np.linalg.LinAlgError("SVD did not converge")
        v_mat = vt.T
        uf = u.T.dot(f_augmented)
        theta = max(0.995, 1 - g_norm)

        actual_reduction = -1
        while actual_reduction <= 0 and nfev < max_nfev:
            p_h, alpha = _solve_trust_region(n, m, uf, s, v_mat, delta, alpha)
            p = d * p_h
            step, step_h, predicted = _select_step(
                x, j_h, diag_h, g_h, p, p_h, d, delta, lb, ub, theta
            )
            x_new = _strictly_feasible(x + step, lb, ub, 0)
            f_new = np.atleast_1d(fun(x_new))
            nfev += 1
            step_h_norm = _norm(step_h)
            if not np.isfinite(f_new).all():
                delta = 0.25 * step_h_norm
                continue
            cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new
            delta_new, ratio = _update_radius(
                delta, actual_reduction, predicted, step_h_norm, step_h_norm > 0.95 * delta
            )
            status = _termination(actual_reduction, cost, _norm(step), _norm(x), ratio, ftol, xtol)
            if status is not None:
                break
            alpha *= delta / delta_new
            delta = delta_new

        if actual_reduction > 0:
            x, f, cost = x_new, f_new, cost_new
            jac_x = np.atleast_2d(jac(x))
            jr, jf, g = _reduce(jac_x, f, reduced)
            scale_inv = np.maximum((jr**2).sum(axis=0) ** 0.5, scale_inv)
            scale = 1 / scale_inv
    return x, cost, nfev, 0 if status is None else status


def _norm(x: FloatArray) -> float:
    return math.sqrt(np.dot(x, x))


def _reduce(
    jac_mat: FloatArray, f: FloatArray, reduced: bool
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """(R, Q^T f, J^T f) of ``J = Q R`` (economic), or (J, f, J^T f) unreduced."""
    m, n = jac_mat.shape
    if not reduced or m <= n:
        return jac_mat, f, jac_mat.T.dot(f)
    qr, tau, _, _ = dgeqrf(jac_mat)
    qtf, _, _ = dormqr("L", "T", qr, tau, f[:, None], 64)
    r = qr[:n] * _upper(n)
    qtf = qtf[:n, 0]
    return r, qtf, r.T.dot(qtf)


@functools.lru_cache(maxsize=32)
def _upper(n: int) -> FloatArray:
    """Mask of the upper triangle of an n x n matrix (read only)."""
    mask = np.triu(np.ones((n, n)))
    mask.flags.writeable = False
    return mask


def _cl_scaling(
    x: FloatArray,
    g: FloatArray,
    lb: FloatArray,
    ub: FloatArray,
    finite_lb: BoolArray,
    finite_ub: BoolArray,
) -> tuple[FloatArray, FloatArray]:
    """Coleman-Li scaling vector ``v`` and its derivative ``dv``."""
    upper = (g < 0) & finite_ub
    lower = (g > 0) & finite_lb
    with np.errstate(invalid="ignore"):  # inf - inf where a bound is infinite
        v = np.where(lower, x - lb, np.where(upper, ub - x, 1.0))
    dv = lower - upper.astype(float)
    return v, dv


def _strictly_feasible(x: FloatArray, lb: FloatArray, ub: FloatArray, rstep: float) -> FloatArray:
    """``x`` moved inside the bounds by ``rstep`` (relative; ``0``: one ulp)."""
    x_new: FloatArray = x.copy()
    if rstep == 0 and ((x > lb) & (x < ub)).all():
        return x_new  # no bound active: nothing moves
    active = _active(x, lb, ub, rstep)
    lower, upper = active == -1, active == 1
    if rstep == 0:
        x_new[lower] = np.nextafter(lb[lower], ub[lower])
        x_new[upper] = np.nextafter(ub[upper], lb[upper])
    else:
        x_new[lower] = lb[lower] + rstep * np.maximum(1, np.abs(lb[lower]))
        x_new[upper] = ub[upper] - rstep * np.maximum(1, np.abs(ub[upper]))
    tight = (x_new < lb) | (x_new > ub)
    x_new[tight] = 0.5 * (lb[tight] + ub[tight])
    return x_new


def _active(x: FloatArray, lb: FloatArray, ub: FloatArray, rtol: float) -> IntArray:
    active = np.zeros_like(x, dtype=int)
    if rtol == 0:
        active[x <= lb] = -1
        active[x >= ub] = 1
        return active
    lower_dist = x - lb
    upper_dist = ub - x
    lower_threshold = rtol * np.maximum(1, np.abs(lb))
    upper_threshold = rtol * np.maximum(1, np.abs(ub))
    active[np.isfinite(lb) & (lower_dist <= np.minimum(upper_dist, lower_threshold))] = -1
    active[np.isfinite(ub) & (upper_dist <= np.minimum(lower_dist, upper_threshold))] = 1
    return active


def _solve_trust_region(
    n: int,
    m: int,
    uf: FloatArray,
    s: FloatArray,
    v_mat: FloatArray,
    delta: float,
    alpha: float,
    rtol: float = 0.01,
    max_iter: int = 10,
) -> tuple[FloatArray, float]:
    """Trust-region subproblem from the SVD of the (scaled) Jacobian (Moré 1977).

    Returns the step and the Levenberg-Marquardt parameter.
    """
    suf = s * uf
    full_rank = s[-1] > EPS * m * s[0] if m >= n else False
    if full_rank:
        p = -v_mat.dot(uf / s)
        if _norm(p) <= delta:
            return p, 0.0

    def phi_and_derivative(alpha: float) -> tuple[float, float]:
        denom = s**2 + alpha
        p_norm = _norm(suf / denom)
        return p_norm - delta, -(suf**2 / denom**3).sum() / p_norm

    alpha_upper = _norm(suf) / delta
    if full_rank:
        phi, phi_prime = phi_and_derivative(0.0)
        alpha_lower = -phi / phi_prime
    else:
        alpha_lower = 0.0
    if not full_rank and alpha == 0:
        alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper) ** 0.5)
    for _ in range(max_iter):
        if alpha < alpha_lower or alpha > alpha_upper:
            alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper) ** 0.5)
        phi, phi_prime = phi_and_derivative(alpha)
        if phi < 0:
            alpha_upper = alpha
        ratio = phi / phi_prime
        alpha_lower = max(alpha_lower, alpha - ratio)
        alpha -= (phi + delta) * ratio / delta
        if np.abs(phi) < rtol * delta:
            break
    p = -v_mat.dot(suf / (s**2 + alpha))
    p *= delta / _norm(p)
    return p, alpha


def _update_radius(
    delta: float, actual: float, predicted: float, step_norm: float, bound_hit: bool
) -> tuple[float, float]:
    if predicted > 0:
        ratio = actual / predicted
    elif predicted == actual == 0:
        ratio = 1
    else:
        ratio = 0
    if ratio < 0.25:
        delta = 0.25 * step_norm
    elif ratio > 0.75 and bound_hit:
        delta *= 2.0
    return delta, ratio


def _termination(
    d_cost: float,
    cost: float,
    dx_norm: float,
    x_norm: float,
    ratio: float,
    ftol: float,
    xtol: float,
) -> int | None:
    ftol_ok = d_cost < ftol * cost and ratio > 0.25
    xtol_ok = dx_norm < xtol * (xtol + x_norm)
    if ftol_ok and xtol_ok:
        return 4
    if ftol_ok:
        return 2
    if xtol_ok:
        return 3
    return None


def _quadratic_1d(
    jac: FloatArray, g: FloatArray, s: FloatArray, diag: FloatArray, s0: FloatArray | None = None
) -> tuple[float, ...]:
    """Coefficients of ``0.5 (s0 + s t)^T (J^T J + diag) (s0 + s t) + g^T (s0 + s t)``."""
    v = jac.dot(s)
    a = np.dot(v, v)
    a += np.dot(s * diag, s)
    a *= 0.5
    b = np.dot(g, s)
    if s0 is None:
        return a, b
    u = jac.dot(s0)
    b += np.dot(u, v)
    c = 0.5 * np.dot(u, u) + np.dot(g, s0)
    b += np.dot(s0 * diag, s)
    c += 0.5 * np.dot(s0 * diag, s0)
    return a, b, c


def _minimize_1d(a: float, b: float, lb: float, ub: float, c: float = 0) -> tuple[float, float]:
    t: list[float] | FloatArray
    t = [lb, ub]
    if a != 0:
        extremum = -0.5 * b / a
        if lb < extremum < ub:
            t.append(extremum)
    t = np.asarray(t)
    y = t * (a * t + b) + c
    i = np.argmin(y)
    return t[i], y[i]


def _evaluate_quadratic(jac: FloatArray, g: FloatArray, s: FloatArray, diag: FloatArray) -> float:
    jac_s = jac.dot(s)
    q = np.dot(jac_s, jac_s)
    q += np.dot(s * diag, s)
    return cast("float", 0.5 * q + np.dot(s, g))


def _step_to_bound(
    x: FloatArray, s: FloatArray, lb: FloatArray, ub: FloatArray
) -> tuple[float, IntArray]:
    non_zero = np.nonzero(s)
    s_non_zero = s[non_zero]
    steps = np.empty_like(x)
    steps.fill(np.inf)
    with np.errstate(over="ignore"):
        steps[non_zero] = np.maximum(
            (lb - x)[non_zero] / s_non_zero, (ub - x)[non_zero] / s_non_zero
        )
    min_step = cast(float, np.min(steps))  # a float64 (x is float64)
    return min_step, np.equal(steps, min_step) * np.sign(s).astype(int)


def _intersect_trust_region(x: FloatArray, s: FloatArray, delta: float) -> tuple[float, float]:
    a = np.dot(s, s)
    if a == 0:
        raise ValueError("`s` is zero.")
    b = np.dot(x, s)
    c = np.dot(x, x) - delta**2
    if c > 0:
        raise ValueError("`x` is not within the trust region.")
    d = np.sqrt(b * b - a * c)
    q = -(b + math.copysign(d, b))
    t1 = q / a
    t2 = c / q
    return (t1, t2) if t1 < t2 else (t2, t1)


def _select_step(
    x: FloatArray,
    jac_h: FloatArray,
    diag_h: FloatArray,
    g_h: FloatArray,
    p: FloatArray,
    p_h: FloatArray,
    d: FloatArray,
    delta: float,
    lb: FloatArray,
    ub: FloatArray,
    theta: float,
) -> tuple[FloatArray, FloatArray, float]:
    """Return the best of three candidate steps.

    The trust-region step (made strictly interior), its reflection on the first bound it hits, and
    the scaled anti-gradient.
    """
    if ((x + p >= lb) & (x + p <= ub)).all():
        p_value = _evaluate_quadratic(jac_h, g_h, p_h, diag_h)
        return p, p_h, -p_value

    p_stride, hits = _step_to_bound(x, p, lb, ub)
    r_h = np.copy(p_h)
    r_h[hits.astype(bool)] *= -1
    r = d * r_h
    p *= p_stride
    p_h *= p_stride
    x_on_bound = x + p
    _, to_tr = _intersect_trust_region(p_h, r_h, delta)
    to_bound, _ = _step_to_bound(x_on_bound, r, lb, ub)
    r_stride = min(to_bound, to_tr)
    if r_stride > 0:
        r_stride_l = (1 - theta) * p_stride / r_stride
        r_stride_u = theta * to_bound if r_stride == to_bound else to_tr
    else:
        r_stride_l = 0
        r_stride_u = -1
    if r_stride_l <= r_stride_u:
        a, b, c = _quadratic_1d(jac_h, g_h, r_h, diag_h, s0=p_h)
        r_stride, r_value = _minimize_1d(a, b, r_stride_l, r_stride_u, c=c)
        r_h *= r_stride
        r_h += p_h
        r = r_h * d
    else:
        r_value = np.inf

    p *= theta
    p_h *= theta
    p_value = _evaluate_quadratic(jac_h, g_h, p_h, diag_h)

    ag_h = -g_h
    ag = d * ag_h
    to_tr = delta / _norm(ag_h)
    to_bound, _ = _step_to_bound(x, ag, lb, ub)
    ag_stride = theta * to_bound if to_bound < to_tr else to_tr
    a, b = _quadratic_1d(jac_h, g_h, ag_h, diag_h)
    ag_stride, ag_value = _minimize_1d(a, b, 0, ag_stride)
    ag_h *= ag_stride
    ag *= ag_stride

    if p_value < r_value and p_value < ag_value:
        return p, p_h, -p_value
    if r_value < p_value and r_value < ag_value:
        return r, r_h, -r_value
    return ag, ag_h, -ag_value
