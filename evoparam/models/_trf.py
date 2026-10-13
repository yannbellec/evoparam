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

import dataclasses
import functools
import math
from collections.abc import Callable
from typing import cast

import numpy as np
from scipy.linalg.lapack import dgeqrf, dgesdd, dgesdd_lwork, dormqr

from evoparam._typing import ArrayLike, BoolArray, FloatArray, IntArray

__all__ = ["TrfOptions", "least_squares_trf"]

EPS = np.finfo(float).eps

# Radius updates of SciPy's trf: a step whose actual reduction is below a quarter of the
# predicted one shrinks the radius to a quarter of the step; one above three quarters that
# reaches the boundary of the trust region (95 % of its radius) doubles it.
_SHRINK = 0.25
_EXPAND = 0.75
_BOUND_HIT = 0.95
# Lower bound of the fraction of the step to the bounds that keeps an iterate interior.
_THETA_MIN = 0.995


@dataclasses.dataclass(frozen=True)
class TrfOptions:
    """Tolerances and options of :func:`least_squares_trf` (those of ``least_squares``).

    ``m``: the number of residuals of the problem, when ``fun`` and ``jac`` return them in
    a reduced orthonormal basis (fewer rows, the same inner products); it sets the rank
    threshold of the trust-region step as in SciPy. By default the rows of ``jac(x0)``.

    ``final_jacobian=False`` leaves out the Jacobian that SciPy computes at the last point,
    after a step met ``ftol`` or ``xtol``: the solution is the same, but the status is then
    the one of that step (SciPy reports 1 when the gradient at the last point is below
    ``gtol``).
    """

    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8
    max_nfev: int | None = None
    reduced: bool = True
    m: int | None = None
    final_jacobian: bool = True


def least_squares_trf(
    fun: Callable[[FloatArray], ArrayLike],
    jac: Callable[[FloatArray], ArrayLike],
    x0: ArrayLike,
    lb: ArrayLike,
    ub: ArrayLike,
    options: TrfOptions | None = None,
) -> tuple[FloatArray, float, int, int]:
    """Minimise ``0.5 * ||fun(x)||^2`` subject to ``lb <= x <= ub``.

    ``jac(x)`` is called only at a point where ``fun`` was just called (the engine keeps
    the linear solve of that point). Returns ``(x, cost, nfev, status)``, the status codes
    of SciPy (0: max_nfev reached, 1: gtol, 2: ftol, 3: xtol, 4: ftol and xtol). Raises
    ValueError, as ``least_squares`` does, for an infeasible start or non-finite initial
    residuals, and LinAlgError when an SVD does not converge. ``options``: see
    :class:`TrfOptions` (its defaults when None).
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    if np.any(lb >= ub):
        raise ValueError("Each lower bound must be strictly less than each upper bound.")
    if not np.all((x0 >= lb) & (x0 <= ub)):
        raise ValueError("Initial guess is outside of provided bounds")
    solver = _Solver(fun, jac, lb, ub, TrfOptions() if options is None else options)
    return solver.run(_strictly_feasible(x0, lb, ub, 1e-10))


@dataclasses.dataclass
class _Point:
    """An iterate: the point, its residuals, cost and (reduced) Jacobian and gradient."""

    x: FloatArray
    f: FloatArray
    cost: float
    jr: FloatArray
    jf: FloatArray
    g: FloatArray


@dataclasses.dataclass
class _Model:
    """The scaled quadratic model of one iteration and the SVD of its augmented Jacobian."""

    d: FloatArray
    diag_h: FloatArray
    g_h: FloatArray
    j_h: FloatArray
    uf: FloatArray
    s: FloatArray
    v_mat: FloatArray
    theta: float


class _Solver:
    """The iteration of :func:`least_squares_trf`: SciPy's ``trf_bounds`` step by step."""

    def __init__(
        self,
        fun: Callable[[FloatArray], ArrayLike],
        jac: Callable[[FloatArray], ArrayLike],
        lb: FloatArray,
        ub: FloatArray,
        options: TrfOptions,
    ) -> None:
        """Keep the problem; the iteration state is set by :meth:`run`."""
        self.fun = fun
        self.jac = jac
        self.lb = lb
        self.ub = ub
        self.options = options
        self.finite_lb, self.finite_ub = np.isfinite(lb), np.isfinite(ub)
        self.nfev = 0
        self.max_nfev = 0
        self.m = 0
        self.n = 0
        self.delta = 0.0
        self.alpha = 0.0
        self.scale_inv: FloatArray = np.ones(0)
        self.scale: FloatArray = np.ones(0)
        self.augmented: FloatArray = np.empty((0, 0))
        self.f_augmented: FloatArray = np.zeros(0)
        self.lwork = 0

    def run(self, x: FloatArray) -> tuple[FloatArray, float, int, int]:
        """Iterate from the strictly feasible ``x``; returns ``(x, cost, nfev, status)``."""
        point = self._start(x)
        status = None
        while True:
            v, dv = _cl_scaling(point.x, point.g, self.lb, self.ub, self.finite_lb, self.finite_ub)
            g_norm = np.abs(point.g * v).max() if self.n else 0.0
            if g_norm < self.options.gtol:
                status = 1
            if status is not None or self.nfev == self.max_nfev:
                break
            model = self._model(point, v, dv, g_norm)
            accepted, status = self._step(point, model)
            if accepted is None:
                continue
            point = accepted
            if status is not None and not self.options.final_jacobian:
                break
            point = self._jacobian_at(point)
            self.scale_inv = np.maximum((point.jr**2).sum(axis=0) ** 0.5, self.scale_inv)
            self.scale = 1 / self.scale_inv
        return point.x, point.cost, self.nfev, 0 if status is None else status

    def _start(self, x: FloatArray) -> _Point:
        """Evaluate the start, set the scaling, the radius and the work arrays."""
        options = self.options
        f = np.atleast_1d(self.fun(x))
        jac_x = np.atleast_2d(self.jac(x))
        if not np.all(np.isfinite(f)):
            raise ValueError("Residuals are not finite in the initial point.")
        self.m = jac_x.shape[0] if options.m is None else int(options.m)
        n = self.n = jac_x.shape[1]
        self.max_nfev = n * 100 if options.max_nfev is None else options.max_nfev
        self.nfev = 1
        cost = 0.5 * np.dot(f, f)
        jr, jf, g = _reduce(jac_x, f, options.reduced)
        scale_inv = (jr**2).sum(axis=0) ** 0.5
        scale_inv[scale_inv == 0] = 1
        self.scale_inv = scale_inv
        self.scale = 1 / scale_inv

        v, dv = _cl_scaling(x, g, self.lb, self.ub, self.finite_lb, self.finite_ub)
        v = np.where(dv != 0, v * scale_inv, v)
        self.delta = _norm(x * scale_inv / v**0.5)
        if self.delta == 0:
            self.delta = 1.0

        rows = jr.shape[0]
        self.augmented = np.empty((rows + n, n), order="F")
        self.lwork = int(dgesdd_lwork(rows + n, n, compute_uv=1, full_matrices=0)[0])
        self.f_augmented = np.zeros(rows + n)
        return _Point(x, f, cost, jr, jf, g)

    def _jacobian_at(self, point: _Point) -> _Point:
        """Return ``point`` with the Jacobian and gradient at its ``x``."""
        jac_x = np.atleast_2d(self.jac(point.x))
        jr, jf, g = _reduce(jac_x, point.f, self.options.reduced)
        return _Point(point.x, point.f, point.cost, jr, jf, g)

    def _model(self, point: _Point, v: FloatArray, dv: FloatArray, g_norm: float) -> _Model:
        """Return the scaled model at ``point`` and the SVD of its augmented Jacobian."""
        v = np.where(dv != 0, v * self.scale_inv, v)
        d = v**0.5 * self.scale
        diag_h = point.g * dv * self.scale
        g_h = d * point.g

        rows = point.jr.shape[0]
        augmented = self.augmented
        self.f_augmented[:rows] = point.jf
        augmented[:rows] = point.jr * d
        j_h = augmented[:rows]
        augmented[rows:] = np.diag(diag_h**0.5)
        u, s, vt, info = dgesdd(augmented, compute_uv=1, full_matrices=0, lwork=self.lwork)
        if info > 0:
            raise np.linalg.LinAlgError("SVD did not converge")
        uf = u.T.dot(self.f_augmented)
        return _Model(d, diag_h, g_h, j_h, uf, s, vt.T, max(_THETA_MIN, 1 - g_norm))

    def _step(self, point: _Point, model: _Model) -> tuple[_Point | None, int | None]:
        """Shrink the trust region until a step lowers the cost.

        Returns the new point (without its Jacobian) or None, and the termination status.
        """
        lb, ub = self.lb, self.ub
        actual_reduction = -1.0
        status = None
        trial = point
        while actual_reduction <= 0 and self.nfev < self.max_nfev:
            p_h, self.alpha = _solve_trust_region(self.m, model, self.delta, self.alpha)
            p = model.d * p_h
            step, step_h, predicted = _select_step(point.x, model, p, p_h, self.delta, (lb, ub))
            x_new = _strictly_feasible(point.x + step, lb, ub, 0)
            f_new = np.atleast_1d(self.fun(x_new))
            self.nfev += 1
            step_h_norm = _norm(step_h)
            if not np.isfinite(f_new).all():
                self.delta = _SHRINK * step_h_norm
                continue
            cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = point.cost - cost_new
            trial = _Point(x_new, f_new, cost_new, point.jr, point.jf, point.g)
            delta_new, ratio = _update_radius(
                self.delta,
                actual_reduction,
                predicted,
                step_h_norm,
                step_h_norm > _BOUND_HIT * self.delta,
            )
            status = _termination(
                actual_reduction,
                point.cost,
                _norm(step),
                _norm(point.x),
                ratio,
                self.options.ftol,
                self.options.xtol,
            )
            if status is not None:
                break
            self.alpha *= self.delta / delta_new
            self.delta = delta_new
        return (trial if actual_reduction > 0 else None), status


def _norm(x: FloatArray) -> float:
    """Return the Euclidean norm of ``x``."""
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
    # the distances are computed only where they are used (finite bound): an infinite bound
    # elsewhere gives no inf - inf
    v = np.ones_like(x)
    np.subtract(x, lb, out=v, where=lower)
    np.subtract(ub, x, out=v, where=upper & ~lower)
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
    """Return -1, 1 or 0 for each entry of ``x`` at its lower bound, upper bound or neither."""
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
    m: int, model: "_Model", delta: float, alpha: float, rtol: float = 0.01, max_iter: int = 10
) -> tuple[FloatArray, float]:
    """Solve the trust-region subproblem from the SVD of the (scaled) Jacobian (Moré 1977).

    ``m``: the number of residuals. Returns the step and the Levenberg-Marquardt parameter.
    """
    uf, s, v_mat = model.uf, model.s, model.v_mat
    n = len(s)
    suf = s * uf
    full_rank = s[-1] > EPS * m * s[0] if m >= n else False
    if full_rank:
        p = -v_mat.dot(uf / s)
        if _norm(p) <= delta:
            return p, 0.0

    def phi_and_derivative(alpha: float) -> tuple[float, float]:
        """Return ``||p(alpha)|| - delta`` and its derivative in ``alpha``."""
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
    """Return the new trust-region radius and the ratio of actual to predicted reduction."""
    if predicted > 0:
        ratio = actual / predicted
    elif predicted == actual == 0:
        ratio = 1
    else:
        ratio = 0
    if ratio < _SHRINK:
        delta = _SHRINK * step_norm
    elif ratio > _EXPAND and bound_hit:
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
    """Return SciPy's termination status of a step (2: ftol, 3: xtol, 4: both) or None."""
    ftol_ok = d_cost < ftol * cost and ratio > _SHRINK
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
    """Return the minimiser of ``a t^2 + b t + c`` on ``[lb, ub]`` and its value."""
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
    """Return the value of the quadratic model at the step ``s``."""
    jac_s = jac.dot(s)
    q = np.dot(jac_s, jac_s)
    q += np.dot(s * diag, s)
    return cast("float", 0.5 * q + np.dot(s, g))


def _quotient(num: FloatArray, den: FloatArray) -> FloatArray:
    """Return ``num / den`` (``den`` non-zero) without computing an overflow.

    ``±inf`` is written directly where the quotient exceeds twice the largest float, the
    IEEE quotient elsewhere.
    """
    huge = np.abs(num) * 0.5 > np.minimum(np.abs(den), 1.0) * np.finfo(np.float64).max
    out = np.copysign(np.inf, num) * np.sign(den)
    np.divide(num, den, out=out, where=~huge)
    return out


def _step_to_bound(
    x: FloatArray, s: FloatArray, lb: FloatArray, ub: FloatArray
) -> tuple[float, IntArray]:
    """Return the step along ``s`` to the nearest bound and the bounds it reaches (signs)."""
    non_zero = np.nonzero(s)
    s_non_zero = s[non_zero]
    steps = np.empty_like(x)
    steps.fill(np.inf)
    steps[non_zero] = np.maximum(
        _quotient((lb - x)[non_zero], s_non_zero), _quotient((ub - x)[non_zero], s_non_zero)
    )
    min_step = cast(float, np.min(steps))  # a float64 (x is float64)
    return min_step, np.equal(steps, min_step) * np.sign(s).astype(int)


def _intersect_trust_region(x: FloatArray, s: FloatArray, delta: float) -> tuple[float, float]:
    """Return the two roots ``t`` of ``||x + s t|| = delta``, in increasing order."""
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
    model: "_Model",
    p: FloatArray,
    p_h: FloatArray,
    delta: float,
    bounds: tuple[FloatArray, FloatArray],
) -> tuple[FloatArray, FloatArray, float]:
    """Return the best of three candidate steps.

    The trust-region step (made strictly interior), its reflection on the first bound it hits, and
    the scaled anti-gradient.
    """
    jac_h, diag_h, g_h, d, theta = model.j_h, model.diag_h, model.g_h, model.d, model.theta
    lb, ub = bounds
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
