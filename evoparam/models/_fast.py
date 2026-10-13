# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# Derived from SciPy (BSD-3-Clause, named in NOTICE): the trust-region reflective solver
#   (scipy/optimize/_lsq/trf.py and common.py, through evoparam/models/_trf.py) and the
#   rational approximations of erfc of the Cephes library as SciPy distributes them
#   (ndtr.c); changes: written as loops over arrays for numba, together with the design
#   columns, the whitening and the Householder QR of the variable projection
"""Compiled fit of one model structure: the ``fast`` extra (numba), spec §9.2.

:func:`optimise` does what ``engine._optimise`` does with the lean trust-region solver
(``models/_trf.py``) on the reduced residual (``_Structure.reduced``): the same algorithm,
the same steps, the same termination tests, written as loops that numba compiles. The
design columns of the Gaussian and the skewed Gaussian, the exponential background (or a
background without non-linear parameter, whose basis is fixed), the AR whitening and the
QR factorisation of the variable projection are evaluated in compiled code, so that one
evaluation costs no Python call.

Every structure the kernel does not cover (other shapes and backgrounds, a robust loss)
goes through the NumPy path, which stays the reference. A design that is not finite or not
clearly of full rank stops the kernel (status ``FALLBACK``), and the NumPy path, which
takes the SVD route there, fits that structure from the start.

The arithmetic follows the NumPy path operation by operation, except for:

- the exponential (:func:`_vexp`, within one ulp) and the error function
  (:func:`_verfc`, Cephes' rational approximations, which SciPy uses) written so that
  their loops vectorise;
- a component taken as 0 more than ``SUPPORT_Z`` widths from its centre, where it is below
  1e-31 of its peak;
- the QR factorisations, unblocked Householder reflections instead of LAPACK's blocked
  ones, and the SVD of the trust-region step, a QR factorisation followed by the one-sided
  Jacobi method instead of LAPACK's ``gesdd``;
- the Jacobian at the last point, which SciPy computes after the termination test and
  does not use, is not computed;
- fused multiply-adds where LLVM contracts a product and a sum, and inner products summed
  in the order of their vectorised loop (``_REASSOCIATED``), which differ from separate
  roundings and from BLAS ``ddot`` at the level of rounding; the order of each loop is
  fixed at compilation, so the results are the same from run to run on one machine.

Results agree to rounding, and the optima to the solver tolerance
(``tests/models/test_fast_kernel.py``).

numba is imported only by :func:`kernel` (through ``utils.require``), the first time a fit
needs it; without numba the engine uses the NumPy path. The functions are compiled with
``error_model="numpy"``: a division by zero gives an infinity or NaN, as in NumPy, which
the checks of the NumPy path then catch, instead of raising.
"""

import math
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from evoparam._typing import FloatArray, IntArray

__all__ = ["kernel", "OK", "FALLBACK", "VALUE_ERROR", "LINALG_ERROR"]

# status of optimise
OK = 0
FALLBACK = 1  # a design not finite or not clearly of full rank: the NumPy path takes over
VALUE_ERROR = 2  # what least_squares_trf raises as ValueError
LINALG_ERROR = 3  # an SVD that did not converge

# shape kinds
GAUSSIAN = 0
SKEWED_GAUSSIAN = 1
# background kinds
BG_FIXED = 0  # linear coefficients only: a fixed basis (none, constant, linear, ...)
BG_EXPONENTIAL = 1  # c0 + c1 exp(-(t - t_ref) / tau)

# |z| beyond which a Gaussian factor is taken as 0: exp(-z^2/2) < 6e-32 there, and the term it
# would add to a whitened residual is below 1e-25 of the residual's scale even for an
# amplitude of 1e6 noise SDs (rounding is 1e-16); the NumPy path computes it up to 38.7
# (shapes.components.SUPPORT_Z), where it underflows to 0
SUPPORT_Z = 12.0
COND_MAX = 1e11  # Frobenius condition number of R above which the SVD route is taken
EPS = float(np.finfo(float).eps)
_SQRT2 = math.sqrt(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
# below this argument erfc takes the rational approximation of the middle range (Cephes' P/Q),
# above it the one of the far range (R/S)
_ERFC_MID = 8.0
# an ordered parameterisation chains two latencies or more
_CHAIN_MIN = 2
# radius updates and termination of the trust-region solver (models/_trf.py)
_SHRINK = 0.25
_EXPAND = 0.75
_BOUND_HIT = 0.95
_THETA_MIN = 0.995

# The kernels take their arrays grouped in tuples, unpacked on entry (numba compiles a tuple
# of arrays as its members):
# - Model: ``(tw, kinds, offsets, bg_kind, lag, fixed)``, the time axis, the kind and offset
#   of each component, the background (kind, lag from its reference time, fixed basis);
# - Whitening: ``(inv_scale, a, post, stride, start, start_rows)``, see ``_whiten``;
# - Prior: ``(prior_cols, prior_diag, target)``, the prior rows and the whitened target;
# - Problem: ``(model, whitening, prior)``, what ``engine._compiled_args`` returns;
# - ColumnWork, FactorWork, JacobianWork: the work arrays (see ``_workspace``).
Model = tuple[FloatArray, IntArray, IntArray, int, FloatArray, FloatArray]
Whitening = tuple[FloatArray, FloatArray, FloatArray, int, FloatArray, IntArray]
Prior = tuple[IntArray, FloatArray, FloatArray]
Problem = tuple[Model, Whitening, Prior]
ColumnWork = tuple[FloatArray, FloatArray, FloatArray, FloatArray, IntArray, FloatArray, IntArray]
FactorWork = tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]
JacobianWork = tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]
Workspace = tuple[ColumnWork, FactorWork, JacobianWork]
Scaled = tuple[FloatArray, FloatArray, FloatArray, FloatArray]
Environment = tuple[Problem, Workspace, IntArray, FloatArray, FloatArray, FloatArray, float, int]


###################################################################################################
## Shapes


def _skew_root(a: float) -> float:
    """``shapes.components._skew_root_scalar`` (the mode of the skewed Gaussian, ``a >= 0``)."""
    low, high = 0.0, 2.0
    z = min(max(0.6 * a / math.sqrt(1.0 + a * a), 0.0), 2.0)
    eps = 4.0 * EPS
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


def _skew_mode(alpha: float) -> tuple[float, float]:
    """``c`` and ``dc / dalpha`` of ``shapes.components.skew_mode`` for one skewness."""
    a = abs(alpha)
    z = _skew_root(a)
    c = math.exp(-0.5 * (z * z)) * math.erfc(-a * z / _SQRT2)
    sign = 1.0 if alpha > 0 else (-1.0 if alpha < 0 else 0.0)
    dc = _SQRT_2_OVER_PI * (sign * z) * math.exp(-0.5 * (z * z) * (1.0 + alpha * alpha))
    return c, dc


# exp: Cody-Waite reduction x = k ln2 + r, |r| <= ln2 / 2, the Taylor polynomial of degree 13
# (truncation 4e-18) and 2^k from its bits: within 1 ulp (0.97 at most against 0.61 for
# NumPy's on 4 000 arguments in [-745, 0], measured with mpmath), and the loops vectorise
_LOG2E = 1.4426950408889634
_LN2_HI = 6.93147180369123816490e-01
_LN2_LO = 1.90821492927058770002e-10
_EXP_LOW = -708.3964185322641  # ln(2^-1022): below, the result is taken as 0
_EXP_TAYLOR = tuple(1.0 / math.factorial(k) for k in range(14))
_E2, _E3, _E4, _E5, _E6, _E7, _E8, _E9, _E10, _E11, _E12, _E13 = _EXP_TAYLOR[2:]

# erfc: the rational approximations of Cephes (ndtr.c, Moshier), which SciPy's erfc uses
_ERF_T = (
    9.60497373987051638749e0,
    9.00260197203842689217e1,
    2.23200534594684319226e3,
    7.00332514112805075473e3,
    5.55923013010394962768e4,
)
_ERF_U = (
    3.35617141647503099647e1,
    5.21357949780152679795e2,
    4.59432382970980127987e3,
    2.26290000613890934246e4,
    4.92673942608635921086e4,
)
_ERFC_P = (
    2.46196981473530512524e-10,
    5.64189564831068821977e-1,
    7.46321056442269912687e0,
    4.86371970985681366614e1,
    1.96520832956077098242e2,
    5.26445194995477358631e2,
    9.34528527171957607540e2,
    1.02755188689515710272e3,
    5.57535335369399327526e2,
)
_ERFC_Q = (
    1.32281951154744992508e1,
    8.67072140885989742329e1,
    3.54937778887819891062e2,
    9.75708501743205489753e2,
    1.82390916687909736289e3,
    2.24633760818710981792e3,
    1.65666309194161350182e3,
    5.57535340817727675546e2,
)
_ERFC_R = (
    5.64189583547755073984e-1,
    1.27536670759978104416e0,
    5.01905042251180477414e0,
    6.16021097993053585195e0,
    7.40974269950448939160e0,
    2.97886665372100240670e0,
)
_ERFC_S = (
    2.26052863220117276590e0,
    9.39603524938001434673e0,
    1.20489539808096656605e1,
    1.70814450747565897222e1,
    9.60896809063285878198e0,
    3.36907645100081516050e0,
)
_T0, _T1, _T2, _T3, _T4 = _ERF_T
_U0, _U1, _U2, _U3, _U4 = _ERF_U
_P0, _P1, _P2, _P3, _P4, _P5, _P6, _P7, _P8 = _ERFC_P
_Q0, _Q1, _Q2, _Q3, _Q4, _Q5, _Q6, _Q7 = _ERFC_Q
_R0, _R1, _R2, _R3, _R4, _R5 = _ERFC_R
_S0, _S1, _S2, _S3, _S4, _S5 = _ERFC_S


def _vexp(x: FloatArray, out: FloatArray, bits: IntArray) -> None:
    """``out[i] = exp(x[i])``; 0 below ``_EXP_LOW``, NaN for NaN.

    For ``x <= 709`` (2^k must be a normal number): every argument here is at most 0.
    ``out`` must not be ``x``: LLVM vectorises the loop only for distinct arrays.
    """
    n = x.shape[0]
    for i in range(n):
        xi = x[i]
        low = xi < _EXP_LOW
        xi = _EXP_LOW if low else xi
        k = math.floor(xi * _LOG2E + 0.5)
        r = (xi - k * _LN2_HI) - k * _LN2_LO
        p = _E13
        p = p * r + _E12
        p = p * r + _E11
        p = p * r + _E10
        p = p * r + _E9
        p = p * r + _E8
        p = p * r + _E7
        p = p * r + _E6
        p = p * r + _E5
        p = p * r + _E4
        p = p * r + _E3
        p = p * r + _E2
        p = p * r + 1.0
        p = p * r + 1.0
        out[i] = 0.0 if low else p
        bits[i] = (np.int64(k) + 1023) << 52
    scale = bits[:n].view(np.float64)
    for i in range(n):
        out[i] *= scale[i]


def _verfc(u: FloatArray, out: FloatArray, tmp: FloatArray, bits: IntArray) -> None:
    """``out[i] = erfc(u[i])`` as Cephes computes it (``tmp``: work space of the same length).

    The three branches are evaluated for every entry and the right one selected, which
    lets the loop vectorise.
    """
    n = u.shape[0]
    for i in range(n):
        out[i] = -u[i] * u[i]
    _vexp(out, tmp, bits)
    for i in range(n):
        a = u[i]
        x = abs(a)
        z = x * x
        num = (((_T0 * z + _T1) * z + _T2) * z + _T3) * z + _T4
        den = ((((z + _U0) * z + _U1) * z + _U2) * z + _U3) * z + _U4
        small = 1.0 - a * num / den
        p_mid = (((((((_P0 * x + _P1) * x + _P2) * x + _P3) * x + _P4) * x + _P5) * x + _P6) * x
                 + _P7) * x + _P8  # fmt: skip
        q_mid = (((((((x + _Q0) * x + _Q1) * x + _Q2) * x + _Q3) * x + _Q4) * x + _Q5) * x
                 + _Q6) * x + _Q7  # fmt: skip
        p_far = ((((_R0 * x + _R1) * x + _R2) * x + _R3) * x + _R4) * x + _R5
        q_far = (((((x + _S0) * x + _S1) * x + _S2) * x + _S3) * x + _S4) * x + _S5
        mid = x < _ERFC_MID
        y = tmp[i] * (p_mid if mid else p_far) / (q_mid if mid else q_far)
        y = 2.0 - y if a < 0 else y
        out[i] = small if x < 1.0 else y


def _span(tw: FloatArray, centre: float, left: float, right: float) -> tuple[int, int]:
    """Return the indices ``[i0, i1)`` of the samples in ``[centre - left, centre + right]``.

    One sample of margin on each side; the component is taken as 0 outside.
    """
    n = tw.shape[0]
    i0 = max(int(np.searchsorted(tw, centre - left)) - 1, 0)
    i1 = min(int(np.searchsorted(tw, centre + right)) + 1, n)
    return i0, max(i1, i0)


def _zero(x: FloatArray) -> None:
    """Set every entry of the vector ``x`` to 0."""
    for i in range(x.shape[0]):
        x[i] = 0.0


def _gaussian_column(
    theta: FloatArray, o: int, tw: FloatArray, row: FloatArray, buf: FloatArray, bits: IntArray
) -> tuple[int, int]:
    """Column of the Gaussian of parameters ``theta[o:o + 2]`` on its span; returns the span."""
    mu, sigma = theta[o], theta[o + 1]
    reach = SUPPORT_Z * sigma
    i0, i1 = _span(tw, mu, reach, reach)
    t = tw[i0:i1]
    arg = buf[0, : i1 - i0]
    inv_sigma = 1.0 / sigma
    for i in range(t.shape[0]):
        z = (t[i] - mu) * inv_sigma
        arg[i] = -0.5 * (z * z)
    _vexp(arg, row[i0:i1], bits)
    return i0, i1


def _skewed_column(
    theta: FloatArray, k: int, o: int, tw: FloatArray, cols: ColumnWork
) -> tuple[int, int]:
    """Column ``k``, a skewed Gaussian of parameters ``theta[o:o + 3]``; returns its span.

    Keeps the numerator, the fast-decaying factor and ``(c, dc)`` in ``cols`` (see
    :func:`_columns`).
    """
    design, numer, fast, skew, _, buf, bits = cols
    xi, omega, alpha = theta[o], theta[o + 1], theta[o + 2]
    c, dc = _skew_mode(alpha)
    skew[k, 0] = c
    skew[k, 1] = dc
    inv_c = 1.0 / c
    reach = SUPPORT_Z * omega
    short = reach / math.sqrt(1.0 + alpha * alpha)
    i0, i1 = _span(tw, xi, short if alpha > 0 else reach, reach if alpha > 0 else short)
    m = i1 - i0
    t = tw[i0:i1]
    arg, u, g, e = buf[0, :m], buf[1, :m], buf[2, :m], buf[3, :m]
    inv_omega = 1.0 / omega
    for i in range(m):
        z = (t[i] - xi) * inv_omega
        arg[i] = -0.5 * (z * z)
        u[i] = -alpha * z / _SQRT2
    _vexp(arg, g, bits)
    _verfc(u, e, arg, bits)  # leaves exp(-u^2) in arg
    numer_k, fast_k = numer[k, i0:i1], fast[k, i0:i1]
    part = design[k, i0:i1]
    for i in range(m):
        value = g[i] * e[i]
        numer_k[i] = value
        part[i] = value * inv_c
        fast_k[i] = g[i] * arg[i]
    return i0, i1


def _columns(theta: FloatArray, model: Model, cols: ColumnWork) -> None:
    """Unwhitened design (columns as rows of ``design``), as ``_columns.component``.

    ``cols`` is ``(design, numer, fast, skew, spans, buf, bits)``: ``numer`` keeps the
    numerator of each skewed Gaussian, ``fast`` its fast-decaying factor
    ``exp(-(1 + alpha^2) z^2 / 2)`` (for :func:`_derivatives`), ``skew`` its ``(c, dc)`` and
    ``spans`` the samples outside which each component is taken as 0 (``SUPPORT_Z``); ``buf``
    (4 x n) and ``bits`` are work space.

    A skewed Gaussian is ``exp(-z^2 / 2) erfc(u)`` with ``u = -alpha z / sqrt(2)``; on the
    side where ``u >= 0``, ``erfc(u) <= exp(-u^2)`` bounds it by
    ``exp(-(1 + alpha^2) z^2 / 2)``, below ``exp(-SUPPORT_Z^2 / 2)`` beyond
    ``SUPPORT_Z / sqrt(1 + alpha^2)`` widths: the span stops there on that side.
    """
    tw, kinds, offsets, bg_kind, lag, fixed = model
    design, spans, buf, bits = cols[0], cols[4], cols[5], cols[6]
    n = tw.shape[0]
    n_comp = kinds.shape[0]
    for k in range(n_comp):
        if kinds[k] == GAUSSIAN:
            i0, i1 = _gaussian_column(theta, offsets[k], tw, design[k], buf, bits)
        else:
            i0, i1 = _skewed_column(theta, k, offsets[k], tw, cols)
        _zero(design[k, :i0])
        _zero(design[k, i1:n])
        spans[k, 0] = i0
        spans[k, 1] = i1
    if bg_kind == BG_EXPONENTIAL:
        tau = theta[offsets[n_comp]]
        inv_tau = 1.0 / tau
        ones, decay = design[n_comp], design[n_comp + 1]
        arg = buf[0, :n]
        for i in range(n):
            ones[i] = 1.0
            arg[i] = -lag[i] * inv_tau
        _vexp(arg, decay, bits)
    else:
        for j in range(fixed.shape[0]):
            src, dst = fixed[j], design[n_comp + j]
            for i in range(n):
                dst[i] = src[i]


def _derivatives(
    theta: FloatArray, beta: FloatArray, model: Model, cols: ColumnWork, out: FloatArray
) -> None:
    """``beta_k`` times the derivatives of each column (rows of ``out``).

    As ``_columns.component_derivatives`` and ``_columns.background_derivatives``; ``out``
    is 0 outside the spans of :func:`_columns`.
    """
    tw, kinds, offsets, bg_kind, lag, _ = model
    design, numer, fast, skew, spans = cols[0], cols[1], cols[2], cols[3], cols[4]
    n_comp = kinds.shape[0]
    length = out.shape[1]
    for k in range(n_comp):
        o = offsets[k]
        b = beta[k]
        i0, i1 = spans[k, 0], spans[k, 1]
        m = i1 - i0
        t = tw[i0:i1]
        for q in range(o, offsets[k + 1]):
            _zero(out[q, :i0])
            _zero(out[q, i1:length])
        if kinds[k] == GAUSSIAN:
            mu, sigma = theta[o], theta[o + 1]
            factor = b / sigma
            inv_sigma = 1.0 / sigma
            row = design[k, i0:i1]
            d_mu, d_sigma = out[o, i0:i1], out[o + 1, i0:i1]
            for i in range(m):
                z = (t[i] - mu) * inv_sigma
                sz = row[i] * z * factor
                d_mu[i] = sz
                d_sigma[i] = sz * z
        else:
            xi, omega, alpha = theta[o], theta[o + 1], theta[o + 2]
            c, dc = skew[k, 0], skew[k, 1]
            factor = -b / (omega * c)
            factor_alpha = b / c
            factor_c = b * dc / (c * c)
            numer_k, gauss = numer[k, i0:i1], fast[k, i0:i1]
            inv_omega = 1.0 / omega
            d_xi, d_omega, d_alpha = out[o, i0:i1], out[o + 1, i0:i1], out[o + 2, i0:i1]
            for i in range(m):
                z = (t[i] - xi) * inv_omega
                numerator = numer_k[i]
                g = _SQRT_2_OVER_PI * gauss[i]
                d_z = alpha * g - z * numerator
                d_xi[i] = d_z * factor
                d_omega[i] = d_z * z * factor
                d_alpha[i] = z * g * factor_alpha - numerator * factor_c
    if bg_kind == BG_EXPONENTIAL:
        o = offsets[n_comp]
        tau = theta[o]
        factor = beta[n_comp + 1] / (tau * tau)
        decay = design[n_comp + 1]
        row = out[o]
        for i in range(lag.shape[0]):
            row[i] = decay[i] * lag[i] * factor
        _zero(out[o, lag.shape[0] : length])


def _whiten(
    src: FloatArray, rows: FloatArray, count: int, n: int, white: Whitening, raw: FloatArray
) -> None:
    """``Whitener.into`` (or ``StridedWhitener``) of ``count`` rows of ``src`` into ``rows``.

    The first ``n`` entries of each row; ``rows`` is another array than ``src``, since LLVM
    vectorises the loops only for distinct arrays.

    ``white`` is ``(inv_scale, a, post, stride, start, start_rows)``: ``inv_scale`` is
    ``1 / scale`` and ``post`` the row weights divided by the innovation SD. The terms of the
    filter are added in the order of the NumPy path (lag 0, 1, ..., p).
    """
    inv_scale, a, post, stride, start, start_rows = white
    p = a.shape[0] - 1
    has_start = start.shape[0] > 0
    a0 = a[0]
    if p == 0 and not has_start:
        for j in range(count):
            row, values = rows[j], src[j]
            for i in range(n):
                row[i] = a0 * (values[i] * inv_scale[i]) * post[i]
        return
    for j in range(count):
        row, values = rows[j], src[j]
        for i in range(n):
            value = values[i] * inv_scale[i]
            raw[i] = value
            row[i] = a0 * value
        for k in range(1, p + 1):
            shift = k * stride
            if shift >= n:
                break
            ak = a[k]
            dst = row[shift:n]
            lagged = raw[: n - shift]
            for i in range(dst.shape[0]):
                dst[i] += ak * lagged[i]
        for i in range(n):
            row[i] *= post[i]
        if has_start:
            _start_rows(row, raw, start, start_rows, p)


def _start_rows(
    row: FloatArray, raw: FloatArray, start: FloatArray, start_rows: IntArray, p: int
) -> None:
    """Write the ``p`` start rows of the exact likelihood (``start`` times the raw samples)."""
    for r in range(p):
        total = 0.0
        for q in range(p):
            total += start[r, q] * raw[start_rows[q]]
        row[start_rows[r]] = total


###################################################################################################
## Householder QR (columns stored as rows: ``mat[j]`` is column j)


def _dot(x: FloatArray, y: FloatArray) -> float:
    """Return the inner product of ``x`` and ``y``.

    Compiled with reassociation allowed (``_REASSOCIATED``), so that LLVM vectorises the
    sum: faster than a BLAS call on vectors of up to a few thousand entries.
    """
    total = 0.0
    for i in range(x.shape[0]):
        total += x[i] * y[i]
    return total


def _householder(mat: FloatArray, n_cols: int, off: int, length: int, tau: FloatArray) -> None:
    """QR factorisation in place, as LAPACK's dgeqr2 (unblocked): R above, reflectors below.

    Column ``j`` of the matrix is ``mat[j, off:length]``. The loops run over views from
    index 0, which numba vectorises.
    """
    for j in range(n_cols):
        alpha = mat[j, off + j]
        tail = mat[j, off + j + 1 : length]
        ss = _dot(tail, tail)
        if ss == 0.0:
            tau[j] = 0.0
            continue
        beta = -math.copysign(math.sqrt(alpha * alpha + ss), alpha)
        tau[j] = (beta - alpha) / beta
        factor = 1.0 / (alpha - beta)
        for i in range(tail.shape[0]):
            tail[i] *= factor
        mat[j, off + j] = beta
        for k in range(j + 1, n_cols):
            other = mat[k, off + j + 1 : length]
            w = (mat[k, off + j] + _dot(tail, other)) * tau[j]
            mat[k, off + j] -= w
            for i in range(other.shape[0]):
                other[i] -= w * tail[i]


def _apply_qt(
    mat: FloatArray, n_cols: int, off: int, length: int, tau: FloatArray, vec: FloatArray
) -> None:
    """``vec <- Q^T vec`` with the reflectors of :func:`_householder`.

    ``vec[i]`` is row ``off + i``.
    """
    rows = length - off
    for j in range(n_cols):
        if tau[j] == 0.0:
            continue
        tail = mat[j, off + j + 1 : length]
        rest = vec[j + 1 : rows]
        w = (vec[j] + _dot(tail, rest)) * tau[j]
        vec[j] -= w
        for i in range(rest.shape[0]):
            rest[i] -= w * tail[i]


###################################################################################################
## Variable projection


def _evaluate(theta: FloatArray, problem: Problem, cols: ColumnWork, fact: FactorWork) -> bool:
    """``_Structure.reduced``: QR of the whitened design; False for the SVD route.

    ``fact`` is ``(qr, tau, qtb, beta, rinv, raw)``. Leaves ``Q^T b`` in ``qtb`` (the residual
    is ``qtb[c:]``) and the linear solution in ``beta``.
    """
    model, white, prior = problem
    prior_cols, prior_diag, target = prior
    qr, tau, qtb, beta, rinv, raw = fact
    n = model[0].shape[0]
    c = qr.shape[0]
    length = qr.shape[1]
    _columns(theta, model, cols)
    _whiten(cols[0], qr, c, n, white, raw)
    for j in range(c):
        _zero(qr[j, n:length])
    for r in range(prior_cols.shape[0]):
        qr[prior_cols[r], n + r] = prior_diag[r]
    if not _all_finite(qr):
        return False
    _householder(qr, c, 0, length, tau)
    if not _well_conditioned(qr, rinv):
        return False
    for i in range(length):
        qtb[i] = target[i]
    _apply_qt(qr, c, 0, length, tau, qtb)
    # R beta = (Q^T b)[:c] (dtrtrs)
    for i in range(c - 1, -1, -1):
        total = qtb[i]
        for k in range(i + 1, c):
            total -= qr[k, i] * beta[k]
        beta[i] = total / qr[i, i]
    return True


def _all_finite(mat: FloatArray) -> bool:
    """Tell whether every entry of the matrix ``mat`` is finite."""
    for j in range(mat.shape[0]):
        for i in range(mat.shape[1]):
            if not math.isfinite(mat[j, i]):
                return False
    return True


def _well_conditioned(qr: FloatArray, rinv: FloatArray) -> bool:
    """Invert ``R`` (dtrtri) into ``rinv``; False for a zero pivot or a large condition number.

    The Frobenius condition number of ``R`` against ``COND_MAX``.
    """
    c = qr.shape[0]
    norm_r = 0.0
    for j in range(c):
        if qr[j, j] == 0.0:
            return False
        for i in range(j + 1):
            norm_r += qr[j, i] * qr[j, i]
    norm_inv = 0.0
    for j in range(c):
        rinv[j, j] = 1.0 / qr[j, j]
        for i in range(j - 1, -1, -1):
            total = 0.0
            for k in range(i + 1, j + 1):
                total += qr[k, i] * rinv[k, j]
            rinv[i, j] = -total / qr[i, i]
        for i in range(j + 1):
            norm_inv += rinv[i, j] * rinv[i, j]
    return math.isfinite(norm_inv) and math.sqrt(norm_r * norm_inv) < COND_MAX


###################################################################################################
## Trust-region reflective solver (models/_trf.py, after SciPy)


def _norm(x: FloatArray) -> float:
    """Return the Euclidean norm of ``x``."""
    total = 0.0
    for i in range(x.shape[0]):
        total += x[i] * x[i]
    return math.sqrt(total)


def _cl_scaling(
    x: FloatArray, g: FloatArray, lb: FloatArray, ub: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return the Coleman-Li scaling vector ``v`` and its derivative ``dv``."""
    n = x.shape[0]
    v = np.ones(n)
    dv = np.zeros(n)
    for i in range(n):
        if g[i] > 0 and math.isfinite(lb[i]):
            v[i] = x[i] - lb[i]
            dv[i] = 1.0
        elif g[i] < 0 and math.isfinite(ub[i]):
            v[i] = ub[i] - x[i]
            dv[i] = -1.0
    return v, dv


def _strictly_feasible(x: FloatArray, lb: FloatArray, ub: FloatArray, rstep: float) -> FloatArray:
    """Return ``x`` moved inside the bounds by ``rstep`` (relative; ``0``: one ulp)."""
    x_new: FloatArray = x.copy()
    n = x.shape[0]
    if rstep == 0:
        for i in range(n):
            if x[i] <= lb[i]:
                x_new[i] = np.nextafter(lb[i], ub[i])
            elif x[i] >= ub[i]:
                x_new[i] = np.nextafter(ub[i], lb[i])
    else:
        for i in range(n):
            lower_dist = x[i] - lb[i]
            upper_dist = ub[i] - x[i]
            lower_threshold = rstep * max(1.0, abs(lb[i]))
            upper_threshold = rstep * max(1.0, abs(ub[i]))
            active = 0
            if math.isfinite(lb[i]) and lower_dist <= min(upper_dist, lower_threshold):
                active = -1
            if math.isfinite(ub[i]) and upper_dist <= min(lower_dist, upper_threshold):
                active = 1
            if active == -1:
                x_new[i] = lb[i] + rstep * max(1.0, abs(lb[i]))
            elif active == 1:
                x_new[i] = ub[i] - rstep * max(1.0, abs(ub[i]))
    for i in range(n):
        if x_new[i] < lb[i] or x_new[i] > ub[i]:
            x_new[i] = 0.5 * (lb[i] + ub[i])
    return x_new


def _phi_and_derivative(
    alpha: float, suf: FloatArray, s: FloatArray, delta: float
) -> tuple[float, float]:
    """Return ``||p(alpha)|| - delta`` and its derivative in ``alpha`` (Moré 1977)."""
    denom = s**2 + alpha
    p_norm = _norm(suf / denom)
    return p_norm - delta, -(suf**2 / denom**3).sum() / p_norm


def _solve_trust_region(
    n: int,
    m: int,
    uf: FloatArray,
    s: FloatArray,
    v_mat: FloatArray,
    delta: float,
    alpha: float,
) -> tuple[FloatArray, float]:
    """Solve the trust-region subproblem from the SVD (``_trf._solve_trust_region``).

    Returns the step and the Levenberg-Marquardt parameter.
    """
    rtol = 0.01
    suf = s * uf
    full_rank = s[-1] > EPS * m * s[0] if m >= n else False
    if full_rank:
        p = -v_mat.dot(uf / s)
        if _norm(p) <= delta:
            return p, 0.0
    alpha_upper = _norm(suf) / delta
    if full_rank:
        phi, phi_prime = _phi_and_derivative(0.0, suf, s, delta)
        alpha_lower = -phi / phi_prime
    else:
        alpha_lower = 0.0
    if not full_rank and alpha == 0:
        alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper) ** 0.5)
    for _ in range(10):
        if alpha < alpha_lower or alpha > alpha_upper:
            alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper) ** 0.5)
        phi, phi_prime = _phi_and_derivative(alpha, suf, s, delta)
        if phi < 0:
            alpha_upper = alpha
        ratio = phi / phi_prime
        alpha_lower = max(alpha_lower, alpha - ratio)
        alpha -= (phi + delta) * ratio / delta
        if abs(phi) < rtol * delta:
            break
    p = -v_mat.dot(suf / (s**2 + alpha))
    p *= delta / _norm(p)
    return p, alpha


def _quadratic_1d(
    jac: FloatArray, g: FloatArray, s: FloatArray, diag: FloatArray, s0: FloatArray
) -> tuple[float, float, float]:
    """Return the coefficients ``a, b, c`` of the model along ``s`` from ``s0``."""
    v = jac.dot(s)
    a = _dot(v, v)
    a += _dot(s * diag, s)
    a *= 0.5
    b = _dot(g, s)
    u = jac.dot(s0)
    b += _dot(u, v)
    c = 0.5 * _dot(u, u) + _dot(g, s0)
    b += _dot(s0 * diag, s)
    c += 0.5 * _dot(s0 * diag, s0)
    return a, b, c


def _quadratic_1d_ab(
    jac: FloatArray, g: FloatArray, s: FloatArray, diag: FloatArray
) -> tuple[float, float]:
    """Return the coefficients ``a, b`` of the model along ``s`` from 0."""
    v = jac.dot(s)
    a = _dot(v, v)
    a += _dot(s * diag, s)
    a *= 0.5
    b = _dot(g, s)
    return a, b


def _minimize_1d(a: float, b: float, lb: float, ub: float, c: float) -> tuple[float, float]:
    """Return the minimiser of ``a t^2 + b t + c`` on ``[lb, ub]`` and its value."""
    best_t = lb
    best_y = lb * (a * lb + b) + c
    y = ub * (a * ub + b) + c
    if y < best_y:
        best_t, best_y = ub, y
    if a != 0:
        extremum = -0.5 * b / a
        if lb < extremum < ub:
            y = extremum * (a * extremum + b) + c
            if y < best_y:
                best_t, best_y = extremum, y
    return best_t, best_y


def _evaluate_quadratic(jac: FloatArray, g: FloatArray, s: FloatArray, diag: FloatArray) -> float:
    """Return the value of the quadratic model at the step ``s`` (``_trf._evaluate_quadratic``)."""
    jac_s = jac.dot(s)
    q = _dot(jac_s, jac_s)
    q += _dot(s * diag, s)
    return 0.5 * q + _dot(s, g)


def _step_to_bound(
    x: FloatArray, s: FloatArray, lb: FloatArray, ub: FloatArray
) -> tuple[float, IntArray]:
    """Return the step along ``s`` to the nearest bound and the bounds it reaches (signs)."""
    n = x.shape[0]
    steps = np.full(n, np.inf)
    for i in range(n):
        if s[i] != 0:
            steps[i] = max((lb[i] - x[i]) / s[i], (ub[i] - x[i]) / s[i])
    min_step = steps.min()
    hits = np.zeros(n, np.int64)
    for i in range(n):
        if steps[i] == min_step:
            hits[i] = 1 if s[i] > 0 else (-1 if s[i] < 0 else 0)
    return min_step, hits


def _intersect_trust_region(x: FloatArray, s: FloatArray, delta: float) -> tuple[float, bool]:
    """Larger root of ``||x + s t|| = delta``; False where least_squares raises ValueError."""
    a = _dot(s, s)
    if a == 0:
        return 0.0, False
    b = _dot(x, s)
    c = _dot(x, x) - delta**2
    if c > 0:
        return 0.0, False
    d = math.sqrt(b * b - a * c)
    q = -(b + math.copysign(d, b))
    t1 = q / a
    t2 = c / q
    return max(t1, t2), True


def _select_step(
    x: FloatArray,
    scaled: Scaled,
    p: FloatArray,
    p_h: FloatArray,
    delta: float,
    lb: FloatArray,
    ub: FloatArray,
    theta: float,
) -> tuple[FloatArray, FloatArray, float, bool]:
    """``_trf._select_step``; ``scaled`` is ``(jac_h, diag_h, g_h, d)``.

    False where ``least_squares`` raises ValueError.
    """
    jac_h, diag_h, g_h, d = scaled
    inside = True
    for i in range(x.shape[0]):
        if not (x[i] + p[i] >= lb[i] and x[i] + p[i] <= ub[i]):
            inside = False
            break
    if inside:
        p_value = _evaluate_quadratic(jac_h, g_h, p_h, diag_h)
        return p, p_h, -p_value, True

    p_stride, hits = _step_to_bound(x, p, lb, ub)
    r_h = p_h.copy()
    for i in range(r_h.shape[0]):
        if hits[i] != 0:
            r_h[i] *= -1
    r = d * r_h
    p = p * p_stride
    p_h = p_h * p_stride
    x_on_bound = x + p
    to_tr, ok = _intersect_trust_region(p_h, r_h, delta)
    if not ok:
        return p, p_h, 0.0, False
    to_bound, _ = _step_to_bound(x_on_bound, r, lb, ub)
    r_stride = min(to_bound, to_tr)
    if r_stride > 0:
        r_stride_l = (1 - theta) * p_stride / r_stride
        r_stride_u = theta * to_bound if r_stride == to_bound else to_tr
    else:
        r_stride_l = 0.0
        r_stride_u = -1.0
    if r_stride_l <= r_stride_u:
        a, b, c = _quadratic_1d(jac_h, g_h, r_h, diag_h, p_h)
        r_stride, r_value = _minimize_1d(a, b, r_stride_l, r_stride_u, c)
        r_h = r_h * r_stride + p_h
        r = r_h * d
    else:
        r_value = np.inf

    p = p * theta
    p_h = p_h * theta
    p_value = _evaluate_quadratic(jac_h, g_h, p_h, diag_h)

    ag_h = -g_h
    ag = d * ag_h
    to_tr = delta / _norm(ag_h)
    to_bound, _ = _step_to_bound(x, ag, lb, ub)
    ag_stride = theta * to_bound if to_bound < to_tr else to_tr
    a, b = _quadratic_1d_ab(jac_h, g_h, ag_h, diag_h)
    ag_stride, ag_value = _minimize_1d(a, b, 0.0, ag_stride, 0.0)
    ag_h = ag_h * ag_stride
    ag = ag * ag_stride

    if p_value < r_value and p_value < ag_value:
        return p, p_h, -p_value, True
    if r_value < p_value and r_value < ag_value:
        return r, r_h, -r_value, True
    return ag, ag_h, -ag_value, True


def _svd_stacked(
    top: FloatArray, diag: FloatArray, f_top: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, bool]:
    """SVD of ``A = [top; diag(diag)]`` (``(r + n) x n``) for the trust-region step.

    Returns ``s`` (descending), ``V`` (the right singular vectors as columns), ``U^T b`` for
    ``b = [f_top; 0]`` (what ``_trf`` takes from LAPACK's ``gesdd``) and False when the
    iteration did not converge (non-finite entries), where ``gesdd`` fails. ``A = Q R`` by
    Householder reflections, then the one-sided Jacobi SVD (Hestenes) of ``R``: accurate for
    the small singular values too, and faster than ``gesdd`` on matrices of a few columns.
    """
    n = top.shape[1]
    b, rhs = _stacked_qr(top, diag, f_top)
    v = np.zeros((n, n))
    for j in range(n):
        v[j, j] = 1.0
    converged = _jacobi_sweeps(b, v)
    norms = np.empty(n)
    for j in range(n):
        norms[j] = math.sqrt(_dot(b[j], b[j]))
    order = np.argsort(-norms, kind="mergesort")
    s = np.empty(n)
    v_mat = np.empty((n, n))
    uf = np.zeros(n)
    for k in range(n):
        j = order[k]
        s[k] = norms[j]
        for i in range(n):
            v_mat[i, k] = v[j, i]
        if norms[j] > 0:
            uf[k] = _dot(b[j], rhs[:n]) / norms[j]
    for k in range(n):
        if not math.isfinite(s[k]):
            converged = False
    return s, v_mat, uf, converged


def _stacked_qr(
    top: FloatArray, diag: FloatArray, f_top: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return ``R`` of ``[top; diag(diag)] = Q R`` (column j as row j) and ``Q^T [f_top; 0]``."""
    r, n = top.shape
    m = r + n
    cols = np.zeros((n, m))  # column j of A as row j
    for j in range(n):
        for i in range(r):
            cols[j, i] = top[i, j]
        cols[j, r + j] = diag[j]
    rhs = np.zeros(m)
    for i in range(r):
        rhs[i] = f_top[i]
    tau = np.zeros(n)
    _householder(cols, n, 0, m, tau)
    _apply_qt(cols, n, 0, m, tau, rhs)
    # B = R (n x n, column j as row j)
    b = np.zeros((n, n))
    for j in range(n):
        for i in range(min(j + 1, m)):
            b[j, i] = cols[j, i]
    return b, rhs


def _jacobi_sweeps(b: FloatArray, v: FloatArray) -> bool:
    """Orthogonalise the rows of ``b`` by Jacobi rotations, accumulated in ``v``.

    At most 60 sweeps; False when a sweep still rotated after the last one.
    """
    n = b.shape[0]
    # squared column norms: exact at the start of each sweep, updated after each rotation
    # (||b_p||^2 - t gamma, ||b_q||^2 + t gamma) within it. Rows are indexed in place: a
    # view per pair costs more than its few products
    sq = np.empty(n)
    for _ in range(60):
        for j in range(n):
            total = 0.0
            for i in range(n):
                total += b[j, i] * b[j, i]
            sq[j] = total
        rotated = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                rotated = _rotate(b, v, sq, p, q) or rotated
        if not rotated:
            return True
    return False


def _rotate(b: FloatArray, v: FloatArray, sq: FloatArray, p: int, q: int) -> bool:
    """Rotate rows ``p`` and ``q`` (Jacobi); False when they are already orthogonal."""
    n = b.shape[0]
    gamma = 0.0
    for i in range(n):
        gamma += b[p, i] * b[q, i]
    norm_p, norm_q = sq[p], sq[q]
    if gamma == 0.0 or abs(gamma) <= EPS * math.sqrt(norm_p * norm_q):
        return False
    zeta = (norm_q - norm_p) / (2.0 * gamma)
    t = math.copysign(1.0, zeta) / (abs(zeta) + math.sqrt(1.0 + zeta * zeta))
    cos = 1.0 / math.sqrt(1.0 + t * t)
    sin = cos * t
    sq[p] = norm_p - t * gamma
    sq[q] = norm_q + t * gamma
    for i in range(n):
        x_p, x_q = b[p, i], b[q, i]
        b[p, i] = cos * x_p - sin * x_q
        b[q, i] = sin * x_p + cos * x_q
    for i in range(n):
        x_p, x_q = v[p, i], v[q, i]
        v[p, i] = cos * x_p - sin * x_q
        v[q, i] = sin * x_p + cos * x_q
    return True


###################################################################################################
## The fit of one structure


def _to_theta(phi: FloatArray, chain: IntArray, theta: FloatArray) -> None:
    """``engine._OrderedMap.to_theta`` (``theta`` is ``phi`` without a chain)."""
    for i in range(phi.shape[0]):
        theta[i] = phi[i]
    n_chain = chain.shape[0]
    if n_chain < _CHAIN_MIN:
        return
    first = phi[chain[0]]
    offset = 0.0
    theta[chain[0]] = first + offset
    for k in range(1, n_chain):
        offset += math.exp(phi[chain[k]])
        theta[chain[k]] = first + offset


def _jacobian(
    x: FloatArray,
    f: FloatArray,
    theta: FloatArray,
    problem: Problem,
    chain: IntArray,
    work: Workspace,
) -> int:
    """Reduced Jacobian at ``x``, the point evaluated last, into ``jr`` and ``jf``.

    ``_Structure.reduced_jacobian`` (``J = -Q2^T W dM``), the chain rule of the ordered
    parameterisation, then ``_trf._reduce`` (``J = Q' R``: ``jr = R``, ``jf = Q'^T f``),
    in place in ``dmat`` (``draw``: the unwhitened derivatives). Returns the number of rows
    of ``jr``.
    """
    model, white, _ = problem
    cols, fact, jac_work = work
    qr, tau, beta, raw = fact[0], fact[1], fact[3], fact[5]
    draw, dmat = jac_work[0], jac_work[1]
    n = model[0].shape[0]
    n_par = dmat.shape[0]
    length = dmat.shape[1]
    c = qr.shape[0]
    rows = length - c
    _derivatives(theta, beta, model, cols, draw)
    _whiten(draw, dmat, n_par, n, white, raw)
    for j in range(n_par):
        _zero(dmat[j, n:length])  # the prior rows do not depend on theta
    for j in range(n_par):
        _apply_qt(qr, c, 0, length, tau, dmat[j])
    # (Q^T W dM)[c:] = -J; the chain rule is linear in the columns
    if chain.shape[0] >= _CHAIN_MIN:
        _chain_rule(x, chain, dmat, c, jac_work[5][:rows])
    return _reduce_jacobian(f, c, jac_work)


def _chain_rule(x: FloatArray, chain: IntArray, dmat: FloatArray, c: int, acc: FloatArray) -> None:
    """Apply the chain rule of the ordered parameterisation to the rows ``c:`` of ``dmat``.

    ``acc`` is work space of the length of those rows.
    """
    n_chain = chain.shape[0]
    rows = acc.shape[0]
    length = dmat.shape[1]
    _zero(acc)
    for k in range(n_chain - 1, -1, -1):
        col = dmat[chain[k], c:length]
        factor = math.exp(x[chain[k]])
        for i in range(rows):
            acc[i] = acc[i] + col[i]
        if k > 0:
            for i in range(rows):
                col[i] = acc[i] * factor
        else:
            for i in range(rows):
                col[i] = acc[i]


def _reduce_jacobian(f: FloatArray, c: int, jac_work: JacobianWork) -> int:
    """``_trf._reduce`` of the Jacobian in ``dmat`` (rows ``c:``) into ``jr`` and ``jf``.

    Returns the number of rows of ``jr``.
    """
    _, dmat, jr, jf, tau_j, work = jac_work
    n_par = dmat.shape[0]
    length = dmat.shape[1]
    rows = length - c
    if rows <= n_par:
        for i in range(rows):
            for j in range(n_par):
                jr[i, j] = -dmat[j, c + i]
            jf[i] = f[i]
        return int(rows)
    _householder(dmat, n_par, c, length, tau_j)
    qtf = work[:rows]
    for i in range(rows):
        qtf[i] = f[i]
    _apply_qt(dmat, n_par, c, length, tau_j, qtf)
    for i in range(n_par):
        jf[i] = qtf[i]
        for j in range(n_par):
            jr[i, j] = -dmat[j, c + i] if i <= j else 0.0
    return int(n_par)


def _workspace(problem: Problem, n_par: int) -> Workspace:
    """Allocate the work arrays of :func:`optimise`: ``(cols, fact, jac_work)``."""
    model, _, prior = problem
    tw, kinds, bg_kind, fixed = model[0], model[1], model[3], model[5]
    n = tw.shape[0]
    n_comp = kinds.shape[0]
    c = n_comp + (2 if bg_kind == BG_EXPONENTIAL else fixed.shape[0])
    length = prior[2].shape[0]
    cols = (
        np.empty((c, n)),  # design
        np.zeros((max(n_comp, 1), n)),  # numer
        np.zeros((max(n_comp, 1), n)),  # fast
        np.zeros((max(n_comp, 1), 2)),  # skew
        np.zeros((max(n_comp, 1), 2), np.int64),  # spans
        np.empty((4, n)),  # buf
        np.empty(n, np.int64),  # bits
    )
    fact = (
        np.empty((c, length)),  # qr
        np.zeros(max(c, 1)),  # tau
        np.empty(length),  # qtb
        np.zeros(c),  # beta
        np.zeros((c, c)),  # rinv
        np.empty(n),  # raw
    )
    jac_work = (
        np.empty((n_par, n)),  # draw
        np.empty((n_par, length)),  # dmat
        np.zeros((n_par, n_par)),  # jr
        np.zeros(n_par),  # jf
        np.zeros(n_par),  # tau_j
        np.empty(length),  # work
    )
    return cols, fact, jac_work


def _infeasible(x0: FloatArray, lb: FloatArray, ub: FloatArray) -> bool:
    """Tell whether a bound pair is empty or ``x0`` lies outside the bounds."""
    for i in range(x0.shape[0]):
        if not (lb[i] < ub[i] and x0[i] >= lb[i] and x0[i] <= ub[i]):
            return True
    return False


def _finite(x: FloatArray) -> bool:
    """Tell whether every entry of the vector ``x`` is finite."""
    finite = True
    for i in range(x.shape[0]):
        if not math.isfinite(x[i]):
            finite = False
            break
    return finite


def _gradient(jac_work: JacobianWork, n_rows: int) -> tuple[FloatArray, FloatArray]:
    """Return the gradient ``J^T f`` and the column norms of ``J``."""
    jr, jf = jac_work[2], jac_work[3]
    jr_used = jr[:n_rows]
    return jr_used.T.dot(jf[:n_rows]), np.sqrt((jr_used**2).sum(axis=0))


def _initial_radius(
    x: FloatArray, g: FloatArray, lb: FloatArray, ub: FloatArray, scale_inv: FloatArray
) -> float:
    """Return the initial trust-region radius of ``_trf``."""
    v, dv = _cl_scaling(x, g, lb, ub)
    for i in range(x.shape[0]):
        if dv[i] != 0:
            v[i] *= scale_inv[i]
    delta = _norm(x * scale_inv / v**0.5)
    if delta == 0:
        delta = 1.0
    return delta


def optimise(
    start: tuple[FloatArray, FloatArray, FloatArray],
    chain: IntArray,
    problem: Problem,
    tol: float,
    max_nfev: int,
) -> tuple[int, FloatArray, int, FloatArray, FloatArray]:
    """Fit one structure: ``least_squares_trf`` on ``_Structure.reduced``.

    ``start`` is ``(x0, lb, ub)``: start and bounds in the optimiser's coordinates (those of
    the ordered parameterisation when ``chain`` has two latencies or more); ``problem`` is
    ``(model, whitening, prior)`` (``engine._compiled_args``). Returns
    ``(status, theta, nfev, beta, residual_w)``: ``theta`` in the model's coordinates,
    ``residual_w`` the whitened residual of every row (``target`` minus the projected
    design). ``status`` is ``OK``, ``FALLBACK``, ``VALUE_ERROR`` or ``LINALG_ERROR``.
    """
    x0, lb, ub = start
    n_par = x0.shape[0]
    empty = np.zeros(0)
    if _infeasible(x0, lb, ub):
        return VALUE_ERROR, x0.copy(), 0, empty, empty
    work = _workspace(problem, n_par)
    theta = np.empty(n_par)
    x = _strictly_feasible(x0, lb, ub, 1e-10)

    # residual and Jacobian at x
    _to_theta(x, chain, theta)
    if not _evaluate(theta, problem, work[0], work[1]):
        return FALLBACK, x0.copy(), 0, empty, empty
    f = work[1][2][work[1][0].shape[0] :].copy()
    if not _finite(f):
        return VALUE_ERROR, x0.copy(), 0, empty, empty
    cost = 0.5 * _dot(f, f)
    n_rows = _jacobian(x, f, theta, problem, chain, work)
    g, scale_inv = _gradient(work[2], n_rows)
    for i in range(n_par):
        if scale_inv[i] == 0:
            scale_inv[i] = 1.0
    delta = _initial_radius(x, g, lb, ub, scale_inv)
    env = (problem, work, chain, theta, lb, ub, tol, max_nfev)
    status, x, nfev, fresh = _iterate((x, f, cost), (g, scale_inv, n_rows), delta, env)
    if status != OK:
        return status, x0.copy(), 0, empty, empty
    return _solution(x, nfev, fresh, x0, problem, work, chain, theta)


def _iterate(
    point: tuple[FloatArray, FloatArray, float],
    gradient: tuple[FloatArray, FloatArray, int],
    delta: float,
    env: Environment,
) -> tuple[int, FloatArray, int, bool]:
    """Iterate as ``_trf`` from ``point = (x, f, cost)``.

    ``gradient`` is ``(g, scale_inv, n_rows)`` at ``x``. Returns ``(status, x, nfev, fresh)``,
    ``fresh`` when the last evaluation was at ``x`` (its design, QR and beta are at hand).
    """
    problem, work, chain, theta, lb, ub, tol, max_nfev = env
    x, f, cost = point
    g, scale_inv, n_rows = gradient
    jac_work = work[2]
    n_par = x.shape[0]
    alpha = 0.0
    status = -1
    fresh = True
    nfev = 1
    while True:
        v, dv = _cl_scaling(x, g, lb, ub)
        g_norm = np.abs(g * v).max() if n_par else 0.0
        if g_norm < tol:
            status = 1
        if status != -1 or nfev == max_nfev:
            break
        scaled, svd, converged = _local_model(v, dv, scale_inv, g, jac_work, n_rows)
        if not converged:
            return LINALG_ERROR, x, nfev, fresh
        theta_step = max(_THETA_MIN, 1 - g_norm)
        code, trial, search = _steps(
            (x, cost), (scaled, svd, theta_step), (delta, alpha, nfev), env
        )
        if code != OK:
            return code, x, nfev, fresh
        x_new, f_new, cost_new, actual_reduction, status, evaluated = trial
        delta, alpha, nfev = search
        if evaluated:
            fresh = False
        if actual_reduction > 0:
            x, f, cost = x_new, f_new, cost_new
            fresh = True
            if status != -1:
                # SciPy computes the Jacobian at the last point too, for nothing: the
                # solution is the same without it
                break
            n_rows = _jacobian(x, f, theta, problem, chain, work)
            g, norms = _gradient(jac_work, n_rows)
            scale_inv = np.maximum(norms, scale_inv)
    return OK, x, nfev, fresh


def _local_model(
    v: FloatArray,
    dv: FloatArray,
    scale_inv: FloatArray,
    g: FloatArray,
    jac_work: JacobianWork,
    n_rows: int,
) -> tuple[Scaled, tuple[FloatArray, FloatArray, FloatArray], bool]:
    """Return the scaled model of one iteration and the SVD of its stacked Jacobian.

    Returns ``((j_h, diag_h, g_h, d), (s, v_mat, uf), converged)``.
    """
    jr, jf = jac_work[2], jac_work[3]
    n_par = v.shape[0]
    scale_x = 1 / scale_inv
    for i in range(n_par):
        if dv[i] != 0:
            v[i] *= scale_inv[i]
    d = v**0.5 * scale_x
    diag_h = g * dv * scale_x
    g_h = d * g
    # the scaled Jacobian and the diagonal of the reflective scaling, stacked; SciPy takes
    # its SVD (gesdd), here a QR and the one-sided Jacobi SVD of R (_svd_stacked)
    j_h = np.empty((n_rows, n_par))
    for i in range(n_rows):
        for j in range(n_par):
            j_h[i, j] = jr[i, j] * d[j]
    s, v_mat, uf, converged = _svd_stacked(j_h, diag_h**0.5, jf[:n_rows])
    return (j_h, diag_h, g_h, d), (s, v_mat, uf), converged


def _steps(
    point: tuple[FloatArray, float],
    local: tuple[Scaled, tuple[FloatArray, FloatArray, FloatArray], float],
    search: tuple[float, float, int],
    env: Environment,
) -> tuple[
    int,
    tuple[FloatArray, FloatArray, float, float, int, bool],
    tuple[float, float, int],
]:
    """Shrink the trust region until a step lowers the cost (the inner loop of ``_trf``).

    Returns ``(code, (x_new, f_new, cost_new, actual_reduction, status, evaluated),
    (delta, alpha, nfev))``; ``code`` is ``OK``, ``FALLBACK`` or ``VALUE_ERROR``.
    """
    problem, work, chain, theta, lb, ub, tol, max_nfev = env
    x, cost = point
    scaled, svd, theta_step = local
    s, v_mat, uf = svd
    d = scaled[3]
    delta, alpha, nfev = search
    n_par = x.shape[0]
    qtb = work[1][2]
    c = work[1][0].shape[0]
    m = qtb.shape[0]
    x_new = x.copy()
    f_new = qtb[c:].copy()
    cost_new = cost
    status = -1
    evaluated = False
    actual_reduction = -1.0
    while actual_reduction <= 0 and nfev < max_nfev:
        p_h, alpha = _solve_trust_region(n_par, m, uf, s, v_mat, delta, alpha)
        p = d * p_h
        step, step_h, predicted, ok = _select_step(x, scaled, p, p_h, delta, lb, ub, theta_step)
        if not ok:
            return (
                VALUE_ERROR,
                (x_new, f_new, cost_new, actual_reduction, status, evaluated),
                (
                    delta,
                    alpha,
                    nfev,
                ),
            )
        x_new = _strictly_feasible(x + step, lb, ub, 0.0)
        _to_theta(x_new, chain, theta)
        if not _evaluate(theta, problem, work[0], work[1]):
            return (
                FALLBACK,
                (x_new, f_new, cost_new, actual_reduction, status, evaluated),
                (
                    delta,
                    alpha,
                    nfev,
                ),
            )
        f_new = qtb[c:].copy()
        nfev += 1
        evaluated = True
        step_h_norm = _norm(step_h)
        if not _finite(f_new):
            delta = _SHRINK * step_h_norm
            continue
        cost_new = 0.5 * _dot(f_new, f_new)
        actual_reduction = cost - cost_new
        delta_new, ratio = _update_radius(
            delta, actual_reduction, predicted, step_h_norm, step_h_norm > _BOUND_HIT * delta
        )
        status = _termination(actual_reduction, cost, _norm(step), _norm(x), ratio, tol)
        if status != -1:
            break
        alpha *= delta / delta_new
        delta = delta_new
    return OK, (x_new, f_new, cost_new, actual_reduction, status, evaluated), (delta, alpha, nfev)


def _update_radius(
    delta: float, actual: float, predicted: float, step_norm: float, bound_hit: bool
) -> tuple[float, float]:
    """``_trf._update_radius``: the new radius and the ratio of actual to predicted reduction."""
    if predicted > 0:
        ratio = actual / predicted
    elif predicted == actual == 0:
        ratio = 1.0
    else:
        ratio = 0.0
    if ratio < _SHRINK:
        delta_new = _SHRINK * step_norm
    elif ratio > _EXPAND and bound_hit:
        delta_new = delta * 2.0
    else:
        delta_new = delta
    return delta_new, ratio


def _termination(
    d_cost: float, cost: float, dx_norm: float, x_norm: float, ratio: float, tol: float
) -> int:
    """``_trf._termination`` with ``ftol = xtol = tol``; -1 to go on."""
    ftol_ok = d_cost < tol * cost and ratio > _SHRINK
    xtol_ok = dx_norm < tol * (tol + x_norm)
    if ftol_ok and xtol_ok:
        return 4
    if ftol_ok:
        return 2
    if xtol_ok:
        return 3
    return -1


def _solution(
    x: FloatArray,
    nfev: int,
    fresh: bool,
    x0: FloatArray,
    problem: Problem,
    work: Workspace,
    chain: IntArray,
    theta: FloatArray,
) -> tuple[int, FloatArray, int, FloatArray, FloatArray]:
    """Return the linear solution and the whitened residual of every row at ``x``."""
    _, white, prior = problem
    prior_cols, prior_diag, target = prior
    cols, fact, _ = work
    design, beta, raw = cols[0], fact[3], fact[5]
    c, n = design.shape
    empty = np.zeros(0)
    _to_theta(x, chain, theta)
    if not fresh and not _evaluate(theta, problem, cols, fact):
        return FALLBACK, x0.copy(), 0, empty, empty
    # residual of every row: target - A beta (A: the whitened design and the prior rows,
    # rebuilt from the design, which the QR overwrote)
    residual = target.copy()
    whitened = np.empty((c, n))
    _whiten(design, whitened, c, n, white, raw)
    for j in range(c):
        column = whitened[j]
        for i in range(n):
            residual[i] -= column[i] * beta[j]
    for r in range(prior_cols.shape[0]):
        residual[n + r] -= prior_diag[r] * beta[prior_cols[r]]
    return OK, theta.copy(), nfev, beta.copy(), residual


###################################################################################################
## Compilation

_LOCK = threading.Lock()
# compiled with "contract" (fused multiply-adds) and the error model of NumPy (no checks of
# division by zero, which keep loops from vectorising); the inner product may also reorder
# its sum
_REASSOCIATED = ("_dot",)
_COMPILED: dict[str, Any] = {}
# compiled in this order (callees first): each name is replaced by its compiled version in
# this module, so that the compiled callers call compiled code
_KERNELS = (
    "_skew_root",
    "_skew_mode",
    "_vexp",
    "_verfc",
    "_span",
    "_zero",
    "_gaussian_column",
    "_skewed_column",
    "_columns",
    "_derivatives",
    "_start_rows",
    "_whiten",
    "_dot",
    "_householder",
    "_apply_qt",
    "_all_finite",
    "_well_conditioned",
    "_evaluate",
    "_norm",
    "_cl_scaling",
    "_strictly_feasible",
    "_phi_and_derivative",
    "_solve_trust_region",
    "_quadratic_1d",
    "_quadratic_1d_ab",
    "_minimize_1d",
    "_evaluate_quadratic",
    "_step_to_bound",
    "_intersect_trust_region",
    "_select_step",
    "_stacked_qr",
    "_rotate",
    "_jacobi_sweeps",
    "_svd_stacked",
    "_to_theta",
    "_chain_rule",
    "_reduce_jacobian",
    "_jacobian",
    "_workspace",
    "_infeasible",
    "_finite",
    "_gradient",
    "_initial_radius",
    "_local_model",
    "_update_radius",
    "_termination",
    "_steps",
    "_solution",
    "_iterate",
    "optimise",
)


def kernel() -> Callable[..., Any] | None:
    """Return the compiled :func:`optimise`, or None when numba (``fast`` extra) is missing.

    Compiled once per process (numba caches the machine code on disk).
    """
    if "optimise" in _COMPILED:
        return _COMPILED["optimise"]  # type: ignore[no-any-return]
    with _LOCK:
        if "optimise" not in _COMPILED:
            try:
                from evoparam.utils import require

                numba = require("numba", extra="fast", min_version="0.59")
            except ImportError:
                _COMPILED["optimise"] = None
                return None
            module = globals()
            for name in _KERNELS:
                flags = {"contract", "reassoc"} if name in _REASSOCIATED else {"contract"}
                module[name] = numba.njit(
                    cache=True, nogil=True, error_model="numpy", fastmath=flags
                )(module[name])
            _COMPILED["optimise"] = module["optimise"]
    return _COMPILED["optimise"]  # type: ignore[no-any-return]
