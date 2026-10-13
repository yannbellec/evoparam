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
  does not use, is not computed.

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
        mid = x < 8.0
        y = tmp[i] * (p_mid if mid else p_far) / (q_mid if mid else q_far)
        y = 2.0 - y if a < 0 else y
        out[i] = small if x < 1.0 else y


def _span(tw: FloatArray, centre: float, width: float) -> tuple[int, int]:
    """Indices ``[i0, i1)`` of the samples where ``|tw - centre| / width <= SUPPORT_Z``.

    One sample of margin on each side; the component is taken as 0 outside.
    """
    n = tw.shape[0]
    reach = SUPPORT_Z * width
    i0 = max(int(np.searchsorted(tw, centre - reach)) - 1, 0)
    i1 = min(int(np.searchsorted(tw, centre + reach)) + 1, n)
    return i0, max(i1, i0)


def _columns(
    theta: FloatArray,
    tw: FloatArray,
    kinds: IntArray,
    offsets: IntArray,
    bg_kind: int,
    lag: FloatArray,
    fixed: FloatArray,
    design: FloatArray,
    numer: FloatArray,
    skew: FloatArray,
    spans: IntArray,
    buf: FloatArray,
    bits: IntArray,
) -> None:
    """Unwhitened design (columns as rows of ``design``), as ``_columns.component``.

    ``numer`` keeps the numerator of each skewed Gaussian, ``skew`` its ``(c, dc)`` and
    ``spans`` the samples outside which each component is taken as 0 (``SUPPORT_Z``);
    ``buf`` (4 x n) and ``bits`` are work space.
    """
    n = tw.shape[0]
    n_comp = kinds.shape[0]
    for k in range(n_comp):
        o = offsets[k]
        row = design[k]
        if kinds[k] == GAUSSIAN:
            mu, sigma = theta[o], theta[o + 1]
            i0, i1 = _span(tw, mu, sigma)
            t = tw[i0:i1]
            arg = buf[0, : i1 - i0]
            for i in range(t.shape[0]):
                z = (t[i] - mu) / sigma
                arg[i] = -0.5 * (z * z)
            _vexp(arg, row[i0:i1], bits)
        else:
            xi, omega, alpha = theta[o], theta[o + 1], theta[o + 2]
            c, dc = _skew_mode(alpha)
            skew[k, 0] = c
            skew[k, 1] = dc
            inv_c = 1.0 / c
            i0, i1 = _span(tw, xi, omega)
            m = i1 - i0
            t = tw[i0:i1]
            arg, u, g, e = buf[0, :m], buf[1, :m], buf[2, :m], buf[3, :m]
            for i in range(m):
                z = (t[i] - xi) / omega
                arg[i] = -0.5 * (z * z)
                u[i] = -alpha * z / _SQRT2
            _vexp(arg, g, bits)
            _verfc(u, e, arg, bits)
            numer_k = numer[k, i0:i1]
            part = row[i0:i1]
            for i in range(m):
                value = g[i] * e[i]
                numer_k[i] = value
                part[i] = value * inv_c
        head = row[:i0]
        for i in range(head.shape[0]):
            head[i] = 0.0
        tail = row[i1:n]
        for i in range(tail.shape[0]):
            tail[i] = 0.0
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
    theta: FloatArray,
    beta: FloatArray,
    tw: FloatArray,
    kinds: IntArray,
    offsets: IntArray,
    bg_kind: int,
    lag: FloatArray,
    design: FloatArray,
    numer: FloatArray,
    skew: FloatArray,
    spans: IntArray,
    out: FloatArray,
    buf: FloatArray,
    bits: IntArray,
) -> None:
    """``beta_k`` times the derivatives of each column (rows of ``out``, first ``n`` entries).

    As ``_columns.component_derivatives`` and ``_columns.background_derivatives``; ``out``
    is 0 outside the spans of :func:`_columns` and on the rows after the ``n`` first.
    """
    n_comp = kinds.shape[0]
    length = out.shape[1]
    for k in range(n_comp):
        o = offsets[k]
        b = beta[k]
        i0, i1 = spans[k, 0], spans[k, 1]
        m = i1 - i0
        t = tw[i0:i1]
        for q in range(o, offsets[k + 1]):
            head = out[q, :i0]
            for i in range(head.shape[0]):
                head[i] = 0.0
            tail = out[q, i1:length]
            for i in range(tail.shape[0]):
                tail[i] = 0.0
        if kinds[k] == GAUSSIAN:
            mu, sigma = theta[o], theta[o + 1]
            factor = b / sigma
            row = design[k, i0:i1]
            d_mu, d_sigma = out[o, i0:i1], out[o + 1, i0:i1]
            for i in range(m):
                z = (t[i] - mu) / sigma
                sz = row[i] * z * factor
                d_mu[i] = sz
                d_sigma[i] = sz * z
        else:
            xi, omega, alpha = theta[o], theta[o + 1], theta[o + 2]
            c, dc = skew[k, 0], skew[k, 1]
            factor = -b / (omega * c)
            factor_alpha = b / c
            factor_c = b * dc / (c * c)
            spread = 1.0 + alpha * alpha
            numer_k = numer[k, i0:i1]
            arg, gauss = buf[0, :m], buf[1, :m]
            for i in range(m):
                z = (t[i] - xi) / omega
                arg[i] = -0.5 * (z * z) * spread
            _vexp(arg, gauss, bits)
            d_xi, d_omega, d_alpha = out[o, i0:i1], out[o + 1, i0:i1], out[o + 2, i0:i1]
            for i in range(m):
                z = (t[i] - xi) / omega
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
        tail = out[o, lag.shape[0] : length]
        for i in range(tail.shape[0]):
            tail[i] = 0.0


def _whiten(
    rows: FloatArray,
    count: int,
    n: int,
    inv_scale: FloatArray,
    a: FloatArray,
    post: FloatArray,
    stride: int,
    start: FloatArray,
    start_rows: IntArray,
    raw: FloatArray,
) -> None:
    """``Whitener.into`` (or ``StridedWhitener``) on the first ``n`` entries of ``count`` rows.

    ``inv_scale`` is ``1 / scale`` and ``post`` the row weights divided by the innovation SD.
    The terms of the filter are added in the order of the NumPy path (lag 0, 1, ..., p).
    """
    p = a.shape[0] - 1
    has_start = start.shape[0] > 0
    a0 = a[0]
    for j in range(count):
        row = rows[j]
        for i in range(n):
            value = row[i] * inv_scale[i]
            raw[i] = value
            row[i] = a0 * value
        for k in range(1, p + 1):
            shift = k * stride
            if shift >= n:
                break
            ak = a[k]
            dst = row[shift:n]
            src = raw[: n - shift]
            for i in range(dst.shape[0]):
                dst[i] += ak * src[i]
        for i in range(n):
            row[i] *= post[i]
        if has_start:
            for r in range(p):
                total = 0.0
                for q in range(p):
                    total += start[r, q] * raw[start_rows[q]]
                row[start_rows[r]] = total


###################################################################################################
## Householder QR (columns stored as rows: ``mat[j]`` is column j)


def _householder(mat: FloatArray, n_cols: int, off: int, length: int, tau: FloatArray) -> None:
    """QR factorisation in place, as LAPACK's dgeqr2 (unblocked): R above, reflectors below.

    Column ``j`` of the matrix is ``mat[j, off:length]``. The inner products are BLAS
    ``ddot`` calls (``np.dot`` on contiguous rows); the loops run over views from index 0,
    which numba vectorises.
    """
    for j in range(n_cols):
        alpha = mat[j, off + j]
        tail = mat[j, off + j + 1 : length]
        ss = np.dot(tail, tail)
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
            w = (mat[k, off + j] + np.dot(tail, other)) * tau[j]
            mat[k, off + j] -= w
            for i in range(other.shape[0]):
                other[i] -= w * tail[i]


def _apply_qt(
    mat: FloatArray, n_cols: int, off: int, length: int, tau: FloatArray, vec: FloatArray
) -> None:
    """``vec <- Q^T vec`` with the reflectors of :func:`_householder` (``vec[i]``: row ``off + i``)."""
    rows = length - off
    for j in range(n_cols):
        if tau[j] == 0.0:
            continue
        tail = mat[j, off + j + 1 : length]
        rest = vec[j + 1 : rows]
        w = (vec[j] + np.dot(tail, rest)) * tau[j]
        vec[j] -= w
        for i in range(rest.shape[0]):
            rest[i] -= w * tail[i]


###################################################################################################
## Variable projection


def _evaluate(
    theta: FloatArray,
    tw: FloatArray,
    kinds: IntArray,
    offsets: IntArray,
    bg_kind: int,
    lag: FloatArray,
    fixed: FloatArray,
    inv_scale: FloatArray,
    a: FloatArray,
    post: FloatArray,
    stride: int,
    start: FloatArray,
    start_rows: IntArray,
    prior_cols: IntArray,
    prior_diag: FloatArray,
    target: FloatArray,
    design: FloatArray,
    numer: FloatArray,
    skew: FloatArray,
    spans: IntArray,
    buf: FloatArray,
    bits: IntArray,
    qr: FloatArray,
    tau: FloatArray,
    qtb: FloatArray,
    beta: FloatArray,
    rinv: FloatArray,
    raw: FloatArray,
) -> bool:
    """``_Structure.reduced``: QR of the whitened design; False for the SVD route.

    Leaves ``Q^T b`` in ``qtb`` (the residual is ``qtb[c:]``) and the linear solution in
    ``beta``.
    """
    n = tw.shape[0]
    c = qr.shape[0]
    length = qr.shape[1]
    _columns(theta, tw, kinds, offsets, bg_kind, lag, fixed, design, numer, skew, spans, buf, bits)
    for j in range(c):
        src = design[j]
        dst = qr[j]
        for i in range(n):
            dst[i] = src[i]
        tail = qr[j, n:length]
        for i in range(tail.shape[0]):
            tail[i] = 0.0
    _whiten(qr, c, n, inv_scale, a, post, stride, start, start_rows, raw)
    for r in range(prior_cols.shape[0]):
        qr[prior_cols[r], n + r] = prior_diag[r]
    for j in range(c):
        for i in range(length):
            if not math.isfinite(qr[j, i]):
                return False
    _householder(qr, c, 0, length, tau)
    # inverse of R (dtrtri) and the Frobenius condition number
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
    if not (math.isfinite(norm_inv) and math.sqrt(norm_r * norm_inv) < COND_MAX):
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


###################################################################################################
## Trust-region reflective solver (models/_trf.py, after SciPy)


def _norm(x: FloatArray) -> float:
    total = 0.0
    for i in range(x.shape[0]):
        total += x[i] * x[i]
    return math.sqrt(total)


def _cl_scaling(
    x: FloatArray, g: FloatArray, lb: FloatArray, ub: FloatArray
) -> tuple[FloatArray, FloatArray]:
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
    x_new = x.copy()
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
    v = jac.dot(s)
    a = np.dot(v, v)
    a += np.dot(s * diag, s)
    a *= 0.5
    b = np.dot(g, s)
    u = jac.dot(s0)
    b += np.dot(u, v)
    c = 0.5 * np.dot(u, u) + np.dot(g, s0)
    b += np.dot(s0 * diag, s)
    c += 0.5 * np.dot(s0 * diag, s0)
    return a, b, c


def _quadratic_1d_ab(
    jac: FloatArray, g: FloatArray, s: FloatArray, diag: FloatArray
) -> tuple[float, float]:
    v = jac.dot(s)
    a = np.dot(v, v)
    a += np.dot(s * diag, s)
    a *= 0.5
    b = np.dot(g, s)
    return a, b


def _minimize_1d(a: float, b: float, lb: float, ub: float, c: float) -> tuple[float, float]:
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
    jac_s = jac.dot(s)
    q = np.dot(jac_s, jac_s)
    q += np.dot(s * diag, s)
    return 0.5 * q + np.dot(s, g)


def _step_to_bound(
    x: FloatArray, s: FloatArray, lb: FloatArray, ub: FloatArray
) -> tuple[float, IntArray]:
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
    a = np.dot(s, s)
    if a == 0:
        return 0.0, False
    b = np.dot(x, s)
    c = np.dot(x, x) - delta**2
    if c > 0:
        return 0.0, False
    d = math.sqrt(b * b - a * c)
    q = -(b + math.copysign(d, b))
    t1 = q / a
    t2 = c / q
    return max(t1, t2), True


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
) -> tuple[FloatArray, FloatArray, float, bool]:
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
    # B = R (n x n, column j as row j) and V = I
    b = np.zeros((n, n))
    for j in range(n):
        for i in range(min(j + 1, m)):
            b[j, i] = cols[j, i]
    v = np.zeros((n, n))
    for j in range(n):
        v[j, j] = 1.0
    # squared column norms: exact at the start of each sweep, updated after each rotation
    # (||b_p||^2 - t gamma, ||b_q||^2 + t gamma) within it. Rows are indexed in place: a
    # view per pair costs more than its few products
    sq = np.empty(n)
    converged = False
    for _ in range(60):
        for j in range(n):
            total = 0.0
            for i in range(n):
                total += b[j, i] * b[j, i]
            sq[j] = total
        rotated = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                gamma = 0.0
                for i in range(n):
                    gamma += b[p, i] * b[q, i]
                alpha, beta = sq[p], sq[q]
                if gamma == 0.0 or abs(gamma) <= EPS * math.sqrt(alpha * beta):
                    continue
                rotated = True
                zeta = (beta - alpha) / (2.0 * gamma)
                t = math.copysign(1.0, zeta) / (abs(zeta) + math.sqrt(1.0 + zeta * zeta))
                cos = 1.0 / math.sqrt(1.0 + t * t)
                sin = cos * t
                sq[p] = alpha - t * gamma
                sq[q] = beta + t * gamma
                for i in range(n):
                    x_p, x_q = b[p, i], b[q, i]
                    b[p, i] = cos * x_p - sin * x_q
                    b[q, i] = sin * x_p + cos * x_q
                for i in range(n):
                    x_p, x_q = v[p, i], v[q, i]
                    v[p, i] = cos * x_p - sin * x_q
                    v[q, i] = sin * x_p + cos * x_q
        if not rotated:
            converged = True
            break
    norms = np.empty(n)
    for j in range(n):
        norms[j] = math.sqrt(np.dot(b[j], b[j]))
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
            uf[k] = np.dot(b[j], rhs[:n]) / norms[j]
    for k in range(n):
        if not math.isfinite(s[k]):
            converged = False
    return s, v_mat, uf, converged


###################################################################################################
## The fit of one structure


def _to_theta(phi: FloatArray, chain: IntArray, theta: FloatArray) -> None:
    """``engine._OrderedMap.to_theta`` (``theta`` is ``phi`` without a chain)."""
    for i in range(phi.shape[0]):
        theta[i] = phi[i]
    n_chain = chain.shape[0]
    if n_chain < 2:
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
    beta: FloatArray,
    tw: FloatArray,
    kinds: IntArray,
    offsets: IntArray,
    bg_kind: int,
    lag: FloatArray,
    design: FloatArray,
    numer: FloatArray,
    skew: FloatArray,
    spans: IntArray,
    buf: FloatArray,
    bits: IntArray,
    inv_scale: FloatArray,
    a: FloatArray,
    post: FloatArray,
    stride: int,
    start: FloatArray,
    start_rows: IntArray,
    qr: FloatArray,
    tau: FloatArray,
    chain: IntArray,
    raw: FloatArray,
    dmat: FloatArray,
    jr: FloatArray,
    jf: FloatArray,
    tau_j: FloatArray,
    work: FloatArray,
) -> int:
    """Reduced Jacobian at ``x``, the point evaluated last, into ``jr`` and ``jf``.

    ``_Structure.reduced_jacobian`` (``J = -Q2^T W dM``), the chain rule of the ordered
    parameterisation, then ``_trf._reduce`` (``J = Q' R``: ``jr = R``, ``jf = Q'^T f``),
    in place in ``dmat``. Returns the number of rows of ``jr``.
    """
    n = tw.shape[0]
    n_par = dmat.shape[0]
    length = dmat.shape[1]
    c = qr.shape[0]
    rows = length - c
    _derivatives(
        theta, beta, tw, kinds, offsets, bg_kind, lag, design, numer, skew, spans, dmat, buf, bits
    )
    _whiten(dmat, n_par, n, inv_scale, a, post, stride, start, start_rows, raw)
    for j in range(n_par):
        _apply_qt(qr, c, 0, length, tau, dmat[j])
    # (Q^T W dM)[c:] = -J; the chain rule is linear in the columns
    n_chain = chain.shape[0]
    if n_chain >= 2:
        acc = work[:rows]
        for i in range(rows):
            acc[i] = 0.0
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
    if rows <= n_par:
        for i in range(rows):
            for j in range(n_par):
                jr[i, j] = -dmat[j, c + i]
            jf[i] = f[i]
        return rows
    _householder(dmat, n_par, c, length, tau_j)
    qtf = work[:rows]
    for i in range(rows):
        qtf[i] = f[i]
    _apply_qt(dmat, n_par, c, length, tau_j, qtf)
    for i in range(n_par):
        jf[i] = qtf[i]
        for j in range(n_par):
            jr[i, j] = -dmat[j, c + i] if i <= j else 0.0
    return n_par


def optimise(
    x0: FloatArray,
    lb: FloatArray,
    ub: FloatArray,
    chain: IntArray,
    tw: FloatArray,
    kinds: IntArray,
    offsets: IntArray,
    bg_kind: int,
    lag: FloatArray,
    fixed: FloatArray,
    inv_scale: FloatArray,
    a: FloatArray,
    post: FloatArray,
    stride: int,
    start: FloatArray,
    start_rows: IntArray,
    prior_cols: IntArray,
    prior_diag: FloatArray,
    target: FloatArray,
    tol: float,
    max_nfev: int,
) -> tuple[int, FloatArray, int, FloatArray, FloatArray]:
    """Fit one structure: ``least_squares_trf`` on ``_Structure.reduced``.

    ``x0``, ``lb``, ``ub``: start and bounds in the optimiser's coordinates (those of the
    ordered parameterisation when ``chain`` has two latencies or more). Returns
    ``(status, theta, nfev, beta, residual_w)``: ``theta`` in the model's coordinates,
    ``residual_w`` the whitened residual of every row (``target`` minus the projected
    design). ``status`` is ``OK``, ``FALLBACK``, ``VALUE_ERROR`` or ``LINALG_ERROR``.
    """
    n = tw.shape[0]
    n_comp = kinds.shape[0]
    c = n_comp + (2 if bg_kind == BG_EXPONENTIAL else fixed.shape[0])
    length = target.shape[0]
    n_par = x0.shape[0]
    m = length
    rows = length - c
    design = np.empty((c, n))
    numer = np.zeros((max(n_comp, 1), n))
    skew = np.zeros((max(n_comp, 1), 2))
    spans = np.zeros((max(n_comp, 1), 2), np.int64)
    buf = np.empty((4, n))
    bits = np.empty(n, np.int64)
    qr = np.empty((c, length))
    tau = np.zeros(max(c, 1))
    qtb = np.empty(length)
    beta = np.zeros(c)
    rinv = np.zeros((c, c))
    raw = np.empty(n)
    dmat = np.empty((n_par, length))
    jr = np.zeros((n_par, n_par))
    jf = np.zeros(n_par)
    tau_j = np.zeros(n_par)
    work = np.empty(length)
    theta = np.empty(n_par)
    empty = np.zeros(0)

    for i in range(n_par):
        if not (lb[i] < ub[i] and x0[i] >= lb[i] and x0[i] <= ub[i]):
            return VALUE_ERROR, x0.copy(), 0, empty, empty
    x = _strictly_feasible(x0, lb, ub, 1e-10)

    # residual and Jacobian at x
    _to_theta(x, chain, theta)
    if not _evaluate(
        theta, tw, kinds, offsets, bg_kind, lag, fixed, inv_scale, a, post, stride, start,
        start_rows, prior_cols, prior_diag, target, design, numer, skew, spans, buf, bits, qr, tau, qtb,
        beta, rinv, raw,
    ):  # fmt: skip
        return FALLBACK, x0.copy(), 0, empty, empty
    f = qtb[c:].copy()
    for i in range(rows):
        if not math.isfinite(f[i]):
            return VALUE_ERROR, x0.copy(), 0, empty, empty

    nfev = 1
    cost = 0.5 * np.dot(f, f)
    n_rows = _jacobian(
        x, f, theta, beta, tw, kinds, offsets, bg_kind, lag, design, numer, skew, spans, buf, bits, inv_scale, a,
        post, stride, start, start_rows, qr, tau, chain, raw, dmat, jr, jf, tau_j, work,
    )  # fmt: skip
    jr_used = jr[:n_rows]
    g = jr_used.T.dot(jf[:n_rows])
    scale_inv = np.sqrt((jr_used**2).sum(axis=0))
    for i in range(n_par):
        if scale_inv[i] == 0:
            scale_inv[i] = 1.0
    scale_x = 1 / scale_inv

    v, dv = _cl_scaling(x, g, lb, ub)
    for i in range(n_par):
        if dv[i] != 0:
            v[i] *= scale_inv[i]
    delta = _norm(x * scale_inv / v**0.5)
    if delta == 0:
        delta = 1.0

    alpha = 0.0
    status = -1
    fresh = True  # the last evaluation was at x (its design, QR and beta are at hand)
    x_new = x.copy()
    f_new = f.copy()
    cost_new = cost
    while True:
        v, dv = _cl_scaling(x, g, lb, ub)
        g_norm = np.abs(g * v).max() if n_par else 0.0
        if g_norm < tol:
            status = 1
        if status != -1 or nfev == max_nfev:
            break
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
        if not converged:
            return LINALG_ERROR, x0.copy(), 0, empty, empty
        theta_step = max(0.995, 1 - g_norm)

        actual_reduction = -1.0
        while actual_reduction <= 0 and nfev < max_nfev:
            p_h, alpha = _solve_trust_region(n_par, m, uf, s, v_mat, delta, alpha)
            p = d * p_h
            step, step_h, predicted, ok = _select_step(
                x, j_h, diag_h, g_h, p, p_h, d, delta, lb, ub, theta_step
            )
            if not ok:
                return VALUE_ERROR, x0.copy(), 0, empty, empty
            x_new = _strictly_feasible(x + step, lb, ub, 0.0)
            _to_theta(x_new, chain, theta)
            if not _evaluate(
                theta, tw, kinds, offsets, bg_kind, lag, fixed, inv_scale, a, post, stride,
                start, start_rows, prior_cols, prior_diag, target, design, numer, skew,
                spans, buf, bits, qr, tau, qtb, beta, rinv, raw,
            ):  # fmt: skip
                return FALLBACK, x0.copy(), 0, empty, empty
            f_new = qtb[c:].copy()
            nfev += 1
            fresh = False
            step_h_norm = _norm(step_h)
            finite = True
            for i in range(rows):
                if not math.isfinite(f_new[i]):
                    finite = False
                    break
            if not finite:
                delta = 0.25 * step_h_norm
                continue
            cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new
            # radius update (_trf._update_radius)
            if predicted > 0:
                ratio = actual_reduction / predicted
            elif predicted == actual_reduction == 0:
                ratio = 1.0
            else:
                ratio = 0.0
            if ratio < 0.25:
                delta_new = 0.25 * step_h_norm
            elif ratio > 0.75 and step_h_norm > 0.95 * delta:
                delta_new = delta * 2.0
            else:
                delta_new = delta
            # termination (_trf._termination)
            ftol_ok = actual_reduction < tol * cost and ratio > 0.25
            xtol_ok = _norm(step) < tol * (tol + _norm(x))
            if ftol_ok and xtol_ok:
                status = 4
            elif ftol_ok:
                status = 2
            elif xtol_ok:
                status = 3
            if status != -1:
                break
            alpha *= delta / delta_new
            delta = delta_new

        if actual_reduction > 0:
            x, f, cost = x_new, f_new, cost_new
            fresh = True
            if status != -1:
                # SciPy computes the Jacobian at the last point too, for nothing: the
                # solution is the same without it
                break
            n_rows = _jacobian(
                    x, f, theta, beta, tw, kinds, offsets, bg_kind, lag, design, numer, skew, spans, buf, bits, inv_scale, a,
                    post, stride, start, start_rows, qr, tau, chain, raw, dmat, jr, jf, tau_j, work,
                )  # fmt: skip
            jr_used = jr[:n_rows]
            g = jr_used.T.dot(jf[:n_rows])
            scale_inv = np.maximum(np.sqrt((jr_used**2).sum(axis=0)), scale_inv)
            scale_x = 1 / scale_inv

    # the linear solution and the whitened residual at the solution
    _to_theta(x, chain, theta)
    if not fresh and not _evaluate(
        theta, tw, kinds, offsets, bg_kind, lag, fixed, inv_scale, a, post, stride, start,
        start_rows, prior_cols, prior_diag, target, design, numer, skew, spans, buf, bits, qr, tau, qtb,
        beta, rinv, raw,
    ):  # fmt: skip
        return FALLBACK, x0.copy(), 0, empty, empty
    # residual of every row: target - A beta (A: the whitened design and the prior rows,
    # rebuilt from the design, which the QR overwrote)
    residual = target.copy()
    column = np.empty(length)
    for j in range(c):
        for i in range(n):
            column[i] = design[j, i]
        for i in range(n, length):
            column[i] = 0.0
        _whiten(
            column.reshape((1, length)), 1, n, inv_scale, a, post, stride, start, start_rows,
            raw,
        )  # fmt: skip
        for i in range(length):
            residual[i] -= column[i] * beta[j]
    for r in range(prior_cols.shape[0]):
        residual[n + r] -= prior_diag[r] * beta[prior_cols[r]]
    return OK, theta.copy(), nfev, beta.copy(), residual


###################################################################################################
## Compilation

_LOCK = threading.Lock()
_COMPILED: dict[str, Any] = {}
# compiled in this order (callees first): each name is replaced by its compiled version in
# this module, so that the compiled callers call compiled code
_KERNELS = (
    "_skew_root",
    "_skew_mode",
    "_vexp",
    "_verfc",
    "_span",
    "_columns",
    "_derivatives",
    "_whiten",
    "_householder",
    "_apply_qt",
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
    "_svd_stacked",
    "_to_theta",
    "_jacobian",
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
                module[name] = numba.njit(cache=True, nogil=True, error_model="numpy")(
                    module[name]
                )
            _COMPILED["optimise"] = module["optimise"]
    return _COMPILED["optimise"]  # type: ignore[no-any-return]
