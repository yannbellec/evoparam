# SPDX-License-Identifier: Apache-2.0
"""Penalised refit of an asymmetric component (``shape_choice="penalised"``).

The least-squares fit of §5.5.3 places the mode of a skewed Gaussian badly when the
deflection is not a skewed Gaussian (a gamma or a biexponential, REQ-106): the mode moves
quickly with the asymmetry ``alpha`` (about 4 % of the FWHM per unit near ``alpha = 4``),
so the noise of ``alpha`` spreads the latency, and the tails, where the skewed Gaussian
departs from such a shape, pull its mode away from the peak (DECISIONS §5.4, CHANGELOG
"REQ-106 with the default pair: a penalised estimate of the asymmetry").

The refit changes only the non-linear parameters of one component, the others and the
background held, and solves every linear coefficient by variable projection, as the
engine does:

1. a Gaussian prior on ``alpha`` (mean 0, SD ``ASYMMETRY_PRIOR_SD``), a pseudo-observation
   added to the whitened residual;
2. a weight ``sqrt(1 + PEAK_WEIGHT s(t)^2)`` on the samples (``s`` the unit component
   curve), which counts the top of the component more than its flanks;
3. the samples below ``LOCAL_LEVEL`` of the component's peak further weighted by
   ``eps = min(1, DISCREPANCY_ENERGY / E)``, ``E`` the whitened energy of the component
   (sum of its squared whitened curve). The misfit of a wrong shape grows with ``E`` while
   the noise does not: a weak component is fitted on all its samples, a strong one mostly
   on its top, where a skewed Gaussian follows any unimodal peak.

The prior and the weights act on the estimate only: the criteria, the residuals and the
uncertainty of the fit are those of the data at the estimate (unweighted, without the prior).
"""

import numpy as np
from scipy.optimize import least_squares

from evoparam._typing import Any, NDArray

# SD of the prior on the asymmetry alpha (mean 0); chosen on development seeds (CHANGELOG)
ASYMMETRY_PRIOR_SD = 2.5
# weight of the top of the component: sqrt(1 + PEAK_WEIGHT * s^2)
PEAK_WEIGHT = 2.0
# fraction of the peak below which a sample counts as the component's tail
LOCAL_LEVEL = 0.3
# whitened energy of the component at which its tail keeps its full weight
DISCREPANCY_ENERGY = 600.0
# asymmetry of the second start of the refit
ALPHA_START = 0.0
_TOL = 1e-10

###################################################################################################
###################################################################################################


def _solve(struct: Any, theta: NDArray[Any], w: NDArray[Any]) -> tuple[Any, Any]:
    """Weighted variable projection at ``theta``: (whitened residual with prior rows, beta)."""
    m = struct.design(theta)
    a = struct.white(m * w[:, None]) if m.shape[1] else np.zeros((len(struct.y), 0))
    b = struct.white(struct.y * w)
    rows = struct.prior_rows(a.shape[1])
    if rows is not None:
        target = np.zeros(len(rows))
        if struct.prior_col is not None:
            position = int(np.sum(np.diag(struct.prior_rows_mask(a.shape[1]))[: struct.prior_col]))
            target[position] = struct.level / float(struct.prior)
        a = np.vstack([a, rows])
        b = np.r_[b, target]
    if a.shape[1] == 0:
        return b, np.zeros(0)
    beta = np.linalg.lstsq(a, b, rcond=None)[0]
    return b - a @ beta, beta


def _limits(struct: Any, theta: NDArray[Any], k: int) -> tuple[NDArray[Any], NDArray[Any]]:
    """Bounds of the parameters of component ``k``; its latency kept between its neighbours'."""
    lo, hi = struct.bounds()
    part = slice(int(struct.offsets[k]), int(struct.offsets[k + 1]))
    lo, hi = lo[part].copy(), hi[part].copy()
    chain = list(getattr(struct, "chain", []))
    shape = struct.shapes[k]
    if chain and shape.latency_param in shape.param_names:
        i = shape.param_names.index(shape.latency_param)
        pos = int(struct.offsets[k]) + i
        if pos in chain:
            j = chain.index(pos)
            if j > 0:
                lo[i] = max(lo[i], float(theta[chain[j - 1]]))
            if j < len(chain) - 1:
                hi[i] = min(hi[i], float(theta[chain[j + 1]]))
    return lo, hi


def penalised_component(struct: Any, theta: Any, k: int) -> NDArray[Any] | None:
    """Return ``theta`` with the parameters of component ``k`` refitted (see the module).

    None when the component has no ``alpha`` or the refit fails; ``theta`` itself is not
    changed.
    """
    shape = struct.shapes[k]
    names = tuple(shape.param_names)
    if "alpha" not in names:
        return None
    ia = names.index("alpha")
    theta = np.asarray(theta, float)
    part = slice(int(struct.offsets[k]), int(struct.offsets[k + 1]))
    lo, hi = _limits(struct, theta, k)
    if not np.all(lo < hi):
        return None
    span = hi - lo
    inner_lo, inner_hi = lo + 1e-9 * span, hi - 1e-9 * span
    tw = struct.tw
    n = len(tw)

    def residual(q: NDArray[Any], w: NDArray[Any]) -> NDArray[Any]:
        th = theta.copy()
        th[part] = q
        r, _ = _solve(struct, th, w)
        out: NDArray[Any] = np.r_[r, q[ia] / ASYMMETRY_PRIOR_SD]
        return out

    def refit(q0: NDArray[Any], w: NDArray[Any]) -> Any:
        q0 = np.minimum(np.maximum(q0, inner_lo), inner_hi)
        return least_squares(
            residual,
            q0,
            args=(w,),
            bounds=(lo, hi),
            method="trf",
            x_scale="jac",
            xtol=_TOL,
            ftol=_TOL,
            gtol=_TOL,
            max_nfev=200 * len(q0),
        )

    try:
        ones = np.ones(n)
        start = theta[part].copy()
        other = start.copy()
        other[ia] = ALPHA_START
        fits = [refit(start, ones), refit(other, ones)]
        q = min(fits, key=lambda f: f.cost).x
        curve = shape.evaluate(tw, q)
        top = np.sqrt(1.0 + PEAK_WEIGHT * curve**2)
        q = refit(q, top).x
        curve = shape.evaluate(tw, q)
        th = theta.copy()
        th[part] = q
        _, beta = _solve(struct, th, ones)
        energy = float(np.sum(struct.white(beta[k] * curve) ** 2))
        eps = min(1.0, DISCREPANCY_ENERGY / energy) if energy > 0 else 1.0
        weights = np.sqrt(1.0 + PEAK_WEIGHT * curve**2) * np.where(curve >= LOCAL_LEVEL, 1.0, eps)
        q = refit(q, weights).x
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not np.all(np.isfinite(q)):
        return None
    out: NDArray[Any] = np.array(theta, dtype=float)
    out[part] = q
    return out
