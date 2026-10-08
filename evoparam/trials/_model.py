"""The trial model: the channel's components, shifted and scaled for each trial.

``x_i(t) = b_i(t) + sum_k a_ik s_k(t; theta_k + delta_ik e_loc) + noise``: for shapes whose
latency parameter is a location (all of the registry), moving it by ``delta`` shifts the
curve by ``delta``. The amplitudes ``a_ik`` and the background's linear coefficients are
solved by least squares for given shifts (variable projection); the shifts (and, in the
``free`` and ``regularized`` modes, the other parameters within +/-25 % of the channel
values) by bounded least squares.
"""

import dataclasses

import numpy as np
from scipy.optimize import least_squares

__all__ = []

FREE_RANGE = 0.25


@dataclasses.dataclass
class ChannelTemplate:
    """What a trial fit needs from the channel fit."""

    shapes: list                 # shape objects, one per component
    params: list                 # channel parameters of each component
    loc_index: list              # index of the location parameter of each component
    max_shift: np.ndarray        # per component (s)
    background: object           # background object
    bg_nonlinear: np.ndarray     # background time constants, fixed
    tw: np.ndarray               # window times

    def basis(self):
        if not getattr(self.background, "n_linear", 0):
            return np.zeros((0, len(self.tw)))
        nl = self.bg_nonlinear if len(self.bg_nonlinear) else None
        return np.atleast_2d(self.background.basis(self.tw, nl))


def _layout(template, mode, movable=None):
    """Free parameters of each component: indices into its parameter vector. With
    ``movable`` (component indices), the other components keep their channel location."""

    if mode in ("free", "regularized"):
        return [list(range(s.n_params)) for s in template.shapes]
    return [[i] if movable is None or k in movable else []
            for k, i in enumerate(template.loc_index)]


def _bounds(template, mode, layout):
    lo, hi = [], []
    for k, (shape, params, free) in enumerate(zip(template.shapes, template.params, layout)):
        s_lo, s_hi = shape.default_bounds(template.tw)
        for i in free:
            if i == template.loc_index[k]:
                a, b = params[i] - template.max_shift[k], params[i] + template.max_shift[k]
            else:
                span = FREE_RANGE * max(abs(params[i]), 1e-12)
                a, b = params[i] - span, params[i] + span
            lo.append(max(a, s_lo[i]))
            hi.append(min(b, s_hi[i]))
    lo, hi = np.array(lo, float), np.array(hi, float)
    bad = hi <= lo
    hi[bad] = lo[bad] + 1e-12
    return lo, hi


def _unpack(template, layout, z):
    out, pos = [], 0
    for params, free in zip(template.params, layout):
        p = np.array(params, float)
        for i in free:
            p[i] = z[pos]
            pos += 1
        out.append(p)
    return out


def fit_trial(template, x, mode="shift_scale", ridge=0.0, valid=None, tol=1e-8, support=None,
              movable=None):
    """Fit one trial (window samples ``x``). Returns a dict of the per-trial numbers.

    ``support`` (bool over the window) restricts the r² to the samples around the components
    of interest: over a long window, ongoing activity would dominate the total variance.
    """

    layout = _layout(template, mode, movable)
    lo, hi = _bounds(template, mode, layout)
    z0 = np.concatenate([[p[i] for i in free] for p, free in zip(template.params, layout)])
    z0 = np.clip(z0, lo, hi)
    basis = template.basis()
    valid = np.ones(len(x), bool) if valid is None else valid
    w = valid.astype(float)
    centre = z0.copy()
    scale = np.where(hi > lo, hi - lo, 1.0)

    def design(z):
        params = _unpack(template, layout, z)
        cols = [s.evaluate(template.tw, p) for s, p in zip(template.shapes, params)]
        return np.column_stack(cols + list(basis)), params

    def solve(z):
        m, params = design(z)
        a = m * w[:, None]
        beta, *_ = np.linalg.lstsq(a, x * w, rcond=None)
        return beta, m, params

    def residual(z):
        beta, m, _ = solve(z)
        r = (x - m @ beta) * w
        if ridge:
            r = np.r_[r, np.sqrt(ridge) * (z - centre) / scale * np.sqrt(len(x))]
        return r

    def jac(z):
        beta, m, params = solve(z)
        cols = []
        pos = 0
        for k, (shape, p, free) in enumerate(zip(template.shapes, params, layout)):
            jk = shape.jacobian(template.tw, p)
            for i in free:
                cols.append(-beta[k] * jk[:, i] * w)
                pos += 1
        j = np.column_stack(cols) if cols else np.zeros((len(x), 0))
        # Kaufman projection: remove the part explained by the linear columns
        a = m * w[:, None]
        q, _ = np.linalg.qr(a)
        j = j - q @ (q.T @ j)
        if ridge:
            j = np.vstack([j, np.sqrt(ridge) * np.diag(1 / scale) * np.sqrt(len(x))])
        return j

    try:
        res = least_squares(residual, z0, jac=jac, bounds=(lo, hi), method="trf", x_scale="jac",
                            ftol=tol, xtol=tol, gtol=tol, max_nfev=200)
        z, converged = res.x, bool(res.status > 0 and np.all(np.isfinite(res.x)))
    except (ValueError, np.linalg.LinAlgError):
        z, converged = z0, False
    beta, m, params = solve(z)
    fitted = m @ beta
    keep = valid if support is None else valid & support
    resid = (x - fitted)[keep]
    tss = np.sum((x[keep] - x[keep].mean()) ** 2)
    r2 = 1 - np.sum(resid ** 2) / tss if tss > 0 else np.nan
    span = hi - lo
    hit = (z - lo <= 1e-3 * span) | (hi - z <= 1e-3 * span)
    # per component: any of its free parameters at a bound
    owner = np.array([k for k, free in enumerate(layout) for _ in free], dtype=int)
    at_bound_k = np.array([bool(np.any(hit[owner == k])) for k in range(len(layout))])
    shifts = np.array([p[i] - c[i] for p, c, i in zip(params, template.params,
                                                       template.loc_index)])
    return {"params": params, "scale": beta[:len(template.shapes)], "shift": shifts,
            "background": beta[len(template.shapes):], "r2": float(r2),
            "converged": converged and np.all(np.isfinite(beta)),
            "at_bound": bool(at_bound_k.any()), "at_bound_k": at_bound_k,
            "fitted": fitted}
