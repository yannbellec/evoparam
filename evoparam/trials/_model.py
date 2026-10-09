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
from scipy.special import erfc

from evoparam.shapes.components import (_SQRT2, _SQRT_2_OVER_PI, Gaussian, SkewedGaussian,
                                        skew_mode)

__all__ = []

FREE_RANGE = 0.25
# initial Levenberg-Marquardt damping (relative to the curvature); 1 chosen on simulated
# CCEPs (simulate_ccep, 3 x 46 channels): 0.4 % of trials end at a worse point than the
# one-trial bounded least-squares solver, against 1.4 % with 1e-3
MU0 = 1.0
# smallest factor of the damping after a good step (Nielsen: 1/3)
MU_DOWN = 1 / 3


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
    ``movable`` (component indices), the other components keep their channel parameters
    (only their amplitude is fitted)."""

    if mode in ("free", "regularized"):
        return [list(range(s.n_params)) if movable is None or k in movable else []
                for k, s in enumerate(template.shapes)]
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


def _value_and_loc_derivative(shape, t, params, loc, jac=True):
    """A shape and its derivative in its location parameter, at times ``t``.

    The Gaussian and the skewed Gaussian (the shapes of the presets) share one pass for both
    (the same formulas as their ``evaluate`` and ``jacobian``); any other shape uses its
    ``evaluate`` and ``jacobian``."""

    kind = type(shape)
    if kind is Gaussian and loc == 0:
        mu, sigma = float(params[0]), float(params[1])
        z = (t - mu) / sigma
        values = np.exp(-0.5 * z * z)
        return values, (values * z / sigma if jac else None)
    if kind is SkewedGaussian and loc == 0:
        xi, omega, alpha = float(params[0]), float(params[1]), float(params[2])
        _, c, _ = skew_mode(alpha)
        c = float(c)
        z = (t - xi) / omega
        gauss_z = np.exp(-0.5 * z * z)
        numerator = gauss_z * erfc(-alpha * z / _SQRT2)
        values = numerator / c
        if not jac:
            return values, None
        gauss = _SQRT_2_OVER_PI * np.exp(-0.5 * z * z * (1.0 + alpha * alpha))
        return values, (z * numerator - alpha * gauss) / (omega * c)
    values = shape.evaluate(t, params)
    if not jac:
        return values, None
    return values, shape.jacobian(t, params)[:, loc]


def _batched_lstsq(a, y, pad=None):
    """Least squares for a stack of small problems: ``a`` (n, T, m), ``y`` (n, T).

    Returns the coefficients (n, m) and a function that removes from a stack of columns
    (n, T, p) their projection on the columns of ``a`` (the Kaufman projection). Normal
    equations on unit-norm columns (Cholesky); a problem whose Gram matrix is not safely
    positive definite is solved by QR, and a rank-deficient one by ``lstsq``."""

    n, _, m = a.shape
    at = np.swapaxes(a, 1, 2)
    # Gram matrix of the unit-norm columns, from the raw columns (no scaled copy of ``a``)
    raw = np.matmul(at, a)
    norms = np.sqrt(np.maximum(np.diagonal(raw, axis1=1, axis2=2), 0.0))
    norms = np.where(norms > 0, norms, 1.0)
    gram = raw / (norms[:, :, None] * norms[:, None, :])
    if pad is not None and pad.any():
        # padding columns are zero: a unit diagonal gives them a zero coefficient
        rows, cols = np.nonzero(pad)
        gram[rows, cols, cols] = 1.0
    good = np.ones(n, bool)
    try:
        chol = np.linalg.cholesky(gram)
        diag = np.diagonal(chol, axis1=1, axis2=2)
        # well conditioned enough for normal equations (condition of b below about 1e6)
        good = diag.min(axis=1) > 1e-6
    except np.linalg.LinAlgError:
        good[:] = False
    beta = np.zeros((n, m))
    if good.any():
        rhs = np.matmul(at[good], y[good][..., None])[..., 0] / norms[good]
        beta[good] = np.linalg.solve(gram[good], rhs[..., None])[..., 0]
    bad = np.flatnonzero(~good)
    q_bad = None
    if len(bad):
        b = a[bad] / norms[bad][:, None, :]
        q_bad, r = np.linalg.qr(b)
        for row, i in enumerate(bad):
            rd = np.abs(np.diag(r[row]))
            if rd.min() > 1e-10 * max(rd.max(), 1e-300):
                beta[i] = np.linalg.solve(r[row], q_bad[row].T @ y[i])
            else:
                beta[i] = np.linalg.lstsq(b[row], y[i], rcond=None)[0]
    beta = beta / norms

    def project_out(j, sel=None):
        """``j`` holds the columns of the problems ``sel`` (default: all), in that order."""

        sel = np.arange(n) if sel is None else np.asarray(sel)
        out = j.copy()
        g_sel = good[sel]
        if g_sel.any():
            rows = sel[g_sel]
            jg = j[g_sel]
            nr = norms[rows][:, :, None]
            coef = np.linalg.solve(gram[rows], np.matmul(at[rows], jg) / nr) / nr
            out[g_sel] = jg - np.matmul(a[rows], coef)
        if not g_sel.all():
            where = np.searchsorted(bad, sel[~g_sel])
            q = q_bad[where]
            jb = j[~g_sel]
            out[~g_sel] = jb - np.matmul(q, np.matmul(np.swapaxes(q, 1, 2), jb))
        return out

    return beta, project_out


def fit_trials_batch(template, data, valid=None, tol=1e-8, support=None, movable=None,
                     max_iter=100):
    """``shift_scale`` fit of every trial at once (rows of ``data``, window samples).

    The same model and bounds as :func:`fit_trial` in ``shift_scale`` mode, solved by a
    projected Levenberg-Marquardt iteration vectorised over trials (variable projection,
    Kaufman Jacobian). Returns one dict per trial, with the keys of :func:`fit_trial`.
    """

    return fit_trials_multi([{"template": template, "data": data, "valid": valid, "tol": tol,
                              "support": support, "movable": movable}], max_iter=max_iter)[0]


def _fast_kind(shape, loc):
    kind = type(shape)
    if kind is Gaussian and loc == 0:
        return "gauss"
    if kind is SkewedGaussian and loc == 0:
        return "skew"
    return None


def fit_trials_multi(problems, max_iter=100, max_rows=1024):
    """:func:`fit_trials_batch` for several problems (channels, null segments) at once.

    ``problems``: dicts with ``template`` and ``data`` and optionally ``valid``, ``tol``,
    ``support`` and ``movable``. The trials of all problems with the same window length are
    solved together, by groups of at most ``max_rows`` trials (each trial's iteration is its
    own: the result of a trial does not depend on the others). Returns one list of dicts
    per problem."""

    out = [None] * len(problems)
    by_len = {}
    for i, pr in enumerate(problems):
        by_len.setdefault(len(pr["template"].tw), []).append(i)
    for idx in by_len.values():
        group, rows = [], 0
        for i in idx:
            n_i = len(np.atleast_2d(problems[i]["data"]))
            if group and rows + n_i > max_rows:
                for j, res in zip(group, _solve_multi([problems[j] for j in group], max_iter)):
                    out[j] = res
                group, rows = [], 0
            group.append(i)
            rows += n_i
        if group:
            for j, res in zip(group, _solve_multi([problems[j] for j in group], max_iter)):
                out[j] = res
    return out


def _solve_multi(problems, max_iter):
    """The trials of ``problems`` (same window length) in one projected Levenberg-Marquardt
    iteration. The columns that do not move (components held at their channel latency, the
    background's linear terms) are projected out once per trial (orthonormal basis from an
    SVD); the iteration then works on the moving columns only."""

    infos = []
    n_t = len(problems[0]["template"].tw)
    for pr in problems:
        tpl = pr["template"]
        data = np.atleast_2d(np.asarray(pr["data"], float))
        layout = _layout(tpl, "shift_scale", pr.get("movable"))
        lo, hi = _bounds(tpl, "shift_scale", layout)
        valid = pr.get("valid")
        valid = np.ones(data.shape, bool) if valid is None else np.asarray(valid, bool)
        basis = tpl.basis()
        fixed = [k for k, free in enumerate(layout) if not free]
        infos.append({"tpl": tpl, "data": data, "layout": layout, "lo": lo, "hi": hi,
                      "basis": basis, "valid": valid, "n": len(data), "fixed": fixed,
                      "tol": float(pr.get("tol", 1e-8)), "support": pr.get("support")})
    n = sum(f["n"] for f in infos)
    p = max(len(f["lo"]) for f in infos)
    n_fix = max(len(f["fixed"]) + len(f["basis"]) for f in infos)
    data = np.concatenate([f["data"] for f in infos])
    w = np.concatenate([f["valid"] for f in infos]).astype(float)
    all_valid = bool(np.all(w == 1.0))
    wx = data * w
    lo, hi = np.zeros((n, p)), np.zeros((n, p))
    z = np.zeros((n, p))
    tol = np.zeros(n)
    pad = np.ones((n, p), bool)              # parameter slots that are padding
    fcols = np.zeros((n, n_t, n_fix))        # columns that do not move (unweighted)
    # one entry per (trial, moving component)
    pr_row, pr_q, pr_prob, pr_comp = [], [], [], []
    start = 0
    for pi, f in enumerate(infos):
        tpl, rows = f["tpl"], slice(start, start + f["n"])
        q_n = len(f["lo"])
        tol[rows] = f["tol"]
        lo[rows, :q_n], hi[rows, :q_n] = f["lo"], f["hi"]
        z[rows, :q_n] = np.clip([tpl.params[k][i] for k, free in enumerate(f["layout"])
                                 for i in free], f["lo"], f["hi"])
        pad[rows, :q_n] = False
        q = 0
        for k, free in enumerate(f["layout"]):
            if free:
                pr_row += range(start, start + f["n"])
                pr_q += [q] * f["n"]
                pr_prob += [pi] * f["n"]
                pr_comp += [k] * f["n"]
                q += 1
        for c, k in enumerate(f["fixed"]):
            fcols[rows, :, c] = tpl.shapes[k].evaluate(tpl.tw, tpl.params[k])
        for i, b in enumerate(f["basis"]):
            fcols[rows, :, len(f["fixed"]) + i] = b
        start += f["n"]
    pr_row, pr_q = np.array(pr_row, int), np.array(pr_q, int)
    pr_prob, pr_comp = np.array(pr_prob, int), np.array(pr_comp, int)
    tws = np.array([f["tpl"].tw for f in infos])
    # fixed columns: orthonormal basis of their (weighted) span, per trial
    # (one basis for all trials when they share their fixed columns and weights)
    shared = len(infos) == 1 and all_valid
    fw = fcols[:1] if shared else fcols * w[:, :, None]
    if n_fix:
        u_f, s_f, vt_f = np.linalg.svd(fw, full_matrices=False)
        keep_f = s_f > 1e-10 * np.maximum(s_f[:, :1], 1e-300)
        u_f = u_f * keep_f[:, None, :]
        if shared:
            u_f, s_f, vt_f = u_f[0], s_f[0], vt_f[0]
    else:
        u_f = np.zeros((n, n_t, 0))
        shared = False

    if shared:
        y_perp = wx - (wx @ u_f) @ u_f.T
    else:
        y_perp = wx - np.matmul(u_f, np.matmul(np.swapaxes(u_f, 1, 2), wx[..., None]))[..., 0]
    # the shapes of the presets are evaluated for all their (trial, component) entries at
    # once, with per-entry parameters; any other shape by (problem, component)
    kinds = {"gauss": [], "skew": []}
    generic = {}
    base_loc = np.zeros(len(pr_row))
    pars = np.zeros((len(pr_row), 4))        # gauss: mu, sigma; skew: xi, omega, alpha, c
    for e in range(len(pr_row)):
        tpl = infos[pr_prob[e]]["tpl"]
        k = pr_comp[e]
        shape, base, loc = tpl.shapes[k], np.asarray(tpl.params[k], float), tpl.loc_index[k]
        base_loc[e] = base[loc]
        kind = _fast_kind(shape, loc)
        if kind == "gauss":
            pars[e, :2] = base[:2]
        elif kind == "skew":
            pars[e, :3] = base[:3]
            pars[e, 3] = float(skew_mode(base[2])[1])
        else:
            generic.setdefault((pr_prob[e], k), []).append(e)
            continue
        kinds[kind].append(e)
    kinds = {key: np.array(v, int) for key, v in kinds.items()}
    generic = {key: np.array(v, int) for key, v in generic.items()}

    def moving(zr, rows, jac):
        """Moving columns (rows, T, p) at ``zr`` and their derivatives in the shift."""

        nr = len(rows)
        local = np.full(n, -1)
        local[rows] = np.arange(nr)
        ms = np.zeros((nr, n_t, p))
        dm = np.zeros((nr, n_t, p)) if jac else None

        def put(entries, values, deriv):
            lr = local[pr_row[entries]]
            ms[lr, :, pr_q[entries]] = values
            if jac:
                dm[lr, :, pr_q[entries]] = deriv

        for kind, entries in kinds.items():
            e = entries[local[pr_row[entries]] >= 0] if len(entries) else entries
            if not len(e):
                continue
            lr = local[pr_row[e]]
            delta = zr[lr, pr_q[e]] - base_loc[e]
            tt = tws[pr_prob[e]] - delta[:, None]
            if kind == "gauss":
                sigma = pars[e, 1, None]
                u = (tt - pars[e, 0, None]) / sigma
                values = np.exp(-0.5 * u * u)
                put(e, values, values * u / sigma if jac else None)
            else:
                omega, alpha, c = pars[e, 1, None], pars[e, 2, None], pars[e, 3, None]
                u = (tt - pars[e, 0, None]) / omega
                gauss_u = np.exp(-0.5 * u * u)
                # erfc (the costly part) only where the Gaussian factor is not negligible
                # (|u| < 9.6: the curve is below 1e-20 of its peak elsewhere)
                near = u * u < 184.0
                numerator = np.zeros_like(u)
                alpha_b = np.broadcast_to(alpha, u.shape)
                numerator[near] = gauss_u[near] * erfc(-alpha_b[near] * u[near] / _SQRT2)
                deriv = None
                if jac:
                    g = _SQRT_2_OVER_PI * np.exp(-0.5 * u * u * (1.0 + alpha * alpha))
                    deriv = (u * numerator - alpha * g) / (omega * c)
                put(e, numerator / c, deriv)
        for (pi, k), entries in generic.items():
            e = entries[local[pr_row[entries]] >= 0]
            if not len(e):
                continue
            tpl = infos[pi]["tpl"]
            base = np.asarray(tpl.params[k], float)
            lr = local[pr_row[e]]
            delta = zr[lr, pr_q[e]] - base_loc[e]
            t = (tpl.tw[None, :] - delta[:, None]).ravel()
            values, d = _value_and_loc_derivative(tpl.shapes[k], t, base, tpl.loc_index[k],
                                                  jac)
            put(e, values.reshape(len(e), n_t), None if d is None else d.reshape(len(e), n_t))
        return ms, dm

    def project_fixed(x, rows):
        if shared:
            # one product for all trials: (T, f) against (T, rows x k)
            nr, _, k = x.shape
            flat = np.swapaxes(x, 0, 1).reshape(n_t, nr * k)
            flat = flat - u_f @ (u_f.T @ flat)
            return np.swapaxes(flat.reshape(n_t, nr, k), 0, 1)
        uf = u_f[rows]
        return x - np.matmul(uf, np.matmul(np.swapaxes(uf, 1, 2), x))

    def evaluate(zr, rows, jac=True):
        ms, dm = moving(zr, rows, jac)
        a = ms if all_valid else ms * w[rows, :, None]
        a = project_fixed(a, rows) if n_fix else a
        beta, project_out = _batched_lstsq(a, y_perp[rows], pad[rows])
        r = y_perp[rows] - np.matmul(a, beta[..., None])[..., 0]
        if not jac:
            return r, beta, ms, None

        def jacobian(sel=None):
            """Projected Jacobian of the problems ``sel`` (default: all): computed only for
            the steps that are kept."""

            sel = np.arange(len(rows)) if sel is None else np.asarray(sel)
            j = -beta[sel][:, None, :] * dm[sel]
            if not all_valid:
                j *= w[rows[sel], :, None]
            if n_fix:
                j = project_fixed(j, rows[sel])
            return project_out(j, sel)

        return r, beta, ms, jacobian

    converged = np.zeros(n, bool)
    finished = np.zeros(n, bool)
    if p:
        mu = np.full(n, MU0)
        r, beta, _, jacobian = evaluate(z, np.arange(n))
        j = jacobian()
        cost = 0.5 * np.sum(r ** 2, axis=1)
        eye = np.eye(p)[None]
        for _ in range(max_iter):
            act = np.flatnonzero(~finished)
            if not len(act):
                break
            ja, ra = j[act], r[act]
            h = np.matmul(np.swapaxes(ja, 1, 2), ja)
            g = np.matmul(np.swapaxes(ja, 1, 2), ra[..., None])[..., 0]
            za, la, ha, ta = z[act], lo[act], hi[act], tol[act, None]
            # active set: a parameter on a bound that the step would push out of the box is
            # held there, and the step is recomputed for the others
            blocked = np.zeros_like(za, dtype=bool)
            g0, h0 = g, h
            for _pass in range(p + 1):
                g = np.where(blocked, 0.0, g0)
                h = np.where(blocked[:, :, None] | blocked[:, None, :], 0.0, h0)
                dh = np.diagonal(h, axis1=1, axis2=2).copy()
                dh_safe = np.where(dh > 0, dh, 1.0)
                damp = h + mu[act, None, None] * (dh_safe[:, :, None] * eye)
                damp = damp + blocked[:, :, None] * eye
                step = -np.linalg.solve(damp, g[..., None])[..., 0]
                out = ~blocked & (((za <= la) & (step < 0)) | ((za >= ha) & (step > 0)))
                if not out.any():
                    break
                blocked |= out
            # first-order optimality (scale-free: gradient over curvature, in parameter units)
            gstop = np.all(np.abs(g) / dh_safe <= ta * (ta + np.abs(za)), axis=1)
            z_new = np.clip(za + step, la, ha)
            r_new, beta_new, _, jacobian_new = evaluate(z_new, act)
            cost_new = 0.5 * np.sum(r_new ** 2, axis=1)
            step = z_new - za
            pred = -np.einsum("np,np->n", g, step) - 0.5 * np.einsum("np,npq,nq->n", step, h,
                                                                     step)
            rho = (cost[act] - cost_new) / np.where(pred > 0, pred, np.inf)
            better = np.isfinite(cost_new) & (cost_new <= cost[act]) & ~gstop
            moved = np.abs(z_new - za)
            xstop = better & np.all(moved <= ta * (ta + np.abs(za)), axis=1)
            fstop = better & (cost[act] - cost_new <= ta[:, 0] * np.maximum(cost[act], 1e-300))
            idx = act[better]
            z[idx], r[idx], beta[idx] = z_new[better], r_new[better], beta_new[better]
            # the Jacobian only where the iteration goes on from the new point
            going = better & ~(xstop | fstop)
            if going.any():
                j[act[going]] = jacobian_new(np.flatnonzero(going))
            cost[idx] = cost_new[better]
            # Nielsen's update of the damping from the gain ratio (overshooting steps along a
            # curved valley raise the damping instead of zig-zagging)
            mu[act] = np.where(better, mu[act] * np.maximum(MU_DOWN, 1 - (2 * np.clip(rho, 0, 1)
                                                                        - 1) ** 3),
                               mu[act] * 4.0)
            stop = gstop | xstop | fstop
            converged[act[stop]] = True
            finished[act[stop | (mu[act] > 1e10)]] = True
    else:
        converged[:] = True
    # the model at the solution: moving coefficients as in the iteration, then the fixed
    # ones by least squares on what the moving columns leave (minimum norm)
    _, beta_s, ms, _ = evaluate(z, np.arange(n), jac=False)
    rest = wx - np.matmul(ms * w[:, :, None], beta_s[..., None])[..., 0]
    if n_fix and shared:
        inv_s = np.where(s_f > 1e-10 * max(s_f[0], 1e-300), 1.0 / np.where(s_f > 0, s_f, 1.0),
                         0.0)
        beta_f = ((rest @ u_f) * inv_s) @ vt_f
    elif n_fix:
        inv_s = np.where(s_f > 1e-10 * np.maximum(s_f[:, :1], 1e-300), 1.0 / np.where(
            s_f > 0, s_f, 1.0), 0.0)
        beta_f = np.matmul(np.swapaxes(vt_f, 1, 2),
                           (inv_s * np.matmul(np.swapaxes(u_f, 1, 2),
                                              rest[..., None])[..., 0])[..., None])[..., 0]
    else:
        beta_f = np.zeros((n, 0))
    fitted = np.matmul(ms, beta_s[..., None])[..., 0] + \
        np.matmul(fcols, beta_f[..., None])[..., 0]
    converged &= np.all(np.isfinite(z), axis=1) & np.all(np.isfinite(beta_s), axis=1) & \
        np.all(np.isfinite(beta_f), axis=1)
    span = hi - lo
    hit = (z - lo <= 1e-3 * span) | (hi - z <= 1e-3 * span)
    results, start = [], 0
    for f in infos:
        tpl, layout = f["tpl"], f["layout"]
        n_comp, q_n, n_fixed = len(tpl.shapes), len(f["lo"]), len(f["fixed"])
        rows = slice(start, start + f["n"])
        moving_k = [k for k, free in enumerate(layout) if free]
        support = f["support"]
        keep = f["valid"] if support is None else f["valid"] & np.asarray(support)[None]
        # r² on the kept samples (support of the reported components), all trials at once
        x, fit_x, kf = f["data"], fitted[rows], keep.astype(float)
        count = kf.sum(axis=1)
        mean = (x * kf).sum(axis=1) / np.maximum(count, 1)
        tss = (((x - mean[:, None]) * kf) ** 2).sum(axis=1)
        rss = (((x - fit_x) * kf) ** 2).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = np.where((tss > 0) & (count > 0), 1 - rss / np.where(tss > 0, tss, 1.0), np.nan)
        owner = np.zeros((q_n, n_comp))
        owner[np.arange(q_n), moving_k] = 1.0
        at_k = (hit[rows, :q_n].astype(float) @ owner) > 0
        scale = np.zeros((f["n"], n_comp))
        scale[:, moving_k] = beta_s[rows, :q_n]
        scale[:, f["fixed"]] = beta_f[rows, :n_fixed]
        background = beta_f[rows, n_fixed:n_fixed + len(f["basis"])]
        params_k = []
        for k, free in enumerate(layout):
            pk = np.tile(np.asarray(tpl.params[k], float), (f["n"], 1))
            if free:
                pk[:, free[0]] = z[rows, moving_k.index(k)]
            params_k.append(pk)
        shifts = np.stack([pk[:, li] - c[li] for pk, c, li in zip(params_k, tpl.params,
                                                                   tpl.loc_index)], axis=1)
        results.append([{"params": [pk[i].copy() for pk in params_k], "scale": scale[i],
                         "shift": shifts[i], "background": background[i].copy(),
                         "r2": float(r2[i]), "converged": bool(converged[start + i]),
                         "at_bound": bool(at_k[i].any()), "at_bound_k": at_k[i],
                         "fitted": fit_x[i]} for i in range(f["n"])])
        start += f["n"]
    return results


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
