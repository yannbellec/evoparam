# SPDX-License-Identifier: Apache-2.0
"""Multi-channel parametric model with shared shape parameters (spec §5.9, method 2).

``y_c(t) = b_c(t) + sum_k A_ck s_k(t - delta_ck; theta_k)``: the shape parameters
``theta_k`` named in ``shared`` are common to the channels, the amplitudes ``A_ck`` and
the background coefficients are solved per channel by least squares (variable projection),
and, with ``free_latency``, each channel's latency moves within ``±max_shift`` of the shared
one. The per-channel fits of the engine (spec §5.5) give the starting point. An optional
graph-Laplacian penalty ``lambda sum_{c~c'} w_cc' (delta_ck - delta_c'k)²`` ties the
latency offsets of neighbouring contacts, with ``lambda`` chosen by leave-one-channel-out
prediction.

The latency of each component is handled as the reported latency (the shape's
``latency(params)``, the mode of a skewed shape) and every channel's latency, shared value
plus offset, is bounded to the fit window. The identifiability rule of the engine applies:
a model in which two components of opposite signs overlap on a channel (latencies closer
than ``overlap_c`` times the sum of their sigmas) is not accepted; one member of the pair
is dropped (the one whose removal fits best) and the model is refitted.
"""

import time as _time
import warnings
from collections import Counter
from typing import cast

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.stats import norm

from evoparam._typing import Any, FloatArray, NDArray
from evoparam.data import Component, Evoked, GoodnessOfFit, GroupFit, ResponseFit, TrialSet
from evoparam.errors import DataError, SettingsError
from evoparam.errors import warn as _warn
from evoparam.models import ResponseModel
from evoparam.models.component_shapes import (
    FWHM_PER_SIGMA,
    compute_features,
    get_background,
    get_shape,
)
from evoparam.models.engine import cancelling_pairs
from evoparam.version import __version__

__all__ = ["fit_shared", "neighbours", "PARAMETER_ROLES"]

# Role of each shape parameter, so that ``shared`` can name roles ("latency", "fwhm",
# "asymmetry") instead of the parameter names of each shape. The location parameter of
# every shape has the role "latency".
PARAMETER_ROLES = {
    "sigma": "fwhm",
    "omega": "fwhm",
    "theta": "fwhm",
    "tau_rise": "fwhm",
    "stretch": "fwhm",
    "alpha": "asymmetry",
    "tau": "asymmetry",
    "kappa": "asymmetry",
    "tau_decay": "asymmetry",
    "freq": "frequency",
    "phase": "phase",
}
LAMBDA_FACTORS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
MAX_ROUNDS = 10
LAMBDA_MARGIN = 0.01


###################################################################################################
## Neighbours


def neighbours(ch_info: Any, names: Any, radius_mm: float = 10.0) -> Any:
    """Return the adjacency weights [n, n] of the contacts within ``radius_mm`` of each other.

    From the ``x, y, z`` columns of ``ch_info``; None when coordinates are missing.

    >>> import pandas as pd
    >>> info = pd.DataFrame({"name": ["a", "b", "c"], "x": [0., 5., 30.], "y": 0., "z": 0.})
    >>> neighbours(info, ["a", "b", "c"]).tolist()
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    """
    if ch_info is None or not {"name", "x", "y", "z"} <= set(ch_info.columns):
        return None
    table = ch_info.set_index(ch_info["name"].astype(str))
    try:
        coords = table.loc[list(names), ["x", "y", "z"]].to_numpy(float)
    except KeyError:
        return None
    if not np.all(np.isfinite(coords)):
        return None
    dist = np.linalg.norm(coords[:, None] - coords[None], axis=2)
    return ((dist <= radius_mm) & (dist > 0)).astype(float)


###################################################################################################
## The model


class _Channel:
    """Data, background and fixed parts of one channel in the window."""

    def __init__(
        self, y: Any, valid: Any, sigma: Any, background: Any, bg_nonlinear: Any, tw: Any
    ) -> None:
        self.y = np.where(valid, y, 0.0)
        self.w = valid.astype(float) / sigma
        self.sigma = sigma
        self.background = background
        self.bg_nonlinear = np.asarray(bg_nonlinear, float)
        self.t_ref = float(background._t_ref(tw)) if hasattr(background, "t_ref") else None
        nl = bg_nonlinear if len(bg_nonlinear) else None
        self.basis = (
            np.atleast_2d(background.basis(tw, nl))
            if background.n_linear
            else np.zeros((0, len(tw)))
        )


class _Layout:
    """Which parameter of which component is shared, local or a latency offset.

    Local and offset parameters are per channel; offsets are within ±max_shift.
    """

    def __init__(self, shapes: Any, shared: Any, free_latency: Any) -> None:
        self.shapes = shapes
        self.shared_idx, self.local_idx = [], []
        for k, shape in enumerate(shapes):
            for i, name in enumerate(shape.param_names):
                role = (
                    "latency" if name == shape.latency_param else PARAMETER_ROLES.get(name, name)
                )
                is_shared = name in shared or role in shared
                if name == shape.latency_param:
                    # The latency is shared (with per-channel offsets when free_latency) or,
                    # when not in ``shared``, local
                    if is_shared:
                        self.shared_idx.append((k, i))
                    else:
                        self.local_idx.append((k, i))
                elif is_shared:
                    self.shared_idx.append((k, i))
                else:
                    self.local_idx.append((k, i))
        self.free_latency = free_latency
        self.shared = shared
        self.loc = [shape.param_names.index(shape.latency_param) for shape in shapes]
        # A latency offset per channel and component when the latency is shared
        self.offset_k = [
            k for k in range(len(shapes)) if free_latency and (k, self.loc[k]) in self.shared_idx
        ]

    def n_local(self) -> int:
        return len(self.local_idx) + len(self.offset_k)

    def params(self, base: Any, z_shared: Any, z_local: Any) -> list[NDArray[Any]]:
        """Return the parameters of each component of one channel.

        ``base`` and the ``z`` hold the latency in place of the location parameter, see
        :func:`_to_latency`.
        """
        params = [np.array(p, float) for p in base]
        for (k, i), value in zip(self.shared_idx, z_shared, strict=False):
            params[k][i] = value
        n = len(self.local_idx)
        for (k, i), value in zip(self.local_idx, z_local[:n], strict=False):
            params[k][i] = value
        for k, delta in zip(self.offset_k, z_local[n:], strict=False):
            params[k][self.loc[k]] += delta
        return [_from_latency(shape, p) for shape, p in zip(self.shapes, params, strict=False)]

    def restricted(
        self, keep: Any, z_shared: Any, z_locals: Any
    ) -> tuple[Any, NDArray[Any], list[NDArray[Any]]]:
        """Return the layout of the components ``keep`` and the matching parameter vectors."""
        layout = _Layout([self.shapes[k] for k in keep], self.shared, self.free_latency)
        z_shared = np.array(
            [v for (k, _), v in zip(self.shared_idx, z_shared, strict=False) if k in keep], float
        )
        n = len(self.local_idx)
        sel = [j for j, (k, _) in enumerate(self.local_idx) if k in keep] + [
            n + j for j, k in enumerate(self.offset_k) if k in keep
        ]
        return layout, z_shared, [np.asarray(z, float)[sel] for z in z_locals]


def _latency_shift(shape: Any, params: Any) -> float:
    """Return the reported latency minus the location parameter.

    It does not depend on the location (the shapes are translation invariant); zero for a shape
    without ``latency``.
    """
    lat = getattr(shape, "latency", None)
    if lat is None:
        return 0.0
    p = np.array(params, float)
    p[shape.param_names.index(shape.latency_param)] = 0.0
    return float(lat(p))


def _to_latency(shape: Any, params: Any) -> NDArray[Any]:
    """Parameters with the location parameter replaced by the reported latency."""
    p = np.array(params, float)
    p[shape.param_names.index(shape.latency_param)] += _latency_shift(shape, p)
    return p


def _from_latency(shape: Any, params: Any) -> NDArray[Any]:
    p = np.array(params, float)
    p[shape.param_names.index(shape.latency_param)] -= _latency_shift(shape, p)
    return p


def _solve(channel: Any, shapes: Any, params: Any, tw: Any) -> tuple[Any, NDArray[Any], Any]:
    """Amplitudes and background coefficients by weighted least squares.

    Returns the coefficients, the design [n_times, p] and the weighted residual.
    """
    cols = [shape.evaluate(tw, p) for shape, p in zip(shapes, params, strict=False)]
    design = np.column_stack(cols + list(channel.basis))
    a = design * channel.w[:, None]
    beta, *_ = np.linalg.lstsq(a, channel.y * channel.w, rcond=None)
    return beta, design, (channel.y - design @ beta) * channel.w


def _bounds(shapes: Any, tw: Any, index: Any) -> tuple[NDArray[Any], NDArray[Any]]:
    """Bounds of the parameters ``index``; a latency is bounded to the fit window."""
    lo, hi = [], []
    for k, i in index:
        if shapes[k].param_names[i] == shapes[k].latency_param:
            lo.append(tw[0])
            hi.append(tw[-1])
            continue
        s_lo, s_hi = shapes[k].default_bounds(tw)
        lo.append(s_lo[i])
        hi.append(s_hi[i])
    return np.array(lo, float), np.array(hi, float)


def _shared_latency(layout: Any, z_shared: Any, k: Any) -> float:
    return float(z_shared[layout.shared_idx.index((k, layout.loc[k]))])


def _local_bounds(layout: Any, tw: Any, max_shift: Any, z_shared: Any) -> tuple[Any, Any]:
    """Bounds of the per-channel parameters.

    A latency offset keeps the channel's latency (shared latency plus offset) in the fit window.
    """
    lo, hi = _bounds(layout.shapes, tw, layout.local_idx)
    off_lo, off_hi = [], []
    for k in layout.offset_k:
        latency = _shared_latency(layout, z_shared, k)
        off_lo.append(max(-max_shift, tw[0] - latency))
        off_hi.append(max(min(max_shift, tw[-1] - latency), off_lo[-1] + 1e-12))
    return np.r_[lo, off_lo], np.r_[hi, off_hi]


def _shared_bounds(layout: Any, tw: Any, z_locals: Any) -> tuple[NDArray[Any], NDArray[Any]]:
    """Bounds of the shared parameters.

    A shared latency keeps every channel's latency (with its offset) in the fit window.
    """
    lo, hi = _bounds(layout.shapes, tw, layout.shared_idx)
    first = layout.n_local() - len(layout.offset_k)
    for j, k in enumerate(layout.offset_k):
        offsets = np.array([z[first + j] for z in z_locals], float)
        i = layout.shared_idx.index((k, layout.loc[k]))
        lo[i] = max(lo[i], tw[0] - offsets.min())
        hi[i] = max(min(hi[i], tw[-1] - offsets.max()), lo[i] + 1e-12)
    return lo, hi


def _clip(z: Any, lo: Any, hi: Any) -> Any:
    span = hi - lo
    return np.clip(z, lo + 1e-9 * span, hi - 1e-9 * span)


###################################################################################################
## Fitting steps


def _shared_step(
    channels: Any, layout: Any, base: Any, z_shared: Any, z_locals: Any, tw: Any
) -> Any:
    if not len(z_shared):
        return z_shared
    lo, hi = _shared_bounds(layout, tw, z_locals)

    def residual(z: Any) -> Any:
        return np.concatenate(
            [
                _solve(ch, layout.shapes, layout.params(base, z, zl), tw)[2]
                for ch, zl in zip(channels, z_locals, strict=False)
            ]
        )

    res = least_squares(
        residual,
        _clip(z_shared, lo, hi),
        bounds=(lo, hi),
        method="trf",
        x_scale="jac",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=200,
    )
    return res.x


def _local_step(
    channels: Any,
    layout: Any,
    base: Any,
    z_shared: Any,
    z_locals: Any,
    tw: Any,
    max_shift: Any,
    laplacian: Any = None,
    lam: float = 0.0,
    z_start: Any = None,
) -> Any:
    """Per-channel parameters; with a penalty, the latency offsets of all channels jointly."""
    n_loc = layout.n_local()
    if n_loc == 0:
        return z_locals
    lo, hi = _local_bounds(layout, tw, max_shift, z_shared)
    start = z_locals if z_start is None else z_start
    if laplacian is None or lam == 0:
        out = []
        for ch, z0 in zip(channels, start, strict=False):
            res = least_squares(
                lambda z, ch=ch: _solve(ch, layout.shapes, layout.params(base, z_shared, z), tw)[
                    2
                ],
                _clip(z0, lo, hi),
                bounds=(lo, hi),
                method="trf",
                x_scale="jac",
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
                max_nfev=200,
            )
            out.append(res.x)
        return out
    n_ch = len(channels)
    n_t = len(tw)
    edges = np.argwhere(np.triu(laplacian > 0, 1))
    n_off = len(layout.offset_k)
    first = n_loc - n_off

    def residual(zz: Any) -> Any:
        z = zz.reshape(n_ch, n_loc)
        parts = [
            _solve(ch, layout.shapes, layout.params(base, z_shared, zc), tw)[2]
            for ch, zc in zip(channels, z, strict=False)
        ]
        if len(edges):
            d = z[edges[:, 0], first:] - z[edges[:, 1], first:]
            parts.append(np.sqrt(lam * laplacian[edges[:, 0], edges[:, 1]])[:, None] * d)
        return np.concatenate([np.ravel(p) for p in parts])

    sparsity = lil_matrix((n_ch * n_t + len(edges) * n_off, n_ch * n_loc), dtype=int)
    for c in range(n_ch):
        sparsity[c * n_t : (c + 1) * n_t, c * n_loc : (c + 1) * n_loc] = 1
    for e, (a, b) in enumerate(edges):
        for j in range(n_off):
            row = n_ch * n_t + e * n_off + j
            sparsity[row, a * n_loc + first + j] = 1
            sparsity[row, b * n_loc + first + j] = 1
    lo_all, hi_all = np.tile(lo, n_ch), np.tile(hi, n_ch)
    z0 = _clip(np.concatenate(start), lo_all, hi_all)
    res = least_squares(
        residual,
        z0,
        bounds=(lo_all, hi_all),
        method="trf",
        x_scale="jac",
        jac_sparsity=sparsity,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=200,
    )
    return list(res.x.reshape(n_ch, n_loc))


def _cost(channels: Any, layout: Any, base: Any, z_shared: Any, z_locals: Any, tw: Any) -> float:
    return float(
        sum(
            np.sum(_solve(ch, layout.shapes, layout.params(base, z_shared, zl), tw)[2] ** 2)
            for ch, zl in zip(channels, z_locals, strict=False)
        )
    )


def _alternate(
    channels: Any,
    layout: Any,
    base: Any,
    z_shared: Any,
    z_locals: Any,
    tw: Any,
    max_shift: Any,
    laplacian: Any = None,
    lam: float = 0.0,
) -> tuple[Any, Any, float, int]:
    cost = _cost(channels, layout, base, z_shared, z_locals, tw)
    rounds = 0
    for rounds in range(1, MAX_ROUNDS + 1):  # noqa: B007  (the count is read after the loop)
        z_shared = _shared_step(channels, layout, base, z_shared, z_locals, tw)
        z_locals = _local_step(
            channels, layout, base, z_shared, z_locals, tw, max_shift, laplacian, lam
        )
        new = _cost(channels, layout, base, z_shared, z_locals, tw)
        if layout.n_local() == 0 or abs(cost - new) <= 1e-6 * max(new, 1e-300):
            cost = new
            break
        cost = new
    return z_shared, z_locals, cost, rounds


###################################################################################################
## Leave-one-channel-out choice of the penalty


def _offset_curvature(
    channels: Any, layout: Any, base: Any, z_shared: Any, z_locals: Any, tw: Any, max_shift: Any
) -> FloatArray:
    """Return the curvature of each channel's weighted residual sum of squares in each offset.

    In each latency offset, by Gauss-Newton with central differences.
    """
    n_off = len(layout.offset_k)
    first = layout.n_local() - n_off
    h = np.zeros((len(channels), n_off))
    step = max(1e-4, 0.05 * max_shift)
    for c, (ch, z) in enumerate(zip(channels, z_locals, strict=False)):
        for j in range(n_off):
            zp, zm = z.copy(), z.copy()
            zp[first + j] += step
            zm[first + j] -= step
            rp = _solve(ch, layout.shapes, layout.params(base, z_shared, zp), tw)[2]
            rm = _solve(ch, layout.shapes, layout.params(base, z_shared, zm), tw)[2]
            jac = (rp - rm) / (2 * step)
            h[c, j] = float(jac @ jac)
    return h


def _smooth_offsets(delta_hat: Any, h: Any, weights: Any, lam: Any) -> Any:
    """Return the minimiser of sum_c h_c (d_c - d̂_c)² + lam sum_{c~c'} w (d_c - d_c')².

    Per component (the quadratic approximation of the penalised problem).
    """
    lap = np.diag(weights.sum(axis=1)) - weights
    out = np.zeros_like(delta_hat)
    for j in range(delta_hat.shape[1]):
        a = np.diag(h[:, j]) + lam * lap
        out[:, j] = np.linalg.lstsq(a, h[:, j] * delta_hat[:, j], rcond=None)[0]
    return out


def _loco(
    channels: Any,
    layout: Any,
    base: Any,
    z_shared: Any,
    z_locals: Any,
    tw: Any,
    weights: Any,
    max_shift: Any,
) -> tuple[list[float], FloatArray, FloatArray]:
    """Held-out error of each lambda.

    Each channel in turn is left out, the offsets of the others are smoothed with lambda, and the
    left-out channel's offset is predicted from them through the penalty:
    ``lambda sum_n w (d - d_n) / (lambda sum_n w + rho)``, where ``rho`` (1e-3 of the median
    curvature) is a weak pull towards the shared latency, so that without coupling (lambda = 0) the
    prediction is the shared latency itself. The left-out channel's data are then fitted with that
    offset (amplitudes and background re-solved) and the weighted residual sums of squares are
    added over the channels.
    """
    n_off = len(layout.offset_k)
    first = layout.n_local() - n_off
    delta_hat = np.array([z[first:] for z in z_locals])
    h = _offset_curvature(channels, layout, base, z_shared, z_locals, tw, max_shift)
    scale = float(np.median(h[h > 0])) if np.any(h > 0) else 1.0
    rho = 1e-3 * scale
    grid = [factor * scale for factor in LAMBDA_FACTORS]
    errors = np.zeros(len(grid))
    n = len(channels)
    for g, lam in enumerate(grid):
        for c in range(n):
            keep = np.flatnonzero(np.arange(n) != c)
            sub = weights[np.ix_(keep, keep)]
            smoothed = (
                _smooth_offsets(delta_hat[keep], h[keep], sub, lam) if lam > 0 else delta_hat[keep]
            )
            w = weights[c, keep]
            pred = lam * (w @ smoothed) / (lam * w.sum() + rho)
            z = z_locals[c].copy()
            lo, hi = _local_bounds(layout, tw, max_shift, z_shared)
            z[first:] = np.clip(pred, lo[first:], hi[first:])
            r = _solve(channels[c], layout.shapes, layout.params(base, z_shared, z), tw)[2]
            errors[g] += float(r @ r)
    return grid, errors, h


def _choose_lambda(grid: Any, errors: Any) -> Any:
    """Return the lambda of smallest held-out error, if it clearly improves on lambda = 0.

    It must improve by more than LAMBDA_MARGIN (relative): a penalty that does not clearly help is
    switched off.
    """
    best = int(np.argmin(errors))
    if errors[best] < (1 - LAMBDA_MARGIN) * errors[0]:
        return grid[best]
    return 0.0


###################################################################################################
## Results


def _cancelling(
    channels: Any,
    layout: Any,
    base: Any,
    z_shared: Any,
    z_locals: Any,
    tw: Any,
    settings: Any,
    sfreq: float,
) -> Any:
    """Count the component pairs that cancel, by number of channels on which they do.

    Engine rule, ``cancelling_pairs``.
    """
    found: Counter[Any]
    found = Counter()
    for ch, z in zip(channels, z_locals, strict=False):
        params = layout.params(base, z_shared, z)
        beta = _solve(ch, layout.shapes, params, tw)[0]
        lat, sig = [], []
        for k, (shape, p) in enumerate(zip(layout.shapes, params, strict=False)):
            f = compute_features(shape, tw, p, beta[k], sfreq=sfreq)
            lat.append(f.latency)
            sig.append(f.fwhm / FWHM_PER_SIGMA if np.isfinite(f.fwhm) else 0.0)
        found.update(cancelling_pairs(lat, sig, beta[: len(layout.shapes)], settings.overlap_c))
    return found


def _gof(y: Any, fitted: Any, valid: Any, sigma: Any, n_params: int) -> "GoodnessOfFit":
    yv, rv = y[valid], (y - fitted)[valid]
    n = len(yv)
    rss = float(np.sum(rv**2))
    tss = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    rmse = float(np.sqrt(rss / n)) if n else np.nan
    span = float(yv.max() - yv.min()) if n else np.nan
    loglik = n * np.log(max(rss / max(n, 1), 1e-300))
    aic = loglik + 2 * n_params
    aicc = aic + 2 * n_params * (n_params + 1) / max(n - n_params - 1, 1)
    lag = float(np.corrcoef(rv[1:], rv[:-1])[0, 1]) if n > 3 and np.std(rv) > 0 else np.nan
    return GoodnessOfFit(
        r2=r2,
        adj_r2=1 - (1 - r2) * (n - 1) / max(n - n_params - 1, 1) if np.isfinite(r2) else np.nan,
        rmse=rmse,
        mae=float(np.mean(np.abs(rv))) if n else np.nan,
        nrmse=rmse / span if span else np.nan,
        aic=aic,
        aicc=aicc,
        bic=loglik + n_params * np.log(max(n, 1)),
        n_params=n_params,
        n_samples=n,
        n_eff=float(n),
        residual_lag1_autocorr=lag,
        baseline_sd=sigma,
    )


def _channel_fit(
    ch: Any,
    layout: Any,
    base: Any,
    z_shared: Any,
    z_local: Any,
    tw: Any,
    y: Any,
    valid: Any,
    settings: Any,
    sfreq: float,
    settings_dict: Any,
    warns: Any,
    provenance: Any,
    n_shared: Any,
) -> "ResponseFit":
    shapes = layout.shapes
    params = layout.params(base, z_shared, z_local)
    beta, design, _ = _solve(ch, shapes, params, tw)
    fitted = design @ beta
    a = design * ch.w[:, None]
    cov = np.linalg.pinv(a.T @ a)
    z = norm.ppf(0.5 + settings.ci_level / 2)
    comps = []
    for k, (shape, p) in enumerate(zip(shapes, params, strict=False)):
        amp = float(beta[k])
        f = compute_features(
            shape,
            tw,
            p,
            amp,
            onset_fraction=settings.onset_fraction,
            sfreq=sfreq,
            window=(tw[0], tw[-1]),
        )
        se = float(np.sqrt(max(cov[k, k], 0.0)))
        comps.append(
            Component(
                shape=shape.name,
                params=p,
                param_names=shape.param_names,
                latency=f.latency,
                amplitude=f.amplitude,
                polarity=1 if f.amplitude >= 0 else -1,
                fwhm=f.fwhm,
                onset=f.onset,
                offset=f.offset,
                area=f.area,
                rise_time=f.rise_time,
                decay_time=f.decay_time,
                symmetry=f.symmetry,
                sharpness=f.sharpness,
                se={"A": se},
                ci={"A": (amp - z * se, amp + z * se)},
            )
        )
    bg = ch.background
    values = dict(zip(bg.linear_names, beta[len(shapes) :], strict=False))
    # the time constants (held from the channel's own fit) and the time origin with the
    # coefficients, so that evaluate_fit gives this channel's fitted background
    values.update(zip(getattr(bg, "nonlinear_names", ()), ch.bg_nonlinear, strict=True))
    background = {"name": bg.name, **{key: float(v) for key, v in values.items()}}
    if ch.t_ref is not None:
        background["t_ref"] = ch.t_ref
    n_params = len(beta) + layout.n_local() + n_shared
    return ResponseFit(
        components=comps,
        background=background,
        gof=_gof(y, fitted, valid, ch.sigma, n_params),
        status="ok",
        message="",
        window=settings.fit_window,
        baseline=settings.baseline if settings.baseline is not None else (np.nan, np.nan),
        settings=settings_dict,
        model=fitted,
        residual=y - fitted,
        warnings=warns,
        provenance=provenance,
    )


###################################################################################################
## Entry point


def _shape_like(
    name: Any, settings: Any
) -> Any:  # a ComponentShape (shapes is reached through models only)
    import dataclasses

    shape = get_shape(name)
    fields = {f.name for f in dataclasses.fields(cast(Any, type(shape)))}
    kwargs = {k: getattr(settings, k) for k in ("min_fwhm", "max_fwhm") if k in fields}
    return get_shape(name, **kwargs)


def fit_shared(  # noqa: C901, PLR0912, PLR0913, PLR0915  (public API, signature fixed by spec §5.9)
    ts: "TrialSet | Evoked",
    *,
    channels: list[str] | None = None,
    preset: Any = None,
    shared: Any = ("latency", "fwhm", "asymmetry"),
    free_latency: bool = False,
    max_shift: float = 0.005,
    smooth: bool = False,
    radius_mm: float = 10.0,
    smooth_lambda: Any = None,
    group_fit: Any = None,
    reject_cancelling: bool = True,
    **model_kw: Any,
) -> "GroupFit":
    """Fit the channels with component shapes shared across them.

    The per-channel fits of the engine (``ResponseModel(preset, **model_kw)``, or
    ``group_fit``) give the number of components (the modal ``K`` of the channels fitted
    with status ``"ok"``; channels with another ``K`` are fitted with the modal one and
    flagged ``"k_mismatch:<K>"``), the shape of each component (by latency order) and the
    starting values (the median over the channels). Then, alternately: the shared
    parameters by bounded least squares over the residuals of all channels (each divided
    by the channel's noise SD), and the per-channel ones (parameters not in ``shared``,
    and latency offsets within ``±max_shift`` when ``free_latency``); amplitudes and
    background coefficients by variable projection, the background time constants fixed
    at the per-channel values.

    ``shared`` names parameters or roles: ``"latency"``, ``"fwhm"``, ``"asymmetry"``
    (see :data:`PARAMETER_ROLES`). ``smooth=True`` (needs ``free_latency``) adds the
    graph-Laplacian penalty on the latency offsets of contacts within ``radius_mm`` (from
    ``ch_info`` ``x, y, z``); ``lambda`` is chosen on a grid (multiples of the median
    curvature of the channels' fit in their offset, 0 included) by leave-one-channel-out
    prediction, unless ``smooth_lambda`` is given.

    Parameters
    ----------
    ts : TrialSet or Evoked
    channels : list of str, optional
        Default: the channels that are not bad nor stimulating contacts (``ch_info``).
    reject_cancelling : bool
        Apply the identifiability rule to the shared model (default True): while two
        components of opposite signs overlap on a channel, the member whose removal fits
        best is dropped and the model refitted (``settings["fit_shared"]["dropped"]``).
        This rule belongs to the shared fit (which has no reseeding); the per-channel fits
        follow the engine's own ``reject_cancelling`` and ``reseed`` settings, passed in
        ``model_kw``.

    Returns
    -------
    GroupFit
        One ResponseFit per channel (condition ""); ``settings["fit_shared"]`` holds the
        shared parameters, the modal K, the flagged channels, the per-channel offsets and
        the penalty grid with its held-out errors.
    """
    start = _time.perf_counter()
    if isinstance(ts, TrialSet):
        evoked = ts.average()
    elif isinstance(ts, Evoked):
        evoked = ts
    else:
        raise DataError("fit_shared expects a TrialSet or an Evoked.")
    if smooth and not free_latency:
        raise SettingsError("smooth=True penalises the latency offsets: set free_latency=True.")
    if max_shift <= 0:
        raise SettingsError("max_shift must be positive.")
    names = list(evoked.ch_names)
    info = ts.ch_info
    if channels is None:
        drop = set()
        if info is not None:
            for column in ("bad", "stim_contact"):
                if column in info.columns:
                    drop |= {
                        n
                        for n, v in zip(info["name"].astype(str), info[column], strict=False)
                        if bool(v)
                    }
        channels = [n for n in names if n not in drop]
    channels = [names[c] if not isinstance(c, str) else c for c in channels]
    if len(channels) < 2:
        raise DataError("fit_shared needs at least two channels.")

    model = ResponseModel(preset, **model_kw)
    settings = model.settings()
    if group_fit is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            group_fit = model.fit_group(ts, picks=channels)
    else:
        settings = model.settings(fit_window=tuple(group_fit.fits[0].window))
    independent = {ch: group_fit.get(ch) for ch in channels}
    counts = Counter(
        len(f.components) for f in independent.values() if f.status == "ok" and f.components
    )
    if not counts:
        raise DataError("fit_shared: no channel has a fitted component.")
    modal_k = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    reference = [
        ch
        for ch in channels
        if independent[ch].status == "ok" and len(independent[ch].components) == modal_k
    ]
    shape_names = [
        Counter(independent[ch].components[k].shape for ch in reference).most_common(1)[0][0]
        for k in range(modal_k)
    ]
    shapes = [_shape_like(name, settings) for name in shape_names]
    base = []
    for k, name in enumerate(shape_names):
        # latency in place of the location parameter (the coordinates of the fit)
        values = [
            _to_latency(shapes[k], independent[ch].components[k].params)
            for ch in reference
            if independent[ch].components[k].shape == name
        ]
        base.append(np.median(np.array(values), axis=0))
    layout = _Layout(shapes, set(shared), free_latency)
    original = list(range(len(shapes)))  # index in the modal-K model of each component

    window = settings.fit_window
    if window is None:
        window = tuple(independent[reference[0]].window)
    win = (evoked.times >= window[0]) & (evoked.times <= window[1])
    tw = evoked.times[win]
    chans, ys, valids, flagged = [], [], [], {}
    for name in channels:
        c: Any
        c = names.index(name)
        fit = independent[name]
        y = evoked.data[c, win]
        valid = np.isfinite(y)
        if evoked.masks is not None:
            valid &= ~evoked.masks[c, win]
        sigma = (
            fit.gof.baseline_sd
            if fit.gof is not None and np.isfinite(fit.gof.baseline_sd) and fit.gof.baseline_sd > 0
            else float(np.nanstd(y)) or 1.0
        )
        bg = fit.background or {}
        background: Any
        if "name" in bg:
            background = get_background(bg["name"])
            nl = np.array([bg[n] for n in background.nonlinear_names], float)
        else:
            background, nl = get_background("constant"), np.zeros(0)
        chans.append(_Channel(y, valid, sigma, background, nl, tw))
        ys.append(np.nan_to_num(y))
        valids.append(valid)
        if not (fit.status == "ok" and len(fit.components) == modal_k):
            flagged[name] = len(fit.components) if fit.status == "ok" else 0

    lo, hi = _bounds(shapes, tw, layout.shared_idx)
    z_shared = _clip(np.array([base[k][i] for k, i in layout.shared_idx], float), lo, hi)
    lo, hi = _local_bounds(layout, tw, max_shift, z_shared)
    z_locals = []
    for name in channels:
        fit = independent[name]
        z = []
        for k, i in layout.local_idx:
            ok = name not in flagged and fit.components[k].shape == shape_names[k]
            z.append(_to_latency(shapes[k], fit.components[k].params)[i] if ok else base[k][i])
        for k in layout.offset_k:
            ok = name not in flagged and fit.components[k].shape == shape_names[k]
            z.append(
                _to_latency(shapes[k], fit.components[k].params)[layout.loc[k]]
                - base[k][layout.loc[k]]
                if ok
                else 0.0
            )
        z_locals.append(_clip(np.array(z, float), lo, hi) if z else np.zeros(0))

    z_shared, z_locals, cost, rounds = _alternate(
        chans, layout, base, z_shared, z_locals, tw, max_shift
    )
    dropped = []
    while reject_cancelling and len(layout.shapes) > 1:
        pairs = _cancelling(chans, layout, base, z_shared, z_locals, tw, settings, evoked.sfreq)
        if not pairs:
            break
        # the pair found on most channels; the member whose removal fits best is dropped
        (a, b), n_channels = pairs.most_common(1)[0]
        best: Any = None
        for drop in (a, b):
            keep = [k for k in range(len(layout.shapes)) if k != drop]
            zl: Any
            zs: Any
            sub, zs, zl = layout.restricted(keep, z_shared, z_locals)
            sub_base = [base[k] for k in keep]
            r: Any
            zs, zl, c, r = _alternate(chans, sub, sub_base, zs, zl, tw, max_shift)
            if best is None or c < best[0]:
                best = (c, drop, keep, sub, sub_base, zs, zl, r)
        cost, drop, keep, layout, base, z_shared, z_locals, r = best
        rounds += r
        dropped.append(
            {
                "component": int(original[drop]),
                "shape": shape_names[original[drop]],
                "reason": "cancelling_pair",
                "n_channels": int(n_channels),
            }
        )
        original = [original[k] for k in keep]
    shapes = layout.shapes
    smooth_info = {}
    if smooth:
        weights = neighbours(info, channels, radius_mm)
        if weights is None or not weights.any():
            _warn(
                "fit_shared: no neighbouring contacts (coordinates missing or no "
                f"pair within {radius_mm} mm); smooth=True has no effect.",
                UserWarning,
                stacklevel=2,
            )
            smooth_info = {"lambda": 0.0, "reason": "no_neighbours"}
        else:
            grid, errors, h = _loco(
                chans, layout, base, z_shared, z_locals, tw, weights, max_shift
            )
            lam = _choose_lambda(grid, errors) if smooth_lambda is None else float(smooth_lambda)
            smooth_info = {
                "lambda": float(lam),
                "grid": [float(g) for g in grid],
                "loco_error": errors.tolist(),
                "n_edges": int(np.triu(weights > 0, 1).sum()),
                "radius_mm": float(radius_mm),
            }
            if lam > 0:
                n_off = len(layout.offset_k)
                first = layout.n_local() - n_off
                delta_hat = np.array([z[first:] for z in z_locals])
                start_offsets = _smooth_offsets(delta_hat, h, weights, lam)
                z_start: Any
                z_start = [
                    np.r_[z[:first], d] for z, d in zip(z_locals, start_offsets, strict=False)
                ]
                for _ in range(3):
                    z_locals = _local_step(
                        chans,
                        layout,
                        base,
                        z_shared,
                        z_locals,
                        tw,
                        max_shift,
                        weights,
                        lam,
                        z_start,
                    )
                    z_start = None
                    z_shared = _shared_step(chans, layout, base, z_shared, z_locals, tw)

    # the latency (the shape's reported latency) under "latency", the others by name
    shared_values = {
        f"{k}:" + ("latency" if i == layout.loc[k] else shapes[k].param_names[i]): float(v)
        for (k, i), v in zip(layout.shared_idx, z_shared, strict=False)
    }
    offsets = {
        name: [float(v) for v in z[layout.n_local() - len(layout.offset_k) :]]
        for name, z in zip(channels, z_locals, strict=False)
    }
    summary = {
        "shared": sorted(shared),
        "shared_values": shared_values,
        "modal_k": modal_k,
        "shapes": [shape_names[k] for k in original],
        "modal_shapes": shape_names,
        "dropped": dropped,
        "reject_cancelling": bool(reject_cancelling),
        "flagged": flagged,
        "free_latency": bool(free_latency),
        "max_shift": float(max_shift),
        "offsets": offsets,
        "rounds": rounds,
        "cost": cost,
        "smooth": smooth_info,
        "channels": channels,
    }
    settings_dict = {
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in settings.to_dict().items()},
        "package_version": __version__,
        "fit_shared": summary,
    }
    provenance = evoked.provenance.with_step(
        "evoparam.spatial.fit_shared",
        {k: v for k, v in summary.items() if k not in ("offsets", "smooth")},
        duration_s=_time.perf_counter() - start,
    )
    n_shared = len(z_shared)
    fits = []
    for name, ch, z, y, valid in zip(channels, chans, z_locals, ys, valids, strict=False):
        warns = [f"k_mismatch:{flagged[name]}"] if name in flagged else []
        fits.append(
            _channel_fit(
                ch,
                layout,
                base,
                z_shared,
                z,
                tw,
                y,
                valid,
                settings,
                evoked.sfreq,
                settings_dict,
                warns,
                provenance,
                n_shared,
            )
        )
    return GroupFit.from_fits(
        fits, channels, [""] * len(fits), settings=settings_dict, provenance=provenance
    )
