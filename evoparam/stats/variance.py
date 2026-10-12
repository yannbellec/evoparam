# SPDX-License-Identifier: Apache-2.0
"""Variance components of nested random effects by restricted maximum likelihood (REML).

The model is the linear mixed model with nested random intercepts and, optionally,
known sampling variances (the two- and three-level meta-analytic model)::

    y = X beta + u_1[unit at level 1] + u_2[unit at level 2] + ... + e + m
    u_l ~ N(0, sigma_l^2),  e ~ N(0, sigma_res^2),  m ~ N(0, se^2) (se known)

Levels are nested: every unit of level ``l + 1`` lies inside one unit of level ``l``
(channels inside subjects, trials inside trains). The observations of different
top-level units are independent, so the covariance is block diagonal, and each block
``V = D + Z G Z'`` (``D`` diagonal, ``G`` the random-effect variances) is inverted with
the Woodbury identity in the small dimension of its random effects. The restricted
log-likelihood is maximised over the standard deviations (bounded at zero) with L-BFGS-B;
one level without sampling variances and an intercept-only design has a closed-form
profile and is maximised in one dimension.

The random effects are predicted by their best linear unbiased predictors (BLUP),
``u = G Z' V^-1 (y - X beta)``, with the standard errors of the prediction errors
``Var(u - u_hat)``, which include the uncertainty of ``beta`` (Henderson). The shrunken
estimate of a unit is ``x_bar' beta + u_1 + ... + u_l`` (its mean design row, and the
effects on its path from the top level), with a standard error that accounts for the
covariance between ``beta`` and the BLUPs; its shrinkage weight
``1 - Var(u_l | y, beta) / sigma_l^2`` is 1 when the unit's own data dominate and 0 when
the estimate is the group value alone.

Use by the empirical-Bayes hierarchical engine (spec §5.8, step 2)
--------------------------------------------------------------------
The engine passes the trial-level estimates of one parameter (from ``evoparam.trials``)
as ``y`` with their standard errors as ``se``, and ``groups=[subject, channel]``:

* ``residual=True`` estimates the trial-to-trial variance beyond the sampling variance
  (``sigma_a`` or the jitter SD), ``variances[0]`` is the between-subject variance and
  ``variances[1]`` the between-channel variance within subjects;
* with one estimate per channel (channel averages), use ``groups=[subject, channel]`` and
  ``residual=False``, or ``groups=[subject]`` and ``residual=True`` (the channel level is
  then the residual);
* ``unit_mean[1]`` / ``unit_mean_se[1]`` are the shrunken channel estimates and their
  standard errors, ``unit_mean[0]`` / ``unit_mean_se[0]`` the subject estimates,
  ``shrinkage`` the shrinkage factors and ``n_obs`` the number of observations behind
  each unit (for the ``"prior_dominated"`` flag);
* subject covariates (``~ age``) enter through ``X``; their estimates and covariance are
  ``beta`` and ``beta_cov``.

Arrays are aligned with ``labels[level]`` (tuples of the labels of the outer levels down
to that level, in order of first appearance). :func:`variance_components` is the same
fit on a DataFrame and returns the per-unit values as tables.
"""

import time as _time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, minimize_scalar

from evoparam._typing import Any, ArrayLike
from evoparam.data import Provenance
from evoparam.errors import DataError, SettingsError
from evoparam.utils._optional import require

__all__ = ["variance_components", "VarianceResult", "reml"]


###################################################################################################
## Numerical core


def _codes(groups: Any) -> tuple[list[Any], list[Any]]:
    """Nested integer codes.

    Unit ``k`` of level ``l`` is the combination of the labels of levels ``0..l``; returns the
    codes and the labels of each unit.
    """
    codes, labels = [], []
    keys: list[tuple[str, ...]] | None = None
    for raw_column in groups:
        column = [str(v) for v in raw_column]
        keys = (
            [(k,) for k in column]
            if keys is None
            else [prev + (k,) for prev, k in zip(keys, column, strict=False)]
        )
        unique = list(dict.fromkeys(keys))
        lookup = {key: i for i, key in enumerate(unique)}
        codes.append(np.array([lookup[key] for key in keys], dtype=np.int64))
        labels.append(unique)
    return codes, labels


class _Blocks:
    """The observations of each top-level unit with their random-effect design."""

    def __init__(self, y: Any, xmat: Any, codes: Any, se2: Any) -> None:
        self.n, self.p = xmat.shape
        self.n_levels = len(codes)
        self.blocks = []
        top = codes[0]
        order = np.argsort(top, kind="stable")
        bounds = np.flatnonzero(np.diff(top[order])) + 1
        for rows in np.split(order, bounds):
            columns, level_of, unit_of = [], [], []
            for level, code in enumerate(codes):
                local = code[rows]
                for unit in dict.fromkeys(local.tolist()):
                    columns.append(local == unit)
                    level_of.append(level)
                    unit_of.append(unit)
            zmat = np.column_stack(columns).astype(float)
            self.blocks.append(
                {
                    "rows": rows,
                    "y": y[rows],
                    "X": xmat[rows],
                    "Z": zmat,
                    "se2": se2[rows],
                    "level": np.array(level_of),
                    "unit": np.array(unit_of),
                }
            )

    def solve(self, sds: Any, sd_res: Any) -> Any:
        """Accumulate X'V^-1 X, X'V^-1 y, y'V^-1 y and log|V| for the given SDs."""
        p = self.p
        xvx, xvy, yvy, logdet = np.zeros((p, p)), np.zeros(p), 0.0, 0.0
        cache = []
        for blk in self.blocks:
            d = blk["se2"] + sd_res**2
            if np.any(d <= 0):
                return None
            lam = sds[blk["level"]]
            zl = blk["Z"] * lam
            dz = zl / d[:, None]
            amat = zl.T @ dz + np.eye(len(lam))
            chol = cho_factor(amat, lower=True)
            logdet += np.sum(np.log(d)) + 2 * np.sum(np.log(np.diag(chol[0])))
            xmat, y = blk["X"], blk["y"]
            dx, dy = xmat / d[:, None], y / d
            zdx, zdy = zl.T @ dx, zl.T @ dy
            ax, ay = cho_solve(chol, zdx), cho_solve(chol, zdy)
            xvx += xmat.T @ dx - zdx.T @ ax
            xvy += xmat.T @ dy - zdx.T @ ay
            yvy += y @ dy - zdy @ ay
            cache.append((d, lam, zl, chol))
        return xvx, xvy, yvy, logdet, cache

    def objective(self, sds: Any, sd_res: Any) -> Any:
        """-2 x restricted log-likelihood (without the constant), and beta."""
        out = self.solve(sds, sd_res)
        if out is None:
            return np.inf, None
        xvx, xvy, yvy, logdet, _ = out
        try:
            chol = cho_factor(xvx, lower=True)
        except np.linalg.LinAlgError:
            return np.inf, None
        beta = cho_solve(chol, xvy)
        quad = yvy - beta @ xvy
        return logdet + 2 * np.sum(np.log(np.diag(chol[0]))) + quad, beta


def _one_level_profile(y: Any, code: Any) -> tuple[Any, Any]:
    """Closed-form REML of y_ij = mu + u_i + e_ij (one level, intercept, no se).

    Returns the variances (between units, residual) at the REML maximum.
    """
    n_units = code.max() + 1
    n_i = np.bincount(code, minlength=n_units).astype(float)
    s_i = np.bincount(code, weights=y, minlength=n_units)
    ybar = s_i / n_i
    within = np.sum((y - ybar[code]) ** 2)
    n_obs = len(y)

    def parts(gamma: Any) -> Any:
        w = n_i / (1 + gamma * n_i)  # 1' H_i^-1 1 for each unit
        mu = np.sum(w * ybar) / np.sum(w)
        quad = within + np.sum(w * (ybar - mu) ** 2)  # r' H^-1 r
        logdet = np.sum(np.log1p(gamma * n_i))
        return mu, quad, logdet, np.sum(w)

    def neg(gamma: Any) -> Any:
        _, quad, logdet, sw = parts(gamma)
        return (n_obs - 1) * np.log(quad / (n_obs - 1)) + logdet + np.log(sw)

    if within <= 0:
        raise DataError("variance_components: no variation within units.")
    # gamma = sigma_u^2 / sigma_e^2: a grid on the log scale, refined around its minimum,
    # and the boundary gamma = 0 (no variance between units)
    grid = np.exp(np.linspace(-14, 10, 97))
    values = [neg(g) for g in grid]
    best = int(np.argmin(values))
    lo, hi = np.log(grid[max(best - 1, 0)]), np.log(grid[min(best + 1, len(grid) - 1)])
    res = minimize_scalar(
        lambda lg: neg(np.exp(lg)), bounds=(lo, hi), method="bounded", options={"xatol": 1e-12}
    )
    gamma, value = (
        (np.exp(res.x), res.fun) if res.fun <= values[best] else (grid[best], values[best])
    )
    if neg(0.0) <= value:
        gamma = 0.0
    _, quad, _, _ = parts(gamma)
    sigma2 = quad / (n_obs - 1)
    return gamma * sigma2, sigma2


def reml(
    y: ArrayLike,
    groups: Any,
    *,
    se: ArrayLike | None = None,
    X: ArrayLike | None = None,  # noqa: N803  (public keyword: the fixed-effect design X)
    residual: bool = True,
    tol: float = 1e-9,
    max_iter: int = 500,
) -> dict[str, Any]:
    """Fit the nested random-intercept model by REML (arrays in, dictionary out).

    Parameters
    ----------
    y : ndarray, shape (n,)
        Observations.
    groups : sequence of array-like
        Unit labels of each level, outermost first (for example ``[subject, channel]``);
        a unit of a level is identified by its label together with those of the outer
        levels, so channel names may repeat across subjects.
    se : ndarray, shape (n,), optional
        Known standard errors of the observations (sampling variances ``se**2``).
    X : ndarray, shape (n, p), optional
        Fixed-effect design; default an intercept.
    residual : bool
        Estimate a residual variance (required without ``se``).

    Returns
    -------
    dict
        ``variances`` (one per level), ``residual`` (variance, 0 when not estimated),
        ``beta``, ``beta_cov`` (fixed effects and their covariance), and per level (lists
        of arrays aligned with ``labels[level]``): ``blup`` (predicted random effects),
        ``blup_se`` (standard errors of the prediction errors), ``unit_mean`` (shrunken
        unit estimates ``x_bar' beta + u_1 + ... + u_l``), ``unit_mean_se`` (their
        standard errors), ``shrinkage`` (weight of the unit's own data, in [0, 1]),
        ``n_obs`` (observations per unit); then ``labels`` (per level: tuples of labels),
        ``loglik`` (restricted log-likelihood), ``converged``, ``n_iter``.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> subject = np.repeat(np.arange(30), 20)
    >>> y = 2.0 * rng.normal(size=30)[subject] + rng.normal(size=600)
    >>> out = reml(y, [subject])
    >>> round(float(np.sqrt(out["variances"][0])), 1), round(float(out["residual"]), 1)
    (1.7, 1.0)
    >>> len(out["unit_mean"][0]), bool(np.all(out["shrinkage"][0] > 0.9))
    (30, True)
    """
    y = np.asarray(y, float).ravel()
    n = y.size
    groups = [np.asarray(g).ravel() for g in groups]
    if not groups:
        raise SettingsError("reml: give at least one grouping level.")
    if any(g.size != n for g in groups):
        raise DataError("reml: every grouping level needs one label per observation.")
    if not np.all(np.isfinite(y)):
        raise DataError("reml: y holds NaN or infinite values; drop them first.")
    se2 = np.zeros(n) if se is None else np.asarray(se, float).ravel() ** 2
    if se2.size != n or not np.all(np.isfinite(se2)):
        raise DataError("reml: se must be finite, one value per observation.")
    if se is None and not residual:
        raise SettingsError("reml: without se the residual variance must be estimated.")
    xmat = np.ones((n, 1)) if X is None else np.asarray(X, float).reshape(n, -1)
    p = xmat.shape[1]
    if n <= p:
        raise DataError(f"reml: {n} observations for {p} fixed effects.")
    codes, labels = _codes(groups)
    n_levels = len(codes)
    blocks = _Blocks(y, xmat, codes, se2)
    intercept_only = p == 1 and np.allclose(xmat, 1.0)

    scale = float(np.std(y)) or 1.0
    n_iter, converged = 0, True
    if n_levels == 1 and se is None and intercept_only:
        var_u, var_res = _one_level_profile(y, codes[0])
        sds, sd_res = np.array([np.sqrt(var_u)]), np.sqrt(var_res)
    else:
        n_par = n_levels + int(residual)

        def fun(theta: Any) -> Any:
            sds_, res_ = theta[:n_levels] * scale, (theta[n_levels] * scale if residual else 0.0)
            return blocks.objective(sds_, res_)[0]

        # without sampling variances, a zero residual would make V singular
        floor = 1e-6 if (residual and not np.any(se2 > 0)) else 0.0
        bounds = [(0.0, None)] * n_levels + [(floor, None)] * int(residual)
        starts = [
            np.full(n_par, 1.0 / np.sqrt(n_par)),
            np.r_[np.full(n_levels, 0.1), [1.0] * int(residual)],
        ]
        best: Any = None
        for x0 in starts:
            res = minimize(
                fun,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": max_iter, "ftol": tol, "gtol": 1e-8},
            )
            n_iter += int(res.nit)
            if best is None or res.fun < best.fun:
                best = res
        converged = bool(best.success) and np.isfinite(best.fun)
        theta = best.x
        sds = theta[:n_levels] * scale
        sd_res = theta[n_levels] * scale if residual else 0.0

    value, beta = blocks.objective(sds, sd_res)
    if beta is None:
        raise DataError("reml: the fixed-effect design is singular.")
    xvx, _, _, _, cache = blocks.solve(sds, sd_res)
    beta_cov = np.linalg.inv(xvx)
    blup, blup_se, unit_mean, unit_mean_se, shrinkage, n_obs = (
        [np.zeros(len(lab)) for lab in labels] for _ in range(6)
    )
    for blk, (d, lam, zl, chol) in zip(blocks.blocks, cache, strict=False):
        xb, zb = blk["X"], blk["Z"]
        r = blk["y"] - xb @ beta
        est = lam * cho_solve(chol, zl.T @ (r / d))
        # prediction errors (Henderson): Var(u - u_hat) = L A^-1 L + H C H' and
        # Cov(beta_hat - beta, u_hat - u) = -C H', with H = G Z' V^-1 X = L A^-1 L Z' D^-1 X
        cond = lam[:, None] * cho_solve(chol, np.eye(len(lam))) * lam[None, :]
        hmat = lam[:, None] * cho_solve(chol, zl.T @ (xb / d[:, None]))
        mmat = cond + hmat @ beta_cov @ hmat.T
        # unit means: x_bar' beta + the effects on the path from the top level to the unit
        counts = zb.sum(axis=0)
        xbar = (zb.T @ xb) / counts[:, None]
        path = ((zb.T @ zb) > 0) & (blk["level"][None, :] <= blk["level"][:, None])
        path = path.astype(float)
        mean = xbar @ beta + path @ est
        xc = xbar @ beta_cov
        mean_var = (
            np.sum(xc * xbar, axis=1)
            + np.sum((path @ mmat) * path, axis=1)
            - 2 * np.sum((xc @ hmat.T) * path, axis=1)
        )
        var_u = lam**2
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = np.where(var_u > 0, 1.0 - np.diag(cond) / var_u, 0.0)
        for level in range(n_levels):
            sel = blk["level"] == level
            units = blk["unit"][sel]
            blup[level][units] = est[sel]
            blup_se[level][units] = np.sqrt(np.clip(np.diag(mmat)[sel], 0, None))
            unit_mean[level][units] = mean[sel]
            unit_mean_se[level][units] = np.sqrt(np.clip(mean_var[sel], 0, None))
            shrinkage[level][units] = np.clip(weight[sel], 0.0, 1.0)
            n_obs[level][units] = counts[sel]
    loglik = -0.5 * (value + (n - p) * np.log(2 * np.pi))
    return {
        "variances": [float(s**2) for s in sds],
        "residual": float(sd_res**2),
        "beta": beta,
        "beta_cov": beta_cov,
        "blup": blup,
        "blup_se": blup_se,
        "unit_mean": unit_mean,
        "unit_mean_se": unit_mean_se,
        "shrinkage": shrinkage,
        "n_obs": [c.astype(np.int64) for c in n_obs],
        "labels": labels,
        "loglik": float(loglik),
        "converged": bool(converged),
        "n_iter": int(n_iter),
    }


###################################################################################################
## Table interface


@dataclass(eq=False)
class VarianceResult:
    """Variance components of a nested random-intercept model, with the shrunken unit effects.

    Attributes
    ----------
    table : DataFrame
        One row per level (outermost first, then ``"residual"`` and, with known standard
        errors, ``"sampling"`` holding the mean sampling variance): ``level``,
        ``variance``, ``sd``, ``n_units``, ``fraction`` (share of the total variance).
    fixed : DataFrame
        Fixed effects: ``term``, ``estimate``, ``se``.
    random_effects : dict of DataFrame
        Per level: the unit labels (one column per level down to it), ``n_obs``,
        ``estimate`` (BLUP of the random effect) and ``se`` (standard error of its
        prediction error), ``shrunken`` and ``shrunken_se`` (the shrunken unit estimate
        and its standard error, see :func:`reml`), ``shrinkage`` (weight of the unit's
        own data: 1 = its data alone, 0 = the group value alone).
    loglik : float
        Restricted log-likelihood.
    converged : bool
    settings : dict
    provenance : Provenance
    """

    table: Any
    fixed: Any
    random_effects: dict[str, Any]
    loglik: float
    converged: bool
    settings: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)

    def variance(self, level: Any) -> float:
        """Variance of one level (or ``"residual"``)."""
        rows = self.table[self.table["level"] == level]
        if len(rows) != 1:
            raise KeyError(f"no level {level!r}; levels: {list(self.table['level'])}.")
        return float(rows["variance"].iloc[0])


def variance_components(
    table: Any,
    *,
    value: str,
    groups: Sequence[str],
    se: str | None = None,
    fixed: Sequence[str] | None = None,
    residual: bool = True,
) -> "VarianceResult":
    """Variance at each level of a nested design, by REML.

    Parameters
    ----------
    table : DataFrame
        One row per observation (rows with a missing value are left out and counted).
    value : str
        Column of the observations.
    groups : sequence of str
        Grouping columns, outermost first (``("subject", "channel")``: channels within
        subjects, trials as the residual).
    se : str, optional
        Column of known standard errors of the observations (the meta-analytic model:
        the residual is then the variance beyond the sampling variance).
    fixed : sequence of str, optional
        Columns entered as fixed effects besides the intercept (numeric columns as they
        are, other columns as treatment-coded factors).
    residual : bool
        Estimate a residual variance (with ``se`` and one observation per unit, set it to
        False).

    Returns
    -------
    VarianceResult

    Examples
    --------
    >>> import pandas as pd
    >>> rng = np.random.default_rng(1)
    >>> subject = np.repeat(np.arange(25), 40)
    >>> df = pd.DataFrame({"subject": subject,
    ...                    "amp": 3 * rng.normal(size=25)[subject] + 5 * rng.normal(size=1000)})
    >>> vc = variance_components(df, value="amp", groups=["subject"])
    >>> list(vc.table["level"])
    ['subject', 'residual']
    """
    start = _time.perf_counter()
    pd = require("pandas", extra="data", caller="evoparam.stats.variance_components")
    groups = [groups] if isinstance(groups, str) else list(groups)
    fixed = [] if fixed is None else ([fixed] if isinstance(fixed, str) else list(fixed))
    needed = [value] + groups + fixed + ([se] if se is not None else [])
    missing = [c for c in needed if c not in table.columns]
    if missing:
        raise SettingsError(f"variance_components: missing columns {missing}.")
    keep = np.isfinite(pd.to_numeric(table[value], errors="coerce").to_numpy(float))
    if se is not None:
        keep &= np.isfinite(pd.to_numeric(table[se], errors="coerce").to_numpy(float))
    data = table.loc[keep]
    design = pd.get_dummies(data[fixed], drop_first=True, dtype=float) if fixed else None
    xmat = (
        np.ones((len(data), 1))
        if design is None
        else np.column_stack([np.ones(len(data)), design.to_numpy(float)])
    )
    names = ["(Intercept)"] + ([] if design is None else list(design.columns))
    out = reml(
        data[value].to_numpy(float),
        [data[g].to_numpy() for g in groups],
        se=None if se is None else data[se].to_numpy(float),
        X=xmat,
        residual=residual,
    )
    rows = [
        {"level": g, "variance": v, "n_units": len(lab)}
        for g, v, lab in zip(groups, out["variances"], out["labels"], strict=False)
    ]
    rows.append({"level": "residual", "variance": out["residual"], "n_units": len(data)})
    if se is not None:
        rows.append(
            {
                "level": "sampling",
                "variance": float(np.mean(data[se].to_numpy(float) ** 2)),
                "n_units": len(data),
            }
        )
    vc = pd.DataFrame(rows)
    vc["sd"] = np.sqrt(vc["variance"])
    total = vc["variance"].sum()
    vc["fraction"] = vc["variance"] / total if total > 0 else np.nan
    vc = vc[["level", "variance", "sd", "n_units", "fraction"]]
    fixed_table = pd.DataFrame(
        {"term": names, "estimate": out["beta"], "se": np.sqrt(np.diag(out["beta_cov"]))}
    )
    effects = {}
    for level, (g, lab) in enumerate(zip(groups, out["labels"], strict=False)):
        frame = pd.DataFrame(list(lab), columns=groups[: level + 1])
        frame["n_obs"] = out["n_obs"][level]
        frame["estimate"] = out["blup"][level]
        frame["se"] = out["blup_se"][level]
        frame["shrunken"] = out["unit_mean"][level]
        frame["shrunken_se"] = out["unit_mean_se"][level]
        frame["shrinkage"] = out["shrinkage"][level]
        effects[g] = frame
    settings = {
        "value": value,
        "groups": groups,
        "se": se,
        "fixed": fixed,
        "residual": residual,
        "method": "reml",
        "n_used": int(keep.sum()),
        "n_excluded": int((~keep).sum()),
    }
    provenance = Provenance().with_step(
        "evoparam.stats.variance_components", settings, duration_s=_time.perf_counter() - start
    )
    return VarianceResult(
        table=vc,
        fixed=fixed_table,
        random_effects=effects,
        loglik=out["loglik"],
        converged=out["converged"],
        settings=settings,
        provenance=provenance,
    )
