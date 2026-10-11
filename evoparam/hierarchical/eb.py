# SPDX-License-Identifier: Apache-2.0
"""Empirical-Bayes engine of the hierarchical model (spec §5.8, step 2).

Three pieces, each closed-form or one-dimensional:

1. **Existence**: the per-channel mean of the template amplitudes ``y_c`` (standard error
   ``se_c`` from the trials) follows a two-component mixture,
   ``(1 - pi_s) N(0, se_c^2) + pi_s N(mu_s, tau^2 + se_c^2)``: no response, or a response
   whose channel amplitude scatters around the subject's ``mu_s`` with the between-channel
   SD ``tau`` (shared by the subjects). ``mu_s ~ N(mu, sigma_g^2)`` ties the subjects
   together. EM gives ``pi_s``, ``mu_s``, ``tau``; ``P(response)`` of a channel is its
   posterior responsibility.
2. **Shrinkage**: the channel estimates of the fitted components (latency, amplitude,
   width; standard errors from their Wald intervals) enter the nested random-effects model
   ``y_sc = x_s' beta + u_s + v_sc + e_sc`` (``e`` with the known standard errors),
   fitted by REML (:func:`evoparam.stats.variance.reml`); the BLUPs are the shrunken
   channel and subject estimates.
3. **Jitter**: the trial latencies of the trusted trials, with the latency noise of each
   channel (the jitter that noise alone gives, §5.7) as known standard error, enter the same
   model with a residual: the residual SD is the trial-to-trial latency jitter beyond noise.
   The right skew of the jitter (ex-Gaussian ``tau``) is estimated from the third moment of
   the centred trial latencies (noise is symmetric and adds none).
"""

import re

import numpy as np
from scipy import stats as sstats
from scipy.optimize import minimize_scalar

from evoparam.errors import SettingsError
from evoparam.stats.variance import reml

__all__ = ["mixture_em", "shrink", "jitter_components", "parse_formula", "design_matrix"]

PARAMETERS = ("latency", "amplitude", "fwhm")
EM_TOL = 1e-9
EM_MAX_ITER = 1000
PI_BOUNDS = (1e-4, 1 - 1e-4)


def _normal_logpdf(y, mean, var):
    return -0.5 * (np.log(2 * np.pi * var) + (y - mean) ** 2 / var)


def mixture_em(y, se, subject, polarity=0, tol=EM_TOL, max_iter=EM_MAX_ITER):
    """Two-component mixture (no response / response) on channel amplitudes.

    Parameters
    ----------
    y, se : ndarray (n,)
        Channel amplitude estimates and their standard errors.
    subject : ndarray (n,) of int
        Subject index of each channel.
    polarity : int
        Expected sign of the response (initialisation only; 0 when unknown).

    Returns
    -------
    dict with ``p_response`` (n,), ``pi``, ``mu`` (per subject), ``tau``, ``mu_group``,
    ``sd_group``, ``loglik``, ``n_iter``, ``converged``.
    """

    y = np.asarray(y, float)
    se2 = np.asarray(se, float) ** 2
    subject = np.asarray(subject, int)
    n_sub = int(subject.max()) + 1
    ok = np.isfinite(y) & np.isfinite(se2) & (se2 > 0)
    sign = polarity if polarity else (np.sign(np.median(y[ok])) or 1.0)
    z = sign * y / np.sqrt(se2)
    # initialise from the channels well beyond their noise
    strong = ok & (z > 3)
    start = np.median(y[strong]) if strong.sum() >= 2 else sign * 3 * np.sqrt(np.median(se2[ok]))
    mu = np.full(n_sub, float(start))
    tau = 0.3 * abs(start)
    pi = np.full(n_sub, max(min(np.mean(strong[ok]) if ok.any() else 0.3, 0.9), 0.05))
    mu_group, var_group = float(start), np.inf
    previous = -np.inf
    converged = False
    for it in range(1, max_iter + 1):
        var1 = tau ** 2 + se2
        l1 = np.log(pi[subject]) + _normal_logpdf(y, mu[subject], var1)
        l0 = np.log(1 - pi[subject]) + _normal_logpdf(y, 0.0, se2)
        top = np.maximum(l1, l0)
        total = top + np.log(np.exp(l1 - top) + np.exp(l0 - top))
        r = np.where(ok, np.exp(l1 - total), 0.0)
        loglik = float(np.sum(total[ok]))
        if abs(loglik - previous) <= tol * max(1.0, abs(loglik)):
            converged = True
            break
        previous = loglik
        counts = np.bincount(subject[ok], minlength=n_sub)
        pi = np.clip(np.bincount(subject, weights=r, minlength=n_sub) / np.maximum(counts, 1),
                     *PI_BOUNDS)
        w = np.where(ok, r / var1, 0.0)
        prior = 0.0 if not np.isfinite(var_group) or var_group <= 0 else 1.0 / var_group
        num = np.bincount(subject, weights=w * np.nan_to_num(y), minlength=n_sub) \
            + prior * mu_group
        den = np.bincount(subject, weights=w, minlength=n_sub) + prior
        mu = np.where(den > 0, num / np.where(den > 0, den, 1.0), mu_group)

        def objective(t):
            v = t ** 2 + se2[ok]
            return -np.sum(r[ok] * _normal_logpdf(y[ok], mu[subject[ok]], v))

        bound = 10 * np.nanmax(np.abs(y[ok])) + 1e-12
        tau = float(minimize_scalar(objective, bounds=(0.0, bound), method="bounded",
                                    options={"xatol": 1e-10 * bound}).x)
        if n_sub > 1:
            mu_group = float(np.mean(mu))
            var_group = max(float(np.var(mu, ddof=1)), 1e-12 * mu_group ** 2 + 1e-300)
    return {"p_response": r, "pi": pi, "mu": mu, "tau": tau, "mu_group": mu_group,
            "sd_group": float(np.sqrt(var_group)) if np.isfinite(var_group) else np.nan,
            "loglik": loglik, "n_iter": it, "converged": converged}


def shrink(y, se, subject, channel, X=None):
    """Shrunken channel and subject estimates of one parameter (REML, nested model).

    With one subject the subject level is dropped (the channels are the only level).
    Returns the :func:`~evoparam.stats.variance.reml` dictionary and the index of the
    channel level in it.
    """

    subject = np.asarray(subject)
    groups = [subject, channel] if len(np.unique(subject)) > 1 else [channel]
    out = reml(y, groups, se=se, X=X, residual=False)
    return out, len(groups) - 1


def jitter_components(latency, noise_sd, subject, channel):
    """Trial-to-trial latency jitter beyond noise: REML residual SD, and the ex-Gaussian
    decomposition (``sigma``, ``tau``) from the second and third moments of the centred
    trial latencies."""

    latency = np.asarray(latency, float)
    noise_sd = np.asarray(noise_sd, float)
    ok = np.isfinite(latency) & np.isfinite(noise_sd) & (noise_sd > 0)
    latency, noise_sd = latency[ok], noise_sd[ok]
    subject, channel = np.asarray(subject)[ok], np.asarray(channel)[ok]
    if latency.size < 5:
        return {"sd": np.nan, "sigma": np.nan, "tau": np.nan, "n_trials": int(latency.size)}
    single = len(np.unique(subject)) == 1
    groups = [channel] if single else [subject, channel]
    out = reml(latency, groups, se=noise_sd, residual=True)
    sd = float(np.sqrt(out["residual"]))
    # third central moment within channels (noise adds nothing to it)
    keys = [f"{a}/{b}" for a, b in zip(subject, channel)]
    centred = np.empty_like(latency)
    for key in dict.fromkeys(keys):
        rows = np.array([k == key for k in keys])
        centred[rows] = latency[rows] - latency[rows].mean()
    m3 = float(np.mean(centred ** 3))
    tau = (m3 / 2) ** (1 / 3) if m3 > 0 else 0.0
    tau = min(tau, sd)
    sigma = float(np.sqrt(max(sd ** 2 - tau ** 2, 0.0)))
    return {"sd": sd, "sigma": sigma, "tau": float(tau), "n_trials": int(latency.size),
            "converged": bool(out["converged"])}


_FORMULA = re.compile(r"^\s*([A-Za-z_]\w*)\s*~\s*(.+?)\s*$")


def parse_formula(formula):
    """``"latency ~ age + state"`` -> ("latency", ["age", "state"]); ``~ 1`` -> no term."""

    match = _FORMULA.match(formula or "")
    if not match:
        raise SettingsError(f"formula must read 'parameter ~ term + term', not {formula!r}.")
    parameter, rhs = match.groups()
    if parameter not in PARAMETERS:
        raise SettingsError(f"formula: the parameter must be one of {PARAMETERS}, not "
                            f"{parameter!r}.")
    terms = [t.strip() for t in rhs.split("+") if t.strip() not in ("", "1")]
    for term in terms:
        if not re.fullmatch(r"[A-Za-z_]\w*", term):
            raise SettingsError(f"formula: only main effects of metadata columns are "
                                f"supported, not {term!r}.")
    return parameter, terms


def design_matrix(values, terms):
    """Design of subject covariates: intercept, numeric columns, treatment-coded factors.

    ``values``: list (one per observation) of dicts term -> value. Returns (X, names).
    """

    columns, names = [np.ones(len(values))], ["Intercept"]
    for term in terms:
        raw = [v[term] for v in values]
        try:
            numeric = np.array(raw, dtype=float)
            columns.append(numeric)
            names.append(term)
        except (TypeError, ValueError):
            levels = sorted(set(map(str, raw)))
            for level in levels[1:]:
                columns.append(np.array([str(r) == level for r in raw], float))
                names.append(f"{term}[T.{level}]")
    return np.column_stack(columns), names


def effects_table(out, names, level=0.95):
    """Fixed effects with standard errors, intervals and two-sided p-values (normal)."""

    import pandas as pd

    beta = np.asarray(out["beta"], float)
    se = np.sqrt(np.diag(np.asarray(out["beta_cov"], float)))
    q = sstats.norm.ppf(0.5 + level / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = beta / se
    return pd.DataFrame({"term": names, "estimate": beta, "se": se, "ci_low": beta - q * se,
                         "ci_high": beta + q * se, "z": z,
                         "p_value": 2 * sstats.norm.sf(np.abs(z))})
