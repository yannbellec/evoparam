# SPDX-License-Identifier: Apache-2.0
"""Public interface of the fitting engine: ResponseModel, fit, fit_response (spec §5.5.8)."""

import dataclasses
import inspect
import logging
import time as _time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.stats import chi2, norm
from scipy.stats import t as student_t

from evoparam._typing import Any, NDArray
from evoparam.data import (
    Component,
    Evoked,
    GoodnessOfFit,
    GroupFit,
    Provenance,
    ResponseFit,
    TrialSet,
)
from evoparam.data.provenance import elapsed
from evoparam.errors import DataError, FitError, SettingsError
from evoparam.errors import warn as _warn
from evoparam.shapes import FWHM_PER_SIGMA, compute_features
from evoparam.utils._estimator import BaseEstimatorMixin, check_is_fitted, validate_array
from evoparam.utils._logging import progress, verbose
from evoparam.utils._numeric import divide
from evoparam.utils._optional import require
from evoparam.version import __version__

from . import engine as _engine
from ._settings import FitSettings, resolve_settings
from .classical import classical_measures
from .labels import label_components

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from evoparam.shapes._base import ComponentFeatures

# diagnostics of the fitting API (spec §9.8: child logger named after the module)
logger: logging.Logger = logging.getLogger("evoparam.models")

__all__ = [
    "ResponseModel",
    "fit",
    "fit_response",
    "compare_models",
    "methods_text",
    "evaluate_fit",
    "fitted_peak",
]

FEATURES_WITH_SE = ("latency", "amplitude", "fwhm", "onset", "offset", "area")

# The lag-1 autocorrelation of the residual is reported from more than this many samples.
_MIN_LAG_SAMPLES = 3
# The expanded percentile interval of a bootstrap needs two trials or more (t_{n-1}).
_MIN_TRIALS_T = 2
# The fewest finite bootstrap draws a standard error and an interval are computed from.
_MIN_BOOTSTRAP_DRAWS = 10
# A curve needs three points for an interior extremum.
_MIN_EXTREMUM_POINTS = 3


###################################################################################################
## Building the result


def _features(
    shape: Any, tw: Any, params: Any, amp: Any, settings: Any, sfreq: float
) -> "ComponentFeatures":
    """Return the features of one component of amplitude ``amp`` on the fit window."""
    return compute_features(
        shape,
        tw,
        params,
        amp,
        onset_fraction=settings.onset_fraction,
        sfreq=sfreq,
        window=(tw[0], tw[-1]),
    )


def _scaled_features(unit: Any, amp: Any) -> Any:
    """Return the features of ``amp * s(t)`` from those of ``s(t)``.

    ``unit`` holds the features measured with amplitude 1: only the amplitude, its sign and the
    area depend on ``amp``, and compute_features gives them as ``amp`` times the unit values (the
    same numbers).
    """
    amp = float(amp)
    signed = amp * unit.amplitude
    return dataclasses.replace(
        unit, amplitude=signed, polarity=1 if signed >= 0 else -1, area=float(amp * unit.area)
    )


def _feature_vector(
    shape: Any, tw: Any, params: Any, amp: Any, settings: Any, sfreq: float, unit: Any = None
) -> NDArray[Any]:
    """Return the features with a standard error as a vector (``FEATURES_WITH_SE`` order)."""
    f = (
        _features(shape, tw, params, amp, settings, sfreq)
        if unit is None
        else _scaled_features(unit, amp)
    )
    return np.array([getattr(f, name) for name in FEATURES_WITH_SE], float)


def _diagnostics(core: Any, _settings: Any) -> dict[str, Any]:
    """Covariance of every parameter, conditioning and sloppy pairs (step 6)."""
    struct, theta, beta = core["struct"], core["theta"], core["beta"]
    jac = struct.full_jacobian(theta, beta)
    p = jac.shape[1]
    rw = struct.white(struct.y - struct.design(theta) @ beta)
    rss_w = float(np.sum(rw**2))
    dof = max(core["n_white"] - p, 1.0)
    s2 = rss_w / dof
    jtj = jac.T @ jac
    cov = s2 * np.linalg.pinv(jtj)
    norms = np.sqrt(np.clip(np.diag(jtj), 1e-300, None))
    scaled = jtj / np.outer(norms, norms)
    eig = np.linalg.eigvalsh(scaled)
    cond = float(eig.max() / max(eig.min(), 1e-300)) if eig.size else 1.0
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    corr = divide(cov, np.outer(sd, sd))
    names = []
    for k, shape in enumerate(core["shapes"]):
        names += [f"{k}:{n}" for n in shape.param_names]
    names += [f"bg:{n}" for n in getattr(struct.background, "nonlinear_names", ())]
    names += [f"{k}:A" for k in range(len(core["shapes"]))]
    names += [f"bg:{n}" for n in getattr(struct.background, "linear_names", ())]
    flagged = [
        (names[i], names[j], float(corr[i, j]))
        for i in range(p)
        for j in range(i + 1, p)
        if np.isfinite(corr[i, j]) and abs(corr[i, j]) > _engine.CORR_LIMIT
    ]
    return {
        "cov": cov,
        "cond": cond,
        "eigvals": eig.tolist(),
        "corr": np.nan_to_num(corr),
        "flagged_pairs": flagged,
        "param_names": names,
        "s2": s2,
        "ill_conditioned": cond > _engine.COND_LIMIT or bool(flagged),
    }


def _component_uncertainty(
    core: Any, diag: Any, k: Any, settings: Any, sfreq: float, z: Any, unit: Any = None
) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """Return Wald intervals for the parameters and delta-method intervals for the features.

    ``unit``: the features of the component with amplitude 1, when already measured.
    """
    struct, theta, beta = core["struct"], core["theta"], core["beta"]
    shape = core["shapes"][k]
    sl = slice(struct.offsets[k], struct.offsets[k + 1])
    params, amp = theta[sl].copy(), float(beta[k])
    idx = list(range(sl.start, sl.stop)) + [struct.n_theta + k]
    sub = diag["cov"][np.ix_(idx, idx)]
    se, ci = {}, {}
    for i, name in enumerate(shape.param_names):
        s = float(np.sqrt(max(sub[i, i], 0)))
        se[name], ci[name] = s, (params[i] - z * s, params[i] + z * s)
    s_amp = float(np.sqrt(max(sub[-1, -1], 0)))
    x0 = np.r_[params, amp]
    if unit is None:
        unit = _features(shape, struct.tw, params, 1.0, settings, sfreq)
    f0 = _feature_vector(shape, struct.tw, params, amp, settings, sfreq, unit=unit)
    grads = np.zeros((len(f0), len(x0)))
    lo, hi = shape.default_bounds(struct.tw)
    for i in range(len(x0)):
        h = 1e-5 * max(abs(x0[i]), 1e-3 if i < len(params) else abs(amp) + 1e-9)
        xp, xm = x0.copy(), x0.copy()
        xp[i] += h
        xm[i] -= h
        if i < len(params):
            xp[i], xm[i] = min(xp[i], hi[i]), max(xm[i], lo[i])
        step = xp[i] - xm[i]
        if step <= 0:
            continue
        # a change of the amplitude alone scales the unit features
        same = None if i < len(params) else unit
        fp = _feature_vector(shape, struct.tw, xp[:-1], xp[-1], settings, sfreq, unit=same)
        fm = _feature_vector(shape, struct.tw, xm[:-1], xm[-1], settings, sfreq, unit=same)
        grads[:, i] = (fp - fm) / step
    for j, name in enumerate(FEATURES_WITH_SE):
        g = np.nan_to_num(grads[j])
        s = float(np.sqrt(max(g @ sub @ g, 0)))
        se[name] = s if name != "amplitude" else max(s, 0.0)
        ci[name] = (f0[j] - z * s, f0[j] + z * s)
    se["A"], ci["A"] = s_amp, (amp - z * s_amp, amp + z * s_amp)
    return se, ci


def _gof(core: Any, y_window: Any, valid: Any, _settings: Any) -> "GoodnessOfFit":
    """Return the goodness of fit of the engine's output on the valid samples of the window."""
    resid = core["residual"][valid]
    yv = y_window[valid]
    n = len(yv)
    p = core["struct"].n_theta + len(core["beta"])
    rss = float(np.sum(resid**2))
    tss = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    struct = core["struct"]
    rw = struct.white(struct.y - struct.design(core["theta"]) @ core["beta"])
    rss_w = float(np.sum(rw**2))
    n_eff = core["n_white"]
    lag = (
        float(np.corrcoef(resid[1:], resid[:-1])[0, 1])
        if n > _MIN_LAG_SAMPLES and np.std(resid) > 0
        else np.nan
    )
    rmse = float(np.sqrt(rss / n))
    span = float(yv.max() - yv.min()) if n else np.nan
    return GoodnessOfFit(
        r2=r2,
        adj_r2=1 - (1 - r2) * (n - 1) / max(n - p - 1, 1) if np.isfinite(r2) else np.nan,
        rmse=rmse,
        mae=float(np.mean(np.abs(resid))),
        nrmse=rmse / span if span else np.nan,
        aic=_engine.criterion(rss_w, n_eff, p, "aic"),
        aicc=_engine.criterion(rss_w, n_eff, p, "aicc"),
        bic=_engine.criterion(rss_w, n_eff, p, "bic"),
        n_params=p,
        n_samples=n,
        n_eff=n_eff,
        residual_lag1_autocorr=lag,
        baseline_sd=core["sigma"],
    )


def _bounds_warnings(core: Any) -> list[Any]:
    """Return a warning for each parameter within ``BOUND_TOL`` of its bounds."""
    out = []
    struct = core["struct"]
    for k, shape in enumerate(core["shapes"]):
        lo, hi = shape.default_bounds(struct.tw)
        params = struct.comp_params(core["theta"], k)
        span = hi - lo
        for i, name in enumerate(shape.param_names):
            if (
                params[i] - lo[i] <= _engine.BOUND_TOL * span[i]
                or hi[i] - params[i] <= _engine.BOUND_TOL * span[i]
            ):
                out.append(f"at_bound:{k}:{name}")
    return out


def _unresolvable(comps: Any, diag: Any, settings: Any) -> list[Any]:
    """Return the indices of the components that cannot be told apart from a neighbour.

    Spec §5.5.3 step 6, risk S1: the amplitudes or latencies of two components correlate beyond
    ``CORR_LIMIT``, or two neighbours of opposite sign are closer than ``overlap_c`` times their
    summed widths (a cancelling pair).
    """
    out = set()
    latency = {}
    for k, comp in enumerate(comps):
        names = comp.param_names
        latency[k] = names[0] if names else None
    for a, b, _ in diag.get("flagged_pairs", []):
        ka, na = a.split(":", 1)
        kb, nb = b.split(":", 1)
        if ka == kb or "bg" in (ka, kb):
            continue
        ka, kb = int(ka), int(kb)
        if (na == "A" and nb == "A") or (na == latency.get(ka) and nb == latency.get(kb)):
            out |= {ka, kb}
    order = sorted(range(len(comps)), key=lambda k: comps[k].latency)
    for i, j in zip(order, order[1:], strict=False):
        a, b = comps[i], comps[j]
        if (
            np.sign(a.amplitude) != np.sign(b.amplitude)
            and abs(b.latency - a.latency)
            < settings.overlap_c * (a.fwhm + b.fwhm) / FWHM_PER_SIGMA
        ):
            out |= {i, j}
    return sorted(out)


def _with_plugins(provenance: Any, settings: Any) -> Any:
    """``provenance.environment["plugins"]``.

    The plugin shapes, background and preset the settings use, ``{name: "dist==version"}`` (spec
    §10.8); unchanged without plugins.
    """
    from evoparam import presets
    from evoparam.shapes import BACKGROUNDS, SHAPES

    used = {}
    for registry, names in (
        (SHAPES, settings.shapes),
        (BACKGROUNDS, (settings.background,)),
        (presets.PRESETS, (settings.preset,)),
    ):
        origins = registry.plugins()
        used.update({name: origins[name] for name in names if name in origins})
    if not used:
        return provenance
    environment = dict(provenance.environment)
    environment["plugins"] = {**(environment.get("plugins") or {}), **used}
    return dataclasses.replace(provenance, environment=environment)


@dataclasses.dataclass(frozen=True)
class _Waveform:
    """A waveform being fitted: time axis, values, sampling rate, its trials and mask."""

    times: Any
    y: Any
    sfreq: float
    trials: Any = None
    mask: Any = None


def _build_fit(core: Any, wave: _Waveform, settings: Any, settings_dict: Any, start: Any) -> Any:
    """Build the :class:`ResponseFit` of the engine's output ``core``."""
    times, y, sfreq, trials, mask = wave.times, wave.y, wave.sfreq, wave.trials, wave.mask
    window = settings.fit_window
    baseline = settings.baseline if settings.baseline is not None else (np.nan, np.nan)
    warns = list(core.get("warnings", []))
    provenance = _with_plugins(
        Provenance().with_step("evoparam.models.fit_waveform", settings_dict, duration_s=None),
        settings,
    )
    if core["status"] in ("skipped", "failed") or "struct" not in core:
        return ResponseFit(
            components=[],
            background={},
            gof=None,
            status=core["status"],
            message=core["message"],
            window=window,
            baseline=baseline,
            settings=settings_dict,
            warnings=warns,
            provenance=provenance,
            diagnostics={
                "sigma": core.get("sigma"),
                "rho": core.get("rho"),
                "n_evaluations": core.get("n_evaluations", 0),
                "noise_reference": core.get("noise_reference", "baseline"),
            },
        )

    struct = core["struct"]
    wmask = core["window_mask"]
    valid = ~(mask[wmask] | ~np.isfinite(y[wmask]))
    z = norm.ppf(0.5 + settings.ci_level / 2)
    diag = (
        _diagnostics(core, settings)
        if core["shapes"]
        else {
            "cov": np.zeros((0, 0)),
            "cond": 1.0,
            "eigvals": [],
            "corr": np.zeros((0, 0)),
            "flagged_pairs": [],
            "param_names": [],
            "ill_conditioned": False,
        }
    )
    if diag["ill_conditioned"]:
        warns.append("ill_conditioned")
    warns += _bounds_warnings(core)

    comps = []
    for k, shape in enumerate(core["shapes"]):
        params = struct.comp_params(core["theta"], k)
        amp = float(core["beta"][k])
        unit = _features(shape, struct.tw, params, 1.0, settings, sfreq)
        f = _scaled_features(unit, amp)
        se, ci = _component_uncertainty(core, diag, k, settings, sfreq, z, unit=unit)
        warns += [f"{w}:{k}" for w in f.warnings]
        comps.append(
            Component(
                shape=shape.name,
                params=params,
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
                se=se,
                ci=ci,
            )
        )

    for k in _unresolvable(comps, diag, settings):
        comps[k] = dataclasses.replace(comps[k], resolvable=False)
        warns.append(f"unresolvable:{k}")

    bg = struct.background
    n_lin = getattr(bg, "n_linear", 0)
    bg_values = dict(
        zip(
            getattr(bg, "linear_names", ()),
            core["beta"][len(comps) : len(comps) + n_lin],
            strict=False,
        )
    )
    bg_values.update(
        zip(getattr(bg, "nonlinear_names", ()), struct.bg_params(core["theta"]), strict=False)
    )
    background = {"name": bg.name, **{k: float(v) for k, v in bg_values.items()}}
    if hasattr(bg, "t_ref"):
        # the time origin of the background's basis (the first sample of the fit window
        # unless the background sets one), so that evaluate_fit gives the fitted curve on
        # any other time axis
        background["t_ref"] = float(bg._t_ref(struct.tw))

    classical = {}
    if settings.classical:
        classical = classical_measures(
            times,
            np.nan_to_num(y),
            window=window,
            polarity="negative",
            trials=trials,
            n_bootstrap=200,
            random_state=settings.random_state,
        )

    fit = ResponseFit(
        components=comps,
        background=background,
        gof=_gof(core, np.where(valid, y[wmask], 0.0), valid, settings),
        status=core["status"],
        message=core["message"],
        window=window,
        baseline=baseline,
        settings=settings_dict,
        classical=classical,
        model=core["model"],
        residual=core["residual"],
        warnings=warns,
        provenance=provenance,
        diagnostics={
            "cond": diag["cond"],
            "eigvals": diag["eigvals"],
            "corr": np.asarray(diag["corr"]).tolist(),
            "flagged_pairs": diag["flagged_pairs"],
            "param_names": diag["param_names"],
            "best_rejected_delta": core["best_rejected_delta"],
            "sigma": core["sigma"],
            "rho": core["rho"],
            "n_evaluations": core.get("n_evaluations", 0),
            "noise_reference": core.get("noise_reference", "baseline"),
            "n_eff": core["n_eff"],
            "n_white": core["n_white"],
            "ar_order": core["ar_order"],
            "noise_stride": core.get("noise_stride", 1),
            "noise_phase": core.get("noise_phase", 0),
            "criterion": core["kind"],
            "uncertainty_method": settings.uncertainty,
            "steps": core["steps"],
            "duration_s": elapsed(start),
        },
    )
    if settings.uncertainty == "bootstrap":
        fit = _bootstrap(fit, core, times, trials, settings, sfreq, z)
    elif settings.uncertainty == "profile":
        fit = _profile(fit, core, settings, sfreq)
    elif settings.uncertainty == "posterior":
        raise SettingsError(
            "uncertainty='posterior' comes with the hierarchical models (build step S11)."
        )
    if settings.windows:
        fit = label_components(fit, settings.windows)
    return fit


def _bootstrap(
    fit: Any, core: Any, _times: Any, trials: Any, settings: Any, sfreq: float, _z: Any
) -> Any:
    """Resample trials, re-average, refit the same model from the full-data solution."""
    if trials is None:
        raise SettingsError(
            "uncertainty='bootstrap' needs the trials (fit a TrialSet, or "
            "pass trials= to fit_waveform)."
        )
    rng = np.random.default_rng(settings.random_state)
    struct = core["struct"]
    wmask = core["window_mask"]
    trials = np.asarray(trials, float)
    n = len(trials)
    draws: dict[int, list[Any]]
    draws = {k: [] for k in range(len(fit.components))}
    for _ in range(settings.n_bootstrap):
        avg = trials[rng.integers(0, n, n)].mean(axis=0)[wmask]
        boot = dataclasses.replace(struct, y=np.nan_to_num(avg))
        theta, beta, _, ok, _ = _engine._optimise(
            boot, core["theta"].copy(), settings, settings.f_scale
        )
        if not ok:
            continue
        for k, shape in enumerate(struct.shapes):
            draws[k].append(
                _feature_vector(
                    shape, struct.tw, boot.comp_params(theta, k), beta[k], settings, sfreq
                )
            )
    # expanded percentile interval (Hesterberg 2015): the plain percentile interval of an
    # average of n trials is too narrow by sqrt((n - 1) / n) and uses z instead of t_{n-1}
    tail = 100 * (1 - settings.ci_level) / 2
    if n >= _MIN_TRIALS_T:
        q = np.sqrt(n / (n - 1)) * student_t.ppf(0.5 + settings.ci_level / 2, n - 1)
        tail = 100 * norm.cdf(-q)
    comps = []
    for k, comp in enumerate(fit.components):
        values = np.array(draws[k])
        se, ci = dict(comp.se), dict(comp.ci)
        if len(values) >= _MIN_BOOTSTRAP_DRAWS:
            for j, name in enumerate(FEATURES_WITH_SE):
                col = values[:, j][np.isfinite(values[:, j])]
                if len(col) < _MIN_BOOTSTRAP_DRAWS:
                    continue
                se[name] = float(np.std(col, ddof=1))
                ci[name] = tuple(np.percentile(col, [tail, 100 - tail]))
        comps.append(dataclasses.replace(comp, se=se, ci=ci))
    diag = dict(fit.diagnostics, n_bootstrap_ok=min((len(v) for v in draws.values()), default=0))
    return dataclasses.replace(fit, components=comps, diagnostics=diag)


def _profile_bounds(grid: Any, stat: Any, limit: Any) -> tuple[Any, Any]:
    """Return the ends of the profile interval.

    Where the statistic crosses ``limit``, linearly interpolated between the outermost grid value
    inside and its neighbour outside (the grid values themselves would shorten the interval by up
    to one grid step per side).
    """
    idx = np.flatnonzero(stat <= limit)
    first, last = idx[0], idx[-1]
    ends = []
    for inner, outer in ((first, first - 1), (last, last + 1)):
        value = grid[inner]
        if 0 <= outer < len(grid) and np.isfinite(stat[outer]) and stat[outer] > stat[inner]:
            frac = (limit - stat[inner]) / (stat[outer] - stat[inner])
            value = grid[inner] + frac * (grid[outer] - grid[inner])
        ends.append(float(value))
    return ends[0], ends[1]


def _profile(fit: Any, core: Any, settings: Any, _sfreq: Any, n_grid: int = 21) -> Any:
    """Profile-likelihood intervals of the non-linear parameters."""
    struct = core["struct"]
    theta, beta = core["theta"], core["beta"]
    lower, upper = struct.bounds()
    rss0 = float(np.sum(struct.white(struct.y - struct.design(theta) @ beta) ** 2))
    s2 = rss0 / max(core["n_white"] - struct.n_theta - len(beta), 1)
    limit = chi2.ppf(settings.ci_level, 1)
    comps, flags = [], []
    for k, comp in enumerate(fit.components):
        ci, se = dict(comp.ci), dict(comp.se)
        for i, name in enumerate(comp.param_names):
            j = struct.offsets[k] + i
            width = 4 * max(se.get(name, 0), 1e-6 * (upper[j] - lower[j]))
            grid = np.clip(theta[j] + np.linspace(-width, width, n_grid), lower[j], upper[j])
            stat = np.full(n_grid, np.inf)
            for g, value in enumerate(grid):
                lo, hi = lower.copy(), upper.copy()
                eps = 1e-9 * (upper[j] - lower[j])
                lo[j], hi[j] = value - eps, value + eps
                fixed = _FixedStructure(struct, lo, hi)
                t0 = theta.copy()
                t0[j] = value
                th, be, rw, ok, _ = _engine._optimise(fixed, t0, settings, settings.f_scale)
                if ok:
                    stat[g] = (np.sum(rw**2) - rss0) / s2
            inside = grid[stat <= limit]
            if len(inside):
                ci[name] = _profile_bounds(grid, stat, limit)
                if len(inside) == n_grid:
                    flags.append(f"structurally_unidentifiable:{k}:{name}")
                elif min(inside) == grid[0] or max(inside) == grid[-1]:
                    flags.append(f"practically_unidentifiable:{k}:{name}")
        comps.append(dataclasses.replace(comp, ci=ci))
    return dataclasses.replace(fit, components=comps, warnings=list(fit.warnings) + flags)


def _cv_selector(trials: Any, times: Any, settings: Any) -> Any:
    """Choose the number of components K by K-fold cross-validation over trials.

    The nested models are K = 0, 1, ... Each model is refitted on the average of the training
    trials and scored on the held-out average; the smallest K within one standard error of the best
    mean error is chosen.
    """

    def select(path: Any, make: Any) -> Any:
        """Return the index on the path that predicts held-out trials best."""
        rng = np.random.default_rng(settings.random_state)
        n = len(trials)
        folds = np.array_split(rng.permutation(n), min(settings.n_folds, n))
        win = (times >= settings.fit_window[0]) & (times <= settings.fit_window[1])
        errors = np.full((len(folds), len(path)), np.nan)
        for i, test in enumerate(folds):
            train = np.setdiff1d(np.arange(n), test)
            if len(train) == 0 or len(test) == 0:
                continue
            y_train = np.nan_to_num(trials[train].mean(axis=0)[win])
            y_test = np.nan_to_num(trials[test].mean(axis=0)[win])
            for k, state in enumerate(path):
                struct = dataclasses.replace(make(state.shapes), y=y_train)
                theta, beta, _, ok, _ = _engine._optimise(
                    struct, state.theta.copy(), settings, settings.f_scale, tol=1e-6
                )
                if ok:
                    pred = struct.design(theta) @ beta
                    errors[i, k] = float(np.mean((y_test - pred) ** 2))
        mean = np.nanmean(errors, axis=0)
        se = np.nanstd(errors, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(errors), axis=0))
        best = int(np.nanargmin(mean))
        limit = mean[best] + np.nan_to_num(se[best])
        return int(np.flatnonzero(mean <= limit)[0])

    return select


class _FixedStructure(_engine._Structure):
    """A structure whose bounds are overridden (to hold one parameter fixed)."""

    def __init__(self, base: Any, lower: Any, upper: Any) -> None:
        """Copy ``base`` with the bounds ``lower`` and ``upper`` in place of its own."""
        super().__init__(
            base.shapes,
            base.background,
            base.tw,
            base.white,
            base.y,
            base.prior,
            base.scale,
            base.edge,
            base.level,
        )
        self._lower, self._upper = lower, upper

    def bounds(self) -> tuple[Any, Any]:
        """Return the fixed bounds."""
        return self._lower, self._upper


###################################################################################################
## Estimator

# Fields that are not settings of the preset: the preset itself and the random seed.
_NOT_PRESET_FIELDS = ("preset", "preset_note", "random_state")


def _changed_from_preset(settings: Any) -> list[Any]:
    """Return the names of the fields whose value differs from the preset's.

    Without a preset, from the package defaults: the settings a methods text must state (REQ-008,
    REQ-340).
    """
    try:
        base = resolve_settings(settings.preset)
    except SettingsError:
        return []
    out = []
    for name in FitSettings.__dataclass_fields__:
        if name in _NOT_PRESET_FIELDS:
            continue
        if getattr(settings, name) != getattr(base, name):
            out.append(name)
    return out


def _draw_seed() -> int:
    """Return a seed drawn from the operating system, recorded in the settings (REQ-295).

    A fit made with ``random_state=None`` can be repeated from its settings (REQ-331).
    """
    return int(np.random.default_rng().integers(2**31 - 1))


def preset_record(settings: Any) -> Any:
    """``{"name", "version"}`` of the preset of resolved settings (REQ-331), or None."""
    import evoparam.presets as _presets

    name = settings.preset
    if not name:
        return None
    version = None
    if name in _presets.PRESETS:
        version = _presets.get(name).version
    else:
        note = settings.preset_note or ""
        if " v" in note:
            version = note.split(" v", 1)[1].split(" ")[0].rstrip(";")
    return {"name": name, "version": version}


def preset_name(value: Any) -> Any:
    """Name of a recorded preset.

    ``settings["preset"]`` is ``{"name", "version"}`` (REQ-331); results written before it hold the
    name alone.
    """
    if isinstance(value, dict):
        return value.get("name")
    return value


def _settings_dict(settings: Any, overrides: Any = ()) -> dict[str, Any]:
    """Return the ``settings`` record of a result (REQ-331).

    The full resolved settings, the preset as ``{"name", "version"}``, ``_overrides`` (the keys the
    user set) and ``_changed_from_preset`` (the keys whose value differs from the preset, stated by
    the methods text).
    """
    out = dataclasses.asdict(settings)
    out["shapes"] = list(settings.shapes)
    out["preset"] = preset_record(settings)
    out["package_version"] = __version__
    out["_overrides"] = list(overrides)
    out["_changed_from_preset"] = _changed_from_preset(settings)
    return out


def evaluate_fit(fit: Any, times: Any, *, components: Any = None, background: bool = True) -> Any:
    """Curve of a ResponseFit on any time grid (all components, or a subset).

    The background is evaluated with the time origin of the fit (``t_ref`` of
    ``fit.background``, the first sample of the fit window): a background whose basis
    depends on time (exponential tail, linear drift) is the fitted one on any grid, not
    only on a grid that starts at the window's first sample. Before that sample, where it
    was not fitted, the background keeps its value there (a recovery tail extrapolated
    backwards grows without bound).
    """
    from evoparam.shapes import get_background, get_shape

    times = np.asarray(times, float)
    total = np.zeros_like(times)
    for k, comp in enumerate(fit.components):
        if components is None or k in components:
            total += _amplitude(fit, k) * get_shape(comp.shape).evaluate(times, comp.params)
    if background and fit.background:
        bg: Any = get_background(fit.background["name"])  # basis: a built-in background
        t_ref = fit.background.get("t_ref")
        bg_times = times
        if (
            t_ref is not None
            and dataclasses.is_dataclass(bg)
            and "t_ref" in {f.name for f in dataclasses.fields(bg)}
        ):
            bg = dataclasses.replace(cast("DataclassInstance", bg), t_ref=float(t_ref))
            bg_times = np.maximum(times, float(t_ref))
        lin = [fit.background[n] for n in getattr(bg, "linear_names", ())]
        nl = [fit.background[n] for n in getattr(bg, "nonlinear_names", ())]
        if lin:
            basis = bg.basis(bg_times, np.array(nl) if nl else None)
            total += np.asarray(lin) @ np.atleast_2d(basis)
    return total


def fitted_peak(
    fit: Any, window: Any, *, polarity: str = "negative", step: float = 1e-4
) -> dict[str, float]:
    """Peak of the fitted waveform (background and components) inside a window.

    The deepest interior local minimum (``polarity="negative"``) or highest interior local
    maximum (``"positive"``) of the fitted curve between ``window[0]`` (or the first sample
    of the fit window, when later) and ``window[1]``, on a grid of ``step`` seconds; a curve
    that is monotone through the window has no peak.
    The value is relative to zero (baseline-corrected data) and must have the sign of the
    polarity.

    Why: when an early component overlaps the recovery tail (the exponential background),
    the split of the curve between them is sloppy (often flagged ``ill_conditioned``), so a
    component's own latency can lie several ms from the trough of the waveform while the
    fitted curve itself is well determined. On ds004080 (the example run and the REQ-371
    runs) the N1 component's latency is within 2 ms of the published rule's peak on 24 %
    (registry row) and 37 % (code) of the channels where both find an N1, the trough of the
    fitted curve on 42 % and 80 %; where that peak is an N1 trough of the rule's average,
    on 31 of 39 and 24 of 24 channels (CHANGELOG, "N1 latency agreement").

    Returns
    -------
    dict
        ``latency`` (s) and ``amplitude`` (data unit), NaN when there is no such peak.
    """
    if polarity not in ("negative", "positive"):
        raise SettingsError("polarity must be 'negative' or 'positive'.")
    out = {"latency": np.nan, "amplitude": np.nan}
    if fit.status != "ok" or not fit.background:
        return out
    # from the first fitted sample: before it the background was not fitted
    # (evaluate_fit holds it there), and a held value next to a falling tail would make a
    # trough at the window's start
    start = max(window[0], fit.background.get("t_ref", window[0]))
    grid = np.arange(start, window[1] + step / 2, step)
    if len(grid) < _MIN_EXTREMUM_POINTS:
        return out
    curve = evaluate_fit(fit, grid)
    s = -curve if polarity == "negative" else curve
    inner = np.flatnonzero((s[1:-1] >= s[:-2]) & (s[1:-1] > s[2:])) + 1
    inner = inner[s[inner] > 0]
    if inner.size:
        j = inner[np.argmax(s[inner])]
        out = {"latency": float(grid[j]), "amplitude": float(curve[j])}
    return out


def _amplitude(fit: Any, k: Any) -> Any:
    """Linear amplitude A of component k: the centre of its (symmetric) Wald interval."""
    comp = fit.components[k]
    ci = comp.ci or {}
    if "A" in ci:
        return 0.5 * (ci["A"][0] + ci["A"][1])
    return comp.amplitude


class ResponseModel(BaseEstimatorMixin):
    """Estimator for evoked waveforms (spec §5.5.8).

    Settings in ``__init__``, learned attributes with a trailing underscore after :meth:`fit`.

    Keywords left at None take the preset's value, or the package default when there is
    no preset (see :class:`~evoparam.models.FitSettings`).
    """

    if TYPE_CHECKING:
        # set from the keywords by __init__ (scikit-learn convention)
        preset: Any
        sfreq: float | None
        tmin: float | None
        n_jobs: int
        random_state: Any
        _overrides_record: list[str]

    def __init__(  # noqa: PLR0913  (spec §5.5.8 fixes 30 parameters; DECISIONS.md §7.2)
        self,
        preset: Any = None,
        *,
        shapes: Any = None,
        background: Any = None,
        max_components: Any = None,
        min_fwhm: Any = None,
        max_fwhm: Any = None,
        min_amplitude: Any = None,
        detection_threshold: Any = None,
        polarity: Any = None,
        fit_window: Any = None,
        baseline: Any = None,
        sfreq: float | None = None,
        tmin: float | None = None,
        selection: Any = None,
        delta: Any = None,
        n_components: Any = None,
        overlap_c: Any = None,
        loss: Any = None,
        f_scale: Any = None,
        max_iter: Any = None,
        n_starts: Any = None,
        onset_fraction: Any = None,
        uncertainty: Any = None,
        n_bootstrap: Any = None,
        ci_level: Any = None,
        classical: Any = None,
        windows: Any = None,
        noise_model: Any = None,
        background_prior: Any = None,
        amplitude_prior: Any = None,
        edge_extension: Any = None,
        background_tau_max: Any = None,
        max_masked_fraction: Any = None,
        n_folds: Any = None,
        reseed: Any = None,
        noise_source: Any = None,
        latency_order: Any = None,
        candidate_search: Any = None,
        reject_cancelling: Any = None,
        threshold_role: Any = None,
        n_jobs: int = 1,
        random_state: Any = None,
        verbose: Any = None,
    ) -> None:
        """Keep every keyword as an attribute (scikit-learn contract: no validation here)."""
        for name, value in locals().items():
            if name != "self":
                setattr(self, name, value)

    @classmethod
    def _param_names(cls) -> list[str]:
        """Return the names of the constructor's parameters."""
        return cls._get_param_names()

    def _explicit(self, **extra: Any) -> Any:
        """Keys of FitSettings that the user set (REQ-331 ``_overrides``), in field order."""
        recorded = getattr(self, "_overrides_record", None)
        if recorded is not None:
            return list(recorded)
        given = {
            k
            for k in FitSettings.__dataclass_fields__
            if getattr(self, k, None) is not None or extra.get(k) is not None
        }
        return [
            k
            for k in FitSettings.__dataclass_fields__
            if k in given and k not in _NOT_PRESET_FIELDS
        ]

    def settings(self, **extra: Any) -> "FitSettings":
        """Return the resolved settings of the estimator, with ``extra`` overrides."""
        keys = set(FitSettings.__dataclass_fields__) - {"preset"}
        overrides = {k: getattr(self, k) for k in keys if hasattr(self, k)}
        overrides.update({k: v for k, v in extra.items() if v is not None})
        return resolve_settings(self.preset, **overrides)

    def fit_waveform(  # noqa: PLR0913  (public keywords; DECISIONS.md §7.2)
        self,
        times: Any,
        y: Any,
        *,
        window: Any = None,
        baseline: Any = None,
        sem: Any = None,
        mask: Any = None,
        trials: Any = None,
        random_state: Any = None,
        noise_ref: Any = None,
        unit: Any = None,
        on_error: str = "record",
    ) -> Any:
        """Fit one waveform; returns a :class:`ResponseFit` (never raises on a bad fit).

        ``random_state`` replaces the estimator's for this call. When both are None a seed
        is drawn from the operating system and recorded in ``settings`` (REQ-295).

        When the optimiser fails (REQ-307), ``on_error="record"`` (default) returns the fit
        with ``status="failed"`` and the exception in ``message``, and logs exactly one
        WARNING naming ``unit`` (the channel, when called by :meth:`fit_group`);
        ``on_error="raise"`` raises :class:`~evoparam.errors.FitError` instead. Data and
        settings errors always propagate.
        """
        start = _time.perf_counter()
        if random_state is None and self.random_state is None:
            random_state = _draw_seed()
        settings = self.settings(fit_window=window, baseline=baseline, random_state=random_state)
        times = np.asarray(times, float)
        y = np.asarray(y, float)
        if times.shape != y.shape:
            raise DataError("times and y must have the same shape.")
        if settings.fit_window is None:
            raise SettingsError("A fit window is needed (fit_window=, window= or a preset).")
        # the end of an epoch is excluded (tmax), so a window that ends at the epoch end
        # lies less than one sample beyond the last one: allowed (the fit uses the samples
        # inside the window)
        step = float(np.median(np.diff(times))) if times.size > 1 else 0.0
        if settings.fit_window[0] < times[0] - step * (1 + 1e-9) or settings.fit_window[1] > times[
            -1
        ] + step * (1 + 1e-9):
            raise DataError(
                f"fit window {settings.fit_window} exceeds the data "
                f"({times[0]:.4f} to {times[-1]:.4f} s)."
            )
        sfreq = 1.0 / np.median(np.diff(times))
        mask = np.zeros(len(y), bool) if mask is None else np.asarray(mask, bool)
        select_k = None
        if (
            noise_ref is None
            and trials is not None
            and _engine.use_plus_minus(settings, times, y, mask)
        ):
            noise_ref = _engine.plus_minus(trials)
        if settings.selection == "cv":
            if trials is None:
                raise SettingsError(
                    "selection='cv' needs the trials (fit a TrialSet, or pass "
                    "trials= to fit_waveform)."
                )
            select_k = _cv_selector(np.asarray(trials, float), times, settings)
        if on_error not in ("record", "raise"):
            raise SettingsError(f"on_error must be 'record' or 'raise', got {on_error!r}.")
        try:
            core = _engine.fit_waveform_core(
                times,
                y,
                settings=settings,
                sfreq=sfreq,
                mask=mask,
                sem=sem,
                select_k=select_k,
                noise_ref=noise_ref,
            )
        except (SettingsError, DataError):
            raise
        except Exception as exc:  # never NaN with status "ok"
            if on_error == "raise":
                raise FitError(
                    f"fit failed on {unit or 'the waveform'}: {type(exc).__name__}: {exc}"
                ) from exc
            core = {"status": "failed", "message": f"{type(exc).__name__}: {exc}", "warnings": []}
            _engine.logger.warning("fit failed on %s: %s", unit or "the waveform", core["message"])
        fit = _build_fit(
            core,
            _Waveform(times, y, sfreq, trials, mask),
            settings,
            _settings_dict(settings, self._explicit(fit_window=window, baseline=baseline)),
            start,
        )
        logger.debug(
            "fit of %s: status %s, %d components, %.3f s",
            unit or "a waveform",
            fit.status,
            len(fit.components),
            _time.perf_counter() - start,
        )
        return fit

    def fit_group(
        self,
        data: Any,
        *,
        window: Any = None,
        baseline: Any = None,
        picks: Any = None,
        n_jobs: Any = None,
        condition: str = "",
        on_error: str = "record",
    ) -> "GroupFit":
        """Fit every channel of an Evoked (or of the average of a TrialSet).

        A channel whose fit fails is kept with ``status="failed"`` and the exception in
        ``message``, with one WARNING record naming the channel (``on_error="record"``, the
        default); ``on_error="raise"`` raises the first :class:`~evoparam.errors.FitError`
        (REQ-307).
        """
        start = _time.perf_counter()
        trials = None
        if isinstance(data, TrialSet):
            trials = data
            evoked = data.average()
        elif isinstance(data, Evoked):
            evoked = data
        else:
            raise DataError("fit_group expects a TrialSet or an Evoked.")
        names = list(evoked.ch_names)
        index = (
            range(len(names))
            if picks is None
            else [names.index(p) if isinstance(p, str) else int(p) for p in picks]
        )

        # one seed for every channel, drawn when none is given and recorded (REQ-295)
        seed = _draw_seed() if self.random_state is None else None

        def one(ch: Any) -> Any:
            """Fit channel ``ch``."""
            mask = None if evoked.masks is None else evoked.masks[ch]
            sem = None if evoked.sem is None else evoked.sem[ch]
            tr = None if trials is None else trials.data[:, ch]
            return self.fit_waveform(
                evoked.times,
                evoked.data[ch],
                window=window,
                baseline=baseline,
                sem=sem,
                mask=mask,
                trials=tr,
                random_state=seed,
                unit=names[ch],
                on_error=on_error,
            )

        jobs = self.n_jobs if n_jobs is None else n_jobs
        if jobs and jobs != 1:
            with ThreadPoolExecutor(max_workers=None if jobs < 0 else jobs) as pool:
                fits = list(progress(pool.map(one, index), total=len(index), desc="fit"))
        else:
            fits = [one(ch) for ch in progress(index, total=len(index), desc="fit")]
        for fit in fits:
            fit.units = evoked.units  # the unit of the amplitudes (REQ-009)
        logger.info(
            "fit of %d channel(s)%s: %s, %.2f s",
            len(fits),
            f" ({condition})" if condition else "",
            ", ".join(
                f"{n} {status}" for status, n in sorted(Counter(f.status for f in fits).items())
            ),
            _time.perf_counter() - start,
        )
        resolved = self.settings(fit_window=window, baseline=baseline, random_state=seed)
        settings = _settings_dict(resolved, self._explicit(fit_window=window, baseline=baseline))
        # the noise reference the fits used (the plus-minus average needs the trials)
        used = {fit.diagnostics.get("noise_reference", "baseline") for fit in fits}
        settings["noise_reference"] = (
            "plusminus"
            if used == {"plusminus"}
            else ("baseline" if used == {"baseline"} else "mixed")
        )
        provenance = _with_plugins(
            evoked.provenance.with_step(
                "evoparam.models.ResponseModel.fit_group",
                settings,
                duration_s=_time.perf_counter() - start,
                preset=preset_record(resolved),
            ),
            resolved,
        )
        return GroupFit.from_fits(
            fits,
            [names[i] for i in index],
            [condition] * len(fits),
            settings=settings,
            provenance=provenance,
        )

    def _as_container(self, x: Any, *, reset: Any) -> Any:
        """X as an Evoked or a TrialSet.

        An array is (n_waveforms, n_times) at ``sfreq`` from ``tmin``, checked the scikit-learn way
        (2-D, dense, real, finite; after ``fit`` as many time points as in ``fit``).
        """
        x = _from_mne(x)
        if isinstance(x, (TrialSet, Evoked)):
            if reset:
                self.n_features_in_: int = int(x.n_times)
            elif x.n_times != getattr(self, "n_features_in_", x.n_times):
                raise DataError(
                    f"X has {x.n_times} time points, but {type(self).__name__} "
                    f"is expecting {self.n_features_in_}."
                )
            return x
        data = validate_array(self, x, reset=reset)
        if self.sfreq is None:
            raise SettingsError("sfreq is required when X is an array.")
        tmin = 0.0 if self.tmin is None else self.tmin
        times = tmin + np.arange(data.shape[-1]) / self.sfreq
        return Evoked(
            data=data, times=times, sfreq=self.sfreq, ch_names=[f"ch{i}" for i in range(len(data))]
        )

    def _fit_container(self, x: Any) -> "GroupFit":
        """GroupFit of a container.

        Without a fit window (no ``fit_window``, no preset) the window is the data range; when that
        range cannot hold a component (shorter than 3 x min_fwhm, or fewer than 5 samples) every
        waveform is returned as skipped rather than raising, so that short tabular inputs keep the
        estimator contract.
        """
        settings = self.settings()
        if settings.fit_window is not None:
            return self.fit_group(x)
        times = np.asarray(x.times, float)
        window = (float(times[0]), float(times[-1]))
        sfreq = float(x.sfreq)
        min_fwhm = max(settings.min_fwhm, 2.0 / sfreq)
        if (
            window[1] - window[0] >= _engine.MIN_WINDOW_FWHMS * min_fwhm
            and len(times) >= _engine.MIN_VALID_SAMPLES
        ):
            return self.fit_group(x, window=window)
        settings = self.settings(fit_window=window)
        evoked = x.average() if isinstance(x, TrialSet) else x
        message = (
            f"skipped: the data range {window} is shorter than 3 x min_fwhm "
            f"({3 * min_fwhm:.4f} s) or has fewer than 5 samples"
        )
        explicit = self._explicit(fit_window=window)
        fits = [
            _build_fit(
                {"status": "skipped", "message": message, "warnings": []},
                _Waveform(times, np.asarray(row, float), sfreq),
                settings,
                _settings_dict(settings, explicit),
                _time.perf_counter(),
            )
            for row in evoked.data
        ]
        for fit in fits:
            fit.units = evoked.units  # the unit of the amplitudes (REQ-009)
        return GroupFit.from_fits(
            fits,
            list(evoked.ch_names),
            [""] * len(fits),
            settings=_settings_dict(settings, explicit),
            provenance=evoked.provenance,
        )

    def fit(self, X: Any, y: Any = None) -> "ResponseModel":  # noqa: N803, ARG002  (scikit-learn API: X, y)
        """Fit every channel.

        Learned: ``fit_`` (GroupFit), ``components_`` (table), ``times_``, ``n_features_in_``
        (samples per waveform) and ``n_iter_`` (most residual evaluations of one waveform's
        least-squares solves, at least 1). X: Evoked, TrialSet, MNE object or array (n_waveforms,
        n_times); ``y`` is ignored.
        """
        container = self._as_container(X, reset=True)
        self.fit_: GroupFit = self._fit_container(container)
        self.components_: Any = self.fit_.to_dataframe()
        self.times_: NDArray[Any] = np.array(container.times, float)
        # the largest number of residual evaluations of one waveform (at least 1)
        self.n_iter_: int = max(
            [1] + [int(f.diagnostics.get("n_evaluations") or 0) for f in self.fit_.fits]
        )
        return self

    def _group(self, x: Any) -> "GroupFit":
        """``fit_`` for X=None, else the fits of X with the same settings."""
        check_is_fitted(self)
        if x is None:
            return self.fit_
        return self._fit_container(self._as_container(x, reset=False))

    def fit_transform(self, X: Any, y: Any = None) -> NDArray[Any]:  # noqa: N803  (scikit-learn API: X)
        """Fit X, then return its feature matrix (see :meth:`transform`)."""
        return self.fit(X, y).transform()

    def predict(self, X: Any = None) -> NDArray[Any]:  # noqa: N803  (scikit-learn API: X)
        """Fitted curves over the time axis, (n_waveforms, n_times).

        NaN outside the fit window and for a waveform without a model. X=None: the waveforms of
        ``fit``; else those of X fitted with the same settings.
        """
        group = self._group(X)
        out = np.full((len(group.fits), len(self.times_)), np.nan)
        for i, fit in enumerate(group.fits):
            if fit.model is not None:
                inside = np.flatnonzero(
                    (self.times_ >= fit.window[0] - 1e-12) & (self.times_ <= fit.window[1] + 1e-12)
                )
                n = min(len(inside), len(fit.model))
                out[i, inside[:n]] = fit.model[:n]
        return out

    def transform(self, X: Any = None) -> NDArray[Any]:  # noqa: N803  (scikit-learn API: X)
        """Feature matrix (n_waveforms, 3).

        Latency, amplitude and width of the largest component of each waveform (NaN without
        component). X=None: the waveforms of ``fit``; else those of X fitted with the same
        settings.
        """
        rows = []
        for fit in self._group(X).fits:
            if fit.components:
                comp = max(fit.components, key=lambda c: abs(c.amplitude))
                rows.append([comp.latency, comp.amplitude, comp.fwhm])
            else:
                rows.append([np.nan] * 3)
        return np.array(rows, dtype=float).reshape(-1, 3)

    def get_feature_names_out(self, input_features: Any = None) -> NDArray[Any]:  # noqa: ARG002  (scikit-learn API)
        """Return the names of the output features."""
        return np.array(["latency", "amplitude", "fwhm"])

    def score(self, X: Any = None, y: Any = None) -> float:  # noqa: N803, ARG002  (scikit-learn API: X, y)
        """Mean r2 over the waveforms that were fitted (of ``fit``, or of X)."""
        values = [f.gof.r2 for f in self._group(X).fits if f.gof is not None]
        return float(np.nanmean(values)) if values else np.nan


###################################################################################################
## Functions


def fit_response(
    times: Any, y: Any, *, window: Any, baseline: Any, preset: Any = None, **overrides: Any
) -> Any:
    """Fit one waveform with a preset and overrides; returns a :class:`ResponseFit`."""
    model = ResponseModel(preset, fit_window=window, baseline=baseline, **overrides)
    return model.fit_waveform(times, y)


def _from_mne(data: Any) -> Any:
    """MNE Epochs, Evoked or a list of Evoked as evoparam containers (others unchanged)."""
    kind = type(data).__module__.split(".")[0]
    if (
        isinstance(data, (list, tuple))
        and data
        and type(data[0]).__module__.split(".")[0] == "mne"
    ):
        return [Evoked.from_mne(evoked) for evoked in data]
    if kind != "mne":
        return data
    require("mne", extra="mne", caller="evoparam.models.api._from_mne")
    import mne

    if isinstance(data, mne.BaseEpochs):
        return TrialSet.from_mne(data)
    if isinstance(data, mne.Evoked):
        return Evoked.from_mne(data)
    raise DataError(f"fit: cannot use an MNE {type(data).__name__}; epoch it first.")


@verbose
def fit(  # noqa: PLR0913  (spec §5.5.8 fixes 14 parameters; DECISIONS.md §7.2)
    data: Any,
    preset: Any = None,
    *,
    window: Any = None,
    baseline: Any = None,
    times: Any = None,
    sfreq: float | None = None,
    picks: Any = None,
    by: Any = None,
    average: bool = True,
    trials: bool = False,
    existence: bool = False,
    n_jobs: int = 1,
    random_state: Any = None,
    verbose: Any = None,  # noqa: ARG001  (read by the @verbose decorator)
    **overrides: Any,
) -> "GroupFit":
    """Fit evoked responses with a preset: the one-call entry point.

    ``evoparam.fit(trials, preset="ccep")`` averages the trials (by condition with ``by``),
    fits every channel and returns a :class:`GroupFit`. ``data`` may be a TrialSet, an
    Evoked, an ``mne.Epochs``, an ``mne.Evoked``, a list of ``mne.Evoked`` (one condition
    each, ``comment`` as the condition label) or an array [n_channels, n_times] with
    ``times``; read files with :func:`evoparam.io.read` first.

    ``trials=True`` (trial-by-trial fits, build step S10) and ``existence=True`` (the
    existence test, build step S8) are not available yet and raise SettingsError.

    ``verbose`` sets the level of the ``evoparam`` logger for the call (spec §9.8): None or
    False inherit, True or ``"info"`` give one INFO record per fitted group, ``"debug"`` one
    DEBUG record per channel, ``"progress"`` adds a progress bar over the channels (extra
    ``progress``).
    """
    if trials:
        raise SettingsError("trials=True needs the trial-by-trial models (build step S10).")
    if existence:
        raise SettingsError(
            "existence=True: use evoparam.fit (evoparam.existence.fit), which "
            "runs the existence test after the fit."
        )
    if not average:
        raise SettingsError("average=False: fit trials one by one with trials=True (S10).")
    model = _model_for_fit(
        preset, window, baseline, n_jobs=n_jobs, random_state=random_state, overrides=overrides
    )
    data = _from_mne(data)
    if isinstance(data, list):
        # MNE Evoked objects, one condition each (their comment is the condition label)
        groups = [
            (evoked.provenance.source.get("comment") or str(k), evoked)
            for k, evoked in enumerate(data)
        ]
        return _fit_conditions(model, groups, picks)
    if isinstance(data, np.ndarray):
        if times is None:
            raise SettingsError("times is required when data is an array.")
        data = np.atleast_2d(data)
        rate = sfreq or 1.0 / np.median(np.diff(times))
        data = Evoked(
            data=data, times=times, sfreq=rate, ch_names=[f"ch{i}" for i in range(len(data))]
        )
    elif not isinstance(data, (TrialSet, Evoked)):
        raise DataError(
            f"fit: cannot use a {type(data).__name__}; convert it first with "
            "evoparam.io.read (MNE objects, files)."
        )
    if by is not None:
        if not isinstance(data, TrialSet):
            raise SettingsError("by= needs a TrialSet.")
        columns = [by] if isinstance(by, str) else list(by)
        groups = [
            (
                "/".join(map(str, key if isinstance(key, tuple) else (key,))),
                data.select(trials=list(rows)),
            )
            for key, rows in data.metadata.groupby(columns, sort=True).indices.items()
        ]
        return _fit_conditions(model, groups, picks)
    return model.fit_group(data, picks=picks)


def _model_for_fit(
    preset: Any, window: Any, baseline: Any, *, n_jobs: int, random_state: Any, overrides: Any
) -> "ResponseModel":
    """Build the model of :func:`fit` from the preset, window, baseline and overrides.

    ``overrides`` may be the settings of a result (REQ-331): the version that recorded them
    is checked (a warning when it is not this one) and the settings the constructor does not
    take are set on the model.
    """
    recorded_version = overrides.pop("package_version", None)
    # what the fits used (noise_source="auto" decides it again from the data): recorded, not
    # a setting, so that ``fit(data, **result.settings)`` takes it back (REQ-331)
    overrides.pop("noise_reference", None)
    if recorded_version is not None and recorded_version != __version__:
        # settings of a result made by another version (REQ-331: func(data, **settings))
        message = (
            f"evoparam.fit: the settings were recorded by evoparam "
            f"{recorded_version}; this is {__version__}."
        )
        _engine.logger.warning(message)
        _warn(message, UserWarning, stacklevel=3)
    if "fit_window" in overrides:
        # the name of the preset field (spec §6.3) and of FitSettings, for ``window``
        given = overrides.pop("fit_window")
        window = given if window is None else window
    # Settings that the constructor does not take (preset_note, freq_min, freq_max) are
    # read by ResponseModel.settings() from the model's attributes, so that the settings
    # of a result can be passed back as they are (REQ-331).
    accepted = set(inspect.signature(ResponseModel.__init__).parameters)
    extra = {
        k: overrides.pop(k)
        for k in list(overrides)
        if k not in accepted and k in FitSettings.__dataclass_fields__
    }
    model = ResponseModel(
        preset,
        fit_window=window,
        baseline=baseline,
        n_jobs=n_jobs,
        random_state=random_state,
        **overrides,
    )
    for key, value in extra.items():
        setattr(model, key, value)
    return model


def _fit_conditions(model: "ResponseModel", groups: Any, picks: Any) -> "GroupFit":
    """Fit each ``(label, data)`` of ``groups``; one GroupFit with the label as condition."""
    fits, chans, conds = [], [], []
    for label, data in groups:
        group = model.fit_group(data, picks=picks, condition=label)
        fits += group.fits
        chans += list(group.index["channel"])
        conds += [label] * len(group.fits)
    return GroupFit.from_fits(
        fits, chans, conds, settings=_settings_dict(model.settings(), model._explicit())
    )


def compare_models(fits: Any) -> Any:
    """Criterion table across fits of the same waveform under different settings."""
    require("pandas", extra="data", caller="evoparam.models.api.compare_models")
    import pandas as pd

    rows = []
    for i, f in enumerate(fits):
        rows.append(
            {
                "model": i,
                "shapes": ",".join(f.settings.get("shapes", [])),
                "background": f.background.get("name", ""),
                "n_components": len(f.components),
                "bic": f.gof.bic if f.gof else np.nan,
                "aicc": f.gof.aicc if f.gof else np.nan,
                "status": f.status,
            }
        )
    table = pd.DataFrame(rows)
    table["delta_bic"] = table["bic"] - table["bic"].min()
    table["delta_aicc"] = table["aicc"] - table["aicc"].min()
    return table


def _article(word: Any) -> Any:
    """Return the indefinite article of ``word``."""
    return "an" if str(word)[:1].lower() in "aeiou" else "a"


def _noise_phrase(settings: Any) -> str:
    """Return the methods-text phrase of the noise model of ``settings``."""
    noise = settings.get("noise_model", "ar")
    if noise == "neff":
        return (
            "residuals scaled by the baseline noise, with the n_eff correction "
            "(effective sample count n (1 - rho)/(1 + rho) of an AR(1) noise) (neff)"
        )
    if noise == "white":
        return "residuals scaled by the baseline noise (white)"
    return f"residuals whitened by an autoregressive model of the baseline noise ({noise})"


def methods_text(fit: Any) -> str:
    """Methods paragraph for a ResponseFit or GroupFit (spec REQ-123)."""
    settings = fit.settings
    shapes = ", ".join(settings.get("shapes", []))
    preset = preset_name(settings.get("preset"))
    window = settings.get("fit_window")
    crit = "BIC" if settings.get("selection") == "bic" else settings.get("selection", "").upper()
    return (
        f"Evoked responses were parameterized with evoparam {settings.get('package_version')}"
        + (f" ({settings.get('preset_note') or f'preset {preset!r}'})" if preset else "")
        + f". In the window {window[0] * 1000:g} to {window[1] * 1000:g} ms, each waveform was "
        f"modelled as {_article(settings.get('background'))} {settings.get('background')} "
        f"background plus up to "
        f"{settings.get('max_components')} components ({shapes}; widths "
        f"{settings.get('min_fwhm') * 1000:g} to {settings.get('max_fwhm') * 1000:g} ms full "
        f"width at half maximum). Amplitudes and background coefficients were solved by "
        f"variable projection and the other parameters by bounded least squares. Components "
        f"were added one at a time and kept when the {crit} decreased by at least "
        f"{settings.get('delta'):g}, computed on {_noise_phrase(settings)}. Uncertainty: "
        f"{settings.get('uncertainty')} ({settings.get('ci_level', 0.95) * 100:g} % intervals)."
    )
