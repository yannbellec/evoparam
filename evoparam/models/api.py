"""Public interface of the fitting engine: ResponseModel, fit, fit_response (spec §5.5.8)."""

import dataclasses
import inspect
import time as _time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.stats import chi2, norm

from evoparam.data import Component, Evoked, GoodnessOfFit, GroupFit, Provenance, ResponseFit
from evoparam.data import TrialSet
from evoparam.data.provenance import elapsed
from evoparam.utils._estimator import BaseEstimatorMixin, check_is_fitted, validate_array
from evoparam.errors import DataError, SettingsError, warn as _warn
from evoparam.shapes import compute_features
from evoparam.version import __version__

from . import engine as _engine
from ._settings import FitSettings, resolve_settings
from .classical import classical_measures
from .labels import label_components

__all__ = ["ResponseModel", "fit", "fit_response", "compare_models", "methods_text",
           "evaluate_fit", "fitted_peak"]

FEATURES_WITH_SE = ("latency", "amplitude", "fwhm", "onset", "offset", "area")


###################################################################################################
## Building the result

def _features(shape, tw, params, amp, settings, sfreq):
    return compute_features(shape, tw, params, amp, onset_fraction=settings.onset_fraction,
                            sfreq=sfreq, window=(tw[0], tw[-1]))


def _feature_vector(shape, tw, params, amp, settings, sfreq):
    f = _features(shape, tw, params, amp, settings, sfreq)
    return np.array([getattr(f, name) for name in FEATURES_WITH_SE], float)


def _diagnostics(core, settings):
    """Covariance of every parameter, conditioning and sloppy pairs (step 6)."""

    struct, theta, beta = core["struct"], core["theta"], core["beta"]
    jac = struct.full_jacobian(theta, beta)
    p = jac.shape[1]
    rw = struct.white(struct.y - struct.design(theta) @ beta)
    rss_w = float(np.sum(rw ** 2))
    dof = max(core["n_white"] - p, 1.0)
    s2 = rss_w / dof
    jtj = jac.T @ jac
    cov = s2 * np.linalg.pinv(jtj)
    norms = np.sqrt(np.clip(np.diag(jtj), 1e-300, None))
    scaled = jtj / np.outer(norms, norms)
    eig = np.linalg.eigvalsh(scaled)
    cond = float(eig.max() / max(eig.min(), 1e-300)) if eig.size else 1.0
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(sd, sd)
    names = []
    for k, shape in enumerate(core["shapes"]):
        names += [f"{k}:{n}" for n in shape.param_names]
    names += [f"bg:{n}" for n in getattr(struct.background, "nonlinear_names", ())]
    names += [f"{k}:A" for k in range(len(core["shapes"]))]
    names += [f"bg:{n}" for n in getattr(struct.background, "linear_names", ())]
    flagged = [(names[i], names[j], float(corr[i, j])) for i in range(p) for j in range(i + 1, p)
               if np.isfinite(corr[i, j]) and abs(corr[i, j]) > _engine.CORR_LIMIT]
    return {"cov": cov, "cond": cond, "eigvals": eig.tolist(), "corr": np.nan_to_num(corr),
            "flagged_pairs": flagged, "param_names": names, "s2": s2,
            "ill_conditioned": cond > _engine.COND_LIMIT or bool(flagged)}


def _component_uncertainty(core, diag, k, settings, sfreq, z):
    """Wald intervals for the parameters and delta-method intervals for the features."""

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
    f0 = _feature_vector(shape, struct.tw, params, amp, settings, sfreq)
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
        fp = _feature_vector(shape, struct.tw, xp[:-1], xp[-1], settings, sfreq)
        fm = _feature_vector(shape, struct.tw, xm[:-1], xm[-1], settings, sfreq)
        grads[:, i] = (fp - fm) / step
    for j, name in enumerate(FEATURES_WITH_SE):
        g = np.nan_to_num(grads[j])
        s = float(np.sqrt(max(g @ sub @ g, 0)))
        se[name] = s if name != "amplitude" else max(s, 0.0)
        ci[name] = (f0[j] - z * s, f0[j] + z * s)
    se["A"], ci["A"] = s_amp, (amp - z * s_amp, amp + z * s_amp)
    return se, ci


def _gof(core, y_window, valid, settings):
    resid = core["residual"][valid]
    yv = y_window[valid]
    n = len(yv)
    p = core["struct"].n_theta + len(core["beta"])
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    struct = core["struct"]
    rw = struct.white(struct.y - struct.design(core["theta"]) @ core["beta"])
    rss_w = float(np.sum(rw ** 2))
    n_eff = core["n_white"]
    lag = (float(np.corrcoef(resid[1:], resid[:-1])[0, 1]) if n > 3 and np.std(resid) > 0
           else np.nan)
    rmse = float(np.sqrt(rss / n))
    span = float(yv.max() - yv.min()) if n else np.nan
    return GoodnessOfFit(
        r2=r2, adj_r2=1 - (1 - r2) * (n - 1) / max(n - p - 1, 1) if np.isfinite(r2) else np.nan,
        rmse=rmse, mae=float(np.mean(np.abs(resid))), nrmse=rmse / span if span else np.nan,
        aic=_engine.criterion(rss_w, n_eff, p, "aic"),
        aicc=_engine.criterion(rss_w, n_eff, p, "aicc"),
        bic=_engine.criterion(rss_w, n_eff, p, "bic"), n_params=p, n_samples=n, n_eff=n_eff,
        residual_lag1_autocorr=lag, baseline_sd=core["sigma"])


def _bounds_warnings(core):
    out = []
    struct = core["struct"]
    for k, shape in enumerate(core["shapes"]):
        lo, hi = shape.default_bounds(struct.tw)
        params = struct.comp_params(core["theta"], k)
        span = hi - lo
        for i, name in enumerate(shape.param_names):
            if params[i] - lo[i] <= _engine.BOUND_TOL * span[i] or \
                    hi[i] - params[i] <= _engine.BOUND_TOL * span[i]:
                out.append(f"at_bound:{k}:{name}")
    return out


def _build_fit(core, times, y, settings, sfreq, trials, mask, start, settings_dict):
    window = settings.fit_window
    baseline = settings.baseline if settings.baseline is not None else (np.nan, np.nan)
    warns = list(core.get("warnings", []))
    provenance = Provenance().with_step("evoparam.models.fit_waveform", settings_dict,
                                        duration_s=None)
    if core["status"] in ("skipped", "failed") or "struct" not in core:
        return ResponseFit(components=[], background={}, gof=None, status=core["status"],
                           message=core["message"], window=window, baseline=baseline,
                           settings=settings_dict, warnings=warns, provenance=provenance,
                           diagnostics={"sigma": core.get("sigma"), "rho": core.get("rho"),
                                        "n_evaluations": core.get("n_evaluations", 0)})

    struct = core["struct"]
    wmask = core["window_mask"]
    valid = ~(mask[wmask] | ~np.isfinite(y[wmask]))
    z = norm.ppf(0.5 + settings.ci_level / 2)
    diag = _diagnostics(core, settings) if core["shapes"] else \
        {"cov": np.zeros((0, 0)), "cond": 1.0, "eigvals": [], "corr": np.zeros((0, 0)),
         "flagged_pairs": [], "param_names": [], "ill_conditioned": False}
    if diag["ill_conditioned"]:
        warns.append("ill_conditioned")
    warns += _bounds_warnings(core)

    comps = []
    for k, shape in enumerate(core["shapes"]):
        params = struct.comp_params(core["theta"], k)
        amp = float(core["beta"][k])
        f = _features(shape, struct.tw, params, amp, settings, sfreq)
        se, ci = _component_uncertainty(core, diag, k, settings, sfreq, z)
        warns += [f"{w}:{k}" for w in f.warnings]
        comps.append(Component(shape=shape.name, params=params, param_names=shape.param_names,
                               latency=f.latency, amplitude=f.amplitude,
                               polarity=1 if f.amplitude >= 0 else -1, fwhm=f.fwhm,
                               onset=f.onset, offset=f.offset, area=f.area,
                               rise_time=f.rise_time, decay_time=f.decay_time,
                               symmetry=f.symmetry, sharpness=f.sharpness, se=se, ci=ci))

    bg = struct.background
    n_lin = getattr(bg, "n_linear", 0)
    bg_values = dict(zip(getattr(bg, "linear_names", ()), core["beta"][len(comps):len(comps) + n_lin]))
    bg_values.update(zip(getattr(bg, "nonlinear_names", ()), struct.bg_params(core["theta"])))
    background = {"name": bg.name, **{k: float(v) for k, v in bg_values.items()}}

    classical = {}
    if settings.classical:
        classical = classical_measures(times, np.nan_to_num(y), window=window,
                                       polarity="negative", trials=trials,
                                       n_bootstrap=200, random_state=settings.random_state)

    fit = ResponseFit(components=comps, background=background,
                      gof=_gof(core, np.where(valid, y[wmask], 0.0), valid, settings),
                      status=core["status"], message=core["message"], window=window,
                      baseline=baseline, settings=settings_dict, classical=classical,
                      model=core["model"], residual=core["residual"], warnings=warns,
                      provenance=provenance,
                      diagnostics={"cond": diag["cond"], "eigvals": diag["eigvals"],
                                   "corr": np.asarray(diag["corr"]).tolist(),
                                   "flagged_pairs": diag["flagged_pairs"],
                                   "param_names": diag["param_names"],
                                   "best_rejected_delta": core["best_rejected_delta"],
                                   "sigma": core["sigma"], "rho": core["rho"],
                                   "n_evaluations": core.get("n_evaluations", 0),
                                   "n_eff": core["n_eff"], "n_white": core["n_white"],
                                   "ar_order": core["ar_order"], "criterion": core["kind"],
                                   "uncertainty_method": settings.uncertainty,
                                   "steps": core["steps"],
                                   "duration_s": elapsed(start)})
    if settings.uncertainty == "bootstrap":
        fit = _bootstrap(fit, core, times, trials, settings, sfreq, z)
    elif settings.uncertainty == "profile":
        fit = _profile(fit, core, settings, sfreq)
    elif settings.uncertainty == "posterior":
        raise SettingsError("uncertainty='posterior' comes with the hierarchical models "
                            "(build step S11).")
    if settings.windows:
        fit = label_components(fit, settings.windows)
    return fit


def _bootstrap(fit, core, times, trials, settings, sfreq, z):
    """Resample trials, re-average, refit the same model from the full-data solution."""

    if trials is None:
        raise SettingsError("uncertainty='bootstrap' needs the trials (fit a TrialSet, or "
                            "pass trials= to fit_waveform).")
    rng = np.random.default_rng(settings.random_state)
    struct = core["struct"]
    wmask = core["window_mask"]
    trials = np.asarray(trials, float)
    n = len(trials)
    draws = {k: [] for k in range(len(fit.components))}
    for _ in range(settings.n_bootstrap):
        avg = trials[rng.integers(0, n, n)].mean(axis=0)[wmask]
        boot = dataclasses.replace(struct, y=np.nan_to_num(avg))
        theta, beta, _, ok, _ = _engine._optimise(boot, core["theta"].copy(), settings,
                                                  settings.f_scale)
        if not ok:
            continue
        for k, shape in enumerate(struct.shapes):
            draws[k].append(_feature_vector(shape, struct.tw, boot.comp_params(theta, k),
                                            beta[k], settings, sfreq))
    alpha = 1 - settings.ci_level
    comps = []
    for k, comp in enumerate(fit.components):
        values = np.array(draws[k])
        se, ci = dict(comp.se), dict(comp.ci)
        if len(values) >= 10:
            for j, name in enumerate(FEATURES_WITH_SE):
                col = values[:, j][np.isfinite(values[:, j])]
                if len(col) < 10:
                    continue
                se[name] = float(np.std(col, ddof=1))
                ci[name] = tuple(np.percentile(col, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
        comps.append(dataclasses.replace(comp, se=se, ci=ci))
    diag = dict(fit.diagnostics, n_bootstrap_ok=min((len(v) for v in draws.values()),
                                                    default=0))
    return dataclasses.replace(fit, components=comps, diagnostics=diag)


def _profile(fit, core, settings, sfreq, n_grid=21):
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
            inside = []
            for value in grid:
                lo, hi = lower.copy(), upper.copy()
                eps = 1e-9 * (upper[j] - lower[j])
                lo[j], hi[j] = value - eps, value + eps
                fixed = _FixedStructure(struct, lo, hi)
                t0 = theta.copy()
                t0[j] = value
                th, be, rw, ok, _ = _engine._optimise(fixed, t0, settings, settings.f_scale)
                if ok and (np.sum(rw ** 2) - rss0) / s2 <= limit:
                    inside.append(value)
            if inside:
                ci[name] = (float(min(inside)), float(max(inside)))
                if len(inside) == n_grid:
                    flags.append(f"structurally_unidentifiable:{k}:{name}")
                elif min(inside) == grid[0] or max(inside) == grid[-1]:
                    flags.append(f"practically_unidentifiable:{k}:{name}")
        comps.append(dataclasses.replace(comp, ci=ci))
    return dataclasses.replace(fit, components=comps, warnings=list(fit.warnings) + flags)


def _cv_selector(trials, times, settings):
    """K-fold cross-validation over trials of the nested models K = 0, 1, ...: each model is
    refitted on the average of the training trials and scored on the held-out average; the
    smallest K within one standard error of the best mean error is chosen."""

    def select(path, make):
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
                theta, beta, _, ok, _ = _engine._optimise(struct, state.theta.copy(), settings,
                                                          settings.f_scale, tol=1e-6)
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

    def __init__(self, base, lower, upper):
        super().__init__(base.shapes, base.background, base.tw, base.white, base.y, base.prior, base.scale, base.edge,
                         base.level)
        self._lower, self._upper = lower, upper

    def bounds(self):
        return self._lower, self._upper


###################################################################################################
## Estimator

# Fields that are not settings of the preset: the preset itself and the random seed.
_NOT_PRESET_FIELDS = ("preset", "preset_note", "random_state")


def _changed_from_preset(settings):
    """Names of the fields whose value differs from the preset's (or, without a preset,
    from the package defaults): the settings a methods text must state (REQ-008, REQ-340)."""

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


def _draw_seed():
    """A seed drawn from the operating system, recorded in the settings (REQ-295): a fit
    made with ``random_state=None`` can be repeated from its settings (REQ-331)."""

    return int(np.random.default_rng().integers(2 ** 31 - 1))


def preset_record(settings):
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


def preset_name(value):
    """Name of a recorded preset: ``settings["preset"]`` is ``{"name", "version"}``
    (REQ-331); results written before it hold the name alone."""

    if isinstance(value, dict):
        return value.get("name")
    return value


def _settings_dict(settings, overrides=()):
    """The ``settings`` record of a result (REQ-331): the full resolved settings, the preset
    as ``{"name", "version"}``, ``_overrides`` (the keys the user set) and
    ``_changed_from_preset`` (the keys whose value differs from the preset, stated by the
    methods text)."""

    out = dataclasses.asdict(settings)
    out["shapes"] = list(settings.shapes)
    out["preset"] = preset_record(settings)
    out["package_version"] = __version__
    out["_overrides"] = list(overrides)
    out["_changed_from_preset"] = _changed_from_preset(settings)
    return out


def evaluate_fit(fit, times, *, components=None, background=True):
    """Curve of a ResponseFit on any time grid (all components, or a subset)."""

    from evoparam.shapes import get_background, get_shape

    times = np.asarray(times, float)
    total = np.zeros_like(times)
    for k, comp in enumerate(fit.components):
        if components is None or k in components:
            total += _amplitude(fit, k) * get_shape(comp.shape).evaluate(times, comp.params)
    if background and fit.background:
        bg = get_background(fit.background["name"])
        lin = [fit.background[n] for n in getattr(bg, "linear_names", ())]
        nl = [fit.background[n] for n in getattr(bg, "nonlinear_names", ())]
        if lin:
            basis = bg.basis(times, np.array(nl) if nl else None)
            total += np.asarray(lin) @ np.atleast_2d(basis)
    return total


def fitted_peak(fit, window, *, polarity="negative", step=1e-4):
    """Peak of the fitted waveform (background and components) inside a window.

    The deepest interior local minimum (``polarity="negative"``) or highest interior local
    maximum (``"positive"``) of the fitted curve between ``window[0]`` and ``window[1]``,
    on a grid of ``step`` seconds; a curve that is monotone through the window has no peak.
    The value is relative to zero (baseline-corrected data) and must have the sign of the
    polarity.

    Why: when an early component overlaps the recovery tail (the exponential background),
    the split of the curve between them is sloppy (often flagged ``ill_conditioned``), so a
    component's own latency can lie several ms from the trough of the waveform while the
    fitted curve itself is well determined. On ds004080 the N1 latency of the component was
    within 2 ms of the published rule's peak on about a quarter of the channels where both
    find an N1, the trough of the fitted curve on about nine in ten (CHANGELOG).

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
    grid = np.arange(window[0], window[1] + step / 2, step)
    if len(grid) < 3:
        return out
    curve = evaluate_fit(fit, grid)
    s = -curve if polarity == "negative" else curve
    inner = np.flatnonzero((s[1:-1] >= s[:-2]) & (s[1:-1] > s[2:])) + 1
    inner = inner[s[inner] > 0]
    if inner.size:
        j = inner[np.argmax(s[inner])]
        out = {"latency": float(grid[j]), "amplitude": float(curve[j])}
    return out


def _amplitude(fit, k):
    """Linear amplitude A of component k: the centre of its (symmetric) Wald interval."""

    comp = fit.components[k]
    ci = comp.ci or {}
    if "A" in ci:
        return 0.5 * (ci["A"][0] + ci["A"][1])
    return comp.amplitude


class ResponseModel(BaseEstimatorMixin):
    """Estimator for evoked waveforms (spec §5.5.8): settings in ``__init__``, learned
    attributes with a trailing underscore after :meth:`fit`.

    Keywords left at None take the preset's value, or the package default when there is
    no preset (see :class:`~evoparam.models.FitSettings`).
    """

    def __init__(self, preset=None, *, shapes=None, background=None, max_components=None,
                 min_fwhm=None, max_fwhm=None, min_amplitude=None, detection_threshold=None,
                 polarity=None, fit_window=None, baseline=None, sfreq=None, tmin=None,
                 selection=None, delta=None, n_components=None, overlap_c=None, loss=None,
                 f_scale=None, max_iter=None, n_starts=None, onset_fraction=None,
                 uncertainty=None, n_bootstrap=None, ci_level=None, classical=None,
                 windows=None, noise_model=None, background_prior=None, amplitude_prior=None,
                 edge_extension=None, background_tau_max=None, max_masked_fraction=None,
                 n_folds=None, n_jobs=1, random_state=None, verbose=None):
        for name, value in locals().items():
            if name != "self":
                setattr(self, name, value)

    @classmethod
    def _param_names(cls):
        return cls._get_param_names()

    def _explicit(self, **extra):
        """Keys of FitSettings that the user set (REQ-331 ``_overrides``), in field order."""

        recorded = getattr(self, "_overrides_record", None)
        if recorded is not None:
            return list(recorded)
        given = {k for k in FitSettings.__dataclass_fields__
                 if getattr(self, k, None) is not None or extra.get(k) is not None}
        return [k for k in FitSettings.__dataclass_fields__
                if k in given and k not in _NOT_PRESET_FIELDS]

    def settings(self, **extra):
        keys = set(FitSettings.__dataclass_fields__) - {"preset"}
        overrides = {k: getattr(self, k) for k in keys if hasattr(self, k)}
        overrides.update({k: v for k, v in extra.items() if v is not None})
        return resolve_settings(self.preset, **overrides)

    def fit_waveform(self, times, y, *, window=None, baseline=None, sem=None, mask=None,
                     trials=None, random_state=None):
        """Fit one waveform; returns a :class:`ResponseFit` (never raises on a bad fit).

        ``random_state`` replaces the estimator's for this call. When both are None a seed
        is drawn from the operating system and recorded in ``settings`` (REQ-295)."""

        start = _time.perf_counter()
        if random_state is None and self.random_state is None:
            random_state = _draw_seed()
        settings = self.settings(fit_window=window, baseline=baseline,
                                 random_state=random_state)
        times = np.asarray(times, float)
        y = np.asarray(y, float)
        if times.shape != y.shape:
            raise DataError("times and y must have the same shape.")
        if settings.fit_window is None:
            raise SettingsError("A fit window is needed (fit_window=, window= or a preset).")
        if settings.fit_window[0] < times[0] or settings.fit_window[1] > times[-1]:
            raise DataError(f"fit window {settings.fit_window} exceeds the data "
                            f"({times[0]:.4f} to {times[-1]:.4f} s).")
        sfreq = 1.0 / np.median(np.diff(times))
        mask = np.zeros(len(y), bool) if mask is None else np.asarray(mask, bool)
        select_k = None
        if settings.selection == "cv":
            if trials is None:
                raise SettingsError("selection='cv' needs the trials (fit a TrialSet, or pass "
                                    "trials= to fit_waveform).")
            select_k = _cv_selector(np.asarray(trials, float), times, settings)
        try:
            core = _engine.fit_waveform_core(times, y, settings=settings, sfreq=sfreq,
                                             mask=mask, sem=sem, select_k=select_k)
        except SettingsError:
            raise
        except Exception as exc:                      # never NaN with status "ok"
            core = {"status": "failed", "message": f"{type(exc).__name__}: {exc}",
                    "warnings": []}
            _engine.logger.warning("fit failed: %s", core["message"])
        return _build_fit(core, times, y, settings, sfreq, trials, mask, start,
                          _settings_dict(settings, self._explicit(fit_window=window,
                                                                  baseline=baseline)))

    def fit_group(self, data, *, window=None, baseline=None, picks=None, n_jobs=None,
                  condition=""):
        """Fit every channel of an Evoked (or of the average of a TrialSet)."""

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
        index = range(len(names)) if picks is None else \
            [names.index(p) if isinstance(p, str) else int(p) for p in picks]

        # one seed for every channel, drawn when none is given and recorded (REQ-295)
        seed = _draw_seed() if self.random_state is None else None

        def one(ch):
            mask = None if evoked.masks is None else evoked.masks[ch]
            sem = None if evoked.sem is None else evoked.sem[ch]
            tr = None if trials is None else trials.data[:, ch]
            return self.fit_waveform(evoked.times, evoked.data[ch], window=window,
                                     baseline=baseline, sem=sem, mask=mask, trials=tr,
                                     random_state=seed)

        jobs = self.n_jobs if n_jobs is None else n_jobs
        if jobs and jobs != 1:
            with ThreadPoolExecutor(max_workers=None if jobs < 0 else jobs) as pool:
                fits = list(pool.map(one, index))
        else:
            fits = [one(ch) for ch in index]
        resolved = self.settings(fit_window=window, baseline=baseline, random_state=seed)
        settings = _settings_dict(resolved, self._explicit(fit_window=window,
                                                           baseline=baseline))
        provenance = evoked.provenance.with_step("evoparam.models.ResponseModel.fit_group",
                                                 settings,
                                                 duration_s=_time.perf_counter() - start,
                                                 preset=preset_record(resolved))
        return GroupFit.from_fits(fits, [names[i] for i in index],
                                  [condition] * len(fits), settings=settings,
                                  provenance=provenance)

    def _as_container(self, X, *, reset):
        """X as an Evoked or a TrialSet. An array is (n_waveforms, n_times) at ``sfreq``
        from ``tmin``, checked the scikit-learn way (2-D, dense, real, finite; after ``fit``
        as many time points as in ``fit``)."""

        X = _from_mne(X)
        if isinstance(X, (TrialSet, Evoked)):
            if reset:
                self.n_features_in_ = int(X.n_times)
            elif X.n_times != getattr(self, "n_features_in_", X.n_times):
                raise DataError(f"X has {X.n_times} time points, but {type(self).__name__} "
                                f"is expecting {self.n_features_in_}.")
            return X
        data = validate_array(self, X, reset=reset)
        if self.sfreq is None:
            raise SettingsError("sfreq is required when X is an array.")
        tmin = 0.0 if self.tmin is None else self.tmin
        times = tmin + np.arange(data.shape[-1]) / self.sfreq
        return Evoked(data=data, times=times, sfreq=self.sfreq,
                      ch_names=[f"ch{i}" for i in range(len(data))])

    def _fit_container(self, X):
        """GroupFit of a container. Without a fit window (no ``fit_window``, no preset) the
        window is the data range; when that range cannot hold a component (shorter than
        3 x min_fwhm, or fewer than 5 samples) every waveform is returned as skipped
        rather than raising, so that short tabular inputs keep the estimator contract."""

        settings = self.settings()
        if settings.fit_window is not None:
            return self.fit_group(X)
        times = np.asarray(X.times, float)
        window = (float(times[0]), float(times[-1]))
        sfreq = float(X.sfreq)
        min_fwhm = max(settings.min_fwhm, 2.0 / sfreq)
        if window[1] - window[0] >= 3 * min_fwhm and len(times) >= 5:
            return self.fit_group(X, window=window)
        settings = self.settings(fit_window=window)
        evoked = X.average() if isinstance(X, TrialSet) else X
        message = (f"skipped: the data range {window} is shorter than 3 x min_fwhm "
                   f"({3 * min_fwhm:.4f} s) or has fewer than 5 samples")
        explicit = self._explicit(fit_window=window)
        fits = [_build_fit({"status": "skipped", "message": message, "warnings": []},
                           times, np.asarray(row, float), settings, sfreq, None, None,
                           _time.perf_counter(), _settings_dict(settings, explicit))
                for row in evoked.data]
        return GroupFit.from_fits(fits, list(evoked.ch_names), [""] * len(fits),
                                  settings=_settings_dict(settings, explicit),
                                  provenance=evoked.provenance)

    def fit(self, X, y=None):
        """Fit every channel. Learned: ``fit_`` (GroupFit), ``components_`` (table),
        ``times_``, ``n_features_in_`` (samples per waveform) and ``n_iter_`` (most residual
        evaluations of one waveform's least-squares solves, at least 1). X: Evoked, TrialSet, MNE
        object or array (n_waveforms, n_times); ``y`` is ignored."""

        X = self._as_container(X, reset=True)
        self.fit_ = self._fit_container(X)
        self.components_ = self.fit_.to_dataframe()
        self.times_ = np.array(X.times, float)
        # the largest number of residual evaluations of one waveform (at least 1)
        self.n_iter_ = max([1] + [int(f.diagnostics.get("n_evaluations") or 0)
                                  for f in self.fit_.fits])
        return self

    def _group(self, X):
        """``fit_`` for X=None, else the fits of X with the same settings."""

        check_is_fitted(self)
        if X is None:
            return self.fit_
        return self._fit_container(self._as_container(X, reset=False))

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform()

    def predict(self, X=None):
        """Fitted curves over the time axis, (n_waveforms, n_times); NaN outside the fit
        window and for a waveform without a model. X=None: the waveforms of ``fit``; else
        those of X fitted with the same settings."""

        group = self._group(X)
        out = np.full((len(group.fits), len(self.times_)), np.nan)
        for i, fit in enumerate(group.fits):
            if fit.model is not None:
                inside = np.flatnonzero((self.times_ >= fit.window[0] - 1e-12) &
                                        (self.times_ <= fit.window[1] + 1e-12))
                n = min(len(inside), len(fit.model))
                out[i, inside[:n]] = fit.model[:n]
        return out

    def transform(self, X=None):
        """Feature matrix (n_waveforms, 3): latency, amplitude and width of the largest
        component of each waveform (NaN without component). X=None: the waveforms of
        ``fit``; else those of X fitted with the same settings."""

        rows = []
        for fit in self._group(X).fits:
            if fit.components:
                comp = max(fit.components, key=lambda c: abs(c.amplitude))
                rows.append([comp.latency, comp.amplitude, comp.fwhm])
            else:
                rows.append([np.nan] * 3)
        return np.array(rows, dtype=float).reshape(-1, 3)

    def get_feature_names_out(self, input_features=None):
        return np.array(["latency", "amplitude", "fwhm"])

    def score(self, X=None, y=None):
        """Mean r2 over the waveforms that were fitted (of ``fit``, or of X)."""

        values = [f.gof.r2 for f in self._group(X).fits if f.gof is not None]
        return float(np.nanmean(values)) if values else np.nan


###################################################################################################
## Functions

def fit_response(times, y, *, window, baseline, preset=None, **overrides):
    """Fit one waveform with a preset and overrides; returns a :class:`ResponseFit`."""

    model = ResponseModel(preset, fit_window=window, baseline=baseline, **overrides)
    return model.fit_waveform(times, y)


def _from_mne(data):
    """MNE Epochs, Evoked or a list of Evoked as evoparam containers (others unchanged)."""

    kind = type(data).__module__.split(".")[0]
    if isinstance(data, (list, tuple)) and data and \
            type(data[0]).__module__.split(".")[0] == "mne":
        return [Evoked.from_mne(evoked) for evoked in data]
    if kind != "mne":
        return data
    import mne

    if isinstance(data, mne.BaseEpochs):
        return TrialSet.from_mne(data)
    if isinstance(data, mne.Evoked):
        return Evoked.from_mne(data)
    raise DataError(f"fit: cannot use an MNE {type(data).__name__}; epoch it first.")


def fit(data, preset=None, *, window=None, baseline=None, times=None, sfreq=None, picks=None,
        by=None, average=True, trials=False, existence=False, n_jobs=1, random_state=None,
        verbose=None, **overrides):
    """Fit evoked responses with a preset: the one-call entry point.

    ``evoparam.fit(trials, preset="ccep")`` averages the trials (by condition with ``by``),
    fits every channel and returns a :class:`GroupFit`. ``data`` may be a TrialSet, an
    Evoked or an array [n_channels, n_times] with ``times`` (convert MNE objects and files
    with :func:`evoparam.io.read` first).

    ``trials=True`` (trial-by-trial fits, build step S10) and ``existence=True`` (the
    existence test, build step S8) are not available yet and raise SettingsError.
    """

    if trials:
        raise SettingsError("trials=True needs the trial-by-trial models (build step S10).")
    if existence:
        raise SettingsError("existence=True: use evoparam.fit (evoparam.existence.fit), which "
                            "runs the existence test after the fit.")
    if not average:
        raise SettingsError("average=False: fit trials one by one with trials=True (S10).")
    recorded_version = overrides.pop("package_version", None)
    if recorded_version is not None and recorded_version != __version__:
        # settings of a result made by another version (REQ-331: func(data, **settings))
        message = (f"evoparam.fit: the settings were recorded by evoparam "
                   f"{recorded_version}; this is {__version__}.")
        _engine.logger.warning(message)
        _warn(message, UserWarning, stacklevel=2)
    if "fit_window" in overrides:
        # the name of the preset field (spec §6.3) and of FitSettings, for ``window``
        given = overrides.pop("fit_window")
        window = given if window is None else window
    # Settings that the constructor does not take (preset_note, freq_min, freq_max) are
    # read by ResponseModel.settings() from the model's attributes, so that the settings
    # of a result can be passed back as they are (REQ-331).
    accepted = set(inspect.signature(ResponseModel.__init__).parameters)
    extra = {k: overrides.pop(k) for k in list(overrides)
             if k not in accepted and k in FitSettings.__dataclass_fields__}
    model = ResponseModel(preset, fit_window=window, baseline=baseline, n_jobs=n_jobs,
                          random_state=random_state, **overrides)
    for key, value in extra.items():
        setattr(model, key, value)
    data = _from_mne(data)
    if isinstance(data, list):
        # MNE Evoked objects, one condition each (their comment is the condition label)
        fits, chans, conds = [], [], []
        for k, evoked in enumerate(data):
            label = evoked.provenance.source.get("comment") or str(k)
            group = model.fit_group(evoked, picks=picks, condition=label)
            fits += group.fits
            chans += list(group.index["channel"])
            conds += [label] * len(group.fits)
        return GroupFit.from_fits(fits, chans, conds, settings=_settings_dict(model.settings(), model._explicit()))
    if isinstance(data, np.ndarray):
        if times is None:
            raise SettingsError("times is required when data is an array.")
        data = np.atleast_2d(data)
        rate = sfreq or 1.0 / np.median(np.diff(times))
        data = Evoked(data=data, times=times, sfreq=rate,
                      ch_names=[f"ch{i}" for i in range(len(data))])
    elif not isinstance(data, (TrialSet, Evoked)):
        raise DataError(f"fit: cannot use a {type(data).__name__}; convert it first with "
                        "evoparam.io.read (MNE objects, files).")
    if by is not None:
        if not isinstance(data, TrialSet):
            raise SettingsError("by= needs a TrialSet.")
        columns = [by] if isinstance(by, str) else list(by)
        fits, chans, conds = [], [], []
        for key, rows in data.metadata.groupby(columns, sort=True).indices.items():
            label = "/".join(map(str, key if isinstance(key, tuple) else (key,)))
            group = model.fit_group(data.select(trials=list(rows)), picks=picks, condition=label)
            fits += group.fits
            chans += list(group.index["channel"])
            conds += [label] * len(group.fits)
        return GroupFit.from_fits(fits, chans, conds, settings=_settings_dict(model.settings(), model._explicit()))
    return model.fit_group(data, picks=picks)


def compare_models(fits):
    """Criterion table across fits of the same waveform under different settings."""

    import pandas as pd

    rows = []
    for i, f in enumerate(fits):
        rows.append({"model": i, "shapes": ",".join(f.settings.get("shapes", [])),
                     "background": f.background.get("name", ""),
                     "n_components": len(f.components),
                     "bic": f.gof.bic if f.gof else np.nan,
                     "aicc": f.gof.aicc if f.gof else np.nan, "status": f.status})
    table = pd.DataFrame(rows)
    table["delta_bic"] = table["bic"] - table["bic"].min()
    table["delta_aicc"] = table["aicc"] - table["aicc"].min()
    return table


def _article(word):
    return "an" if str(word)[:1].lower() in "aeiou" else "a"


def methods_text(fit):
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
        f"{settings.get('delta'):g}, computed on residuals whitened by an autoregressive "
        f"model of the baseline noise ({settings.get('noise_model', 'ar')}). Uncertainty: "
        f"{settings.get('uncertainty')} ({settings.get('ci_level', 0.95) * 100:g} % intervals)."
    )
