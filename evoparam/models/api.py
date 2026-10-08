"""Public interface of the fitting engine: ResponseModel, fit, fit_response (spec §5.5.8)."""

import dataclasses
import inspect
import time as _time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.stats import chi2, norm

from evoparam.data import Component, Evoked, GoodnessOfFit, GroupFit, Provenance, ResponseFit
from evoparam.data import TrialSet
from evoparam.errors import DataError, SettingsError
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
                           diagnostics={"sigma": core.get("sigma"), "rho": core.get("rho")})

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
                                   "n_eff": core["n_eff"], "n_white": core["n_white"],
                                   "ar_order": core["ar_order"], "criterion": core["kind"],
                                   "uncertainty_method": settings.uncertainty,
                                   "steps": core["steps"],
                                   "duration_s": _time.perf_counter() - start})
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

def _settings_dict(settings):
    out = dataclasses.asdict(settings)
    out["shapes"] = list(settings.shapes)
    out["package_version"] = __version__
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


class ResponseModel:
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

    # scikit-learn conventions
    @classmethod
    def _param_names(cls):
        return [p for p in inspect.signature(cls.__init__).parameters if p != "self"]

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params):
        for name, value in params.items():
            if name not in self._param_names():
                raise SettingsError(f"Unknown parameter {name!r}.")
            setattr(self, name, value)
        return self

    def settings(self, **extra):
        keys = set(FitSettings.__dataclass_fields__) - {"preset"}
        overrides = {k: getattr(self, k) for k in keys if hasattr(self, k)}
        overrides.update({k: v for k, v in extra.items() if v is not None})
        return resolve_settings(self.preset, **overrides)

    def fit_waveform(self, times, y, *, window=None, baseline=None, sem=None, mask=None,
                     trials=None):
        """Fit one waveform; returns a :class:`ResponseFit` (never raises on a bad fit)."""

        start = _time.perf_counter()
        settings = self.settings(fit_window=window, baseline=baseline)
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
                          _settings_dict(settings))

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

        def one(ch):
            mask = None if evoked.masks is None else evoked.masks[ch]
            sem = None if evoked.sem is None else evoked.sem[ch]
            tr = None if trials is None else trials.data[:, ch]
            return self.fit_waveform(evoked.times, evoked.data[ch], window=window,
                                     baseline=baseline, sem=sem, mask=mask, trials=tr)

        jobs = self.n_jobs if n_jobs is None else n_jobs
        if jobs and jobs != 1:
            with ThreadPoolExecutor(max_workers=None if jobs < 0 else jobs) as pool:
                fits = list(pool.map(one, index))
        else:
            fits = [one(ch) for ch in index]
        settings = _settings_dict(self.settings(fit_window=window, baseline=baseline))
        provenance = evoked.provenance.with_step("evoparam.models.ResponseModel.fit_group",
                                                 settings,
                                                 duration_s=_time.perf_counter() - start)
        return GroupFit.from_fits(fits, [names[i] for i in index],
                                  [condition] * len(fits), settings=settings,
                                  provenance=provenance)

    def fit(self, X, y=None):
        """Fit every channel; learned attributes ``fit_``, ``components_``,
        ``n_features_in_``."""

        if isinstance(X, np.ndarray):
            if self.sfreq is None:
                raise SettingsError("sfreq is required when X is an array.")
            data = np.atleast_2d(X)
            tmin = 0.0 if self.tmin is None else self.tmin
            times = tmin + np.arange(data.shape[-1]) / self.sfreq
            X = Evoked(data=data, times=times, sfreq=self.sfreq,
                       ch_names=[f"ch{i}" for i in range(len(data))])
        self.fit_ = self.fit_group(X)
        self.components_ = self.fit_.to_dataframe()
        self.n_features_in_ = len(self.fit_.fits)
        return self

    def predict(self, X=None):
        """Fitted curves of the window of every channel."""

        return [fit.model for fit in self.fit_.fits]

    def transform(self, X=None):
        """Feature matrix: latency and amplitude of the largest component per channel."""

        rows = []
        for fit in self.fit_.fits:
            if fit.components:
                comp = max(fit.components, key=lambda c: abs(c.amplitude))
                rows.append([comp.latency, comp.amplitude, comp.fwhm])
            else:
                rows.append([np.nan] * 3)
        return np.array(rows)

    def get_feature_names_out(self, input_features=None):
        return np.array(["latency", "amplitude", "fwhm"])

    def score(self, X=None, y=None):
        """Mean r2 over the channels that were fitted."""

        values = [f.gof.r2 for f in self.fit_.fits if f.gof is not None]
        return float(np.nanmean(values)) if values else np.nan


###################################################################################################
## Functions

def fit_response(times, y, *, window, baseline, preset=None, **overrides):
    """Fit one waveform with a preset and overrides; returns a :class:`ResponseFit`."""

    model = ResponseModel(preset, fit_window=window, baseline=baseline, **overrides)
    return model.fit_waveform(times, y)


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
    if "fit_window" in overrides:
        # the name of the preset field (spec §6.3) and of FitSettings, for ``window``
        given = overrides.pop("fit_window")
        window = given if window is None else window
    model = ResponseModel(preset, fit_window=window, baseline=baseline, n_jobs=n_jobs,
                          random_state=random_state, **overrides)
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
        return GroupFit.from_fits(fits, chans, conds, settings=_settings_dict(model.settings()))
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


def methods_text(fit):
    """Methods paragraph for a ResponseFit or GroupFit (spec REQ-123)."""

    settings = fit.settings
    shapes = ", ".join(settings.get("shapes", []))
    preset = settings.get("preset")
    window = settings.get("fit_window")
    crit = "BIC" if settings.get("selection") == "bic" else settings.get("selection", "").upper()
    return (
        f"Evoked responses were parameterized with evoparam {settings.get('package_version')}"
        + (f" ({settings.get('preset_note') or f'preset {preset!r}'})" if preset else "")
        + f". In the window {window[0] * 1000:g} to {window[1] * 1000:g} ms, each waveform was "
        f"modelled as a {settings.get('background')} background plus up to "
        f"{settings.get('max_components')} components ({shapes}; widths "
        f"{settings.get('min_fwhm') * 1000:g} to {settings.get('max_fwhm') * 1000:g} ms full "
        f"width at half maximum). Amplitudes and background coefficients were solved by "
        f"variable projection and the other parameters by bounded least squares. Components "
        f"were added one at a time and kept when the {crit} decreased by at least "
        f"{settings.get('delta'):g}, with the number of samples corrected for the lag-1 "
        f"autocorrelation of the baseline (effective sample size). Uncertainty: "
        f"{settings.get('uncertainty')} ({settings.get('ci_level', 0.95) * 100:g} % intervals)."
    )
