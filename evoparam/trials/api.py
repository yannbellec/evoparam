"""Trial-by-trial parameters, jitter, realignment and covariates (spec §5.7)."""

import dataclasses
import os
import time as _time
import warnings

import numpy as np

from evoparam.data import Evoked, Provenance, TrialFit, TrialSet
from evoparam.errors import DataError, SettingsError, warn as _warn
from evoparam.models.component_shapes import (FWHM_PER_SIGMA, Template, compute_features,
                                               get_background, get_shape)
from evoparam.utils._optional import require
from evoparam.version import __version__

from ._model import ChannelTemplate, fit_trial

__all__ = ["fit_trials", "TrialFitSet", "jitter", "realign", "dose_response",
           "prestimulus_dependence", "MODES", "state_dependence", "COVARIATES"]

MODES = ("shift_scale", "free", "template", "regularized")
COVARIATES = ("trend", "dose", "prestim", "state")
SCALP_PRESETS = ("erp", "p300_bci", "tep", "sep")
JITTER_REFUSE = 1.2
NULL_REPEATS = 5
RIDGE_GRID = (0.0, 0.01, 0.1, 1.0, 10.0)
MODALITY_PRESETS = {"spes": "ccep", "erp": "erp", "tep": "tep", "erna": "erna", "sep": "sep",
                    "ecap": "ecap", "lfp_ep": "rodent_ep"}


###################################################################################################
## Container

@dataclasses.dataclass(eq=False)
class TrialFitSet:
    """Trial fits of every channel (and condition): ``fits`` (list of :class:`TrialFit`),
    the tidy ``table`` of all of them, ``skipped_channels`` and ``settings``."""

    fits: list
    table: object
    skipped_channels: list
    settings: dict
    provenance: Provenance = dataclasses.field(default_factory=Provenance)

    def to_dataframe(self):
        return self.table.copy()

    def get(self, channel, condition=None):
        for fit in self.fits:
            if fit.channel == channel and (condition is None or
                                           fit.settings.get("condition") == condition):
                return fit
        raise KeyError(f"No trial fit for channel {channel!r}.")

    def save(self, path, overwrite=False):
        """Save to a folder: one ``.evoparam`` container per channel fit, and the table."""

        import json

        if os.path.exists(path) and not overwrite and os.listdir(path):
            raise DataError(f"{path} exists and is not empty (overwrite=True to replace).")
        os.makedirs(path, exist_ok=True)
        for i, fit in enumerate(self.fits):
            fit.save(os.path.join(path, f"fit_{i:04d}.evoparam"), overwrite=True)
        self.table.to_csv(os.path.join(path, "table.csv"), index=False)
        with open(os.path.join(path, "set.json"), "w", encoding="utf-8") as fobj:
            json.dump({"n_fits": len(self.fits), "skipped_channels": self.skipped_channels,
                       "package_version": __version__}, fobj)
        return path

    @classmethod
    def load(cls, path):
        import json

        pd = require("pandas", extra="data", caller="evoparam.trials.TrialFitSet.load")
        with open(os.path.join(path, "set.json"), encoding="utf-8") as fobj:
            meta = json.load(fobj)
        fits = [TrialFit.load(os.path.join(path, f"fit_{i:04d}.evoparam"))
                for i in range(meta["n_fits"])]
        table = pd.concat([f.table for f in fits], ignore_index=True) if fits else \
            pd.read_csv(os.path.join(path, "table.csv"))
        return cls(fits=fits, table=table, skipped_channels=meta["skipped_channels"],
                   settings={})


###################################################################################################
## Helpers

def _shape_for(comp, settings):
    name = comp.shape
    fields = {f.name for f in dataclasses.fields(type(get_shape(name)))}
    kwargs = {k: settings[k] for k in ("min_fwhm", "max_fwhm") if k in fields and k in settings}
    return get_shape(name, **kwargs)


def _amplitude_of(comp):
    ci = comp.ci or {}
    return 0.5 * (ci["A"][0] + ci["A"][1]) if "A" in ci else comp.amplitude


def _template_for(fit, picks, times, max_shift, factor):
    """Channel template restricted to the chosen components."""

    settings = fit.settings
    window = fit.window
    tw = times[(times >= window[0]) & (times <= window[1])]
    shapes, params, locs, widths = [], [], [], []
    for k in picks:
        comp = fit.components[k]
        shape = _shape_for(comp, settings)
        shapes.append(shape)
        params.append(np.array(comp.params, float))
        locs.append(shape.param_names.index(shape.latency_param))
        widths.append(comp.fwhm)
    if max_shift == "auto":
        shifts = np.array([factor * w for w in widths], float)
    else:
        shifts = np.full(len(picks), float(max_shift))
    # A trial's component must not reach its neighbour: at most half the distance
    lats = [fit.components[k].latency for k in picks]
    for i in range(len(picks)):
        gaps = [abs(lats[i] - lats[j]) for j in range(len(picks)) if j != i]
        if gaps and shifts[i] > 0.5 * min(gaps):
            _warn(f"max_shift of component {picks[i]} clipped to half the distance to "
                  "its neighbour.", UserWarning, stacklevel=3)
            shifts[i] = 0.5 * min(gaps)
    bg = fit.background or {"name": "none"}
    background = get_background(bg["name"])
    nl = np.array([bg[n] for n in getattr(background, "nonlinear_names", ())], float)
    return ChannelTemplate(shapes, params, locs, np.maximum(shifts, 1e-6), background, nl, tw)


def _template_mode(template, fit, average, times, picks):
    """Replace each component by the channel average over its support (Woody template)."""

    tw = template.tw
    win = (times >= tw[0]) & (times <= tw[-1])
    y = average[win]
    curves = [s.evaluate(tw, p) * _amplitude_of(fit.components[k])
              for s, p, k in zip(template.shapes, template.params, picks)]
    total = np.sum(curves, axis=0)
    shapes, params, locs = [], [], []
    for i, k in enumerate(picks):
        comp = fit.components[k]
        own = y - (total - curves[i])
        support = (tw >= comp.onset) & (tw <= comp.offset)
        if support.sum() < 4:
            support = np.abs(curves[i]) >= 0.1 * np.max(np.abs(curves[i]))
        values = np.where(support, own, 0.0)
        values[0] = values[-1] = 0.0
        shape = Template(template_times=tw, template_values=values,
                         max_shift=float(template.max_shift[i]))
        shapes.append(shape)
        params.append(np.array([0.0]))
        locs.append(0)
    return dataclasses.replace(template, shapes=shapes, params=params, loc_index=locs)


def _fit_all(template, data, valid, mode, ridge, support=None):
    return [fit_trial(template, data[i], mode=mode, ridge=ridge, valid=valid[i],
                      support=support) for i in range(len(data))]


def _choose_ridge(template, data, valid):
    """Penalty chosen by split-sample prediction error: fit on even samples, score on odd."""

    even = np.zeros(data.shape[1], bool)
    even[::2] = True
    best, best_err = 0.0, np.inf
    for lam in RIDGE_GRID:
        err = 0.0
        for i in range(len(data)):
            res = fit_trial(template, data[i], mode="regularized", ridge=lam,
                            valid=valid[i] & even, tol=1e-6)
            odd = valid[i] & ~even
            err += np.sum((data[i][odd] - res["fitted"][odd]) ** 2)
        if err < best_err:
            best, best_err = lam, err
    return best


def _null_jitter(template, fit, trials, times, baseline, picks, support=None, r2_min=0.0,
                 report=None):
    """Jitter found by the same procedure on noise only: the channel curve plus each trial's
    baseline segment (no response)."""

    tw = template.tw
    base = (times >= baseline[0]) & (times <= baseline[1])
    n_win = len(tw)
    model = sum(s.evaluate(tw, p) * _amplitude_of(fit.components[k])
                for s, p, k in zip(template.shapes, template.params, picks))
    shifts = []
    for x in trials:
        noise_all = np.nan_to_num(x[base])
        if len(noise_all) < 4:
            return np.full(len(picks), np.nan), 0
        # Several segments of each trial's baseline (circular offsets): the null SD is then
        # estimated from more values than the trial SD it is compared with
        for offset in np.linspace(0, len(noise_all), NULL_REPEATS, endpoint=False).astype(int):
            idx = (offset + np.arange(n_win)) % len(noise_all)
            noise = noise_all[idx] - noise_all.mean()
            # only the reported components move: the others stay at their channel values
            res = fit_trial(template, model + noise, mode="shift_scale", tol=1e-6,
                            support=support, movable=report)
            bound = res["at_bound_k"][report].any() if report is not None else res["at_bound"]
            # the same trust rule as the real trials, so that both SDs describe the same set
            if res["converged"] and not bound and \
                    np.nan_to_num(res["r2"], nan=-1) >= r2_min:
                shifts.append(res["shift"])
    if len(shifts) < 3:
        return np.full(len(picks), np.nan), 0
    # Repeats from one trial's baseline are not independent: count trials, not repeats
    return np.nanstd(np.array(shifts), axis=0, ddof=1), int(np.ceil(len(shifts) / NULL_REPEATS))


def _prestim(trials, times, band, window):
    """Pre-stimulus phase (three estimators), its circular SD, and power, per trial."""

    from scipy.signal import butter, hilbert, lfilter, sosfiltfilt

    inside = (times >= window[0]) & (times <= window[1])
    sfreq = 1.0 / np.median(np.diff(times))
    if inside.sum() / sfreq < 3.0 / band[0]:
        raise SettingsError("prestim: the pre-stimulus window is shorter than three cycles of "
                            "the band's lowest frequency.")
    seg = np.nan_to_num(trials[:, inside])
    b, a = butter(2, band, btype="bandpass", fs=sfreq)
    causal = lfilter(b, a, seg, axis=-1)
    phase1 = np.angle(hilbert(causal, axis=-1))[:, -1]
    mirrored = np.concatenate([seg, seg[:, ::-1]], axis=-1)
    sos = butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    phase2 = np.angle(hilbert(sosfiltfilt(sos, mirrored, axis=-1), axis=-1))[:, seg.shape[1] - 1]
    # Autoregressive forward prediction of the filtered signal, one cycle past the end
    phase3 = []
    for row in causal:
        p = 8
        lagged = np.column_stack([row[p - j - 1:len(row) - j - 1] for j in range(p)])
        coef, *_ = np.linalg.lstsq(lagged, row[p:], rcond=None)
        ext = list(row)
        for _ in range(int(sfreq / band[0])):
            ext.append(np.dot(coef, ext[-1:-p - 1:-1]))
        phase3.append(np.angle(hilbert(np.array(ext)))[len(row) - 1])
    phases = np.column_stack([phase1, phase2, np.array(phase3)])
    resultant = np.abs(np.mean(np.exp(1j * phases), axis=1))
    circ_sd = np.sqrt(-2 * np.log(np.clip(resultant, 1e-12, 1)))
    phase = np.angle(np.mean(np.exp(1j * phases), axis=1))
    power = np.mean(np.abs(hilbert(sosfiltfilt(sos, seg, axis=-1), axis=-1)) ** 2, axis=-1)
    return phase, circ_sd, power


###################################################################################################
## Main entry point

def fit_trials(ts, group_fit=None, *, mode="shift_scale", components=None, max_shift="auto",
               stretch=False, r2_min=0.2, realign=False, realign_on=None, covariates=(),
               prestim_band=(8.0, 13.0), prestim_window=(-0.5, -0.01), by=("stim_site",),
               preset=None, n_jobs=1, random_state=None, verbose=None):
    """Latency, amplitude (and in the free modes, width) of each component in every trial.

    The channel fit (``group_fit``, or ``evoparam.models.fit`` of the trial average with
    ``preset``, by default the one of the TrialSet's modality) gives the components; each
    trial is then fitted with them, shifted (``max_shift``, by default a quarter of the
    component's width for intracranial presets and half for scalp ones, never more than half
    the distance to a neighbour) and scaled:

    - ``"shift_scale"``: shift and amplitude only (Woody filtering with a parametric template);
    - ``"free"``: every parameter within +/-25 % of the channel value;
    - ``"template"``: the component's own segment of the channel average as template;
    - ``"regularized"``: ``"free"`` with a ridge penalty towards the channel values, the
      penalty chosen by split-sample prediction error (even samples to fit, odd to score;
      the specification's leave-one-trial-out would refit the channel model per trial).

    A trial is ``converged`` when the optimiser ends normally with finite numbers, and
    ``trusted`` when also its r² >= ``r2_min`` and its shift is not at a bound; untrusted
    trials keep their numbers. ``jitter_sd`` (SD of the shifts over trusted trials) is
    reported with ``jitter_sd_null``, the same procedure on the baseline noise added to the
    channel curve; realignment is refused (warning) when ``jitter_sd <= 1.2 jitter_sd_null``.

    Covariates: ``"trend"`` (amplitude against the position in the train), ``"dose"``
    (against ``stim_current_mA``, see :func:`dose_response`), ``"prestim"`` (pre-stimulus
    phase with its uncertainty across three estimators, and power, in ``prestim_band``),
    ``"state"`` (amplitude and latency per level of ``metadata["state"]``, see
    :func:`state_dependence`; a default covariate of the ``rodent_ep`` and ``primate_ep``
    presets, spec §6.2).

    Returns
    -------
    TrialFitSet
    """

    pd = require("pandas", extra="data", caller="evoparam.trials.fit_trials")
    start = _time.perf_counter()
    if not isinstance(ts, TrialSet):
        raise DataError("fit_trials expects a TrialSet.")
    if mode not in MODES:
        raise SettingsError(f"mode must be one of {MODES}, not {mode!r}.")
    unknown = set(covariates) - set(COVARIATES)
    if unknown:
        raise SettingsError(f"Unknown covariates {sorted(unknown)}.")
    if group_fit is None:
        from evoparam.models import fit as fit_models
        preset = preset or MODALITY_PRESETS.get(ts.modality)
        if preset is None:
            raise SettingsError("Give group_fit, or preset= (the TrialSet has no modality).")
        group_fit = fit_models(ts, preset, by=[c for c in by if ts.metadata is not None
                                               and c in ts.metadata.columns] or None)
    settings0 = group_fit.settings or {}
    preset0 = settings0.get("preset")
    preset0 = preset0.get("name") if isinstance(preset0, dict) else preset0
    factor = 0.5 if preset0 in SCALP_PRESETS else 0.25
    baseline = tuple(settings0.get("baseline") or (ts.times[0], ts.times[0] + 0.1))

    meta = ts.metadata
    columns = [c for c in (by or ()) if meta is not None and c in meta.columns]
    if columns:
        labels = meta[columns].astype(str).agg("/".join, axis=1).to_numpy()
    else:
        labels = np.array([""] * ts.n_trials)

    fits, skipped, rows = [], [], []
    for label in dict.fromkeys(labels):
        trial_idx = np.flatnonzero(labels == label)
        sub = ts.data[trial_idx]
        sub_masks = None if ts.masks is None else ts.masks[trial_idx]
        for ch, name in enumerate(ts.ch_names):
            try:
                fit = group_fit.get(name, condition=label if label else None)
            except KeyError:
                try:
                    fit = group_fit.get(name)
                except KeyError:
                    continue
            if fit.status != "ok" or not fit.components:
                skipped.append(name)
                continue
            picks = list(range(len(fit.components)))
            if components is not None:
                picks = [k for k, c in enumerate(fit.components)
                         if k in components or c.label in components]
                if not picks:
                    skipped.append(name)
                    continue
            every = list(range(len(fit.components)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore" if verbose is False else "default")
                # The trial model keeps every component of the channel (a complete model);
                # only the chosen ones are reported
                template = _template_for(fit, every, ts.times, max_shift, factor)
            report = [every.index(k) for k in picks]
            support = np.zeros(len(template.tw), bool)
            for k in picks:
                comp = fit.components[k]
                support |= (template.tw >= comp.onset - comp.fwhm) & \
                    (template.tw <= comp.offset + comp.fwhm)
            win = (ts.times >= template.tw[0]) & (ts.times <= template.tw[-1])
            x = np.nan_to_num(sub[:, ch][:, win])
            valid = np.ones(x.shape, bool) if sub_masks is None else ~sub_masks[:, ch][:, win]
            channel_template = template
            if mode == "template":
                average = np.nanmean(sub[:, ch], axis=0)
                template = _template_mode(template, fit, average, ts.times, every)
            if stretch and mode == "shift_scale":
                fit_mode = "free"
            else:
                fit_mode = "free" if mode == "free" else mode
            ridge = _choose_ridge(template, x, valid) if mode == "regularized" else 0.0
            results = _fit_all(template, x, valid,
                               "shift_scale" if mode == "template" else fit_mode, ridge, support)

            shift = np.array([r["shift"] for r in results])[:, report]
            scale = np.array([r["scale"] for r in results])[:, report]
            r2 = np.array([r["r2"] for r in results])
            converged = np.array([r["converged"] for r in results])
            # Trust looks at the reported components only: a neighbour held by a tight bound
            # does not make the trial's N1 untrustworthy
            at_bound = np.array([r["at_bound_k"][report].any() for r in results])
            trusted = converged & (np.nan_to_num(r2, nan=-1) >= r2_min) & ~at_bound
            peaks = np.array([fit.components[k].amplitude / _amplitude_of(fit.components[k])
                              if _amplitude_of(fit.components[k]) else 1.0 for k in picks])
            amplitude = scale * peaks[None]
            latency = shift + np.array([fit.components[k].latency for k in picks])[None]
            fwhm = np.tile([fit.components[k].fwhm for k in picks], (len(results), 1))
            if fit_mode in ("free", "regularized") and mode != "template":
                for i, r in enumerate(results):
                    for j, k in enumerate(report):
                        f = compute_features(channel_template.shapes[k], channel_template.tw,
                                             r["params"][k], 1.0, sfreq=ts.sfreq)
                        fwhm[i, j] = f.fwhm
            params = np.array([np.concatenate([r["params"][k] for k in report])
                               for r in results])

            jit = {}
            for j, k in enumerate(picks):
                good = trusted
                jit[k] = float(np.std(shift[good, j], ddof=1)) if good.sum() >= 3 else np.nan
            null, n_null = _null_jitter(channel_template, fit, sub[:, ch], ts.times, baseline,
                                        every, support, r2_min, report)
            null = null[report]
            jit_null = {k: float(null[j]) for j, k in enumerate(picks)}
            corrected = {}
            for j, k in enumerate(picks):
                comp = fit.components[k]
                sigma = comp.fwhm / FWHM_PER_SIGMA
                jj = jit[k] if np.isfinite(jit[k]) else 0.0
                corrected[k] = float(comp.amplitude * np.sqrt(sigma ** 2 + jj ** 2) / sigma)

            realigned = None
            if realign:
                target = picks[0] if realign_on is None else (
                    realign_on if isinstance(realign_on, int) else
                    next(k for k in picks if fit.components[k].label == realign_on))
                j = picks.index(target)
                # Refuse unless the jitter clearly exceeds what noise alone gives: above
                # 1.2 x the noise-only SD, and above the 95 % quantile of the ratio of two
                # SDs of noise (F distribution), so that jitter-free data are refused in
                # 95 % of runs whatever the number of trials
                from scipy.stats import f as f_dist
                n_t = int(trusted.sum())
                limit = JITTER_REFUSE
                if n_t > 2 and n_null > 2:
                    limit = max(limit, float(np.sqrt(f_dist.ppf(0.95, n_t - 1, n_null - 1))))
                if not jit[target] > limit * jit_null[target]:
                    _warn(f"{name}: jitter {jit[target]:.4f} s is not above "
                          f"{limit:.2f} x the noise-only jitter "
                          f"{jit_null[target]:.4f} s; realignment refused.",
                          UserWarning, stacklevel=2)
                else:
                    realigned = _realigned(ts.times, sub[:, ch], shift[:, j], trusted,
                                           name, ts)

            table = []
            for i, trial in enumerate(trial_idx):
                for j, k in enumerate(picks):
                    row = {"trial": int(trial), "channel": name, "condition": label,
                           "component": int(k), "label": fit.components[k].label or "",
                           "shift": shift[i, j], "scale": scale[i, j],
                           "latency": latency[i, j], "amplitude": amplitude[i, j],
                           "fwhm": fwhm[i, j], "r2": r2[i], "converged": bool(converged[i]),
                           "trusted": bool(trusted[i])}
                    if meta is not None:
                        for col in meta.columns:
                            row.setdefault(col, meta[col].iloc[trial])
                    table.append(row)
            frame = pd.DataFrame(table)
            if "prestim" in covariates:
                phase, circ_sd, power = _prestim(sub[:, ch], ts.times, prestim_band,
                                                 prestim_window)
                frame["prestim_phase"] = np.repeat(phase, len(picks))
                frame["prestim_phase_sd"] = np.repeat(circ_sd, len(picks))
                frame["prestim_power"] = np.repeat(power, len(picks))
            rows.append(frame)
            fits.append(TrialFit(channel=name, template=fit, shift=shift, scale=scale,
                                 params=params, latency=latency, amplitude=amplitude, fwhm=fwhm,
                                 gof=r2, converged=converged, jitter_sd=jit,
                                 jitter_sd_null=jit_null, amplitude_jitter_corrected=corrected,
                                 trusted=trusted, table=frame, realigned=realigned,
                                 settings={"mode": mode, "condition": label, "picks": picks,
                                           "ridge": ridge, "max_shift":
                                           channel_template.max_shift.tolist(),
                                           "r2_min": r2_min, "package_version": __version__}))
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    covariate_tables = {}
    if "trend" in covariates and len(table):
        covariate_tables["trend"] = _trend(table)
    if "dose" in covariates and len(table):
        covariate_tables["dose"] = dose_response(table)
    if "state" in covariates and len(table):
        covariate_tables["state"] = state_dependence(table)
    settings = {"mode": mode, "max_shift": max_shift, "r2_min": r2_min, "by": list(by or ()),
                "covariates": list(covariates), "package_version": __version__,
                "covariate_tables": {k: v.to_dict("list") for k, v in covariate_tables.items()}}
    provenance = ts.provenance.with_step("evoparam.trials.fit_trials",
                                         {k: v for k, v in settings.items()
                                          if k != "covariate_tables"},
                                         duration_s=_time.perf_counter() - start)
    return TrialFitSet(fits=fits, table=table, skipped_channels=sorted(set(skipped)),
                       settings=settings, provenance=provenance)


def _realigned(times, trials, shifts, trusted, name, ts):
    data = []
    for x, d in zip(trials[trusted], shifts[trusted]):
        data.append(np.interp(times, times - d, np.nan_to_num(x)))
    data = np.array(data)
    return Evoked(data=data.mean(axis=0)[None], times=times, sfreq=ts.sfreq, ch_names=[name],
                  units=ts.units, n_trials=int(trusted.sum()),
                  provenance=ts.provenance.with_step("evoparam.trials.realign",
                                                     {"channel": name}))


###################################################################################################
## Companions

def jitter(trial_fit, component=0):
    """Per channel: ``jitter_sd`` (with a bootstrap interval over trusted trials),
    ``jitter_sd_null``, ``jitter_sd_excess = sqrt(jitter_sd² - jitter_sd_null²)`` (the jitter
    left once the part that noise alone produces is removed; the best estimate of the true
    jitter at low SNR) and ``amplitude_jitter_corrected`` of one component (index or label)."""

    pd = require("pandas", extra="data", caller="evoparam.trials.jitter")
    rng = np.random.default_rng(0)
    rows = []
    for fit in trial_fit.fits:
        picks = fit.settings["picks"]
        if isinstance(component, str):
            matches = [k for k in picks if fit.template.components[k].label == component]
            if not matches:
                continue
            k = matches[0]
        else:
            if component not in picks:
                continue
            k = component
        j = picks.index(k)
        values = fit.shift[fit.trusted, j]
        boots = [np.std(rng.choice(values, len(values)), ddof=1) for _ in range(500)] \
            if len(values) >= 3 else [np.nan]
        raw, null = fit.jitter_sd[k], fit.jitter_sd_null[k]
        excess = float(np.sqrt(max(raw ** 2 - null ** 2, 0.0))) if np.isfinite(raw) and \
            np.isfinite(null) else np.nan
        rows.append({"channel": fit.channel, "condition": fit.settings["condition"],
                     "component": k, "jitter_sd": raw, "jitter_sd_excess": excess,
                     "ci_low": float(np.nanpercentile(boots, 2.5)),
                     "ci_high": float(np.nanpercentile(boots, 97.5)),
                     "jitter_sd_null": fit.jitter_sd_null[k],
                     "amplitude_jitter_corrected": fit.amplitude_jitter_corrected[k],
                     "n_trusted": int(fit.trusted.sum())})
    return pd.DataFrame(rows)


def realign(ts, trial_fit, component=0):
    """Average of the trusted trials, each shifted back by its latency shift of
    ``component`` (the Woody-filter average); one Evoked per channel fit."""

    out = []
    for fit in trial_fit.fits:
        picks = fit.settings["picks"]
        k = component if isinstance(component, int) else next(
            (p for p in picks if fit.template.components[p].label == component), None)
        if k not in picks:
            continue
        ch = list(ts.ch_names).index(fit.channel)
        out.append(_realigned(ts.times, ts.data[:, ch], fit.shift[:, picks.index(k)],
                              fit.trusted, fit.channel, ts))
    return out


def _trend(table):
    pd = require("pandas", extra="data", caller="evoparam.trials")
    order_col = "index_in_train" if "index_in_train" in table.columns else "trial"
    rows = []
    for (ch, cond, k), g in table[table["trusted"]].groupby(["channel", "condition",
                                                            "component"]):
        if len(g) < 3:
            continue
        x = g[order_col].to_numpy(float)
        y = g["scale"].to_numpy(float)
        slope, intercept = np.polyfit(x - x.min(), y, 1)
        rows.append({"channel": ch, "condition": cond, "component": k,
                     "slope": slope / intercept if intercept else np.nan,
                     "intercept": intercept, "x": order_col})
    return pd.DataFrame(rows)


def state_dependence(trial_fit, *, column="state", reference=None):
    """Amplitude and latency per behavioural or brain state (``metadata["state"]``: sleep,
    wake, anaesthesia...), per channel and component, on the trusted trials.

    State changes the gain of evoked field potentials up to about three-fold (spec §6.2,
    rodent and primate presets), so the trial parameters are summarised per state level:
    ``n_trials``, the median and mean amplitude and latency, the ``gain`` (median absolute
    amplitude over that of the ``reference`` state, by default the most frequent one) and,
    on the reference row's channel and component, the Kruskal-Wallis p-values of amplitude
    and latency across the states (``p_amplitude``, ``p_latency``; NaN with one state).

    Raises
    ------
    SettingsError
        The trial table has no ``column``.
    """

    from scipy.stats import kruskal

    pd = require("pandas", extra="data", caller="evoparam.trials.state_dependence")
    table = trial_fit.table if isinstance(trial_fit, TrialFitSet) else trial_fit
    if column not in table.columns:
        raise SettingsError(f"state_dependence: the trial table has no column {column!r} "
                            f"(metadata['{column}']).")
    columns = ["channel", "component", "label", column, "n_trials", "amplitude_median",
               "amplitude_mean", "latency_median", "latency_mean", "gain", "reference",
               "p_amplitude", "p_latency"]
    rows = []
    trusted = table[table["trusted"].astype(bool) & table[column].notna()]
    for (ch, k), g in trusted.groupby(["channel", "component"], sort=False):
        states = g[column].astype(str)
        counts = states.value_counts()
        ref = str(reference) if reference is not None else str(counts.index[0])
        groups = {state: g[states == state] for state in sorted(counts.index)}
        ref_amp = (np.median(np.abs(groups[ref]["amplitude"].to_numpy(float)))
                   if ref in groups else np.nan)
        p_amp = p_lat = np.nan
        usable = [v for v in groups.values() if len(v) >= 2]
        if len(usable) >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    p_amp = float(kruskal(*[v["amplitude"].to_numpy(float)
                                            for v in usable]).pvalue)
                    p_lat = float(kruskal(*[v["latency"].to_numpy(float)
                                            for v in usable]).pvalue)
                except ValueError:          # all values identical
                    pass
        for state, v in groups.items():
            amp, lat = v["amplitude"].to_numpy(float), v["latency"].to_numpy(float)
            rows.append({"channel": ch, "component": int(k),
                         "label": str(v["label"].iloc[0]) if "label" in v else "",
                         column: state, "n_trials": int(len(v)),
                         "amplitude_median": float(np.median(amp)),
                         "amplitude_mean": float(np.mean(amp)),
                         "latency_median": float(np.median(lat)),
                         "latency_mean": float(np.mean(lat)),
                         "gain": float(np.median(np.abs(amp)) / ref_amp) if ref_amp else np.nan,
                         "reference": ref, "p_amplitude": p_amp, "p_latency": p_lat})
    return pd.DataFrame(rows, columns=columns)


def _hill(i, a_max, i50, h):
    return a_max * i ** h / (i50 ** h + i ** h)


def dose_response(trial_fit, *, x="stim_current_mA", model="auto"):
    """Amplitude against stimulation dose per channel and component.

    ``model="hill"``: ``a(I) = a_max I^h / (I50^h + I^h)``, ``h`` in [0.5, 6], with the
    threshold current ``I_thr = I50 (0.1/0.9)^(1/h)``; ``"linear"``: a straight line;
    ``"auto"``: Hill with four currents or more, else linear.
    """

    from scipy.optimize import curve_fit

    pd = require("pandas", extra="data", caller="evoparam.trials.dose_response")
    table = trial_fit.table if isinstance(trial_fit, TrialFitSet) else trial_fit
    if x not in table.columns:
        raise SettingsError(f"dose_response: the trial table has no column {x!r}.")
    rows = []
    for (ch, k), g in table[table["trusted"]].groupby(["channel", "component"]):
        dose = g[x].to_numpy(float)
        amp = np.abs(g["amplitude"].to_numpy(float))
        levels = np.unique(dose[np.isfinite(dose)])
        if len(levels) < 2:
            raise SettingsError("dose_response needs at least two doses.")
        kind = model if model != "auto" else ("hill" if len(levels) >= 4 else "linear")
        row = {"channel": ch, "component": k, "model": kind, "n_doses": len(levels)}
        if kind == "hill":
            try:
                popt, _ = curve_fit(_hill, dose, amp, p0=[amp.max(), np.median(levels), 2.0],
                                    bounds=([0, levels.min() / 10, 0.5],
                                            [10 * amp.max() + 1e-12, levels.max() * 10, 6.0]),
                                    maxfev=20000)
                a_max, i50, h = popt
                row.update(a_max=a_max, i50=i50, h=h, i_thr=i50 * (0.1 / 0.9) ** (1 / h))
            except RuntimeError:
                row.update(a_max=np.nan, i50=np.nan, h=np.nan, i_thr=np.nan)
        else:
            slope, intercept = np.polyfit(dose, amp, 1)
            row.update(slope=slope, intercept=intercept)
        rows.append(row)
    return pd.DataFrame(rows)


def prestimulus_dependence(trial_fit, *, band=(8., 13.), n_permutations=1000,
                           random_state=None):
    """Amplitude against pre-stimulus phase: ``a = c0 + c1 cos(phi - phi0)``, with a
    permutation test of ``c1`` (phases shuffled across trials). Needs ``covariates=("prestim",)``
    in :func:`fit_trials`."""

    pd = require("pandas", extra="data", caller="evoparam.trials.prestimulus_dependence")
    table = trial_fit.table if isinstance(trial_fit, TrialFitSet) else trial_fit
    if "prestim_phase" not in table.columns:
        raise SettingsError("Run fit_trials with covariates=('prestim',) first.")
    rng = np.random.default_rng(random_state)
    rows = []
    for (ch, k), g in table[table["trusted"]].groupby(["channel", "component"]):
        phi = g["prestim_phase"].to_numpy(float)
        amp = g["scale"].to_numpy(float)

        def harmonic(p):
            design = np.column_stack([np.ones_like(p), np.cos(p), np.sin(p)])
            coef, *_ = np.linalg.lstsq(design, amp, rcond=None)
            return coef

        c0, cc, cs = harmonic(phi)
        c1 = np.hypot(cc, cs)
        null = np.array([np.hypot(*harmonic(rng.permutation(phi))[1:])
                         for _ in range(n_permutations)])
        rows.append({"channel": ch, "component": k, "c0": c0, "c1": c1,
                     "modulation": c1 / abs(c0) if c0 else np.nan,
                     "phi0": float(np.arctan2(cs, cc)),
                     "p_value": (1 + np.sum(null >= c1)) / (1 + n_permutations),
                     "n_trials": len(g)})
    return pd.DataFrame(rows)
