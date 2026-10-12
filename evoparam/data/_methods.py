# SPDX-License-Identifier: Apache-2.0
"""Methods paragraph generated from a result's provenance and settings (spec §5.13, REQ-338).

The paragraph is built from frozen templates, one per processing step recorded in
``provenance.steps``, in the order the steps were run. It reads only ``provenance``,
``settings`` (and ``gof``, which no template uses), never the data, so a result loaded
from disk gives the same paragraph as the live object, and two runs with the same settings
give the same words whatever the data.

Steps that only bring data in (readers, epoching, simulators) or reshape them without
changing a number (selection, cropping, conversions) are not described: the data
description belongs to the author. Steps without a template are described by their
function name and their scalar settings, so that no step is silently left out.

The public entry point is :func:`evoparam.reports.methods_text`; the result classes call
:func:`methods_text` of this module (``ResponseFit.methods_text()``,
``GroupFit.methods_text()``), because the data containers may not import the reports
module (spec §4.2).
"""

import math

from evoparam.data._citation import citation_text, read_citation

__all__ = ["methods_text", "step_sentence", "SILENT_PREFIXES"]

# Steps that bring data in or reshape them without changing any number.
SILENT_PREFIXES = (
    "evoparam.io.",
    "evoparam.sim.",
    "evoparam.data.TrialSet.select",
    "evoparam.data.TrialSet.crop",
    "evoparam.data.TrialSet.as_evoked",
    "evoparam.data.Evoked.as_trialset",
    "evoparam.data.Evoked.select",
    "evoparam.data.migrate",
    "evoparam.stats.to_long",
    "evoparam.data.TrialSet.from_mne",
    "evoparam.data.Evoked.from_mne",
)

_FIT_STEPS = ("evoparam.models.ResponseModel.fit_group", "evoparam.models.fit_waveform")

_ARTIFACT_NAMES = {
    "interpolate": "interpolation across the artifact window",
    "template": "template subtraction",
    "tail": "template subtraction with a fitted recovery tail",
    "period_template": "subtraction of a template locked to the stimulation period",
    "lowrank": "low-rank (singular value) artifact modelling",
    "subspace": "subspace projection",
    "polynomial": "local polynomial detrending",
    "matching_pursuit": "matching pursuit over an artifact dictionary",
    "paradigm": "paradigm subtraction (alternating polarity or forward masking)",
    "learned": "a learned artifact model",
    "none": "no artifact removal",
}

_REFERENCE_NAMES = {
    "monopolar": "the recording reference was kept",
    "bipolar": "channels were re-referenced to bipolar derivations of adjacent contacts",
    "car": "channels were re-referenced to the common average",
    "car_adaptive": "channels were re-referenced to an adjusted common average from which "
    "channels with a response were excluded",
    "laplacian": "channels were re-referenced to a local (Laplacian) average of their neighbours",
    "average": "channels were re-referenced to the common average",
}

_CORRECTIONS = {
    "fdr_bh": "Benjamini-Hochberg false-discovery-rate correction",
    "fdr_by": "Benjamini-Yekutieli false-discovery-rate correction",
    "bonferroni": "Bonferroni correction",
    "holm": "Holm correction",
    "max_stat": "maximum-statistic correction",
    "none": "no correction",
}

_SELECTIONS = {
    "bic": "BIC",
    "aicc": "AICc",
    "cv": "cross-validated prediction error",
    "fixed": "a fixed number of components",
}

_UNCERTAINTIES = {
    "jacobian": "Jacobian-based (Wald) intervals",
    "profile": "profile-likelihood intervals",
    "bootstrap": "trial-bootstrap intervals",
    "posterior": "posterior credible intervals",
}

# Methods with a published reference, cited in the closing sentence.
_REFERENCES = {
    "fdr_bh": "Benjamini and Hochberg (1995) for the false-discovery-rate correction",
    "fdr_by": "Benjamini and Yekutieli (2001) for the false-discovery-rate correction",
}


###################################################################################################
## Formatting helpers


def _num(value, digits=6):
    """Return a setting printed without float noise: 0.0090000001 -> 0.009."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return str(value)
    return f"{round(value, digits):g}"


def _ms(value):
    return _num(float(value) * 1000.0, 4)


def _window(value):
    if value is None:
        return "none"
    return f"{_ms(value[0])} to {_ms(value[1])} ms"


def _setting(value):
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_setting(v) for v in value) + ")"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_setting(v)}" for k, v in value.items()) + "}"
    return (
        _num(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else ("yes" if value is True else "no" if value is False else str(value))
    )


def _polarity(value):
    return "both polarities" if value == "both" else f"{value} polarity only"


def _join(items):
    items = [str(item) for item in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


###################################################################################################
## One sentence per step


def _artifact(settings):
    method = settings.get("method", "?")
    name = _ARTIFACT_NAMES.get(method, f"the '{method}' method")
    if method == "none":
        return "No stimulation-artifact removal was applied."
    text = (
        f"The stimulation artifact was removed by {name} (evoparam.artifact.remove_artifact, "
        f"method '{method}') in the window {_window(settings.get('window'))}"
    )
    details = []
    if method == "interpolate":
        details.append(f"{settings.get('interpolation', 'linear')} interpolation")
    if method in ("template", "tail"):
        details.append(f"{settings.get('template', 'global')} template")
        group_by = settings.get("group_by") or []
        if group_by:
            details.append(
                f"templates grouped by the metadata columns {_join(group_by)} (those present)"
            )
        if settings.get("scale"):
            details.append("scaled to each trial and channel")
        if settings.get("tail") == "joint":
            details.append(
                "recovery tail left to the response model and fitted jointly with its background"
            )
        elif settings.get("tail") not in (None, "none"):
            details.append(
                f"recovery tail subtracted (double exponential over "
                f"{_ms(settings.get('tail_length', 0))} ms)"
            )
    if method == "period_template" and settings.get("period") is not None:
        details.append(f"period {_ms(settings['period'])} ms")
    if method == "lowrank" and settings.get("n_components") is not None:
        details.append(f"{settings['n_components']} components")
    if method == "subspace":
        rule = settings.get("n_components")
        # results saved before the option existed projected every sample
        extent = settings.get("subspace_extent", "all")
        if rule == "energy":
            details.append("directions holding 95 % of the whitened window energy, at most 6")
        elif rule in (None, "noise"):
            details.append("directions whose whitened window energy exceeds twice the noise level")
        else:
            details.append(f"{rule} directions")
        details.append(
            "window samples projected" if extent == "window" else "every sample projected"
        )
    if method == "paradigm" and settings.get("scheme"):
        details.append(f"scheme '{settings['scheme']}'")
    if details:
        text += " (" + "; ".join(details) + ")"
    text += "."
    if settings.get("quality", True):
        text += " A quality score between 0 and 1 was computed for every trial."
    return text


def _paradigm(settings):
    scheme = settings.get("scheme", "?")
    return (
        f"Stimulation artifacts were cancelled by paradigm subtraction (scheme "
        f"'{scheme}', evoparam.artifact.paradigm_subtract)."
    )


def _baseline(settings):
    mode = settings.get("mode", "mean")
    what = {"mean": "the mean", "median": "the median", "linear": "a straight line fitted to"}.get(
        mode, f"the {mode} of"
    )
    return (
        f"Each trial and channel was baseline-corrected by subtracting {what} of "
        f"{_window(settings.get('baseline'))}"
        + (
            " of its own trial"
            if settings.get("baseline_source", "own") == "own"
            else f" ({settings.get('baseline_source')} baseline)"
        )
        + "."
    )


def _reference(settings):
    scheme = settings.get("scheme", "?")
    used = settings.get("used") or scheme
    if scheme == "car_adaptive" and used == "car":
        # Too few channels without a response: rereference fell back to the plain average.
        return (
            "Channels were re-referenced to the common average (an adjusted common "
            "average excluding channels with a response was requested, but too few "
            "channels were available for it)."
        )
    text = _REFERENCE_NAMES.get(scheme, f"channels were re-referenced (scheme '{scheme}')")
    text = text[0].upper() + text[1:]
    if scheme == "car_adaptive" and settings.get("response_window") is not None:
        text += f" (responses judged in {_window(settings['response_window'])})"
    excluded = settings.get("exclude") or []
    if excluded:
        text += f"; channels excluded from the reference: {_join(excluded)}"
    return text + "."


def _reject(settings):
    keys = sorted(k for k in settings if k not in ("capped_channels", "n_rejected", "rejected"))
    rules = ", ".join(
        f"{k} = {_setting(settings[k])}" for k in keys if not isinstance(settings[k], dict)
    )
    return f"Trials were flagged for rejection (evoparam.prep.reject_trials; {rules})."


def _resample(settings):
    rate = settings.get("sfreq") or settings.get("new_sfreq") or settings.get("target")
    return (
        f"Data were resampled to {_num(rate)} Hz."
        if rate is not None
        else "Data were resampled (evoparam.prep.resample)."
    )


def _average(settings):
    method = settings.get("method")
    if method is None:
        method = "median" if settings.get("robust") else "mean"
    return f"Trials were averaged per channel ({method})."


def _fit(settings):
    preset_window = settings.get("fit_window")
    shapes = _join(str(s).replace("_", " ") for s in settings.get("shapes", []))
    background = str(settings.get("background", "?")).replace("_", " ")
    crit = _SELECTIONS.get(settings.get("selection"), str(settings.get("selection")))
    article = "an" if background[:1] in "aeiou" else "a"
    text = (
        f"In the window {_window(preset_window)}, each waveform was modelled as {article} "
        f"{background} background plus up to {settings.get('max_components')} components "
        f"({shapes}; full width at half maximum {_ms(settings.get('min_fwhm'))} to "
        f"{_ms(settings.get('max_fwhm'))} ms, "
        f"{_polarity(settings.get('polarity', 'both'))}). "
        f"Amplitudes and background coefficients were solved by variable projection and "
        f"the other parameters by bounded least squares ({settings.get('loss', 'linear')} "
        f"loss, {settings.get('n_starts')} starts)."
    )
    if settings.get("selection") == "fixed":
        text += f" Each waveform was fitted with {settings.get('n_components')} components."
    elif settings.get("selection") == "cv":
        text += (
            f" The number of components was chosen by {settings.get('n_folds')}-fold "
            f"cross-validation over trials (smallest model within one standard error of "
            f"the best)."
        )
    else:
        noise = settings.get("noise_model", "ar")
        reference = settings.get("noise_reference")
        source = (
            "the plus-minus average of the trials (alternating signs, whole epoch)"
            if reference == "plusminus"
            else "the plus-minus average of the trials where the baseline held fewer than "
            "30 independent samples, of the baseline elsewhere"
            if reference == "mixed"
            else "the baseline"
        )
        how = {
            "ar": f"residuals whitened by an autoregressive model of the noise of {source}",
            "neff": f"residuals scaled by the noise of {source} with the effective sample "
            "count of an AR(1) noise, n_eff = n (1 - rho)/(1 + rho) (n_eff "
            "correction)",
        }.get(noise, f"residuals scaled by the noise of {source} (white noise)")
        text += (
            f" Components were added one at a time and kept when the {crit} decreased "
            f"by at least {_num(settings.get('delta'))}, computed on {how}."
        )
        if settings.get("reject_cancelling"):
            text += (
                " A model holding two overlapping components of opposite sign (a "
                "cancelling pair) was not accepted."
            )
        reseed = settings.get("reseed", False)
        if reseed:
            text += (
                " "
                + ("Every" if reseed == "always" else "A")
                + " final model "
                + (
                    ""
                    if reseed == "always"
                    else "holding two overlapping components of opposite sign "
                )
                + "was refitted from starts on the extrema of the data and replaced "
                "when the refit was better; a component of the refitted model that "
                "no longer lowered the criterion by the same amount was removed."
            )
    level = settings.get("ci_level", 0.95)
    kind = settings.get("uncertainty")
    text += (
        f" Uncertainty: {_UNCERTAINTIES.get(kind, kind)} at {_num(level * 100)} %"
        + (
            f" ({settings.get('n_bootstrap')} resamples)"
            if settings.get("uncertainty") == "bootstrap"
            else ""
        )
        + "."
    )
    text += (
        f" Onset and offset are the times where a component reaches "
        f"{_num(settings.get('onset_fraction', 0.1) * 100)} % of its peak amplitude."
    )
    windows = settings.get("windows") or {}
    if windows:
        labels = []
        for label, value in windows.items():
            sign = {1: "positive", -1: "negative"}.get(
                int(value[2]) if len(value) > 2 else 0, "either polarity"
            )
            parts = str(label).split("/")
            if (
                len(parts) == 2
                and sign == "either polarity"
                and sorted(p[:1] for p in parts) == ["N", "P"]
            ):
                # "P1/N1 by depth" (rodent and primate presets): named by its sign
                positive, negative = sorted(parts, key=lambda p: p[:1] != "P")
                sign = (
                    f"named {positive} when positive and {negative} when negative, the "
                    f"sign of the field potential reversing with recording depth"
                )
            labels.append(f"{label} ({_window(value[:2])}, {sign})")
        text += f" Components were labelled by latency window: {_join(labels)}."
    if settings.get("classical"):
        text += (
            " Classical measures (peak, mean amplitude, area and fractional-area "
            "latencies) were computed on the same window."
        )
    return text


def _existence(settings):
    method = settings.get("method")
    window = _window(settings.get("window"))
    baseline = _window(settings.get("baseline"))
    if method == "threshold":
        spec = settings.get("rule_spec") or {}
        rule = settings.get("rule") or spec.get("name", "?")
        return (
            f"Responses were detected with the published rule '{rule}' "
            f"({spec.get('source', 'custom rule')}): {spec.get('statistic', '?')} above "
            f"{_num(spec.get('k'))} in {window}, baseline {baseline}."
        )
    what = {
        "baseline_permutation": f"a sign-flip permutation test of the trials "
        f"({settings.get('statistic', 'max_t')} statistic, "
        f"{settings.get('n_permutations')} permutations)",
        "projection": "the cross-projection test of the trials",
        "model_comparison": f"a model comparison (change of BIC) with a parametric "
        f"bootstrap ({settings.get('n_permutations')} samples)",
    }.get(method, f"the '{method}' test")
    correction = _CORRECTIONS.get(settings.get("correction"), settings.get("correction"))
    by = settings.get("by") or []
    return (
        f"The presence of a response in {window} was tested on every channel with {what} "
        f"against the baseline {baseline} ({_polarity(settings.get('polarity', 'both'))}, "
        f"alpha = {_num(settings.get('alpha'))}, {correction} across channels"
        + (f", separately for each {_join(by)}" if by else "")
        + ")."
    )


def _trials(settings):
    mode = settings.get("mode", "?")
    max_shift = settings.get("max_shift")
    shift = (
        "automatic maximum shift"
        if max_shift in (None, "auto")
        else f"maximum shift {_ms(max_shift)} ms"
    )
    text = (
        f"Single-trial parameters were estimated by refitting the channel's components to "
        f"every trial with the '{mode}' trial model ({shift}); trials whose fit explained "
        f"less than {_num(float(settings.get('r2_min', 0)) * 100)} % of the variance were "
        f"marked as untrusted."
    )
    covariates = settings.get("covariates") or []
    if covariates:
        text += f" Trial covariates: {_join(covariates)}."
    return text


def _realign(_settings):
    return "Trials were realigned on the estimated single-trial latencies (Woody-style)."


def _hierarchical(settings):
    engine = settings.get("engine", "eb")
    name = {
        "eb": "empirical Bayes (restricted maximum likelihood variance components)",
        "numpyro": "full Bayesian inference (NUTS, numpyro)",
        "pymc": "full Bayesian inference (NUTS, PyMC)",
    }.get(engine, engine)
    levels = settings.get("levels")
    return (
        "Parameters were pooled across levels"
        + (f" ({' > '.join(map(str, levels))})" if levels else "")
        + f" with a hierarchical model fitted by {name}."
    )


def _variance(_settings):
    return (
        "Variance components were estimated by restricted maximum likelihood with the "
        "known sampling variances of the estimates."
    )


def _tf(settings):
    tp = settings.get("transform_params") or {}
    transform = settings.get("transform", "?")
    if transform == "morse":
        what = (
            f"generalised Morse wavelets (gamma = {_num(tp.get('gamma'))}, "
            f"beta = {_num(tp.get('beta'))})"
        )
    elif transform == "morlet":
        cycles = tp.get("n_cycles")
        values = cycles if isinstance(cycles, list) else [cycles]
        what = "complex Morlet wavelets" + (
            f" ({_join(sorted({_num(c, 3) for c in values}))} cycles)"
            if cycles is not None
            else ""
        )
    elif transform == "multitaper":
        what = (
            f"DPSS multitapers (time-bandwidth product {_num(tp.get('time_bandwidth'))}, "
            f"{tp.get('n_tapers')} tapers)"
        )
    else:
        what = f"the '{transform}' transform"
    freqs = settings.get("freqs") or []
    span = (
        f" at {len(freqs)} frequencies from {_num(freqs[0], 3)} to {_num(freqs[-1], 3)} Hz"
        if freqs
        else ""
    )
    return (
        f"Time-frequency power was computed with {what}{span}, normalised "
        f"({settings.get('baseline_mode')}) to the baseline "
        f"{_window(settings.get('baseline'))}, and tested in "
        f"{_window(settings.get('window'))} ({settings.get('stats')}, alpha = "
        f"{_num(settings.get('alpha'))})."
    )


def _shared_shapes(settings):
    return (
        f"Response shapes shared across "
        f"{'stimulation sites' if settings.get('mode') == 'convergent' else 'channels'} "
        f"were extracted with the '{settings.get('method', '?')}' method (at most "
        f"{settings.get('max_shapes', '?')} shapes)."
    )


def _fit_shared(settings):
    shared = settings.get("shared") or []
    return (
        f"Component parameters were fitted jointly across channels with "
        f"{_join(shared) if shared else 'no'} parameters shared."
    )


def _events(settings):
    method = settings.get("method", "threshold+template")
    text = (
        f"Stimulation pulses were detected from their artifact in the signal "
        f"(evoparam.events.detect_pulses, method '{method}'): onsets where the robust "
        f"z-score of the median absolute first difference across channels exceeded "
        f"{_num(settings.get('threshold_sd'))}"
    )
    if "template" in method:
        text += (
            f", kept when strong or when repeated with the same artifact shape "
            f"(normalised cross-correlation of at least {_num(settings.get('rho_min'))})"
        )
    interval = settings.get("min_interval")
    if interval is not None:
        text += f", at least {_ms(interval)} ms apart"
    if settings.get("expected_rate"):
        text += f", at an expected rate of {_num(settings['expected_rate'])} Hz"
    text += "."
    extra = []
    if settings.get("saturation", True):
        extra.append("amplifier saturation was detected per channel and pulse")
    if settings.get("site"):
        vote = " and voted across the pulses of each train" if settings.get("vote") else ""
        extra.append(
            f"the stimulated contact pair was identified from the artifact "
            f"('{settings.get('site_method', 'evidence')}' rule{vote})"
        )
    extra.append("pulses were grouped into trains")
    text += " " + _join(extra)[0].upper() + _join(extra)[1:] + "."
    return text


def _filter(kind):
    def sentence(settings):
        if kind == "notch":
            freqs = settings.get("frequencies") or [settings.get("freq")]
            return (
                f"Line noise was removed with zero-phase notch filters at "
                f"{_join(_num(f) for f in freqs)} Hz (quality factor "
                f"{_num(settings.get('q'))})."
            )
        order = settings.get("order", 4)
        return (
            f"A zero-phase Butterworth {kind} filter (order {order}, applied forward and "
            f"backward) with a cutoff of {_num(settings.get('cutoff'))} Hz was applied."
        )

    return sentence


def _apply_mask(_settings):
    return (
        "Samples given by a user-supplied mask were excluded from every later step "
        "(evoparam.artifact.apply_mask)."
    )


def _train_baseline(settings):
    source = {
        "continuous": " (from the continuous recording)",
        "epochs": " (from the epoch of the first pulse of the train)",
    }.get(settings.get("source"), "")
    return (
        f"Each stimulation train was baseline-corrected: per channel, the "
        f"{settings.get('mode', 'mean')} of the last {_ms(settings.get('length'))} ms "
        f"before the train{source} was subtracted from every pulse of that train "
        f"(evoparam.prep.train_baseline)."
    )


def _reference_sensitivity(settings):
    schemes = list(settings.get("schemes") or [])
    names = {
        "bipolar": "bipolar",
        "car": "common-average",
        "car_adaptive": "adjusted common-average",
        "laplacian": "Laplacian",
        "monopolar": "recording",
    }
    detector = settings.get("detector")
    return (
        f"The dependence of the detections on the reference was assessed by repeating "
        f"the detection under the {_join(names.get(s, s) for s in schemes)} references "
        f"and reporting the pairwise Cohen's kappa of the detections and the latency "
        f"differences (evoparam.prep.reference_sensitivity"
        + (f", detector {detector}" if detector else "")
        + ")."
    )


def _bootstrap(settings):
    by = settings.get("by") or []
    window = settings.get("window")
    method = {"bca": "bias-corrected and accelerated", "percentile": "percentile"}.get(
        settings.get("method"), str(settings.get("method"))
    )
    return (
        f"Intervals of the {str(settings.get('statistic')).replace('_', ' ')} were "
        f"obtained by bootstrap over trials ({settings.get('n_boot')} resamples, "
        f"{method} intervals at {_num(float(settings.get('ci_level', 0.95)) * 100)} %"
        + (f", resampled within each {_join(by)}" if by else "")
        + (f", window {_window(window)}" if window else "")
        + ")."
    )


def _compare(settings):
    conditions = settings.get("conditions") or []
    return (
        f"The {settings.get('parameter')} was compared between "
        + (
            f"the conditions {_join(conditions)}"
            if conditions
            else f"the levels of '{settings.get('condition')}'"
        )
        + f" by a {'paired' if settings.get('paired') else 'unpaired'} permutation test at "
        f"the {settings.get('unit')} level ({settings.get('n_permutations')} permutations, "
        f"{_CORRECTIONS.get(settings.get('correction'), settings.get('correction'))}), "
        f"with bootstrap intervals of the difference ({settings.get('n_boot')} resamples, "
        f"{_num(float(settings.get('ci_level', 0.95)) * 100)} %)."
    )


def _reliability(settings):
    method = settings.get("method", "both")
    parts = []
    if method in ("both", "split_half"):
        parts.append(
            f"split-half correlation over {settings.get('n_splits')} random splits "
            f"(Spearman-Brown corrected)"
        )
    if method in ("both", "dependability"):
        parts.append(
            f"the dependability coefficient of generalisability theory over the "
            f"facets {_join(settings.get('facets') or [])}, with the number of "
            f"trials needed for a dependability of {_num(settings.get('target_phi'))}"
        )
    window = settings.get("window")
    return (
        f"The reliability of the {str(settings.get('scorer')).replace('_', ' ')}"
        + (f" in {_window(window)}" if window else "")
        + f" was estimated by {_join(parts)}."
    )


def _trials_needed(settings):
    scorer = str(settings.get("scorer"))
    what = (
        "the probability of detecting the response"
        if scorer == "existence"
        else f"the precision of the {scorer.replace('_', ' ')}"
    )
    return (
        f"The growth of {what} with the number of trials was estimated by drawing "
        f"{settings.get('n_resamples')} random subsets of trials at each count."
    )


def _jackknife(settings):
    return (
        f"Latencies were scored with the {str(settings.get('scorer')).replace('_', ' ')} "
        f"on the leave-one-out grand averages of the {settings.get('n_subjects')} "
        f"participants (jackknife)"
        + (
            ", and each participant's score was retrieved from them"
            if settings.get("retrieve")
            else ""
        )
        + "."
    )


def _group_summary(settings):
    by = settings.get("by")
    by = _join(by) if isinstance(by, (list, tuple)) else by
    return (
        f"Parameters were summarised per {by} with bootstrap intervals "
        f"({settings.get('n_boot')} resamples, "
        f"{_num(float(settings.get('ci_level', 0.95)) * 100)} %)."
    )


def _apply(settings):
    return (
        f"A user-supplied transform ({settings.get('func')}) was applied to the trials "
        f"(evoparam.data.TrialSet.apply, axis '{settings.get('axis')}')."
    )


def _generic(function, settings):
    scalars = [
        f"{key} = {_setting(value)}"
        for key, value in sorted(settings.items())
        if isinstance(value, (str, int, float, bool)) and key != "package_version"
    ]
    return (
        f"The data were processed with {function}"
        + (f" ({'; '.join(scalars)})" if scalars else "")
        + "."
    )


_TEMPLATES = {
    "evoparam.artifact.remove_artifact": _artifact,
    "evoparam.artifact.paradigm_subtract": _paradigm,
    "evoparam.prep.baseline_correct": _baseline,
    "evoparam.prep.rereference": _reference,
    "evoparam.prep.reject_trials": _reject,
    "evoparam.prep.resample": _resample,
    "evoparam.prep.average": _average,
    "evoparam.data.TrialSet.average": _average,
    "evoparam.models.ResponseModel.fit_group": _fit,
    "evoparam.models.fit_waveform": _fit,
    "evoparam.existence.test_response": _existence,
    "evoparam.trials.fit_trials": _trials,
    "evoparam.trials.realign": _realign,
    "evoparam.hierarchical.fit_hierarchical": _hierarchical,
    "evoparam.stats.variance_components": _variance,
    "evoparam.tf.tf_fit": _tf,
    "evoparam.spatial.shared_shapes": _shared_shapes,
    "evoparam.spatial.fit_shared": _fit_shared,
    "evoparam.events.detect_pulses": _events,
    "evoparam.prep.highpass": _filter("high-pass"),
    "evoparam.prep.lowpass": _filter("low-pass"),
    "evoparam.prep.notch": _filter("notch"),
    "evoparam.prep.train_baseline": _train_baseline,
    "evoparam.prep.reference_sensitivity": _reference_sensitivity,
    "evoparam.artifact.apply_mask": _apply_mask,
    "evoparam.data.TrialSet.apply": _apply,
    "evoparam.stats.bootstrap": _bootstrap,
    "evoparam.stats.compare_conditions": _compare,
    "evoparam.stats.reliability": _reliability,
    "evoparam.stats.trials_needed": _trials_needed,
    "evoparam.stats.jackknife_latency": _jackknife,
    "evoparam.stats.group_summary": _group_summary,
}


def step_sentence(step):
    """Return the sentence of one provenance step, or None for a step that is not described."""
    function = step.get("function", "")
    if function.startswith(SILENT_PREFIXES):
        return None
    settings = step.get("settings") or {}
    template = _TEMPLATES.get(function)
    return template(settings) if template is not None else _generic(function, settings)


###################################################################################################
## The paragraph


def _fit_settings(steps, settings):
    """Return the settings of the last fit step (the result's own settings when it is a fit)."""
    if settings and "fit_window" in settings and "shapes" in settings:
        return settings
    for step in reversed(steps):
        if step.get("function") in _FIT_STEPS:
            return step.get("settings") or {}
    return {}


def _overrides_sentence(fit_settings):
    # the keys whose value differs from the preset (REQ-008); results written before
    # REQ-331 held them in ``_overrides``
    overrides = fit_settings.get("_changed_from_preset")
    if overrides is None:
        overrides = fit_settings.get("_overrides") or []
    if not overrides:
        return ""
    changed = [f"{key} = {_setting(fit_settings.get(key))}" for key in overrides]
    return f" Settings changed from the preset (times in s): {'; '.join(changed)}."


def _seed(provenance, settings, steps):
    for source in (settings or {}, *(step.get("settings") or {} for step in reversed(steps))):
        if isinstance(source, dict) and source.get("random_state") is not None:
            return source["random_state"]
    return getattr(provenance, "random_state", None)


def methods_text(obj, cite=True):
    """Methods paragraph of a result (or container) from its provenance and settings.

    Parameters
    ----------
    obj : result dataclass or container
        Anything with ``provenance`` (a :class:`~evoparam.data.Provenance`) and,
        optionally, ``settings``.
    cite : bool
        End with the package citation and the citations of the methods used.

    Returns
    -------
    str
        The paragraph: package, version and preset; one sentence per processing step;
        the settings changed from the preset; the random seed and how to regenerate the
        result; the citations.
    """
    provenance = getattr(obj, "provenance", None)
    steps = list(getattr(provenance, "steps", None) or [])
    settings = getattr(obj, "settings", None) or {}
    if not isinstance(settings, dict):
        settings = {}
    version = getattr(provenance, "package_version", None) or settings.get("package_version")
    fit_settings = _fit_settings(steps, settings)
    if not steps and fit_settings:
        steps = [{"function": "evoparam.models.fit_waveform", "settings": fit_settings}]

    preset_note = fit_settings.get("preset_note")
    opening = f"Evoked responses were analysed with evoparam (version {version})"
    if preset_note:
        opening += f" using the {preset_note}"
    elif fit_settings.get("preset"):
        preset = fit_settings["preset"]
        name = preset.get("name") if isinstance(preset, dict) else preset
        opening += f" using the preset '{name}'"
    sentences = [opening + "."]

    seen = set()
    references = []
    for step in steps:
        sentence = step_sentence(step)
        if sentence is None:
            continue
        # the same step recorded twice (for example the average done inside a group fit
        # and recorded again) is described once
        if sentence in seen:
            continue
        seen.add(sentence)
        sentences.append(sentence)
        correction = (step.get("settings") or {}).get("correction")
        if (
            step.get("function") == "evoparam.existence.test_response"
            and correction in _REFERENCES
            and _REFERENCES[correction] not in references
        ):
            references.append(_REFERENCES[correction])
        rule = (step.get("settings") or {}).get("rule_spec") or {}
        if rule.get("source") and rule["source"] not in references:
            references.append(rule["source"])
    overrides = _overrides_sentence(fit_settings)
    if overrides:
        sentences.append(overrides.strip())

    seed = _seed(provenance, settings, steps)
    if seed is None:
        sentences.append(
            "No random seed was set, so results that use random draws "
            "(permutation p-values, bootstrap intervals) can differ slightly "
            "between runs; every other setting is recorded in the result's "
            "provenance, from which evoparam.reproduce regenerates it."
        )
    else:
        sentences.append(
            f"Random draws used the seed {seed}; the result can be regenerated "
            f"from its recorded provenance with evoparam.reproduce."
        )
    text = " ".join(sentences)
    if not cite:
        return text
    try:
        software = citation_text(read_citation())
    except OSError:
        software = "evoparam (see CITATION.cff)"
    text += f" Software: {software}"
    if references:
        text += " Methods: " + "; ".join(references) + "."
    return text
