# SPDX-License-Identifier: Apache-2.0
"""remove_artifact: one entry point for every method, plus compare_methods."""

import time as _time
import warnings

import numpy as np

from evoparam.data import ArtifactResult
from evoparam.errors import ArtifactError, SettingsError
from evoparam.utils._interp import INTERPOLATIONS, interpolate_window
from evoparam.utils._optional import require

from ._common import GROUP_BY, base_masks, check_trialset, finish, window_slice
from .paradigm import SCHEMES, paradigm_subtract
from .period import estimate_period, period_filter, period_lags
from .quality import artifact_quality

__all__ = ["remove_artifact", "compare_methods", "METHODS"]

METHODS = ("interpolate", "template", "tail", "period_template", "lowrank", "subspace",
           "polynomial", "matching_pursuit", "paradigm", "learned", "none")

# Methods that change samples outside the declared window (spec REQ-060).
GLOBAL_METHODS = ("period_template", "lowrank", "subspace", "polynomial")


def _check_window_vs_rate(ts, window, method):
    """Refuse a window longer than the interval between pulses of a train."""

    rate = None
    if ts.metadata is not None and "stim_frequency_hz" in ts.metadata.columns:
        values = ts.metadata["stim_frequency_hz"].to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        rate = float(np.max(values)) if len(values) else None
    if rate and window[1] - window[0] >= 1.0 / rate:
        raise ArtifactError(f"The window {tuple(window)} s is longer than the interval between "
                            f"pulses of a {rate:g} Hz train; method {method!r} cannot work "
                            "on one pulse at a time. Use method='period_template'.")


def remove_artifact(ts, *, method="template", window=(-0.002, 0.008), buffer=3,
                    interpolation="linear", template="global", group_by=GROUP_BY,
                    n_neighbors=10, scale=True, fallback=False, tail="none",
                    tail_length=0.05, tau_bounds=(0.0005, 0.05), tail_part="full",
                    period=None,
                    expected_rate=None, n_bins=None, n_skip=None, d_period=None, causal=False,
                    pulse_times=None, ranks=(4, 2, 4), post_window=0.03,
                    baseline=(-0.5, -0.05), n_components=None, subspace_extent="window",
                    ridge=1e-3,
                    poly_halfwidth=0.003, poly_extent="local", line_freq=50.0, n_atoms=20,
                    pursuit_length=0.1,
                    scheme=None, pairing="auto", mask_subtracted=False, quality=True,
                    inplace=False,
                    random_state=None, verbose=None):
    """Remove the stimulation artifact, mark replaced samples and score each trial.

    Methods (spec §5.3; each implemented from the published equations):

    - ``"interpolate"``: replace the window by a linear, cubic or PCHIP interpolation
      between ``buffer`` samples on each side. Replaced samples are masked.
    - ``"template"``: subtract the median artifact of each group of pulses (``group_by``;
      ``template="local"``: the ``n_neighbors`` nearest pulses in time; or a given array),
      scaled to each trial by least squares (``scale``). Groups with fewer than 3 pulses
      raise :class:`ArtifactError`, or are interpolated with ``fallback=True``.
    - ``"tail"``: interpolate the window, then fit ``r(t) = c0 + c1 e^(-t/tau1) +
      c2 e^(-t/tau2)`` on ``tail_length`` after it and subtract it (``tail_part="full"``,
      the specification; ``"joined"`` subtracts ``r(t) - r(t_end)`` instead, which keeps
      the level reached at the end of the tail: better for a response inside the tail
      window, worse after it, CHANGELOG); ``tail="joint"`` leaves the tail to the response
      model (background ``"exponential"``).
    - ``"period_template"``: for high-frequency trains: subtract from every sample the mean
      of the samples one or more periods away (fractional period from ``period`` or
      :func:`estimate_period` around ``expected_rate``); ``causal`` uses past samples only.
    - ``"lowrank"``: sequential low-rank regression across channels, pulses and trials
      (``ranks``; pulses at ``pulse_times`` or every ``1 / expected_rate`` from 0).
    - ``"subspace"``: whitening with the ``baseline`` covariance and projection away from
      the artifact directions of the window. By default (``n_components=None`` for this
      method) the directions above the whitened noise level are removed
      (``n_components="noise"``) and only the window is projected
      (``subspace_extent="window"``); the specification's version (95 % of the window
      energy, at most 6 directions, every sample projected) is
      ``n_components="energy", subspace_extent="all"`` (REQ-064: on real TMS and SPES
      artifact templates 45 to 60 dB of SIR improvement against 0.2 to 18 dB, CHANGELOG).
    - ``"polynomial"``: local cubic fit (half-width ``poly_halfwidth``) subtracted; zeros
      while railed (samples in ``ts.masks``), one-sided fits after. ``poly_extent="local"``
      (default) subtracts it from the window start to the end of the artifact (where the
      fitted curve is back within 3 robust SD of its baseline level), with half-cosine
      ramps; ``"global"`` subtracts it at every sample, as published (a high-pass filter
      of the whole epoch: the baseline RMS falls by 9 % on white noise at 2048 Hz).
    - ``"matching_pursuit"``: greedy decomposition into line-noise, response and artifact
      atoms; the artifact atoms are subtracted (patent note, spec §13).
    - ``"paradigm"``: alternating polarity or forward masking (``scheme``; see
      :func:`paradigm_subtract`); the output has one trial per pair or set.
    - ``"learned"``: an artifact basis per group (PCA, 95 % of the energy, at most 8), ridge-
      regressed out of each window; the basis is returned (``result.basis``).
    - ``"none"``: no change (quality only).

    Masks: replaced samples (interpolated, zeroed) are added to ``cleaned.masks``, with the
    existing masks (for example saturation). Methods that subtract an estimate (template,
    learned, the tail of ``tail``, matching pursuit) keep the corrected samples as
    measurements and do not mask them: a fit that starts inside the window then uses them
    (median latency error 0.09 ms against 0.64 ms with them masked, response onset at
    the window end; CHANGELOG). ``mask_subtracted=True`` masks them, as the specification
    writes ("replaced samples are exactly the masked samples").

    Parameters
    ----------
    ts : TrialSet
        Epochs, onset at t = 0.
    method : str
        One of :data:`METHODS`.
    window : (float, float)
        Artifact window ``(a, b)``, in seconds.
    quality : bool
        Compute the quality score (always written to ``metadata["quality"]`` when True).
    inplace : bool
        Accepted for the uniform signature; the input is never modified.
    random_state, verbose
        Unused by the current methods (all deterministic); recorded in the settings.
    Other parameters
        See the method descriptions above and spec §5.3.

    Returns
    -------
    ArtifactResult
        ``cleaned`` (TrialSet), ``method``, ``window``, ``template`` (stacked group templates,
        or None), ``quality`` (per trial), ``quality_components``, ``basis`` (``learned``),
        ``settings`` (every argument, enough to re-create the result) and ``provenance``.

    Raises
    ------
    ArtifactError
        Window outside the epoch or longer than the pulse interval of a train; too few
        pulses for a template; ``lowrank`` without enough channels, pulses or trials;
        ``period_template`` without ``period`` or ``expected_rate``; ``paradigm`` without
        the needed metadata.
    SettingsError
        Unknown method or invalid setting.
    """

    start = _time.perf_counter()
    check_trialset(ts)
    settings = {key: value for key, value in locals().items() if key not in ("ts", "start")}
    if method not in METHODS:
        raise SettingsError(f"method must be one of {METHODS}, not {method!r}.")
    if interpolation not in INTERPOLATIONS:
        raise SettingsError(f"interpolation must be one of {INTERPOLATIONS}.")
    if tail not in ("none", "subtract", "joint"):
        raise SettingsError("tail must be 'none', 'subtract' or 'joint'.")
    window = (float(window[0]), float(window[1]))
    inds = window_slice(ts, window, buffer if method in ("interpolate", "tail") else 0)
    if method in ("interpolate", "template", "learned", "tail"):
        _check_window_vs_rate(ts, window, method)

    masks = base_masks(ts)
    saturation = ts.masks
    info = {"template": None, "basis": None, "fit_r2": None}

    def interp(values):
        out, _ = interpolate_window(ts.times, values, window, buffer, interpolation)
        return out

    if method == "none":
        data = ts.data.copy()
    elif method == "interpolate":
        data = interp(ts.data)
        masks[:, :, inds] = True
    elif method == "template":
        from .template import subtract_template
        data, extra = subtract_template(ts, inds, template=template, group_by=group_by,
                                        n_neighbors=n_neighbors, scale=scale,
                                        fallback=fallback, interpolate=interp)
        if extra["fallback_trials"]:
            masks[np.ix_(extra["fallback_trials"], np.arange(ts.n_channels), inds)] = True
        if mask_subtracted:
            masks[:, :, inds] = True
        info.update(template=extra["template"], fit_r2=extra["fit_r2"],
                    template_response_correlation=extra["template_response_correlation"],
                    fallback_trials=extra["fallback_trials"])
    elif method == "learned":
        from .template import subtract_learned
        data, extra = subtract_learned(ts, inds, group_by=group_by, n_components=n_components,
                                       ridge=ridge)
        if mask_subtracted:
            masks[:, :, inds] = True
        info.update(basis=extra["basis"], fit_r2=extra["fit_r2"],
                    basis_groups=extra["basis_groups"])
    elif method == "tail":
        from .tail import fit_tail
        data = interp(ts.data)
        masks[:, :, inds] = True
        if tail != "joint":
            fitted, tail_inds, params = fit_tail(ts.times, data, inds[-1] + 1, tail_length,
                                                 tau_bounds, part=tail_part)
            data[:, :, tail_inds] -= fitted
            if mask_subtracted:
                masks[:, :, tail_inds] = True
            info.update(tail_tau1=params["tau1"], tail_tau2=params["tau2"])
    elif method == "period_template":
        data, extra = _period(ts, period, expected_rate, n_bins, n_skip, d_period, causal)
        info.update(extra)
    elif method == "lowrank":
        from .lowrank import lowrank_clean
        if pulse_times is None:
            if not expected_rate:
                raise ArtifactError("method 'lowrank' needs pulse_times or expected_rate.")
            pulse_times = np.arange(0.0, ts.times[-1] - post_window, 1.0 / expected_rate)
        data, extra = lowrank_clean(ts.data, ts.sfreq, ts.times, pulse_times, ranks=ranks,
                                    post_window=post_window)
        info.update(removed_fraction=extra["removed_fraction"])
    elif method == "subspace":
        from .subspace import subspace_clean
        data, extra = subspace_clean(ts.data, inds, window_slice(ts, baseline),
                                     n_components="noise" if n_components is None
                                     else n_components, extent=subspace_extent)
        info.update(n_directions=extra["n_directions"],
                    window_energy_removed=extra["window_energy_removed"])
    elif method == "polynomial":
        from .polynomial import polynomial_clean
        data, zeroed = polynomial_clean(ts.data, ts.sfreq, saturation,
                                        halfwidth=poly_halfwidth, extent=poly_extent,
                                        window_inds=inds,
                                        baseline_inds=np.flatnonzero(
                                            (ts.times >= baseline[0])
                                            & (ts.times <= baseline[1])))
        masks |= zeroed
    elif method == "matching_pursuit":
        from .pursuit import pursuit_clean, warn_patent
        warn_patent()
        seg = np.flatnonzero((ts.times >= window[0]) & (ts.times <= window[1] + pursuit_length))
        data = ts.data.copy()
        flat = data[:, :, seg].reshape(-1, len(seg))
        cleaned_seg, _ = pursuit_clean(flat, ts.times[seg], window[1], line_freq=line_freq,
                                       n_iter=n_atoms)
        data[:, :, seg] = cleaned_seg.reshape(ts.n_trials, ts.n_channels, len(seg))
        if mask_subtracted:
            masks[:, :, seg] = True
    elif method == "paradigm":
        if scheme not in SCHEMES:
            raise ArtifactError(f"method 'paradigm' needs scheme in {SCHEMES}.")
        out = paradigm_subtract(ts, scheme=scheme, pairing=pairing)
        data, masks, saturation = out.data, base_masks(out), out.masks
        ts_out = out
    if method != "paradigm":
        ts_out = None

    reference = ts if ts_out is None else ts_out
    cleaned = finish(reference, data, masks, "evoparam.artifact.remove_artifact",
                     _jsonable_settings(settings), start)

    components = None
    scores = np.ones(cleaned.n_trials)
    if quality:
        components = artifact_quality(cleaned, window=window, baseline=baseline,
                                      saturation=saturation, fit_r2=info["fit_r2"])
        scores = components["quality"].to_numpy()
        pd = require("pandas", extra="data", caller="evoparam.artifact.remove_artifact")
        meta = (cleaned.metadata.copy() if cleaned.metadata is not None
                else pd.DataFrame(index=range(cleaned.n_trials)))
        meta["quality"] = scores
        from dataclasses import replace
        cleaned = replace(cleaned, metadata=meta)

    extras = {key: value for key, value in info.items()
              if key not in ("template", "basis", "fit_r2")}
    result_settings = dict(settings, **{"_info": extras})
    return ArtifactResult(cleaned=cleaned, method=method, window=window,
                          template=info["template"], quality=scores,
                          quality_components=components, basis=info["basis"],
                          settings=result_settings, provenance=cleaned.provenance.copy())


def _jsonable_settings(settings):
    out = {}
    for key, value in settings.items():
        if isinstance(value, np.ndarray):
            value = {"array_shape": list(value.shape), "array_sum": float(np.sum(value))}
        out[key] = value
    return out


def _period(ts, period, expected_rate, n_bins, n_skip, d_period, causal):
    if period is None and not expected_rate:
        raise ArtifactError("method 'period_template' needs period (s) or expected_rate (Hz).")
    n_trials, n_channels, n_times = ts.data.shape
    data = np.empty_like(ts.data)
    periods = []
    skip = int(round(0.05 * ts.sfreq)) if n_skip is None else int(n_skip)
    for trial in range(n_trials):
        if period is not None:
            samples = float(period) * ts.sfreq
        else:
            strongest = int(np.argmax(np.std(ts.data[trial], axis=-1)))
            samples = estimate_period(ts.data[trial, strongest], ts.sfreq, expected_rate)
        periods.append(samples)
        lags, used_bins = period_lags(samples, n_times, n_bins=n_bins, n_skip=skip,
                                      d_period=d_period)
        data[trial], _ = period_filter(ts.data[trial], lags, causal=causal)
    return data, {"period_samples": periods, "n_bins": used_bins, "n_skip": skip}


def compare_methods(ts, methods, *, truth=None, window_eval=(0.008, 0.05)):
    """Run several artifact-removal settings and compare them.

    Parameters
    ----------
    ts : TrialSet
    methods : list of dict
        Keyword arguments of :func:`remove_artifact`, one dict per run (``"method"``
        included). A ``"label"`` key names the row.
    truth : TrialSet, optional
        Artifact-free data (synthetic overlay benchmark). With it, each row reports the
        relative RMS error ``rrmse = ||x_hat - x|| / ||x||`` on ``window_eval`` and the
        signal-to-interference ratio before and after cleaning (dB) on the artifact window
        plus ``window_eval``.
    window_eval : (float, float)
        Evaluation window, in seconds.

    Returns
    -------
    pandas.DataFrame
        One row per run: ``label``, ``method``, ``median_quality``, ``mean_residual_ratio``,
        ``runtime_s`` and, with ``truth``, ``rrmse``, ``sir_before_db``, ``sir_after_db``,
        ``sir_improvement_db``.
    """

    pd = require("pandas", extra="data", caller="evoparam.artifact.compare_methods")
    check_trialset(ts)
    rows = []
    for spec in methods:
        spec = dict(spec)
        label = spec.pop("label", spec.get("method", "template"))
        begin = _time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = remove_artifact(ts, **spec)
        row = {"label": label, "method": result.method,
               "median_quality": float(np.median(result.quality)),
               "mean_residual_ratio": float(result.quality_components["residual_ratio"].mean()),
               "runtime_s": _time.perf_counter() - begin}
        if truth is not None and result.cleaned.data.shape == truth.data.shape:
            times = ts.times
            ev = (times >= window_eval[0]) & (times <= window_eval[1])
            full = (times >= result.window[0]) & (times <= window_eval[1])
            clean, ref, raw = result.cleaned.data, truth.data, ts.data
            row["rrmse"] = float(np.linalg.norm(clean[..., ev] - ref[..., ev])
                                 / np.linalg.norm(ref[..., ev]))
            power = np.sum(ref[..., full] ** 2)
            row["sir_before_db"] = float(10 * np.log10(power / np.sum((raw - ref)[..., full] ** 2)))
            row["sir_after_db"] = float(10 * np.log10(power / np.sum((clean - ref)[..., full] ** 2)))
            row["sir_improvement_db"] = row["sir_after_db"] - row["sir_before_db"]
        rows.append(row)
    return pd.DataFrame(rows)
