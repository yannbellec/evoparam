# SPDX-License-Identifier: Apache-2.0
"""Assembly shared by the generators: seeds, noise, SNR, trial variability, artifact, truth.

A generator describes its components as a list of :class:`Source` (one per
component and channel) and hands them to :func:`assemble` with the time axis,
the noise model, the artifact and the metadata. :func:`assemble` draws the
noise, sets the amplitudes from the requested signal-to-noise ratios, draws the
trial-to-trial variability, sums everything and returns the
:class:`~evoparam.data.TrialSet` and its :class:`GroundTruth`.
"""

import time as _time
from dataclasses import dataclass
from typing import cast

import numpy as np

from evoparam._typing import Any, FloatArray, NDArray
from evoparam.data import Provenance, TrialSet
from evoparam.errors import SettingsError
from evoparam.shapes import compute_features, get_shape
from evoparam.utils._optional import require

from ._truth import GroundTruth
from .noise import make_noise, parse_kind

__all__ = []

# Independent random streams of a simulation, spawned from one seed, so that changing
# the noise model does not change the components and the reverse.
STREAMS = ("design", "trial", "noise", "artifact")

JITTER_DISTRIBUTIONS = ("normal", "exgaussian")

# fewest samples of an epoch and of a baseline
_MIN_SAMPLES = 2
# a sample at t > -1e-12 s is at or after the onset (rounding of k / sfreq)
_ONSET_TOLERANCE_S = -1e-12


@dataclass
class Source:
    """One component on one channel, before the amplitudes are set.

    ``amplitude`` is the nominal signed amplitude (data units); when ``parent`` is
    set, it is instead a multiple of the parent's final amplitude (the response to
    a neighbouring event copies the response to the trial's own event). ``rel`` is
    the spatial weight relative to the largest one, applied to the requested SNR.
    ``gain``, ``present`` and ``offset`` are per-trial arrays (amplitude factor,
    presence, latency offset in s) or None. ``group`` names the components that
    share their latency jitter and amplitude variability (default: the label).
    """

    label: str
    shape: str
    params: NDArray[Any]
    amplitude: float
    channel: int
    role: str = "target"
    event: int = 0
    rel: float = 1.0
    gain: NDArray[Any] | None = None
    present: NDArray[Any] | None = None
    offset: NDArray[Any] | None = None
    group: str | None = None
    parent: int | None = None


###################################################################################################
## Seeds and settings


def seed_streams(seed: Any) -> tuple[int, dict[str, np.random.Generator]]:
    """Seed actually used and one Generator per stream of :data:`STREAMS`.

    An int is used as is; ``None`` draws a seed from the operating system; a
    Generator gives a seed drawn from it. The returned seed regenerates the same
    simulation.
    """
    if isinstance(seed, np.random.Generator):
        entropy = int(seed.integers(0, 2**63 - 1))
    elif seed is None:
        entropy = int(cast(int, np.random.SeedSequence().entropy))  # an int when drawn
    elif isinstance(seed, (int, np.integer)) and not isinstance(seed, bool) and seed >= 0:
        entropy = int(seed)
    else:
        raise SettingsError(
            f"seed must be a non-negative int, a numpy Generator or None, got {seed!r}."
        )
    children = np.random.SeedSequence(entropy).spawn(len(STREAMS))
    return entropy, {
        name: np.random.default_rng(child) for name, child in zip(STREAMS, children, strict=False)
    }


def time_axis(sfreq: float, tmin: float, tmax: float) -> FloatArray:
    """Return the samples ``tmin <= t < tmax`` on the grid ``k / sfreq`` (0.0 is a sample)."""
    if not (np.isfinite(sfreq) and sfreq > 0):
        raise SettingsError(f"sfreq must be a positive number, got {sfreq}.")
    first = int(np.ceil(tmin * sfreq - 1e-9))
    last = int(np.ceil(tmax * sfreq - 1e-9))
    if last - first < _MIN_SAMPLES:
        raise SettingsError(
            f"the epoch ({tmin}, {tmax}) s holds fewer than 2 samples at {sfreq} Hz."
        )
    return np.arange(first, last) / float(sfreq)


def jsonable_settings(settings: Any) -> dict[Any, Any]:
    """Return the settings with specifications, arrays and containers replaced by plain values."""
    out = {}
    for key, raw_value in settings.items():
        value = raw_value
        if hasattr(value, "to_dict") and not isinstance(value, TrialSet):
            value = value.to_dict()
        elif isinstance(value, TrialSet):
            value = {
                "TrialSet": value.provenance.source.get("dataset"),
                "shape": list(value.data.shape),
                "sfreq": value.sfreq,
                "data_hash": value.provenance.data_hash,
            }
        elif isinstance(value, np.random.Generator):
            value = "numpy.random.Generator"
        elif isinstance(value, (list, tuple)):
            value = [v.to_dict() if hasattr(v, "to_dict") else v for v in value]
        elif isinstance(value, dict):
            value = jsonable_settings(value)
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        out[key] = value
    return out


def table(columns: Any, n_rows: Any) -> Any:
    """Return a pandas DataFrame from a dict of columns, or None when pandas is missing."""
    if not columns:
        return None
    try:
        pd = require("pandas", extra="data", caller="evoparam.sim")
    except ImportError:
        return None
    return pd.DataFrame(
        {
            key: (np.asarray(value) if not isinstance(value, list) else value)
            for key, value in columns.items()
        },
        index=pd.RangeIndex(n_rows),
    )


###################################################################################################
## Amplitudes from signal-to-noise ratios


def _check_snr_value(value: Any, what: Any) -> float:
    """Return ``value`` as a float, checked to be a positive finite SNR."""
    value = float(value)
    if not (np.isfinite(value) and value > 0):
        raise SettingsError(f"snr for {what} must be a positive number, got {value}.")
    return cast("float", value)


def _snr_from_dict(
    snr: dict[Any, Any], sources: Any, ch_names: Any, primary: list[int], out: list[float | None]
) -> None:
    """Fill ``out`` from a dict of SNRs keyed by component label or channel name."""
    labels = {sources[k].label for k in primary}
    names = set(ch_names)
    with_component = {ch_names[sources[k].channel] for k in primary}
    for raw_key, raw_value in snr.items():
        key = str(raw_key)
        if key in labels and key in names:
            raise SettingsError(f"snr key '{key}' is both a component label and a channel.")
        if key in names:
            if key not in with_component:
                raise SettingsError(f"snr requested for channel '{key}', which has no component.")
            value = _check_snr_value(raw_value, f"channel '{key}'")
            for k in primary:
                if ch_names[sources[k].channel] == key:
                    out[k] = value
        elif key in labels:
            value = _check_snr_value(raw_value, f"component '{key}'")
            for k in primary:
                if sources[k].label == key and out[k] is None:
                    out[k] = value * sources[k].rel
        else:
            raise SettingsError(
                f"snr key '{key}' is neither a component label "
                f"{sorted(labels)} nor a channel name."
            )


def _snr_from_sequence(
    snr: Any, sources: Any, primary: list[int], out: list[float | None]
) -> None:
    """Fill ``out`` from one SNR per component label, in order of first appearance."""
    labels = list(dict.fromkeys(sources[k].label for k in primary))
    if len(snr) != len(labels):
        raise SettingsError(f"snr: {len(snr)} values for the {len(labels)} components {labels}.")
    values = {
        label: _check_snr_value(v, f"component '{label}'")
        for label, v in zip(labels, snr, strict=False)
    }
    for k in primary:
        out[k] = values[sources[k].label] * sources[k].rel


def requested_snr(snr: Any, sources: Any, ch_names: Any) -> list[float | None]:
    """Return the requested SNR of each source (None when the amplitude is given in data units).

    ``snr`` is None, a number (every component, times its relative spatial weight),
    a sequence (one value per component label, in order of first appearance) or a
    dict whose keys are labels or channel names. A channel key sets the SNR of every
    component on that channel and must name a channel that has one.
    """
    out: list[float | None] = [None] * len(sources)
    primary = [k for k, src in enumerate(sources) if src.parent is None]
    if snr is None:
        return out
    if isinstance(snr, dict):
        _snr_from_dict(snr, sources, ch_names, primary, out)
        return out
    if isinstance(snr, (list, tuple, np.ndarray)):
        _snr_from_sequence(snr, sources, primary, out)
        return out
    value = _check_snr_value(snr, "every component")
    for k in primary:
        out[k] = value * sources[k].rel
    return out


###################################################################################################
## Trial variability


def _per_label(value: Any, label: Any, name: Any) -> float:
    """Return the value of ``name`` for ``label`` (a scalar or a per-label dict), checked >= 0."""
    if isinstance(value, dict):
        value = value.get(label, 0.0)
    value = float(value)
    if not (np.isfinite(value) and value >= 0):
        raise SettingsError(f"{name} must be >= 0, got {value} for '{label}'.")
    return cast("float", value)


def trial_variability(
    sources: Any,
    n_trials: int,
    rng: np.random.Generator,
    jitter_sd: Any,
    jitter: Any,
    jitter_tau: Any,
    amplitude_cv: Any,
) -> tuple[FloatArray, FloatArray]:
    """Latency shift and amplitude factor of each source in each trial.

    A scalar ``jitter_sd`` moves every component of one event by the same shift in
    a trial (the whole response moves); a dict ``{label: sd}`` draws one shift per
    component label. Shifts are normal, or ex-Gaussian (normal plus an exponential
    of mean ``jitter_tau``). Amplitude factors are log-normal with mean 1 and
    coefficient of variation ``amplitude_cv`` (scalar or per label), one per label.
    Both are shared across channels.
    """
    if jitter not in JITTER_DISTRIBUTIONS:
        raise SettingsError(f"jitter must be one of {JITTER_DISTRIBUTIONS}, got {jitter!r}.")
    shifts, factors = {}, {}
    shift = np.zeros((n_trials, len(sources)))
    factor = np.ones((n_trials, len(sources)))
    for k, src in enumerate(sources):
        label = src.group or src.label
        jkey = (src.event, "*" if not isinstance(jitter_sd, dict) else label)
        if jkey not in shifts:
            sd = _per_label(jitter_sd, label, "jitter_sd")
            tau = 0.0
            if jitter == "exgaussian":
                tau = sd if jitter_tau is None else _per_label(jitter_tau, label, "jitter_tau")
            draw = np.zeros(n_trials)
            if sd > 0:
                draw += rng.normal(0.0, sd, n_trials)
            if tau > 0:
                draw += rng.exponential(tau, n_trials)
            shifts[jkey] = draw
        akey = (src.event, label)
        if akey not in factors:
            cv = _per_label(amplitude_cv, label, "amplitude_cv")
            if cv > 0:
                sigma2 = np.log1p(cv**2)
                factors[akey] = np.exp(rng.normal(-0.5 * sigma2, np.sqrt(sigma2), n_trials))
            else:
                factors[akey] = np.ones(n_trials)
        shift[:, k] = shifts[jkey]
        factor[:, k] = factors[akey]
    return shift, factor


###################################################################################################
## Assembly


def baseline_index(times: Any) -> Any:
    """Return the samples before the onset (t < 0).

    Every sample when fewer than 2 precede it.
    """
    index = np.flatnonzero(times < _ONSET_TOLERANCE_S)
    return index if index.size >= _MIN_SAMPLES else np.arange(times.size)


@dataclass
class Design:
    """What one simulation is drawn from.

    The generator's name and settings, the seed and its random streams
    (:func:`seed_streams`), the time axis, the channels, the sources and the number of
    trials. ``start`` is the ``time.perf_counter()`` value at which the generator
    started (the duration recorded in the provenance runs from there).
    """

    generator: str
    settings: Any
    seed: Any
    rngs: Any
    times: Any
    sfreq: float
    ch_names: Any
    sources: Any
    n_trials: int
    start: Any = None


@dataclass
class NoiseSpec:
    """The noise of a simulation: a model of :func:`make_noise`, or given data.

    ``noise_data`` / ``noise_info`` replace the noise model (used by ``inject``, whose
    noise is the real data).
    """

    noise: str = "pink"
    noise_sd: float = 1.0
    noise_kw: Any = None
    noise_data: Any = None
    noise_info: Any = None


@dataclass
class Variability:
    """Amplitudes from signal-to-noise ratios and the trial-to-trial variability."""

    snr: Any = None
    jitter_sd: float | dict[str, Any] = 0.0
    jitter: str = "normal"
    jitter_tau: Any = None
    amplitude_cv: float | dict[str, Any] = 0.0


@dataclass
class Recording:
    """What the simulated recording carries besides the signal and the noise.

    The artifact (data, description and amplifier rail), the trial metadata, the
    channel table, the event times, the condition of each trial, the species,
    modality and unit, and whether the truth keeps the noise-free arrays. ``base`` is
    the TrialSet whose metadata, masks and provenance are kept (``inject``).
    """

    artifact: Any = None
    artifact_info: Any = None
    rail: Any = None
    metadata: Any = None
    ch_info: Any = None
    event_times: Any = None
    conditions: Any = None
    species: str = "human"
    modality: Any = None
    units: str = "uV"
    keep_arrays: bool = True
    base: Any = None


_FEATURE_NAMES = (
    "latency",
    "amplitude",
    "polarity",
    "fwhm",
    "onset",
    "offset",
    "area",
    "rise_time",
    "decay_time",
    "symmetry",
    "sharpness",
    "peak_time",
)


def _draw_noise(
    design: Design, spec: NoiseSpec, units: str
) -> tuple[Any, dict[str, Any], FloatArray]:
    """Return the noise of every trial, its description and the baseline noise SD per channel."""
    noise_data, noise_info = spec.noise_data, spec.noise_info
    times = design.times
    if noise_data is None:
        shape = (design.n_trials, len(design.ch_names), times.size)
        broadband, with_line = parse_kind(spec.noise)
        if broadband is None and not with_line:
            noise_data = np.zeros(shape)
            noise_info = {"kind": "none"}
        else:
            kwargs = dict(spec.noise_kw or {})
            if broadband != "real" or "sd" in kwargs:
                kwargs.setdefault("sd", spec.noise_sd)
            noise_data, noise_info = make_noise(
                times.size,
                design.sfreq,
                spec.noise,
                size=(design.n_trials, len(design.ch_names)),
                units=units,
                seed=design.rngs["noise"],
                return_info=True,
                **kwargs,
            )
    base_idx = baseline_index(times)
    trial_sd = np.nanstd(noise_data[:, :, base_idx], axis=-1)
    baseline_sd = np.nanmean(trial_sd, axis=0)
    noise_info = dict(noise_info)
    noise_info["baseline_sd"] = baseline_sd.tolist()
    noise_info["baseline_window"] = [float(times[base_idx[0]]), float(times[base_idx[-1]])]
    return noise_data, noise_info, baseline_sd


def _amplitudes(
    design: Design, snr: Any, baseline_sd: Any
) -> tuple[FloatArray, list[float | None]]:
    """Return the amplitude of each source and the SNR recorded for it.

    A source with a requested SNR gets ``SNR x baseline SD`` with the sign of its
    nominal amplitude; a source with a parent gets a multiple of the parent's amplitude.
    """
    sources, ch_names = design.sources, design.ch_names
    targets = requested_snr(snr, sources, ch_names)
    amplitude = np.zeros(len(sources))
    record_snr: list[float | None] = [None] * len(sources)
    for k, src in enumerate(sources):
        if src.parent is not None:
            continue
        target = targets[k]
        if target is None:
            amplitude[k] = float(src.amplitude)
            continue
        sd = baseline_sd[src.channel]
        if not (np.isfinite(sd) and sd > 0):
            raise SettingsError(
                f"snr cannot be set on channel '{ch_names[src.channel]}': its "
                "baseline noise SD is zero (no noise)."
            )
        sign = -1.0 if src.amplitude < 0 else 1.0
        amplitude[k] = sign * target * sd
        record_snr[k] = target
    for k, src in enumerate(sources):
        if src.parent is not None:
            amplitude[k] = amplitude[src.parent] * float(src.amplitude)
            if record_snr[src.parent] is not None:
                record_snr[k] = record_snr[src.parent] * abs(float(src.amplitude))
    return amplitude, record_snr


def _trial_draws(
    design: Design, variability: Variability
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """Return the latency shift, the amplitude scale and the presence per trial and source."""
    sources, n_trials = design.sources, design.n_trials
    shift, factor = trial_variability(
        sources,
        n_trials,
        design.rngs["trial"],
        variability.jitter_sd,
        variability.jitter,
        variability.jitter_tau,
        variability.amplitude_cv,
    )
    present = np.ones((n_trials, len(sources)), dtype=bool)
    for k, src in enumerate(sources):
        if src.offset is not None:
            shift[:, k] += np.asarray(src.offset, dtype=float)
        if src.gain is not None:
            factor[:, k] *= np.asarray(src.gain, dtype=float)
        if src.present is not None:
            present[:, k] = np.asarray(src.present, dtype=bool)
    scale = np.where(present, factor, 0.0)
    return shift, scale, present


def _component_record(
    src: Source,
    shape_obj: Any,
    params: Any,
    feats: dict[str, Any],
    amplitude: float,
    snr: float | None,
    design: Design,
) -> dict[str, Any]:
    """Return the truth record of one source (nominal features scaled by its amplitude)."""
    feats = dict(feats)
    feats["amplitude"] = feats["amplitude"] * amplitude
    feats["area"] = feats["area"] * amplitude
    feats["polarity"] = 1 if feats["amplitude"] >= 0 else -1
    return {
        "label": src.label,
        "shape": src.shape,
        "param_names": list(shape_obj.param_names),
        "params": params.tolist(),
        "loc_param": shape_obj.latency_param,
        "amplitude": float(amplitude),
        "channel": design.ch_names[src.channel],
        "role": src.role,
        "event": int(src.event),
        "snr": snr,
        "features": {name: feats[name] for name in _FEATURE_NAMES},
    }


def _signal(
    design: Design, amplitude: FloatArray, record_snr: list[float | None], shift: Any, scale: Any
) -> tuple[FloatArray, list[dict[str, Any]]]:
    """Return the noise-free signal of every trial and the truth record of each source."""
    times, n_trials = design.times, design.n_trials
    signal = np.zeros((n_trials, len(design.ch_names), times.size))
    feature_cache = {}
    components = []
    for k, src in enumerate(design.sources):
        shape_obj = get_shape(src.shape)
        params = np.asarray(src.params, dtype=np.float64)
        loc = shape_obj.param_names.index(shape_obj.latency_param)
        trial_params = np.tile(params, (n_trials, 1))
        trial_params[:, loc] += shift[:, k]
        weights = amplitude[k] * scale[:, k]
        if np.any(weights != 0):
            signal[:, src.channel] += weights[:, None] * shape_obj.evaluate(times, trial_params)
        key = (src.shape, params.tobytes())
        if key not in feature_cache:
            feature_cache[key] = compute_features(
                shape_obj, times, params, 1.0, sfreq=design.sfreq
            ).to_dict()
        components.append(
            _component_record(
                src, shape_obj, params, feature_cache[key], amplitude[k], record_snr[k], design
            )
        )
    return signal, components


def _add_artifact(data: Any, recording: Recording) -> tuple[Any, Any, Any]:
    """Return the data with the artifact added and clipped at the rail, masks and description."""
    artifact_info = None if recording.artifact_info is None else dict(recording.artifact_info)
    if recording.artifact is not None:
        data = data + recording.artifact
    masks = None if recording.base is None else recording.base.masks
    rail = recording.rail
    if rail is not None:
        clipped = np.abs(data) >= rail
        data = np.clip(data, -rail, rail)
        masks = clipped if masks is None else (masks | clipped)
        if artifact_info is not None:
            artifact_info["rail"] = float(rail)
            artifact_info["n_clipped_samples"] = int(clipped.sum())
    if masks is not None and not masks.any():
        masks = None
    return data, masks, artifact_info


def _trialset(
    design: Design, recording: Recording, data: Any, masks: Any, settings: Any
) -> TrialSet:
    """Return the simulated TrialSet, with the generator step in its provenance."""
    duration = _time.perf_counter() - design.start
    generator, seed, base = design.generator, design.seed, recording.base
    if base is None:
        provenance = Provenance(
            source={"format": "simulation", "generator": generator}, random_state=seed
        )
        provenance = provenance.with_step(
            generator, settings, duration_s=duration, random_state=seed
        )
        n_trials, n_channels = design.n_trials, len(design.ch_names)
        return TrialSet(
            data=data,
            times=design.times,
            sfreq=design.sfreq,
            ch_names=list(design.ch_names),
            units=recording.units,
            metadata=table(recording.metadata, n_trials),
            ch_info=table(recording.ch_info, n_channels),
            masks=masks,
            event_times=recording.event_times,
            species=recording.species,
            modality=recording.modality,
            provenance=provenance,
        )
    provenance = base.provenance.with_step(
        generator, settings, duration_s=duration, random_state=seed
    )
    provenance.data_hash = None
    from dataclasses import replace

    out: TrialSet = replace(base, data=data, masks=masks, provenance=provenance)
    return out


def assemble(
    design: Design,
    noise: NoiseSpec | None = None,
    variability: Variability | None = None,
    recording: Recording | None = None,
) -> tuple[TrialSet, GroundTruth]:
    """Build the TrialSet and the GroundTruth of one simulation (see module docstring)."""
    noise = NoiseSpec() if noise is None else noise
    variability = Variability() if variability is None else variability
    recording = Recording() if recording is None else recording
    if design.start is None:
        design.start = _time.perf_counter()
    noise_data, noise_info, baseline_sd = _draw_noise(design, noise, recording.units)
    amplitude, record_snr = _amplitudes(design, variability.snr, baseline_sd)
    shift, scale, present = _trial_draws(design, variability)
    signal, components = _signal(design, amplitude, record_snr, shift, scale)
    data, masks, artifact_info = _add_artifact(signal + noise_data, recording)
    settings = jsonable_settings(design.settings)
    ts = _trialset(design, recording, data, masks, settings)
    conditions = recording.conditions
    if conditions is None:
        conditions = [""] * design.n_trials
    keep = recording.keep_arrays
    truth = GroundTruth(
        generator=design.generator,
        ch_names=list(design.ch_names),
        times=design.times,
        sfreq=design.sfreq,
        units=recording.units,
        components=components,
        shift=shift,
        scale=scale,
        present=present,
        conditions=list(conditions),
        noise=noise_info,
        artifact=artifact_info,
        seed=design.seed,
        settings=settings,
        signal=signal if keep else None,
        artifact_data=recording.artifact if (keep and recording.artifact is not None) else None,
        provenance=ts.provenance.copy(),
    )
    return ts, truth
