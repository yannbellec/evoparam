"""The six generators of spec §5.12: ERP, CCEP, ERNA, TEP, ECAP and animal evoked LFP.

Each returns a :class:`~evoparam.data.TrialSet` and the :class:`GroundTruth` of
what was put in it. Every generator takes ``seed`` (an int, a
``numpy.random.Generator`` or None) and is deterministic for a given int seed.
Randomness comes from four independent streams spawned from the seed (design of
the simulation, trial variability, noise, artifact), so that changing the noise
model leaves the components unchanged.

Amplitudes are either given in data units (``snr=None``) or set from a
signal-to-noise ratio: the peak amplitude of a component divided by the noise
SD of its channel, measured per trial on the simulated baseline (t < 0) and
averaged over trials (spec §5.12).
"""

import time as _time

import numpy as np

from evoparam.errors import SettingsError
from evoparam.utils._optional import require

from ._engine import Source, assemble, seed_streams, time_axis
from .artifacts import (dipole_gain, grid_layout, line_layout, pulse_shape, scalp_layout,
                        seeg_layout, tail_shape)
from .components import (CCEP_DELAYED, CCEP_N1, CCEP_N2, ERP_COMPONENTS, TEP_COMPONENTS,
                         ComponentSpec, as_spec)

__all__ = ["simulate_erp", "simulate_ccep", "simulate_erna", "simulate_tep", "simulate_ecap",
           "simulate_animal_ep", "GENERATORS"]


###################################################################################################
## Helpers

def _channel_indices(channels, ch_names, label):
    if channels is None:
        return list(range(len(ch_names)))
    lookup = {name: index for index, name in enumerate(ch_names)}
    out = []
    for channel in channels:
        if isinstance(channel, str):
            if channel not in lookup:
                raise SettingsError(f"component '{label}': unknown channel '{channel}'.")
            out.append(lookup[channel])
        elif isinstance(channel, (int, np.integer)) and -len(ch_names) <= channel < len(ch_names):
            out.append(int(channel) % len(ch_names))
        else:
            raise SettingsError(f"component '{label}': cannot use {channel!r} as a channel.")
    return out


def _spec_sources(spec, rng, ch_names, event=0, **extra):
    """One Source per channel of a specification (drawn once, shared by its channels)."""

    params, amplitude = spec.resolve(rng)
    channels = _channel_indices(spec.channels, ch_names, spec.label)
    weights = np.ones(len(channels)) if spec.weights is None else \
        np.asarray(spec.weights, dtype=float)
    if spec.weights is not None and spec.channels is None and weights.size != len(ch_names):
        raise SettingsError(f"component '{spec.label}': {weights.size} weights for "
                            f"{len(ch_names)} channels.")
    peak = np.max(np.abs(weights)) if weights.size else 0.0
    if not peak > 0:
        raise SettingsError(f"component '{spec.label}': every weight is zero.")
    out = []
    for channel, weight in zip(channels, weights):
        if weight == 0:
            continue
        out.append(Source(label=spec.label, shape=spec.shape, params=params,
                          amplitude=amplitude * weight / peak, channel=channel, role=spec.role,
                          event=event, rel=abs(weight) / peak, **extra))
    return out


def _start():
    return _time.perf_counter()


def _settings(local, drop=()):
    return {key: value for key, value in local.items()
            if key not in ("start",) + tuple(drop) and not key.startswith("_")}


def _check_count(name, value, minimum=1):
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < minimum:
        raise SettingsError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return int(value)


def _ch_info(names, coords, groups, kind, stim=(), coord_space="simulated"):
    n_channels = len(names)
    return {"name": list(names), "type": [kind] * n_channels, "group": list(groups),
            "x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2],
            "coord_space": [coord_space] * n_channels, "reference": [""] * n_channels,
            "bad": np.zeros(n_channels, dtype=bool),
            "stim_contact": np.array([name in stim for name in names], dtype=bool),
            "tissue": ["unknown"] * n_channels, "region": [""] * n_channels}


def _stim_metadata(n_trials, site, current, width, polarity, frequency=np.nan, n_pulses=1):
    return {"trial": np.arange(n_trials, dtype=np.int64), "condition": [""] * n_trials,
            "stim_site": [site] * n_trials,
            "stim_current_mA": np.full(n_trials, float(current)),
            "stim_pulse_width_us": np.full(n_trials, float(width)),
            "stim_polarity": list(polarity) if not isinstance(polarity, str)
            else [polarity] * n_trials,
            "stim_frequency_hz": np.full(n_trials, float(frequency)),
            "stim_n_pulses": np.full(n_trials, int(n_pulses), dtype=np.int64),
            "stim_charge_uC": np.full(n_trials, float(current) * float(width) * 1e-3)}


def _overlap_offsets(overlap, n_trials, rng):
    """SOA of the previous and next events per trial, and the amplitude ratio."""

    if isinstance(overlap, dict):
        soa = overlap.get("soa")
        ratio = float(overlap.get("scale", 1.0))
    else:
        soa, ratio = overlap, 1.0
    if isinstance(soa, (tuple, list)):
        low, high = float(soa[0]), float(soa[1])
        if not 0 < low <= high:
            raise SettingsError(f"overlap: the SOA range must be 0 < low <= high, got {soa}.")
        draw = lambda: rng.uniform(low, high, n_trials)  # noqa: E731
    else:
        value = float(soa)
        if not value > 0:
            raise SettingsError(f"overlap: the SOA must be positive (s), got {soa}.")
        draw = lambda: np.full(n_trials, value)  # noqa: E731
    return draw(), draw(), ratio


###################################################################################################
## ERP

def simulate_erp(n_trials=60, *, n_channels=1, sfreq=500., tmin=-0.2, tmax=0.8,
                 components=("P1", "N1", "P3"), jitter_sd=0., amplitude_cv=0., noise="pink",
                 snr=5., overlap=None, seed=None, jitter="normal", jitter_tau=None,
                 noise_sd=10., noise_kw=None, ch_names=None, keep_arrays=True):
    """Simulate scalp event-related potentials with a known truth.

    Parameters
    ----------
    n_trials : int
        Number of trials.
    n_channels : int
        Number of channels (10-20 names, midline first).
    sfreq : float
        Sampling frequency (Hz).
    tmin, tmax : float
        Epoch ``[tmin, tmax)`` (s).
    components : sequence
        Components: preset names of :data:`~evoparam.sim.ERP_COMPONENTS` (``"P1"``,
        ``"N1"``, ``"N170"``, ``"P3"``, ``"N400"``, ``"MMN"``, ``"ERN"``),
        :class:`~evoparam.sim.ComponentSpec` objects or dicts of their fields (a dict
        whose ``label`` is a preset name overrides that preset). Ranges are drawn
        once per simulation. An empty sequence simulates noise only.
    jitter_sd : float or dict
        Latency jitter SD (s): one shift per trial for the whole response, or
        ``{label: sd}`` for independent shifts per component.
    amplitude_cv : float or dict
        Coefficient of variation of the log-normal amplitude factor (mean 1).
    noise : str
        Noise model of :func:`~evoparam.sim.make_noise` (``"white"``, ``"ar1"``,
        ``"pink"``, ``"real"``, ``"line"``, combinations such as ``"pink+line"``),
        or ``"none"``.
    snr : float, sequence, dict or None
        Peak amplitude over the per-trial baseline noise SD: one value for every
        component, one per component, or a dict keyed by component labels or
        channel names. None keeps the amplitudes of the specifications (µV).
    overlap : float, (float, float), dict or None
        Stimulus-onset asynchrony (s) of the neighbouring events: the response to
        the previous event (at ``-SOA``) and to the next one (at ``+SOA``) are added,
        with SOAs drawn uniformly per trial from a range; a dict
        ``{"soa": ..., "scale": 1.0}`` also scales their amplitude. Their components
        have role ``"overlap"`` and event -1 / +1 in the truth.
    seed : int, numpy.random.Generator or None
        Random seed.
    jitter : {"normal", "exgaussian"}
        Distribution of the latency shifts; ``"exgaussian"`` adds an exponential of
        mean ``jitter_tau`` (default ``jitter_sd``).
    jitter_tau : float or dict, optional
        Mean of the exponential part (s).
    noise_sd : float
        SD of the synthetic noise (µV).
    noise_kw : dict, optional
        Further arguments of :func:`~evoparam.sim.make_noise` (``chi``, ``rho``,
        ``line_freq``, ``source`` for real noise, ...).
    ch_names : list of str, optional
        Channel names.
    keep_arrays : bool
        Keep the noise-free signal in the truth.

    Returns
    -------
    TrialSet
        Units µV, modality ``"erp"``; metadata ``trial`` and ``condition`` (and the
        SOAs with ``overlap``).
    GroundTruth

    Raises
    ------
    SettingsError
        Invalid setting, or an SNR requested for a channel without a component.

    Examples
    --------
    >>> ts, truth = simulate_erp(20, components=["N1", "P3"], snr=5., seed=1)
    >>> ts.data.shape
    (20, 1, 500)
    """

    start = _start()
    settings = _settings(locals())
    n_trials = _check_count("n_trials", n_trials)
    n_channels = _check_count("n_channels", n_channels)
    entropy, rngs = seed_streams(seed)
    times = time_axis(sfreq, tmin, tmax)
    names, coords, groups = scalp_layout(n_channels)
    if ch_names is not None:
        if len(ch_names) != n_channels:
            raise SettingsError(f"ch_names: {len(ch_names)} names for {n_channels} channels.")
        names = [str(name) for name in ch_names]
    specs = [as_spec(item, ERP_COMPONENTS) for item in (components or ())]
    sources = []
    for spec in specs:
        sources.extend(_spec_sources(spec, rngs["design"], names))
    metadata = {"trial": np.arange(n_trials, dtype=np.int64), "condition": [""] * n_trials}
    if overlap is not None:
        previous, following, ratio = _overlap_offsets(overlap, n_trials, rngs["design"])
        metadata.update(soa_previous=previous, soa_next=following)
        for k in range(len(sources)):
            src = sources[k]
            for event, offset in ((-1, -previous), (1, following)):
                sources.append(Source(label=src.label, shape=src.shape, params=src.params,
                                      amplitude=ratio, channel=src.channel, role="overlap",
                                      event=event, offset=offset, parent=k))
    ch_info = _ch_info(names, coords, groups, "eeg", coord_space="unit_disc")
    return assemble(generator="evoparam.sim.simulate_erp", settings=settings, seed=entropy,
                    rngs=rngs, times=times, sfreq=sfreq, ch_names=names, sources=sources,
                    n_trials=n_trials, noise=noise, noise_sd=noise_sd, noise_kw=noise_kw,
                    snr=snr, jitter_sd=jitter_sd, jitter=jitter, jitter_tau=jitter_tau,
                    amplitude_cv=amplitude_cv, metadata=metadata, ch_info=ch_info,
                    modality="erp", keep_arrays=keep_arrays, start=start)


###################################################################################################
## CCEP

def _stim_pair(names, groups, coords, rng, layout):
    """Two adjacent contacts of one shaft (sEEG) or one grid row."""

    pairs = []
    for i in range(len(names) - 1):
        same = groups[i] == groups[i + 1]
        if layout == "grid":
            same = same and np.isclose(coords[i, 1], coords[i + 1, 1])
        if same:
            pairs.append((i, i + 1))
    if not pairs:
        raise SettingsError("simulate_ccep: the layout has no pair of adjacent contacts.")
    return pairs[int(rng.integers(len(pairs)))]


def _ccep_spec(value, default, name):
    if value is None:
        return default
    if isinstance(value, ComponentSpec):
        return value
    if isinstance(value, dict):
        from dataclasses import replace
        return replace(default, **value)
    raise SettingsError(f"{name} must be None, a dict of ComponentSpec fields or a "
                        "ComponentSpec.")


def _train_layout(n_trials, train_structure, rng):
    """Event times, train columns and the adaptation gain of each pulse."""

    if train_structure is None:
        return {"event_times": 5.0 * np.arange(n_trials), "train_id": np.zeros(n_trials, int),
                "index_in_train": np.arange(n_trials), "rate": np.nan, "n_pulses": 1,
                "gain": None}
    structure = dict(train_structure)
    n_pulses = _check_count("train_structure['n_pulses']", structure.get("n_pulses", 10))
    rate = structure.get("rate_hz", (0.2, 1.0))
    rate = float(rng.uniform(*rate)) if isinstance(rate, (tuple, list)) else float(rate)
    if not rate > 0:
        raise SettingsError("train_structure['rate_hz'] must be positive.")
    adaptation = float(structure.get("adaptation", 0.0))
    gap = float(structure.get("gap_s", 10.0))
    index = np.arange(n_trials) % n_pulses
    train = np.arange(n_trials) // n_pulses
    event_times = train * (n_pulses / rate + gap) + index / rate
    gain = np.maximum(0.0, 1.0 + adaptation * index)
    return {"event_times": event_times, "train_id": train, "index_in_train": index,
            "rate": rate, "n_pulses": n_pulses, "gain": gain, "adaptation": adaptation}


def stimulation_artifact(times, sfreq, coords, *, source, sink=None, current_mA,
                         pulse_width_us, biphasic=True, polarity=-1, onset=0.0,
                         scale_uV_per_mA=25000.0, tail_fraction=0.02, tau1=0.0015,
                         tau2=0.015, tail_weight=0.7, radius=1.0):
    """Artifact template of one pulse on every channel, shape (n_channels, n_times).

    ``scale_uV_per_mA × current × (r0/r+ − r0/r−)`` times the recorded pulse plus a
    recovery tail of relative size ``tail_fraction``.
    """

    gain = scale_uV_per_mA * float(current_mA) * dipole_gain(coords, source, sink, radius)
    width = float(pulse_width_us)
    shape = pulse_shape(times, sfreq, pulse_width_us=width,
                        kind="biphasic" if biphasic else "monophasic", polarity=polarity,
                        onset=onset)
    if tail_fraction:
        end = onset + (2 if biphasic else 1) * width * 1e-6
        shape = shape + polarity * tail_fraction * tail_shape(times, sfreq, start=end, tau1=tau1,
                                                              tau2=tau2, weight=tail_weight)
    return gain[:, None] * shape[None, :]


def simulate_ccep(n_trials=10, *, n_channels=46, sfreq=2048., stim_current_mA=4.,
                  pulse_width_us=200., biphasic=True, n1=None, n2=None, p_response=0.3,
                  delayed=False, artifact=True, saturation=False, train_structure=None,
                  seed=None, tmin=-0.5, tmax=1.0, layout="seeg", p_n2=0.8, p_delayed=0.5,
                  polarity="negative", noise="pink", noise_sd=20., noise_kw=None, snr=None,
                  jitter_sd=0., amplitude_cv=0., artifact_scale_uV_per_mA=100000.,
                  artifact_cv=0.02, tail_fraction=0.02, tail_tau=(0.0015, 0.015),
                  keep_arrays=True):
    """Simulate cortico-cortical evoked potentials of single-pulse stimulation.

    One stimulation site (two adjacent contacts) and ``n_channels`` recording
    contacts on sEEG shafts (3.5 mm spacing) or an ECoG grid (10 mm). Each
    non-stimulated contact responds with probability ``p_response``: an N1 (latency
    15-40 ms, FWHM 5-15 ms), an N2 with probability ``p_n2`` (80-250 ms, FWHM
    40-150 ms) and, with ``delayed=True``, a delayed response (100 ms-1 s) with
    probability ``p_delayed``; parameters are drawn per channel.

    The artifact is the potential of the stimulating current (source and sink at the
    two contacts, ``∝ 1/r+ − 1/r−``, so it decays with distance and changes sign)
    times the rectangular pulse seen through the amplifier's anti-aliasing filter,
    followed by a double-exponential recovery tail; it varies by ``artifact_cv``
    between pulses. ``saturation`` clips the data at the rails (True: 30 % of the
    largest artifact; or a value in µV) and marks the clipped samples in ``masks``.
    ``train_structure`` groups pulses in trains, for example ``{"n_pulses": 10,
    "rate_hz": 0.5, "adaptation": -0.03}`` (rate drawn in 0.2-1 Hz when a range; the
    response gain is ``1 + adaptation × index_in_train``).

    Parameters
    ----------
    n_trials : int
        Number of pulses.
    n1, n2 : dict or ComponentSpec, optional
        Overrides of :data:`~evoparam.sim.CCEP_N1` and :data:`~evoparam.sim.CCEP_N2`.
    polarity : {"negative", "positive", "random"}
        Sign of the responses (random: per channel).
    snr : float, sequence, dict or None
        As in :func:`simulate_erp`; None keeps the drawn amplitudes (µV).
    seed : int, numpy.random.Generator or None
        Random seed.
    Other parameters
        See the description above and :func:`simulate_erp`.

    Returns
    -------
    TrialSet
        Units µV, modality ``"spes"``; metadata with the stimulation columns of spec
        Table 4.1 (``stim_site``, ``stim_current_mA``, ``stim_pulse_width_us``,
        ``stim_polarity``, ``stim_frequency_hz``, ``stim_n_pulses``,
        ``stim_charge_uC``) plus ``train_id`` and ``index_in_train``; ``ch_info`` with
        coordinates and ``stim_contact``.
    GroundTruth
        ``artifact`` holds the stimulation site, the geometry and the waveform
        parameters; ``artifact_data`` the artifact itself.

    Examples
    --------
    >>> ts, truth = simulate_ccep(10, n_channels=12, seed=0)
    >>> ts.data.shape, float(ts.metadata["stim_current_mA"].iloc[0])
    ((10, 12, 3072), 4.0)
    """

    start = _start()
    settings = _settings(locals())
    n_trials = _check_count("n_trials", n_trials)
    n_channels = _check_count("n_channels", n_channels, 3)
    if layout not in ("seeg", "grid"):
        raise SettingsError(f"layout must be 'seeg' or 'grid', got {layout!r}.")
    if polarity not in ("negative", "positive", "random"):
        raise SettingsError("polarity must be 'negative', 'positive' or 'random'.")
    for name, value in (("p_response", p_response), ("p_n2", p_n2), ("p_delayed", p_delayed)):
        if not 0 <= value <= 1:
            raise SettingsError(f"{name} must be in [0, 1], got {value}.")
    entropy, rngs = seed_streams(seed)
    design = rngs["design"]
    times = time_axis(sfreq, tmin, tmax)
    names, coords, groups = (seeg_layout if layout == "seeg" else grid_layout)(n_channels)
    anode, cathode = _stim_pair(names, groups, coords, design, layout)
    site = f"{names[anode]}-{names[cathode]}"
    trains = _train_layout(n_trials, train_structure, design)

    specs = {"N1": _ccep_spec(n1, CCEP_N1, "n1"), "N2": _ccep_spec(n2, CCEP_N2, "n2"),
             "delayed": CCEP_DELAYED}
    sources = []
    responders = []
    for channel in range(n_channels):
        if channel in (anode, cathode) or design.random() >= p_response:
            continue
        responders.append(names[channel])
        flip = {"negative": 1.0, "positive": -1.0,
                "random": -1.0 if design.random() < 0.5 else 1.0}[polarity]
        chosen = ["N1"]
        if design.random() < p_n2:
            chosen.append("N2")
        if delayed and design.random() < p_delayed:
            chosen.append("delayed")
        for label in chosen:
            params, amplitude = specs[label].resolve(design)
            sources.append(Source(label=specs[label].label, shape=specs[label].shape,
                                  params=params, amplitude=flip * amplitude, channel=channel,
                                  gain=trains["gain"]))

    artifact_data, artifact_info, rail = None, None, None
    polarity_name = "biphasic_cathodic_first" if biphasic else "monophasic_cathodic"
    if artifact:
        template = stimulation_artifact(times, sfreq, coords, source=coords[anode],
                                        sink=coords[cathode], current_mA=stim_current_mA,
                                        pulse_width_us=pulse_width_us, biphasic=biphasic,
                                        scale_uV_per_mA=artifact_scale_uV_per_mA,
                                        tail_fraction=tail_fraction, tau1=tail_tau[0],
                                        tau2=tail_tau[1])
        variation = 1.0 + artifact_cv * rngs["artifact"].standard_normal(n_trials)
        artifact_data = variation[:, None, None] * template[None]
        artifact_info = {"model": "dipole_pulse_tail", "stim_site": site,
                         "stim_contacts": [names[anode], names[cathode]],
                         "current_mA": float(stim_current_mA),
                         "pulse_width_us": float(pulse_width_us), "biphasic": bool(biphasic),
                         "scale_uV_per_mA": float(artifact_scale_uV_per_mA),
                         "tail_fraction": float(tail_fraction), "tail_tau": list(tail_tau),
                         "filter": {"type": "butterworth_lowpass", "order": 4,
                                    "cutoff_hz": 0.25 * sfreq},
                         "trial_scale": variation.tolist(),
                         "peak_uV": np.max(np.abs(template), axis=1).tolist()}
        if saturation is not False and saturation is not None:
            rail = 0.3 * float(np.max(np.abs(template))) if saturation is True \
                else float(saturation)
    metadata = _stim_metadata(n_trials, site, stim_current_mA, pulse_width_us, polarity_name,
                              trains["rate"], trains["n_pulses"])
    metadata.update(train_id=np.asarray(trains["train_id"], dtype=np.int64),
                    index_in_train=np.asarray(trains["index_in_train"], dtype=np.int64))
    ch_info = _ch_info(names, coords, groups, "seeg" if layout == "seeg" else "ecog",
                       stim=(names[anode], names[cathode]))
    ts, truth = assemble(generator="evoparam.sim.simulate_ccep", settings=settings, seed=entropy,
                         rngs=rngs, times=times, sfreq=sfreq, ch_names=names, sources=sources,
                         n_trials=n_trials, noise=noise, noise_sd=noise_sd, noise_kw=noise_kw,
                         snr=snr, jitter_sd=jitter_sd, amplitude_cv=amplitude_cv,
                         artifact=artifact_data, artifact_info=artifact_info, rail=rail,
                         metadata=metadata, ch_info=ch_info,
                         event_times=np.asarray(trains["event_times"], dtype=float),
                         modality="spes", keep_arrays=keep_arrays, start=start)
    truth.settings["responders"] = responders
    truth.settings["stim_site"] = site
    if train_structure is not None:
        truth.settings["train_rate_hz"] = trains["rate"]
    return ts, truth


###################################################################################################
## ERNA

def simulate_erna(n_pulses=10, *, n_trials=20, sfreq=4096., f=(200., 450.), tau=0.004,
                  pulse_rate=130., artifact=True, n_channels=4, tmin=-0.02, tmax=None,
                  onset_delay=0.001, buildup=2.0, amplitude=(-300., -100.),
                  stim_current_mA=3., pulse_width_us=60., noise="pink", noise_sd=5.,
                  noise_kw=None, snr=None, jitter_sd=0., amplitude_cv=0., seed=None,
                  keep_arrays=True):
    """Simulate evoked resonant neural activity (ERNA) of deep brain stimulation.

    Each trial is a burst of ``n_pulses`` pulses at ``pulse_rate`` Hz on a DBS lead
    (``n_channels`` contacts, 2 mm apart; the first is the stimulating contact).
    After each pulse, a damped oscillation (``damped_oscillation`` shape, onset
    ``onset_delay`` after the pulse, frequency drawn once in ``f``, decay ``tau``)
    whose amplitude builds up over the burst as ``1 - exp(-(k + 1) / buildup)``
    (normalised to 1 at the last pulse) and decays with the distance from the
    second contact. The artifact repeats at the stimulation rate.

    Parameters
    ----------
    n_pulses : int
        Pulses per burst.
    sfreq : float
        Sampling frequency (Hz), 4096 or higher.
    f : float or (float, float)
        ERNA frequency (Hz), or a range drawn once.
    tau : float or (float, float)
        Decay time constant (s).
    pulse_rate : float
        Stimulation rate (Hz).
    amplitude : float or (float, float)
        Amplitude of the last pulse's response at the nearest contact (µV), when
        ``snr`` is None.
    Other parameters
        As in :func:`simulate_ccep`.

    Returns
    -------
    TrialSet
        Modality ``"erna"``; one component per pulse in the truth, with ``event``
        the pulse index.
    GroundTruth

    Examples
    --------
    >>> ts, truth = simulate_erna(5, n_trials=4, seed=0)
    >>> sorted(set(truth.to_dataframe()["event"]))
    [0, 1, 2, 3, 4]
    """

    start = _start()
    settings = _settings(locals())
    n_pulses = _check_count("n_pulses", n_pulses)
    n_trials = _check_count("n_trials", n_trials)
    n_channels = _check_count("n_channels", n_channels, 2)
    if sfreq < 4096:
        raise SettingsError(f"simulate_erna: sfreq must be 4096 Hz or higher, got {sfreq}.")
    if not pulse_rate > 0:
        raise SettingsError("pulse_rate must be positive.")
    entropy, rngs = seed_streams(seed)
    design = rngs["design"]
    if tmax is None:
        tmax = n_pulses / pulse_rate + 0.03
    times = time_axis(sfreq, tmin, tmax)
    names, coords, groups = line_layout(n_channels, 2.0, "L")
    freq = float(design.uniform(*f)) if isinstance(f, (tuple, list)) else float(f)
    decay = float(design.uniform(*tau)) if isinstance(tau, (tuple, list)) else float(tau)
    peak = float(design.uniform(*amplitude)) if isinstance(amplitude, (tuple, list)) \
        else float(amplitude)
    if freq * 2.5 > sfreq:
        raise SettingsError(f"simulate_erna: {freq} Hz is too fast for {sfreq} Hz sampling.")
    distance = dipole_gain(coords, coords[1], radius=2.0)
    build = 1.0 - np.exp(-(np.arange(n_pulses) + 1.0) / float(buildup))
    build = build / build.max()
    sources = []
    for channel in range(1, n_channels):
        rel_space = distance[channel] / distance[1:].max()
        for pulse in range(n_pulses):
            params = np.array([pulse / pulse_rate + onset_delay, decay, freq, -np.pi / 2])
            sources.append(Source(label="ERNA", shape="damped_oscillation", params=params,
                                  amplitude=peak * rel_space * build[pulse], channel=channel,
                                  event=pulse, rel=rel_space * build[pulse], group="ERNA"))
    artifact_data, artifact_info = None, None
    if artifact:
        template = np.zeros((n_channels, times.size))
        for pulse in range(n_pulses):
            template += stimulation_artifact(times, sfreq, coords, source=coords[0],
                                             current_mA=stim_current_mA,
                                             pulse_width_us=pulse_width_us,
                                             onset=pulse / pulse_rate, tail_fraction=0.05,
                                             tau1=0.0003, tau2=0.001)
        artifact_data = np.repeat(template[None], n_trials, axis=0)
        artifact_info = {"model": "monopolar_pulse_train", "stim_contact": names[0],
                         "pulse_rate_hz": float(pulse_rate), "n_pulses": n_pulses,
                         "current_mA": float(stim_current_mA),
                         "pulse_width_us": float(pulse_width_us),
                         "peak_uV": np.max(np.abs(template), axis=1).tolist()}
    metadata = _stim_metadata(n_trials, names[0], stim_current_mA, pulse_width_us,
                              "biphasic_cathodic_first", pulse_rate, n_pulses)
    ch_info = _ch_info(names, coords, groups, "dbs", stim=(names[0],))
    ts, truth = assemble(generator="evoparam.sim.simulate_erna", settings=settings, seed=entropy,
                         rngs=rngs, times=times, sfreq=sfreq, ch_names=names, sources=sources,
                         n_trials=n_trials, noise=noise, noise_sd=noise_sd, noise_kw=noise_kw,
                         snr=snr, jitter_sd=jitter_sd, amplitude_cv=amplitude_cv,
                         artifact=artifact_data, artifact_info=artifact_info, metadata=metadata,
                         ch_info=ch_info, event_times=2.0 * np.arange(n_trials),
                         modality="erna", keep_arrays=keep_arrays, start=start)
    truth.settings.update(erna_freq_hz=freq, erna_tau_s=decay)
    return ts, truth


###################################################################################################
## TEP

AUDITORY_COMPONENTS = {
    "AEP_N100": ComponentSpec("AEP_N100", latency=(0.090, 0.120), fwhm=(0.030, 0.050),
                              amplitude=(-6.0, -2.0), role="confound"),
    "AEP_P200": ComponentSpec("AEP_P200", latency=(0.170, 0.230), fwhm=(0.050, 0.080),
                              amplitude=(3.0, 6.0), role="confound"),
}

# Spatial spread (head radii) of each TEP component around the stimulated site.
_TEP_SPREAD = {"N15": 0.25, "P30": 0.25, "N45": 0.35, "P60": 0.35, "N100": 0.6, "P180": 0.6}


def _blob(coords, centre, spread, floor=0.05):
    weights = np.exp(-np.sum((coords[:, :2] - centre[:2]) ** 2, axis=1) / (2 * spread ** 2))
    weights[weights < floor] = 0.0
    return weights


def simulate_tep(n_trials=40, *, n_channels=32, sfreq=5000., tmin=-0.2, tmax=0.4,
                 components=tuple(TEP_COMPONENTS), stim_site="C3", auditory=True,
                 pulse_artifact=True, decay_artifact=True, muscle_artifact=True, noise="pink",
                 noise_sd=5., noise_kw=None, snr=None, jitter_sd=0., amplitude_cv=0.,
                 seed=None, keep_arrays=True):
    """Simulate TMS-evoked potentials with their artifacts and the auditory confound.

    Components N15, P30, N45, P60, N100 and P180 (latency ranges of
    [LIT-domains-1], :data:`~evoparam.sim.TEP_COMPONENTS`) centred on the stimulated
    site with a spread growing with latency; the auditory response to the coil
    click (``AEP_N100``, ``AEP_P200``, role ``"confound"``, centred on the vertex);
    a TMS pulse artifact (large, recovering within 5-10 ms), a decay artifact
    (exponential, 10-50 ms) and a muscle artifact (high-frequency burst 10-30 ms
    after the pulse) near the coil [LIT-domains-12].

    Parameters
    ----------
    components : sequence
        TEP components (names of :data:`~evoparam.sim.TEP_COMPONENTS`, specs or dicts).
    stim_site : str
        Channel under the coil.
    auditory, pulse_artifact, decay_artifact, muscle_artifact : bool
        Include each confound.
    Other parameters
        As in :func:`simulate_erp`.

    Returns
    -------
    TrialSet
        Modality ``"tep"``.
    GroundTruth
        ``artifact`` holds the parameters of the three artifacts.

    Examples
    --------
    >>> ts, truth = simulate_tep(4, n_channels=8, sfreq=2000., seed=0)
    >>> "AEP_N100" in set(truth.to_dataframe()["label"])
    True
    """

    start = _start()
    settings = _settings(locals())
    n_trials = _check_count("n_trials", n_trials)
    n_channels = _check_count("n_channels", n_channels)
    entropy, rngs = seed_streams(seed)
    design, art_rng = rngs["design"], rngs["artifact"]
    times = time_axis(sfreq, tmin, tmax)
    names, coords, groups = scalp_layout(n_channels)
    if stim_site not in names:
        raise SettingsError(f"stim_site '{stim_site}' is not one of the {n_channels} channels "
                            f"{names}.")
    centre = coords[names.index(stim_site)]
    sources = []
    for item in components:
        spec = as_spec(item, TEP_COMPONENTS)
        if spec.weights is None and spec.channels is None:
            from dataclasses import replace
            spec = replace(spec, weights=tuple(_blob(coords, centre,
                                                     _TEP_SPREAD.get(spec.label, 0.4))))
        sources.extend(_spec_sources(spec, design, names))
    if auditory:
        vertex = np.zeros(3)
        for spec in AUDITORY_COMPONENTS.values():
            from dataclasses import replace
            spec = replace(spec, weights=tuple(_blob(coords, vertex, 0.5)))
            sources.extend(_spec_sources(spec, design, names))

    distance = np.linalg.norm(coords[:, :2] - centre[:2], axis=1)
    near = np.exp(-distance ** 2 / (2 * 0.25 ** 2))
    template_parts, info = {}, {"model": "tms", "stim_site": stim_site}
    if pulse_artifact:
        peak = float(art_rng.uniform(2000.0, 8000.0))
        recovery = float(art_rng.uniform(0.001, 0.002))
        shape = pulse_shape(times, sfreq, pulse_width_us=150.0, kind="biphasic", polarity=1)
        shape = shape / max(np.max(np.abs(shape)), 1e-12)
        shape = shape + tail_shape(times, sfreq, start=0.0003, tau1=recovery, tau2=recovery,
                                   weight=1.0)
        template_parts["pulse"] = (peak * (0.1 + 0.9 * near))[:, None] * shape[None]
        info["pulse"] = {"peak_uV": peak, "recovery_tau_s": recovery}
    if decay_artifact:
        tau = float(art_rng.uniform(0.010, 0.050))
        amp = art_rng.uniform(10.0, 60.0) * np.sign(art_rng.standard_normal())
        shape = tail_shape(times, sfreq, start=0.001, tau1=tau, tau2=tau, weight=1.0)
        template_parts["decay"] = (amp * near)[:, None] * shape[None]
        info["decay"] = {"amplitude_uV": float(amp), "tau_s": tau}
    if muscle_artifact:
        from evoparam.shapes import get_shape
        params = np.array([art_rng.uniform(0.010, 0.020), art_rng.uniform(0.003, 0.006),
                           art_rng.uniform(60.0, 120.0), art_rng.uniform(-np.pi, np.pi)])
        if params[2] * 2.5 > sfreq:
            params[2] = sfreq / 2.5
        amp = float(art_rng.uniform(20.0, 150.0))
        shape = get_shape("gabor").evaluate(times, params)
        lateral = np.exp(-distance ** 2 / (2 * 0.3 ** 2))
        template_parts["muscle"] = (amp * lateral)[:, None] * shape[None]
        info["muscle"] = {"amplitude_uV": amp, "shape": "gabor",
                          "params": dict(zip(("t0", "sigma", "freq", "phase"), params.tolist()))}
    artifact_data = None
    if template_parts:
        cvs = {"pulse": 0.02, "decay": 0.1, "muscle": 0.2}
        artifact_data = np.zeros((n_trials, n_channels, times.size))
        for name, part in template_parts.items():
            factor = 1.0 + cvs[name] * art_rng.standard_normal(n_trials)
            artifact_data += factor[:, None, None] * part[None]
            info[name]["trial_cv"] = cvs[name]
    else:
        info = None
    metadata = {"trial": np.arange(n_trials, dtype=np.int64), "condition": [""] * n_trials,
                "stim_site": [stim_site] * n_trials}
    ch_info = _ch_info(names, coords, groups, "eeg", coord_space="unit_disc")
    return assemble(generator="evoparam.sim.simulate_tep", settings=settings, seed=entropy,
                    rngs=rngs, times=times, sfreq=sfreq, ch_names=names, sources=sources,
                    n_trials=n_trials, noise=noise, noise_sd=noise_sd, noise_kw=noise_kw,
                    snr=snr, jitter_sd=jitter_sd, amplitude_cv=amplitude_cv,
                    artifact=artifact_data, artifact_info=info, metadata=metadata,
                    ch_info=ch_info, modality="tep", keep_arrays=keep_arrays, start=start)


###################################################################################################
## ECAP

_ECAP_KINDS = {
    # N1 latency at the nearest recording contact (s), N1 -> P2 delay (s), FWHMs (s),
    # contact spacing (mm), conduction velocity (mm/s, None: no propagation), space constant.
    "spinal": {"n1": (0.0008, 0.0012), "p2_delay": (0.0003, 0.0006),
               "n1_fwhm": (0.00015, 0.0003), "p2_fwhm": (0.0003, 0.0006), "spacing": 7.0,
               "velocity": 60000.0, "space": 20.0},
    "cochlear": {"n1": (0.0003, 0.0005), "p2_delay": (0.0002, 0.0004),
                 "n1_fwhm": (0.0001, 0.0002), "p2_fwhm": (0.0002, 0.0004), "spacing": 0.75,
                 "velocity": None, "space": 2.0},
}
PARADIGMS = (None, "alternating", "forward_masking")


def simulate_ecap(n_trials=40, *, kind="spinal", n_channels=8, sfreq=30000., tmin=-0.001,
                  tmax=0.003, paradigm=None, masker_probe_interval=0.0005, masking=1.0,
                  stim_current_mA=1., pulse_width_us=50., artifact=True,
                  amplitude=((-200., -50.), (40., 150.)), noise="white", noise_sd=5.,
                  noise_kw=None, snr=None, jitter_sd=0., amplitude_cv=0.,
                  artifact_scale_uV_per_mA=20000., tail_fraction=0.05, tail_tau=0.0002,
                  switch_artifact_uV=20., seed=None, keep_arrays=True):
    """Simulate evoked compound action potentials (spinal cord or cochlea).

    The response is an N1-P2 complex of two ``biexponential`` components (N1 at
    0.3-1.5 ms) on the contacts of a lead (the first contact stimulates). On a
    spinal lead the latency grows with distance (conduction at 60 m/s) and the
    amplitude decays slowly; in the cochlea the latency is the same everywhere and
    the amplitude decays faster. The artifact (pulse plus capacitive tail)
    overlaps the N1.

    Paradigms (spec §5.3 method 9):

    - ``"alternating"``: trials alternate cathodic-first and anodic-first pulses
      (``stim_polarity``, ``pair_id``): the artifact inverts, the response does not.
    - ``"forward_masking"``: sets of four trials A (probe), B (masker then probe,
      ``masker_probe_interval`` apart), C (masker), D (no stimulus) in
      ``paradigm_condition``, with a switch-on artifact in every trial;
      ``A - B + C - D`` leaves the probe response times ``masking``. The masker's
      response has role ``"masker"`` (event -1) in the truth.

    Parameters
    ----------
    kind : {"spinal", "cochlear"}
    paradigm : {None, "alternating", "forward_masking"}
    masking : float
        Fraction of the probe response suppressed by the masker (1: complete).
    amplitude : ((float, float), (float, float))
        Ranges of the N1 and P2 amplitudes at the nearest contact (µV).
    Other parameters
        As in :func:`simulate_ccep`.

    Returns
    -------
    TrialSet
        Modality ``"ecap"``.
    GroundTruth
        Conditions ``"A"`` to ``"D"`` with forward masking.

    Examples
    --------
    >>> ts, truth = simulate_ecap(8, paradigm="alternating", seed=0)
    >>> list(ts.metadata["stim_polarity"][:2])
    ['biphasic_cathodic_first', 'biphasic_anodic_first']
    """

    start = _start()
    settings = _settings(locals())
    n_trials = _check_count("n_trials", n_trials)
    n_channels = _check_count("n_channels", n_channels, 2)
    if kind not in _ECAP_KINDS:
        raise SettingsError(f"kind must be one of {sorted(_ECAP_KINDS)}, got {kind!r}.")
    if paradigm not in PARADIGMS:
        raise SettingsError(f"paradigm must be one of {PARADIGMS}, got {paradigm!r}.")
    if paradigm == "alternating" and n_trials % 2:
        raise SettingsError("paradigm 'alternating' needs an even number of trials.")
    if paradigm == "forward_masking" and n_trials % 4:
        raise SettingsError("paradigm 'forward_masking' needs a multiple of 4 trials.")
    if not 0 <= masking <= 1:
        raise SettingsError(f"masking must be in [0, 1], got {masking}.")
    if paradigm is not None:
        require("pandas", extra="data", caller="evoparam.sim.simulate_ecap (paradigms)")
    entropy, rngs = seed_streams(seed)
    design = rngs["design"]
    form = _ECAP_KINDS[kind]
    times = time_axis(sfreq, tmin, tmax)
    names, coords, groups = line_layout(n_channels, form["spacing"], "E", depth=False)
    distance = np.abs(coords[:, 0] - coords[0, 0])

    if paradigm == "forward_masking":
        condition = np.array(["A", "B", "C", "D"] * (n_trials // 4))
        pair_id = np.repeat(np.arange(n_trials // 4), 4)
    elif paradigm == "alternating":
        condition = np.array([""] * n_trials)
        pair_id = np.repeat(np.arange(n_trials // 2), 2)
    else:
        condition = np.array([""] * n_trials)
        pair_id = np.arange(n_trials)
    sign = np.where((np.arange(n_trials) % 2 == 1) & (paradigm == "alternating"), 1, -1)
    polarity_names = ["biphasic_anodic_first" if s > 0 else "biphasic_cathodic_first"
                      for s in sign]

    n1_latency = float(design.uniform(*form["n1"]))
    p2_delay = float(design.uniform(*form["p2_delay"]))
    fwhm = (float(design.uniform(*form["n1_fwhm"])), float(design.uniform(*form["p2_fwhm"])))
    peaks = (float(design.uniform(*amplitude[0])), float(design.uniform(*amplitude[1])))
    probe_present = condition != "C"
    probe_present &= condition != "D"
    probe_gain = np.where(condition == "B", 1.0 - masking, 1.0)
    if masking == 1.0:
        probe_present &= condition != "B"
    masker_present = (condition == "B") | (condition == "C")
    from .components import shape_params
    sources = []
    for channel in range(1, n_channels):
        delay = 0.0 if form["velocity"] is None else \
            (distance[channel] - distance[1]) / form["velocity"]
        space = np.exp(-(distance[channel] - distance[1]) / form["space"])
        for label, latency, width, peak in (("N1", n1_latency, fwhm[0], peaks[0]),
                                            ("P2", n1_latency + p2_delay, fwhm[1], peaks[1])):
            params = shape_params("biexponential", latency + delay, width, tau_decay_ratio=3.0)
            sources.append(Source(label=label, shape="biexponential", params=params,
                                  amplitude=peak * space, channel=channel, rel=space,
                                  present=probe_present, gain=probe_gain))
            if paradigm == "forward_masking":
                sources.append(Source(label=label, shape="biexponential", params=params,
                                      amplitude=1.0, channel=channel, role="masker", event=-1,
                                      present=masker_present,
                                      offset=np.full(n_trials, -masker_probe_interval),
                                      parent=len(sources) - 1))

    artifact_data, artifact_info = None, None
    if artifact:
        def pulse(onset, polarity):
            return stimulation_artifact(times, sfreq, coords, source=coords[0],
                                        current_mA=stim_current_mA,
                                        pulse_width_us=pulse_width_us, polarity=polarity,
                                        onset=onset,
                                        scale_uV_per_mA=artifact_scale_uV_per_mA,
                                        tail_fraction=tail_fraction, tau1=tail_tau,
                                        tau2=tail_tau)
        probe = {s: pulse(0.0, s) for s in (-1, 1)}
        artifact_data = np.zeros((n_trials, n_channels, times.size))
        for trial in range(n_trials):
            if condition[trial] in ("", "A", "B"):
                artifact_data[trial] += probe[sign[trial]]
        if paradigm == "forward_masking":
            masker = pulse(-masker_probe_interval, -1)
            switch = switch_artifact_uV * tail_shape(times, sfreq, start=tmin + 0.1 * (0 - tmin),
                                                     tau1=0.0002, tau2=0.0002, weight=1.0)
            for trial in range(n_trials):
                if condition[trial] in ("B", "C"):
                    artifact_data[trial] += masker
                artifact_data[trial] += switch[None, :]
        artifact_info = {"model": "monopolar_pulse_tail", "stim_contact": names[0],
                         "current_mA": float(stim_current_mA),
                         "pulse_width_us": float(pulse_width_us),
                         "scale_uV_per_mA": float(artifact_scale_uV_per_mA),
                         "tail_fraction": float(tail_fraction), "tail_tau_s": float(tail_tau),
                         "paradigm": paradigm,
                         "masker_probe_interval_s": float(masker_probe_interval),
                         "switch_artifact_uV": float(switch_artifact_uV)
                         if paradigm == "forward_masking" else None}
    metadata = _stim_metadata(n_trials, names[0], stim_current_mA, pulse_width_us,
                              polarity_names)
    metadata["condition"] = list(condition)
    if paradigm is not None:
        metadata["pair_id"] = pair_id.astype(np.int64)
    if paradigm == "forward_masking":
        metadata["paradigm_condition"] = list(condition)
    ch_info = _ch_info(names, coords, groups, "ecap", stim=(names[0],))
    ts, truth = assemble(generator="evoparam.sim.simulate_ecap", settings=settings, seed=entropy,
                         rngs=rngs, times=times, sfreq=sfreq, ch_names=names, sources=sources,
                         n_trials=n_trials, noise=noise, noise_sd=noise_sd, noise_kw=noise_kw,
                         snr=snr, jitter_sd=jitter_sd, amplitude_cv=amplitude_cv,
                         artifact=artifact_data, artifact_info=artifact_info, metadata=metadata,
                         ch_info=ch_info, conditions=list(condition), modality="ecap",
                         keep_arrays=keep_arrays, start=start)
    return ts, truth


###################################################################################################
## Animal evoked LFP

ANIMAL_COMPONENTS = {
    "visual": (ComponentSpec("N1", latency=(0.040, 0.060), fwhm=(0.020, 0.035),
                             amplitude=(-300.0, -100.0)),
               ComponentSpec("P2", latency=(0.080, 0.120), fwhm=(0.040, 0.070),
                             amplitude=(100.0, 250.0))),
    "electrical": (ComponentSpec("early", latency=(0.003, 0.008), fwhm=(0.002, 0.005),
                                 amplitude=(-300.0, -100.0)),
                   ComponentSpec("late", latency=(0.015, 0.030), fwhm=(0.010, 0.020),
                                 amplitude=(50.0, 200.0))),
}


def simulate_animal_ep(n_trials=100, *, species="mouse", stimulus="visual", n_channels=32,
                       spacing_um=40., sfreq=1250., tmin=-0.2, tmax=0.5, components=None,
                       reversal_um=(300., 500.), reversal_width_um=100.,
                       states=("awake", "nrem"), state_gain=None, artifact=None,
                       noise="pink", noise_sd=30., noise_kw=None, snr=None, jitter_sd=0.,
                       amplitude_cv=0., seed=None, keep_arrays=True):
    """Simulate visual or electrically evoked LFP on a laminar probe.

    Each component has a depth profile ``tanh((z_rev - depth) / width)``: one
    polarity above the reversal depth ``z_rev`` (drawn in ``reversal_um`` per
    component), the other below. Trials are split between ``states`` (metadata
    ``state``, used as the truth's condition) and the response is multiplied by
    ``state_gain[state]`` (default 1 awake, 2 in NREM sleep; up to ×3 between
    states in [LIT-variability-50]). 1250 Hz is the sampling rate of the public
    Neuropixels LFP [LIT-datasets-50].

    Parameters
    ----------
    species : str
        Lower-case species ("mouse", "rat", ...).
    stimulus : {"visual", "electrical"}
        Flash-evoked or electrically evoked response; an electrical stimulus adds a
        stimulation artifact (``artifact`` default True for it).
    n_channels, spacing_um : int, float
        Contacts and their spacing along the probe (µm).
    components : sequence, optional
        Specifications (default: :data:`ANIMAL_COMPONENTS` of the stimulus).
    states : tuple of str
        States the trials are split between, in equal parts at random.
    state_gain : dict, optional
        Response gain per state.
    Other parameters
        As in :func:`simulate_erp`.

    Returns
    -------
    TrialSet
        Modality ``"lfp_ep"``, ``species`` set; metadata ``state``.
    GroundTruth

    Examples
    --------
    >>> ts, truth = simulate_animal_ep(8, n_channels=8, seed=0)
    >>> sorted(set(ts.metadata["state"]))
    ['awake', 'nrem']
    """

    start = _start()
    settings = _settings(locals())
    n_trials = _check_count("n_trials", n_trials)
    n_channels = _check_count("n_channels", n_channels, 2)
    if stimulus not in ANIMAL_COMPONENTS:
        raise SettingsError(f"stimulus must be one of {sorted(ANIMAL_COMPONENTS)}.")
    states = tuple(str(s) for s in states)
    if not states:
        raise SettingsError("states must name at least one state.")
    gains = {"awake": 1.0, "nrem": 2.0} if state_gain is None else dict(state_gain)
    missing = [s for s in states if s not in gains]
    if missing:
        raise SettingsError(f"state_gain has no value for {missing}.")
    entropy, rngs = seed_streams(seed)
    design = rngs["design"]
    times = time_axis(sfreq, tmin, tmax)
    names, coords, groups = line_layout(n_channels, spacing_um / 1000.0, "D")
    depth_um = -coords[:, 2] * 1000.0
    state_of = np.array(states)[design.permutation(np.arange(n_trials) % len(states))]
    trial_gain = np.array([float(gains[s]) for s in state_of])
    specs = components if components is not None else ANIMAL_COMPONENTS[stimulus]
    sources = []
    for item in specs:
        spec = as_spec(item, {s.label: s for s in ANIMAL_COMPONENTS[stimulus]})
        reversal = float(design.uniform(*reversal_um)) if isinstance(reversal_um, (tuple, list)) \
            else float(reversal_um)
        if spec.weights is None and spec.channels is None:
            from dataclasses import replace
            profile = np.tanh((reversal - depth_um) / float(reversal_width_um))
            spec = replace(spec, weights=tuple(profile))
        sources.extend(_spec_sources(spec, design, names, gain=trial_gain))
    use_artifact = (stimulus == "electrical") if artifact is None else bool(artifact)
    artifact_data, artifact_info = None, None
    if use_artifact:
        site = np.array([0.5, 0.0, -0.4])
        template = stimulation_artifact(times, sfreq, coords, source=site, current_mA=0.1,
                                        pulse_width_us=100.0, scale_uV_per_mA=20000.0,
                                        radius=0.05, tail_fraction=0.05, tau1=0.001, tau2=0.005)
        variation = 1.0 + 0.02 * rngs["artifact"].standard_normal(n_trials)
        artifact_data = variation[:, None, None] * template[None]
        artifact_info = {"model": "monopolar_pulse_tail", "site_mm": site.tolist(),
                         "current_mA": 0.1, "pulse_width_us": 100.0,
                         "trial_scale": variation.tolist()}
    metadata = {"trial": np.arange(n_trials, dtype=np.int64), "condition": state_of.tolist(),
                "state": state_of.tolist()}
    ch_info = _ch_info(names, coords, groups, "lfp")
    ts, truth = assemble(generator="evoparam.sim.simulate_animal_ep", settings=settings,
                         seed=entropy, rngs=rngs, times=times, sfreq=sfreq, ch_names=names,
                         sources=sources, n_trials=n_trials, noise=noise, noise_sd=noise_sd,
                         noise_kw=noise_kw, snr=snr, jitter_sd=jitter_sd,
                         amplitude_cv=amplitude_cv, artifact=artifact_data,
                         artifact_info=artifact_info, metadata=metadata, ch_info=ch_info,
                         conditions=state_of.tolist(), species=species, modality="lfp_ep",
                         keep_arrays=keep_arrays, start=start)
    return ts, truth


GENERATORS = {"erp": simulate_erp, "ccep": simulate_ccep, "erna": simulate_erna,
              "tep": simulate_tep, "ecap": simulate_ecap, "animal": simulate_animal_ep}
