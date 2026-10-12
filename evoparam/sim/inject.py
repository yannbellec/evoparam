# SPDX-License-Identifier: Apache-2.0
"""Inject simulated components into real recordings (real-noise benchmark, spec §5.12)."""

import time as _time

import numpy as np

from evoparam.data import TrialSet
from evoparam.errors import DataError

from ._engine import assemble, seed_streams
from .components import ERP_COMPONENTS, as_spec
from .generators import _channel_indices, _spec_sources
from .noise import check_license

__all__ = ["inject"]


def _segments(ts, record):
    """One record per trial of ``ts``: dataset, subject, run and sample offsets."""
    from .noise import _source_value

    segments = []
    for trial in range(ts.n_trials):
        absolute = None
        if ts.event_times is not None:
            onset = ts.event_times[trial] * ts.sfreq
            first = int(round(onset + ts.times[0] * ts.sfreq))
            absolute = [first, first + ts.n_times]
        segments.append(
            {
                "segment": trial,
                **record,
                "subject": _source_value(ts, trial, "subject", "subject"),
                "session": _source_value(ts, trial, "session", "session"),
                "run": _source_value(ts, trial, "run", "run"),
                "source_trial": trial,
                "source_channels": list(ts.ch_names),
                "epoch_samples": [0, ts.n_times],
                "recording_samples": absolute,
                "source_sfreq": ts.sfreq,
            }
        )
    return segments


def inject(
    ts_real,
    components,
    *,
    snr,
    channels=None,
    seed=None,
    jitter_sd=0.0,
    jitter="normal",
    jitter_tau=None,
    amplitude_cv=0.0,
    dataset=None,
    keep_arrays=True,
):
    """Add simulated components to real recordings, whose own signal becomes the noise.

    Parameters
    ----------
    ts_real : TrialSet
        Real epochs (for example baselines, or trials without a response).
    components : sequence
        Components as in :func:`~evoparam.sim.simulate_erp` (preset names, specs or
        dicts).
    snr : float, sequence, dict or None
        Peak amplitude over the per-trial SD of the real data on its baseline
        (t < 0), averaged over trials, per channel; None keeps the amplitudes of
        the specifications (in the units of ``ts_real``).
    channels : sequence of str or int, optional
        Channels that receive the components (default: all); a specification's own
        ``channels`` are used as given.
    seed : int, numpy.random.Generator or None
        Random seed (parameter ranges, jitter, amplitude variability).
    jitter_sd, jitter, jitter_tau, amplitude_cv
        Trial variability, as in :func:`~evoparam.sim.simulate_erp`.
    dataset : str, optional
        Registry identifier of ``ts_real`` when its provenance does not name it.

    Returns
    -------
    TrialSet
        ``ts_real`` with the components added (metadata, masks and provenance kept,
        one provenance step added).
    GroundTruth
        Its noise record lists every trial of ``ts_real`` as a segment (dataset,
        version, subject, run, sample offsets).

    Raises
    ------
    SettingsError
        The data's dataset is in the registry under a licence other than CC0, CC BY
        or CC BY-SA; an SNR is requested for a channel without a component.

    Notes
    -----
    Data whose dataset is not named (the user's own recordings) are accepted and
    recorded with ``dataset = None``; a named dataset must be in the registry with an
    allowed licence. NaN samples (inside masks) are left untouched.
    """
    start = _time.perf_counter()
    settings = {key: value for key, value in locals().items() if key != "start"}
    if not isinstance(ts_real, TrialSet):
        raise DataError(f"inject: expected a TrialSet, got {type(ts_real).__name__}.")
    entropy, rngs = seed_streams(seed)
    name = dataset or ts_real.provenance.source.get("dataset")
    if name:
        record = check_license(name)
    else:
        record = {"dataset": None, "version": None, "doi": None, "license": None}
    targets = _channel_indices(channels, ts_real.ch_names, "inject")
    specs = [as_spec(item, ERP_COMPONENTS) for item in components]
    sources = []
    for spec in specs:
        if spec.channels is None:
            from dataclasses import replace

            weights = spec.weights
            if weights is not None and len(weights) == ts_real.n_channels:
                weights = tuple(weights[c] for c in targets)
            spec = replace(spec, channels=tuple(targets), weights=weights)
        sources.extend(_spec_sources(spec, rngs["design"], ts_real.ch_names))
    noise_data = np.array(ts_real.data, dtype=np.float64)
    noise_info = {
        "kind": "real",
        "dataset": record,
        "segments": _segments(ts_real, record),
        "resampling": None,
        "source_units": ts_real.units,
        "source_path": ts_real.provenance.source.get("path"),
    }
    ts, truth = assemble(
        generator="evoparam.sim.inject",
        settings=settings,
        seed=entropy,
        rngs=rngs,
        times=ts_real.times,
        sfreq=ts_real.sfreq,
        ch_names=ts_real.ch_names,
        sources=sources,
        n_trials=ts_real.n_trials,
        snr=snr,
        jitter_sd=jitter_sd,
        jitter=jitter,
        jitter_tau=jitter_tau,
        amplitude_cv=amplitude_cv,
        units=ts_real.units,
        keep_arrays=keep_arrays,
        noise_data=noise_data,
        noise_info=noise_info,
        base=ts_real,
        start=start,
    )
    return ts, truth
