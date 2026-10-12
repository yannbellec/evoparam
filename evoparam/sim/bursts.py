# SPDX-License-Identifier: Apache-2.0
"""Induced oscillatory bursts with a known truth (spec §8.1, T7).

For the time-frequency task of the benchmark.

An induced burst is an oscillation whose phase is random from trial to trial, so that it
leaves no trace in the average and is seen only in the power of single trials (spec
§5.10). Each burst is a cosine at ``freq`` Hz under a Hann envelope of ``duration`` s
centred at ``centre`` s, with amplitude ``amplitude`` (data units) and a phase drawn
uniformly per trial; its time-frequency zone is the rectangle ``[centre - duration / 2,
centre + duration / 2] x [freq]``. The bursts are added to noise of
:func:`~evoparam.sim.make_noise` (default AR(1), ``rho = 0.9``).
"""

import time as _time
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np

from evoparam._typing import Any, ArrayLike, FloatArray, RandomState
from evoparam.errors import SettingsError

from ._engine import assemble, seed_streams, time_axis

if TYPE_CHECKING:
    from evoparam.data import TrialSet
    from evoparam.sim._truth import GroundTruth

__all__ = ["simulate_bursts", "burst_waveform"]


def burst_waveform(
    times: Any, *, freq: Any, centre: Any, duration: Any, amplitude: Any, phase: ArrayLike = 0.0
) -> FloatArray:
    """One burst on ``times``: ``amplitude * hann(t) * cos(2 pi freq t + phase)``.

    ``phase`` may be an array of shape (n_trials,), the result then has shape
    (n_trials, n_times).
    """
    times = np.asarray(times, float)
    rel = times - float(centre)
    envelope = np.where(
        np.abs(rel) <= duration / 2.0, 0.5 + 0.5 * np.cos(2.0 * np.pi * rel / float(duration)), 0.0
    )
    phases = np.asarray(phase, float)
    return cast(
        "FloatArray",
        float(amplitude)
        * envelope
        * np.cos(2.0 * np.pi * float(freq) * times + phases[..., None]),
    )


def _check_burst(burst: Any, n_channels: int, ch_names: Any, sfreq: float) -> dict[str, Any]:
    keys = {"freq", "centre", "duration", "amplitude"}
    missing = keys - set(burst)
    if missing:
        raise SettingsError(f"simulate_bursts: burst {burst} lacks {sorted(missing)}.")
    out: dict[str, Any] = {key: float(burst[key]) for key in keys}  # floats, and the channel
    if not 0 < out["freq"] < sfreq / 2:
        raise SettingsError(f"simulate_bursts: freq {out['freq']} Hz must be in (0, sfreq/2).")
    if not out["duration"] > 0:
        raise SettingsError("simulate_bursts: duration must be positive.")
    channel = burst.get("channel", 0)
    if isinstance(channel, str):
        if channel not in ch_names:
            raise SettingsError(f"simulate_bursts: unknown channel {channel!r}.")
        channel = ch_names.index(channel)
    if not 0 <= int(channel) < n_channels:
        raise SettingsError(f"simulate_bursts: channel {channel} out of range.")
    out["channel"] = int(channel)
    return out


def simulate_bursts(
    n_trials: int = 30,
    *,
    n_channels: int = 1,
    sfreq: float = 1000.0,
    tmin: float = -0.6,
    tmax: float = 0.8,
    bursts: Any = (),
    noise: str = "ar1",
    noise_sd: float = 10.0,
    noise_kw: dict[str, Any] | None = None,
    seed: RandomState | None = None,
    keep_arrays: bool = True,
) -> tuple["TrialSet", "GroundTruth"]:
    """Simulate trials with induced oscillatory bursts (random phase per trial).

    Parameters
    ----------
    n_trials : int
        Number of trials.
    n_channels : int
        Number of channels (named ``c0``, ``c1``, ...).
    sfreq : float
        Sampling frequency (Hz).
    tmin, tmax : float
        Epoch ``[tmin, tmax)`` (s).
    bursts : sequence of dict
        Each with ``freq`` (Hz), ``centre`` (s), ``duration`` (s, the full width of the Hann
        envelope), ``amplitude`` (data units) and ``channel`` (index or name, default 0).
        An empty sequence simulates noise only (the null cells of a type-I error estimate).
    noise : str
        Noise model of :func:`~evoparam.sim.make_noise`; default ``"ar1"``.
    noise_sd : float
        SD of the noise (µV).
    noise_kw : dict, optional
        Further arguments of :func:`~evoparam.sim.make_noise` (default ``{"rho": 0.9}`` for
        AR(1) noise).
    seed : int, numpy.random.Generator or None
        Random seed.
    keep_arrays : bool
        Keep the noise-free signal in the truth.

    Returns
    -------
    TrialSet
        Units µV, modality ``"erp"``.
    GroundTruth
        No evoked component; ``settings["bursts"]`` lists the bursts with their channel
        name and their zone (``t_start``, ``t_end``, ``freq``), and ``settings["phases"]``
        the phase of each burst in each trial ([n_bursts][n_trials], rad).

    Examples
    --------
    >>> ts, truth = simulate_bursts(10, bursts=[{"freq": 80., "centre": 0.25,
    ...                                          "duration": 0.1, "amplitude": 10.}], seed=0)
    >>> ts.data.shape, truth.settings["bursts"][0]["freq"]
    ((10, 1, 1400), 80.0)
    """
    start = _time.perf_counter()
    settings = {
        "n_trials": n_trials,
        "n_channels": n_channels,
        "sfreq": sfreq,
        "tmin": tmin,
        "tmax": tmax,
        "bursts": [dict(b) for b in bursts],
        "noise": noise,
        "noise_sd": noise_sd,
        "noise_kw": noise_kw,
        "seed": seed,
        "keep_arrays": keep_arrays,
    }
    if int(n_trials) < 1 or int(n_channels) < 1:
        raise SettingsError("simulate_bursts: n_trials and n_channels must be >= 1.")
    n_trials, n_channels = int(n_trials), int(n_channels)
    entropy, rngs = seed_streams(seed)
    times = time_axis(sfreq, tmin, tmax)
    names = [f"c{i}" for i in range(n_channels)]
    checked = [_check_burst(b, n_channels, names, sfreq) for b in bursts]
    kw = dict(noise_kw or {})
    if noise == "ar1":
        kw.setdefault("rho", 0.9)
    ts, truth = assemble(
        generator="evoparam.sim.simulate_bursts",
        settings=settings,
        seed=entropy,
        rngs=rngs,
        times=times,
        sfreq=sfreq,
        ch_names=names,
        sources=[],
        n_trials=n_trials,
        noise=noise,
        noise_sd=noise_sd,
        noise_kw=kw,
        metadata={"trial": np.arange(n_trials, dtype=np.int64), "condition": [""] * n_trials},
        modality="erp",
        keep_arrays=keep_arrays,
        start=start,
    )
    signal = np.zeros(ts.data.shape)
    phases, records = [], []
    for burst in checked:
        phase = rngs["trial"].uniform(0.0, 2.0 * np.pi, n_trials)
        signal[:, burst["channel"]] += burst_waveform(
            times,
            freq=burst["freq"],
            centre=burst["centre"],
            duration=burst["duration"],
            amplitude=burst["amplitude"],
            phase=phase,
        )
        phases.append(phase.tolist())
        records.append(
            {
                **burst,
                "channel": names[burst["channel"]],
                "t_start": burst["centre"] - burst["duration"] / 2.0,
                "t_end": burst["centre"] + burst["duration"] / 2.0,
            }
        )
    provenance = ts.provenance.copy()
    provenance.data_hash = None
    ts = replace(ts, data=ts.data + signal, provenance=provenance)
    truth.settings["bursts"] = records
    truth.settings["phases"] = phases
    if keep_arrays:
        truth.signal = signal
    return ts, truth
