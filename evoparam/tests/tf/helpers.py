# SPDX-License-Identifier: Apache-2.0
"""Simulated trials for the time-frequency tests: AR(1) noise, a sharp N1, gamma bursts."""

from typing import Any

import numpy as np

from evoparam.data import TrialSet

# Dimensions of single-channel data, [n_trials, n_times].
ONE_CHANNEL_NDIM = 2

SFREQ = 1000.0
TIMES = np.arange(-0.6, 0.8, 1 / SFREQ)


def ar1(rng: Any, shape: Any, phi: Any = 0.9, sd: Any = 10.0) -> Any:
    """AR(1) noise with stationary standard deviation ``sd``."""

    e = rng.normal(0, sd * np.sqrt(1 - phi**2), shape)
    out = np.empty(shape)
    out[..., 0] = rng.normal(0, sd, shape[:-1])
    for k in range(1, shape[-1]):
        out[..., k] = phi * out[..., k - 1] + e[..., k]
    return out


def n1(latency: Any = 0.025, fwhm: Any = 0.008, amp: Any = -60.0, times: Any = TIMES) -> Any:
    return amp * np.exp(-0.5 * ((times - latency) / (fwhm / 2.3548)) ** 2)


def burst(
    rng: Any,
    n: Any,
    freq: Any = 80.0,
    centre: Any = 0.25,
    duration: Any = 0.1,
    amp: Any = 3.0,
    times: Any = TIMES,
) -> Any:
    """Gamma bursts with random phase: Hann envelope of ``duration`` s at ``centre``."""

    env = np.where(
        np.abs(times - centre) <= duration / 2,
        0.5 + 0.5 * np.cos(2 * np.pi * (times - centre) / duration),
        0.0,
    )
    phase = rng.uniform(0, 2 * np.pi, n)[:, None]
    return amp * env[None] * np.cos(2 * np.pi * freq * times[None] + phase)


def trialset(data: Any, modality: Any = "spes", metadata: Any = None, masks: Any = None) -> Any:
    data = np.asarray(data, float)
    if data.ndim == ONE_CHANNEL_NDIM:
        data = data[:, None]
    return TrialSet(
        data=data,
        times=TIMES,
        sfreq=SFREQ,
        ch_names=[f"c{i}" for i in range(data.shape[1])],
        modality=modality,
        metadata=metadata,
        masks=masks,
    )
