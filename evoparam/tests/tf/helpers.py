# SPDX-License-Identifier: Apache-2.0
"""Simulated trials for the time-frequency tests: AR(1) noise, a sharp N1, gamma bursts."""

import numpy as np

from evoparam.data import TrialSet

SFREQ = 1000.
TIMES = np.arange(-0.6, 0.8, 1 / SFREQ)


def ar1(rng, shape, phi=0.9, sd=10.0):
    """AR(1) noise with stationary standard deviation ``sd``."""

    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), shape)
    out = np.empty(shape)
    out[..., 0] = rng.normal(0, sd, shape[:-1])
    for k in range(1, shape[-1]):
        out[..., k] = phi * out[..., k - 1] + e[..., k]
    return out


def n1(latency=0.025, fwhm=0.008, amp=-60.0, times=TIMES):
    return amp * np.exp(-0.5 * ((times - latency) / (fwhm / 2.3548)) ** 2)


def burst(rng, n, freq=80.0, centre=0.25, duration=0.1, amp=3.0, times=TIMES):
    """Gamma bursts with random phase: Hann envelope of ``duration`` s at ``centre``."""

    env = np.where(np.abs(times - centre) <= duration / 2,
                   0.5 + 0.5 * np.cos(2 * np.pi * (times - centre) / duration), 0.0)
    phase = rng.uniform(0, 2 * np.pi, n)[:, None]
    return amp * env[None] * np.cos(2 * np.pi * freq * times[None] + phase)


def trialset(data, modality="spes", metadata=None, masks=None):
    data = np.asarray(data, float)
    if data.ndim == 2:
        data = data[:, None]
    return TrialSet(data=data, times=TIMES, sfreq=SFREQ,
                    ch_names=[f"c{i}" for i in range(data.shape[1])], modality=modality,
                    metadata=metadata, masks=masks)
