# SPDX-License-Identifier: Apache-2.0
"""Synthetic trials for the existence tests."""

import numpy as np
from scipy.signal import lfilter

from evoparam.data import TrialSet

SFREQ = 1000.
TIMES = np.arange(-0.5, 0.6, 1 / SFREQ)


def noise_trials(rng, n_trials, n_channels, rho=0.0, sd=1.0):
    e = rng.normal(0, 1, (n_trials, n_channels, len(TIMES)))
    x = lfilter([1], [1, -rho], e, axis=-1) if rho else e
    x = x - x[..., (TIMES >= -0.5) & (TIMES <= -0.05)].mean(axis=-1, keepdims=True)
    return sd * x / np.std(x)


def trialset(data, metadata=None):
    return TrialSet(data=data, times=TIMES, sfreq=SFREQ,
                    ch_names=[f"c{i}" for i in range(data.shape[1])], metadata=metadata)


def gaussian(latency, fwhm):
    return np.exp(-0.5 * ((TIMES - latency) / (fwhm / 2.3548)) ** 2)


def plateau(onset, offset, ramp=0.002):
    up = np.clip((TIMES - onset) / ramp, 0, 1)
    down = np.clip((offset - TIMES) / ramp, 0, 1)
    return np.minimum(up, down)
