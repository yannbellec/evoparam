# SPDX-License-Identifier: Apache-2.0
"""Synthetic trials for the existence tests."""

from typing import Any

import numpy as np
from scipy.signal import lfilter

from evoparam.data import TrialSet

SFREQ = 1000.0
TIMES = np.arange(-0.5, 0.6, 1 / SFREQ)


def noise_trials(rng: Any, n_trials: Any, n_channels: Any, rho: Any = 0.0, sd: Any = 1.0) -> Any:
    e = rng.normal(0, 1, (n_trials, n_channels, len(TIMES)))
    x = lfilter([1], [1, -rho], e, axis=-1) if rho else e
    x = x - x[..., (TIMES >= -0.5) & (TIMES <= -0.05)].mean(axis=-1, keepdims=True)
    return sd * x / np.std(x)


def trialset(data: Any, metadata: Any = None) -> Any:
    return TrialSet(
        data=data,
        times=TIMES,
        sfreq=SFREQ,
        ch_names=[f"c{i}" for i in range(data.shape[1])],
        metadata=metadata,
    )


def gaussian(latency: Any, fwhm: Any) -> Any:
    return np.exp(-0.5 * ((TIMES - latency) / (fwhm / 2.3548)) ** 2)


def plateau(onset: Any, offset: Any, ramp: Any = 0.002) -> Any:
    up = np.clip((TIMES - onset) / ramp, 0, 1)
    down = np.clip((offset - TIMES) / ramp, 0, 1)
    return np.minimum(up, down)
