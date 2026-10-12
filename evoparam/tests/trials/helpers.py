# SPDX-License-Identifier: Apache-2.0
"""Simulated trials with known per-trial shifts, amplitudes, doses and phases."""

import warnings
from typing import Any

import numpy as np

from evoparam.data import TrialSet
from evoparam.models import fit as fit_models

SFREQ = 1000.0
TIMES = np.arange(-0.5, 1.0, 1 / SFREQ)


def gauss(latency: Any, fwhm: Any, times: Any = TIMES) -> Any:
    return np.exp(-0.5 * ((times - latency) / (fwhm / 2.3548)) ** 2)


def simulate(
    n: Any = 40,
    jitter: Any = 0.002,
    amp: Any = -30.0,
    cv: Any = 0.3,
    fwhm: Any = 0.02,
    noise: Any = 10.0,
    latency: Any = 0.08,
    seed: Any = 0,
    channels: Any = 1,
    metadata: Any = None,
) -> Any:
    rng = np.random.default_rng(seed)
    shifts = rng.normal(0, jitter, n) if jitter else np.zeros(n)
    scales = rng.normal(1, cv, n) if cv else np.ones(n)
    data = rng.normal(0, noise, (n, channels, len(TIMES)))
    for i in range(n):
        data[i, 0] += amp * scales[i] * gauss(latency + shifts[i], fwhm)
    ts = TrialSet(
        data=data,
        times=TIMES,
        sfreq=SFREQ,
        ch_names=[f"c{i}" for i in range(channels)],
        metadata=metadata,
    )
    return ts, shifts, scales


def channel_fit(ts: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_models(ts, "ccep", **kwargs)
