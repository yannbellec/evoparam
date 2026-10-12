# SPDX-License-Identifier: Apache-2.0
"""Simulated pulses for the online mode: epochs with known responses, and their channel fits."""

import warnings
from typing import Any

import numpy as np

from evoparam.data import TrialSet
from evoparam.models import fit as fit_models

SFREQ = 2048.0
TIMES = np.arange(-1024, 2048) / SFREQ  # -0.5 to 1 s, a sample at t = 0
N1 = (-40.0, 0.03, 0.02)  # amplitude (uV), latency (s), FWHM (s)


def no_pulses(_samples: Any, _first: Any) -> Any:
    """A detector that finds nothing: the pulses come from push_pulse."""

    return []


def gauss(times: Any, latency: Any, fwhm: Any) -> Any:
    return np.exp(-0.5 * ((times - latency) / (fwhm / 2.3548)) ** 2)


def simulate(
    n_trials: Any = 200,
    n_channels: Any = 2,
    responsive: Any = None,
    jitter: Any = 0.0015,
    cv: Any = 0.2,
    noise: Any = 10.0,
    seed: Any = 0,
) -> Any:
    """Epochs with a jittered, scaled N1 on the ``responsive`` channels (default all)."""

    rng = np.random.default_rng(seed)
    responsive = range(n_channels) if responsive is None else responsive
    data = rng.normal(0, noise, (n_trials, n_channels, len(TIMES)))
    for i in range(n_trials):
        for c in responsive:
            data[i, c] += (
                N1[0]
                * (1 + cv * rng.normal())
                * gauss(TIMES, N1[1] + rng.normal(0, jitter), N1[2])
            )
    return TrialSet(
        data=data, times=TIMES, sfreq=SFREQ, ch_names=[f"c{i}" for i in range(n_channels)]
    )


def channel_fits(ts: Any) -> Any:
    """Channel fits of the trial average with the ccep preset (a GroupFit)."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_models(ts, "ccep")
