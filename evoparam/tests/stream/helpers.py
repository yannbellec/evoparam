# SPDX-License-Identifier: Apache-2.0
"""Simulated pulses for the online mode: epochs with known responses, and their channel fits."""

import warnings

import numpy as np

from evoparam.data import TrialSet
from evoparam.models import fit as fit_models

SFREQ = 2048.0
TIMES = np.arange(-1024, 2048) / SFREQ  # -0.5 to 1 s, a sample at t = 0
N1 = (-40.0, 0.03, 0.02)  # amplitude (uV), latency (s), FWHM (s)


def no_pulses(_samples, _first):
    """A detector that finds nothing: the pulses come from push_pulse."""

    return []


def gauss(times, latency, fwhm):
    return np.exp(-0.5 * ((times - latency) / (fwhm / 2.3548)) ** 2)


def simulate(
    n_trials=200, n_channels=2, responsive=None, jitter=0.0015, cv=0.2, noise=10.0, seed=0
):
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


def channel_fits(ts):
    """Channel fits of the trial average with the ccep preset (a GroupFit)."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_models(ts, "ccep")
