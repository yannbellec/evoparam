# SPDX-License-Identifier: Apache-2.0
"""Simulated trials with known per-trial shifts, amplitudes, doses and phases."""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from evoparam.data import TrialSet
from evoparam.models import fit as fit_models

SFREQ = 1000.0
TIMES = np.arange(-0.5, 1.0, 1 / SFREQ)


def gauss(latency: Any, fwhm: Any, times: Any = TIMES) -> Any:
    return np.exp(-0.5 * ((times - latency) / (fwhm / 2.3548)) ** 2)


@dataclass(frozen=True)
class TrialsDesign:
    """The keywords of :func:`simulate`: the simulated trials of one response."""

    n: Any = 40
    jitter: Any = 0.002
    amp: Any = -30.0
    cv: Any = 0.3
    fwhm: Any = 0.02
    noise: Any = 10.0
    latency: Any = 0.08
    seed: Any = 0
    channels: Any = 1
    metadata: Any = None


def simulate(**keywords: Any) -> Any:
    """Trials of one jittered, scaled response on channel 0 in white noise: the TrialSet, the
    shifts and the scales. The keywords are the fields of :class:`TrialsDesign`."""

    design = TrialsDesign(**keywords)
    rng = np.random.default_rng(design.seed)
    shifts = rng.normal(0, design.jitter, design.n) if design.jitter else np.zeros(design.n)
    scales = rng.normal(1, design.cv, design.n) if design.cv else np.ones(design.n)
    data = rng.normal(0, design.noise, (design.n, design.channels, len(TIMES)))
    for i in range(design.n):
        data[i, 0] += design.amp * scales[i] * gauss(design.latency + shifts[i], design.fwhm)
    ts = TrialSet(
        data=data,
        times=TIMES,
        sfreq=SFREQ,
        ch_names=[f"c{i}" for i in range(design.channels)],
        metadata=design.metadata,
    )
    return ts, shifts, scales


def channel_fit(ts: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_models(ts, "ccep", **kwargs)
