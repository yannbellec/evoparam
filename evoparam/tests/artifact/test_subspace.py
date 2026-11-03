# SPDX-License-Identifier: Apache-2.0
"""Whitening and projection away from the artifact subspace (REQ-064)."""

from typing import Any

import numpy as np

from evoparam.artifact import compare_methods
from evoparam.data import TrialSet

from .helpers import artifact_shape, response, times_axis

MIN_GAIN_DB = 30

###################################################################################################
###################################################################################################


def _overlay(seed: Any) -> Any:
    rng = np.random.default_rng(seed)
    sfreq, n_trials, n_channels = 2048.0, 30, 32
    times = times_axis(sfreq)
    mixing = rng.normal(0, 1, (n_channels, n_channels)) / np.sqrt(n_channels)
    eeg = np.einsum("ij,kjt->kit", mixing, rng.normal(0, 5, (n_trials, n_channels, len(times))))
    topo_response = rng.normal(0, 1, n_channels)
    clean = eeg + topo_response[None, :, None] * response(times, 0.012, -20.0)[None, None]
    topo_art = rng.normal(0, 1, (n_channels, 2))
    shapes = np.array(
        [
            artifact_shape(times, sfreq, 3000.0, 400.0, 0.002),
            artifact_shape(times, sfreq, 800.0, 0.0, 0.002) * 0.5,
        ]
    )
    gains = rng.uniform(0.9, 1.1, (n_trials, 1, 1))
    noisy = clean + gains * (topo_art @ shapes)[None]
    names = [f"E{i}" for i in range(n_channels)]

    def make(d):
        return TrialSet(data=d, times=times, sfreq=sfreq, ch_names=names)

    return make(noisy), make(clean)


def _gain(seed: Any, **options: Any) -> Any:
    noisy, clean = _overlay(seed)
    table = compare_methods(
        noisy,
        [{"method": "subspace", "window": (-0.002, 0.008), **options}],
        truth=clean,
        window_eval=(0.008, 0.05),
    )
    return table["sir_improvement_db"].iloc[0]


def test_req064_signal_to_interference_improvement() -> None:

    gains = [_gain(seed) for seed in range(4, 14)]
    assert min(gains) >= MIN_GAIN_DB, np.round(gains, 1)
    assert _gain(4, n_components="energy", subspace_extent="all") >= MIN_GAIN_DB
