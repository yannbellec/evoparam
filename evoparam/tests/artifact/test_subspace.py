"""Whitening and projection away from the artifact subspace (REQ-064)."""

import numpy as np

from evoparam.artifact import compare_methods
from evoparam.data import TrialSet

from .helpers import artifact_shape, response, times_axis

###################################################################################################
###################################################################################################

def test_req064_signal_to_interference_improvement():

    rng = np.random.default_rng(4)
    sfreq, n_trials, n_channels = 2048.0, 30, 32
    times = times_axis(sfreq)
    # Spatially correlated background EEG
    mixing = rng.normal(0, 1, (n_channels, n_channels)) / np.sqrt(n_channels)
    eeg = np.einsum("ij,kjt->kit", mixing, rng.normal(0, 5, (n_trials, n_channels, len(times))))
    topo_response = rng.normal(0, 1, n_channels)
    clean = eeg + topo_response[None, :, None] * response(times, 0.012, -20.0)[None, None]
    # One stimulation site: two fixed artifact topographies
    topo_art = rng.normal(0, 1, (n_channels, 2))
    shapes = np.array([artifact_shape(times, sfreq, 3000.0, 400.0, 0.002),
                       artifact_shape(times, sfreq, 800.0, 0.0, 0.002) * 0.5])
    gains = rng.uniform(0.9, 1.1, (n_trials, 1, 1))
    noisy = clean + gains * (topo_art @ shapes)[None]
    names = [f"E{i}" for i in range(n_channels)]
    make = lambda d: TrialSet(data=d, times=times, sfreq=sfreq, ch_names=names)

    table = compare_methods(make(noisy), [{"method": "subspace", "window": (-0.002, 0.008)}],
                            truth=make(clean), window_eval=(0.008, 0.05))
    assert table["sir_improvement_db"].iloc[0] >= 30
