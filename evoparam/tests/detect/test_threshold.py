"""Tests for evoparam.detect.threshold."""

import numpy as np

from evoparam.detect.threshold import detect_stim_onsets, match_onsets

###################################################################################################
###################################################################################################

def make_recording(onsets, sfreq=1000., duration=20., n_channels=4, seed=0):

    rng = np.random.default_rng(seed)
    data = rng.normal(0, 1, (n_channels, int(duration * sfreq)))
    for onset in onsets:
        ind = int(round(onset * sfreq))
        data[:, ind:ind + 3] += 500.
    return data

def test_detect_stim_onsets():

    sfreq, onsets = 1000., np.arange(1, 19, 2.5)
    data = make_recording(onsets, sfreq)

    detected = detect_stim_onsets(data, sfreq)
    results = match_onsets(detected, onsets, tolerance=0.002)
    assert results['hits'] == len(onsets)
    assert results['false_alarms'] == 0

    # Clock offset is carried through
    detected = detect_stim_onsets(data, sfreq, t0=100.)
    assert np.allclose(detected - 100., onsets, atol=0.002)

def test_match_onsets():

    results = match_onsets([1.0, 2.001, 5.0], [1.0, 2.0, 3.0], tolerance=0.01)
    assert results['hits'] == 2
    assert results['misses'] == 1
    assert results['false_alarms'] == 1
    assert np.allclose(results['errors'], [0, 0.001])
