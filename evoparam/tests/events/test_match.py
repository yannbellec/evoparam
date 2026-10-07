"""Matching detections to reference onsets, and the system delay."""

import numpy as np

from evoparam.events import estimate_delay, match_onsets

###################################################################################################
###################################################################################################

def test_match_onsets():

    results = match_onsets([1.0, 2.001, 5.0], [1.0, 2.0, 3.0], tolerance=0.01)
    assert results["hits"] == 2
    assert results["misses"] == 1
    assert results["false_alarms"] == 1
    assert np.allclose(results["errors"], [0, 0.001])
    assert np.isclose(results["recall"], 2 / 3)
    assert np.isclose(results["precision"], 2 / 3)


def test_match_onsets_empty():

    results = match_onsets([], [1.0, 2.0])
    assert results["hits"] == 0 and results["misses"] == 2
    assert np.isnan(results["precision"])


def test_estimate_delay_over_recordings():

    rng = np.random.default_rng(0)
    refs = [np.arange(1, 20, 5.0), np.arange(3, 40, 5.0)]
    dets = [ref + 0.001 + rng.normal(0, 1e-5, len(ref)) for ref in refs]
    delay, spread = estimate_delay(dets, refs)
    assert abs(delay - 0.001) < 5e-5
    assert spread < 5e-5
    assert np.isnan(estimate_delay([5.0], [1.0])[0])
