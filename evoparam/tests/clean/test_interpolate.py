"""Tests for evoparam.clean.interpolate."""

import numpy as np
from pytest import raises

from evoparam.clean.interpolate import interpolate_artifact

###################################################################################################
###################################################################################################

def test_interpolate_artifact():

    times = np.arange(-100, 101) / 1000.
    line = 2 * times + 1
    data = np.tile(line, (3, 2, 1))
    data[..., (times >= -0.002) & (times <= 0.010)] = 1e6

    cleaned = interpolate_artifact(times, data, window=(-0.002, 0.010), buffer=3)
    assert cleaned.shape == data.shape
    assert np.allclose(cleaned, np.tile(line, (3, 2, 1)))

    # The input is not changed
    assert data.max() == 1e6

    # 1d input works too
    assert np.allclose(interpolate_artifact(times, data[0, 0]), line)

def test_interpolate_artifact_errors():

    times = np.arange(10) / 1000.
    with raises(ValueError):
        interpolate_artifact(times, np.zeros(10), window=(0.5, 0.6))
    with raises(ValueError):
        interpolate_artifact(times, np.zeros(10), window=(0.0, 0.002), buffer=3)
