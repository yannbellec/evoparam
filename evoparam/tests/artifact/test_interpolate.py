"""Interpolation (method 1, inherited from the skeleton's clean.interpolate)."""

import numpy as np
import pytest

from evoparam.artifact import remove_artifact
from evoparam.errors import ArtifactError, SettingsError
from evoparam.utils._interp import interpolate_window

from .helpers import make_overlay

###################################################################################################
###################################################################################################

def test_interpolate_window_line():

    times = np.arange(-100, 101) / 1000.
    line = 2 * times + 1
    data = np.tile(line, (3, 2, 1))
    data[..., (times >= -0.002) & (times <= 0.010)] = 1e6
    for kind in ("linear", "cubic", "pchip"):
        cleaned, inds = interpolate_window(times, data, (-0.002, 0.010), buffer=3, kind=kind)
        assert np.allclose(cleaned, np.tile(line, (3, 2, 1)))
        assert len(inds) == 13
    assert data.max() == 1e6                                   # input unchanged
    assert np.allclose(interpolate_window(times, data[0, 0], (-0.002, 0.010))[0], line)


def test_interpolate_window_errors():

    times = np.arange(10) / 1000.
    with pytest.raises(ArtifactError):
        interpolate_window(times, np.zeros(10), (0.5, 0.6))
    with pytest.raises(ArtifactError):
        interpolate_window(times, np.zeros(10), (0.0, 0.002), buffer=3)
    with pytest.raises(SettingsError):
        interpolate_window(times, np.zeros(10), (0.004, 0.005), kind="spline")
    with pytest.raises(SettingsError):
        interpolate_window(times, np.zeros(10), (0.004, 0.005), buffer=1, kind="cubic")


@pytest.mark.parametrize("kind", ["linear", "cubic", "pchip"])
def test_remove_artifact_interpolate(kind):

    noisy, truth = make_overlay(onset=0.02)
    result = remove_artifact(noisy, method="interpolate", interpolation=kind)
    inds = np.flatnonzero((noisy.times >= -0.002) & (noisy.times <= 0.008))
    assert result.cleaned.masks[:, :, inds].all()
    assert result.method == "interpolate"
    assert result.window == (-0.002, 0.008)
    # The artifact peak is gone
    assert np.abs(result.cleaned.data[:, :, inds]).max() < 0.2 * np.abs(noisy.data[:, :, inds]).max()


def test_window_outside_epoch():

    noisy, _ = make_overlay()
    with pytest.raises(ArtifactError):
        remove_artifact(noisy, method="interpolate", window=(0.6, 0.7))
