# SPDX-License-Identifier: Apache-2.0
"""Interpolation (method 1, inherited from the skeleton's clean.interpolate)."""

from typing import Any

import numpy as np
import pytest

from evoparam.artifact import remove_artifact
from evoparam.errors import ArtifactError, SettingsError
from evoparam.utils._interp import interpolate_window

from .helpers import make_overlay

# Window (s) interpolated over, the samples it holds with the buffer, and the value
# written inside it.
WINDOW_START = -0.002
WINDOW_END = 0.010
N_REPLACED = 13
INSIDE = 1e6
# Default window (s) of remove_artifact.
DEFAULT_START = -0.002
DEFAULT_END = 0.008

###################################################################################################
###################################################################################################


def test_interpolate_window_line() -> None:

    times = np.arange(-100, 101) / 1000.0
    line = 2 * times + 1
    data = np.tile(line, (3, 2, 1))
    data[..., (times >= WINDOW_START) & (times <= WINDOW_END)] = INSIDE
    for kind in ("linear", "cubic", "pchip"):
        cleaned, inds = interpolate_window(
            times, data, (WINDOW_START, WINDOW_END), buffer=3, kind=kind
        )
        assert np.allclose(cleaned, np.tile(line, (3, 2, 1)))
        assert len(inds) == N_REPLACED
    assert data.max() == INSIDE  # input unchanged
    assert np.allclose(interpolate_window(times, data[0, 0], (WINDOW_START, WINDOW_END))[0], line)


def test_interpolate_window_errors() -> None:

    times = np.arange(10) / 1000.0
    with pytest.raises(ArtifactError):
        interpolate_window(times, np.zeros(10), (0.5, 0.6))
    with pytest.raises(ArtifactError):
        interpolate_window(times, np.zeros(10), (0.0, 0.002), buffer=3)
    with pytest.raises(SettingsError):
        interpolate_window(times, np.zeros(10), (0.004, 0.005), kind="spline")
    with pytest.raises(SettingsError):
        interpolate_window(times, np.zeros(10), (0.004, 0.005), buffer=1, kind="cubic")


@pytest.mark.parametrize("kind", ["linear", "cubic", "pchip"])
def test_remove_artifact_interpolate(kind: Any) -> None:

    noisy, truth = make_overlay(onset=0.02)
    result = remove_artifact(noisy, method="interpolate", interpolation=kind)
    inds = np.flatnonzero((noisy.times >= DEFAULT_START) & (noisy.times <= DEFAULT_END))
    assert result.cleaned.masks[:, :, inds].all()
    assert result.method == "interpolate"
    assert result.window == (-0.002, 0.008)
    # The artifact peak is gone
    assert (
        np.abs(result.cleaned.data[:, :, inds]).max() < 0.2 * np.abs(noisy.data[:, :, inds]).max()
    )


def test_window_outside_epoch() -> None:

    noisy, _ = make_overlay()
    with pytest.raises(ArtifactError):
        remove_artifact(noisy, method="interpolate", window=(0.6, 0.7))
