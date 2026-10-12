# SPDX-License-Identifier: Apache-2.0
"""The numbers of the signal plots (REQ-349): the average of a set of waveforms with its
spread band, and the positions of a strip plot, computed in :mod:`evoparam.stats`."""

import numpy as np
import pytest

from evoparam.errors import SettingsError
from evoparam.stats import strip_jitter, waveform_band


def test_waveform_band_mean_std_sem_median():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(12, 30))
    center, half = waveform_band(x)
    assert np.allclose(center, x.mean(axis=0))
    assert np.allclose(half, x.std(axis=0))
    center, half = waveform_band(x, spread="sem", average="median", scale=2.0)
    assert np.allclose(center, np.median(x, axis=0))
    assert np.allclose(half, 2.0 * x.std(axis=0, ddof=1) / np.sqrt(12))
    center, half = waveform_band(x[0], spread=np.ones(30))
    assert np.array_equal(center, x[0])
    assert np.array_equal(half, np.ones(30))
    with pytest.raises(SettingsError):
        waveform_band(x[0])
    with pytest.raises(SettingsError):
        waveform_band(x, spread="iqr")


def test_strip_jitter_is_seeded_and_continues_a_generator():
    a = strip_jitter(20, x=1.0, random_state=3)
    assert np.array_equal(a, strip_jitter(20, x=1.0, random_state=3))
    assert np.allclose(a, 1.0 + np.random.default_rng(3).normal(0.0, 0.025, 20))
    rng = np.random.default_rng(5)
    first, second = strip_jitter(4, random_state=rng), strip_jitter(6, random_state=rng)
    assert np.array_equal(np.r_[first, second], strip_jitter(10, random_state=5))
