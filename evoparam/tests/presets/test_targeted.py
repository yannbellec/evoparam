# SPDX-License-Identifier: Apache-2.0
"""presets.targeted: one expected component measured on every waveform."""

import warnings

import numpy as np
import pytest

from evoparam import fit, presets, sim
from evoparam.errors import SettingsError


def test_targeted_measures_the_component_on_every_channel():

    p3 = presets.targeted("erp", "P3", (0.3, 0.6), "positive")
    assert p3.name == "erp_P3" and p3.parent == "erp" and p3.selection == "fixed"
    assert p3.fit_window == pytest.approx((0.25, 0.65))
    ts, truth = sim.simulate_erp(60, n_channels=4, components=("P1", "N1", "P3"), seed=0, snr=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = fit(ts, p3, random_state=0, existence=False)
    for result in group.fits:
        assert len(result.components) == 1
        comp = result.components[0]
        assert comp.label == "P3" and comp.amplitude > 0
    true = truth.to_dataframe()
    latency = float(true[true["label"] == "P3"]["latency"].median())
    fitted = np.median([f.components[0].latency for f in group.fits])
    assert abs(fitted - latency) < 0.03
    with pytest.raises(SettingsError):
        presets.targeted("erp", "P3", (0.3, 0.6), "up")
