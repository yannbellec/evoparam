# SPDX-License-Identifier: Apache-2.0
"""Montage sensitivity on OpenNeuro ds004080 sub-ccepAgeUMCU01 (REQ-084).

Uses the 40 s segment cached by the events validation (tests/events/test_ds004080_events.py);
runs only with EVOPARAM_NETWORK_TESTS=1.
"""

import os
import re
from typing import Any

import numpy as np
import pytest

from evoparam.data import TrialSet
from evoparam.existence import existence_detector
from evoparam.io import epoch_data
from evoparam.prep import baseline_correct, reference_sensitivity, threshold_detector

pd = pytest.importorskip("pandas")

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
        reason="set EVOPARAM_NETWORK_TESTS=1 to read OpenNeuro",
    ),
]

RUN = "sub-ccepAgeUMCU01/ses-1/ieeg/sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448"


def _epochs() -> Any:
    from evoparam.tests.events.test_ds004080_events import _segment

    data, sfreq, names, t0, onsets, sites = _segment(RUN)
    keep = [
        i
        for i, n in enumerate(names)
        if re.match(r"^[A-Za-z]+\d+$", n)
        and not n.upper().startswith(("MKR", "ECG", "EMG", "EOG"))
    ]
    names = [names[i] for i in keep]
    times, epochs, kept = epoch_data(data[keep], sfreq, onsets, -0.5, 1.0, t0=t0)
    meta = pd.DataFrame({"stim_site": np.asarray(sites)[kept]})
    inside = (times >= -0.002) & (times <= 0.010)
    masks = np.broadcast_to(inside, epochs.shape).copy()
    epochs[..., inside] = 0.0
    ts = TrialSet(
        data=epochs, times=times, sfreq=sfreq, ch_names=names, metadata=meta, masks=masks
    )
    return baseline_correct(ts, check=False)


def test_req084_three_schemes_under_two_minutes() -> None:
    """With the default detector, the existence test of spec §5.6 (method 3)."""

    ts = _epochs()
    table = reference_sensitivity(ts, reference_kwargs={"random_state": 0, "n_boot": 50})
    assert table.attrs["detector"] == "existence_detector"
    assert table.attrs["duration_s"] < 120
    assert len(table) == 3
    assert set(zip(table["scheme_a"], table["scheme_b"], strict=False)) == {
        ("bipolar", "car"),
        ("bipolar", "car_adaptive"),
        ("car", "car_adaptive"),
    }
    assert table["kappa"].between(-1, 1).all()
    assert (table["n_channels"] > 20).all()
    channels = table.attrs["channels"]
    assert {"bipolar", "car", "car_adaptive"} <= set(channels.columns.get_level_values(0))


def test_existence_detector_finds_responses() -> None:

    detected, latency, groups = existence_detector(_epochs())
    assert groups == ["PT01-PT02"]
    assert detected.any()
    assert not detected.all()
    assert np.all((latency[detected] >= 0.009) & (latency[detected] <= 0.5))
    assert np.all(np.isnan(latency[~detected]))


def test_threshold_detector_finds_responses() -> None:

    detected, latency = threshold_detector(_epochs())
    assert detected.any()
    assert np.all((latency[detected] >= 0.009) & (latency[detected] <= 0.1))
