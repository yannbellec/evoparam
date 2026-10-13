# SPDX-License-Identifier: Apache-2.0
"""Adjusted common average against the authors' CARLA code on ds004977 (REQ-372); runs with
EVOPARAM_NETWORK_TESTS=1. The authors' channel lists come from
``validation/carla_ds004977.py`` (their MATLAB code, run as an external comparator)."""

import json
import os
import pathlib

import numpy as np
import pytest

# Jaccard index of the channels left by our CARLA and the published one.
MIN_JACCARD = 0.8

ROOT = pathlib.Path(__file__).parents[3]
REFERENCE = ROOT / "validation" / "layer3" / "carla_ds004977.json"
pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
        reason="set EVOPARAM_NETWORK_TESTS=1 to read OpenNeuro",
    ),
    pytest.mark.skipif(not REFERENCE.exists(), reason="comparator file missing"),
]


def test_req372_left_out_channels_agree_with_the_authors_code() -> None:

    from evoparam.prep.reference import car_adaptive_select
    from evoparam.tests.prep.test_reference_ds004977 import load_trials

    reference = json.loads(REFERENCE.read_text())
    for site in reference["sites"]:
        trials = load_trials(site["subject"], site["run"], site["site"])
        index = np.array([trials.ch_names.index(c) for c in site["channels"]])
        chosen = car_adaptive_select(
            trials.data[:, index], trials.times, n_boot=100, random_state=0
        )
        ours = {site["channels"][i] for i in chosen["members"]}
        left_ours = set(site["channels"]) - ours
        left_theirs = set(site["channels"]) - set(site["carla_used"])
        jaccard = len(left_ours & left_theirs) / len(left_ours | left_theirs)
        assert jaccard >= MIN_JACCARD, (site["site"], jaccard)
