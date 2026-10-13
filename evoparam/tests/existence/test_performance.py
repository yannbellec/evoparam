# SPDX-License-Identifier: Apache-2.0
"""Speed of the permutation test (REQ-138); runs only with EVOPARAM_PERF_TESTS=1."""

import os
import time

import numpy as np
import pytest

from evoparam.data import TrialSet
from evoparam.existence import test_response as run_test

# Time limit (s) of the existence test on 100 channels.
MAX_SECONDS = 20

pytestmark = pytest.mark.skipif(
    os.environ.get("EVOPARAM_PERF_TESTS") != "1",
    reason="set EVOPARAM_PERF_TESTS=1 to time the test",
)


def test_req138_100_channels_50_trials_1000_permutations() -> None:

    rng = np.random.default_rng(0)
    times = np.arange(-0.5, 0.524, 1 / 1000.0)[:1024]
    ts = TrialSet(
        data=rng.normal(size=(50, 100, 1024)),
        times=times,
        sfreq=1000.0,
        ch_names=[f"c{i}" for i in range(100)],
    )
    start = time.perf_counter()
    run_test(
        ts, window=(0.009, 0.5), baseline=(-0.5, -0.05), n_permutations=1000, by=(), random_state=0
    )
    assert time.perf_counter() - start <= MAX_SECONDS
