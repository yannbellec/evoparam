# SPDX-License-Identifier: Apache-2.0
"""Speed of the time-frequency parameterisation (REQ-218); runs only with
EVOPARAM_PERF_TESTS=1."""

import os
import pathlib
import sys

import pytest

# Longest time (s) of the time-frequency benchmark.
MAX_SECONDS = 60

pytestmark = pytest.mark.skipif(
    os.environ.get("EVOPARAM_PERF_TESTS") != "1",
    reason="set EVOPARAM_PERF_TESTS=1 to time the transform",
)


def test_req218_100_channels_50_trials_2048_samples_40_freqs() -> None:

    sys.path.insert(0, str(pathlib.Path(__file__).parents[3] / "benchmarks"))
    from bench_tf import run

    duration, tf = run("morse", "fdr")
    assert len(tf.zones) >= 1
    assert duration <= MAX_SECONDS
