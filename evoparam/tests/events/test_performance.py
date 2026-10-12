# SPDX-License-Identifier: Apache-2.0
"""Speed and memory of pulse detection (REQ-045); runs only with EVOPARAM_PERF_TESTS=1.

The target is for one core of the reference machine (spec §9.1); shared CI runners are
slower, so the check is opt-in.
"""

import os
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("EVOPARAM_PERF_TESTS") != "1",
    reason="set EVOPARAM_PERF_TESTS=1 to time detect_pulses",
)


def test_req045_runtime_and_memory():

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "benchmarks"))
    from bench_events import make_data, measure

    data, sfreq = make_data()
    names = [f"C{i:03d}" for i in range(data.shape[0])]
    for kwargs in ({}, {"ch_names": names}):
        elapsed, peak, events = measure(data, sfreq, **kwargs)
        assert len(events) == 120
        assert elapsed <= 1.0
        assert peak <= 2 * data.nbytes
