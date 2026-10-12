# SPDX-License-Identifier: Apache-2.0
"""Speed of every artifact-removal method (REQ-068); runs only with EVOPARAM_PERF_TESTS=1."""

import os
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("EVOPARAM_PERF_TESTS") != "1",
    reason="set EVOPARAM_PERF_TESTS=1 to time remove_artifact",
)


def test_req068_runtime():

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "benchmarks"))
    from bench_artifact import make_data, run

    for label, _, elapsed, limit in run(make_data()):
        assert elapsed <= limit, label
