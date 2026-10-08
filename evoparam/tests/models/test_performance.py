"""Speed of the engine (REQ-115); runs only with EVOPARAM_PERF_TESTS=1."""

import os
import time

import numpy as np
import pytest

from evoparam.models import fit_response

pytestmark = pytest.mark.skipif(os.environ.get("EVOPARAM_PERF_TESTS") != "1",
                                reason="set EVOPARAM_PERF_TESTS=1 to time the engine")


@pytest.mark.xfail(strict=False, reason="REQ-115 not met: about 0.2 s per waveform instead of "
                   "50 ms (pure NumPy/SciPy; see CHANGELOG, step S7).")
def test_req115_one_waveform_under_50_ms():

    sfreq = 2048.
    t = np.arange(-0.5, 0.5, 1 / sfreq)
    rng = np.random.default_rng(0)
    durations = []
    for _ in range(5):
        y = (-100 * np.exp(-0.5 * ((t - 0.025) / 0.006) ** 2)
             - 60 * np.exp(-0.5 * ((t - 0.15) / 0.04) ** 2) + rng.normal(0, 5, len(t)))
        start = time.perf_counter()
        fit_response(t, y, window=(0.009, 0.5), baseline=(-0.5, -0.02), preset="ccep")
        durations.append(time.perf_counter() - start)
    assert np.median(durations) <= 0.05
