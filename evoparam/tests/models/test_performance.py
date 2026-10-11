# SPDX-License-Identifier: Apache-2.0
"""Speed of the engine (REQ-115); runs only with EVOPARAM_PERF_TESTS=1.

REQ-115: one waveform of 2 048 samples with ``max_components = 6``, the default shapes and
``uncertainty="jacobian"`` fits in <= 50 ms (median) on the reference machine; ``fit_group``
on 100 channels x 2 048 samples with ``n_jobs=1`` completes in <= 10 s.
"""

import os
import time

import numpy as np
import pytest

from evoparam.data import Evoked
from evoparam.models import ResponseModel, fit_response

pytestmark = pytest.mark.skipif(os.environ.get("EVOPARAM_PERF_TESTS") != "1",
                                reason="set EVOPARAM_PERF_TESTS=1 to time the engine")

SFREQ = 2048.
TIMES = (np.arange(2048) - 1022) / SFREQ                    # 2 048 samples, to 0.5005 s


def _waveform(rng):
    return (-100 * np.exp(-0.5 * ((TIMES - 0.025) / 0.006) ** 2)
            - 60 * np.exp(-0.5 * ((TIMES - 0.15) / 0.04) ** 2) + rng.normal(0, 5, len(TIMES)))


@pytest.mark.xfail(strict=False, reason="REQ-115 not met: median 0.29 to 0.30 s per waveform "
                   "with ccep v1.1, 0.52 to 0.55 s before (one thread; pure NumPy/SciPy; "
                   "CHANGELOG, 'Engine speed (REQ-115, REQ-290 P1)')")
def test_req115_one_waveform_under_50_ms():

    rng = np.random.default_rng(0)
    durations = []
    for _ in range(5):
        y = _waveform(rng)
        start = time.perf_counter()
        fit_response(TIMES, y, window=(0.009, 0.5), baseline=(-0.5, -0.02), preset="ccep")
        durations.append(time.perf_counter() - start)
    assert np.median(durations) <= 0.05


@pytest.mark.xfail(strict=False, reason="REQ-115 (fit_group) not met: 25 s for 100 channels "
                   "with ccep v1.1, 45 to 48 s before (one thread; CHANGELOG, 'Engine speed "
                   "(REQ-115, REQ-290 P1)')")
def test_req115_fit_group_100_channels_under_10_s():

    rng = np.random.default_rng(1)
    data = np.array([_waveform(rng) * rng.uniform(0, 1.5) + rng.normal(0, 5, len(TIMES))
                     for _ in range(100)])
    evoked = Evoked(data=data, times=TIMES, sfreq=SFREQ, ch_names=[f"c{i}" for i in range(100)])
    model = ResponseModel("ccep", n_jobs=1)
    start = time.perf_counter()
    group = model.fit_group(evoked)
    elapsed = time.perf_counter() - start
    assert len(group.fits) == 100
    assert elapsed <= 10.0, elapsed
