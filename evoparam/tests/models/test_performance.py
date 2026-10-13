# SPDX-License-Identifier: Apache-2.0
"""Speed of the engine (REQ-115); runs only with EVOPARAM_PERF_TESTS=1.

REQ-115: one waveform of 2 048 samples with ``max_components = 6``, the default shapes and
``uncertainty="jacobian"`` fits in <= 50 ms (median) on the reference machine; ``fit_group``
on 100 channels x 2 048 samples with ``n_jobs=1`` completes in <= 10 s.

Both are met with the compiled fit of the ``fast`` extra (numba, ``models/_fast.py``); without
numba the engine takes the NumPy path, which does not meet them (CHANGELOG, "Engine speed,
part 2"). The time of one waveform is measured as §9.1 measures P1: 20 repetitions after one
warm-up call (the first fit of a process loads the compiled code, about 0.7 s).
"""

import importlib.util
import os
import time
from typing import Any

import numpy as np
import pytest

from evoparam.data import Evoked
from evoparam.models import ResponseModel, fit_response

pytestmark = pytest.mark.skipif(
    os.environ.get("EVOPARAM_PERF_TESTS") != "1",
    reason="set EVOPARAM_PERF_TESTS=1 to time the engine",
)

SFREQ = 2048.0
TIMES = (np.arange(2048) - 1022) / SFREQ  # 2 048 samples, to 0.5005 s


def _waveform(rng: Any) -> Any:
    return (
        -100 * np.exp(-0.5 * ((TIMES - 0.025) / 0.006) ** 2)
        - 60 * np.exp(-0.5 * ((TIMES - 0.15) / 0.04) ** 2)
        + rng.normal(0, 5, len(TIMES))
    )


WITHOUT_NUMBA = importlib.util.find_spec("numba") is None


@pytest.mark.xfail(
    WITHOUT_NUMBA,
    strict=False,
    reason="REQ-115 is met with the fast extra only: without numba, median 0.29 to 0.30 s per "
    "waveform (one thread; CHANGELOG, 'Engine speed (REQ-115, REQ-290 P1)')",
)
def test_req115_one_waveform_under_50_ms() -> None:

    rng = np.random.default_rng(0)
    waveforms = [_waveform(rng) for _ in range(5)]

    def fit(y: Any) -> None:
        fit_response(TIMES, y, window=(0.009, 0.5), baseline=(-0.5, -0.02), preset="ccep")

    fit(waveforms[0])  # warm-up (§9.1)
    durations = []
    for _ in range(4):
        for y in waveforms:
            start = time.perf_counter()
            fit(y)
            durations.append(time.perf_counter() - start)
    assert np.median(durations) <= 0.05, np.median(durations)


@pytest.mark.xfail(
    WITHOUT_NUMBA,
    strict=False,
    reason="REQ-115 (fit_group) is met with the fast extra only: without numba, 25 s for 100 "
    "channels (one thread; CHANGELOG, 'Engine speed (REQ-115, REQ-290 P1)')",
)
def test_req115_fit_group_100_channels_under_10_s() -> None:

    rng = np.random.default_rng(1)
    data = np.array(
        [_waveform(rng) * rng.uniform(0, 1.5) + rng.normal(0, 5, len(TIMES)) for _ in range(100)]
    )
    evoked = Evoked(data=data, times=TIMES, sfreq=SFREQ, ch_names=[f"c{i}" for i in range(100)])
    model = ResponseModel("ccep", n_jobs=1)
    start = time.perf_counter()
    group = model.fit_group(evoked)
    elapsed = time.perf_counter() - start
    assert len(group.fits) == 100
    assert elapsed <= 10.0, elapsed
