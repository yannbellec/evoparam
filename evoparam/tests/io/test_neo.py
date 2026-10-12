# SPDX-License-Identifier: Apache-2.0
"""Neo reader (REQ-319)."""

import numpy as np
import pytest

neo = pytest.importorskip("neo")
pq = pytest.importorskip("quantities")

from evoparam.io import read_neo  # noqa: E402


def _block():
    block = neo.Block()
    for run in range(2):
        seg = neo.Segment()
        data = np.zeros((2000, 3))
        data[520:540, 0] = 0.05  # 0.05 mV = 50 µV
        signal = neo.AnalogSignal(
            data,
            units="mV",
            sampling_rate=1000 * pq.Hz,
            t_start=0 * pq.s,
            array_annotations={"channel_names": ["a", "b", "c"]},
        )
        seg.analogsignals.append(signal)
        seg.events.append(
            neo.Event(times=[0.5, 1.0, 1.5] * pq.s, labels=["left", "right", "left"], name="stim")
        )
        block.segments.append(seg)
    return block


def test_req319_block_with_two_segments():

    ts = read_neo(_block(), tmin=-0.1, tmax=0.2)
    assert ts.data.shape == (6, 3, 300)
    assert ts.ch_names == ["a", "b", "c"]
    window = (ts.times >= 0.021) & (ts.times <= 0.038)
    assert np.allclose(ts.data[0, 0, window], 50.0)  # mV -> µV, x 1000
    assert list(ts.metadata["condition"]) == ["left", "right", "left"] * 2
    assert list(ts.metadata["run"]) == ["0", "0", "0", "1", "1", "1"]
    one = read_neo(_block(), segment=1, events="stim", tmin=-0.1, tmax=0.2)
    assert one.n_trials == 3 and set(one.metadata["run"]) == {"1"}
