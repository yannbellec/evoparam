# SPDX-License-Identifier: Apache-2.0
"""LSL adapters (REQ-325): a local outlet with a synthetic signal, read through LSLInlet into
a StreamFitter; the outlet of pulse estimates carries the intended-use metadata.

Skipped when pylsl or the liblsl library is missing (extra ``stream``).
"""

import json
import threading
import time

import numpy as np
import pytest

from evoparam.stream import FIELDS, INTENDED_USE, LSLInlet, LSLOutlet, StreamFitter

from .helpers import N1, SFREQ, channel_fits, gauss, simulate

try:
    import pylsl

    pylsl.local_clock()
except Exception as exc:  # noqa: BLE001  (missing package or missing library)
    pylsl = None
    REASON = f"pylsl unavailable: {exc}"

pytestmark = [
    pytest.mark.lsl,
    pytest.mark.skipif(pylsl is None, reason=REASON if pylsl is None else ""),
]

N_CHANNELS = 4
N_PULSES = 10


def _signal(seed=0):
    """12 s of 4 channels, a pulse (artifact spike and N1) every second from 1.5 s."""

    rng = np.random.default_rng(seed)
    n = int(12 * SFREQ)
    data = rng.normal(0, 10, (n, N_CHANNELS))
    onsets = (np.arange(N_PULSES + 1) + 1.5) * SFREQ
    times = np.arange(n) / SFREQ
    for onset in onsets.astype(int):
        data[onset : onset + 3] += np.array([3000.0, -2000.0, 500.0])[:, None]
        data[:, :2] += (N1[0] * gauss(times - onset / SFREQ, N1[1], N1[2]))[:, None]
    return data, onsets.astype(int)


def test_req325_inlet_fitter_and_outlet_metadata():

    data, _ = _signal()
    info = pylsl.StreamInfo(
        "evoparam-test-signal", "EEG", N_CHANNELS, SFREQ, "float32", "evoparam-test-signal"
    )
    channels = info.desc().append_child("channels")
    for i in range(N_CHANNELS):
        channels.append_child("channel").append_child_value("label", f"c{i}")
    outlet = pylsl.StreamOutlet(info)
    inlet = LSLInlet(name="evoparam-test-signal")
    assert inlet.sfreq == SFREQ
    assert inlet.ch_names == [f"c{i}" for i in range(N_CHANNELS)]

    template = channel_fits(simulate(n_trials=40, n_channels=1, seed=6)).fits[0]
    fitter = StreamFitter(
        template,
        sfreq=inlet.sfreq,
        ch_names=inlet.ch_names,
        artifact_window=(-0.002, 0.008),
        existence_max_pulses=N_PULSES,
    )

    def send():
        start = pylsl.local_clock()
        for first in range(0, len(data), 256):  # 8 times faster than real time
            chunk = data[first : first + 256]
            stamps = start + (first + np.arange(len(chunk))) / SFREQ
            outlet.push_chunk(chunk.tolist(), stamps.tolist())
            time.sleep(256 / SFREQ / 8)

    sender = threading.Thread(target=send)
    sender.start()
    estimates = inlet.feed(fitter, max_pulses=N_PULSES, duration=30.0)
    sender.join()
    inlet.close()
    pulse_times = sorted({e.pulse_time for e in estimates})
    assert len(pulse_times) == N_PULSES
    # onsets on the stream clock: one second apart, as sent
    assert np.allclose(np.diff(pulse_times), 1.0, atol=1.5 / SFREQ)
    n1 = [e for e in estimates if e.channel in ("c0", "c1")]
    assert np.allclose([e.latency for e in n1], N1[1], atol=0.002)
    assert np.allclose([e.amplitude for e in n1], N1[0], rtol=0.25)

    # Outlet of the estimates: metadata and rows
    out = LSLOutlet(fitter, name="evoparam-test-pulses")
    found = pylsl.resolve_byprop("name", "evoparam-test-pulses", 1, 5.0)
    assert found
    reader = pylsl.StreamInlet(found[0])
    desc = reader.info(5.0).desc()
    assert desc.child_value("intended_use") == INTENDED_USE
    meta = json.loads(desc.child_value("evoparam"))
    assert meta["intended_use"] == INTENDED_USE
    assert meta["fields"] == list(FIELDS)
    reader.open_stream(5.0)
    time.sleep(0.5)
    out.push(estimates)
    rows, stamps = [], []
    deadline = time.monotonic() + 10
    while len(rows) < len(estimates) and time.monotonic() < deadline:
        chunk, ts = reader.pull_chunk(timeout=0.5)
        rows += chunk
        stamps += ts
    assert len(rows) == len(estimates)
    assert np.allclose(stamps, [e.pulse_time for e in estimates])
    assert np.allclose(
        [r[FIELDS.index("amplitude")] for r in rows], [e.amplitude for e in estimates], rtol=1e-6
    )
