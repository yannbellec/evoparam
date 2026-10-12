# SPDX-License-Identifier: Apache-2.0
"""REQ-317: reading epochs of a 1 GB ElectricalSeries keeps the memory low.

A synthetic file of 1 GB (64 channels x 4,194,304 samples of float32 at 4096 Hz, fixed seed)
is generated once, chunk by chunk, under ``$EVOPARAM_DATA/nwb-memory`` (default
``~/.cache/evoparam``); epochs covering 10 % of the samples are read in a fresh Python
process, whose peak resident memory may grow by at most 0.25 GB (from a baseline taken
after a one-epoch warm-up read, so that the modules loaded on first use are not counted). Runs with
``EVOPARAM_SLOW_TESTS=1`` (the file takes about a minute to write the first time).
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

pynwb = pytest.importorskip("pynwb")

SLOW = os.environ.get("EVOPARAM_SLOW_TESTS") == "1"
N_CH = 64
N_TIMES = 4 * 1024 * 1024
SFREQ = 4096.0
CHUNK = 65536
LIMIT_GB = 0.25


def _path():
    root = os.environ.get("EVOPARAM_DATA", os.path.expanduser("~/.cache/evoparam"))
    folder = os.path.join(root, "nwb-memory")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "synthetic_1gb_seed0.nwb")


def _make(path):
    from datetime import datetime, timezone

    from hdmf.data_utils import GenericDataChunkIterator
    from pynwb import NWBFile, NWBHDF5IO
    from pynwb.ecephys import ElectricalSeries

    class Synthetic(GenericDataChunkIterator):
        """Gaussian noise (20 µV), block k drawn from the seed (0, k): any block can be
        regenerated alone, and the whole file never sits in memory."""

        def _get_data(self, selection):
            rows = selection[0]
            block = rows.start // CHUNK
            data = np.random.default_rng((0, block)).normal(0, 20e-6, (CHUNK, N_CH))
            return data.astype(np.float32)[: rows.stop - rows.start][:, selection[1]]

        def _get_maxshape(self):
            return (N_TIMES, N_CH)

        def _get_dtype(self):
            return np.dtype("float32")

    nwb = NWBFile(
        session_description="1 GB synthetic",
        identifier="evoparam-1gb",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    device = nwb.create_device(name="probe")
    group = nwb.create_electrode_group(name="shank", description="s", location="x", device=device)
    for _ in range(N_CH):
        nwb.add_electrode(group=group, location="x")
    region = nwb.create_electrode_table_region(list(range(N_CH)), "all")
    data = Synthetic(
        buffer_shape=(CHUNK, N_CH), chunk_shape=(CHUNK // 16, N_CH), display_progress=False
    )
    nwb.add_acquisition(
        ElectricalSeries(name="lfp", data=data, electrodes=region, rate=SFREQ, starting_time=0.0)
    )
    tmp = path + ".part"
    with NWBHDF5IO(tmp, "w") as io:
        io.write(nwb)
    os.replace(tmp, path)


SCRIPT = textwrap.dedent("""
    import json, resource, sys
    import numpy as np
    import evoparam.io as io_module
    def peak():
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if sys.platform == "darwin" else value * 1024
    # warm-up: one epoch, so that the modules imported on first use (pandas, pynwb, the
    # NWB namespaces) are in memory before the baseline; the growth is that of the read
    io_module.read_nwb(sys.argv[1], events=np.array([1.0]), tmin=0.0, tmax=0.25)
    base = peak()
    n_times, sfreq = int(sys.argv[2]), float(sys.argv[3])
    width = 0.25                                         # s per epoch
    n_events = int(0.10 * n_times / (width * sfreq))     # 10 % of the samples
    onsets = np.linspace(1.0, n_times / sfreq - 1.0, n_events)
    ts = io_module.read_nwb(sys.argv[1], events=onsets, tmin=0.0, tmax=width)
    print(json.dumps({"base": base, "peak": peak(), "shape": list(ts.data.shape),
                      "epoch_bytes": int(ts.data.nbytes)}))
""")


@pytest.mark.skipif(not SLOW, reason="EVOPARAM_SLOW_TESTS=1 (1 GB file)")
def test_req317_peak_memory_reading_10_percent_of_1gb():

    path = _path()
    if not os.path.exists(path):
        _make(path)
    assert os.path.getsize(path) >= N_TIMES * N_CH * 4
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT, path, str(N_TIMES), str(SFREQ)],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    import json

    result = json.loads(proc.stdout.strip().splitlines()[-1])
    grown = (result["peak"] - result["base"]) / 1e9
    assert result["shape"][1] == N_CH
    covered = result["shape"][0] * result["shape"][2] / N_TIMES
    assert 0.09 <= covered <= 0.11
    # the epochs themselves (float64, µV) are part of the growth
    assert grown < LIMIT_GB, f"peak RSS grew by {grown:.3f} GB"
