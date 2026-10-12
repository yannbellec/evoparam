# SPDX-License-Identifier: Apache-2.0
"""Remote reads by HTTP byte ranges (REQ-021), against local HTTP servers.

The servers run on 127.0.0.1: one honours Range headers, one ignores them (fallback to a
whole-file download), one fails before answering (retries). The test of the real
OpenNeuro dataset runs only when EVOPARAM_NETWORK_TESTS=1.
"""

import hashlib
import os
import time

import numpy as np
import pytest

import evoparam.io._fetch as fetch_module
from evoparam.errors import ReaderError
from evoparam.io import RemoteFile, read_bids, read_brainvision
from evoparam.tests.io.helpers import http_server

pd = pytest.importorskip("pandas")

###################################################################################################
###################################################################################################

DATA_FILE = "sub-01/ieeg/sub-01_task-spes_run-01_ieeg.eeg"


@pytest.fixture
def delays(monkeypatch):
    """Record the backoff delays instead of sleeping."""

    recorded = []
    monkeypatch.setattr(fetch_module, "_sleep", recorded.append)
    return recorded


def test_remote_read_matches_local_read(bids_root):

    local = read_bids(bids_root, "01", task="spes", run="01")
    os.path.getsize(os.path.join(bids_root, *DATA_FILE.split("/")))
    with http_server(bids_root) as (base, stats):
        remote = read_bids(base, "01", task="spes", run="01", remote=True)

    assert np.array_equal(remote.data, local.data)
    pd.testing.assert_frame_equal(remote.metadata, local.metadata)
    pd.testing.assert_frame_equal(remote.ch_info, local.ch_info)

    # Only byte ranges of the data file were transferred, and they are recorded
    source = remote.provenance.source
    data_requests = [r for r in stats.requests if r[1].endswith(".eeg") and r[0] == "GET"]
    assert data_requests
    assert all(r[2] is not None and r[3] == 206 for r in data_requests)
    assert source["bytes_read"] == stats.bytes_sent
    assert source["path"] == base + DATA_FILE
    assert source["full_download"] is False
    covered = sum(stop - start for start, stop in source["ranges"])
    assert covered == source["data_bytes_read"]

    # sha256 of the bytes read: one line per chunk, recomputed from the file
    with open(os.path.join(bids_root, *DATA_FILE.split("/")), "rb") as file:
        content = file.read()
    chunk = 1 << 16
    lines = []
    for start, stop in source["ranges"]:
        for lo in range(start, stop, chunk):
            hi = min(lo + chunk, stop)
            lines.append(f"{lo}-{hi}:{hashlib.sha256(content[lo:hi]).hexdigest()}")
    assert source["sha256"] == hashlib.sha256("\n".join(lines).encode("ascii")).hexdigest()


def test_remote_read_transfers_only_the_epochs(bids_root):

    size = os.path.getsize(os.path.join(bids_root, *DATA_FILE.split("/")))
    with http_server(bids_root) as (base, stats):
        trials = read_bids(
            base,
            "01",
            task="spes",
            run="01",
            remote=True,
            select="onset == 23.0",
            epoch=(-0.1, 0.1),
        )
    assert trials.n_trials == 1
    source = trials.provenance.source
    assert source["data_bytes_read"] <= 2 * (1 << 16) < size / 2
    assert source["bytes_read"] < size / 2


def test_remote_with_base_address_and_session(bids_root):

    parent, name = os.path.split(bids_root)
    with http_server(parent) as (base, _):
        trials = read_bids(name, "sub-02", session="1", task="spes", run="01", remote=base)
    assert trials.n_trials == 15
    assert set(trials.metadata["stim_polarity"]) == {"biphasic_cathodic_first"}


def test_server_ignoring_range_falls_back_to_full_download(bids_root, tmp_path):

    local = read_bids(bids_root, "01", task="spes", run="01")
    size = os.path.getsize(os.path.join(bids_root, *DATA_FILE.split("/")))
    with http_server(bids_root, honour_range=False) as (base, stats):
        with pytest.warns(UserWarning, match=r"ignored the byte-range request.*\(0\.7 MB\)"):
            remote = read_bids(
                base, "01", task="spes", run="01", remote=True, cache_dir=str(tmp_path / "cache")
            )
    assert np.array_equal(remote.data, local.data)
    source = remote.provenance.source
    assert source["full_download"] is True
    assert source["ranges"] == [[0, size]]
    assert source["sha256"] == local.provenance.source["sha256"]  # whole-file hash
    assert len(os.listdir(tmp_path / "cache")) == 1
    # The whole file was fetched once
    assert sum(1 for r in stats.requests if r[1].endswith(".eeg") and r[0] == "GET") == 1


def test_retries_with_backoff(bids_root, delays):

    with http_server(bids_root, failures=2) as (base, _):
        data, _, _, _ = read_brainvision(
            base + "sub-01/ieeg/sub-01_task-spes_run-01_ieeg.vhdr", tmin=1.0, tmax=1.1
        )
    assert data.shape == (3, 100)
    assert delays == [2.0, 4.0]


def test_gives_up_after_four_retries(bids_root, delays):

    with http_server(bids_root, failures=100) as (base, stats):
        with pytest.raises(ReaderError, match="HTTP 503"):
            read_bids(base, "01", task="spes", run="01", remote=True)
    assert delays == [2.0, 4.0, 8.0, 16.0]
    assert len(stats.requests) == 5


def test_missing_file_and_offline_mode(bids_root, delays, monkeypatch):

    with http_server(bids_root) as (base, stats):
        with pytest.raises(ReaderError, match="HTTP 404"):
            read_bids(base, "01", task="other", run="01", remote=True)
        assert delays == []  # a 404 is not retried
        with pytest.raises(ReaderError, match="task="):
            read_bids(base, "01", remote=True)

        monkeypatch.setenv("EVOPARAM_OFFLINE", "1")
        n_requests = len(stats.requests)
        with pytest.raises(ReaderError, match="EVOPARAM_OFFLINE"):
            read_bids(base, "01", task="spes", run="01", remote=True)
        assert len(stats.requests) == n_requests


def test_remote_file_chunk_cache(tmp_path):

    content = bytes(range(256)) * 400  # 102 400 bytes
    (tmp_path / "blob.bin").write_bytes(content)
    with http_server(tmp_path) as (base, stats):
        with RemoteFile(base + "blob.bin", chunk_size=4096, max_chunks=4) as remote:
            assert remote.size == len(content)
            assert remote.read(5000, 3000) == content[5000:8000]  # chunk 1
            n_requests = len(stats.requests)
            assert remote.read(6000, 100) == content[6000:6100]  # cached
            assert len(stats.requests) == n_requests
            assert remote.read(102300, 500) == content[102300:]  # end of file
            assert remote.read(200000, 10) == b""
            assert remote.read(0, 20000) == content[:20000]  # more than the cache
            assert remote.ranges == [[0, 20480], [98304, 102400]]
            assert remote.bytes_read == 6 * 4096  # chunks 0-4 and 24
            digest = remote.sha256()

        # Same chunks read in another order and grouping: same sha256
        with RemoteFile(base + "blob.bin", chunk_size=4096) as remote:
            for offset in (98304, 0, 16384, 4096):
                remote.read(offset, 4096)
            remote.read(8192, 8192)
            assert remote.ranges == [[0, 20480], [98304, 102400]]
            assert remote.sha256() == digest

    with pytest.raises(ReaderError):
        RemoteFile("file:///etc/passwd")


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
    reason="set EVOPARAM_NETWORK_TESTS=1 to read OpenNeuro over the internet",
)
def test_openneuro_ds004080_ten_trials():
    """REQ-021: ten trials (−0.5, 1.0) s of ds004080 sub-ccepAgeUMCU01 run-021448, < 60 MB."""

    start = time.perf_counter()
    trials = read_bids(
        "ds004080",
        "sub-ccepAgeUMCU01",
        session="1",
        task="SPESclin",
        run="021448",
        remote=True,
        select={"stim_site": "PT01-PT02"},
        epoch=(-0.5, 1.0),
    )
    elapsed = time.perf_counter() - start
    source = trials.provenance.source
    print(
        "\nds004080 remote read: {} trials x {} channels, {:.1f} MB transferred "
        "({:.1f} MB of data ranges), {:.1f} s".format(
            trials.n_trials,
            trials.n_channels,
            source["bytes_read"] / 1e6,
            source["data_bytes_read"] / 1e6,
            elapsed,
        )
    )
    assert trials.n_trials == 10
    assert source["bytes_read"] < 60e6
    assert source["doi"] == "10.18112/openneuro.ds004080.v1.2.4"
    assert set(trials.metadata["stim_site"]) == {"PT01-PT02"}
    assert np.allclose(trials.metadata["stim_current_mA"], 7.0)
    assert trials.times[0] == -0.5
    assert trials.times[-1] < 1.0
