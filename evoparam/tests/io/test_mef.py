# SPDX-License-Identifier: Apache-2.0
"""MEF 3.0 reader on synthetic sessions written by the test writer (helpers.write_mef).

The writer is an independent port of the meflib encoder (byte-by-byte CRC, range encoder of
RED_encode_exec, file layouts); the reader is checked against it for lossless and lossy
blocks, segments, gaps, NaN and infinity codes, units, encryption and corrupted files,
locally and through a local HTTP server.
"""

import copy
import os
import shutil
import subprocess
import sys
import textwrap
from typing import Any

import numpy as np
import pytest

from evoparam.data import TrialSet
from evoparam.errors import DataError, ReaderError
from evoparam.io import mef as mef_module
from evoparam.io import read, read_bids, read_mef, read_mef_header
from evoparam.tests.io.helpers import (
    CH_NAMES,
    MEF_NAN,
    MEF_NEG_INF,
    MEF_POS_INF,
    corrupt_byte,
    http_server,
    make_mef_bids,
    mef_crc,
    mef_round,
    red_encode,
    signal,
    write_mef,
)
from evoparam.tests.io.helpers import SFREQ as BIDS_SFREQ

# Entries 1 and 128 of meflib's CRC table (CRC_KOOPMAN32_KEY).
CRC_ENTRY_1 = 0x9695C4CA
CRC_ENTRY_128 = 0xEB31D82E
# Start time (us) and units factor of the session written, and its blocks per channel.
START_TIME_US = 1_600_000_000_000_000
FACTOR = 0.1
N_BLOCKS = 12
# Largest error of the lossy RED options (2 steps of 0.1 uV, scale factor 4).
MAX_LOSSY_ERROR = 0.25
# NaN samples written (one, then a whole block of 500).
N_NAN = 501
# Blocks of the three channels, read for their CRC.
N_BLOCKS_READ = 36
# HTTP status of a missing file.
NOT_FOUND = 404
# HTTP status of a byte-range answer.
PARTIAL_CONTENT = 206

###################################################################################################
###################################################################################################

SFREQ = 1000.0
NAMES = ["A10", "A2", "B1"]  # natural order on reading: A2, A10, B1
ORDER = [1, 0, 2]


def make_signal(n_times: Any = 6000, seed: Any = 0) -> Any:
    """Random walks (µV) with large jumps (keysamples), a constant run and a NaN sample."""

    rng = np.random.default_rng(seed)
    data = np.cumsum(rng.normal(0, 2, (3, n_times)), axis=1)
    data[1, 1000:1200] += rng.normal(0, 800, 200)
    data[2, 2000:3000] = 12.5  # two constant blocks of 500 samples
    data[0, 4321] = np.nan
    return data


def stored(data: Any, factor: Any = 0.1) -> Any:
    """Values after the round trip: round(x / factor) * factor, NaN kept."""

    out = mef_round(np.nan_to_num(data) / factor) * factor
    return np.where(np.isnan(data), np.nan, out)


@pytest.fixture
def session(tmp_path: Any) -> Any:
    data = make_signal()
    return write_mef(tmp_path, "sub-01_task-spes_run-01_ieeg", data, SFREQ, NAMES), data


def test_crc_matches_the_bytewise_crc() -> None:

    rng = np.random.default_rng(1)
    buf = rng.integers(0, 256, 20000, dtype=np.uint8)
    starts = rng.integers(0, 10000, 40)
    lengths = np.concatenate([[4, 5, 255, 256, 257, 511, 512, 513], rng.integers(4, 9000, 32)])
    got = mef_module.crc32_koopman(buf, starts, lengths)
    expected = [mef_crc(buf[s : s + n].tobytes()) for s, n in zip(starts, lengths, strict=False)]
    assert got.tolist() == expected
    # Table entries of meflib (CRC_KOOPMAN32_KEY)
    assert mef_crc(b"\x00") != mef_crc(b"\x01")
    assert mef_module._crc_tables()["table"][1] == CRC_ENTRY_1
    assert mef_module._crc_tables()["table"][128] == CRC_ENTRY_128


def test_red_blocks_round_trip() -> None:

    rng = np.random.default_rng(2)
    cases = [
        np.cumsum(rng.integers(-100, 100, 3000)),  # differences only
        rng.integers(-(2**30), 2**30, 700),  # keysamples only
        np.full(1500, -7),  # one symbol (stats > 255)
        np.array([5]),  # one sample
        np.array([MEF_NAN, 0, MEF_POS_INF, MEF_NEG_INF, -128, 127, -127, 128]),
        np.concatenate([np.zeros(400), rng.integers(-5000, 5000, 400)]),
    ]
    blocks = [red_encode(case, 1000 * k) for k, case in enumerate(cases)]
    buf = np.frombuffer(b"".join(blocks) + bytes(8), dtype=np.uint8)
    starts = np.cumsum([0] + [len(b) for b in blocks[:-1]])
    rows = buf[starts[:, None] + np.arange(304)]
    headers = np.frombuffer(rows.tobytes(), dtype=mef_module.BLOCK_DTYPE)
    assert all(len(block) % 8 == 0 for block in blocks)
    values, offsets = mef_module.red_decode(buf, starts, headers)
    for case, offset in zip(cases, offsets, strict=False):
        assert np.array_equal(values[offset : offset + case.size], case)
    crc = mef_module.crc32_koopman(buf, starts + 4, headers["block_bytes"].astype(int) - 4)
    assert np.array_equal(crc, headers["crc"])


def test_read_whole_session(session: Any) -> None:

    root, data = session
    header = read_mef_header(root)
    assert header["ch_names"] == [NAMES[i] for i in ORDER]
    assert header["sfreq"] == SFREQ
    assert header["n_samples"] == data.shape[1]
    assert header["start_time"] == START_TIME_US
    assert header["session"] == "sub-01_task-spes_run-01_ieeg"
    channel = header["channels"][0]
    assert channel["units"] == "microvolts"
    assert channel["factor"] == FACTOR
    assert channel["to_uv"] == 1.0
    assert len(channel["blocks"]) == N_BLOCKS

    out, sfreq, names, t0 = read_mef(root)
    assert sfreq == SFREQ
    assert t0 == 0.0
    assert names == header["ch_names"]
    expected = stored(data)[ORDER]
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-9)
    assert np.isnan(out[1, 4321])
    assert np.isnan(out).sum() == 1


def test_window_and_picks(session: Any) -> None:

    root, data = session
    expected = stored(data)
    out, sfreq, names, t0 = read_mef(root, tmin=1.2345, tmax=2.5, picks=["B1", "A10"])
    assert names == ["B1", "A10"]
    assert t0 == pytest.approx(1.235)
    np.testing.assert_allclose(out, expected[[2, 0], 1235:2500], atol=1e-9)
    out, _, names, _ = read_mef(root, tmin=5.9, picks=[0, -1])
    assert names == ["A2", "B1"]
    assert out.shape == (2, 100)
    np.testing.assert_allclose(out, expected[[1, 2], 5900:], atol=1e-9)
    with pytest.raises(ReaderError, match="no channel"):
        read_mef(root, picks=["C1"])
    with pytest.raises(DataError, match="picks"):
        read_mef(root, picks=[7])
    with pytest.raises(ReaderError, match="outside the recording"):
        read_mef(root, tmin=6.0, tmax=7.0)


def test_segments_give_the_same_data(tmp_path: Any) -> None:

    data = make_signal()
    one = write_mef(tmp_path, "one", data, SFREQ, NAMES, block_samples=700)
    three = write_mef(
        tmp_path, "three", data, SFREQ, NAMES, block_samples=700, segment_starts=[1500, 4200]
    )
    header = read_mef_header(three)
    assert [s["number"] for s in header["channels"][0]["segments"]] == [0, 1, 2]
    assert [s["start_sample"] for s in header["channels"][0]["segments"]] == [0, 1500, 4200]
    a, *_ = read_mef(one)
    b, *_ = read_mef(three)
    np.testing.assert_array_equal(a, b)
    c, *_ = read_mef(three, tmin=1.4, tmax=4.3)
    np.testing.assert_array_equal(c, a[:, 1400:4300])


def test_gaps_are_nan(tmp_path: Any) -> None:
    """A gap is placed by the block start times: flagged discontinuities, or unflagged
    jumps of more than 1 s (shorter unflagged jumps are time-stamp jitter)."""

    data = make_signal()
    root = write_mef(tmp_path, "gap", data, SFREQ, NAMES, gaps={3000: 0.25, 5000: 0.0105})
    out, *_ = read_mef(root)
    assert out.shape == (3, 6000 + 250 + 10)  # 10.5 ms = 10.5 samples
    expected = stored(data)[ORDER]
    np.testing.assert_allclose(out[:, :3000], expected[:, :3000], atol=1e-9)
    assert np.isnan(out[:, 3000:3250]).all()
    np.testing.assert_allclose(out[:, 3250:5250], expected[:, 3000:5000], atol=1e-9)
    assert np.isnan(out[:, 5250:5260]).all()
    np.testing.assert_allclose(out[:, 5260:], expected[:, 5000:], atol=1e-9)

    unflagged = write_mef(
        tmp_path, "unflagged", data, SFREQ, NAMES, gaps={2000: 1.5, 4000: 0.25}, flag_gaps=False
    )
    other, *_ = read_mef(unflagged)
    assert other.shape == (3, 7500)
    assert np.isnan(other[:, 2000:3500]).all()
    assert np.isnan(other).sum() == 3 * 1500 + 1
    np.testing.assert_allclose(other[:, 3500:], expected[:, 2000:], atol=1e-9)

    # read() marks the gap in masks, for the continuous recording and for epochs
    trials = read(root)
    assert isinstance(trials, TrialSet)
    assert trials.masks is not None
    assert np.array_equal(trials.masks[0], np.isnan(out))
    epochs = read(root, onsets=[2.0, 3.1], epoch=(-0.2, 0.3))
    assert epochs.data.shape == (2, 3, 500)
    # The second epoch, from grid sample 2900 to 3400, holds the gap (3000 to 3250)
    assert not epochs.masks[0].any()
    assert epochs.masks[1, :, 100:350].all()
    assert not epochs.masks[1, :, :100].any()
    assert not epochs.masks[1, :, 350:].any()
    assert np.isnan(epochs.data[1, :, 100:350]).all()
    np.testing.assert_allclose(epochs.data[0], out[:, 1800:2300], atol=1e-12)


def test_lossy_blocks(tmp_path: Any) -> None:
    """Scale factor and detrending, restored as meflib's RED_unscale and RED_retrend."""

    data = make_signal()[:2]
    data[0, 4321] = 0.0
    root = write_mef(
        tmp_path,
        "lossy",
        data,
        SFREQ,
        ["X", "Y"],
        factor=0.1,
        block_samples=600,
        scale_factor=4.0,
        detrend=True,
    )
    out, *_ = read_mef(root)
    expected = np.empty_like(data)
    for ch in range(2):
        x = mef_round(data[ch] / 0.1)
        for first in range(0, data.shape[1], 600):
            seg = x[first : first + 600]
            c = np.arange(1, seg.size + 1, dtype=float)
            n = seg.size
            sx, sxx = n * (n + 1) / 2.0, n * (n + 1) * (2 * n + 1) / 6.0
            sy, sxy = float(np.sum(seg)), float(np.sum(seg * c))
            slope = float(np.float32(((sx * sy / n) - sxy) / ((sx * sx / n) - sxx)))
            intercept = float(np.float32(sy / n - slope * sx / n))
            scaled = mef_round(mef_round(seg - slope * c - intercept) / 4.0)
            back = mef_round(mef_round(scaled * 4.0) + slope * c + intercept)
            expected[ch, first : first + 600] = back * 0.1
    np.testing.assert_allclose(out, expected, atol=1e-9)
    assert 0 < np.max(np.abs(out - data)) < MAX_LOSSY_ERROR  # lossy, within 2 steps


def test_nan_and_infinity_codes(tmp_path: Any) -> None:

    data = np.zeros((1, 1000))
    data[0, 10] = np.nan
    data[0, 20] = np.inf
    data[0, 30] = -np.inf
    data[0, 500:] = np.nan  # a whole NaN block
    root = write_mef(tmp_path, "codes", data, SFREQ, ["X"], block_samples=500)
    out, *_ = read_mef(root)
    assert np.isnan(out[0, 10])
    assert out[0, 20] == np.inf
    assert out[0, 30] == -np.inf
    assert np.isnan(out[0, 500:]).all()
    assert np.isnan(out).sum() == N_NAN


def test_units_conversion(tmp_path: Any) -> None:

    data = make_signal()[:1]
    root = write_mef(tmp_path, "mv", data, SFREQ, ["X"], units="millivolts", factor=1e-4)
    out, *_ = read_mef(root)
    expected = mef_round(np.nan_to_num(data[0]) / 1e3 / 1e-4) * 1e-4 * 1e3
    np.testing.assert_allclose(np.nan_to_num(out[0]), expected, atol=1e-9)

    root = write_mef(tmp_path, "counts", data, SFREQ, ["X"], units="counts", factor=1.0)
    with pytest.warns(UserWarning, match="not a voltage"):
        out, *_ = read_mef(root)
    np.testing.assert_allclose(np.nan_to_num(out[0]), mef_round(np.nan_to_num(data[0])))


def test_encrypted_sessions_raise(tmp_path: Any) -> None:

    data = make_signal()[:1]
    root = write_mef(tmp_path, "metadata", data, SFREQ, ["X"], encrypt="metadata")
    with pytest.raises(ReaderError, match=r"encrypted \(level 1\).*unencrypted MEF 3.0 only"):
        read_mef(root)
    root = write_mef(tmp_path, "blocks", data, SFREQ, ["X"], encrypt="data")
    with pytest.raises(ReaderError, match="data \\(RED blocks\\) is encrypted"):
        read_mef(root)


def test_crc_errors(session: Any) -> None:

    root, data = session
    reference, *_ = read_mef(root)
    seg = os.path.join(root, "A2.timd", "A2-000000.segd", "A2-000000")
    # A byte of the discretionary region of the first block: the decoding is unchanged
    corrupt_byte(seg + ".tdat", 1024 + 10)
    with pytest.raises(ReaderError, match="block 0 of channel A2 has a wrong CRC"):
        read_mef(root)
    out, *_ = read_mef(root, check_crc=False)
    np.testing.assert_array_equal(out, reference)
    corrupt_byte(seg + ".tmet", 5000)
    with pytest.raises(ReaderError, match="CRC of the file body"):
        read_mef(root)
    other = os.path.join(root, "B1.timd", "B1-000000.segd", "B1-000000")
    corrupt_byte(other + ".tidx", 100)
    with pytest.raises(ReaderError, match="B1-000000.tidx: the CRC of the universal header"):
        read_mef(root, picks=["B1"])


def test_bad_files(tmp_path: Any, session: Any) -> None:

    root, _ = session
    with pytest.raises(ReaderError, match="not found"):
        read_mef(tmp_path / "missing.mefd")
    os.makedirs(tmp_path / "empty.mefd")
    with pytest.raises(ReaderError, match="no time-series channel"):
        read_mef(tmp_path / "empty.mefd")
    seg = os.path.join(root, "B1.timd", "B1-000000.segd", "B1-000000")
    with open(seg + ".tmet", "r+b") as file:
        file.seek(15)
        file.write(b"\x00")  # big endian
    with pytest.raises(ReaderError, match="big-endian"):
        read_mef(root, check_crc=False)
    os.replace(seg + ".tidx", seg + ".tmet")
    with pytest.raises(ReaderError, match="not a MEF 3.0 tmet file"):
        read_mef(root)


def test_constant_blocks_are_not_decoded(tmp_path: Any) -> None:
    """Censored data (zeros, as in ds004977) are filled from the index extrema."""

    data = make_signal()
    data[:, :4000] = 0.0
    root = write_mef(tmp_path, "censored", data, SFREQ, NAMES)
    out, *_ = read_mef(root)
    np.testing.assert_allclose(out, stored(data)[ORDER], atol=1e-9)
    # 12 blocks of 500 samples per channel, the first 8 constant: read for their CRC only,
    # not read at all without CRC checks
    _, _, _, _, checked = mef_module._read_with_source(root)
    _, _, _, _, unchecked = mef_module._read_with_source(root, check_crc=False)
    assert checked["blocks_read"] == N_BLOCKS_READ
    assert unchecked["blocks_read"] == N_BLOCKS_READ - 3 * 8
    out, *_ = read_mef(root, check_crc=False)
    np.testing.assert_allclose(out, stored(data)[ORDER], atol=1e-9)


def test_remote_matches_local(tmp_path: Any) -> None:

    data = make_signal(20000)
    root = write_mef(tmp_path, "sub-01_run-01_ieeg", data, SFREQ, NAMES, block_samples=1000)
    with open(tmp_path / "sub-01_run-01_channels.tsv", "w", encoding="utf-8") as file:
        file.write("name\ttype\n" + "".join(f"{name}\tSEEG\n" for name in ["B1", "A10", "A2"]))
    local, sfreq, names, t0, source = mef_module._read_with_source(root, 3.2, 5.0)
    assert names == ["B1", "A10", "A2"]  # channels.tsv order
    with http_server(tmp_path) as (base, stats):
        url = base + "sub-01_run-01_ieeg.mefd"
        remote, _, rnames, rt0, rsource = mef_module._read_with_source(url, 3.2, 5.0)
        data_requests = [r for r in stats.requests if r[1].endswith(".tdat")]
        sent = stats.bytes_sent
    np.testing.assert_array_equal(remote, local)
    assert rnames == names
    assert rt0 == t0
    assert rsource["sha256"] == source["sha256"]
    assert rsource["remote"] is True
    assert rsource["bytes_read"] == sent
    assert data_requests
    assert all(r[2] is not None and r[3] == PARTIAL_CONTENT for r in data_requests)
    sizes = sum(
        os.path.getsize(os.path.join(dirpath, f))
        for dirpath, _, files in os.walk(root)
        for f in files
        if f.endswith(".tdat")
    )
    assert rsource["data_bytes_read"] < sizes / 3  # 3 of 20 blocks per channel

    # Without channels.tsv and without listing, the channel names must be given
    os.remove(tmp_path / "sub-01_run-01_channels.tsv")
    with http_server(tmp_path) as (base, stats):
        with pytest.raises(ReaderError, match="cannot be listed"):
            read_mef(base + "sub-01_run-01_ieeg.mefd")
        out, _, names, _ = read_mef(base + "sub-01_run-01_ieeg.mefd", picks=["A2"], tmax=1.0)
    np.testing.assert_allclose(out, stored(data)[[1], :1000], atol=1e-9)


def test_remote_s3_listing(tmp_path: Any, monkeypatch: Any) -> None:
    """With the S3 listing (OpenNeuro), names and segments come from one request."""

    data = make_signal()
    write_mef(tmp_path, "ses", data, SFREQ, NAMES, segment_starts=[3000])
    local, *_ = read_mef(tmp_path / "ses.mefd")
    with http_server(tmp_path) as (base, stats):
        monkeypatch.setattr(mef_module, "_s3_listing", lambda _url: (base.rstrip("/"), "ses.mefd"))
        remote, _, names, _ = read_mef(base + "ses.mefd")
        requests = list(stats.requests)
    np.testing.assert_array_equal(remote, local)
    assert names == ["A2", "A10", "B1"]
    assert sum("list-type=2" in r[1] for r in requests) == 1
    assert all(
        r[3] in (200, 206)
        for r in requests
        if "list-type" not in r[1] and not r[1].endswith("_channels.tsv")
    )


def test_read_dispatch(session: Any) -> None:

    root, data = session
    trials = read(root, tmin=1.0, tmax=2.0)
    assert isinstance(trials, TrialSet)
    assert trials.units == "uV"
    assert trials.masks is None
    np.testing.assert_allclose(trials.data[0], stored(data)[ORDER][:, 1000:2000], atol=1e-9)
    np.testing.assert_allclose(trials.times, 1.0 + np.arange(1000) / SFREQ)
    source = trials.provenance.source
    assert source["format"] == "mef"
    assert source["reader"] == "evoparam.io.read"
    assert source["units"] == ["microvolts"]
    assert source["units_conversion_factor"] == [0.1]
    assert trials.provenance.steps[-1]["settings"]["format"] == "mef"
    epochs = read(root, onsets=[1.0, 2.5], epoch=(-0.1, 0.2), picks=["B1"])
    np.testing.assert_allclose(
        epochs.data[:, 0], stored(data)[2, [range(900, 1200), range(2400, 2700)]], atol=1e-9
    )
    same = read(root, tmin=1.0, tmax=2.0)
    assert np.array_equal(same.data, trials.data)
    assert same.provenance.source == copy.deepcopy(trials.provenance.source)


def test_light_core_mef_in_a_subprocess(tmp_path: Any) -> None:
    """REQ-025: the MEF reader needs numpy only."""

    code = textwrap.dedent("""
        import sys
        for name in ("pandas", "pyarrow", "mne", "edfio", "pyedflib", "pynwb", "neo"):
            sys.modules[name] = None
        import numpy as np
        from evoparam.io import read, read_mef
        from evoparam.tests.io.helpers import write_mef
        root = write_mef(sys.argv[1], "light", np.arange(3000.).reshape(1, -1) % 97, 500.,
                         ["X"])
        data, sfreq, names, t0 = read_mef(root)
        assert data.shape == (1, 3000) and sfreq == 500. and names == ["X"]
        trials = read(root)
        assert trials.metadata is None and trials.data.shape == (1, 1, 3000)
        print("ok")
    """)
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


###################################################################################################
## BIDS


def test_read_bids_mef_matches_brainvision(bids_root: Any, tmp_path: Any) -> None:
    """The mini BIDS dataset with MEF recordings reads as the BrainVision one."""

    pd = pytest.importorskip("pandas")
    root = make_mef_bids(tmp_path / "mef")
    for subject, session in (("01", None), ("02", "1")):
        reference = read_bids(bids_root, subject, session=session, task="spes", run="01")
        trials = read_bids(root, subject, session=session, task="spes", run="01")
        np.testing.assert_array_equal(trials.data, reference.data)
        pd.testing.assert_frame_equal(trials.metadata, reference.metadata)
        pd.testing.assert_frame_equal(trials.ch_info, reference.ch_info)
        assert trials.masks is None
        assert trials.modality == "spes"
        source = trials.provenance.source
        assert source["format"] == "mef"
        assert source["path"].endswith("_ieeg.mefd")
        assert source["reader"] == "evoparam.io.read_bids"

    # Remotely, with the channel names of channels.tsv: only the picked channels are read
    local = read_bids(root, "01", task="spes", run="01", picks=["A03", "A01"])
    with http_server(root) as (base, stats):
        remote = read_bids(base, "01", task="spes", run="01", remote=True, picks=["A03", "A01"])
        requests = list(stats.requests)
        sent = stats.bytes_sent
    np.testing.assert_array_equal(remote.data, local.data)
    assert remote.ch_names == ["A03", "A01"]
    assert remote.provenance.source["sha256"] == local.provenance.source["sha256"]
    assert remote.provenance.source["bytes_read"] == sent
    assert not any("A02" in r[1] for r in requests)
    assert any(r[1].endswith("_ieeg.vhdr") and r[3] == NOT_FOUND for r in requests)


def test_read_bids_mef_gap_is_masked(tmp_path: Any) -> None:

    pytest.importorskip("pandas")
    root = make_mef_bids(tmp_path / "mef")
    folder = os.path.join(root, "sub-01", "ieeg")
    stem = "sub-01_task-spes_run-01_ieeg"
    shutil.rmtree(os.path.join(folder, stem + ".mefd"))
    write_mef(
        folder,
        stem,
        signal(len(CH_NAMES), 60000),
        BIDS_SFREQ,
        CH_NAMES,
        factor=0.5,
        block_samples=1000,
        gaps={5200: 0.1},
    )
    trials = read_bids(root, "01", task="spes", run="01")
    assert trials.masks is not None
    # The epoch of the onset at 5.0 s holds the gap, from 5.2 to 5.3 s
    row = int(np.flatnonzero(np.isclose(trials.event_times, 5.0))[0])
    gap = (trials.times >= 0.2 - 1e-9) & (trials.times < 0.3 - 1e-9)
    assert trials.masks[row][:, gap].all()
    assert not trials.masks[row][:, ~gap].any()
    assert trials.masks.sum() == gap.sum() * 3
    assert np.isnan(trials.data[trials.masks]).all()
