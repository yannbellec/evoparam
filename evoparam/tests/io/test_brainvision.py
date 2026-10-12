# SPDX-License-Identifier: Apache-2.0
"""Tests for evoparam.io.brainvision."""

import numpy as np

from evoparam.io.brainvision import *

###################################################################################################
###################################################################################################


def write_brainvision(folder, data, sfreq, ch_names, resolution=0.5):
    """Write a small multiplexed float32 BrainVision recording, and return the .vhdr path."""

    lines = [
        "Brain Vision Data Exchange Header File Version 1.0",
        "; test file",
        "",
        "[Common Infos]",
        "DataFile=test.eeg",
        "DataFormat=BINARY",
        "DataOrientation=MULTIPLEXED",
        f"NumberOfChannels={len(ch_names)}",
        f"SamplingInterval={1e6 / sfreq}",
        "",
        "[Binary Infos]",
        "BinaryFormat=IEEE_FLOAT_32",
        "",
        "[Channel Infos]",
    ]
    lines += [f"Ch{ind + 1}={name},,{resolution},µV" for ind, name in enumerate(ch_names)]
    (folder / "test.vhdr").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (data.T / resolution).astype("<f4").tofile(folder / "test.eeg")

    return str(folder / "test.vhdr")


def test_read_brainvision(tmp_path):

    sfreq, ch_names = 100.0, ["A1", "A2", "A3"]
    data = np.arange(3 * 500, dtype=float).reshape(3, 500)
    vhdr = write_brainvision(tmp_path, data, sfreq, ch_names)

    header = read_brainvision_header(vhdr)
    assert header["n_channels"] == 3
    assert header["sfreq"] == sfreq
    assert header["ch_names"] == ch_names

    out, out_sfreq, out_names, t0 = read_brainvision_segment(vhdr, 1.0, 2.0)
    assert out_sfreq == sfreq and out_names == ch_names and t0 == 1.0
    assert np.allclose(out, data[:, 100:201])

    out, _, out_names, _ = read_brainvision_segment(vhdr, 0, 0.5, picks=["A3", "A1"])
    assert out_names == ["A3", "A1"]
    assert np.allclose(out, data[[2, 0], 0:51])
