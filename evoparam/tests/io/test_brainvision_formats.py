# SPDX-License-Identifier: Apache-2.0
"""BrainVision formats: INT_16 / IEEE_FLOAT_32, multiplexed / vectorized, units, picks."""

import pathlib

import numpy as np
import pytest

from evoparam.errors import DataError, ReaderError
from evoparam.io import read_brainvision, read_brainvision_header, read_brainvision_segment
from evoparam.tests.io.helpers import write_brainvision

###################################################################################################
###################################################################################################

SFREQ = 250.0
NAMES = ["C1", "C2", "C3"]


def small_signal(n_times=1000):
    # Integers below 2**14 so that INT_16 with resolution 0.5 is exact
    return (np.arange(3)[:, None] * 1000 + np.arange(n_times)[None, :] % 997).astype(float)


@pytest.mark.parametrize("fmt", ["INT_16", "IEEE_FLOAT_32", "INT_32"])
@pytest.mark.parametrize("orientation", ["MULTIPLEXED", "VECTORIZED"])
def test_formats_and_orientations(tmp_path, fmt, orientation):

    data = small_signal()
    vhdr = write_brainvision(
        tmp_path,
        "rec",
        data,
        SFREQ,
        NAMES,
        fmt=fmt,
        orientation=orientation,
        resolutions=[0.5, 0.5, 0.5],
    )
    header = read_brainvision_header(vhdr)
    assert header["orientation"] == orientation
    assert header["dtype"] == {"INT_16": "<i2", "IEEE_FLOAT_32": "<f4", "INT_32": "<i4"}[fmt]

    out, sfreq, names, t0 = read_brainvision(vhdr)
    assert sfreq == SFREQ
    assert names == NAMES
    assert t0 == 0
    assert np.array_equal(out, data)

    # Window [1.0, 2.0) s: samples 250 to 499
    out, _, names, t0 = read_brainvision(vhdr, tmin=1.0, tmax=2.0, picks=["C3", "C1"])
    assert names == ["C3", "C1"]
    assert t0 == 1.0
    assert np.array_equal(out, data[[2, 0], 250:500])


def test_per_channel_resolution_and_units(tmp_path):

    data = small_signal()
    vhdr = write_brainvision(
        tmp_path,
        "rec",
        data,
        SFREQ,
        NAMES,
        fmt="INT_16",
        resolutions=[0.5, 0.001, 1e-6],
        units=["µV", "mV", "V"],
    )
    with pytest.warns(UserWarning, match="converted to µV: C2, C3"):
        out, _, _, _ = read_brainvision(vhdr)
    assert np.allclose(out, data, rtol=1e-3, atol=1e-6)

    header = read_brainvision_header(vhdr)
    assert list(header["resolutions"]) == [0.5, 0.001, 1e-6]
    assert list(header["to_uv"]) == [1.0, 1e3, 1e6]


def test_non_voltage_unit_and_greek_mu(tmp_path):

    data = small_signal(100)
    vhdr = write_brainvision(tmp_path, "rec", data, SFREQ, NAMES, units=["μV", "µV", "Adim."])
    with pytest.warns(UserWarning, match="not a voltage were left unconverted: C3"):
        out, _, _, _ = read_brainvision(vhdr)
    assert np.allclose(out, data)


def test_duplicated_names_get_a_suffix(tmp_path):

    vhdr = write_brainvision(tmp_path, "rec", small_signal(100), SFREQ, ["X", "Y", "X"])
    with pytest.warns(UserWarning, match="Duplicated channel names"):
        header = read_brainvision_header(vhdr)
    assert header["ch_names"] == ["X-1", "Y", "X-2"]


def test_segment_keeps_inherited_semantics(tmp_path):

    data = small_signal()
    vhdr = write_brainvision(tmp_path, "rec", data, SFREQ, NAMES)
    out, _, _, t0 = read_brainvision_segment(vhdr, 1.0, 2.0)
    assert t0 == 1.0
    assert out.shape == (3, 251)
    assert np.array_equal(out, data[:, 250:501])


def test_errors(tmp_path):

    vhdr = write_brainvision(tmp_path, "rec", small_signal(100), SFREQ, NAMES)
    with pytest.raises(DataError, match="unknown channel"):
        read_brainvision(vhdr, picks=["nope"])
    with pytest.raises(ReaderError, match="outside the recording"):
        read_brainvision(vhdr, tmin=10.0, tmax=11.0)
    with pytest.raises(ReaderError):
        read_brainvision(str(tmp_path / "missing.vhdr"))

    text = (
        pathlib.Path(vhdr)
        .read_text(encoding="utf-8")
        .replace("DataFormat=BINARY", "DataFormat=ASCII")
    )
    (tmp_path / "ascii.vhdr").write_text(text, encoding="utf-8")
    with pytest.raises(ReaderError, match="only binary"):
        read_brainvision_header(str(tmp_path / "ascii.vhdr"))

    text = pathlib.Path(vhdr).read_text(encoding="utf-8").replace("IEEE_FLOAT_32", "UINT_8")
    (tmp_path / "uint8.vhdr").write_text(text, encoding="utf-8")
    with pytest.raises(ReaderError, match="not supported"):
        read_brainvision_header(str(tmp_path / "uint8.vhdr"))


def test_latin1_header(tmp_path):

    vhdr = write_brainvision(tmp_path, "rec", small_signal(100), SFREQ, NAMES)
    text = pathlib.Path(vhdr).read_text(encoding="utf-8")
    with open(vhdr, "wb") as file:
        file.write(text.encode("latin-1"))
    header = read_brainvision_header(vhdr)
    assert header["units"] == ["µV"] * 3
