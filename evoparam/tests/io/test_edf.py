# SPDX-License-Identifier: Apache-2.0
"""read_edf; runs only when edfio is installed (extra "edf")."""

import numpy as np
import pytest

from evoparam.io import read_edf

edfio = pytest.importorskip("edfio")

###################################################################################################
###################################################################################################


def test_read_edf(tmp_path):

    sfreq, n_times = 256, 256 * 4
    values = np.vstack(
        [np.sin(np.arange(n_times) / 10.0) * 100.0, np.cos(np.arange(n_times) / 7.0)]
    )
    signals = [
        edfio.EdfSignal(values[0], sampling_frequency=sfreq, label="A1", physical_dimension="uV"),
        edfio.EdfSignal(values[1], sampling_frequency=sfreq, label="A2", physical_dimension="mV"),
    ]
    path = str(tmp_path / "rec.edf")
    edfio.Edf(signals).write(path)

    data, out_sfreq, names, t0 = read_edf(path, tmin=1.0, tmax=2.0, picks=["A2", "A1"])
    assert out_sfreq == sfreq
    assert names == ["A2", "A1"]
    assert t0 == 1.0
    assert data.shape == (2, sfreq)
    assert np.allclose(data[0], values[1, sfreq : 2 * sfreq] * 1e3, rtol=1e-3, atol=1.0)
    assert np.allclose(data[1], values[0, sfreq : 2 * sfreq], rtol=1e-3, atol=0.1)
