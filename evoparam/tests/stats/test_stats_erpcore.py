# SPDX-License-Identifier: Apache-2.0
"""Reliability of the ERP CORE N170 (REQ-233, real-data part).

ERP CORE (Kappenman et al. 2021; OSF, DOI 10.18115/D5JW4R; CC-BY-4.0, recorded in the
dataset registry) is not redistributed by the test suite: the test reads a local copy
fetched from OSF (node pfde9, the N170 folder: ``<participant>/*N170*_epoch_interp_ar.set``
and ``.fdt``), given by EVOPARAM_ERPCORE_N170. The N170 is scored as the mean amplitude
at PO8 from 110 to 150 ms (the ERP CORE measurement window), over every epoch of each
participant. It needs the
extra ``mne`` (EEGLAB reader) and runs only with EVOPARAM_NETWORK_TESTS=1.
"""

import glob
import os
from typing import Any

import numpy as np
import pandas as pd
import pytest

from evoparam.data import TrialSet
from evoparam.stats import reliability

from .helpers import independent_dependability, independent_split_half

ROOT = os.environ.get("EVOPARAM_ERPCORE_N170", "")

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
        reason="set EVOPARAM_NETWORK_TESTS=1 for real-data checks",
    ),
    pytest.mark.skipif(
        not ROOT,
        reason="set EVOPARAM_ERPCORE_N170 to a local copy "
        "of the ERP CORE N170 epochs (not redistributed)",
    ),
]

WINDOW = (0.110, 0.150)

###################################################################################################
###################################################################################################


def _load() -> Any:
    mne = pytest.importorskip("mne")
    files = sorted(glob.glob(os.path.join(ROOT, "**", "*N170*epoch*.set"), recursive=True))
    assert len(files) >= 10, f"found {len(files)} epoched N170 files under {ROOT}"
    data, subjects, times = [], [], None
    for path in files:
        epochs = mne.read_epochs_eeglab(path, verbose="error")
        x = epochs.get_data(picks=["PO8"]) * 1e6  # V -> uV
        if times is None:
            times = epochs.times
        assert np.allclose(epochs.times, times), path
        data.append(x)
        subjects += [os.path.basename(path).split("_")[0]] * len(x)
    return TrialSet(
        data=np.concatenate(data),
        times=times,
        sfreq=1.0 / np.median(np.diff(times)),
        ch_names=["PO8"],
        metadata=pd.DataFrame({"subject": subjects}),
    )


def test_req233_erpcore_n170() -> None:

    ts = _load()
    out = reliability(ts, window=WINDOW, n_splits=1000, random_state=0).iloc[0]
    inside = (ts.times >= WINDOW[0]) & (ts.times <= WINDOW[1])
    split = independent_split_half(ts, lambda avg: avg[inside].mean())
    phi, _, _ = independent_dependability(ts, WINDOW)
    assert out["split_half_odd_even"] == pytest.approx(split, abs=0.01)
    assert out["dependability"] == pytest.approx(phi, abs=0.01)
    assert out["dependability"] > 0.7  # a strong, reliable component
