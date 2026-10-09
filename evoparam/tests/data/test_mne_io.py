"""MNE round trip and fitting MNE objects (REQ-310, REQ-311)."""

import warnings

import numpy as np
import pandas as pd
import pytest

mne = pytest.importorskip("mne")

from evoparam.data import Evoked, TrialSet  # noqa: E402


def _epochs():
    info = mne.create_info(["E1", "G1", "S1", "D1"], 1000.0, ["eeg", "ecog", "seeg", "dbs"])
    with info._unlock():
        for i, ch in enumerate(info["chs"]):
            ch["loc"][:3] = [0.01 * i, 0.02, 0.03]
            ch["coord_frame"] = 4
    info["bads"] = ["G1"]
    rng = np.random.default_rng(0)
    times = np.arange(-0.05, 0.3, 0.001)
    data = rng.normal(size=(6, 4, times.size)) * 1e-6
    data[:, 0] -= 30e-6 * np.exp(-0.5 * ((times - 0.1) / 0.01) ** 2)
    events = np.column_stack([np.arange(6) * 500 + 1000, np.zeros(6, int), [1, 2] * 3])
    metadata = pd.DataFrame({"stim_current_mA": np.arange(6, dtype=float)})
    return mne.EpochsArray(data, info, events=events, tmin=-0.05, event_id={"a": 1, "b": 2},
                           metadata=metadata, verbose="error")


def test_req310_round_trip():

    e = _epochs()
    ts = TrialSet.from_mne(e)
    assert ts.units == "uV" and np.allclose(ts.data, e.get_data() * 1e6)
    assert list(ts.metadata["condition"]) == ["a", "b"] * 3
    assert list(ts.ch_info["type"]) == ["eeg", "ecog", "seeg", "dbs"]
    assert list(ts.ch_info["bad"]) == [False, True, False, False]
    back = ts.to_mne()
    np.testing.assert_allclose(back.get_data(), e.get_data(), rtol=1e-12)
    assert back.info["bads"] == e.info["bads"]
    pd.testing.assert_frame_equal(back.metadata.reset_index(drop=True),
                                  e.metadata.reset_index(drop=True))
    assert back.event_id == e.event_id
    np.testing.assert_allclose([c["loc"][:3] for c in back.info["chs"]],
                               [c["loc"][:3] for c in e.info["chs"]], atol=1e-9)
    ev = e.average()
    again = Evoked.from_mne(ev).to_mne()
    np.testing.assert_allclose(again.data, ev.data, rtol=1e-12)


def test_req311_fit_dispatch():

    import evoparam

    e = _epochs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = evoparam.fit(e, "erp", fit_window=(0.05, 0.29))
        converted = evoparam.fit(TrialSet.from_mne(e), "erp", fit_window=(0.05, 0.29))
        pd.testing.assert_frame_equal(direct.to_dataframe(), converted.to_dataframe())
        evokeds = [e[c].average() for c in ("a", "b")]
        for ev, name in zip(evokeds, ("a", "b")):
            ev.comment = name
        grouped = evoparam.fit(evokeds, "erp", fit_window=(0.05, 0.29))
    assert set(grouped.index["condition"]) == {"a", "b"}
    annotations = direct.to_annotations()
    assert isinstance(annotations, mne.Annotations)
    assert all("|" in d for d in annotations.description)
    model = direct.to_evoked_model()
    assert isinstance(model, mne.EvokedArray) and model.comment == "evoparam model"
