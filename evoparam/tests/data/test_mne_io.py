# SPDX-License-Identifier: Apache-2.0
"""MNE round trip and fitting MNE objects (REQ-310, REQ-311)."""

import warnings
from typing import Any

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from evoparam.data import Evoked, TrialSet  # noqa: E402  (after pytest.importorskip)

pd = pytest.importorskip("pandas")


def _epochs() -> Any:
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
    return mne.EpochsArray(
        data,
        info,
        events=events,
        tmin=-0.05,
        event_id={"a": 1, "b": 2},
        metadata=metadata,
        verbose="error",
    )


def test_req310_round_trip() -> None:

    e = _epochs()
    ts = TrialSet.from_mne(e)
    assert ts.units == "uV"
    assert np.allclose(ts.data, e.get_data() * 1e6)
    assert list(ts.metadata["condition"]) == ["a", "b"] * 3
    assert list(ts.ch_info["type"]) == ["eeg", "ecog", "seeg", "dbs"]
    assert list(ts.ch_info["bad"]) == [False, True, False, False]
    back = ts.to_mne()
    np.testing.assert_allclose(back.get_data(), e.get_data(), rtol=1e-12)
    assert back.info["bads"] == e.info["bads"]
    pd.testing.assert_frame_equal(
        back.metadata.reset_index(drop=True), e.metadata.reset_index(drop=True)
    )
    assert back.event_id == e.event_id
    # event times survive (the samples of MNE already include first_samp)
    np.testing.assert_allclose(ts.event_times, e.events[:, 0] / 1000.0)
    np.testing.assert_array_equal(back.events, e.events)
    # stim_columns="auto": standard columns added (REQ-313), not given back to MNE
    assert ts.metadata["stim_current_mA"].tolist() == list(np.arange(6.0))
    assert {"stim_site", "stim_pulse_width_us", "stim_polarity"} <= set(ts.metadata)
    assert (
        "stim_current_mA" in ts.provenance.source["harmonization"]["mapping"].values()
        or ts.provenance.source["harmonization"]["mapping"]["stim_current_mA"] == "stim_current_mA"
    )
    raw = TrialSet.from_mne(e, stim_columns=None)
    assert "stim_site" not in raw.metadata
    np.testing.assert_allclose(
        [c["loc"][:3] for c in back.info["chs"]], [c["loc"][:3] for c in e.info["chs"]], atol=1e-9
    )
    ev = e.average()
    again = Evoked.from_mne(ev).to_mne()
    np.testing.assert_allclose(again.data, ev.data, rtol=1e-12)


def test_req310_stim_columns_mapping() -> None:

    e = _epochs()
    e.metadata = pd.DataFrame({"amp": ["4 mA", "4 mA", "2 mA"] * 2, "pair": ["PT01-PT02"] * 6})
    ts = TrialSet.from_mne(e, stim_columns={"amp": "stim_current_mA", "pair": "stim_site"})
    assert ts.metadata["stim_current_mA"].tolist() == [4.0, 4.0, 2.0] * 2
    assert set(ts.metadata["stim_site"]) == {"PT01-PT02"}
    assert list(ts.to_mne().metadata.columns) == ["amp", "pair"]


def test_req311_fit_dispatch() -> None:

    import evoparam

    e = _epochs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = evoparam.fit(e, "erp", fit_window=(0.05, 0.29))
        converted = evoparam.fit(TrialSet.from_mne(e), "erp", fit_window=(0.05, 0.29))
        pd.testing.assert_frame_equal(direct.to_dataframe(), converted.to_dataframe())
        average = e.average()
        # evoparam.fit runs the existence test, whose null draws a seed when none is given
        pd.testing.assert_frame_equal(
            evoparam.fit(average, "erp", fit_window=(0.05, 0.29), random_state=0).to_dataframe(),
            evoparam.fit(
                Evoked.from_mne(average), "erp", fit_window=(0.05, 0.29), random_state=0
            ).to_dataframe(),
        )
        evokeds = [e[c].average() for c in ("a", "b")]
        for ev, name in zip(evokeds, ("a", "b"), strict=False):
            ev.comment = name
        grouped = evoparam.fit(evokeds, "erp", fit_window=(0.05, 0.29))
    assert set(grouped.index["condition"]) == {"a", "b"}
    annotations = direct.to_annotations()
    assert isinstance(annotations, mne.Annotations)
    assert all("|" in d for d in annotations.description)
    model = direct.to_evoked_model()
    assert isinstance(model, mne.EvokedArray)
    assert model.comment == "evoparam model"
