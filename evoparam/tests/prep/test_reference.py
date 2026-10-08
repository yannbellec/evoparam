"""Re-referencing (REQ-081, REQ-082)."""


import numpy as np
import pandas as pd
import pytest

from evoparam.errors import DataError, SettingsError
from evoparam.prep import car_adaptive_select, rereference

from .helpers import epochs, sim_carla

###################################################################################################
###################################################################################################

NAMES = ["LA1", "LA2", "LA3", "LA5", "LA6", "RB1", "RB2", "RB3"]


def test_req082_bipolar_matches_mne():

    mne = pytest.importorskip("mne")
    ts = epochs(names=NAMES, n_trials=3)
    with pytest.warns(UserWarning, match="missing contacts"):
        bip = rereference(ts, "bipolar")
    assert bip.ch_names == ["LA1-LA2", "LA2-LA3", "LA5-LA6", "RB1-RB2", "RB2-RB3"]
    assert (bip.ch_info["reference"] == "bipolar").all()

    info = mne.create_info(NAMES, ts.sfreq, "seeg")
    epochs_mne = mne.EpochsArray(ts.data * 1e-6, info, tmin=ts.times[0], verbose=False)
    anode = [n.split("-")[0] for n in bip.ch_names]
    cathode = [n.split("-")[1] for n in bip.ch_names]
    ref = mne.set_bipolar_reference(epochs_mne, anode, cathode, verbose=False)
    ref = ref.pick([n for n in ref.ch_names if "-" in n])
    assert ref.ch_names == bip.ch_names
    assert np.allclose(ref.get_data() * 1e6, bip.data)


def test_bipolar_uses_groups_and_masks():

    ts = epochs(names=["x1", "x2", "y1", "y2"], n_trials=2)
    groups = {"x1": "shaft", "x2": "shaft", "y1": "other", "y2": "other"}
    out = rereference(ts, "bipolar", groups=groups)
    assert out.ch_names == ["x1-x2", "y1-y2"]
    with pytest.raises(DataError):
        rereference(epochs(names=["a", "b"]), "bipolar")


def test_car_excludes_stimulated_and_bad_channels():

    meta = pd.DataFrame({"stim_site": ["A1-A2"] * 4})
    info = pd.DataFrame({"name": [f"A{i}" for i in range(1, 9)],
                         "bad": [False] * 7 + [True]})
    ts = epochs(n_trials=4, metadata=meta, ch_info=info)
    out = rereference(ts, "car")
    expected = ts.data - ts.data[:, 2:7].mean(axis=1, keepdims=True)
    assert np.allclose(out.data, expected)
    assert (out.ch_info["reference"] == "car").all()
    out = rereference(ts, "car", exclude=["A3"])
    assert np.allclose(out.data, ts.data - ts.data[:, 3:7].mean(axis=1, keepdims=True))


def test_req081_car_adaptive_simulation():
    """Median false negatives 0 up to 84 % responsive channels, false positives <= 3.5."""

    for fraction in (0.0, 0.5, 0.84):
        fn, fp = [], []
        for seed in range(5):
            ts, responsive = sim_carla(fraction, seed)
            chosen = car_adaptive_select(ts.data, ts.times, n_boot=30, random_state=seed)
            members = set(chosen["members"].tolist())
            fn.append(len(members & responsive))
            fp.append(len(set(range(50)) - responsive - members))
        assert np.median(fn) == 0, fraction
        assert np.median(fp) <= 3.5, fraction


def test_car_adaptive_records_choice_and_falls_back():

    ts, responsive = sim_carla(0.3, 1)
    out = rereference(ts, "car_adaptive", n_boot=20, random_state=0)
    step = out.provenance.steps[-1]["settings"]
    assert step["used"] == "car_adaptive"
    assert len(step["members"]) == step["n_star"]
    assert not {ts.ch_names[i] for i in responsive} & set(step["members"])
    small = epochs(names=[f"A{i}" for i in range(1, 6)])
    with pytest.warns(UserWarning, match="plain common average"):
        out = rereference(small, "car_adaptive")
    assert out.provenance.steps[-1]["settings"]["used"] == "car"


def test_laplacian_and_monopolar():

    info = pd.DataFrame({"name": ["G1", "G2", "G3", "G4"], "x": [0., 10., 20., 100.],
                         "y": [0.] * 4, "z": [0.] * 4})
    ts = epochs(names=["G1", "G2", "G3", "G4"], ch_info=info, n_trials=2)
    with pytest.warns(UserWarning, match="no neighbour"):
        out = rereference(ts, "laplacian", radius=12.0)
    assert np.allclose(out.data[:, 1], ts.data[:, 1] - (ts.data[:, 0] + ts.data[:, 2]) / 2)
    assert np.allclose(out.data[:, 3], ts.data[:, 3])
    assert out.ch_info["reference"].tolist() == ["laplacian"] * 3 + ["monopolar"]
    with pytest.raises(DataError):
        rereference(epochs(), "laplacian")
    mono = rereference(ts, "monopolar")
    assert np.array_equal(mono.data, ts.data)
    assert (mono.ch_info["reference"] == "monopolar").all()
    with pytest.raises(SettingsError):
        rereference(ts, "average")
