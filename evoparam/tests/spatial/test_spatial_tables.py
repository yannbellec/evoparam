# SPDX-License-Identifier: Apache-2.0
"""Parameter maps and the connectivity export (REQ-195, REQ-196).

REQ-196 reads six stimulation sites of one ds004080 run (OpenNeuro, remote byte ranges;
the epochs are cached in the evoparam data folder, about 4 MB) and runs only with
EVOPARAM_NETWORK_TESTS=1.
"""

import copy
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from evoparam.data import TrialSet
from evoparam.errors import DataError, SettingsError
from evoparam.existence import test_response as run_test
from evoparam.models import fit as fit_models
from evoparam.spatial import (MAP_COLUMNS, MARKERS, connectivity_array, connectivity_from_long,
                              connectivity_matrix, connectivity_to_long, parameter_map,
                              plot_parameter_map)

from .helpers import BASELINE, SFREQ, TIMES, gauss

###################################################################################################
###################################################################################################

SITES = ("A01-A02", "A03-A04", "B01-B02")


def _scene(seed=0, n_trials=8):
    """Three sites into eight contacts on two shafts (4 mm apart); every site drives the
    contacts of its own shaft (an N1 whose latency grows with distance), not the others."""

    rng = np.random.default_rng(seed)
    names = [f"{s}{i:02d}" for s in "AB" for i in range(1, 5)]
    coords = np.array([(0.0 if n[0] == "A" else 20.0, 0.0, -4.0 * int(n[1:])) for n in names])
    data, sites = [], []
    for site in SITES:
        stim = site.split("-")
        centre = coords[[names.index(s) for s in stim]].mean(axis=0)
        for _ in range(n_trials):
            trial = rng.normal(0, 5, (len(names), len(TIMES)))
            for c, name in enumerate(names):
                if name in stim:
                    trial[c] += 2000.0
                elif name[0] == site[0]:
                    r = np.linalg.norm(coords[c] - centre)
                    trial[c] -= 80 * gauss(0.02 + r / 4000, 0.012)
            data.append(trial)
            sites.append(site)
    info = pd.DataFrame({"name": names, "x": coords[:, 0], "y": coords[:, 1],
                         "z": coords[:, 2], "coord_space": "ACPC", "group": [n[0] for n in names],
                         "bad": [n == "B04" for n in names],
                         "stim_contact": [n in ("A01", "A02", "A03", "A04", "B01", "B02")
                                          for n in names]})
    return TrialSet(data=np.array(data), times=TIMES, sfreq=SFREQ, ch_names=names,
                    ch_info=info, metadata=pd.DataFrame({"stim_site": sites}))


@pytest.fixture(scope="module")
def scene():
    ts = _scene()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = fit_models(ts, "ccep", baseline=BASELINE, by="stim_site", classical=False)
    return ts, group


def _largest(table):
    """Per (channel, condition): the component of largest |amplitude| from to_dataframe."""

    table = table.assign(abs_amp=table["amplitude"].abs())
    return {(r.channel, r.condition): r for r in
            table.sort_values("abs_amp", ascending=False, na_position="last")
            .drop_duplicates(["channel", "condition"]).itertuples()}


def test_req195_connectivity_round_trip_and_cells(scene):

    _, group = scene
    for parameter in ("amplitude", "latency", "fwhm"):
        wide = connectivity_matrix(group, parameter)
        assert list(wide.index) == list(SITES)
        assert list(wide.columns) == group.ch_names
        long = connectivity_matrix(group, parameter, long=True)
        assert len(long) == wide.size and set(long["parameter"]) == {parameter}
        # wide -> long -> wide, and long -> wide
        back = connectivity_from_long(connectivity_to_long(wide, parameter))
        pd.testing.assert_frame_equal(back, wide, check_exact=False, rtol=1e-12)
        pd.testing.assert_frame_equal(connectivity_from_long(long), wide, check_exact=False,
                                      rtol=1e-12)
        # Cell by cell against GroupFit.to_dataframe()
        largest = _largest(group.to_dataframe())
        for site in wide.index:
            for ch in wide.columns:
                row = largest[(ch, site)]
                expected = getattr(row, parameter)
                if row.status != "ok" or row.component < 0:
                    assert np.isnan(wide.loc[site, ch])
                else:
                    assert wide.loc[site, ch] == pytest.approx(expected, rel=1e-12, abs=1e-15)
        array, sites, channels = connectivity_array(wide)
        assert array.shape == (len(sites), len(channels))
        assert np.array_equal(np.isnan(array), wide.isna().to_numpy())


def test_parameter_map_columns_markers_and_coordinates(scene):

    ts, group = scene
    table = parameter_map(group, ch_info=ts)
    assert list(table.columns) == list(MAP_COLUMNS)
    assert len(table) == len(SITES) * ts.n_channels
    assert not table.duplicated(["stim_site", "channel"]).any()
    assert set(table["marker"]) <= set(MARKERS)
    rows = table.set_index(["stim_site", "channel"])
    assert rows.loc[("A01-A02", "A01"), "marker"] == "stim_contact"
    assert rows.loc[("A01-A02", "B04"), "marker"] == "bad"
    # A responsive contact: an N1 near 20 ms, coordinates from ch_info
    a04 = rows.loc[("A01-A02", "A04")]
    assert a04["marker"] == "response" and a04["amplitude"] < -40
    assert abs(a04["latency"] - 0.0225) < 0.004
    assert (a04["x"], a04["z"], a04["coord_space"]) == (0.0, -16.0, "ACPC")
    assert (table["value"] == table["amplitude"]).sum() == table["amplitude"].notna().sum()
    ok = table["status"] == "ok"
    assert table.loc[ok, "amplitude"].notna().all()
    # One site, another parameter
    one = parameter_map(group, parameter="latency", stim_site="B01-B02", ch_info=ts)
    assert set(one["stim_site"]) == {"B01-B02"} and len(one) == ts.n_channels
    assert np.allclose(one["value"], one["latency"], equal_nan=True)
    with pytest.raises(SettingsError):
        parameter_map(group, parameter="colour")
    with pytest.raises(DataError):
        parameter_map(ts)


def test_parameter_map_without_coordinates_warns(scene):

    _, group = scene
    with pytest.warns(UserWarning, match="no channel coordinates"):
        table = parameter_map(group)
    assert table["x"].isna().all()


def test_parameter_map_with_existence_and_shapes(scene):

    ts, group = scene
    test = run_test(ts, window=(0.009, 0.2), baseline=BASELINE, n_permutations=200,
                    n_boot=0, random_state=0)
    table = parameter_map(group, ch_info=ts, existence=test, parameter="existence_p")
    rows = table.set_index(["stim_site", "channel"])
    assert rows.loc[("A01-A02", "A04"), "existence_significant"]
    assert not rows.loc[("A01-A02", "B03"), "existence_significant"]
    assert np.allclose(table["value"], table["existence_p"], equal_nan=True)
    # A hierarchical-style table of P(response)
    probs = pd.DataFrame({"channel": ["A04", "B03"], "stim_site": ["A01-A02"] * 2,
                          "p_response": [0.97, 0.02]})
    hier = type("Hier", (), {"p_response": probs})()
    rows = parameter_map(group, ch_info=ts, existence=hier).set_index(["stim_site", "channel"])
    assert rows.loc[("A01-A02", "A04"), "p_response"] == 0.97
    assert np.isnan(rows.loc[("A03-A04", "A04"), "p_response"])
    # Shape weights of a divergent basis of one site
    basis = {"A01-A02": copy.deepcopy(shared_shapes_stub())}
    rows = parameter_map(group, ch_info=ts, basis=basis).set_index(["stim_site", "channel"])
    assert rows.loc[("A01-A02", "A03"), "shape_id"] == 0
    assert rows.loc[("A01-A02", "A03"), "shape_weight"] == 2.0
    assert rows.loc[("A03-A04", "A03"), "shape_id"] == -1


def shared_shapes_stub():
    from evoparam.data import ShapeBasis

    assignment = pd.DataFrame({"group": ["A03", "A04"], "shape_id": [0, 0],
                               "weight": [2.0, 1.0], "r2": [0.9, 0.8], "n_trials": [8, 8],
                               "used": [True, True], "nmf_loading": [0.7, 0.7]})
    return ShapeBasis(curves=np.ones((1, 3)) / np.sqrt(3), times=np.arange(3.0),
                      weights=np.array([[2.0], [1.0]]), r2=np.array([0.9, 0.8]),
                      assignment=assignment, stability=np.ones(1), n_shapes=1, method="nmf",
                      settings={"stim_site": "A01-A02"})


def test_plot_parameter_map(scene):

    pytest.importorskip("matplotlib")
    ts, group = scene
    figure = plot_parameter_map(parameter_map(group, ch_info=ts), projection=("x", "z"))
    assert figure.axes
    import matplotlib.pyplot as plt
    plt.close(figure)


def _mni(table, space="MNI152NLin2009aSym"):
    """The table with its coordinates moved into a plausible MNI range (left temporal)."""

    out = table.copy()
    out["x"], out["y"], out["z"] = -40.0 + out["x"] / 4, -20.0 + out["y"], 10.0 + out["z"]
    out["coord_space"] = space
    out.attrs.update(table.attrs)
    return out


def test_parameter_map_glass_brain_with_nilearn(scene):
    """Spec §5.9 method 3: a glass-brain figure, the nilearn plotting backend used when
    installed (MNI-like coordinates), the stimulation contacts, bad and masked channels as
    distinct markers."""

    pytest.importorskip("matplotlib")
    pytest.importorskip("nilearn")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection

    ts, group = scene
    table = _mni(parameter_map(group, ch_info=ts))
    site = table["stim_site"].iloc[0]
    figure = plot_parameter_map(table, stim_site=site)          # backend "auto" -> nilearn
    # the glass brain: one axes per view (ortho: sagittal, coronal, axial) plus the colour
    # bar, each view with the response markers and the stimulation contacts
    views = [a for a in figure.axes if any(isinstance(c, PathCollection)
                                           for c in a.collections)]
    assert len(views) >= 3
    markers = table[table["stim_site"] == site]["marker"]
    assert (markers == "stim_contact").any()
    legend = [t.get_text() for t in figure.legends[0].get_texts()]
    assert "stim_contact" in legend
    plt.close(figure)
    with pytest.raises(SettingsError, match="MNI-like"):
        plot_parameter_map(parameter_map(group, ch_info=ts), backend="nilearn")


def test_parameter_map_electrode_grid_fallback(scene, monkeypatch):
    """Without nilearn (or in subject space) the electrode-grid figure is drawn."""

    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    import evoparam.spatial.maps as maps

    ts, group = scene
    table = _mni(parameter_map(group, ch_info=ts))
    monkeypatch.setattr(maps, "_have_nilearn", lambda: False)
    figure = plot_parameter_map(table, projection=("x", "z"))
    assert len(figure.axes) == 2                                 # the grid and its colour bar
    assert figure.axes[0].get_xlabel().startswith("x (mm, MNI152")
    plt.close(figure)
    monkeypatch.undo()
    subject = table.assign(coord_space="ACPC")
    figure = plot_parameter_map(subject)
    assert len(figure.axes) == 2
    plt.close(figure)


###################################################################################################
## REQ-196 on OpenNeuro ds004080

network = pytest.mark.skipif(os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
                             reason="set EVOPARAM_NETWORK_TESTS=1 to read OpenNeuro")
RUN = dict(subject="ccepAgeUMCU01", session="1", task="SPESclin", run="021448")
SITES_4080 = ["PT01-PT02", "PT03-PT02", "PT03-PT04", "PT05-PT04", "PT05-PT06", "PT07-PT06"]
EPOCH = (-0.6, 0.6)


def load_ds004080():
    """Epochs of six sites of one run (cached), with ch_info from the BIDS tables."""

    from evoparam.io import read_bids
    from evoparam.io._fetch import data_folder

    cache = os.path.join(data_folder(), "spatial-validation", "ds004080")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, "sub-{subject}_run-{run}.npz".format(**RUN))
    info_path = path.replace(".npz", "_ch_info.csv")
    if not os.path.exists(path):
        ts = read_bids("ds004080", RUN["subject"], session=RUN["session"], task=RUN["task"],
                       run=RUN["run"], remote=True, picks="ecog", epoch=EPOCH,
                       select={"stim_site": SITES_4080})
        ts.ch_info.to_csv(info_path, index=False)
        np.savez_compressed(path, data=ts.data.astype(np.float32), times=ts.times,
                            sfreq=ts.sfreq, names=np.array(ts.ch_names),
                            sites=ts.metadata["stim_site"].to_numpy(dtype=str))
    saved = np.load(path)
    info = pd.read_csv(info_path, keep_default_na=False, na_values=[""])
    for column in ("x", "y", "z"):
        info[column] = pd.to_numeric(info[column], errors="coerce")
    return TrialSet(data=saved["data"].astype(float), times=saved["times"],
                    sfreq=float(saved["sfreq"]), ch_names=[str(n) for n in saved["names"]],
                    metadata=pd.DataFrame({"stim_site": saved["sites"].astype(str)}),
                    ch_info=info)


def _electrodes():
    from evoparam.io import openneuro_url
    from evoparam.io._fetch import data_folder

    cache = os.path.join(data_folder(), "spatial-validation", "ds004080",
                         "sub-{subject}_electrodes.tsv".format(**RUN))
    if not os.path.exists(cache):
        import urllib.request

        url = openneuro_url("ds004080", "sub-{subject}/ses-{session}/ieeg/sub-{subject}_"
                            "ses-{session}_electrodes.tsv".format(**RUN))
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        with open(cache, "wb") as fobj:
            fobj.write(content)
    return pd.read_csv(cache, sep="\t", na_values=["n/a"])


@pytest.mark.network
@network
def test_req196_parameter_map_on_ds004080():

    from evoparam.artifact import remove_artifact
    from evoparam.prep import baseline_correct

    ts = load_ds004080()
    assert set(ts.metadata["stim_site"]) == set(SITES_4080)
    # Three of the six cached sites keep the fits under five minutes (about 0.8 s per fit)
    sites = SITES_4080[:3]
    ts = ts.select(trials=np.flatnonzero(ts.metadata["stim_site"].isin(sites)))
    baseline = (-0.5, -0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = remove_artifact(ts, method="template", window=(-0.002, 0.008),
                             baseline=baseline).cleaned
        ts = baseline_correct(ts, baseline, check=False)
        group = fit_models(ts, "ccep", baseline=baseline, by="stim_site", classical=False)
    table = parameter_map(group, ch_info=ts)
    # One row per (stim_site, channel)
    assert len(table) == len(sites) * ts.n_channels
    assert not table.duplicated(["stim_site", "channel"]).any()
    assert set(table["stim_site"]) == set(sites)
    # Coordinates from electrodes.tsv (mm, fsaverage)
    electrodes = _electrodes().set_index("name")
    rows = table.drop_duplicates("channel").set_index("channel")
    shared = [ch for ch in rows.index if ch in electrodes.index]
    assert len(shared) == ts.n_channels
    for axis in "xyz":
        assert np.allclose(rows.loc[shared, axis], electrodes.loc[shared, axis],
                           equal_nan=True, atol=1e-6)
    assert set(rows["coord_space"]) == {"fsaverage"}
    # No NaN amplitude where the fit status is "ok"
    ok = table["status"] == "ok"
    assert ok.sum() >= 100
    assert table.loc[ok, "amplitude"].notna().all()
    stim = table[[ch in site.split("-") for ch, site in zip(table["channel"],
                                                             table["stim_site"])]]
    assert (stim["marker"] == "stim_contact").all() and len(stim) == 2 * len(sites)
