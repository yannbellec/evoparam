# SPDX-License-Identifier: Apache-2.0
"""REQ-001 (one model) and REQ-002 (one call).

REQ-001: the same fitting engine and result types serve every preset of §6; one simulated
waveform of each modality goes through ``evoparam.fit`` with only the preset name changed,
and every result has the same type and columns.

REQ-002: ``evoparam.fit(data, preset=...)`` on an ``mne.Epochs``, an ``mne.Evoked``, a
``TrialSet``, a BIDS path read by ``evoparam.read`` or a NumPy array with ``times`` returns a
``GroupFit`` with existence results, components, uncertainties and a methods paragraph, with
no other argument required.
"""

import warnings
from typing import Any

import pytest

import evoparam
from evoparam import presets, sim
from evoparam.data import Component, GroupFit, ResponseFit, ResponseTest
from evoparam.data.results import FIT_STATUSES

# Shortest methods text.
MIN_TEXT = 200

pytest.importorskip("pandas")

NO_ARTIFACT = {"pulse_artifact": False, "decay_artifact": False, "muscle_artifact": False}

# Median-nerve SEP (N20, P25, N33; as in the REQ-277 case of test_presets_sim): the P1, N1
# and P3 of the ERP defaults lie after the 10-60 ms window of the sep preset, where a fit
# rightly finds no response.
SEP_COMPONENTS = (
    {"label": "N20", "latency": 0.020, "fwhm": 0.004, "amplitude": -1.0},
    {"label": "P25", "latency": 0.025, "fwhm": 0.006, "amplitude": 0.8},
    {"label": "N33", "latency": 0.033, "fwhm": 0.008, "amplitude": -0.6},
)

# One simulated waveform per modality (§5.12 generators): the epoch covers the preset's
# window. TEP at 1 kHz: the 5 kHz default takes about 70 s per fit (measured), the same
# waveform at 1 kHz about 2 s.
MODALITIES = {
    "ccep": lambda: sim.simulate_ccep(
        n_trials=10, n_channels=4, p_response=1.0, layout="grid", seed=0
    ),
    "ccep_seeg": lambda: sim.simulate_ccep(
        n_trials=10, n_channels=4, p_response=1.0, layout="seeg", seed=0
    ),
    "ecap": lambda: sim.simulate_ecap(
        n_trials=20, n_channels=3, kind="spinal", tmax=0.0035, seed=0
    ),
    "ecap_ci": lambda: sim.simulate_ecap(
        n_trials=20, n_channels=3, kind="cochlear", tmax=0.0035, seed=0
    ),
    "erna": lambda: sim.simulate_erna(n_trials=10, n_channels=2, seed=0),
    "erp": lambda: sim.simulate_erp(n_trials=30, seed=0),
    "p300_bci": lambda: sim.simulate_erp(n_trials=30, seed=0),
    "sep": lambda: sim.simulate_erp(
        n_trials=30, sfreq=5000.0, tmin=-0.05, tmax=0.1, components=SEP_COMPONENTS, seed=0
    ),
    "primate_ep": lambda: sim.simulate_animal_ep(
        n_trials=30, n_channels=4, species="macaque", seed=0
    ),
    "rodent_ep": lambda: sim.simulate_animal_ep(
        n_trials=30, n_channels=4, species="mouse", seed=0
    ),
    "tep": lambda: sim.simulate_tep(
        n_trials=20, n_channels=4, sfreq=1000.0, seed=0, **NO_ARTIFACT
    ),
}


def test_every_preset_has_a_simulated_modality() -> None:
    assert set(MODALITIES) == set(presets.PRESETS)


def _one_waveform(name: Any) -> Any:
    trials, truth = MODALITIES[name]()
    best = max(truth.components, key=lambda c: abs(c["amplitude"]))["channel"]
    return trials.select(picks=[best])


@pytest.fixture(scope="module")
def results() -> Any:
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in sorted(MODALITIES):
            out[name] = evoparam.fit(_one_waveform(name), preset=name)
    return out


def test_req001_same_types_for_every_preset(results: Any) -> None:
    for name, group in results.items():
        assert type(group) is GroupFit, name
        for fit_ in group.fits:
            assert type(fit_) is ResponseFit, name
            assert fit_.status in FIT_STATUSES, name
            assert all(type(c) is Component for c in fit_.components), name
            assert type(fit_.existence) is ResponseTest, name
        assert group.fits[0].status == "ok", (name, group.fits[0].message)


def test_req001_same_columns_for_every_preset(results: Any) -> None:
    columns = {name: list(group.to_dataframe().columns) for name, group in results.items()}
    reference = columns["ccep"]
    for name, names in columns.items():
        assert names == reference, name
    fit_columns = {
        name: list(group.to_dataframe(level="fit").columns) for name, group in results.items()
    }
    for name, names in fit_columns.items():
        assert names == fit_columns["ccep"], name


def test_req001_only_the_preset_differs(results: Any) -> None:
    for name, group in results.items():
        assert group.settings["preset"]["name"] == name


# REQ-002 -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trials() -> Any:
    ts, _ = sim.simulate_ccep(n_trials=10, n_channels=3, p_response=1.0, seed=4)
    return ts


def _bids(trials: Any, tmp_path_factory: Any) -> Any:
    from evoparam.sim.bids import write_bids

    _, truth = sim.simulate_ccep(n_trials=10, n_channels=3, p_response=1.0, seed=4)
    root = tmp_path_factory.mktemp("one_call") / "bids"
    write_bids(trials, truth, str(root), subject="01", task="ccep")
    return evoparam.read(str(root), subject="01", task="ccep")


def _inputs(kind: Any, trials: Any, tmp_path_factory: Any) -> Any:
    if kind == "TrialSet":
        return trials
    if kind == "numpy":
        return trials.average().data
    if kind == "BIDS":
        return _bids(trials, tmp_path_factory)
    pytest.importorskip("mne")
    epochs = trials.to_mne()
    return epochs if kind == "mne.Epochs" else epochs.average()


@pytest.mark.parametrize("kind", ["mne.Epochs", "mne.Evoked", "TrialSet", "BIDS", "numpy"])
def test_req002_one_call(kind: Any, trials: Any, tmp_path_factory: Any) -> None:
    data = _inputs(kind, trials, tmp_path_factory)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "numpy":
            group = evoparam.fit(data, preset="ccep", times=trials.times)
        else:
            group = evoparam.fit(data, preset="ccep")
    assert type(group) is GroupFit
    # existence results
    assert all(type(f.existence) is ResponseTest for f in group.fits)
    table = group.to_dataframe()
    assert table["existence_p"].notna().all()
    # components with their uncertainties
    ok = table[table["status"] == "ok"]
    assert len(ok) >= 1
    for name in ("latency", "amplitude", "fwhm"):
        assert ok[f"{name}_ci_low"].notna().all()
        assert ok[f"{name}_ci_high"].notna().all()
    # the methods paragraph
    text = group.methods_text()
    assert "evoparam" in text
    assert "ccep" in text
    assert len(text) > MIN_TEXT
    assert "presence of a response" in text
