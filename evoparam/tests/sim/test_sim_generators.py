# SPDX-License-Identifier: Apache-2.0
"""Generators: containers, provenance, settings, presets and the behaviour of each simulator."""

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from evoparam import sim
from evoparam.data._jsonutil import dumps
from evoparam.errors import SettingsError
from evoparam.shapes import compute_features, get_shape
from evoparam.sim.components import SHAPE_FORMS
from evoparam.tests.sim._helpers import SMALL

pd = pytest.importorskip("pandas")

MODALITIES = {
    "erp": "erp",
    "ccep": "spes",
    "erna": "erna",
    "tep": "tep",
    "ecap": "ecap",
    "animal": "lfp_ep",
}


###################################################################################################
## Every generator


@pytest.mark.parametrize("name", sorted(sim.GENERATORS))
def test_containers_and_provenance(name):
    gen = sim.GENERATORS[name]
    ts, truth = gen(seed=7, **SMALL[name])
    ts.validate()
    assert ts.units == "uV" == truth.units
    assert ts.modality == MODALITIES[name]
    assert ts.ch_names == truth.ch_names
    assert ts.sfreq == truth.sfreq
    np.testing.assert_array_equal(ts.times, truth.times)
    # Time axis on the grid k / sfreq, tmax excluded.
    np.testing.assert_allclose(ts.times * ts.sfreq, np.round(ts.times * ts.sfreq), atol=1e-6)
    assert ts.times[0] < 0 < ts.times[-1]
    assert truth.signal.shape == ts.data.shape
    assert truth.n_trials == ts.n_trials
    assert truth.n_components > 0
    assert len(truth.conditions) == ts.n_trials
    assert list(ts.metadata["trial"]) == list(range(ts.n_trials))
    assert list(ts.metadata["condition"]) == truth.conditions
    assert list(ts.ch_info["name"]) == ts.ch_names
    source = ts.provenance.source
    assert source["format"] == "simulation"
    assert source["generator"] == truth.generator == f"evoparam.sim.{gen.__name__}"
    step = ts.provenance.steps[-1]
    assert step["function"] == truth.generator
    json.loads(dumps(step["settings"]))  # settings are strict JSON
    assert step["settings"]["seed"] == 7
    for key, value in SMALL[name].items():
        assert truth.settings[key] == value
    assert np.isfinite(ts.data).all()


@pytest.mark.parametrize("name", sorted(sim.GENERATORS))
def test_truth_features_are_the_nominal_curves(name):
    _, truth = sim.GENERATORS[name](seed=7, **SMALL[name])
    for comp in truth.components[:6]:
        shape = get_shape(comp["shape"])
        assert list(comp["param_names"]) == list(shape.param_names)
        assert comp["loc_param"] in comp["param_names"]
        feats = compute_features(
            shape, truth.times, np.asarray(comp["params"]), comp["amplitude"], sfreq=truth.sfreq
        ).to_dict()
        assert np.isclose(comp["features"]["latency"], feats["latency"], atol=1e-12)
        assert np.isclose(comp["features"]["amplitude"], feats["amplitude"], rtol=1e-12)
        assert comp["role"] in sim.ROLES


@pytest.mark.parametrize("name", sorted(sim.GENERATORS))
def test_noise_level_is_noise_sd(name):
    """With snr=None the amplitudes are in µV and the noise SD is noise_sd."""

    ts, truth = sim.GENERATORS[name](seed=3, snr=None, noise="white", noise_sd=4.0, **SMALL[name])
    noise = ts.data - truth.signal
    if truth.artifact_data is not None:
        noise = noise - truth.artifact_data
    assert abs(noise.std() / 4.0 - 1) < 0.05
    assert np.allclose(truth.noise["baseline_sd"], 4.0, rtol=0.15)
    assert all(comp["snr"] is None for comp in truth.components)


def test_bad_counts():
    for gen, key in (
        (sim.simulate_erp, "n_trials"),
        (sim.simulate_ccep, "n_trials"),
        (sim.simulate_tep, "n_trials"),
        (sim.simulate_ecap, "n_trials"),
        (sim.simulate_animal_ep, "n_trials"),
        (sim.simulate_erna, "n_pulses"),
    ):
        for value in (0, 1.5, True):
            with pytest.raises(SettingsError, match=key):
                gen(**{key: value}, seed=0)


###################################################################################################
## Components and presets


@pytest.mark.parametrize("shape", sorted(SHAPE_FORMS))
def test_shape_params_reach_latency_and_fwhm(shape):
    params = sim.shape_params(shape, 0.2, 0.04)
    times = np.arange(-0.5, 1.5, 0.001)
    feats = compute_features(get_shape(shape), times, params, 1.0, sfreq=1000.0).to_dict()
    assert abs(feats["latency"] - 0.2) < 1e-6
    assert abs(feats["fwhm"] - 0.04) < 1e-5


def test_shape_params_errors():
    with pytest.raises(SettingsError):
        sim.shape_params("gaussian", 0.1, -0.01)
    with pytest.raises(SettingsError):
        sim.shape_params("no_such_shape", 0.1, 0.02)


def test_component_spec():
    spec = sim.ComponentSpec("X", latency=(0.1, 0.2), fwhm=0.03, amplitude=(-3.0, -1.0))
    rng = np.random.default_rng(0)
    for _ in range(20):
        params, amplitude = spec.resolve(rng)
        feats = compute_features(
            get_shape("gaussian"), np.arange(0, 0.5, 0.001), params, amplitude, sfreq=1000.0
        ).to_dict()
        assert 0.1 <= feats["latency"] <= 0.2
        assert -3.0 <= amplitude <= -1.0
    json.dumps(spec.to_dict())
    with pytest.raises(SettingsError):
        sim.ComponentSpec("")
    with pytest.raises(SettingsError):
        sim.ComponentSpec("X", role="noise")
    with pytest.raises(SettingsError):
        sim.ComponentSpec("X", channels=("Cz",), weights=(1.0, 2.0))
    with pytest.raises(SettingsError):
        sim.ComponentSpec("X", latency=(0.2, 0.1)).resolve(rng)
    explicit = sim.ComponentSpec("X", shape="gaussian", params=(0.1, 0.01), amplitude=2.0)
    params, amplitude = explicit.resolve(rng)
    assert list(params) == [0.1, 0.01]
    assert amplitude == 2.0
    with pytest.raises(SettingsError):
        sim.ComponentSpec("X", shape="gaussian", params=(0.1,)).resolve(rng)


def test_presets():
    assert set(sim.ERP_COMPONENTS) == {"P1", "N1", "N170", "MMN", "N400", "P3", "ERN"}
    assert list(sim.TEP_COMPONENTS) == ["N15", "P30", "N45", "P60", "N100", "P180"]
    assert sim.CCEP_N1.latency == (0.015, 0.040)
    assert sim.CCEP_N1.fwhm == (0.005, 0.015)
    assert sim.CCEP_N2.latency == (0.080, 0.250)
    assert sim.CCEP_N2.fwhm == (0.040, 0.150)
    assert sim.CCEP_DELAYED.latency == (0.100, 1.000)
    assert sim.ROLES == ("target", "overlap", "confound", "masker")


###################################################################################################
## ERP


def test_erp_components_and_overrides():
    ts, truth = sim.simulate_erp(
        seed=1,
        n_trials=5,
        n_channels=4,
        components=[
            "P1",
            {
                "label": "P3",
                "latency": 0.35,
                "fwhm": 0.2,
                "channels": ("Pz", "Cz"),
                "weights": (1.0, 0.5),
            },
        ],
    )
    assert ts.ch_names == ["Cz", "Pz", "Fz", "C3"]
    p3 = [c for c in truth.components if c["label"] == "P3"]
    assert {c["channel"] for c in p3} == {"Pz", "Cz"}
    for comp in p3:
        assert np.isclose(comp["features"]["latency"], 0.35)
        assert np.isclose(comp["features"]["fwhm"], 0.2, rtol=1e-4)
    p1 = [c for c in truth.components if c["label"] == "P1"]
    assert len(p1) == 4
    assert all(0.08 <= c["features"]["latency"] <= 0.13 for c in p1)
    ts, truth = sim.simulate_erp(seed=1, n_trials=3, n_channels=2, ch_names=["a", "b"])
    assert ts.ch_names == ["a", "b"]
    with pytest.raises(SettingsError):
        sim.simulate_erp(seed=1, n_trials=3, n_channels=2, ch_names=["a"])
    with pytest.raises(SettingsError, match="preset"):
        sim.simulate_erp(seed=1, n_trials=3, components=["N2pc"])
    with pytest.raises(SettingsError, match="channel"):
        sim.simulate_erp(seed=1, n_trials=3, components=[{"label": "P3", "channels": ("O9",)}])


def test_erp_noise_only():
    ts, truth = sim.simulate_erp(seed=1, n_trials=4, components=[], snr=None)
    assert truth.n_components == 0
    assert not np.any(truth.signal)


def test_erp_jitter_distributions():
    _, normal = sim.simulate_erp(
        seed=2, n_trials=400, components=["P3"], jitter_sd=0.02, noise="none", snr=None
    )
    shift = normal.shift[:, 0]
    assert abs(shift.std() - 0.02) < 0.003
    assert abs(shift.mean()) < 0.004
    _, exg = sim.simulate_erp(
        seed=2,
        n_trials=400,
        components=["P3"],
        jitter_sd=0.01,
        jitter="exgaussian",
        jitter_tau=0.03,
        noise="none",
        snr=None,
    )
    shift = exg.shift[:, 0]
    assert abs(shift.mean() - 0.03) < 0.006
    skew = np.mean((shift - shift.mean()) ** 3) / shift.std() ** 3
    assert skew > 0.5
    with pytest.raises(SettingsError):
        sim.simulate_erp(seed=2, n_trials=4, jitter="uniform", jitter_sd=0.01)


def test_erp_jitter_per_component():
    _, shared = sim.simulate_erp(seed=2, n_trials=50, components=["N1", "P3"], jitter_sd=0.01)
    np.testing.assert_array_equal(shared.shift[:, 0], shared.shift[:, 1])
    _, separate = sim.simulate_erp(
        seed=2, n_trials=50, components=["N1", "P3"], jitter_sd={"N1": 0.005, "P3": 0.02}
    )
    assert separate.shift[:, 0].std() < separate.shift[:, 1].std()


def test_erp_overlap():
    ts, truth = sim.simulate_erp(
        seed=4, n_trials=10, components=["P3"], overlap={"soa": (0.3, 0.5), "scale": 0.5}, snr=None
    )
    soa_prev = ts.metadata["soa_previous"].to_numpy()
    assert np.all((soa_prev >= 0.3) & (soa_prev <= 0.5))
    roles = {(c["role"], c["event"]) for c in truth.components}
    assert roles == {("target", 0), ("overlap", -1), ("overlap", 1)}
    main, previous = truth.components[0], truth.components[1]
    assert np.isclose(previous["amplitude"], 0.5 * main["amplitude"])
    np.testing.assert_allclose(truth.shift[:, 1] - truth.shift[:, 0], -soa_prev)
    with pytest.raises(SettingsError):
        sim.simulate_erp(seed=4, n_trials=3, overlap=-0.1)


###################################################################################################
## CCEP


def test_ccep_design():
    ts, truth = sim.simulate_ccep(
        seed=9, n_trials=4, n_channels=20, p_response=1.0, delayed=True, p_delayed=1.0, tmax=1.2
    )
    stim = truth.artifact["stim_contacts"]
    responders = truth.settings["responders"]
    assert not set(stim) & set(responders)
    assert len(responders) == 18
    labels = {c["label"] for c in truth.components}
    assert labels in ({"N1", "N2", "delayed"}, {"N1", "delayed"})
    for comp in truth.components:
        if comp["label"] == "N1":
            assert 0.015 - 1e-9 <= comp["features"]["latency"] <= 0.040 + 1e-9
            assert 0.005 - 1e-6 <= comp["features"]["fwhm"] <= 0.015 + 1e-6
            assert comp["amplitude"] < 0
    assert ts.metadata["stim_site"].iloc[0] == "-".join(stim)
    assert np.isclose(ts.metadata["stim_charge_uC"].iloc[0], 4.0 * 200 * 1e-3)
    assert ts.ch_info["stim_contact"].sum() == 2
    _, none = sim.simulate_ccep(seed=9, n_trials=2, n_channels=8, p_response=0.0, tmax=0.2)
    assert none.n_components == 0
    _, positive = sim.simulate_ccep(
        seed=9, n_trials=2, n_channels=8, p_response=1.0, polarity="positive", tmax=0.3
    )
    assert all(c["amplitude"] > 0 for c in positive.components)


def test_ccep_grid_and_errors():
    ts, _ = sim.simulate_ccep(seed=0, n_trials=2, n_channels=16, layout="grid", tmax=0.2)
    assert ts.ch_names[0] == "G01"
    assert set(ts.ch_info["type"]) == {"ecog"}
    with pytest.raises(SettingsError):
        sim.simulate_ccep(seed=0, layout="strip")
    with pytest.raises(SettingsError):
        sim.simulate_ccep(seed=0, polarity="both")
    with pytest.raises(SettingsError):
        sim.simulate_ccep(seed=0, p_response=1.5)
    with pytest.raises(SettingsError):
        sim.simulate_ccep(seed=0, n_channels=2)
    with pytest.raises(SettingsError):
        sim.simulate_ccep(seed=0, n1=3.0)


def test_ccep_overrides():
    _, truth = sim.simulate_ccep(
        seed=1,
        n_trials=2,
        n_channels=10,
        p_response=1.0,
        p_n2=0.0,
        n1={"latency": 0.02, "fwhm": 0.01, "amplitude": -100.0},
        tmax=0.2,
    )
    for comp in truth.components:
        assert comp["label"] == "N1"
        assert np.isclose(comp["features"]["latency"], 0.02)
        assert np.isclose(comp["amplitude"], -100.0)


def test_ccep_trains():
    ts, truth = sim.simulate_ccep(
        20,
        seed=0,
        n_channels=10,
        p_response=1.0,
        tmax=0.2,
        train_structure={"n_pulses": 10, "rate_hz": 0.5, "adaptation": -0.03},
    )
    meta = ts.metadata
    assert list(meta["index_in_train"][:10]) == list(range(10))
    assert list(meta["train_id"][[0, 9, 10]]) == [0, 0, 1]
    assert np.isclose(meta["stim_frequency_hz"].iloc[0], 0.5)
    np.testing.assert_allclose(np.diff(ts.event_times[:10]), 2.0)
    assert ts.event_times[10] - ts.event_times[9] > 2.0
    index = 0
    expected = 1.0 - 0.03 * np.tile(np.arange(10), 2)
    np.testing.assert_allclose(truth.scale[:, index], expected)
    with pytest.raises(SettingsError):
        sim.simulate_ccep(4, seed=0, train_structure={"n_pulses": 2, "rate_hz": -1.0})


###################################################################################################
## ERNA, TEP, ECAP, animal


def test_erna_pulses():
    ts, truth = sim.simulate_erna(n_pulses=5, n_trials=2, seed=0, noise="none", artifact=False)
    assert ts.modality == "erna"
    events = sorted({c["event"] for c in truth.components})
    assert events == list(range(5))
    freq = {c["params"][2] for c in truth.components}
    assert len(freq) == 1
    assert 200.0 <= freq.pop() <= 450.0
    first = [c for c in truth.components if c["channel"] == ts.ch_names[1]]
    amplitudes = [abs(c["amplitude"]) for c in sorted(first, key=lambda c: c["event"])]
    assert all(np.diff(amplitudes) > 0)  # build-up over the train
    assert truth.components[0]["shape"] == "damped_oscillation"
    assert ts.metadata["stim_n_pulses"].iloc[0] == 5
    with pytest.raises(SettingsError, match="4096"):
        sim.simulate_erna(n_pulses=2, sfreq=2048.0)
    with pytest.raises(SettingsError):
        sim.simulate_erna(n_pulses=2, pulse_rate=0.0)


def test_tep_components_and_site():
    ts, truth = sim.simulate_tep(
        seed=0,
        n_trials=3,
        n_channels=8,
        sfreq=2000.0,
        noise="none",
        auditory=False,
        pulse_artifact=False,
        decay_artifact=False,
        muscle_artifact=False,
    )
    assert {c["label"] for c in truth.components} == set(sim.TEP_COMPONENTS)
    assert all(c["role"] == "target" for c in truth.components)
    n100 = [c for c in truth.components if c["label"] == "N100"]
    peak = max(n100, key=lambda c: abs(c["amplitude"]))
    assert peak["channel"] == "C3"
    with pytest.raises(SettingsError, match="stim_site"):
        sim.simulate_tep(seed=0, n_trials=2, n_channels=4, stim_site="O2")


def test_ecap_masking_and_paradigms():
    ts, truth = sim.simulate_ecap(8, seed=0, paradigm="forward_masking", masking=1.0, noise="none")
    probe = [i for i, c in enumerate(truth.components) if c["role"] == "target"]
    # Full masking: no probe response in condition B (masker + probe).
    b_trials = [t for t, cond in enumerate(truth.conditions) if cond == "B"]
    assert not truth.present[np.ix_(b_trials, probe)].any()
    _, half = sim.simulate_ecap(8, seed=0, paradigm="forward_masking", masking=0.5, noise="none")
    np.testing.assert_allclose(half.scale[np.ix_(b_trials, probe)], 0.5)
    maskers = [i for i, c in enumerate(truth.components) if c["role"] == "masker"]
    for trial, cond in enumerate(truth.conditions):
        assert truth.present[trial, maskers].all() == (cond in ("B", "C"))
        assert truth.present[trial, probe].all() == (cond == "A")
    assert list(ts.metadata["pair_id"][:4]) == [0, 0, 0, 0]
    with pytest.raises(SettingsError):
        sim.simulate_ecap(6, seed=0, paradigm="forward_masking")
    with pytest.raises(SettingsError):
        sim.simulate_ecap(5, seed=0, paradigm="alternating")
    with pytest.raises(SettingsError):
        sim.simulate_ecap(8, seed=0, masking=2.0)


def test_animal_depth_reversal_and_states():
    ts, truth = sim.simulate_animal_ep(
        seed=2, n_trials=6, n_channels=24, spacing_um=40.0, noise="none", reversal_um=400.0
    )
    assert ts.species == "mouse"
    assert set(ts.metadata["state"]) == {"awake", "nrem"}
    first = truth.components[0]["label"]
    amps = [c["amplitude"] for c in truth.components if c["label"] == first]
    assert np.sign(amps[0]) == -np.sign(amps[-1])  # polarity reverses with depth
    _, custom = sim.simulate_animal_ep(
        seed=2,
        n_trials=6,
        n_channels=4,
        states=("a", "b"),
        state_gain={"a": 1.0, "b": 3.0},
        noise="none",
    )
    assert set(custom.conditions) == {"a", "b"}
    ts, truth = sim.simulate_animal_ep(
        seed=2, n_trials=4, n_channels=4, stimulus="electrical", noise="none"
    )
    assert truth.artifact is not None
    assert truth.artifact_data.any()
    with pytest.raises(SettingsError):
        sim.simulate_animal_ep(seed=2, stimulus="auditory")
    with pytest.raises(SettingsError):
        sim.simulate_animal_ep(seed=2, states=("awake", "rem"))
    with pytest.raises(SettingsError):
        sim.simulate_animal_ep(seed=2, states=())


###################################################################################################
## Light core


def test_generators_without_pandas_in_a_subprocess():
    """Optional dependencies stay optional: the simulators run without pandas (metadata and
    ch_info are then None) and the tables raise ImportError naming the extra."""

    code = textwrap.dedent("""
        import sys
        for name in ("pandas", "pyarrow", "mne"):
            sys.modules[name] = None
        import numpy as np
        from evoparam import sim
        ts, truth = sim.simulate_erp(4, seed=0)
        ts.validate()
        assert ts.metadata is None and ts.ch_info is None
        ts, truth = sim.simulate_ccep(2, n_channels=8, seed=0, tmax=0.2)
        assert ts.data.shape[:2] == (2, 8)
        noise = sim.make_noise(100, 100., "pink", seed=0)
        assert noise.shape == (100,)
        truth.to_json()
        try:
            truth.to_dataframe()
        except ImportError as exc:
            assert 'pip install "evoparam[data]"' in str(exc), str(exc)
        else:
            raise AssertionError("to_dataframe did not raise ImportError")
        print("ok")
    """)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
