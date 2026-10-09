"""GroundTruth: REQ-255 (same schema as GroupFit) and REQ-256 (real noise record and licences)."""

import copy

import numpy as np
import pytest

from evoparam import sim
from evoparam.data.results import COMPONENT_COLUMNS
from evoparam.errors import DataError, SettingsError
from evoparam.shapes import compute_features, get_shape
from evoparam.tests.sim._helpers import SMALL, fits_from_truth, real_baselines

pd = pytest.importorskip("pandas")


###################################################################################################
## REQ-255: shared columns and units

def _kind(series):
    kind = series.dtype.kind
    return {"i": "int", "u": "int", "f": "float", "b": "bool"}.get(kind, "object")


@pytest.mark.parametrize("name", sorted(sim.GENERATORS))
def test_truth_and_groupfit_share_columns_and_units(name):
    _, truth = sim.GENERATORS[name](seed=8, **SMALL[name])
    fit_table = fits_from_truth(truth).to_dataframe()
    for level in ("component", "trial"):
        table = truth.to_dataframe(level)
        common = [c for c in fit_table.columns if c in table.columns]
        # Every GroupFit component column but the fit status is in the truth.
        assert common == [c for c in COMPONENT_COLUMNS if c != "status"]
        assert list(table.columns[:len(common)]) == common
        for column in common:
            assert table.attrs["units"][column] == fit_table.attrs["units"][column], column
            if len(table) and len(fit_table):
                assert _kind(table[column]) == _kind(fit_table[column]), column
        assert table.attrs["data_units"] == truth.units
        for column in table.columns:
            assert column in table.attrs["units"]
            if column.startswith("param_"):
                assert table.attrs["units"][column] in ("s", "", "Hz", "rad")


def test_perfect_fit_table_equals_truth_table():
    """A fit made of the true components gives the truth's values in every shared column."""

    _, truth = sim.simulate_ccep(seed=2, **SMALL["ccep"])
    truth_table = truth.to_dataframe()
    fit_table = fits_from_truth(truth).to_dataframe()
    keys = ["channel", "latency"]
    merged = truth_table.merge(fit_table, on=keys, suffixes=("_t", "_f"))
    assert len(merged) == len(truth_table)
    for column in ("amplitude", "fwhm", "onset", "offset", "area", "polarity"):
        np.testing.assert_allclose(merged[f"{column}_t"], merged[f"{column}_f"])


###################################################################################################
## Exactness of the truth

@pytest.mark.parametrize("name, jitter", [("erp", 0.004), ("ccep", 0.004), ("ecap", 0.0001)])
def test_trial_rows_are_the_features_of_the_trial_curves(name, jitter):
    """Features of a trial are those of its curve (components inside the epoch)."""

    kwargs = dict(SMALL[name], jitter_sd=jitter, amplitude_cv=0.2)
    ts, truth = sim.GENERATORS[name](seed=4, **kwargs)
    table = truth.to_dataframe("trial")
    assert len(table) == int(truth.present.sum())
    order = {i: c for i, c in enumerate(truth.components)}
    rows = table.to_dict("records")
    checked = 0
    for index, comp in order.items():
        params = truth.trial_params(index)
        shape = get_shape(comp["shape"])
        for trial in np.flatnonzero(truth.present[:, index])[:3]:
            amp = comp["amplitude"] * truth.scale[trial, index]
            feats = compute_features(shape, truth.times, params[trial], amp,
                                     sfreq=truth.sfreq).to_dict()
            match = [r for r in rows if r["trial"] == trial and r["channel"] == comp["channel"]
                     and r["label"] == comp["label"] and r["event"] == comp["event"]
                     and np.isclose(r["shift"], truth.shift[trial, index])]
            assert match, (comp["label"], trial)
            row = match[0]
            np.testing.assert_allclose(row["latency"], feats["latency"], rtol=0, atol=1e-9)
            np.testing.assert_allclose(row["amplitude"], feats["amplitude"], rtol=1e-9)
            for column in ("fwhm", "onset", "offset"):
                np.testing.assert_allclose(row[column], feats[column], rtol=0,
                                           atol=0.2 / truth.sfreq)
            checked += 1
    assert checked > 0


def test_signal_is_the_sum_of_the_waveforms():
    ts, truth = sim.simulate_erp(seed=6, n_trials=8, n_channels=2, jitter_sd=0.01,
                                 amplitude_cv=0.3, overlap=0.4)
    total = np.zeros_like(truth.signal)
    for index, comp in enumerate(truth.components):
        total[:, truth.ch_names.index(comp["channel"])] += truth.waveform(index)
    np.testing.assert_allclose(total, truth.signal, atol=1e-9 * np.abs(truth.signal).max())
    np.testing.assert_allclose(truth.waveform(0, trial=3), truth.waveform(0)[3])
    assert {c["event"] for c in truth.components} == {-1, 0, 1}
    assert {"soa_previous", "soa_next"} <= set(ts.metadata.columns)


def test_component_level_averages_the_trials():
    _, truth = sim.simulate_erp(seed=6, n_trials=30, jitter_sd=0.005, amplitude_cv=0.2,
                                components=["P3"])
    row = truth.to_dataframe().iloc[0]
    comp = truth.components[0]
    assert row["n_trials"] == 30
    assert np.isclose(row["shift"], truth.shift[:, 0].mean())
    assert np.isclose(row["latency"], comp["features"]["latency"] + truth.shift[:, 0].mean())
    assert np.isclose(row["amplitude"], comp["amplitude"] * truth.scale[:, 0].mean())
    assert np.isclose(row["snr"], comp["snr"] * truth.scale[:, 0].mean())
    assert abs(np.log(truth.scale).std() - 0.2) < 0.1


def test_conditions_split_the_component_level():
    _, truth = sim.simulate_animal_ep(seed=1, **SMALL["animal"])
    table = truth.to_dataframe()
    assert set(table["condition"]) == {"awake", "nrem"}
    one = table[(table["channel"] == table["channel"].iloc[0])]
    awake = one[one["condition"] == "awake"]["amplitude"].to_numpy()
    nrem = one[one["condition"] == "nrem"]["amplitude"].to_numpy()
    np.testing.assert_allclose(nrem, 2.0 * awake)


def test_json_round_trip():
    _, truth = sim.simulate_tep(seed=3, **SMALL["tep"])
    expected = copy.deepcopy(truth)
    expected.signal, expected.artifact_data = None, None
    again = sim.GroundTruth.from_json(truth.to_json())
    assert again == expected
    with_arrays = sim.GroundTruth.from_dict(truth.to_dict(arrays=True))
    assert with_arrays == copy.deepcopy(truth)
    import json
    json.loads(truth.to_json())                      # strict JSON, no NaN tokens
    assert "NaN" not in truth.to_json()


@pytest.mark.parametrize("suffix", ["", ".zip"])
def test_container_round_trip(tmp_path, suffix):
    """Spec §4.3.5: the ground truth is saved like every result, as a ``.evoparam``
    container (manifest, arrays in npz, provenance and settings), and loads back equal,
    arrays included, with GroundTruth.load and with evoparam.load_object."""

    import json

    from evoparam import load_object

    for name in ("tep", "ccep"):
        _, truth = sim.GENERATORS[name](seed=3, **SMALL[name])
        path = truth.save(str(tmp_path / f"truth_{name}{suffix}"))
        assert path.endswith(".evoparam" + suffix if not suffix else suffix)
        again = sim.GroundTruth.load(path)
        assert again == truth
        assert load_object(path) == truth
        if not suffix:
            with open(f"{path}/manifest.json", encoding="utf-8") as fobj:
                assert json.load(fobj)["object"] == "GroundTruth"
    with pytest.raises(DataError, match="not a GroundTruth"):
        trials, _ = sim.simulate_erp(seed=0, n_trials=3)
        sim.GroundTruth.load(trials.save(str(tmp_path / "trials")))


def test_truth_validation_and_repr():
    _, truth = sim.simulate_erp(seed=0, n_trials=3)
    assert "3 trials" in repr(truth) and "seed 0" in repr(truth)
    bad = truth.to_dict(arrays=False)
    bad["shift"] = [[0.0]]
    with pytest.raises(DataError):
        sim.GroundTruth.from_dict(bad)
    with pytest.raises(DataError):
        truth.to_dataframe("channel")
    empty = sim.simulate_erp(seed=0, n_trials=3, components=[], snr=None)[1]
    assert empty.n_components == 0 and len(empty.to_dataframe()) == 0


def test_keep_arrays_false():
    ts, truth = sim.simulate_erp(seed=0, n_trials=3, keep_arrays=False)
    assert truth.signal is None and truth.artifact_data is None
    ts_full, truth_full = sim.simulate_erp(seed=0, n_trials=3)
    assert ts.data.tobytes() == ts_full.data.tobytes()


###################################################################################################
## REQ-256: real noise records and licences

def test_real_noise_records_every_segment():
    real = real_baselines(n_trials=6, n_channels=3, sfreq=500., tmin=-2.0)
    ts, truth = sim.simulate_erp(seed=2, n_trials=5, n_channels=3, noise="real",
                                 noise_kw={"source": real}, snr=3.0)
    record = truth.noise
    assert record["kind"] == "real"
    assert record["dataset"]["dataset"] == "ds004080"
    assert record["dataset"]["version"] == "1.2.4"
    assert record["dataset"]["license"] == "CC0"
    segments = record["segments"]
    assert len(segments) == ts.n_trials
    for trial, segment in enumerate(segments):
        for key in ("dataset", "version", "subject", "run", "source_trial", "epoch_samples",
                    "recording_samples", "source_channels"):
            assert segment[key] is not None, key
        assert segment["dataset"] == "ds004080" and segment["version"] == "1.2.4"
        assert segment["subject"] == "ccepAgeUMCU01" and segment["run"] == "021448"
        start, stop = segment["epoch_samples"]
        assert stop - start == ts.n_times
        epoch_start = int(round(real.event_times[segment["source_trial"]] * real.sfreq
                                + real.times[0] * real.sfreq))
        assert segment["recording_samples"] == [epoch_start + start, epoch_start + stop]
        # The noise in the data is exactly this segment (demeaned, scaled to noise_sd).
        noise = ts.data[trial] - truth.signal[trial]
        piece = real.data[segment["source_trial"], :, start:stop]
        piece = piece - piece.mean(axis=-1, keepdims=True)
        ratio = noise / piece
        assert np.allclose(ratio, ratio.flat[0], rtol=1e-9)
    json_record = sim.GroundTruth.from_json(truth.to_json()).noise
    assert json_record["segments"][0]["subject"] == "ccepAgeUMCU01"


def test_real_noise_resampling_is_recorded():
    real = real_baselines(n_trials=4, n_channels=2, sfreq=1000., tmin=-2.0)
    ts, truth = sim.simulate_erp(seed=2, n_trials=3, n_channels=2, sfreq=500., noise="real",
                                 noise_kw={"source": real}, snr=3.0)
    resampling = truth.noise["resampling"]
    assert resampling["from_sfreq"] == 1000.0 and resampling["to_sfreq"] == 500.0
    assert (resampling["up"], resampling["down"]) == (1, 2)
    start, stop = truth.noise["segments"][0]["epoch_samples"]
    assert stop - start == 2 * ts.n_times                   # in source samples


def test_real_noise_unit_conversion():
    real = real_baselines(n_trials=4, n_channels=2, units="V")
    noise, info = sim.make_noise(400, 500., "real", size=(2, 2), source=real, units="uV",
                                 seed=0, return_info=True)
    segment = info["segments"][0]
    start, stop = segment["epoch_samples"]
    piece = real.data[segment["source_trial"], :, start:stop] * 1e6
    np.testing.assert_allclose(noise[0], piece - piece.mean(axis=-1, keepdims=True))
    assert info["source_units"] == "V"


@pytest.mark.parametrize("dataset", ["localize-mi"])
def test_registry_datasets_without_an_allowed_licence_are_refused(dataset):
    real = real_baselines(dataset=dataset)
    with pytest.raises(SettingsError, match="licence"):
        sim.simulate_erp(seed=0, n_trials=3, n_channels=3, noise="real",
                         noise_kw={"source": real})
    with pytest.raises(SettingsError, match="licence"):
        sim.inject(real, ["N1"], snr=2.0, seed=0)


def test_unknown_or_unnamed_dataset_is_refused_for_real_noise():
    with pytest.raises(SettingsError, match="not in the registry"):
        sim.make_noise(400, 500., "real", size=(1, 3), source=real_baselines(dataset="ds999999"))
    with pytest.raises(SettingsError, match="does not name"):
        sim.make_noise(400, 500., "real", size=(1, 3), source=real_baselines(dataset=None))
    # Naming the dataset explicitly is enough.
    noise = sim.make_noise(400, 500., "real", size=(1, 3), source=real_baselines(dataset=None),
                           dataset="ds004080", seed=0)
    assert noise.shape == (1, 3, 400)


@pytest.mark.parametrize("licence, allowed", [("CC-BY-NC-4.0", False), ("CC BY-NC-SA 4.0", False),
                                              ("proprietary", False), (None, False),
                                              ("CC-BY-4.0", True), ("CC BY-SA 4.0", True),
                                              ("CC0 1.0", True)])
def test_licence_rule(monkeypatch, licence, allowed):
    from evoparam.io import registry
    original = registry.dataset_info

    def patched(dataset_id):
        entry = original("ds004080")
        entry["license"] = licence
        return entry

    monkeypatch.setattr(registry, "dataset_info", patched)
    real = real_baselines()
    if allowed:
        _, truth = sim.inject(real, ["N1"], snr=2.0, seed=0)
        assert truth.noise["dataset"]["license"] == licence
    else:
        with pytest.raises(SettingsError, match="licence"):
            sim.inject(real, ["N1"], snr=2.0, seed=0)


def test_normalize_license():
    assert sim.noise.normalize_license("CC0 1.0") == "CC0"
    assert sim.noise.normalize_license("cc-by-4.0") == "CC-BY"
    assert sim.noise.normalize_license("CC BY-SA 4.0") == "CC-BY-SA"
    assert sim.noise.normalize_license("CC_BY_NC") == "CC-BY-NC"
    assert sim.noise.normalize_license(None) is None
    assert sim.noise.normalize_license("4.0") is None
    assert sim.ALLOWED_LICENSES == ("CC0", "CC-BY", "CC-BY-SA")


def test_inject_records_and_adds_components():
    real = real_baselines(n_trials=6, n_channels=3)
    ts, truth = sim.inject(real, ["N1", {"label": "P3", "channels": ("R01",)}], snr=4.0,
                           seed=1)
    np.testing.assert_allclose(ts.data - truth.signal, real.data)
    assert ts.metadata.equals(real.metadata)
    assert ts.provenance.steps[-1]["function"] == "evoparam.sim.inject"
    segments = truth.noise["segments"]
    assert [s["source_trial"] for s in segments] == list(range(6))
    assert segments[0]["recording_samples"] == [int(round(10.0 * 500 - 2.0 * 500)),
                                                int(round(10.0 * 500 - 2.0 * 500)) + real.n_times]
    labels = [(c["label"], c["channel"]) for c in truth.components]
    assert ("P3", "R01") in labels and ("N1", "R00") in labels and ("N1", "R02") in labels
    for index, comp in enumerate(truth.components):
        sd = np.std(real.data[:, truth.ch_names.index(comp["channel"]), real.times < 0],
                    axis=-1).mean()
        peak = np.max(np.abs(truth.waveform(index)), axis=-1).mean()
        assert abs(peak / sd / comp["snr"] - 1) < 0.02
    # The user's own data (no dataset named) are accepted and recorded as such.
    own = real_baselines(dataset=None)
    _, truth = sim.inject(own, ["N1"], snr=2.0, seed=0)
    assert truth.noise["dataset"]["dataset"] is None
    with pytest.raises(DataError):
        sim.inject(real.data, ["N1"], snr=2.0)
