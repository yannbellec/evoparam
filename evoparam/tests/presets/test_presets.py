"""Presets: the table of spec §6.1, citations, derive and the generated page (REQ-270 to
REQ-272, REQ-279, REQ-280)."""

import pathlib
import re
import warnings

import numpy as np
import pytest

from evoparam import presets
from evoparam.errors import SettingsError

ROOT = pathlib.Path(__file__).parents[3]
SPEC = ROOT / "docs" / "spec" / "SPEC.md"
LITERATURE = ROOT / "docs" / "spec" / "LITERATURE.md"
needs_docs = pytest.mark.skipif(not SPEC.exists(), reason="the specification is not in this "
                                "checkout (installed package)")

# Values that differ from the table on purpose; each is explained in the preset's notes.
DEVIATIONS = {("sep", "background"): "constant",
              ("tep", "shapes"): ("gaussian", "skewed_gaussian")}

ARTIFACT = {"none": None, "`template` + `tail=\"joint\"`": "template+tail:joint",
            "`interpolate`": "interpolate", "`interpolate` then `subspace`": "interpolate+subspace",
            "`period_template`": "period_template",
            "`paradigm` (alternating polarity)": "paradigm:alternating",
            "`paradigm` (forward masking)": "paradigm:forward_masking", "`template`": "template"}
REFERENCE = {"as recorded": "as_recorded", "`car_adaptive`": "car_adaptive",
             "`bipolar`": "bipolar", "average": "average",
             "bipolar (adjacent contacts)": "bipolar"}


def _number(text):
    return float(text.replace("−", "-"))


def _tuples(cell):
    return [(_number(a), _number(b)) for a, b in
            re.findall(r"\(([−\-\d.]+),\s*([−\-\d.]+)\)", cell)]


def _table():
    lines = SPEC.read_text("utf-8").split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("### 6.1"))
    rows = [l for l in lines[start:start + 40] if l.startswith("| ")]
    names = [c.split()[0].strip("`") for c in rows[0].strip("|").split("|")[1:]]
    out = {n: {} for n in names}
    for row in rows[1:]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        for name, cell in zip(names, cells[1:]):
            out[name][cells[0]] = cell
    # "as rodent", "as `ccep`": copy the referenced column
    for name, values in out.items():
        for key, cell in values.items():
            match = re.fullmatch(r"as (`?)(\w+)\1", cell)
            if match and key not in ("reference",) and match.group(2) in ("rodent", "ccep"):
                ref = "rodent_ep" if match.group(2) == "rodent" else "ccep"
                values[key] = out[ref][key]
    return out


def _expected(name, row):
    window = lambda cell: (_tuples(cell) or [None])[0]
    labels = {}
    for label, lo, hi in re.findall(r"([A-Za-z0-9/]+)(?: by depth)? \(([\d.]+)–([\d.]+)\)",
                                    row["component labels (s)"]):
        labels[label] = (float(lo), float(hi))
    keys = tuple(re.findall(r"\[(LIT-[\w-]+)\]", row["key sources"]))
    return {"epoch": window(row["epoch (s)"]), "baseline": window(row["baseline (s)"]),
            "artifact_window": window(row["artifact window (s)"]),
            "artifact_method": ARTIFACT.get(row["artifact method"].split(";")[0].strip(),
                                            row["artifact method"]),
            "reference": REFERENCE.get(row["reference"], row["reference"]),
            "fit_window": window(row["fit window (s)"]),
            "shapes": tuple(s.strip() for s in row["shapes"].split(",")),
            "background": row["background"], "max_components": int(row["max components"]),
            "min_fwhm": float(row["min FWHM (s)"]), "max_fwhm": float(row["max FWHM (s)"]),
            "component_labels": labels, "trial_fit": row["trial fit"].startswith("yes"),
            "species": tuple(s.strip() for s in row["species"].split(",")),
            "citation": keys}


@needs_docs
def test_req270_every_preset_matches_the_table():

    table = _table()
    assert set(table) == set(presets.PRESETS)
    for name, row in table.items():
        p = presets.get(name)
        assert dataclass_frozen(p)
        for field, value in _expected(name, row).items():
            got = getattr(p, field)
            if (name, field) in DEVIATIONS:
                assert got == DEVIATIONS[(name, field)]
                assert field in p.notes, f"{name}.{field}: deviation not explained"
                continue
            if field == "citation":
                assert set(got) == set(value), (name, field)
            elif isinstance(value, tuple) and value and isinstance(value[0], float):
                assert np.allclose(got, value), (name, field, got, value)
            else:
                assert got == value, (name, field, got, value)


def dataclass_frozen(p):
    with pytest.raises(Exception):
        p.name = "other"
    return True


@needs_docs
def test_req271_citations_resolve_and_choices_are_explained():
    """REQ-271, and REQ-014 of §3.1 (presets cite their sources), which is REQ-271."""

    keys = set(re.findall(r"^- \*\*\[(LIT-[\w-]+)\]\*\*", LITERATURE.read_text("utf-8"),
                          flags=re.M))
    assert len(keys) > 500
    for p in presets.PRESETS.values():
        assert p.citation, p.name
        assert set(p.citation) <= keys, (p.name, set(p.citation) - keys)
        for field, reason in p.choices.items():
            assert field in p.fields and reason.strip(), (p.name, field)


def test_derive_and_registry():

    parent = presets.get("ccep")
    before = parent.to_dict()
    child = presets.derive("ccep", fit_window=(0.012, 0.5), name="ccep_late")
    assert parent.to_dict() == before
    assert child.parent == "ccep" and child.overrides["fit_window"] == (0.012, 0.5)
    assert child.fit_window == (0.012, 0.5) and parent.fit_window == (0.009, 0.5)
    with pytest.raises(SettingsError, match="Unknown preset field"):
        presets.derive("ccep", fit_windw=(0.0, 1.0))
    with pytest.raises(SettingsError, match="known"):
        presets.get("nope")
    with pytest.raises(SettingsError, match="exists"):
        presets.register(presets.derive("ccep"))
    presets.register(child)
    try:
        assert presets.get("ccep_late") is child
    finally:
        presets.PRESETS.pop("ccep_late")
    assert presets.from_json(presets.to_json("tep")) == presets.get("tep")


def _waveform():
    from evoparam.data import Evoked

    sfreq = 2048.0
    times = np.arange(-0.5, 0.6, 1 / sfreq)
    rng = np.random.default_rng(0)
    y = -40 * np.exp(-0.5 * ((times - 0.025) / 0.004) ** 2) \
        - 20 * np.exp(-0.5 * ((times - 0.12) / 0.03) ** 2) + rng.normal(0, 3, times.size)
    return Evoked(data=y[None], times=times, sfreq=sfreq, ch_names=["c0"])


def test_req272_fit_with_overrides_equals_fit_with_derived_preset():

    from evoparam.models import fit

    ev = _waveform()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = fit(ev, "ccep", fit_window=(0.012, 0.5)).fits[0]
        b = fit(ev, presets.derive("ccep", fit_window=(0.012, 0.5))).fits[0]
    assert len(a.components) == len(b.components) >= 1
    for ca, cb in zip(a.components, b.components):
        assert ca.label == cb.label
        assert np.allclose(ca.params, cb.params) and ca.amplitude == cb.amplitude


def test_req279_simulation_only_presets_say_so_in_the_methods_text():

    from evoparam.models import methods_text, resolve_settings

    for name in ("ecap", "ecap_ci", "erna"):
        assert presets.get(name).validation == "simulation_only"
        assert "simulations only" in resolve_settings(name).preset_note
    from evoparam.models import fit

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = fit(_waveform(), "ccep")
    assert "preset 'ccep' v1.1" in methods_text(group.fits[0])


@needs_docs
def test_req280_presets_page_is_generated():

    import importlib.util

    spec = importlib.util.spec_from_file_location("gen", ROOT / "tools" / "gen_presets_doc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    page = (ROOT / "docs" / "presets.md").read_text("utf-8")
    assert page == module.render(), "run python tools/gen_presets_doc.py"


def test_fit_presets_view_follows_the_files():

    from evoparam.models import FIT_PRESETS

    assert set(FIT_PRESETS) == set(presets.PRESETS)
    assert FIT_PRESETS["ccep"]["windows"]["N1"] == (0.009, 0.05, -1)
    assert FIT_PRESETS["erna"]["freq_min"] == 200.0


###################################################################################################
## Preset behaviours of spec §6.2 (audit F58), applied by the preset pipeline

def test_f58_preset_fields():

    assert presets.get("ccep_seeg").reference_sensitivity
    assert not any(presets.get(n).reference_sensitivity for n in presets.PRESETS
                   if n != "ccep_seeg")
    assert presets.get("rodent_ep").trial_covariates == ("state",)
    assert presets.get("primate_ep").trial_covariates == ("state",)
    assert presets.get("erna").train_baseline == 0.05 and presets.get("erna").baseline is None
    assert presets.from_json(presets.to_json("erna")) == presets.get("erna")
    with pytest.raises(SettingsError, match="train_baseline"):
        presets.derive("erna", train_baseline=-1.0)


def test_f58_ccep_seeg_attaches_the_reference_sensitivity_report():
    """§6.2: for ccep_seeg, reference_sensitivity (§5.4) is run by default and its kappa
    report attached."""

    from evoparam import reports, sim

    ts, _ = sim.simulate_ccep(n_trials=8, n_channels=8, sfreq=1024.0, p_response=0.5, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = reports.run_pipeline(ts, "ccep_seeg", trial_fits=False)
        plain = reports.run_pipeline(ts, "ccep_seeg", trial_fits=False,
                                     reference_sensitivity=False)
        ccep = reports.run_pipeline(ts, "ccep", trial_fits=False)
    report = result.reference_report
    assert list(report[["scheme_a", "scheme_b"]].itertuples(index=False, name=None)) == [
        ("bipolar", "car"), ("bipolar", "car_adaptive"), ("car", "car_adaptive")]
    assert report["kappa"].notna().any() and "channels" in report.attrs
    assert plain.reference_report is None and ccep.reference_report is None
    text = reports.methods_text(result)
    assert "dependence of the detections on the reference" in text
    assert "dependence of the detections" not in reports.methods_text(ccep)
    # the report does not change the fits (it runs on a copy of the cleaned trials)
    assert np.allclose(result.group.to_dataframe()["latency"].to_numpy(float),
                       plain.group.to_dataframe()["latency"].to_numpy(float), equal_nan=True)


def test_f58_rodent_state_is_a_default_trial_covariate():
    """§6.2: metadata["state"] is a default covariate in the trial table (rodent_ep and
    primate_ep)."""

    import dataclasses

    from evoparam import reports, sim

    ts, _ = sim.simulate_animal_ep(n_trials=40, n_channels=4, species="mouse", seed=0)
    assert set(ts.metadata["state"]) == {"awake", "nrem"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = reports.run_pipeline(ts, "rodent_ep")
    fits = result.trial_fits
    assert fits.table.groupby("state")["trusted"].sum().min() >= 2
    assert fits.settings["covariates"] == ["state"]
    table = fits.settings["covariate_tables"]["state"]
    assert set(table["state"]) == {"awake", "nrem"}
    gains = np.asarray(table["gain"], float)
    ref = np.asarray(table["state"]) == np.asarray(table["reference"])
    assert np.allclose(gains[ref], 1.0) and np.isfinite(table["p_amplitude"]).all()
    assert "state" in fits.table.columns
    assert "Trial covariates: state." in reports.methods_text(result)
    # without the column: a warning, and the trial fits run without the covariate
    no_state = dataclasses.replace(ts, metadata=ts.metadata.drop(columns="state"))
    with pytest.warns(UserWarning, match="metadata\\['state'\\] is missing"):
        result = reports.run_pipeline(no_state, "rodent_ep", existence=False)
    assert result.trial_fits.settings["covariates"] == []


def _pulse_trials(sfreq=4096.0, tmin=-0.002, tmax=0.008, levels=(5.0, -3.0, 8.0)):
    """Per-pulse epochs of three 130 Hz trains of 10 pulses, 1 s apart, in a recording whose
    DC level differs before each train (and stays during it), with noise from each train's
    onset on; returns the epochs, the recording and the epochs of the noise alone."""

    import dataclasses

    from evoparam.io import epoch

    n = int(4.0 * sfreq)
    raw = np.zeros((2, n))
    noise = np.zeros((2, n))
    rng = np.random.default_rng(0)
    onsets = []
    for k, level in enumerate(levels):
        start = 0.5 + k * 1.0
        raw[:, int((start - 0.1) * sfreq):int((start + 0.2) * sfreq)] = level * np.array(
            [[1.0], [2.0]])
        first, last = int(np.ceil(start * sfreq)), int((start + 0.2) * sfreq)
        noise[:, first:last] = rng.normal(0, 1, (2, last - first))
        onsets += [start + i / 130.0 for i in range(10)]
    raw = raw + noise
    ts = epoch(raw, sfreq, np.array(onsets), tmin, tmax, ch_names=["c0", "c1"])
    expected = epoch(noise, sfreq, np.array(onsets), tmin, tmax, ch_names=["c0", "c1"]).data
    meta = ts.metadata.assign(train_id=np.repeat(np.arange(3), 10))
    return dataclasses.replace(ts, metadata=meta), raw, expected


def test_f58_erna_baseline_per_train():
    """§6.1/§6.2: erna's baseline is per train, the last 50 ms before the train."""

    from evoparam import reports
    from evoparam.errors import DataError
    from evoparam.prep import train_baseline

    ts, raw, expected = _pulse_trials()
    out = train_baseline(ts, 0.05, raw=raw, sfreq=4096.0)
    assert np.allclose(out.data, expected)         # each train's level removed from its pulses
    step = out.provenance.steps[-1]
    assert step["function"] == "evoparam.prep.train_baseline"
    assert step["settings"]["source"] == "continuous" and step["settings"]["n_trains"] == 3
    # without train_id: trains from the gaps between pulses (no pulse in the 50 ms before)
    import dataclasses

    bare = dataclasses.replace(ts, metadata=ts.metadata.drop(columns="train_id"))
    assert np.allclose(train_baseline(bare, 0.05, raw=raw, sfreq=4096.0).data, expected)
    # from the epochs, when the first pulse's epoch holds the 50 ms before the train
    long, _, expected_long = _pulse_trials(tmin=-0.06)
    assert np.allclose(train_baseline(long, 0.05).data, expected_long)
    with pytest.raises(DataError, match="epochs from -0.05 s"):
        train_baseline(ts, 0.05)
    # the preset pipeline applies it, given the recording, and says so in the methods text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = reports.run_pipeline(ts, "erna", artifact=False, reference=False,
                                      existence=False, trial_fits=False,
                                      continuous=(raw, 4096.0, 0.0))
    functions = [s["function"] for s in result.group.provenance.steps]
    assert "evoparam.prep.train_baseline" in functions
    assert np.allclose(result.trials.data, expected)
    assert "the last 50 ms before the train (from the continuous recording)" in \
        reports.methods_text(result)
    with pytest.warns(UserWarning, match="baseline per train was not applied"):
        reports.run_pipeline(ts, "erna", artifact=False, reference=False, existence=False,
                             trial_fits=False)


def test_f58_rodent_p1_n1_named_by_sign_at_each_depth():
    """§6.1: rodent_ep and primate_ep label "P1/N1 by depth": the early field potential
    reverses with depth, so each contact names its component by its own sign."""

    from evoparam.data import Evoked
    from evoparam.models import fit, label_components
    from evoparam.models.labels import sign_name

    assert (sign_name("P1/N1", 1), sign_name("P1/N1", -1)) == ("P1", "N1")
    assert sign_name("N1/P1", 1) == "P1" and sign_name("N2", 1) == "N2"
    assert sign_name("a/b", 1) == "a/b"
    sfreq = 1250.0
    times = np.arange(-0.1, 0.5, 1 / sfreq)
    rng = np.random.default_rng(1)
    early = np.exp(-0.5 * ((times - 0.025) / 0.005) ** 2)
    # a laminar pair: positive above the reversal, negative below
    data = np.array([60 * early, -60 * early]) + rng.normal(0, 1, (2, times.size))
    ev = Evoked(data=data, times=times, sfreq=sfreq, ch_names=["L02", "L10"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = fit(ev, "rodent_ep")
    labels = [[c.label for c in f.components] for f in group.fits]
    assert "P1" in labels[0] and "N1" in labels[1], labels
    kept = label_components(group, presets.fit_fields("rodent_ep")["windows"],
                            polarity_rule="none")
    assert "P1/N1" in [c.label for c in kept.fits[0].components]
    with pytest.raises(SettingsError):
        label_components(group, "rodent_ep", polarity_rule="depth")
