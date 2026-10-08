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
DEVIATIONS = {("sep", "background"): "constant"}

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
    assert "preset 'ccep' v1.0" in methods_text(group.fits[0])


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
