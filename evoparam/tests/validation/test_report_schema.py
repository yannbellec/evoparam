# SPDX-License-Identifier: Apache-2.0
"""Layer-3 comparison records (REQ-374): every layer-3 comparison reports the comparator's
number, evoparam's number, the difference with its interval, and the exact versions of the
data and of the comparator definition (registry rule id).

The records are ``validation/layer3/comparisons.json``, written by
``validation/layer3_report.py`` from the layer-3 result files."""

import hashlib
import importlib.util
import json
import math
from typing import Any

import pytest

from ..benchmark.helpers import ROOT

LAYER3 = ROOT / "validation" / "layer3"
NOT_RESULTS = {"tolerances.json", "comparisons.json"}


def _module(path: Any, name: Any) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report() -> Any:
    return _module(ROOT / "validation" / "layer3_report.py", "layer3_report")


@pytest.fixture(scope="module")
def records() -> Any:
    return json.loads((LAYER3 / "comparisons.json").read_text(encoding="utf-8"))


def test_req374_every_layer3_file_has_comparison_records(records: Any) -> None:
    """REQ-374: each layer-3 result file is a source of the records, and gives at least one
    record."""

    results = sorted(p.name for p in LAYER3.glob("*.json") if p.name not in NOT_RESULTS)
    assert results
    for name in results:
        assert name in records["sources"], name
        prefix = name[: -len(".json")] + "/"
        assert any(r["id"].startswith(prefix) for r in records["comparisons"]), name


def test_req374_records_give_both_numbers_the_difference_and_its_interval(
    records: Any, report: Any
) -> None:
    """REQ-374: the comparator's number, evoparam's number, the difference (evoparam minus
    comparator) with its interval, the data (dataset id and DOI) and the comparator
    definition (registry rule id, or the definition written out) in every record."""

    rules = json.loads((ROOT / "evoparam" / "existence" / "rules.json").read_text("utf-8"))
    for r in records["comparisons"]:
        assert set(report.REQUIRED) <= set(r), r["id"]
        values = [r["comparator_value"], r["evoparam_value"], r["difference"], *r["difference_ci"]]
        assert all(isinstance(v, float) and math.isfinite(v) for v in values), r["id"]
        lo, hi = r["difference_ci"]
        assert lo <= r["difference"] <= hi, r["id"]
        assert r["interval"]
        assert r["n"] > 0
        assert r["data"]["dataset"], r["id"]
        assert r["data"]["doi"], r["id"]
        definition = r["comparator_definition"]
        if "registry_rule" in definition:
            assert definition["registry_rule"] in rules["rules"], r["id"]
            entry = dict(rules["rules"][definition["registry_rule"]], **definition["overrides"])
            assert definition["version"] == report.digest(entry), r["id"]
        else:
            assert definition["id"], r["id"]


def test_req374_records_are_those_of_the_result_files(report: Any) -> None:
    """REQ-374: the committed records are the ones the result files give (regenerated with
    ``python validation/layer3_report.py``)."""

    assert (LAYER3 / "comparisons.json").read_text(encoding="utf-8") == report.render()


def test_req374_n1_difference_is_the_difference_of_the_fractions(records: Any) -> None:
    """REQ-374: on ds004080 the fractions are those of the counts of n1_ds004080.json."""

    n1 = json.loads((LAYER3 / "n1_ds004080.json").read_text(encoding="utf-8"))
    for reading, m in n1["metrics"].items():
        r = next(
            x for x in records["comparisons"] if x["id"] == f"n1_ds004080/{reading}/n1_fraction"
        )
        assert r["comparator_value"] == pytest.approx(m["n_rule"] / m["n_channels"], abs=1e-6)
        assert r["evoparam_value"] == pytest.approx(m["n_model"] / m["n_channels"], abs=1e-6)
        assert r["data"]["version"] == "1.2.4"


def test_req374_erpcore_records_the_files_it_reads(tmp_path: Any, monkeypatch: Any) -> None:
    """REQ-374: the ERP CORE run records the SHA-256 of every file it reads (the OSF deposit
    has no version number)."""

    erpcore = _module(ROOT / "validation" / "erpcore.py", "erpcore_validation")
    folder = tmp_path / "P3" / "1"
    folder.mkdir(parents=True)
    (folder / "1_P3_epoch_interp_ar.set").write_bytes(b"set")
    (folder / "1_P3_epoch_interp_ar.fdt").write_bytes(b"fdt")
    monkeypatch.setattr(erpcore, "ROOT", str(tmp_path))
    hashes = erpcore.files_sha256("P3", 1)
    assert sorted(hashes) == ["P3/1/1_P3_epoch_interp_ar.fdt", "P3/1/1_P3_epoch_interp_ar.set"]
    assert hashes["P3/1/1_P3_epoch_interp_ar.set"] == hashlib.sha256(b"set").hexdigest()


def test_req374_exact_versions_of_data_and_comparator(records: Any) -> None:
    """REQ-374: every record names the exact version of its data (a version number, or the
    SHA-256 of the files read) and of its comparator definition."""

    missing = [
        r["id"]
        for r in records["comparisons"]
        if not (r["data"].get("version") or r["data"].get("files_sha256"))
        or not r["comparator_definition"].get("version")
    ]
    assert missing == []
