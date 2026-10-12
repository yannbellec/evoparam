# SPDX-License-Identifier: Apache-2.0
"""REQ-399 (spec §13.7): the SOUP inventory ``soup.json`` (package, version range, licence,
anomaly source URL, functions used) is generated and diffed in CI; every algorithm records
its version in ``Provenance.steps``; the requirement → test traceability report of §14 has
no untested requirement. REQ-396: ``THIRD_PARTY.md`` comes from the same inventory.

The numerical-regression half of REQ-399 (outputs that change without an algorithm version
bump fail) is ``test_determinism.py::test_req295_fixture_matches_the_reference_values``.
"""

import ast
import importlib.util
import json
import pathlib
import re
import sys
from importlib import metadata
from typing import Any

import pytest

import evoparam

ROOT = pathlib.Path(evoparam.__file__).resolve().parent.parent

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(
        not (ROOT / "tools" / "gen_soup.py").exists(), reason="needs the source tree"
    ),
]


def _tool(name: Any) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _soup() -> Any:
    return json.loads((ROOT / "soup.json").read_text(encoding="utf-8"))


def test_req399_soup_and_third_party_are_up_to_date() -> None:
    if sys.version_info < (3, 11):
        pytest.importorskip("tomli")  # the generator reads pyproject.toml
    assert _tool("gen_soup").main(["--check"]) == 0, "run python tools/gen_soup.py"


def test_req399_soup_fields() -> None:
    soup = _soup()
    assert soup["schema"] == "evoparam-soup/1"
    names = {row["package"] for row in soup["packages"]}
    assert {"numpy", "scipy", "mne", "pandas", "matplotlib"} <= names
    for row in soup["packages"]:
        assert set(row) >= {
            "package",
            "version_range",
            "licence",
            "anomaly_url",
            "functions_used",
            "role",
            "locked_version",
        }, row["package"]
        assert row["anomaly_url"].startswith("https://"), row["package"]
        assert row["licence"], row["package"]
        assert "GPL" not in row["licence"], row["package"]
        assert row["role"] in ("core", "optional", "transitive", "tooling")
    core = {row["package"]: row for row in soup["packages"] if row["role"] == "core"}
    assert sorted(core) == ["numpy", "scipy"]
    assert core["numpy"]["version_range"] == ">=1.24,<3"
    assert {"linalg.lstsq", "asarray"} & set(core["numpy"]["functions_used"])
    assert {"optimize.least_squares"} <= set(core["scipy"]["functions_used"])


def _family(text: Any) -> Any:
    text = text.lower()
    out = set()
    for family, words in (
        ("bsd", ("bsd",)),
        ("mit", ("mit",)),
        ("apache", ("apache",)),
        ("mpl", ("mpl", "mozilla")),
        ("psf", ("psf", "python software foundation")),
    ):
        if any(w in text for w in words):
            out.add(family)
    return out


def test_req399_licences_agree_with_the_installed_metadata() -> None:
    checked = 0
    for row in _soup()["packages"]:
        try:
            meta = metadata.metadata(row["package"])
        except metadata.PackageNotFoundError:
            continue
        declared = " ".join(
            filter(
                None,
                [
                    meta.get("License-Expression"),
                    meta.get("License") if len(meta.get("License") or "") <= 200 else None,
                    *[c for c in meta.get_all("Classifier") or [] if c.startswith("License")],
                ],
            )
        )
        if not _family(declared):
            continue  # the metadata names no licence
        assert _family(row["licence"]) & _family(declared), (row["package"], declared)
        checked += 1
    assert checked >= 5


def _third_party_imports() -> Any:
    # tomllib is standard from Python 3.11 (the package falls back to tomli on 3.10)
    stdlib = set(sys.stdlib_module_names) | {"tomllib"}
    found = set()
    for path in (ROOT / "evoparam").rglob("*.py"):
        if "tests" in path.relative_to(ROOT).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                found.add(node.module.split(".")[0])
            elif (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "require"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                found.add(str(node.args[0].value).split(".")[0])
    return {n for n in found if n not in stdlib and n != "evoparam" and not n.startswith("_")}


def test_req399_every_third_party_import_is_in_the_inventory() -> None:
    inventoried = {row["import_name"].split(".")[0] for row in _soup()["packages"]}
    # ``ndx_evoparam`` is evoparam's own NWB extension (evoparam/io/ndx_evoparam)
    missing = sorted(_third_party_imports() - inventoried - {"ndx_evoparam"})
    assert missing == []


def test_req399_algorithm_version_in_every_step() -> None:
    from evoparam import sim
    from evoparam.version import ALGORITHM_VERSIONS

    trials, _ = sim.simulate_ccep(n_trials=4, n_channels=3, seed=0, artifact=False, p_response=1.0)
    steps = trials.provenance.steps
    assert steps
    assert all(re.fullmatch(r"\d+\.\d+", s["algorithm_version"]) for s in steps)
    assert steps[-1]["algorithm_version"] == ALGORITHM_VERSIONS["evoparam.sim.simulate_ccep"]


def test_req399_traceability_generator() -> None:
    tool = _tool("gen_traceability")
    report = tool.report()
    assert report["n_requirements"] >= 220
    reqs = {row["req"] for row in report["requirements"]}
    assert {"REQ-290", "REQ-375", "REQ-376", "REQ-399"} <= reqs
    row = next(r for r in report["requirements"] if r["req"] == "REQ-399")
    assert "evoparam/tests/test_soup_inventory.py" in row["tests_naming_it"]


# requirements that no test names: none since 2026-10-11 (the last four, of earlier build
# steps, are named by tests now; some of those tests are strict expected failures whose
# reasons give what is missing, see CHANGELOG)
UNTESTED = []


def test_req399_no_untested_requirement() -> None:
    assert _tool("gen_traceability").report()["untested"] == []


def test_req399_the_untested_list_is_the_measured_one() -> None:
    """Keeps the list above honest: the untested requirements are exactly those."""

    assert _tool("gen_traceability").report()["untested"] == UNTESTED
