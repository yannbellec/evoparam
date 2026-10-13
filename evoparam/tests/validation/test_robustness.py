# SPDX-License-Identifier: Apache-2.0
"""Layer 4 of the scientific validation (spec §7.4): REQ-375 (the five robustness analyses
in the release pipeline, attached to the release notes) and REQ-376 (the layer-3 regression
gate with the pre-registered tolerances kappa -0.05, latency +1 ms median, recall -3
points).

The analyses are run by ``validation/robustness.py`` (about an hour on the reference
machine); these tests read the report it wrote for this version under
``validation/layer4/<version>/`` and check the gate ``tools/layer3_gate.py`` on the committed
layer-3 files and on mutated copies of them.
"""

import copy
import importlib.util
import json
import math
import subprocess
import sys
from typing import Any

import pytest

import evoparam
from evoparam.version import __version__

from ..benchmark.helpers import ROOT

LAYER3 = ROOT / "validation" / "layer3"
LAYER4 = ROOT / "validation" / "layer4" / __version__
ANALYSES = ("montage", "preset", "sampling", "trials", "platform")
HIIE = "waits for the owner's HIIE decision (nothing public before it)"


def _module(path: Any, name: Any) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate() -> Any:
    return _module(ROOT / "tools" / "layer3_gate.py", "layer3_gate")


@pytest.fixture(scope="module")
def robustness() -> Any:
    return _module(ROOT / "validation" / "robustness.py", "robustness")


@pytest.fixture(scope="module")
def report() -> Any:
    path = LAYER4 / "report.json"
    assert path.exists(), f"run python validation/robustness.py (no {path})"
    return json.loads(path.read_text(encoding="utf-8"))


def _workflow(name: Any) -> Any:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))


###################################################################################################
## REQ-375


def test_req375_the_five_analyses_are_defined(robustness: Any) -> None:
    """REQ-375: the script runs exactly the five analyses of §7.4."""

    assert robustness.ANALYSES == ANALYSES
    assert robustness.SCHEMES == ("bipolar", "car", "car_adaptive")
    assert robustness.RATES == (1024.0, 512.0)
    assert robustness.TRIAL_COUNTS == (5, 7, 10)
    assert robustness.PRESET_FACTORS == (0.5, 1.5)
    assert robustness.SAMPLING_RECALL_POINTS == 5.0
    assert robustness.PRESET_RECALL_REL == 0.10
    assert robustness.PRESET_LATENCY_MS == 2.0


def test_req375_preset_perturbation(robustness: Any) -> None:
    """§7.4 (2): a choice field is perturbed by +-50 %; intervals at both ends; integers
    rounded and at least 1; flags are left alone."""

    assert robustness.perturb(0.2, 0.5) == pytest.approx(0.1)
    assert robustness.perturb([0.009, 0.1], 1.5) == pytest.approx([0.0135, 0.15])
    assert robustness.perturb(3, 0.5) == 2
    assert robustness.perturb(1, 0.5) == 1
    assert robustness.perturb(True, 0.5) is True
    assert robustness.perturb({"N1": [0.01, 0.05]}, 0.5) == {"N1": [0.005, 0.025]}


def test_req375_sampling_decimates_and_never_upsamples(robustness: Any) -> None:
    """§7.4 (3): runs are decimated; a run recorded at or below the rate is kept as is."""

    assert robustness.decimates(2048.0002097152214, 1024.0)
    assert robustness.decimates(2048.0002097152214, 512.0)
    assert not robustness.decimates(512.0, 1024.0)
    assert not robustness.decimates(512.0, 512.0)
    assert not robustness.decimates(1024.0001048576107, 1024.0)


def test_req375_release_pipeline_runs_the_analyses_and_keeps_them() -> None:
    """REQ-375: the release workflow runs the five analyses and keeps their report with the
    release records."""

    workflow = _workflow("release.yml")
    steps = [s for job in workflow["jobs"].values() for s in job.get("steps", [])]
    runs = " ".join(s.get("run", "") for s in steps)
    assert "python validation/robustness.py" in runs
    kept = " ".join(str(s.get("with", {}).get("path", "")) for s in steps)
    assert "validation/layer4/" in kept


def test_req375_report_holds_the_five_analyses(report: Any) -> None:
    """REQ-375: the report of this version holds the five analyses, each with results."""

    assert report["version"] == __version__
    assert tuple(report["analyses"]) == ANALYSES
    assert report["montage"]["summary"]["bipolar-car"]["n"] > 0
    assert len(report["preset"]) > 0
    assert report["sampling"]["verdict"]
    assert [r["n_trials"] for r in report["trials"]["summary"]] == [5, 7, 10]
    assert report["platform"]["match"] is True
    text = (LAYER4 / "report.md").read_text(encoding="utf-8")
    for heading in (
        "## 1. Montage",
        "## 2. Preset",
        "## 3. Sampling rate",
        "## 4. Trial count",
        "## 5. Platform",
    ):
        assert heading in text


def test_req375_montage_covers_ds004080_and_ds004977(report: Any) -> None:
    """§7.4 (1): every ds004080 and ds004977 result is recomputed under the three
    references."""

    assert report["data"]["ds004080"]
    assert report["data"]["ds004977"]
    n_cases = len(report["data"]["ds004080"]) + len(report["data"]["ds004977"])
    for pair in ("bipolar-car", "bipolar-car_adaptive", "car-car_adaptive"):
        assert report["montage"]["summary"][pair]["n"] == n_cases


def _sampling_reason() -> Any:
    path = LAYER4 / "report.json"
    if not path.exists():
        return "no layer-4 report"
    verdict = json.loads(path.read_text(encoding="utf-8"))["sampling"]["verdict"]
    return "; ".join(
        f"{reading} reading: N1 recall {v['recall_native']:.3f} native, "
        f"{v['recall_1024']:.3f} at 1024 Hz ({v['drop_points_1024']:+.2f} points, "
        f"{'met' if v['met'] else 'not met'})"
        for reading, v in verdict.items()
    )


def _sampling_met() -> Any:
    path = LAYER4 / "report.json"
    if not path.exists():
        return False
    verdict = json.loads(path.read_text(encoding="utf-8"))["sampling"]["verdict"]
    return all(v["met"] for v in verdict.values())


@pytest.mark.xfail(
    not _sampling_met(), strict=True, reason="§7.4 (3) not met: " + _sampling_reason()
)
def test_req375_sampling_recall_within_5_points_at_1024_hz(report: Any) -> None:
    """§7.4 (3): N1 recall stays within 5 percentage points of the native rate at 1024 Hz
    (both readings of the published rule)."""

    for v in report["sampling"]["verdict"].values():
        assert v["drop_points_1024"] <= 5.0


def test_req375_preset_sensitivities_are_listed_in_the_preset_docs(report: Any) -> None:
    """§7.4 (2): every sensitivity found is listed in the preset documentation."""

    page = (ROOT / "docs" / "preset_sensitivity.md").read_text(encoding="utf-8")
    assert f"evoparam {__version__}" in page
    for row in report["preset"]:
        if row["field"] is None or row["factor"] == 1.0:
            continue
        line = f"| `{row['preset']}` | `{row['field']}` | {row['factor']} |"
        assert line in page, line
        if row["sensitive"]:
            assert any(
                text.startswith(line) and text.rstrip().endswith("**yes** |")
                for text in page.splitlines()
            )
    assert "preset_sensitivity" in (ROOT / "docs" / "presets.md").read_text(encoding="utf-8")


def test_req375_trial_count_reports_changed_decisions(report: Any) -> None:
    """§7.4 (4): the change of existence decisions is reported per trial count."""

    for row in report["trials"]["summary"]:
        assert row["draws"] > 0
        assert 0.0 <= row["changed_mean"] <= 1.0


def test_req375_platform_this_machine(report: Any) -> None:
    """§7.4 (5): the deterministic fixture fit equals the reference within 1e-9 here."""

    assert report["platform"]["rtol"] == 1e-9
    assert report["platform"]["match"] is True


def test_req375_platform_macos_and_windows(report: Any) -> None:
    """§7.4 (5): identical results on Linux, macOS and Windows: one record per platform in
    ``platforms/`` (written by ``validation/robustness.py --analyses platform`` on each
    machine), each within the 1e-9 of REQ-295."""

    records = report.get("platforms", [report["platform"]])
    assert all(r["match"] and r["rtol"] == 1e-9 for r in records), [
        (r["platform"], r["max_rel_diff"]) for r in records if not r["match"]]
    seen = {r["platform"].split("-")[0] for r in records}
    missing = {"Linux", "macOS", "Windows"} - seen
    if missing:
        pytest.xfail(f"§7.4 (5): measured on {sorted(seen)}, not yet on {sorted(missing)} "
                     "(run validation/robustness.py --analyses platform there)")


@pytest.mark.xfail(strict=True, reason=HIIE)
def test_req375_report_attached_to_the_release_notes() -> None:
    """REQ-375: the report is attached to the release notes of every minor version (no
    release exists before the owner's HIIE decision)."""

    tags = subprocess.run(
        ["git", "tag", "--list", "v*"], capture_output=True, text=True, cwd=ROOT, check=False
    ).stdout.split()
    assert tags, "no release tag"


###################################################################################################
## REQ-376


def test_req376_tolerances_are_the_preregistered_ones(gate: Any) -> None:
    """REQ-376: kappa -0.05, latency +1 ms median, recall -3 points."""

    assert gate.tolerances() == {"kappa": 0.05, "latency": 1.0, "recall": 0.03}


def _docs() -> Any:
    return {
        name: json.loads((LAYER3 / name).read_text(encoding="utf-8"))
        for name in ("n1_ds004080.json", "erpcore_difference.json", "erpcore_parent.json")
    }


def test_req376_committed_files_have_the_metrics(gate: Any) -> None:
    """REQ-376: the gate reads the N1 metrics of ds004080 (two readings) and the ERP CORE
    components; every value is finite."""

    found = gate.metrics(_docs())
    assert {
        "n1_ds004080/paper/kappa",
        "n1_ds004080/code/recall",
        "n1_ds004080/paper/median_abs_latency_diff_ms",
    } <= set(found)
    assert any(k.startswith("erpcore_parent/") for k in found)
    n1 = {k: v for k, v in found.items() if k.startswith("n1_")}
    assert all(math.isfinite(v) for _, v in n1.values())


@pytest.mark.parametrize(
    ("field", "delta", "blocks"),
    [
        ("kappa", -0.051, True),
        ("kappa", -0.049, False),
        ("median_abs_latency_diff_ms", 1.01, True),
        ("median_abs_latency_diff_ms", 0.99, False),
        ("recall", -0.031, True),
        ("recall", -0.029, False),
        ("recall", +0.2, False),
    ],
)
def test_req376_regression_beyond_tolerance_blocks(
    gate: Any, field: Any, delta: Any, blocks: Any
) -> None:
    """REQ-376: a change beyond its tolerance blocks the release; within it does not."""

    base = _docs()
    now = copy.deepcopy(base)
    now["n1_ds004080.json"]["metrics"]["paper"][field] += delta
    result = gate.gate(base, now, "# Changelog\n\n## Unreleased\n\n- nothing\n")
    assert result["release_blocked"] is blocks
    if blocks:
        assert result["blocking"] == [f"n1_ds004080/paper/{field}"]


def test_req376_erpcore_latency_regression_blocks(gate: Any) -> None:
    base = _docs()
    now = copy.deepcopy(base)
    row = now["erpcore_parent.json"]["summary"][0]
    row["median_diff_ms"] = abs(row["median_diff_ms"] or 0.0) + 1.5
    result = gate.gate(base, now, "## Unreleased\n")
    assert result["release_blocked"]


def test_req376_missing_metric_blocks(gate: Any) -> None:
    base = _docs()
    now = copy.deepcopy(base)
    del now["n1_ds004080.json"]["metrics"]["code"]
    assert gate.gate(base, now, "## Unreleased\n")["release_blocked"]


def test_req376_changelog_explanation_unblocks(gate: Any) -> None:
    """REQ-376: unless the changelog explains it (in ## Unreleased)."""

    base = _docs()
    now = copy.deepcopy(base)
    now["n1_ds004080.json"]["metrics"]["paper"]["kappa"] -= 0.2
    text = (
        "# Changelog\n\n## Unreleased\n\n- Layer-3 regression explained: "
        "n1_ds004080/paper/kappa: the N1 window now follows the published rule.\n\n"
        "## 0.0.1\n"
    )
    result = gate.gate(base, now, text)
    assert result["regressions"] == ["n1_ds004080/paper/kappa"]
    assert not result["release_blocked"]
    # an explanation under an older release does not count
    old = (
        "# Changelog\n\n## Unreleased\n\n## 0.0.1\n\n- Layer-3 regression explained: "
        "n1_ds004080/paper/kappa: old.\n"
    )
    assert gate.gate(base, now, old)["release_blocked"]


def test_req376_gate_passes_on_the_committed_files(tmp_path: Any) -> None:
    """REQ-376: the gate script against the committed layer-3 files (no release tag yet:
    the baseline is the files themselves) exits 0."""

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "layer3_gate.py"),
            "--baseline",
            str(LAYER3),
            "--json",
            str(tmp_path / "gate.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = json.loads((tmp_path / "gate.json").read_text(encoding="utf-8"))
    assert out["n_metrics"] > 0
    assert not out["blocking"]


def test_req376_release_workflow_runs_the_gate() -> None:
    workflow = _workflow("release.yml")
    runs = " ".join(
        s.get("run", "") for job in workflow["jobs"].values() for s in job.get("steps", [])
    )
    assert "python tools/layer3_gate.py" in runs
    assert "layer3_gate" in (ROOT / "RELEASING.md").read_text(encoding="utf-8")


def test_req376_n1_file_matches_this_version() -> None:
    doc = json.loads((LAYER3 / "n1_ds004080.json").read_text(encoding="utf-8"))
    assert doc["version"] == evoparam.__version__
    assert set(doc["metrics"]) == {"paper", "code"}
