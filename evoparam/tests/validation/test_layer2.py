# SPDX-License-Identifier: Apache-2.0
"""The layer-2 runner (REQ-370): the smoke design runs and reports every metric."""

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parents[3]
RUNNER = ROOT / "validation" / "layer2.py"
pytestmark = pytest.mark.skipif(not RUNNER.exists(), reason="repository files not in this "
                                "checkout")


def _runner():
    spec = importlib.util.spec_from_file_location("layer2", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_weekly_design_covers_every_factor_level():

    layer2 = _runner()
    cells = layer2.cells("weekly")
    for factor, levels in layer2.FACTORS.items():
        for modality in layer2.MODALITIES:
            seen = {c[factor] for m, c in cells if m == modality}
            assert set(levels) <= seen, (modality, factor)


def test_smoke_run_writes_the_report(tmp_path):

    layer2 = _runner()
    report = layer2.main(["--design", "smoke", "--replicates", "2", "--jobs", "1",
                          "--no-tests", "--out", str(tmp_path)])
    saved = json.loads((tmp_path / "report.json").read_text())
    assert len(saved["grid"]) == len(report["grid"]) == 4
    with_components = [row for row in saved["grid"] if row["n_components"] > 0]
    assert all(row["f1"] >= 0.5 for row in with_components)
    assert all(row["false_per_waveform"] is not None for row in saved["grid"])
    assert (tmp_path / "report.md").read_text().startswith("# Layer-2 validation")
