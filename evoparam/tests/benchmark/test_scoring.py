# SPDX-License-Identifier: Apache-2.0
"""REQ-352: scoring is deterministic and the signed JSON is reproducible byte for byte from
the same inputs.

The byte-for-byte check scores the same predictions twice, in two separate processes of
the same environment (``evoparam benchmark score``), and compares the files. The metrics
themselves are compared with a relative tolerance of 1e-9.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from evoparam.benchmark import (BASELINES, TASKS, get_baseline, read_result, score,
                                score_outputs, verify_result, write_output, write_result)
from evoparam.benchmark.scoring import HEADLINE, RESULT_SCHEMA, auc_score
from evoparam.benchmark.tasks import output_path
from evoparam.errors import DataError
from evoparam.version import __version__

from .helpers import ROOT, inputs, perfect_t1, perfect_t2, suite

pytest.importorskip("jsonschema")


def _predict(tmp_path, task, method="vanblooijs2023", n=None):
    """Write a baseline's outputs for the public cells of ``core``; returns the folder."""

    folder = tmp_path / method / task
    baseline = get_baseline(method)
    for ti in inputs(task, n):
        out = baseline.run({task: ti})
        write_output(task, out[task], output_path(str(folder), task, ti.cell))
    return folder


def _cli_score(folder, out, task="T1"):
    env = dict(os.environ,
               PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run([sys.executable, "-m", "evoparam", "benchmark", "score", "--task",
                           task, "--predictions", str(folder), "--suite", "core",
                           "--method-name", "vanblooijs2023", "--method-version", "1",
                           "--out", str(out), "--json"],
                          capture_output=True, text=True, env=env, cwd=str(out.parent),
                          check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return out.read_bytes()


def test_req352_signed_json_is_byte_identical_across_two_runs(tmp_path):
    folder = _predict(tmp_path, "T1")
    first = _cli_score(folder, tmp_path / "first.json")
    second = _cli_score(folder, tmp_path / "second.json")
    assert first == second
    document = json.loads(first)
    assert verify_result(document)
    assert str(tmp_path) not in first.decode("utf-8")       # no machine path in the result


def test_req352_in_process_scoring_matches_the_command_line(tmp_path):
    folder = _predict(tmp_path, "T1")
    cli = _cli_score(folder, tmp_path / "cli.json")
    method = {"name": "vanblooijs2023", "version": "1", "citation": "", "kind": "submission",
              "settings": {}}
    document = score("T1", str(folder), "core", method=method)
    path = write_result(document, str(tmp_path / "api.json"))
    with open(path, "rb") as fobj:
        assert fobj.read() == cli


def test_req352_result_document_is_signed_and_complete(tmp_path):
    folder = _predict(tmp_path, "T2")
    document = score("T2", str(folder), "core", method=get_baseline("vanblooijs2023").info())
    assert document["schema"] == RESULT_SCHEMA
    assert document["task"] == "T2"
    assert document["task_schema"] == "benchmark-task/T2/1"
    assert document["suite"]["name"] == "core"
    assert document["suite"]["sha256"] == suite("core").sha256
    assert document["package"] == {"name": "evoparam", "version": __version__}
    cells = suite("core").cells("T2", split="public")
    assert [c["cell"] for c in document["cells"]] == [c.id for c in cells]
    assert all(len(c["input_sha256"]) == 64 and len(c["output_sha256"]) == 64
               for c in document["cells"])
    assert len(document["inputs_sha256"]) == len(document["outputs_sha256"]) == 64
    assert set(HEADLINE["T2"]) <= set(document["headline"])
    assert "created" not in json.dumps(document)
    assert verify_result(document)
    path = write_result(document, str(tmp_path / "r.json"))
    assert read_result(path) == json.loads(open(path, encoding="utf-8").read())


def test_req352_a_changed_number_breaks_the_signature(tmp_path):
    folder = _predict(tmp_path, "T1", n=2)
    document = score("T1", str(folder), "core", inputs=list(inputs("T1", 2)))
    path = write_result(document, str(tmp_path / "r.json"))
    text = open(path, encoding="utf-8").read()
    data = json.loads(text)
    data["metrics"]["f1"] = 0.999
    with open(path, "w", encoding="utf-8") as fobj:
        json.dump(data, fobj, indent=1, sort_keys=True)
    assert not verify_result(data)
    with pytest.raises(DataError, match="signature"):
        read_result(path)


def test_req352_a_changed_output_changes_the_result(tmp_path):
    folder = _predict(tmp_path, "T1", n=2)
    ins = list(inputs("T1", 2))
    before = score("T1", str(folder), "core", inputs=ins)
    path = folder / f"{ins[0].cell}.tsv"
    lines = path.read_text(encoding="utf-8").split("\n")
    fields = lines[1].split("\t")
    fields[2] = "1" if fields[2] == "0" else "0"
    lines[1] = "\t".join(fields)
    path.write_text("\n".join(lines), encoding="utf-8")
    after = score("T1", str(folder), "core", inputs=ins)
    assert after["cells"][0]["output_sha256"] != before["cells"][0]["output_sha256"]
    assert after["cells"][1] == before["cells"][1]
    assert after["signature"]["digest"] != before["signature"]["digest"]


@pytest.mark.parametrize("task", list(TASKS))
def test_req352_scoring_is_deterministic_for_every_task(task):
    """Same inputs and outputs, scored twice: the same document (metrics within rtol 1e-9
    and the same signature)."""

    ti = inputs(task, 1)[0]
    method = next(b for b in BASELINES.values() if task in b.tasks
                  and b.name not in ("evoparam-preset", "tf-cluster")) \
        if task != "T7" else None
    if method is None:
        rows = [{"channel": ti.descriptor["ch_names"][0], "zone": 0, "sign": 1,
                 "freq_centroid": 80.0, "time_centroid": 0.25}]
        outputs = {ti.cell: rows}
    else:
        outputs = {ti.cell: method.run({task: ti})[task]}
    a = score_outputs(task, [ti], outputs, suite=suite("core"))
    b = score_outputs(task, [ti], outputs, suite=suite("core"))
    assert a["signature"] == b["signature"]
    for key, value in a["metrics"].items():
        if isinstance(value, float):
            np.testing.assert_allclose(b["metrics"][key], value, rtol=1e-9)


def test_req352_perfect_answers_score_perfectly():
    t1 = list(inputs("T1"))
    t2 = list(inputs("T2"))
    r1 = score_outputs("T1", t1, {ti.cell: perfect_t1(ti) for ti in t1}, suite=suite("core"))
    for key in ("auc", "f1", "precision", "recall", "specificity"):
        np.testing.assert_allclose(r1["metrics"][key], 1.0, rtol=1e-9)
    assert r1["metrics"]["fpr_at_0.05"] == 0.0
    r2 = score_outputs("T2", t2, {ti.cell: perfect_t2(ti) for ti in t2}, suite=suite("core"))
    assert r2["metrics"]["reported_fraction"] == 1.0
    assert r2["metrics"]["latency_mae"] == pytest.approx(0.0, abs=1e-12)
    assert r2["metrics"]["amplitude_mae"] == pytest.approx(0.0, abs=1e-12)
    shifted = score_outputs("T2", t2, {ti.cell: perfect_t2(ti, shift=0.002) for ti in t2},
                            suite=suite("core"))
    np.testing.assert_allclose(shifted["metrics"]["latency_mae"], 0.002, rtol=1e-9)
    np.testing.assert_allclose(shifted["metrics"]["latency_bias"], 0.002, rtol=1e-9)


def test_req352_t4_no_removal_and_exact_removal():
    ti = inputs("T4", 1)[0]
    none = score_outputs("T4", [ti], {ti.cell: ti.arrays["data"]}, suite=suite("core"))
    exact = score_outputs("T4", [ti], {ti.cell: ti.truth["clean"]}, suite=suite("core"))
    np.testing.assert_allclose(none["metrics"]["artifact_residual_rel"], 1.0, rtol=1e-9)
    assert exact["metrics"]["artifact_residual_rms"] == pytest.approx(0.0, abs=1e-12)


def test_req352_auc_is_the_mann_whitney_statistic():
    assert auc_score([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2]) == 1.0
    assert auc_score([1, 1, 0, 0], [0.1, 0.2, 0.9, 0.8]) == 0.0
    assert auc_score([1, 0], [0.5, 0.5]) == 0.5
    np.testing.assert_allclose(auc_score([1, 0, 1, 0, 1], [3, 1, 2, 2, 1]), 4 / 6, rtol=1e-9)


def test_req352_missing_output_is_an_error_unless_allowed(tmp_path):
    folder = _predict(tmp_path, "T1", n=2)
    ins = list(inputs("T1", 3))
    with pytest.raises(DataError, match="no output for cell"):
        score("T1", str(folder), "core", inputs=ins)
    document = score("T1", str(folder), "core", inputs=ins, allow_missing=True)
    assert document["missing_cells"] == [ins[2].cell]
