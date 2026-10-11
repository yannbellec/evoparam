# SPDX-License-Identifier: Apache-2.0
"""REQ-350 (``evoparam benchmark run --suite core`` completes in at most 30 min and covers
T1-T7) and REQ-353 (the six published threshold rules and the cross-projection test are
baselines, and their T1/T2 results on the simulated ``core`` suite are in the first public
report).

The timing runs only with EVOPARAM_PERF_TESTS=1 and the re-run of the committed results
with EVOPARAM_SLOW_TESTS=1. The committed run (``docs/benchmark/run-core.json``) records
the measured duration.
"""

import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from evoparam.benchmark import (BASELINES, RULES, TASKS, get_baseline, list_suites,
                                load_results, make_inputs, read_output, read_result,
                                run_suite, simulate_cell, write_output)
from evoparam.benchmark._canon import array_sha256
from evoparam.benchmark.tasks import output_path

from .helpers import DOCS, PERF, RESULTS, ROOT, SLOW, inputs, suite

pytest.importorskip("jsonschema")

LIMIT_S = 30 * 60
PUBLISHED_RULES = ("vanblooijs2023", "hays2021", "keller2014", "silverstein2020", "ftract",
                   "prime2020")


def test_req350_core_covers_t1_to_t7():
    core = suite("core")
    assert list_suites() == ["core", "full"]
    assert core.task_ids() == list(TASKS)
    for task in TASKS:
        assert len(core.cells(task, split="public")) == 8, task
        assert len(core.cells(task, split="hidden")) == 2, task
    cells = core.cells(split="all")
    assert len(cells) == 60 == core.hidden_split["n_cells"]
    assert len({c.id for c in cells}) == 60
    seeds = [c.seed for c in core.cells(split="public")]
    assert len(set(seeds)) == len(seeds) == 48
    full = suite("full")
    assert full.task_ids() == list(TASKS)
    assert len(full.cells(split="public")) == 240 and len(full.cells(split="hidden")) == 60
    assert not {c.seed for c in full.cells(split="public")} & set(seeds)


def test_req350_suite_files_are_what_the_builder_writes():
    """The frozen files are those of ``tools/benchmark_suites.py`` (the commitment aside,
    which needs the hidden seeds document)."""

    spec = importlib.util.spec_from_file_location("benchmark_suites",
                                                  ROOT / "tools" / "benchmark_suites.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, replicates, hidden, base in (("core", 1, 2, 8_000_000),
                                          ("full", 5, 10, 9_000_000)):
        doc, hidden_ids = module.build(name, replicates, hidden, base)
        path = ROOT / "evoparam" / "benchmark" / "suites" / f"{name}.json"
        frozen = json.loads(path.read_text(encoding="utf-8"))
        frozen.pop("hidden_split")
        assert json.loads(json.dumps(doc)) == frozen
        assert sorted(hidden_ids) == sorted(c.id for c in suite(name).cells(split="hidden"))


@pytest.mark.parametrize("task", list(TASKS))
def test_req350_cells_are_generated_on_the_fly_and_deterministic(task):
    cell = suite("core").cells(task)[0]
    a, truth_a = simulate_cell(cell)
    b, _ = simulate_cell(cell)
    assert array_sha256(a.data) == array_sha256(b.data)
    np.testing.assert_allclose(a.data, b.data, rtol=1e-9, atol=0)
    ti = make_inputs(cell, a, truth_a, tasks=[task])[task]
    assert ti.descriptor["suite"] == {"name": "core", "version": "1.0.0"}
    assert ti.descriptor["split"] == "public"
    assert ti.truth is not None


def test_req350_every_baseline_answers_its_tasks(tmp_path):
    """Each baseline on one cell of each task it answers: its output validates. The slow
    ones (evoparam-preset on T1/T2/T7, tf-cluster) run in the slow test below."""

    slow = {("evoparam-preset", "T1"), ("evoparam-preset", "T2"), ("evoparam-preset", "T7"),
            ("tf-cluster", "T7")}
    for name, baseline in BASELINES.items():
        for task in baseline.tasks:
            if (name, task) in slow:
                continue
            out = baseline.run({task: inputs(task, 1)[0]})
            path = write_output(task, out[task], output_path(str(tmp_path / name), task,
                                                             inputs(task, 1)[0].cell))
            assert read_output(task, path) is not None, (name, task)


def test_req353_rules_and_projection_are_baselines():
    assert tuple(RULES) == PUBLISHED_RULES
    for name in PUBLISHED_RULES + ("projection",):
        baseline = get_baseline(name)
        assert set(baseline.tasks) == {"T1", "T2"}
        assert baseline.info()["citation"]
    assert {"classical", "template", "evoparam-preset"} <= set(BASELINES)


def test_req353_committed_results_on_core():
    docs = {(d["method"]["name"], d["task"]): d for d in load_results(str(RESULTS))}
    core = suite("core")
    for name in PUBLISHED_RULES + ("projection",):
        for task in ("T1", "T2"):
            d = docs[(name, task)]
            assert d["suite"]["name"] == "core" and d["suite"]["sha256"] == core.sha256
            assert d["missing_cells"] == []
            assert len(d["cells"]) == len(core.cells(task, split="public"))
    for task in TASKS:
        assert any(t == task for _, t in docs), task
    for path in RESULTS.glob("*.json"):
        read_result(str(path))                              # signatures hold


def test_req350_committed_run_covers_t1_to_t7_within_the_limit():
    run = json.loads((DOCS / "run-core.json").read_text(encoding="utf-8"))
    assert run["suite"]["name"] == "core" and run["tasks"] == list(TASKS)
    assert run["failures"] == []
    assert run["n_cells"] == 48
    assert run["duration_s"]["total"] <= LIMIT_S


@PERF
def test_req350_core_run_completes_within_30_minutes(tmp_path):
    env = dict(os.environ,
               PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    start = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", "evoparam", "benchmark", "run", "--suite",
                           "core", "--out", str(tmp_path / "run"), "--json"],
                          capture_output=True, text=True, env=env, check=False)
    duration = time.perf_counter() - start
    assert proc.returncode == 0, proc.stderr[-3000:]
    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert summary["tasks"] == list(TASKS) and summary["failures"] == []
    assert {json.loads(open(p, encoding="utf-8").read())["task"] for p in summary["results"]} \
        == set(TASKS)
    assert duration <= LIMIT_S, f"core run took {duration:.0f} s"


@SLOW
def test_req353_rerun_reproduces_the_committed_results(tmp_path):
    """The baselines of T1 and T2 run again on core give the committed metrics (rtol
    1e-9)."""

    methods = list(PUBLISHED_RULES) + ["projection", "vanblooijs2023_code",
                                       "evoparam-preset"]
    run_suite("core", str(tmp_path), tasks=["T1", "T2"], methods=methods, page=False)
    for name in methods:
        for task in ("T1", "T2"):
            new = read_result(str(tmp_path / "results" / f"{name}__{task}.json"))
            old = read_result(str(RESULTS / f"{name}__{task}.json"))
            assert [c["output_sha256"] for c in new["cells"]] == \
                [c["output_sha256"] for c in old["cells"]], (name, task)
            for key, value in old["metrics"].items():
                if isinstance(value, float):
                    np.testing.assert_allclose(new["metrics"][key], value, rtol=1e-9,
                                               err_msg=f"{name} {task} {key}")
