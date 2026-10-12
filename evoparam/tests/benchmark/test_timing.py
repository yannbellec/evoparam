# SPDX-License-Identifier: Apache-2.0
"""REQ-290 and REQ-299: the performance workloads P1–P6, their results file, and the
comparison with the CI baseline.

The comparison is checked on documents built here (a pull request that slows P1 by 2x
fails with a message naming the workload and the ratio; 1.10x to 1.25x warns), the suite
runner on its two cheapest workloads, and the absolute targets of §9.1 on the results
committed under ``benchmarks/results/`` (measured on the reference machine of
``benchmarks/README.md``). A target that is not met is a strict expected failure whose
reason gives the measured value.
"""

import copy
import glob
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

from evoparam.benchmark import timing

from .helpers import ROOT

BENCH = ROOT / "benchmarks"
FIELDS = ("id", "median_s", "p90_s", "peak_rss_bytes", "n_repeats", "environment")


def _doc(p1=0.05, p2=3.0, p2_rss=200 * 2**20):
    rows = [
        {
            "id": "P1",
            "workload": "one waveform",
            "median_s": p1,
            "p90_s": p1,
            "peak_rss_bytes": 3 * 2**20,
            "n_repeats": 5,
            "target_met": None,
        },
        {
            "id": "P2",
            "workload": "fit_trials",
            "median_s": p2,
            "p90_s": p2,
            "peak_rss_bytes": p2_rss,
            "n_repeats": 1,
            "target_met": None,
        },
    ]
    return {
        "schema": timing.SCHEMA,
        "suite": "ci",
        "environment": {"python": "3.x"},
        "results": rows,
    }


def test_req299_p1_twice_slower_fails_naming_workload_and_ratio():
    """REQ-299 acceptance: a pull request that slows P1 by 2x fails the perf job with a
    message naming the workload and the ratio."""

    failures, notes = timing.compare(_doc(p1=0.10), _doc(p1=0.05))
    assert len(failures) == 1
    assert not notes
    assert failures[0].startswith("P1 (one waveform)")
    assert "2.00 x the baseline" in failures[0]


@pytest.mark.parametrize(
    ("ratio", "failed", "warned"), [(1.05, False, False), (1.20, False, True), (1.30, True, False)]
)
def test_req299_tolerance_and_warning_band(ratio, failed, warned):
    """REQ-299 / §9.1: fail above 1.25x, warn (no failure) between 1.10x and 1.25x."""

    failures, notes = timing.compare(_doc(p2=3.0 * ratio), _doc(p2=3.0))
    assert bool(failures) is failed
    assert bool(notes) is warned


def test_req299_memory_tolerance_and_p3_bound():
    """REQ-299: 10 % tolerance on peak memory; a P3 figure above its bound fails."""

    failures, _ = timing.compare(_doc(p2_rss=230 * 2**20), _doc(p2_rss=200 * 2**20))
    assert failures
    assert "peak memory 1.15 x" in failures[0]
    failures, _ = timing.compare(_doc(p2_rss=215 * 2**20), _doc(p2_rss=200 * 2**20))
    assert not failures
    run = _doc()
    run["results"].append(
        {
            "id": "P3",
            "workload": "fit",
            "median_s": 1.0,
            "p90_s": 1.0,
            "peak_rss_bytes": 300 * 2**20,
            "bound_bytes": 200 * 2**20,
            "n_repeats": 1,
            "target_met": False,
        }
    )
    failures, _ = timing.compare(run, _doc())
    assert failures == ["P3: peak RSS increase 300 MiB above its bound 200 MiB"]


def test_req299_baseline_is_the_median_of_runs():
    """REQ-299: the baseline is regenerated from three runs (median)."""

    runs = [_doc(p1=v)["results"] for v in (0.04, 0.09, 0.05)]
    rows = timing._median_of_runs(runs)
    assert rows[0]["median_s"] == pytest.approx(0.05)
    assert rows[0]["n_runs"] == 3


def _load_script():
    spec = importlib.util.spec_from_file_location("run_benchmarks", BENCH / "run_benchmarks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_req299_script_exit_code_and_annotations(tmp_path, monkeypatch, capsys):
    """REQ-299: ``run_benchmarks.py --compare`` exits 1 on a 2x P1 and prints a GitHub
    error annotation naming P1 and the ratio; a 1.2x P2 is a warning annotation."""

    script = _load_script()
    baseline = _doc(p1=0.05, p2=3.0)
    (tmp_path / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
    slow = _doc(p1=0.10, p2=3.6)
    monkeypatch.setattr(script.timing, "run_suite", lambda *_a, **_k: copy.deepcopy(slow))
    code = script.main(
        [
            "--suite",
            "ci",
            "--out",
            str(tmp_path / "results"),
            "--compare",
            str(tmp_path / "baseline.json"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "::error::P1 (one waveform): median 0.1000 s is 2.00 x the baseline" in out
    assert "::warning::P2: median 1.20 x the baseline" in out
    assert glob.glob(str(tmp_path / "results" / "*-ci.json"))


def test_req290_results_file_fields(tmp_path):
    """REQ-290 / §9.1: the results file holds {id, median_s, p90_s, peak_rss_bytes,
    n_repeats, environment} per workload (run here on P4 and P6 of the ci suite)."""

    document = timing.run_suite("ci", only=["P4", "P6"])
    path = timing.write_results(document, timing.results_path(document, str(tmp_path)))
    assert os.path.basename(path).endswith("-ci.json")
    rows = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))["results"]
    assert [r["id"] for r in rows] == ["P4", "P6"]
    p6 = rows[1]
    for field in FIELDS:
        assert field in p6
    assert p6["n_repeats"] == 5
    assert p6["median_s"] > 0
    assert p6["environment"]["python"]
    assert p6["environment"]["numpy"]


def test_req290_cli_runs_the_perf_suite(tmp_path):
    """§9.1: the workloads are also reachable as ``evoparam benchmark run --suite perf``
    (here ``--suite ci --task P4``, the workload that only records its status)."""

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "evoparam",
            "benchmark",
            "run",
            "--suite",
            "ci",
            "--task",
            "P4",
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    files = glob.glob(str(tmp_path / "*-ci.json"))
    assert len(files) == 1
    rows = json.loads(pathlib.Path(files[0]).read_text(encoding="utf-8"))["results"]
    assert rows[0]["id"] == "P4"
    assert rows[0]["status"] == "not_available"


def test_req299_baseline_is_committed():
    """REQ-299: the CI baseline is committed as ``benchmarks/baseline-ci.json``."""

    baseline = json.loads((BENCH / "baseline-ci.json").read_text(encoding="utf-8"))
    assert baseline["suite"] == "ci"
    assert baseline["repeat_runs"] == 3
    assert {r["id"] for r in baseline["results"]} >= {"P1", "P2", "P3", "P5", "P6"}


###################################################################################################
## Absolute targets on the reference machine (committed results)


def _committed(suite):
    suffix = "-ci.json" if suite == "ci" else ".json"
    paths = sorted(
        p
        for p in glob.glob(str(BENCH / "results" / "*.json"))
        if p.endswith(suffix) and (suite == "ci" or not p.endswith("-ci.json"))
    )
    assert paths, f"no committed {suite} results under benchmarks/results/"
    rows = json.loads(pathlib.Path(paths[-1]).read_text(encoding="utf-8"))["results"]
    return {timing._key(r): r for r in rows}


# measured on the reference machine (benchmarks/README.md), load average 9 to 11 on 4 cores
MEASURED = {
    "P1": "median 1.30 s over 20 repetitions (target < 0.050 s); one fit of the ccep preset "
    "with 3 starts and max_components=6 on a 1 s window at 2 048 Hz",
    "P3": "peak RSS increase 331 MiB (347 MB) for 105 MB of trials in memory (3.3 x, "
    "target < 2 x; median of 3 runs), "
    "measured on the ci size (50 x 64 x 4 096); the full size (4.19 GB) was not run",
    "P4": "not available: no GPU backend in this version (NumPy/SciPy engine)",
    "P6": "median 0.909 s over 10 runs for import evoparam (target < 0.300 s); "
    "scipy.stats and scipy.interpolate dominate the import time",
}


def _target(key, suite="perf"):
    row = _committed(suite)[key]
    assert row.get("target_met") is True, row


def test_req290_p2_trial_fits_within_60_s():
    """REQ-290 P2: fit_trials on 100 channels x 50 trials with n_jobs=4 in < 60 s."""

    _target("P2")


def test_req290_p3_memory_mapped_bound():
    """REQ-290 P3 (memory-mapped): peak RSS increase < 0.25 x data size + 512 MB."""

    _target("P3/mmap", "ci")


def test_req290_p5_reproduce_within_1_5_x():
    """REQ-290 P5: evoparam.reproduce of a saved 256-channel GroupFit < 1.5 x the fit."""

    _target("P5")


@pytest.mark.xfail(strict=True, reason="REQ-290 P1 not met: " + MEASURED["P1"])
def test_req290_p1_one_waveform_under_50_ms():
    """REQ-290 P1: one averaged waveform fitted in a median < 50 ms."""

    _target("P1")


@pytest.mark.xfail(strict=True, reason="REQ-290 P3 not met: " + MEASURED["P3"])
def test_req290_p3_in_memory_under_twice_the_data():
    """REQ-290 P3 (in memory): peak RSS increase < 2 x data size."""

    _target("P3", "ci")


@pytest.mark.xfail(strict=True, reason="REQ-290 P4 not met: " + MEASURED["P4"])
def test_req290_p4_gpu_equals_cpu():
    """REQ-290 P4: the GPU backend equals the CPU within 1e-6 relative on P1 and P2."""

    _target("P4")


@pytest.mark.xfail(strict=True, reason="REQ-290 P6 not met: " + MEASURED["P6"])
def test_req290_p6_import_under_300_ms():
    """REQ-290 P6: python -X importtime -c "import evoparam", median of 10 < 300 ms."""

    _target("P6")
