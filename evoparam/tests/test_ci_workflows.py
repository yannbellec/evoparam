# SPDX-License-Identifier: Apache-2.0
"""CI workflows, checked statically (spec §9.5 test map: ci.yml, nightly.yml, docs.yml,
release.yml, smoke.yml, perf.yml; REQ-297, REQ-298, REQ-299, REQ-300, REQ-302, REQ-305,
REQ-309 c, REQ-375).

GitHub Actions is not used to run them yet (owner decision D16): the YAML is parsed and the
jobs, matrices and commands the requirements name are checked. Branch protection (required
checks) is a repository setting that cannot be read from the files.
"""

import os
import pathlib
import re
import subprocess
import sys
from typing import Any

import pytest

import evoparam

yaml = pytest.importorskip("yaml")

# Time limits (min) of the test and documentation jobs.
TEST_TIMEOUT = 25
DOCS_TIMEOUT = 20
# Thresholds of REQ-305: line length, complexity, arguments and statements.
LINE_LENGTH = 99
MAX_COMPLEXITY = 12
MAX_ARGS = 8
MAX_STATEMENTS = 60

ROOT = pathlib.Path(evoparam.__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"

pytestmark = pytest.mark.skipif(not WORKFLOWS.is_dir(), reason="needs the source tree")


def _load(name: Any) -> Any:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _on(doc: Any) -> Any:
    return doc.get("on", doc.get(True))  # YAML 1.1 reads the key `on` as True


def _runs(job: Any) -> Any:
    return "\n".join(
        str(step.get("run", "")) + " " + str(step.get("uses", "")) for step in job.get("steps", [])
    )


@pytest.mark.parametrize("name", sorted(p.name for p in WORKFLOWS.glob("*.yml")))
def test_every_workflow_parses(name: Any) -> None:
    doc = _load(name)
    assert doc.get("jobs")
    assert _on(doc)


def test_req297_ci_jobs_and_triggers() -> None:
    doc = _load("ci.yml")
    on = _on(doc)
    assert "pull_request" in on
    assert on["push"]["branches"] == ["main"]
    jobs = doc["jobs"]
    for job in (
        "lint",
        "format",
        "typecheck",
        "pyright",
        "tests",
        "core-only",
        "lowest-floors",
        "doctests",
        "docs",
        "import-graph",
        "build",
    ):
        assert job in jobs, job
    assert "ruff check" in _runs(jobs["lint"])
    assert "ruff format --check" in _runs(jobs["format"])
    assert "mypy --strict" in _runs(jobs["typecheck"])
    assert jobs["pyright"].get("continue-on-error") is True
    assert "pyright --verifytypes evoparam" in _runs(jobs["pyright"])
    assert "--doctest-modules evoparam" in _runs(jobs["doctests"])
    assert "test_import_graph.py" in _runs(jobs["import-graph"])
    assert jobs["docs"]["uses"] == "./.github/workflows/docs.yml"


def test_req297_tests_matrix_lock_and_timeouts() -> None:
    job = _load("ci.yml")["jobs"]["tests"]
    matrix = job["strategy"]["matrix"]
    assert matrix["python"] == ["3.10", "3.11", "3.12", "3.13"]
    assert matrix["os"] == ["ubuntu-latest", "macos-latest", "windows-latest"]
    assert job["name"] == "tests (${{ matrix.python }}, ${{ matrix.os }})"
    assert job["timeout-minutes"] == TEST_TIMEOUT
    runs = _runs(job)
    assert "--require-hashes -r ci/locks/py${{ matrix.python }}.lock" in runs
    assert '".[all]"' in runs
    assert "--timeout=120" in runs
    assert "--cov=evoparam" in runs
    for version in matrix["python"]:
        assert (ROOT / "ci" / "locks" / f"py{version}.lock").is_file()


def test_req297_core_only_and_lowest_floors() -> None:
    jobs = _load("ci.yml")["jobs"]
    core = _runs(jobs["core-only"])
    assert "pip install ." in core
    assert "evoparam.selftest()" in core
    assert "-m core" in core
    floors = _runs(jobs["lowest-floors"])
    assert '"numpy==1.24.*"' in floors
    assert '"scipy==1.10.*"' in floors


def test_req297_build_job() -> None:
    runs = _runs(_load("ci.yml")["jobs"]["build"])
    for command in (
        "python -m build",
        "twine check",
        "check-wheel-contents",
        "py.typed",
        "py3-none-any",
    ):
        assert command in runs, command


def test_req298_coverage_gates() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r"\[tool\.coverage\.report\][^\[]*fail_under = 90", text)
    job = _load("ci.yml")["jobs"]["tests"]
    gates = [s for s in job["steps"] if "coverage report" in str(s.get("run", ""))]
    assert any("--fail-under=90" in s["run"] for s in gates)
    assert any('--include="evoparam/shapes/*" --fail-under=100' in s["run"] for s in gates)
    assert any("--fail-under=80" in s["run"] for s in gates)
    for step in gates:
        assert step["if"] == "matrix.python == '3.12' && matrix.os == 'ubuntu-latest'"
    # the gates measure what REQ-298 names: no gated module is left out of the measurement,
    # the 90 % report covers the core packages, the 80 % report the outer layers
    try:
        import tomllib
    except ImportError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    with open(ROOT / "pyproject.toml", "rb") as fobj:
        coverage = tomllib.load(fobj)["tool"]["coverage"]
    assert coverage["run"]["branch"] is True
    assert coverage["run"]["omit"] == ["evoparam/tests/*"]
    core = coverage["report"]["include"]
    for package in (
        "data",
        "events",
        "artifact",
        "prep",
        "shapes",
        "models",
        "existence",
        "trials",
        "presets",
        "utils",
    ):
        assert f"evoparam/{package}/*" in core, package
    for reader in ("readers", "bids", "brainvision", "mef", "nwb", "neo", "epochs"):
        assert f"evoparam/io/{reader}.py" in core, reader
    outer = next(s["run"] for s in gates if "--fail-under=80" in s["run"])
    for layer in ("plots/*", "cli.py", "stream/*", "benchmark/*", "io/_fetch.py", "io/fetch.py"):
        assert f"evoparam/{layer}" in outer, layer


@pytest.mark.skipif(
    os.environ.get("EVOPARAM_SLOW_TESTS") != "1",
    reason="runs the shapes tests under coverage; set EVOPARAM_SLOW_TESTS=1",
)
def test_req298_shapes_coverage_is_100_percent(tmp_path: Any) -> None:
    pytest.importorskip("coverage")
    env = dict(
        os.environ,
        COVERAGE_FILE=str(tmp_path / ".coverage"),
        PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])),
    )
    run = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--rcfile=/dev/null",
        "--branch",
        "--source=evoparam/shapes",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        str(ROOT / "evoparam" / "tests" / "shapes"),
    ]
    subprocess.run(
        run, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1200, check=True
    )
    report = subprocess.run(
        [sys.executable, "-m", "coverage", "report", "--rcfile=/dev/null", "--fail-under=100"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert report.returncode == 0, report.stdout[-1500:]


def test_req299_perf_workflow() -> None:
    doc = _load("perf.yml")
    on = _on(doc)
    paths = on["pull_request"]["paths"]
    for folder in (
        "evoparam/shapes/**",
        "evoparam/models/**",
        "evoparam/trials/**",
        "evoparam/artifact/**",
        "benchmarks/**",
    ):
        assert folder in paths
    assert "schedule" in on
    job = doc["jobs"]["perf"]
    assert job["runs-on"] == "ubuntu-22.04"
    assert job["env"]["OMP_NUM_THREADS"] == "1"
    runs = _runs(job)
    assert "benchmarks/run_benchmarks.py --suite ci" in runs
    assert "benchmarks/baseline-ci.json" in runs
    assert "--time-tolerance 1.25" in runs
    assert "--memory-tolerance 1.10" in runs
    update = doc["jobs"]["baseline-update"]
    assert "perf-baseline-update" in update["if"]
    assert "--repeat-runs 3" in _runs(update)


def test_req300_nightly_smoke_and_secrets() -> None:
    nightly = _load("nightly.yml")
    assert "schedule" in _on(nightly)
    runs = _runs(nightly["jobs"]["pre-release"])
    assert (
        "--pre --extra-index-url https://pypi.anaconda.org/scientific-python-nightly-wheels"
        "/simple numpy scipy" in runs
    )
    assert "mne-python@main" in runs
    assert "upstream-breakage" in runs
    smoke = _load("smoke.yml")
    assert {"release", "schedule"} <= set(_on(smoke))
    runs = _runs(smoke["jobs"]["smoke"])
    assert "python -m venv" in runs
    assert "pip install" in runs
    assert "evoparam.selftest()" in runs
    assert "evoparam fit --help" in runs
    hits = [p.name for p in WORKFLOWS.glob("*.yml") if "secrets." in p.read_text("utf-8")]
    assert not hits


def test_req309_monthly_lock_refresh() -> None:
    runs = _runs(_load("nightly.yml")["jobs"]["refresh-locks"])
    assert "uv pip compile pyproject.toml" in runs
    assert "--generate-hashes" in runs
    assert "gh pr create" in runs
    lock = (ROOT / "ci" / "locks" / "py3.12.lock").read_text(encoding="utf-8")
    assert "--hash=sha256:" in lock
    assert re.search(r"^numpy==", lock, re.MULTILINE)


def test_req302_docs_workflow() -> None:
    job = _load("docs.yml")["jobs"]["build"]
    runs = _runs(job)
    assert "sphinx-build -W --keep-going -b html" in runs
    assert "sphinx-build -b linkcheck" in runs
    assert job["timeout-minutes"] <= DOCS_TIMEOUT
    conf = (ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
    for ext in (
        "numpydoc",
        "sphinx_gallery.gen_gallery",
        "sphinx.ext.intersphinx",
        "sphinxcontrib.bibtex",
        "pydata_sphinx_theme",
    ):
        assert ext in conf, ext
    assert "run_stale_examples" in conf
    assert "references.bib" in conf
    assert (ROOT / "docs" / "references.bib").is_file()


@pytest.mark.skipif(
    os.environ.get("EVOPARAM_DOCS_TESTS") != "1",
    reason="REQ-302: builds the whole site (several minutes, runs the gallery); set "
    "EVOPARAM_DOCS_TESTS=1 with the docs extra installed (pip install evoparam[docs], as "
    "in the docs CI job); the MNE example downloads the MNE sample dataset",
)
def test_req302_docs_build_with_warnings_as_errors(tmp_path: Any) -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-b",
            "html",
            str(ROOT / "docs"),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_req305_precommit_and_pr_template() -> None:
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    hooks = {h["id"] for repo in config["repos"] for h in repo["hooks"]}
    assert {
        "ruff",
        "ruff-format",
        "mypy",
        "interrogate",
        "check-yaml",
        "end-of-file-fixer",
        "nbstripout",
    } <= hooks
    template = (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")
    for item in (
        "tests added",
        "CHANGELOG entry",
        "docs updated",
        "provenance step added for new public functions",
        "no new hard dependency",
    ):
        assert item in template, item


def test_req305_ruff_configuration() -> None:
    try:
        import tomllib
    except ImportError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    with open(ROOT / "pyproject.toml", "rb") as fobj:
        ruff = tomllib.load(fobj)["tool"]["ruff"]
    assert ruff["line-length"] == LINE_LENGTH
    lint = ruff["lint"]
    for code in (
        "E",
        "W",
        "F",
        "I",
        "N",
        "UP",
        "B",
        "A",
        "C4",
        "SIM",
        "PT",
        "RET",
        "ARG",
        "PL",
        "NPY",
        "D",
        "E722",
        "S110",
        "BLE001",
    ):
        assert code in lint["select"], code
    assert lint["pydocstyle"]["convention"] == "numpy"
    assert lint["mccabe"]["max-complexity"] == MAX_COMPLEXITY
    assert lint["pylint"]["max-args"] == MAX_ARGS
    assert lint["pylint"]["max-statements"] == MAX_STATEMENTS


def _ruff(*args: Any) -> Any:
    import shutil
    import subprocess

    exe = shutil.which("ruff")
    if exe is None:
        pytest.skip("ruff is not installed")
    return subprocess.run(
        [exe, *args], cwd=ROOT, capture_output=True, text=True, timeout=300, check=False
    )


def test_req305_ruff_check_passes() -> None:
    result = _ruff("check", "evoparam", "benchmarks", "docs/tutorials", "examples")
    assert result.returncode == 0, result.stdout[-1500:]


def test_req305_ruff_format_passes() -> None:
    result = _ruff("format", "--check", "evoparam", "benchmarks", "docs/tutorials", "examples")
    assert result.returncode == 0, result.stdout[-1500:]


def test_req305_docstring_coverage() -> None:
    import shutil
    import subprocess
    import sys

    exe = shutil.which("interrogate") or str(pathlib.Path(sys.executable).parent / "interrogate")
    if not pathlib.Path(exe).exists():
        pytest.skip("interrogate is not installed")
    result = subprocess.run(
        [exe, "--fail-under", "95", "evoparam", "-e", "evoparam/tests"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-500:]


def test_req307_ruff_blind_except_rules() -> None:
    """E722, S110 and BLE001 report nothing in the package (14 findings before the
    qualite-statique branch, 6 before reglages-ignores: every ``except Exception`` names
    its exception classes, or records or re-raises the error with a justified noqa)."""
    result = _ruff(
        "check", "--select", "E722,S110,BLE001", "evoparam", "--exclude", "evoparam/tests"
    )
    assert result.returncode == 0, result.stdout[-1500:]


def test_req375_release_pipeline_runs_the_robustness_layer() -> None:
    jobs = _load("release.yml")["jobs"]
    runs = _runs(jobs["validation"])
    assert "validation/robustness.py" in runs
    assert "tools/layer3_gate.py" in runs
    assert any(
        "validation/layer4/" in str(s.get("with", {}).get("path", ""))
        for s in jobs["validation"]["steps"]
    )
