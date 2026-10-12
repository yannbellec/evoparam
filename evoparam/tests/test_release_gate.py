# SPDX-License-Identifier: Apache-2.0
"""REQ-019 (no public step before the institutional decision), REQ-017 (the N1 tests in every
CI build) and REQ-018 (every gap traced).

REQ-019: while the gate of RELEASING.md says that the HIIE decision is not on record, nothing
in the repository may publish: the package carries the classifier that PyPI refuses, no
workflow uploads a package, deploys pages or deposits to Zenodo, ``make publish`` stops, and
the citation file announces no DOI. Once the decision is on record, the gate must name the
date and the reference of each record.
"""

import pathlib
import re
import subprocess
import sys

import pytest

import evoparam

pytestmark = pytest.mark.core

ROOT = pathlib.Path(evoparam.__file__).resolve().parent.parent
PUBLISHING = re.compile(
    r"pypa/gh-action-pypi-publish|twine upload(?!.*printf)|"
    r"actions/deploy-pages|peaceiris/actions-gh-pages|zenodo|"
    r"conda-forge.*upload|anaconda upload|flit publish|poetry publish|"
    r"uv publish|hatch publish",
    re.IGNORECASE,
)


def gate():
    text = (ROOT / "RELEASING.md").read_text(encoding="utf-8")
    block = text.split("<!-- release-gate -->")[1].split("<!-- end-release-gate -->")[0]
    items = dict(re.findall(r"^- ([^:]+): (.+)$", block, re.MULTILINE))
    assert set(items) == {"HIIE decision", "Invention disclosure (M0)", "Decision record (M1)"}
    return items


def _decided():
    return gate()["HIIE decision"].startswith("on record")


def test_req019_the_gate_is_readable():
    items = gate()
    if _decided():
        assert re.search(r"on record, \d{4}-\d{2}-\d{2}, \S+", items["HIIE decision"])
        assert items["Decision record (M1)"] != "none"
    else:
        assert items["HIIE decision"] == "not on record"


def test_req019_no_upload_before_the_decision():
    if _decided():
        pytest.skip("the HIIE decision is on record (RELEASING.md): public steps are allowed")
    try:
        import tomllib
    except ImportError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    with open(ROOT / "pyproject.toml", "rb") as fobj:
        project = tomllib.load(fobj)["project"]
    assert "Private :: Do Not Upload" in project["classifiers"]
    hits = []
    for path in sorted((ROOT / ".github").rglob("*.y*ml")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if PUBLISHING.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not hits, hits
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert not re.search(r"^\s*doi:", citation, re.MULTILINE)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="make is not a Windows tool")
def test_req019_make_publish_stops_before_the_decision():
    if _decided():
        pytest.skip("the HIIE decision is on record (RELEASING.md)")
    result = subprocess.run(
        ["make", "-s", "publish"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode != 0
    assert "HIIE decision" in result.stdout + result.stderr


def test_req017_n1_tests_run_in_every_build():
    workflow = (ROOT / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")
    triggers = workflow.split("\njobs:")[0]
    assert "push:" in triggers
    assert "pull_request:" in triggers
    build = workflow.split("\n  build:")[1].split("\n  docs:")[0]
    assert "\n    if:" not in build.split("steps:")[0]  # the job runs on every trigger
    step = build.split("REQ-113 and REQ-114 in every build (REQ-017)")[1].split("- name:")[0]
    assert 'EVOPARAM_NETWORK_TESTS: "1"' in step
    assert "evoparam/tests/models/test_engine_ccep.py" in step
    assert "\n      if:" not in step
    engine = (ROOT / "evoparam" / "tests" / "models" / "test_engine_ccep.py").read_text(
        encoding="utf-8"
    )
    assert re.search(r"def test_req113_", engine)
    assert re.search(r"def test_req114_", engine)


def test_req018_every_addressed_gap_is_traced():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_traceability.py")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "49 gaps marked Addressed" in result.stdout


def test_req018_the_check_finds_a_broken_mapping(tmp_path):
    spec = tmp_path / "docs" / "spec"
    spec.mkdir(parents=True)
    (spec / "GAPS.md").write_text(
        (ROOT / "docs" / "spec" / "GAPS.md").read_text("utf-8"), encoding="utf-8"
    )
    text = (ROOT / "docs" / "spec" / "SPEC.md").read_text("utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| GAP-03 |"))
    cells = row.split("|")
    cells[6] = " "  # no test file for GAP-03
    (spec / "SPEC.md").write_text(text.replace(row, "|".join(cells)), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_traceability.py"), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 1
    assert "GAP-03: no test file" in result.stdout


###################################################################################################
## REQ-398: the release workflow refuses to publish without the recorded decision


def _workflow(name):
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))


def test_req398_release_workflow_starts_with_the_gate():
    jobs = _workflow("release.yml")["jobs"]
    steps = " ".join(str(s.get("run", "")) for s in jobs["gate"]["steps"])
    assert "tools/release_gate.py" in steps
    for name, job in jobs.items():
        if name != "gate":
            needs = job.get("needs", [])
            needs = [needs] if isinstance(needs, str) else needs
            assert "gate" in needs, f"job {name} does not wait for the gate"


def test_req398_the_gate_refuses_now():
    if _decided():
        pytest.skip("the HIIE decision is on record (RELEASING.md)")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "release_gate.py")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 1
    assert "GOVERNANCE.md" in result.stdout
    assert "RELEASING.md" in result.stdout


def test_req398_the_gate_passes_with_a_recorded_decision(tmp_path):
    governance = (ROOT / "GOVERNANCE.md").read_text(encoding="utf-8")
    governance = governance.replace(
        "- Invention disclosure: not filed",
        "- Invention disclosure: HIIE disclosure 2027-001, 2027-01-15",
    ).replace(
        "- HIIE decision reference: none",
        "- HIIE decision reference: HIIE ref. 2027-123, 2027-03-01 (patent: no)",
    )
    releasing = (
        (ROOT / "RELEASING.md")
        .read_text(encoding="utf-8")
        .replace(
            "- HIIE decision: not on record",
            "- HIIE decision: on record, 2027-03-01, HIIE-2027-123",
        )
    )
    (tmp_path / "GOVERNANCE.md").write_text(governance, encoding="utf-8")
    (tmp_path / "RELEASING.md").write_text(releasing, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "release_gate.py"), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout


def test_req398_no_repository_secret_in_any_workflow():
    """REQ-300 (d): no workflow uses a repository secret (trusted publishing needs none)."""

    hits = [
        f"{p.name}:{n}"
        for p in sorted((ROOT / ".github").rglob("*.y*ml"))
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if "secrets." in line
    ]
    assert not hits, hits


def test_req398_the_upload_job_waits_outside_the_workflows():
    """The trusted-publishing job is kept in ci/ until the decision (RELEASING.md step 9)."""

    job = (ROOT / "ci" / "release-publish-job.yml").read_text(encoding="utf-8")
    assert "pypa/gh-action-pypi-publish" in job
    assert "id-token: write" in job
    assert "step 9" in (ROOT / "RELEASING.md").read_text(
        encoding="utf-8"
    ).lower() or "ci/release-publish-job.yml" in (ROOT / "RELEASING.md").read_text(
        encoding="utf-8"
    )


@pytest.mark.xfail(
    strict=True,
    reason="REQ-398: the invention disclosure and the HIIE "
    "written decision are not on record; waits for the owner's HIIE decision "
    "(nothing public before it)",
)
def test_req398_decision_on_record():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "release_gate.py")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0
