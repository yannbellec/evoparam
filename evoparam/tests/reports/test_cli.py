"""The evoparam command (REQ-322, REQ-323, REQ-345, REQ-346)."""

import hashlib
import json
import os
import subprocess
import sys

import pytest

from evoparam.cli import main


def _run(*args, cwd=None):
    proc = subprocess.run([sys.executable, "-m", "evoparam", *args], capture_output=True,
                          text=True, cwd=cwd)
    return proc


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli")
    proc = _run("simulate", "--modality", "ccep", "--n-trials", "10", "--n-channels", "5",
                "--random-state", "0", "--out", str(root / "bids"), "--json")
    assert proc.returncode == 0, proc.stderr
    return root


@pytest.fixture(scope="module")
def fitted(dataset):
    out = dataset / "deriv"
    proc = _run("fit", str(dataset / "bids"), "--sub", "01", "--task", "ccep", "--preset",
                "ccep", "--out", str(out), "--json")
    return proc, out


def test_req345_fit_writes_the_derivatives(fitted):

    proc, out = fitted
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["command"] == "fit" and summary["exit_code"] == 0
    names = {os.path.basename(p) for p in summary["outputs"]}
    for suffix in ("responses.tsv", "responses.json", "existence.tsv", "trials.tsv",
                   "artifact.json"):
        assert f"sub-01_task-ccep_desc-evoparam_{suffix}" in names
    assert sum(summary["n_units"].values()) == 5
    description = json.loads((out / "dataset_description.json").read_text())
    assert description["DatasetType"] == "derivative"
    assert description["GeneratedBy"][0]["Name"] in out.name or "evoparam" in \
        description["GeneratedBy"][0]["Name"]


def test_req345_cli_equals_python(fitted, dataset, tmp_path):

    from evoparam import presets
    from evoparam.cli import run_fit
    from evoparam.io import read_bids, write_bids_derivatives

    _, out = fitted
    p = presets.get("ccep")
    ts = read_bids(str(dataset / "bids"), "01", task="ccep", datatype="ieeg", epoch=p.epoch)
    results, _ = run_fit(ts, p, random_state=0)
    written = write_bids_derivatives(results, str(dataset / "bids"), subject="01", task="ccep",
                                     out=str(tmp_path), overwrite=True)
    for path in written:
        if path.endswith(".tsv") and "sub-01" in path:
            cli = out / os.path.relpath(path, tmp_path)
            assert hashlib.sha256(open(path, "rb").read()).hexdigest() == \
                hashlib.sha256(open(cli, "rb").read()).hexdigest(), path


def test_report_reproduce_migrate(fitted, tmp_path):

    _, out = fitted
    for fmt in ("html", "md", "pdf"):
        proc = _run("report", str(out), "--format", fmt, "--out", str(tmp_path), "--json")
        assert proc.returncode == 0, proc.stderr
        assert json.loads(proc.stdout)["outputs"][0].endswith(f".{fmt}")
    container = next(str(p) for p in out.rglob("*.evoparam"))
    proc = _run("reproduce", container, "--json")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["result"]["identical"] is True
    proc = _run("migrate", container, "--out", str(tmp_path / "copy.evoparam"), "--json")
    assert proc.returncode == 0, proc.stderr


def test_small_commands(capsys):

    assert main(["--version"]) == 0
    assert "research use" in capsys.readouterr().out
    assert main(["cite"]) == 0
    assert capsys.readouterr().out.startswith("Bellec Y.")
    assert main(["presets", "list", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["result"]["presets"]) >= 11
    assert main(["presets", "show", "ccep"]) == 0
    assert main(["info", "--json"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["result"]["evoparam"]


def test_req323_exit_codes(fitted, tmp_path, monkeypatch, capsys):

    _, out = fitted
    assert main(["fit"]) == 2                                   # usage
    assert main(["benchmark", "--json"]) == 1                    # runtime (not built yet)
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["exit_code"] == 1 and summary["status"] == "error"
    bad = tmp_path / "bad.evoparam"
    bad.mkdir()
    (bad / "manifest.json").write_text("{not json")
    assert main(["report", str(bad)]) == 5                       # input error
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    container = next(str(p) for p in out.rglob("*.evoparam"))
    assert main(["report", container, "--format", "pdf", "--out", str(tmp_path)]) == 4
    monkeypatch.delitem(sys.modules, "matplotlib")
    # a changed setting in the saved provenance: reproduction differs (exit 6)
    import shutil

    copy = tmp_path / "changed.evoparam"
    shutil.copytree(container, copy)
    # the result's provenance lives in structure.json (provenance.json is a readable copy)
    structure = (copy / "structure.json").read_text()
    assert '"max_components": 6' in structure
    (copy / "structure.json").write_text(structure.replace('"max_components": 6',
                                                           '"max_components": 1'))
    manifest = json.loads((copy / "manifest.json").read_text())
    manifest["files"]["structure.json"] = hashlib.sha256(
        (copy / "structure.json").read_bytes()).hexdigest()
    (copy / "manifest.json").write_text(json.dumps(manifest))
    assert main(["reproduce", str(copy)]) == 6
