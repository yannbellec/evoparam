# SPDX-License-Identifier: Apache-2.0
"""The evoparam command (REQ-322, REQ-323, REQ-345, REQ-346)."""

import hashlib
import json
import os
import subprocess
import sys

import pytest

from evoparam.cli import main


def _run(*args, cwd=None):
    proc = subprocess.run(
        [sys.executable, "-m", "evoparam", *args], capture_output=True, text=True, cwd=cwd
    )
    return proc


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli")
    proc = _run(
        "simulate",
        "--modality",
        "ccep",
        "--n-trials",
        "10",
        "--n-channels",
        "5",
        "--random-state",
        "0",
        "--out",
        str(root / "bids"),
        "--json",
    )
    assert proc.returncode == 0, proc.stderr
    return root


@pytest.fixture(scope="module")
def fitted(dataset):
    out = dataset / "deriv"
    proc = _run(
        "fit",
        str(dataset / "bids"),
        "--sub",
        "01",
        "--task",
        "ccep",
        "--preset",
        "ccep",
        "--out",
        str(out),
        "--json",
    )
    return proc, out


def test_req345_fit_writes_the_derivatives(fitted):

    proc, out = fitted
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["command"] == "fit" and summary["exit_code"] == 0
    names = {os.path.basename(p) for p in summary["outputs"]}
    for suffix in (
        "responses.tsv",
        "responses.json",
        "existence.tsv",
        "trials.tsv",
        "artifact.json",
    ):
        assert f"sub-01_task-ccep_desc-evoparam_{suffix}" in names
    assert sum(summary["n_units"].values()) == 5
    description = json.loads((out / "dataset_description.json").read_text())
    assert description["DatasetType"] == "derivative"
    assert (
        description["GeneratedBy"][0]["Name"] in out.name
        or "evoparam" in description["GeneratedBy"][0]["Name"]
    )


def test_req345_cli_equals_python(fitted, dataset, tmp_path):

    from evoparam import presets
    from evoparam.cli import run_fit
    from evoparam.io import read_bids, write_bids_derivatives

    _, out = fitted
    p = presets.get("ccep")
    ts = read_bids(str(dataset / "bids"), "01", task="ccep", datatype="ieeg", epoch=p.epoch)
    results, _ = run_fit(ts, p, random_state=0)
    written = write_bids_derivatives(
        results,
        str(dataset / "bids"),
        subject="01",
        task="ccep",
        out=str(tmp_path / "evoparam"),
        overwrite=True,
        trials=ts,
    )
    for path in written:
        if path.endswith(".tsv") and "sub-01" in path:
            cli = out / os.path.relpath(path, tmp_path / "evoparam")
            assert (
                hashlib.sha256(open(path, "rb").read()).hexdigest()
                == hashlib.sha256(open(cli, "rb").read()).hexdigest()
            ), path


def test_report_reproduce_migrate(fitted, tmp_path):

    _, out = fitted
    for fmt in ("html", "md", "pdf"):
        proc = _run("report", str(out), "--format", fmt, "--out", str(tmp_path), "--json")
        assert proc.returncode == 0, proc.stderr
        assert json.loads(proc.stdout)["outputs"][0].endswith(f".{fmt}")
    container = next(str(p) for p in out.rglob("*_responses.evoparam"))
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
    assert main(["fit"]) == 2  # usage
    spec = tmp_path / "missing_tool.json"  # runtime: the external
    spec.write_text(
        json.dumps(
            {
                "name": "missing",
                "tasks": ["T7"],  # tool cannot be found
                "command": [str(tmp_path / "no-such-tool")],
            }
        )
    )
    assert (
        main(
            [
                "benchmark",
                "run",
                "--suite",
                "core",
                "--tasks",
                "T7",
                "--methods",
                "",
                "--adapter",
                str(spec),
                "--out",
                str(tmp_path / "bench"),
                "--json",
            ]
        )
        == 1
    )
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["exit_code"] == 1 and summary["status"] == "error"
    assert main(["benchmark", "--json"]) == 2  # usage: no action
    capsys.readouterr()
    bad = tmp_path / "bad.evoparam"
    bad.mkdir()
    (bad / "manifest.json").write_text("{not json")
    assert main(["report", str(bad)]) == 5  # input error
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    container = next(str(p) for p in out.rglob("*_responses.evoparam"))
    assert main(["report", container, "--format", "pdf", "--out", str(tmp_path)]) == 4
    monkeypatch.delitem(sys.modules, "matplotlib")
    # a changed setting in the saved provenance: reproduction differs (exit 6)
    import shutil

    copy = tmp_path / "changed.evoparam"
    shutil.copytree(container, copy)
    # the result's provenance lives in structure.json (provenance.json is a readable copy)
    structure = (copy / "structure.json").read_text()
    assert '"max_components": 6' in structure
    (copy / "structure.json").write_text(
        structure.replace('"max_components": 6', '"max_components": 1')
    )
    manifest = json.loads((copy / "manifest.json").read_text())
    manifest["files"]["structure.json"] = hashlib.sha256(
        (copy / "structure.json").read_bytes()
    ).hexdigest()
    (copy / "manifest.json").write_text(json.dumps(manifest))
    assert main(["reproduce", str(copy)]) == 6


def test_req323_partial_and_interrupted(fitted, monkeypatch, capsys):
    """Exit 3 with a channel whose fit failed, 130 when interrupted."""

    import dataclasses

    from evoparam import cli

    _, out = fitted
    dataset = out.parent
    real = cli.run_fit

    def one_channel_fails(ts, preset, **kwargs):
        results, notes = real(ts, preset, **kwargs)
        group = results[0]
        broken = dataclasses.replace(
            group.fits[0], status="failed", components=[], message="forced failure (test)"
        )
        results[0] = dataclasses.replace(group, fits=[broken] + list(group.fits[1:]))
        return results, notes

    monkeypatch.setattr(cli, "run_fit", one_channel_fails)
    code = main(
        [
            "fit",
            str(dataset / "bids"),
            "--sub",
            "01",
            "--task",
            "ccep",
            "--preset",
            "ccep",
            "--out",
            str(dataset / "partial"),
            "--json",
        ]
    )
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert code == 3 and summary["status"] == "partial" and summary["n_units"]["failed"] >= 1

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_fit", interrupted)
    code = main(
        [
            "fit",
            str(dataset / "bids"),
            "--sub",
            "01",
            "--task",
            "ccep",
            "--out",
            str(dataset / "stopped"),
            "--json",
        ]
    )
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert code == 130 and summary["status"] == "interrupted"


def test_req322_single_file_default_preset_and_toml(dataset, tmp_path, capsys):
    """fit on one recording (its BIDS events.tsv next to it), no --preset (ccep for iEEG),
    options from a TOML file with a table for the command."""

    vhdr = next((dataset / "bids").rglob("*_ieeg.vhdr"))
    config = tmp_path / "options.toml"
    config.write_text("random-state = 0\n[fit]\nno-existence = true\n")
    code = main(
        ["fit", str(vhdr), "--out", str(tmp_path / "evoparam"), "--config", str(config), "--json"]
    )
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert code in (0, 3), summary
    names = {os.path.basename(p) for p in summary["outputs"]}
    assert "sub-01_task-ccep_desc-evoparam_responses.tsv" in names
    assert not any(n.endswith("_existence.tsv") for n in names)  # from [fit]
    from evoparam.io import read_bids_derivatives

    group = read_bids_derivatives(str(tmp_path / "evoparam"), subject="01", task="ccep")
    from evoparam import presets

    # the preset is recorded as {"name", "version"} (REQ-331)
    assert group.settings["preset"] == {"name": "ccep", "version": presets.get("ccep").version}

    # .npz recording: data [trials, channels, times], sfreq, tmin
    import numpy as np

    from evoparam import sim

    trials, _ = sim.simulate_ccep(n_trials=6, n_channels=3, seed=1)
    npz = tmp_path / "sub-07_task-spes_ieeg.npz"
    np.savez(
        npz,
        data=trials.data,
        sfreq=trials.sfreq,
        tmin=trials.times[0],
        ch_names=np.array(trials.ch_names),
    )
    code = main(["test", str(npz), "--out", str(tmp_path / "evoparam-npz"), "--json"])
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert code == 0, summary
    assert any(
        p.endswith("sub-07_task-spes_desc-evoparam_existence.tsv") for p in summary["outputs"]
    )


def test_req322_clean_writes_epochs_and_info_backends(dataset, tmp_path, capsys):

    code = main(
        [
            "clean",
            str(dataset / "bids"),
            "--sub",
            "01",
            "--task",
            "ccep",
            "--method",
            "template",
            "--window",
            "-0.002",
            "0.010",
            "--out",
            str(tmp_path / "evoparam"),
            "--json",
        ]
    )
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert code == 0, summary
    epochs = [p for p in summary["outputs"] if p.endswith(("_epo.fif", "_epo.npz"))]
    assert len(epochs) == 1 and os.path.isfile(epochs[0])
    assert any(p.endswith("_artifact.json") for p in summary["outputs"])
    assert main(["info", "--json"]) == 0
    info = json.loads(capsys.readouterr().out.splitlines()[-1])["result"]
    assert info["backends"]["fit"] == {"numpy": True}
    assert set(info["backends"]["hierarchical"]) == {"eb", "numpyro", "pymc"}
    assert (
        main(
            [
                "simulate",
                "--modality",
                "ccep",
                "--n-trials",
                "4",
                "--n-channels",
                "3",
                "--seed",
                "5",
                "--out",
                str(tmp_path / "sim"),
                "--json",
            ]
        )
        == 0
    )


def test_req322_fit_on_a_fif_file(tmp_path, capsys):

    mne = pytest.importorskip("mne")
    from evoparam import sim

    trials, _ = sim.simulate_ccep(n_trials=8, n_channels=3, seed=2)
    path = tmp_path / "sub-02_task-spes_epo.fif"
    trials.to_mne().save(str(path), verbose="error")
    code = main(
        [
            "fit",
            str(path),
            "--preset",
            "ccep",
            "--no-existence",
            "--out",
            str(tmp_path / "evoparam"),
            "--json",
        ]
    )
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert code in (0, 3), summary
    assert any(
        p.endswith("sub-02_task-spes_desc-evoparam_responses.tsv") for p in summary["outputs"]
    )
    del mne
