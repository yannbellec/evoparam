# SPDX-License-Identifier: Apache-2.0
"""Every command of ``evoparam`` through ``subprocess``, compared with the Python API with the
same seed (REQ-345), and ``evoparam cite`` (REQ-346).

Every output file must be byte-identical, JSON included. The creation time and the step
durations are the only parts of a result that depend on when and how fast it was made;
with ``SOURCE_DATE_EPOCH`` set (the reproducible-builds convention, read by
:func:`evoparam.data.provenance.fixed_clock`) they are fixed, in the command line and in
the Python API alike.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

import evoparam
from evoparam import __version__, presets, reports

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(evoparam.__file__)))
EPOCH = "1767225600"                      # 2026-01-01T00:00:00Z
SIM = ["--modality", "ccep", "--n-trials", "8", "--n-channels", "3", "--random-state", "3"]


def run(*args, code=0):
    """``python -m evoparam <args> --json``: the parsed summary (exit code checked)."""

    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYTHONWARNINGS="ignore")
    proc = subprocess.run([sys.executable, "-m", "evoparam", *map(str, args), "--json"],
                          capture_output=True, text=True, env=env, cwd=ROOT, timeout=900)
    assert proc.returncode == code, (args, proc.stdout[-2000:], proc.stderr[-2000:])
    summary = json.loads(proc.stdout.splitlines()[-1])
    assert summary["exit_code"] == code and summary["version"] == __version__
    return summary


def run_text(*args):
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYTHONWARNINGS="ignore")
    proc = subprocess.run([sys.executable, "-m", "evoparam", *map(str, args)],
                          capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    return proc.returncode, proc.stdout


@pytest.fixture(scope="module", autouse=True)
def fixed_clock():
    """The same fixed clock for the command line (inherited environment) and the API."""

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SOURCE_DATE_EPOCH", EPOCH)
        yield


def _files(root):
    out = {}
    for folder, _, names in os.walk(root):
        for name in names:
            path = os.path.join(folder, name)
            out[os.path.relpath(path, root)] = path
    return out


def assert_same_tree(cli_root, api_root, only=None):
    cli, api = _files(cli_root), _files(api_root)
    if only:
        cli = {k: v for k, v in cli.items() if k.endswith(only)}
        api = {k: v for k, v in api.items() if k.endswith(only)}
    assert sorted(cli) == sorted(api)
    for name, path in cli.items():
        with open(path, "rb") as a, open(api[name], "rb") as b:
            left, right = a.read(), b.read()
        assert left == right, name


@pytest.fixture(scope="module")
def bids(tmp_path_factory):
    root = tmp_path_factory.mktemp("cmd") / "bids"
    run("simulate", *SIM, "--out", root)
    return root


def _trials(bids, preset="ccep"):
    from evoparam.io import read_bids

    return read_bids(str(bids), "01", task="ccep", datatype="ieeg",
                     epoch=presets.get(preset).epoch)


def test_req345_simulate(bids, tmp_path):

    from evoparam.sim import GENERATORS
    from evoparam.sim.bids import write_bids

    trials, truth = GENERATORS["ccep"](n_trials=8, n_channels=3, seed=3)
    write_bids(trials, truth, str(tmp_path / "bids"), subject="01", task="ccep")
    assert_same_tree(bids, tmp_path / "bids")


def test_req345_test(bids, tmp_path):

    from evoparam.existence import test_response
    from evoparam.io import write_bids_derivatives
    from evoparam.prep import baseline_correct

    run("test", bids, "--sub", "01", "--task", "ccep", "--preset", "ccep",
        "--out", tmp_path / "cli")
    p = presets.get("ccep")
    trials = baseline_correct(_trials(bids), p.baseline, check=False)
    test = test_response(trials, method="baseline_permutation", window=p.fit_window,
                         baseline=p.baseline, random_state=0, by=())
    write_bids_derivatives(test, str(bids), subject="01", task="ccep", datatype="ieeg",
                           out=str(tmp_path / "api"), overwrite=True, trials=trials)
    assert_same_tree(tmp_path / "cli", tmp_path / "api", only=("_existence.tsv",
                                                                "_existence.json"))


def test_req345_clean(bids, tmp_path):

    from evoparam.artifact import remove_artifact
    from evoparam.data import load_object
    from evoparam.io import write_bids_derivatives

    summary = run("clean", bids, "--sub", "01", "--task", "ccep", "--preset", "ccep",
                  "--out", tmp_path / "cli")
    p = presets.get("ccep")
    source = _trials(bids)
    result = remove_artifact(source, method="template", window=p.artifact_window,
                             random_state=0)
    write_bids_derivatives(result, str(bids), subject="01", task="ccep", datatype="ieeg",
                           out=str(tmp_path / "api"), overwrite=True, trials=source)
    assert_same_tree(tmp_path / "cli", tmp_path / "api", only=("_artifact.json",))
    cleaned = next(p for p in summary["outputs"] if p.endswith("cleaned.evoparam"))
    assert load_object(cleaned).data.tobytes() == result.cleaned.data.tobytes()


def test_req345_detect_events(bids, tmp_path):

    import pandas as pd

    from evoparam.events import detect_events
    from evoparam.io import read_brainvision

    vhdr = bids / "sub-01" / "ieeg" / "sub-01_task-ccep_ieeg.vhdr"
    out = tmp_path / "events.tsv"
    summary = run("detect-events", vhdr, "--out", out)
    data, sfreq, names, *_ = read_brainvision(str(vhdr))
    events = detect_events(data, sfreq=sfreq, ch_names=names)
    table = pd.read_csv(out, sep="\t")
    assert list(table.columns[:2]) == ["onset", "duration"]
    assert list(table.columns[2:]) == list(events.columns)
    assert summary["n_units"]["ok"] == len(events) == 8
    assert table["onset"].tolist() == pytest.approx(events["onset_time"].tolist(), abs=0)
    sidecar = json.loads((tmp_path / "events.json").read_text())
    assert set(sidecar) == set(table.columns) and sidecar["onset"]["Units"] == "s"


def test_req345_report_json_and_migrate(bids, tmp_path):

    from evoparam.data import load_object

    run("fit", bids, "--sub", "01", "--task", "ccep", "--preset", "ccep", "--out",
        tmp_path / "deriv")
    container = next(str(p) for p in (tmp_path / "deriv").rglob("*.evoparam"))
    summary = run("report", container, "--format", "json", "--out", tmp_path / "rep")
    group = load_object(container)
    assert reports.validate_json(summary["outputs"][0])
    again = reports.from_json(summary["outputs"][0])
    assert again.to_dataframe().equals(group.to_dataframe())
    copy = tmp_path / "copy.evoparam"
    shutil.copytree(container, copy)
    summary = run("migrate", copy)
    assert summary["result"]["to_schema"].startswith("evoparam-result/1")
    assert load_object(str(copy)) == group


def test_req345_information_commands():

    info = run("info")
    assert info["result"]["evoparam"] == __version__
    listing = run("presets", "list")
    assert {row["name"] for row in listing["result"]["presets"]} == set(presets.PRESETS)
    shown = run("presets", "show", "ccep")
    assert shown["result"]["preset"]["name"] == "ccep"
    assert run_text("benchmark")[0] == 2                          # an action is required
    # the repository is not a BIDS dataset: input error (REQ-314, tests/io/test_bids_validator.py)
    assert run("validate-bids", ROOT, code=5)["status"] == "error"
    code, stdout = run_text("--version")
    assert code == 0 and __version__ in stdout and "research use" in stdout.lower()


def test_req346_cite_prints_citation_cff():

    code, stdout = run_text("cite")
    assert code == 0 and stdout.strip() == reports.citation()
    code, stdout = run_text("cite", "--bibtex")
    assert code == 0 and stdout.strip() == reports.citation(bibtex=True).strip()
    assert run("cite")["result"]["citation"] == reports.citation()


def test_methods_text_is_logged_at_info_level(bids, tmp_path):
    """Spec §5.13: the methods paragraph ends a fit when verbose is INFO."""

    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYTHONWARNINGS="ignore")
    proc = subprocess.run([sys.executable, "-m", "evoparam", "fit", str(bids), "--sub", "01",
                           "--task", "ccep", "--preset", "ccep", "--out",
                           str(tmp_path / "d"), "--verbose", "info"],
                          capture_output=True, text=True, env=env, cwd=ROOT, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "Evoked responses were analysed with evoparam" in proc.stderr
    assert "Software:" in proc.stderr
