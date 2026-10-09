"""BIDS derivatives (REQ-336, REQ-342) and the simulated raw dataset they come from.

The reference BIDS validator runs when it is available: set ``EVOPARAM_BIDS_VALIDATOR`` to
its command (for example ``npx bids-validator@2`` or ``deno run -A jsr:@bids/validator``),
or put ``bids-validator`` on the PATH; the CI job of REQ-342 installs it.
"""

import csv
import json
import os
import shlex
import shutil
import subprocess
import warnings

import numpy as np
import pytest

from evoparam import __version__, reports, sim
from evoparam.data import Provenance
from evoparam.io import read_bids, write_bids_derivatives
from evoparam.sim.bids import write_bids

TSV_SUFFIXES = ("_responses.tsv", "_existence.tsv", "_trials.tsv")


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    """A simulated BIDS-iEEG dataset, the ccep pipeline run on it, and its derivatives."""

    root = tmp_path_factory.mktemp("bidsderiv") / "bids"
    trials, truth = sim.simulate_ccep(n_trials=8, n_channels=3, sfreq=1024.0, seed=2)
    write_bids(trials, truth, str(root), subject="01", task="spes")
    read = read_bids(str(root), "01", task="spes", datatype="ieeg", epoch=(-0.5, 1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = reports.run_pipeline(read, "ccep", trial_fits=True)
    out = root / "derivatives" / f"evoparam-{__version__}"
    files = write_bids_derivatives(result.results, str(root), subject="01", task="spes",
                                   datatype="ieeg", out=str(out))
    return {"root": root, "out": out, "files": files, "result": result, "trials": trials,
            "read": read}


def _json(files, suffix):
    path = next(f for f in files if f.endswith(suffix))
    with open(path, encoding="utf-8") as fobj:
        return json.load(fobj)


def test_provenance_sidecar(written):
    """REQ-336: the sidecars and dataset_description GeneratedBy hold the provenance."""

    group = written["result"].group
    sidecar = _json(written["files"], "_responses.json")
    assert Provenance.from_dict(sidecar["Provenance"]) == group.provenance
    assert sidecar["PackageVersion"] == __version__
    assert sidecar["Preset"] == {"Name": "ccep", "Version": "1.0"}
    assert sidecar["Settings"]["max_components"] == group.settings["max_components"]
    description = _json(written["files"], "dataset_description.json")
    assert description["DatasetType"] == "derivative"
    generated = description["GeneratedBy"][0]
    assert generated["Name"] == "evoparam" and generated["Version"] == __version__
    assert Provenance.from_dict(generated["Provenance"]) == group.provenance
    existence = _json(written["files"], "_existence.json")
    test = written["result"].test
    assert Provenance.from_dict(existence["Provenance"]) == test.provenance
    artifact = _json(written["files"], "_artifact.json")
    last = written["result"].artifact[-1]
    assert Provenance.from_dict(artifact["Provenance"]) == last.provenance


def test_nwb_provenance(written):
    """REQ-336: the NWB processing module holds the settings and provenance of each result."""

    pynwb = pytest.importorskip("pynwb")
    from datetime import datetime, timezone

    from evoparam import io

    nwbfile = pynwb.NWBFile(session_description="evoparam test", identifier="evoparam-test",
                            session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc))
    group, test = written["result"].group, written["result"].test
    io.write_nwb([group, test], nwbfile)
    description = json.loads(nwbfile.processing["evoparam"].description)
    assert [Provenance.from_dict(p) for p in description["provenance"]] == \
        [group.provenance, test.provenance]
    assert description["settings"][0]["max_components"] == group.settings["max_components"]


def test_req342_every_column_has_a_description_and_unit(written):

    for suffix in TSV_SUFFIXES:
        path = next(f for f in written["files"] if f.endswith(suffix))
        with open(path, encoding="utf-8", newline="") as fobj:
            header = next(csv.reader(fobj, delimiter="\t"))
        sidecar = _json(written["files"], suffix[:-4] + ".json")
        for column in header:
            entry = sidecar.get(column)
            assert isinstance(entry, dict) and entry.get("Description"), (suffix, column)
            assert entry.get("Units"), (suffix, column)
    responses = _json(written["files"], "_responses.json")
    assert responses["latency"]["Units"] == "s" and responses["amplitude"]["Units"] == "uV"
    assert responses["symmetry"]["Units"] == "n/a"


def test_artifact_record(written):

    record = _json(written["files"], "_artifact.json")
    artifact = written["result"].artifact[-1]
    assert record["Method"] == artifact.method and record["Window"] == list(artifact.window)
    assert record["QualitySummary"]["median"] == artifact.quality_summary()["median"]
    assert record["PackageVersion"] == __version__


def test_two_artifact_steps_make_one_record(tmp_path):
    """The tep preset chains interpolation and subspace projection: one artifact.json
    with every step."""

    from evoparam.artifact import remove_artifact

    trials, _ = sim.simulate_tep(n_trials=6, n_channels=3, stim_site="Cz", seed=0)
    first = remove_artifact(trials, method="interpolate", window=(-0.002, 0.012))
    second = remove_artifact(first.cleaned, method="subspace", window=(-0.002, 0.012))
    files = write_bids_derivatives([first, second], subject="01", task="tep", datatype="eeg",
                                   out=str(tmp_path))
    record = _json(files, "_artifact.json")
    assert record["Method"] == "subspace"
    assert [step["Method"] for step in record["Steps"]] == ["interpolate", "subspace"]


def test_simulated_dataset_round_trip(written):
    """write_bids then read_bids returns the simulated epochs (float32 precision)."""

    trials, read = written["trials"], written["read"]
    assert read.ch_names == trials.ch_names and read.n_trials == trials.n_trials
    index = int(np.argmin(np.abs(read.times - trials.times[0])))
    part = read.data[:, :, index:index + trials.times.size]
    scale = float(np.nanmax(np.abs(trials.data)))
    assert np.nanmax(np.abs(part - trials.data)) <= 1e-6 * scale


def _validator():
    command = os.environ.get("EVOPARAM_BIDS_VALIDATOR")
    if command:
        return shlex.split(command)
    found = shutil.which("bids-validator")
    return [found] if found else None


@pytest.mark.skipif(_validator() is None,
                    reason="bids-validator not installed (set EVOPARAM_BIDS_VALIDATOR)")
@pytest.mark.parametrize("which", ["raw", "derivatives"])
def test_req342_bids_validator(written, which):

    folder = written["root"] if which == "raw" else written["out"]
    # REQ-342: the derivatives are validated in derivative mode
    mode = ["--datasetTypes", "derivative"] if which == "derivatives" else []
    proc = subprocess.run(_validator() + mode + [str(folder), "--format", "json"],
                          capture_output=True, text=True, timeout=900)
    report = json.loads(proc.stdout)
    issues = report["issues"]["issues"] if isinstance(report.get("issues"), dict) \
        else report.get("issues", [])
    errors = [(i.get("code"), i.get("location")) for i in issues
              if i.get("severity") == "error"]
    assert errors == []


# Spec §9.2 and REQ-315: the reference validator is the Node.js bids-validator@2, run with
# npx, in derivative mode.
REFERENCE_VALIDATOR = ["npx", "--yes", "bids-validator@2", "--datasetTypes", "derivative"]


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("EVOPARAM_NETWORK_TESTS") != "1" or not shutil.which("npx"),
                    reason="needs npx and EVOPARAM_NETWORK_TESTS=1 (npx fetches the validator)")
def test_req315_reference_validator_as_written(written):
    """REQ-315 as written: ``npx bids-validator@2 --datasetTypes derivative`` reports no
    error on the derivatives.

    On 2026-10-09 the npm registry has no 2.x release of ``bids-validator``: its latest
    version is 1.15.0 and npx stops with ``ETARGET No matching version found for
    bids-validator@2``. The 2.x validator is published on JSR only, as ``@bids/validator``,
    and test_req342_bids_validator runs it. The test therefore expects failure while npx
    cannot resolve the version, and it passes as soon as npm has a 2.x release."""

    proc = subprocess.run(REFERENCE_VALIDATOR + [str(written["out"]), "--json"],
                          capture_output=True, text=True, timeout=900)
    if "ETARGET" in proc.stderr or "No matching version found" in proc.stderr:
        pytest.xfail("npm has no bids-validator@2 release (ETARGET); measured 2026-10-09, "
                     "latest 1.15.0")
    report = json.loads(proc.stdout)
    issues = report["issues"]["issues"] if isinstance(report.get("issues"), dict) \
        else report.get("issues", [])
    errors = [(i.get("code"), i.get("location")) for i in issues
              if i.get("severity") == "error"]
    assert errors == []
