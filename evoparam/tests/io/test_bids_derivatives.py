# SPDX-License-Identifier: Apache-2.0
"""BIDS derivatives (REQ-336, REQ-342) and the simulated raw dataset they come from.

The reference BIDS validator runs when it is available: set ``EVOPARAM_BIDS_VALIDATOR`` to
its command, put ``bids-validator`` on the PATH, or install Deno (the test then runs
``deno run -A jsr:@bids/validator@3.0.2``); derivatives are validated with
``--datasetTypes derivative`` (REQ-315). The Python schema validator ``bidsschematools``
(extra ``validate``) checks the file names and the column registry.
"""

import csv
import json
import os
import shlex
import shutil
import subprocess
import warnings
from typing import Any

import numpy as np
import pytest

from evoparam import __version__, reports, sim
from evoparam.data import Provenance
from evoparam.io import read_bids, read_bids_derivatives, write_bids_derivatives
from evoparam.io.derivatives import CLASSICAL_COLUMNS, RESPONSE_COLUMNS, STIM_COLUMNS
from evoparam.sim.bids import write_bids

# Fewest files listed by the BIDS validator in the written derivatives.
MIN_LISTED = 5

TSV_SUFFIXES = ("_responses.tsv", "_existence.tsv", "_trials.tsv")


@pytest.fixture(scope="module")
def written(tmp_path_factory: Any) -> Any:
    """A simulated BIDS-iEEG dataset, the ccep pipeline run on it, and its derivatives."""

    root = tmp_path_factory.mktemp("bidsderiv") / "bids"
    trials, truth = sim.simulate_ccep(n_trials=8, n_channels=3, sfreq=1024.0, seed=2)
    write_bids(trials, truth, str(root), subject="01", task="spes")
    read = read_bids(str(root), "01", task="spes", datatype="ieeg", epoch=(-0.5, 1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = reports.run_pipeline(read, "ccep", trial_fits=True)
    out = root / "derivatives" / f"evoparam-{__version__}"
    files = write_bids_derivatives(
        result.results, str(root), subject="01", task="spes", out=str(out), trials=result.trials
    )
    return {
        "root": root,
        "out": out,
        "files": files,
        "result": result,
        "trials": trials,
        "read": read,
    }


def _json(files: Any, suffix: Any) -> Any:
    path = next(f for f in files if f.endswith(suffix))
    with open(path, encoding="utf-8") as fobj:
        return json.load(fobj)


def test_provenance_sidecar(written: Any) -> None:
    """REQ-336: the sidecars and dataset_description GeneratedBy hold the provenance."""

    group = written["result"].group
    sidecar = _json(written["files"], "_responses.json")
    assert Provenance.from_dict(sidecar["Provenance"]) == group.provenance
    assert sidecar["PackageVersion"] == __version__
    assert sidecar["Preset"] == {"Name": "ccep", "Version": "1.1"}
    assert sidecar["Settings"]["max_components"] == group.settings["max_components"]
    description = _json(written["files"], "dataset_description.json")
    assert description["DatasetType"] == "derivative"
    generated = description["GeneratedBy"][0]
    assert generated["Name"] == "evoparam"
    assert generated["Version"] == __version__
    assert Provenance.from_dict(generated["Provenance"]) == group.provenance
    existence = _json(written["files"], "_existence.json")
    test = written["result"].test
    assert Provenance.from_dict(existence["Provenance"]) == test.provenance
    artifact = _json(written["files"], "_artifact.json")
    last = written["result"].artifact[-1]
    assert Provenance.from_dict(artifact["Provenance"]) == last.provenance


def test_nwb_provenance(written: Any) -> None:
    """REQ-336: the NWB processing module holds the settings and provenance of each result."""

    pynwb = pytest.importorskip("pynwb")
    from datetime import datetime, timezone

    from evoparam import io

    nwbfile = pynwb.NWBFile(
        session_description="evoparam test",
        identifier="evoparam-test",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    group, test = written["result"].group, written["result"].test
    io.write_nwb([group, test], nwbfile)
    description = json.loads(nwbfile.processing["evoparam"].description)
    assert [Provenance.from_dict(p) for p in description["provenance"]] == [
        group.provenance,
        test.provenance,
    ]
    assert description["settings"][0]["max_components"] == group.settings["max_components"]


def test_req342_every_column_has_a_description_and_unit(written: Any) -> None:

    for suffix in TSV_SUFFIXES:
        path = next(f for f in written["files"] if f.endswith(suffix))
        with open(path, encoding="utf-8", newline="") as fobj:
            header = next(csv.reader(fobj, delimiter="\t"))
        sidecar = _json(written["files"], suffix[:-4] + ".json")
        for column in header:
            entry = sidecar.get(column)
            assert isinstance(entry, dict), (suffix, column)
            assert entry.get("Description"), (suffix, column)
            assert entry.get("Units"), (suffix, column)
    responses = _json(written["files"], "_responses.json")
    assert responses["latency"]["Units"] == "s"
    assert responses["amplitude"]["Units"] == "uV"
    assert responses["symmetry"]["Units"] == "n/a"


def test_artifact_record(written: Any) -> None:

    record = _json(written["files"], "_artifact.json")
    artifact = written["result"].artifact[-1]
    assert record["Method"] == artifact.method
    assert record["Window"] == list(artifact.window)
    assert record["QualitySummary"]["median"] == artifact.quality_summary()["median"]
    assert record["PackageVersion"] == __version__


def test_two_artifact_steps_make_one_record(tmp_path: Any) -> None:
    """The tep preset chains interpolation and subspace projection: one artifact.json
    with every step."""

    from evoparam.artifact import remove_artifact

    trials, _ = sim.simulate_tep(n_trials=6, n_channels=3, stim_site="Cz", seed=0)
    first = remove_artifact(trials, method="interpolate", window=(-0.002, 0.012))
    second = remove_artifact(first.cleaned, method="subspace", window=(-0.002, 0.012))
    files = write_bids_derivatives(
        [first, second], subject="01", task="tep", datatype="eeg", out=str(tmp_path / "evoparam")
    )
    record = _json(files, "_artifact.json")
    assert record["Method"] == "subspace"
    assert [step["Method"] for step in record["Steps"]] == ["interpolate", "subspace"]


def test_simulated_dataset_round_trip(written: Any) -> None:
    """write_bids then read_bids returns the simulated epochs (float32 precision)."""

    trials, read = written["trials"], written["read"]
    assert read.ch_names == trials.ch_names
    assert read.n_trials == trials.n_trials
    index = int(np.argmin(np.abs(read.times - trials.times[0])))
    part = read.data[:, :, index : index + trials.times.size]
    scale = float(np.nanmax(np.abs(trials.data)))
    assert np.nanmax(np.abs(part - trials.data)) <= 1e-6 * scale


def _validator() -> Any:
    command = os.environ.get("EVOPARAM_BIDS_VALIDATOR")
    if command:
        return shlex.split(command)
    found = shutil.which("bids-validator")
    if found:
        return [found]
    deno = shutil.which("deno")
    return [deno, "run", "-A", "jsr:@bids/validator@3.0.2"] if deno else None


@pytest.mark.skipif(
    _validator() is None, reason="bids-validator not installed (set EVOPARAM_BIDS_VALIDATOR)"
)
@pytest.mark.parametrize("which", ["raw", "derivatives"])
def test_req342_bids_validator(written: Any, which: Any) -> None:
    """REQ-024 and REQ-342: the output of ``write_bids_derivatives`` passes the BIDS
    validator in derivative mode (the simulated raw data in raw mode), when the validator
    is installed."""

    folder = written["root"] if which == "raw" else written["out"]
    # REQ-342: the derivatives are validated in derivative mode (the raw data in raw mode)
    types = ["--datasetTypes", "raw" if which == "raw" else "derivative"]
    proc = subprocess.run(
        _validator() + [str(folder), "--format", "json"] + types,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    report = json.loads(proc.stdout)
    issues = (
        report["issues"]["issues"]
        if isinstance(report.get("issues"), dict)
        else report.get("issues", [])
    )
    errors = [(i.get("code"), i.get("location")) for i in issues if i.get("severity") == "error"]
    assert errors == []


# Spec §9.2 and REQ-315: the reference validator is bids-validator >= 2.0 in derivative mode,
# named in the spec as the Node.js package run with npx (``npx bids-validator@2``). npm has
# no 2.x release of ``bids-validator`` (measured 2026-10-09 and 2026-10-13: latest 1.15.0,
# ``ETARGET``); the BIDS maintainers publish 2.x and 3.x on JSR as ``@bids/validator``. The
# test runs npx when npm resolves a 2.x release and the pinned JSR release otherwise
# (docs/spec/DECISIONS.md, "REQ-315").
REFERENCE_VALIDATOR = ["npx", "--yes", "bids-validator@2", "--datasetTypes", "derivative"]
JSR_VALIDATOR = ["run", "-A", "jsr:@bids/validator@3.0.2", "--datasetTypes", "derivative"]


def _reference_validator() -> Any:
    if shutil.which("npx"):
        probe = subprocess.run(
            ["npm", "view", "bids-validator@2", "version"],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            return REFERENCE_VALIDATOR
    deno = shutil.which("deno")
    return [deno] + JSR_VALIDATOR if deno else None


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("EVOPARAM_NETWORK_TESTS") != "1"
    or not (shutil.which("npx") or shutil.which("deno")),
    reason="needs npx or deno and EVOPARAM_NETWORK_TESTS=1 (the validator is fetched)",
)
def test_req315_reference_validator_as_written(written: Any) -> None:
    """REQ-315: bids-validator >= 2.0 in derivative mode reports no error on the
    derivatives (npx when npm has a 2.x release, the JSR release of the BIDS maintainers
    otherwise)."""

    command = _reference_validator()
    if command is None:
        pytest.skip("neither npm's bids-validator@2 nor deno is available")
    output = ["--json"] if command is REFERENCE_VALIDATOR else ["--format", "json"]
    proc = subprocess.run(
        command + [str(written["out"])] + output,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    report = json.loads(proc.stdout)
    issues = (
        report["issues"]["issues"]
        if isinstance(report.get("issues"), dict)
        else report.get("issues", [])
    )
    errors = [(i.get("code"), i.get("location")) for i in issues if i.get("severity") == "error"]
    assert errors == []


def test_req315_tree_columns_and_generated_by(written: Any) -> None:

    out = written["out"]
    description = _json(written["files"], "dataset_description.json")
    generated = description["GeneratedBy"][0]
    assert generated["Name"] in os.path.basename(str(out))  # BIDS substring rule
    assert set(generated) >= {"Name", "Version", "CodeURL", "Description"}
    assert description["SourceDatasets"] == [] or "URL" in description["SourceDatasets"][0]
    for name in ("descriptions.tsv", "README", ".bidsignore"):
        assert os.path.isfile(os.path.join(out, name))
    with open(os.path.join(out, "README"), encoding="utf-8") as fobj:
        readme = fobj.read()
    assert "Methods (generated by evoparam" in readme
    assert "evoparam" in readme
    # datatype taken from the source run
    assert any(f"{os.sep}ieeg{os.sep}" in f for f in written["files"])

    path = next(f for f in written["files"] if f.endswith("_responses.tsv"))
    with open(path, encoding="utf-8") as fobj:
        header = fobj.readline().rstrip("\n").split("\t")
    stim = [c for c in STIM_COLUMNS if c in header]
    assert header[: len(RESPONSE_COLUMNS)] == list(RESPONSE_COLUMNS)
    classical = [c for c in header if c.startswith("classical_")]
    assert classical == [c for c in CLASSICAL_COLUMNS if c in classical]
    assert classical
    assert header == list(RESPONSE_COLUMNS) + classical + stim + [
        c for c in header if c.startswith("electrical_stimulation_")
    ]
    assert {"evoparam_stim_current_mA", "evoparam_stim_pulse_width_us"} <= set(stim)
    sidecar = _json(written["files"], "_responses.json")
    assert sidecar["status"]["Levels"].keys() == {"ok", "no_response", "failed", "skipped"}
    assert sidecar["latency"]["LongName"] == "Latency"
    assert sidecar["evoparam_stim_current_mA"]["Units"] == "mA"


def test_req316_stimulation_columns(written: Any) -> None:
    """Prefixed columns hold the harmonised stimulation of the run; the standard example
    columns are copied unchanged; every written column outside the BIDS column registry
    carries the evoparam_ prefix or is an evoparam table column defined in the sidecar."""

    meta = written["result"].trials.metadata
    path = next(f for f in written["files"] if f.endswith("_responses.tsv"))
    with open(path, encoding="utf-8") as fobj:
        rows = list(csv.DictReader(fobj, delimiter="\t"))
    sites = set(meta["stim_site"])
    if len(sites) == 1:
        assert {r["stim_site"] for r in rows} == sites
    current = set(meta["stim_current_mA"])
    if len(current) == 1:
        assert {float(r["evoparam_stim_current_mA"]) for r in rows} == current
    for column in ("electrical_stimulation_site", "electrical_stimulation_current"):
        if column in meta and len(set(meta[column].astype(str))) == 1:
            assert {r[column] for r in rows} == set(meta[column].astype(str))
    schema = pytest.importorskip("bidsschematools.schema")
    registry = {c.get("name", k) for k, c in schema.load_schema().objects.columns.items()}
    sidecar = _json(written["files"], "_responses.json")
    for column in rows[0]:
        if (
            column in registry
            or column in RESPONSE_COLUMNS
            or column in CLASSICAL_COLUMNS
            or column.startswith("electrical_stimulation_")
        ):
            continue
        assert column.startswith("evoparam_"), column
        assert sidecar[column]["Description"], column


def test_req315_bidsschematools_validation(written: Any) -> None:
    """The Python schema validator: every file of the raw dataset and every derivative file
    that BIDS defines matches a schema rule; the evoparam tables are declared in
    .bidsignore (no BIDS suffix exists for them, spec §10.2)."""

    validator = pytest.importorskip("bidsschematools.validator")
    raw = validator.validate_bids([str(written["root"])], accept_non_bids_dir=True)
    assert raw["path_tracking"] == []
    assert len(raw["path_listing"]) > MIN_LISTED
    with open(os.path.join(written["out"], ".bidsignore"), encoding="utf-8") as fobj:
        ignored = [line.strip() for line in fobj if line.strip()]
    import fnmatch

    files = [str(p) for p in written["out"].rglob("*") if p.is_file()]
    kept = [
        f
        for f in files
        if not any(
            fnmatch.fnmatch(os.path.basename(f), pattern) or ".evoparam" in f
            for pattern in ignored
        )
        and not os.path.basename(f).startswith(".")
    ]
    assert {os.path.basename(f) for f in kept} == {
        "dataset_description.json",
        "README",
        "descriptions.tsv",
    }
    derived = validator.validate_bids(kept, accept_non_bids_dir=True)
    assert [os.path.basename(p) for p in derived["path_tracking"]] in ([], ["descriptions.tsv"])


def test_req315_read_back_equal_results(written: Any) -> None:

    result = written["result"]
    root = written["root"]
    group = read_bids_derivatives(str(root), subject="01", task="spes")
    assert group == result.group  # same tables, same provenance
    assert group.to_dataframe().equals(result.group.to_dataframe())
    assert group.provenance == result.group.provenance
    everything = read_bids_derivatives(
        str(written["out"]), subject="sub-01", task="spes", kind="all"
    )
    assert everything["existence"] == result.test
    assert [a.method for a in everything["artifact"]] == [a.method for a in result.artifact]
    assert everything["artifact"][-1] == result.artifact[-1]
    trial = everything["trials"]
    from evoparam.trials import TrialFitSet

    fits = TrialFitSet(**trial)
    assert fits.settings == result.trial_fits.settings
    assert fits.provenance == result.trial_fits.provenance
    assert fits.table.equals(result.trial_fits.table)


def test_req315_edited_table_is_detected(written: Any, tmp_path: Any) -> None:

    from evoparam.errors import DataError

    copy = tmp_path / "deriv"
    shutil.copytree(written["out"], copy)
    path = next(
        p for p in (copy / "sub-01" / "ieeg").iterdir() if p.name.endswith("_responses.tsv")
    )
    lines = path.read_text().splitlines()
    cells = lines[1].split("\t")
    header = lines[0].split("\t")
    cells[header.index("r2")] = "0.123"
    lines[1] = "\t".join(cells)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(DataError, match="edited"):
        read_bids_derivatives(str(copy), subject="01", task="spes")


def test_req316_docs_page_lists_the_columns() -> None:

    import evoparam

    page = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(evoparam.__file__))),
        "docs",
        "bids_stimulation_columns.rst",
    )
    if not os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(page)), "pyproject.toml")):
        pytest.skip("needs the source tree")
    with open(page, encoding="utf-8") as fobj:
        text = fobj.read()
    for column in STIM_COLUMNS:
        assert f"``{column}``" in text, column
    for level in ("monophasic_cathodic", "biphasic_anodic_first", "alternating", "unknown"):
        assert level in text
    assert "BEP021" in text
    assert "BEP037" in text
