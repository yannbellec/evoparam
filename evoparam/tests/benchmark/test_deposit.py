"""REQ-355 (deposit of the simulated suites as NWB/BIDS with a DOI, with the generator
version and the seeds of the public split) and REQ-356 (the hidden split's seeds committed
as a SHA-256 hash at deposit time, revealed at paper acceptance).

What can be done before the deposit is tested here: the local preparation of the deposit
(BIDS and NWB files, manifest with the generator version and the public seeds, file
hashes) and the commitment (the suite files hold the SHA-256 of the hidden seeds document,
never the seeds; a document that does not match is refused). The deposit itself is a
public step that has not been made: the two acceptance criteria stay strict xfails until
it is.
"""

import copy
import hashlib
import json
import os

import numpy as np
import pytest

from evoparam.benchmark import (hidden_commitment, load_hidden_seeds, load_suite,
                                prepare_deposit, simulate_cell)
from evoparam.benchmark.deposit import DEPOSIT_SCHEMA
from evoparam.benchmark.suite import HIDDEN_SCHEMA
from evoparam.errors import DataError
from evoparam.version import __version__

from .helpers import ROOT, hidden_seeds_path, suite

DEPOSIT_REASON = ("deposit not made: waits for the owner's HIIE decision (nothing public "
                  "before it)")


@pytest.fixture(scope="module")
def deposit(tmp_path_factory):
    pytest.importorskip("pynwb")
    pytest.importorskip("jsonschema")
    out = tmp_path_factory.mktemp("deposit") / "core"
    path = prepare_deposit("core", str(out), tasks=["T6"])
    with open(path, encoding="utf-8") as fobj:
        return out, json.load(fobj)


def test_req355_manifest_has_generator_version_and_public_seeds(deposit):
    out, manifest = deposit
    core = suite("core")
    assert manifest["schema"] == DEPOSIT_SCHEMA
    assert manifest["generator"]["package"] == "evoparam"
    assert manifest["generator"]["version"] == __version__
    assert manifest["generator"]["functions"] == ["evoparam.sim.simulate_erp"]
    assert manifest["suite"] == {"name": "core", "version": core.version,
                                 "sha256": core.sha256, "file": "suite-core.json"}
    public = {c.id: c for c in core.cells("T6", split="public")}
    cells = manifest["public_split"]["cells"]
    assert [c["id"] for c in cells] == list(public)
    for entry in cells:
        assert entry["seed"] == public[entry["id"]].seed
        assert entry["generator"] == "simulate_erp"
        assert entry["settings"] == public[entry["id"]].settings
    assert manifest["doi"] is None
    assert "not deposited" in manifest["status"]


def test_req355_files_and_hashes(deposit):
    out, manifest = deposit
    assert (out / "suite-core.json").read_bytes() == \
        (ROOT / "evoparam" / "benchmark" / "suites" / "core.json").read_bytes()
    for rel, digest in manifest["files"].items():
        assert hashlib.sha256((out / rel).read_bytes()).hexdigest() == digest, rel
    on_disk = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert on_disk - set(manifest["files"]) == {"manifest.json"}
    for entry in manifest["public_split"]["cells"]:
        kinds = {rel.split("/")[0] for rel in entry["files"]}
        assert kinds == {"bids", "nwb", "truth", "inputs"}
    assert (out / "bids" / "dataset_description.json").is_file()


def test_req355_nwb_holds_the_simulated_trials(deposit):
    from evoparam.io import read_nwb

    out, manifest = deposit
    entry = manifest["public_split"]["cells"][0]
    cell = next(c for c in suite("core").cells("T6") if c.id == entry["id"])
    ts, _ = simulate_cell(cell)
    back = read_nwb(str(out / "nwb" / f"{cell.id}.nwb"), tmin=float(ts.times[0]),
                    tmax=float(ts.times[-1]) + 0.5 / ts.sfreq)
    assert back.data.shape == ts.data.shape
    np.testing.assert_allclose(back.data, ts.data, rtol=1e-9, atol=1e-9)
    assert list(back.ch_names) == list(ts.ch_names)


def test_req356_hidden_split_is_committed_not_written(deposit):
    out, manifest = deposit
    core = suite("core")
    hidden = manifest["hidden_split"]
    assert hidden["commitment_sha256"] == core.hidden_split["commitment_sha256"]
    assert hidden["n_cells"] == len(core.cells(split="hidden")) == 12
    assert all("seed" not in c for c in hidden["cells"])
    assert hidden["verified_against_seeds"] is None
    text = (out / "suite-core.json").read_text(encoding="utf-8")
    for group in json.loads(text)["groups"].values():
        for cell in group["cells"]:
            assert ("seed" in cell) == (cell["split"] == "public")


def test_req356_commitment_is_the_same_for_both_suites():
    core, full = suite("core"), load_suite("full")
    assert core.hidden_split["commitment_sha256"] == full.hidden_split["commitment_sha256"]
    assert len(core.hidden_split["commitment_sha256"]) == 64
    assert core.hidden_split["fraction"] == full.hidden_split["fraction"] == 0.2
    assert all(c.seed is None for c in full.cells(split="hidden"))


def _fake_document(core, full):
    return {"schema": HIDDEN_SCHEMA, "nonce": "00" * 32,
            "seeds": {"core": {c.id: 1 for c in core.cells(split="hidden")},
                      "full": {c.id: 2 for c in full.cells(split="hidden")}}}


def test_req356_a_different_seeds_document_is_refused(tmp_path):
    core, full = suite("core"), load_suite("full")
    document = _fake_document(core, full)
    assert hidden_commitment(document) != core.hidden_split["commitment_sha256"]
    with pytest.raises(DataError, match="is not the commitment"):
        core.with_hidden_seeds(document)
    with pytest.raises(DataError, match="does not match the commitment"):
        prepare_deposit(core, str(tmp_path / "d"), tasks=["T6"], formats=(),
                        hidden_seeds=document)
    with pytest.raises(DataError, match="hidden cell without its seed"):
        simulate_cell(core.cells(split="hidden")[0])


def test_req356_commitment_is_the_hash_of_the_canonical_document():
    core, full = suite("core"), load_suite("full")
    document = _fake_document(core, full)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode("utf-8")
    assert hidden_commitment(document) == hashlib.sha256(canonical).hexdigest()
    # key order and whitespace do not change the commitment; any seed does
    shuffled = json.loads(json.dumps(document, indent=3))
    assert hidden_commitment(shuffled) == hidden_commitment(document)
    changed = copy.deepcopy(document)
    changed["seeds"]["core"][next(iter(changed["seeds"]["core"]))] = 3
    assert hidden_commitment(changed) != hidden_commitment(document)
    with pytest.raises(DataError):
        load_hidden_seeds({"schema": "something-else", "seeds": {}})


def test_req356_the_kept_seeds_match_the_commitment(tmp_path):
    """With the hidden seeds document (kept outside the repository): it matches the
    commitment of both suites and gives every hidden cell its seed."""

    path = hidden_seeds_path()
    document = load_hidden_seeds(path)
    assert len(document["nonce"]) == 64
    for name in ("core", "full"):
        loaded = load_suite(name, hidden_seeds=path)
        hidden = loaded.cells(split="hidden")
        assert hidden and all(isinstance(c.seed, int) for c in hidden)
        assert hidden_commitment(document) == loaded.hidden_split["commitment_sha256"]
    manifest_path = prepare_deposit("core", str(tmp_path / "d"), tasks=["T6"], formats=(),
                                    hidden_seeds=path)
    with open(manifest_path, encoding="utf-8") as fobj:
        manifest = json.load(fobj)
    assert manifest["hidden_split"]["verified_against_seeds"] is True
    text = json.dumps(manifest)
    for seed in document["seeds"]["core"].values():
        assert f'"seed": {seed}' not in text


@pytest.mark.xfail(strict=True, reason=DEPOSIT_REASON)
def test_req355_suites_deposited_with_a_doi(deposit):
    """REQ-355: the deposit (NWB/BIDS files with a DOI) is made before the benchmark paper
    is submitted. Measured: the prepared manifest has ``doi: null``; no deposit exists."""

    _, manifest = deposit
    assert manifest["doi"] is not None


@pytest.mark.xfail(strict=True, reason=DEPOSIT_REASON)
def test_req356_commitment_published_at_deposit_time():
    """REQ-356: the commitment is published with the deposit (its DOI recorded with the
    suites). Measured: no deposit, so the commitment is published nowhere."""

    deposit = suite("core").hidden_split.get("deposit_doi")
    assert deposit is not None and os.environ.get("EVOPARAM_DEPOSIT_DOI") == deposit
