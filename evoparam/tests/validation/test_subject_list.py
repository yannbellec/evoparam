"""The ds004080 subject list of the layer-3 validation (REQ-371)."""

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parents[3]
TOOL = ROOT / "tools" / "ds004080_subjects.py"
LIST = ROOT / "validation" / "ds004080_subjects.txt"
pytestmark = pytest.mark.skipif(not TOOL.exists(), reason="repository files not in this checkout")


def _tool():
    spec = importlib.util.spec_from_file_location("ds004080_subjects", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake(n_trials, sfreq):
    """A listing of four subjects and a fetch that serves their sidecars and events."""

    listing, files = {}, {}
    for i, (count, rate) in enumerate(zip(n_trials, sfreq), start=1):
        stem = f"sub-{i:02d}/ses-1/ieeg/sub-{i:02d}_ses-1_task-SPES_run-1"
        listing[stem + "_events.tsv"] = 1
        listing[stem + "_ieeg.json"] = 1
        files[stem + "_ieeg.json"] = json.dumps({"SamplingFrequency": rate}).encode()
        rows = "\n".join("electrical_stimulation\tA1-A2" for _ in range(count))
        files[stem + "_events.tsv"] = ("trial_type\telectrical_stimulation_site\n" + rows).encode()
    files["participants.tsv"] = b"participant_id\n"

    def fetch(url):
        return files[url.split("/ds004080/")[1]]
    return listing, fetch


def test_rule_selects_by_trials_and_rate():

    tool = _tool()
    listing, fetch = _fake([12, 9, 10, 20], [2048.0, 2048.0, 2048.0, 512.0])
    tool.N_SUBJECTS = 2
    tool.EXAMPLE = "sub-04"
    chosen, sha = tool.select(fetch=fetch, listing=listing)
    assert [s for s, _ in chosen] == ["sub-01", "sub-03", "sub-04"]
    assert len(sha) == 64


def test_list_is_committed_with_version_and_checksum():

    lines = LIST.read_text("utf-8").splitlines()
    assert lines[0].startswith("# ds004080 v1.2.4; participants.tsv sha256 ")
    subjects = [l.split("\t")[0] for l in lines if not l.startswith("#")]
    assert len(subjects) >= 5 and "sub-ccepAgeUMCU01" in subjects
