# SPDX-License-Identifier: Apache-2.0
"""REQ-008 (methods text): every result produces a methods paragraph naming the package
version, the preset, the settings that differ from the preset, the statistical choices and
the citations (REQ-340, REQ-390)."""

import re
import warnings

import pytest

import evoparam
from evoparam import sim
from evoparam.reports import methods_text

pytest.importorskip("pandas")


@pytest.fixture(scope="module")
def results():
    trials, _ = sim.simulate_ccep(n_trials=10, n_channels=3, p_response=1.0, seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = evoparam.fit(trials, preset="ccep", max_components=2, random_state=0)
        return {
            "GroupFit": group,
            "ResponseFit": group.fits[0],
            "ResponseTest": evoparam.test_response(
                trials, window=(0.009, 0.5), baseline=(-0.5, -0.05), by=(), random_state=0
            ),
            "ArtifactResult": evoparam.artifact.remove_artifact(
                trials, method="interpolate", window=(-0.002, 0.006)
            ),
            "TrialFitSet": evoparam.trials.fit_trials(trials, group, random_state=0),
        }


# words that name the statistical (or estimation) choice of each kind of result
CHOICES = {
    "GroupFit": ("BIC", "intervals", "permutation test", "alpha", "correction"),
    "ResponseFit": ("BIC", "intervals", "permutation test"),
    "ResponseTest": ("permutation test", "alpha", "correction"),
    "ArtifactResult": ("interpolation", "quality score"),
    "TrialFitSet": ("trial model", "variance"),
}


@pytest.mark.parametrize("kind", sorted(CHOICES))
def test_req008_methods_paragraph(kind, results):
    text = methods_text(results[kind])
    # the package version
    assert re.search(r"evoparam \(version " + re.escape(evoparam.__version__) + r"\)", text)
    # the statistical choices
    for word in CHOICES[kind]:
        assert word in text, word
    # the citations: the software, and the method papers when a method has one
    assert "Bellec Y.V. evoparam" in text
    if kind in ("GroupFit", "ResponseFit", "ResponseTest"):
        assert "Benjamini and Hochberg (1995)" in text


@pytest.mark.parametrize("kind", ["GroupFit", "ResponseFit"])
def test_req008_preset_and_changed_settings(kind, results):
    text = methods_text(results[kind])
    assert "preset 'ccep'" in text
    assert "Settings changed from the preset" in text
    assert "max_components = 2" in text


def test_req008_no_changed_settings_sentence_without_changes():
    trials, _ = sim.simulate_ccep(n_trials=10, n_channels=3, p_response=1.0, seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = methods_text(evoparam.fit(trials, preset="ccep", random_state=0))
    assert "preset 'ccep'" in text
    assert "Settings changed from the preset" not in text


# ---------------------------------------------------------------------------------------
# REQ-390: the methods paragraph is stored in the saved result and in the derivatives
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(CHOICES))
def test_req390_methods_text_stored_in_saved_result(kind, results, tmp_path):
    import json
    import os

    from evoparam.data import load_object, save_object

    obj = results[kind]
    if kind == "TrialFitSet":  # a list of TrialFit: each one is saved
        obj = obj.fits[0]
    path = save_object(obj, str(tmp_path / kind))
    with open(os.path.join(path, "methods.txt"), encoding="utf-8") as fobj:
        stored = fobj.read()
    assert stored == obj.methods_text() + "\n"
    with open(os.path.join(path, "manifest.json"), encoding="utf-8") as fobj:
        assert "methods.txt" in json.load(fobj)["files"]  # checked at loading
    assert load_object(path).methods_text() + "\n" == stored  # byte-identical round trip


def test_req390_methods_text_in_nwb_and_bids_derivatives(results, tmp_path):
    import json

    group = results["GroupFit"]
    pynwb = pytest.importorskip("pynwb")
    from datetime import datetime, timezone

    from evoparam.io import write_bids_derivatives, write_nwb

    nwb = pynwb.NWBFile(
        session_description="req390",
        identifier="req390",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_nwb(group, nwb)
    description = json.loads(nwb.processing["evoparam"].description)
    assert description["methods"] == [group.methods_text()]
    written = write_bids_derivatives(group, out=str(tmp_path / "derivatives"), subject="01")
    readme = next(p for p in written if p.endswith("README"))
    with open(readme, encoding="utf-8") as fobj:
        assert methods_text(group) in fobj.read()
    containers = [p for p in written if p.endswith(".evoparam")]
    for container in containers:  # the result next to its table
        with open(f"{container}/methods.txt", encoding="utf-8") as fobj:
            assert fobj.read().strip()


@pytest.mark.xfail(
    strict=True,
    reason="REQ-390: the versioned Zenodo DOI of the release in the "
    "methods paragraph waits for the owner's HIIE decision (nothing public "
    "before it); CITATION.cff has no doi until then (REQ-019)",
)
def test_req390_methods_text_names_the_versioned_doi(results):
    assert re.search(r"10\.5281/zenodo\.\d+", results["GroupFit"].methods_text())
