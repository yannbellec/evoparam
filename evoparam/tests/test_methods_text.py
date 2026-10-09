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
            "ResponseTest": evoparam.test_response(trials, window=(0.009, 0.5),
                                                   baseline=(-0.5, -0.05), by=(),
                                                   random_state=0),
            "ArtifactResult": evoparam.artifact.remove_artifact(
                trials, method="interpolate", window=(-0.002, 0.006)),
            "TrialFitSet": evoparam.trials.fit_trials(trials, group, random_state=0),
        }


# words that name the statistical (or estimation) choice of each kind of result
CHOICES = {"GroupFit": ("BIC", "intervals", "permutation test", "alpha", "correction"),
           "ResponseFit": ("BIC", "intervals", "permutation test"),
           "ResponseTest": ("permutation test", "alpha", "correction"),
           "ArtifactResult": ("interpolation", "quality score"),
           "TrialFitSet": ("trial model", "variance")}


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
    assert "Settings changed from the preset" in text and "max_components = 2" in text


def test_req008_no_changed_settings_sentence_without_changes():
    trials, _ = sim.simulate_ccep(n_trials=10, n_channels=3, p_response=1.0, seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = methods_text(evoparam.fit(trials, preset="ccep", random_state=0))
    assert "preset 'ccep'" in text and "Settings changed from the preset" not in text
