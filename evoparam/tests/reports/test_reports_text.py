"""Frozen methods text of the reference pipelines (REQ-340).

The frozen files are in ``methods/``; the package version is written ``{version}``. After
a deliberate change of a template or of a preset, regenerate them with
``python tools/freeze_methods_text.py``, review the diff and name the change in the
CHANGELOG. The ds004080 pipeline is also run on the real recording (network test): the
methods text depends only on the settings, so it must give the frozen words too. The ERP
CORE P3 files are not on OpenNeuro and need MNE to read, so that pipeline is checked on its
stand-in only.
"""

import os
import re
import warnings
from importlib import resources

import numpy as np
import pytest

from evoparam import __version__, reports

FOLDER = resources.files("evoparam.tests.reports").joinpath("methods")


def _frozen(name):
    return FOLDER.joinpath(name + ".txt").read_text("utf-8").replace("{version}", __version__)


def _text(result):
    return reports.methods_text(result) + "\n"


@pytest.mark.parametrize("name", sorted(reports.REFERENCE_PIPELINES))
def test_req340_reference_pipelines_match_frozen_text(name):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = reports.run_reference_pipeline(name)
    text = _text(result)
    frozen = _frozen(name)
    if text != frozen:
        # Name the first differing sentence, so that a failure is easy to read.
        new, old = text.split(". "), frozen.split(". ")
        first = next(i for i, (a, b) in enumerate(zip(new + [""], old + [""])) if a != b)
        pytest.fail(f"{name}: methods text differs from methods/{name}.txt at sentence "
                    f"{first}:\n  frozen: {old[first] if first < len(old) else ''}\n"
                    f"  now:    {new[first] if first < len(new) else ''}\n"
                    "Regenerate with tools/freeze_methods_text.py if the change is wanted.")


def test_req340_frozen_text_has_no_run_specific_content():

    for name in reports.REFERENCE_PIPELINES:
        raw = FOLDER.joinpath(name + ".txt").read_text("utf-8")
        # The version appears only as a placeholder (the citation names the cited release).
        assert "{version}" in raw and __version__ not in raw.split("Software:")[0]
        assert not re.search(r"\d{4}-\d{2}-\d{2}", raw)              # no date
        assert not re.search(r"\d+\.\d{5,}", raw)                     # no long float
        assert "duration" not in raw and "/home/" not in raw and "/tmp/" not in raw
        assert raw.count("\n") == 1


def test_req340_every_changed_setting_is_in_the_text(erp_run):

    trials = reports.reference_input("erpcore_p3")
    changed = {"max_components": 3, "fit_window": (0.1, 0.6), "loss": "soft_l1",
               "min_fwhm": 0.02, "n_starts": 5}
    result = reports.run_pipeline(trials, "erp", **changed)
    text = reports.methods_text(result)
    sentence = next(s for s in text.split(". ") if s.startswith("Settings changed"))
    for key, value in changed.items():
        expected = f"({value[0]:g}, {value[1]:g})" if isinstance(value, tuple) else f"{value}"
        assert f"{key} = {expected}" in sentence, (key, sentence)
    # The settings of the preset run are not listed, and the run without changes has none.
    assert "Settings changed" not in reports.methods_text(erp_run)
    assert set(result.group.settings["_overrides"]) == set(changed)


def test_req340_seed_sentence_follows_the_seed():

    trials = reports.reference_input("erpcore_p3")
    text = reports.methods_text(reports.run_pipeline(trials, "erp", random_state=11))
    assert "Random draws used the seed 11;" in text


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
                    reason="set EVOPARAM_NETWORK_TESTS=1 to read OpenNeuro ds004080")
def test_req340_ds004080_real_recording_gives_the_frozen_text():
    """sub-ccepAgeUMCU01 run-021448, site PT01-PT02 (120 channels, 8 pulses; ~3 min)."""

    import pandas as pd

    from evoparam.data import TrialSet
    from evoparam.io import epoch_data
    from evoparam.tests.events.test_ds004080_events import _segment

    run = "sub-ccepAgeUMCU01/ses-1/ieeg/sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448"
    data, sfreq, names, t0, onsets, sites = _segment(run)
    site = "PT01-PT02"
    onsets = onsets[np.asarray(sites) == site]
    keep = [i for i, n in enumerate(names) if re.match(r"^[A-Za-z]+\d+$", n)
            and not n.upper().startswith(("MKR", "ECG")) and n not in site.split("-")]
    times, epochs, _ = epoch_data(data[keep], sfreq, onsets, -1.0, 1.0, t0=t0)
    trials = TrialSet(data=epochs, times=times, sfreq=sfreq, ch_names=[names[i] for i in keep],
                      metadata=pd.DataFrame({"stim_site": [site] * len(epochs)}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = reports.run_reference_pipeline("ds004080_ccep", trials)
    assert _text(result) == _frozen("ds004080_ccep")

