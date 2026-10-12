# SPDX-License-Identifier: Apache-2.0
"""Frozen methods text of the reference pipelines (REQ-340).

The frozen files are in ``methods/``; the package version is written ``{version}``. After
a deliberate change of a template or of a preset, regenerate them with
``python tools/freeze_methods_text.py``, review the diff and name the change in the
CHANGELOG. The ds004080 and ERP CORE P3 pipelines are also run on their real recordings
(network tests, ``EVOPARAM_NETWORK_TESTS=1``): the methods text depends only on the
settings, so it must give the frozen words too. ERP CORE (CC-BY-4.0) is downloaded from its
OSF project into ``EVOPARAM_ERPCORE`` (default ``~/.cache/evoparam/erpcore``, the layout of
``validation/erpcore.py``) and never stored in the repository; it needs MNE to read.
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
        first = next(
            i for i, (a, b) in enumerate(zip(new + [""], old + [""], strict=False)) if a != b
        )
        pytest.fail(
            f"{name}: methods text differs from methods/{name}.txt at sentence "
            f"{first}:\n  frozen: {old[first] if first < len(old) else ''}\n"
            f"  now:    {new[first] if first < len(new) else ''}\n"
            "Regenerate with tools/freeze_methods_text.py if the change is wanted."
        )


def test_req340_frozen_text_has_no_run_specific_content():

    for name in reports.REFERENCE_PIPELINES:
        raw = FOLDER.joinpath(name + ".txt").read_text("utf-8")
        # The version appears only as a placeholder (the citation names the cited release).
        assert "{version}" in raw
        assert __version__ not in raw.split("Software:")[0]
        assert not re.search(r"\d{4}-\d{2}-\d{2}", raw)  # no date
        assert not re.search(r"\d+\.\d{5,}", raw)  # no long float
        assert "duration" not in raw
        assert "/home/" not in raw
        assert "/tmp/" not in raw
        assert raw.count("\n") == 1


def test_req340_every_changed_setting_is_in_the_text(erp_run):

    trials = reports.reference_input("erpcore_p3")
    changed = {
        "max_components": 3,
        "fit_window": (0.1, 0.6),
        "loss": "soft_l1",
        "min_fwhm": 0.02,
        "n_starts": 5,
    }
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
@pytest.mark.skipif(
    os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
    reason="set EVOPARAM_NETWORK_TESTS=1 to read OpenNeuro ds004080",
)
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
    keep = [
        i
        for i, n in enumerate(names)
        if re.match(r"^[A-Za-z]+\d+$", n)
        and not n.upper().startswith(("MKR", "ECG"))
        and n not in site.split("-")
    ]
    times, epochs, _ = epoch_data(data[keep], sfreq, onsets, -1.0, 1.0, t0=t0)
    trials = TrialSet(
        data=epochs,
        times=times,
        sfreq=sfreq,
        ch_names=[names[i] for i in keep],
        metadata=pd.DataFrame({"stim_site": [site] * len(epochs)}),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = reports.run_reference_pipeline("ds004080_ccep", trials)
    assert _text(result) == _frozen("ds004080_ccep")


###################################################################################################
## ERP CORE P3 (Kappenman et al. 2021, OSF DOI 10.18115/D5JW4R, CC-BY-4.0)

# OSF node of the P3 component; its folder "P3 All Data and Scripts" holds one folder per
# participant with the epochs after the published pipeline (ICA, interpolation, flags)
ERPCORE_P3_NODE = "etdkz"
ERPCORE_SUFFIXES = ("_epoch_interp_ar.set", "_epoch_interp_ar.fdt")
# SHA-256 of participant 1's files, recorded on 2026-10-08 (also checked against the hash
# OSF publishes for every file)
ERPCORE_P3_SHA256 = {
    "1_P3_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.fdt": "f62f325b0c579481829afe6cb78ba3b742e7460f1383168385cff26e2ebda431",
    "1_P3_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.set": "546dd1bedb039f2a5cc8db620c957d4d2db6623fe914198986c1b1b674e2a6e6",
}


def _osf_pages(url):
    import json

    from evoparam.io._fetch import fetch_bytes

    while url:
        page = json.loads(fetch_bytes(url))
        yield from page["data"]
        url = page["links"].get("next")


def fetch_erpcore_p3(participant=1):
    """Path of the epoched ``.set`` of one ERP CORE P3 participant, downloaded from OSF
    into ``$EVOPARAM_ERPCORE/P3/<participant>/`` when it is not there, and checked
    against the recorded and the published SHA-256."""

    import glob

    from evoparam.io._fetch import download, sha256_file

    root = os.environ.get("EVOPARAM_ERPCORE") or os.path.expanduser("~/.cache/evoparam/erpcore")
    target = os.path.join(root, "P3", str(int(participant)))
    local = lambda suffix: sorted(glob.glob(os.path.join(target, "*" + suffix)))
    published = {}
    if any(not local(suffix) for suffix in ERPCORE_SUFFIXES):
        base = f"https://api.osf.io/v2/nodes/{ERPCORE_P3_NODE}/files/osfstorage/"
        folder = next(
            f for f in _osf_pages(base) if f["attributes"]["name"].endswith("All Data and Scripts")
        )
        related = lambda item: item["relationships"]["files"]["links"]["related"]["href"]
        subject = next(
            s
            for s in _osf_pages(related(folder) + "?page[size]=100")
            if s["attributes"]["name"] == str(int(participant))
        )
        for item in _osf_pages(related(subject) + "?page[size]=100"):
            name = item["attributes"]["name"]
            if name.endswith(ERPCORE_SUFFIXES):
                hashes = (item["attributes"].get("extra") or {}).get("hashes") or {}
                published[name] = hashes.get("sha256")
                download(item["links"]["download"], os.path.join(target, name))
    for suffix in ERPCORE_SUFFIXES:
        path = local(suffix)[0]
        name = os.path.basename(path)
        want = ERPCORE_P3_SHA256.get(name) if participant == 1 else published.get(name)
        if want:
            assert sha256_file(path) == want, f"{path}: SHA-256 differs from {want}"
    return local(ERPCORE_SUFFIXES[0])[0]


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
    reason="set EVOPARAM_NETWORK_TESTS=1 to read ERP CORE from OSF",
)
def test_req340_erpcore_p3_real_recording_gives_the_frozen_text():
    """ERP CORE P3 (active visual oddball), participant 1, channel Pz, all epochs (rare and
    frequent) after the published pipeline, in µV: the data the frozen text describes."""

    mne = pytest.importorskip("mne")
    import pandas as pd

    from evoparam.data import TrialSet

    path = fetch_erpcore_p3(1)
    epochs = mne.read_epochs_eeglab(path, verbose="error")
    inverse = {code: label for label, code in epochs.event_id.items()}
    trials = TrialSet(
        data=epochs.get_data(picks=["Pz"]) * 1e6,
        times=epochs.times,
        sfreq=float(epochs.info["sfreq"]),
        ch_names=["Pz"],
        metadata=pd.DataFrame({"event": [inverse[c] for c in epochs.events[:, 2]]}),
    )
    assert trials.n_trials > 100
    assert trials.times[0] < 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = reports.run_reference_pipeline("erpcore_p3", trials)
    assert _text(result) == _frozen("erpcore_p3")
