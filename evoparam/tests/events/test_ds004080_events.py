"""Pulse detection and site identification on OpenNeuro ds004080 (REQ-040, REQ-042).

25 runs of 21 patients (Utrecht SPES, ECoG, 512 or 2048 Hz). For each run, 40 s of
the recording are read from the first pulse on (8 pulses per run, 200 in all) and
compared with ``events.tsv``. Segments are cached in the evoparam data folder, so
only the first run downloads them (about 2 GB). Runs only with EVOPARAM_NETWORK_TESTS=1.
"""

import os

import numpy as np
import pytest

from evoparam.events import detect_pulses, estimate_delay, match_onsets

pytestmark = [pytest.mark.network,
              pytest.mark.skipif(os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
                                 reason="set EVOPARAM_NETWORK_TESTS=1 to read OpenNeuro")]

RUNS = [
    "sub-ccepAgeUMCU01/ses-1/ieeg/sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448",
    "sub-ccepAgeUMCU01/ses-1/ieeg/sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-031422",
    "sub-ccepAgeUMCU04/ses-1/ieeg/sub-ccepAgeUMCU04_ses-1_task-SPESclin_run-031148",
    "sub-ccepAgeUMCU07/ses-1/ieeg/sub-ccepAgeUMCU07_ses-1_task-SPESclin_run-021647",
    "sub-ccepAgeUMCU07/ses-1/ieeg/sub-ccepAgeUMCU07_ses-1_task-SPESclin_run-021752",
    "sub-ccepAgeUMCU10/ses-1/ieeg/sub-ccepAgeUMCU10_ses-1_task-SPESclin_run-031402",
    "sub-ccepAgeUMCU13/ses-1/ieeg/sub-ccepAgeUMCU13_ses-1_task-SPESclin_run-061646",
    "sub-ccepAgeUMCU16/ses-1/ieeg/sub-ccepAgeUMCU16_ses-1_task-SPESclin_run-021727",
    "sub-ccepAgeUMCU19/ses-1/ieeg/sub-ccepAgeUMCU19_ses-1_task-SPESclin_run-040955",
    "sub-ccepAgeUMCU22/ses-1/ieeg/sub-ccepAgeUMCU22_ses-1_task-SPESclin_run-011714",
    "sub-ccepAgeUMCU25/ses-1/ieeg/sub-ccepAgeUMCU25_ses-1_task-SPESclin_run-031716",
    "sub-ccepAgeUMCU25/ses-1/ieeg/sub-ccepAgeUMCU25_ses-1_task-SPESclin_run-031729",
    "sub-ccepAgeUMCU28/ses-1/ieeg/sub-ccepAgeUMCU28_ses-1_task-SPESclin_run-021305",
    "sub-ccepAgeUMCU31/ses-1/ieeg/sub-ccepAgeUMCU31_ses-1_task-SPESclin_run-031740",
    "sub-ccepAgeUMCU34/ses-1/ieeg/sub-ccepAgeUMCU34_ses-1_task-SPESclin_run-011520",
    "sub-ccepAgeUMCU37/ses-1/ieeg/sub-ccepAgeUMCU37_ses-1_task-SPESclin_run-021647",
    "sub-ccepAgeUMCU40/ses-1/ieeg/sub-ccepAgeUMCU40_ses-1_task-SPESclin_run-031553",
    "sub-ccepAgeUMCU43/ses-1/ieeg/sub-ccepAgeUMCU43_ses-1_task-SPESclin_run-021101",
    "sub-ccepAgeUMCU46/ses-1/ieeg/sub-ccepAgeUMCU46_ses-1_task-SPESclin_run-021157",
    "sub-ccepAgeUMCU49/ses-1/ieeg/sub-ccepAgeUMCU49_ses-1_task-SPESclin_run-021549",
    "sub-ccepAgeUMCU49/ses-1/ieeg/sub-ccepAgeUMCU49_ses-1_task-SPESclin_run-031049",
    "sub-ccepAgeUMCU55/ses-1/ieeg/sub-ccepAgeUMCU55_ses-1_task-SPESclin_run-031717",
    "sub-ccepAgeUMCU61/ses-1/ieeg/sub-ccepAgeUMCU61_ses-1_task-SPESclin_run-021244",
    "sub-ccepAgeUMCU67/ses-1/ieeg/sub-ccepAgeUMCU67_ses-1_task-SPESclin_run-021704",
    "sub-ccepAgeUMCU67/ses-1/ieeg/sub-ccepAgeUMCU67_ses-1_task-SPESclin_run-021808",
]
SPAN = 40.0

###################################################################################################
###################################################################################################

def _segment(stem):
    """40 s of one run (cached), its reference onsets and sites."""

    from evoparam.io import openneuro_url, read_bids_events, read_brainvision_segment
    from evoparam.io._fetch import data_folder

    cache = os.path.join(data_folder(), "events-validation", "ds004080")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, os.path.basename(stem) + ".npz")
    if not os.path.exists(path):
        events = read_bids_events(openneuro_url("ds004080", stem + "_events.tsv"))
        stim = [ev for ev in events if ev.get("trial_type") == "electrical_stimulation"]
        onsets = np.array([float(ev["onset"]) for ev in stim])
        tmin = max(onsets.min() - 1.0, 0.0)
        data, sfreq, names, t0 = read_brainvision_segment(
            openneuro_url("ds004080", stem + "_ieeg.vhdr"), tmin, tmin + SPAN)
        keep = (onsets >= t0) & (onsets <= t0 + data.shape[1] / sfreq)
        sites = np.array([str(ev.get("electrical_stimulation_site", "")) for ev in stim])
        np.savez_compressed(path, data=data.astype(np.float32), sfreq=sfreq,
                            names=np.array(names), t0=t0, onsets=onsets[keep], sites=sites[keep])
    seg = np.load(path)
    return (seg["data"].astype(float), float(seg["sfreq"]), list(seg["names"]),
            float(seg["t0"]), seg["onsets"], seg["sites"])


@pytest.fixture(scope="module")
def results():
    out = []
    for stem in RUNS:
        data, sfreq, names, t0, onsets, sites = _segment(stem)
        events = detect_pulses(data, sfreq, t0=t0, ch_names=names)
        out.append((stem, sfreq, events, onsets, sites))
    return out


def test_req040_recall_precision_and_onset(results):

    assert len({stem.split("/")[0] for stem, *_ in results}) >= 5
    hits = misses = false_alarms = 0
    for _, _, events, onsets, _ in results:
        match = match_onsets(events["onset_time"], onsets, tolerance=0.01)
        hits, misses = hits + match["hits"], misses + match["misses"]
        false_alarms += match["false_alarms"]
    assert hits / (hits + misses) >= 0.99
    assert hits / (hits + false_alarms) >= 0.99

    # Onset error: the trigger in events.tsv precedes the artifact by a constant delay
    # (about 1 ms). Estimate it on the even runs, check the odd runs within 1 sample.
    calib = [(ev["onset_time"].to_numpy(), on) for i, (_, _, ev, on, _) in enumerate(results)
             if i % 2 == 0]
    delay, _ = estimate_delay([c[0] for c in calib], [c[1] for c in calib])
    assert 0 < delay < 0.0015
    for i, (stem, sfreq, events, onsets, _) in enumerate(results):
        if i % 2 == 0:
            continue
        errors = match_onsets(events["onset_time"] - delay, onsets, tolerance=0.01)["errors"]
        assert np.median(np.abs(errors)) <= 1.0 / sfreq + 1e-6, stem


def _site_agreement(results):
    agree, confident = [], []
    for _, _, events, onsets, sites in results:
        times = events["onset_time"].to_numpy()
        for onset, site in zip(onsets, sites):
            k = int(np.argmin(np.abs(times - onset)))
            if abs(times[k] - onset) > 0.01:
                continue
            same = (set(events["stim_site_guess"].iloc[k].upper().split("-"))
                    == set(site.upper().split("-")))
            agree.append(same)
            if events["site_confidence"].iloc[k] >= 0.5:
                confident.append(same)
    return np.mean(agree), np.mean(confident), len(confident) / len(agree)


def test_req042_confidence_is_calibrated(results):

    _, accuracy, fraction = _site_agreement(results)
    assert accuracy >= 0.98
    assert fraction >= 0.4


def test_req042_site_agreement(results):

    agreement, _, _ = _site_agreement(results)
    assert agreement >= 0.95
