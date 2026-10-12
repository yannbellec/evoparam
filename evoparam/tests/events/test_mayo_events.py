# SPDX-License-Identifier: Apache-2.0
"""Pulse detection and site identification on the Mayo Clinic MEF 3.0 datasets (REQ-040,
REQ-042).

The four Mayo CCEP datasets of OpenNeuro store their recordings as MEF 3.0 sessions:
ds003708 (1 run, ECoG, 2048 Hz), ds004457 (5 runs, sEEG, 2048 Hz), ds004696 (8 runs, of
which 3 repeat ds004457 sub-1, sub-2 and sub-4: the 5 others are used) and ds004977
(4800 Hz; censored by its authors with zeros outside one site per run, so one run per
subject is used, at its uncensored site, and a reference pulse whose data are zeros on
every channel is not counted: none in the segments used).
For each run, ``SPAN`` s are read from 1 s before the first pulse of each listed site,
and compared with ``events.tsv`` (172 pulses: 152 at 2048 Hz, 20 in ds004977).
``detect_pulses`` runs with its default settings and channel names only, as on ds004080
(``test_ds004080_events.py``). Segments are cached in the evoparam data folder (float32),
so only the first run of the test downloads them (about 0.7 GB cached, 10-15 min). Runs only with
EVOPARAM_NETWORK_TESTS=1.
"""

import csv
import io
import os

import numpy as np
import pytest

from evoparam.events import detect_pulses, match_onsets

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
        reason="set EVOPARAM_NETWORK_TESTS=1 to read OpenNeuro",
    ),
]

# (dataset, run, sites): the first pulse of each site starts a segment. The sites of the
# 2048 Hz runs are the 10th, 50th and 90th percentiles of the order of first stimulation.
RUNS = [
    (
        "ds003708",
        "sub-01/ses-ieeg01/ieeg/sub-01_ses-ieeg01_task-ccep_run-01",
        ["LTG6-LTG7", "LTG28-LTG29", "LPS1-LPS2"],
    ),
    (
        "ds004457",
        "sub-1/ses-ieeg01/ieeg/sub-1_ses-ieeg01_task-ccep_run-01",
        ["LB3-LB4", "LJ3-LJ4", "LB13-LB14"],
    ),
    (
        "ds004457",
        "sub-2/ses-ieeg01/ieeg/sub-2_ses-ieeg01_task-ccep_run-01",
        ["RK3-RK4", "RS1-RS2", "RA1-RA2"],
    ),
    (
        "ds004457",
        "sub-3/ses-ieeg01/ieeg/sub-3_ses-ieeg01_task-ccep_run-01",
        ["LA7-LA8", "LC9-LC10", "LTM2-LTM3"],
    ),
    (
        "ds004457",
        "sub-4/ses-ieeg01/ieeg/sub-4_ses-ieeg01_task-ccep_run-01",
        ["RB2-RB3", "RP3-RP4", "ROC8-ROC9"],
    ),
    (
        "ds004457",
        "sub-5/ses-ieeg01/ieeg/sub-5_ses-ieeg01_task-ccep_run-01",
        ["RK5-RK6", "RO1-RO2", "RMO2-RMO3"],
    ),
    (
        "ds004696",
        "sub-02/ses-ieeg01/ieeg/sub-02_ses-ieeg01_task-ccep_run-01",
        ["RA2-RA3", "RZ2-RZ3", "LX6-LX7"],
    ),
    (
        "ds004696",
        "sub-03/ses-ieeg01/ieeg/sub-03_ses-ieeg01_task-ccep_run-01",
        ["RB12-RB13", "RA11-RA12", "RF7-RF8"],
    ),
    (
        "ds004696",
        "sub-06/ses-ieeg01/ieeg/sub-06_ses-ieeg01_task-ccep_run-01",
        ["LD10-LD11", "LZ2-LZ3", "LB11-LB12"],
    ),
    (
        "ds004696",
        "sub-07/ses-ieeg01/ieeg/sub-07_ses-ieeg01_task-ccep_run-01",
        ["LK4-LK5", "LZ3-LZ4", "LB1-LB2"],
    ),
    (
        "ds004696",
        "sub-08/ses-ieeg01/ieeg/sub-08_ses-ieeg01_task-ccep_run-01",
        ["RA1-RA2", "RPO5-RPO6", "RF6-RF7"],
    ),
    ("ds004977", "sub-1/ses-ieeg01/ieeg/sub-1_ses-ieeg01_task-ccep_run-01", ["ROC11-ROC12"]),
    ("ds004977", "sub-2/ses-ieeg01/ieeg/sub-2_ses-ieeg01_task-ccep_run-12", ["RZ4-RZ5"]),
    ("ds004977", "sub-3/ses-ieeg01/ieeg/sub-3_ses-ieeg01_task-ccep_run-03", ["LA2-LA3"]),
    ("ds004977", "sub-4/ses-ieeg01/ieeg/sub-4_ses-ieeg01_task-ccep_run-02", ["RO3-RO4"]),
]
SPAN = 20.0

###################################################################################################
###################################################################################################


def _table(dataset, rel):
    from evoparam.io import openneuro_url
    from evoparam.io._fetch import fetch_bytes

    text = fetch_bytes(openneuro_url(dataset, rel)).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def run_segments(dataset, stem, sites, span=SPAN):
    """Segments of ``span`` s from 1 s before the first pulse of each site (cached).

    Returns a list of dicts: data (µV), sfreq, names, t0, onsets and sites of the
    stimulation events inside the segment, types and status of channels.tsv.
    """

    from evoparam.io import openneuro_url
    from evoparam.io._fetch import data_folder
    from evoparam.io.mef import _read_grid, read_mef_header

    cache = os.path.join(data_folder(), "events-validation", dataset)
    os.makedirs(cache, exist_ok=True)
    paths = [
        os.path.join(cache, f"{os.path.basename(stem)}_{site}_{span:g}s.npz") for site in sites
    ]
    if not all(os.path.exists(path) for path in paths):
        events = [
            row
            for row in _table(dataset, stem + "_events.tsv")
            if row.get("trial_type") == "electrical_stimulation"
        ]
        onsets = np.array([float(row["onset"]) for row in events])
        labels = np.array([row["electrical_stimulation_site"] for row in events])
        channels = _table(dataset, stem + "_channels.tsv")
        header = read_mef_header(openneuro_url(dataset, stem + "_ieeg.mefd"))
        sfreq = header["sfreq"]
        firsts = [float(onsets[labels == site].min()) - 1.0 for site in sites]
        starts = np.round(np.array(firsts) * sfreq).astype(np.int64)
        data, _ = _read_grid(
            header, list(range(len(header["ch_names"]))), starts, int(round(span * sfreq))
        )
        lookup = {row["name"]: row for row in channels}
        types = np.array([lookup.get(name, {}).get("type", "") for name in header["ch_names"]])
        status = np.array([lookup.get(name, {}).get("status", "") for name in header["ch_names"]])
        for path, start, segment in zip(paths, starts, data):
            t0 = start / sfreq
            keep = (onsets >= t0) & (onsets <= t0 + span)
            np.savez_compressed(
                path,
                data=segment.astype(np.float32),
                sfreq=sfreq,
                names=np.array(header["ch_names"]),
                t0=t0,
                onsets=onsets[keep],
                sites=labels[keep],
                types=types,
                status=status,
            )
    out = []
    for path in paths:
        seg = np.load(path)
        out.append(
            {
                "data": seg["data"].astype(float),
                "sfreq": float(seg["sfreq"]),
                "names": [str(n) for n in seg["names"]],
                "t0": float(seg["t0"]),
                "onsets": seg["onsets"],
                "sites": seg["sites"].astype(str),
                "types": seg["types"].astype(str),
                "status": seg["status"].astype(str),
            }
        )
    return out


def channel_table(segment):
    """Channel table from channels.tsv: types (sEEG, ECoG, EEG, ...) and bad channels."""

    pd = pytest.importorskip("pandas")
    kinds = {"SEEG": "seeg", "ECOG": "ecog", "EEG": "eeg"}
    return pd.DataFrame(
        {
            "name": segment["names"],
            "type": [kinds.get(t.upper(), "misc") for t in segment["types"]],
            "bad": [s.lower() == "bad" for s in segment["status"]],
        }
    )


def detect(seg, **kwargs):
    """detect_pulses with the default settings; no pulse found gives an empty table."""

    from evoparam.errors import EventError

    pd = pytest.importorskip("pandas")
    try:
        return detect_pulses(
            seg["data"], seg["sfreq"], t0=seg["t0"], ch_names=seg["names"], **kwargs
        )
    except EventError:
        return pd.DataFrame({"onset_time": [], "stim_site_guess": [], "site_confidence": []})


def uncensored(seg, half_width=0.1):
    """Reference pulses whose data are not zeros on every channel within 100 ms (ds004977
    is censored with zeros by its authors)."""

    keep = []
    span = int(round(half_width * seg["sfreq"]))
    for onset in seg["onsets"]:
        i0 = int(round((onset - seg["t0"]) * seg["sfreq"]))
        keep.append(bool(np.any(seg["data"][:, max(0, i0 - span) : i0 + span] != 0)))
    return np.array(keep, dtype=bool)


@pytest.fixture(scope="module")
def results():
    out = []
    for dataset, stem, sites in RUNS:
        for seg in run_segments(dataset, stem, sites):
            keep = uncensored(seg)
            out.append(
                (dataset, stem, seg["sfreq"], detect(seg), seg["onsets"][keep], seg["sites"][keep])
            )
    return out


def subset(results, *, mayo_2048):
    return [res for res in results if (res[0] != "ds004977") == mayo_2048]


def match_all(results):
    hits = misses = false_alarms = 0
    errors = []
    for _, _, _, events, onsets, _ in results:
        match = match_onsets(events["onset_time"], onsets, tolerance=0.01)
        hits, misses = hits + match["hits"], misses + match["misses"]
        false_alarms += match["false_alarms"]
        errors += list(match["errors"])
    return hits, misses, false_alarms, np.array(errors)


def site_agreement(results, column="stim_site_guess"):
    agree, confident = [], []
    for _, _, _, events, onsets, sites in results:
        # No vote (and no per-pulse column) when a single pulse was detected
        name = column if column in events.columns else "stim_site_guess"
        times = events["onset_time"].to_numpy()
        for onset, site in zip(onsets, sites):
            if times.size == 0:
                continue
            k = int(np.argmin(np.abs(times - onset)))
            if abs(times[k] - onset) > 0.01:
                continue
            same = set(str(events[name].iloc[k]).upper().split("-")) == set(
                site.upper().split("-")
            )
            agree.append(same)
            if events["site_confidence"].iloc[k] >= 0.5:
                confident.append(same)
    return (
        np.mean(agree),
        (np.mean(confident) if confident else np.nan),
        len(confident) / len(agree),
        len(agree),
    )


def test_mayo_onsets_match_events_tsv(results):
    """The reader puts the pulses where events.tsv says (2048 Hz runs): every pulse is
    found, within one sample (the onsets of events.tsv are sample numbers / sfreq)."""

    mayo = subset(results, mayo_2048=True)
    assert len({dataset + stem for dataset, stem, *_ in mayo}) == 11
    hits, misses, _, errors = match_all(mayo)
    assert hits + misses >= 150
    assert hits / (hits + misses) >= 0.99
    assert np.median(np.abs(errors)) <= 1 / 2048 + 1e-6  # rounding of the onsets


@pytest.mark.xfail(
    strict=False,
    reason="precision 0.84 on the Mayo 2048 Hz runs (152 hits, "
    "29 false alarms: small transients, 0.1-3 mV against "
    "35-45 mV pulses, in ds004457 sub-2 and ds004696 "
    "sub-07); REQ-040 is specified on ds004080 only",
)
def test_mayo_precision(results):

    hits, _, false_alarms, _ = match_all(subset(results, mayo_2048=True))
    assert hits / (hits + false_alarms) >= 0.99


@pytest.mark.xfail(
    strict=False,
    reason="ds004977: the default threshold (20 robust SD) finds "
    "no pulse in 2 of 4 runs (largest score 14.9 and 10.9; "
    "recall 0.50); with sEEG picks and threshold_sd=7 all 20 "
    "are found, 2.1 ms after the events.tsv onsets",
)
def test_ds004977_pulses_found_with_defaults(results):

    hits, misses, _, _ = match_all(subset(results, mayo_2048=False))
    assert hits / (hits + misses) >= 0.99


def test_req042_mayo_site_agreement(results):
    """Agreement with events.tsv on every detected pulse of the Mayo sets (measured: 0.99,
    160 of 162 pulses; 0.99 on the 2048 Hz runs, 1.0 on ds004977). The stimulating contacts
    reach the amplifier's range (34.8 mV) in ds003708, ds004457 and ds004696, and are silent
    (switched to the stimulator) in ds004977: the two kinds of evidence of
    identify_stim_site added for these sets. Before them: 0.67 (0.72 and 0). Held-out
    check, sites not used here (30th and 70th percentiles of the order of first stimulation
    of the 11 runs at 2048 Hz, 101 pulses): 0.93 (0.49 before), every pulse with
    site_confidence >= 0.5 right; 4 of the 7 errors in ds004457 sub-4."""

    agreement, _, _, n = site_agreement(results)
    assert n >= 160
    assert agreement >= 0.95
    assert site_agreement(subset(results, mayo_2048=False))[0] >= 0.95


def test_req042_mayo_confidence_is_calibrated(results):
    """Measured: accuracy 1.0 for site_confidence >= 0.5, on 41 % of the pulses."""

    _, accuracy, fraction, _ = site_agreement(results)
    assert accuracy >= 0.98
    assert fraction >= 0.4
