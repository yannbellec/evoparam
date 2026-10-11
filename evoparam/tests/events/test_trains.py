# SPDX-License-Identifier: Apache-2.0
"""Train grouping and metadata (REQ-044)."""

import os

import numpy as np
import pandas as pd
import pytest

from evoparam.errors import EventError
from evoparam.events import events_to_metadata, group_trains, trains

###################################################################################################
###################################################################################################

def test_req044_synthetic_design_exactly():
    """1 Hz x 30 pulses x 20 sites, 4 s between sites."""

    onsets, sites = [], []
    t = 0.0
    for site in range(20):
        for _ in range(30):
            onsets.append(t)
            sites.append(f"S{site:02d}-S{site:02d}b")
            t += 1.0
        t += 3.0
    events = pd.DataFrame({"onset_time": onsets, "stim_site_guess": sites})

    for kwargs in ({"expected_rate": 1.0}, {}, {"site_column": None}):
        grouped = group_trains(events.sample(frac=1, random_state=0), **kwargs)
        assert grouped["train_id"].nunique() == 20
        assert (grouped.groupby("train_id").size() == 30).all()
        assert grouped["index_in_train"].max() == 29
        assert (np.diff(grouped["onset_time"]) > 0).all()

    summary = trains(group_trains(events))
    assert len(summary) == 20
    assert (summary["n_pulses"] == 30).all()
    assert np.allclose(summary["mean_interval"], 1.0)
    assert summary["site"].tolist() == sorted(set(sites))


def test_site_change_splits_even_without_gap():

    events = pd.DataFrame({"onset_time": np.arange(10) * 5.0,
                           "stim_site_guess": ["A-B"] * 5 + ["C-D"] * 5})
    assert group_trains(events)["train_id"].tolist() == [0] * 5 + [1] * 5
    assert group_trains(events, site_column=None)["train_id"].nunique() == 1


def test_trains_needs_train_id():

    with pytest.raises(EventError):
        trains(pd.DataFrame({"onset_time": [0.0]}))


def test_events_to_metadata():

    events = group_trains(pd.DataFrame({
        "onset_time": [1.0, 2.0, 10.0], "stim_site_guess": ["A-B", "A-B", "C-D"],
        "detector": ["threshold"] * 3, "stim_current_mA": [4.0, 4.0, "6"]}))
    meta = events_to_metadata(events)
    assert meta["trial"].tolist() == [0, 1, 2]
    assert meta["stim_site"].tolist() == ["A-B", "A-B", "C-D"]
    assert meta["block"].tolist() == [0, 0, 1]
    assert meta["stim_current_mA"].dtype == float


# Every events.tsv of the four Mayo CCEP datasets of OpenNeuro (ds004696 sub-01, 04 and 05
# repeat ds004457 sub-2, sub-1 and sub-4; ds004977 keeps the events of every site although its
# signal is censored outside one site per run, which does not matter here: only the event
# times and sites are used).
MAYO_EVENTS = [
    "ds003708/sub-01/ses-ieeg01/ieeg/sub-01_ses-ieeg01_task-ccep_run-01",
    *(f"ds004457/sub-{k}/ses-ieeg01/ieeg/sub-{k}_ses-ieeg01_task-ccep_run-01"
      for k in range(1, 6)),
    *(f"ds004696/sub-0{k}/ses-ieeg01/ieeg/sub-0{k}_ses-ieeg01_task-ccep_run-01"
      for k in range(1, 9)),
    *(f"ds004977/sub-1/ses-ieeg01/ieeg/sub-1_ses-ieeg01_task-ccep_run-0{k}" for k in (1, 2, 3)),
    "ds004977/sub-2/ses-ieeg01/ieeg/sub-2_ses-ieeg01_task-ccep_run-12",
    "ds004977/sub-3/ses-ieeg01/ieeg/sub-3_ses-ieeg01_task-ccep_run-03",
    "ds004977/sub-4/ses-ieeg01/ieeg/sub-4_ses-ieeg01_task-ccep_run-02",
]
# The one run whose own events.tsv has more than 12 pulses per site (median 13 pulses per
# block, 378 pulses on 27 sites).
MAYO_LONG = "ds004696/sub-07/ses-ieeg01/ieeg/sub-07_ses-ieeg01_task-ccep_run-01"

network = pytest.mark.skipif(os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
                             reason="set EVOPARAM_NETWORK_TESTS=1 to read OpenNeuro")


def _mayo_events(stem):
    """Stimulation events of one Mayo run (events.tsv, cached in the evoparam data folder)."""

    from evoparam.io import openneuro_url
    from evoparam.io._fetch import data_folder, fetch_bytes

    dataset, rel = stem.split("/", 1)
    cache = os.path.join(data_folder(), "events-validation", "mayo-events")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"{dataset}_{os.path.basename(rel)}_events.tsv")
    if not os.path.exists(path):
        with open(path, "wb") as handle:
            handle.write(fetch_bytes(openneuro_url(dataset, rel + "_events.tsv")))
    raw = pd.read_csv(path, sep="\t")
    raw = raw[raw["trial_type"] == "electrical_stimulation"]
    return pd.DataFrame({"onset_time": raw["onset"].astype(float).to_numpy(),
                         "stim_site_guess": raw["electrical_stimulation_site"].to_numpy()})


def _mayo_trains(stem):
    events = _mayo_events(stem)
    grouped = group_trains(events)
    return events, grouped, trains(grouped)


@pytest.mark.network
@network
@pytest.mark.parametrize("stem", [s for s in MAYO_EVENTS if s != MAYO_LONG])
def test_req044_mayo_protocol(stem):
    """Mayo SPES, every run of the four datasets: about 0.2 Hz, 10-12 pulses per site.

    The Mayo protocol sometimes pauses inside a site (tens of seconds); the site, not
    the pause, defines the train.
    """

    events, grouped, summary = _mayo_trains(stem)
    sites = grouped["stim_site_guess"].to_numpy()
    blocks = np.r_[0, np.cumsum(sites[1:] != sites[:-1])]
    # Every train holds one site, and never spans two blocks of consecutive pulses
    assert (grouped.groupby("train_id")["stim_site_guess"].nunique() == 1).all()
    assert (pd.Series(blocks).groupby(grouped["train_id"].to_numpy()).nunique() == 1).all()
    # Each block of consecutive pulses on one site is exactly one train
    assert (grouped["train_id"].to_numpy() == blocks).all()
    assert len(grouped) == len(events)
    # About 0.2 Hz and 10-12 pulses per site
    assert 10 <= summary["n_pulses"].median() <= 12
    assert 2.5 <= summary["mean_interval"].median() <= 6.0


@pytest.mark.network
@network
@pytest.mark.xfail(strict=True, reason="ds004696 sub-07: the run's own events.tsv has a median "
                                       "of 13 pulses per site (378 pulses on 27 sites; 14 per "
                                       "site counting all its blocks), above the protocol's "
                                       "10-12; group_trains reproduces its blocks exactly")
def test_req044_mayo_protocol_ds004696_sub07():

    events, grouped, summary = _mayo_trains(MAYO_LONG)
    sites = grouped["stim_site_guess"].to_numpy()
    blocks = np.r_[0, np.cumsum(sites[1:] != sites[:-1])]
    assert (grouped["train_id"].to_numpy() == blocks).all()
    assert 2.5 <= summary["mean_interval"].median() <= 6.0
    assert 10 <= summary["n_pulses"].median() <= 12


@pytest.mark.network
@network
def test_req044_mayo_protocol_all_runs_together():
    """The protocol over all the trains of the 20 runs: median 10-12 pulses, about 0.2 Hz."""

    counts, intervals = [], []
    for stem in MAYO_EVENTS:
        _, _, summary = _mayo_trains(stem)
        counts += summary["n_pulses"].tolist()
        intervals += summary["mean_interval"].dropna().tolist()
    assert 10 <= np.median(counts) <= 12
    assert 2.5 <= np.median(intervals) <= 6.0
