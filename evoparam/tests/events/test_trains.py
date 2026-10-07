"""Train grouping and metadata (REQ-044)."""

import glob
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


MAYO = os.environ.get("EVOPARAM_MAYO_EVENTS", "")


@pytest.mark.skipif(not MAYO, reason="set EVOPARAM_MAYO_EVENTS to a folder of Mayo events.tsv")
def test_req044_mayo_protocol():
    """Mayo SPES (ds003708, ds004457): about 0.2 Hz, 10-12 pulses per site.

    The Mayo protocol sometimes pauses inside a site (tens of seconds); the site, not
    the pause, defines the train.
    """

    for path in sorted(glob.glob(os.path.join(MAYO, "*_events.tsv"))):
        raw = pd.read_csv(path, sep="\t")
        raw = raw[raw["trial_type"] == "electrical_stimulation"]
        events = pd.DataFrame({"onset_time": raw["onset"].astype(float),
                               "stim_site_guess": raw["electrical_stimulation_site"]})
        grouped = group_trains(events)
        sites = grouped["stim_site_guess"].to_numpy()
        blocks = np.r_[0, np.cumsum(sites[1:] != sites[:-1])]
        # Every train holds one site, and never spans two blocks of consecutive pulses
        assert (grouped.groupby("train_id")["stim_site_guess"].nunique() == 1).all(), path
        assert (pd.Series(blocks).groupby(grouped["train_id"].to_numpy()).nunique() == 1).all()
        # Each block of consecutive pulses on one site is exactly one train
        assert (grouped["train_id"].to_numpy() == blocks).all(), path
        # About 0.2 Hz and 10-12 pulses per site
        summary = trains(grouped)
        assert 9 <= summary["n_pulses"].median() <= 12, path
        assert 2.5 <= summary["mean_interval"].median() <= 6.0, path
