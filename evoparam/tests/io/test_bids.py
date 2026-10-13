# SPDX-License-Identifier: Apache-2.0
"""Tests for evoparam.io.bids."""

from typing import Any

from evoparam.io.bids import read_bids_events, select_stimulation_events

# Rows of the events.tsv written, onset (s) of the first, and stimulation rows.
N_EVENTS = 3
FIRST_ONSET = 1.5
N_STIMULATIONS = 2

###################################################################################################
###################################################################################################


def test_read_and_select_events(tmp_path: Any) -> None:

    text = (
        "onset\tduration\ttrial_type\telectrical_stimulation_site\n"
        "1.5\t0.001\telectrical_stimulation\tA1-A2\n"
        "2.5\tn/a\tartefact\tn/a\n"
        "6.5\t0.001\telectrical_stimulation\tA3-A4\n"
    )
    path = tmp_path / "events.tsv"
    path.write_text(text, encoding="utf-8")

    events = read_bids_events(str(path))
    assert len(events) == N_EVENTS
    assert events[0]["onset"] == FIRST_ONSET
    assert events[1]["duration"] is None

    assert len(select_stimulation_events(events)) == N_STIMULATIONS
    selected = select_stimulation_events(events, site="A3-A4")
    assert [event["onset"] for event in selected] == [6.5]
