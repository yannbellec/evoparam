# SPDX-License-Identifier: Apache-2.0
"""Tests for evoparam.io.bids."""

from evoparam.io.bids import *

###################################################################################################
###################################################################################################


def test_read_and_select_events(tmp_path):

    text = (
        "onset\tduration\ttrial_type\telectrical_stimulation_site\n"
        "1.5\t0.001\telectrical_stimulation\tA1-A2\n"
        "2.5\tn/a\tartefact\tn/a\n"
        "6.5\t0.001\telectrical_stimulation\tA3-A4\n"
    )
    path = tmp_path / "events.tsv"
    path.write_text(text, encoding="utf-8")

    events = read_bids_events(str(path))
    assert len(events) == 3
    assert events[0]["onset"] == 1.5
    assert events[1]["duration"] is None

    assert len(select_stimulation_events(events)) == 2
    selected = select_stimulation_events(events, site="A3-A4")
    assert [event["onset"] for event in selected] == [6.5]
