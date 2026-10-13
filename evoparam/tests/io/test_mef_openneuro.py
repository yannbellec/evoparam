# SPDX-License-Identifier: Apache-2.0
"""Reading the Mayo Clinic MEF 3.0 recordings of OpenNeuro (network).

ds003708 (ECoG, 2048 Hz) and ds004977 (sEEG, 4800 Hz, stored in nV with a conversion
factor of 0.001 to µV, censored with zeros outside one stimulation site per run) are read
by HTTP byte ranges: only the metadata of the channels read and the blocks under the
window are transferred (about 25 MB for 20 s of the 89 ds003708 channels, most of it the
block indexes; about 13 MB for the 273 ds004977 headers). Nothing is cached: these tests
check the reader itself. Runs only with EVOPARAM_NETWORK_TESTS=1.
"""

import csv
import io
import os
from typing import Any

import numpy as np
import pytest

from evoparam.io import openneuro_url, read, read_mef, read_mef_header

# Sampling rates (Hz) of ds003708 and ds004977.
DS003708_SFREQ = 2048.0
DS004977_SFREQ = 4800.0
# Most bytes transferred for the 20 s window: data ranges, and in all.
MAX_DATA_BYTES = 10e6
MAX_BYTES = 40e6
# Fewest pulses in the window.
MIN_PULSES = 3
# Channels of ds004977, units factor of its LFP channel, smallest SD of the data (uV),
# largest onset error (s) of the artifact, and smallest artifact (uV).
N_DS004977_CHANNELS = 273
LFP_FACTOR = 0.001
MIN_SD = 10
MAX_ONSET_ERROR = 0.003
MIN_ARTIFACT = 10_000

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
        reason="set EVOPARAM_NETWORK_TESTS=1 to read OpenNeuro",
    ),
]

DS003708 = ("ds003708", "sub-01/ses-ieeg01/ieeg/sub-01_ses-ieeg01_task-ccep_run-01")
DS004977 = ("ds004977", "sub-1/ses-ieeg01/ieeg/sub-1_ses-ieeg01_task-ccep_run-01")

###################################################################################################
###################################################################################################


def _table(dataset: Any, stem: Any, suffix: Any) -> Any:
    from evoparam.io._fetch import fetch_bytes

    text = fetch_bytes(openneuro_url(dataset, stem + suffix)).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def _stim_onsets(dataset: Any, stem: Any) -> Any:
    rows = _table(dataset, stem, "_events.tsv")
    stim = [row for row in rows if row["trial_type"] == "electrical_stimulation"]
    return (
        np.array([float(row["onset"]) for row in stim]),
        [row["electrical_stimulation_site"] for row in stim],
    )


def test_ds003708_header_matches_channels_tsv() -> None:

    dataset, stem = DS003708
    header = read_mef_header(openneuro_url(dataset, stem + "_ieeg.mefd"))
    rows = _table(dataset, stem, "_channels.tsv")
    assert header["sfreq"] == DS003708_SFREQ
    assert header["ch_names"] == [row["name"] for row in rows]
    assert {float(row["sampling_frequency"]) for row in rows} == {2048.0}
    assert all(channel["voltage"] for channel in header["channels"])
    assert not header["section_3_encrypted"]


def test_ds003708_window_matches_events_tsv() -> None:

    from evoparam.events import detect_pulses, match_onsets

    dataset, stem = DS003708
    onsets, _ = _stim_onsets(dataset, stem)
    tmin = float(np.sort(onsets)[1]) - 1.0  # the first pulse is isolated
    trials = read(openneuro_url(dataset, stem + "_ieeg.mefd"), tmin=tmin, tmax=tmin + 20.0)
    source = trials.provenance.source
    assert source["format"] == "mef"
    assert source["remote"]
    assert trials.sfreq == DS003708_SFREQ
    assert trials.data.shape[1:] == (89, 40960)
    assert np.isfinite(trials.data).all()
    # Only the blocks under the window are transferred
    assert source["data_bytes_read"] < MAX_DATA_BYTES
    assert source["bytes_read"] < MAX_BYTES

    t0 = trials.times[0]
    assert abs(t0 - tmin) < 1.0 / trials.sfreq
    events = detect_pulses(trials.data[0], trials.sfreq, t0=t0)
    inside = onsets[(onsets >= t0) & (onsets < t0 + 20.0)]
    match = match_onsets(events["onset_time"], inside, tolerance=0.01)
    assert len(inside) >= MIN_PULSES
    assert match["hits"] == len(inside)
    assert match["false_alarms"] == 0
    # events.tsv onsets are sample numbers divided by the sampling frequency
    assert np.median(np.abs(match["errors"])) <= 1.0 / trials.sfreq + 1e-6


def test_ds004977_header_units_and_censoring() -> None:

    dataset, stem = DS004977
    url = openneuro_url(dataset, stem + "_ieeg.mefd")
    header = read_mef_header(url)
    rows = _table(dataset, stem, "_channels.tsv")
    assert header["sfreq"] == DS004977_SFREQ
    assert header["ch_names"] == [row["name"] for row in rows]
    assert len(header["ch_names"]) == N_DS004977_CHANNELS
    lfp = header["channels"][header["ch_names"].index("ROC13")]
    assert lfp["units"].strip().lower() == "microvolts"
    assert lfp["factor"] == LFP_FACTOR

    onsets, sites = _stim_onsets(dataset, stem)
    first = float(onsets[[site == "ROC11-ROC12" for site in sites]].min())
    picks = ["ROC13", "RA8", "RA1", "RT2"]
    data, sfreq, names, t0 = read_mef(url, tmin=first - 0.5, tmax=first + 0.5, picks=picks)
    assert names == picks
    assert sfreq == DS004977_SFREQ
    assert data.shape == (4, 4800)
    assert np.isfinite(data).all()
    assert (data.std(axis=1) > MIN_SD).all()
    # The stimulation artifact (tens of mV near the site) starts at the events.tsv onset
    jump = int(np.argmax(np.abs(np.diff(data[0])))) + 1
    assert abs(t0 + jump / sfreq - first) < MAX_ONSET_ERROR
    assert np.abs(data[0]).max() > MIN_ARTIFACT

    # Outside the uncensored span the recording is zeros
    censored, *_ = read_mef(url, tmin=100.0, tmax=105.0, picks=picks)
    assert censored.shape == (4, 24000)
    assert (censored == 0).all()
