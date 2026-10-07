"""Amplifier saturation (REQ-043)."""

import numpy as np
import pytest

from evoparam.errors import DataError
from evoparam.events import detect_pulses, detect_saturation

from .helpers import make_noise, make_recording

###################################################################################################
###################################################################################################

RAIL = 50.0


def clipped_recording(lengths, sfreq=2048., seed=0):
    """Noise with clipping runs of the given lengths at +/-RAIL, on rotating channels."""

    rng = np.random.default_rng(seed)
    data = make_noise(8, 60., sfreq, seed=seed + 1, dtype=np.float64)
    starts = np.round(np.linspace(1., 59., len(lengths)) * sfreq).astype(int)
    runs = []
    for ind, (start, length) in enumerate(zip(starts, lengths)):
        ch = ind % 8
        sign = 1. if rng.random() < 0.5 else -1.
        data[ch, start:start + length] = sign * RAIL
        runs.append((ch, start, length))
    return data, runs


def found(table, ch, start, sfreq):
    rows = table[table["channel"] == str(ch)]
    return bool(np.any(np.abs(rows["start_time"].to_numpy() - start / sfreq) < 1.5 / sfreq))


def test_req043_detection_rates():

    sfreq = 2048.
    lengths = [3, 4, 5, 8, 12] * 20 + [2] * 100
    data, runs = clipped_recording(lengths, sfreq)
    _, table = detect_saturation(data, sfreq)

    long_runs = [run for run in runs if run[2] >= 3]
    short_runs = [run for run in runs if run[2] == 2]
    assert all(found(table, ch, start, sfreq) for ch, start, _ in long_runs)
    assert np.mean([found(table, ch, start, sfreq) for ch, start, _ in short_runs]) >= 0.95


def test_req043_false_positives_on_clean_data():

    # 10 minutes of clean data, including quantised (integer-coded) noise
    sfreq = 2048.
    clean = make_noise(8, 600., sfreq, sd=3.0, seed=5, dtype=np.float64)
    quantised = np.round(make_noise(8, 600., sfreq, sd=3.0, seed=6, dtype=np.float64))
    for data in (clean, quantised):
        _, table = detect_saturation(data, sfreq)
        assert len(table) <= 1


def test_mask_window_includes_recovery():

    sfreq = 2048.
    data, runs = clipped_recording([5], sfreq)
    mask, table = detect_saturation(data, sfreq, recovery=0.005)
    ch, start, length = runs[0]
    stop = start + length + int(round(0.005 * sfreq))
    assert mask[ch, start:stop].all()
    assert not mask[ch, stop + 1:].any()
    assert not mask[ch, :start].any()
    assert np.isclose(table["end_time"].iloc[0], stop / sfreq)
    assert table["n_samples"].iloc[0] == length


def test_known_rail_flags_single_samples():

    sfreq = 2048.
    data = make_noise(2, 5., sfreq, dtype=np.float64)
    data[0, 1000] = 100.
    _, table = detect_saturation(data, sfreq, rail=100.)
    assert len(table) == 1 and table["n_samples"].iloc[0] == 1
    _, table = detect_saturation(data, sfreq)          # estimated rail needs 2 samples
    assert len(table) == 0


def test_stuck_code_flat_run():

    sfreq = 2048.
    data = make_noise(2, 5., sfreq, dtype=np.float64)
    data[1, 3000] = 80.                                 # sets the estimated rail
    data[1, 5000:5004] = 60.                            # stuck below the rail, above half
    _, table = detect_saturation(data, sfreq)
    assert (table["kind"] == "flat").any()


def test_per_pulse_search_and_event_link():

    sfreq = 2048.
    data, onsets, _ = make_recording(n_channels=8, duration=6., seed=31)
    data = data.astype(np.float64)
    rail = 1.5 * np.abs(data).max()
    # Channel 3 sticks at the rail right after the third pulse only
    data[3, onsets[2] + 1:onsets[2] + 6] = rail
    events = detect_pulses(data, sfreq, saturation=False)
    mask, table = detect_saturation(data, sfreq, events=events, rail=rail)
    assert set(table["event_index"]) == {2}
    assert mask[3].any() and not mask[[0, 1, 2, 4, 5, 6, 7]].any()

    events = detect_pulses(data, sfreq)
    assert events["saturated"].tolist() == [i == 2 for i in range(len(events))]
    assert events.loc[2, "n_saturated_channels"] == 1


def test_invalid_input():

    with pytest.raises(DataError):
        detect_saturation(np.zeros((2, 100)), 1000., min_run=1)
    with pytest.raises(DataError):
        detect_saturation(np.zeros((2, 100)), 1000., ch_names=["a"])
