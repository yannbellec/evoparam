# SPDX-License-Identifier: Apache-2.0
"""Rate-constrained detection (REQ-041) and interval flags."""

import numpy as np

from evoparam.events import detect_pulses

from .helpers import add_pulse, make_noise, make_recording, spatial_gains

###################################################################################################
###################################################################################################


def test_expected_rate_sets_min_interval_for_trains():

    # 50 Hz train: the default 0.5 s refractory period would merge the pulses
    sfreq = 2048.0
    data, onsets, _ = make_recording(n_channels=8, duration=2.0, interval=0.02, first=0.1, seed=21)
    merged = detect_pulses(data, sfreq)
    assert len(merged) < len(onsets)

    events = detect_pulses(data, sfreq, expected_rate=50.0)
    assert (events["onset_sample"].to_numpy() == onsets).all()
    assert (events["interval_flag"] == "").all()


def test_interval_flags():

    sfreq = 2048.0
    rng = np.random.default_rng(22)
    data = make_noise(8, 12.0, sfreq, seed=23)
    gains = spatial_gains(8, (2, 3), rng)
    times = [0.5, 1.5, 2.5, 3.5, 6.5, 7.5, 7.8, 8.5]  # one long gap, one short interval
    onsets = np.round(np.array(times) * sfreq).astype(int)
    for onset in onsets:
        add_pulse(data, onset, 300.0, gains)

    events = detect_pulses(data, sfreq, expected_rate=1.0, min_interval=0.2)
    flags = dict(zip(np.round(events["onset_time"], 1), events["interval_flag"], strict=False))
    assert flags[6.5] == "long"
    assert flags[7.8] == "short"
    assert flags[1.5] == ""
