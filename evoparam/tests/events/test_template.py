# SPDX-License-Identifier: Apache-2.0
"""Template confirmation and recovery (REQ-041)."""

import numpy as np

from evoparam.events import detect_pulses, match_onsets

from .helpers import add_pulse, make_noise, make_recording, spatial_gains

###################################################################################################
###################################################################################################


def test_req041_synthetic_pulses_line_noise_and_transients():
    """100 channels, 2048 Hz, 200 pulses (artifact >= 10x baseline SD on the median
    channel), 50 Hz line noise, 20 movement transients of 5x SD."""

    sfreq = 2048.0
    data, onsets, transients = make_recording(
        n_channels=100,
        sfreq=sfreq,
        duration=100.3,
        interval=0.5,
        first=0.3,
        amplitude=(30.0, 300.0),
        line_hz=50.0,
        line_amp=2.0,
        n_transients=20,
        transient_sd=5.0,
        seed=11,
    )
    assert len(onsets) == 200

    events = detect_pulses(data, sfreq, method="threshold+template", expected_rate=2.0)
    results = match_onsets(events["onset_sample"] / sfreq, onsets / sfreq, tolerance=0.002)
    assert results["recall"] >= 0.99
    assert results["precision"] >= 0.99
    # No transient is detected as a pulse
    detected = events["onset_sample"].to_numpy()
    for start in transients:
        assert np.all(np.abs(detected - start) > 0.03 * sfreq)


def test_sharp_transients_far_weaker_than_pulses_are_rejected():

    # Electrode pops: one-sample steps of 40 SD on most channels. They cross the threshold
    # but are an order of magnitude weaker than the pulses, as on ds004080 (pulses reach
    # hundreds to thousands of robust SD, transients tens)
    sfreq = 2048.0
    data, onsets, _ = make_recording(
        n_channels=32, duration=40.0, interval=2.0, amplitude=(1500.0, 2500.0), seed=12
    )
    rng = np.random.default_rng(3)
    pops = []
    for _ in range(10):
        while True:
            start = int(rng.integers(100, data.shape[1] - 100))
            if np.min(np.abs(onsets - start)) > 0.6 * sfreq:
                break
        data[:, start:] += 40.0 * (rng.random(32)[:, None] < 0.9)
        pops.append(start)

    threshold_only = detect_pulses(data, sfreq, method="threshold")
    assert len(threshold_only) > len(onsets)  # the pops cross the threshold

    events = detect_pulses(data, sfreq)
    results = match_onsets(events["onset_sample"] / sfreq, onsets / sfreq, tolerance=0.002)
    assert results["hits"] == len(onsets)
    assert results["false_alarms"] == 0


def test_weak_pulses_are_recovered_by_template():

    # Same site, two currents: the weak pulses are under the strength gate but match
    sfreq = 2048.0
    rng = np.random.default_rng(13)
    data = make_noise(32, 30.0, sfreq, seed=14)
    gains = spatial_gains(32, (10, 11), rng, floor=0.6)
    strong = np.round(np.arange(0.5, 15, 1.0) * sfreq).astype(int)
    weak = np.round(np.arange(15.5, 29.5, 1.0) * sfreq).astype(int)
    for onset in strong:
        add_pulse(data, onset, 400.0, gains)
    for onset in weak:
        add_pulse(data, onset, 25.0, gains)

    events = detect_pulses(data, sfreq)
    found = events["onset_sample"].to_numpy()
    assert set(strong) <= set(found)
    assert set(weak) <= set(found)
    recovered = events.loc[events["onset_sample"].isin(weak), "detector"]
    assert (recovered == "template").all()


def test_grey_zone_pulses_confirmed_by_repetition():

    # Pulses at an eighth of the strong ones' size: under the gate, but they repeat
    sfreq = 2048.0
    rng = np.random.default_rng(18)
    data = make_noise(32, 20.0, sfreq, seed=19)
    gains = spatial_gains(32, (10, 11), rng, floor=0.6)
    strong = np.round(np.arange(0.5, 10, 1.0) * sfreq).astype(int)
    grey = np.round(np.arange(10.5, 19.5, 1.0) * sfreq).astype(int)
    for onset in strong:
        add_pulse(data, onset, 400.0, gains)
    for onset in grey:
        add_pulse(data, onset, 50.0, gains)
    events = detect_pulses(data, sfreq)
    assert set(grey) <= set(events["onset_sample"])


def test_alternating_polarity_forms_two_classes():

    sfreq = 2048.0
    rng = np.random.default_rng(15)
    data = make_noise(16, 20.0, sfreq, seed=16)
    gains = spatial_gains(16, (4, 5), rng)
    onsets = np.round(np.arange(0.5, 19.5, 1.0) * sfreq).astype(int)
    for ind, onset in enumerate(onsets):
        add_pulse(data, onset, 300.0, gains, polarity=1.0 if ind % 2 == 0 else -1.0)

    events = detect_pulses(data, sfreq)
    assert (events["onset_sample"].to_numpy() == onsets).all()
    classes = events["template_class"].to_numpy()
    assert len(set(classes[::2])) == 1 and len(set(classes[1::2])) == 1
    assert classes[0] != classes[1]


def test_template_method_with_user_template():

    sfreq = 2048.0
    data, onsets, _ = make_recording(n_channels=8, duration=10.0, seed=17)
    from evoparam.events._score import shifted_snippets, snippet_lengths

    n_pre, n_post, _ = snippet_lengths(sfreq)
    template = shifted_snippets(data, np.arange(8), onsets[:1], n_pre, n_post, 0)[0, 0]

    events = detect_pulses(data, sfreq, method="template", template=template)
    assert (events["onset_sample"].to_numpy() == onsets).all()
    assert (events["detector"] == "template").all()
