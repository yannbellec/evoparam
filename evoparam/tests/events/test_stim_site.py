"""Stimulated-site identification (REQ-042, synthetic part)."""

import numpy as np
import pandas as pd
import pytest

from evoparam.events import contact_group, detect_pulses, identify_stim_site, vote_sites

from .helpers import add_pulse, make_noise

###################################################################################################
###################################################################################################

NAMES = [f"A{i:02d}" for i in range(1, 9)] + [f"B{i:02d}" for i in range(1, 9)] + ["MKR1+"]


def bipolar_recording(pair, n_pulses=10, sfreq=2048., seed=0, marker=True):
    """Artifact of opposite sign on the two contacts of ``pair``, weaker copies elsewhere."""

    rng = np.random.default_rng(seed)
    data = make_noise(len(NAMES), n_pulses + 1., sfreq, seed=seed + 1, dtype=np.float64)
    onsets = np.round((np.arange(n_pulses) + 0.5) * sfreq).astype(int)
    gains = np.full(len(NAMES), 0.2) * rng.uniform(0.8, 1.2, len(NAMES))
    signs = np.ones(len(NAMES))
    gains[pair[0]], gains[pair[1]] = 1.0, 0.9
    signs[pair[1]] = -1.
    for onset in onsets:
        add_pulse(data, onset, 500., gains * signs)
        if marker:
            data[-1, onset:onset + 3] += 5000.            # trigger channel: largest of all
    return data, onsets


def table(onsets):
    return pd.DataFrame({"onset_sample": onsets, "onset_time": onsets / 2048.})


def test_identifies_pair_with_confidence():

    data, onsets = bipolar_recording((2, 3))
    located = identify_stim_site(data, 2048., table(onsets), ch_names=NAMES)
    assert (located["stim_site_guess"] == "A03-A04").all()
    assert (located["polarity_check"]).all()
    assert (located["site_confidence"] > 0.5).all()


def test_shaft_check_halves_confidence():

    data, onsets = bipolar_recording((7, 8))           # A08 and B01: different electrodes
    located = identify_stim_site(data, 2048., table(onsets), ch_names=NAMES)
    assert (located["stim_site_guess"] == "A08-B01").all()
    assert (located["site_confidence"] <= 0.5).all()

    groups = pd.DataFrame({"name": NAMES, "group": ["shaft"] * len(NAMES)})
    located = identify_stim_site(data, 2048., table(onsets), ch_names=NAMES, ch_info=groups)
    assert (located["site_confidence"] > 0.5).all()


def test_channel_types_exclude_non_neural():

    data, onsets = bipolar_recording((2, 3))
    info = pd.DataFrame({"name": NAMES, "type": ["seeg"] * 16 + ["trig"]})
    located = identify_stim_site(data, 2048., table(onsets), ch_names=NAMES, ch_info=info)
    assert (located["stim_site_guess"] == "A03-A04").all()

    located = identify_stim_site(data, 2048., table(onsets), ch_names=NAMES, exclude=None)
    assert located["stim_site_guess"].str.contains("MKR1+", regex=False).all()


def test_unrecorded_contacts_give_low_confidence():

    # The stimulating contacts are not in the recording: every channel sees the same copy
    rng = np.random.default_rng(3)
    data = make_noise(16, 6., 2048., seed=4, dtype=np.float64)
    onsets = np.round((np.arange(5) + 0.5) * 2048.).astype(int)
    for onset in onsets:
        add_pulse(data, onset, 300., rng.uniform(0.95, 1.05, 16))
    located = identify_stim_site(data, 2048., table(onsets), ch_names=NAMES[:16])
    assert (located["site_confidence"] < 0.5).all()


def test_high_frequency_train_warns():

    data = make_noise(4, 1., 2048., dtype=np.float64)
    onsets = np.arange(100, 2000, 8)
    with pytest.warns(UserWarning, match="closer than"):
        located = identify_stim_site(data, 2048., table(onsets), ch_names=NAMES[:4])
    assert (located["stim_site_guess"] == "").all()


def test_vote_corrects_isolated_wrong_partner():

    events = pd.DataFrame({
        "onset_time": np.arange(8) * 5.0,
        "stim_site_guess": ["A01-A02", "A01-A02", "A01-B07", "A01-A02",
                            "C01-C02", "C01-C02", "C02-C05", "C01-C02"],
        "site_confidence": [1, 1, 0.5, 1, 1, 1, 0.5, 1.0]})
    voted = vote_sites(events)
    assert voted["stim_site_guess"].tolist() == ["A01-A02"] * 4 + ["C01-C02"] * 4
    assert voted["stim_site_pulse"].tolist() == events["stim_site_guess"].tolist()
    assert voted["site_block"].tolist() == [0] * 4 + [1] * 4
    assert np.allclose(voted["site_confidence"].iloc[0], 0.75 * np.mean([1, 1, 0.5, 1]))


def test_detect_pulses_fills_site_columns():

    data, onsets = bipolar_recording((10, 11))
    events = detect_pulses(data, 2048., ch_names=NAMES)
    assert (events["stim_site_guess"] == "B03-B04").all()
    assert "stim_site_pulse" in events.columns


def test_contact_group():

    assert contact_group("PT01") == "PT"
    assert contact_group("LA'12") == "LA'"
    assert contact_group("Grid_3") == "Grid"
    assert contact_group("Cz") == "Cz"
