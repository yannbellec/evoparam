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


def held_recording(pair=(2, 3), step=300., hold=True, lead=True, rail_neighbour=None,
                   n_pulses=6, sfreq=2048., seed=0):
    """Common-reference artifact on every contact, and a stimulating pair that the
    amplifier holds and/or that moves before the common artifact (as on ds004080).

    The pair's own step (``step``) is smaller than the common artifact, so the largest
    artifacts are on other contacts. ``rail_neighbour`` is a contact that saturates: a
    large artifact, then flat at the rail, without moving before the onset. The last
    contact is dead (all zeros).
    """

    rng = np.random.default_rng(seed)
    n_channels = 16
    data = make_noise(n_channels, n_pulses + 1., sfreq, seed=seed + 1, dtype=np.float64)
    data[-1] = 0.0
    onsets = np.round((np.arange(n_pulses) + 0.5) * sfreq).astype(int)
    early, duration = int(round(0.0025 * sfreq)), int(round(0.3 * sfreq))
    gains = rng.uniform(0.8, 1.2, n_channels)
    gains[-1] = 0.0
    for onset in onsets:
        add_pulse(data, onset, 600., gains)
        for contact, sign in zip(pair, (1., -1.)):
            start = onset - early if lead else onset
            if hold:
                data[contact, start:start + duration] = data[contact, start - 1] + sign * step
            else:
                data[contact, start:onset + 8] += sign * step
        if rail_neighbour is not None:
            data[rail_neighbour, onset + 1:onset + duration] = 3000.
    return data, onsets


def test_held_and_leading_contacts_win_over_larger_artifacts():

    data, onsets = held_recording(rail_neighbour=4)
    names = NAMES[:16]
    located = identify_stim_site(data, 2048., table(onsets), ch_names=names)
    assert (located["stim_site_guess"] == "A03-A04").all()
    assert (located["polarity_check"]).all()
    # Pair: held and leading; third: the saturated neighbour, held only
    assert np.allclose(located["site_confidence"], 0.5)

    events = detect_pulses(data, 2048., ch_names=names)
    assert np.abs(events["onset_sample"].to_numpy() - onsets).max() <= 1
    assert (events["stim_site_guess"] == "A03-A04").all()
    assert (events["stim_site_pulse"] == "A03-A04").all()


@pytest.mark.parametrize("hold, lead", [(True, False), (False, True)])
def test_held_or_leading_contacts_alone(hold, lead):

    data, onsets = held_recording(hold=hold, lead=lead)
    located = identify_stim_site(data, 2048., table(onsets), ch_names=NAMES[:16])
    assert (located["stim_site_guess"] == "A03-A04").all()
    assert (located["site_confidence"] > 0.9).all()

    # The plain largest-artifact rule picks other contacts: no lead and no hold window
    plain = identify_stim_site(data, 2048., table(onsets), ch_names=NAMES[:16], lead=0.001,
                               hold=(0.005, 0.005))
    assert not (plain["stim_site_guess"] == "A03-A04").any()


def test_saturated_neighbour_and_dead_channel_are_not_chosen():

    # Without lead, the pair (held) and the saturated neighbour (held) tie on evidence;
    # the neighbour's larger artifact wins, but at low confidence.
    data, onsets = held_recording(lead=False, rail_neighbour=4)
    located = identify_stim_site(data, 2048., table(onsets), ch_names=NAMES[:16])
    assert located["stim_site_guess"].str.contains("A05").all()
    assert not located["stim_site_guess"].str.contains("B08").any()
    assert (located["site_confidence"] < 0.5).all()


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


def test_pair_at_the_amplifier_range_wins_over_quiet_channels(monkeypatch):

    # Mayo recordings: the stimulating pair reaches the amplifier's range (clipped), while
    # two very quiet contacts have the largest artifact relative to their noise.
    rng = np.random.default_rng(5)
    data = make_noise(16, 6., 2048., seed=6, dtype=np.float64)
    data[8:10] *= 0.3
    onsets = np.round((np.arange(5) + 0.5) * 2048.).astype(int)
    gains = rng.uniform(0.15, 0.25, 16)
    gains[2], gains[3] = 2.0, -2.0
    gains[8:10] = 0.4
    for onset in onsets:
        add_pulse(data, onset, 3000., gains)
    data = np.clip(data, -2500., 2500.)
    located = identify_stim_site(data, 2048., table(onsets), ch_names=NAMES[:16])
    assert (located["stim_site_guess"] == "A03-A04").all()
    assert (located["site_confidence"] >= 0.5).all()

    from evoparam.events import site
    monkeypatch.setattr(site, "RAIL_TOLERANCE", -1.0)         # never at the range
    plain = identify_stim_site(data, 2048., table(onsets), ch_names=NAMES[:16])
    assert (plain["stim_site_guess"] == "B01-B02").all()


def test_silent_pair_switched_to_the_stimulator():

    # ds004977: the stimulating contacts are disconnected from the amplifier (almost no
    # noise, no artifact); the largest artifact is on the contacts next to them.
    rng = np.random.default_rng(7)
    data = make_noise(16, 6., 2048., seed=8, dtype=np.float64)
    onsets = np.round((np.arange(5) + 0.5) * 2048.).astype(int)
    gains = rng.uniform(0.25, 0.35, 16)
    gains[1], gains[4] = 1.0, -1.0
    for onset in onsets:
        add_pulse(data, onset, 500., gains)
    data[2:4] = rng.normal(0, 0.02, (2, data.shape[1]))
    located = identify_stim_site(data, 2048., table(onsets), ch_names=NAMES[:16])
    assert (located["stim_site_guess"] == "A03-A04").all()
    assert (located["site_confidence"] >= 0.5).all()

    # A quiet pair elsewhere (not next to the largest artifact) is not a site
    data[2:4] = make_noise(2, 6., 2048., seed=9, dtype=np.float64)
    for onset in onsets:
        add_pulse(data[2:4], onset, 500., gains[2:4])
    data[12:14] = rng.normal(0, 0.02, (2, data.shape[1]))
    located = identify_stim_site(data, 2048., table(onsets), ch_names=NAMES[:16])
    assert (located["stim_site_guess"] == "A02-A05").all()
