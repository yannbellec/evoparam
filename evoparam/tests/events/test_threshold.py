# SPDX-License-Identifier: Apache-2.0
"""Robust threshold detection (inherited from the skeleton's detect_stim_onsets)."""

from typing import Any

import numpy as np
import pytest

from evoparam.errors import DataError, EventError, SettingsError
from evoparam.events import EVENT_COLUMNS, detect_pulses, match_onsets

from .helpers import make_recording

###################################################################################################
###################################################################################################


def legacy_recording(
    onsets: Any, sfreq: Any = 1000.0, duration: Any = 20.0, n_channels: Any = 4, seed: Any = 0
) -> Any:
    """The skeleton's test recording: a 3-sample block of 500 on every channel."""

    rng = np.random.default_rng(seed)
    data = rng.normal(0, 1, (n_channels, int(duration * sfreq)))
    for onset in onsets:
        ind = int(round(onset * sfreq))
        data[:, ind : ind + 3] += 500.0
    return data


@pytest.mark.parametrize("method", ["threshold", "threshold+template"])
def test_detects_legacy_recording(method: Any) -> None:

    sfreq, onsets = 1000.0, np.arange(1, 19, 2.5)
    data = legacy_recording(onsets, sfreq)

    events = detect_pulses(data, sfreq, method=method)
    results = match_onsets(events["onset_time"], onsets, tolerance=0.001)
    assert results["hits"] == len(onsets)
    assert results["false_alarms"] == 0
    # The onset is the first sample of the artifact
    assert np.allclose(results["errors"], 0)

    # Clock offset is carried through
    events = detect_pulses(data, sfreq, method=method, t0=100.0)
    assert np.allclose(events["onset_time"] - 100.0, onsets, atol=0.001)


def test_event_table_columns_and_types() -> None:

    data, onsets, _ = make_recording(duration=6.0, seed=3)
    events = detect_pulses(data, 2048.0)
    assert list(events.columns[: len(EVENT_COLUMNS)]) == list(EVENT_COLUMNS)
    assert events["onset_sample"].dtype == np.int64
    assert events["onset_time"].dtype == np.float64
    assert events["train_id"].dtype == np.int64
    assert events["saturated"].dtype == bool
    assert (events["onset_sample"].to_numpy() == onsets).all()
    assert set(events["detector"]) <= {"threshold", "template"}


def test_onset_is_first_artifact_sample_with_precursor() -> None:

    # A small switching transient 2 samples before the pulse must not move the onset
    sfreq = 2048.0
    data, onsets, _ = make_recording(duration=6.0, seed=4)
    for onset in onsets:
        data[:, onset - 3 : onset - 1] += 4.0
    events = detect_pulses(data, sfreq)
    assert (events["onset_sample"].to_numpy() == onsets).all()


def test_strongest_wins_in_refractory_period() -> None:

    # A click 100 ms before each pulse, strong enough to cross the threshold
    sfreq = 2048.0
    data, onsets, _ = make_recording(n_channels=16, duration=10.0, interval=2.0, seed=5)
    click = int(0.1 * sfreq)
    for onset in onsets:
        data[:, onset - click] += 40.0
        data[:, onset - click + 1] -= 40.0
    events = detect_pulses(data, sfreq, min_interval=0.5)
    assert (events["onset_sample"].to_numpy() == onsets).all()


def test_single_channel_and_1d_input() -> None:

    data, onsets, _ = make_recording(n_channels=1, duration=6.0, seed=6)
    events = detect_pulses(data[0], 2048.0)
    assert (events["onset_sample"].to_numpy() == onsets).all()


def test_no_pulse_raises_with_diagnostics() -> None:

    rng = np.random.default_rng(0)
    with pytest.raises(EventError, match="largest score"):
        detect_pulses(rng.normal(size=(4, 4096)), 2048.0)


def test_invalid_settings() -> None:

    data, _, _ = make_recording(duration=3.0, seed=7)
    with pytest.raises(SettingsError):
        detect_pulses(data, 2048.0, method="magic")
    with pytest.raises(SettingsError):
        detect_pulses(data, 2048.0, threshold_sd=0)
    with pytest.raises(SettingsError):
        detect_pulses(data, 2048.0, rho_min=1.5)
    with pytest.raises(DataError):
        detect_pulses(np.zeros((2, 3, 4)), 2048.0)
    with pytest.raises(DataError):
        detect_pulses(data, 2048.0, picks=["nope"])


def test_picks_restrict_channels() -> None:

    data, onsets, _ = make_recording(n_channels=8, duration=6.0, seed=8)
    # Destroy the artifact on half of the channels: picking the other half still works
    data[:4] = np.random.default_rng(1).normal(size=data[:4].shape)
    events = detect_pulses(data, 2048.0, picks=[4, 5, 6, 7])
    assert (events["onset_sample"].to_numpy() == onsets).all()


def test_return_scores() -> None:

    data, _, _ = make_recording(duration=4.0, seed=9)
    events, z, rho = detect_pulses(data, 2048.0, return_scores=True)
    assert len(z) == data.shape[1] - 1
    assert len(rho) == len(events)


@pytest.mark.parametrize("method", ["threshold", "threshold+template", "template"])
def test_spec_procedure_on_the_legacy_recording(method: Any) -> None:
    """procedure="spec": the specification's methods 1 and 2 as written. The onset is the
    last pre-artifact sample, one sample before the default onset."""

    sfreq, onsets = 1000.0, np.arange(1, 19, 2.5)
    data = legacy_recording(onsets, sfreq)
    events = detect_pulses(data, sfreq, method=method, procedure="spec")
    results = match_onsets(events["onset_time"], onsets, tolerance=0.002)
    assert results["hits"] == len(onsets)
    assert results["false_alarms"] == 0
    assert np.allclose(results["errors"], -1.0 / sfreq)
    if method != "threshold":
        assert (events["rho"] >= 0.8).all()


def test_spec_procedure_template_pass_recovers_and_rejects() -> None:
    """Method 2 as a second pass: a weaker pulse with the same artifact is added, a large
    transient of another shape is rejected."""

    sfreq = 1000.0
    onsets = np.arange(1, 19, 2.0)
    data = legacy_recording(onsets, sfreq)
    weak = 19.5
    data[:, int(weak * sfreq) : int(weak * sfreq) + 3] += 6.0  # below the threshold
    data[:, 1500:1520] += 200.0 * (-1.0) ** np.arange(20)  # a fast oscillating burst
    threshold_only = detect_pulses(data, sfreq, method="threshold", procedure="spec")
    both = detect_pulses(data, sfreq, method="threshold+template", procedure="spec")
    truth = np.r_[onsets, weak]
    assert match_onsets(both["onset_time"], truth, 0.002)["recall"] == 1.0
    assert match_onsets(both["onset_time"], truth, 0.002)["false_alarms"] == 0
    alone = match_onsets(threshold_only["onset_time"], truth, 0.002)
    assert alone["hits"] == len(onsets)
    assert alone["false_alarms"] == 1
    assert (both["detector"] == "template").sum() >= 1


def test_unknown_procedure() -> None:

    with pytest.raises(SettingsError, match="procedure"):
        detect_pulses(legacy_recording([1.0]), 1000.0, procedure="other")
