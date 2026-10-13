# SPDX-License-Identifier: Apache-2.0
"""Template subtraction (REQ-061)."""

import warnings
from typing import Any

import numpy as np
import pytest

from evoparam.artifact import estimate_artifact_template, remove_artifact
from evoparam.errors import ArtifactError, SettingsError

from .helpers import make_overlay

# Residual reduction (dB) required (REQ-061), and the latency tolerance (s) of the response.
MIN_REDUCTION_DB = 20
LATENCY_TOLERANCE = 0.0005

###################################################################################################
###################################################################################################

WINDOW = (-0.002, 0.008)


def residual_db(noisy: Any, truth: Any, cleaned: Any, window: Any = WINDOW) -> Any:
    inside = (noisy.times >= window[0]) & (noisy.times <= window[1])
    before = np.sum((noisy.data - truth.data)[..., inside] ** 2)
    after = np.sum((cleaned.data - truth.data)[..., inside] ** 2)
    return 10 * np.log10(before / after)


@pytest.mark.parametrize("template", ["global", "local"])
def test_req061_template_reduces_window_residual_by_20_db(template: Any) -> None:

    # Template SNR >= 10: 20 trials, artifact thousands of uV, noise 5 uV
    noisy, truth = make_overlay(onset=0.012, n_trials=20)
    result = remove_artifact(noisy, method="template", template=template, window=WINDOW)
    assert residual_db(noisy, truth, result.cleaned) >= MIN_REDUCTION_DB


@pytest.mark.parametrize("onset", [0.008, 0.012, 0.020])
def test_req061_responses_after_the_window_are_untouched(onset: Any) -> None:

    # Artifact without a tail: it ends inside the window, so after the window the cleaned
    # data must equal the truth plus noise, and the response must not move
    noisy, truth = make_overlay(onset=onset, art_tail=0.0)
    result = remove_artifact(noisy, method="template", window=WINDOW)
    after = noisy.times > WINDOW[1]
    assert np.array_equal(result.cleaned.data[..., after], noisy.data[..., after])
    # Peak latency and amplitude of the mean response are those of the truth
    mean_clean = result.cleaned.data.mean(axis=0)
    mean_truth = truth.data.mean(axis=0)
    win = (noisy.times > onset) & (noisy.times < onset + 0.04)
    lat_clean = noisy.times[win][np.argmin(mean_clean[:, win], axis=1)]
    lat_truth = noisy.times[win][np.argmin(mean_truth[:, win], axis=1)]
    assert np.all(np.abs(lat_clean - lat_truth) <= LATENCY_TOLERANCE)


def test_scaling_follows_per_trial_amplitude() -> None:

    noisy, truth = make_overlay(jitter=0.3, seed=3)
    scaled = remove_artifact(noisy, method="template", window=WINDOW, scale=True)
    fixed = remove_artifact(noisy, method="template", window=WINDOW, scale=False)
    assert residual_db(noisy, truth, scaled.cleaned) > residual_db(noisy, truth, fixed.cleaned)


def test_user_template_and_estimate() -> None:

    noisy, truth = make_overlay()
    templates = estimate_artifact_template(noisy, WINDOW)
    assert list(templates) == [("C1-C2", 4.0, "biphasic_cathodic_first")]
    tpl = templates[("C1-C2", 4.0, "biphasic_cathodic_first")]
    result = remove_artifact(noisy, method="template", window=WINDOW, template=tpl)
    assert residual_db(noisy, truth, result.cleaned) >= MIN_REDUCTION_DB
    with pytest.raises(SettingsError):
        remove_artifact(noisy, method="template", window=WINDOW, template=np.zeros((2, 2)))


def test_small_groups_raise_or_fall_back() -> None:

    noisy, truth = make_overlay(n_trials=10)
    noisy.metadata.loc[:1, "stim_site"] = "C3-C4"  # a group of 2 pulses
    with pytest.raises(ArtifactError, match="at least 3"):
        remove_artifact(noisy, method="template", window=WINDOW)
    with pytest.warns(UserWarning, match="interpolated"):
        result = remove_artifact(noisy, method="template", window=WINDOW, fallback=True)
    inside = (noisy.times >= WINDOW[0]) & (noisy.times <= WINDOW[1])
    assert result.cleaned.masks[:2][..., inside].all()
    assert not result.cleaned.masks[2:].any()


def test_response_inside_window_warns() -> None:

    # The response starts inside the window: the template contains it
    noisy, _ = make_overlay(
        onset=0.0, amplitude=-3000.0, width=0.02, art_amplitude=200.0, art_tail=0.0
    )
    with pytest.warns(UserWarning, match="correlates with the response"):
        remove_artifact(noisy, method="template", window=(-0.002, 0.004))


def test_window_longer_than_train_interval_is_refused() -> None:

    noisy, _ = make_overlay()
    noisy.metadata["stim_frequency_hz"] = 130.0
    with pytest.raises(ArtifactError, match="period_template"):
        remove_artifact(noisy, method="template", window=(-0.002, 0.008))


def test_saturated_pulses_left_out_of_template() -> None:

    from evoparam.artifact import apply_mask

    noisy, truth = make_overlay(n_trials=12)
    inside = np.flatnonzero((noisy.times >= WINDOW[0]) & (noisy.times <= WINDOW[1]))
    data = noisy.data.copy()
    data[0, :, inside] = 9e4  # a clipped pulse
    sat = np.zeros(data.shape, dtype=bool)
    sat[0, :, inside] = True
    from dataclasses import replace

    corrupted = apply_mask(replace(noisy, data=data), sat)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = remove_artifact(corrupted, method="template", window=WINDOW)
    clean_rows = slice(1, None)
    before = np.sum((noisy.data - truth.data)[clean_rows][..., inside] ** 2)
    after = np.sum((result.cleaned.data - truth.data)[clean_rows][..., inside] ** 2)
    assert 10 * np.log10(before / after) >= MIN_REDUCTION_DB
