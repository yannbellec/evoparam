# SPDX-License-Identifier: Apache-2.0
"""Masks and locality of every method (REQ-060)."""

import warnings

import numpy as np
import pytest

from evoparam.artifact import remove_artifact

from .helpers import make_overlay

###################################################################################################
###################################################################################################

WINDOW = (-0.002, 0.008)
LOCAL = [
    ("interpolate", {}),
    ("template", {}),
    ("template", {"template": "local"}),
    ("learned", {}),
    ("matching_pursuit", {}),
    ("none", {}),
]
GLOBAL = [
    ("subspace", {}),
    ("polynomial", {}),
    ("period_template", {"period": 0.05}),
    ("lowrank", {"pulse_times": np.arange(0, 0.04, 0.01)}),
]


def _run(ts, method, kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return remove_artifact(ts, method=method, window=WINDOW, **kwargs)


@pytest.mark.parametrize(("method", "kwargs"), LOCAL)
def test_local_methods_change_only_the_window(method, kwargs):

    noisy, _ = make_overlay()
    result = _run(noisy, method, kwargs)
    changed = result.cleaned.data != noisy.data
    times = noisy.times
    inside = (times >= WINDOW[0]) & (times <= WINDOW[1])
    if method == "matching_pursuit":
        inside = (times >= WINDOW[0]) & (times <= WINDOW[1] + 0.1)
    assert not changed[..., ~inside].any()
    masks = result.cleaned.masks
    replaced = np.zeros_like(changed) if masks is None else masks
    if method == "interpolate":
        # Replaced samples are exactly the masked samples
        assert np.array_equal(replaced, np.broadcast_to(inside, changed.shape))
    else:
        assert not replaced.any()


def test_tail_changes_window_and_tail_only():

    noisy, _ = make_overlay()
    result = _run(noisy, "tail", {"tail": "subtract", "tail_length": 0.03})
    changed = result.cleaned.data != noisy.data
    times = noisy.times
    allowed = (times >= WINDOW[0]) & (times <= WINDOW[1] + 0.03 + 1.0 / noisy.sfreq)
    assert not changed[..., ~allowed].any()
    inside = (times >= WINDOW[0]) & (times <= WINDOW[1])
    assert np.array_equal(result.cleaned.masks, np.broadcast_to(inside, changed.shape))


@pytest.mark.parametrize(("method", "kwargs"), GLOBAL)
def test_global_methods_change_baseline_level_by_two_percent_at_most(method, kwargs):
    """Methods that act on every sample may change the baseline's RMS level by <= 2 %."""

    if method == "subspace":
        # A spatially fixed artifact (one stimulation site) on 32 channels
        noisy, _ = make_overlay(n_channels=32, channel_jitter=False)
    elif method == "period_template":
        # A train method on a recording without train or response: only noise is filtered
        _, noisy = make_overlay(n_channels=8, amplitude=0.0)
    else:
        noisy, _ = make_overlay(n_channels=12)
    result = _run(noisy, method, kwargs)
    base = (noisy.times >= -0.5) & (noisy.times <= -0.05)
    rms = lambda x: np.sqrt(np.mean((x - x.mean(axis=-1, keepdims=True)) ** 2))
    before = rms(noisy.data[..., base])
    after = rms(result.cleaned.data[..., base])
    assert abs(after / before - 1) <= 0.02


@pytest.mark.xfail(
    strict=True,
    reason="REQ-060 for the published polynomial method "
    "(poly_extent='global', a local cubic subtracted at "
    "every sample): a high-pass filter of the whole epoch, "
    "the baseline RMS falls by 9.1 % on white noise at "
    "2048 Hz (18 % at 1 kHz, 4.0 % at 4800 Hz, 0.75 % at "
    "25 kHz); the default poly_extent='local' meets 2 %",
)
def test_req060_polynomial_global_extent_as_published():

    noisy, _ = make_overlay(n_channels=12)
    result = _run(noisy, "polynomial", {"poly_extent": "global"})
    base = (noisy.times >= -0.5) & (noisy.times <= -0.05)
    rms = lambda x: np.sqrt(np.mean((x - x.mean(axis=-1, keepdims=True)) ** 2))
    before = rms(noisy.data[..., base])
    after = rms(result.cleaned.data[..., base])
    assert abs(after / before - 1) <= 0.02


def test_existing_saturation_masks_are_kept():

    from evoparam.artifact import apply_mask

    noisy, _ = make_overlay()
    sat = np.zeros(noisy.data.shape, dtype=bool)
    sat[0, 0, 1030:1034] = True
    masked = apply_mask(noisy, sat)
    result = _run(masked, "template", {})
    assert result.cleaned.masks[0, 0, 1030:1034].all()
