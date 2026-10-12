# SPDX-License-Identifier: Apache-2.0
"""Local cubic fit with recovery after saturation (REQ-065)."""

import numpy as np
from scipy.signal import savgol_filter

from evoparam.artifact import apply_mask, remove_artifact
from evoparam.data import TrialSet

###################################################################################################
###################################################################################################

SFREQ = 25000.0


def _railed_recording():
    """MEA-like recording at 25 kHz: spikes, and a 600 mV pulse that rails the amplifier."""

    rng = np.random.default_rng(5)
    times = np.arange(int(-0.05 * SFREQ), int(0.1 * SFREQ)) / SFREQ
    n = len(times)
    truth = rng.normal(0, 5, (3, 4, n))
    # Spikes everywhere, including just after the pulse
    wave = -80 * np.hanning(int(0.001 * SFREQ))
    for at in np.arange(int(0.054 * SFREQ), n - 100, int(0.006 * SFREQ)):
        truth[..., at : at + len(wave)] += wave

    # A 600 mV pulse: the amplifier rails at 2 mV for 1 ms, then recovers exponentially
    i0 = np.searchsorted(times, 0.0)
    rail_end = i0 + int(0.001 * SFREQ)
    noisy = truth.copy()
    noisy[..., i0:rail_end] = 2000.0
    rel = (np.arange(n - rail_end)) / SFREQ
    noisy[..., rail_end:] += 1500.0 * np.exp(-rel / 0.002) + 300.0 * np.exp(-rel / 0.02)
    railed = np.zeros(noisy.shape, dtype=bool)
    railed[..., i0:rail_end] = True

    names = ["a", "b", "c", "d"]
    ts = apply_mask(TrialSet(data=noisy, times=times, sfreq=SFREQ, ch_names=names), railed)
    return ts, truth, i0, rail_end


def test_req065_recovery_within_2_ms_and_zeros_while_railed():
    """The published method (the fit subtracted at every sample)."""

    ts, truth, i0, rail_end = _railed_recording()
    n = len(ts.times)
    result = remove_artifact(
        ts, method="polynomial", window=(0.0, 0.001), quality=False, poly_extent="global"
    )
    cleaned = result.cleaned.data

    # Zeros while railed, and masked
    assert np.all(cleaned[..., i0:rail_end] == 0.0)
    assert result.cleaned.masks[..., i0:rail_end].all()

    # From 2 ms after the pulse, the cleaned signal matches the truth after the same filter
    half = int(round(0.003 * SFREQ))
    reference = truth - savgol_filter(truth, 2 * half + 1, 3, axis=-1)
    after = slice(i0 + int(0.002 * SFREQ), n - half)
    error = cleaned[..., after] - reference[..., after]
    assert np.sqrt(np.mean(error**2)) <= 0.5 * np.std(truth[..., :i0])


def test_req065_local_extent_zeros_while_railed_and_keeps_more_of_the_signal():
    """The default extent: zeros while railed, the published output inside the artifact,
    and closer to the true signal from 2 ms after the pulse (9.3 against 9.7 uV RMS: the
    published filter also removes the slow part of the spikes everywhere)."""

    ts, truth, i0, rail_end = _railed_recording()
    kwargs = dict(method="polynomial", window=(0.0, 0.001), quality=False)
    local = remove_artifact(ts, **kwargs).cleaned
    published = remove_artifact(ts, poly_extent="global", **kwargs).cleaned
    assert np.all(local.data[..., i0:rail_end] == 0.0)
    assert local.masks[..., i0:rail_end].all()
    assert np.allclose(
        local.data[..., i0 : i0 + int(0.01 * SFREQ)],
        published.data[..., i0 : i0 + int(0.01 * SFREQ)],
    )
    # The baseline before the ramp is untouched
    half = int(round(0.003 * SFREQ))
    assert np.array_equal(local.data[..., : i0 - half], ts.data[..., : i0 - half])
    after = slice(i0 + int(0.002 * SFREQ), len(ts.times) - half)
    rms = lambda x: np.sqrt(np.mean((x[..., after] - truth[..., after]) ** 2))
    assert rms(local.data) < rms(published.data)


def test_local_extent_keeps_the_response_after_the_artifact():
    """Overlay at 2048 Hz, response onset 20 ms: the published filter removes the response
    (RRMSE 0.99 on 8-50 ms), the local extent keeps it (0.02) and the baseline (0 %)."""

    from evoparam.artifact import compare_methods

    from .helpers import make_overlay

    noisy, truth = make_overlay(n_channels=12, onset=0.02, seed=0)
    table = compare_methods(
        noisy,
        [
            {"method": "polynomial", "label": "local"},
            {"method": "polynomial", "poly_extent": "global", "label": "global"},
        ],
        truth=truth,
    ).set_index("label")
    assert table.loc["local", "rrmse"] < 0.05
    assert table.loc["global", "rrmse"] > 0.9


def test_polynomial_without_rails_is_a_filter():

    rng = np.random.default_rng(6)
    times = np.arange(-1000, 1000) / SFREQ
    ts = TrialSet(data=rng.normal(0, 1, (1, 1, 2000)), times=times, sfreq=SFREQ, ch_names=["a"])
    result = remove_artifact(
        ts,
        method="polynomial",
        window=(0.0, 0.001),
        quality=False,
        baseline=(-0.03, -0.001),
        poly_extent="global",
    )
    expected = ts.data - savgol_filter(ts.data, 151, 3, axis=-1)
    assert np.allclose(result.cleaned.data, expected)
    assert result.cleaned.masks is None
