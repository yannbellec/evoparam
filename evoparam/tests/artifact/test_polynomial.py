"""Local cubic fit with recovery after saturation (REQ-065)."""

import numpy as np
from scipy.signal import savgol_filter

from evoparam.artifact import apply_mask, remove_artifact
from evoparam.data import TrialSet

###################################################################################################
###################################################################################################

SFREQ = 25000.0


def test_req065_recovery_within_2_ms_and_zeros_while_railed():

    rng = np.random.default_rng(5)
    times = np.arange(int(-0.05 * SFREQ), int(0.1 * SFREQ)) / SFREQ
    n = len(times)
    truth = rng.normal(0, 5, (3, 4, n))
    # Spikes everywhere, including just after the pulse
    wave = -80 * np.hanning(int(0.001 * SFREQ))
    for at in np.arange(int(0.054 * SFREQ), n - 100, int(0.006 * SFREQ)):
        truth[..., at:at + len(wave)] += wave

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
    result = remove_artifact(ts, method="polynomial", window=(0.0, 0.001), quality=False)
    cleaned = result.cleaned.data

    # Zeros while railed, and masked
    assert np.all(cleaned[..., i0:rail_end] == 0.0)
    assert result.cleaned.masks[..., i0:rail_end].all()

    # From 2 ms after the pulse, the cleaned signal matches the truth after the same filter
    half = int(round(0.003 * SFREQ))
    reference = truth - savgol_filter(truth, 2 * half + 1, 3, axis=-1)
    after = slice(i0 + int(0.002 * SFREQ), n - half)
    error = cleaned[..., after] - reference[..., after]
    assert np.sqrt(np.mean(error ** 2)) <= 0.5 * np.std(truth[..., :i0])


def test_polynomial_without_rails_is_a_filter():

    rng = np.random.default_rng(6)
    times = np.arange(-1000, 1000) / SFREQ
    ts = TrialSet(data=rng.normal(0, 1, (1, 1, 2000)), times=times, sfreq=SFREQ, ch_names=["a"])
    result = remove_artifact(ts, method="polynomial", window=(0.0, 0.001), quality=False,
                             baseline=(-0.03, -0.001))
    expected = ts.data - savgol_filter(ts.data, 151, 3, axis=-1)
    assert np.allclose(result.cleaned.data, expected)
    assert result.cleaned.masks is None
