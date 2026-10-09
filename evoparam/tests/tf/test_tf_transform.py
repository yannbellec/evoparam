"""Transforms and the exact footprint / residual decomposition (REQ-210, REQ-211)."""

import warnings

import numpy as np
import pytest

from evoparam.models import fit as fit_models
from evoparam.tests.tf.helpers import SFREQ, TIMES, ar1, n1, trialset
from evoparam.tf import TRANSFORMS, tf_fit
from evoparam.tf.transforms import (apply_kernels, log_freqs, make_kernels, morse_peak,
                                    morse_reference_modulus, morse_spectrum)


def test_req210_morse_matches_the_analytic_modulus():
    """A cosine gives, far from the edges, half the wavelet's gain at its frequency."""

    n = 4096
    t = np.arange(n) / SFREQ
    freqs = log_freqs(4, 250, 40)
    kernels = make_kernels("morse", freqs, SFREQ, n, gamma=3.0, p2=30.0)
    for f_sig in (10.0, 37.5, 80.0, 150.0):
        coef = apply_kernels(kernels, np.cos(2 * np.pi * f_sig * t))[0]
        for j in range(0, 40, 3):
            if kernels.half_support[j] > 1000:
                continue
            expected = morse_reference_modulus(f_sig, freqs[j], 3.0, 10.0)
            got = np.abs(coef[j, n // 2])
            assert abs(got - expected) <= 1e-6 * max(expected, 1e-3), (f_sig, freqs[j])


def test_morse_peak_and_normalisation():

    beta, gamma = 10.0, 3.0
    peak = morse_peak(gamma, beta)
    omega = np.linspace(0.01, 4, 4001)
    psi = morse_spectrum(omega, gamma, beta)
    assert abs(omega[np.argmax(psi)] - peak) < 1e-3
    assert abs(morse_spectrum(np.array([peak]), gamma, beta)[0] - 2.0) < 1e-12
    assert psi.max() <= 2.0
    assert np.all(morse_spectrum(-omega, gamma, beta) == 0)


def test_morse_chirp_ridge_follows_the_instantaneous_frequency():

    n = 4096
    t = np.arange(n) / SFREQ
    f0, f1 = 20.0, 200.0
    rate = (f1 - f0) / t[-1]
    x = np.cos(2 * np.pi * (f0 * t + 0.5 * rate * t ** 2))
    freqs = log_freqs(10, 300, 80)
    kernels = make_kernels("morse", freqs, SFREQ, n)
    power = np.abs(apply_kernels(kernels, x)[0]) ** 2
    for k in range(800, 3300, 250):
        ridge = freqs[np.argmax(power[:, k])]
        inst = f0 + rate * t[k]
        assert abs(np.log2(ridge / inst)) < 0.1


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_every_transform_has_unit_gain_at_its_frequency(transform):

    n = 2048
    t = np.arange(n) / SFREQ
    freqs = log_freqs(8, 200, 12)
    kernels = make_kernels(transform, freqs, SFREQ, n)
    for j in (4, 8):
        coef = apply_kernels(kernels, 3.0 * np.cos(2 * np.pi * freqs[j] * t))
        power = np.mean(np.abs(coef[:, j, n // 2 - 20:n // 2 + 20]) ** 2, axis=0)
        # multitaper: the higher tapers have lower gain, so the mean power is below 9
        assert 0.3 * 9 <= power.mean() <= 9 * 1.001


def test_req210_cone_of_influence_is_masked():

    rng = np.random.default_rng(0)
    ts = trialset(ar1(rng, (12, len(TIMES))))
    tf = tf_fit(ts, freqs=(4, 250, 20), window=(-0.6, 0.8), baseline=(-0.5, -0.1),
                zones="none", stats="none")
    half = np.array(tf.settings["half_support_s"])
    for j in range(len(tf.freqs)):
        edge = (tf.times - TIMES[0] < half[j] - 1e-9) | (TIMES[-1] - tf.times < half[j] - 1e-9)
        assert np.all(np.isnan(tf.maps["total"][0, j, edge]))
        assert np.all(np.isfinite(tf.maps["total"][0, j, ~edge]))


def _fitted(seed=0, n=20):
    rng = np.random.default_rng(seed)
    ts = trialset(ar1(rng, (n, len(TIMES))) + n1()[None])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ts, fit_models(ts, "ccep")


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_req211_decomposition_is_exact(transform):

    ts, group = _fitted()
    tf = tf_fit(ts, transform=transform, freqs=(4, 250, 20), group_fit=group, zones="none",
                stats="none")
    total = tf.maps["total"]
    parts = tf.maps["evoked_footprint"] + tf.maps["residual"] + tf.maps["cross"]
    ok = np.isfinite(total)
    assert ok.any()
    assert np.all(np.isfinite(parts[ok]))
    assert np.max(np.abs(parts[ok] - total[ok]) / np.maximum(np.abs(total[ok]), 1e-300)) <= 1e-9


def test_power_maps_and_baseline_normalised_maps_are_both_given():
    """Owner decision D12: the exact identity holds for power (REQ-211), the maps of §4.3.4
    are baseline-normalised; both are in TFFit.maps, the normalised ones on the total's
    baseline (evoked level, dB: 10 log10 of the map over the total's baseline mean)."""

    ts, group = _fitted()
    tf = tf_fit(ts, freqs=(4, 250, 20), group_fit=group, zones="none", stats="none",
                baseline_level="evoked", baseline_mode="db")
    for key in ("total", "evoked_footprint", "residual", "cross", "total_norm",
                "footprint_norm", "residual_norm"):
        assert key in tf.maps and tf.maps[key].shape == tf.maps["total"].shape
    base = (tf.times >= -0.5) & (tf.times <= -0.1)
    ref = np.nanmean(tf.maps["total"][0][:, base], axis=1)[:, None]
    ok = np.isfinite(tf.maps["total_norm"][0])
    assert ok.any()
    expected = 10 * np.log10(tf.maps["total"][0] / ref)
    assert np.allclose(tf.maps["total_norm"][0][ok], expected[ok], rtol=1e-9)
    expected = 10 * np.log10(tf.maps["evoked_footprint"][0] / ref)
    fp = np.isfinite(tf.maps["footprint_norm"][0])
    assert np.allclose(tf.maps["footprint_norm"][0][fp], expected[fp], rtol=1e-9)


def test_masked_samples_never_reach_valid_pixels():

    rng = np.random.default_rng(1)
    data = ar1(rng, (15, len(TIMES)))
    masks = np.zeros((15, 1, len(TIMES)), bool)
    artifact = (TIMES >= -0.002) & (TIMES <= 0.008)
    masks[:, 0, artifact] = True
    data[:, artifact] = 1e6
    ts = trialset(data, masks=masks)
    tf = tf_fit(ts, freqs=(4, 250, 20), zones="none", stats="none")
    half = np.array(tf.settings["half_support_s"])
    for j in range(len(tf.freqs)):
        near = (tf.times >= -0.002 - half[j]) & (tf.times <= 0.008 + half[j])
        assert np.all(np.isnan(tf.maps["total"][0, j, near]))
        far = np.isfinite(tf.maps["total"][0, j])
        # the 1e6 artifact does not leak into the valid pixels
        assert np.all(tf.maps["total"][0, j, far] < 1e4)
