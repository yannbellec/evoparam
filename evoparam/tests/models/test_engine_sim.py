"""The engine on simulations with known components (REQ-104 to REQ-107, REQ-110 to REQ-112,
REQ-116, REQ-121, REQ-122)."""

import numpy as np
import pytest

from .helpers import TIMES, gauss, match, model, noise, random_components

###################################################################################################
###################################################################################################

def _waveforms(n, seed=0, **kwargs):
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        truth = random_components(rng, **kwargs)
        yield sum(gauss(*c) for c in truth) + noise(rng), truth


def test_req105_recall_and_precision():

    tp = fn = fp = 0
    for y, truth in _waveforms(60):
        a, b, c, _, _ = match(model().fit_waveform(TIMES, y), truth)
        tp, fn, fp = tp + a, fn + b, fp + c
    assert tp / (tp + fn) >= 0.95
    assert tp / (tp + fp) >= 0.95


def test_req106_accuracy_at_high_snr():

    lat, amp = [], []
    for y, truth in _waveforms(40, seed=100, snr=(10, 15)):
        _, _, _, l, a = match(model(shapes=("gaussian",)).fit_waveform(TIMES, y), truth)
        lat += l
        amp += a
    assert np.mean(np.abs(lat)) * 1000 <= 0.5           # samples at 1 kHz
    assert abs(np.median(amp) - 1) <= 0.05


def test_req104_detection_threshold_changes_no_decision():

    same = 0
    total = 0
    for y, truth in _waveforms(20, seed=200):
        counts = [len(model(detection_threshold=d).fit_waveform(TIMES, y).components)
                  for d in (1.0, 2.0, 3.0)]
        same += len(set(counts)) == 1
        total += 1
    assert same / total >= 0.95


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_req107_req112_no_component_in_pure_noise(rho):

    rate = np.mean([model().fit_waveform(TIMES, noise(np.random.default_rng(i), rho)).status
                    == "no_response" for i in range(40)])
    assert rate >= 0.95


def test_req107_whitening_is_what_controls_false_components():

    accepted = [model(noise_model="white").fit_waveform(TIMES, noise(np.random.default_rng(i),
                                                                     0.9)).status == "ok"
                for i in range(20)]
    assert np.mean(accepted) >= 0.3


def test_req110_edge_components():

    hits = 0
    for seed in range(20):
        rng = np.random.default_rng(300 + seed)
        latency = 0.02 + rng.uniform(0, 0.03)                  # within one FWHM of the start
        y = gauss(latency, -8, 0.03) + noise(rng)
        fit = model().fit_waveform(TIMES, y)
        hits += any(abs(c.latency - latency) <= 0.015 and c.amplitude < 0
                    for c in fit.components)
    assert hits / 20 >= 0.95


def test_req111_resolution_and_req121_conditioning():

    two = 0
    for seed in range(10):
        rng = np.random.default_rng(400 + seed)
        y = gauss(0.2, -10, 0.04) + gauss(0.24, -10, 0.04) + noise(rng)
        two += len(model(shapes=("gaussian",)).fit_waveform(TIMES, y).components) == 2
    assert two / 10 >= 0.9
    flagged = 0
    for seed in range(10):
        rng = np.random.default_rng(500 + seed)
        y = gauss(0.2, -10, 0.04) + gauss(0.212, -10, 0.04) + noise(rng)
        # Two components forced into a window not much wider than the pair: the fit cannot
        # separate them, and must say so
        fit = model(shapes=("gaussian",), selection="fixed", n_components=2,
                    fit_window=(0.17, 0.24), edge_extension=0.0).fit_waveform(TIMES, y)
        flagged += "ill_conditioned" in fit.warnings
        assert fit.diagnostics["flagged_pairs"]
    assert flagged == 10


def test_req116_coverage_of_jacobian_intervals():

    inside_lat = inside_amp = n = 0
    for seed in range(60):
        rng = np.random.default_rng(600 + seed)
        y = gauss(0.2, -8, 0.04) + noise(rng)
        fit = model(shapes=("gaussian",)).fit_waveform(TIMES, y)
        if len(fit.components) != 1:
            continue
        comp = fit.components[0]
        n += 1
        inside_lat += comp.ci["latency"][0] <= 0.2 <= comp.ci["latency"][1]
        inside_amp += comp.ci["amplitude"][0] <= -8 <= comp.ci["amplitude"][1]
    assert 0.85 <= inside_lat / n <= 1.0
    assert 0.85 <= inside_amp / n <= 1.0


def test_req122_fixed_and_cross_validated_selection():

    rng = np.random.default_rng(7)
    y = gauss(0.2, -8, 0.04) + noise(rng)
    fit = model(selection="fixed", n_components=3).fit_waveform(TIMES, y)
    assert len(fit.components) == 3
    good = 0
    for seed in range(10):
        rng = np.random.default_rng(700 + seed)
        sig = gauss(0.1, -6, 0.028) + gauss(0.3, 5, 0.07)
        trials = sig + rng.normal(0, 5, (40, len(TIMES)))
        m = model(shapes=("gaussian",), selection="cv", max_components=4, random_state=0)
        good += len(m.fit_waveform(TIMES, trials.mean(0), trials=trials).components) == 2
    assert good / 10 >= 0.85
