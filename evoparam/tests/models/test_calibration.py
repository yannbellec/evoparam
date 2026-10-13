# SPDX-License-Identifier: Apache-2.0
"""The calibrated acceptance of a component (models/_calibration.py), the final shape
choice and the resolution check (REQ-105, REQ-106, REQ-111, REQ-112; CHANGELOG
"Calibrated acceptance")."""

from typing import Any

import numpy as np
import pytest
from scipy.stats import chi2

from evoparam.models import ResponseModel
from evoparam.models._calibration import Calibration, scan_level
from evoparam.models._settings import FIXED_DELTA, resolve_settings
from evoparam.models.engine import Whitener, ar_model

from .helpers import BASELINE, TIMES, WINDOW, gauss, model, noise

###################################################################################################
###################################################################################################


def test_scan_level_without_search_is_the_chi_square_quantile() -> None:
    # no latency or width to search over: P(|Z| >= u) = alpha
    assert scan_level(0.05, 0.0, 0.0) == pytest.approx(chi2.ppf(0.95, 1), rel=1e-8)
    # a larger search space needs a higher level; a lower alpha too
    assert scan_level(0.05, 100.0, 300.0) > scan_level(0.05, 10.0, 30.0)
    assert scan_level(0.01, 100.0, 300.0) > scan_level(0.05, 100.0, 300.0)


def test_inflation_for_an_uncertain_spectrum() -> None:
    assert Calibration.inflate(18.0, 0.0) == 18.0
    # small variance: c0 + c0^2 v / 4
    assert Calibration.inflate(18.0, 1e-4) == pytest.approx(18.0 + 18.0**2 * 1e-4 / 4, rel=1e-3)
    assert Calibration.inflate(18.0, 0.1) > Calibration.inflate(18.0, 0.05) > 18.0


def _calibration(rho: Any, seed: int = 0) -> Calibration:
    y = noise(np.random.default_rng(seed), rho)
    t = np.asarray(TIMES)
    base = (t >= BASELINE[0]) & (t <= BASELINE[1])
    window = (t >= WINDOW[0]) & (t <= WINDOW[1])
    sigma = 1.4826 * np.median(np.abs(y[base] - np.median(y[base])))
    a, innovation, acov = ar_model(y[base] / sigma, return_acov=True)
    tw = TIMES[window]
    white = Whitener(np.full(len(tw), sigma), a, innovation, np.ones(len(tw), bool), acov=acov)
    ones = white(np.ones((len(tw), 1)))
    return Calibration(
        0.03, white, tw, ones, 0.002, 0.3, white.n, a, innovation, acov, int(base.sum())
    )


def test_white_noise_curvatures_match_the_continuous_formula() -> None:
    """For white noise the scan field of Gaussian templates has the metric
    ``diag(1 / (2 sigma^2), 1 / 2)`` in (latency, log width): ``L2 = n / 2 (1/s1 - 1/s2)``."""

    tw = np.arange(0, 0.58, 0.001)
    white = Whitener(np.ones(len(tw)), [1.0], 1.0, np.ones(len(tw), bool))
    cal = Calibration(0.03, white, tw, None, 0.004, 0.1, len(tw), [1.0], 1.0, None, 0)
    s1, s2 = 4 / 2.3548, 100 / 2.3548
    assert cal.l2 == pytest.approx(len(tw) / 2 * (1 / s1 - 1 / s2), rel=0.05)
    assert cal.cov is None  # white noise: no autoregressive coefficient to be uncertain about
    assert cal.required(white(np.c_[gauss(0.3, 1, 0.02, tw), np.ones(len(tw))]), 0) == (
        pytest.approx(cal.delta0)
    )


def test_coloured_noise_needs_more_evidence() -> None:
    """AR(1) rho = 0.9: the whitened scan is rougher (larger curvatures) and the slow part of
    the spectrum is known to about 40 %, so broad components need a larger fall of BIC."""

    t = np.asarray(TIMES)
    tw = t[(t >= WINDOW[0]) & (t <= WINDOW[1])]
    columns = np.c_[gauss(0.3, 1, 0.08, tw), gauss(0.3, 1, 0.004, tw), np.ones(len(tw))]
    spread = {}
    for rho in (0.0, 0.9):
        cal = _calibration(rho)
        design = cal.white(columns)
        spread[rho] = (cal.l2, cal.c0, cal.required(design, 0) - cal.required(design, 1))
    assert spread[0.9][0] > spread[0.0][0]
    assert spread[0.9][1] > spread[0.0][1]
    # broad minus narrow: large for the red noise, about zero for the white one
    assert spread[0.9][2] > 3 * abs(spread[0.0][2])


def test_delta_none_is_calibrated_and_a_number_is_the_fixed_rule() -> None:
    y = gauss(0.2, -8, 0.04) + noise(np.random.default_rng(1))
    fit = model().fit_waveform(TIMES, y)
    acc = fit.diagnostics["acceptance"]
    assert acc["rule"] == "calibrated"
    assert acc["alpha"] == 0.03
    assert np.isfinite(acc["scan_level"])
    assert np.isfinite(acc["delta"])
    fixed = model(delta=10).fit_waveform(TIMES, y)
    assert fixed.diagnostics["acceptance"] == {"rule": "fixed", "delta": 10.0}
    # not calibrated without the whitened likelihood: the fixed 10 of spec §5.5.4
    assert model(noise_model="neff").fit_waveform(TIMES, y).diagnostics["acceptance"] == {
        "rule": "fixed",
        "delta": FIXED_DELTA,
    }
    assert "calibrated on each waveform's noise model" in fit.methods_text()
    assert "decreased by at least 10" in fixed.methods_text()


def test_presets_keep_the_rule_they_were_validated_with() -> None:
    for name in ("ccep", "erp", "tep", "ecap"):
        s = resolve_settings(name)
        assert s.delta == 10.0
        assert s.shape_choice == "bic"
        assert not s.resolution_check
    s = resolve_settings(None)
    assert s.delta is None
    assert s.shape_choice == "aic"
    assert s.resolution_check


def test_settings_validation() -> None:
    from evoparam.errors import SettingsError

    with pytest.raises(SettingsError):
        ResponseModel(fit_window=WINDOW, shape_choice="bayes").settings()
    with pytest.raises(SettingsError):
        ResponseModel(fit_window=WINDOW, alpha=1.5).settings()
    with pytest.raises(SettingsError):
        ResponseModel(fit_window=WINDOW, delta=float("nan")).settings()


def test_resolution_check_flags_an_unresolvable_pair_only() -> None:
    """Two equal Gaussians 0.5 FWHM apart (one component, ``merged``); 1 FWHM apart (two
    components, neither flagged); one strong narrow component at SNR 200 (resolvable: not
    flagged)."""

    rng = np.random.default_rng(12)
    close = gauss(0.2, -10, 0.04) + gauss(0.22, -10, 0.04) + noise(rng)
    fit = model(shapes=("gaussian",)).fit_waveform(TIMES, close)
    assert len(fit.components) == 1
    assert "merged:0,1" in fit.warnings
    apart = gauss(0.2, -10, 0.04) + gauss(0.24, -10, 0.04) + noise(rng)
    fit = model(shapes=("gaussian",)).fit_waveform(TIMES, apart)
    assert len(fit.components) == 2
    assert not any(w.startswith("merged:") for w in fit.warnings)
    strong = gauss(0.3, 200, 0.02) + noise(rng)
    fit = model(shapes=("gaussian",)).fit_waveform(TIMES, strong)
    assert len(fit.components) == 1
    assert not any(w.startswith("merged:") for w in fit.warnings)
    off = model(shapes=("gaussian",), resolution_check=False).fit_waveform(TIMES, close)
    assert not any(w.startswith("merged:") for w in off.warnings)


def test_final_shape_choice_finds_the_asymmetry() -> None:
    """A clearly skewed component (alpha 4, SNR 20, FWHM 30 ms) is reported with the skewed
    shape under shape_choice="aic", whatever shape the quick fits of the search preferred."""

    from evoparam.shapes import get_shape

    shape = get_shape("skewed_gaussian")
    params = np.array([0.25, 0.02, 4.0])
    y = 20 * shape.evaluate(TIMES, params) + noise(np.random.default_rng(5))
    fit = model().fit_waveform(TIMES, y)
    assert [c.shape for c in fit.components] == ["skewed_gaussian"]
    fine = np.arange(0.2, 0.3, 1e-6)
    truth = float(fine[np.argmax(shape.evaluate(fine, params))])  # the mode
    assert abs(fit.components[0].latency - truth) < 0.002
