# SPDX-License-Identifier: Apache-2.0
"""The calibrated acceptance of a component (models/_calibration.py), the final shape
choice and the resolution check (REQ-105, REQ-106, REQ-111, REQ-112; CHANGELOG
"Calibrated acceptance")."""

from typing import Any

import numpy as np
import pytest
from scipy.stats import chi2

from evoparam import presets
from evoparam.errors import SettingsError
from evoparam.models import ResponseModel
from evoparam.models._calibration import SPECTRAL_SEGMENTS, Calibration, scan_level
from evoparam.models._settings import CALIBRATED, FIXED_DELTA, FitSettings, resolve_settings
from evoparam.models.engine import Whitener, ar_model

from .helpers import BASELINE, TIMES, WINDOW, gauss, model, noise

###################################################################################################
###################################################################################################

ALPHA = 0.04  # the package default
FIXED = 10.0  # spec §5.5.4, every preset
C0 = 18.0  # a scan level
LATENCY_TOL = 0.002  # s


def test_scan_level_without_search_is_the_chi_square_quantile() -> None:
    # no latency or width to search over: P(|Z| >= u) = alpha
    assert scan_level(0.05, 0.0, 0.0) == pytest.approx(chi2.ppf(0.95, 1), rel=1e-8)
    # a larger search space needs a higher level; a lower alpha too
    assert scan_level(0.05, 100.0, 300.0) > scan_level(0.05, 10.0, 30.0)
    assert scan_level(0.01, 100.0, 300.0) > scan_level(0.05, 100.0, 300.0)


def test_inflation_for_an_uncertain_spectrum() -> None:
    assert Calibration.inflate(C0, 0.0) == C0
    # small variance: c0 + c0^2 v / 4
    assert Calibration.inflate(C0, 1e-4) == pytest.approx(C0 + C0**2 * 1e-4 / 4, rel=1e-3)
    assert Calibration.inflate(C0, 0.1) > Calibration.inflate(C0, 0.05) > C0


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
        ALPHA, white, tw, ones, 0.002, 0.3, white.n, a, innovation, acov, int(base.sum())
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


def _ar1(rng: Any, rho: float, n: int) -> Any:
    e = rng.normal(0, 1, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x / np.std(x)


def test_a_changing_spectrum_needs_more_evidence() -> None:
    """A baseline whose spectrum drifts (AR(1) 0.3, then 0.6, then 0.9) widens the threshold
    of a broad component more than a stationary baseline of the same mean spectrum; the
    narrow component's threshold moves less (CHANGELOG "Calibrated acceptance: the change
    of the noise spectrum")."""

    rng = np.random.default_rng(8)
    tw = np.arange(0, 0.5, 0.001)
    columns = np.c_[gauss(0.25, 1, 0.1, tw), gauss(0.25, 1, 0.004, tw), np.ones(len(tw))]
    needed = {}
    for name, z in (
        ("stationary", _ar1(rng, 0.6, 1500)),
        ("drifting", np.r_[_ar1(rng, 0.3, 500), _ar1(rng, 0.6, 500), _ar1(rng, 0.9, 500)]),
    ):
        a, innovation, acov = ar_model(z, return_acov=True)
        white = Whitener(np.ones(len(tw)), a, innovation, np.ones(len(tw), bool), acov=acov)
        ones = white(np.ones((len(tw), 1)))
        cal = Calibration(
            ALPHA, white, tw, ones, 0.002, 0.3, white.n, a, innovation, acov, len(z), baseline=z
        )
        assert len(cal.segments) == SPECTRAL_SEGMENTS
        design = white(columns)
        needed[name] = (cal.required(design, 0) - cal.delta0, cal.required(design, 1) - cal.delta0)
    assert needed["drifting"][0] > needed["stationary"][0]
    assert needed["drifting"][0] > needed["drifting"][1]


def test_calibrated_on_request_and_a_number_is_the_fixed_rule() -> None:
    y = gauss(0.2, -8, 0.04) + noise(np.random.default_rng(1))
    fit = model(delta=CALIBRATED).fit_waveform(TIMES, y)
    assert fit.settings["delta"] == CALIBRATED
    acc = fit.diagnostics["acceptance"]
    assert acc["rule"] == "calibrated"
    assert acc["alpha"] == ALPHA
    assert np.isfinite(acc["scan_level"])
    assert np.isfinite(acc["delta"])
    fixed = model(delta=10).fit_waveform(TIMES, y)
    assert fixed.diagnostics["acceptance"] == {"rule": "fixed", "delta": FIXED}
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
        assert s.delta == FIXED
        assert s.shape_choice == "bic"
        assert not s.resolution_check
    s = resolve_settings(None)
    assert s.delta == FIXED  # the fixed rule stays the package default (DECISIONS §5.3)
    assert s.shape_choice == "hqc"
    assert s.resolution_check
    # None given to the dataclass reads as "not given": the default
    given = FitSettings(delta=None)
    given.validate()
    assert given.delta == FIXED


def test_a_preset_can_ask_for_the_calibrated_acceptance() -> None:
    """``delta="calibrated"`` overrides a preset's fixed 10: the fit records the sentinel,
    reports the calibrated rule, and differs from the preset's own fit only by the
    acceptance; ``delta=None`` keeps the preset's value (keywords left at None)."""

    times, y = TIMES, gauss(0.2, -4, 0.02) + noise(np.random.default_rng(3))
    window = {"fit_window": WINDOW, "baseline": BASELINE}
    own = ResponseModel("erp", **window).fit_waveform(times, y)
    assert own.diagnostics["acceptance"] == {"rule": "fixed", "delta": FIXED}
    assert ResponseModel("erp", delta=None, **window).settings().delta == FIXED
    cal = ResponseModel("erp", delta=CALIBRATED, **window)
    assert cal.settings().delta == CALIBRATED
    assert cal.settings().shape_choice == "bic"  # the preset's other settings stay
    fit = cal.fit_waveform(times, y)
    assert fit.settings["delta"] == CALIBRATED
    assert fit.diagnostics["acceptance"]["rule"] == "calibrated"
    assert fit.diagnostics["acceptance"]["delta"] < FIXED
    assert "calibrated on each waveform's noise model" in fit.methods_text()
    # a preset of its own may set it too
    derived = presets.derive("erp", name="erp_calibrated", delta=CALIBRATED)
    assert presets.from_json(presets.to_json(derived)).delta == CALIBRATED
    again = ResponseModel(derived, **window).fit_waveform(times, y)
    assert again.diagnostics["acceptance"] == fit.diagnostics["acceptance"]
    with pytest.raises(SettingsError):
        presets.derive("erp", delta="adaptive")


def test_settings_validation() -> None:
    with pytest.raises(SettingsError):
        ResponseModel(fit_window=WINDOW, shape_choice="bayes").settings()
    with pytest.raises(SettingsError):
        ResponseModel(fit_window=WINDOW, alpha=1.5).settings()
    with pytest.raises(SettingsError):
        ResponseModel(fit_window=WINDOW, delta=float("nan")).settings()
    with pytest.raises(SettingsError):
        ResponseModel(fit_window=WINDOW, delta="adaptive").settings()


def test_resolution_check_flags_an_unresolvable_pair_only() -> None:
    """Two equal Gaussians 0.5 FWHM apart (one component, ``merged``); 1 FWHM apart (two
    components, neither flagged); one strong narrow component at SNR 400 (resolvable at the
    fixed delta 10 of the default: not flagged; at SNR 200 only with the calibrated
    acceptance, whose threshold is lower)."""

    rng = np.random.default_rng(12)
    close = gauss(0.2, -10, 0.04) + gauss(0.22, -10, 0.04) + noise(rng)
    fit = model(shapes=("gaussian",)).fit_waveform(TIMES, close)
    assert len(fit.components) == 1
    assert "merged:0,1" in fit.warnings
    apart = gauss(0.2, -10, 0.04) + gauss(0.24, -10, 0.04) + noise(rng)
    fit = model(shapes=("gaussian",)).fit_waveform(TIMES, apart)
    assert len(fit.components) == 2  # noqa: PLR2004  (the two Gaussians)
    assert not any(w.startswith("merged:") for w in fit.warnings)
    strong = gauss(0.3, 400, 0.02) + noise(rng)
    fit = model(shapes=("gaussian",)).fit_waveform(TIMES, strong)
    assert len(fit.components) == 1
    assert not any(w.startswith("merged:") for w in fit.warnings)
    off = model(shapes=("gaussian",), resolution_check=False).fit_waveform(TIMES, close)
    assert not any(w.startswith("merged:") for w in off.warnings)


def test_resolution_check_does_not_depend_on_signs() -> None:
    """REQ-003: a component with a neighbour within twice its FWHM is not checked, whatever
    the neighbour's sign, and the sign-flipped waveform gets the same components and
    warnings."""

    rng = np.random.default_rng(0)
    y = gauss(0.2, -8, 0.03) + gauss(0.25, 6, 0.03) + noise(rng)
    fit, flipped = model().fit_waveform(TIMES, y), model().fit_waveform(TIMES, -y)
    assert len(fit.components) == len(flipped.components) == 2  # noqa: PLR2004  (the pair)
    assert not any(w.startswith("merged:") for w in fit.warnings)
    assert fit.warnings == flipped.warnings
    for c, f in zip(fit.components, flipped.components, strict=True):
        assert abs(c.latency - f.latency) < 1e-9  # noqa: PLR2004  (REQ-003 tolerance)
        assert abs(c.amplitude + f.amplitude) < 1e-9  # noqa: PLR2004


def test_final_shape_choice_finds_the_asymmetry() -> None:
    """A clearly skewed component (alpha 4, SNR 20, FWHM 30 ms) is reported with the skewed
    shape under the final shape choice (the default "hqc"), whatever shape the quick fits of
    the search preferred."""

    from evoparam.shapes import get_shape

    shape = get_shape("skewed_gaussian")
    params = np.array([0.25, 0.02, 4.0])
    y = 20 * shape.evaluate(TIMES, params) + noise(np.random.default_rng(5))
    fit = model().fit_waveform(TIMES, y)
    assert [c.shape for c in fit.components] == ["skewed_gaussian"]
    fine = np.arange(0.2, 0.3, 1e-6)
    truth = float(fine[np.argmax(shape.evaluate(fine, params))])  # the mode
    assert abs(fit.components[0].latency - truth) < LATENCY_TOL
