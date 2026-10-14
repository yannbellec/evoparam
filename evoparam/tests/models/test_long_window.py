# SPDX-License-Identifier: Apache-2.0
"""Decimated search of a long fit window (spec §5.5.9)."""

from typing import Any

import numpy as np

from evoparam.models import ResponseModel
from evoparam.models import engine as eng

# Sampling rate (Hz), epoch (s), fit window (s) and baseline (s): a 12 s window at 2 kHz
# holds 24 000 samples, more than the 20 000 of the search.
SFREQ = 2000.0
EPOCH = (-2.0, 13.0)
WINDOW = (0.0, 12.0)
BASELINE = (-2.0, 0.0)
# Three slow components: (latency s, amplitude, FWHM s), in noise of SD 1.
TRUTH = ((1.5, -8.0, 0.4), (4.0, 6.0, 1.0), (8.0, -5.0, 1.5))
# Spec §5.5.9: the window length (s) and rate (Hz) above which the search is decimated, and
# the most samples it runs on.
LONG_WINDOW = 10.0
HIGH_RATE = 1000.0
SEARCH_MAX = 20_000
# Latency tolerance (s) against the truth; agreement with the search at full resolution, in
# standard errors (two ends of the optimiser from different starts) and relative on BIC.
LATENCY_TOLERANCE = 0.02
SAME_SE = 1e-3
SAME_BIC = 1e-6
# Decimation factors: a 12 s window at 2 kHz (24 000 samples) and a 30 s one (60 000).
FACTOR_12S = 2
FACTOR_30S = 3

###################################################################################################
###################################################################################################


def _waveform(seed: int = 0) -> tuple[Any, Any]:
    times = np.arange(EPOCH[0], EPOCH[1], 1 / SFREQ)
    y = sum(a * np.exp(-0.5 * ((times - m) / (w / 2.3548)) ** 2) for m, a, w in TRUTH)
    return times, y + np.random.default_rng(seed).normal(0, 1, len(times))


def _model(**kwargs: Any) -> ResponseModel:
    return ResponseModel(
        shapes="gaussian",
        background="constant",
        fit_window=kwargs.pop("fit_window", WINDOW),
        baseline=BASELINE,
        min_fwhm=0.01,
        max_fwhm=3.0,
        random_state=0,
        **kwargs,
    )


def test_long_window_is_searched_decimated_and_refitted_at_full_resolution(
    monkeypatch: Any,
) -> None:
    """A window longer than 10 s at 1 kHz or more: the stepwise search runs on at most
    20 000 samples, the model is refitted at full resolution, and "decimated_search" is noted
    in the provenance."""

    times, y = _waveform()
    n_window = int(np.count_nonzero((times >= WINDOW[0]) & (times <= WINDOW[1])))
    assert WINDOW[1] - WINDOW[0] > LONG_WINDOW
    assert SFREQ >= HIGH_RATE
    assert n_window > SEARCH_MAX
    fit = _model().fit_waveform(times, y)
    note = fit.provenance.steps[-1]["decimated_search"]
    assert note["n_samples"] <= SEARCH_MAX < note["n_samples_full"] == n_window
    assert note["factor"] == FACTOR_12S
    assert fit.diagnostics["decimated_search"] == note
    # the result is at full resolution: model and residual on every sample of the window
    assert len(fit.model) == len(fit.residual) == n_window
    assert fit.status == "ok"
    assert len(fit.components) == len(TRUTH)
    for comp, (latency, amplitude, _) in zip(fit.components, TRUTH, strict=True):
        assert abs(comp.latency - latency) <= LATENCY_TOLERANCE
        assert np.sign(comp.amplitude) == np.sign(amplitude)
    # the refit reaches the optimum the search at full resolution reaches
    monkeypatch.setattr(eng, "DECIMATE_WINDOW", np.inf)
    full = _model().fit_waveform(times, y)
    assert "decimated_search" not in full.provenance.steps[-1]
    assert len(full.components) == len(fit.components)
    for a, b in zip(fit.components, full.components, strict=True):
        for name in ("latency", "amplitude", "fwhm"):
            assert abs(getattr(a, name) - getattr(b, name)) <= SAME_SE * b.se[name], name
    assert np.isclose(fit.gof.bic, full.gof.bic, rtol=SAME_BIC)


def test_short_or_slow_windows_are_not_decimated() -> None:
    """The rule of spec §5.5.9 applies only above 10 s and at 1 kHz or more."""

    times, _ = _waveform()
    settings = _model().settings()
    assert eng.decimation_factor(times, settings, SFREQ) == FACTOR_12S
    short = _model(fit_window=(0.0, 9.5)).settings()
    assert eng.decimation_factor(times, short, SFREQ) == 1
    assert eng.decimation_factor(times[::4], settings, SFREQ / 4) == 1
    # 15 s at 1 kHz: 15 000 samples, already at most 20 000
    slow = np.arange(-2.0, 18.0, 1e-3)
    assert eng.decimation_factor(slow, _model(fit_window=(0.0, 15.0)).settings(), 1000.0) == 1
    # 30 s at 2 kHz: 60 000 samples, factor 3
    wide = np.arange(-2.0, 31.0, 1 / SFREQ)
    factor = eng.decimation_factor(wide, _model(fit_window=(0.0, 30.0)).settings(), SFREQ)
    assert factor == FACTOR_30S
    assert eng._decimated_count(wide, (0.0, 30.0), factor) <= SEARCH_MAX


def test_decimated_copy_leaves_masked_samples_out() -> None:
    """The block means of the decimated copy ignore masked samples; an all-masked block
    is masked."""

    times = np.arange(8) / SFREQ
    y = np.arange(8.0)
    mask = np.array([False, True, True, True, False, False, False, False])
    t_d, y_d, m_d, sem_d, ref_d = eng._decimate(times, y, mask, None, None, 2)
    assert np.allclose(t_d, times.reshape(-1, 2).mean(1))
    assert y_d[0] == 0.0
    assert m_d.tolist() == [False, True, False, False]
    assert y_d[2:].tolist() == [4.5, 6.5]
    assert sem_d is None
    assert ref_d is None
