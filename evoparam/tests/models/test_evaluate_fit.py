# SPDX-License-Identifier: Apache-2.0
"""evaluate_fit and fitted_peak give the fitted curve on any time grid.

Defect: the exponential background is ``c0 + c1 exp(-(t - t_ref)/tau)`` with ``t_ref`` the
first sample of the fit window, but the fit did not record ``t_ref`` and evaluate_fit took
the first time of the grid it was given. On a grid starting elsewhere (a fine grid from the
window's nominal start, the whole epoch, a plot axis) the tail was scaled by
``exp((t_grid - t_ref)/tau)``: on ds004080 the curve drawn from 5 ms ran hundreds of uV
below the data, and the trough of ``fitted_peak`` (grid from the window's nominal start,
up to one sample before the first fitted one) moved by up to 0.7 ms (one channel 15 ms).
"""

import warnings

import numpy as np

from evoparam.models import ResponseModel, evaluate_fit, fitted_peak

SFREQ = 2048.0


def _tail_fit():
    rng = np.random.default_rng(3)
    times = np.arange(-0.5, 0.3, 1 / SFREQ)
    t = np.clip(times, 0, None)
    y = (600 * np.exp(-t / 0.004) * (times >= 0)
         - 250 * np.exp(-0.5 * ((times - 0.021) / 0.003) ** 2)
         - 120 * np.exp(-0.5 * ((times - 0.08) / 0.02) ** 2)
         + rng.normal(0, 5, len(times)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ResponseModel("ccep", fit_window=(0.009, 0.25), uncertainty="jacobian",
                            sfreq=SFREQ, tmin=times[0]).fit_waveform(times, y)
    return times, y, fit


def test_background_time_origin_is_recorded():

    times, _, fit = _tail_fit()
    assert fit.status == "ok"
    assert fit.background["name"] == "exponential"
    first = times[(times >= fit.window[0]) & (times <= fit.window[1])][0]
    assert fit.background["t_ref"] == first


def test_evaluate_fit_is_the_fitted_model_on_any_grid():

    times, y, fit = _tail_fit()
    w = (times >= fit.window[0]) & (times <= fit.window[1])
    assert np.allclose(evaluate_fit(fit, times[w]), fit.model, rtol=1e-9, atol=1e-6)
    # a grid that starts elsewhere: the whole epoch, or later in the window
    for grid in (times, times[w][40:]):
        inside = (grid >= times[w][0]) & (grid <= times[w][-1])
        curve = evaluate_fit(fit, grid)
        assert np.all(np.isfinite(curve))
        ref = fit.model[np.isin(times[w], grid[inside])]
        assert np.allclose(curve[inside], ref, rtol=1e-9, atol=1e-6)
    # before the first fitted sample the background keeps its value there
    bg = evaluate_fit(fit, times, components=[])
    assert np.allclose(bg[times < times[w][0]], bg[w][0])


def test_fitted_peak_is_the_trough_of_the_fitted_model():

    times, _, fit = _tail_fit()
    peak = fitted_peak(fit, (0.009, 0.05))
    assert abs(peak["latency"] - 0.021) <= 1 / SFREQ
    w = (times >= fit.window[0]) & (times <= 0.05)
    j = np.argmin(fit.model[:w.sum()])
    assert abs(peak["latency"] - times[w][j]) <= 1 / SFREQ


def test_evaluate_fit_on_a_shared_fit():
    """fit_shared kept the background's coefficients only (no time constant, no time
    origin): evaluate_fit failed on its channels (KeyError 'tau')."""

    from evoparam.spatial import fit_shared
    from evoparam.tests.spatial.helpers import BASELINE, TIMES, shared_channels

    draw, _, _ = shared_channels(0, n=6, snr=6.0, latency=0.03, fwhm=0.02)
    evoked = draw()
    evoked.data[:, TIMES >= 0] += 200 * np.exp(-TIMES[TIMES >= 0] / 0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = fit_shared(evoked, preset="ccep", baseline=BASELINE)
    times, checked = TIMES, 0
    for fit in group.fits:
        if fit.status != "ok" or fit.background.get("name") != "exponential":
            continue
        w = (times >= fit.window[0]) & (times <= fit.window[1])
        assert np.allclose(evaluate_fit(fit, times)[w], fit.model, rtol=1e-7, atol=1e-6)
        checked += 1
    assert checked
