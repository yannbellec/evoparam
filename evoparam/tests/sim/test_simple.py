# SPDX-License-Identifier: Apache-2.0
"""gen_response and report: one example response and its fit in a few lines (tutorial 01)."""

import numpy as np
import pytest

from evoparam import report
from evoparam.errors import SettingsError
from evoparam.sim import gen_response

# Largest latency (ms) and relative amplitude error accepted on the example of tutorial 01.
LATENCY_TOLERANCE_MS = 1.0
AMPLITUDE_TOLERANCE = 0.1


def test_gen_response_puts_the_bumps_where_asked() -> None:
    times, signal = gen_response([[25, -150, 10]], background=None, noise=0)
    assert times[0] == pytest.approx(-0.5)
    assert times[-1] == pytest.approx(1.0)
    peak = np.argmin(signal)
    assert times[peak] == pytest.approx(0.025, abs=1 / 2048)
    assert signal[peak] == pytest.approx(-150, rel=1e-3)
    half = times[signal <= signal[peak] / 2]
    assert half[-1] - half[0] == pytest.approx(0.010, abs=2 / 2048)


def test_gen_response_background_and_noise() -> None:
    times, signal = gen_response([], background=(60, 30), noise=0)
    assert signal[times < 0] == pytest.approx(0)
    at = np.searchsorted(times, 0.030)
    assert signal[at] == pytest.approx(60 * np.exp(-times[at] / 0.030))
    _, noisy = gen_response([], background=None, noise=8, seed=0)
    assert noisy.std() == pytest.approx(8, rel=0.15)  # pink noise: the SD of one draw varies
    _, again = gen_response([], background=None, noise=8, seed=0)
    assert np.array_equal(noisy, again)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"components": [[25, -150]]},
        {"components": [[25, -150, 0]]},
        {"components": [], "background": (60, 0)},
        {"components": [], "tmin": 10},
        {"components": [], "noise": -1},
    ],
)
def test_gen_response_refuses_bad_settings(kwargs: dict) -> None:
    with pytest.raises(SettingsError):
        gen_response(**kwargs)


def test_report_finds_the_tutorial_bumps() -> None:
    times, signal = gen_response([[25, -150, 10], [150, -80, 80]], seed=3)
    fit = report(times, signal, plot=False)
    assert fit.units == "uV"
    found = {c.label: c for c in fit.components}
    for label, latency, amplitude in (("N1", 25, -150), ("N2", 150, -80)):
        assert abs(1000 * found[label].latency - latency) <= LATENCY_TOLERANCE_MS * (
            1 if label == "N1" else 5
        )
        assert found[label].amplitude == pytest.approx(amplitude, rel=AMPLITUDE_TOLERANCE)
    text = str(fit)
    assert "RESPONSE MODEL" in text
    assert "N1" in text
    assert "µV" in text
    assert fit._repr_html_().startswith("<pre>")


def test_report_draws_the_fit() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, signal = gen_response([[25, -150, 10]], seed=0)
    report(times, signal)
    assert plt.gcf().axes[0].get_ylabel() == "amplitude (µV)"
    plt.close("all")


def test_report_refuses_mismatched_arrays() -> None:
    with pytest.raises(SettingsError):
        report(np.arange(3), np.arange(4))
