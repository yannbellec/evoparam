# SPDX-License-Identifier: Apache-2.0
"""REQ-061, second part: bias of the fitted components after cleaning.

Overlay benchmark at 2048 Hz (``helpers.make_overlay``): 20 trials, 4 channels, a negative
component of known onset, and an artifact that ends inside the artifact window ``(a, b) =
(-2, 8) ms`` (recovery tail with a 1 ms time constant: 1.2 uV left at ``b``), three noise
seeds. Every channel average is fitted with :class:`~evoparam.models.ResponseModel` (default
settings) and the largest negative component is compared with the same fit of the
artifact-free data: the bias is what the cleaning adds. Channels where the artifact-free fit
finds no negative component are left out (with the fit window starting at the onset, 8 ms,
the fitter finds it on 7 of 12); a component lost after cleaning counts as an infinite error.

- Template subtraction and the learned basis, components with onset >= b (8, 12 and 20 ms,
  the benchmark onsets of spec §5.3 that are not inside the window), fit window from ``b``:
  median latency error <= 0.5 ms and median relative amplitude error <= 5 %. Measured with
  ``shape_choice="penalised"``: at most 0.074 ms and 2.1 % (onsets 8, 12, 20 ms: 0.043, 0.013,
  0.074 ms and 1.1, 0.26, 2.1 %; the methods agree).
- Plain blanking (``interpolate``: the window replaced and masked, the fit from ``a`` with the
  masked samples at zero weight), onset within 2 ms of ``b`` (6 to 10 ms): the specification
  states an amplitude bias >= 20 %. Measured median relative amplitude errors with
  ``shape_choice="penalised"``: 37 %, lost (component not found on most channels), 66 %, 42 %
  and 23 % (onsets 6, 7, 8, 9, 10 ms). With the earlier least-squares shape choice the 10 ms
  onset gave 15 %, below 20 %.
"""

import functools
import warnings
from typing import Any

import numpy as np
import pytest

from evoparam.artifact import remove_artifact
from evoparam.models import ResponseModel

from .helpers import make_overlay

# Median latency error (samples) and relative amplitude error after cleaning (REQ-061), and
# the amplitude bias that plain blanking is required to show.
MAX_LATENCY_ERROR = 0.5
MAX_AMPLITUDE_ERROR = 0.05
MIN_BLANKING_BIAS = 0.20

###################################################################################################
###################################################################################################

WINDOW = (-0.002, 0.008)
SEEDS = (0, 1, 2)
N_CHANNELS = 4


def _main_component(fit: Any) -> Any:
    """Latency and amplitude of the largest negative component (NaN without one)."""

    negative = [c for c in fit.components if c.polarity < 0]
    if not negative:
        return np.nan, np.nan
    comp = max(negative, key=lambda c: abs(c.amplitude))
    return comp.latency, comp.amplitude


def _fit(ts: Any, fit_window: Any) -> Any:
    model = ResponseModel(fit_window=fit_window, baseline=(-0.5, -0.05))
    avg = ts.average()
    return np.array(
        [
            _main_component(
                model.fit_waveform(
                    avg.times, avg.data[ch], mask=None if avg.masks is None else avg.masks[ch]
                )
            )
            for ch in range(avg.data.shape[0])
        ]
    )


def _overlay(onset: Any, seed: Any) -> Any:
    return make_overlay(onset=onset, seed=seed, n_channels=N_CHANNELS, art_tau=0.001)


@functools.cache
def _truth_fit(onset: Any, seed: Any, fit_window: Any) -> Any:
    return _fit(_overlay(onset, seed)[1], fit_window)


def _errors(onset: Any, fit_window: Any, **kwargs: Any) -> Any:
    """Latency errors (ms) and relative amplitude errors over channels and seeds."""

    lat, amp = [], []
    for seed in SEEDS:
        noisy, _ = _overlay(onset, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cleaned = remove_artifact(noisy, window=WINDOW, **kwargs).cleaned
        got, ref = _fit(cleaned, fit_window), _truth_fit(onset, seed, fit_window)
        found = np.isfinite(ref[:, 0])
        lat += list((got[found, 0] - ref[found, 0]) * 1000)
        amp += list(got[found, 1] / ref[found, 1] - 1)
    # A component found without the artifact but lost after cleaning counts as the worst error.
    lat, amp = np.abs(np.array(lat)), np.abs(np.array(amp))
    return np.where(np.isfinite(lat), lat, np.inf), np.where(np.isfinite(amp), amp, np.inf)


@pytest.mark.parametrize(
    "kwargs",
    [{"method": "template"}, {"method": "template", "template": "local"}, {"method": "learned"}],
    ids=["template", "template-local", "learned"],
)
@pytest.mark.parametrize("onset", [0.008, 0.012, 0.020])
def test_req061_fitted_components_after_cleaning(onset: Any, kwargs: Any) -> None:

    lat, amp = _errors(onset, (WINDOW[1], 0.1), **kwargs)
    assert np.median(lat) <= MAX_LATENCY_ERROR
    assert np.median(amp) <= MAX_AMPLITUDE_ERROR


def test_req061_blanking_amplitude_bias_near_window_end() -> None:

    for onset in (0.006, 0.007, 0.008, 0.009, 0.010):
        _, amp = _errors(onset, (WINDOW[0], 0.1), method="interpolate")
        assert np.median(amp) >= MIN_BLANKING_BIAS, onset
