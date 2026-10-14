# SPDX-License-Identifier: Apache-2.0
"""The penalised estimate of an asymmetric component (``shape_choice="penalised"``,
models/_penalised.py; REQ-106 with the default pair)."""

import dataclasses

import numpy as np
import pytest

from evoparam.errors import SettingsError
from evoparam.models import ResponseModel
from evoparam.models._penalised import penalised_component
from evoparam.shapes import get_shape

from .helpers import BASELINE, WINDOW, erp_waveform, match

# Gamma deflections of the REQ-106 design at 12 and 25 samples (where the least-squares mode
# is furthest from the truth's), SNR 10 and 20, on development seeds (not the REQ's).
SEED = 900_000
N_WAVEFORMS = 16
# latency MAE of REQ-106 (samples); measured 0.40 penalised, 0.87 with "hqc"
MAX_MAE_SAMPLES = 0.5

###################################################################################################
###################################################################################################


def _latency_mae(shape_choice: str) -> float:
    errors = []
    for i in range(N_WAVEFORMS):
        times, y, truth = erp_waveform(
            SEED + i, 1, (10, 20)[i % 2], [(12, 25)[(i // 2) % 2]], shape="gamma", span=(0.1, 0.5)
        )
        fit = ResponseModel(
            fit_window=WINDOW, baseline=BASELINE, shape_choice=shape_choice
        ).fit_waveform(times, y)
        errors += [abs(v) * 1000 for v in match(fit, truth)[3]]  # samples at 1 kHz
    return float(np.mean(errors))


def test_penalised_is_a_shape_choice() -> None:

    assert ResponseModel(shape_choice="penalised").settings().shape_choice == "penalised"
    with pytest.raises(SettingsError):
        ResponseModel(shape_choice="penalized").settings()


def test_penalised_mode_of_a_gamma_deflection() -> None:
    """The penalised skewed Gaussian puts the mode of wide gamma deflections closer to the
    truth's than the least-squares one (``"hqc"``, the same shape choice) does, within the
    MAE of REQ-106."""

    penalised, least_squares = _latency_mae("penalised"), _latency_mae("hqc")
    assert penalised <= MAX_MAE_SAMPLES
    assert penalised < least_squares


def test_penalised_component_leaves_symmetric_shapes() -> None:
    """A component without an asymmetry parameter is not refitted (None)."""

    class _Struct:
        shapes = (get_shape("gaussian"),)
        offsets = np.array([0, 2])

    assert penalised_component(_Struct(), np.array([0.3, 0.01]), 0) is None


def test_methods_text_names_the_prior() -> None:
    """The methods text states the SD of the prior the engine uses."""

    from evoparam.data._methods import _fit
    from evoparam.models._penalised import ASYMMETRY_PRIOR_SD

    settings = dataclasses.asdict(ResponseModel().settings())
    assert settings["shape_choice"] == "penalised"
    assert f"SD {ASYMMETRY_PRIOR_SD:g})" in _fit(settings)
    settings["shape_choice"] = "hqc"
    assert "SD" not in _fit(settings)
