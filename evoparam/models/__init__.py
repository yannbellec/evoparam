# SPDX-License-Identifier: Apache-2.0
"""Models: the fitting engine for evoked waveforms (spec §5.5)."""

from evoparam.shapes import list_backgrounds, list_shapes, register_shape

from ._settings import FIT_PRESETS, FitSettings, resolve_settings
from .api import (
    ResponseModel,
    compare_models,
    evaluate_fit,
    fit,
    fit_response,
    fitted_peak,
    methods_text,
)
from .classical import classical_measures
from .labels import label_components
from .windows import ComponentWindows

__all__ = [
    "ResponseModel",
    "fit",
    "fit_response",
    "compare_models",
    "methods_text",
    "evaluate_fit",
    "fitted_peak",
    "classical_measures",
    "label_components",
    "ComponentWindows",
    "FitSettings",
    "FIT_PRESETS",
    "resolve_settings",
    "list_shapes",
    "list_backgrounds",
    "register_shape",
]
