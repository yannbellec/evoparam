"""Models: the fitting engine for evoked waveforms (spec §5.5)."""

from ._settings import FIT_PRESETS, FitSettings, resolve_settings
from .api import (ResponseModel, compare_models, evaluate_fit, fit, fit_response,
                  methods_text)
from .classical import classical_measures
from .labels import label_components

__all__ = ["ResponseModel", "fit", "fit_response", "compare_models", "methods_text",
           "evaluate_fit", "classical_measures", "label_components", "FitSettings",
           "FIT_PRESETS", "resolve_settings"]
