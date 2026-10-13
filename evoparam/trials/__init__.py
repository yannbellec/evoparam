# SPDX-License-Identifier: Apache-2.0
"""Trials: trial-by-trial parameters, latency jitter, realignment and covariates (spec §5.7)."""

from .api import (
    COVARIATES,
    MODES,
    TrialFitSet,
    dose_response,
    fit_trials,
    jitter,
    prestimulus_dependence,
    realign,
    state_dependence,
)
from .estimator import TrialFitter
from .online import FixedShapeModel

__all__ = [
    "TrialFitter",
    "FixedShapeModel",
    "fit_trials",
    "TrialFitSet",
    "jitter",
    "realign",
    "dose_response",
    "prestimulus_dependence",
    "state_dependence",
    "MODES",
    "COVARIATES",
]
