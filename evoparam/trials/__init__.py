"""Trials: trial-by-trial parameters, latency jitter, realignment and covariates (spec §5.7)."""

from .api import (MODES, TrialFitSet, dose_response, fit_trials, jitter,
                  prestimulus_dependence, realign)

__all__ = ["fit_trials", "TrialFitSet", "jitter", "realign", "dose_response",
           "prestimulus_dependence", "MODES"]
