"""Prep: the light preparation that changes evoked-response numbers (baseline, reference,
trial rejection, resampling, averaging, filters); everything else is left to MNE-Python."""

from .average import AVERAGE_METHODS, average
from .baseline import baseline_correct, train_baseline
from .filters import highpass, lowpass, notch
from .reference import SCHEMES, car_adaptive_select, rereference
from .reject import reject_trials
from .resample import resample
from .sensitivity import (DETECTORS, reference_sensitivity, register_detector,
                          threshold_detector)

__all__ = ["baseline_correct", "train_baseline", "rereference", "car_adaptive_select",
           "reference_sensitivity", "threshold_detector", "register_detector", "DETECTORS",
           "reject_trials", "resample", "average", "highpass", "lowpass", "notch", "SCHEMES",
           "AVERAGE_METHODS"]
