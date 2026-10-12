# SPDX-License-Identifier: Apache-2.0
"""The shape and background tools that modules built on the engine use (trials, spatial).

Re-exported here because, in the dependency table of spec §4.2, those modules depend on
``models`` and not on ``shapes`` directly.
"""

from evoparam.shapes import FWHM_PER_SIGMA, Template, compute_features, get_background, get_shape
from evoparam.shapes.components import Gaussian, SkewedGaussian, skew_mode

__all__ = [
    "FWHM_PER_SIGMA",
    "Template",
    "compute_features",
    "get_background",
    "get_shape",
    "Gaussian",
    "SkewedGaussian",
    "skew_mode",
]
