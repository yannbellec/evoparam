# SPDX-License-Identifier: Apache-2.0
"""Component shapes and backgrounds with analytic Jacobians (spec §5.5.1, §5.5.2).

This module depends on numpy and scipy only.
"""

from evoparam.shapes._base import (ComponentShape, Background, ComponentFeatures,
                                   compute_features, FWHM_PER_SIGMA, ONSET_FRACTION)
from evoparam.shapes.components import (Gaussian, SkewedGaussian, ExGaussian, Gamma,
                                        Biexponential, DampedOscillation, Gabor, Template)
from evoparam.shapes import backgrounds
from evoparam.shapes.backgrounds import (NoBackground, ConstantBackground, LinearBackground,
                                         ExponentialBackground, DoubleExponentialBackground,
                                         PSplineBackground, ArtifactBasisBackground)
from evoparam.shapes.registry import (SHAPES, BACKGROUNDS, register_shape, unregister_shape,
                                      check_shape, get_shape, get_background, list_shapes,
                                      list_backgrounds)
