# SPDX-License-Identifier: Apache-2.0
"""Component shapes and backgrounds with analytic Jacobians (spec §5.5.1, §5.5.2).

This module depends on numpy and scipy only.
"""

from evoparam.shapes import backgrounds
from evoparam.shapes._base import (
    FWHM_PER_SIGMA,
    ONSET_FRACTION,
    Background,
    ComponentFeatures,
    ComponentShape,
    compute_features,
)
from evoparam.shapes.backgrounds import (
    ArtifactBasisBackground,
    ConstantBackground,
    DoubleExponentialBackground,
    ExponentialBackground,
    LinearBackground,
    NoBackground,
    PSplineBackground,
)
from evoparam.shapes.components import (
    Biexponential,
    DampedOscillation,
    ExGaussian,
    Gabor,
    Gamma,
    Gaussian,
    SkewedGaussian,
    Template,
)
from evoparam.shapes.registry import (
    BACKGROUNDS,
    SHAPES,
    check_shape,
    get_background,
    get_shape,
    list_backgrounds,
    list_shapes,
    register_shape,
    unregister_shape,
)
