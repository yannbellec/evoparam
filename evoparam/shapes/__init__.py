# SPDX-License-Identifier: Apache-2.0
"""Component shapes and backgrounds with analytic Jacobians (spec §5.5.1, §5.5.2).

This module depends on numpy and scipy only.
"""

from evoparam.shapes import backgrounds as backgrounds
from evoparam.shapes._base import FWHM_PER_SIGMA as FWHM_PER_SIGMA
from evoparam.shapes._base import ONSET_FRACTION as ONSET_FRACTION
from evoparam.shapes._base import Background as Background
from evoparam.shapes._base import ComponentFeatures as ComponentFeatures
from evoparam.shapes._base import ComponentShape as ComponentShape
from evoparam.shapes._base import compute_features as compute_features
from evoparam.shapes.backgrounds import ArtifactBasisBackground as ArtifactBasisBackground
from evoparam.shapes.backgrounds import ConstantBackground as ConstantBackground
from evoparam.shapes.backgrounds import DoubleExponentialBackground as DoubleExponentialBackground
from evoparam.shapes.backgrounds import ExponentialBackground as ExponentialBackground
from evoparam.shapes.backgrounds import LinearBackground as LinearBackground
from evoparam.shapes.backgrounds import NoBackground as NoBackground
from evoparam.shapes.backgrounds import PSplineBackground as PSplineBackground
from evoparam.shapes.components import Biexponential as Biexponential
from evoparam.shapes.components import DampedOscillation as DampedOscillation
from evoparam.shapes.components import ExGaussian as ExGaussian
from evoparam.shapes.components import Gabor as Gabor
from evoparam.shapes.components import Gamma as Gamma
from evoparam.shapes.components import Gaussian as Gaussian
from evoparam.shapes.components import SkewedGaussian as SkewedGaussian
from evoparam.shapes.components import Template as Template
from evoparam.shapes.registry import BACKGROUNDS as BACKGROUNDS
from evoparam.shapes.registry import SHAPES as SHAPES
from evoparam.shapes.registry import check_shape as check_shape
from evoparam.shapes.registry import get_background as get_background
from evoparam.shapes.registry import get_shape as get_shape
from evoparam.shapes.registry import list_backgrounds as list_backgrounds
from evoparam.shapes.registry import list_shapes as list_shapes
from evoparam.shapes.registry import register_shape as register_shape
from evoparam.shapes.registry import unregister_shape as unregister_shape
