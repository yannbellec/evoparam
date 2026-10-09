"""evoparam - Parameterizing evoked neural responses"""

from .version import __version__

from . import errors
from .data import TrialSet, Evoked, Provenance, load_object
from . import shapes

from .bands import Bands
from .objs import evoparam, evoparamGroup
from .objs.utils import fit_evoparam_3d
from .models import ResponseModel
from .existence import fit, test_response

from . import io, events, artifact, prep, param, trials, tf, spatial, hierarchical, presets
from .reports import reproduce
from .events import detect_events

# The existence test is the default detector of prep.reference_sensitivity (spec §5.4, method
# 3); prep sits below the existence test in the dependency table, so the package root
# registers it.
prep.register_detector("existence", existence.existence_detector)
