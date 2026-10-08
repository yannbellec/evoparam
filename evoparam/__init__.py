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

from . import io, events, artifact, prep, param, trials, tf, spatial
from .events import detect_events
