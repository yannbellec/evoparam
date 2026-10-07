"""evoparam - Parameterizing evoked neural responses"""

from .version import __version__

from . import errors
from .data import TrialSet, Evoked, Provenance, load_object

from .bands import Bands
from .objs import evoparam, evoparamGroup
from .objs.utils import fit_evoparam_3d

from . import io, detect, clean, param
