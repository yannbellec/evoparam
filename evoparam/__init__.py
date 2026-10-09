"""evoparam - Parameterizing evoked neural responses.

For research use only; not a medical device. Its outputs are not intended for diagnosis or
treatment decisions.
"""

from .version import __version__

from . import errors
from .data import TrialSet, Evoked, Provenance, load_object
from . import shapes

from .models import ResponseModel
from .existence import fit, test_response
from .io import read

from . import io, events, artifact, prep, existence, trials, tf, spatial, hierarchical, presets
from .reports import reproduce
from .events import detect_events

# The existence test is the default detector of prep.reference_sensitivity (spec §5.4, method
# 3); prep sits below the existence test in the dependency table, so the package root
# registers it.
prep.register_detector("existence", existence.existence_detector)
