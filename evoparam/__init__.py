# SPDX-License-Identifier: Apache-2.0
"""evoparam - Parameterizing evoked neural responses.

For research use only; not a medical device. Its outputs are not intended for diagnosis or
treatment decisions.
"""

import typing as _typing
from typing import TYPE_CHECKING

from evoparam._typing import Any

from . import errors as errors
from .data import Evoked as Evoked
from .data import Provenance as Provenance
from .data import TrialSet as TrialSet
from .data import load_object as load_object
from .utils import set_log_file as set_log_file  # REQ-304
from .utils import set_log_level as set_log_level  # REQ-304
from .utils import verbose as verbose  # REQ-304
from .version import __version__ as __version__

# The subpackages and the functions of the package root are imported on first use (PEP 562),
# so that ``import evoparam`` stays under 300 ms (REQ-290 P6): importing them all, with
# scipy.stats, took about 0.95 s.
if TYPE_CHECKING:  # the names imported on first use, for type checkers and IDEs
    from . import artifact as artifact
    from . import benchmark as benchmark
    from . import events as events
    from . import existence as existence
    from . import hierarchical as hierarchical
    from . import io as io
    from . import models as models
    from . import plots as plots
    from . import presets as presets
    from . import reports as reports
    from . import shapes as shapes
    from . import sim as sim
    from . import spatial as spatial
    from . import stats as stats
    from . import stream as stream
    from . import tf as tf
    from . import trials as trials
    from .events import detect_events as detect_events
    from .existence import fit as fit
    from .existence import test_response as test_response
    from .io import read as read
    from .models import ResponseModel as ResponseModel
    from .reports import reproduce as reproduce

_SUBPACKAGES = frozenset({
    "artifact", "benchmark", "events", "existence", "hierarchical", "io", "models", "plots",
    "presets", "reports", "shapes", "sim", "spatial", "stats", "stream", "tf",
    "trials",
})
_ROOT_FUNCTIONS = {
    "detect_events": ("events", "detect_events"),
    "fit": ("existence", "fit"),
    "test_response": ("existence", "test_response"),
    "read": ("io", "read"),
    "ResponseModel": ("models", "ResponseModel"),
    "reproduce": ("reports", "reproduce"),
}


def _existence_detector(*args: Any, **kwargs: Any) -> Any:
    """The existence test as the detector of prep.reference_sensitivity (spec §5.4, method 3),
    imported on first call: prep sits below the existence test in the dependency table, so
    the package root registers it."""
    from .existence import existence_detector

    return existence_detector(*args, **kwargs)


from .prep import register_detector as _register_detector  # noqa: E402  (13 ms)

_existence_detector.__name__ = _existence_detector.__qualname__ = "existence_detector"
_register_detector("existence", _existence_detector)

def selftest(*, tolerance: float = 0.001) -> float:
    """Fit one bundled synthetic CCEP and check the recovered N1 latency (REQ-291).

    The CCEP is simulated with :func:`evoparam.sim.simulate_ccep` (``seed=0``, 3 channels,
    10 trials, no artifact) and fitted with the ``ccep`` preset; the latency of the N1 of
    the first channel must be within ``tolerance`` (1 ms) of the simulated one. Needs NumPy
    and SciPy only.

    Returns
    -------
    float
        The absolute latency error, in seconds.

    Raises
    ------
    AssertionError
        When the N1 is not found or is further than ``tolerance`` from the truth.
    """
    import warnings

    from .models import ResponseModel
    from .sim import simulate_ccep

    trials, truth = simulate_ccep(
        n_trials=10, n_channels=3, p_response=1.0, seed=0, artifact=False
    )
    channel = trials.ch_names[0]
    true = next(c for c in truth.components if c["channel"] == channel and c["label"] == "N1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # one waveform (a ResponseFit), so that only NumPy and SciPy are needed
        fit = ResponseModel("ccep", random_state=0).fit_waveform(
            trials.times, trials.data[:, 0].mean(axis=0), trials=trials.data[:, 0]
        )
    found = [c for c in fit.components if 0.009 <= c.latency <= 0.05 and c.amplitude < 0]
    assert found, "selftest: no N1 found on the bundled CCEP"
    n1 = max(found, key=lambda c: abs(c.amplitude))
    error = abs(n1.latency - true["features"]["latency"])
    assert error <= tolerance, f"selftest: N1 latency error {error * 1000:.3f} ms > 1 ms"
    return float(error)


def cite(*, bibtex: bool = False, file: "_typing.TextIO | None" = None) -> str:
    """Print and return how to cite evoparam (REQ-391).

    The software citation of ``CITATION.cff`` with the version, its DOIs when they exist and the
    preferred paper citation when there is one. ``bibtex=True`` gives a BibTeX entry.
    """
    import sys

    from .reports import citation

    text = citation(bibtex=bibtex)
    print(text, file=file or sys.stdout)
    return text


def __getattr__(name: Any) -> Any:
    """Import the subpackages and root functions on first use, and serve the deprecated public
    names (REQ-293) from ``evoparam/_deprecated.py`` with an
    :class:`~evoparam.utils.EvoparamDeprecationWarning`.
    """
    import importlib

    if name in _SUBPACKAGES:
        return importlib.import_module(f"{__name__}.{name}")
    if name in _ROOT_FUNCTIONS:
        package, attribute = _ROOT_FUNCTIONS[name]
        value = getattr(importlib.import_module(f"{__name__}.{package}"), attribute)
        globals()[name] = value
        return value
    from ._deprecated import REGISTRY

    if name in REGISTRY:
        from .utils._deprecation import shim

        return shim(f"evoparam.{name}", *REGISTRY[name])
    raise AttributeError(f"module 'evoparam' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SUBPACKAGES | set(_ROOT_FUNCTIONS))
