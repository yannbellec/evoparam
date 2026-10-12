# SPDX-License-Identifier: Apache-2.0
"""evoparam - Parameterizing evoked neural responses.

For research use only; not a medical device. Its outputs are not intended for diagnosis or
treatment decisions.
"""

import typing as _typing

from . import (
    artifact,
    errors,
    events,
    existence,
    hierarchical,
    io,
    prep,
    presets,
    shapes,
    spatial,
    tf,
    trials,
)
from .data import Evoked, Provenance, TrialSet, load_object
from .events import detect_events
from .existence import fit, test_response
from .io import read
from .models import ResponseModel
from .reports import reproduce
from .version import __version__

# The existence test is the default detector of prep.reference_sensitivity (spec §5.4, method
# 3); prep sits below the existence test in the dependency table, so the package root
# registers it.
prep.register_detector("existence", existence.existence_detector)

from .utils import set_log_file, set_log_level, verbose  # noqa: E402  (REQ-304)


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
    """Print and return how to cite evoparam (REQ-391): the software citation of
    ``CITATION.cff`` with the version, its DOIs when they exist and the preferred paper
    citation when there is one. ``bibtex=True`` gives a BibTeX entry.
    """
    import sys

    from .reports import citation

    text = citation(bibtex=bibtex)
    print(text, file=file or sys.stdout)
    return text


def __getattr__(name):
    """Deprecated public names (REQ-293): served from ``evoparam/_deprecated.py`` with an
    :class:`~evoparam.utils.EvoparamDeprecationWarning`.
    """
    from ._deprecated import REGISTRY

    if name in REGISTRY:
        from .utils._deprecation import shim

        return shim(f"evoparam.{name}", *REGISTRY[name])
    raise AttributeError(f"module 'evoparam' has no attribute {name!r}")
