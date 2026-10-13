# SPDX-License-Identifier: Apache-2.0
"""Exception and warning hierarchy for evoparam.

Every error raised by the package derives from :class:`EvoparamError`. A
function never returns a result while hiding an exception: when a failure on
one channel or one trial is acceptable, the result records ``status="failed"``
with a message instead, and a WARNING log record names the unit.

This module imports nothing from the rest of the package, so every module may
import it.
"""

import logging
import os
import sys
import warnings

# The package logs to "evoparam" and installs a NullHandler only (spec §9.8, REQ-304): the
# application decides where records go, and nothing is printed twice when it does not.
logging.getLogger("evoparam").addHandler(logging.NullHandler())

__all__ = [
    "EvoparamError",
    "DataError",
    "ReaderError",
    "EventError",
    "ArtifactError",
    "FitError",
    "SettingsError",
    "EvoparamVersionError",
    "NoResponseWarning",
    "ConvergenceWarning",
    "warn",
]


class EvoparamError(Exception):
    """Base class for all errors raised by evoparam."""


class DataError(EvoparamError, ValueError):
    """Malformed container, unit mismatch, NaN outside masks, failed checksum."""


class ReaderError(EvoparamError):
    """Unreadable file, missing sidecar, refused remote range request."""


class EventError(EvoparamError):
    """No pulses found, inconsistent train structure."""


class ArtifactError(EvoparamError):
    """Window outside the epoch, template cannot be estimated."""


class FitError(EvoparamError):
    """Optimiser failure after all restarts. Never silently caught."""


class SettingsError(EvoparamError, ValueError):
    """Invalid setting or preset override (for example min_fwhm > max_fwhm)."""


class EvoparamVersionError(EvoparamError):
    """A saved container or preset version cannot be read by this version."""


class NoResponseWarning(UserWarning):
    """The existence test is negative and a fit was still requested."""


class ConvergenceWarning(UserWarning):
    """Trial fits or MCMC chains did not converge."""


# Files whose frames ``warn`` does not count in ``stacklevel`` (Python >= 3.12): the
# ``verbose`` decorator and the ``n_jobs`` helper wrap the public functions. The prefixes
# stop before the extension: CPython's C implementation compares them with the file name
# without its last character.
_SKIP = tuple(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", name)
    for name in ("_logging.", "_parallel.")
)


def warn(
    message: str | Warning, category: type[Warning] = UserWarning, stacklevel: int = 2
) -> None:
    """Issue a warning and log it on the ``evoparam`` logger at WARNING level.

    Every warning of the package goes through this function, so that it is also logged
    (REQ-005): a pipeline run without a console still keeps its warnings in the log.
    ``stacklevel`` counts from the caller, as for :func:`warnings.warn`.
    """
    text = str(message)
    name = category.__name__ if isinstance(category, type) else type(message).__name__
    logging.getLogger("evoparam").warning("%s: %s", name, text)
    if sys.version_info >= (3, 12):
        # the frames of the verbose decorator and of the n_jobs helper are not counted, so
        # that a warning points to the same line as without them
        warnings.warn(message, category, stacklevel=stacklevel + 1, skip_file_prefixes=_SKIP)
    else:
        warnings.warn(message, category, stacklevel=stacklevel + 1)
