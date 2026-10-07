"""Exception and warning hierarchy for evoparam.

Every error raised by the package derives from :class:`EvoparamError`. A
function never returns a result while hiding an exception: when a failure on
one channel or one trial is acceptable, the result records ``status="failed"``
with a message instead, and a WARNING log record names the unit.

This module imports nothing from the rest of the package, so every module may
import it.
"""

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
