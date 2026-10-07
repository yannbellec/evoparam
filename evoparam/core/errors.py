"""Custom error definitions for evoparam (inherited fitting engine).

These classes derive from :class:`evoparam.errors.EvoparamError` so that one
``except EvoparamError`` catches both. They are removed with the inherited
engine at build step S7.
"""

from evoparam.errors import EvoparamError


class evoparamError(EvoparamError):
    """Base class for errors in the evoparam module."""

class FitError(evoparamError):
    """Error for a failure to fit."""

class NoDataError(evoparamError):
    """Error for if data is missing."""

class DataError(evoparamError):
    """Error for if there is a problem with the data."""

class InconsistentDataError(evoparamError):
    """Error for if the data is inconsistent."""

class IncompatibleSettingsError(evoparamError):
    """Error for if settings are incompatible."""

class NoModelError(evoparamError):
    """Error for if the model is not fit."""
