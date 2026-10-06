"""Custom error definitions for evoparam."""

class evoparamError(Exception):
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
