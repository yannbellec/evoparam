# SPDX-License-Identifier: Apache-2.0
"""The failure of one unit of work: a channel or a trial (spec REQ-307 ii).

Where the failure of one channel or one trial is acceptable (``fit_group``, ``fit_trials``,
``test_response``, ``remove_artifact``), ``on_error="record"`` (the default) keeps the
unit's result with ``status="failed"`` and the exception in ``message``, and logs exactly
one WARNING record naming the unit; ``on_error="raise"`` raises the first
:class:`~evoparam.errors.FitError` instead. The errors of the data or of the settings of the
call are never recorded as the failure of a unit: they propagate (REQ-307 i).
"""

import logging

from evoparam.errors import (
    ArtifactError,
    DataError,
    EventError,
    EvoparamVersionError,
    FitError,
    ReaderError,
    SettingsError,
)

__all__ = ["ON_ERROR", "PROPAGATE", "check_on_error", "describe", "fit_error", "record_failure"]

ON_ERROR = ("record", "raise")

#: Errors of the data or of the settings of the call (REQ-307 i, the documented
#: ``ArtifactError`` and ``EventError`` of the calls, and the ``ImportError`` of a missing
#: optional package): they always propagate.
PROPAGATE = (
    ImportError,
    DataError,
    SettingsError,
    ReaderError,
    EvoparamVersionError,
    ArtifactError,
    EventError,
)


def check_on_error(on_error: str) -> None:
    """Raise a :class:`SettingsError` unless ``on_error`` is ``"record"`` or ``"raise"``."""
    if on_error not in ON_ERROR:
        raise SettingsError(f"on_error must be 'record' or 'raise', got {on_error!r}.")


def describe(exc: BaseException) -> str:
    """Return the ``message`` of a failed unit: the exception's class and text.

    Examples
    --------
    >>> describe(ValueError("no convergence"))
    'ValueError: no convergence'
    """
    return f"{type(exc).__name__}: {exc}"


def fit_error(what: str, unit: str, exc: BaseException) -> FitError:
    """Return the :class:`FitError` that ``on_error="raise"`` raises (``raise ... from``)."""
    return FitError(f"{what} failed on {unit}: {describe(exc)}")


def record_failure(logger: logging.Logger, what: str, unit: str, exc: BaseException) -> str:
    """Log the one WARNING record of a failed unit and return its ``message``."""
    message = describe(exc)
    logger.warning("%s failed on %s: %s", what, unit, message)
    return message
