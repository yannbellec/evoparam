# SPDX-License-Identifier: Apache-2.0
"""Logging through the standard library (spec §9.8, REQ-304).

All diagnostics go to ``logging.getLogger("evoparam")`` and its children (``evoparam.models``,
...); the package installs a ``NullHandler`` only (``evoparam.errors``). The ``verbose``
argument of a public function maps to a level for the duration of the call through
:func:`verbose`; :func:`set_log_level` and :func:`set_log_file` are the only global
switches.
"""

import contextlib
import functools
import inspect
import logging
import os
from collections.abc import Callable
from typing import Any, Union

__all__ = ["verbose", "set_log_level", "set_log_file", "LEVELS"]

LOGGER = "evoparam"
# the values ``verbose`` accepts (a logging level as an int is also accepted)
VerboseLevel = None | bool | int | str
# verbose value -> logging level (None: inherit, the logger keeps its level)
LEVELS = {
    None: None,
    False: None,
    True: logging.INFO,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "progress": logging.INFO,
}


def _level(value):
    key = value.lower() if isinstance(value, str) else value
    if isinstance(key, int) and not isinstance(key, bool):
        return key
    if key not in LEVELS:
        from evoparam.errors import SettingsError

        raise SettingsError(
            f"verbose must be one of None, False, True, 'info', 'debug', "
            f"'warning', 'error', 'progress' or a logging level; got {value!r}."
        )
    return LEVELS[key]


class _Verbose(contextlib.ContextDecorator):
    """Set the level of the ``evoparam`` logger for the duration of a block or a call."""

    def __init__(self, level=None):
        self.value = level
        self.level = _level(level)
        self._saved = []

    def __enter__(self):
        logger = logging.getLogger(LOGGER)
        self._saved.append(logger.level)
        if self.level is not None:
            logger.setLevel(self.level)
        return self

    def __exit__(self, *exc):
        logging.getLogger(LOGGER).setLevel(self._saved.pop())
        return False


def verbose(level: VerboseLevel | Callable[..., Any] = None) -> Any:
    """Context manager, decorator, or decorator factory mapping ``verbose`` to a level.

    - ``with verbose("debug"): ...`` sets the ``evoparam`` logger to DEBUG in the block;
    - ``@verbose`` on a function with a ``verbose`` keyword applies the caller's value for
      the duration of each call;
    - ``@verbose("info")`` applies a fixed level to every call.

    ``None``/``False`` inherit, ``True``/``"info"`` → INFO, ``"debug"`` → DEBUG,
    ``"warning"`` → WARNING, ``"error"`` → ERROR, ``"progress"`` → INFO (plus progress bars
    when ``tqdm`` is installed).
    """
    if callable(level) and not isinstance(level, (str, bool)):
        func = level
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            value = bound.arguments.get("verbose", kwargs.get("verbose"))
            with _Verbose(value):
                return func(*args, **kwargs)

        return wrapper
    return _Verbose(level)


def set_log_level(level: VerboseLevel) -> int:
    """Set the level of the ``evoparam`` logger (``verbose`` values or a logging level).

    Returns the previous level.
    """
    logger = logging.getLogger(LOGGER)
    previous = logger.level
    value = _level(level)
    logger.setLevel(logging.NOTSET if value is None else value)
    return previous


def set_log_file(
    path: Union[str, "os.PathLike[str]", None] = None,
    *,
    mode: str = "a",
    fmt: str = "%(asctime)s %(name)s %(levelname)s %(message)s",
) -> logging.FileHandler | None:
    """Write the ``evoparam`` log to ``path`` (a ``FileHandler``).

    ``None`` removes the file handler set by a previous call. Returns the handler (or None).
    """
    logger = logging.getLogger(LOGGER)
    for handler in list(logger.handlers):
        if getattr(handler, "_evoparam_log_file", False):
            logger.removeHandler(handler)
            handler.close()
    if path is None:
        return None
    handler = logging.FileHandler(path, mode=mode, encoding="utf-8")
    handler.setFormatter(logging.Formatter(fmt))
    handler._evoparam_log_file = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return handler
