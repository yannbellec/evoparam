# SPDX-License-Identifier: Apache-2.0
"""Deprecation shims (spec §9.3, REQ-293).

A renamed or removed public name keeps a shim for two minor versions. The shims are listed
in ``evoparam/_deprecated.py`` (``REGISTRY = {old_name: (new_name, since, remove_in)}``);
the package root serves each old name through :func:`shim`, which emits
:class:`EvoparamDeprecationWarning` naming the replacement and the version of removal and
forwards every argument.
"""

import functools
import importlib
from typing import Any

from evoparam.errors import warn

__all__ = ["EvoparamDeprecationWarning", "shim", "resolve", "minor_gap", "version_tuple"]

# Numbers of a version read: major, minor, patch.
_VERSION_PARTS = 3


class EvoparamDeprecationWarning(DeprecationWarning):
    """A public name of evoparam is deprecated (a ``DeprecationWarning`` subclass)."""


def version_tuple(text: object) -> tuple[int, ...]:
    """(major, minor, patch) of a version string (pre-release and local parts dropped)."""
    import re

    parts = []
    for piece in str(text).split("+")[0].split("."):
        match = re.match(r"\d+", piece)
        if not match:
            break
        parts.append(int(match.group()))
        if len(parts) == _VERSION_PARTS:
            break
    return tuple(parts + [0] * (3 - len(parts)))


def minor_gap(since: str, remove_in: str) -> int:
    """Return the number of minor versions between ``since`` and ``remove_in`` (same major)."""
    a, b = version_tuple(since), version_tuple(remove_in)
    if a[0] != b[0]:
        return (b[0] - a[0]) * 1000 + b[1] - a[1]
    return b[1] - a[1]


def resolve(target: str) -> Any:
    """Return the object named ``"package.module:attribute"`` or ``"package.module.attribute"``."""
    if ":" in target:
        module, _, attribute = target.partition(":")
    else:
        module, _, attribute = target.rpartition(".")
    obj = importlib.import_module(module)
    for part in attribute.split("."):
        obj = getattr(obj, part)
    return obj


def message(old: str, new: str, since: str, remove_in: str) -> str:
    return (
        f"{old} is deprecated since evoparam {since} and will be removed in evoparam "
        f"{remove_in}; use {new} instead."
    )


def shim(old: str, new: str, since: str, remove_in: str) -> Any:
    """Return a callable (or the object itself, for a non-callable) that warns and forwards."""
    target = resolve(new)
    text = message(old, new, since, remove_in)
    if not callable(target):
        warn(text, EvoparamDeprecationWarning, stacklevel=3)
        return target

    @functools.wraps(target)
    def forward(*args: Any, **kwargs: Any) -> Any:
        warn(text, EvoparamDeprecationWarning, stacklevel=2)
        return target(*args, **kwargs)

    forward.__deprecated__ = text  # type: ignore[attr-defined]  # read by the tests and docs
    return forward
