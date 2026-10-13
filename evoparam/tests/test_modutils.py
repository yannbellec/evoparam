# SPDX-License-Identifier: Apache-2.0
"""The module utilities kept as is from the inherited code (spec §4.5,
``evoparam/utils/_modutils.py``): optional imports and docstring copies."""

import importlib
from types import ModuleType
from typing import Any

import pytest

from evoparam.utils import _modutils
from evoparam.utils._modutils import (
    check_dependency,
    copy_doc_class,
    copy_doc_func_to_method,
    docs_append_to_section,
    docs_drop_param,
    safe_import,
)

pytestmark = pytest.mark.core


def test_safe_import() -> None:
    assert safe_import("json") is importlib.import_module("json")
    assert safe_import(".decoder", "json") is importlib.import_module("json.decoder")
    assert safe_import("evoparam_no_such_module") is False


def test_safe_import_retries_after_importing_the_parent(monkeypatch: Any) -> None:
    """Python before 3.5.4 raised SystemError for a submodule whose parent was not imported."""
    calls = []

    def flaky(*args: str) -> ModuleType:
        calls.append(args)
        if len(calls) == 1:
            raise SystemError("parent module not loaded")
        return importlib.import_module(*args)

    monkeypatch.setattr(_modutils, "import_module", flaky)
    assert safe_import(".decoder", "json") is importlib.import_module("json.decoder")
    assert calls == [(".decoder", "json"), ("json",), (".decoder", "json")]

    def broken(*args: str) -> ModuleType:
        if args == ("json",):
            raise ImportError("missing")
        raise SystemError("parent module not loaded")

    monkeypatch.setattr(_modutils, "import_module", broken)
    assert safe_import(".decoder", "json") is False


def test_check_dependency() -> None:
    @check_dependency(importlib.import_module("json"), "json")
    def available(x: int) -> int:
        """Double ``x``."""
        return 2 * x

    @check_dependency(False, "nothing")
    def missing() -> None:
        raise AssertionError("not called")

    assert available(3) == 2 * 3
    assert available.__doc__ == "Double ``x``."
    with pytest.raises(ImportError, match="Optional evoparam dependency nothing is required"):
        missing()


DOC = """Summary.

Parameters
----------
obj : Fit
    The object.
n : int
    A number.

Returns
-------
int
"""


def test_docstring_helpers() -> None:
    dropped = docs_drop_param(DOC)
    assert "obj : Fit" not in dropped
    assert "The object." not in dropped
    assert "n : int\n    A number." in dropped
    assert dropped.startswith("Summary.\n\nParameters\n----------\nn : int")
    added = docs_append_to_section(DOC, "Returns", "\n    The number.")
    assert added.endswith("Returns\n-------\nint\n\n    The number.")
    assert added.count("The number.") == 1


def test_docstring_copies() -> None:
    def source(_obj: Any, n: int) -> int:
        return n

    source.__doc__ = DOC

    @copy_doc_func_to_method(source)
    def method(n: int) -> int:
        return n

    assert method.__doc__ == docs_drop_param(DOC)

    class Base:
        __doc__ = DOC

    @copy_doc_class(Base, section="Parameters", add="\nextra : str")
    class Derived(Base):
        pass

    assert Derived.__doc__ == docs_append_to_section(DOC, "Parameters", "\nextra : str")
