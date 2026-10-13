# SPDX-License-Identifier: Apache-2.0
"""Deep equality of containers and results (``evoparam.data._compare``): arrays, tables,
dataclasses and nested values, NaN equal to NaN, missing values equal to each other."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from evoparam.data._compare import DeepEqualMixin, deep_equal
from evoparam.data._table import Table

pytestmark = pytest.mark.core


@dataclass(eq=False)
class Box(DeepEqualMixin):
    value: Any


@dataclass(eq=False)
class Other(DeepEqualMixin):
    value: Any


def test_arrays() -> None:
    assert deep_equal(np.array([1.0, np.nan]), np.array([1.0, np.nan]))
    assert deep_equal(np.array([1 + 1j, np.nan]), np.array([1 + 1j, np.nan]))
    assert not deep_equal(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    assert not deep_equal(np.zeros(2), np.zeros(3))
    assert not deep_equal(np.zeros(2), np.zeros(2, dtype=int))  # another dtype kind
    assert deep_equal(np.array(["a", "b"]), np.array(["a", "b"]))
    assert deep_equal(np.array([1, 2]), [1, 2])


def test_scalars_and_containers() -> None:
    nan = float("nan")
    assert deep_equal(nan, np.float64("nan"))
    assert deep_equal(1, 1.0)
    assert not deep_equal(1.0, 1.5)
    assert deep_equal(True, np.bool_(True))
    assert not deep_equal(True, 1)  # a flag is not a number
    assert not deep_equal(1, True)
    assert deep_equal({"a": [1.0, nan]}, {"a": [1.0, nan]})
    assert not deep_equal({"a": 1}, {"b": 1})
    assert not deep_equal([1, 2], (1, 2))  # list and tuple differ
    assert not deep_equal([1, 2], [1, 2, 3])
    assert deep_equal("x", "x")
    assert not deep_equal("x", "y")
    assert not deep_equal(None, 0)
    assert not deep_equal(0, None)
    marker = object()
    assert deep_equal(marker, marker)


def test_dataclasses() -> None:
    assert Box(np.array([np.nan])) == Box(np.array([np.nan]))
    assert Box(1.0) != Box(2.0)
    assert not deep_equal(Box(1.0), Other(1.0))
    assert Box(1.0).__eq__(Other(1.0)) is NotImplemented
    assert Box(1.0) != 1.0
    with pytest.raises(TypeError, match="unhashable"):
        hash(Box(1.0))


def test_tables() -> None:
    table = Table({"x": [1.0, np.nan], "s": ["a", None]})
    assert deep_equal(table, Table({"x": [1.0, np.nan], "s": ["a", None]}))
    assert not deep_equal(table, Table({"x": [1.0, np.nan]}))
    assert not deep_equal(table, Table({"x": [1.0, 2.0], "s": ["a", None]}))
    assert not deep_equal(table, Table({"x": [1, 2], "s": ["a", None]}))  # int and float
    assert not deep_equal(table, Table({"x": [1.0, np.nan], "s": ["a", "b"]}))
    assert not deep_equal(table, Table({"x": [1.0], "s": ["a"]}))
    assert not deep_equal(table, {"x": [1.0, np.nan], "s": ["a", None]})
    assert not deep_equal({"x": [1.0]}, table)


def test_dataframes() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"x": [1.0, np.nan], "s": ["a", None], "n": [1, 2]})
    assert deep_equal(frame, frame.copy())
    assert not deep_equal(frame, frame[["x", "s"]])
    assert not deep_equal(frame, frame.set_axis([5, 6]))
    assert not deep_equal(frame, frame.astype({"n": float}))
    assert not deep_equal(frame, frame.assign(x=[1.0, 2.0]))
    assert not deep_equal(frame, frame.assign(x=[1.5, np.nan]))
    assert not deep_equal(frame, frame.assign(s=["b", None]))
    assert not deep_equal(frame, frame.iloc[:1])
    assert not deep_equal(frame, Table({"x": [1.0, np.nan]}))
    assert not deep_equal(Table({"x": [1.0, np.nan]}), frame)
