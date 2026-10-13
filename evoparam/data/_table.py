# SPDX-License-Identifier: Apache-2.0
"""Tables of the core without pandas (spec REQ-010).

The core functions (``detect_events``, ``remove_artifact``, ``fit``, ``test_response``,
``fit_trials``, saving and loading results) run with NumPy and SciPy only. Their tables
(event tables, quality tables, the index of a :class:`~evoparam.data.GroupFit`, trial
tables) are pandas DataFrames when pandas is installed (extra ``data``), with the same
columns and dtypes as always, and a :class:`Table` otherwise: named NumPy columns of equal
length, with the part of the DataFrame interface that the package itself uses (columns,
length, column access and assignment, row selection, ``iloc``, ``copy``, ``attrs``,
``to_dict``). :func:`new_table` makes the right one; :meth:`Table.to_pandas` converts.
"""

import importlib.util
from collections import namedtuple
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from evoparam.utils._optional import require

_SortKind = Literal["stable", "quicksort", "mergesort", "heapsort"]

__all__ = [
    "Column",
    "Table",
    "as_text",
    "concat_tables",
    "has_pandas",
    "is_missing",
    "is_table",
    "new_table",
    "row_labels",
]


def has_pandas() -> bool:
    """Return True when pandas can be imported (the extra ``data``)."""
    try:
        return importlib.util.find_spec("pandas") is not None
    except (ImportError, ValueError):  # an import hook that hides it, or a broken spec
        return False


def is_table(obj: object) -> bool:
    """Return True for a pandas DataFrame or a :class:`Table`."""
    kind = type(obj)
    return isinstance(obj, Table) or (
        kind.__name__ == "DataFrame" and kind.__module__.startswith("pandas")
    )


def is_missing(value: Any) -> bool:
    """Return True for a missing value: None, NaN, ``pd.NA`` or ``pd.NaT``."""
    if value is None or type(value).__name__ in ("NAType", "NaTType"):
        return True
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def as_text(values: Any) -> NDArray[Any]:
    """Values as text, a missing value (None, NaN, ``pd.NA``) as ``""``.

    The same as ``series.fillna("").astype(str).to_numpy()`` of pandas, for a column of a
    DataFrame or of a :class:`Table`.

    Examples
    --------
    >>> as_text(["A1-A2", None, float("nan"), 3]).tolist()
    ['A1-A2', '', '', '3']
    """
    return np.array(
        ["" if is_missing(v) else str(v) for v in np.asarray(values, dtype=object)], dtype=object
    )


def row_labels(table: Any, columns: Sequence[str]) -> NDArray[Any]:
    """Return the values of ``columns`` of each row joined by ``/`` (object array of str).

    The same as ``table[columns].astype(str).agg("/".join, axis=1).to_numpy()`` of pandas.

    Examples
    --------
    >>> table = Table({"site": ["A1-A2", "B1-B2"], "current": [1.0, 2.0]})
    >>> row_labels(table, ["site", "current"]).tolist()
    ['A1-A2/1.0', 'B1-B2/2.0']
    """
    values = [np.asarray(table[name], dtype=object) for name in columns]
    return np.array(
        ["/".join(str(column[row]) for column in values) for row in range(len(table))],
        dtype=object,
    )


class Column(np.ndarray[Any, Any]):  # a NumPy array subclass
    """One column of a :class:`Table`: a 1-D NumPy array with ``to_numpy`` and ``iloc``.

    Examples
    --------
    >>> col = Table({"a": [3, 1, 2]})["a"]
    >>> col.to_numpy().tolist(), int(col.iloc[0])
    ([3, 1, 2], 3)
    """

    def __new__(cls, values: Any) -> "Column":
        """Wrap ``values`` (any 1-D array-like) as a column."""
        return np.asarray(values).view(cls)

    def to_numpy(self, dtype: Any = None, copy: bool = False) -> NDArray[Any]:
        """Return the values as a plain NumPy array (a copy when ``copy`` or ``dtype``)."""
        plain = self.view(np.ndarray)
        if dtype is not None:
            return plain.astype(dtype, copy=copy)
        return plain.copy() if copy else plain

    @property
    def iloc(self) -> NDArray[Any]:
        """Positional access, as ``Series.iloc``."""
        return self.view(np.ndarray)

    @property
    def values(self) -> NDArray[Any]:
        """The values as a plain NumPy array, as ``Series.values``."""
        return self.view(np.ndarray)

    def unique(self) -> NDArray[Any]:
        """Distinct values in order of first appearance, as ``Series.unique``."""
        _, first = np.unique(self.view(np.ndarray), return_index=True)
        return self.view(np.ndarray)[np.sort(first)]


def _scalar(values: Any) -> bool:
    return np.ndim(values) == 0 and not isinstance(values, (list, tuple))


def _as_column(values: Any, n_rows: int | None) -> NDArray[Any]:
    if _scalar(values):
        if n_rows is None:
            raise ValueError("a scalar column needs the number of rows")
        array = np.full(n_rows, values, dtype=None if not isinstance(values, str) else object)
    else:
        array = np.asarray(values)
        if array.dtype.kind == "U":  # text as Python strings, as pandas holds it
            # from the values themselves: NumPy would turn a NaN or None among text into "nan"
            array = np.empty(len(array), dtype=object)
            array[:] = list(values)
        if array.ndim != 1:
            raise ValueError(f"a column must be 1-D, got shape {array.shape}")
        if n_rows is not None and len(array) != n_rows:
            raise ValueError(f"a column of {len(array)} values for {n_rows} rows")
    return array


class _ILoc:
    """Row selection by position: ``table.iloc[rows]`` (an int gives the row as a dict)."""

    def __init__(self, table: "Table") -> None:
        self._table = table

    def __getitem__(self, rows: Any) -> Any:
        if isinstance(rows, (int, np.integer)):
            return {name: values[int(rows)] for name, values in self._table._data.items()}
        return self._table._take(rows)


class Table:
    """Named columns of equal length (the pandas-free table of the core, REQ-010).

    Parameters
    ----------
    data : mapping of column name to 1-D values, or a sequence of row mappings
    columns : sequence of str, optional
        Column order (and the columns of an empty table).

    Examples
    --------
    >>> t = Table({"onset": [0.5, 1.0, 1.5], "site": ["A1-A2", "B1-B2", "A1-A2"]})
    >>> len(t), list(t.columns)
    (3, ['onset', 'site'])
    >>> t[t["onset"] > 0.7]["site"].tolist()
    ['B1-B2', 'A1-A2']
    """

    def __init__(self, data: Any = None, columns: Sequence[str] | None = None) -> None:
        self._data: dict[str, NDArray[Any]] = {}
        self.attrs: dict[str, Any] = {}
        if data is None:
            data = {}
        if isinstance(data, Table):
            data = data._data
        if not isinstance(data, Mapping):  # rows
            rows = list(data)
            names = (
                list(columns)
                if columns is not None
                else list(dict.fromkeys(key for row in rows for key in row))
            )
            data = {name: [row.get(name, np.nan) for row in rows] for name in names}
        names = list(columns) if columns is not None else list(data)
        lengths = [len(data[name]) for name in names if name in data and not _scalar(data[name])]
        n_rows = lengths[0] if lengths else 0
        for name in names:
            values = data[name] if name in data else np.full(n_rows, np.nan)
            self._data[str(name)] = _as_column(values, n_rows)

    # ------------------------------------------------------------------ shape
    @property
    def columns(self) -> list[str]:
        """Column names, in order."""
        return list(self._data)

    def __len__(self) -> int:
        """Return the number of rows."""
        for values in self._data.values():
            return len(values)
        return 0

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, columns)."""
        return len(self), len(self._data)

    @property
    def empty(self) -> bool:
        """True when the table has no row."""
        return len(self) == 0

    # ------------------------------------------------------------------ access
    def __getitem__(self, key: Any) -> Any:
        """Return a column (a name), a table of columns (a list of names) or rows (a mask)."""
        if isinstance(key, str):
            if key not in self._data:
                raise KeyError(key)
            return Column(self._data[key])
        if isinstance(key, list) and all(isinstance(k, str) for k in key) and key:
            missing = [k for k in key if k not in self._data]
            if missing:
                raise KeyError(missing)
            out = Table({k: self._data[k] for k in key})
            out.attrs = dict(self.attrs)
            return out
        return self._take(key)

    def __setitem__(self, name: str, values: Any) -> None:
        """Add or replace a column (a scalar is repeated on every row)."""
        n_rows = len(self) if self._data else None
        if n_rows is None and np.ndim(values) == 0:
            n_rows = 0
        self._data[str(name)] = _as_column(values, n_rows)

    def __contains__(self, name: object) -> bool:
        """Return True when ``name`` is a column, as for a DataFrame."""
        return name in self._data

    def __iter__(self) -> Iterator[str]:
        """Iterate over the column names, as for a DataFrame."""
        return iter(self._data)

    @property
    def iloc(self) -> _ILoc:
        """Row selection by position."""
        return _ILoc(self)

    def _take(self, rows: Any) -> "Table":
        index = np.arange(len(self))[np.asarray(rows) if not isinstance(rows, slice) else rows]
        out = Table({name: values[index] for name, values in self._data.items()})
        out.attrs = dict(self.attrs)
        return out

    # ------------------------------------------------------------------ copies
    def copy(self, deep: bool = True) -> "Table":
        """Return a copy (``deep``: the columns are copied too, else shared)."""
        out = Table(
            {name: values.copy() if deep else values for name, values in self._data.items()}
        )
        out.attrs = dict(self.attrs)
        return out

    def reset_index(self, drop: bool = True) -> "Table":
        """Return a copy (a Table has no index; for code written for DataFrames)."""
        if not drop:
            raise ValueError("a Table has no index to keep")
        return self.copy()

    def sort_values(self, by: str, kind: _SortKind = "stable") -> "Table":
        """Return the rows in the order of column ``by`` (stable)."""
        return self._take(np.argsort(self._data[by], kind=kind))

    # ------------------------------------------------------------------ export
    def to_dict(self, orient: str = "dict") -> Any:
        """Return the table as Python objects, as ``DataFrame.to_dict`` (RangeIndex).

        ``"dict"``: {column: {row: value}}; ``"list"``: {column: list}; ``"records"``: a
        list of row dicts.
        """
        if orient == "dict":
            return {name: dict(enumerate(values.tolist())) for name, values in self._data.items()}
        if orient == "list":
            return {name: values.tolist() for name, values in self._data.items()}
        if orient == "records":
            return [
                {name: values[i] for name, values in self._data.items()} for i in range(len(self))
            ]
        raise ValueError(f"orient must be 'dict', 'list' or 'records', got {orient!r}")

    def itertuples(self, index: bool = True, name: str | None = None) -> Iterable[tuple[Any, ...]]:
        """Iterate over the rows as tuples (the position first when ``index``).

        ``name`` gives named tuples of that name, as pandas does; None plain tuples.
        """
        columns = list(self._data.values())
        fields = (["Index"] if index else []) + self.columns
        kind = None if name is None else namedtuple(name, fields, rename=True)
        for i in range(len(self)):
            row = (
                (i, *(values[i] for values in columns)) if index else tuple(v[i] for v in columns)
            )
            yield row if kind is None else kind(*row)

    def to_pandas(self) -> Any:
        """Return the same table as a pandas DataFrame (extra ``data``)."""
        pd = require("pandas", extra="data", caller="evoparam.data.Table.to_pandas")
        frame = pd.DataFrame(dict(self._data), columns=self.columns)
        frame.attrs.update(self.attrs)
        return frame

    def __repr__(self) -> str:
        """Column names and number of rows."""
        return f"<Table: {len(self)} rows x {len(self._data)} columns {self.columns}>"


def new_table(data: Any, columns: Sequence[str] | None = None) -> Any:
    """Return a pandas DataFrame of ``data`` when pandas is installed, else a :class:`Table`.

    ``data`` is a mapping of column name to values, or a sequence of row mappings, as for
    ``pandas.DataFrame``.
    """
    if has_pandas():
        pd = require("pandas", extra="data", caller="evoparam.data.new_table")
        return pd.DataFrame(data, columns=columns)
    return Table(data, columns=columns)


def concat_tables(tables: Sequence[Any]) -> Any:
    """Rows of ``tables`` one after the other, as ``pandas.concat(..., ignore_index=True)``."""
    tables = list(tables)
    if tables and not isinstance(tables[0], Table):
        pd = require("pandas", extra="data", caller="evoparam.data.concat_tables")
        return pd.concat(tables, ignore_index=True)
    if not tables:
        return new_table({})
    names = list(dict.fromkeys(name for table in tables for name in table.columns))
    return Table(
        {
            name: np.concatenate(
                [
                    table._data[name]
                    if name in table._data
                    else np.full(len(table), np.nan, dtype=object)
                    for table in tables
                ]
            )
            for name in names
        }
    )
