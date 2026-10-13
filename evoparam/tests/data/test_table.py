# SPDX-License-Identifier: Apache-2.0
"""REQ-010 (light core): the pandas-free :class:`evoparam.data.Table`.

Each operation that the package uses on its tables gives, on a Table, what it gives on the
pandas DataFrame of the same data; containers whose tables are Tables save and load
without pandas (CSV tables) with the same dtypes, missing values and units.
"""

import importlib.util
import json
import pathlib
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_equal

from evoparam.data import TrialSet, _table, load_object, save_object, serialize
from evoparam.data._table import (
    Column,
    Table,
    as_text,
    concat_tables,
    has_pandas,
    is_missing,
    is_table,
    new_table,
    row_labels,
)
from evoparam.errors import DataError

pytestmark = pytest.mark.core

DATA = {
    "onset": [1.5, 0.5, 1.0, 0.5],
    "site": ["B1-B2", "A1-A2", None, "A1-A2"],
    "n": np.array([3, 1, 2, 4], dtype="int64"),
    "ok": [True, False, True, True],
}


@pytest.fixture
def table() -> Table:
    out = Table(DATA)
    out.attrs["units"] = {"onset": "s"}
    return out


@pytest.fixture
def without_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Behave as if pandas were not installed (the CSV reader and the table factory)."""
    monkeypatch.setattr(_table, "has_pandas", lambda: False)
    monkeypatch.setattr(serialize, "has_pandas", lambda: False)


def test_construction_from_columns_and_from_rows(table: Table) -> None:
    assert table.columns == ["onset", "site", "n", "ok"]
    assert table.shape == (len(DATA["n"]), len(DATA))
    assert not table.empty
    # text is held as Python strings (object), as pandas holds it
    assert table["site"].dtype == object
    assert table["n"].dtype == np.int64
    rows = Table([{"a": 1.0, "b": "x"}, {"a": 2.0}])
    assert rows.columns == ["a", "b"]
    assert rows["a"].tolist() == [1.0, 2.0]
    assert rows["b"][0] == "x"
    assert is_missing(rows["b"][1])
    ordered = Table({"a": [1, 2], "b": [3, 4]}, columns=["b", "a", "c"])
    assert ordered.columns == ["b", "a", "c"]
    assert np.isnan(ordered["c"]).all()
    assert Table(table).columns == table.columns
    scalar = Table({"a": [1, 2, 3], "b": "x"})
    assert scalar["b"].tolist() == ["x", "x", "x"]
    empty = Table()
    assert empty.empty
    assert len(empty) == 0
    assert empty.columns == []


def test_construction_errors() -> None:
    with pytest.raises(ValueError, match="1-D"):
        Table({"a": np.zeros((2, 2))})
    with pytest.raises(ValueError, match="3 values for 2 rows"):
        Table({"a": [1, 2], "b": [1, 2, 3]})
    with pytest.raises(ValueError, match="number of rows"):
        _table._as_column(1.0, None)


def test_column_access_and_selection(table: Table) -> None:
    onset = table["onset"]
    assert isinstance(onset, Column)
    assert type(onset.to_numpy()) is np.ndarray
    assert onset.to_numpy(dtype=int).tolist() == [1, 0, 1, 0]
    copied = onset.to_numpy(copy=True)
    copied[0] = 99.0
    assert_equal(table["onset"][0], DATA["onset"][0])
    assert_equal(float(onset.iloc[1]), DATA["onset"][1])
    assert type(onset.values) is np.ndarray
    assert onset.unique().tolist() == [1.5, 0.5, 1.0]  # order of first appearance
    sub = table[["site", "onset"]]
    assert sub.columns == ["site", "onset"]
    assert sub.attrs == table.attrs
    picked = table[table["onset"] < table["onset"][0]]
    assert picked["n"].tolist() == [1, 2, 4]
    assert picked.attrs == table.attrs
    assert table.iloc[1] == {"onset": 0.5, "site": "A1-A2", "n": 1, "ok": False}
    assert table.iloc[1:3]["n"].tolist() == [1, 2]
    assert table.iloc[np.array([3, 0])]["n"].tolist() == [4, 3]
    assert "site" in table
    assert "x" not in table
    assert list(table) == table.columns
    with pytest.raises(KeyError):
        table["x"]
    with pytest.raises(KeyError):
        table[["onset", "x"]]


def test_assignment(table: Table) -> None:
    table["quality"] = [0.1, 0.2, 0.3, 0.4]
    table["flag"] = "x"
    assert table.columns[-2:] == ["quality", "flag"]
    assert table["flag"].tolist() == ["x"] * 4
    with pytest.raises(ValueError, match="2 values for 4 rows"):
        table["bad"] = [1, 2]
    fresh = Table()
    fresh["a"] = [1, 2]
    assert len(fresh) == len([1, 2])
    blank = Table()
    blank["a"] = 1.0  # a scalar in a table without rows: no row
    assert len(blank) == 0


def test_copies_order_and_export(table: Table) -> None:
    deep, shallow = table.copy(), table.copy(deep=False)
    table["onset"][0] = 7.0  # in place in the column array
    assert_equal(deep["onset"][0], DATA["onset"][0])
    assert_equal(shallow["onset"][0], 7.0)
    assert deep.attrs == table.attrs
    assert deep.attrs is not table.attrs
    assert table.reset_index(drop=True)["n"].tolist() == table["n"].tolist()
    with pytest.raises(ValueError, match="no index"):
        table.reset_index(drop=False)
    ordered = deep.sort_values("onset")
    assert ordered["n"].tolist() == [1, 4, 2, 3]  # stable: the two 0.5 keep their order
    assert deep.to_dict("list")["n"] == [3, 1, 2, 4]
    assert deep.to_dict("records")[0] == {"onset": 1.5, "site": "B1-B2", "n": 3, "ok": True}
    assert deep.to_dict()["onset"] == {0: 1.5, 1: 0.5, 2: 1.0, 3: 0.5}
    with pytest.raises(ValueError, match="orient"):
        deep.to_dict("split")
    rows = list(deep.itertuples())
    assert rows[1] == (1, 0.5, "A1-A2", 1, False)
    assert list(deep.itertuples(index=False))[0] == (1.5, "B1-B2", 3, True)
    named = next(iter(deep.itertuples(name="Row")))
    assert named.Index == 0
    assert named.site == "B1-B2"
    assert repr(deep) == "<Table: 4 rows x 4 columns ['onset', 'site', 'n', 'ok']>"


def test_helpers() -> None:
    for value in (None, float("nan"), np.float32("nan")):
        assert is_missing(value)
    for value in (0.0, "", "nan"):
        assert not is_missing(value)
    assert as_text(["a", None, np.nan, 2]).tolist() == ["a", "", "", "2"]
    labels = row_labels(Table({"site": ["A", "B"], "i": [1, 2]}), ["site", "i"])
    assert labels.tolist() == ["A/1", "B/2"]
    assert labels.dtype == object
    assert is_table(Table())
    assert not is_table({"a": [1]})
    assert not is_table(np.zeros(2))


def test_has_pandas_with_a_hidden_package(monkeypatch: pytest.MonkeyPatch) -> None:
    def hidden(name: str) -> Any:
        raise ImportError(f"hidden: {name}")

    monkeypatch.setattr(importlib.util, "find_spec", hidden)
    assert has_pandas() is False


@pytest.mark.usefixtures("without_pandas")
def test_new_and_concat_without_pandas() -> None:
    made = new_table({"a": [1.0, 2.0]})
    assert isinstance(made, Table)
    assert isinstance(new_table([{"a": 1}], columns=["a"]), Table)
    joined = concat_tables([Table({"a": [1.0], "b": ["x"]}), Table({"a": [2.0, 3.0]})])
    assert isinstance(joined, Table)
    assert joined["a"].tolist() == [1.0, 2.0, 3.0]
    assert joined["b"][0] == "x"
    assert all(is_missing(v) for v in joined["b"][1:])
    empty = concat_tables([])
    assert isinstance(empty, Table)
    assert empty.empty


def test_table_and_dataframe_agree(table: Table) -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(DATA)
    assert isinstance(new_table(DATA), pd.DataFrame)
    converted = table.to_pandas()
    pd.testing.assert_frame_equal(converted, frame)
    assert converted.attrs == table.attrs
    assert is_table(frame)
    for name in ("onset", "n", "ok"):
        assert table[name].tolist() == frame[name].tolist()
        assert table[name].dtype == frame[name].to_numpy().dtype
    # text: the same values, a missing value as missing (None here, NaN or None in pandas)
    assert as_text(table["site"]).tolist() == as_text(frame["site"]).tolist()
    assert [is_missing(v) for v in table["site"]] == frame["site"].isna().tolist()
    mask = table["onset"].to_numpy() < table["onset"][0]
    assert table[mask]["n"].tolist() == frame[mask]["n"].tolist()
    assert (
        table.sort_values("onset")["n"].tolist()
        == frame.sort_values("onset", kind="stable")["n"].tolist()
    )
    numbers = ["onset", "n", "ok"]
    assert table[numbers].to_dict("list") == frame[numbers].to_dict("list")
    assert table[numbers].to_dict() == frame[numbers].to_dict()
    assert table[numbers].to_dict("records") == frame[numbers].to_dict("records")
    assert list(table[numbers].itertuples(index=False)) == list(
        frame[numbers].itertuples(index=False, name=None)
    )
    assert table["onset"].unique().tolist() == frame["onset"].unique().tolist()
    assert as_text(table["site"]).tolist() == frame["site"].fillna("").astype(str).tolist()
    expected = frame[["onset", "n"]].astype(str).agg("/".join, axis=1).tolist()
    assert row_labels(table, ["onset", "n"]).tolist() == expected
    assert is_missing(pd.NA)
    assert is_missing(pd.NaT)
    joined = concat_tables([frame, frame])
    assert isinstance(joined, pd.DataFrame)
    assert list(joined.index) == list(range(2 * len(frame)))


def _trials(metadata: Any) -> TrialSet:
    rng = np.random.default_rng(0)
    return TrialSet(
        data=rng.standard_normal((4, 2, 20)),
        times=-0.1 + np.arange(20) / 50.0,
        sfreq=50.0,
        ch_names=["A1-A2", "B1-B2"],
        metadata=metadata,
    )


@pytest.mark.usefixtures("without_pandas")
def test_container_with_a_table_saves_and_loads_without_pandas(
    table: Table, tmp_path: pathlib.Path
) -> None:
    table["count"] = np.array([1.0, np.nan, 2.0, 3.0])
    trials = _trials(table)
    path = save_object(trials, tmp_path / "trials.evoparam", table_format="csv")
    again = load_object(path)
    meta = again.metadata
    assert isinstance(meta, Table)
    assert meta.columns == table.columns
    assert meta.attrs["units"]["onset"] == "s"
    assert meta["n"].dtype == np.int64
    assert meta["ok"].dtype == bool
    assert meta["onset"].tolist() == table["onset"].tolist()
    assert_equal(meta["count"], [1.0, np.nan, 2.0, 3.0])
    assert meta["site"].tolist() == ["B1-B2", "A1-A2", None, "A1-A2"]
    assert again == trials
    folder = pathlib.Path(path) / "tables"
    sidecar = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
    assert sidecar["columns"]["onset"] == {"dtype": "float64", "unit": "s"}


def test_light_table_reads_what_pandas_wrote(tmp_path: pathlib.Path) -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "onset": [0.5, np.nan],
            "site": ["A1-A2", None],
            "n": pd.array([1, None], dtype="Int64"),
            "flag": pd.array([True, None], dtype="boolean"),
        }
    )
    trials = _trials(pd.concat([frame, frame], ignore_index=True))
    path = save_object(trials, tmp_path / "t", table_format="csv")
    folder = pathlib.Path(path) / "tables"
    key = next(folder.glob("*.json")).stem
    sidecar = json.loads((folder / f"{key}.json").read_text(encoding="utf-8"))
    light = serialize._read_light_table(str(folder), key, sidecar)
    assert_equal(light["onset"].to_numpy(), [0.5, np.nan, 0.5, np.nan])
    # the missing text value as pandas wrote it (NaN or None, recorded in the sidecar)
    heavy = serialize._read_table(str(folder), key)
    assert light["site"][0] == "A1-A2"
    assert is_missing(light["site"][1])
    assert type(light["site"][1]) is type(heavy["site"].to_numpy(dtype=object)[1])
    assert light["n"].tolist() == [1, None, 1, None]
    assert light["flag"].tolist() == [True, None, True, None]


def test_light_table_errors(tmp_path: pathlib.Path) -> None:
    folder = str(tmp_path)
    with pytest.raises(DataError, match="unsupported dtype"):
        serialize._write_light_table(Table({"t": np.array([1, 2], dtype="m8[s]")}), folder, "k")
    with pytest.raises(DataError, match="mixes strings"):
        serialize._write_light_table(Table({"s": np.array(["a", 1], dtype=object)}), folder, "k")
    with pytest.raises(DataError, match="reserved value"):
        serialize._write_light_table(Table({"s": [serialize.MISSING]}), folder, "k")
    with pytest.raises(DataError, match="different kinds"):
        serialize._write_light_table(
            Table({"s": np.array(["a", None, np.nan], dtype=object)}), folder, "k"
        )
    with pytest.raises(DataError, match="missing values"):
        serialize._light_column([serialize.MISSING], "int64", "n", "k", None)
