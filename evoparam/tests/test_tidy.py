"""REQ-332 (tidy tables): the column set of ``GroupFit.to_dataframe("component")`` equals
the documented list in ``docs/reference/tables.rst``, and every column has a unit entry
(in ``table.attrs["units"]`` and in the sidecar of the saved container)."""

import json
import pathlib
import re
import warnings

import pytest

import evoparam
from evoparam import sim

pytest.importorskip("pandas")

DOC = pathlib.Path(evoparam.__file__).parent.parent / "docs" / "reference" / "tables.rst"
UNITS = {"(none)": "", "data unit": "data", "data unit × s": "data·s"}


def documented(anchor):
    """{column: unit} of the list-table that follows ``.. _<anchor>:``."""

    if not DOC.is_file():
        pytest.skip("docs/ not in this checkout")
    text = DOC.read_text(encoding="utf-8").split(f".. _{anchor}:")[1].split(".. _")[0]
    rows = re.findall(r"^   \* - ``([^`]+)``\n     - ([^\n]*)$", text, flags=re.MULTILINE)
    return {name: UNITS.get(unit.strip(), unit.strip()) for name, unit in rows}


@pytest.fixture(scope="module")
def group():
    trials, _ = sim.simulate_ccep(n_trials=8, n_channels=3, sfreq=1024.0, seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return evoparam.fit(trials, preset="ccep", existence=True, random_state=0)


def test_component_columns_are_the_documented_list(group, tmp_path):

    doc = documented("table-component")
    table = group.to_dataframe("component")
    assert list(table.columns) == list(doc)
    assert table.attrs["units"] == doc
    assert set(table.attrs["units"]) == set(table.columns)
    # the saved container: every column of every table has a unit entry in its sidecar
    path = pathlib.Path(group.save(str(tmp_path / "group")))
    sidecars = list((path / "tables").glob("*.json"))
    assert sidecars
    for sidecar in sidecars:
        columns = json.loads(sidecar.read_text(encoding="utf-8"))["columns"]
        assert columns and all("unit" in entry for entry in columns.values()), sidecar.name


def test_fit_columns_are_the_documented_list(group):

    doc = documented("table-fit")
    table = group.to_dataframe("fit")
    assert list(table.columns) == list(doc)
    assert table.attrs["units"] == doc


def test_one_row_per_channel_condition_component(group):

    table = group.to_dataframe("component")
    keys = table[["channel", "condition", "component"]]
    assert not keys.duplicated().any()
    n_rows = sum(max(len(f.components), 1) for f in group.fits)
    assert len(table) == n_rows
