# SPDX-License-Identifier: Apache-2.0
"""Tests for the result dataclasses (spec §4.3.4)."""

import copy

import numpy as np
import pytest

from evoparam.data import GroupFit, ResponseFit
from evoparam.data.results import (COMPONENT_COLUMNS, EXISTENCE_COLUMNS, FIT_COLUMNS,
                                   INTERVAL_COLUMNS)
from evoparam.errors import DataError
from evoparam.tests.data.factories import make_group_fit, make_response_fit, make_component


def test_components_sorted_by_latency():
    fit = make_response_fit()
    assert [c.label for c in fit.components] == ["N1", "N2"]


def test_status_rules():
    with pytest.raises(DataError, match="status"):
        ResponseFit(components=[], background={}, gof=None, status="maybe", message="x",
                    window=(0, 1), baseline=(-1, 0), settings={})
    with pytest.raises(DataError, match="message"):
        ResponseFit(components=[], background={}, gof=None, status="failed", message="",
                    window=(0, 1), baseline=(-1, 0), settings={})


def test_component_checks():
    with pytest.raises(DataError, match="polarity"):
        comp = make_component()
        type(comp)(**{**comp.to_dict(), "polarity": 0})
    with pytest.raises(DataError, match="param_names"):
        comp = make_component()
        type(comp)(**{**comp.to_dict(), "params": np.zeros(2)})


def test_group_tables_are_tidy_with_units():
    group = make_group_fit()
    components = group.to_dataframe("component")
    # The component columns, then the existence test of each channel (spec §5.6)
    assert list(components.columns) == list(COMPONENT_COLUMNS) + list(INTERVAL_COLUMNS) \
        + list(EXISTENCE_COLUMNS)
    assert set(components.attrs["units"]) == set(components.columns)
    assert len(components) == 2 + 1 + 2
    assert components.loc[components["channel"] == "CH01", "status"].item() == "no_response"
    fits = group.to_dataframe("fit")
    assert list(fits.columns) == list(FIT_COLUMNS)
    assert np.isnan(fits.loc[1, "r2"])
    with pytest.raises(DataError):
        group.to_dataframe("trial")


def test_group_get():
    group = make_group_fit()
    assert group.get("CH01").status == "no_response"
    assert group.get("CH00", condition="3mA") is group.fits[0]
    with pytest.raises(KeyError):
        group.get("CH09")
    with pytest.raises(DataError):
        GroupFit.from_fits([make_response_fit()], ["A", "B"])


def test_equality_is_nan_aware():
    first = make_response_fit()
    # A deep copy, not a second factory call: nested objects carry their own creation
    # time, which can differ by a millisecond between two calls.
    second = copy.deepcopy(first)
    assert second is not first and second.existence is not first.existence
    assert first == second
    other = make_response_fit()
    other.components[0].latency += 1e-6
    assert other != make_response_fit()
