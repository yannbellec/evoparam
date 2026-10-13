# SPDX-License-Identifier: Apache-2.0
"""REQ-006 (uncertainty everywhere): every reported latency, amplitude, width, onset, area and
existence decision has an interval or a p-value in the same table row."""

import warnings
from typing import Any

import numpy as np
import pytest

import evoparam
from evoparam import sim
from evoparam.data.results import INTERVAL_COLUMNS, INTERVAL_FEATURES

pytest.importorskip("pandas")

REPORTED = ("latency", "amplitude", "fwhm", "onset", "area")


@pytest.fixture(scope="module", params=["trials", "evoked"])
def table(request: Any) -> Any:
    trials, _ = sim.simulate_ccep(n_trials=10, n_channels=4, seed=3)
    data = trials if request.param == "trials" else trials.average()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = evoparam.fit(data, preset="ccep", random_state=0)
    return group.to_dataframe()


def test_req006_interval_columns_for_every_reported_value(table: Any) -> None:
    assert set(REPORTED) <= set(INTERVAL_FEATURES)
    for name in REPORTED:
        assert {f"{name}_ci_low", f"{name}_ci_high"} <= set(table.columns), name
    assert set(INTERVAL_COLUMNS) <= set(table.columns)
    assert {"existence_p", "existence_significant"} <= set(table.columns)


def test_req006_every_reported_value_has_its_interval_in_the_row(table: Any) -> None:
    ok = table[table["status"] == "ok"]
    assert len(ok) >= 1
    for name in REPORTED:
        value = ok[name].to_numpy(float)
        low = ok[f"{name}_ci_low"].to_numpy(float)
        high = ok[f"{name}_ci_high"].to_numpy(float)
        reported = np.isfinite(value)
        assert reported.any(), name
        assert np.all(np.isfinite(low[reported]) & np.isfinite(high[reported])), name
        assert np.all(low[reported] <= high[reported]), name


def test_req006_every_existence_decision_has_its_p_value(table: Any) -> None:
    decided = table["existence_significant"].notna()
    assert decided.all()
    p = table.loc[decided, "existence_p"].to_numpy(float)
    assert np.all(np.isfinite(p) & (p >= 0) & (p <= 1))


def _durations(data: Any) -> Any:
    from evoparam.existence import response_duration

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return response_duration(
            data, window=(0.009, 0.5), baseline=(-0.5, -0.05), by=(), random_state=0
        )


def test_req006_existence_table_with_trials() -> None:
    trials, _ = sim.simulate_ccep(n_trials=10, n_channels=4, seed=3)
    frame = _durations(trials)
    assert frame["p_value"].notna().all()
    assert frame["significant"].notna().all()
    responding = frame[frame["significant"].astype(bool)]
    assert len(responding) >= 1
    for name in ("onset", "offset", "duration"):
        low = responding[f"{name}_ci_low"].to_numpy(float)
        high = responding[f"{name}_ci_high"].to_numpy(float)
        assert np.all(np.isfinite(low) & np.isfinite(high) & (low <= high)), name


def test_req006_existence_table_of_an_evoked() -> None:
    trials, _ = sim.simulate_ccep(n_trials=10, n_channels=4, seed=3)
    frame = _durations(trials.average())
    assert frame["p_value"].notna().all()
    responding = frame[frame["significant"].astype(bool)]
    assert len(responding) >= 1
    for name in ("onset", "offset", "duration"):
        low = responding[f"{name}_ci_low"].to_numpy(float)
        high = responding[f"{name}_ci_high"].to_numpy(float)
        assert np.all(np.isfinite(low) & np.isfinite(high) & (low <= high)), name
