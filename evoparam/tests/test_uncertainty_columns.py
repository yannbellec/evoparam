# SPDX-License-Identifier: Apache-2.0
"""REQ-006 (uncertainty everywhere): every reported latency, amplitude, width, onset, area and
existence decision has an interval or a p-value in the same table row."""

import warnings

import numpy as np
import pytest

import evoparam
from evoparam import sim
from evoparam.data.results import INTERVAL_COLUMNS, INTERVAL_FEATURES

pytest.importorskip("pandas")

REPORTED = ("latency", "amplitude", "fwhm", "onset", "area")


@pytest.fixture(scope="module", params=["trials", "evoked"])
def table(request):
    trials, _ = sim.simulate_ccep(n_trials=10, n_channels=4, seed=3)
    data = trials if request.param == "trials" else trials.average()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = evoparam.fit(data, preset="ccep", random_state=0)
    return group.to_dataframe()


def test_req006_interval_columns_for_every_reported_value(table):
    assert set(REPORTED) <= set(INTERVAL_FEATURES)
    for name in REPORTED:
        assert {f"{name}_ci_low", f"{name}_ci_high"} <= set(table.columns), name
    assert set(INTERVAL_COLUMNS) <= set(table.columns)
    assert {"existence_p", "existence_significant"} <= set(table.columns)


def test_req006_every_reported_value_has_its_interval_in_the_row(table):
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


def test_req006_every_existence_decision_has_its_p_value(table):
    decided = table["existence_significant"].notna()
    assert decided.all()
    p = table.loc[decided, "existence_p"].to_numpy(float)
    assert np.all(np.isfinite(p) & (p >= 0) & (p <= 1))


def _durations(data):
    from evoparam.existence import response_duration

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return response_duration(
            data, window=(0.009, 0.5), baseline=(-0.5, -0.05), by=(), random_state=0
        )


def test_req006_existence_table_with_trials():
    trials, _ = sim.simulate_ccep(n_trials=10, n_channels=4, seed=3)
    frame = _durations(trials)
    assert frame["p_value"].notna().all() and frame["significant"].notna().all()
    responding = frame[frame["significant"].astype(bool)]
    assert len(responding) >= 1
    for name in ("onset", "offset", "duration"):
        low = responding[f"{name}_ci_low"].to_numpy(float)
        high = responding[f"{name}_ci_high"].to_numpy(float)
        assert np.all(np.isfinite(low) & np.isfinite(high) & (low <= high)), name


@pytest.mark.xfail(
    strict=True,
    reason="An Evoked has no trials to resample: the onset, "
    "offset and duration of its existence test (baseline windows of the "
    "average, §5.6) are reported without an interval (measured: NaN on all "
    "4 channels of the simulated CCEP). Its decision has its p-value.",
)
def test_req006_existence_table_of_an_evoked():
    trials, _ = sim.simulate_ccep(n_trials=10, n_channels=4, seed=3)
    frame = _durations(trials.average())
    assert frame["p_value"].notna().all()
    responding = frame[frame["significant"].astype(bool)]
    assert len(responding) >= 1
    assert responding["onset_ci_low"].notna().all()
