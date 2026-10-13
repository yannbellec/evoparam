# SPDX-License-Identifier: Apache-2.0
"""Scorers: Hungarian matching of fitted to true components, and their summary."""

from typing import Any

import numpy as np
import pytest

from evoparam import sim
from evoparam.errors import SettingsError
from evoparam.tests.sim._helpers import SMALL, fits_from_truth

pd = pytest.importorskip("pandas")


def _truth_table(
    latencies: Any, fwhm: Any = 0.02, channel: Any = "c", amplitude: Any = -5.0
) -> Any:
    return pd.DataFrame(
        {
            "channel": [channel] * len(latencies),
            "latency": latencies,
            "amplitude": [amplitude] * len(latencies),
            "fwhm": [fwhm] * len(latencies),
        }
    )


def _fit_table(latencies: Any, channel: Any = "c", amplitude: Any = -5.0, **columns: Any) -> Any:
    table = pd.DataFrame(
        {
            "channel": [channel] * len(latencies),
            "component": np.arange(len(latencies)),
            "latency": latencies,
            "amplitude": [amplitude] * len(latencies),
            "fwhm": [0.02] * len(latencies),
            "polarity": [int(np.sign(amplitude))] * len(latencies),
        }
    )
    for key, value in columns.items():
        table[key] = value
    return table


@pytest.mark.parametrize("name", sorted(sim.GENERATORS))
def test_perfect_fit_scores_one(name: Any) -> None:
    _, truth = sim.GENERATORS[name](seed=5, **SMALL[name])
    fits = fits_from_truth(truth)
    score = sim.score_fits(truth, fits).iloc[0]
    assert score["recall"] == 1.0
    assert score["precision"] == 1.0
    assert score["f1"] == 1.0
    assert score["n_matched"] == score["n_true"] == truth.to_dataframe().shape[0]
    assert score["latency_mae"] < 1e-12
    assert score["amplitude_rel_mae"] < 1e-12
    assert score["n_false"] == 0
    assert list(sim.score_fits(truth, fits).columns) == list(sim.SCORE_COLUMNS)


def test_errors_and_coverage() -> None:
    _, truth = sim.simulate_ccep(seed=3, **SMALL["ccep"])
    fits = fits_from_truth(truth, latency_error=0.001, amplitude_factor=1.1, ci_halfwidth=0.002)
    table = sim.match_components(truth, fits)
    matched = table[table["kind"] == "match"]
    np.testing.assert_allclose(matched["latency_error"], 0.001, atol=1e-12)
    np.testing.assert_allclose(matched["amplitude_rel_error"], 0.1, rtol=1e-9)
    assert table.attrs["units"]["latency_error"] == "s"
    score = sim.score_fits(truth, fits).iloc[0]
    assert np.isclose(score["latency_bias"], 0.001)
    assert np.isclose(score["latency_mae"], 0.001)
    assert np.isclose(score["latency_mae_samples"], 0.001 * truth.sfreq)
    assert np.isclose(score["amplitude_rel_bias"], 0.1)
    assert score["coverage_latency"] == 1.0
    assert np.isnan(score["coverage_amplitude"])  # no amplitude interval
    narrow = fits_from_truth(truth, latency_error=0.001, ci_halfwidth=0.0005)
    assert sim.score_fits(truth, narrow).iloc[0]["coverage_latency"] == 0.0


def test_tolerance_misses_and_false_components() -> None:
    truth = _truth_table([0.1])
    for latency, kinds in ((0.109, ["match"]), (0.111, ["miss", "false"])):
        out = sim.match_components(truth, _fit_table([latency]))
        assert sorted(out["kind"]) == sorted(kinds)
    out = sim.match_components(truth, _fit_table([0.111]), tolerance="fwhm/1")
    assert list(out["kind"]) == ["match"]
    out = sim.match_components(truth, _fit_table([0.111]), tolerance="1*fwhm")
    assert list(out["kind"]) == ["match"]
    out = sim.match_components(truth, _fit_table([0.111]), tolerance=0.005)
    assert sorted(out["kind"]) == ["false", "miss"]
    out = sim.match_components(truth, _fit_table([0.5]), tolerance=None)
    assert list(out["kind"]) == ["match"]
    with pytest.raises(SettingsError):
        sim.match_components(truth, _fit_table([0.1]), tolerance="half")
    with pytest.raises(SettingsError):
        sim.match_components(truth, _fit_table([0.1]), tolerance=-1.0)


def test_hungarian_assignment_is_optimal() -> None:
    """Greedy matching would pair 0.100 with 0.105 and leave 0.110 without a partner; the
    optimal assignment (total |Δ| 0.017) pairs 0.100 with 0.088 and 0.110 with 0.105."""

    truth = _truth_table([0.100, 0.110], fwhm=0.04)
    fit = _fit_table([0.105, 0.088])
    out = sim.match_components(truth, fit, tolerance=0.015)
    matched = out[out["kind"] == "match"].sort_values("latency_true")
    assert len(matched) == 2
    np.testing.assert_allclose(matched["latency_fit"], [0.088, 0.105])


def test_polarity_and_roles() -> None:
    truth = _truth_table([0.1, 0.3])
    truth["role"] = ["target", "confound"]
    fit = _fit_table([0.1], amplitude=5.0)
    assert list(sim.match_components(truth.iloc[:1], fit)["kind"]) == ["match"]
    out = sim.match_components(truth.iloc[:1], fit, match_polarity=True)
    assert sorted(out["kind"]) == ["false", "miss"]
    out = sim.match_components(truth, fit, roles="target")
    assert list(out["kind"]) == ["match"]
    out = sim.match_components(truth, fit, window=(0.2, 0.4))
    assert sorted(out["kind"]) == ["false", "miss"]


def test_false_rate_on_empty_waveforms() -> None:
    _, truth = sim.simulate_ccep(seed=3, **SMALL["ccep"])
    silent = [name for name in truth.ch_names if name not in truth.settings["responders"]]
    assert len(silent) >= 2
    fits = fits_from_truth(truth, extra={(silent[0], ""): [0.05, 0.2]})
    score = sim.score_fits(truth, fits).iloc[0]
    assert score["n_false"] == 2
    assert score["recall"] == 1.0
    assert np.isclose(score["false_rate_empty"], 2 / len(silent))
    assert np.isclose(score["false_per_waveform"], 2 / len(truth.ch_names))
    assert np.isclose(score["precision"], score["n_matched"] / (score["n_matched"] + 2))
    by_channel = sim.score_fits(truth, fits, by="channel")
    row = by_channel[by_channel["channel"] == silent[0]].iloc[0]
    assert row["n_false"] == 2
    assert row["false_rate_empty"] == 2.0


def test_missed_components_lower_recall() -> None:
    _, truth = sim.simulate_erp(seed=1, n_trials=6, n_channels=2, components=["N1", "P3"])
    fits = fits_from_truth(truth, drop=[(truth.ch_names[0], "P3")])
    score = sim.score_fits(truth, fits).iloc[0]
    assert score["recall"] == 0.75
    assert score["precision"] == 1.0
    assert np.isclose(score["f1"], 2 * 3 / (4 + 3))
    by_label = sim.score_fits(truth, fits, by="label").set_index("label")
    assert by_label.loc["P3", "recall"] == 0.5
    assert by_label.loc["N1", "recall"] == 1.0


def test_conditions() -> None:
    _, truth = sim.simulate_animal_ep(seed=1, **SMALL["animal"])
    fits = fits_from_truth(truth)
    assert fits.conditions == ["awake", "nrem"]
    assert sim.score_fits(truth, fits).iloc[0]["recall"] == 1.0
    one = fits.fits[fits.index["position"][(fits.index["condition"] == "awake")].iloc[0]]
    with pytest.raises(SettingsError, match="several conditions"):
        sim.match_components(
            truth,
            pd.DataFrame(
                {"channel": ["L00"], "latency": [0.05], "amplitude": [1.0], "fwhm": [0.01]}
            ),
        )
    out = sim.match_components(
        truth.to_dataframe().query("condition == 'awake'"), one, channel=truth.ch_names[0]
    )
    assert (out["kind"] == "match").sum() >= 1


def test_response_fit_and_dataframe_inputs() -> None:
    _, truth = sim.simulate_erp(seed=1, n_trials=6, components=["N1", "P3"])
    fits = fits_from_truth(truth)
    single = fits.fits[0]
    out = sim.match_components(truth, single)
    assert list(out["kind"]) == ["match", "match"]
    table = fits.to_dataframe()
    out = sim.match_components(truth, table)
    assert list(out["kind"]) == ["match", "match"]
    _, two = sim.simulate_erp(seed=1, n_trials=6, n_channels=2)
    with pytest.raises(SettingsError, match="name the channel"):
        sim.match_components(two, single)
    with pytest.raises(SettingsError):
        sim.match_components(truth, [1, 2])
    with pytest.raises(SettingsError):
        sim.match_components([1], table)
    with pytest.raises(SettingsError):
        sim.match_components(truth, table, level="channel")
    # A fit without any component (component == -1 row) counts as fitted, with no rows.
    empty = pd.DataFrame(
        {
            "channel": [truth.ch_names[0]],
            "component": [-1],
            "latency": [np.nan],
            "amplitude": [np.nan],
            "fwhm": [np.nan],
            "polarity": [0],
        }
    )
    out = sim.match_components(truth, empty)
    assert list(out["kind"]) == ["miss", "miss"]


def test_trial_level_matching() -> None:
    _, truth = sim.simulate_erp(seed=2, n_trials=4, components=["P3"], jitter_sd=0.02)
    trials = truth.to_dataframe("trial")
    fit = trials[
        ["channel", "trial", "component", "latency", "amplitude", "fwhm", "polarity"]
    ].copy()
    out = sim.match_components(truth, fit, level="trial")
    assert list(out["kind"]) == ["match"] * 4
    assert list(out["trial"]) == [0, 1, 2, 3]
    np.testing.assert_allclose(out["latency_error"], 0.0, atol=1e-12)
    swapped = fit.assign(trial=fit["trial"].to_numpy()[::-1])
    out = sim.match_components(truth, swapped, level="trial", tolerance=1e-4)
    assert (out["kind"] == "match").sum() < 4
    with pytest.raises(SettingsError, match="trial"):
        sim.match_components(truth, fit.drop(columns="trial"), level="trial")
