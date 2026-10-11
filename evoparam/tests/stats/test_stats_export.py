# SPDX-License-Identifier: Apache-2.0
"""Tidy export (REQ-235), model formulas (REQ-236) and group summaries (REQ-237)."""

import importlib.util
import json
import warnings

import numpy as np
import pandas as pd
import pytest

import evoparam
from evoparam.data import GroupFit, HierarchicalFit, ResponseFit
from evoparam.errors import DataError, SettingsError
from evoparam.sim import simulate_erp
from evoparam.stats import (column_dictionary, fit_mixed, group_summary, mixed_formula,
                            to_long)

from .helpers import N1

HAS_STATSMODELS = importlib.util.find_spec("statsmodels") is not None

###################################################################################################
###################################################################################################

def _group_fit():
    """Fits of a simulated recording, plus a failed and a skipped channel."""

    comps = [dict(N1, channels=("Fz", "Cz")), {"label": "P3", "latency": 0.3, "fwhm": 0.1,
                                                "amplitude": 6.0, "channels": ("Pz",)}]
    ts, _ = simulate_erp(30, n_channels=4, components=comps, snr=None, noise="white",
                         noise_sd=2., sfreq=250., tmin=-0.2, tmax=0.6, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = evoparam.fit(ts, preset="erp", window=(0.0, 0.5), baseline=(-0.2, 0.0),
                           existence=True, existence_kwargs={"n_permutations": 200,
                                                             "baseline": (-0.2, 0.0)})
    extra = [ResponseFit(components=[], background={}, gof=None, status=status,
                         message=message, window=(0.0, 0.5), baseline=(-0.2, 0.0), settings={})
             for status, message in (("failed", "the optimiser raised"),
                                     ("skipped", "stimulating channel"))]
    fits = list(fit.fits) + extra
    names = list(fit.index["channel"]) + ["X1", "X2"]
    return GroupFit.from_fits(fits, names, settings=fit.settings, provenance=fit.provenance)


def _scalar(value):
    return not isinstance(value, (list, tuple, dict, set, np.ndarray, pd.DataFrame, pd.Series))


def test_req235_long_table():

    group = _group_fit()
    long = to_long(group, units="uV", metadata={"subject": "01"})
    assert all(_scalar(v) for v in long.to_numpy().ravel())
    assert list(long.columns[:2]) == ["subject", "condition"]
    assert list(long.columns[-7:]) == ["parameter", "value", "unit", "se", "ci_low", "ci_high",
                                      "weight"]
    sidecar = long.attrs["sidecar"]
    assert list(sidecar) == list(long.columns)
    assert all(entry["Description"] and entry["Units"] for entry in sidecar.values())
    json.loads(json.dumps(sidecar))
    assert set(sidecar["status"]["Levels"]) == {"ok", "no_response", "failed", "skipped"}
    units = dict(zip(long["parameter"], long["unit"]))
    assert units["latency"] == "s" and units["amplitude"] == "uV" and units["area"] == "uV·s"
    ok = long[(long["status"] == "ok") & (long["parameter"] == "latency")]
    assert (ok["se"] > 0).all()
    assert np.allclose(ok["weight"], 1 / ok["se"] ** 2)
    # channels without a component keep their rows, with their status, and no value
    for channel, status in (("X1", "failed"), ("X2", "skipped")):
        rows = long[long["channel"] == channel]
        assert len(rows) == 10 and set(rows["status"]) == {status}
        assert rows["value"].isna().all() and set(rows["component"]) == {-1}
    n_comp = sum(max(len(f.components), 1) for f in group.fits)
    assert len(long) == 10 * n_comp
    assert long.attrs["provenance"]["steps"][-1]["function"] == "evoparam.stats.to_long"


def test_req235_wide_table_and_other_results():

    group = _group_fit()
    wide = to_long(group, wide=True, units="uV")
    assert len(wide) == sum(max(len(f.components), 1) for f in group.fits)
    assert {"latency", "latency_se", "latency_weight", "amplitude_ci_low"} <= set(wide.columns)
    sidecar = wide.attrs["sidecar"]
    assert list(sidecar) == list(wide.columns)
    assert sidecar["latency_se"]["Units"] == "s"
    assert sidecar["amplitude"]["Units"] == "uV"
    assert sidecar["latency_weight"]["Units"] == "1/s^2"
    assert all(_scalar(v) for v in wide.to_numpy().ravel())
    # a trial table (TrialFit / TrialFitSet layout) keeps every metadata column
    trials = pd.DataFrame({"trial": [0, 0, 1, 1], "channel": "Cz", "condition": "",
                           "component": [0, 1, 0, 1], "label": ["N1", "P3"] * 2,
                           "shift": [0.001, 0.001, -0.002, -0.002], "scale": [1.1, 1.1, 0.9, 0.9],
                           "latency": [0.1, 0.3, 0.098, 0.299], "amplitude": [-5, 6, -4, 5.5],
                           "fwhm": [0.04, 0.1, 0.04, 0.1], "r2": [0.8, 0.8, 0.7, 0.7],
                           "converged": True, "trusted": True, "stim_current_mA": 3.0,
                           "tags": [["a"], ["a"], ["b"], ["b"]]})
    long = to_long(trials, units="uV")
    assert set(long["parameter"]) == {"shift", "scale", "latency", "amplitude", "fwhm", "r2"}
    assert {"stim_current_mA", "trusted", "trial"} <= set(long.columns)
    assert long["tags"].iloc[0] == '["a"]'                         # one value per cell
    assert long.attrs["sidecar"]["stim_current_mA"]["Description"].startswith("Column")
    summary = pd.DataFrame({"level": ["channel", "subject"], "unit": ["Cz", "01"],
                            "component": ["N1", "N1"], "parameter": ["latency", "latency"],
                            "mean": [0.1, 0.11], "sd": [0.002, 0.004],
                            "hdi_low": [0.096, 0.102], "hdi_high": [0.104, 0.118]})
    hier = HierarchicalFit(levels=("channel", "subject"), summary=summary, level_tables={},
                           p_response=pd.DataFrame(), variance_components=pd.DataFrame(),
                           effects=None, engine="eb", diagnostics={})
    long = to_long(hier)
    assert long["value"].tolist() == [0.1, 0.11] and long["unit"].tolist() == ["s", "s"]
    assert np.allclose(long["weight"], [1 / 0.002 ** 2, 1 / 0.004 ** 2])
    with pytest.raises(DataError):
        to_long(42)


def test_column_dictionary_unknown_columns():

    table = pd.DataFrame({"channel": ["A"], "my_score": [1.0]})
    table.attrs["units"] = {"my_score": "uV"}
    out = column_dictionary(table)
    assert out["my_score"] == {"Description": "Column 'my_score' of the input table.",
                               "Units": "uV"}
    assert out["channel"]["Units"] == "n/a"


# Frozen templates (REQ-236).
FORMULA_CASES = [
    ({"response": "value", "fixed": ["condition"], "groups": ["subject"]},
     {"lme4": "value ~ condition + (1 | subject)",
      "brms": "value ~ condition + (1 | subject)",
      "statsmodels": "value ~ C(condition)"}),
    ({"response": "value", "fixed": ["condition", "stim_current_mA"],
      "groups": ["subject", "channel"]},
     {"lme4": "value ~ condition + stim_current_mA + (1 | subject) + (1 | subject:channel)",
      "brms": "value ~ condition + stim_current_mA + (1 | subject) + (1 | subject:channel)",
      "statsmodels": "value ~ C(condition) + stim_current_mA"}),
    ({"response": "value", "fixed": [], "groups": ["subject", "stim site"], "se": "se"},
     {"lme4": "value ~ 1 + (1 | subject) + (1 | subject:`stim site`)",
      "brms": "value | se(se, sigma = TRUE) ~ 1 + (1 | subject) + (1 | subject:`stim site`)",
      "statsmodels": "value ~ 1"}),
    ({"response": "N1 latency", "fixed": ["if"], "groups": ["subject"]},
     {"lme4": "`N1 latency` ~ `if` + (1 | subject)",
      "brms": "`N1 latency` ~ `if` + (1 | subject)",
      "statsmodels": 'Q("N1 latency") ~ C(Q("if"))'}),
]


def _formula_table():
    rng = np.random.default_rng(0)
    n = 240
    subject = np.repeat([f"s{i}" for i in range(12)], 20)
    channel = np.tile(np.repeat(["A", "B", "C", "D"], 5), 12)
    condition = np.tile(["x", "y"], n // 2)
    value = rng.normal(size=12)[np.repeat(np.arange(12), 20)] + np.where(condition == "y", 1, 0) \
        + rng.normal(size=n)
    return pd.DataFrame({"value": value, "condition": condition, "subject": subject,
                         "channel": channel, "stim site": channel,
                         "stim_current_mA": rng.choice([1.0, 3.0], n), "se": 0.5,
                         "N1 latency": value, "if": condition})


@pytest.mark.parametrize("case, expected", FORMULA_CASES)
def test_req236_frozen_templates(case, expected):

    table = _formula_table()
    for engine, text in expected.items():
        kwargs = dict(case)
        se = kwargs.pop("se", None)
        assert mixed_formula(table, engine=engine, se=se, **kwargs) == text, engine


@pytest.mark.skipif(not HAS_STATSMODELS, reason="REQ-236 statsmodels part: the optional "
                    "package statsmodels (extra 'stats') is not installed")
@pytest.mark.parametrize("case, expected", FORMULA_CASES)
def test_req236_statsmodels_parses(case, expected):

    import statsmodels.formula.api as smf
    table = _formula_table()
    formula = mixed_formula(table, engine="statsmodels",
                            **{k: v for k, v in case.items() if k != "se"})
    model = smf.mixedlm(formula, table, groups=table[case["groups"][0]])
    assert model.exog.shape[0] == len(table)
    result = fit_mixed(table, formula, case["groups"][0], nested=case["groups"][1:])
    assert np.all(np.isfinite(result.fe_params))


@pytest.mark.skipif(HAS_STATSMODELS, reason="statsmodels is installed")
def test_fit_mixed_names_the_extra():

    with pytest.raises(ImportError, match=r"evoparam\[stats\]"):
        fit_mixed(_formula_table(), "value ~ 1", "subject")


def test_formula_errors():

    table = _formula_table()
    with pytest.raises(SettingsError, match="missing"):
        mixed_formula(table, response="value", fixed=["age"], groups=["subject"])
    with pytest.raises(SettingsError):
        mixed_formula(table, response="value", fixed=[], groups=[], engine="lme4")
    with pytest.raises(SettingsError):
        mixed_formula(table, response="value", fixed=[], groups=["subject"], engine="nlme")


def test_req237_status_counts():

    group = _group_fit()
    table = group.to_dataframe()
    info = pd.DataFrame({"name": group.ch_names,
                         "region": ["frontal", "central", "parietal", "occipital", "frontal",
                                    "central"]})
    for by in (["channel"], ["region"], ["condition"]):
        out = group_summary(table, by=by, ch_info=info, random_state=0)
        counts = out.drop_duplicates(by)
        total = counts[["n_ok", "n_no_response", "n_failed", "n_skipped"]].sum(axis=1)
        assert (total == counts["n_channels"]).all(), by
    out = group_summary(table, by=["condition"], random_state=0)
    first = out.iloc[0]
    statuses = [f.status for f in group.fits]
    assert first["n_channels"] == len(group.fits)
    for status in ("ok", "no_response", "failed", "skipped"):
        assert first[f"n_{status}"] == statuses.count(status)
    lat = out[out["parameter"] == "latency"].iloc[0]
    ok_lat = table.loc[table["status"] == "ok", "latency"].dropna()
    assert lat["n"] == len(ok_lat) and lat["mean"] == pytest.approx(ok_lat.mean())
    assert lat["ci_low"] <= lat["mean"] <= lat["ci_high"]
    # the long format gives the same counts
    long = to_long(group, units="uV")
    out_long = group_summary(long, by=["condition"], parameters=["latency"], random_state=0)
    assert out_long[["n_ok", "n_failed", "n_skipped", "n_channels"]].values.tolist() == \
        out[out["parameter"] == "latency"][["n_ok", "n_failed", "n_skipped",
                                            "n_channels"]].values.tolist()
    with pytest.raises(SettingsError, match="region"):
        group_summary(table, by=["region"])
    bad = table.assign(status="maybe")
    with pytest.raises(DataError):
        group_summary(bad, by=["channel"])
