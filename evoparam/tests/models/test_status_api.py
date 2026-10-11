# SPDX-License-Identifier: Apache-2.0
"""Status semantics, reproducibility, tidy output, methods text and the API (REQ-118 to
REQ-120, REQ-123)."""

import numpy as np
import pandas as pd
import pytest

import evoparam
from evoparam.data import GroupFit, TrialSet
from evoparam.errors import DataError, SettingsError
from evoparam.models import (ResponseModel, classical_measures, compare_models, fit,
                             fit_response, label_components, methods_text)

from .helpers import BASELINE, SFREQ, TIMES, WINDOW, gauss, model, noise

###################################################################################################
###################################################################################################

def _trials(n=12, seed=0, channels=3):
    rng = np.random.default_rng(seed)
    sig = gauss(0.025, -100, 0.012) + gauss(0.15, -50, 0.08)
    data = sig[None, None] + rng.normal(0, 10, (n, channels, len(TIMES)))
    data[:, -1] = rng.normal(0, 10, (n, len(TIMES)))            # a channel without response
    return TrialSet(data=data, times=TIMES, sfreq=SFREQ, ch_names=[f"c{i}" for i in range(channels)])


def test_req118_status_semantics():

    rng = np.random.default_rng(0)
    none = model().fit_waveform(TIMES, noise(rng))
    assert none.status == "no_response" and "best rejected" in none.message
    assert "best_rejected_delta" in none.diagnostics
    flat = model().fit_waveform(TIMES, np.zeros(len(TIMES)))
    assert flat.status == "skipped" and flat.message == "flat"
    masked = model().fit_waveform(TIMES, noise(rng), mask=TIMES > 0)
    assert masked.status == "skipped"
    ok = model().fit_waveform(TIMES, gauss(0.2, -8, 0.04) + noise(rng))
    assert ok.status == "ok"
    for comp in ok.components:
        assert np.all(np.isfinite(comp.params)) and np.isfinite(comp.latency)


def test_req119_req120_reproducible_tidy_and_saved(tmp_path):

    ts = _trials()
    a = fit(ts, "ccep", random_state=0)
    b = fit(ts, "ccep", random_state=0)
    assert a.to_dataframe().equals(b.to_dataframe())
    table = a.to_dataframe()
    assert set(table["channel"]) == {"c0", "c1", "c2"}
    assert not any(isinstance(v, (list, dict)) for v in table.iloc[0])
    path = a.save(tmp_path / "fit.evoparam")
    loaded = GroupFit.load(path)
    assert loaded.to_dataframe().equals(table)
    labels = a.get("c0").components
    assert any(c.label == "N1" for c in labels)
    assert a.get("c2").status == "no_response"


def test_req119_reproducible_with_parallel_jobs():
    """REQ-119 with n_jobs > 1: two runs with the same random_state give byte-identical
    to_dataframe() output, and the same output as n_jobs=1."""

    ts = _trials(channels=4)
    tables = [fit(ts, "ccep", random_state=0, n_jobs=jobs).to_dataframe()
              for jobs in (2, 2, 4, 1)]
    text = [t.to_csv(float_format="%.17g").encode() for t in tables]
    assert text[0] == text[1] == text[2] == text[3]
    assert all(t.equals(tables[0]) for t in tables)


def test_neff_noise_model_is_available():
    """F64: the noise model of spec §5.5.3 step 1 (residuals over sigma_n, the AR(1)
    effective count n_eff in the criterion and the covariance) is selectable and named in the
    methods text."""

    rng = np.random.default_rng(5)
    y = gauss(0.2, -8, 0.04) + noise(rng, 0.9)
    spec = model(noise_model="neff").fit_waveform(TIMES, y)
    rho = spec.diagnostics["rho"]
    n = int(np.sum((TIMES >= WINDOW[0]) & (TIMES <= WINDOW[1])))
    assert spec.diagnostics["n_white"] == pytest.approx(n * (1 - rho) / (1 + rho))
    assert spec.gof.n_eff == pytest.approx(n * (1 - rho) / (1 + rho))
    assert "n_eff correction" in methods_text(spec)
    with pytest.raises(SettingsError):
        model(noise_model="nope").fit_waveform(TIMES, y)


def test_req123_methods_text():

    fit_ = fit_response(TIMES, gauss(0.2, -8, 0.04) + noise(np.random.default_rng(1)),
                        window=WINDOW, baseline=BASELINE, preset="erp")
    text = methods_text(fit_)
    assert text.startswith("Evoked responses were parameterized with evoparam")
    assert "preset 'erp'" in text and "BIC" in text and "constant background" in text


def test_top_level_fit_inputs_and_errors():

    ts = _trials()
    from evoparam.existence import fit as fit_with_existence
    assert evoparam.fit is fit_with_existence
    by = TrialSet(data=ts.data, times=ts.times, sfreq=ts.sfreq, ch_names=ts.ch_names,
                  metadata=pd.DataFrame({"condition": ["a", "b"] * 6}))
    grouped = fit(by, "ccep", by="condition")
    assert set(grouped.index["condition"]) == {"a", "b"}
    arr = fit(ts.data.mean(0), "ccep", times=TIMES)
    assert len(arr.fits) == 3
    with pytest.raises(SettingsError):
        fit(ts, "ccep", trials=True)
    with pytest.raises(SettingsError):
        fit(ts, "nope")
    with pytest.raises(SettingsError):
        ResponseModel(fit_window=(0.0, 0.002), baseline=BASELINE).fit_waveform(TIMES, noise(
            np.random.default_rng(0)))
    with pytest.raises(DataError):
        ResponseModel(fit_window=(0.5, 2.0)).fit_waveform(TIMES, noise(np.random.default_rng(0)))


def test_estimator_contract_and_compare():

    m = ResponseModel("ccep", max_components=4)
    assert m.get_params()["max_components"] == 4
    m.set_params(max_components=3)
    assert m.get_params()["max_components"] == 3
    trials = _trials()
    m.fit(trials)
    # scikit-learn semantics (spec §10.4): samples are the channels, features the time points
    assert m.n_features_in_ == trials.n_times and m.transform().shape == (3, 3)
    assert np.isfinite(m.score())
    rng = np.random.default_rng(2)
    y = gauss(0.2, -8, 0.04) + noise(rng)
    table = compare_models([model(shapes=(s,)).fit_waveform(TIMES, y)
                            for s in ("gaussian", "skewed_gaussian")])
    assert list(table["delta_bic"]).count(0.0) == 1


def test_uncertainty_levels():

    ts = _trials(n=20)
    m = ResponseModel("ccep", uncertainty="bootstrap", n_bootstrap=30, random_state=0)
    group = m.fit_group(ts, picks=["c0"])
    comp = group.fits[0].components[0]
    assert comp.ci["latency"][0] < comp.latency < comp.ci["latency"][1]
    y = gauss(0.2, -8, 0.04) + noise(np.random.default_rng(3))
    prof = model(shapes=("gaussian",), uncertainty="profile").fit_waveform(TIMES, y)
    lo, hi = prof.components[0].ci["mu"]
    assert lo < prof.components[0].params[0] < hi
    with pytest.raises(SettingsError):
        model(uncertainty="posterior").fit_waveform(TIMES, y)


def test_classical_measures_and_labels():

    rng = np.random.default_rng(4)
    trials = gauss(0.1, -8, 0.04)[None] + rng.normal(0, 2, (30, len(TIMES)))
    y = trials.mean(0)
    scores = classical_measures(TIMES, y, window=(0.05, 0.15), trials=trials, n_bootstrap=100,
                                random_state=0)
    assert abs(scores["peak_latency"] - 0.1) <= 0.01
    inside = (TIMES >= 0.05) & (TIMES <= 0.15)
    assert np.isclose(scores["sme"]["mean_amplitude"],
                      np.std(trials[:, inside].mean(1), ddof=1) / np.sqrt(30))
    assert scores["definition"]["peak_latency"].startswith("local_peak(n_neighbors=3)")
    clean = gauss(0.1, -8, 0.04)
    assert abs(classical_measures(TIMES, clean, window=(0.0, 0.2),
                                  measures=("fractional_area_latency",))
               ["fractional_area_latency"] - 0.1) < 1e-3
    fit_ = model().fit_waveform(TIMES, gauss(0.025, -8, 0.012) + gauss(0.15, -6, 0.08)
                                + noise(rng))
    labelled = label_components(fit_, "ccep")
    assert {c.label for c in labelled.components} >= {"N1", "N2"}
