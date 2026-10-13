# SPDX-License-Identifier: Apache-2.0
"""Trial table, modes, persistence and edge cases (REQ-157)."""

import warnings
from typing import Any

import numpy as np
import pytest

from evoparam.errors import SettingsError
from evoparam.trials import TrialFitSet, fit_trials, realign

from .helpers import channel_fit, gauss, simulate

pd = pytest.importorskip("pandas")

###################################################################################################
###################################################################################################


def test_req157_table_shape_flags_and_round_trip(tmp_path: Any) -> None:

    meta = pd.DataFrame({"stim_site": ["A-B"] * 30, "index_in_train": np.arange(30)})
    ts, _, _ = simulate(n=30, channels=3, metadata=meta, seed=1)
    ts.data[:, 1] += 20 * gauss(0.2, 0.05)[None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = channel_fit(ts)
        tf = fit_trials(ts, group, covariates=("trend",))
    n_rows = sum(len(f.settings["picks"]) * 30 for f in tf.fits)
    assert len(tf.table) == n_rows
    assert not tf.table[["converged", "trusted"]].isna().any().any()
    assert "c2" in tf.skipped_channels
    assert {"index_in_train", "stim_site"} <= set(tf.table.columns)
    assert "trend" in tf.settings["covariate_tables"]
    path = tf.save(str(tmp_path / "trials"))
    loaded = TrialFitSet.load(path)
    pd.testing.assert_frame_equal(
        loaded.table.reset_index(drop=True), tf.table.reset_index(drop=True), check_dtype=False
    )
    assert np.array_equal(loaded.fits[0].shift, tf.fits[0].shift)


@pytest.mark.parametrize("mode", ["free", "template", "regularized"])
def test_other_modes_run(mode: Any) -> None:

    ts, shifts, _ = simulate(n=20, jitter=0.003, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tf = fit_trials(ts, channel_fit(ts), mode=mode, r2_min=0.0)
    fit = tf.fits[0]
    assert fit.converged.mean() >= 0.9
    # the template mode uses the jitter-blurred channel average as template: less precise
    assert np.corrcoef(fit.shift[:, 0], shifts)[0, 1] >= (0.35 if mode == "template" else 0.5)


def test_realign_and_errors() -> None:

    ts, _, _ = simulate(n=40, jitter=0.006, cv=0.0, fwhm=0.03, noise=3.0, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tf = fit_trials(ts, channel_fit(ts), max_shift=0.02)
    evoked = realign(ts, tf)[0]
    plain = ts.data[:, 0].mean(axis=0)
    assert np.min(evoked.data[0]) < np.min(plain)  # sharper, deeper peak
    with pytest.raises(SettingsError):
        fit_trials(ts, mode="magic")
    with pytest.raises(SettingsError):
        fit_trials(ts, channel_fit(ts), covariates=("weather",))
    with pytest.raises(SettingsError, match="three cycles"):
        fit_trials(
            ts,
            channel_fit(ts),
            covariates=("prestim",),
            prestim_window=(-0.1, -0.01),
            prestim_band=(2.0, 4.0),
        )


def test_options_are_checked_and_recorded() -> None:

    ts, _, _ = simulate(n=20, jitter=0.003, seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = channel_fit(ts)
        for name in ("trial_model", "null_rule", "ridge_cv"):
            with pytest.raises(SettingsError, match=name):
                fit_trials(ts, group, **{name: "magic"})
        tf = fit_trials(ts, group, trial_model="selected", null_rule="spec", r2_min=0.0)
    assert tf.settings["trial_model"] == "selected"
    assert tf.settings["null_rule"] == "spec"
    assert tf.settings["ridge_cv"] == "loto"
    assert np.isfinite(tf.fits[0].jitter_sd_null[0])


def test_selected_trial_model_keeps_only_the_chosen_components() -> None:

    ts, shifts, _ = simulate(n=30, jitter=0.003, seed=5)
    ts.data[:, 0] += 15 * gauss(0.2, 0.06)[None]  # a second, later component
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = channel_fit(ts)
        comps = group.fits[0].components
        assert len(comps) >= 2
        first = int(np.argmin([abs(c.latency - 0.08) for c in comps]))
        full = fit_trials(ts, group, components=[first], r2_min=0.0)
        only = fit_trials(ts, group, components=[first], r2_min=0.0, trial_model="selected")
    # both report the chosen component only, but the models differ
    assert full.fits[0].shift.shape == only.fits[0].shift.shape == (30, 1)
    assert not np.allclose(full.fits[0].shift, only.fits[0].shift)
    for fit in (full, only):
        assert np.corrcoef(fit.fits[0].shift[:, 0], shifts)[0, 1] >= 0.5


@pytest.mark.parametrize("anchored", [False, True])
def test_vectorised_solver_matches_trial_by_trial_solver(anchored: Any) -> None:
    """The shift_scale solver vectorised over trials reaches the minimum that the bounded
    least-squares solver of one trial reaches (same model, same bounds), with every
    component moving or with the first one only (trial_model="anchored")."""

    from evoparam.trials._model import fit_trial, fit_trials_batch
    from evoparam.trials.api import _template_for

    ts, _, _ = simulate(n=30, jitter=0.004, seed=6)
    ts.data[:, 0] += 15 * gauss(0.2, 0.06)[None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = channel_fit(ts).fits[0]
        template = _template_for(fit, list(range(len(fit.components))), ts.times, 0.01, 0.25)
    window = (ts.times >= template.tw[0]) & (ts.times <= template.tw[-1])
    x = ts.data[:, 0][:, window]
    movable = [0] if anchored else None
    batch = fit_trials_batch(template, x, movable=movable)
    for xi, b in zip(x, batch, strict=False):
        one = fit_trial(template, xi, movable=movable)
        assert b["converged"]
        assert one["converged"]
        cost_b, cost_one = np.sum((xi - b["fitted"]) ** 2), np.sum((xi - one["fitted"]) ** 2)
        assert cost_b <= cost_one * (1 + 1e-6)
        assert np.allclose(b["shift"], one["shift"], atol=2e-5)
