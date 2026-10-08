"""Trial table, modes, persistence and edge cases (REQ-157)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from evoparam.errors import SettingsError
from evoparam.trials import TrialFitSet, fit_trials, realign

from .helpers import channel_fit, gauss, simulate

###################################################################################################
###################################################################################################

def test_req157_table_shape_flags_and_round_trip(tmp_path):

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
    pd.testing.assert_frame_equal(loaded.table.reset_index(drop=True),
                                  tf.table.reset_index(drop=True), check_dtype=False)
    assert np.array_equal(loaded.fits[0].shift, tf.fits[0].shift)


@pytest.mark.parametrize("mode", ["free", "template", "regularized"])
def test_other_modes_run(mode):

    ts, shifts, _ = simulate(n=20, jitter=0.003, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tf = fit_trials(ts, channel_fit(ts), mode=mode, r2_min=0.0)
    fit = tf.fits[0]
    assert fit.converged.mean() >= 0.9
    # the template mode uses the jitter-blurred channel average as template: less precise
    assert np.corrcoef(fit.shift[:, 0], shifts)[0, 1] >= (0.35 if mode == "template" else 0.5)


def test_realign_and_errors():

    ts, _, _ = simulate(n=40, jitter=0.006, cv=0.0, fwhm=0.03, noise=3.0, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tf = fit_trials(ts, channel_fit(ts), max_shift=0.02)
    evoked = realign(ts, tf)[0]
    plain = ts.data[:, 0].mean(axis=0)
    assert np.min(evoked.data[0]) < np.min(plain)          # sharper, deeper peak
    with pytest.raises(SettingsError):
        fit_trials(ts, mode="magic")
    with pytest.raises(SettingsError):
        fit_trials(ts, channel_fit(ts), covariates=("weather",))
    with pytest.raises(SettingsError, match="three cycles"):
        fit_trials(ts, channel_fit(ts), covariates=("prestim",), prestim_window=(-0.1, -0.01),
                   prestim_band=(2.0, 4.0))
