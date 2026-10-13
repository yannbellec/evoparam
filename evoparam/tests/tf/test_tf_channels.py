# SPDX-License-Identifier: Apache-2.0
"""Corrections across channels of the time-frequency statistics (spec §5.10 method 4: "as in
§5.6", whose default is fdr_bh)."""

import inspect
from typing import Any

import numpy as np
import pytest

from evoparam.errors import SettingsError
from evoparam.tests.tf.helpers import TIMES, ar1, burst, trialset
from evoparam.tf import tf_fit
from evoparam.tf.stats import bh_adjusted, fdr_significant, row_levels

# Channels of the correction test, the error bound over 30 runs, the most channels
# selected, the responsive channel, and alpha.
N_CHANNELS = 2
MAX_FAMILY_ERROR = 0.1
MAX_SELECTED = 2
RESPONSIVE = 2
ALPHA = 0.05

FREQS = (4, 250, 20)


def test_default_is_fdr_bh_and_every_correction_of_5_6_is_offered() -> None:

    assert inspect.signature(tf_fit).parameters["channel_correction"].default == "fdr_bh"
    ts = trialset(ar1(np.random.default_rng(0), (10, 2, len(TIMES))))
    for method in ("fdr_bh", "holm", "bonferroni", "none"):
        tf = tf_fit(ts, freqs=(4, 250, 6), zones="none", channel_correction=method)
        assert tf.settings["channel_correction"] == method
        assert len(tf.settings["row_alpha"]) == N_CHANNELS
    with pytest.raises(SettingsError, match="channel_correction"):
        tf_fit(ts, freqs=(4, 250, 6), zones="none", channel_correction="sidak")


def test_row_levels() -> None:

    p = np.array([0.001, 0.02, 0.035, 0.3, np.nan])
    assert np.allclose(row_levels(p, 0.05, "none"), [0.05] * 4 + [0])
    assert np.allclose(row_levels(p, 0.05, "bonferroni"), [0.0125] * 4 + [0])
    # Holm: 0.001 <= 0.05 / 4, 0.02 > 0.05 / 3: stop
    assert np.allclose(row_levels(p, 0.05, "holm"), [0.0125, 0, 0, 0, 0])
    # BH over the four rows: 0.035 <= 3 * 0.05 / 4 -> three rows selected, at 3 * 0.05 / 4
    assert np.allclose(row_levels(p, 0.05, "fdr_bh"), [0.0375] * 3 + [0, 0])
    assert np.allclose(row_levels([0.2, 0.3], 0.05, "fdr_bh"), [0, 0])


def test_bh_adjusted_matches_the_step_up_procedure() -> None:

    rng = np.random.default_rng(1)
    p = np.concatenate([rng.uniform(0, 1, 300), rng.uniform(0, 1e-3, 20), [np.nan]])
    for q in (0.01, 0.05, 0.2):
        assert np.array_equal(bh_adjusted(p) <= q, fdr_significant(p, q))


def test_bonferroni_is_alpha_over_rows_on_each_row() -> None:
    """Bonferroni keeps the rule of the first version: Benjamini-Hochberg over the pixels of
    each row at alpha / n_rows."""

    rng = np.random.default_rng(2)
    data = ar1(rng, (30, 3, len(TIMES)))
    data[:, 0] += burst(rng, 30, amp=12.0)
    tf = tf_fit(
        trialset(data), freqs=FREQS, zones="none", channel_correction="bonferroni", random_state=0
    )
    assert tf.significant[0].any()
    assert np.allclose(tf.settings["row_alpha"], 0.05 / 3)


def _null_family_error(method: Any, runs: Any = 30, n_channels: Any = 8) -> Any:
    hits = 0
    for seed in range(runs):
        rng = np.random.default_rng(3000 + seed)
        ts = trialset(ar1(rng, (30, n_channels, len(TIMES))))
        tf = tf_fit(ts, freqs=FREQS, zones="none", channel_correction=method, random_state=seed)
        hits += bool(tf.significant.any())
    return hits / runs


def test_fdr_bh_across_channels_controls_errors_on_null_channels() -> None:
    """Under the complete null every selected channel is a false discovery: the
    false-discovery rate over channels is the family-wise error, at most alpha (0.05 plus
    the sampling error of 30 runs); without correction it grows with the channels."""

    corrected, uncorrected = _null_family_error("fdr_bh"), _null_family_error("none")
    assert corrected <= MAX_FAMILY_ERROR, corrected
    assert uncorrected >= corrected


def test_fdr_bh_keeps_a_responsive_channel_among_null_ones() -> None:

    rng = np.random.default_rng(4)
    data = ar1(rng, (30, 8, len(TIMES)))
    data[:, 2] += burst(rng, 30, amp=12.0)
    tf = tf_fit(trialset(data), freqs=FREQS, zones="none", random_state=0)
    responsive = tf.significant.reshape(8, -1).any(axis=1)
    assert responsive[2]
    assert responsive.sum() <= MAX_SELECTED
    p = np.array(tf.settings["row_p_value"], float)
    assert np.nanargmin(p) == RESPONSIVE
    # the pixels of the selected row are those of Benjamini-Hochberg at its level
    level = tf.settings["row_alpha"][2]
    assert level == pytest.approx(responsive.sum() * 0.05 / 8)
    assert "Benjamini and Bogomolov" in tf.methods_text()


def test_one_row_is_tested_at_alpha() -> None:

    rng = np.random.default_rng(5)
    data = ar1(rng, (30, 1, len(TIMES))) + burst(rng, 30, amp=12.0)[:, None]
    ts = trialset(data)
    tf = tf_fit(ts, freqs=FREQS, zones="none", random_state=0)
    assert np.isfinite(tf.settings["row_p_value"][0])
    assert tf.settings["row_p_value"][0] <= ALPHA
    # one row: fdr_bh across channels is Benjamini-Hochberg over the row's pixels at alpha
    assert tf.settings["row_alpha"] == [0.05]
