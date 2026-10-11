# SPDX-License-Identifier: Apache-2.0
"""tf_fit options, edge cases, conditions, zone methods, comparisons and round trips."""

import warnings

import numpy as np
import pandas as pd
import pytest

from evoparam.errors import SettingsError
from evoparam.models import fit as fit_models
from evoparam.tests.tf.helpers import SFREQ, TIMES, ar1, burst, n1, trialset
from evoparam.tf import tf_compare, tf_fit, tf_zones


def _burst_set(seed=0, n=30, amp=10.0, channels=1, freq=80.0):
    rng = np.random.default_rng(seed)
    data = ar1(rng, (n, channels, len(TIMES)))
    data[:, 0] += burst(rng, n, freq=freq, amp=amp)
    return trialset(data)


def test_frequencies_above_sfreq_over_2_5_are_dropped():

    ts = _burst_set()
    with pytest.warns(UserWarning, match="dropped"):
        tf = tf_fit(ts, freqs=np.array([10.0, 100.0, 390.0, 450.0]), zones="none")
    assert np.all(tf.freqs <= SFREQ / 2.5)


def test_full_epoch_zscore_is_undefined():

    with pytest.raises(SettingsError, match="full_epoch"):
        tf_fit(_burst_set(), baseline_level="full_epoch", baseline_mode="zscore")


def test_group_fit_on_another_channel_set_is_refused():

    ts = _burst_set(channels=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = fit_models(ts, "ccep", picks=["c0"])
    with pytest.raises(SettingsError, match="different channel set"):
        tf_fit(ts, group_fit=group)


def test_cluster_with_few_trials_warns():

    with pytest.warns(UserWarning, match="few distinct patterns"):
        tf_fit(_burst_set(n=8), stats="cluster", n_permutations=50, zones="none")


@pytest.mark.parametrize("mode", ["db", "percent", "zscore", "logratio", "ratio"])
@pytest.mark.parametrize("level", ["single_trial", "evoked", "full_epoch"])
def test_every_normalisation_finds_the_burst(mode, level):

    if level == "full_epoch" and mode == "zscore":
        pytest.skip("undefined")
    tf = tf_fit(_burst_set(amp=12.0), baseline_mode=mode, baseline_level=level, zones="none",
                freqs=(4, 250, 30))
    box = ((tf.freqs[:, None] >= 65) & (tf.freqs[:, None] <= 100)
           & (tf.times[None] >= 0.2) & (tf.times[None] <= 0.3))
    assert tf.significant[0][box].mean() > 0.3
    centre = 1.0 if mode == "ratio" else 0.0
    assert np.nanmean(tf.maps["total_norm"][0][box]) > centre


def test_conditions_are_rows_and_zone_metrics_join_trials():

    rng = np.random.default_rng(5)
    data = ar1(rng, (40, 2, len(TIMES)))
    data[:20, 0] += burst(rng, 20, amp=12.0)
    meta = pd.DataFrame({"stim_site": ["A"] * 20 + ["B"] * 20})
    ts = trialset(data, metadata=meta)
    tf = tf_fit(ts, by_trial=True, freqs=(4, 250, 30), random_state=0)
    assert tf.settings["rows"] == [["c0", "A"], ["c1", "A"], ["c0", "B"], ["c1", "B"]]
    assert tf.tmaps.shape[0] == 4
    assert len(tf.zones) >= 1
    m = tf.zone_metrics
    zone0 = m[(m["zone_id"] == 0) & (m["trial"] < 0)].set_index(["channel", "condition"])
    assert zone0.loc[("c0", "A"), "t"] > 3
    assert zone0.loc[("c0", "A"), "value"] > zone0.loc[("c0", "B"), "value"]
    trials = m[(m["zone_id"] == 0) & (m["trial"] >= 0) & (m["channel"] == "c0")]
    assert sorted(trials["trial"]) == list(range(40))
    assert tf.zones["p_value"].iloc[0] < 0.05


def test_two_separate_bursts_give_two_zones_and_cp_agrees():

    rng = np.random.default_rng(2)
    n = 30
    data = ar1(rng, (n, 3, len(TIMES)))
    for c in range(3):
        data[:, c] += burst(rng, n, freq=80.0, centre=0.15, amp=12.0)
        data[:, c] += burst(rng, n, freq=20.0, centre=0.45, duration=0.2, amp=12.0)
    tf = tf_fit(trialset(data), freqs=(4, 250, 40), random_state=0)
    pos = tf.zones[tf.zones["sign"] > 0]
    assert len(pos) >= 2
    centroids = sorted(pos["freq_centroid"].iloc[:2])
    assert abs(np.log2(centroids[0] / 20.0)) < 0.3 and abs(np.log2(centroids[1] / 80.0)) < 0.3
    cp = tf_zones(tf, method="cp", random_state=0)
    cp_pos = cp[cp["sign"] > 0]
    assert len(cp_pos) >= 2
    assert sorted(np.round(np.log2(cp_pos["freq_centroid"].iloc[:2] / 20.0))) == [0.0, 2.0]


def test_null_data_gives_no_zones():

    ts = trialset(ar1(np.random.default_rng(11), (30, len(TIMES))))
    tf = tf_fit(ts, freqs=(4, 250, 30), random_state=0)
    assert len(tf.zones) == 0
    assert len(tf.zone_metrics) == 0


def test_footprint_maps_and_residual_of_a_sharp_n1():

    rng = np.random.default_rng(4)
    ts = trialset(ar1(rng, (30, len(TIMES))) + n1()[None])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = fit_models(ts, "ccep")
    tf = tf_fit(ts, group_fit=group, freqs=(4, 250, 30), zones="none", random_state=0)
    high = tf.freqs >= 50
    at_n1 = np.argmin(np.abs(tf.times - 0.025))
    assert np.nanmax(tf.maps["footprint_norm"][0][high, at_n1]) > 3
    assert np.nanmax(np.abs(tf.maps["residual_t"][0][high])) < 6


def test_compare_two_conditions():

    a = tf_fit(_burst_set(seed=1, channels=4, amp=12.0), zones="none", freqs=(4, 250, 20))
    rng = np.random.default_rng(9)
    b = tf_fit(trialset(ar1(rng, (30, 4, len(TIMES)))), zones="none", freqs=(4, 250, 20))
    table = tf_compare(a, b, n_permutations=100, random_state=0)
    assert list(table.columns) == ["sign", "t_start", "t_end", "f_low", "f_high", "mass",
                                   "n_pixels", "p_value"]
    with pytest.raises(SettingsError):
        tf_compare(a, tf_fit(_burst_set(), zones="none", freqs=(4, 250, 10)))


def test_tffit_round_trip(tmp_path):

    from evoparam.data.serialize import load_object, save_object

    tf = tf_fit(_burst_set(amp=12.0), freqs=(4, 250, 20), random_state=0)
    path = save_object(tf, tmp_path / "tf.evoparam")
    back = load_object(path)
    assert np.array_equal(back.significant, tf.significant)
    np.testing.assert_allclose(back.maps["total_norm"], tf.maps["total_norm"], equal_nan=True)
    assert back.settings["rows"] == tf.settings["rows"]
    pd.testing.assert_frame_equal(back.zones.reset_index(drop=True),
                                  tf.zones.reset_index(drop=True), check_dtype=False)


def test_zone_metrics_export_to_the_long_format():

    from evoparam.stats import to_long

    tf = tf_fit(_burst_set(amp=12.0), freqs=(4, 250, 20), by_trial=True, random_state=0)
    table = to_long(tf)
    values = table[table["parameter"] == "value"]
    assert set(values["unit"]) == {"dB"}
    assert {"channel", "condition", "trial", "zone_id"} <= set(table.columns)
    assert len(values) == 31 * len(tf.zones)


def _blob(freq_index, time_index, shape=(16, 60)):
    f = np.exp(-0.5 * ((np.arange(shape[0]) - freq_index) / 1.5) ** 2)
    t = np.exp(-0.5 * ((np.arange(shape[1]) - time_index) / 4.0) ** 2)
    return np.outer(f, t)


def test_cp_rank_is_chosen_on_split_halves():
    """Spec §5.10 method 5: the CP rank by split-half stability, on halves of each row's
    trials (default) or of the rows (channels x conditions, the sample mode of the
    tensor). Two parts present in eight rows with
    their own weights give rank 2 on halves of the trials and of the rows; the rule of the
    first version (restarts) stays available."""

    from evoparam.tf.zones import cp_split_values, select_rank

    rng = np.random.default_rng(0)
    a, b = _blob(3, 15), _blob(11, 40)
    weights = rng.uniform(0.5, 1.5, (2, 8))
    tensor = a[..., None] * weights[0] + b[..., None] * weights[1]
    group = tensor.mean(axis=-1)
    rank, stability = select_rank(group, None, 4, np.random.default_rng(1), method="cp",
                                  tensor=tensor)
    assert rank == 2 and stability[2] >= 0.8
    values = cp_split_values(tensor, None, 2, np.random.default_rng(2))
    assert len(values) == 20 and np.mean(values) >= 0.8
    # trial halves of every row (the default when the TFFit holds them)
    noisy = [tensor * rng.uniform(0.8, 1.2, tensor.shape) for _ in range(40)]
    row_halves = np.stack(noisy).reshape(20, 2, *tensor.shape)
    rank_t, stability_t = select_rank(group, None, 4, np.random.default_rng(1), method="cp",
                                      tensor=tensor, row_halves=row_halves)
    assert rank_t == 2 and stability_t[2] >= 0.8
    rank_r, _ = select_rank(group, None, 4, np.random.default_rng(1), method="cp",
                            tensor=tensor, cp_stability="restarts")
    assert rank_r >= 1
    # a single row carrying the map: halves of the trials, else restarts
    single = np.zeros_like(tensor)
    single[..., 0] = a + b
    assert cp_split_values(single, None, 2, rng) is None
    halves = np.stack([np.stack([a + b, a + b])] * 20)
    assert np.mean(cp_split_values(single, halves, 2, rng)) >= 0.8
    with pytest.raises(SettingsError, match="cp_stability"):
        select_rank(group, None, 4, rng, method="cp", tensor=tensor, cp_stability="both")
