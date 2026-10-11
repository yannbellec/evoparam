# SPDX-License-Identifier: Apache-2.0
"""Published zones registry (REQ-217) and the compatibility layer."""

import numpy as np
import pytest

from evoparam.errors import SettingsError
from evoparam.tests.tf.helpers import TIMES, ar1, burst, trialset
from evoparam.tf import list_published_zones, tf_fit
from evoparam.tf.stats import baseline_normal_map
from evoparam.tf.zones import load_zones, resolve_published, volume_conduction_weight, zone_mask

VERIFIED = "brinyark2026_delayed_high_gamma"
UNVERIFIED = "brinyark2026_TN1"
# The eleven zones of Brinyark et al. 2026 ([LIT-tf-4], §2.4 and Fig. 2): five bands x N1
# (15-50 ms) and N2 (50-500 ms), and the whole response (4-250 Hz x 15-500 ms).
BANDS = {"T": (4.0, 8.0), "A": (8.0, 13.0), "B": (13.0, 30.0), "Gl": (30.0, 50.0),
         "Gh": (50.0, 250.0)}
INTERVALS = {"N1": (0.015, 0.05), "N2": (0.05, 0.5)}


def test_registry_lists_every_zone_with_status_and_source():

    table = list_published_zones()
    assert set(table["name"]) == set(load_zones())
    assert table.set_index("name").loc[VERIFIED, "verified"]
    assert not table.set_index("name").loc[UNVERIFIED, "verified"]
    assert all(table["source"].str.len() > 0)


def test_req217_unverified_zone_needs_allow_unverified():

    ts = trialset(ar1(np.random.default_rng(0), (10, len(TIMES))))
    with pytest.raises(SettingsError, match=UNVERIFIED) as err:
        tf_fit(ts, zones=f"published:{UNVERIFIED}")
    assert "Brinyark" in str(err.value)
    assert "allow_unverified=True" in str(err.value)


def test_zone_without_bounds_is_refused_even_when_allowed(monkeypatch):

    import evoparam.tf.zones as zones

    registry = load_zones()
    registry["empty"] = {"time": None, "freq": None, "verified": False, "source": "nowhere"}
    monkeypatch.setattr(zones, "load_zones", lambda: registry)
    with pytest.raises(SettingsError, match="no bounds"):
        resolve_published("empty", allow_unverified=True)


def test_the_eleven_published_zones_have_the_article_bounds():

    registry = load_zones()
    for band, freq in BANDS.items():
        for interval, time in INTERVALS.items():
            zone = registry[f"brinyark2026_{band}{interval}"]
            assert zone["freq"] == list(freq) and zone["time"] == list(time)
            assert "10.1016/j.clinph.2026.2111855" in zone["source"]
    whole = registry["brinyark2026_WR"]
    assert whole["freq"] == [4.0, 250.0] and whole["time"] == [0.015, 0.5]
    # the ten other zones are unverified (spec §5.10: confirmed by the user before use)
    others = [n for n in registry if n.startswith("brinyark2026_") and
              registry[n].get("alias_of") is None and n != VERIFIED]
    assert len(others) == 10 and not any(registry[n]["verified"] for n in others)
    table = list_published_zones().set_index("name")
    assert table.loc["brinyark2026_GhN1", "intervals"] == "[)x[]"


def test_ghn2_is_the_verified_delayed_high_gamma_zone():

    alias = resolve_published("brinyark2026_GhN2")
    zone = resolve_published(VERIFIED)
    assert alias["time"] == zone["time"] == [0.05, 0.5]
    assert alias["freq"] == zone["freq"] == [50.0, 250.0]
    assert alias["settings"] == zone["settings"]
    freqs, times = np.geomspace(4, 250, 40), np.arange(0, 0.6, 0.001)
    assert np.array_equal(zone_mask(alias, freqs, times), zone_mask(zone, freqs, times))


def test_band_and_interval_zones_tile_the_whole_response():
    """Half-open convention: 50 ms belongs to N2 and each band edge to the higher band, so
    every pixel of the whole response lies in exactly one of the ten zones."""

    registry = load_zones()
    freqs = np.array([4.0, 6.0, 8.0, 10.0, 13.0, 20.0, 30.0, 40.0, 50.0, 100.0, 250.0])
    times = np.round(np.arange(0.0, 0.6, 0.001), 6)
    count = sum(zone_mask(registry[f"brinyark2026_{b}{i}"], freqs, times).astype(int)
                for b in BANDS for i in INTERVALS)
    whole = zone_mask(registry["brinyark2026_WR"], freqs, times)
    assert np.array_equal(count, whole.astype(int))
    at_50ms = np.flatnonzero(times == 0.05)[0]
    assert zone_mask(registry["brinyark2026_GhN2"], freqs, times)[-1, at_50ms]
    assert not zone_mask(registry["brinyark2026_GhN1"], freqs, times)[-1, at_50ms]
    eight = np.flatnonzero(freqs == 8.0)[0]
    assert zone_mask(registry["brinyark2026_AN2"], freqs, times)[eight, 100]
    assert not zone_mask(registry["brinyark2026_TN2"], freqs, times)[eight, 100]


def test_article_zone_uses_the_published_statistic_and_significance_fraction():

    rng = np.random.default_rng(7)
    ts = trialset(ar1(rng, (30, len(TIMES))) + burst(rng, 30, freq=100.0, centre=0.03,
                                                      duration=0.04, amp=15.0))
    with pytest.raises(SettingsError, match="unverified"):
        tf_fit(ts, zones="published:brinyark2026_GhN1")
    tf = tf_fit(ts, zones="published:brinyark2026_GhN1", allow_unverified=True,
                freqs=(4, 250, 40), random_state=0)
    s = tf.settings
    assert s["stats"] == "baseline_normal" and s["baseline"] == [-0.45, -0.15]
    assert s["channel_correction"] == "none" and s["window"][0] == 0.015
    assert s["transform"] == "morse" and s["transform_params"]["beta"] == 10.0
    region = zone_mask(resolve_published("brinyark2026_GhN1", True), tf.freqs, tf.times)
    cells = region & np.isfinite(tf.maps["total_norm"][0])
    row = tf.zone_metrics.iloc[0]
    # significance fraction: significant pixels / pixels of the zone (Fig. 2C)
    assert row["fraction_significant"] == pytest.approx(tf.significant[0][cells].mean())
    assert row["fraction_significant"] > 0.2
    assert "baseline of its frequency" in tf.methods_text()


def test_baseline_normal_map_against_an_independent_computation():

    rng = np.random.default_rng(8)
    power = rng.gamma(2.0, 1.0, (12, 3, 50))
    baseline = np.zeros(50, bool)
    baseline[:20] = True
    window = np.zeros((3, 50), bool)
    window[:, 25:] = True
    z, p = baseline_normal_map(power, baseline, window)
    from statistics import NormalDist, mean, stdev

    db = (10 * np.log10(power)).mean(axis=0)
    for f in range(3):
        mu, sd = mean(db[f, :20]), stdev(db[f, :20])
        for k in (25, 49):
            ref = (db[f, k] - mu) / sd
            assert z[f, k] == pytest.approx(ref, rel=1e-12)
            assert p[f, k] == pytest.approx(2 * (1 - NormalDist().cdf(abs(ref))), abs=1e-12)
    assert np.isnan(z[:, :25]).all()


def test_volume_conduction_weights():

    w = volume_conduction_weight([5.0, 10.0, 15.0, 20.0, 30.0])
    assert np.allclose(w, [0.0, 0.0, (225 - 100) / 300, 1.0, 1.0])


def test_unknown_zone_is_refused():

    with pytest.raises(SettingsError, match="Unknown published zone"):
        resolve_published("nope")


def test_published_zone_sets_the_source_settings_and_measures_the_zone():

    rng = np.random.default_rng(3)
    ts = trialset(ar1(rng, (30, len(TIMES))) + burst(rng, 30, freq=100.0, centre=0.2, amp=10.0))
    tf = tf_fit(ts, zones=f"published:{VERIFIED}", freqs=(4, 250, 40), by_trial=True,
                random_state=0)
    assert tf.method == "published"
    assert tf.settings["transform"] == "morse"
    assert tf.settings["transform_params"]["beta"] == 10.0
    assert tf.settings["baseline"] == [-0.5, -0.1]
    assert tf.settings["published_zone"]["name"] == VERIFIED
    row = tf.zone_metrics[tf.zone_metrics["trial"] < 0].iloc[0]
    assert row["value"] > 0 and row["t"] > 3 and row["p_value"] < 0.01
    assert 0 < row["fraction_increase"] <= row["fraction_significant"] <= 1
    per_trial = tf.zone_metrics[tf.zone_metrics["trial"] >= 0]
    assert sorted(per_trial["trial"]) == list(range(30))
    assert "brinyark2026_delayed_high_gamma" in tf.methods_text()
