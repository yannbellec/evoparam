"""Published zones registry (REQ-217) and the compatibility layer."""

import numpy as np
import pytest

from evoparam.errors import SettingsError
from evoparam.tests.tf.helpers import TIMES, ar1, burst, trialset
from evoparam.tf import list_published_zones, tf_fit
from evoparam.tf.zones import load_zones, resolve_published

VERIFIED = "brinyark2026_delayed_high_gamma"
UNVERIFIED = "brinyark2026_other_zones"


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


def test_zone_without_bounds_is_refused_even_when_allowed():

    with pytest.raises(SettingsError, match="no bounds"):
        resolve_published(UNVERIFIED, allow_unverified=True)


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
