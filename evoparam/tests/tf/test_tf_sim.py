"""Time-frequency simulations with known truth (REQ-212 to REQ-215)."""

import warnings

import numpy as np

from evoparam.models import fit as fit_models
from evoparam.tests.tf.helpers import TIMES, ar1, burst, n1, trialset
from evoparam.tf import tf_fit

FREQS = (4, 250, 40)
# A burst adding 20 % to the signal power (variance) over its 100 ms: with AR(1) noise of
# standard deviation 10 and a Hann envelope (mean square 3/8), amplitude**2 * 3 / 16 = 20.
BURST_AMP = np.sqrt(0.2 * 100 * 16 / 3)


def test_req212_footprint_removes_the_sharp_n1_confound():
    """A sharp N1 (FWHM 8 ms) puts significant power in high gamma in the total map; the
    residual (trials minus the fitted N1) has none."""

    runs, total, residual = 40, 0, 0
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        ts = trialset(ar1(rng, (30, len(TIMES))) + n1(fwhm=0.008, amp=-60.0)[None])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            group = fit_models(ts, "ccep")
        tf = tf_fit(ts, freqs=FREQS, group_fit=group, zones="none", random_state=seed)
        high = (tf.freqs >= 50) & (tf.freqs <= 250)
        total += bool(tf.significant[0][high].any())
        residual += bool(tf.maps["residual_significant"][0][high].astype(bool).any())
    assert total / runs >= 0.8
    assert residual / runs <= 0.05


def test_req213_zone_recovery_of_induced_gamma_bursts():

    runs, recovered = 20, 0
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        ts = trialset(ar1(rng, (30, len(TIMES))) + burst(rng, 30, amp=BURST_AMP))
        tf = tf_fit(ts, freqs=FREQS, zone_method="nmf", random_state=seed)
        zones = tf.zones[tf.zones["sign"] > 0]
        if len(zones):
            best = zones.iloc[0]
            recovered += (abs(best["freq_centroid"] / 80.0 - 1) <= 0.1
                          and abs(best["time_centroid"] - 0.25) <= 0.02)
    assert recovered / runs >= 0.9


def test_narrow_band_twenty_percent_is_below_pixel_resolution():
    """Documented limit: +20 % of the local 80 Hz power (0.8 dB) with 30 trials gives a
    pixel t of about 0.8; pixel tests cannot find it (REQ-213 is met with the burst adding
    20 % of the signal power)."""

    runs, found = 20, 0
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        ts = trialset(ar1(rng, (30, len(TIMES))) + burst(rng, 30, amp=1.32))
        tf = tf_fit(ts, freqs=FREQS, zones="none", random_state=seed)
        burst_box = ((tf.freqs[:, None] >= 65) & (tf.freqs[:, None] <= 100)
                     & (tf.times[None] >= 0.2) & (tf.times[None] <= 0.3))
        found += bool(tf.significant[0][burst_box].any())
    assert found / runs <= 0.1


def _null_rate(stats, runs, n_permutations=200):
    hits = 0
    for seed in range(runs):
        rng = np.random.default_rng(1000 + seed)
        ts = trialset(ar1(rng, (30, len(TIMES))))
        tf = tf_fit(ts, freqs=FREQS, zones="none", stats=stats, n_permutations=n_permutations,
                    random_state=seed)
        hits += bool(tf.significant[0].any())
    return hits / runs


def test_req214_fdr_type_one_error():
    # Under the complete null the false-discovery rate is the family-wise error, which
    # bounds the pixel-wise error from above.
    assert _null_rate("fdr", 200) <= 0.06


def test_req214_cluster_family_wise_error():
    assert _null_rate("cluster", 100) <= 0.06


def test_req214_tfce_family_wise_error():
    assert _null_rate("tfce", 40, n_permutations=100) <= 0.06


def test_req215_single_trial_normalisation_resists_a_noisy_trial():

    ratios = []
    region = None
    for seed in range(10):
        rng = np.random.default_rng(seed)
        clean = ar1(rng, (30, len(TIMES))) + burst(rng, 30, amp=BURST_AMP)
        noisy = clean.copy()
        noisy[0] = ar1(rng, (1, len(TIMES)), sd=100.0)[0]
        change = {}
        for level in ("single_trial", "evoked"):
            values = []
            for data in (clean, noisy):
                tf = tf_fit(trialset(data), freqs=FREQS, zones="none", stats="none",
                            baseline_level=level)
                if region is None:
                    region = ((tf.freqs[:, None] >= 65) & (tf.freqs[:, None] <= 100)
                              & (tf.times[None] >= 0.2) & (tf.times[None] <= 0.3))
                values.append(np.nanmean(tf.maps["total_norm"][0][region]))
            change[level] = abs(values[1] - values[0])
        ratios.append(change["single_trial"] / change["evoked"])
    assert np.median(ratios) <= 0.5
