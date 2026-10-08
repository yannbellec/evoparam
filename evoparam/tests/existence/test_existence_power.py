"""Power (REQ-131), duration (REQ-132), projection duration, discordance (REQ-139)."""

import numpy as np
import pandas as pd

from evoparam.existence import test_response as run_test, train_discordance

from .helpers import gaussian, noise_trials, plateau, trialset

###################################################################################################
###################################################################################################

def _power(signal, statistic, n_sets=100, n_trials=10, seed=0):
    rng = np.random.default_rng(seed)
    data = noise_trials(rng, n_trials, n_sets) + signal[None, None]
    test = run_test(trialset(data), statistic=statistic, correction="none",
                         n_permutations=500, n_boot=0, by=(), random_state=0)
    return np.mean(test.significant)


def test_req131_power_transient_and_sustained():

    transient = 2.0 * gaussian(0.03, 0.010)
    assert _power(transient, "max_t") >= 0.9
    # A sustained 100 ms response: at the spec's SNR of 1, both statistics already reach a
    # power near 1 with 10 trials; the advantage of the cluster statistic shows at 0.7
    sustained = 0.7 * plateau(0.05, 0.15)
    assert _power(sustained, "cluster", seed=1) >= 0.9
    assert _power(sustained, "max_t", seed=1) < 0.7


def test_req132_duration_and_onset():

    rng = np.random.default_rng(4)
    data = noise_trials(rng, 20, 50) + 3.0 * plateau(0.05, 0.15)[None, None]
    test = run_test(trialset(data), n_permutations=500, n_boot=50, by=(), random_state=0)
    assert abs(np.median(test.duration) - 0.10) <= 0.01
    assert np.median(np.abs(test.onset - 0.05)) <= 0.002
    ci = np.asarray(test.settings["duration_ci"])
    assert np.all(ci[:, 0] <= ci[:, 1])


def test_projection_duration_and_significance():

    rng = np.random.default_rng(5)
    data = noise_trials(rng, 20, 10) + 3.0 * plateau(0.009, 0.12)[None, None]
    data[:, 5:] = noise_trials(rng, 20, 5)
    test = run_test(trialset(data), method="projection", by=(), random_state=0)
    assert np.all(test.significant[:5]) and np.sum(test.significant[5:]) <= 1
    assert np.median(np.abs(test.offset[:5] - 0.12)) <= 0.01


def test_req139_train_discordance():

    rng = np.random.default_rng(6)
    n_trains, per = 6, 10
    data = noise_trials(rng, n_trains * per, 2)
    signal = 3.0 * gaussian(0.03, 0.01)
    data[:, 0] += signal                                         # every train responds
    for train in range(0, n_trains, 2):                          # every other train responds
        data[train * per:(train + 1) * per, 1] += signal
    meta = pd.DataFrame({"train_id": np.repeat(np.arange(n_trains), per)})
    ts = trialset(data, meta)
    test = run_test(ts, n_permutations=200, n_boot=0, by=(), random_state=0)
    table = train_discordance(ts, test)
    assert table["discordance"].iloc[0] == 0.0
    assert table["discordance"].iloc[1] >= 0.9
