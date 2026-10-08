"""Methods text (REQ-219)."""

import numpy as np
import pytest

from evoparam.tests.tf.helpers import TIMES, ar1, trialset
from evoparam.tf import methods_text, tf_fit


@pytest.mark.parametrize("transform,stats,zones,words", [
    ("morse", "fdr", "auto", ["Morse", "gamma = 3", "beta = 10", "false-discovery",
                              "non-negative matrix factorisation"]),
    ("morlet", "cluster", "none", ["Morlet", "7 cycles", "cluster-mass", "No zones were computed"]),
    ("multitaper", "tfce", "none", ["multitapers", "time-bandwidth product 4",
                                    "threshold-free"]),
    ("stft", "none", "none", ["short-time Fourier", "0.2 s"]),
])
def test_req219_methods_text_names_every_choice(transform, stats, zones, words):

    ts = trialset(ar1(np.random.default_rng(0), (12, len(TIMES))))
    tf = tf_fit(ts, transform=transform, stats=stats, zones=zones, freqs=(4, 250, 10),
                n_permutations=20, baseline=(-0.4, -0.1), baseline_mode="percent",
                random_state=1)
    text = methods_text(tf)
    assert text == tf.methods_text()
    for word in words + ["-400 to -100 ms", "(percent)", "alpha = 0.05", "evoparam",
                         "each trial by its own baseline"]:
        assert word in text, word
