# SPDX-License-Identifier: Apache-2.0
"""Calibration under the null (REQ-130) and corrections (REQ-136)."""

from typing import Any

import numpy as np
import pytest

from evoparam.existence import correct
from evoparam.existence import test_response as run_test

from .helpers import noise_trials, trialset

# Level of the corrections.
ALPHA = 0.05

###################################################################################################
###################################################################################################


@pytest.mark.parametrize("rho", [0.0, 0.9])
@pytest.mark.parametrize("statistic", ["max_t", "cluster"])
def test_req130_type_one_error(rho: Any, statistic: Any) -> None:

    rng = np.random.default_rng(1 if rho else 2)
    ts = trialset(noise_trials(rng, 20, 2000, rho))
    test = run_test(
        ts,
        window=(0.009, 0.5),
        baseline=(-0.5, -0.05),
        correction="none",
        statistic=statistic,
        n_permutations=500,
        n_boot=0,
        by=(),
        random_state=0,
    )
    for alpha in (0.05, 0.01):
        rate = np.mean(test.p_value_raw <= alpha)
        # [0.8 alpha, 1.2 alpha], widened by 2 binomial standard errors for 2000 channels
        se = np.sqrt(alpha * (1 - alpha) / 2000)
        assert 0.8 * alpha - 2 * se <= rate <= 1.2 * alpha + 2 * se, (alpha, rate)


def test_req136_corrections() -> None:

    rng = np.random.default_rng(3)
    fdr, fwe = [], []
    for _ in range(2000):
        p = rng.uniform(size=1000)
        fdr.append(np.mean(correct(p, "fdr_bh") <= ALPHA))
        fwe.append(np.any(correct(p, "bonferroni") <= ALPHA))
    se = np.sqrt(ALPHA * 0.95 / 2000)
    assert np.mean(fdr) <= ALPHA
    assert np.mean(fwe) <= ALPHA + 2 * se  # at most 1 family-wise error in 95 % of runs
    p = np.array([0.01, 0.02, 0.03, 0.5, np.nan])
    assert np.allclose(correct(p, "holm")[:4], [0.04, 0.06, 0.06, 0.5])
    assert np.allclose(correct(p, "fdr_bh")[:4], [0.04, 0.04, 0.04, 0.5])
    assert np.isnan(correct(p, "bonferroni")[4])
