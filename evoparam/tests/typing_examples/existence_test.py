# SPDX-License-Identifier: Apache-2.0
"""The existence test of every channel."""

import numpy as np
import numpy.typing as npt

import evoparam
from evoparam.data import ResponseTest, TrialSet


def responsive(trials: TrialSet) -> npt.NDArray[np.bool_]:
    test: ResponseTest = evoparam.test_response(
        trials, window=(0.009, 0.5), baseline=(-0.5, -0.05), random_state=0
    )
    return np.asarray(test.significant, dtype=bool)
