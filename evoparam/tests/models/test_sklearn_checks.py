# SPDX-License-Identifier: Apache-2.0
"""REQ-321: the scikit-learn ``check_estimator`` subset."""

import warnings
from typing import Any

import pytest

from evoparam.artifact import ArtifactRemover
from evoparam.models import ResponseModel
from evoparam.trials import TrialFitter
from evoparam.utils._sklearn_checks import EXPECTED, FORBIDDEN, expected_failed_checks

sklearn_checks = pytest.importorskip("sklearn.utils.estimator_checks")

# Most scikit-learn checks expected to fail.
MAX_EXPECTED_FAILURES = 15

ESTIMATORS = [
    ResponseModel(sfreq=1000.0, tmin=0.0),
    TrialFitter(sfreq=1000.0, tmin=0.0),
    ArtifactRemover(method="interpolate", sfreq=1000.0, tmin=0.0, window=(0.0, 0.002)),
]


def test_expected_list_is_short_and_allowed() -> None:

    assert len(EXPECTED) <= MAX_EXPECTED_FAILURES
    assert not set(EXPECTED) & set(FORBIDDEN)
    assert all(isinstance(reason, str) and reason for reason in EXPECTED.values())


@sklearn_checks.parametrize_with_checks(ESTIMATORS, expected_failed_checks=expected_failed_checks)
def test_sklearn_compatible(estimator: Any, check: Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check(estimator)
