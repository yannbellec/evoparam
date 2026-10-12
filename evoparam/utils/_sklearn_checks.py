# SPDX-License-Identifier: Apache-2.0
"""Checks of scikit-learn's ``check_estimator`` that the estimators are excused from
(spec §10.4, REQ-321): ``{check name: reason}``, at most 15 entries.

None is needed: ``ResponseModel``, ``TrialFitter`` and ``ArtifactRemover`` pass every
check. Short tabular inputs (fewer time points than a component needs) give skipped fits
instead of errors; integer data are converted to float.
"""

__all__ = ["EXPECTED", "FORBIDDEN", "expected_failed_checks"]

EXPECTED = {}

# checks that must never be excused (REQ-321)
FORBIDDEN = (
    "check_get_params_invariance",
    "check_set_params",
    "check_no_attributes_set_in_init",
    "check_dont_overwrite_parameters",
    "check_fit_idempotent",
    "check_estimators_unfitted",
    "check_parameters_default_constructible",
    "check_estimators_overwrite_params",
    "check_n_features_in",
)


def expected_failed_checks(estimator):
    return dict(EXPECTED)
