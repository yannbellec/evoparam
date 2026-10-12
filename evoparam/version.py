# SPDX-License-Identifier: Apache-2.0
"""Package version and algorithm versions (spec §13.7, REQ-399)."""

__version__ = "0.1.0.dev0"

# Version of each algorithm (spec §13.7, REQ-399): recorded in every provenance step as
# ``algorithm_version``. Bump the entry of a function whenever a change alters its numerical
# output; ``evoparam/tests/test_determinism.py`` fails when the reference fixture's values
# change while the versions recorded with them do not. Functions not listed are at the
# default version.
DEFAULT_ALGORITHM_VERSION = "1.0"
ALGORITHM_VERSIONS = {
    "evoparam.models.ResponseModel.fit_group": "1.1",
    "evoparam.models.ResponseModel.fit_waveform": "1.1",
    "evoparam.artifact.remove_artifact": "1.0",
    "evoparam.existence.test_response": "1.1",
    "evoparam.trials.fit_trials": "1.1",
    "evoparam.prep.rereference": "1.0",
    "evoparam.events.detect_events": "1.0",
    "evoparam.sim.simulate_ccep": "1.0",
    # 1.1: the fitted background is evaluated from the fit's own time origin (the first
    # sample of the fit window) on any time grid, not from the grid's first time
    "evoparam.models.evaluate_fit": "1.1",
    "evoparam.models.fitted_peak": "1.1",
    "evoparam.tf.tf_fit": "1.1",
    # 1.1: the standard error of the channel latency is the noise of the average (Wald)
    # plus the mean of the jitter beyond noise, truncated to the label window (REQ-171)
    "evoparam.hierarchical.fit_hierarchical": "1.1",
}


def algorithm_version(function):
    """Version string of the algorithm behind ``function`` (a dotted public name)."""
    return ALGORITHM_VERSIONS.get(function, DEFAULT_ALGORITHM_VERSION)
