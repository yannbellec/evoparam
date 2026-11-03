# SPDX-License-Identifier: Apache-2.0
"""Package version and algorithm versions (spec §13.7, REQ-399)."""

__version__ = "0.1.0.dev0"

# default version.
DEFAULT_ALGORITHM_VERSION = "1.0"
ALGORITHM_VERSIONS = {
    "evoparam.models.ResponseModel.fit_group": "1.6",
    "evoparam.models.ResponseModel.fit_waveform": "1.6",
    "evoparam.models.fit_waveform": "1.6",
    "evoparam.artifact.remove_artifact": "1.0",
    "evoparam.existence.test_response": "1.6",
    "evoparam.trials.fit_trials": "1.4",
    "evoparam.prep.rereference": "1.0",
    "evoparam.events.detect_events": "1.0",
    "evoparam.sim.simulate_ccep": "1.0",
    "evoparam.models.evaluate_fit": "1.1",
    "evoparam.models.fitted_peak": "1.1",
    "evoparam.tf.tf_fit": "1.1",
    "evoparam.hierarchical.fit_hierarchical": "1.4",
    "evoparam.spatial.fit_shared": "1.3",
}


def algorithm_version(function: str) -> str:
    """Version string of the algorithm behind ``function`` (a dotted public name)."""
    return ALGORITHM_VERSIONS.get(function, DEFAULT_ALGORITHM_VERSION)
