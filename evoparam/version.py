__version__ = '0.1.0.dev0'

# Version of each algorithm (spec §13.7, REQ-399): recorded in every provenance step as
# ``algorithm_version``. Bump the entry of a function whenever a change alters its numerical
# output; ``evoparam/tests/test_determinism.py`` fails when the reference fixture's values
# change while the versions recorded with them do not. Functions not listed are at the
# default version.
DEFAULT_ALGORITHM_VERSION = "1.0"
ALGORITHM_VERSIONS = {
    "evoparam.models.ResponseModel.fit_group": "1.0",
    "evoparam.models.ResponseModel.fit_waveform": "1.0",
    "evoparam.artifact.remove_artifact": "1.0",
    "evoparam.existence.test_response": "1.0",
    "evoparam.trials.fit_trials": "1.0",
    "evoparam.prep.rereference": "1.0",
    "evoparam.events.detect_events": "1.0",
    "evoparam.sim.simulate_ccep": "1.0",
}


def algorithm_version(function):
    """Version string of the algorithm behind ``function`` (a dotted public name)."""

    return ALGORITHM_VERSIONS.get(function, DEFAULT_ALGORITHM_VERSION)
