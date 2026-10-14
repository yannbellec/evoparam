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
    # 1.2: the search of a further component stops after the six strongest residual extrema
    # when none of them improves the criterion (engine.CANDIDATE_SCAN; REQ-290 P1)
    # 1.3: package defaults (no preset): calibrated acceptance (delta "calibrated", alpha
    # 0.03), final shape choice by AIC and the resolution check; the presets are unchanged
    # 1.4: package defaults: the final shape choice refits every component, whatever its
    # width, and compares the shapes by HQC; the resolution check passes over a component
    # with a neighbour of either sign (REQ-003, REQ-004); the presets are unchanged
    # 1.5: package default shape_choice "penalised" (as "hqc", a skewed component
    # estimated with a prior on its asymmetry and weights on its top); presets unchanged
    "evoparam.models.ResponseModel.fit_group": "1.5",
    "evoparam.models.ResponseModel.fit_waveform": "1.5",
    # the provenance step of a waveform fit is named after the function, not the method
    "evoparam.models.fit_waveform": "1.5",
    "evoparam.artifact.remove_artifact": "1.0",
    # 1.3: method="model_comparison" fits with the engine's package defaults (above)
    # 1.4: the same with the engine 1.4
    # 1.5: the same with the engine 1.5
    "evoparam.existence.test_response": "1.5",
    # 1.3: without a preset, the fits of the package defaults (models 1.5)
    "evoparam.trials.fit_trials": "1.3",
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
    # 1.2: the fits of the channels (models 1.2)
    # 1.3: without a preset, the fits of the package defaults (models 1.5)
    "evoparam.hierarchical.fit_hierarchical": "1.3",
    # 1.1: the independent fits of the channels (models 1.2)
    # 1.2: without a preset, the fits of the package defaults (models 1.5)
    "evoparam.spatial.fit_shared": "1.2",
}


def algorithm_version(function: str) -> str:
    """Version string of the algorithm behind ``function`` (a dotted public name)."""
    return ALGORITHM_VERSIONS.get(function, DEFAULT_ALGORITHM_VERSION)
