# SPDX-License-Identifier: Apache-2.0
"""Package version and algorithm versions (spec §13.7, REQ-399)."""

__version__ = "0.1.0.dev0"

# Algorithm versions (spec §13.7, REQ-399), recorded as ``algorithm_version``. Bump on any
# change of numerical output; tests/test_determinism.py checks it. Unlisted: default.
DEFAULT_ALGORITHM_VERSION = "1.0"
ALGORITHM_VERSIONS = {
    # 1.2: the search of a further component stops after the six strongest residual extrema
    # when none of them improves the criterion (engine.CANDIDATE_SCAN; REQ-290 P1)
    # 1.3: package defaults (no preset): final shape choice by AIC and the resolution check;
    # the default delta stays the fixed 10, the calibrated acceptance is on request only
    # (delta "calibrated", alpha 0.04); the presets are unchanged
    # 1.4: package defaults: the final shape choice refits every component, whatever its
    # width, and compares the shapes by HQC; the resolution check passes over a component
    # with a neighbour of either sign (REQ-003, REQ-004); the presets are unchanged
    # 1.5: package default shape_choice "penalised" (as "hqc", a skewed component
    # estimated with a prior on its asymmetry and weights on its top); presets unchanged
    # 1.6: the intervals of a component whose onset lies in a masked interval are widened by
    # the masked fraction of its support (spec §5.3 item 12); a window longer than 10 s at
    # 1 kHz or more is searched on a decimated copy and refitted at full resolution
    # (§5.5.9); uncertainty="bootstrap" gives intervals on every parameter and the bootstrap
    # distribution (§5.5.5); bootstrap and profile intervals follow the engine's component
    "evoparam.models.ResponseModel.fit_group": "1.6",
    "evoparam.models.ResponseModel.fit_waveform": "1.6",
    # the provenance step of a waveform fit is named after the function, not the method
    "evoparam.models.fit_waveform": "1.6",
    "evoparam.artifact.remove_artifact": "1.0",
    # 1.3: method="model_comparison" fits with the engine's package defaults (above)
    # 1.4: the same with the engine 1.4
    # 1.5: the same with the engine 1.5
    # 1.6: the same with the engine 1.6 (decimated search of a window longer than 10 s)
    "evoparam.existence.test_response": "1.6",
    # 1.3: without a preset, the fits of the package defaults (models 1.5)
    # 1.4: the channel fits of models 1.6 (widened intervals, bootstrap interval of "A")
    "evoparam.trials.fit_trials": "1.4",
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
    # 1.4: the channel fits of models 1.6 (intervals widened under a masked onset)
    "evoparam.hierarchical.fit_hierarchical": "1.4",
    # 1.1: the independent fits of the channels (models 1.2)
    # 1.2: without a preset, the fits of the package defaults (models 1.5)
    # 1.3: the independent fits of the channels (models 1.6)
    "evoparam.spatial.fit_shared": "1.3",
}


def algorithm_version(function: str) -> str:
    """Version string of the algorithm behind ``function`` (a dotted public name)."""
    return ALGORITHM_VERSIONS.get(function, DEFAULT_ALGORITHM_VERSION)
