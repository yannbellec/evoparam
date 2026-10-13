# SPDX-License-Identifier: Apache-2.0
"""Artifact: remove the stimulation artifact, mark replaced samples, score each trial."""

from ._common import apply_mask
from .api import METHODS, compare_methods, remove_artifact
from .estimator import ArtifactRemover
from .paradigm import paradigm_subtract
from .period import estimate_period
from .quality import artifact_quality
from .template import estimate_artifact_template

__all__ = [
    "ArtifactRemover",
    "remove_artifact",
    "estimate_artifact_template",
    "estimate_period",
    "artifact_quality",
    "paradigm_subtract",
    "apply_mask",
    "compare_methods",
    "METHODS",
]
