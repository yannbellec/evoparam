# SPDX-License-Identifier: Apache-2.0
"""Data containers, results, provenance and the ``.evoparam`` container format."""

from .containers import Evoked, TrialSet
from .provenance import Provenance, capture_environment, compute_data_hash
from .results import (
    ArtifactResult,
    Component,
    GoodnessOfFit,
    GroupFit,
    HierarchicalFit,
    ResponseFit,
    ResponseTest,
    ShapeBasis,
    TFFit,
    TrialFit,
)
from .serialize import load_object, save_object
