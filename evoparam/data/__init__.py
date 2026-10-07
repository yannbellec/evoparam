"""Data containers, results, provenance and the ``.evoparam`` container format."""

from .provenance import Provenance, capture_environment, compute_data_hash
from .containers import TrialSet, Evoked
from .results import (Component, GoodnessOfFit, ResponseFit, GroupFit, TrialFit,
                      HierarchicalFit, ResponseTest, ArtifactResult, ShapeBasis, TFFit)
from .serialize import save_object, load_object

# Settings and result tuples of the inherited fitting engine; removed at build step S7.
from .data import evoparamSettings, evoparamMetaData, evoparamResults, SimParams
