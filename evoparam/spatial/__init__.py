# SPDX-License-Identifier: Apache-2.0
"""Spatial: shared shapes across channels, maps and the convergent paradigm (spec §5.9).

- :func:`shared_shapes` (:func:`convergent`, :func:`divergent`): the few response shapes
  shared across stimulation sites into one channel, or across channels for one site
  (basis-profile-curve method, re-implemented from its published description), returned as
  a :class:`~evoparam.data.ShapeBasis`; :func:`as_template` turns a curve into a
  ``template`` component shape.
- :func:`fit_shared`: the parametric model of the engine with shape parameters shared
  across channels, optional per-channel latency offsets and a graph-Laplacian penalty on
  neighbouring contacts.
- :func:`parameter_map`, :func:`connectivity_matrix` (and its long form): per (site,
  channel) tables of the fitted parameters with coordinates and markers.
- :func:`screen_volume_conduction`: local versus volume-conducted responses.
"""

from .bpc import (
    BASIS_COLUMNS,
    adjusted_rand_index,
    as_template,
    convergent,
    divergent,
    shared_shapes,
)
from .maps import (
    MAP_COLUMNS,
    MARKERS,
    connectivity_array,
    connectivity_from_long,
    connectivity_matrix,
    connectivity_to_long,
    parameter_map,
    plot_parameter_map,
)
from .shared import PARAMETER_ROLES, fit_shared, neighbours
from .volume import RISKS, SCREEN_COLUMNS, screen_volume_conduction

__all__ = [
    "shared_shapes",
    "convergent",
    "divergent",
    "as_template",
    "fit_shared",
    "neighbours",
    "parameter_map",
    "connectivity_matrix",
    "connectivity_to_long",
    "connectivity_from_long",
    "connectivity_array",
    "plot_parameter_map",
    "screen_volume_conduction",
    "adjusted_rand_index",
    "BASIS_COLUMNS",
    "PARAMETER_ROLES",
    "MAP_COLUMNS",
    "MARKERS",
    "SCREEN_COLUMNS",
    "RISKS",
]
