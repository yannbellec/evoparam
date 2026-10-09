# Decisions on the reviewed specification

Points where two requirements of `SPEC.md` cannot both be read literally, and how they are
resolved. `SPEC.md` itself is not edited.

## `prep.reference_sensitivity` runs the existence test (§5.4) — import graph (§4.2)

- §5.4, method 3: `reference_sensitivity` "runs the existence test (§5.6) under each
  scheme". §4.2: `prep` may import `data`, `utils` and `shapes` only, and `models`
  imports `prep`, while `existence` imports `models`: an import `prep -> existence` would
  break the rule and close a cycle.
- Resolution (the feature wins, the graph stays acyclic): `evoparam.prep` keeps a registry
  of detectors, `prep.DETECTORS`, filled with `prep.register_detector(name, detector)`.
  `evoparam/__init__.py`, the package root (outside the per-module rules), registers
  `existence.existence_detector` as `"existence"` once both subpackages are imported, and
  `reference_sensitivity(..., detector="existence")` looks it up by name. The interim
  threshold detector stays available as `detector="threshold"`; a callable is also
  accepted.
- Resolved import table for this point: `prep -> {data, utils, shapes}` (unchanged);
  `existence -> {data, models, utils}` (unchanged); root `evoparam -> prep, existence`
  (registration only).
- Tests: `tests/prep/test_reference.py::test_reference_sensitivity_runs_the_existence_test`
  (the table equals the existence test's detections) and
  `tests/prep/test_reference_sensitivity.py` (REQ-084 on ds004080, `attrs["detector"] ==
  "existence_detector"`); `tests/test_import_graph.py` unchanged.
