"""Import-graph test (REQ-296): module dependencies follow the table of spec §4.2.

Every module of the package is parsed with ``ast``. An import from one
top-level module of the package to another is an edge. Edges between the
modules of the new architecture must be allowed by RULES. The inherited
modules (``LEGACY``) are removed or rewritten at later build steps; until then
the only edges they may add into the new modules are listed in TRANSITIONAL,
with the step that removes them.
"""

import ast
import pathlib
from collections import defaultdict

import evoparam

PACKAGE = pathlib.Path(evoparam.__file__).parent

# Leaf modules that every module may import.
LEAVES = {"errors", "version"}
# Pure-data modules that every module may import (spec §4.2: "presets ... everything").
PURE_DATA = {"presets"}

# Spec §4.2: module -> modules it may import (besides LEAVES).
ANY = "*"
RULES = {
    "data": set(),
    "io": {"data", "utils"},
    "events": {"data", "utils", "shapes"},
    "artifact": {"data", "utils", "shapes"},
    "prep": {"data", "utils", "shapes"},
    "shapes": set(),
    "models": {"data", "shapes", "prep", "utils"},
    "existence": {"data", "models", "utils"},
    "trials": {"data", "models", "existence", "utils"},
    "hierarchical": {"data", "models", "trials", "stats", "utils"},
    "spatial": {"data", "models", "utils"},
    "tf": {"data", "models", "utils"},
    # stats -> existence: trials_needed resamples the existence test (spec §5.11, method 5)
    # and compare_conditions uses its corrections ("as in §5.6"); existence imports nothing
    # from stats, so no cycle arises.
    "stats": {"data", "models", "existence", "utils"},
    # sim -> io: the licence of a real-noise dataset is read from the dataset registry
    # (spec REQ-256); io imports nothing below data and utils.
    "sim": {"data", "shapes", "artifact", "utils", "io"},
    "presets": set(),
    "utils": {"data"},
    "reports": {ANY}, "plots": {ANY}, "cli": {ANY}, "stream": {ANY}, "benchmark": {ANY},
}
FORBIDDEN_BETWEEN = {"reports", "plots", "cli", "stream", "benchmark"}

# Inherited modules, rewritten or removed by the build steps of spec §12.
LEGACY = {"analysis", "bands", "core", "objs", "param", "plts", "_legacy"}

# (source file relative to the package, imported module) -> build step that removes the edge.
TRANSITIONAL = {
    ("data/conversions.py", "analysis"): "S7",
    ("data/conversions.py", "core"): "S7",
    ("data/conversions.py", "__init__"): "S7",
    ("utils", "core"): "S7",
    ("utils", "objs"): "S7",
}


def _module_name(path):
    rel = path.relative_to(PACKAGE).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ["evoparam"] + parts


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    own = _module_name(path)
    package = own if path.name == "__init__.py" else own[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - (node.level - 1)]
                prefix = ".".join(base + ([node.module] if node.module else []))
                if node.module is None:
                    for alias in node.names:
                        yield f"{prefix}.{alias.name}"
                else:
                    yield prefix
            elif node.module:
                yield node.module


def _edges():
    edges = defaultdict(set)
    for path in PACKAGE.rglob("*.py"):
        rel = path.relative_to(PACKAGE)
        if rel.parts[0] == "tests":
            continue
        source = rel.parts[0][:-3] if len(rel.parts) == 1 else rel.parts[0]
        for target in _imports(path):
            parts = target.split(".")
            if parts[0] != "evoparam":
                continue
            dest = parts[1] if len(parts) > 1 else "__init__"
            if dest != source:
                edges[(source, rel.as_posix())].add(dest)
    return edges


def _transitional(rel_path, source, dest):
    return (rel_path, dest) in TRANSITIONAL or (source, dest) in TRANSITIONAL


def test_edges_follow_spec_table():
    violations = []
    for (source, rel_path), dests in _edges().items():
        if source in LEGACY or source in ("__init__", "version", "errors"):
            continue
        assert source in RULES, f"module '{source}' is not in the dependency table of §4.2"
        for dest in dests:
            if dest in LEAVES or dest in PURE_DATA or _transitional(rel_path, source, dest):
                continue
            allowed = RULES[source]
            if ANY in allowed:
                if source in FORBIDDEN_BETWEEN and dest in FORBIDDEN_BETWEEN \
                        and not (source, dest) == ("benchmark", "reports"):
                    violations.append(f"{rel_path} -> {dest}")
                continue
            if dest not in allowed:
                violations.append(f"{rel_path} -> {dest}")
    assert not violations, "imports not allowed by spec §4.2:\n" + "\n".join(sorted(violations))


def test_leaves_import_nothing_from_the_package():
    for name in LEAVES:
        for target in _imports(PACKAGE / f"{name}.py"):
            assert not target.startswith("evoparam"), f"{name}.py imports {target}"


def test_no_cycles_between_new_modules():
    graph = defaultdict(set)
    for (source, rel_path), dests in _edges().items():
        if source in LEGACY or source not in RULES:
            continue
        for dest in dests:
            if dest in RULES and not _transitional(rel_path, source, dest):
                graph[source].add(dest)
    visiting, done = set(), set()

    def visit(node, trail):
        if node in done:
            return
        assert node not in visiting, "import cycle: " + " -> ".join(trail + [node])
        visiting.add(node)
        for nxt in graph[node]:
            visit(nxt, trail + [node])
        visiting.discard(node)
        done.add(node)

    for node in list(graph):
        visit(node, [])


# Modules that hold a fitting engine: the new one (spec §5.5) and the inherited fitter with
# the modules built on it (removed at build step S7).
FITTERS = ("models", "objs", "param", "_legacy")


def _module_files(dotted):
    """Files executed by importing ``dotted`` (package __init__ files included), without the
    package root, whose own imports are the public API of the whole package."""

    parts = dotted.split(".")[1:]
    files = []
    for depth in range(1, len(parts) + 1):
        base = PACKAGE.joinpath(*parts[:depth])
        if base.with_suffix(".py").is_file():
            files.append(base.with_suffix(".py"))
            break
        if (base / "__init__.py").is_file():
            files.append(base / "__init__.py")
        else:
            break
    return files


def test_sim_does_not_reach_the_fitter():
    """REQ-254: nothing that evoparam.sim imports, directly or through other modules, is a
    fitting engine, so the ground truth cannot depend on the estimator."""

    start = sorted((PACKAGE / "sim").rglob("*.py"))
    seen, queue, trail = set(start), list(start), {path: None for path in start}
    while queue:
        path = queue.pop()
        for target in _imports(path):
            if target.split(".")[0] != "evoparam":
                continue
            for dest in _module_files(target):
                if dest not in seen:
                    seen.add(dest)
                    trail[dest] = path
                    queue.append(dest)
    reached = [path for path in seen if path.relative_to(PACKAGE).parts[0] in FITTERS]

    def chain(path):
        out = []
        while path is not None:
            out.append(path.relative_to(PACKAGE).as_posix())
            path = trail[path]
        return " <- ".join(out)

    assert not reached, "evoparam.sim reaches a fitter:\n" + "\n".join(map(chain, reached))
    modules = {path.relative_to(PACKAGE).parts[0] for path in seen}
    assert "sim" in modules and "shapes" in modules and "data" in modules


def test_stats_is_below_hierarchical():
    """Spec §4.2: hierarchical imports stats (variance components), never the reverse, and
    stats does not reach the trial-level module."""

    edges = _edges()
    stats_dests = set().union(*[dests for (source, _), dests in edges.items()
                                if source == "stats"])
    assert "stats" in {source for source, _ in edges}
    assert not stats_dests & {"hierarchical", "trials", "spatial", "tf", "sim"}, stats_dests
    for (source, _), dests in edges.items():
        if "stats" in dests:
            assert source in ("hierarchical", "reports", "plots", "cli", "stream", "benchmark"), \
                source


def test_spatial_reaches_shapes_and_prep_through_models():
    """evoparam.spatial (build step S12) imports data, models and utils only: the shapes
    through ``models.component_shapes`` and the reference schemes through
    ``models.references`` (spec §4.2)."""

    dests = set()
    for (source, _), targets in _edges().items():
        if source == "spatial":
            dests |= targets
    assert dests and dests - LEAVES <= RULES["spatial"]
    assert {"data", "models"} <= dests
