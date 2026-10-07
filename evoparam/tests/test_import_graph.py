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
    "stats": {"data", "models", "utils"},
    "sim": {"data", "shapes", "artifact", "utils"},
    "presets": set(),
    "utils": {"data"},
    "reports": {ANY}, "plots": {ANY}, "cli": {ANY}, "stream": {ANY}, "benchmark": {ANY},
}
FORBIDDEN_BETWEEN = {"reports", "plots", "cli", "stream", "benchmark"}

# Inherited modules, rewritten or removed by the build steps of spec §12.
LEGACY = {"analysis", "bands", "clean", "core", "objs", "param", "plts", "_legacy"}

# (source file relative to the package, imported module) -> build step that removes the edge.
TRANSITIONAL = {
    ("data/conversions.py", "analysis"): "S7",
    ("data/conversions.py", "core"): "S7",
    ("data/conversions.py", "__init__"): "S7",
    ("sim/gen.py", "core"): "S9",
    ("sim/gen.py", "data"): "S9",
    ("sim/params.py", "core"): "S9",
    ("sim/params.py", "data"): "S9",
    ("sim/utils.py", "core"): "S9",
    ("utils", "clean"): "S5",
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
            if dest in LEAVES or _transitional(rel_path, source, dest):
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
