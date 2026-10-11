# SPDX-License-Identifier: Apache-2.0
"""Import-graph test (REQ-296): module dependencies follow the table of spec §4.2.

Every module of the package is parsed with ``ast`` and the directed graph of intra-package
imports is built; ``from evoparam import plots`` is an edge to ``plots`` (not to the
package root). The test fails on (i) any edge not allowed by the table, (ii) any cycle and
(iii) any top-level import of an optional package. No module is exempt.

The table is the reviewed table of spec §4.2 with the owner's resolutions of D3, D4 and D5
(the features required by other sections win). Every edge that the reviewed table does not
list is written next to the rule that requires it; ``docs/spec/DECISIONS.md`` holds the same
resolved table with its reasons. ``test_resolved_table_matches_decisions`` keeps the two in
step.
"""

import ast
import pathlib
import re
import sys
from collections import defaultdict

import pytest

import evoparam

pytestmark = pytest.mark.core

PACKAGE = pathlib.Path(evoparam.__file__).parent
DECISIONS = PACKAGE.parent / "docs" / "spec" / "DECISIONS.md"

# Leaf modules that import nothing from the package and that every module may import:
# the error hierarchy that §4.6 asks every module to raise (D4: data raises DataError) and
# the package version that every provenance record carries (§4.1 principle 3).
LEAVES = {"errors", "version"}

# Every top-level module of the package (subpackages and single files).
MODULES = sorted({p.name if p.is_dir() else p.stem for p in PACKAGE.iterdir()
                  if (p.is_dir() and (p / "__init__.py").is_file())
                  or (p.suffix == ".py" and p.stem != "__init__")} - {"tests"})

ANY = "*"
TOP = {"reports", "plots", "cli", "stream", "benchmark"}

# Spec §4.2 with D3/D4/D5 resolved: module -> modules it may import (besides LEAVES).
RULES = {
    # D4/D5: the plugins used for a result are recorded in its provenance (REQ-328).
    "data": {"_plugins"},
    # D5: the reader registry loads the evoparam.readers entry points (REQ-328).
    "io": {"data", "utils", "_plugins"},
    "events": {"data", "utils", "shapes"},
    "artifact": {"data", "utils", "shapes"},
    "prep": {"data", "utils", "shapes"},
    # D5: the shape registry loads the evoparam.shapes entry points (REQ-328).
    "shapes": {"_plugins"},
    # D5: evoparam.fit(data, preset="ccep") resolves the preset (§6.3).
    "models": {"data", "shapes", "prep", "utils", "presets"},
    "existence": {"data", "models", "utils"},
    "trials": {"data", "models", "existence", "utils"},
    # D5: optional packages are imported through utils.require only (§9.2, REQ-292).
    "hierarchical": {"data", "models", "trials", "stats", "utils"},
    "spatial": {"data", "models", "utils"},
    "tf": {"data", "models", "utils"},
    # D5: trials_needed resamples the existence test (§5.11).
    "stats": {"data", "models", "utils", "existence"},
    # D5: the licence of a real-noise dataset is read from the registry of io (REQ-256);
    # optional packages through utils.require (§9.2).
    "sim": {"data", "shapes", "artifact", "utils", "io"},
    # D5: the preset registry loads the evoparam.presets entry points (REQ-328).
    "presets": {"_plugins"},
    # Cross-cutting helpers (§4.2 figure): the leaves only.
    "utils": set(),
    # Entry-point loader of §10.8: the leaves only.
    "_plugins": set(),
    # Data tables of §9.2 and §9.3 (REQ-292 extras, REQ-293 deprecated names).
    "_extras": set(), "_deprecated": set(),
    "reports": {ANY}, "plots": {ANY}, "cli": {ANY}, "stream": {ANY}, "benchmark": {ANY},
    # The package root re-exports the public API; ``python -m evoparam`` runs the cli.
    "__init__": {ANY},
    "__main__": {"cli"},
    "errors": set(),
    "version": set(),
}
# Modules whose only allowed use of utils is utils.require (the rows of §4.2 do not list
# utils for them; §9.2 makes require the only way to import an optional package).
REQUIRE_ONLY = {"hierarchical": {"evoparam.utils._optional", "evoparam.utils.require"},
                "sim": {"evoparam.utils._optional", "evoparam.utils.require"}}
# Spec §4.2: the top layers do not import each other, except benchmark -> reports.
# D3 (owner): the HTML/PDF reports embed the figures and the command line runs the reports
# (§5.13), so the top layers are ordered plots < reports < cli, benchmark.
# S19: the command line runs the benchmark (§8.4 `evoparam benchmark score ...`, §10.5), so
# cli -> benchmark; the benchmark never imports the command line.
ALLOWED_BETWEEN = {("benchmark", "reports"), ("reports", "plots"), ("cli", "reports"),
                   ("cli", "plots"), ("benchmark", "plots"), ("cli", "benchmark")}

# Import names of the packages of the extras of spec §9.2 (and of the test tool chain).
OPTIONAL_IMPORTS = {"mne", "mne_bids", "bids", "pynwb", "hdmf", "ndx_events", "neo", "edfio",
                    "pyedflib", "numpyro", "jax", "arviz", "pymc", "cupy", "numba",
                    "matplotlib", "pandas", "pyarrow", "xarray", "statsmodels", "formulaic",
                    "bidsschematools", "nwbinspector", "jsonschema", "tqdm", "joblib", "pylsl",
                    "sklearn", "h5py", "remfile", "quantities", "mpmath", "pytest",
                    "hypothesis", "click"}

# (iii) packages that may be imported at the top level of a module: the standard library
# and the two hard dependencies (REQ-291).
HARD = {"numpy", "scipy", "evoparam", "__future__"}


def _module_name(path):
    rel = path.relative_to(PACKAGE).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ["evoparam"] + parts


def _imports(path, source=None):
    """Dotted names imported by a file; ``from a import b`` yields ``a.b`` so that a
    submodule imported from its package is seen."""

    tree = ast.parse(source if source is not None else path.read_text(encoding="utf-8"))
    own = _module_name(path)
    package = own if path.name == "__init__.py" else own[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - (node.level - 1)]
                module = base + (node.module.split(".") if node.module else [])
            else:
                module = node.module.split(".") if node.module else []
            for alias in node.names:
                yield ".".join(module + [alias.name])


def _target(dotted):
    """Top-level module of the package that a dotted import reaches (``__init__`` for the
    package root itself), or None outside the package."""

    parts = dotted.split(".")
    if parts[0] != "evoparam":
        return None
    if len(parts) > 1 and parts[1] in MODULES:
        return parts[1]
    return "__init__"


def _source(path):
    rel = path.relative_to(PACKAGE)
    return rel.parts[0][:-3] if len(rel.parts) == 1 else rel.parts[0]


def _files():
    return [p for p in sorted(PACKAGE.rglob("*.py"))
            if p.relative_to(PACKAGE).parts[0] != "tests"]


def _edges(sources=None):
    """{(source module, file): {(target module, dotted import)}}; ``sources`` maps a file to
    replacement text (used by the mutation check)."""

    edges = defaultdict(set)
    sources = sources or {}
    for path in _files() + [p for p in sources if not p.exists()]:
        source = _source(path)
        for dotted in _imports(path, sources.get(path)):
            dest = _target(dotted)
            if dest is not None and dest != source:
                edges[(source, path.relative_to(PACKAGE).as_posix())].add((dest, dotted))
    return edges


def _violations(edges):
    out = []
    for (source, rel_path), targets in edges.items():
        if source not in RULES:
            out.append(f"{rel_path}: module '{source}' is not in the dependency table")
            continue
        allowed = RULES[source]
        for dest, dotted in sorted(targets):
            if dest in LEAVES and source not in LEAVES:
                continue
            if ANY in allowed:
                if dest == "__init__" and source != "__init__":
                    out.append(f"{rel_path} -> package root ({dotted})")
                elif source in TOP and dest in TOP and (source, dest) not in ALLOWED_BETWEEN:
                    out.append(f"{rel_path} -> {dest} ({dotted})")
                continue
            if dest not in allowed:
                out.append(f"{rel_path} -> {dest} ({dotted})")
            elif dest == "utils" and source in REQUIRE_ONLY and \
                    not any(dotted == ok or dotted.startswith(ok + ".")
                            for ok in REQUIRE_ONLY[source]):
                out.append(f"{rel_path} -> {dotted} (only utils.require is allowed)")
    return sorted(out)


def _cycle(edges):
    graph = defaultdict(set)
    for (source, _), targets in edges.items():
        for dest, _ in targets:
            if source != "__init__" and dest != "__init__":
                graph[source].add(dest)
    visiting, done = set(), set()

    def visit(node, trail):
        if node in done:
            return None
        if node in visiting:
            return trail[trail.index(node):] + [node]
        visiting.add(node)
        for nxt in sorted(graph[node]):
            found = visit(nxt, trail + [node])
            if found:
                return found
        visiting.discard(node)
        done.add(node)
        return None

    for node in sorted(graph):
        found = visit(node, [])
        if found:
            return found
    return None


def test_every_module_is_in_the_table():
    assert set(MODULES) | {"__init__"} <= set(RULES), sorted(set(MODULES) - set(RULES))


def test_edges_follow_spec_table():
    """(i) no edge outside the resolved table of §4.2, no module exempt."""

    violations = _violations(_edges())
    assert not violations, "imports not allowed by spec §4.2:\n" + "\n".join(violations)


def test_no_cycles():
    """(ii) the module graph has no cycle."""

    cycle = _cycle(_edges())
    assert cycle is None, "import cycle: " + " -> ".join(cycle)


def _top_level_imports(tree):
    """Imports executed when the module is imported: module and class bodies, including
    ``try``/``if`` blocks, except ``if TYPE_CHECKING:``; function bodies are skipped."""

    def walk(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield alias.name, node.lineno
                continue
            if isinstance(node, ast.ImportFrom):
                if not node.level and node.module:
                    yield node.module, node.lineno
                continue
            if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
                yield from walk(node.orelse)
                continue
            for name in ("body", "orelse", "finalbody"):
                yield from walk(getattr(node, name, None) or [])
            for handler in getattr(node, "handlers", None) or []:
                yield from walk(handler.body)

    yield from walk(tree.body)


def _optional_top_level(sources=None):
    sources = sources or {}
    stdlib = set(sys.stdlib_module_names)
    found = []
    for path in _files():
        tree = ast.parse(sources.get(path, path.read_text(encoding="utf-8")))
        for name, line in _top_level_imports(tree):
            root = name.split(".")[0]
            if root not in stdlib and root not in HARD:
                found.append(f"{path.relative_to(PACKAGE).as_posix()}:{line}: import {name}")
    return found


def test_no_top_level_import_of_optional_packages():
    """(iii) optional packages are imported inside functions only (REQ-296, REQ-292)."""

    found = _optional_top_level()
    assert not found, "top-level imports of optional packages:\n" + "\n".join(found)


def test_mutation_forbidden_import_is_detected():
    """REQ-296 mutation check: ``import evoparam.models`` added to shapes/__init__.py, or
    ``from evoparam import models`` (the form the earlier test missed), makes the test fail;
    so do a cycle and a top-level ``import pandas``."""

    init = PACKAGE / "shapes" / "__init__.py"
    text = init.read_text(encoding="utf-8")
    for line in ("import evoparam.models\n", "from evoparam import models\n",
                 "from .. import models\n"):
        violations = _violations(_edges({init: text + line}))
        assert any(v.startswith("shapes/__init__.py -> models") for v in violations), line
    # data imports nothing but the leaves and _plugins
    data_init = PACKAGE / "data" / "__init__.py"
    violations = _violations(_edges({data_init: data_init.read_text() + "import evoparam\n"}))
    assert any("data/__init__.py -> __init__" in v or "package root" in v for v in violations)
    # a cycle: models -> existence -> models
    models_init = PACKAGE / "models" / "__init__.py"
    edges = _edges({models_init: models_init.read_text() + "from evoparam import existence\n"})
    assert _cycle(edges) is not None
    # a top-level optional import, including one guarded by try/except
    guarded = text + "try:\n    import pandas\nexcept ImportError:\n    pandas = None\n"
    assert any("import pandas" in f for f in _optional_top_level({init: guarded}))


def test_light_core_import_pulls_no_optional_package():
    """REQ-010 / F66: ``import evoparam`` loads no optional package and nothing of the
    inherited code (removed, spec §4.5 and §12 rule 1)."""

    import subprocess

    code = ("import sys, evoparam\n"
            "print(','.join(sorted({m.split('.')[0] for m in sys.modules})))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(PACKAGE.parent), check=True).stdout.strip()
    loaded = sorted(set(out.split(",")) & OPTIONAL_IMPORTS)
    assert loaded == [], loaded
    for legacy in ("objs", "plts", "core", "analysis", "bands", "param"):
        assert not list((PACKAGE / legacy).rglob("*.py")), legacy
    for legacy in ("data.py", "params.py", "io.py", "reports.py", "download.py", "debug.py"):
        assert not (PACKAGE / "utils" / legacy).exists(), legacy
    for name in ("Bands", "evoparamGroup", "fit_evoparam_3d", "evoparam"):
        assert not hasattr(evoparam, name), name


def test_resolved_table_matches_decisions():
    """docs/spec/DECISIONS.md lists the same resolved table (D3/D4/D5)."""

    if not DECISIONS.is_file():
        pytest.skip("docs/ not in this checkout")
    text = DECISIONS.read_text(encoding="utf-8")
    section = text.split("<!-- import-table -->")[1]
    table = {}
    for row in section.splitlines():
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) < 2 or not cells[0].startswith("`"):
            continue
        names = re.findall(r"`([^`]+)`", cells[0])
        allowed = set(re.findall(r"`([^`]+)`", cells[1]))
        if "any" in cells[1].lower():
            allowed = {ANY}
        for name in names:
            table[name] = allowed
    for name, allowed in RULES.items():
        if name in LEAVES:
            continue
        assert table.get(name) == allowed, (name, table.get(name), allowed)


def test_sim_does_not_reach_the_fitter():
    """REQ-254: nothing that evoparam.sim imports, directly or through other modules, is a
    fitting engine, so the ground truth cannot depend on the estimator."""

    def files_of(dotted):
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

    start = sorted((PACKAGE / "sim").rglob("*.py"))
    seen, queue, trail = set(start), list(start), {path: None for path in start}
    while queue:
        path = queue.pop()
        for target in _imports(path):
            if target.split(".")[0] != "evoparam":
                continue
            for dest in files_of(target):
                if dest not in seen:
                    seen.add(dest)
                    trail[dest] = path
                    queue.append(dest)
    reached = [path for path in seen if path.relative_to(PACKAGE).parts[0] == "models"]

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
    stats_dests = {dest for (source, _), targets in edges.items() if source == "stats"
                   for dest, _ in targets}
    assert stats_dests
    assert not stats_dests & {"hierarchical", "trials", "spatial", "tf", "sim"}, stats_dests
    for (source, _), targets in edges.items():
        if "stats" in {dest for dest, _ in targets}:
            assert source in {"hierarchical", "__init__"} | TOP, source


def test_spatial_reaches_shapes_and_prep_through_models():
    """evoparam.spatial imports data, models and utils only (spec §4.2)."""

    dests = {dest for (source, _), targets in _edges().items() if source == "spatial"
             for dest, _ in targets}
    assert dests and dests - LEAVES <= RULES["spatial"]
    assert {"data", "models"} <= dests
