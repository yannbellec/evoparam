# SPDX-License-Identifier: Apache-2.0
"""Import-graph test (REQ-296): module dependencies follow the table of spec §4.2.

Every module of the package is parsed with ``ast`` and the directed graph of intra-package
imports is built; ``from evoparam import plots`` is an edge to ``plots`` (not to the
package root). The test fails on (i) any edge not allowed by the table, (ii) any cycle and
(iii) any top-level import of an optional package. No module is exempt. Extended for REQ-395
with a licence allow-list: every third-party package the package imports has an allowed
licence in the SOUP inventory (``soup.json``).

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
from typing import Any

import pytest

import evoparam

# Cells of a row of the import table (module, imports).
MIN_CELLS = 2

pytestmark = pytest.mark.core

PACKAGE = pathlib.Path(evoparam.__file__).parent
DECISIONS = PACKAGE.parent / "docs" / "spec" / "DECISIONS.md"

# Leaf modules every module may import: the error hierarchy (§4.6, D4), the version
# (§4.1 principle 3) and the array type aliases (REQ-301).
LEAVES = {"errors", "version", "_typing"}

# Every top-level module of the package (subpackages and single files).
MODULES = sorted(
    {
        p.name if p.is_dir() else p.stem
        for p in PACKAGE.iterdir()
        if (p.is_dir() and (p / "__init__.py").is_file())
        or (p.suffix == ".py" and p.stem != "__init__")
    }
    - {"tests"}
)

ANY = "*"
TOP = {"reports", "plots", "cli", "stream", "benchmark"}

# Spec §4.2 with D3/D4/D5 resolved: module -> modules it may import (besides LEAVES).
RULES = {
    # D4/D5: the plugins used for a result are recorded in its provenance (REQ-328);
    # optional packages through utils.require only (§9.2, REQ-292).
    "data": {"_plugins", "utils"},
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
    # D5: the preset registry loads the evoparam.presets entry points (REQ-328);
    # optional packages through utils.require only (§9.2, REQ-292).
    "presets": {"_plugins", "utils"},
    # Cross-cutting helpers (§4.2 figure): the leaves only.
    "utils": set(),
    # Entry-point loader of §10.8: the leaves only.
    "_plugins": set(),
    # Data tables of §9.2 and §9.3 (REQ-292 extras, REQ-293 deprecated names).
    "_extras": set(),
    "_deprecated": set(),
    "reports": {ANY},
    "plots": {ANY},
    "cli": {ANY},
    "stream": {ANY},
    "benchmark": {ANY},
    # The package root re-exports the public API; ``python -m evoparam`` runs the cli.
    "__init__": {ANY},
    "__main__": {"cli"},
    "errors": set(),
    "version": set(),
    "_typing": set(),
}
# Modules whose only allowed use of utils is utils.require (the rows of §4.2 do not list
# utils for them; §9.2 makes require the only way to import an optional package).
REQUIRE_ONLY = {
    "data": {"evoparam.utils._optional", "evoparam.utils.require"},
    "presets": {"evoparam.utils._optional", "evoparam.utils.require"},
    "hierarchical": {"evoparam.utils._optional", "evoparam.utils.require"},
    "sim": {"evoparam.utils._optional", "evoparam.utils.require"},
}
# Spec §4.2: top layers do not import each other, except plots < reports < cli (D3) and
# cli -> benchmark -> reports (S19).
ALLOWED_BETWEEN = {
    ("benchmark", "reports"),
    ("reports", "plots"),
    ("cli", "reports"),
    ("cli", "plots"),
    ("benchmark", "plots"),
    ("cli", "benchmark"),
}

# Import names of the packages of the extras of spec §9.2 (and of the test tool chain).
OPTIONAL_IMPORTS = {
    "mne",
    "mne_bids",
    "bids",
    "pynwb",
    "hdmf",
    "ndx_events",
    "neo",
    "edfio",
    "pyedflib",
    "numpyro",
    "jax",
    "arviz",
    "pymc",
    "cupy",
    "numba",
    "matplotlib",
    "pandas",
    "pyarrow",
    "xarray",
    "statsmodels",
    "formulaic",
    "bidsschematools",
    "nwbinspector",
    "jsonschema",
    "tqdm",
    "joblib",
    "pylsl",
    "sklearn",
    "h5py",
    "remfile",
    "quantities",
    "mpmath",
    "pytest",
    "hypothesis",
    "click",
}

# (iii) packages that may be imported at the top level of a module: the standard library
# and the two hard dependencies (REQ-291).
HARD = {"numpy", "scipy", "evoparam", "__future__"}


def _module_name(path: Any) -> Any:
    rel = path.relative_to(PACKAGE).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ["evoparam"] + parts


def _imports(path: Any, source: Any = None) -> Any:
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
                base = package[: len(package) - (node.level - 1)]
                module = base + (node.module.split(".") if node.module else [])
            else:
                module = node.module.split(".") if node.module else []
            for alias in node.names:
                yield ".".join(module + [alias.name])


def _target(dotted: Any) -> Any:
    """Top-level module of the package that a dotted import reaches (``__init__`` for the
    package root itself), or None outside the package."""

    parts = dotted.split(".")
    if parts[0] != "evoparam":
        return None
    if len(parts) > 1 and parts[1] in MODULES:
        return parts[1]
    return "__init__"


def _source(path: Any) -> Any:
    rel = path.relative_to(PACKAGE)
    return rel.parts[0][:-3] if len(rel.parts) == 1 else rel.parts[0]


def _files() -> Any:
    return [p for p in sorted(PACKAGE.rglob("*.py")) if p.relative_to(PACKAGE).parts[0] != "tests"]


def _edges(sources: Any = None) -> Any:
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


def _violations(edges: Any) -> Any:
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
            elif (
                dest == "utils"
                and source in REQUIRE_ONLY
                and not any(
                    dotted == ok or dotted.startswith(ok + ".") for ok in REQUIRE_ONLY[source]
                )
            ):
                out.append(f"{rel_path} -> {dotted} (only utils.require is allowed)")
    return sorted(out)


def _cycle(edges: Any) -> Any:
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
            return trail[trail.index(node) :] + [node]
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


def test_every_module_is_in_the_table() -> None:
    assert set(MODULES) | {"__init__"} <= set(RULES), sorted(set(MODULES) - set(RULES))


def test_edges_follow_spec_table() -> None:
    """(i) no edge outside the resolved table of §4.2, no module exempt."""

    violations = _violations(_edges())
    assert not violations, "imports not allowed by spec §4.2:\n" + "\n".join(violations)


def test_no_cycles() -> None:
    """(ii) the module graph has no cycle."""

    cycle = _cycle(_edges())
    assert cycle is None, "import cycle: " + " -> ".join(cycle)


def _top_level_imports(tree: Any) -> Any:
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


def _optional_top_level(sources: Any = None) -> Any:
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


def test_no_top_level_import_of_optional_packages() -> None:
    """(iii) optional packages are imported inside functions only (REQ-296, REQ-292)."""

    found = _optional_top_level()
    assert not found, "top-level imports of optional packages:\n" + "\n".join(found)


def test_mutation_forbidden_import_is_detected() -> None:
    """REQ-296 mutation check: ``import evoparam.models`` added to shapes/__init__.py, or
    ``from evoparam import models`` (the form the earlier test missed), makes the test fail;
    so do a cycle and a top-level ``import pandas``."""

    init = PACKAGE / "shapes" / "__init__.py"
    text = init.read_text(encoding="utf-8")
    for line in (
        "import evoparam.models\n",
        "from evoparam import models\n",
        "from .. import models\n",
    ):
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


###################################################################################################
## REQ-395: licence allow-list of the third-party packages the package imports

# Licences (SPDX) a third-party package imported by evoparam may have: permissive licences
# and file-level MPL-2.0 (tqdm, hypothesis); never a copyleft licence (spec §13.7, REQ-395).
LICENCE_ALLOW_LIST = {"Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "MIT", "PSF-2.0", "MPL-2.0"}
# typing-only module of the type checkers (imported under ``if TYPE_CHECKING:``)
TYPING_ONLY = {"_typeshed"}
SOUP = PACKAGE.parent / "soup.json"


# modules of the standard library of a later supported Python than the running one
# (sys.stdlib_module_names is that of the running Python): tomllib from Python 3.11
LATER_STDLIB = frozenset({"tomllib"})


def _third_party_imports(sources: Any = None) -> Any:
    """{import name: [module, ...]} of every third-party package the package imports: import
    statements anywhere in a module and ``require("name", ...)`` calls. Scripts of the
    repository (``benchmarks/``, ``tools/``) that the tests import are first-party."""

    repo = PACKAGE.parent
    out: dict[str, list[str]] = defaultdict(list)
    for path, text in (sources or dict.fromkeys(sorted(PACKAGE.rglob("*.py")))).items():
        tree = ast.parse(text if text is not None else path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            elif (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", "")) == "require"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                names = [str(node.args[0].value)]
            for name in names:
                top = name.split(".")[0]
                if (
                    top in sys.stdlib_module_names
                    or top in LATER_STDLIB
                    or top in ("evoparam", "__future__")
                    or top in TYPING_ONLY
                    or any((repo / d / f"{top}.py").is_file() for d in ("benchmarks", "tools"))
                ):
                    continue
                out[top].append(path.relative_to(PACKAGE).as_posix())
    return out


def _licence_violations(imports: Any, soup: Any = None) -> Any:
    import json

    if soup is None:
        soup = {
            p["import_name"]: p["licence"]
            for p in json.loads(SOUP.read_text(encoding="utf-8"))["packages"]
        }
    bad = []
    for name, modules in sorted(imports.items()):
        licence = soup.get(name)
        if licence is None:
            bad.append(f"{name} (in {modules[0]}): not in soup.json, licence unknown")
            continue
        terms = set(re.split(r"\s+(?:AND|OR)\s+", licence))
        if not terms <= LICENCE_ALLOW_LIST:
            bad.append(f"{name} (in {modules[0]}): {licence}")
    return bad


@pytest.mark.skipif(not SOUP.is_file(), reason="soup.json not in this checkout")
def test_imported_packages_have_allowed_licences() -> None:
    """REQ-395: every third-party package that a module of the package imports (at the top
    level, inside a function or through ``evoparam.utils.require``) is in the SOUP inventory
    with a licence of ``LICENCE_ALLOW_LIST``."""

    imports = _third_party_imports()
    assert {"numpy", "scipy", "pandas", "mne"} <= set(imports)
    bad = _licence_violations(imports)
    assert not bad, bad


@pytest.mark.skipif(not SOUP.is_file(), reason="soup.json not in this checkout")
def test_licence_allow_list_finds_a_mutation() -> None:
    """An import of a copyleft package (here ``ieegprep``, not in the inventory) or of
    an inventoried package whose licence is not allowed makes the check fail."""

    path = PACKAGE / "shapes" / "__init__.py"
    text = path.read_text(encoding="utf-8")
    bad = _licence_violations(_third_party_imports({path: text + "import ieegprep\n"}))
    assert any(b.startswith("ieegprep") for b in bad)
    bad = _licence_violations(
        _third_party_imports({path: text + "def f():\n    return require('ieegprep')\n"})
    )
    assert any(b.startswith("ieegprep") for b in bad)
    assert _licence_violations({"numpy": ["x.py"]}, soup={"numpy": "Unknown-1.0"})
    assert not _licence_violations({"numpy": ["x.py"]}, soup={"numpy": "BSD-3-Clause"})


def test_light_core_import_pulls_no_optional_package() -> None:
    """REQ-010 / F66: ``import evoparam`` loads no optional package and nothing of the
    inherited code (removed, spec §4.5 and §12 rule 1)."""

    import subprocess

    code = (
        "import sys, evoparam\nprint(','.join(sorted({m.split('.')[0] for m in sys.modules})))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PACKAGE.parent),
        check=True,
    ).stdout.strip()
    loaded = sorted(set(out.split(",")) & OPTIONAL_IMPORTS)
    assert loaded == [], loaded
    for legacy in ("objs", "plts", "core", "analysis", "bands", "param"):
        assert not list((PACKAGE / legacy).rglob("*.py")), legacy
    for legacy in ("data.py", "params.py", "io.py", "reports.py", "download.py", "debug.py"):
        assert not (PACKAGE / "utils" / legacy).exists(), legacy
    for name in ("Bands", "evoparamGroup", "fit_evoparam_3d", "evoparam"):
        assert not hasattr(evoparam, name), name


def test_resolved_table_matches_decisions() -> None:
    """docs/spec/DECISIONS.md lists the same resolved table (D3/D4/D5)."""

    if not DECISIONS.is_file():
        pytest.skip("docs/ not in this checkout")
    text = DECISIONS.read_text(encoding="utf-8")
    section = text.split("<!-- import-table -->")[1]
    table = {}
    for row in section.splitlines():
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) < MIN_CELLS or not cells[0].startswith("`"):
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


def test_sim_does_not_reach_the_fitter() -> None:
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
    seen, queue, trail = set(start), list(start), dict.fromkeys(start)
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
    assert "sim" in modules
    assert "shapes" in modules
    assert "data" in modules


def test_stats_is_below_hierarchical() -> None:
    """Spec §4.2: hierarchical imports stats (variance components), never the reverse, and
    stats does not reach the trial-level module."""

    edges = _edges()
    stats_dests = {
        dest for (source, _), targets in edges.items() if source == "stats" for dest, _ in targets
    }
    assert stats_dests
    assert not stats_dests & {"hierarchical", "trials", "spatial", "tf", "sim"}, stats_dests
    for (source, _), targets in edges.items():
        if "stats" in {dest for dest, _ in targets}:
            assert source in {"hierarchical", "__init__"} | TOP, source


def test_spatial_reaches_shapes_and_prep_through_models() -> None:
    """evoparam.spatial imports data, models and utils only (spec §4.2)."""

    dests = {
        dest
        for (source, _), targets in _edges().items()
        if source == "spatial"
        for dest, _ in targets
    }
    assert dests
    assert dests - LEAVES <= RULES["spatial"]
    assert {"data", "models"} <= dests
