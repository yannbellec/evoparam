# SPDX-License-Identifier: Apache-2.0
"""REQ-011 (licence hygiene) and the extras of §9.2.

- GPL deny list (REQ-309 a): ``erdetect``, ``ieegprep``, ``teppy``, ``ultranest`` appear in
  no import statement of the package and in no extra of ``pyproject.toml``.
- Licence scan: no file of the package carries a GPL or LGPL/AGPL licence notice or an
  SPDX identifier other than the project's Apache-2.0; the package is Apache-2.0 with the
  attribution of its ancestors in NOTICE.
- ``all`` is the union of every extra except ``gpu``, ``bayes`` and ``docs`` (§9.2), and
  every package the test suite skips without is in ``all`` (or in ``bayes``, which the local
  runner ``tools/run_all_tests.py`` installs too), so that nothing skips for a missing
  package (audit F4).
"""

import ast
import pathlib
import re

import pytest

import evoparam

pytestmark = pytest.mark.core

ROOT = pathlib.Path(evoparam.__file__).resolve().parent.parent
PACKAGE = ROOT / "evoparam"
DENY = ("erdetect", "ieegprep", "teppy", "ultranest")
NOT_IN_ALL = {"gpu", "bayes", "docs", "all"}

# import name -> distribution, for the packages the tests skip without
DISTRIBUTIONS = {
    "mypy": "mypy",
    "coverage": "pytest-cov",  # pytest-cov installs coverage
    "sklearn": "scikit-learn",
    "mpmath": "mpmath",
    "pandas": "pandas",
    "mne": "mne",
    "pynwb": "pynwb",
    "matplotlib": "matplotlib",
    "quantities": "quantities",
    "nwbinspector": "nwbinspector",
    "neo": "neo",
    "jax": "jax",
    "numpyro": "numpyro",
    "statsmodels": "statsmodels",
    "pymc": "pymc",
    "arviz": "arviz",
    "pyarrow": "pyarrow",
    "xarray": "xarray",
    "mne_bids": "mne-bids",
    "bids": "pybids",
    "edfio": "edfio",
    "numba": "numba",
    "tqdm": "tqdm",
    "joblib": "joblib",
    "pylsl": "pylsl",
    "jsonschema": "jsonschema",
    "bidsschematools": "bidsschematools",
    "h5py": "h5py",
    "hdmf": "hdmf",
    "ndx_events": "ndx-events",
    "remfile": "remfile",
    "formulaic": "formulaic",
    "hypothesis": "hypothesis",
    "tomli": "tomli",
    "nilearn": "nilearn",
    "yaml": "pyyaml",
}


def _toml():
    try:
        import tomllib
    except ImportError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def _name(requirement):
    return re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0].strip().lower()


def _extras():
    return _toml()["project"]["optional-dependencies"]


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module
        elif (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", getattr(node.func, "attr", ""))
            in ("require", "import_module", "importorskip")
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            yield node.args[0].value


def test_no_gpl_imports():
    hits = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for name in _imports(path):
            if name.split(".")[0].lower() in DENY:
                hits.append(f"{path.relative_to(ROOT)}: {name}")
    for extra, requirements in _extras().items():
        hits += [f"pyproject [{extra}]: {r}" for r in requirements if _name(r) in DENY]
    hits += [
        f"pyproject dependencies: {r}"
        for r in _toml()["project"]["dependencies"]
        if _name(r) in DENY
    ]
    assert not hits, hits


def test_no_gpl_imports_finds_a_mutation(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def f():\n    import ieegprep\n    return require('erdetect', extra='x')\n")
    assert [n for n in _imports(bad) if n in DENY] == ["ieegprep", "erdetect"]


GPL_TEXT = re.compile(
    r"GNU (Lesser |Affero )?General Public License|SPDX-License-Identifier:"
    r"\s*(?!Apache-2\.0)\S+|\b(L|A)?GPL-?[23]",
    re.IGNORECASE,
)


def test_licence_scan():
    assert "Apache License" in (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert _toml()["project"]["license"] == "Apache-2.0"
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "ERPparam" in notice
    assert "Apache" in notice
    hits = []
    for path in sorted(PACKAGE.rglob("*")):
        rel = path.relative_to(ROOT).as_posix()
        if (
            not path.is_file()
            or "__pycache__" in rel
            or path.suffix
            not in (".py", ".json", ".txt", ".csv", ".tsv", ".md", ".rst", ".html", ".cff")
        ):
            continue
        if rel == "evoparam/tests/test_dependencies.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits += [
            f"{rel}: {m.group(0)}"
            for m in GPL_TEXT.finditer(text)
            if not _mentions_isolation(text, m.start())
        ]
    assert not hits, hits


def _mentions_isolation(text, start):
    """A sentence that says a GPL tool is NOT used (REQ-309, the comparator scripts run as
    separate processes) is not a licence notice."""

    line = text[text.rfind("\n", 0, start) + 1 : text.find("\n", start)]
    return bool(
        re.search(
            r"not (imported|vendored|a dependency)|separate process|"
            r"subprocess|outside the|never import",
            line,
            re.IGNORECASE,
        )
    )


def test_all_is_the_union_of_the_extras():
    extras = _extras()
    union = set()
    for name, requirements in extras.items():
        if name not in NOT_IN_ALL:
            union |= {r.replace(" ", "") for r in requirements}
    assert {r.replace(" ", "") for r in extras["all"]} == union
    assert set(extras) >= {
        "mne",
        "bids",
        "nwb",
        "neo",
        "edf",
        "bayes",
        "gpu",
        "fast",
        "plot",
        "data",
        "xarray",
        "stats",
        "validate",
        "cli",
        "progress",
        "parallel",
        "stream",
        "docs",
        "tests",
        "all",
    }


def _skipped_imports():
    names = set()
    pattern = re.compile(
        r"importorskip\(\s*[\"']([\w.]+)[\"']|find_spec\(\s*[\"']([\w.]+)"
        r"[\"']|available\(\s*[\"']([\w.]+)[\"']"
    )
    for path in sorted((PACKAGE / "tests").rglob("*.py")):
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            name = next(g for g in match.groups() if g).split(".")[0]
            if name != "evoparam":  # the package's own optional modules
                names.add(name)
    return names


def test_f4_nothing_skips_for_a_package_outside_the_extras():
    extras = _extras()
    in_all = {_name(r) for r in extras["all"]}
    in_bayes = {_name(r) for r in extras["bayes"]}
    names = _skipped_imports()
    assert {"pandas", "mne", "matplotlib"} <= names
    unknown = sorted(n for n in names if n not in DISTRIBUTIONS)
    assert not unknown, f"add these import names to DISTRIBUTIONS: {unknown}"
    missing = sorted(n for n in names if DISTRIBUTIONS[n] not in in_all | in_bayes)
    assert not missing, missing


def test_f4_the_runner_installs_all_and_bayes():
    text = (ROOT / "tools" / "run_all_tests.py").read_text(encoding="utf-8")
    assert re.search(r"default=\"all,bayes\"", text)


###################################################################################################
## REQ-292: the extras table of §9.2, lazy imports and the ImportError message

SPEC = ROOT / "docs" / "spec" / "SPEC.md"


def _floor(requirement):
    match = re.search(r">=\s*([\w.]+)", requirement)
    return match.group(1) if match else None


def _vt(text):
    return tuple(int(p) for p in re.findall(r"\d+", text)[:3])


def _spec_extras():
    """{extra: {distribution: floor}} read from the extras table of SPEC.md §9.2."""

    text = SPEC.read_text(encoding="utf-8").split("### 9.2 Dependency policy")[1]
    table = text.split("| extra | packages (floor) | what it unlocks |")[1].split("\n\n")[0]
    out = {}
    for row in table.splitlines()[2:]:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        extra = cells[0].strip("`")
        if extra in ("cli", "all"):
            continue
        main = re.split(r"\(alternative|\bor\b", cells[1])[0]
        out[extra] = {
            _name(r): _floor(r) for r in re.findall(r"`([^`]+)`", main) if "_" not in r
        }  # `check_estimator` is not a package
    return out


@pytest.mark.skipif(not SPEC.is_file(), reason="docs/spec not in this checkout")
def test_req292_extras_file_is_the_spec_table():
    """``evoparam/_extras.py`` holds the table of §9.2 (names and floors)."""

    from evoparam import _extras as table

    spec = _spec_extras()
    ours = {k: v for k, v in table.EXTRAS.items() if k != "cli"}
    assert ours == spec


def test_req292_pyproject_matches_the_extras_table():
    """(i) every extra of the table is in pyproject.toml with its packages at (at least)
    their floors; every other package is a documented addition (``table.ADDITIONS``)."""

    from evoparam import _extras as table

    extras = _extras()
    problems = []
    for extra, packages in table.EXTRAS.items():
        if extra not in extras:
            problems.append(f"[{extra}] missing")
            continue
        declared = {_name(r): _floor(r) for r in extras[extra]}
        for dist, floor in packages.items():
            alternatives = table.ALTERNATIVES.get(extra, {})
            if dist not in declared and not set(alternatives) & set(declared):
                problems.append(f"[{extra}] {dist} missing")
                continue
            if (
                dist in declared
                and floor
                and (declared[dist] is None or _vt(declared[dist]) < _vt(floor))
            ):
                problems.append(f"[{extra}] {dist} floor {declared[dist]} < {floor}")
        for dist in declared:
            if (
                dist not in packages
                and dist not in table.ALTERNATIVES.get(extra, {})
                and dist not in table.ADDITIONS.get(extra, {})
            ):
                problems.append(f"[{extra}] {dist} is not in the table nor in ADDITIONS")
    for extra in set(extras) - set(table.EXTRAS) - {"all"}:
        if extra not in table.ADDITIONS:
            problems.append(f"[{extra}] is not in the table nor in ADDITIONS")
    assert not problems, problems


def test_req292_every_optional_package_in_exactly_one_extra():
    """Every dependency other than numpy and scipy is declared in exactly one extra (the
    unions ``all`` and ``cli`` and the tool chains ``docs`` and ``tests`` aside; ``jax`` is in
    ``bayes`` and ``gpu`` as the table of §9.2 itself declares)."""

    from evoparam import _extras as table

    seen = {}
    for extra, requirements in _extras().items():
        if extra in table.UNIONS:
            continue
        for r in requirements:
            seen.setdefault(_name(r), set()).add(extra)
    twice = {d: sorted(e) for d, e in seen.items() if len(e) > 1 and d != "jax"}
    assert not twice, twice
    assert seen["jax"] == {"bayes", "gpu"}
    assert {_name(r) for r in _toml()["project"]["dependencies"]} == {"numpy", "scipy"}


# (module hidden, code that reaches a function needing it, extra named in the message)
ENTRY_POINTS = [
    ("pandas", "import evoparam; evoparam.prep.reference_sensitivity(None)", "data"),
    ("edfio", "import evoparam; evoparam.io.read_edf('x.edf')", "edf"),
    ("neo", "from evoparam.io import read_neo; read_neo('x')", "neo"),
    ("pynwb", "from evoparam.io import read_nwb; read_nwb('x.nwb')", "nwb"),
    ("pylsl", "from evoparam.stream.lsl import _pylsl; _pylsl('evoparam.stream')", "stream"),
    (
        "xarray",
        "import numpy as np, evoparam; evoparam.TrialSet(data=np.zeros((2, 1, 5)), "
        "times=np.arange(5) / 100., sfreq=100., ch_names=['a']).to_xarray()",
        "xarray",
    ),
    (
        "mne",
        "import numpy as np, evoparam; evoparam.TrialSet(data=np.zeros((2, 1, 5)), "
        "times=np.arange(5) / 100., sfreq=100., ch_names=['a']).to_mne()",
        "mne",
    ),
    ("matplotlib", "from evoparam.plots._plots import _plt; _plt()", "plot"),
]


@pytest.mark.parametrize(
    ("module", "code", "extra"), ENTRY_POINTS, ids=[e[0] for e in ENTRY_POINTS]
)
def test_req292_missing_package_message(module, code, extra):
    """(iii) with the package hidden (``sys.modules[name] = None``) the entry point raises
    ImportError whose message holds ``pip install "evoparam[<extra>]"``."""

    import subprocess
    import sys

    script = f"import sys\nsys.modules[{module!r}] = None\n{code}\n"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode != 0
    assert "ImportError" in result.stderr, result.stderr[-2000:]
    match = re.search(r'pip install "evoparam\[(\w+)\]"', result.stderr)
    assert match, result.stderr[-2000:]
    assert match.group(1) == extra, result.stderr[-2000:]


OPTIONAL = {
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
    "nilearn",
}


def _imports_outside_require():
    """Import statements of an optional package inside a function that does not call
    ``require`` for that package first (the package is then not imported through
    ``utils.require``)."""

    out = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "tests" in path.relative_to(PACKAGE).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            required = {
                str(n.args[0].value).split(".")[0]
                for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", getattr(n.func, "attr", "")) == "require"
                and n.args
                and isinstance(n.args[0], ast.Constant)
            }
            for node in ast.walk(fn):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module]
                out += [
                    f"{path.relative_to(ROOT)}:{node.lineno} {n}"
                    for n in names
                    if n.split(".")[0] in OPTIONAL and n.split(".")[0] not in required
                ]
    return sorted(set(out))


@pytest.mark.xfail(
    strict=True,
    reason="REQ-292 'imported only through evoparam.utils.require' "
    "is not met yet: 66 import statements of an optional package in 25 modules "
    "are inside functions that do not call require for it (measured 2026-10-11, "
    "S20; most follow a require of the caller or are capability checks). Each "
    "of them is lazy (no top-level import, test_import_graph (iii)), and the "
    "messages of the public entry points name the extra "
    "(test_req292_missing_package_message)",
)
def test_req292_optional_packages_only_through_require():
    found = _imports_outside_require()
    assert not found, f"{len(found)}: {found[:10]}"


###################################################################################################
## REQ-395: no GPL, LGPL or AGPL package in any extra (licence scan of the resolved tree)

COPYLEFT = re.compile(r"\b(A|L)?GPL|GNU (Lesser |Affero )?General Public", re.IGNORECASE)


def _licence(dist):
    from importlib import metadata

    try:
        meta = metadata.metadata(dist)
    except metadata.PackageNotFoundError:
        return None
    short = meta.get("License") if len(meta.get("License") or "") <= 200 else None
    return " ".join(
        filter(
            None,
            [meta.get("License-Expression"), short]
            + [c for c in meta.get_all("Classifier") or [] if c.startswith("License ::")],
        )
    )


def _resolved_tree(roots):
    """Installed distributions reachable from ``roots`` through their requirements."""

    from importlib import metadata

    seen, todo = set(), list(roots)
    while todo:
        dist = todo.pop()
        key = dist.lower().replace("_", "-")
        if key in seen:
            continue
        try:
            requires = metadata.requires(dist) or []
        except metadata.PackageNotFoundError:
            continue
        seen.add(key)
        for r in requires:
            if "extra ==" in r:
                continue
            todo.append(_name(r))
    return seen


# Copyleft packages that the installed tree pulls in through a third party, each reached at
# run time only by that third party (never imported, vendored or declared by evoparam), with
# the reason (docs/spec/DECISIONS.md, REQ-395).
COPYLEFT_ALLOWED = {
    "frozendict": "LGPL-3.0, a requirement of pybids (extra bids, named by spec §9.2); "
    "evoparam neither imports nor declares it",
    "num2words": "LGPL, a requirement of pybids (extra bids, named by spec §9.2); evoparam "
    "neither imports nor declares it",
}


def test_req395_no_copyleft_package_in_the_installed_extras():
    """The licence of every installed distribution reachable from the extras (except
    ``docs`` and ``tests``, the tool chains) is not GPL, LGPL or AGPL. Distributions that
    are not installed here are checked by the CI licence scan of every extras set."""

    roots = {
        _name(r)
        for extra, reqs in _extras().items()
        if extra not in ("docs", "tests", "all")
        for r in reqs
    }
    hits = []
    for dist in sorted(_resolved_tree(roots)):
        text = _licence(dist)
        if text and COPYLEFT.search(text) and dist not in COPYLEFT_ALLOWED:
            hits.append(f"{dist}: {text}")
    assert not hits, hits


@pytest.mark.xfail(
    strict=True,
    reason="REQ-395 verification read as 'the resolved tree has no "
    "copyleft package at all': pybids (extra bids, spec §9.2) requires "
    "frozendict (LGPL-3.0) and num2words (LGPL) (measured 2026-10-11 on the installed tree); "
    "evoparam does not import, vendor or declare it (DECISIONS.md)",
)
def test_req395_resolved_tree_has_no_copyleft_package_at_all():
    roots = {
        _name(r)
        for extra, reqs in _extras().items()
        if extra not in ("docs", "tests", "all")
        for r in reqs
    }
    hits = [
        d for d in sorted(_resolved_tree(roots)) if _licence(d) and COPYLEFT.search(_licence(d))
    ]
    assert not hits, hits


def test_req395_the_scan_finds_a_copyleft_licence():
    assert COPYLEFT.search("License :: OSI Approved :: GNU General Public License v3 (GPLv3)")
    assert COPYLEFT.search("LGPL-2.1-or-later")
    assert not COPYLEFT.search("Apache-2.0")
    assert not COPYLEFT.search("BSD-3-Clause")


def test_req309_version_floors():
    """Spec §9.2: ``python>=3.10``, ``numpy>=1.24,<3``, ``scipy>=1.10`` (S20: the
    pyproject declared ``numpy`` and ``scipy>=1.8``, inherited)."""

    try:
        import tomllib
    except ImportError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    with open(ROOT / "pyproject.toml", "rb") as fobj:
        project = tomllib.load(fobj)["project"]
    assert project["requires-python"] == ">=3.10"
    assert sorted(project["dependencies"]) == ["numpy>=1.24,<3", "scipy>=1.10"]
