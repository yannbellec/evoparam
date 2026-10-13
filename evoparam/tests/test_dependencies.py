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
from typing import Any

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
    "sphinx": "sphinx",  # the documentation build (REQ-302), extra `docs`
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


def _toml() -> Any:
    try:
        import tomllib
    except ImportError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def _name(requirement: Any) -> Any:
    return re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0].strip().lower()


def _extras() -> Any:
    return _toml()["project"]["optional-dependencies"]


def _imports(path: Any) -> Any:
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


def test_no_gpl_imports() -> None:
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


def test_no_gpl_imports_finds_a_mutation(tmp_path: Any) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("def f():\n    import ieegprep\n    return require('erdetect', extra='x')\n")
    assert [n for n in _imports(bad) if n in DENY] == ["ieegprep", "erdetect"]


GPL_TEXT = re.compile(
    r"GNU (Lesser |Affero )?General Public License|SPDX-License-Identifier:"
    r"\s*(?!Apache-2\.0)\S+|\b(L|A)?GPL-?[23]",
    re.IGNORECASE,
)


def test_licence_scan() -> None:
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


def _mentions_isolation(text: Any, start: Any) -> Any:
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


def test_all_is_the_union_of_the_extras() -> None:
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


def _skipped_imports() -> Any:
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


def test_f4_nothing_skips_for_a_package_outside_the_extras() -> None:
    extras = _extras()
    in_all = {_name(r) for r in extras["all"]}
    in_bayes = {_name(r) for r in extras["bayes"]}
    # the documentation build (REQ-302) needs the `docs` extra, as the docs job of REQ-297
    in_docs = {_name(r) for r in extras["docs"]}
    names = _skipped_imports()
    assert {"pandas", "mne", "matplotlib"} <= names
    unknown = sorted(n for n in names if n not in DISTRIBUTIONS)
    assert not unknown, f"add these import names to DISTRIBUTIONS: {unknown}"
    missing = sorted(n for n in names if DISTRIBUTIONS[n] not in in_all | in_bayes | in_docs)
    assert not missing, missing


def test_f4_the_runner_installs_all_and_bayes() -> None:
    text = (ROOT / "tools" / "run_all_tests.py").read_text(encoding="utf-8")
    assert re.search(r"default=\"all,bayes\"", text)


###################################################################################################
## REQ-292: the extras table of §9.2, lazy imports and the ImportError message

SPEC = ROOT / "docs" / "spec" / "SPEC.md"


def _floor(requirement: Any) -> Any:
    match = re.search(r">=\s*([\w.]+)", requirement)
    return match.group(1) if match else None


def _vt(text: Any) -> Any:
    return tuple(int(p) for p in re.findall(r"\d+", text)[:3])


def _spec_extras() -> Any:
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
def test_req292_extras_file_is_the_spec_table() -> None:
    """``evoparam/_extras.py`` holds the table of §9.2 (names and floors)."""

    from evoparam import _extras as table

    spec = _spec_extras()
    ours = {k: v for k, v in table.EXTRAS.items() if k != "cli"}
    assert ours == spec


def test_req292_pyproject_matches_the_extras_table() -> None:
    """(i) every extra of the table is in pyproject.toml with its packages at (at least)
    their floors; every other package is a documented addition (``table.ADDITIONS``). A
    package of the table that is not declared is a documented withholding
    (``table.WITHHELD``, with its reason: docs/spec/DECISIONS.md, section 5.7)."""

    from evoparam import _extras as table

    extras = _extras()
    problems = []
    for extra, packages in table.EXTRAS.items():
        if extra not in extras:
            problems.append(f"[{extra}] missing")
            continue
        declared = {_name(r): _floor(r) for r in extras[extra]}
        withheld = table.WITHHELD.get(extra, {})
        problems += [
            f"[{extra}] {d} is declared and withheld" for d in set(withheld) & set(declared)
        ]
        for dist, floor in packages.items():
            if dist in withheld:
                continue
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


def test_req292_every_optional_package_in_exactly_one_extra() -> None:
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
def test_req292_missing_package_message(module: Any, code: Any, extra: Any) -> None:
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


def _imports_outside_require() -> Any:
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


def test_req292_optional_packages_only_through_require() -> None:
    found = _imports_outside_require()
    assert not found, f"{len(found)}: {found[:10]}"


###################################################################################################
## REQ-395: no GPL, LGPL or AGPL package in any extra (licence scan of the resolved tree)


def _scan_tool() -> Any:
    """``tools/licence_scan.py``, the scan the CI job ``licence-scan`` runs on each extra."""

    import importlib.util

    path = ROOT / "tools" / "licence_scan.py"
    if not path.is_file():
        pytest.skip("tools/ not in this checkout")
    spec = importlib.util.spec_from_file_location("licence_scan", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_req395_resolved_tree_has_no_copyleft_package() -> None:
    """The licence of every installed distribution reachable from every extra of
    ``pyproject.toml`` (the tool chains ``docs`` and ``tests`` included) is not copyleft. A
    distribution that puts a copyleft licence on named files only is accepted when none of them
    is installed (``FILE_SCOPED``, checked on the installed files). Distributions that are
    not installed here are checked by the CI job ``licence-scan``, one environment per
    extra."""

    tool = _scan_tool()
    hits, _, n = tool.scan(sorted(_extras()))
    assert n > 0
    assert not hits, hits


def test_req395_the_scan_finds_a_copyleft_licence() -> None:
    tool = _scan_tool()
    assert tool.COPYLEFT.search("License :: OSI Approved :: GNU General Public License v3 (GPLv3)")
    assert tool.COPYLEFT.search("LGPL-2.1-or-later")
    assert tool.COPYLEFT.search("License :: OSI Approved :: GNU Lesser General Public License v3")
    assert not tool.COPYLEFT.search("Apache-2.0")
    assert not tool.COPYLEFT.search("BSD-3-Clause")
    assert not tool.COPYLEFT.search("MPL-2.0")
    # only the distributions named in FILE_SCOPED can be accepted, and only without the files
    assert not tool.file_scoped_ok("numpy")


def test_req395_licence_scan_job_covers_every_extra() -> None:
    """The CI job ``licence-scan`` installs each extra of ``pyproject.toml`` in its own
    environment and runs the scan on it."""

    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8"))
    job = doc["jobs"]["licence-scan"]
    assert set(job["strategy"]["matrix"]["extra"]) == set(_extras())
    runs = "\n".join(str(step.get("run", "")) for step in job["steps"])
    assert 'pip install ".[${{ matrix.extra }}]"' in runs
    assert "tools/licence_scan.py --extra ${{ matrix.extra }}" in runs


def test_req395_pull_request_checklist() -> None:
    """The pull-request template asks the REQ-395 question."""

    text = (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")
    assert re.search(r"- \[ \] no copyleft package .*REQ-395", text)


def test_req395_withheld_packages_are_documented() -> None:
    """A package of the §9.2 table that pyproject does not declare has its reason in
    ``evoparam/_extras.py`` and in docs/spec/DECISIONS.md."""

    from evoparam import _extras as table

    decisions = (ROOT / "docs" / "spec" / "DECISIONS.md").read_text(encoding="utf-8")
    for extra, packages in table.WITHHELD.items():
        for dist, reason in packages.items():
            assert reason.strip(), (extra, dist)
            assert f"`{dist}`" in decisions, dist


def test_req309_version_floors() -> None:
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
