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
DISTRIBUTIONS = {"sklearn": "scikit-learn", "mpmath": "mpmath", "pandas": "pandas",
                 "mne": "mne", "pynwb": "pynwb", "matplotlib": "matplotlib",
                 "quantities": "quantities", "nwbinspector": "nwbinspector", "neo": "neo",
                 "jax": "jax", "numpyro": "numpyro", "statsmodels": "statsmodels",
                 "pymc": "pymc", "arviz": "arviz", "pyarrow": "pyarrow", "xarray": "xarray",
                 "mne_bids": "mne-bids", "bids": "pybids", "edfio": "edfio",
                 "numba": "numba", "tqdm": "tqdm", "joblib": "joblib", "pylsl": "pylsl",
                 "jsonschema": "jsonschema", "bidsschematools": "bidsschematools",
                 "h5py": "h5py", "hdmf": "hdmf", "ndx_events": "ndx-events",
                 "remfile": "remfile", "formulaic": "formulaic", "hypothesis": "hypothesis",
                 "tomli": "tomli", "nilearn": "nilearn"}


def _toml():
    try:
        import tomllib
    except ImportError:                  # Python 3.10
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
        elif isinstance(node, ast.Call) and getattr(node.func, "id", getattr(
                node.func, "attr", "")) in ("require", "import_module", "importorskip") \
                and node.args and isinstance(node.args[0], ast.Constant) and \
                isinstance(node.args[0].value, str):
            yield node.args[0].value


def test_no_gpl_imports():
    hits = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for name in _imports(path):
            if name.split(".")[0].lower() in DENY:
                hits.append(f"{path.relative_to(ROOT)}: {name}")
    for extra, requirements in _extras().items():
        hits += [f"pyproject [{extra}]: {r}" for r in requirements if _name(r) in DENY]
    hits += [f"pyproject dependencies: {r}" for r in _toml()["project"]["dependencies"]
             if _name(r) in DENY]
    assert not hits, hits


def test_no_gpl_imports_finds_a_mutation(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def f():\n    import ieegprep\n    return require('erdetect', extra='x')\n")
    assert [n for n in _imports(bad) if n in DENY] == ["ieegprep", "erdetect"]


GPL_TEXT = re.compile(r"GNU (Lesser |Affero )?General Public License|SPDX-License-Identifier:"
                      r"\s*(?!Apache-2\.0)\S+|\b(L|A)?GPL-?[23]", re.IGNORECASE)


def test_licence_scan():
    assert "Apache License" in (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert _toml()["project"]["license"] == "Apache-2.0"
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "ERPparam" in notice and "Apache" in notice
    hits = []
    for path in sorted(PACKAGE.rglob("*")):
        rel = path.relative_to(ROOT).as_posix()
        if not path.is_file() or "__pycache__" in rel or path.suffix not in (
                ".py", ".json", ".txt", ".csv", ".tsv", ".md", ".rst", ".html", ".cff"):
            continue
        if rel == "evoparam/tests/test_dependencies.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits += [f"{rel}: {m.group(0)}" for m in GPL_TEXT.finditer(text)
                 if not _mentions_isolation(text, m.start())]
    assert not hits, hits


def _mentions_isolation(text, start):
    """A sentence that says a GPL tool is NOT used (REQ-309, the comparator scripts run as
    separate processes) is not a licence notice."""

    line = text[text.rfind("\n", 0, start) + 1:text.find("\n", start)]
    return bool(re.search(r"not (imported|vendored|a dependency)|separate process|"
                          r"subprocess|outside the|never import", line, re.IGNORECASE))


def test_all_is_the_union_of_the_extras():
    extras = _extras()
    union = set()
    for name, requirements in extras.items():
        if name not in NOT_IN_ALL:
            union |= {r.replace(" ", "") for r in requirements}
    assert {r.replace(" ", "") for r in extras["all"]} == union
    assert set(extras) >= {"mne", "bids", "nwb", "neo", "edf", "bayes", "gpu", "fast", "plot",
                           "data", "xarray", "stats", "validate", "cli", "progress",
                           "parallel", "stream", "docs", "tests", "all"}


def _skipped_imports():
    names = set()
    pattern = re.compile(r"importorskip\(\s*[\"']([\w.]+)[\"']|find_spec\(\s*[\"']([\w.]+)"
                         r"[\"']|available\(\s*[\"']([\w.]+)[\"']")
    for path in sorted((PACKAGE / "tests").rglob("*.py")):
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            name = next(g for g in match.groups() if g).split(".")[0]
            if name != "evoparam":                # the package's own optional modules
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
