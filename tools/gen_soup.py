# SPDX-License-Identifier: Apache-2.0
"""Generate the SOUP inventory ``soup.json`` and ``THIRD_PARTY.md`` (spec §13.7, REQ-396,
REQ-399).

SOUP (software of unknown provenance, IEC 62304) is every third-party package evoparam runs
with. For each package declared in ``pyproject.toml`` (the hard dependencies, every optional
extra and, marked as tooling, the ``docs`` and ``tests`` chains) the inventory records:

- ``package`` and ``import_name``;
- ``role``: ``core``, ``optional`` (with the ``extras`` that declare it), ``transitive``
  (imported by the source, installed by another package: ``installed_by``) or ``tooling``;
- ``version_range``: the specifiers of ``pyproject.toml`` (union over the extras);
- ``locked_version``: the version pinned in ``ci/locks/py3.12.lock`` (REQ-309);
- ``licence`` (SPDX) and ``licence_checked``: whether the licence was read from the
  package's own metadata when the table below was written (S20, 2026-10-11) or comes from
  the project's repository only;
- ``anomaly_url``: where the package's known anomalies (bugs) are tracked;
- ``imported_by_source`` and ``functions_used``: whether evoparam's source imports the
  package, and the names of the package it uses, found by parsing every module of the
  package (``import``, ``from ... import``, attribute chains on the imported name, and
  names bound by ``evoparam.utils.require`` or by a helper that returns it).

The file depends only on the repository, so that ``--check`` gives the same answer on every
machine; ``evoparam/tests/test_soup_inventory.py`` compares the licence column with the
metadata of the installed packages.

Usage::

    python tools/gen_soup.py [--check]
"""

import argparse
import ast
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOUP = ROOT / "soup.json"
THIRD_PARTY = ROOT / "THIRD_PARTY.md"
LOCK = ROOT / "ci" / "locks" / "py3.12.lock"
TOOLING = ("docs", "tests")
UNIONS = ("all",)

# package -> (import name, SPDX licence, licence read from the installed metadata, tracker)
PACKAGES = {
    "numpy": ("numpy", "BSD-3-Clause", True, "https://github.com/numpy/numpy/issues"),
    "scipy": ("scipy", "BSD-3-Clause", True, "https://github.com/scipy/scipy/issues"),
    "mne": ("mne", "BSD-3-Clause", True, "https://github.com/mne-tools/mne-python/issues"),
    "mne-bids": ("mne_bids", "BSD-3-Clause", True, "https://github.com/mne-tools/mne-bids/issues"),
    "pybids": ("bids", "MIT", True, "https://github.com/bids-standard/pybids/issues"),
    "pynwb": ("pynwb", "BSD-3-Clause", True,
              "https://github.com/NeurodataWithoutBorders/pynwb/issues"),
    "hdmf": ("hdmf", "BSD-3-Clause", True, "https://github.com/hdmf-dev/hdmf/issues"),
    "ndx-events": ("ndx_events", "BSD-3-Clause", True, "https://github.com/rly/ndx-events/issues"),
    "h5py": ("h5py", "BSD-3-Clause", True, "https://github.com/h5py/h5py/issues"),
    "remfile": ("remfile", "Apache-2.0", False, "https://github.com/magland/remfile/issues"),
    "neo": ("neo", "BSD-3-Clause", True, "https://github.com/NeuralEnsemble/python-neo/issues"),
    "quantities": ("quantities", "BSD-3-Clause", False,
                   "https://github.com/python-quantities/python-quantities/issues"),
    "edfio": ("edfio", "Apache-2.0", True, "https://github.com/the-siesta-group/edfio/issues"),
    "numpyro": ("numpyro", "Apache-2.0", True, "https://github.com/pyro-ppl/numpyro/issues"),
    "jax": ("jax", "Apache-2.0", True, "https://github.com/jax-ml/jax/issues"),
    "arviz": ("arviz", "Apache-2.0", True, "https://github.com/arviz-devs/arviz/issues"),
    "pymc": ("pymc", "Apache-2.0", True, "https://github.com/pymc-devs/pymc/issues"),
    "numba": ("numba", "BSD-2-Clause", False, "https://github.com/numba/numba/issues"),
    "matplotlib": ("matplotlib", "PSF-2.0", True,
                   "https://github.com/matplotlib/matplotlib/issues"),
    "nilearn": ("nilearn", "BSD-3-Clause", True, "https://github.com/nilearn/nilearn/issues"),
    "pandas": ("pandas", "BSD-3-Clause", True, "https://github.com/pandas-dev/pandas/issues"),
    "pyarrow": ("pyarrow", "Apache-2.0", True, "https://github.com/apache/arrow/issues"),
    "xarray": ("xarray", "Apache-2.0", True, "https://github.com/pydata/xarray/issues"),
    "statsmodels": ("statsmodels", "BSD-3-Clause", True,
                    "https://github.com/statsmodels/statsmodels/issues"),
    "formulaic": ("formulaic", "MIT", True, "https://github.com/matthewwardrop/formulaic/issues"),
    "bidsschematools": ("bidsschematools", "MIT", True,
                        "https://github.com/bids-standard/bids-specification/issues"),
    "nwbinspector": ("nwbinspector", "BSD-3-Clause", True,
                     "https://github.com/NeurodataWithoutBorders/nwbinspector/issues"),
    "jsonschema": ("jsonschema", "MIT", True,
                   "https://github.com/python-jsonschema/jsonschema/issues"),
    "tqdm": ("tqdm", "MPL-2.0 AND MIT", True, "https://github.com/tqdm/tqdm/issues"),
    "tomli": ("tomli", "MIT", False, "https://github.com/hukkin/tomli/issues"),
    "joblib": ("joblib", "BSD-3-Clause", True, "https://github.com/joblib/joblib/issues"),
    "pylsl": ("pylsl", "MIT", True, "https://github.com/labstreaminglayer/pylsl/issues"),
    "scikit-learn": ("sklearn", "BSD-3-Clause", True,
                     "https://github.com/scikit-learn/scikit-learn/issues"),
    "sphinx": ("sphinx", "BSD-2-Clause", False, "https://github.com/sphinx-doc/sphinx/issues"),
    "pydata-sphinx-theme": ("pydata_sphinx_theme", "BSD-3-Clause", False,
                            "https://github.com/pydata/pydata-sphinx-theme/issues"),
    "sphinx-gallery": ("sphinx_gallery", "BSD-3-Clause", False,
                       "https://github.com/sphinx-gallery/sphinx-gallery/issues"),
    "numpydoc": ("numpydoc", "BSD-3-Clause", False, "https://github.com/numpy/numpydoc/issues"),
    "sphinx-copybutton": ("sphinx_copybutton", "MIT", False,
                          "https://github.com/executablebooks/sphinx-copybutton/issues"),
    "sphinxcontrib-bibtex": ("sphinxcontrib.bibtex", "BSD-2-Clause", False,
                             "https://github.com/mcmtroffaes/sphinxcontrib-bibtex/issues"),
    "myst-parser": ("myst_parser", "MIT", False,
                    "https://github.com/executablebooks/MyST-Parser/issues"),
    "pytest": ("pytest", "MIT", True, "https://github.com/pytest-dev/pytest/issues"),
    "pytest-cov": ("pytest_cov", "MIT", True, "https://github.com/pytest-dev/pytest-cov/issues"),
    "pytest-timeout": ("pytest_timeout", "MIT", True,
                       "https://github.com/pytest-dev/pytest-timeout/issues"),
    "pytest-xdist": ("xdist", "MIT", True, "https://github.com/pytest-dev/pytest-xdist/issues"),
    "hypothesis": ("hypothesis", "MPL-2.0", True,
                   "https://github.com/HypothesisWorks/hypothesis/issues"),
    "mypy": ("mypy", "MIT", True, "https://github.com/python/mypy/issues"),
    "ruff": ("ruff", "MIT", True, "https://github.com/astral-sh/ruff/issues"),
    "interrogate": ("interrogate", "MIT", True, "https://github.com/econchick/interrogate/issues"),
    "mpmath": ("mpmath", "BSD-3-Clause", True, "https://github.com/mpmath/mpmath/issues"),
    "pyyaml": ("yaml", "MIT", True, "https://github.com/yaml/pyyaml/issues"),
    # imported by evoparam's source but installed by another package (TRANSITIVE)
    "packaging": ("packaging", "Apache-2.0 OR BSD-2-Clause", True,
                  "https://github.com/pypa/packaging/issues"),
    "tabulate": ("tabulate", "MIT", True, "https://github.com/astanin/python-tabulate/issues"),
    "patsy": ("patsy", "BSD-2-Clause", True, "https://github.com/pydata/patsy/issues"),
    "pytensor": ("pytensor", "BSD-3-Clause", True, "https://github.com/pymc-devs/pytensor/issues"),
}

# packages the source imports that pyproject.toml does not declare: each comes with another
# package, and the import is guarded or lives in that package's extra
TRANSITIVE = {
    "packaging": "dependency of pip and most extras; evaluates the markers of the lock file "
                 "(Provenance.environment['lock_sha256']); without it no lock is claimed",
    "tabulate": "optional dependency of pandas (DataFrame.to_markdown) in the Markdown "
                "reports; plain text is written without it",
    "patsy": "dependency of statsmodels (extra stats): design matrices of the mixed-model export",
    "pytensor": "dependency of pymc (extra bayes): tensors of the PyMC engine",
}


def _pyproject():
    try:
        import tomllib
    except ImportError:                       # Python 3.10
        import tomli as tomllib
    with open(ROOT / "pyproject.toml", "rb") as fobj:
        return tomllib.load(fobj)["project"]


def _name(requirement):
    name = re.match(r"[A-Za-z0-9_.\-]+", requirement).group(0).lower()
    return name


def _specifier(requirement):
    body = requirement.split(";")[0]
    body = re.sub(r"^[A-Za-z0-9_.\-]+(\[[^\]]*\])?", "", body).strip()
    marker = requirement.split(";", 1)[1].strip() if ";" in requirement else ""
    return body, marker


def declared():
    """{package: {"extras": [...], "ranges": {...}, "role": ...}} from pyproject.toml."""

    project = _pyproject()
    out = {}

    def note(requirement, extra):
        name = _name(requirement)
        row = out.setdefault(name, {"extras": [], "ranges": set(), "markers": set()})
        spec, marker = _specifier(requirement)
        if spec:
            row["ranges"].add(spec)
        if marker:
            row["markers"].add(marker)
        if extra and extra not in row["extras"]:
            row["extras"].append(extra)

    for requirement in project["dependencies"]:
        note(requirement, None)
        out[_name(requirement)]["core"] = True
    for extra, requirements in project["optional-dependencies"].items():
        if extra in UNIONS:
            continue
        for requirement in requirements:
            note(requirement, extra)
    for name, via in TRANSITIVE.items():
        out.setdefault(name, {"extras": [], "ranges": set(), "markers": set()})["via"] = via
    for name, row in out.items():
        if row.get("via"):
            row["role"] = "transitive"
        elif row.get("core"):
            row["role"] = "core"
        elif any(e not in TOOLING for e in row["extras"]):
            row["role"] = "optional"
        else:
            row["role"] = "tooling"
    return out


def locked_versions(lock=LOCK):
    text = lock.read_text(encoding="utf-8") if lock.exists() else ""
    return {m.group(1).lower(): m.group(2)
            for m in re.finditer(r"^([A-Za-z0-9_.\-]+)==([^\s;\\]+)", text, flags=re.MULTILINE)}


class _Uses(ast.NodeVisitor):
    """Names used from each imported top-level package in one module."""

    def __init__(self, tree=None):
        self.aliases = {}                          # local name -> dotted package path
        self.used = {}                             # top-level package -> set of names
        # functions of the module that return require("<package>", ...)
        self.loaders = {}
        for node in ast.walk(tree) if tree is not None else ():
            if isinstance(node, ast.FunctionDef):
                for sub in ast.walk(node):
                    package = self._required(getattr(sub, "value", None)) \
                        if isinstance(sub, ast.Return) else None
                    if package:
                        self.loaders[node.name] = package

    @staticmethod
    def _required(call):
        """The package of a ``require("<package>", ...)`` call, else None."""

        if (isinstance(call, ast.Call) and getattr(call.func, "id", getattr(call.func, "attr",
                                                                          None)) == "require"
                and call.args and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)):
            return call.args[0].value
        return None

    def _add(self, dotted):
        top, _, rest = dotted.partition(".")
        self.used.setdefault(top, set())
        if rest:
            self.used[top].add(rest)

    def visit_Import(self, node):
        for alias in node.names:
            if alias.asname:                       # import a.b as c: c is a.b
                self.aliases[alias.asname] = alias.name
            else:                                  # import a.b: a is a
                self.aliases[alias.name.split(".")[0]] = alias.name.split(".")[0]
            self._add(alias.name)

    def visit_ImportFrom(self, node):
        if node.level or not node.module:
            return
        for alias in node.names:
            dotted = f"{node.module}.{alias.name}"
            self.aliases[alias.asname or alias.name] = dotted
            self._add(dotted)

    def visit_Assign(self, node):
        call = node.value
        package = self._required(call)
        if package is None and isinstance(call, ast.Call):
            package = self.loaders.get(getattr(call.func, "id", None))
        if package:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.aliases[target.id] = package
            self._add(package)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        chain = []
        current = node
        while isinstance(current, ast.Attribute):
            chain.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name) and current.id in self.aliases:
            chain.reverse()
            self._add(".".join([self.aliases[current.id]] + chain[:2]))
            return
        self.generic_visit(node)


def functions_used(import_names):
    """{import name: (sorted names used, imported at all)} over the package's modules (tests
    excluded)."""

    uses = {name: set() for name in import_names}
    imported = set()
    tops = {name.split(".")[0]: name for name in import_names}
    for path in sorted((ROOT / "evoparam").rglob("*.py")):
        if "tests" in path.relative_to(ROOT).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        visitor = _Uses(tree)
        visitor.visit(tree)
        for top, names in visitor.used.items():
            if top in tops:
                key = tops[top]
                imported.add(key)
                prefix = key.split(".", 1)[1] + "." if "." in key else ""
                uses[key] |= {n[len(prefix):] if prefix and n.startswith(prefix) else n
                              for n in names if n}
    return {name: (sorted(values), name in imported) for name, values in uses.items()}


def inventory():
    rows = declared()
    unknown = sorted(set(rows) - set(PACKAGES))
    if unknown:
        raise SystemExit(f"tools/gen_soup.py: add {unknown} to PACKAGES (licence, tracker)")
    locks = locked_versions()
    used = functions_used([PACKAGES[name][0] for name in rows])
    items = []
    for name in sorted(rows):
        import_name, licence, checked, tracker = PACKAGES[name]
        row = rows[name]
        items.append({
            "package": name, "import_name": import_name, "role": row["role"],
            "extras": sorted(row["extras"]),
            "version_range": ", ".join(sorted(row["ranges"])) or "any",
            "markers": sorted(row["markers"]),
            "locked_version": locks.get(name),
            "licence": licence,
            "licence_checked": "package metadata" if checked else "project repository",
            "anomaly_url": tracker,
            "imported_by_source": used[import_name][1] if row["role"] != "tooling" else False,
            "functions_used": used[import_name][0] if row["role"] != "tooling" else [],
            **({"installed_by": row["via"]} if row.get("via") else {}),
        })
    return {"schema": "evoparam-soup/1",
            "description": "SOUP inventory of evoparam (spec REQ-399), generated by "
                           "tools/gen_soup.py from pyproject.toml, ci/locks/py3.12.lock and "
                           "the package source.",
            "packages": items}


def _used(row):
    if row["role"] == "tooling":
        return "tooling"
    if not row["imported_by_source"]:
        return "not imported by the package source"
    return str(len(row["functions_used"]))


def third_party_md(soup):
    lines = ["# Third-party packages", "",
             "Generated by `tools/gen_soup.py` with `soup.json` (spec REQ-396, REQ-399); do not "
             "edit by hand. Every package evoparam declares, with its version range "
             "(`pyproject.toml`), the version of the lock file `ci/locks/py3.12.lock`, its "
             "licence, and where its known anomalies are tracked. No package is copied into "
             "this repository; they are installed by `pip`. The code evoparam inherited is "
             "named in `NOTICE`; the algorithms written from papers are listed in "
             "`REIMPLEMENTED.md`.", ""]
    for role, title in (("core", "Hard dependencies"), ("optional", "Optional extras"),
                        ("transitive", "Imported, installed by another package"),
                        ("tooling", "Documentation and test tooling (not installed with "
                                    "evoparam)")):
        lines += [f"## {title}", "",
                  "| package | extras | version range | locked | licence | anomalies | "
                  "names used |", "|---|---|---|---|---|---|---|"]
        for row in soup["packages"]:
            if row["role"] != role:
                continue
            extras = ", ".join(row["extras"]) or "-"
            lines.append(f"| {row['package']} | {extras} | {row['version_range']} | "
                         f"{row['locked_version'] or 'not locked'} | {row['licence']} | "
                         f"<{row['anomaly_url']}> | {_used(row)} |")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def render():
    soup = inventory()
    return json.dumps(soup, indent=1, ensure_ascii=False) + "\n", third_party_md(soup)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="fail when soup.json or THIRD_PARTY.md is out of date")
    args = parser.parse_args(argv)
    soup, md = render()
    if args.check:
        stale = [p.name for p, text in ((SOUP, soup), (THIRD_PARTY, md))
                 if not p.exists() or p.read_text(encoding="utf-8") != text]
        if stale:
            print(f"out of date: {', '.join(stale)}; run tools/gen_soup.py")
            return 1
        return 0
    SOUP.write_text(soup, encoding="utf-8")
    THIRD_PARTY.write_text(md, encoding="utf-8")
    print(f"wrote soup.json ({len(json.loads(soup)['packages'])} packages) and THIRD_PARTY.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
