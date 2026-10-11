# SPDX-License-Identifier: Apache-2.0
"""REQ-012: no inherited spectral vocabulary.

Public names, docstrings and documentation contain none of the prohibited words of the
internal style guide (the *Vocabulary* section of CONTRIBUTING.md); this test greps the
package and the documentation. The ancestors are acknowledged only in NOTICE; the
specification (docs/spec) and the change log quote them as history, and the tests name the
removed objects to check that they are gone.
"""

import importlib
import pathlib
import pkgutil
import re

import pytest

import evoparam

pytestmark = pytest.mark.core

ROOT = pathlib.Path(evoparam.__file__).resolve().parent.parent
GUIDE = ROOT / "CONTRIBUTING.md"
CASE_SENSITIVE = {"Bands"}
TEXT_SUFFIXES = {".py", ".rst", ".md", ".txt", ".json", ".toml", ".cff", ".yml", ".yaml",
                 ".cfg", ".in", ".html", ".tsv"}


def prohibited_words():
    """The first column of the table between the markers of CONTRIBUTING.md."""

    text = GUIDE.read_text(encoding="utf-8")
    block = text.split("<!-- prohibited-words -->")[1].split("<!-- end-prohibited-words -->")[0]
    words = []
    for line in block.splitlines()[2:]:
        if line.startswith("|"):
            words += re.findall(r"`([^`]+)`", line.split("|")[1])
    return words


WORDS = prohibited_words()


def pattern(word):
    flags = 0 if word in CASE_SENSITIVE else re.IGNORECASE
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(word) + r"(?![A-Za-z0-9_])", flags)


def scanned_files():
    files = [ROOT / "README.rst", ROOT / "CITATION.cff", ROOT / "pyproject.toml"]
    for folder in ("evoparam", "docs", "examples", "tools", "benchmarks", "validation"):
        for path in sorted((ROOT / folder).rglob("*")):
            rel = path.relative_to(ROOT).as_posix()
            if (path.is_file() and path.suffix in TEXT_SUFFIXES
                    and not rel.startswith(("evoparam/tests/", "docs/spec/"))
                    and "__pycache__" not in rel and ".egg-info" not in rel):
                files.append(path)
    return [path for path in files if path.exists()]


def test_the_style_guide_lists_the_words():
    assert len(WORDS) >= 15
    for word in ("ERPparam", "FOOOF", "specparam", "aperiodic", "peak_width_limits",
                 "max_n_peaks", "power spectrum", "knee"):
        assert word in WORDS


def test_req012_package_and_documentation():
    hits = []
    patterns = [(word, pattern(word)) for word in WORDS]
    for path in scanned_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            hits += [f"{path.relative_to(ROOT)}:{number}: {word!r}"
                     for word, regex in patterns if regex.search(line)]
    assert not hits, "\n".join(hits)


def test_req012_contributing_outside_the_list():
    text = GUIDE.read_text(encoding="utf-8")
    outside = text.split("<!-- prohibited-words -->")[0] + \
        text.split("<!-- end-prohibited-words -->")[1]
    hits = [word for word in WORDS if pattern(word).search(outside)]
    assert not hits


def _public_modules():
    names = ["evoparam"]
    for info in pkgutil.walk_packages(evoparam.__path__, "evoparam."):
        parts = info.name.split(".")
        if "tests" in parts or any(p.startswith("_") for p in parts[1:]):
            continue
        names.append(info.name)
    return names


def test_req012_public_names():
    hits = []
    patterns = [(word, pattern(word)) for word in WORDS if " " not in word]
    for name in _public_modules():
        try:
            module = importlib.import_module(name)
        except ImportError:
            # a module of an optional extra: its source is scanned by the test above
            continue
        public = getattr(module, "__all__", [n for n in dir(module) if not n.startswith("_")])
        for attr in public:
            hits += [f"{name}.{attr}" for word, regex in patterns if regex.search(attr)]
            obj = getattr(module, attr, None)
            for param in _parameters(obj):
                hits += [f"{name}.{attr}({param}=)" for word, regex in patterns
                         if regex.search(param)]
    assert not hits, hits


def _parameters(obj):
    import inspect

    if not callable(obj):
        return []
    try:
        return list(inspect.signature(obj).parameters)
    except (TypeError, ValueError):
        return []
