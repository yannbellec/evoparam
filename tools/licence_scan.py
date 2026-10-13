# SPDX-License-Identifier: Apache-2.0
"""Licence scan of the resolved dependency tree of the extras (spec §13.7, REQ-395).

REQ-395: no module of evoparam imports, vendors or declares as an extra a package under
GPL, LGPL or AGPL. This script checks the installed environment: from the packages that
the chosen extras of ``pyproject.toml`` declare, it follows the requirements of every
installed distribution (the resolved tree) and fails when the licence of one of them
(``License-Expression``, a short ``License`` field, or a ``License ::`` classifier) is GPL,
LGPL or AGPL.

One kind of entry is not a failure, and it is checked rather than assumed: a distribution
whose own licence file puts GPL on named files only (``FILE_SCOPED``), when none of these
files is installed. docutils (a requirement of Sphinx, extra ``docs``) carries a GPL
classifier for ``tools/editors/emacs/rst.el``, which its wheel does not install; the rest of
docutils is public domain or BSD.

The CI job ``licence-scan`` (``.github/workflows/ci.yml``) installs every extra in its own
virtual environment and runs this script there, so that each extras set is scanned on its
own resolved tree. ``evoparam/tests/test_dependencies.py`` runs the same scan on the
environment of the tests.

Usage::

    python tools/licence_scan.py --extra bids        # one extra
    python tools/licence_scan.py --all-extras        # every extra of pyproject.toml

The exit code is 1 when a copyleft distribution is found, 0 otherwise.
"""

import argparse
import pathlib
import re
import sys
from importlib import metadata

ROOT = pathlib.Path(__file__).resolve().parent.parent

# a longer ``License`` field is the whole licence text, read through the classifiers instead
SHORT_LICENCE = 200

COPYLEFT = re.compile(r"\b(A|L)?GPL|GNU (Lesser |Affero )?General Public", re.IGNORECASE)

# distribution -> the files that its licence file puts under the GPL (and nothing else)
FILE_SCOPED = {
    "docutils": ("tools/editors/emacs/rst.el",),
}


def name(requirement: str) -> str:
    """Normalised distribution name of a requirement string."""
    return re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0].strip().lower().replace("_", "-")


def extras() -> dict[str, list[str]]:
    """Return the ``[project.optional-dependencies]`` table of ``pyproject.toml``."""
    try:
        import tomllib
    except ImportError:  # Python 3.10
        import tomli as tomllib
    with open(ROOT / "pyproject.toml", "rb") as fobj:
        table: dict[str, list[str]] = tomllib.load(fobj)["project"]["optional-dependencies"]
    return table


def licence(dist: str) -> str | None:
    """Licence text of an installed distribution (expression, short field, classifiers)."""
    try:
        meta = metadata.metadata(dist)
    except metadata.PackageNotFoundError:
        return None
    short = meta.get("License") if len(meta.get("License") or "") <= SHORT_LICENCE else None
    classifiers = [c for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    return " ".join(filter(None, [meta.get("License-Expression"), short, *classifiers]))


def resolved_tree(roots: set[str]) -> set[str]:
    """Return the installed distributions reachable from ``roots``.

    Their requirements are followed, not the requirements of their own extras.
    """
    seen: set[str] = set()
    todo = list(roots)
    while todo:
        dist = name(todo.pop())
        if dist in seen:
            continue
        try:
            requires = metadata.requires(dist) or []
        except metadata.PackageNotFoundError:
            continue
        seen.add(dist)
        todo += [r for r in requires if "extra ==" not in r]
    return seen


def file_scoped_ok(dist: str) -> bool:
    """Return True when ``dist`` puts GPL on named files only and none is installed."""
    scoped = FILE_SCOPED.get(dist)
    if not scoped:
        return False
    files = [str(f).replace("\\", "/") for f in metadata.distribution(dist).files or []]
    return not any(
        f.endswith(s) or f.endswith(s.rsplit("/", 1)[-1]) for f in files for s in scoped
    )


def scan(chosen: list[str]) -> tuple[list[str], list[str], int]:
    """(copyleft hits, accepted file-scoped entries, number of distributions scanned)."""
    table = extras()
    unknown = [e for e in chosen if e not in table]
    if unknown:
        raise SystemExit(f"unknown extra(s): {unknown}")
    roots = {name(r) for e in chosen for r in table[e]}
    hits, scoped = [], []
    tree = sorted(resolved_tree(roots))
    for dist in tree:
        text = licence(dist)
        if not text or not COPYLEFT.search(text):
            continue
        if file_scoped_ok(dist):
            scoped.append(f"{dist}: {text} (GPL on {', '.join(FILE_SCOPED[dist])}, not installed)")
        else:
            hits.append(f"{dist}: {text}")
    return hits, scoped, len(tree)


def main(argv: list[str] | None = None) -> int:
    """Scan the chosen extras; return 1 when a copyleft distribution is found."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--extra", action="append", help="extra to scan (repeatable)")
    group.add_argument("--all-extras", action="store_true", help="every extra of pyproject")
    args = parser.parse_args(argv)
    chosen = sorted(extras()) if args.all_extras else args.extra
    hits, scoped, n = scan(chosen)
    print(f"extras {', '.join(chosen)}: {n} installed distribution(s) in the resolved tree")
    for line in scoped:
        print(f"  accepted, file-scoped: {line}")
    for line in hits:
        print(f"  COPYLEFT: {line}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
