# SPDX-License-Identifier: Apache-2.0
"""Add the SPDX licence header to every Python source file (spec §13.7, REQ-396).

Every ``.py`` file of the repository (``adapters/`` excepted: its files carry their own
GPL-3.0-or-later header, spec REQ-395) starts with::

    # SPDX-License-Identifier: Apache-2.0

(after a ``#!`` line when there is one). A file that keeps code of the inherited project
(``INHERITED`` below, found with ``tools/check_no_inherited_code.py`` against the first
commit of this repository) carries a second line::

    # Modified from <upstream> (Apache-2.0); changes: <changes>

where ``<upstream>`` points to the project and commit named in ``NOTICE``, the single place
that names the inherited project (REQ-012; docs/spec/DECISIONS.md).
A file derived from third-party code under another licence (``DERIVED`` below, named with
its licence text in ``NOTICE``) names both licences on its first line, for example
``# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause``.

Usage::

    python tools/add_spdx_headers.py [--check]

``--check`` lists the files without the header (or without the modification notice) and
exits with status 1 when there is one; ``evoparam/tests/test_spdx_headers.py`` runs it.
"""

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPDX = "# SPDX-License-Identifier: Apache-2.0"
EXCLUDED = (
    "adapters",
    "build",
    "dist",
    ".git",
    "docs/_build",
    "docs/auto_tutorials",
    "docs/auto_examples",
    "docs/generated",
)
UPSTREAM = "the upstream project and commit named in NOTICE"
# file -> changes made to the inherited code (found by tools/check_no_inherited_code.py,
# S20; the shape equations come from the inherited shape functions)
INHERITED = {
    "evoparam/utils/_modutils.py": "moved from core/modutils.py into utils/ (spec §4.5); "
    "module docstring rewritten; functions unchanged",
    "evoparam/plots/_signals.py": "signal and summary figures of plts/signals.py and "
    "plts/templates.py rewritten: each returns the Figure, "
    "accepts ax=, never calls show(), uses the Okabe and Ito "
    "palette and draws numbers computed by evoparam.stats",
    "evoparam/shapes/components.py": "Gaussian and skewed-Gaussian equations of core/funcs.py "
    "rewritten with unit amplitude, vectorised over parameter "
    "sets, with analytic Jacobians; the other shapes are new",
}
# file → licence of the third-party code it derives from (named, with its licence text, in
# NOTICE); its SPDX header names both licences
DERIVED = {
    "evoparam/models/_trf.py": "BSD-3-Clause",  # SciPy's trust-region reflective solver
    "evoparam/models/_fast.py": "BSD-3-Clause",  # the same solver; Cephes' erfc (SciPy)
}


def spdx(rel):
    """Return the SPDX header line of the file ``rel`` (relative to the repository root)."""
    return f"{SPDX} AND {DERIVED[rel]}" if rel in DERIVED else SPDX


def notice(changes):
    """Return the modification notice, wrapped at 99 characters (continuation ``#   ``)."""
    import textwrap

    return "\n".join(
        textwrap.wrap(
            f"Modified from {UPSTREAM} (Apache-2.0); changes: {changes}",
            width=99,
            initial_indent="# ",
            subsequent_indent="#   ",
        )
    )


def source_files(root=ROOT):
    """Return the Python files of ``root`` that carry the header (EXCLUDED and caches left out)."""
    out = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if any(rel == e or rel.startswith(e + "/") for e in EXCLUDED):
            continue
        if "__pycache__" in rel or ".egg-info" in rel or "/." in "/" + rel:
            continue
        out.append(path)
    return out


def _header(lines):
    """Index of the first line after a ``#!`` line."""
    return 1 if lines and lines[0].startswith("#!") else 0


def problems(path, root=ROOT):
    """Return what the header of ``path`` lacks (empty when it is right)."""
    rel = path.relative_to(root).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()
    start = _header(lines)
    found = []
    if len(lines) <= start or lines[start] != spdx(rel):
        found.append("no SPDX header on its first line")
    if rel in INHERITED and (
        len(lines) <= start + 1 or not lines[start + 1].startswith("# Modified from ")
    ):
        found.append("inherited file without the modification notice")
    return found


def add(path, root=ROOT):
    """Add the missing header lines to ``path``; True when the file changed."""
    rel = path.relative_to(root).as_posix()
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    start = _header(lines)
    changed = False
    if len(lines) <= start or lines[start].rstrip("\n") != spdx(rel):
        lines.insert(start, spdx(rel) + "\n")
        changed = True
    if rel in INHERITED and not (
        len(lines) > start + 1 and lines[start + 1].startswith("# Modified from ")
    ):
        lines.insert(start + 1, notice(INHERITED[rel]) + "\n")
        changed = True
    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return changed


def main(argv=None):
    """Add the missing headers, or with ``--check`` list the files without one."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="list the files without the header and fail if there is one",
    )
    args = parser.parse_args(argv)
    files = source_files()
    if args.check:
        bad = [(p, problems(p)) for p in files]
        bad = [(p, why) for p, why in bad if why]
        for path, why in bad:
            print(f"{path.relative_to(ROOT).as_posix()}: {'; '.join(why)}")
        print(f"{len(files) - len(bad)} of {len(files)} Python files have the SPDX header")
        return 1 if bad else 0
    changed = sum(add(p) for p in files)
    print(f"header added to {changed} of {len(files)} Python files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
