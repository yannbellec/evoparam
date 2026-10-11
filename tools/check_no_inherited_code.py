# SPDX-License-Identifier: Apache-2.0
"""Fail when a source file reuses code inherited under Apache-2.0 (spec §10.1, REQ-312).

Code proposed upstream (MNE, MNE-BIDS) under BSD-3-Clause must be written new, never moved
or adapted from the inherited files (the origin named in NOTICE, with its repository and
commit; NOTICE is the single place that names it). This script compares
every window of 8 consecutive lines of the checked sources with every window of the
inherited Python files, after normalising whitespace (each line stripped, runs of spaces
and tabs reduced to one space, blank lines dropped), and exits with status 1 when any
window matches.

Usage::

    python tools/check_no_inherited_code.py SOURCES... [--inherited DIR] [--window 8]

``SOURCES`` are files or folders (``*.py``). ``--inherited`` is a checkout of the inherited
code; without it, the repository of the NOTICE origin is cloned at the NOTICE commit into
``~/.cache/evoparam/inherited`` (``git`` needed). The ``evoparam-upstream`` repository runs
this script in its CI.
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys

NOTICE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "NOTICE")
WINDOW = 8


def origin(notice=NOTICE):
    """(repository URL, commit) of the first origin listed in NOTICE."""

    with open(notice, encoding="utf-8") as fobj:
        text = fobj.read().split("Origin", 1)[1]
    url = re.search(r"https://\S+", text).group(0)
    commit = re.search(r"commit ([0-9a-f]{40})", text).group(1)
    return url + ".git", commit


def normalise(text):
    """The non-blank lines of a source, whitespace normalised, with their line numbers."""

    out = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = re.sub(r"[ \t]+", " ", line.strip())
        if line:
            out.append((number, line))
    return out


def windows(lines, size=WINDOW):
    """{digest of the window: first line number} of every window of ``size`` lines."""

    out = {}
    for i in range(len(lines) - size + 1):
        chunk = "\n".join(line for _, line in lines[i:i + size])
        out.setdefault(hashlib.sha256(chunk.encode("utf-8")).hexdigest(), lines[i][0])
    return out


def python_files(paths):
    for path in paths:
        if os.path.isdir(path):
            for folder, dirs, names in os.walk(path):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for name in sorted(names):
                    if name.endswith(".py"):
                        yield os.path.join(folder, name)
        elif path.endswith(".py"):
            yield path


def inherited_index(folder, size=WINDOW):
    """{digest: (file, line)} of every window of the inherited Python files."""

    index = {}
    for path in python_files([folder]):
        with open(path, encoding="utf-8", errors="replace") as fobj:
            for digest, line in windows(normalise(fobj.read()), size).items():
                index.setdefault(digest, (os.path.relpath(path, folder), line))
    return index


def fetch_inherited(target):
    """Clone the NOTICE origin at the NOTICE commit (once)."""

    upstream, commit = origin()
    if not os.path.isdir(os.path.join(target, ".git")):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", upstream, target], check=True)
    subprocess.run(["git", "-C", target, "checkout", "--quiet", commit], check=True)
    return target


def check(sources, inherited, size=WINDOW):
    """List of (source file, line, inherited file, line) of the matching windows."""

    index = inherited_index(inherited, size)
    found = []
    for path in python_files(sources):
        with open(path, encoding="utf-8", errors="replace") as fobj:
            for digest, line in windows(normalise(fobj.read()), size).items():
                if digest in index:
                    found.append((path, line) + index[digest])
    return sorted(found)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--inherited", help="checkout of the inherited code")
    parser.add_argument("--window", type=int, default=WINDOW)
    args = parser.parse_args(argv)
    inherited = args.inherited or fetch_inherited(
        os.path.expanduser(os.path.join("~", ".cache", "evoparam", "inherited", "upstream")))
    matches = check(args.sources, inherited, args.window)
    for path, line, origin, origin_line in matches:
        print(f"{path}:{line}: {args.window} lines identical to {origin}:{origin_line} "
              f"(inherited, Apache-2.0)")
    if matches:
        print(f"{len(matches)} window(s) reuse inherited code; write that code new.")
        return 1
    print("no inherited code found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
