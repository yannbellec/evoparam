"""Fail when a source file reuses code inherited under Apache-2.0 (spec §10.1, REQ-312).

Code proposed upstream (MNE, MNE-BIDS) under BSD-3-Clause must be written new, never moved
or adapted from the files inherited from ERPparam (NOTICE: ERPparam at commit
98e52e3bf538031987ce2986bd93141efc475002, itself derived from FOOOF). This script compares
every window of 8 consecutive lines of the checked sources with every window of the
inherited Python files, after normalising whitespace (each line stripped, runs of spaces
and tabs reduced to one space, blank lines dropped), and exits with status 1 when any
window matches.

Usage::

    python tools/check_no_inherited_code.py SOURCES... [--inherited DIR] [--window 8]

``SOURCES`` are files or folders (``*.py``). ``--inherited`` is a checkout of the inherited
code; without it, ERPparam is cloned at the NOTICE commit into
``~/.cache/evoparam/inherited`` (``git`` needed). The ``evoparam-upstream`` repository runs
this script in its CI.
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys

UPSTREAM = "https://github.com/voytekresearch/ERPparam.git"
COMMIT = "98e52e3bf538031987ce2986bd93141efc475002"
WINDOW = 8


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
    """Clone ERPparam at the NOTICE commit (once)."""

    if not os.path.isdir(os.path.join(target, ".git")):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", UPSTREAM, target], check=True)
    subprocess.run(["git", "-C", target, "checkout", "--quiet", COMMIT], check=True)
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
        os.path.expanduser(os.path.join("~", ".cache", "evoparam", "inherited", "ERPparam")))
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
