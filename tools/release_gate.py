# SPDX-License-Identifier: Apache-2.0
"""Refuse to release before the HIIE decision is on record (spec §13.7, REQ-398).

The release workflow (``.github/workflows/release.yml``) runs this first. It exits with
status 1, naming what is missing, unless

- ``GOVERNANCE.md`` (block ``<!-- hiie-decision -->``) holds the invention disclosure and
  the HIIE decision reference, each with a date (``YYYY-MM-DD``), and
- the gate of ``RELEASING.md`` says ``HIIE decision: on record, <date>, <reference>``.

Usage::

    python tools/release_gate.py [--root PATH]
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def _block(text, start, end):
    if start not in text or end not in text:
        return None
    return text.split(start, 1)[1].split(end, 1)[0]


def check(root=ROOT):
    """List of problems (empty when the release may proceed)."""

    root = pathlib.Path(root)
    problems = []
    governance = (root / "GOVERNANCE.md").read_text(encoding="utf-8")
    block = _block(governance, "<!-- hiie-decision -->", "<!-- end-hiie-decision -->")
    if block is None:
        problems.append("GOVERNANCE.md has no hiie-decision block")
    else:
        items = dict(re.findall(r"^- ([^:]+): (.+)$", block, re.MULTILINE))
        for key in ("Invention disclosure", "HIIE decision reference"):
            value = items.get(key, "")
            if not value or value.strip() in ("none", "not filed") or not DATE.search(value):
                problems.append(f"GOVERNANCE.md: '{key}' has no reference and date "
                                f"({value or 'missing'})")
    releasing = (root / "RELEASING.md").read_text(encoding="utf-8")
    gate = _block(releasing, "<!-- release-gate -->", "<!-- end-release-gate -->") or ""
    match = re.search(r"^- HIIE decision: (.+)$", gate, re.MULTILINE)
    if not match or not re.match(r"on record, \d{4}-\d{2}-\d{2}, \S+", match.group(1)):
        problems.append("RELEASING.md: the gate does not say 'HIIE decision: on record, "
                        f"<date>, <reference>' ({match.group(1) if match else 'missing'})")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)
    problems = check(args.root)
    for problem in problems:
        print(f"release refused: {problem}")
    if not problems:
        print("HIIE decision on record: the release may proceed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
