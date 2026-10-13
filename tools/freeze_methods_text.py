# SPDX-License-Identifier: Apache-2.0
"""Freeze the methods text of the reference pipelines (REQ-340).

Runs every pipeline of ``evoparam.reports.REFERENCE_PIPELINES`` on its stand-in input and
writes its methods paragraph to ``evoparam/tests/reports/methods/<name>.txt``, with the
package version replaced by ``{version}`` so that a version bump alone does not change
the frozen files. Run it after a deliberate change of a template or of a preset, review
the diff, and commit the files with the change (the CHANGELOG names the change).

    python tools/freeze_methods_text.py            # every pipeline
    python tools/freeze_methods_text.py sim_tep    # one pipeline
    python tools/freeze_methods_text.py --check    # exit 1 when a file differs
"""

import argparse
import os
import sys

from evoparam import __version__, reports

FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "evoparam",
    "tests",
    "reports",
    "methods",
)


def frozen_text(name):
    """Return the methods paragraph of reference pipeline ``name``.

    The version is replaced by ``{version}``, and the text ends with a newline.
    """
    text = reports.methods_text(reports.run_reference_pipeline(name))
    return text.replace(__version__, "{version}") + "\n"


def main(argv=None):
    """Write the frozen methods texts, or with ``--check`` compare them; return the status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="pipelines (default: all)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare with the frozen files instead of writing them",
    )
    args = parser.parse_args(argv)
    names = args.names or list(reports.REFERENCE_PIPELINES)
    status = 0
    os.makedirs(FOLDER, exist_ok=True)
    for name in names:
        path = os.path.join(FOLDER, name + ".txt")
        text = frozen_text(name)
        if args.check:
            with open(path, encoding="utf-8") as fobj:
                same = fobj.read() == text
            print(f"{name}: {'same' if same else 'DIFFERS'}")
            status |= not same
        else:
            with open(path, "w", encoding="utf-8", newline="\n") as fobj:
                fobj.write(text)
            print(f"{name}: written to {path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
