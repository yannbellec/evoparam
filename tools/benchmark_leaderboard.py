# SPDX-License-Identifier: Apache-2.0
"""Write the benchmark leaderboard page and report from the result JSONs (spec §8.4, REQ-357).

The page (``docs/benchmark/leaderboard.html``) and the report
(``docs/benchmark/report-core.md``) are generated from the signed result files of
``docs/benchmark/results/`` only: every number comes from a result file, a result whose
signature does not match its content is refused, and nothing is edited by hand. A new entry
is a pull request adding its result JSON (and the code or container that produced it);
this script is then run again.

Usage::

    python tools/benchmark_leaderboard.py            # write the page and the report
    python tools/benchmark_leaderboard.py --check    # exit 1 when they are not up to date

Exit code 0 on success; 1 with ``--check`` when a file differs from what the results give.
"""

import argparse
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evoparam.benchmark import write_leaderboard, write_report  # noqa: E402

DOCS = ROOT / "docs" / "benchmark"


def build(results, page, report):
    """Write the page and the report; returns their paths."""

    return write_leaderboard(str(results), str(page)), write_report(str(results), str(report))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(DOCS / "results"))
    parser.add_argument("--page", default=str(DOCS / "leaderboard.html"))
    parser.add_argument("--report", default=str(DOCS / "report-core.md"))
    parser.add_argument("--check", action="store_true",
                        help="compare with the files in place instead of writing them")
    args = parser.parse_args(argv)
    if not args.check:
        for path in build(args.results, args.page, args.report):
            print(f"wrote {os.path.relpath(path, ROOT)}")
        return 0
    stale = []
    with tempfile.TemporaryDirectory() as tmp:
        page, report = build(args.results, os.path.join(tmp, "page.html"),
                             os.path.join(tmp, "report.md"))
        for new, old in ((page, args.page), (report, args.report)):
            with open(new, "rb") as a:
                fresh = a.read()
            current = pathlib.Path(old).read_bytes() if os.path.isfile(old) else None
            if fresh != current:
                stale.append(old)
    for path in stale:
        print(f"not up to date: {os.path.relpath(path, ROOT)}")
    print("leaderboard: " + ("ok" if not stale else f"{len(stale)} file(s) to regenerate"))
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
