# SPDX-License-Identifier: Apache-2.0
r"""Run the performance workloads P1–P6 (spec §9.1, REQ-290) against a baseline (REQ-299).

A thin wrapper around :func:`evoparam.benchmark.timing.run_suite` (also reachable as
``evoparam benchmark run --suite perf``). The results go to
``benchmarks/results/<platform>-<python>-<commit>.json``.

Usage::

    OMP_NUM_THREADS=1 python benchmarks/run_benchmarks.py --suite perf
    python benchmarks/run_benchmarks.py --suite ci --compare benchmarks/baseline-ci.json \
        --time-tolerance 1.25 --memory-tolerance 1.10 --warn-above 1.10
    python benchmarks/run_benchmarks.py --suite ci --repeat-runs 3 \
        --write-baseline benchmarks/baseline-ci.json

Exit code 1 when a comparison fails (a median time above ``--time-tolerance`` x its
baseline, a peak memory above ``--memory-tolerance`` x, or a P3 figure above its bound);
the warnings between ``--warn-above`` and ``--time-tolerance`` are GitHub annotations.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from evoparam.benchmark import timing  # noqa: E402  (after the sys.path insert above)


def main(argv=None):
    """Run the benchmark suite from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suite", default="perf", choices=sorted(timing.SUITES))
    parser.add_argument("--only", help="comma-separated workload ids (default: P1 to P6)")
    parser.add_argument(
        "--out", default=os.path.join(HERE, "results"), help="folder of the results file"
    )
    parser.add_argument("--compare", help="baseline results file")
    parser.add_argument("--time-tolerance", type=float, default=1.25)
    parser.add_argument("--memory-tolerance", type=float, default=1.10)
    parser.add_argument("--warn-above", type=float, default=1.10)
    parser.add_argument("--repeat-runs", type=int, default=1)
    parser.add_argument("--write-baseline", help="write the run as a baseline file here")
    args = parser.parse_args(argv)
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    document = timing.run_suite(args.suite, only=only, repeat_runs=args.repeat_runs, log=print)
    path = timing.write_results(document, timing.results_path(document, args.out))
    print(f"results: {path}")
    if args.write_baseline:
        timing.write_results(document, args.write_baseline)
        print(f"baseline: {args.write_baseline}")
    if args.compare:
        with open(args.compare, encoding="utf-8") as fobj:
            baseline = json.load(fobj)
        failures, notes = timing.compare(
            document,
            baseline,
            time_tolerance=args.time_tolerance,
            memory_tolerance=args.memory_tolerance,
            warn_above=args.warn_above,
        )
        for note in notes:
            print(f"::warning::{note}")
        for failure in failures:
            print(f"::error::{failure}")
        if failures:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
