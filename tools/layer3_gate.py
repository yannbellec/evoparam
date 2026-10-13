# SPDX-License-Identifier: Apache-2.0
"""Release gate on the layer-3 metrics (spec §7.4, REQ-376).

A regression of any layer-3 metric larger than its pre-registered tolerance blocks the
release unless the CHANGELOG explains it. The tolerances are those of §7.4, kept in
``validation/layer3/tolerances.json``: kappa -0.05, median latency difference +1 ms,
recall -3 percentage points.

Metrics compared (``validation/layer3``):

- ``n1_ds004080.json`` (written by ``validation/robustness.py``): per reading of the
  published N1 rule (``paper``, ``code``), ``kappa`` (kappa), ``recall`` (recall) and
  ``median_abs_latency_diff_ms`` (latency);
- ``erpcore_difference.json``, ``erpcore_parent.json`` (``validation/erpcore.py``): per
  component, ``labelled`` and ``within_10ms`` (recall) and the absolute median latency
  difference ``abs_median_diff_ms`` (latency).

The baseline is the same files at a git reference (default: the latest ``v*`` tag, else
``HEAD``) or in a folder; the current values are the working tree's. A metric present in the
baseline and absent (or not finite) in the current files is a regression too.

A regression is explained when the ``## Unreleased`` section of ``CHANGELOG.md`` has a line
``Layer-3 regression explained: <metric id>: <reason>``.

Usage::

    python tools/layer3_gate.py [--baseline REF_OR_DIR] [--current DIR] [--changelog PATH]
                                [--json OUT]

Exit code 0 when nothing blocks, 1 when an unexplained regression blocks the release.
"""

import argparse
import json
import math
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER3 = "validation/layer3"
FILES = ("n1_ds004080.json", "erpcore_difference.json", "erpcore_parent.json")
TOLERANCES = ROOT / LAYER3 / "tolerances.json"
EXPLAINED = re.compile(r"Layer-3 regression explained: (\S+?):\s*\S")


def tolerances(path=TOLERANCES):
    """Return the tolerances of the layer-3 metrics read from ``path``."""
    with open(path, encoding="utf-8") as fobj:
        return json.load(fobj)["tolerances"]


def metrics(documents):
    """{metric id: (kind, value)} from the layer-3 documents ``{file name: dict}``."""
    out = {}
    n1 = documents.get("n1_ds004080.json")
    if n1:
        for reading, row in n1.get("metrics", {}).items():
            out[f"n1_ds004080/{reading}/kappa"] = ("kappa", row.get("kappa"))
            out[f"n1_ds004080/{reading}/recall"] = ("recall", row.get("recall"))
            out[f"n1_ds004080/{reading}/median_abs_latency_diff_ms"] = (
                "latency",
                row.get("median_abs_latency_diff_ms"),
            )
    for name in ("erpcore_difference.json", "erpcore_parent.json"):
        doc = documents.get(name)
        if not doc:
            continue
        stem = name[: -len(".json")]
        for row in doc.get("summary", []):
            component = row["component"]
            out[f"{stem}/{component}/labelled"] = ("recall", row.get("labelled"))
            out[f"{stem}/{component}/within_10ms"] = ("recall", row.get("within_10ms"))
            diff = row.get("median_diff_ms")
            out[f"{stem}/{component}/abs_median_diff_ms"] = (
                "latency",
                None if diff is None else abs(diff),
            )
    return out


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def compare(baseline, current, tol):
    """Rows ``{metric, kind, baseline, current, change, tolerance, regression}``."""
    rows = []
    for key, (kind, base) in sorted(baseline.items()):
        if not _finite(base):
            continue
        now = current.get(key, (kind, None))[1]
        if not _finite(now):
            rows.append(
                {
                    "metric": key,
                    "kind": kind,
                    "baseline": base,
                    "current": None,
                    "change": None,
                    "tolerance": tol[kind],
                    "regression": True,
                }
            )
            continue
        change = now - base
        if kind == "latency":
            regression = change > tol["latency"] + 1e-12
        elif kind == "recall":
            regression = change < -tol["recall"] - 1e-12
        else:
            regression = change < -tol["kappa"] - 1e-12
        rows.append(
            {
                "metric": key,
                "kind": kind,
                "baseline": base,
                "current": now,
                "change": change,
                "tolerance": tol[kind],
                "regression": bool(regression),
            }
        )
    return rows


def explained(changelog_text):
    """Metric ids explained in the ``## Unreleased`` section of the CHANGELOG."""
    _, found, section = changelog_text.partition("## Unreleased")
    if not found:
        return set()
    body = re.split(r"\n## ", section, maxsplit=1)[0]
    return set(EXPLAINED.findall(body))


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def default_baseline():
    """Return the latest ``v*`` tag, else HEAD."""
    tags = _git("tag", "--list", "v*", "--sort=-v:refname").stdout.split()
    return tags[0] if tags else "HEAD"


def load(source):
    """{file name: document} from a folder or a git reference."""
    docs = {}
    path = pathlib.Path(source)
    if path.is_dir():
        for name in FILES:
            if (path / name).exists():
                docs[name] = json.loads((path / name).read_text(encoding="utf-8"))
        return docs
    for name in FILES:
        shown = _git("show", f"{source}:{LAYER3}/{name}")
        if shown.returncode == 0:
            docs[name] = json.loads(shown.stdout)
    return docs


def gate(baseline_docs, current_docs, changelog_text, tol=None):
    """Return the gate report: the compared metrics, the regressions and the blocking ones."""
    tol = tol or tolerances()
    rows = compare(metrics(baseline_docs), metrics(current_docs), tol)
    ok = explained(changelog_text)
    for row in rows:
        row["explained"] = row["metric"] in ok
    blocking = [r for r in rows if r["regression"] and not r["explained"]]
    return {
        "tolerances": tol,
        "n_metrics": len(rows),
        "rows": rows,
        "regressions": [r["metric"] for r in rows if r["regression"]],
        "blocking": [r["metric"] for r in blocking],
        "release_blocked": bool(blocking),
    }


def main(argv=None):
    """Compare the layer-3 results with the baseline; exit 1 when the release is blocked."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--baseline",
        default=None,
        help="git reference or folder (default: latest v* tag, else HEAD)",
    )
    parser.add_argument("--current", default=str(ROOT / LAYER3))
    parser.add_argument("--changelog", default=str(ROOT / "CHANGELOG.md"))
    parser.add_argument("--json", default=None, help="write the report here")
    args = parser.parse_args(argv)
    baseline = args.baseline or default_baseline()
    report = gate(
        load(baseline),
        load(args.current),
        pathlib.Path(args.changelog).read_text(encoding="utf-8"),
    )
    report["baseline"] = baseline
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for row in report["rows"]:
        if row["regression"]:
            state = "explained in CHANGELOG" if row["explained"] else "BLOCKS THE RELEASE"
            print(
                f"{row['metric']}: {row['baseline']} -> {row['current']} "
                f"({row['kind']} tolerance {row['tolerance']}): {state}"
            )
    print(
        f"layer-3 gate against {baseline}: {report['n_metrics']} metrics, "
        f"{len(report['regressions'])} regressions, {len(report['blocking'])} blocking"
    )
    return 1 if report["release_blocked"] else 0


if __name__ == "__main__":
    sys.exit(main())
