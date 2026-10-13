# SPDX-License-Identifier: Apache-2.0
"""Run the whole test suite, network, slow and performance tests included, on one Python.

Spec §12 asks for every step to end "with its REQs green in CI", and REQ-017 asks for
REQ-113/REQ-114 in every build. Many of these tests read OpenNeuro (``EVOPARAM_NETWORK_TESTS``),
take long (``EVOPARAM_SLOW_TESTS``) or time the code (``EVOPARAM_PERF_TESTS``), and the
default pytest run skips them. This runner is the local job that runs all of them (decision
D16, docs/spec/DECISIONS.md). The GitHub workflow defines the same job
(``network-slow-perf`` in ``.github/workflows/build.yml``).

What it does:

1. with ``--venv DIR``, creates a virtual environment from ``--python`` and installs the
   package with the extras of ``--extras`` (default ``all,bayes``: ``all`` is the union of
   §9.2 without ``bayes``, which the NUTS tests need);
2. checks that every optional package the tests use can be imported by that Python, and
   lists the ones that cannot (their tests would be skipped);
3. runs ``python -m pytest`` with the three flags set (unless turned off) and writes a
   JUnit XML report;
4. reads the report and writes ``summary.md`` and ``summary.json``: totals, failures,
   errors, expected failures with their reasons, and every skip with its reason. A skip
   caused by a missing package is listed apart.

Examples::

    python tools/run_all_tests.py                              # this Python, as installed
    python tools/run_all_tests.py --python python3.12 --venv /tmp/ev312
    python tools/run_all_tests.py --no-perf --out reports/run1 evoparam/tests/models
    python tools/run_all_tests.py --env EVOPARAM_ERPCORE=/data/erpcore

The exit code is pytest's (0 when every test passed or was an expected failure).
"""

import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Import names of the optional packages the test suite uses, and the extra holding each.
OPTIONAL = {
    "pandas": "data", "pyarrow": "data", "matplotlib": "plot", "mne": "mne",
    "mne_bids": "bids", "bids": "bids", "pynwb": "nwb", "hdmf": "nwb", "ndx_events": "nwb",
    "h5py": "nwb", "remfile": "nwb", "neo": "neo", "quantities": "neo", "edfio": "edf",
    "statsmodels": "stats", "formulaic": "stats", "jsonschema": "validate",
    "bidsschematools": "validate", "nwbinspector": "validate", "tqdm": "progress",
    "joblib": "parallel", "xarray": "xarray", "numba": "fast", "sklearn": "sklearn",
    "mpmath": "tests", "hypothesis": "tests", "pytest": "tests", "pytest_cov": "tests",
    "pytest_timeout": "tests", "numpyro": "bayes", "jax": "bayes", "arviz": "bayes",
    "pymc": "bayes",
}

FLAGS = {"network": "EVOPARAM_NETWORK_TESTS", "slow": "EVOPARAM_SLOW_TESTS",
         "perf": "EVOPARAM_PERF_TESTS"}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", default=["evoparam"],
                        help="test paths (default: the whole package)")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter to run the tests with (default: this one)")
    parser.add_argument("--venv", help="create (or reuse) a virtual environment here from "
                                       "--python and install the package into it")
    parser.add_argument("--extras", default="all,bayes",
                        help="extras installed with --venv (default: all,bayes)")
    parser.add_argument("--out", help="folder for junit.xml, summary.md and summary.json "
                                      "(default: test-reports/<python>-<time>)")
    for name in FLAGS:
        parser.add_argument(f"--no-{name}", action="store_true",
                            help=f"do not set {FLAGS[name]}=1")
    parser.add_argument("--data", help="download cache (EVOPARAM_DATA)")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="more environment variables (for example EVOPARAM_ERPCORE)")
    parser.add_argument("--doctests", action="store_true",
                        help="also run the doctests of the package")
    parser.add_argument("--pytest-args", default="",
                        help="extra arguments passed to pytest (one string)")
    return parser


def _run(command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    return subprocess.run(command, **kwargs)


def _venv_python(venv):
    folder = "Scripts" if os.name == "nt" else "bin"
    name = "python.exe" if os.name == "nt" else "python"
    return str(pathlib.Path(venv) / folder / name)


def prepare(args):
    """The interpreter to test with (creating and filling the venv when asked)."""

    if not args.venv:
        return args.python
    python = _venv_python(args.venv)
    if not pathlib.Path(python).exists():
        _run([args.python, "-m", "venv", args.venv], check=True)
        _run([python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    spec = f"{ROOT}[{args.extras}]" if args.extras else str(ROOT)
    _run([python, "-m", "pip", "install", "-e", spec], check=True)
    return python


def missing_packages(python):
    """Optional packages that ``python`` cannot import: {name: extra}."""

    code = ("import importlib.util, json, sys\n"
            "names = json.loads(sys.argv[1])\n"
            "print(json.dumps([n for n in names if importlib.util.find_spec(n) is None]))\n")
    proc = subprocess.run([python, "-c", code, json.dumps(sorted(OPTIONAL))],
                          capture_output=True, text=True, check=True)
    return {name: OPTIONAL[name] for name in json.loads(proc.stdout)}


def python_version(python):
    proc = subprocess.run([python, "-c", "import platform; print(platform.python_version())"],
                          capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def summarise(junit, missing):
    """Totals and the lists of failures, errors, expected failures and skips."""

    root = ET.parse(junit).getroot()
    out = {"passed": [], "failed": [], "errors": [], "skipped": [], "xfailed": [],
           "skipped_missing_package": []}
    for case in root.iter("testcase"):
        name = f"{case.get('classname')}::{case.get('name')}"
        failure, error, skipped = (case.find(tag) for tag in ("failure", "error", "skipped"))
        if failure is not None:
            out["failed"].append({"test": name, "message": (failure.get("message") or "")[:500]})
        elif error is not None:
            out["errors"].append({"test": name, "message": (error.get("message") or "")[:500]})
        elif skipped is not None:
            message = skipped.get("message") or ""
            entry = {"test": name, "reason": message[:500]}
            if skipped.get("type") == "pytest.xfail" or message.startswith("xfail"):
                out["xfailed"].append(entry)
            elif "could not import" in message or "No module named" in message:
                out["skipped_missing_package"].append(entry)
            else:
                out["skipped"].append(entry)
        else:
            out["passed"].append(name)
    totals = {key: len(value) for key, value in out.items()}
    totals["passed"] = len(out["passed"])
    out["passed"] = totals["passed"]
    out["totals"] = totals
    out["missing_packages"] = missing
    return out


def write_summary(folder, summary, header):
    (folder / "summary.json").write_text(json.dumps({**header, **summary}, indent=2),
                                         encoding="utf-8")
    lines = [f"# Test run {header['started']}", "",
             f"- Python {header['python_version']} (`{header['python']}`)",
             f"- flags: {', '.join(header['flags']) or 'none'}",
             f"- paths: {' '.join(header['paths'])}",
             f"- pytest exit code: {header['exit_code']}", "",
             "| passed | failed | errors | xfailed | skipped | skipped (missing package) |",
             "|---|---|---|---|---|---|",
             "| {passed} | {failed} | {errors} | {xfailed} | {skipped} | "
             "{skipped_missing_package} |".format(**summary["totals"]), ""]
    if summary["missing_packages"]:
        lines += ["## Optional packages missing (their tests skip)", ""]
        lines += [f"- `{name}` (extra `{extra}`)"
                  for name, extra in sorted(summary["missing_packages"].items())]
        lines.append("")
    for key, title, field in (("failed", "Failures", "message"), ("errors", "Errors", "message"),
                              ("xfailed", "Expected failures", "reason"),
                              ("skipped_missing_package", "Skipped: missing package", "reason"),
                              ("skipped", "Skipped", "reason")):
        if summary[key]:
            lines += [f"## {title}", ""]
            lines += [f"- `{entry['test']}`: {' '.join(entry[field].split())}"
                      for entry in summary[key]]
            lines.append("")
    (folder / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    args = _parser().parse_args(argv)
    python = prepare(args)
    version = python_version(python)
    started = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = pathlib.Path(args.out or ROOT / "test-reports" / f"py{version}-{started}")
    out.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    flags = []
    for name, variable in FLAGS.items():
        if not getattr(args, f"no_{name}"):
            env[variable] = "1"
            flags.append(variable)
    if args.data:
        env["EVOPARAM_DATA"] = args.data
    for item in args.env:
        key, _, value = item.partition("=")
        env[key] = value
    missing = missing_packages(python)
    if missing:
        print("optional packages missing (their tests will skip): "
              + ", ".join(f"{k} [{v}]" for k, v in sorted(missing.items())), flush=True)

    junit = out / "junit.xml"
    command = [python, "-m", "pytest", *args.paths, "-p", "no:cacheprovider", "-rfEsx",
               f"--junitxml={junit}", "-o", "junit_family=xunit2"]
    command += args.pytest_args.split()
    code = _run(command, cwd=str(ROOT), env=env).returncode
    if args.doctests:
        doc_junit = out / "doctests.xml"
        doc = _run([python, "-m", "pytest", "--doctest-modules", "--ignore=evoparam/tests",
                    "evoparam", "-p", "no:cacheprovider", f"--junitxml={doc_junit}"],
                   cwd=str(ROOT), env=env).returncode
        code = code or doc

    header = {"started": started, "python": python, "python_version": version,
              "flags": flags, "paths": args.paths, "exit_code": code}
    if junit.exists():
        summary = summarise(junit, missing)
        write_summary(out, summary, header)
        totals = summary["totals"]
        print(f"summary: {totals} -> {out / 'summary.md'}", flush=True)
    else:
        print(f"pytest wrote no report (exit code {code})", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
