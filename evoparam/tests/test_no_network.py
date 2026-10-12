# SPDX-License-Identifier: Apache-2.0
"""REQ-016: no network by default.

No function accesses the network unless it is a documented fetch or remote-read function
called with remote options; there is no telemetry (§9.10).

- Static: only ``evoparam/io/_fetch.py`` (the documented download and HTTP range-read
  helpers, used by ``fetch_dataset``, the remote BIDS, BrainVision and MEF readers) imports
  a network module, and ``remfile`` is reached only by the remote NWB read.
- Dynamic: in a subprocess whose sockets refuse to connect (and record every attempt), the
  package is imported with every module, then a simulated dataset goes through the CLI
  (``simulate``, ``fit``, ``presets list``, ``cite``, ``--version``) and through
  the API (``fit`` with its existence test, save and load, the HTML and Markdown reports,
  ``reproduce``, local reads). No attempt is recorded.
"""

import ast
import json
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

import evoparam

pytestmark = pytest.mark.core

PACKAGE = pathlib.Path(evoparam.__file__).resolve().parent
NETWORK_MODULES = {
    "socket",
    "urllib.request",
    "http.client",
    "http",
    "requests",
    "urllib3",
    "httpx",
    "aiohttp",
    "ftplib",
    "smtplib",
    "fsspec",
    "pooch",
    "remfile",
    "websocket",
    "telnetlib",
    "xmlrpc.client",
    "ssl",
}
ALLOWED = {
    "evoparam/io/_fetch.py": {"urllib.request", "http.client"},
    "evoparam/io/nwb.py": {"remfile"},
}  # read_nwb of an http(s) address only


def _network_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        elif (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", getattr(node.func, "attr", ""))
            in ("require", "import_module")
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            names = [str(node.args[0].value)]
        for name in names:
            if name in NETWORK_MODULES or name.split(".")[0] in NETWORK_MODULES - {"http"}:
                found.add(name)
    return found


def test_req016_only_the_fetch_helpers_import_network_modules():
    hits = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE.parent).as_posix()
        if "/tests/" in rel:
            continue
        found = _network_imports(path) - ALLOWED.get(rel, set())
        if found:
            hits[rel] = sorted(found)
    assert not hits, hits


def test_req016_remote_reads_are_opt_in():
    import inspect

    from evoparam.io import read_bids

    # remote reading of a BIDS dataset is an explicit option, off by default
    assert inspect.signature(read_bids).parameters["remote"].default is False


GUARD = textwrap.dedent("""
    import socket
    import sys

    ATTEMPTS = []

    def refuse(*args, **kwargs):
        ATTEMPTS.append(repr(args[:2])[:120])
        raise OSError("network access refused by the REQ-016 test")

    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse
    socket.create_connection = refuse
    socket.getaddrinfo = refuse
""")

SCRIPT = textwrap.dedent("""
    import contextlib
    import importlib
    import io
    import json
    import os
    import pkgutil
    import tempfile
    import warnings

    warnings.simplefilter("ignore")
    os.environ.pop("EVOPARAM_NETWORK_TESTS", None)
    import evoparam

    imported = []
    for info in pkgutil.walk_packages(evoparam.__path__, "evoparam."):
        if ".tests" in info.name or info.name.endswith("__main__"):
            continue
        try:
            importlib.import_module(info.name)
            imported.append(info.name)
        except ImportError:
            pass

    from evoparam.cli import main

    def cli(*args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                code = main([str(a) for a in args])
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue()

    folder = tempfile.mkdtemp()
    bids = os.path.join(folder, "bids")
    codes = {}
    codes["simulate"] = cli("simulate", "--modality", "ccep", "--n-trials", "8",
                            "--n-channels", "3", "--random-state", "3", "--out", bids)[0]
    codes["fit"] = cli("fit", bids, "--sub", "01", "--task", "ccep", "--preset", "ccep",
                       "--out", os.path.join(folder, "derivatives"))[0]
    codes["presets"] = cli("presets", "list")[0]
    codes["cite"] = cli("cite")[0]
    codes["version"] = cli("--version")[0]

    from evoparam import sim
    from evoparam.data import load_object, save_object

    trials, truth = sim.simulate_ccep(n_trials=8, n_channels=3, seed=1)
    group = evoparam.fit(trials, preset="ccep", random_state=0)
    path = os.path.join(folder, "group.evoparam")
    save_object(group, path)
    back = load_object(path)
    assert type(back) is type(group)
    evoparam.reproduce(path, data=trials)
    evoparam.reports.html(group, os.path.join(folder, "report.html"))
    evoparam.reports.markdown(group)
    evoparam.io.read(os.path.join(bids), subject="01", task="ccep")
    print(json.dumps({"attempts": ATTEMPTS, "codes": codes, "n_modules": len(imported)}))
""")


def test_req016_no_network_without_remote_options(tmp_path):
    pytest.importorskip("pandas")
    env = dict(
        os.environ, PYTHONWARNINGS="ignore", MPLBACKEND="Agg", EVOPARAM_DATA=str(tmp_path / "data")
    )
    env.pop("EVOPARAM_NETWORK_TESTS", None)
    result = subprocess.run(
        [sys.executable, "-c", GUARD + SCRIPT],
        capture_output=True,
        text=True,
        timeout=1200,
        env=env,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["attempts"] == []
    assert report["n_modules"] >= 60
    assert all(code == 0 for code in report["codes"].values()), report["codes"]


def test_req016_the_guard_records_an_attempt():
    code = GUARD + textwrap.dedent("""
        from evoparam.io._fetch import fetch_bytes
        try:
            fetch_bytes("https://example.org/x", timeout=1)
        except Exception:
            pass
        print(len(ATTEMPTS))
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env=dict(os.environ, EVOPARAM_OFFLINE=""),
    )
    assert int(result.stdout.strip().splitlines()[-1]) >= 1, result.stderr[-2000:]
