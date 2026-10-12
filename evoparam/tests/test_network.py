# SPDX-License-Identifier: Apache-2.0
"""REQ-306: no network, no telemetry.

- The tests of the ``core`` marker set run with a socket guard (``conftest.py``): only tests
  marked ``network`` may connect. Checked here from inside a core test.
- Only ``evoparam/io/_fetch.py`` imports a network or serialisation-by-pickle module;
  ``np.load`` is always called with ``allow_pickle=False``; no update check, no telemetry.
- The checksum and offline rules of the fetch functions are tested in
  ``tests/io/test_fetch.py`` (one corrupted byte: ``ReaderError`` and the file deleted;
  ``EVOPARAM_OFFLINE=1``); the whole-package dynamic check is ``tests/test_no_network.py``.
"""

import ast
import pathlib
import re
import socket
from typing import Any

import pytest

import evoparam

pytestmark = pytest.mark.core

PACKAGE = pathlib.Path(evoparam.__file__).resolve().parent
NETWORK = {
    "socket",
    "urllib.request",
    "urllib3",
    "http.client",
    "requests",
    "httpx",
    "aiohttp",
    "ftplib",
    "smtplib",
    "pickle",
    "dill",
    "cloudpickle",
    "shelve",
}


def test_core_tests_run_with_the_socket_guard() -> None:
    with pytest.raises(OSError, match="REQ-306"):
        socket.create_connection(("127.0.0.1", 9), timeout=0.1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="REQ-306"):
            sock.connect(("127.0.0.1", 9))
    finally:
        sock.close()


def _imports(path: Any) -> Any:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module
            yield from (f"{node.module}.{a.name}" for a in node.names)


def test_only_the_fetch_helpers_import_network_modules() -> None:
    hits = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix()
        if rel.startswith("tests/") or rel == "io/_fetch.py":
            continue
        for name in set(_imports(path)):
            if name in NETWORK or name.split(".")[0] in NETWORK - {"urllib"}:
                hits.append(f"{rel}: {name}")
    assert not hits, hits


def test_np_load_never_allows_pickle() -> None:
    hits = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix()
        if rel.startswith("tests/"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "load"
                and getattr(node.func.value, "id", "") in ("np", "numpy")
            ):
                flag = [k for k in node.keywords if k.arg == "allow_pickle"]
                if not flag or not (
                    isinstance(flag[0].value, ast.Constant) and flag[0].value.value is False
                ):
                    hits.append(f"{rel}:{node.lineno}")
    assert not hits, hits


def test_no_telemetry_or_update_check() -> None:
    pattern = re.compile(
        r"telemetry|analytics|update[_ ]check|check_for_update|pypi\.org/pypi", re.IGNORECASE
    )
    hits = [
        f"{p.relative_to(PACKAGE)}"
        for p in PACKAGE.rglob("*.py")
        if "tests" not in p.relative_to(PACKAGE).parts
        and pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert not hits, hits


def _grep_as_written() -> Any:
    """``grep -rn "pickle\\|urllib\\|requests\\|httpx" evoparam/`` outside the fetch helpers
    and the ``allow_pickle=False`` lines."""

    hits = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix()
        if rel.startswith("tests/") or rel == "io/_fetch.py":
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"pickle|urllib|requests|httpx", line) and not re.search(
                r"allow_pickle=False", line
            ):
                hits.append(f"{rel}:{n}")
    return hits


@pytest.mark.xfail(
    strict=True,
    reason="REQ-306 grep as written matches lines that are not "
    "network access: 'HTTP range requests' in docstrings of the remote readers "
    "(brainvision, mef), 'from urllib.parse import quote, unquote' in io/mef.py "
    "(URL text only); 7 lines in 2 modules measured on "
    "2026-10-11 (S20). The import-level check above holds",
)
def test_grep_as_written() -> None:
    assert not _grep_as_written(), _grep_as_written()
