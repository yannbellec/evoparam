# SPDX-License-Identifier: Apache-2.0
"""Configuration file for pytest."""

import importlib.util
import socket
from typing import Any

import pytest


def pytest_configure() -> None:
    if importlib.util.find_spec("matplotlib") is not None:
        import matplotlib

        matplotlib.use("agg")


class _NetworkRefusedError(OSError):
    pass


def _refuse(*_args: Any, **_kwargs: Any) -> None:
    raise _NetworkRefusedError(
        "network access refused: tests of the core suite run with a socket "
        "guard (REQ-306); mark the test 'network' to connect"
    )


@pytest.fixture(autouse=True)
def _socket_guard(request: Any, monkeypatch: Any) -> Any:
    """REQ-306: every test of the ``core`` marker set runs with a socket guard; only tests
    marked ``network`` may connect. Local (AF_UNIX) sockets are left alone."""

    if (
        request.node.get_closest_marker("core") is None
        or request.node.get_closest_marker("network") is not None
    ):
        yield
        return
    original = socket.socket.connect

    def connect(self, address):
        if getattr(socket, "AF_UNIX", None) is not None and self.family == socket.AF_UNIX:
            return original(self, address)
        return _refuse()

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", lambda _self, _address: _refuse())
    monkeypatch.setattr(socket, "create_connection", _refuse)
    yield
