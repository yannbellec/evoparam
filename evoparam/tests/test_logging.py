# SPDX-License-Identifier: Apache-2.0
"""REQ-304: logging through the standard library.

A failed channel emits exactly one WARNING record naming the channel (with REQ-307);
``verbose="debug"`` gives DEBUG records from ``evoparam.models``; the ``verbose`` mapping is
implemented once (``evoparam.utils.verbose``); ``set_log_level`` and ``set_log_file`` are
the global switches; no ``print`` outside the command line.
"""

import ast
import logging
import pathlib

import numpy as np
import pytest

import evoparam
from evoparam.models import ResponseModel
from evoparam.models import engine as _engine
from evoparam.utils import set_log_file, set_log_level, verbose

PACKAGE = pathlib.Path(evoparam.__file__).resolve().parent


def _evoked():
    from evoparam.data import Evoked

    rng = np.random.default_rng(0)
    times = np.arange(-0.2, 0.5, 1 / 512)
    data = rng.normal(0, 1, (3, times.size))
    data += -20 * np.exp(-0.5 * ((times - 0.03) / 0.004) ** 2)
    return Evoked(data=data, times=times, sfreq=512.0, ch_names=["A1", "A2", "A3"])


def test_package_logger_has_a_null_handler_only():
    handlers = logging.getLogger("evoparam").handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)
    assert not [
        h
        for h in handlers
        if not isinstance(h, logging.NullHandler) and not getattr(h, "_evoparam_log_file", False)
    ]


def test_failed_channel_emits_one_warning_with_its_name(caplog, monkeypatch):
    evoked = _evoked()
    original = _engine.fit_waveform_core

    def failing(times, y, **kwargs):
        if np.isclose(y[0], evoked.data[1, 0]):
            raise FloatingPointError("injected failure")
        return original(times, y, **kwargs)

    monkeypatch.setattr(_engine, "fit_waveform_core", failing)
    with caplog.at_level(logging.WARNING, logger="evoparam"):
        group = ResponseModel("ccep", max_components=1).fit_group(evoked)
    records = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "injected failure" in r.getMessage()
    ]
    assert len(records) == 1
    assert "A2" in records[0].getMessage()
    assert [f.status for f in group.fits].count("failed") == 1


def test_verbose_debug_gives_debug_records_from_models(caplog):
    with caplog.at_level(logging.NOTSET), verbose("debug"):
        assert logging.getLogger("evoparam").getEffectiveLevel() == logging.DEBUG
        ResponseModel("ccep", max_components=1).fit_group(_evoked())
    assert any(
        r.levelno == logging.DEBUG and r.name.startswith("evoparam") for r in caplog.records
    )


@pytest.mark.parametrize(
    "value,level",
    [
        (True, logging.INFO),
        ("info", logging.INFO),
        ("debug", logging.DEBUG),
        ("warning", logging.WARNING),
        ("error", logging.ERROR),
        ("progress", logging.INFO),
    ],
)
def test_verbose_mapping(value, level):
    logger = logging.getLogger("evoparam")
    before = logger.level
    with verbose(value):
        assert logger.level == level
    assert logger.level == before
    with verbose(None), verbose(False):
        assert logger.level == before


def test_verbose_decorator_reads_the_argument():
    seen = []

    @verbose
    def func(x, verbose=None):
        seen.append(logging.getLogger("evoparam").level)
        return x

    assert func(1, verbose="debug") == 1
    assert seen == [logging.DEBUG]


def test_global_switches(tmp_path):
    previous = set_log_level("warning")
    try:
        assert logging.getLogger("evoparam").level == logging.WARNING
        path = tmp_path / "evoparam.log"
        set_log_file(path)
        logging.getLogger("evoparam.models").warning("written to the file")
        set_log_file(None)
        assert "written to the file" in path.read_text(encoding="utf-8")
    finally:
        logging.getLogger("evoparam").setLevel(previous)
    assert evoparam.set_log_level is set_log_level and evoparam.set_log_file is set_log_file


def _prints():
    hits = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE)
        if rel.parts[0] == "tests" or rel.name == "cli.py" or rel.name == "__main__.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print":
                hits.append(f"{rel}:{node.lineno}")
    return hits


def test_no_print_outside_the_cli():
    """``print`` is used by the command line only (``cli.py`` and the ``__main__`` scripts,
    which are command lines too); ``evoparam.cite()`` prints the citation by design and
    nothing else does."""

    assert _prints() == ["__init__.py:" + str(_cite_print_line())]


def _cite_print_line():
    tree = ast.parse((PACKAGE / "__init__.py").read_text(encoding="utf-8"))
    cite = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cite")
    return next(
        n.lineno
        for n in ast.walk(cite)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "print"
    )
