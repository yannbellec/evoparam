# SPDX-License-Identifier: Apache-2.0
"""``n_jobs`` (spec §5 conventions, §9.2): what a unit run in a worker process carries back.

A unit run in a joblib worker returns its result with its warnings and its ``evoparam`` log
records, which the calling process issues again in the order of the units (REQ-307: the
WARNING record of a failed unit reaches the caller as in a serial run). The worker side
runs in the test process here, so that what it collects can be inspected; without joblib
the units run in a thread pool.
"""

import logging
import os
import warnings
from typing import Any

import pytest

from evoparam.utils import _parallel, _rungs
from evoparam.utils._parallel import _captured, _replay, parallel_map

pytestmark = [pytest.mark.core, pytest.mark.usefixtures("logger_level")]

LOG = logging.getLogger("evoparam.tests.unit")


@pytest.fixture
def logger_level() -> Any:
    """A worker sets the level of the evoparam logger: restore it after the test."""
    logger = logging.getLogger("evoparam")
    level = logger.level
    yield
    logger.setLevel(level)


def unit(item: int) -> int:
    LOG.warning("unit %d of %s", item, "run")
    warnings.warn(f"unit {item}", UserWarning, stacklevel=1)
    return 10 * item


def unit_pointing_above(item: int) -> int:
    # stacklevel 3: two frames above the unit, on the second rung in a worker
    warnings.warn(f"above {item}", UserWarning, stacklevel=3)
    return item


def test_a_unit_in_a_worker_collects_its_warnings_and_records() -> None:
    other = os.getpid() + 1  # as if this process were a worker of another parent
    result, issued, records = _captured(unit, 2, logging.WARNING, other)
    assert result == 10 * 2
    (message, category, filename, lineno, steps) = issued[0]
    assert str(message) == "unit 2"
    assert category is UserWarning
    assert filename == __file__
    assert steps == 0
    assert lineno == unit.__code__.co_firstlineno + 2
    (record,) = records
    assert record.getMessage() == "unit 2 of run"
    assert record.args is None  # formatted in the worker
    assert record.levelno == logging.WARNING
    # the handler is removed after the unit
    assert not logging.getLogger("evoparam").handlers or all(
        type(h).__name__ != "_Collect" for h in logging.getLogger("evoparam").handlers
    )


def test_a_warning_above_the_unit_is_located_above_the_caller() -> None:
    _, issued, _ = _captured(unit_pointing_above, 1, logging.WARNING, os.getpid() + 1)
    (_, _, filename, lineno, steps) = issued[0]
    assert filename == _rungs.FILE
    assert steps == _rungs.LEVELS[lineno]
    assert steps == 3 - 1  # stacklevel 3: two frames above the unit


def test_replay_issues_the_records_and_warnings(caplog: Any) -> None:
    _, issued, records = _captured(unit, 3, logging.WARNING, os.getpid() + 1)
    caplog.clear()
    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.WARNING):
        warnings.simplefilter("always")
        _replay(issued, records)
    assert [r.getMessage() for r in caplog.records] == ["unit 3 of run"]
    assert [str(w.message) for w in caught] == ["unit 3"]
    assert caught[0].filename == __file__
    _, issued, _ = _captured(unit_pointing_above, 4, logging.WARNING, os.getpid() + 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _replay(issued, [])
    assert [str(w.message) for w in caught] == ["above 4"]
    assert caught[0].filename != _rungs.FILE  # located above the caller, not on a rung


def test_in_the_calling_process_a_unit_runs_as_is() -> None:
    with pytest.warns(UserWarning, match="unit 5"):
        result, issued, records = _captured(unit, 5, logging.WARNING, os.getpid())
    assert (result, issued, records) == (50, [], [])


def test_without_joblib_the_units_run_in_threads(monkeypatch: Any) -> None:
    def no_joblib(*_args: Any, **_kwargs: Any) -> Any:
        raise ImportError("joblib hidden")

    monkeypatch.setattr(_parallel, "require", no_joblib)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert parallel_map(unit, range(6), n_jobs=3) == [0, 10, 20, 30, 40, 50]
