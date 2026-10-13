# SPDX-License-Identifier: Apache-2.0
"""``n_jobs``: run independent units of work in parallel (spec §5 conventions, §9.2).

``n_jobs`` uses ``joblib`` (extra ``parallel``, processes) when it is installed, otherwise a
thread pool. A unit of work never draws random numbers of its own: the caller draws them
in the order of the serial loop and hands them to the units, so that the result does not
depend on ``n_jobs``. In worker processes the warnings and the ``evoparam`` log records of
each unit are collected and issued again in the calling process, in the order of the
units, so that a unit's WARNING record (REQ-307) and its warnings reach the caller as in a
serial run, at the same lines. In the thread pool the warnings are issued by the threads
themselves: one that points past its unit points into the pool.
"""

import logging
import os
import sys
import warnings
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from evoparam.errors import _frame_above, _warn_at

from . import _rungs
from ._logging import LOGGER, progress
from ._optional import require

__all__ = ["n_workers", "parallel_map"]

_T = TypeVar("_T")
_R = TypeVar("_R")


def n_workers(n_jobs: int | None, n_items: int) -> int:
    """Return the number of workers for ``n_items`` units (``n_jobs`` as in joblib).

    ``None``, 0 and 1 mean one worker (the serial loop); ``-1`` one per core, ``-2`` all
    cores but one, and so on; never more than ``n_items``.
    """
    if n_jobs is None or n_jobs == 0:
        jobs = 1
    elif n_jobs < 0:
        jobs = max(1, (os.cpu_count() or 1) + 1 + int(n_jobs))
    else:
        jobs = int(n_jobs)
    return max(1, min(jobs, n_items))


class _Collect(logging.Handler):
    """Keep the records of a unit run in a worker process."""

    def __init__(self) -> None:
        super().__init__(logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        # the message is formatted here: the arguments may not survive pickling
        record.msg, record.args = record.getMessage(), None
        record.exc_info, record.exc_text = None, None
        self.records.append(record)


def _captured(
    function: Callable[[Any], Any], item: Any, level: int, parent: int
) -> tuple[Any, list[tuple[Any, ...]], list[logging.LogRecord]]:
    """Run one unit; in a worker process, also return its warnings and log records.

    The unit runs below the rungs of :mod:`evoparam.utils._rungs`: a warning located above
    the unit (``stacklevel`` reaching past the unit function) carries how many frames above
    the unit it points to, so that the calling process locates it as many frames above the
    caller of :func:`parallel_map`.
    """
    if os.getpid() == parent:  # a thread backend chosen by the caller: nothing to carry
        return function(item), [], []
    logger = logging.getLogger(LOGGER)
    logger.setLevel(level)
    handler = _Collect()
    logger.addHandler(handler)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _rungs.climb(function, item)
    finally:
        logger.removeHandler(handler)
    issued = [
        (
            w.message,
            w.category,
            w.filename,
            w.lineno,
            _rungs.LEVELS.get(w.lineno, 0) if w.filename == _rungs.FILE else 0,
        )
        for w in caught
    ]
    return result, issued, handler.records


def _replay(issued: Sequence[tuple[Any, ...]], records: Sequence[logging.LogRecord]) -> None:
    for record in records:
        logging.getLogger(record.name).handle(record)
    here = sys._getframe()
    for message, category, filename, lineno, steps in issued:
        # steps frames above the caller of parallel_map (the frames of this module skipped)
        frame = _frame_above(here, steps) if steps else None
        if frame is not None:
            _warn_at(message, category, frame)
        else:
            warnings.warn_explicit(message, category, filename, lineno)


def parallel_map(
    function: Callable[[_T], _R],
    items: Iterable[_T],
    n_jobs: int | None = 1,
    *,
    desc: str | None = None,
) -> list[_R]:
    """``[function(item) for item in items]``, in ``n_jobs`` workers, in the order of ``items``.

    One worker runs the plain loop. Otherwise the units run in ``joblib`` processes when
    joblib is installed (``function`` and the items are serialised), else in a thread pool.
    With ``verbose="progress"`` a bar counts the units (:func:`evoparam.utils.verbose`).
    The first exception of a unit propagates.
    """
    units = list(items)
    workers = n_workers(n_jobs, len(units))
    if workers == 1:
        return [function(item) for item in progress(units, total=len(units), desc=desc)]
    try:
        require("joblib", extra="parallel", caller="evoparam.utils._parallel.parallel_map")
        import joblib
    except ImportError:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(progress(pool.map(function, units), total=len(units), desc=desc))
    level = logging.getLogger(LOGGER).getEffectiveLevel()
    parent = os.getpid()
    tasks = (joblib.delayed(_captured)(function, item, level, parent) for item in units)
    results: list[_R] = []
    outputs = joblib.Parallel(n_jobs=workers, return_as="generator")(tasks)
    for result, issued, records in progress(outputs, total=len(units), desc=desc):
        _replay(issued, records)
        results.append(result)
    return results
