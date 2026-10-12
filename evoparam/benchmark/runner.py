# SPDX-License-Identifier: Apache-2.0
"""Run methods on a suite and score them (``evoparam benchmark run``, REQ-350).

``run_suite`` generates every cell of the suite from its seed, writes the task inputs,
runs the baselines (and the adapters of external tools) on them, writes their outputs,
scores every method on every task it answers, and writes the signed results, a Markdown
report and the leaderboard page. Layout of ``out``::

    inputs/<task>/<cell>.json, <cell>.npz        task inputs (spec §8.4)
    predictions/<method>/<task>/<cell>.tsv       outputs (.npz for T4)
    results/<method>__<task>.json                 signed results
    report.md, leaderboard.html, summary.json     report, page, timings and failures
"""

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

from evoparam.errors import DataError

from ._canon import clean
from .adapters import Adapter, load_adapter
from .baselines import BASELINES
from .scoring import score, write_result
from .suite import load_suite, make_inputs, simulate_cell
from .tasks import output_path, write_input, write_output

__all__ = ["run_suite", "result_name"]


def result_name(method, task):
    """File name of a result: ``<method>__<task>.json``."""
    return f"{method}__{task}.json"


def _cell_job(args):
    """Simulate one cell, write its inputs, run the baselines; returns timings and
    failures (runs in a worker process when ``n_jobs > 1``).
    """
    cell, tasks, methods, out = args
    timings, failures = {}, []
    start = time.perf_counter()
    ts, truth = simulate_cell(cell)
    inputs = make_inputs(cell, ts, truth, tasks=tasks)
    timings["simulate"] = time.perf_counter() - start
    for task, task_input in inputs.items():
        write_input(task_input, os.path.join(out, "inputs", task))
    for name in methods:
        baseline = BASELINES[name]
        wanted = {t: ti for t, ti in inputs.items() if t in baseline.tasks}
        if not wanted:
            continue
        start = time.perf_counter()
        try:
            outputs = baseline.run(wanted)
        except Exception as err:  # a failing method is recorded, the run goes on
            failures.append(
                {
                    "cell": cell.id,
                    "method": name,
                    "error": f"{type(err).__name__}: {err}",
                    "trace": traceback.format_exc(limit=3),
                }
            )
            continue
        timings[name] = time.perf_counter() - start
        for task, value in outputs.items():
            write_output(
                task,
                value,
                output_path(os.path.join(out, "predictions", name, task), task, cell.id),
            )
    return cell.id, timings, failures


def run_suite(
    suite="core",
    out=".",
    *,
    tasks=None,
    methods=None,
    adapters=(),
    hidden_seeds=None,
    split="public",
    n_jobs=1,
    page=True,
    log=None,
):
    """Run baselines and adapters on a suite, score them and write the results.

    Parameters
    ----------
    suite : str or Suite
        ``"core"``, ``"full"``, a suite file or a loaded suite.
    out : str
        Output folder (created).
    tasks : sequence of str, optional
        Default: every task of the suite.
    methods : sequence of str, optional
        Baseline names (default: every baseline answering one of the tasks); an empty
        sequence writes the inputs only (for external methods).
    adapters : sequence of Adapter or str
        External tools (objects or specification files).
    hidden_seeds : dict or str, optional
        The hidden seeds document; with ``split="all"`` the hidden cells are run too.
    split : {"public", "hidden", "all"}
    n_jobs : int
        Cells run in parallel worker processes.
    page : bool
        Write ``report.md`` and ``leaderboard.html``.
    log : callable, optional
        Called with progress lines.

    Returns
    -------
    dict
        The summary also written to ``summary.json``: tasks, methods, cells, duration (s)
        per step, failures and the result files.
    """
    from .leaderboard import write_leaderboard
    from .report import write_report

    log = log or (lambda message: None)
    start = time.perf_counter()
    if not hasattr(suite, "cells"):
        suite = load_suite(suite, hidden_seeds=hidden_seeds)
    elif hidden_seeds is not None:
        suite = suite.with_hidden_seeds(hidden_seeds)
    tasks = list(tasks) if tasks else suite.task_ids()
    for task in tasks:
        if task not in suite.tasks:
            raise DataError(f"suite {suite.name!r} has no task {task}.")
    if methods is None:
        methods = [name for name, b in BASELINES.items() if set(b.tasks) & set(tasks)]
    unknown = [m for m in methods if m not in BASELINES]
    if unknown:
        raise DataError(f"unknown baselines {unknown}; baselines: {sorted(BASELINES)}.")
    adapters = [a if isinstance(a, Adapter) else load_adapter(a) for a in adapters]
    for adapter in adapters:
        adapter.check()  # a missing program stops the run before any work
    os.makedirs(out, exist_ok=True)
    cells = {}
    for task in tasks:
        for cell in suite.cells(task, split=split):
            cells.setdefault(cell.id, cell)
    jobs = [(cell, [t for t in tasks if t in cell.tasks], methods, out) for cell in cells.values()]
    timings, failures = {}, []
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            for cell_id, cell_times, cell_failures in pool.map(_cell_job, jobs):
                timings[cell_id] = cell_times
                failures += cell_failures
                log(f"{cell_id}: {sum(cell_times.values()):.1f} s")
    else:
        for job in jobs:
            cell_id, cell_times, cell_failures = _cell_job(job)
            timings[cell_id] = cell_times
            failures += cell_failures
            log(f"{cell_id}: {sum(cell_times.values()):.1f} s")
    run_seconds = time.perf_counter() - start

    adapter_seconds = 0.0
    for adapter in adapters:
        for task in tasks:
            if adapter.tasks and task not in adapter.tasks:
                continue
            folder = os.path.join(out, "predictions", adapter.name, task)
            for cell in suite.cells(task, split=split):
                t0 = time.perf_counter()
                try:
                    adapter.run(os.path.join(out, "inputs", task, f"{cell.id}.json"), folder)
                except Exception as err:  # recorded; the cell is listed as missing
                    failures.append(
                        {
                            "cell": cell.id,
                            "method": adapter.name,
                            "error": f"{type(err).__name__}: {err}",
                        }
                    )
                adapter_seconds += time.perf_counter() - t0
            log(f"adapter {adapter.name}, {task}: done")

    t0 = time.perf_counter()
    results = []
    method_info = {name: BASELINES[name].info() for name in methods}
    method_info.update({a.name: a.info() for a in adapters})
    for task in tasks:
        task_inputs = None
        for name in list(methods) + [a.name for a in adapters]:
            folder = os.path.join(out, "predictions", name, task)
            if not os.path.isdir(folder):
                continue
            if task_inputs is None:
                task_inputs = [
                    make_inputs(cell, tasks=[task])[task]
                    for cell in suite.cells(task, split=split)
                ]
            document = score(
                task,
                folder,
                suite,
                method=method_info[name],
                allow_missing=True,
                inputs=task_inputs,
            )
            path = write_result(document, os.path.join(out, "results", result_name(name, task)))
            results.append(path)
            log(f"scored {name} on {task}")
    score_seconds = time.perf_counter() - t0
    pages = []
    if page and results:
        pages.append(write_report(os.path.join(out, "results"), os.path.join(out, "report.md")))
        pages.append(
            write_leaderboard(os.path.join(out, "results"), os.path.join(out, "leaderboard.html"))
        )
    summary = {
        "suite": {"name": suite.name, "version": suite.version, "sha256": suite.sha256},
        "split": split,
        "tasks": tasks,
        "methods": list(methods),
        "adapters": [a.name for a in adapters],
        "n_cells": len(cells),
        "n_jobs": n_jobs,
        "duration_s": {
            "cells": run_seconds,
            "adapters": adapter_seconds,
            "scoring": score_seconds,
            "total": time.perf_counter() - start,
        },
        "cell_timings_s": timings,
        "failures": failures,
        "results": results,
        "pages": pages,
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fobj:
        json.dump(clean(summary), fobj, indent=1, sort_keys=True)
        fobj.write("\n")
    return summary
