# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the benchmark tests: the suites, cached task inputs, outputs built from
the truth, and the environment switches."""

import functools
import os
import pathlib
from typing import Any

import pytest

import evoparam
from evoparam.benchmark import load_suite, make_inputs

ROOT = pathlib.Path(evoparam.__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "benchmark"
RESULTS = DOCS / "results"
ADAPTERS = ROOT / "adapters"

SLOW = pytest.mark.skipif(
    os.environ.get("EVOPARAM_SLOW_TESTS") != "1",
    reason="set EVOPARAM_SLOW_TESTS=1 to run the long benchmark tests",
)
PERF = pytest.mark.skipif(
    os.environ.get("EVOPARAM_PERF_TESTS") != "1",
    reason="set EVOPARAM_PERF_TESTS=1 to time the core suite",
)
NETWORK = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
        reason="set EVOPARAM_NETWORK_TESTS=1 to read public data",
    ),
]


def hidden_seeds_path() -> Any:
    """The hidden seeds document, kept outside the repository: its path is given by
    ``EVOPARAM_BENCHMARK_HIDDEN_SEEDS`` (the tests that need it skip without it)."""

    path = os.environ.get("EVOPARAM_BENCHMARK_HIDDEN_SEEDS")
    if not path or not os.path.isfile(path):
        pytest.skip("set EVOPARAM_BENCHMARK_HIDDEN_SEEDS to the hidden seeds document")
    return path


@functools.cache
def suite(name: Any = "core") -> Any:
    return load_suite(name)


@functools.cache
def inputs(task: Any, n: Any = None) -> Any:
    """The task inputs (with their truth) of the first ``n`` public cells of ``core``."""

    cells = suite("core").cells(task, split="public")[:n]
    return tuple(make_inputs(cell, tasks=[task])[task] for cell in cells)


def perfect_t1(ti: Any, p_value: Any = 1e-6) -> Any:
    """T1 rows that give the true decision for every scored channel."""

    truth = ti.truth
    stim = set(truth["stim_channels"])
    return [
        {
            "site": ti.descriptor["site"],
            "channel": c,
            "responsive": int(c in truth["responders"]),
            "p_value": p_value if c in truth["responders"] else 0.5,
        }
        for c in ti.descriptor["ch_names"]
        if c not in stim
    ]


def perfect_t2(ti: Any, shift: Any = 0.0) -> Any:
    """T2 rows with the true N1 of every true responder (latency shifted by ``shift``)."""

    return [
        {
            "site": ti.descriptor["site"],
            "channel": c,
            "latency": v["latency"] + shift,
            "amplitude": v["amplitude"],
        }
        for c, v in sorted(ti.truth["n1"].items())
    ]
