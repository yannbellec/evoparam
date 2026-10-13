# SPDX-License-Identifier: Apache-2.0
"""Reading a file, reproducing a saved result, citing, the self-test, logging."""

import evoparam
from evoparam.data import TrialSet


def read(path: str) -> TrialSet:
    data = evoparam.read(path)
    assert isinstance(data, TrialSet)
    return data


def check(path: str) -> bool:
    report = evoparam.reproduce(path)
    return bool(report.identical)


def housekeeping() -> str:
    evoparam.set_log_level("warning")
    with evoparam.verbose("debug"):
        error: float = evoparam.selftest()
    return f"{error:.2e} " + evoparam.cite(file=None)
