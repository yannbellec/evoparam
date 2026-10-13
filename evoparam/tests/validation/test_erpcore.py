# SPDX-License-Identifier: Apache-2.0
"""ERP CORE, 40 participants x 7 components (REQ-274), on a local copy (CC-BY 4.0, OSF
DOI 10.18115/D5JW4R) downloaded by ``validation/erpcore.py``'s documented layout; runs only
when ``EVOPARAM_ERPCORE`` names that folder.

Measured on 2026-10-13 (difference waves on the published channels; the ``erp`` preset
targeted at the paradigm's component, ``presets.targeted``: three components of its
polarity in the published window widened by 25 ms, no background; the component is the sum
of its labelled parts, and its latency is the 50 % area latency of that sum, compared with
the same measure on the average): within 10 ms for ERN 95 %, LRP 95 %, MMN 100 %, N170 95 %,
N2pc 92 % (36 of 39), N400 95 %, P3 90 %. Every wave gets a value with a fixed number of
components; whether the component is there is the existence test's answer.
"""

import importlib.util
import os
import pathlib
from typing import Any

import pytest

ROOT = os.environ.get("EVOPARAM_ERPCORE", "")
SCRIPT = pathlib.Path(__file__).parents[3] / "validation" / "erpcore.py"
pytestmark = pytest.mark.skipif(
    not (ROOT and os.path.isdir(ROOT) and SCRIPT.exists()),
    reason="set EVOPARAM_ERPCORE to the local ERP CORE copy",
)


@pytest.fixture(scope="module")
def table() -> Any:
    pytest.importorskip("mne")
    spec = importlib.util.spec_from_file_location("erpcore", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summary(module.run())


def test_req274_lateralized_readiness_potential(table: Any) -> None:

    assert table.loc["LRP", "n"] == 40
    assert table.loc["LRP", "within_10ms"] >= 0.9


@pytest.mark.parametrize("component", ["ERN", "N170", "N400", "MMN", "N2pc", "P3"])
def test_req274_other_components(table: Any, component: Any) -> None:

    assert table.loc[component, "within_10ms"] >= 0.9
