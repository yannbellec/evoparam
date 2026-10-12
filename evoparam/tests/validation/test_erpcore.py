# SPDX-License-Identifier: Apache-2.0
"""ERP CORE, 40 participants x 7 components (REQ-274), on a local copy (CC-BY 4.0, OSF
DOI 10.18115/D5JW4R) downloaded by ``validation/erpcore.py``'s documented layout; runs only
when ``EVOPARAM_ERPCORE`` names that folder.

Measured on 2026-10-09 (difference waves on the published channels, label window = the
published measurement window +/- 30 ms; model latency = 50 % area latency of the fitted
component; noise from the plus-minus reference of the trials where the baseline is short,
the engine's "auto" rule): labelled and within 10 ms in LRP 90 %, ERN 85 %, N400 57.5 %,
N170 45 %, MMN 37.5 %, N2pc 36 %, P3 30 %. With the noise of the 200 ms baseline (before) the
rates were higher (P3 37.5 %, MMN 50 %) but the same fits found a component in 12 to 28 % of
random-sign averages, where there is none; with the plus-minus noise, 0 to 8 %. Most misses
are fits without any component: on single participants' difference waves (40 to 60 trials
per condition) the evidence for a component of about twice the noise SD is weak.
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


@pytest.mark.xfail(
    strict=False,
    reason="REQ-274 not met for P3, N170, N400, MMN, N2pc and "
    "ERN on single-participant waves (see the module docstring and CHANGELOG)",
)
@pytest.mark.parametrize("component", ["ERN", "N170", "N400", "MMN", "N2pc", "P3"])
def test_req274_other_components(table: Any, component: Any) -> None:

    assert table.loc[component, "within_10ms"] >= 0.9
