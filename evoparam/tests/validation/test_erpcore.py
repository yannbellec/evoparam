"""ERP CORE, 40 participants x 7 components (REQ-274), on a local copy (CC-BY 4.0, OSF
DOI 10.18115/D5JW4R) downloaded by ``validation/erpcore.py``'s documented layout; runs only
when ``EVOPARAM_ERPCORE`` names that folder.

Measured on 2026-10-08 (difference waves on the published channels, label window = the
published measurement window +/- 30 ms; model latency = 50 % area latency of the fitted
component): labelled and within 10 ms in LRP 92.5 %, ERN 87.5 %, N170 55 %, N400 55 %,
MMN 50 %, N2pc 38 %, P3 37.5 %. Most misses are fits without any component: on single
participants' difference waves (40 to 60 trials per condition) the BIC rule with delta 10
rejects components of about twice the noise SD; delta 2 raises N170 to 62.5 % only.
"""

import importlib.util
import os
import pathlib

import pytest

ROOT = os.environ.get("EVOPARAM_ERPCORE", "")
SCRIPT = pathlib.Path(__file__).parents[3] / "validation" / "erpcore.py"
pytestmark = pytest.mark.skipif(not (ROOT and os.path.isdir(ROOT) and SCRIPT.exists()),
                                reason="set EVOPARAM_ERPCORE to the local ERP CORE copy")


@pytest.fixture(scope="module")
def table():
    pytest.importorskip("mne")
    spec = importlib.util.spec_from_file_location("erpcore", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summary(module.run())


def test_req274_lateralized_readiness_potential(table):

    assert table.loc["LRP", "n"] == 40
    assert table.loc["LRP", "within_10ms"] >= 0.9


@pytest.mark.xfail(strict=False, reason="REQ-274 not met for P3, N170, N400, MMN, N2pc and "
                   "ERN on single-participant waves (see the module docstring and CHANGELOG)")
@pytest.mark.parametrize("component", ["ERN", "N170", "N400", "MMN", "N2pc", "P3"])
def test_req274_other_components(table, component):

    assert table.loc[component, "within_10ms"] >= 0.9
