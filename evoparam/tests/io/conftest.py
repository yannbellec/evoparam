# SPDX-License-Identifier: Apache-2.0
"""Fixtures of the io tests: a mini BIDS dataset and an isolated download folder."""

from typing import Any

import pytest

from evoparam.tests.io.helpers import make_bids

###################################################################################################
###################################################################################################


@pytest.fixture(scope="session")
def bids_root(tmp_path_factory: Any) -> Any:
    """Mini BIDS-iEEG dataset (2 subjects, 3 channels, 20 events per run), read only."""

    return make_bids(tmp_path_factory.mktemp("bids") / "mini")


@pytest.fixture(autouse=True)
def isolated_data_folder(tmp_path: Any, monkeypatch: Any) -> None:
    """Downloads go to a temporary folder; never to the user's cache."""

    monkeypatch.setenv("EVOPARAM_DATA", str(tmp_path / "evoparam-data"))
    monkeypatch.delenv("EVOPARAM_OFFLINE", raising=False)
