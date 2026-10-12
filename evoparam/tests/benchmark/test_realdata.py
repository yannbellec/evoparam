# SPDX-License-Identifier: Apache-2.0
"""Real-data suites (spec §8.2): index files and loaders only, no copy of the data.

The indices are checked offline; the loaders read public data and run only with
EVOPARAM_NETWORK_TESTS=1.
"""

import numpy as np
import pytest

from evoparam.benchmark import TASKS, list_indices, load_index, real_inputs
from evoparam.benchmark.realdata import REALDATA_SCHEMA
from evoparam.benchmark.suite import ERPCORE
from evoparam.errors import DataError
from evoparam.io.registry import dataset_info

from .helpers import NETWORK, ROOT

pytest.importorskip("jsonschema")


def test_indices_name_registered_datasets():
    assert list_indices() == ["ds002094", "ds003708", "ds004080", "ds004977", "erpcore"]
    for name in list_indices():
        index = load_index(name)
        assert index["schema"] == REALDATA_SCHEMA
        info = dataset_info(index["dataset"])
        assert info["id"] in (index["dataset"], name) or index["dataset"] in info["aliases"]
        assert index["licence"] and index["redistribution"]
        assert index["tasks"] and set(index["tasks"]) <= set(TASKS)
        if name != "ds002094":
            assert index["entries"]


def test_indices_hold_no_data():
    folder = ROOT / "evoparam" / "benchmark" / "realdata"
    files = sorted(p.name for p in folder.iterdir() if p.is_file())
    assert all(f.endswith(".json") for f in files)
    assert sum((folder / f).stat().st_size for f in files) < 200_000


def test_ds004080_entries_follow_the_fixed_selection():
    index = load_index("ds004080")
    lines = [
        line.split("\t")
        for line in (ROOT / "validation" / "ds004080_subjects.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line and not line.startswith("#")
    ]
    expected = [(s.replace("sub-", ""), r.rsplit("run-", 1)[1]) for s, r in lines]
    assert [(e["subject"], e["run"]) for e in index["entries"]] == expected
    assert index["truth"]["min_agree"] == 2
    assert set(index["truth"]["rules"]) <= {
        "vanblooijs2023",
        "hays2021",
        "keller2014",
        "silverstein2020",
        "ftract",
        "prime2020",
    }


def test_erpcore_index_uses_the_reference_definitions():
    index = load_index("erpcore")
    assert index["tasks"] == ["T6"]
    assert {e["component"] for e in index["entries"]} == set(ERPCORE["components"])
    assert len(index["entries"]) == 5 * 40
    assert "referenced only" in index["redistribution"]


def test_unknown_index_and_task():
    with pytest.raises(DataError, match="Unknown real-data index"):
        load_index("ds999999")
    with pytest.raises(DataError, match="does not serve task T6"):
        real_inputs("ds004080", 0, task="T6")


def _network(func):
    for mark in reversed(NETWORK):
        func = mark(func)
    return func


@_network
def test_bids_loader_reads_one_entry(tmp_path):
    out = real_inputs("ds004080", 0, task="T1", cache_dir=str(tmp_path))
    assert out
    for ti in out:
        assert ti.descriptor["split"] == "real"
        assert ti.descriptor["source"]["dataset"] == "ds004080"
        assert not set(ti.descriptor["site"].split("-")) & set(ti.descriptor["ch_names"])
        assert np.all(np.isfinite(ti.arrays["data"]))
        truth = ti.truth
        assert not set(truth["responders"]) & set(truth["excluded"])
        assert set(truth["n1"]) <= set(truth["responders"])


@_network
def test_erpcore_loader_reads_one_entry():
    pytest.importorskip("mne")
    ti = real_inputs("erpcore", 0, task="T6")[0]
    entry = load_index("erpcore")["entries"][0]
    assert ti.descriptor["definitions"][0]["condition"] == entry["component"]
    np.testing.assert_allclose(ti.arrays["average"], ti.arrays["data"].mean(axis=0), rtol=1e-9)
