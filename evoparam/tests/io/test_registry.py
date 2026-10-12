# SPDX-License-Identifier: Apache-2.0
"""Dataset registry (REQ-023)."""

import importlib
import json
from importlib import resources

import pytest

from evoparam.errors import ReaderError
from evoparam.io import dataset_info, list_datasets
from evoparam.io.registry import EVENTS_STATUS, match_dataset

###################################################################################################
###################################################################################################

SEEDED = (
    "ds004080",
    "ds003708",
    "ds004457",
    "ds004696",
    "ds004977",
    "ds006254",
    "ds002094",
    "ds005691",
    "ds006519",
    "erp-core",
    "localize-mi",
    "bigp3bci",
    "allen-visual-coding-neuropixels",
    "f-tract",
)

KEYS = {
    "id",
    "aliases",
    "title",
    "repository",
    "url",
    "doi",
    "version",
    "publications",
    "modality",
    "recording",
    "species",
    "sfreq",
    "n_subjects",
    "size_gb",
    "license",
    "events_status",
    "format",
    "loader",
    "quirks",
    "references",
    "files",
}


def registry():
    text = resources.files("evoparam.io").joinpath("registry.json").read_text(encoding="utf-8")
    return json.loads(text)


def test_registry_file_is_versioned_and_complete():

    content = registry()
    assert content["schema"] == "evoparam-registry/1" and content["version"]
    ids = [entry["id"] for entry in content["datasets"]]
    assert len(ids) == len(set(ids))
    assert set(SEEDED) <= set(ids)
    for entry in content["datasets"]:
        assert set(entry) == KEYS, entry["id"]
        assert entry["events_status"] in EVENTS_STATUS
        assert entry["license"] is None or isinstance(entry["license"], str)
        assert entry["size_gb"] is None or entry["size_gb"] > 0
        assert entry["sfreq"] is None or entry["sfreq"] > 0
        assert entry["references"], entry["id"]


def test_every_loader_entry_point_resolves():

    for entry in registry()["datasets"]:
        if entry["loader"] is None:
            continue
        module, _, name = entry["loader"].rpartition(".")
        assert callable(getattr(importlib.import_module(module), name)), entry["loader"]


def test_dataset_info():

    info = dataset_info("DS004080")
    assert info["sfreq"] == 2048.0 and info["n_subjects"] == 74 and info["size_gb"] == 269.0
    assert info["license"] == "CC0" and info["events_status"] == "present"
    assert info["doi"] == "10.18112/openneuro.ds004080.v1.2.4"
    tep = dataset_info("ds002094")  # read from its files (REQ-275)
    assert tep["license"] == "CC0" and tep["sfreq"] == 5000.0 and tep["n_subjects"] == 20
    assert dataset_info("erpcore")["id"] == "erp-core"

    # A copy: changing it does not change the registry
    info["sfreq"] = 1.0
    assert dataset_info("ds004080")["sfreq"] == 2048.0

    with pytest.raises(ReaderError, match="not in the registry"):
        dataset_info("ds999999")


def test_list_datasets():

    pytest.importorskip("pandas")
    table = list_datasets()
    assert len(table) >= len(SEEDED)
    for column in ("id", "license", "size_gb", "sfreq", "events_status", "loader"):
        assert column in table.columns
    assert table.attrs["registry_version"] == registry()["version"]

    spes = list_datasets(modality="spes", species="human")
    assert "ds004080" in set(spes["id"]) and set(spes["modality"]) == {"spes"}
    assert set(list_datasets(modality=["erp", "tep"])["modality"]) == {"erp", "tep"}
    small = list_datasets(max_size_gb=300)
    assert list(small["id"]) == ["ds004080", "ds002094"]  # the datasets with a stated size
    assert list(list_datasets(max_size_gb=100)["id"]) == ["ds002094"]
    assert len(list_datasets(max_size_gb=1)) == 0


def test_match_dataset():

    assert match_dataset("https://s3.amazonaws.com/openneuro.org/ds004080/")["id"] == "ds004080"
    assert match_dataset("/data/ds003708")["id"] == "ds003708"
    assert match_dataset("/data/other") is None
