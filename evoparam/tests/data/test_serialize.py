# SPDX-License-Identifier: Apache-2.0
"""Tests for the .evoparam container (REQ-020 save/load round trip, REQ-330, REQ-332, REQ-334)."""

import json
import os
import zipfile
from typing import Any

import numpy as np
import pandas as pd
import pytest

from evoparam.data import GroupFit, TrialSet, load_object, save_object, serialize
from evoparam.errors import DataError, EvoparamVersionError
from evoparam.tests.data.factories import all_results, make_group_fit, make_trialset

RESULTS = all_results()
FORMATS = ["csv"] + (["parquet"] if serialize._have_pyarrow() else [])


def _strict_load(path: Any) -> Any:
    def refuse(name):
        raise ValueError(name)

    with open(path, encoding="utf-8") as fobj:
        return json.loads(fobj.read(), parse_constant=refuse)


@pytest.mark.parametrize("table_format", FORMATS)
@pytest.mark.parametrize("name", sorted(RESULTS))
def test_round_trip_every_type(tmp_path: Any, name: Any, table_format: Any) -> None:
    obj = RESULTS[name]
    path = save_object(obj, tmp_path / name, table_format=table_format)
    loaded = load_object(path)
    assert type(loaded) is type(obj)
    assert loaded == obj


def test_round_trip_zip(tmp_path: Any) -> None:
    obj = make_group_fit()
    path = save_object(obj, tmp_path / "group.zip")
    assert zipfile.is_zipfile(path)
    assert load_object(path) == obj


def test_trialset_data_exact_and_mmap(tmp_path: Any) -> None:
    trials = make_trialset(with_masks=True)
    path = trials.save(tmp_path / "trials")
    assert path.endswith(".evoparam")
    loaded = TrialSet.load(path, mmap=True)
    assert isinstance(loaded.data, np.memmap)
    assert not loaded.data.flags.writeable
    assert loaded == trials
    np.testing.assert_array_equal(loaded.data, trials.data)
    pd.testing.assert_frame_equal(loaded.metadata, trials.metadata)


def test_manifest_keys(tmp_path: Any) -> None:
    path = save_object(make_group_fit(), tmp_path / "g")
    manifest = _strict_load(os.path.join(path, "manifest.json"))
    assert set(manifest) == {
        "schema",
        "object",
        "package_version",
        "created",
        "table_format",
        "files",
    }
    assert manifest["schema"] == serialize.SCHEMA
    assert manifest["object"] == "GroupFit"
    for name in ("structure.json", "data.npz", "provenance.json", "settings.json"):
        assert name in manifest["files"]


def test_checksum_detects_one_changed_byte(tmp_path: Any) -> None:
    path = save_object(make_group_fit(), tmp_path / "g")
    target = os.path.join(path, "data.npz")
    with open(target, "r+b") as fobj:
        fobj.seek(os.path.getsize(target) // 2)
        byte = fobj.read(1)
        fobj.seek(-1, os.SEEK_CUR)
        fobj.write(bytes([byte[0] ^ 0xFF]))
    with pytest.raises(DataError, match="checksum"):
        load_object(path)


def test_json_valid_with_nan_and_inf(tmp_path: Any) -> None:
    fit = make_group_fit()
    fit.settings["nan_value"] = float("nan")
    fit.settings["inf_value"] = float("inf")
    path = save_object(fit, tmp_path / "g", table_format="csv")
    for root, _, names in os.walk(path):
        for name in names:
            if name.endswith(".json"):
                _strict_load(os.path.join(root, name))
    settings = _strict_load(os.path.join(path, "settings.json"))
    assert settings["nan_value"] is None
    assert settings["inf_value"] is None
    loaded = load_object(path)
    assert np.isnan(loaded.settings["nan_value"])
    assert np.isinf(loaded.settings["inf_value"])


def test_table_sidecar_units(tmp_path: Any) -> None:
    table = make_group_fit().to_dataframe("component")
    obj = make_trialset()
    obj.metadata.attrs["units"] = {"stim_current_mA": "mA"}
    path = save_object(obj, tmp_path / "t", table_format="csv")
    sidecar = _strict_load(os.path.join(path, "tables", "metadata.json"))
    assert sidecar["columns"]["stim_current_mA"]["unit"] == "mA"
    assert sidecar["columns"]["condition"]["unit"] == ""
    assert set(table.attrs["units"]) == set(table.columns)


def test_newer_major_raises(tmp_path: Any) -> None:
    path = save_object(make_trialset(), tmp_path / "t")
    manifest_path = os.path.join(path, "manifest.json")
    manifest = _strict_load(manifest_path)
    manifest["schema"] = "evoparam-result/99"
    with open(manifest_path, "w", encoding="utf-8") as fobj:
        json.dump(manifest, fobj)
    with pytest.raises(EvoparamVersionError, match="99"):
        load_object(path)


def test_older_major_without_migration_raises(tmp_path: Any) -> None:
    path = save_object(make_trialset(), tmp_path / "t")
    manifest_path = os.path.join(path, "manifest.json")
    manifest = _strict_load(manifest_path)
    manifest["schema"] = "evoparam-result/0.3"
    with open(manifest_path, "w", encoding="utf-8") as fobj:
        json.dump(manifest, fobj)
    with pytest.raises(EvoparamVersionError, match="migration"):
        load_object(path)


def test_no_overwrite_by_default(tmp_path: Any) -> None:
    path = save_object(make_trialset(), tmp_path / "t")
    with pytest.raises(FileExistsError):
        save_object(make_trialset(), path)
    replacement = make_trialset(seed=3)
    save_object(replacement, path, overwrite=True)
    assert load_object(path) == replacement


def test_light_save_drops_curves(tmp_path: Any) -> None:
    fit = make_group_fit()
    path = fit.save(tmp_path / "g", light=True)
    loaded = GroupFit.load(path)
    assert all(f.model is None and f.residual is None for f in loaded.fits)
    assert loaded.fits[0].components == fit.fits[0].components


def test_load_checks_type(tmp_path: Any) -> None:
    path = save_object(make_group_fit(), tmp_path / "g")
    with pytest.raises(DataError, match="not a TrialSet"):
        TrialSet.load(path)


def test_non_container_raises(tmp_path: Any) -> None:
    with pytest.raises(DataError):
        load_object(tmp_path)


def test_object_arrays_refused(tmp_path: Any) -> None:
    fit = make_group_fit()
    fit.settings["bad"] = np.array([object()], dtype=object)
    with pytest.raises(DataError, match="dtype"):
        save_object(fit, tmp_path / "g")
