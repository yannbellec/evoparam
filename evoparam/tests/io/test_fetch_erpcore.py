# SPDX-License-Identifier: Apache-2.0
"""``fetch_erpcore`` against a local copy of the OSF API (no Internet): the listing pages
are followed, the files downloaded and checked against the SHA-256 published by OSF or
recorded in the dataset registry, and a file that does not match is deleted."""

import hashlib
import json
import pathlib
from typing import Any

import pytest

from evoparam.errors import ReaderError
from evoparam.io import _fetch, fetch
from evoparam.tests.io.helpers import http_server

SET, FDT = b"set file", b"fdt file"


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _osf(root: pathlib.Path, base: str, participant: str = "1", fdt: bool = True) -> None:
    """Pages of a fake OSF project for ERP CORE P3: node, data folder, participants, files."""
    node = root / "v2" / "nodes" / fetch.ERPCORE_NODES["P3"] / "files"
    node.mkdir(parents=True)
    folder = {"href": base + "subjects1.json"}
    (node / "osfstorage.json").write_text(
        json.dumps(
            {
                "data": [
                    {"attributes": {"name": "P3 Raw Data"}},
                    {
                        "attributes": {"name": "P3 All Data and Scripts"},
                        "relationships": {"files": {"links": {"related": folder}}},
                    },
                ],
                "links": {"next": None},
            }
        ),
        encoding="utf-8",
    )
    # two pages of participants (the second through links.next)
    (root / "subjects1.json").write_text(
        json.dumps(
            {
                "data": [{"attributes": {"name": "40"}}],
                "links": {"next": base + "subjects2.json"},
            }
        ),
        encoding="utf-8",
    )
    files = {"href": base + "files.json"}
    (root / "subjects2.json").write_text(
        json.dumps(
            {
                "data": [
                    {
                        "attributes": {"name": participant},
                        "relationships": {"files": {"links": {"related": files}}},
                    }
                ],
                "links": {},
            }
        ),
        encoding="utf-8",
    )
    items = [
        {
            "attributes": {
                "name": "1_P3_epoch_interp_ar.set",
                "extra": {"hashes": {"sha256": _sha(SET)}},
            },
            "links": {"download": base + "data/1.set"},
        },
        {
            "attributes": {"name": "1_P3_shifted_ds.set", "extra": {}},
            "links": {"download": base + "data/other.set"},
        },
    ]
    if fdt:
        items.append(
            {
                "attributes": {
                    "name": "1_P3_epoch_interp_ar.fdt",
                    "extra": {"hashes": {"sha256": _sha(FDT)}},
                },
                "links": {"download": base + "data/1.fdt"},
            }
        )
    (root / "files.json").write_text(json.dumps({"data": items, "links": {}}), encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "1.set").write_bytes(SET)
    (root / "data" / "1.fdt").write_bytes(FDT)


@pytest.fixture
def osf(tmp_path: pathlib.Path, monkeypatch: Any) -> Any:
    """Serve a fake OSF API; the fixed address of the API is mapped onto it."""
    served = tmp_path / "served"
    served.mkdir()
    real = _fetch.fetch_bytes
    with http_server(served) as (base, stats):

        def local(url: str, **kwargs: Any) -> Any:
            api = "https://api.osf.io/"
            if url.startswith(api):  # the first page: a static file of the local server
                url = base + url[len(api) :].rstrip("/") + ".json"
            return real(url, **kwargs)

        monkeypatch.setattr(_fetch, "fetch_bytes", local)
        monkeypatch.setattr(fetch, "_registry_files", lambda _dataset: {})
        monkeypatch.delenv("EVOPARAM_OFFLINE", raising=False)
        yield served, base, stats


def test_fetch_erpcore_downloads_and_checks(osf: Any, tmp_path: pathlib.Path) -> None:
    served, base, stats = osf
    _osf(served, base)
    path = fetch.fetch_erpcore("P3", 1, folder=str(tmp_path / "erpcore"))
    target = tmp_path / "erpcore" / "P3" / "1"
    assert path == str(target / "1_P3_epoch_interp_ar.set")
    assert (target / "1_P3_epoch_interp_ar.set").read_bytes() == SET
    assert (target / "1_P3_epoch_interp_ar.fdt").read_bytes() == FDT
    assert not (target / "1_P3_shifted_ds.set").exists()  # not one of the epoched files
    # a second call finds the files and checks them again against OSF without downloading
    before = len(stats.requests)
    assert fetch.fetch_erpcore("P3", 1, folder=str(tmp_path / "erpcore")) == path
    assert len(stats.requests) > before  # the listing is read again (no registry hashes)
    assert not any("/data/" in str(request[1]) for request in stats.requests[before:])


def test_fetch_erpcore_deletes_a_file_that_does_not_match(
    osf: Any, tmp_path: pathlib.Path
) -> None:
    served, base, _ = osf
    _osf(served, base)
    (served / "data" / "1.set").write_bytes(b"corrupted")
    with pytest.raises(ReaderError, match="does not match the expected"):
        fetch.fetch_erpcore("P3", 1, folder=str(tmp_path / "erpcore"))
    assert not (tmp_path / "erpcore" / "P3" / "1" / "1_P3_epoch_interp_ar.set").exists()


def test_fetch_erpcore_missing_participant_or_file(osf: Any, tmp_path: pathlib.Path) -> None:
    served, base, _ = osf
    _osf(served, base, participant="2", fdt=False)
    with pytest.raises(ReaderError, match="no participant 1 on OSF"):
        fetch.fetch_erpcore("P3", 1, folder=str(tmp_path / "a"))
    (served / "subjects2.json").write_text(
        (served / "subjects2.json").read_text(encoding="utf-8").replace('"2"', '"1"'),
        encoding="utf-8",
    )
    with pytest.raises(ReaderError, match=r"no \*_epoch_interp_ar.fdt on OSF"):
        fetch.fetch_erpcore("P3", 1, folder=str(tmp_path / "b"))


def test_fetch_erpcore_from_the_registry_without_connecting(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    target = tmp_path / "P3" / "7"
    target.mkdir(parents=True)
    (target / "7_P3_epoch_interp_ar.set").write_bytes(SET)
    (target / "7_P3_epoch_interp_ar.fdt").write_bytes(FDT)
    registry = {
        "P3/7/7_P3_epoch_interp_ar.set": _sha(SET),
        "P3/7/7_P3_epoch_interp_ar.fdt": _sha(FDT),
        "N170/7/7_N170_epoch_interp_ar.set": "0" * 64,
    }
    monkeypatch.setattr(fetch, "_registry_files", lambda _dataset: registry)
    monkeypatch.setenv("EVOPARAM_OFFLINE", "1")  # any connection would raise
    path = fetch.fetch_erpcore("P3", 7, folder=str(tmp_path))
    assert path == str(target / "7_P3_epoch_interp_ar.set")
    monkeypatch.setenv("EVOPARAM_ERPCORE", str(tmp_path))
    assert fetch.fetch_erpcore("P3", 7) == path
    registry["P3/7/7_P3_epoch_interp_ar.fdt"] = "f" * 64
    with pytest.raises(ReaderError, match="does not match the expected"):
        fetch.fetch_erpcore("P3", 7, folder=str(tmp_path))
    assert not (target / "7_P3_epoch_interp_ar.fdt").exists()


def test_fetch_erpcore_unknown_component() -> None:
    with pytest.raises(ReaderError, match="ERP CORE component must be one of"):
        fetch.fetch_erpcore("P300")


def test_registry_files() -> None:
    from evoparam.io.registry import _load_registry

    entries = _load_registry()["datasets"]
    first = entries[0]
    assert fetch._registry_files(first["id"].upper()) == (first.get("files") or {})
    assert fetch._registry_files("no-such-dataset") is None
