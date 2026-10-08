"""Schema versioning and migration of the ``.evoparam`` container (REQ-334).

``containers/`` keeps one frozen container per released schema major. Each must load and
equal the object that :func:`expected` builds. When a new major is released, freeze a
container of it next to the older ones (run this file as a script) and register the
migration of the older ones in ``evoparam/data/_migrations.py``.
"""

import dataclasses
import json
import os
import shutil

import numpy as np
import pytest

from evoparam.data import Provenance, load_object, save_object
from evoparam.data import _migrations
from evoparam.data.serialize import SCHEMA, SCHEMA_MAJOR
from evoparam.errors import EvoparamVersionError
from evoparam.reports import migrate
from evoparam.tests.data.factories import make_group_fit, make_trialset

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "containers")
FROZEN = {1: {"groupfit": "groupfit-1.evoparam", "trialset": "trialset-1.evoparam"}}


def _provenance():
    return Provenance(package_version="0.1.0.dev0", created="2026-01-01T00:00:00+00:00",
                      source={"path": "synthetic", "format": "array"}, steps=[],
                      environment={}, random_state=0, data_hash=None)


def expected(kind):
    """The object frozen in the container of ``kind``, with a fixed provenance."""

    if kind == "trialset":
        return dataclasses.replace(make_trialset(), provenance=_provenance())
    group = make_group_fit()
    fits = []
    for fit in group.fits:
        test = fit.existence
        if test is not None:
            test = dataclasses.replace(test, provenance=_provenance())
        fits.append(dataclasses.replace(fit, provenance=_provenance(), existence=test))
    return dataclasses.replace(group, fits=fits, provenance=_provenance())


@pytest.mark.parametrize("major", sorted(FROZEN))
@pytest.mark.parametrize("kind", ["groupfit", "trialset"])
def test_req334_frozen_containers_load_and_equal(major, kind):

    path = os.path.join(FOLDER, FROZEN[major][kind])
    with open(os.path.join(path, "manifest.json"), encoding="utf-8") as fobj:
        assert json.load(fobj)["schema"].startswith(f"evoparam-result/{major}.")
    obj = load_object(path)
    reference = expected(kind)
    if kind == "trialset":
        # The factory's data differ from the frozen ones in the last bits from one CPU to
        # another, and the data hash with them: compare the values, then the rest exactly.
        np.testing.assert_allclose(obj.data, reference.data, rtol=1e-9, atol=0)
        reference = dataclasses.replace(reference, data=obj.data, provenance=_provenance())
    assert obj == reference
    assert obj.provenance == reference.provenance


def test_req334_every_released_major_is_frozen():

    assert SCHEMA_MAJOR in FROZEN
    for major in FROZEN:
        assert major == SCHEMA_MAJOR or (major, major + 1) in _migrations.MIGRATIONS


def test_req334_newer_major_raises(tmp_path):

    path = shutil.copytree(os.path.join(FOLDER, FROZEN[1]["trialset"]), tmp_path / "t.evoparam")
    manifest = os.path.join(path, "manifest.json")
    with open(manifest, encoding="utf-8") as fobj:
        data = json.load(fobj)
    data["schema"] = "evoparam-result/99"
    with open(manifest, "w", encoding="utf-8") as fobj:
        json.dump(data, fobj)
    with pytest.raises(EvoparamVersionError, match="99"):
        load_object(path)


def test_req334_older_major_is_migrated_and_recorded(tmp_path, monkeypatch):

    path = str(shutil.copytree(os.path.join(FOLDER, FROZEN[1]["groupfit"]),
                               tmp_path / "old.evoparam"))
    manifest = os.path.join(path, "manifest.json")
    with open(manifest, encoding="utf-8") as fobj:
        data = json.load(fobj)
    data["schema"] = f"evoparam-result/{SCHEMA_MAJOR - 1}.0"
    with open(manifest, "w", encoding="utf-8") as fobj:
        json.dump(data, fobj)
    calls = []
    monkeypatch.setitem(_migrations.MIGRATIONS, (SCHEMA_MAJOR - 1, SCHEMA_MAJOR),
                        lambda structure: calls.append(1) or structure)
    obj = load_object(path)
    assert calls == [1]
    step = obj.provenance.steps[-1]
    assert step["function"] == "evoparam.data.migrate"
    assert step["settings"] == {"from_schema": SCHEMA_MAJOR - 1, "to_schema": SCHEMA_MAJOR}
    report = migrate(path)
    assert report == {"path": path, "from_schema": f"evoparam-result/{SCHEMA_MAJOR - 1}.0",
                      "to_schema": SCHEMA, "migrated": True}
    with open(manifest, encoding="utf-8") as fobj:
        assert json.load(fobj)["schema"] == SCHEMA
    again = load_object(path)                          # verified checksums, current schema
    assert again == obj and again.provenance.steps[-1]["function"] == "evoparam.data.migrate"
    assert migrate(path)["migrated"] is False
    assert sorted(os.listdir(tmp_path)) == ["old.evoparam"]   # no temporary file left


def _freeze():
    """Write the frozen containers of the current major (run once per released major)."""

    os.makedirs(FOLDER, exist_ok=True)
    for kind, name in FROZEN[SCHEMA_MAJOR].items():
        target = os.path.join(FOLDER, name)
        if os.path.exists(target):
            shutil.rmtree(target)
        save_object(expected(kind), target, table_format="csv")


if __name__ == "__main__":
    _freeze()
