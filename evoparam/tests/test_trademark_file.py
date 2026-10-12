# SPDX-License-Identifier: Apache-2.0
"""REQ-397: TRADEMARK.md states who owns the name (per the HIIE decision), what forks may be
called, how plugins and products may reference the name, and that the licence grants no
trademark rights (Apache-2.0 §6); it is linked from the README (spec §13.7)."""

import pathlib
import re

import pytest

import evoparam

ROOT = pathlib.Path(evoparam.__file__).resolve().parent.parent
TRADEMARK = ROOT / "TRADEMARK.md"

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not (ROOT / "pyproject.toml").exists(), reason="needs the source tree"),
]


def test_req397_trademark_file_states_the_policy() -> None:
    text = TRADEMARK.read_text(encoding="utf-8")
    for heading in (
        "Who owns the name",
        "The licence grants no trademark rights",
        "Forks",
        "Plugins, extensions and products",
    ):
        assert f"## {heading}" in text, heading
    assert "HIIE" in text
    assert "GOVERNANCE.md" in text
    assert "Section 6" in text
    assert "Apache License, Version 2.0" in text
    assert "for evoparam" in text
    assert "built on evoparam" in text
    forks = text.split("## Forks", 1)[1].split("## ", 1)[0]
    assert "different name" in forks
    assert "NOTICE" in forks


def test_req397_linked_from_the_readme() -> None:
    readme = (ROOT / "README.rst").read_text(encoding="utf-8")
    assert "`TRADEMARK.md <TRADEMARK.md>`_" in readme


def test_req397_reviewed_at_each_major_release() -> None:
    assert re.search(
        r"major release.*TRADEMARK\.md.*reviewed",
        (ROOT / "RELEASING.md").read_text(encoding="utf-8"),
    )


@pytest.mark.xfail(
    strict=True,
    reason="REQ-397: the owner of the name is set by the HIIE "
    "decision; waits for the owner's HIIE decision (nothing public before it)",
)
def test_req397_owner_named_from_the_recorded_decision() -> None:
    text = TRADEMARK.read_text(encoding="utf-8")
    owner = text.split("## Who owns the name", 1)[1].split("## ", 1)[0]
    assert re.search(r"(?m)^The name is owned by \S", owner)
