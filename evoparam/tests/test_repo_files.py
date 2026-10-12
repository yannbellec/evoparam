# SPDX-License-Identifier: Apache-2.0
"""REQ-327: community files, their policies and links (source tree only)."""

import os

import pytest

import evoparam

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(evoparam.__file__)))
FILES = ("CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "GOVERNANCE.md", "SECURITY.md", "SUPPORT.md")
TEMPLATES = ("bug.yml", "feature.yml", "new_preset.yml", "new_dataset.yml", "config.yml")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(ROOT, "pyproject.toml")), reason="needs the source tree"
)


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fobj:
        return fobj.read()


def test_files_exist_and_are_linked():

    for name in FILES:
        assert os.path.isfile(os.path.join(ROOT, name)), name
    readme = _read("README.rst")
    for name in FILES:
        assert name in readme, f"README does not link {name}"
    assert os.path.isfile(os.path.join(ROOT, "docs", "contributing", "add_shape.rst"))


def test_policies():

    contributing = _read("CONTRIBUTING.md")
    assert "Developer Certificate of Origin" in contributing
    assert "Signed-off-by" in contributing
    for topic in ("evoparam.shapes", "evoparam.presets", "evoparam.readers", "dataset"):
        assert topic in contributing
    governance = _read("GOVERNANCE.md")
    assert "steering group of three" in governance
    assert "lazy consensus" in governance
    assert "7-day" in governance
    assert "Contributor Covenant, version 2.1" in _read("CODE_OF_CONDUCT.md")


def test_issue_templates():

    folder = os.path.join(ROOT, ".github", "ISSUE_TEMPLATE")
    assert sorted(os.listdir(folder)) == sorted(TEMPLATES)
    assert "evoparam info" in _read(".github", "ISSUE_TEMPLATE", "bug.yml")
    preset = _read(".github", "ISSUE_TEMPLATE", "new_preset.yml")
    assert "Literature keys" in preset
    assert "Test dataset" in preset
    assert "discussions" in _read(".github", "ISSUE_TEMPLATE", "config.yml")
