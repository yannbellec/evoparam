# SPDX-License-Identifier: Apache-2.0
"""REQ-393 (spec §11.8): the figures are regenerated from the tag by one command, and
``make figures`` on the tag reproduces every figure file hash. The preprint that names the
tag waits for the owner's HIIE decision."""

import importlib.util
import json
import pathlib
import re
import subprocess

import pytest

import evoparam

ROOT = pathlib.Path(evoparam.__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "make_figures.py"

pytestmark = pytest.mark.skipif(not TOOL.exists(), reason="needs the source tree")


def _tool():
    spec = importlib.util.spec_from_file_location("make_figures", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_req393_one_command_makes_the_figures():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert re.search(r"^figures:\n\t@python tools/make_figures\.py$", makefile, re.MULTILINE)
    manifest = json.loads((ROOT / "figures" / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["figures"]) == set(_tool().FIGURES)
    for name in manifest["figures"]:
        assert (ROOT / "figures" / name).is_file(), name


def test_req393_figures_reproduce_their_hashes(tmp_path):
    """Every figure regenerated from this tree has the SHA-256 of the manifest (same
    matplotlib version: SVG output is byte-identical only within one version)."""

    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    tool = _tool()
    recorded = json.loads((ROOT / "figures" / "manifest.json").read_text(encoding="utf-8"))
    if matplotlib.__version__ != recorded["matplotlib"]:
        pytest.skip(f"manifest written with matplotlib {recorded['matplotlib']}")
    made = tool.make(tmp_path)
    assert made["figures"] == recorded["figures"]


def _tags():
    done = subprocess.run(
        ["git", "-C", str(ROOT), "tag", "--list", "v*"],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout.split()


@pytest.mark.xfail(
    strict=True,
    reason="REQ-393: the immutable, archived release tag that the "
    "preprint names, and the preprint itself, wait for the owner's HIIE "
    "decision (nothing public before it)",
)
def test_req393_make_figures_on_the_release_tag():
    tags = _tags()
    assert tags, "no release tag"
    manifest = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{tags[-1]}:figures/manifest.json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert manifest.returncode == 0
