# SPDX-License-Identifier: Apache-2.0
"""REQ-329: the 1.0 release checklist; REQ-312: the inherited-code check."""

import os
import subprocess
import sys
from typing import Any

import pytest

import evoparam

# Links of the 1.0 checklist.
MIN_LINKS = 3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(evoparam.__file__)))
pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(ROOT, "RELEASING.md")), reason="needs the source tree"
)


def test_req329_checklist_has_the_three_items() -> None:

    with open(os.path.join(ROOT, "RELEASING.md"), encoding="utf-8") as fobj:
        text = fobj.read()
    section = text.split("## 1.0 checklist", 1)[1].split("\n## ", 1)[0]
    for item in ("MNE example", "DANDI / NWB listing", "conda-forge feedstock"):
        assert f"**{item}**" in section, item
    assert section.count("Link") >= MIN_LINKS
    for target in ("MNE-BIDS", "SpikeInterface", "pynapple", "Elephant / Neo", "MOABB", "MATLAB"):
        assert target in text, target


def test_req312_inherited_code_is_found(tmp_path: Any) -> None:

    inherited = tmp_path / "inherited"
    inherited.mkdir()
    original = (
        "\n".join(f"def f{i}(x):\n    return x * {i}  # original line {i}" for i in range(10))
        + "\n"
    )
    (inherited / "module.py").write_text(original)
    fresh = tmp_path / "fresh.py"
    fresh.write_text("\n".join(f"value_{i} = {i}" for i in range(20)) + "\n")
    copied = tmp_path / "copied.py"
    # the same code, re-indented with tabs and spaces and blank lines added
    copied.write_text(
        "\n\n".join(line.replace("    ", "\t  ") for line in original.splitlines()) + "\n"
    )
    tool = os.path.join(ROOT, "tools", "check_no_inherited_code.py")
    ok = subprocess.run(
        [sys.executable, tool, str(fresh), "--inherited", str(inherited)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0
    assert "no inherited code" in ok.stdout
    bad = subprocess.run(
        [sys.executable, tool, str(copied), "--inherited", str(inherited)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode == 1
    assert "module.py" in bad.stdout


def test_req312_new_modules_hold_no_inherited_code() -> None:
    """Modules written for build step S17 against the inherited ERPparam (needs the checkout:
    EVOPARAM_INHERITED=<path to ERPparam at the NOTICE commit>)."""

    inherited = os.environ.get("EVOPARAM_INHERITED")
    if not inherited:
        pytest.skip("EVOPARAM_INHERITED not set")
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    try:
        import check_no_inherited_code as tool
    finally:
        sys.path.pop(0)
    sources = [
        os.path.join(ROOT, "evoparam", p)
        for p in (
            "io/_validate.py",
            "_plugins.py",
            "io/ndx_evoparam",
            "io/derivatives.py",
            "utils/_estimator.py",
            "trials/estimator.py",
            "artifact/estimator.py",
        )
    ]
    assert tool.check(sources, inherited) == []
