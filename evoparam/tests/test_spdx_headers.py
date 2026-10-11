# SPDX-License-Identifier: Apache-2.0
"""REQ-396: provenance map. SPDX header in every source file, modification notice on the
inherited files, NOTICE, THIRD_PARTY.md and REIMPLEMENTED.md (spec §13.7).

Source tree only (the tools and the repository files are not in the wheel).
"""

import ast
import importlib.util
import pathlib
import re
import subprocess
import tarfile

import pytest

import evoparam

ROOT = pathlib.Path(evoparam.__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "add_spdx_headers.py"
FIRST_COMMIT = "e6def07"            # the copy of the inherited project, renamed (NOTICE)

pytestmark = [pytest.mark.core,
              pytest.mark.skipif(not TOOL.exists(), reason="needs the source tree")]


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_req396_every_python_file_has_the_spdx_header(capsys):
    tool = _tool("add_spdx_headers")
    files = tool.source_files()
    assert len(files) > 400
    bad = {p.relative_to(ROOT).as_posix(): tool.problems(p) for p in files}
    assert {k: v for k, v in bad.items() if v} == {}
    assert tool.main(["--check"]) == 0


def test_req396_check_fails_on_a_file_without_the_header(tmp_path):
    tool = _tool("add_spdx_headers")
    (tmp_path / "pkg").mkdir()
    bare = tmp_path / "pkg" / "bare.py"
    bare.write_text('"""A module."""\n', encoding="utf-8")
    assert tool.problems(bare, root=tmp_path) == ["no SPDX header on its first line"]
    assert tool.add(bare, root=tmp_path)
    assert tool.problems(bare, root=tmp_path) == []
    assert bare.read_text(encoding="utf-8").splitlines()[0] == tool.SPDX
    script = tmp_path / "pkg" / "script.py"
    script.write_text("#!/usr/bin/env python\nprint(1)\n", encoding="utf-8")
    tool.add(script, root=tmp_path)
    assert script.read_text(encoding="utf-8").splitlines()[:2] == ["#!/usr/bin/env python",
                                                                    tool.SPDX]


def test_req396_inherited_files_carry_the_modification_notice():
    tool = _tool("add_spdx_headers")
    assert tool.INHERITED
    for rel, changes in tool.INHERITED.items():
        lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
        assert lines[0] == tool.SPDX, rel
        assert lines[1].startswith("# Modified from the upstream project and commit named in "
                                   "NOTICE (Apache-2.0); changes: "), rel
        assert changes.split()[0] in " ".join(lines[1:6]), rel
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert re.search(r"commit [0-9a-f]{40}", notice) and "Apache License, Version 2.0" in notice


def test_req396_files_derived_from_third_party_code_name_it_in_notice():
    tool = _tool("add_spdx_headers")
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    for rel, licence in tool.DERIVED.items():
        first = (ROOT / rel).read_text(encoding="utf-8").splitlines()[0]
        assert first == f"{tool.SPDX} AND {licence}", rel
        assert rel in notice, rel
    assert "Redistribution and use in source and binary forms" in notice


def _inherited_export(tmp_path):
    archive = tmp_path / "first.tar"
    done = subprocess.run(["git", "-C", str(ROOT), "archive", "-o", str(archive),
                           FIRST_COMMIT], capture_output=True, text=True)
    if done.returncode != 0:
        pytest.skip(f"needs the git history ({done.stderr.strip()[:80]})")
    with tarfile.open(archive) as tar:
        tar.extractall(tmp_path / "first", filter="data")
    return tmp_path / "first"


def test_req396_every_file_with_inherited_code_is_marked(tmp_path):
    """CI fails on an inherited file without the notice: every file that shares a window of
    8 normalised lines with the first commit (the inherited copy) is in INHERITED."""

    checker = _tool("check_no_inherited_code")
    inherited = _inherited_export(tmp_path)
    matches = checker.check([str(ROOT / "evoparam"), str(ROOT / "tools"),
                             str(ROOT / "benchmarks"), str(ROOT / "validation"),
                             str(ROOT / "examples")], str(inherited))
    files = {pathlib.Path(m[0]).resolve().relative_to(ROOT).as_posix() for m in matches}
    assert files, "the check finds the inherited module utilities"
    assert sorted(files - set(_tool("add_spdx_headers").INHERITED)) == []


def _implemented_from_a_paper():
    out = []
    for path in sorted((ROOT / "evoparam").rglob("*.py")):
        if "tests" in path.parts or path.name == "__init__.py":    # packages only summarise
            continue
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
        if re.search(r"\b(Re-)?[Ii]mplemented from\b|\bSpecified only from\b", doc):
            out.append(path.relative_to(ROOT / "evoparam").as_posix())
    return out


def test_req396_reimplemented_lists_every_algorithm_written_from_a_paper():
    text = (ROOT / "REIMPLEMENTED.md").read_text(encoding="utf-8")
    assert "No code was copied" in text
    modules = _implemented_from_a_paper()
    assert len(modules) >= 7
    assert [m for m in modules if f"`{m}`" not in text] == []
    table_rows = [line for line in text.splitlines() if line.startswith("| ")
                  and not line.startswith("| method")]
    for row in table_rows:                               # paper and patent column filled
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert len(cells) == 4 and all(cells), row
    assert "US 2022/0338786 A1" in text and "US 2023/0240618 A1" in text
    period = next(r for r in table_rows if "artifact/period.py" in r)
    pursuit = next(r for r in table_rows if "artifact/pursuit.py" in r)
    assert "US 2022/0338786 A1" in period and "US 2023/0240618 A1" in pursuit
    bib = (ROOT / "docs" / "references.bib").read_text(encoding="utf-8")
    for key in set(re.findall(r"LIT-[a-z]+-\d+", text)):
        assert f"{{{key}," in bib, key


def test_req396_third_party_and_notice_exist():
    for name in ("NOTICE", "THIRD_PARTY.md", "REIMPLEMENTED.md", "LICENSE"):
        assert (ROOT / name).is_file(), name
    third = (ROOT / "THIRD_PARTY.md").read_text(encoding="utf-8")
    for package in ("numpy", "scipy", "mne", "pandas", "matplotlib", "pynwb"):
        assert f"| {package} |" in third, package
