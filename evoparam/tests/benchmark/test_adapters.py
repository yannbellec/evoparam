# SPDX-License-Identifier: Apache-2.0
"""REQ-354: the adapter mechanism runs an external command-line tool on task files without
importing it; the adapters for GPL tools live outside the Apache-licensed package tree
(checked by a licence-scan test)."""

import ast
import json
import os
import re
import sys

import pytest

from evoparam.benchmark import (Adapter, AdapterError, BenchmarkValidationError, load_adapter,
                                read_output, run_suite, write_input)
from evoparam.errors import DataError
from evoparam.tests.test_dependencies import GPL_TEXT, _mentions_isolation

from .helpers import ADAPTERS, ROOT, inputs

pytest.importorskip("jsonschema")

FAKE_TOOL = '''
import json, os, sys
input_path, output_path = sys.argv[1], sys.argv[2]
with open(input_path) as fobj:
    d = json.load(fobj)
mode = os.environ.get("FAKE_MODE", "ok")
with open(output_path + ".pid", "w") as fobj:
    fobj.write(str(os.getpid()))
if mode == "fail":
    sys.stderr.write("fake tool: cannot read the recording")
    sys.exit(3)
if mode == "sleep":
    import time
    time.sleep(30)
if mode == "silent":
    sys.exit(0)
with open(output_path, "w") as out:
    out.write("site\\tchannel\\tresponsive\\tp_value\\n")
    stim = set(d["stim_channels"])
    for k, c in enumerate(d["ch_names"]):
        if c in stim:
            continue
        p = "2.0" if mode == "invalid" else repr(0.5 / (k + 1))
        out.write(f"{d['site']}\\t{c}\\t{int(k % 2 == 0)}\\t{p}\\n")
'''


@pytest.fixture
def fake(tmp_path):
    tool = tmp_path / "tool" / "fake_detector.py"
    tool.parent.mkdir()
    tool.write_text(FAKE_TOOL, encoding="utf-8")
    ti = inputs("T1", 1)[0]
    path = write_input(ti, str(tmp_path / "inputs" / "T1"))
    return tool, path, ti


def _adapter(tool, **kw):
    return Adapter(name="fake", command=["{python}", str(tool)], tasks=("T1",), version="0.1",
                   citation="a fake detector", **kw)


def test_req354_adapter_runs_the_tool_in_a_separate_process(fake, tmp_path):
    tool, path, ti = fake
    output = _adapter(tool).run(path, str(tmp_path / "out"))
    assert output == str(tmp_path / "out" / f"{ti.cell}.tsv")
    rows = read_output("T1", output)
    stim = set(ti.descriptor["stim_channels"])
    assert [r["channel"] for r in rows] == [c for c in ti.descriptor["ch_names"]
                                            if c not in stim]
    with open(output + ".pid", encoding="utf-8") as fobj:
        assert int(fobj.read()) != os.getpid()            # another process ran the tool
    assert "fake_detector" not in sys.modules             # nothing of the tool was imported


def test_req354_placeholders_and_appended_paths(fake, tmp_path):
    tool, path, _ = fake
    explicit = Adapter(name="fake", command=["{python}", "{here}/fake_detector.py", "{input}",
                                             "{output}"], here=str(tool.parent))
    assert read_output("T1", explicit.run(path, str(tmp_path / "a")))
    argv = _adapter(tool)._argv("in.json", "out.tsv")
    assert argv == [sys.executable, str(tool), "in.json", "out.tsv"]


@pytest.mark.parametrize("mode, error, message", [
    ("fail", AdapterError, "exit code 3 on .*fake tool: cannot read the recording"),
    ("silent", AdapterError, "no output written"),
    ("invalid", BenchmarkValidationError, "line 2, column 'p_value': 2.0 is greater than "
                                          "the maximum of 1"),
])
def test_req354_failures_are_reported(fake, tmp_path, mode, error, message):
    tool, path, _ = fake
    with pytest.raises(error, match=message):
        _adapter(tool, env={"FAKE_MODE": mode}).run(path, str(tmp_path / "out"))


def test_req354_time_out(fake, tmp_path):
    tool, path, _ = fake
    with pytest.raises(AdapterError, match="did not finish within 1.0 s"):
        _adapter(tool, env={"FAKE_MODE": "sleep"}, timeout=1.0).run(path, str(tmp_path / "o"))


def test_req354_task_and_program_checks(fake, tmp_path):
    tool, path, _ = fake
    with pytest.raises(DataError, match="does not answer task T1"):
        Adapter(name="t7", command=["{python}", str(tool)], tasks=("T7",)).run(path)
    with pytest.raises(DataError, match="unknown tasks"):
        Adapter(name="x", command=["true"], tasks=("T9",))
    with pytest.raises(AdapterError, match="not found"):
        Adapter(name="x", command=["no-such-program-on-this-machine"]).check()
    _adapter(tool).check()


def test_req354_run_suite_scores_an_adapter(fake, tmp_path):
    tool, _, _ = fake
    spec = tmp_path / "tool" / "fake.json"
    spec.write_text(json.dumps({"name": "fake", "version": "0.1", "citation": "a fake detector",
                                "tasks": ["T1"],
                                "command": ["{python}", "{here}/fake_detector.py"]}),
                    encoding="utf-8")
    summary = run_suite("core", str(tmp_path / "run"), tasks=["T1"], methods=[],
                        adapters=[str(spec)], page=True)
    assert summary["failures"] == []
    assert summary["adapters"] == ["fake"]
    with open(tmp_path / "run" / "results" / "fake__T1.json", encoding="utf-8") as fobj:
        document = json.load(fobj)
    assert document["method"]["kind"] == "adapter"
    assert document["method"]["citation"] == "a fake detector"
    assert document["missing_cells"] == []
    assert "fake" in (tmp_path / "run" / "leaderboard.html").read_text(encoding="utf-8")


def test_req354_shipped_adapter_specifications():
    for name, tasks in (("erdetect", ["T1", "T2"]), ("crp", ["T1", "T2"])):
        adapter = load_adapter(str(ADAPTERS / f"{name}.json"))
        assert list(adapter.tasks) == tasks
        assert adapter.citation and "GPL" in adapter.citation
        script = adapter._argv("i", "o")[1]
        assert os.path.isfile(script) and script.startswith(str(ADAPTERS))


def test_req354_erdetect_adapter_with_a_stand_in_detector(tmp_path):
    """The ER-detect adapter's reading of the input and writing of the output, with a
    stand-in for the tool's detection function (the tool itself is not installed here)."""

    standin = tmp_path / "standin" / "standin_detector.py"
    standin.parent.mkdir()
    standin.write_text(
        "import numpy as np\n"
        "def detect(data, onset, sfreq):\n"
        "    a, b = onset + int(0.009 * sfreq), onset + int(0.1 * sfreq)\n"
        "    sd = data[:, :, onset - int(sfreq):onset - int(0.1 * sfreq)].std(axis=2)\n"
        "    j = a + np.argmin(data[:, :, a:b], axis=2)\n"
        "    v = np.take_along_axis(data, j[..., None], axis=2)[..., 0]\n"
        "    hit = v < -3.4 * sd\n"
        "    return np.where(hit, j, np.nan), np.where(hit, v, np.nan)\n", encoding="utf-8")
    adapter = load_adapter(str(ADAPTERS / "erdetect.json"))
    adapter.env = {"ERDETECT_FUNCTION": "standin_detector:detect",
                   "PYTHONPATH": str(standin.parent)}
    for task in ("T1", "T2"):
        ti = inputs(task, 1)[0]
        path = write_input(ti, str(tmp_path / "inputs" / task))
        rows = read_output(task, adapter.run(path, str(tmp_path / "out" / task)))
        if task == "T1":
            assert len(rows) == len(ti.descriptor["ch_names"]) - 2
            assert {r["responsive"] for r in rows} == {0, 1}
        else:
            assert rows and all(0.009 <= r["latency"] <= 0.1 for r in rows)
    assert "standin_detector" not in sys.modules


def test_req354_erdetect_adapter_without_the_tool(tmp_path):
    adapter = load_adapter(str(ADAPTERS / "erdetect.json"))
    adapter.env = {"ERDETECT_FUNCTION": "no_such_module_here:detect"}
    path = write_input(inputs("T1", 1)[0], str(tmp_path / "inputs"))
    with pytest.raises(AdapterError, match="cannot load no_such_module_here:detect"):
        adapter.run(path, str(tmp_path / "out"))


###################################################################################################
## Licence scan

# The notice the adapters carry, outside the package tree (this line is not a notice).
SPDX_GPL = "SPDX-License-Identifier: GPL-3.0-or-later"  # of the adapters outside the package
GNU_GPL = "GNU General Public License"  # the adapters' licence, outside the package


def test_req354_gpl_adapters_live_outside_the_package():
    licence = (ADAPTERS / "LICENSE").read_text(encoding="utf-8")
    assert GNU_GPL in licence
    assert SPDX_GPL in licence
    assert "NOT part of the evoparam package" in licence
    files = sorted(p for p in ADAPTERS.iterdir() if p.is_file())
    code = [p for p in files if p.suffix in (".py", ".m")]
    assert {p.name for p in code} >= {"erdetect_adapter.py", "crp_adapter.py", "crp_wrapper.m",
                                      "_taskio.py"}
    for path in code:
        head = path.read_text(encoding="utf-8").splitlines()[:2]
        assert any(SPDX_GPL in line for line in head), path
    # nothing of the adapters is inside the package tree
    names = {p.stem for p in code}
    inside = [p for p in (ROOT / "evoparam").rglob("*") if p.stem in names]
    assert not inside, inside
    # the adapters are not shipped with the package
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert re.search(r"^prune adapters\s*$", manifest, re.MULTILINE)
    assert not (ADAPTERS / "__init__.py").exists()


def test_req354_package_has_no_gpl_notice_and_imports_no_adapter():
    adapter_modules = {p.stem for p in ADAPTERS.glob("*.py")} | {"adapters", "erdetect"}
    hits = []
    for path in sorted((ROOT / "evoparam").rglob("*")):
        rel = path.relative_to(ROOT).as_posix()
        if not path.is_file() or "__pycache__" in rel:
            continue
        if path.suffix in (".py", ".json", ".md", ".rst", ".txt", ".html") \
                and rel != "evoparam/tests/test_dependencies.py":
            text = path.read_text(encoding="utf-8", errors="replace")
            hits += [f"{rel}: {m.group(0)}" for m in GPL_TEXT.finditer(text)
                     if not _mentions_isolation(text, m.start())]
        if path.suffix == ".py":
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                         [node.module or ""] if isinstance(node, ast.ImportFrom)
                         and node.level == 0 else [])
                hits += [f"{rel}: import {n}" for n in names
                         if n.split(".")[0] in adapter_modules]
    assert not hits, hits


def test_req354_licence_scan_finds_a_mutation(tmp_path):
    text = "# SPDX-License-Identifier: GPL-3.0-only\nimport numpy\n"  # not imported: a probe
    assert [m.group(0) for m in GPL_TEXT.finditer(text)
            if not _mentions_isolation(text, m.start())]
