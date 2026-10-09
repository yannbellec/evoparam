"""REQ-326 (and the tutorial files of REQ-303): the tutorials exist, end with the methods
paragraph, name only registry datasets with checksums, and run.

The tutorials run with ``EVOPARAM_SLOW_TESTS=1`` (the simulations, about three minutes) and
``EVOPARAM_NETWORK_TESTS=1`` (those that read public data); the docs build runs them all.
"""

import ast
import json
import os
import re
import subprocess
import sys

import pytest

import evoparam
from evoparam import presets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(evoparam.__file__)))
TUTORIALS = os.path.join(ROOT, "docs", "tutorials")
INTEGRATIONS = ("mne", "bids", "nwb", "neo", "sklearn", "cli", "streaming")
# the online mode is build step S18: its tutorial comes with it
LATER = {"integration_streaming": "S18"}
NETWORK = {"preset_ccep", "preset_ccep_seeg", "preset_erp", "integration_bids"}
SLOW = os.environ.get("EVOPARAM_SLOW_TESTS") == "1"
ONLINE = os.environ.get("EVOPARAM_NETWORK_TESTS") == "1"

pytestmark = pytest.mark.skipif(not os.path.isdir(TUTORIALS), reason="needs the source tree")


def _names():
    return [f"preset_{name}" for name in sorted(presets.PRESETS.builtin)] + \
        [f"integration_{name}" for name in INTEGRATIONS] + ["reporting_your_results"]


def _source(name):
    with open(os.path.join(TUTORIALS, f"{name}.py"), encoding="utf-8") as fobj:
        return fobj.read()


def test_every_tutorial_exists():

    missing = [n for n in _names() if not os.path.isfile(os.path.join(TUTORIALS, f"{n}.py"))]
    assert missing == [n for n in LATER if not _stream_exists()]


def _stream_exists():
    import importlib.util

    return importlib.util.find_spec("evoparam.stream") is not None


@pytest.mark.parametrize("name", [n for n in _names() if n not in LATER])
def test_tutorial_structure(name):

    source = _source(name)
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree)
    assert docstring and "=====" in docstring, name                 # sphinx-gallery title
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    called = {getattr(c.func, "attr", getattr(c.func, "id", None)) for c in calls}
    assert "methods_text" in called, name
    # the last cell prints the methods paragraph
    assert "methods_text" in source.rsplit("# %%", 1)[1], name
    # downloads only through fetch_* (no urllib, requests or curl)
    assert not re.search(r"\b(urllib|requests|httpx|wget|curl)\b", source), name


def test_datasets_named_are_in_the_registry_with_checksums():

    with open(os.path.join(ROOT, "evoparam", "io", "registry.json"), encoding="utf-8") as f:
        registry = {d["id"]: d for d in json.load(f)["datasets"]}
    for name in _names():
        if name in LATER:
            continue
        source = _source(name)
        # the datasets a tutorial reads: DATASET = "..." or the first argument of a reader
        named = set(re.findall(r'DATASET = "(ds\d{6})"', source)) | \
            set(re.findall(r'(?:read_bids|fetch_openneuro_file)\("(ds\d{6})"', source))
        if "fetch_erpcore" in source:
            named.add("erp-core")
        for dataset in named:
            assert dataset in registry, (name, dataset)
            assert registry[dataset].get("files"), (name, dataset, "no sha256 recorded")
            for path, digest in registry[dataset]["files"].items():
                assert re.fullmatch(r"[0-9a-f]{64}", digest), (dataset, path)
        # every file a tutorial fetches from OpenNeuro is in the registry
        for stem, suffix in re.findall(r'fetch_openneuro_file\(DATASET, (STEM) \+ "([^"]+)"',
                                       source):
            dataset = re.search(r'DATASET = "(ds\d{6})"', source).group(1)
            path = re.search(r'STEM = "([^"]+)"', source).group(1) + suffix
            assert path in registry[dataset]["files"], (name, path)


@pytest.mark.skipif(not SLOW, reason="EVOPARAM_SLOW_TESTS=1 runs the tutorials")
@pytest.mark.parametrize("name", [n for n in _names() if n not in LATER])
def test_tutorial_runs(name, tmp_path):

    if name in NETWORK and not ONLINE:
        pytest.skip("reads public data: EVOPARAM_NETWORK_TESTS=1")
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONWARNINGS="ignore",
               PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run([sys.executable, os.path.join(TUTORIALS, f"{name}.py")],
                          capture_output=True, text=True, cwd=str(tmp_path), env=env,
                          timeout=1800)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "were analysed with evoparam" in proc.stdout, name
