# SPDX-License-Identifier: Apache-2.0
"""Documentation content (spec §10.7).

REQ-326 (and the tutorial files of REQ-303): the tutorials exist, end with the methods
paragraph, name only registry datasets with checksums, and run. REQ-325: the streaming
tutorial states the intended use.

The tutorials run with ``EVOPARAM_SLOW_TESTS=1`` (the simulations, about three minutes) and
``EVOPARAM_NETWORK_TESTS=1`` (those that read public data); the docs build runs them all.
Source tree only.
"""

import ast
import json
import os
import re
import runpy
import subprocess
import sys

import pytest

import evoparam
from evoparam import presets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(evoparam.__file__)))
TUTORIALS = os.path.join(ROOT, "docs", "tutorials")
INTEGRATIONS = ("mne", "bids", "nwb", "neo", "sklearn", "cli", "streaming")
# tutorials that come with a later build step (the online mode, S18, is merged: none left)
LATER = {}
NETWORK = {"preset_ccep", "preset_ccep_seeg", "preset_erp", "integration_bids"}
SLOW = os.environ.get("EVOPARAM_SLOW_TESTS") == "1"
ONLINE = os.environ.get("EVOPARAM_NETWORK_TESTS") == "1"

STREAM_TUTORIAL = os.path.join(ROOT, "tutorials", "92_online_mapping_with_lsl.py")

pytestmark = [
    pytest.mark.skipif(not os.path.isdir(TUTORIALS), reason="needs the source tree"),
    pytest.mark.skipif(
        not os.path.isfile(os.path.join(ROOT, "pyproject.toml")), reason="needs the source tree"
    ),
]


def _names():
    return (
        [f"preset_{name}" for name in sorted(presets.PRESETS.builtin)]
        + [f"integration_{name}" for name in INTEGRATIONS]
        + ["reporting_your_results"]
    )


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
    assert docstring and "=====" in docstring, name  # sphinx-gallery title
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
        named = set(re.findall(r'DATASET = "(ds\d{6})"', source)) | set(
            re.findall(r'(?:read_bids|fetch_openneuro_file)\("(ds\d{6})"', source)
        )
        if "fetch_erpcore" in source:
            named.add("erp-core")
        for dataset in named:
            assert dataset in registry, (name, dataset)
            assert registry[dataset].get("files"), (name, dataset, "no sha256 recorded")
            for path, digest in registry[dataset]["files"].items():
                assert re.fullmatch(r"[0-9a-f]{64}", digest), (dataset, path)
        # every file a tutorial fetches from OpenNeuro is in the registry
        for stem, suffix in re.findall(
            r'fetch_openneuro_file\(DATASET, (STEM) \+ "([^"]+)"', source
        ):
            dataset = re.search(r'DATASET = "(ds\d{6})"', source).group(1)
            path = re.search(r'STEM = "([^"]+)"', source).group(1) + suffix
            assert path in registry[dataset]["files"], (name, path)


@pytest.mark.skipif(not SLOW, reason="EVOPARAM_SLOW_TESTS=1 runs the tutorials")
@pytest.mark.parametrize("name", [n for n in _names() if n not in LATER])
def test_tutorial_runs(name, tmp_path):

    if name in NETWORK and not ONLINE:
        pytest.skip("reads public data: EVOPARAM_NETWORK_TESTS=1")
    env = dict(
        os.environ,
        MPLBACKEND="Agg",
        PYTHONWARNINGS="ignore",
        PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )
    proc = subprocess.run(
        [sys.executable, os.path.join(TUTORIALS, f"{name}.py")],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "were analysed with evoparam" in proc.stdout, name


def test_req325_streaming_tutorial_states_intended_use():

    with open(STREAM_TUTORIAL, encoding="utf-8") as fobj:
        text = fobj.read()
    assert "not a medical device" in text
    assert "not validated for clinical" in text


def test_streaming_tutorial_runs(capsys):

    runpy.run_path(STREAM_TUTORIAL, run_name="__main__")
    out = capsys.readouterr().out
    assert "c0: response" in out and "c7: no response" in out


# ---------------------------------------------------------------------------------------
# REQ-302, REQ-303: documentation pages (S20)
# ---------------------------------------------------------------------------------------

DOCS = os.path.join(ROOT, "docs")


def _tool(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "tools", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _page(name):
    with open(os.path.join(DOCS, name), encoding="utf-8") as fobj:
        return fobj.read()


def test_req303_pages_are_in_the_toctree():
    index = _page("index.rst")
    for page in (
        "getting_started",
        "auto_tutorials/index",
        "auto_examples/index",
        "reporting",
        "faq",
        "presets",
        "preset_sensitivity",
        "datasets",
        "api",
        "glossary",
    ):
        assert f"\n   {page}\n" in index, page
        if "/" not in page:
            assert any(os.path.isfile(os.path.join(DOCS, page + ext)) for ext in (".rst", ".md"))


def test_req303_getting_started():
    text = _page("getting_started.rst")
    for item in (
        "pip install evoparam",
        "evoparam.selftest()",
        "What the numbers mean",
        "latency",
        "amplitude",
        "polarity",
        "fwhm",
        "status",
        "r2",
    ):
        assert item in text, item
    example = text.split("A ten-line example", 1)[1].split(".. code-block:: python", 1)[1]
    example = example.split("\n\n", 2)[1]
    assert len([line for line in example.splitlines() if line.strip()]) <= 10


def test_req303_reporting_page():
    text = _page("reporting.rst")
    for item in (
        "settings",
        "preset",
        "package version",
        "goodness of fit",
        "existence test",
        "artifact method",
        "montage",
        "methods_text()",
        "CITATION.cff",
        "Zenodo",
        "evoparam.cite()",
        "methods.txt",
    ):
        assert item in text, item


FAQ = (
    "Which preset for my data?",
    "Why is polarity a separate field?",
    "Why is ``r2`` not the inherited correlation?",
    "My N1 is missing",
    "GPU gives slightly different numbers",
    "How do I add a shape?",
    "Is this a medical device?",
)


def test_req303_faq_has_the_seven_questions():
    text = _page("faq.rst")
    headings = re.findall(r"^(.+)\n-{3,}$", text, flags=re.MULTILINE)
    assert [h for h in headings if h in FAQ] == list(FAQ)
    medical = text.split("Is this a medical device?", 1)[1]
    assert "Section 13" in medical and "research use only" in medical
    for name in presets.PRESETS.builtin:  # the preset question covers them all
        assert f"``{name}``" in text.split("Why is polarity", 1)[0], name


def test_req303_every_public_symbol_in_the_api_index():
    tool = _tool("gen_api_index")
    index = _page("api.rst")
    sections = dict(
        re.findall(
            r"^\.\. currentmodule:: (\S+)\n\n\.\. autosummary::\n"
            r"   :toctree: generated/\n\n((?:   \S+\n)+)",
            index,
            flags=re.MULTILINE,
        )
    )
    missing = []
    for module, names in tool.public_symbols().items():
        listed = set(sections.get(module, "").split())
        missing += [f"{module}.{n}" for n in names if n != "__version__" and n not in listed]
    assert missing == []
    assert tool.main(["--check"]) == 0, "docs/api.rst is out of date"


def test_req303_glossary_holds_every_term_of_section_4():
    tool = _tool("gen_glossary")
    text = _page("glossary.rst")
    terms = set(re.findall(r"^   (\S.*)$", text, flags=re.MULTILINE))
    expected = tool.terms()
    assert len(expected) > 150
    assert [t for t in expected if t not in terms] == []
    for name in (
        "TrialSet",
        "Evoked",
        "Provenance",
        "ResponseFit",
        "GroupFit",
        "TrialFit",
        "ResponseTest",
        "ArtifactResult",
        "Preset",
        "Component.polarity",
        "GoodnessOfFit.r2",
        "FitError",
        "stim_site",
        "ch_info.stim_contact",
    ):
        assert name in terms, name
    assert tool.main(["--check"]) == 0, "docs/glossary.rst is out of date"


def test_req303_datasets_page_generated_from_the_registry():
    tool = _tool("gen_registry_doc")
    with open(os.path.join(ROOT, "evoparam", "io", "registry.json"), encoding="utf-8") as fobj:
        registry = json.load(fobj)
    text = _page("datasets.rst")
    for item in registry["datasets"]:
        assert f"``{item['id']}``" in text, item["id"]
        if item.get("doi"):
            assert item["doi"] in text
    for column in ("DOI", "size (GB)", "sampling rate (Hz)", "licence", "events", "loader"):
        assert column in text
    assert tool.main(["--check"]) == 0, "docs/datasets.rst is out of date"


def _cited_keys():
    keys = set()
    for folder, _, files in os.walk(DOCS):
        if os.path.relpath(folder, DOCS).split(os.sep)[0] in ("spec", "_build"):
            continue
        for name in files:
            if name.endswith((".rst", ".md", ".py", ".txt")) and name != "references.bib":
                with open(os.path.join(folder, name), encoding="utf-8") as fobj:
                    keys |= set(re.findall(r"LIT-[a-z]+-\d+", fobj.read()))
    return keys


def test_req302_every_cited_literature_key_is_in_references_bib():
    bib = set(
        re.findall(r"^@misc\{(LIT-[a-z]+-\d+),", _page("references.bib"), flags=re.MULTILINE)
    )
    cited = _cited_keys()
    assert cited, "the documentation cites no literature key"
    assert sorted(cited - bib) == []
    assert _tool("gen_references_bib").main(["--check"]) == 0, "references.bib is out of date"
