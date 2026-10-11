# SPDX-License-Identifier: Apache-2.0
"""REQ-308 (release process and citation metadata) and REQ-391 (CITATION.cff, DOIs, cite).

``CITATION.cff`` is read without cffconvert (the package's own YAML-subset reader, and
PyYAML when installed, must agree). Before the HIIE decision (RELEASING.md) there is no
DOI and no release: the checks that need them are strict expected failures that wait for
the owner's decision.
"""

import pathlib
import re
import subprocess

import pytest

import evoparam
from evoparam.data._citation import citation_text, parse_cff, read_citation

ROOT = pathlib.Path(evoparam.__file__).resolve().parent.parent
HIIE = "waits for the owner's HIIE decision (nothing public before it)"

pytestmark = pytest.mark.skipif(not (ROOT / "CITATION.cff").is_file(),
                                reason="needs the source tree")


def _cff():
    return parse_cff((ROOT / "CITATION.cff").read_text(encoding="utf-8"))


def _release_version(version):
    """``0.1.0.dev0`` -> ``0.1.0`` (the version being prepared); a final version is itself."""

    return re.match(r"\d+(\.\d+)*", version).group(0)


def test_cff_parses_the_same_with_pyyaml():
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    ours = _cff()
    assert {k: str(v) for k, v in data.items() if not isinstance(v, (list, dict))} == \
        {k: v for k, v in ours.items() if not isinstance(v, (list, dict))}
    assert data["cff-version"] == "1.2.0"


def test_req308_version_matches_the_package():
    """``version`` is ``evoparam.__version__`` on a release; on a development build
    (``X.Y.Z.devN``) it is the version being prepared (DECISIONS.md, REQ-308)."""

    version = _cff()["version"]
    assert version == _release_version(evoparam.__version__)
    if re.fullmatch(r"\d+\.\d+\.\d+", evoparam.__version__):
        assert version == evoparam.__version__


def test_req308_date_released_is_the_tag_date():
    tag = subprocess.run(["git", "describe", "--exact-match", "--tags", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True)
    cff = _cff()
    if tag.returncode != 0:
        assert "date-released" not in cff, "date-released only on a tagged commit"
        return
    date = subprocess.run(["git", "log", "-1", "--format=%cs", tag.stdout.strip()], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    assert str(cff["date-released"]) == date


def test_req308_first_author():
    assert _cff()["authors"][0]["family-names"] == "Bellec"


def test_package_copy_is_the_root_file():
    assert (ROOT / "evoparam" / "CITATION.cff").read_text("utf-8") == \
        (ROOT / "CITATION.cff").read_text("utf-8")


@pytest.mark.xfail(strict=True, reason=f"REQ-308/REQ-391: the Zenodo concept DOI {HIIE}")
def test_req308_doi_is_a_zenodo_doi():
    assert str(_cff().get("doi", "")).startswith("10.5281/zenodo.")


def test_req308_releasing_has_the_steps():
    text = (ROOT / "RELEASING.md").read_text(encoding="utf-8")
    section = text.split("## Making a release", 1)[1].split("\n## ", 1)[0]
    for step in ("freeze the CHANGELOG", "bump the version", "benchmarks/run_benchmarks.py",
                 "reference machine", "-W", "tag", "PyPI smoke test", "Zenodo DOI",
                 "announce", "validation/robustness.py", "tools/layer3_gate.py"):
        assert step.lower() in section.lower(), step


def test_req391_cite_names_the_version(capsys):
    text = evoparam.cite()
    assert f"version {_cff()['version']}" in text
    assert text in capsys.readouterr().out


def test_req391_both_dois_and_the_paper_when_they_exist():
    """With a concept DOI, a version DOI and a preferred citation in the file, the
    citation (``cite()``, ``evoparam cite``, the methods text) names all three."""

    cff = dict(read_citation(), doi="10.5281/zenodo.1000",
               identifiers=[{"type": "doi", "value": "10.5281/zenodo.1001",
                             "description": "The version DOI of this release"}])
    cff["preferred-citation"] = {"type": "article", "title": "Evoked response "
                                 "parameterization", "authors": [{"family-names": "Bellec",
                                                                   "given-names": "Yann V."}],
                                 "journal": "A Journal", "year": "2027",
                                 "doi": "10.1000/xyz"}
    text = citation_text(cff)
    assert "https://doi.org/10.5281/zenodo.1001" in text
    assert "https://doi.org/10.5281/zenodo.1000" in text
    assert "Evoked response parameterization" in text and "10.1000/xyz" in text


@pytest.mark.xfail(strict=True, reason=f"REQ-391: the version DOI of each release {HIIE}")
def test_req391_cite_contains_the_version_doi():
    assert re.search(r"10\.5281/zenodo\.\d+", evoparam.cite())


@pytest.mark.xfail(strict=True, reason=f"REQ-391: the preferred paper citation needs a paper, "
                   f"which {HIIE}")
def test_req391_preferred_citation():
    assert "preferred-citation" in _cff()
