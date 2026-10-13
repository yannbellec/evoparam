# SPDX-License-Identifier: Apache-2.0
"""The package citation, read from ``CITATION.cff`` (the single source of truth, spec §11).

``CITATION.cff`` is YAML. The package keeps numpy and scipy as its only required
dependencies, so this module reads the subset of YAML that citation files use (nested
mappings, lists of scalars or mappings, quoted or plain scalars, comments) with the
standard library. The copy shipped inside the package (``evoparam/CITATION.cff``) must
be identical to the one at the root of the repository; a test checks it.
"""

import os
import re
from importlib import resources
from typing import Any, cast

__all__ = ["read_citation", "citation_text", "citation_bibtex", "parse_cff"]


_KEY = re.compile(r"^[A-Za-z0-9_.-]+:(\s|$)")
_QUOTED_MIN = 2  # an opening and a closing quote


def _scalar(text: str) -> str | None:
    """Return a YAML scalar without its quotes or trailing comment (None when empty)."""
    text = text.strip()
    if not text:
        return None
    if text[0] in "\"'" and text[-1] == text[0] and len(text) >= _QUOTED_MIN:
        inner = text[1:-1]
        return inner.replace("''", "'") if text[0] == "'" else inner.replace('\\"', '"')
    if " #" in text:
        text = text.split(" #", 1)[0].rstrip()
    return text


def _lines(text: str) -> list[tuple[int, str]]:
    """Return the indentation and text of each line that is not blank, a comment or ``---``."""
    out = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or stripped == "---":
            continue
        out.append((len(raw) - len(raw.lstrip(" ")), raw.strip()))
    return out


def _parse_block(lines: list[tuple[int, str]], start: int, indent: int) -> tuple[Any, int]:
    """Parse the block of ``lines`` that starts at ``start`` with indentation ``indent``."""
    if start >= len(lines):
        return None, start
    if lines[start][1].startswith("- "):
        items, i = [], start
        while i < len(lines) and lines[i][0] == indent and lines[i][1].startswith("- "):
            body = lines[i][1][2:]
            if _KEY.match(body):
                # a mapping whose first key is on the dash line
                sub = [(indent + 2, body)]
                j = i + 1
                while j < len(lines) and lines[j][0] > indent:
                    sub.append(lines[j])
                    j += 1
                value, _ = _parse_block(sub, 0, indent + 2)
                items.append(value)
                i = j
            else:
                items.append(_scalar(body))
                i += 1
        return items, i
    mapping, i = {}, start
    key: str | None
    while i < len(lines) and lines[i][0] == indent:
        key, _, rest = lines[i][1].partition(":")
        key = _scalar(key)
        if rest.strip():
            mapping[key] = _scalar(rest)
            i += 1
        elif i + 1 < len(lines) and lines[i + 1][0] > indent:
            mapping[key], i = _parse_block(lines, i + 1, lines[i + 1][0])
        elif i + 1 < len(lines) and lines[i + 1][0] == indent and lines[i + 1][1].startswith("- "):
            mapping[key], i = _parse_block(lines, i + 1, indent)
        else:
            mapping[key] = None
            i += 1
    return mapping, i


def parse_cff(text: str) -> Any:
    """Parse the YAML subset of a citation file into dictionaries, lists and strings."""
    lines = _lines(text)
    if not lines:
        return {}
    value, _ = _parse_block(lines, 0, lines[0][0])
    return value


def _cff_text() -> str:
    """Return the text of CITATION.cff (installed with the package, else in the source tree)."""
    try:
        return resources.files("evoparam").joinpath("CITATION.cff").read_text("utf-8")
    except (FileNotFoundError, OSError):
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "CITATION.cff",
        )
        with open(root, encoding="utf-8") as fobj:
            return fobj.read()


def read_citation() -> dict[str, Any]:
    """Return the parsed ``CITATION.cff`` of the package (a dictionary)."""
    return cast("dict[str, Any]", parse_cff(_cff_text()))


def _authors(cff: dict[str, Any]) -> list[tuple[str, str]]:
    """Return the (family name, given names) of each author of the citation file."""
    names = []
    for author in cff.get("authors") or []:
        if not isinstance(author, dict):
            continue
        if author.get("family-names"):
            given = author.get("given-names") or ""
            names.append((author["family-names"], given))
        elif author.get("name"):
            names.append((author["name"], ""))
    return names


def citation_text(cff: dict[str, Any] | None = None) -> str:
    """One-line citation.

    Authors (family name and initials), title, version, then the DOI or else the repository.
    """
    cff = read_citation() if cff is None else cff
    authors = ", ".join(
        " ".join([family] + ["".join(f"{n[0]}." for n in given.replace("-", " ").split())]).strip()
        for family, given in _authors(cff)
    )
    text = f"{authors.rstrip('.')}. " if authors else ""
    text += f"{cff.get('title', 'evoparam')}"
    text += f" (version {cff['version']})." if cff.get("version") else "."
    version_doi = _version_doi(cff)
    if cff.get("doi") and version_doi:
        # REQ-391: the concept DOI (every version) and the DOI of this version
        text += (
            f" https://doi.org/{version_doi} (this version; all versions: "
            f"https://doi.org/{cff['doi']})"
        )
    elif cff.get("doi"):
        text += f" https://doi.org/{cff['doi']}"
    elif cff.get("repository-code"):
        text += f" {cff['repository-code']}"
    paper = _preferred(cff)
    if paper:
        text += f" Please also cite: {paper}"
    return text


def _version_doi(cff: dict[str, Any]) -> str | None:
    """Return the DOI of this version.

    An ``identifiers`` entry of type ``doi`` whose description names the version (Zenodo's version
    DOI), when the citation file has one.
    """
    for item in cff.get("identifiers") or []:
        if (
            isinstance(item, dict)
            and item.get("type") == "doi"
            and "version" in str(item.get("description", "")).lower()
        ):
            return item.get("value")
    return None


def _preferred(cff: dict[str, Any]) -> str | None:
    """Return the ``preferred-citation`` (the paper), formatted on one line, or None."""
    paper = cff.get("preferred-citation")
    if not isinstance(paper, dict) or not paper.get("title"):
        return None
    authors = ", ".join(
        " ".join([family] + [f"{n[0]}." for n in given.replace("-", " ").split()]).strip()
        for family, given in _authors(paper)
    )
    parts = [authors.rstrip(".") + "." if authors else "", f"{paper['title']}."]
    venue = paper.get("journal") or paper.get("collection-title") or paper.get("publisher")
    if isinstance(venue, dict):
        venue = venue.get("name")
    if venue:
        parts.append(f"{venue}")
    if paper.get("year"):
        parts.append(f"({paper['year']}).")
    if paper.get("doi"):
        parts.append(f"https://doi.org/{paper['doi']}")
    return " ".join(p for p in parts if p)


def citation_bibtex(cff: dict[str, Any] | None = None) -> str:
    """BibTeX entry (``@software``) built from the citation file."""
    cff = read_citation() if cff is None else cff
    names = _authors(cff)
    authors = " and ".join(f"{family}, {given}".rstrip(", ") for family, given in names)
    fields = [
        ("title", cff.get("title")),
        ("author", authors),
        ("version", cff.get("version")),
        ("doi", cff.get("doi")),
        ("url", cff.get("repository-code")),
        ("license", cff.get("license")),
    ]
    if cff.get("date-released"):
        fields.append(("year", str(cff["date-released"])[:4]))
    key = f"{names[0][0].lower()}_evoparam" if names else "evoparam"
    body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields if value)
    return f"@software{{{key},\n" + body + ",\n}"
