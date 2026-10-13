# SPDX-License-Identifier: Apache-2.0
"""Write ``docs/glossary.rst`` from section 4 of the specification (spec §9.7, REQ-303 (f)).

The glossary holds every term that section 4 (Architecture) defines:

- the design principles of §4.1 (bold titles);
- the modules of §4.2 (``evoparam.<module>``);
- every container, protocol and field of §4.3 (the ``class`` and ``name: type  # meaning``
  lines of its code blocks), with the type and the meaning given there;
- the metadata columns of Table 4.1 and the channel-table columns of Table 4.2;
- the new names of §4.4 (with the inherited name they replace);
- the errors and warnings of §4.6.

``evoparam/tests/test_docs.py`` checks that the file is up to date (``--check``).

Usage::

    python tools/gen_glossary.py [--check]
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC = ROOT / "docs" / "spec" / "SPEC.md"
TARGET = ROOT / "docs" / "glossary.rst"
# Columns of the table of §4.4: inherited name, new name, note.
TABLE_COLUMNS = 3

CLASS = re.compile(r"^class (\w+)(?:\(([\w, ]*)\))?:\s*(?:\.\.\.)?\s*(?:#\s*(.*))?$")
FIELD = re.compile(r"^    (\w+): ([^#=]+?)(?:\s*=\s*([^#]+?))?\s*(?:#\s*(.*))?$")
ROW = re.compile(r"^\| (`[^|]+`(?:, `[^|]+`)*) \| ([^|]*) \| (.*) \|$")


def section4(text):
    """Return section 4 (Architecture) of the specification text."""
    start = text.index("## 4. Architecture")
    end = text.index("\n## 5. ", start)
    return text[start:end]


def subsections(text):
    """{heading number: body} for the ``### 4.x`` parts."""
    parts = re.split(r"^### (4\.\d) ", text, flags=re.MULTILINE)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def _code_blocks(text):
    return re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)


def rst(text):
    """Inline text safe for reStructuredText: code as literals, markup characters escaped."""
    out = []
    for i, piece in enumerate(re.split(r"(`[^`]+`)", text.strip())):
        if i % 2:
            out.append(f"``{piece.strip('`')}``")
        else:
            escaped = piece.replace("\\", "\\\\").replace("*", "\\*").replace("|", "\\|")
            out.append(re.sub(r"(\w)_(?=\W|$)", r"\1\\_", escaped))
    return "".join(out).strip()


def principles(body):
    """Return the (name, text) of each design principle of §4.1."""
    return [
        (m.group(1).rstrip("."), rst(m.group(2)))
        for m in re.finditer(r"^\d+\. \*\*(.+?)\*\* (.+)$", body, flags=re.MULTILINE)
    ]


def modules(body):
    """Return the (name, text) of each module named in the section, with its import rules."""
    names = sorted(set(re.findall(r"evoparam\.([a-z_]+)", body)))
    rules = {}
    for row in re.findall(
        r"^\| (`[^|]+`(?:, `[^|]+`)*) \| ([^|]*) \| ([^|]*) \|$", body, flags=re.MULTILINE
    ):
        for name in re.findall(r"`(\w+)`", row[0]):
            rules[name] = (row[1].strip(), row[2].strip())
    out = []
    for name in names:
        text = f"Module ``evoparam.{name}``."
        if name in rules:
            text += f" May import: {rst(rules[name][0])}. Must not import: {rst(rules[name][1])}."
        out.append((f"evoparam.{name}", text))
    return out


def classes(body):
    """(class terms, field terms) from the code blocks of §4.3."""
    class_terms, field_terms = [], []
    for block in _code_blocks(body):
        current = None
        for line in block.splitlines():
            match = CLASS.match(line)
            if match:
                current = match.group(1)
                base, comment = match.group(2), match.group(3)
                text = rst(comment) if comment else "Container of the data model (§4.3)."
                if base:
                    text = (
                        f"{text} ({rst('`' + base + '`')})"
                        if comment
                        else f"Protocol or class based on ``{base}`` (§4.3)."
                    )
                class_terms.append((current, text))
                continue
            match = FIELD.match(line)
            if match and current:
                name, kind, default, comment = match.groups()
                text = f"``{kind.strip()}``"
                if default:
                    text += f", default ``{default.strip()}``"
                if comment:
                    text += f"; {rst(comment)}"
                field_terms.append((f"{current}.{name}", text))
            elif line and not line.startswith((" ", "#", "@")):
                current = None
    return class_terms, field_terms


def table_columns(body, title):
    """Rows of the table that follows ``title`` in ``body``."""
    start = body.index(title)
    rows = []
    for line in body[start:].splitlines()[1:]:
        if not line.strip() and rows:
            break
        match = ROW.match(line)
        if match:
            for name in re.findall(r"`(\w+)`", match.group(1)):
                rows.append((name, f"{rst(match.group(2))}; {rst(match.group(3))}"))
    return rows


def prohibited_words():
    """Return the words of the vocabulary rule (the table of CONTRIBUTING.md, REQ-012)."""
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    block = text.split("<!-- prohibited-words -->")[1].split("<!-- end-prohibited-words -->")[0]
    words = []
    for line in block.splitlines()[2:]:
        if line.startswith("|"):
            words += re.findall(r"`([^`]+)`", line.split("|")[1])
    return words


def _clean(text, words):
    """Return False when ``text`` holds a word of the vocabulary rule (never in the docs)."""
    return not any(
        re.search(
            r"(?<![A-Za-z0-9_])" + re.escape(w) + r"(?![A-Za-z0-9_])",
            text,
            0 if w[:1].isupper() and w[1:].islower() else re.IGNORECASE,
        )
        for w in words
    )


def renames(body):
    """Return the new names of §4.4, without the inherited names they replace.

    REQ-012: the docs never name them; the specification keeps the table.
    """
    words = prohibited_words()
    out = []
    for line in body.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if (
            len(cells) != TABLE_COLUMNS
            or not line.startswith("| ")
            or cells[0] in ("inherited name", "---")
        ):
            continue
        new = re.findall(r"`([\w.]+)`", cells[1])
        if not new or cells[1].startswith("removed"):
            continue
        text = "Name chosen in place of an inherited one (spec §4.4)"
        if _clean(cells[1], words):
            text += f": {rst(cells[1])}"
        if cells[2] and _clean(cells[2], words) and "inherited" not in cells[2]:
            text += f". Reason: {rst(cells[2])}"
        out.append((new[0], text + "."))
    return out


def errors(body):
    """Return the (name, text) of each error class of the section's code blocks."""
    out = []
    for block in _code_blocks(body):
        for line in block.splitlines():
            match = CLASS.match(line)
            if match:
                base, comment = match.group(2), match.group(3)
                text = f"Subclass of ``{base}``."
                if comment:
                    text = f"{rst(comment[0].upper() + comment[1:])}. {text}"
                out.append((match.group(1), text))
    return out


def _glossary(title, entries):
    lines = [title, "-" * len(title), "", ".. glossary::", ""]
    seen = set()
    for term, text in entries:
        if term in seen:
            continue
        seen.add(term)
        lines += [f"   {term}", f"      {text}", ""]
    return lines


def render(spec_text):
    """Return the text of the glossary built from the specification text."""
    parts = subsections(section4(spec_text))
    class_terms, field_terms = classes(parts["4.3"])
    out = [
        "Glossary",
        "========",
        "",
        ".. Generated by tools/gen_glossary.py from section 4 of docs/spec/SPEC.md; do not edit.",
        "",
        "Every term defined in section 4 (Architecture) of the specification: design "
        "principles, modules, containers and their fields, the standard metadata and "
        "channel columns, the names that replace inherited ones, and the errors.",
        "",
    ]
    out += _glossary("Design principles", principles(parts["4.1"]))
    out += _glossary("Modules", modules(parts["4.2"]))
    out += _glossary("Containers, protocols and results", class_terms)
    out += _glossary("Fields", field_terms)
    out += _glossary(
        "Trial metadata columns (Table 4.1)", table_columns(parts["4.3"], "(Table 4.1)")
    )
    out += _glossary(
        "Channel table columns (Table 4.2)",
        [(f"ch_info.{n}", t) for n, t in table_columns(parts["4.3"], "(Table 4.2)")],
    )
    out += _glossary("Names that replace inherited ones", renames(parts["4.4"]))
    out += _glossary("Errors and warnings", errors(parts["4.6"]))
    while out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"


def terms(spec_text=None):
    """Every term the glossary must hold (used by the tests)."""
    spec_text = spec_text or SPEC.read_text(encoding="utf-8")
    parts = subsections(section4(spec_text))
    class_terms, field_terms = classes(parts["4.3"])
    return [
        t
        for t, _ in principles(parts["4.1"])
        + modules(parts["4.2"])
        + class_terms
        + field_terms
        + errors(parts["4.6"])
    ]


def main(argv=None):
    """Write the glossary, or with ``--check`` fail when it is out of date."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if the file is out of date")
    args = parser.parse_args(argv)
    text = render(SPEC.read_text(encoding="utf-8"))
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != text:
            import difflib

            diff = difflib.unified_diff(current.splitlines(True), text.splitlines(True))
            print("".join(list(diff)[:40]), end="")
            print(f"{TARGET.relative_to(ROOT)} is out of date: run tools/gen_glossary.py")
            return 1
        return 0
    TARGET.write_text(text, encoding="utf-8")
    count = len(re.findall(r"^   \S", text, flags=re.MULTILINE))
    print(f"wrote {TARGET.relative_to(ROOT)} ({count} terms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
