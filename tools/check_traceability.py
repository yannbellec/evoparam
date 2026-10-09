"""Check the traceability of the gaps (REQ-018).

Every gap of ``docs/spec/GAPS.md`` marked "Addressed" must map to at least one requirement,
one module, one test file and one validation item in the traceability matrix of the
specification (``docs/spec/SPEC.md`` §14.1), and the requirements of the gap in GAPS.md
must be those of §14.1. Every requirement named must exist in §14.2.

``--files`` also lists the test files of §14 that are not in the repository (information:
the build steps that write them may still be to come); ``--strict`` makes them errors.

Usage::

    python tools/check_traceability.py [--files] [--strict] [--root PATH]

Exit code 0 when the mapping is complete, 1 otherwise; the problems are printed.
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _rows(text, start, stop=None):
    """Cells of the markdown table rows between the headings ``start`` and ``stop``."""

    body = text.split(start, 1)[1]
    if stop:
        body = body.split(stop, 1)[0]
    rows = []
    for line in body.splitlines():
        if line.startswith("|") and not re.match(r"^\|\s*-", line):
            rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows[1:]                                   # without the header


def gaps(root):
    """{GAP id: (status, set of REQ ids)} from the master table of GAPS.md."""

    text = (root / "docs" / "spec" / "GAPS.md").read_text(encoding="utf-8")
    out = {}
    for cells in _rows(text, "## Master table"):
        if not cells or not cells[0].startswith("GAP-"):
            continue
        does = cells[3]
        status = ("Partly addressed" if does.startswith("Partly addressed") else
                  "Not addressed" if does.startswith("Not addressed") else
                  "Addressed" if does.startswith("Addressed") else "?")
        out[cells[0]] = (status, set(re.findall(r"REQ-\d{3}", cells[-1])))
    return out


def matrix(root):
    """§14.1 rows {GAP id: dict} and the set of REQ ids of §14.2."""

    text = (root / "docs" / "spec" / "SPEC.md").read_text(encoding="utf-8")
    rows = {}
    for cells in _rows(text, "### 14.1 Gap → requirement", "### 14.2"):
        if len(cells) < 7 or not cells[0].startswith("GAP-"):
            continue
        rows[cells[0]] = {"status": cells[2], "reqs": set(re.findall(r"REQ-\d{3}", cells[3])),
                          "modules": [m for m in re.split(r",\s*", cells[4]) if m not in
                                      ("", "—", "-")],
                          "tests": re.findall(r"`([^`]+)`", cells[5]),
                          "validation": cells[6] if cells[6] not in ("", "—", "-") else ""}
    reqs = {}
    for cells in _rows(text, "### 14.2 Requirement", "\n## "):
        if cells and re.match(r"REQ-\d{3}$", cells[0]):
            reqs[cells[0]] = re.findall(r"`([^`]+)`", cells[3]) if len(cells) > 3 else []
    return rows, reqs


def check(root=ROOT, files=False, strict=False):
    """(errors, notes): the problems of the mapping, and the information lines."""

    errors, notes = [], []
    table = gaps(root)
    rows, reqs = matrix(root)
    addressed = sorted(g for g, (status, _) in table.items() if status == "Addressed")
    if not addressed:
        errors.append("GAPS.md: no gap marked Addressed was found (format changed?)")
    for gap in addressed:
        row = rows.get(gap)
        if row is None:
            errors.append(f"{gap}: Addressed in GAPS.md but missing from §14.1")
            continue
        if not row["status"].startswith("Addressed"):
            errors.append(f"{gap}: status '{row['status']}' in §14.1, Addressed in GAPS.md")
        for key, what in (("reqs", "requirement"), ("modules", "module"),
                          ("tests", "test file"), ("validation", "validation item")):
            if not row[key]:
                errors.append(f"{gap}: no {what} in §14.1")
        if row["reqs"] != table[gap][1]:
            errors.append(f"{gap}: requirements differ between GAPS.md "
                          f"({sorted(table[gap][1])}) and §14.1 ({sorted(row['reqs'])})")
        unknown = sorted(r for r in row["reqs"] if r not in reqs)
        if unknown:
            errors.append(f"{gap}: {', '.join(unknown)} not in §14.2")
    if files:
        present = {p.name for p in root.rglob("*.py") if ".git" not in p.parts}
        named = sorted({pathlib.PurePosixPath(t).name for g in addressed if g in rows
                        for t in rows[g]["tests"]} |
                       {pathlib.PurePosixPath(t).name for tests in reqs.values() for t in tests
                        if t.endswith(".py")})
        missing = [name for name in named if name.endswith(".py") and name not in present]
        for name in missing:
            (errors if strict else notes).append(f"test file of §14 not in the repository: "
                                                 f"{name}")
        notes.append(f"{len(named) - len(missing)} of {len(named)} test files of §14 exist")
    notes.insert(0, f"{len(addressed)} gaps marked Addressed, {len(rows)} rows in §14.1, "
                    f"{len(reqs)} requirements in §14.2")
    return errors, notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--files", action="store_true",
                        help="also list the test files of §14 missing from the repository")
    parser.add_argument("--strict", action="store_true",
                        help="with --files, a missing test file is an error")
    args = parser.parse_args(argv)
    errors, notes = check(pathlib.Path(args.root), files=args.files, strict=args.strict)
    for line in notes:
        print(line)
    for line in errors:
        print("ERROR:", line)
    print("traceability: " + ("ok" if not errors else f"{len(errors)} problem(s)"))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
