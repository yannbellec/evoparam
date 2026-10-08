"""reports, plots and cli compute no scientific quantity (REQ-349): they import no numerical
library and call no numerical routine of NumPy."""

import ast
import pathlib

import evoparam

PACKAGE = pathlib.Path(evoparam.__file__).parent
MODULES = ["reports", "plots", "cli.py"]
FORBIDDEN_IMPORTS = ("scipy",)
FORBIDDEN_CALLS = ("linalg", "fft", "polyfit", "lstsq", "convolve", "gradient")


def _files():
    for name in MODULES:
        path = PACKAGE / name
        yield from ([path] if path.suffix == ".py" else sorted(path.rglob("*.py")))


def test_no_numerical_code_in_the_presentation_layers():

    problems = []
    for path in _files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) \
                    else [node.module or ""]
                if any(n.split(".")[0] in FORBIDDEN_IMPORTS for n in names):
                    problems.append(f"{path.name}: imports {names}")
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_CALLS:
                problems.append(f"{path.name}:{node.lineno}: uses .{node.attr}")
    assert not problems, problems
