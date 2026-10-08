"""reports, plots and cli compute no scientific quantity of their own (REQ-349).

Together with the import-graph rule (tests/test_import_graph.py), this test reads the source of
these modules and fails on numerical code: scipy, the numerical parts of numpy (linear
algebra, Fourier transforms, statistics, fitting, filtering), and the private numerical
modules of the analysis packages. Every number they show must come from the result or from
a public function of spec §5.1–§5.12. (``evoparam.stream`` is built at step S18.)
"""

import ast
import os

import pytest

PACKAGE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAYERS = ("reports", "plots", "cli.py")

# numpy functions that compute a statistic, a transform or a fit.
NUMERICAL = {
    "linalg", "fft", "polyfit", "polyval", "polynomial", "convolve", "correlate", "gradient",
    "trapz", "trapezoid", "cumsum", "mean", "nanmean", "median", "nanmedian", "std",
    "nanstd", "var", "nanvar", "percentile", "nanpercentile", "quantile", "nanquantile",
    "cov", "corrcoef", "histogram", "lstsq", "interp", "exp", "log", "log10", "random",
    "average", "einsum", "dot", "matmul", "outer", "sinc", "hanning", "hamming", "kaiser",
}
# Analysis packages whose private modules hold the numerical code.
ANALYSIS = ("models", "existence", "artifact", "trials", "prep", "events", "tf", "spatial",
            "hierarchical", "stats", "shapes", "sim", "bands", "analysis")
# Private modules that hold settings or text, not numbers.
ALLOWED_PRIVATE = {"evoparam.models._settings"}


def _files():
    for layer in LAYERS:
        path = os.path.join(PACKAGE, layer)
        if path.endswith(".py"):
            yield path
            continue
        for folder, _, names in os.walk(path):
            for name in sorted(names):
                if name.endswith(".py"):
                    yield os.path.join(folder, name)


def _numpy_attribute(node):
    """The attribute name of ``np.<name>`` / ``numpy.<name>``, else None."""

    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
            and node.value.id in ("np", "numpy"):
        return node.attr
    return None


def _violations(path):
    with open(path, encoding="utf-8") as fobj:
        tree = ast.parse(fobj.read(), path)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "scipy" and alias.name != "scipy":
                    found.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
            if module.split(".")[0] == "scipy":
                found.append((node.lineno, f"from {module} import ..."))
            parts = module.split(".")
            if parts[0] == "numpy" and len(parts) > 1 and parts[1] in NUMERICAL:
                found.append((node.lineno, f"from {module} import ..."))
            if parts[0] == "evoparam" and len(parts) > 2 and parts[1] in ANALYSIS \
                    and parts[2].startswith("_") and module not in ALLOWED_PRIVATE:
                found.append((node.lineno, f"from {module} import ... (private)"))
        name = _numpy_attribute(node)
        if name in NUMERICAL:
            found.append((node.lineno, f"np.{name}"))
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "scipy" and node.attr != "__version__":
            found.append((node.lineno, f"scipy.{node.attr}"))
    return found


@pytest.mark.parametrize("path", list(_files()),
                         ids=lambda p: os.path.relpath(p, PACKAGE).replace(os.sep, "/"))
def test_req349_no_numerical_code(path):

    found = _violations(path)
    assert found == [], (f"{os.path.relpath(path, PACKAGE)}: numerical code in a "
                         f"presentation layer: {found}. Use (or add) a public function of "
                         "the analysis modules instead.")


def test_req349_scanner_detects_numerical_code(tmp_path):

    sample = tmp_path / "sample.py"
    sample.write_text("import numpy as np\nfrom scipy import signal\n"
                      "from evoparam.models._engine import solve\n"
                      "x = np.median([1, 2])\ny = np.linalg.norm([1])\n", encoding="utf-8")
    lines = sorted(line for line, _ in _violations(str(sample)))
    assert lines == [2, 3, 4, 5]
