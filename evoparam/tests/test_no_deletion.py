# SPDX-License-Identifier: Apache-2.0
"""REQ-004: no deletion heuristics.

Components are accepted or rejected only by the model-selection criterion (§5.5.4). This
test reads the source of ``evoparam/models`` and forbids threshold-based deletion of fitted
components: removing a component (``del``, ``.remove``, ``.pop``, a filtering
comprehension) or skipping one under a comparison of its size (amplitude, height, SNR,
area, width) with a threshold. The one exception is the user option ``min_amplitude`` of
§5.5.10 (data units, ``0`` = no restriction, the default), which the test checks is off by
default. A mutation sample checks that the scan finds each forbidden form.
"""

import ast
import pathlib
import re
from typing import Any

import pytest

import evoparam
from evoparam.models import FitSettings

# Fewest modules of evoparam.models.
MIN_FILES = 5

pytestmark = pytest.mark.core

MODELS = pathlib.Path(evoparam.__file__).resolve().parent / "models"
HOLDERS = re.compile(r"comp|shape|peak|cand", re.IGNORECASE)
SIZES = re.compile(r"amplitude|height|snr|area|fwhm|width|beta|prominence|power", re.IGNORECASE)
THRESHOLDS = re.compile(r"thresh|min_|max_|limit|cutoff|cut_off|floor|tol", re.IGNORECASE)
ALLOWED = {"settings.min_amplitude"}  # §5.5.10: user option, 0 = no restriction


def _text(node: Any) -> Any:
    return ast.unparse(node)


def _names(node: Any) -> Any:
    return {_text(n) for n in ast.walk(node) if isinstance(n, (ast.Name, ast.Attribute))}


def _size_threshold_compare(test: Any) -> Any:
    for node in ast.walk(test):
        if isinstance(node, ast.Compare) and any(
            isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in node.ops
        ):
            names = _names(node)
            if names & ALLOWED:
                continue
            sized = any(SIZES.search(n) for n in names)
            bounded = any(THRESHOLDS.search(n) for n in names) or any(
                isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and n.value != 0
                for n in ast.walk(node)
            )
            if sized and bounded:
                return True
    return False


def _drops(body: Any) -> Any:
    """The statements remove or skip something: continue, del, .remove(), .pop()."""

    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, (ast.Continue, ast.Delete)):
                return True
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("remove", "pop")
                and HOLDERS.search(_text(node.func.value))
            ):
                return True
    return False


def scan(source: Any, name: Any = "<source>") -> Any:
    """Forbidden forms in ``source``, as 'name:line: form'."""

    hits = []
    for node in ast.walk(ast.parse(source)):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Delete) and any(HOLDERS.search(_text(t)) for t in node.targets):
            hits.append(f"{name}:{line}: del {_text(node)}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "remove"
            and HOLDERS.search(_text(node.func.value))
        ):
            hits.append(f"{name}:{line}: {_text(node)}")
        elif isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)):
            for gen in node.generators:
                if HOLDERS.search(_text(gen.iter)) and any(
                    _size_threshold_compare(c) for c in gen.ifs
                ):
                    hits.append(f"{name}:{line}: filter {_text(node)[:80]}")
        elif isinstance(node, ast.If) and _size_threshold_compare(node.test) and _drops(node.body):
            hits.append(f"{name}:{line}: if {_text(node.test)[:80]}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "filter"
            and node.args
            and HOLDERS.search(_text(node.args[-1]))
        ):
            hits.append(f"{name}:{line}: {_text(node)[:80]}")
    return hits


def test_req004_models_have_no_deletion_heuristic() -> None:
    hits = []
    files = sorted(MODELS.rglob("*.py"))
    assert len(files) >= MIN_FILES
    for path in files:
        hits += scan(path.read_text(encoding="utf-8"), path.relative_to(MODELS.parent).as_posix())
    assert not hits, "\n".join(hits)


def test_req004_min_amplitude_is_off_by_default() -> None:
    # the only size rule of the engine is the user's min_amplitude (§5.5.10): 0 = none
    assert FitSettings().min_amplitude == 0.0
    engine = (MODELS / "engine.py").read_text(encoding="utf-8")
    # the gate is the truthiness of min_amplitude, whatever the line layout (ruff format)
    assert re.search(r"if\s*\(?\s*settings\.min_amplitude\s+and\b", engine)


MUTATIONS = {
    "del": "def f(components):\n    del components[0]\n",
    "remove": "def f(comps, c):\n    comps.remove(c)\n",
    "filter": "def f(components, threshold):\n"
    "    return [c for c in components if c.amplitude > threshold]\n",
    "snr": "def f(components):\n    return [c for c in components if c.snr >= 3.0]\n",
    "skip": "def f(components, min_height):\n"
    "    out = []\n"
    "    for c in components:\n"
    "        if abs(c.amplitude) < min_height:\n"
    "            continue\n"
    "        out.append(c)\n"
    "    return out\n",
    "pop": "def f(state, peak_threshold):\n"
    "    if state.beta[-1] < peak_threshold:\n"
    "        state.shapes.pop()\n",
    "builtin": "def f(components):\n"
    "    return list(filter(lambda c: c.fwhm > 0.002, components))\n",
}


@pytest.mark.parametrize("form", sorted(MUTATIONS))
def test_req004_scan_finds_each_forbidden_form(form: Any) -> None:
    assert scan(MUTATIONS[form], form)


def test_req004_scan_finds_a_mutation_of_the_engine() -> None:
    engine = (MODELS / "engine.py").read_text(encoding="utf-8")
    # the allowed user option replaced by a threshold of the engine: the scan must see it
    mutated = engine.replace("settings.min_amplitude", "settings.detection_threshold")
    assert mutated != engine
    assert scan(mutated, "engine.py")
