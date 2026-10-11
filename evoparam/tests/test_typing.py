# SPDX-License-Identifier: Apache-2.0
"""REQ-301: static typing of the public API (spec §9.6).

The distribution ships ``evoparam/py.typed``; ``mypy`` checks one example per public entry
point (``typing_examples/*.py``, a user shape class that satisfies ``ComponentShape``
without inheriting included) and must reject the deliberately wrong examples
(``typing_examples/wrong_*.py``). The strict criteria (``mypy --strict evoparam``, the
examples under ``--strict``, ``pyright --verifytypes`` >= 95 % known symbols) are measured
and kept as strict expected failures with the measured values.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

import evoparam

ROOT = pathlib.Path(evoparam.__file__).resolve().parent.parent
EXAMPLES = pathlib.Path(__file__).resolve().parent / "typing_examples"
GOOD = sorted(p for p in EXAMPLES.glob("*.py") if not p.name.startswith("wrong_"))
WRONG = sorted(EXAMPLES.glob("wrong_*.py"))

pytestmark = pytest.mark.skipif(not (ROOT / "pyproject.toml").exists(),
                                reason="needs the source tree")


def _mypy(files, tmp_path, *flags, config=None):
    pytest.importorskip("mypy")
    if config is None:
        config = tmp_path / "mypy.ini"
        config.write_text("[mypy]\n", encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [
        str(ROOT), os.environ.get("PYTHONPATH")])))
    done = subprocess.run([sys.executable, "-m", "mypy", "--config-file", str(config),
                           "--cache-dir", str(tmp_path / "cache"), "--no-error-summary",
                           *flags, *map(str, files)],
                          capture_output=True, text=True, cwd=str(ROOT), env=env,
                          timeout=600)
    return done.returncode, done.stdout + done.stderr


def test_req301_py_typed_is_shipped():
    assert (ROOT / "evoparam" / "py.typed").is_file()
    try:
        import tomllib
    except ImportError:                  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    with open(ROOT / "pyproject.toml", "rb") as fobj:
        config = tomllib.load(fobj)["tool"]
    assert "py.typed" in config["setuptools"]["package-data"]["evoparam"]
    mypy = config["mypy"]
    assert mypy["strict"] is True
    for override in mypy["overrides"]:              # only optional packages are ignored
        assert override["ignore_missing_imports"] is True
        assert "evoparam" not in " ".join(override["module"])


def test_req301_examples_cover_the_entry_points():
    text = "\n".join(p.read_text(encoding="utf-8") for p in GOOD)
    for entry in ("evoparam.fit", "ResponseModel", "TrialSet", "Evoked", "test_response",
                  "remove_artifact", "fit_trials", "evoparam.read", "evoparam.reproduce",
                  "evoparam.selftest", "evoparam.cite", "evoparam.verbose", "register_shape",
                  "ComponentShape", "load_object"):
        assert entry in text, entry
    user_shape = (EXAMPLES / "user_shape.py").read_text(encoding="utf-8")
    assert "class Triangle:" in user_shape            # no base class: structural typing


def test_req301_good_examples_type_check(tmp_path):
    code, out = _mypy(GOOD, tmp_path, "--disallow-untyped-defs", "--warn-unused-ignores",
                      "--ignore-missing-imports")
    assert code == 0, out


@pytest.mark.parametrize("wrong", [p.name for p in WRONG])
def test_req301_wrong_example_fails(wrong, tmp_path):
    code, out = _mypy([EXAMPLES / wrong], tmp_path, "--ignore-missing-imports")
    assert code == 1 and f"{wrong}:" in out and "error:" in out, out


def test_req301_the_user_shape_satisfies_the_protocol_at_run_time():
    import importlib.util

    from evoparam.shapes import ComponentShape

    spec = importlib.util.spec_from_file_location("user_shape", EXAMPLES / "user_shape.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert isinstance(module.Triangle(), ComponentShape)


@pytest.mark.xfail(strict=True, reason="REQ-301: the good examples do not pass mypy --strict "
                   "(S20, mypy 2.4.0): most public functions are not annotated, so every "
                   "call is 'Call to untyped function ... in typed context' [no-untyped-call]")
def test_req301_good_examples_pass_under_strict(tmp_path):
    code, out = _mypy(GOOD, tmp_path, "--strict", "--ignore-missing-imports")
    assert code == 0, out


@pytest.mark.xfail(strict=True, reason="REQ-301: mypy --strict evoparam reports 5220 errors in "
                   "142 of 163 modules (S20, 2026-10-11, mypy 2.4.0: 2797 no-untyped-call, "
                   "1661 no-untyped-def, 271 attr-defined, 127 type-arg, ...); annotating the "
                   "package is a step of its own")
@pytest.mark.skipif(os.environ.get("EVOPARAM_SLOW_TESTS") != "1",
                    reason="about two minutes; set EVOPARAM_SLOW_TESTS=1")
def test_req301_mypy_strict_on_the_package(tmp_path):
    code, out = _mypy([ROOT / "evoparam"], tmp_path, "--strict", "--exclude",
                      "evoparam/tests/", config=ROOT / "pyproject.toml")
    assert code == 0, out[-2000:]


@pytest.mark.xfail(strict=True, reason="REQ-301: pyright 1.1.414 --verifytypes evoparam "
                   "--ignoreexternal: 35.0 % of the exported symbols have a known type (1307 "
                   "known, 121 ambiguous, 2310 unknown; S20, 2026-10-11) against >= 95 %")
def test_req301_pyright_verifytypes_known_share():
    pyright = shutil.which("pyright") or os.path.expanduser("~/.local/bin/pyright")
    if not os.path.exists(pyright):
        pytest.skip("pyright is not installed")
    done = subprocess.run([pyright, "--pythonpath", sys.executable, "--verifytypes",
                           "evoparam", "--ignoreexternal", "--outputjson"],
                          capture_output=True, text=True, cwd=str(ROOT), timeout=900,
                          env=dict(os.environ, PYTHONPATH=str(ROOT)))
    score = json.loads(done.stdout)["typeCompleteness"]["completenessScore"]
    assert score >= 0.95, score
