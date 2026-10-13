# SPDX-License-Identifier: Apache-2.0
"""REQ-301: static typing of the public API (spec §9.6).

The distribution ships ``evoparam/py.typed``; ``mypy`` checks one example per public entry
point (``typing_examples/*.py``, a user shape class that satisfies ``ComponentShape``
without inheriting included) and must reject the deliberately wrong examples
(``typing_examples/wrong_*.py``). The examples pass under ``--strict`` and
``pyright --verifytypes`` reports >= 95 % known symbols; ``mypy --strict evoparam`` passes
with the type stubs of the lock files.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any

import pytest

import evoparam

ROOT = pathlib.Path(evoparam.__file__).resolve().parent.parent
EXAMPLES = pathlib.Path(__file__).resolve().parent / "typing_examples"
GOOD = sorted(p for p in EXAMPLES.glob("*.py") if not p.name.startswith("wrong_"))
WRONG = sorted(EXAMPLES.glob("wrong_*.py"))

pytestmark = pytest.mark.skipif(
    not (ROOT / "pyproject.toml").exists(), reason="needs the source tree"
)


def _mypy(files: Any, tmp_path: Any, *flags: Any, config: Any = None) -> Any:
    pytest.importorskip("mypy")
    if config is None:
        config = tmp_path / "mypy.ini"
        config.write_text("[mypy]\n", encoding="utf-8")
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])),
    )
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(config),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--no-error-summary",
            *flags,
            *map(str, files),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
        timeout=600,
        check=False,
    )
    return done.returncode, done.stdout + done.stderr


def test_req301_py_typed_is_shipped() -> None:
    assert (ROOT / "evoparam" / "py.typed").is_file()
    try:
        import tomllib
    except ImportError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    with open(ROOT / "pyproject.toml", "rb") as fobj:
        config = tomllib.load(fobj)["tool"]
    assert "py.typed" in config["setuptools"]["package-data"]["evoparam"]
    mypy = config["mypy"]
    assert mypy["strict"] is True
    for override in mypy["overrides"]:  # only optional packages are ignored
        assert override["ignore_missing_imports"] is True
        assert "evoparam" not in " ".join(override["module"])


def test_req301_examples_cover_the_entry_points() -> None:
    text = "\n".join(p.read_text(encoding="utf-8") for p in GOOD)
    for entry in (
        "evoparam.fit",
        "ResponseModel",
        "TrialSet",
        "Evoked",
        "test_response",
        "remove_artifact",
        "fit_trials",
        "evoparam.read",
        "evoparam.reproduce",
        "evoparam.selftest",
        "evoparam.cite",
        "evoparam.verbose",
        "register_shape",
        "ComponentShape",
        "load_object",
    ):
        assert entry in text, entry
    user_shape = (EXAMPLES / "user_shape.py").read_text(encoding="utf-8")
    assert "class Triangle:" in user_shape  # no base class: structural typing


def test_req301_good_examples_type_check(tmp_path: Any) -> None:
    code, out = _mypy(
        GOOD,
        tmp_path,
        "--disallow-untyped-defs",
        "--warn-unused-ignores",
        "--ignore-missing-imports",
    )
    assert code == 0, out


@pytest.mark.parametrize("wrong", [p.name for p in WRONG])
def test_req301_wrong_example_fails(wrong: Any, tmp_path: Any) -> None:
    code, out = _mypy([EXAMPLES / wrong], tmp_path, "--ignore-missing-imports")
    assert code == 1, out
    assert f"{wrong}:" in out, out
    assert "error:" in out, out


def test_req301_the_user_shape_satisfies_the_protocol_at_run_time() -> None:
    import importlib.util

    from evoparam.shapes import ComponentShape

    spec = importlib.util.spec_from_file_location("user_shape", EXAMPLES / "user_shape.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert isinstance(module.Triangle(), ComponentShape)


def test_req301_good_examples_pass_under_strict(tmp_path: Any) -> None:
    code, out = _mypy(GOOD, tmp_path, "--strict", "--ignore-missing-imports")
    assert code == 0, out


def _stubs_missing() -> list[str]:
    """The type-stub distributions of the CI lock files (ci/locks/) that are not installed."""
    from importlib import metadata

    missing = []
    for name in ("scipy-stubs", "types-PyYAML", "types-tabulate"):
        try:
            metadata.distribution(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
    return missing


@pytest.mark.skipif(
    os.environ.get("EVOPARAM_SLOW_TESTS") != "1",
    reason="about two minutes; set EVOPARAM_SLOW_TESTS=1",
)
def test_req301_mypy_strict_on_the_package(tmp_path: Any) -> None:
    """``mypy --strict evoparam`` passes in the environment of the lock files, which hold the
    stubs of scipy, PyYAML and tabulate (47 [import-untyped] errors without them)."""
    missing = _stubs_missing()
    if missing:
        pytest.skip(f"the stubs of the lock files are not installed: {missing}")
    code, out = _mypy(
        [ROOT / "evoparam"],
        tmp_path,
        "--strict",
        "--exclude",
        "evoparam/tests/",
        config=ROOT / "pyproject.toml",
    )
    assert code == 0, out[-2000:]


def test_req301_pyright_verifytypes_known_share() -> None:
    pyright = shutil.which("pyright") or os.path.expanduser("~/.local/bin/pyright")
    if not os.path.exists(pyright):
        pytest.skip("pyright is not installed")
    done = subprocess.run(
        [
            pyright,
            "--pythonpath",
            sys.executable,
            "--verifytypes",
            "evoparam",
            "--ignoreexternal",
            "--outputjson",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=900,
        env=dict(os.environ, PYTHONPATH=str(ROOT)),
        check=False,
    )
    score = json.loads(done.stdout)["typeCompleteness"]["completenessScore"]
    assert score >= 0.95, score
