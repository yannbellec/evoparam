# SPDX-License-Identifier: Apache-2.0
"""evoparam.reproduce (REQ-333): a saved result is regenerated from its provenance."""

import json
import os
import shutil
import subprocess
import sys
import warnings

import pytest

import evoparam
from evoparam import reports
from evoparam.errors import EvoparamVersionError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(evoparam.__file__)))


@pytest.fixture(scope="module")
def saved(ccep_run, tmp_path_factory):
    """The group fit of the chain simulate -> remove_artifact -> rereference ->
    baseline_correct -> fit -> test_response, saved as a container."""

    folder = tmp_path_factory.mktemp("reproduce")
    return ccep_run.group.save(str(folder / "group"))


def _copy(path, tmp_path):
    target = tmp_path / os.path.basename(path)
    shutil.copytree(path, target)
    return str(target)


def _edit_settings(path, **changes):
    file = os.path.join(path, "settings.json")
    with open(file, encoding="utf-8") as fobj:
        settings = json.load(fobj)
    settings.update(changes)
    with open(file, "w", encoding="utf-8") as fobj:
        json.dump(settings, fobj)


def test_req333_reproduce_is_identical(saved):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        report = reports.reproduce(saved, rtol=1e-9)
    assert report.identical, report.differences[:5]
    assert report.max_rel_diff == 0.0 or report.max_rel_diff <= 1e-9
    assert report.checksums_ok
    assert report.settings_differences == []
    functions = [step["function"] for step in report.steps]
    assert functions[0] == "evoparam.sim.simulate_ccep"
    assert "evoparam.models.ResponseModel.fit_group" in functions
    assert functions[-1] == "evoparam.existence.test_response"
    assert all(step["status"] == "replayed" for step in report.steps)
    assert evoparam.reproduce is reports.reproduce


def test_req333_changed_setting_is_detected_and_named(saved, tmp_path):

    path = _copy(saved, tmp_path)
    _edit_settings(path, max_components=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        report = reports.reproduce(path)
    assert not report.identical
    assert report.settings_differences == ["max_components"]
    fit_step = next(
        s for s in report.steps if s["function"] == "evoparam.models.ResponseModel.fit_group"
    )
    assert "max_components" in fit_step["message"]
    assert report.differences


def test_req333_removed_preset_version_raises(saved, tmp_path):

    path = _copy(saved, tmp_path)
    _edit_settings(path, preset_note="preset 'ccep' v0.1; validation: none")
    with pytest.raises(EvoparamVersionError, match="v0.1"):
        reports.reproduce(path)


def test_req333_environment_differences_are_reported(saved, monkeypatch):
    """A result made under another numpy: reproduce warns, names the difference, and
    still regenerates the numbers."""

    from evoparam.reports import _reproduce

    current = _reproduce.capture_environment()
    monkeypatch.setattr(_reproduce, "capture_environment", lambda: dict(current, numpy="0.0.1"))
    with pytest.warns(UserWarning, match="environment differs"):
        report = reports.reproduce(saved)
    assert "numpy" in report.environment_differences
    assert report.checksums_ok
    assert report.identical


def test_req333_cli_returns_the_same_report(saved):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        report = reports.reproduce(saved).to_dict()
    env = dict(
        os.environ,
        PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
        PYTHONWARNINGS="ignore",
    )
    out = subprocess.run(
        [sys.executable, "-m", "evoparam", "reproduce", saved, "--json"],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        timeout=600,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    summary = json.loads(out.stdout)
    assert summary["command"] == "reproduce"
    assert summary["exit_code"] == 0
    assert summary["result"] == json.loads(json.dumps(report))
