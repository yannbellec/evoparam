# SPDX-License-Identifier: Apache-2.0
"""REQ-005 (no silent failure): every public function either returns a result whose
``status`` explains a non-result or raises a documented exception; no NaN parameter appears
with ``status="ok"`` (REQ-118); every warning is also logged."""

import ast
import logging
import pathlib
import warnings
from typing import Any

import numpy as np
import pytest

import evoparam
from evoparam import errors, sim
from evoparam.data import TrialSet
from evoparam.data.results import FIT_STATUSES

# Statuses of the case (at least two kinds).
MIN_STATUSES = 2

PACKAGE = pathlib.Path(evoparam.__file__).resolve().parent
DOCUMENTED = tuple(
    getattr(errors, name)
    for name in errors.__all__
    if isinstance(getattr(errors, name), type) and issubclass(getattr(errors, name), Exception)
)


def _warn_calls(path: Any) -> Any:
    """Calls of warnings.warn (or a bare warn imported from warnings) in a module."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    from_warnings = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "warnings"
        for alias in node.names
        if alias.name == "warn"
    }
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            (
                isinstance(func, ast.Attribute)
                and func.attr == "warn"
                and isinstance(func.value, ast.Name)
                and func.value.id == "warnings"
            )
            or isinstance(func, ast.Name)
            and func.id in from_warnings
        ):
            lines.append(node.lineno)
    return lines


def _modules() -> Any:
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE.parent).as_posix()
        if "/tests/" not in rel and rel != "evoparam/errors.py":
            yield rel, path


def test_req005_every_warning_goes_through_the_logging_helper() -> None:
    hits = [f"{rel}:{line}" for rel, path in _modules() for line in _warn_calls(path)]
    assert not hits, "use evoparam.errors.warn (warns and logs): " + ", ".join(hits)


def test_req005_nwb_reader_warnings_are_logged() -> None:
    path = PACKAGE / "io" / "nwb.py"
    assert not _warn_calls(path)


def test_req005_the_helper_warns_and_logs(caplog: Any) -> None:
    with (
        caplog.at_level(logging.WARNING, logger="evoparam"),
        pytest.warns(errors.NoResponseWarning, match="nothing here"),
    ):
        errors.warn("nothing here", errors.NoResponseWarning)
    assert any("NoResponseWarning: nothing here" in r.getMessage() for r in caplog.records)


def test_req005_warnings_of_a_pipeline_are_logged(caplog: Any) -> None:
    trials, _ = sim.simulate_ccep(n_trials=8, n_channels=3, seed=0)
    evoked = trials.average()
    with (
        caplog.at_level(logging.WARNING, logger="evoparam"),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        # an Evoked: the existence test warns that its null is weaker (no trials)
        evoparam.fit(evoked, preset="ccep", random_state=0)
    logged = " ".join(r.getMessage() for r in caplog.records)
    # the package's warnings are UserWarning subclasses (NumPy's RuntimeWarnings are not)
    ours = [w for w in caught if issubclass(w.category, UserWarning)]
    assert ours
    for w in ours:
        assert str(w.message).splitlines()[0] in logged, str(w.message)


def _statuses_case() -> Any:
    rng = np.random.default_rng(3)
    trials, _ = sim.simulate_ccep(n_trials=10, n_channels=4, seed=3)
    data = trials.data.copy()
    data[:, 1] = rng.normal(0, 1, data[:, 1].shape)  # noise only
    data[:, 2] = 0.0  # flat
    return TrialSet(data=data, times=trials.times, sfreq=trials.sfreq, ch_names=trials.ch_names)


def test_req005_status_explains_every_non_result() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = evoparam.fit(_statuses_case(), preset="ccep", random_state=0)
    statuses = {f.status for f in group.fits}
    assert statuses <= set(FIT_STATUSES)
    assert len(statuses) >= MIN_STATUSES
    for fit_ in group.fits:
        if fit_.status != "ok":
            assert fit_.message, fit_.status
        else:
            # REQ-118: no NaN parameter with status "ok"
            assert fit_.components
            for comp in fit_.components:
                values = [comp.latency, comp.amplitude, comp.fwhm, *np.ravel(comp.params)]
                assert np.all(np.isfinite(values)), comp
    table = group.to_dataframe()
    ok = table[table["status"] == "ok"]
    for column in ("latency", "amplitude", "fwhm"):
        assert ok[column].notna().all(), column


def _bad_calls() -> Any:
    trials, _ = sim.simulate_ccep(n_trials=4, n_channels=3, seed=0)
    times = trials.times
    from evoparam.artifact import remove_artifact
    from evoparam.prep import baseline_correct

    return {
        "fit without times": lambda: evoparam.fit(np.zeros((2, 50)), preset="erp"),
        "fit of a string": lambda: evoparam.fit("not data", preset="erp"),
        "fit unknown preset": lambda: evoparam.fit(trials, preset="no such preset"),
        "fit window outside the data": lambda: evoparam.fit(
            trials, preset="ccep", window=(5.0, 6.0), existence=False
        ),
        "existence with too few trials": lambda: evoparam.test_response(
            trials, window=(0.01, 0.3), baseline=(-0.4, -0.05)
        ),
        "unknown artifact method": lambda: remove_artifact(trials, method="nope"),
        "baseline outside the data": lambda: baseline_correct(trials, (5.0, 6.0)),
        "read an unknown format": lambda: evoparam.read("nothing.unknown-format"),
        "TrialSet with mismatched times": lambda: TrialSet(
            data=np.zeros((2, 3, 10)),
            times=times[:5],
            sfreq=trials.sfreq,
            ch_names=["a", "b", "c"],
        ),
        "unknown preset": lambda: evoparam.presets.get("no such preset"),
    }


@pytest.mark.parametrize("case", sorted(_bad_calls()))
def test_req005_failures_raise_documented_exceptions(case: Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(DOCUMENTED):
            _bad_calls()[case]()
