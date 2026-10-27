# SPDX-License-Identifier: Apache-2.0
"""REQ-307: no silent failure.

A failure injected on one channel is recorded on that channel (``status="failed"``, a
message, one WARNING naming it) while the other channels are fitted; ``on_error="raise"``
propagates a ``FitError``; data and settings errors propagate; no bare ``except`` and no
``except Exception: pass``; ``np.errstate(all="ignore")`` is never used.
"""

import ast
import logging
import pathlib
import re
from typing import Any

import numpy as np
import pytest

import evoparam
from evoparam.errors import DataError, FitError, SettingsError
from evoparam.models import ResponseModel
from evoparam.models import engine as _engine

# Channel whose fit fails, and the most lines with numpy.errstate.
FAILED_CHANNEL = 2
MAX_ERRSTATE_LINES = 3

PACKAGE = pathlib.Path(evoparam.__file__).resolve().parent


@pytest.fixture
def evoked() -> Any:
    from evoparam.data import Evoked

    rng = np.random.default_rng(1)
    times = np.arange(-0.2, 0.5, 1 / 512)
    data = rng.normal(0, 1, (4, times.size)) - 25 * np.exp(-0.5 * ((times - 0.03) / 0.004) ** 2)
    return Evoked(data=data, times=times, sfreq=512.0, ch_names=["C1", "C2", "C3", "C4"])


@pytest.fixture
def inject(monkeypatch: Any, evoked: Any) -> None:
    """The fit of channel C3 raises (an exception from inside the optimiser path)."""

    original = _engine.fit_waveform_core

    def failing(times, y, **kwargs):
        if np.isclose(y[0], evoked.data[2, 0]):
            raise np.linalg.LinAlgError("injected: singular Jacobian")
        return original(times, y, **kwargs)

    monkeypatch.setattr(_engine, "fit_waveform_core", failing)


@pytest.mark.usefixtures("inject")
def test_failed_unit_is_recorded(evoked: Any, caplog: Any) -> None:
    with caplog.at_level(logging.WARNING, logger="evoparam"):
        group = ResponseModel("ccep", max_components=2).fit_group(evoked)
    status = [f.status for f in group.fits]
    assert status[FAILED_CHANNEL] == "failed"
    assert group.fits[FAILED_CHANNEL].message
    assert "singular Jacobian" in group.fits[FAILED_CHANNEL].message
    assert all(s != "failed" for i, s in enumerate(status) if i != FAILED_CHANNEL)
    warned = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "singular Jacobian" in r.getMessage()
    ]
    assert len(warned) == 1
    assert "C3" in warned[0].getMessage()


@pytest.mark.usefixtures("inject")
def test_on_error_raise_propagates(evoked: Any) -> None:
    with pytest.raises(FitError, match="C3"):
        ResponseModel("ccep", max_components=2).fit_group(evoked, on_error="raise")
    with pytest.raises(SettingsError):
        ResponseModel("ccep").fit_group(evoked, on_error="ignore")


def test_data_and_settings_errors_propagate(evoked: Any) -> None:
    with pytest.raises(DataError):
        ResponseModel("ccep").fit_waveform(evoked.times, evoked.data[0][:-1])
    with pytest.raises(SettingsError):
        ResponseModel(max_components=2).fit_waveform(evoked.times, evoked.data[0])


def _sources() -> Any:
    for path in sorted(PACKAGE.rglob("*.py")):
        if "tests" not in path.relative_to(PACKAGE).parts:
            yield path, path.read_text(encoding="utf-8")


def test_no_bare_except_and_no_swallowed_exception() -> None:
    hits = []
    for path, text in _sources():
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    hits.append(f"{path.name}:{node.lineno} bare except")
                elif (
                    getattr(node.type, "id", "") in ("Exception", "BaseException")
                    and len(node.body) == 1
                    and isinstance(node.body[0], ast.Pass)
                ):
                    hits.append(f"{path.name}:{node.lineno} except Exception: pass")
    assert not hits, hits


def test_errstate_never_ignores_everything() -> None:
    hits = [
        f"{path.name}:{n}"
        for path, text in _sources()
        for n, line in enumerate(text.splitlines(), 1)
        if re.search(r"errstate\([^)]*all\s*=\s*['\"]ignore", line)
    ]
    assert not hits, hits


def _errstate_lines() -> Any:
    """``grep -rn "errstate" evoparam/``: every text file of the package (the spec keeps its
    tests outside ``evoparam/``, so ``evoparam/tests`` is left out; compiled files too)."""
    hits = []
    for path in sorted(PACKAGE.rglob("*")):
        rel = path.relative_to(PACKAGE)
        if not path.is_file() or rel.parts[0] == "tests" or "__pycache__" in rel.parts:
            continue
        raw = path.read_bytes()
        if b"\0" in raw:  # grep -I
            continue
        text = raw.decode("utf-8", errors="replace")
        hits += [
            f"{rel.as_posix()}:{n}"
            for n, line in enumerate(text.splitlines(), 1)
            if "errstate" in line
        ]
    return hits


def test_errstate_at_most_three_lines_in_the_engine() -> None:
    lines = _errstate_lines()
    assert len(lines) <= MAX_ERRSTATE_LINES, f"{len(lines)} lines"
    assert all(line.startswith("models/_engine.py") for line in lines), f"{len(lines)} lines"


###################################################################################################
## REQ-307 (ii): fit_trials, test_response, remove_artifact fail on C3 only, as fit_group


C3 = 2  # the index of the failing channel
WINDOW = (-0.002, 0.008)  # the artifact window of remove_artifact
TRIALS_NDIM = 3  # trials x channels x times


@pytest.fixture
def trials() -> Any:
    from evoparam.data import TrialSet

    rng = np.random.default_rng(2)
    times = np.arange(-0.2, 0.5, 1 / 512)
    wave = -25 * np.exp(-0.5 * ((times - 0.03) / 0.004) ** 2)
    data = rng.normal(0, 3, (20, 4, times.size)) + wave
    return TrialSet(data=data, times=times, sfreq=512.0, ch_names=["C1", "C2", "C3", "C4"])


def _has_c3(values: Any, trials: Any) -> bool:
    """Whether the array of trials ``values`` [..., n_times] holds the trials of C3."""
    target = trials.data[:, C3]
    values = np.asarray(values)
    if values.ndim == TRIALS_NDIM:
        return any(np.array_equal(values[:, c], target) for c in range(values.shape[1]))
    return np.array_equal(values, target)


def _injector(monkeypatch: Any, module: Any, name: Any, position: Any, trials: Any) -> Any:
    """Make ``module.name`` raise when its argument ``position`` holds the trials of C3."""
    original = getattr(module, name)
    error: dict[str, Any] = {"raise": np.linalg.LinAlgError("injected: singular matrix")}

    def failing(*args: Any, **kwargs: Any) -> Any:
        if _has_c3(args[position], trials):
            raise error["raise"]
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, failing)
    return error


def _warned_once(caplog: Any) -> None:
    warned = [
        r for r in caplog.records if r.levelno == logging.WARNING and "injected" in r.getMessage()
    ]
    assert len(warned) == 1, [r.getMessage() for r in warned]
    assert "C3" in warned[0].getMessage()


def _unit_function(kind: str, monkeypatch: Any, trials: Any) -> Any:
    """(call, error switch) for the unit function ``kind``, with C3 failing."""
    if kind == "fit_trials":
        from evoparam.trials import api as trials_api

        group = ResponseModel("ccep", max_components=2, random_state=0).fit_group(trials)
        error = _injector(monkeypatch, trials_api, "_null_jitter", 2, trials)
        return (
            lambda **kw: trials_api.fit_trials(trials, group, random_state=0, **kw),
            error,
        )
    if kind == "test_response":
        from evoparam.existence import api as existence_api

        error = _injector(monkeypatch, existence_api, "permutation_test", 0, trials)
        return (
            lambda **kw: existence_api.test_response(
                trials, n_permutations=99, n_boot=20, random_state=0, **kw
            ),
            error,
        )
    from evoparam.artifact import api as artifact_api

    error = _injector(monkeypatch, artifact_api, "interpolate_window", 1, trials)
    return (
        lambda **kw: artifact_api.remove_artifact(
            trials, method="interpolate", window=WINDOW, **kw
        ),
        error,
    )


UNIT_FUNCTIONS = ("fit_trials", "test_response", "remove_artifact")


@pytest.mark.parametrize("kind", UNIT_FUNCTIONS)
def test_failed_unit_is_recorded_by_every_unit_function(
    kind: str, monkeypatch: Any, trials: Any, caplog: Any
) -> None:
    call, _ = _unit_function(kind, monkeypatch, trials)
    with caplog.at_level(logging.WARNING, logger="evoparam"):
        result = call()
    _warned_once(caplog)
    if kind == "fit_trials":
        by_channel = {fit.channel: fit for fit in result.fits}
        failed = by_channel["C3"]
        assert failed.status == "failed"
        assert "singular matrix" in failed.message
        assert np.isnan(failed.latency).all()
        assert not failed.trusted.any()
        assert "C3" not in set(result.table["channel"])
        others = [fit for name, fit in by_channel.items() if name != "C3"]
        assert others
        assert all(fit.status == "ok" for fit in others)
        assert all(not fit.message for fit in others)
        assert all(np.isfinite(fit.latency).any() for fit in others)
    elif kind == "test_response":
        assert result.status.tolist() == ["ok", "ok", "failed", "ok"]
        assert "singular matrix" in result.message[2]
        assert np.isnan(result.p_value[2])
        assert not result.significant[2]
        assert np.isfinite(np.delete(result.p_value, 2)).all()
        assert all(not m for i, m in enumerate(result.message) if i != C3)
    else:
        assert result.status.tolist() == ["ok", "ok", "failed", "ok"]
        assert "singular matrix" in result.message[2]
        window = (trials.times >= WINDOW[0]) & (trials.times <= WINDOW[1])
        np.testing.assert_array_equal(result.cleaned.data[:, 2], trials.data[:, 2])
        assert result.cleaned.masks[:, 2][:, window].all()
        # the other channels are cleaned as when nothing fails
        monkeypatch.undo()
        clean = call()
        keep = [0, 1, 3]
        np.testing.assert_array_equal(result.cleaned.data[:, keep], clean.cleaned.data[:, keep])
        np.testing.assert_array_equal(result.cleaned.masks[:, keep], clean.cleaned.masks[:, keep])


@pytest.mark.parametrize("kind", UNIT_FUNCTIONS)
def test_on_error_raise_propagates_on_every_unit_function(
    kind: str, monkeypatch: Any, trials: Any
) -> None:
    call, _ = _unit_function(kind, monkeypatch, trials)
    with pytest.raises(FitError, match="C3"):
        call(on_error="raise")
    with pytest.raises(SettingsError):
        call(on_error="ignore")


@pytest.mark.parametrize("kind", UNIT_FUNCTIONS)
def test_data_errors_are_not_recorded_by_unit_functions(
    kind: str, monkeypatch: Any, trials: Any
) -> None:
    call, error = _unit_function(kind, monkeypatch, trials)
    error["raise"] = DataError("injected: bad data")
    with pytest.raises(DataError, match="bad data"):
        call()


def test_on_error_on_every_unit_function() -> None:
    import inspect

    from evoparam.artifact import remove_artifact
    from evoparam.existence import test_response
    from evoparam.trials import fit_trials

    for func in (fit_trials, test_response, remove_artifact):
        parameter = inspect.signature(func).parameters["on_error"]
        assert parameter.default == "record", func.__name__
