# SPDX-License-Identifier: Apache-2.0
"""Arguments that were accepted but not used (CHANGELOG, static-quality pass).

- ``verbose`` maps to the level of the ``evoparam`` logger for the duration of the call
  (REQ-304, spec §9.8) in ``evoparam.fit``, ``evoparam.models.fit``, ``test_response``,
  ``fit_trials``, ``fit_hierarchical``, ``tf_fit`` and ``detect_pulses``; ``"progress"``
  adds a ``tqdm`` bar (not in ``fit_hierarchical``, which spec §4.2 keeps away from utils).
- ``n_jobs`` runs the units of work of ``test_response``, ``fit_trials`` and
  ``shared_shapes`` in parallel (joblib when installed, else threads; spec §5 conventions)
  with results identical to ``n_jobs=1``.
- ``prestimulus_dependence(band=)``, ``GroupFit.to_annotations(what=)``,
  ``artifact_quality(template=)``, ``reports.markdown(data=)``, ``cli.run_fit(log=)`` and
  ``fetch_openneuro_file(folder=None)`` do what their docstrings say.
"""

import dataclasses
import json
import logging
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

import evoparam
from evoparam.errors import SettingsError
from evoparam.utils._logging import progress, verbose
from evoparam.utils._parallel import n_workers, parallel_map

LOGGER = logging.getLogger("evoparam")


@pytest.fixture(scope="module")
def ccep() -> Any:
    from evoparam.sim import simulate_ccep

    ts, _ = simulate_ccep(20, n_channels=8, seed=3, artifact=False)
    return ts


@pytest.fixture(scope="module")
def group(ccep: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evoparam.fit(ccep, "ccep", random_state=0, existence=False)


@pytest.fixture(params=["joblib", "threads"])
def backend(request: Any, monkeypatch: Any) -> str:
    """Run the parallel units in joblib processes, or in threads (joblib not importable)."""

    if request.param == "joblib":
        pytest.importorskip("joblib")
    else:
        monkeypatch.setitem(sys.modules, "joblib", None)  # import joblib raises ImportError
    return str(request.param)


def _info_records(caplog: Any, prefix: str) -> list[Any]:
    return [r for r in caplog.records if r.levelno == logging.INFO and r.name.startswith(prefix)]


###################################################################################################
## verbose (REQ-304)


@pytest.fixture
def quiet_logger() -> Any:
    """The ``evoparam`` logger at WARNING for the test, its level restored after."""

    previous = LOGGER.level
    LOGGER.setLevel(logging.WARNING)
    yield LOGGER
    LOGGER.setLevel(previous)


def _check_verbose(caplog: Any, prefix: str, call: Any) -> None:
    """INFO records with verbose="info", none with verbose=None; the level is restored."""

    with caplog.at_level(logging.NOTSET), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        call(None)
        assert not _info_records(caplog, prefix)
        assert LOGGER.level == logging.WARNING
        call("info")
        assert _info_records(caplog, prefix), prefix
        assert LOGGER.level == logging.WARNING


@pytest.mark.usefixtures("quiet_logger")
def test_verbose_fit(caplog: Any, ccep: Any) -> None:
    evoked = ccep.average()
    _check_verbose(
        caplog,
        "evoparam.models",
        lambda v: evoparam.models.fit(evoked, "ccep", random_state=0, verbose=v),
    )
    caplog.clear()
    # evoparam.fit: the fit and the existence test, and the methods paragraph at INFO
    _check_verbose(
        caplog,
        "evoparam.existence",
        lambda v: evoparam.fit(ccep, "ccep", random_state=0, verbose=v),
    )
    assert any("Methods:" in r.getMessage() for r in caplog.records)


@pytest.mark.usefixtures("quiet_logger")
def test_verbose_debug_reaches_the_channel_records(caplog: Any) -> None:
    times = np.arange(-0.2, 0.5, 1 / 512)
    y = -20 * np.exp(-0.5 * ((times - 0.03) / 0.004) ** 2)
    with caplog.at_level(logging.NOTSET), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evoparam.models.fit(y[None], "ccep", times=times, random_state=0, verbose="debug")
    assert any(
        r.levelno == logging.DEBUG and r.name.startswith("evoparam.models") for r in caplog.records
    )
    assert LOGGER.level == logging.WARNING


@pytest.mark.usefixtures("quiet_logger")
def test_verbose_test_response(caplog: Any, ccep: Any) -> None:
    from evoparam.existence import test_response

    _check_verbose(
        caplog,
        "evoparam.existence",
        lambda v: test_response(ccep, random_state=0, n_permutations=100, n_boot=10, verbose=v),
    )


@pytest.mark.usefixtures("quiet_logger")
def test_verbose_fit_trials(caplog: Any, ccep: Any, group: Any) -> None:
    from evoparam.trials import fit_trials

    _check_verbose(caplog, "evoparam.trials", lambda v: fit_trials(ccep, group, verbose=v))


@pytest.mark.usefixtures("quiet_logger")
def test_verbose_fit_hierarchical(caplog: Any) -> None:
    from evoparam.hierarchical import fit_hierarchical
    from evoparam.tests.hierarchical.helpers import simulate_group

    sets, _ = simulate_group(n_subjects=2, n_channels=4, n_trials=10, snr=3.0, seed=1)
    _check_verbose(
        caplog,
        "evoparam.hierarchical",
        lambda v: fit_hierarchical(sets, components=["N1"], verbose=v),
    )
    # spec §4.2 keeps utils out of hierarchical: its own reading of verbose, same values
    with pytest.raises(SettingsError, match="verbose must be one of"):
        fit_hierarchical(sets, components=["N1"], verbose="loud")
    assert LOGGER.level == logging.WARNING


@pytest.mark.usefixtures("quiet_logger")
def test_verbose_tf_fit(caplog: Any) -> None:
    from evoparam.tests.tf.helpers import TIMES, ar1, burst, trialset
    from evoparam.tf import tf_fit

    rng = np.random.default_rng(0)
    data = ar1(rng, (12, 1, len(TIMES)))
    data[:, 0] += burst(rng, 12, freq=80.0, amp=10.0)
    ts = trialset(data)
    _check_verbose(
        caplog,
        "evoparam.tf",
        lambda v: tf_fit(ts, freqs=np.array([60.0, 80.0, 100.0]), zones="none", verbose=v),
    )


@pytest.mark.usefixtures("quiet_logger")
def test_verbose_detect_pulses(caplog: Any) -> None:
    from evoparam.events import detect_pulses

    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, (4, 20_000))
    for onset in np.arange(1, 19, 2.5):
        data[:, int(onset * 1000) : int(onset * 1000) + 3] += 500.0
    _check_verbose(
        caplog,
        "evoparam.events",
        lambda v: detect_pulses(data, 1000.0, method="threshold", verbose=v),
    )
    caplog.clear()
    with caplog.at_level(logging.NOTSET):
        detect_pulses(data, 1000.0, method="threshold", verbose="debug")
    assert any(r.levelno == logging.DEBUG for r in caplog.records)


@pytest.mark.usefixtures("quiet_logger")
def test_verbose_progress_shows_a_bar(capsys: Any, ccep: Any) -> None:
    pytest.importorskip("tqdm")
    evoked = ccep.average()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evoparam.models.fit(evoked, "ccep", random_state=0, verbose=None)
        assert "fit" not in capsys.readouterr().err
        evoparam.models.fit(evoked, "ccep", random_state=0, verbose="progress")
    assert "fit" in capsys.readouterr().err
    assert LOGGER.level == logging.WARNING


def test_progress_is_transparent() -> None:
    items = [3, 1, 2]
    assert list(progress(items)) == items
    with verbose("info"):
        assert list(progress(items, total=3, desc="x")) == items
    with verbose("progress"):
        assert list(progress(iter(items), total=3, desc="x")) == items
        with verbose("info"):  # an explicit level of an inner call: no bar
            assert list(progress(items)) == items


###################################################################################################
## n_jobs: the helper


def test_n_workers() -> None:
    assert n_workers(1, 10) == 1
    assert n_workers(None, 10) == 1
    assert n_workers(0, 10) == 1
    two = 2
    assert n_workers(4, two) == two
    assert n_workers(-1, 1000) >= 1
    assert n_workers(3, 0) == 1


def _square(x: int) -> int:
    if x < 0:
        raise ValueError(f"negative {x}")
    return x * x


def _warn_and_log(x: int) -> int:
    from evoparam.errors import warn

    warn(f"unit {x} warns", UserWarning)
    logging.getLogger("evoparam.tests").info("unit %d logs", x)
    return x


@pytest.mark.usefixtures("backend")
def test_parallel_map_order_errors_warnings_and_records(caplog: Any) -> None:
    assert parallel_map(_square, range(7), 3) == [x * x for x in range(7)]
    with pytest.raises(ValueError, match="negative"):
        parallel_map(_square, [1, -2, 3], 2)
    with (
        caplog.at_level(logging.INFO, logger="evoparam"),
        pytest.warns(UserWarning, match="unit") as caught,
    ):
        assert parallel_map(_warn_and_log, [1, 2, 3], 2) == [1, 2, 3]
    assert sorted(str(w.message) for w in caught) == [f"unit {x} warns" for x in (1, 2, 3)]
    messages = [r.getMessage() for r in caplog.records]
    for x in (1, 2, 3):
        assert f"unit {x} logs" in messages
        assert sum(f"unit {x} warns" in m for m in messages) == 1  # one WARNING per unit


###################################################################################################
## n_jobs: identical results


def _same_test(a: Any, b: Any) -> None:
    for name in ("statistic", "p_value", "p_value_raw", "onset", "offset", "duration", "snr_db"):
        assert np.array_equal(getattr(a, name), getattr(b, name), equal_nan=True), name
    assert json.dumps(a.settings, default=str) == json.dumps(b.settings, default=str)


@pytest.mark.parametrize(
    "options",
    [
        {"method": "baseline_permutation"},
        {"method": "baseline_permutation", "statistic": "cluster", "by": ()},
        {"method": "projection"},
    ],
)
@pytest.mark.usefixtures("backend")
def test_test_response_n_jobs(options: Any, ccep: Any) -> None:
    from evoparam.existence import test_response

    kwargs = {"random_state": 5, "n_permutations": 200, "n_boot": 30, **options}
    _same_test(test_response(ccep, **kwargs), test_response(ccep, n_jobs=2, **kwargs))


def test_test_response_n_jobs_evoked_and_model_comparison(ccep: Any) -> None:
    from evoparam.existence import test_response

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evoked = ccep.average()
        _same_test(
            test_response(evoked, random_state=2, n_permutations=300),
            test_response(evoked, random_state=2, n_permutations=300, n_jobs=3),
        )
        kwargs = {"method": "model_comparison", "random_state": 2, "n_permutations": 3}
        kwargs["picks"] = [0, 1, 2]
        _same_test(test_response(ccep, **kwargs), test_response(ccep, n_jobs=3, **kwargs))


@pytest.mark.usefixtures("backend")
def test_fit_trials_n_jobs(ccep: Any, group: Any) -> None:
    from evoparam.trials import fit_trials

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = fit_trials(ccep, group, covariates=("prestim",))
        b = fit_trials(ccep, group, covariates=("prestim",), n_jobs=2)
    pd.testing.assert_frame_equal(a.table, b.table)
    assert [f.channel for f in a.fits] == [f.channel for f in b.fits]
    for fa, fb in zip(a.fits, b.fits, strict=True):
        for name in ("shift", "scale", "latency", "amplitude", "fwhm", "gof", "params"):
            assert np.array_equal(getattr(fa, name), getattr(fb, name), equal_nan=True)
        assert fa.jitter_sd == fb.jitter_sd or np.isnan(list(fa.jitter_sd.values())).all()
        assert fa.settings == fb.settings
    assert a.skipped_channels == b.skipped_channels
    assert json.dumps(a.settings, default=str) == json.dumps(b.settings, default=str)


def test_fit_trials_n_jobs_warnings_reach_the_caller(
    ccep: Any, group: Any, caplog: Any, recwarn: Any
) -> None:
    """A refused realignment warns in the unit of its channel (REQ-307: one WARNING)."""
    from evoparam.trials import fit_trials

    warnings.simplefilter("always")
    with caplog.at_level(logging.WARNING, logger="evoparam"):
        fit_trials(ccep, group, realign=True)
    serial = sorted(str(w.message) for w in recwarn if "realignment refused" in str(w.message))
    records = sorted(r.getMessage() for r in caplog.records if "realignment" in r.getMessage())
    recwarn.clear()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="evoparam"):
        fit_trials(ccep, group, realign=True, n_jobs=2)
    assert serial
    assert (
        sorted(str(w.message) for w in recwarn if "realignment refused" in str(w.message))
        == serial
    )
    assert sorted(r.getMessage() for r in caplog.records if "realignment" in r.getMessage()) == (
        records
    )


@pytest.mark.parametrize("method", ["nmf", "kpca"])
@pytest.mark.usefixtures("backend")
def test_shared_shapes_n_jobs(method: Any) -> None:
    from evoparam.spatial import shared_shapes
    from evoparam.tests.spatial.helpers import multi_site

    ts, _ = multi_site(4)
    a = shared_shapes(ts, random_state=11, method=method)
    b = shared_shapes(ts, random_state=11, method=method, n_jobs=3)
    assert a.n_shapes == b.n_shapes
    assert np.array_equal(a.curves, b.curves)
    assert np.array_equal(a.weights, b.weights)
    pd.testing.assert_frame_equal(a.assignment, b.assignment)
    assert json.dumps(a.settings, default=str) == json.dumps(b.settings, default=str)


###################################################################################################
## Arguments that are used now


def test_prestimulus_dependence_band(ccep: Any, group: Any) -> None:
    from evoparam.trials import fit_trials, prestimulus_dependence

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tf = fit_trials(ccep, group, covariates=("prestim",), prestim_band=(10.0, 20.0))
    assert tf.settings["prestim_band"] == [10.0, 20.0]
    with pytest.raises(SettingsError, match=r"prestim_band=\(10.0, 20.0\)"):
        prestimulus_dependence(tf, n_permutations=10)  # default band (8, 13)
    table = prestimulus_dependence(tf, band=(10.0, 20.0), n_permutations=10, random_state=0)
    assert table.attrs["band"] == [10.0, 20.0]
    # a table carries no band: it is recorded, not checked
    assert prestimulus_dependence(tf.table, n_permutations=10).attrs["band"] == [8.0, 13.0]
    with pytest.raises(SettingsError, match="band must be"):
        prestimulus_dependence(tf, band=(13.0, 8.0))


def test_to_annotations_what(group: Any) -> None:
    with pytest.raises(SettingsError, match="what must be 'components'"):
        group.to_annotations(what="windows")
    pytest.importorskip("mne")
    annotations = group.to_annotations(what="components")
    assert len(annotations) == sum(len(f.components) for f in group.fits)


def test_artifact_quality_template_gives_the_fit_r2() -> None:
    from evoparam.artifact import artifact_quality, remove_artifact
    from evoparam.artifact._common import window_slice
    from evoparam.sim import simulate_ccep

    ts, _ = simulate_ccep(12, n_channels=4, seed=1, artifact=True)
    window = (-0.002, 0.008)
    win = window_slice(ts, window)
    tpl = np.median(ts.data[:, :, win], axis=0)  # (channels, samples)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = remove_artifact(ts, method="template", window=window, template=tpl, scale=False)
    cleaned = result.cleaned
    expected = np.asarray(result.quality_components["template_fit_r2"], float)
    table = artifact_quality(cleaned, window=window, template=tpl)
    assert np.allclose(table["template_fit_r2"], expected)
    # given fit_r2 wins; no template, no r2
    assert np.allclose(
        artifact_quality(cleaned, window=window, template=tpl, fit_r2=np.ones(12))[
            "template_fit_r2"
        ],
        1.0,
    )
    assert artifact_quality(cleaned, window=window)["template_fit_r2"].isna().all()
    with pytest.raises(SettingsError, match="broadcast"):
        artifact_quality(cleaned, window=window, template=np.ones(3))


def test_markdown_data(group: Any, ccep: Any) -> None:
    from evoparam import reports

    masks = np.zeros(ccep.data.shape, bool)
    masks[:, :, :10] = True
    masked = dataclasses.replace(ccep, masks=masks)
    text = reports.markdown(group, data=masked)
    assert "masked fraction" in text
    assert "masked fraction" not in reports.markdown(group)
    with pytest.raises(SettingsError, match="TrialSet or Evoked"):
        reports.markdown(group, data=np.zeros(3))


@pytest.mark.usefixtures("quiet_logger")
def test_cli_run_fit_log(ccep: Any) -> None:
    from evoparam.cli import run_fit

    lines: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_fit(ccep, "ccep", random_state=0, existence=False, log=lines.append)
    assert any(line.startswith("INFO evoparam.models") for line in lines), lines
    assert LOGGER.level == logging.WARNING
    assert not [h for h in LOGGER.handlers if type(h).__name__ == "_Forward"]


def test_fetch_openneuro_file_folder_none(tmp_path: Any, monkeypatch: Any) -> None:
    from evoparam.io import fetch_openneuro_file
    from evoparam.io._fetch import sha256_file

    monkeypatch.setenv("EVOPARAM_DATA", str(tmp_path))
    target = tmp_path / "dsTEST" / "sub-01" / "events.tsv"
    target.parent.mkdir(parents=True)
    target.write_text("onset\n1.0\n", encoding="utf-8")
    path = fetch_openneuro_file(
        "dsTEST", "sub-01/events.tsv", None, sha256=sha256_file(str(target))
    )
    assert path == str(target)


def test_background_without_basis_is_refused() -> None:
    from evoparam.shapes.registry import BACKGROUNDS, check_background

    for name in BACKGROUNDS.builtin:
        check_background(BACKGROUNDS[name])  # the class: some need settings to build

    class ProtocolOnly:
        name = "protocol_only"
        param_names = ("c0",)
        n_params = 1

        def evaluate(self, times: Any, params: Any) -> Any:
            return np.broadcast_to(params[..., :1], params.shape[:-1] + (len(times),))

        def jacobian(self, times: Any, params: Any) -> Any:
            return np.ones(params.shape[:-1] + (len(times), 1))

        def default_bounds(self, times: Any, data_range: Any = None) -> Any:  # noqa: ARG002  (protocol)
            return np.array([-1.0]), np.array([1.0])

        def initial_guess(self, times: Any, waveform: Any) -> Any:  # noqa: ARG002  (protocol)
            return np.zeros(1)

    with pytest.raises(SettingsError, match="basis"):
        check_background(ProtocolOnly())
