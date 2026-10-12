# SPDX-License-Identifier: Apache-2.0
"""REQ-007 (provenance everywhere): every result can be regenerated from its saved
provenance with ``evoparam.reproduce`` (REQ-333).

Each result type is made with its defaults (``random_state=None``: the seed drawn is
recorded), saved, and regenerated; the regenerated result must equal the saved one to 1e-9.
"""

import os
import warnings
from typing import Any

import pytest

import evoparam
from evoparam import sim
from evoparam.data import save_object

pytest.importorskip("pandas")


@pytest.fixture(scope="module")
def trials() -> Any:
    ts, _ = sim.simulate_ccep(n_trials=10, n_channels=3, p_response=1.0, seed=4)
    return ts


def _group(ts: Any) -> Any:
    return evoparam.fit(ts, preset="ccep")


BUILDERS = {
    "GroupFit": _group,
    "ResponseTest": lambda ts: evoparam.test_response(
        ts, window=(0.009, 0.5), baseline=(-0.5, -0.05), by=()
    ),
    "ArtifactResult": lambda ts: evoparam.artifact.remove_artifact(
        ts, method="interpolate", window=(-0.002, 0.006)
    ),
    "Evoked": lambda ts: ts.average(),
}


def _regenerate(obj: Any, path: Any, ts: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evoparam.reproduce(save_object(obj, path), data=ts, rtol=1e-9)


@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_req007_result_is_regenerated(kind: Any, trials: Any, tmp_path: Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = BUILDERS[kind](trials)
    report = _regenerate(obj, str(tmp_path / f"{kind}.evoparam"), trials)
    assert report.identical, report.differences[:5]
    assert report.steps
    assert all(s["status"] in ("replayed", "skipped") for s in report.steps)


@pytest.mark.xfail(
    strict=True,
    reason="fit_trials records no provenance step in its "
    "TrialFit containers (measured: reproduce returns '<not regenerated>' with "
    "no step), and TrialFitSet is saved as a folder of TrialFit containers, not "
    "as one .evoparam result; evoparam/trials is outside this change.",
)
def test_req007_trial_fits_are_regenerated(trials: Any, tmp_path: Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = evoparam.trials.fit_trials(trials, _group(trials))
        folder = fits.save(str(tmp_path / "set"))
        report = evoparam.reproduce(
            os.path.join(folder, "fit_0000.evoparam"), data=trials, rtol=1e-9
        )
    assert report.identical, report.differences[:5]


@pytest.mark.xfail(
    strict=True,
    reason="tf_fit does not draw and record a seed when "
    "random_state is None, and its replay differs (measured 2026-10-09, one "
    "channel, 8 frequencies: 'freqs', 'maps.total' and 'maps.total_norm' "
    "differ; settings random_state None). evoparam/tf is outside this change.",
)
def test_req007_time_frequency_fit_is_regenerated(trials: Any, tmp_path: Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tf = evoparam.tf.tf_fit(
            trials, freqs=(4.0, 100.0, 8), n_permutations=100, max_zones=2, picks=["A01"]
        )
    report = _regenerate(tf, str(tmp_path / "tf.evoparam"), trials)
    assert report.identical, report.differences[:5]
