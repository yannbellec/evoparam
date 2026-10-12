# SPDX-License-Identifier: Apache-2.0
"""fit_shared keeps every latency in the fit window and applies the identifiability rule of
the engine (no two components of opposite signs overlapping on a channel).

Defect reproduced on OpenNeuro ds004977 (sub-1, run-01, the uncensored stimulation site
ROC11-ROC12, bipolar channels of shaft ROC, preset ``ccep_seeg``): the shared fit of
origin/main gave components at -44 ms (before the stimulation, outside the 9-490 ms window)
and at 543 ms, and pairs of opposite signs at the same latency with amplitudes of 10^4 to
10^5 uV on all 11 channels. The shared latency was bounded by the shape's default bounds
(the window widened by half the largest sigma, 53 ms) and the offsets by ``±max_shift``
only. The real-data test runs with EVOPARAM_NETWORK_TESTS=1.
"""

import os
import warnings
from typing import Any

import numpy as np
import pytest

from evoparam.models import ResponseModel
from evoparam.shapes import FWHM_PER_SIGMA
from evoparam.spatial import fit_shared

from .helpers import BASELINE, shared_channels


def _outside(group: Any) -> Any:
    lo, hi = group.fits[0].window
    return [
        (name, c.latency)
        for name, f in zip(group.settings["fit_shared"]["channels"], group.fits, strict=False)
        for c in f.components
        if not lo <= c.latency <= hi
    ]


def _pairs(group: Any, overlap_c: Any = 0.5) -> Any:
    out = []
    for name, f in zip(group.settings["fit_shared"]["channels"], group.fits, strict=False):
        comps = f.components
        for i, a in enumerate(comps):
            for b in comps[i + 1 :]:
                if (
                    a.amplitude * b.amplitude < 0
                    and abs(a.latency - b.latency) < overlap_c * (a.fwhm + b.fwhm) / FWHM_PER_SIGMA
                ):
                    out.append((name, a.latency, a.amplitude, b.latency, b.amplitude))
    return out


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("free_latency", [False, True])
def test_latencies_stay_in_the_fit_window(seed: Any, free_latency: Any) -> None:
    """A response cut by the window start (true latency 11 ms, FWHM 30 ms, window from
    9 ms): the per-channel fits may put it before the window (edge components, REQ-110),
    the shared fit keeps the latency of every channel in the window."""

    draw, _, _ = shared_channels(seed, n=10, snr=6.0, latency=0.011, fwhm=0.03)
    evoked = draw()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = fit_shared(
            evoked, preset="ccep", baseline=BASELINE, free_latency=free_latency, max_shift=0.005
        )
    assert not _outside(group)


def _ecap(seed: Any) -> Any:
    from evoparam.sim import simulate_ecap

    ts, _ = simulate_ecap(40, kind="cochlear", artifact=False, seed=seed, tmax=0.0035)
    channels = [c for c in ts.ch_names if c != "E00"][:6]
    evoked = ts.average()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # per-channel fits without the rule and without reseeding: they hold cancelling
        # pairs (seeds 0 and 3)
        independent = ResponseModel(
            "ecap_ci", reject_cancelling=False, reseed=False, latency_order="sorted"
        ).fit_group(evoked, picks=channels)
    return evoked, channels, independent


@pytest.mark.parametrize("seed", [0, 3])
def test_no_cancelling_pair_in_the_shared_fit(seed: Any) -> None:

    evoked, channels, independent = _ecap(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = fit_shared(evoked, channels=channels, preset="ecap_ci", group_fit=independent)
        reproduced = fit_shared(
            evoked,
            channels=channels,
            preset="ecap_ci",
            group_fit=independent,
            reject_cancelling=False,
        )
    assert _pairs(reproduced)  # the defect without the rule
    assert not _pairs(group)
    dropped = group.settings["fit_shared"]["dropped"]
    assert dropped
    assert all(d["reason"] == "cancelling_pair" for d in dropped)
    assert not _outside(group)


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
    reason="set EVOPARAM_NETWORK_TESTS=1 to read OpenNeuro",
)
@pytest.mark.parametrize("free_latency", [False, True])
def test_ds004977_shared_fit_in_window_without_cancelling_pairs(free_latency: Any) -> None:

    from evoparam.prep import baseline_correct, rereference
    from evoparam.tests.prep.test_reference_ds004977 import load_trials

    ts = load_trials("1", "01", "ROC11-ROC12")
    ts = ts.select(picks=[n for n in ts.ch_names if n.startswith("ROC")])
    bipolar = rereference(ts, "bipolar", groups=dict.fromkeys(ts.ch_names, "ROC"))
    bipolar = baseline_correct(bipolar, (-0.2, -0.02), check=False)
    average = bipolar.average()
    # the channels that are not the stimulated pair and not censored (zeros)
    channels = [
        n
        for i, n in enumerate(bipolar.ch_names)
        if "ROC11" not in n and "ROC12" not in n and np.nanstd(average.data[i]) > 0
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = fit_shared(
            bipolar,
            channels=channels,
            preset="ccep_seeg",
            fit_window=(0.009, 0.49),
            baseline=(-0.2, -0.02),
            free_latency=free_latency,
        )
    assert not _outside(group)
    assert not _pairs(group)
