# SPDX-License-Identifier: Apache-2.0
"""Resampling (REQ-085), averaging and filters."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from evoparam.artifact import apply_mask
from evoparam.errors import DataError, SettingsError
from evoparam.prep import average, highpass, lowpass, notch, resample

from .helpers import ccep, epochs

pd = pytest.importorskip("pandas")

###################################################################################################
###################################################################################################


@pytest.mark.parametrize("new_sfreq", [500.0, 512.0, 2048.0])
def test_req085_latency_and_event_times(new_sfreq: Any) -> None:

    ts = epochs(n_trials=3, sfreq=2000.0, noise=0.0, event_times=np.array([1.5, 3.25, 7.125]))
    data = ts.data + ccep(ts.times, latency=0.0237, width=0.006)[None, None]
    out = resample(replace(ts, data=data), new_sfreq)
    assert out.sfreq == pytest.approx(new_sfreq)
    assert np.array_equal(out.event_times, ts.event_times)
    win = (out.times > 0.005) & (out.times < 0.05)
    latency = out.times[win][np.argmin(out.data[0, 0, win])]
    assert abs(latency - 0.0237) <= 1 / (2 * new_sfreq) + 1e-12
    assert np.min(np.abs(out.times)) < 1e-9  # t = 0 stays on the grid


def test_resample_carries_masks() -> None:

    ts = epochs(n_trials=1, sfreq=1000.0)
    mask = np.zeros(ts.data.shape, dtype=bool)
    mask[..., 500:510] = True
    out = resample(apply_mask(ts, mask), 500.0)
    assert out.masks[0, 0].sum() == 5
    with pytest.raises(SettingsError):
        resample(ts, 0)


def test_average_methods() -> None:

    meta = pd.DataFrame(
        {
            "quality": np.linspace(0.1, 1, 10),
            "rejected": [False] * 9 + [True],
            "condition": ["a", "b"] * 5,
        }
    )
    ts = epochs(n_trials=10, metadata=meta)
    data = ts.data.copy()
    data[-1] += 1e4  # the rejected trial
    ts = replace(ts, data=data)
    mean = average(ts)
    assert np.allclose(mean.data, ts.data[:9].mean(axis=0))
    assert np.allclose(average(ts, method="median").data, np.median(ts.data[:9], axis=0))
    w = meta["quality"].to_numpy()[:9][:, None, None]
    assert np.allclose(average(ts, method="weighted").data, (w * ts.data[:9]).sum(0) / w.sum())
    assert average(ts, method="trimmed").data.shape == (8, ts.n_times)
    assert np.allclose(average(ts, use_rejected=True).data, ts.data.mean(axis=0))
    groups = average(ts, by="condition", method="median")
    assert len(groups) == 2
    median = average(ts, method="median")
    assert np.all(median.n_trials == 9)
    with pytest.raises(SettingsError):
        average(ts, method="mode")
    with pytest.raises(DataError):
        average(replace(ts, metadata=None), method="weighted")


def test_filters() -> None:

    ts = epochs(n_trials=2, sfreq=1000.0, noise=0.0)
    t = ts.times
    data = np.sin(2 * np.pi * 2 * t) + np.sin(2 * np.pi * 60 * t) + np.sin(2 * np.pi * 200 * t)
    ts = replace(ts, data=np.broadcast_to(data, ts.data.shape).copy())
    mid = slice(300, 1200)
    hp = highpass(ts, 20.0)
    assert (
        np.abs(
            hp.data[0, 0, mid] - np.sin(2 * np.pi * 60 * t[mid]) - np.sin(2 * np.pi * 200 * t[mid])
        ).max()
        < 0.1
    )
    lp = lowpass(ts, 100.0)
    assert (
        np.abs(
            lp.data[0, 0, mid] - np.sin(2 * np.pi * 2 * t[mid]) - np.sin(2 * np.pi * 60 * t[mid])
        ).max()
        < 0.1
    )
    nt = notch(ts, 60.0, harmonics=3)
    assert nt.provenance.steps[-1]["settings"]["frequencies"] == [60.0, 120.0, 180.0]
    assert (
        np.abs(
            nt.data[0, 0, mid] - np.sin(2 * np.pi * 2 * t[mid]) - np.sin(2 * np.pi * 200 * t[mid])
        ).max()
        < 0.1
    )
    mask = np.zeros(ts.data.shape, dtype=bool)
    mask[..., 500] = True
    with pytest.warns(UserWarning, match="ring around"):
        lowpass(apply_mask(ts, mask), 100.0)
    with pytest.raises(SettingsError):
        highpass(ts, 600.0)
