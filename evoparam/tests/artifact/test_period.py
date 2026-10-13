# SPDX-License-Identifier: Apache-2.0
"""Period-based reconstruction for high-frequency trains (REQ-062)."""

from typing import Any

import numpy as np
import pytest
from scipy.stats import ranksums

from evoparam.artifact import estimate_period, remove_artifact
from evoparam.data import TrialSet
from evoparam.errors import ArtifactError, SettingsError

# SD and bias (samples) of the period estimates, tolerance of the period found, and the
# level of the rank-sum test.
MAX_PERIOD_SD = 1e-4
MAX_PERIOD_ERROR = 1e-3
ALPHA = 0.05

###################################################################################################
###################################################################################################

SFREQ, FREQ = 1000.0, 130.2  # a real DBS frequency is not a round number


def train(n: Any, period: Any, rng: Any, amplitude: Any = 300.0) -> Any:
    """A repeating artifact: a decaying biphasic shape every ``period`` samples."""

    phase = np.mod(np.arange(n) + rng.uniform(0, period), period)
    return amplitude * np.exp(-phase / 1.2) * np.sin(2 * np.pi * phase / period * 2)


def test_req062_fractional_period_precision() -> None:

    rng = np.random.default_rng(0)
    period = SFREQ / FREQ
    estimates = []
    for _ in range(100):
        n = int(3 * SFREQ)
        t = np.arange(n) / SFREQ
        x = (
            train(n, period, rng)
            + 20 * np.sin(2 * np.pi * 10 * t)
            + 10 * np.sin(2 * np.pi * 50 * t)
            + rng.normal(0, 5, n)
        )
        estimates.append(estimate_period(x, SFREQ, 130.0))
    assert np.std(estimates) <= MAX_PERIOD_SD
    assert abs(np.mean(estimates) - period) <= MAX_PERIOD_ERROR


def test_req062_residual_like_prestimulation_noise() -> None:

    rng = np.random.default_rng(1)
    n_pre, n_stim = int(5 * SFREQ), int(20 * SFREQ)
    n = n_pre + n_stim
    times = np.arange(n) / SFREQ - 5.0
    signal = 20 * np.sin(2 * np.pi * 10 * times) + 10 * np.sin(2 * np.pi * 50 * times)
    noise = rng.normal(0, 5, (2, n))
    art = np.zeros((2, n))
    art[:, n_pre:] = np.array([[1.0], [0.6]]) * train(n_stim, SFREQ / FREQ, rng)
    ts = TrialSet(data=(signal + noise + art)[None], times=times, sfreq=SFREQ, ch_names=["a", "b"])

    result = remove_artifact(
        ts, method="period_template", expected_rate=130.0, baseline=(-4.9, -0.1)
    )
    period = result.settings["_info"]["period_samples"][0]
    assert abs(period - SFREQ / FREQ) < MAX_PERIOD_ERROR

    margin = int(3 * SFREQ)  # the filter reaches about 2.5 s to each side
    stim = slice(n_pre + margin, n - margin)
    error = (result.cleaned.data[0] - signal)[:, stim]
    pre = noise[:, : n_pre - margin]
    step = 13  # thin out to roughly independent samples
    for ch in range(2):
        stat = ranksums(np.abs(error[ch, ::step]), np.abs(pre[ch, ::step]))
        assert stat.pvalue > ALPHA
    # The artifact was large: it is gone
    assert np.std(error) < 1.1 * np.std(pre)


def test_causal_variant_uses_past_only() -> None:

    rng = np.random.default_rng(2)
    n = int(8 * SFREQ)
    x = train(n, SFREQ / FREQ, rng) + rng.normal(0, 5, n)
    ts = TrialSet(
        data=np.tile(x, (1, 1, 1)), times=np.arange(n) / SFREQ, sfreq=SFREQ, ch_names=["a"]
    )
    result = remove_artifact(
        ts,
        method="period_template",
        period=1 / FREQ,
        causal=True,
        baseline=(0.0, 1.0),
        quality=False,
    )
    # A change after sample t never alters the cleaned value at t
    later = ts.data.copy()
    later[..., 5000:] += 1000.0
    from dataclasses import replace

    moved = remove_artifact(
        replace(ts, data=later),
        method="period_template",
        period=1 / FREQ,
        causal=True,
        baseline=(0.0, 1.0),
        quality=False,
    )
    assert np.allclose(moved.cleaned.data[..., :5000], result.cleaned.data[..., :5000])


def test_period_errors() -> None:

    with pytest.raises(ArtifactError, match="not resolvable"):
        estimate_period(np.zeros(1000), 250.0, 130.0)
    with pytest.raises(ArtifactError, match="10 periods"):
        estimate_period(np.zeros(20), 1000.0, 130.0)
    with pytest.raises(SettingsError):
        estimate_period(np.zeros((2, 100)), 1000.0, 130.0)
    ts = TrialSet(
        data=np.zeros((1, 1, 1000)), times=np.arange(1000) / SFREQ, sfreq=SFREQ, ch_names=["a"]
    )
    with pytest.raises(ArtifactError, match="period"):
        remove_artifact(ts, method="period_template", baseline=(0, 0.5))
