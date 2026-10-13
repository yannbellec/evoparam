# SPDX-License-Identifier: Apache-2.0
"""Overlay benchmark for the artifact tests: artifact-free epochs with known responses,
plus an artifact of known shape (spec §5.3 acceptance criteria)."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from evoparam.data import TrialSet

pd = pytest.importorskip("pandas")

###################################################################################################
###################################################################################################


def times_axis(sfreq: Any, tmin: Any = -0.5, tmax: Any = 0.5) -> Any:
    return np.arange(int(round(tmin * sfreq)), int(round(tmax * sfreq))) / sfreq


def response(times: Any, onset: Any = 0.012, amplitude: Any = -80.0, width: Any = 0.004) -> Any:
    """A negative component that starts at ``onset`` (gamma-like rise, smooth decay)."""

    rel = np.clip(times - onset, 0, None)
    shape = (rel / width) ** 2 * np.exp(-rel / width)
    return amplitude * shape / shape.max()


def artifact_shape(
    times: Any, sfreq: Any, amplitude: Any = 3000.0, tail: Any = 500.0, tau: Any = 0.002
) -> Any:
    """Sharp biphasic pulse at t = 0 and an exponential recovery tail."""

    art = np.zeros_like(times)
    i0 = int(np.searchsorted(times, 0.0))
    pulse = np.array([1.0, -0.7, 0.3, -0.1])
    width = max(1, int(round(0.0005 * sfreq)))
    pulse = np.repeat(pulse, width)
    art[i0 : i0 + len(pulse)] = amplitude * pulse
    start = i0 + len(pulse)
    art[start:] += tail * np.exp(-(times[start:] - times[start]) / tau)
    return art


@dataclass(frozen=True)
class OverlayDesign:
    """The keywords of :func:`make_overlay`: the simulated recording, artifact and response."""

    sfreq: Any = 2048.0
    n_trials: Any = 20
    n_channels: Any = 8
    onset: Any = 0.012
    amplitude: Any = -80.0
    width: Any = 0.004
    noise: Any = 5.0
    art_amplitude: Any = 3000.0
    art_tail: Any = 500.0
    art_tau: Any = 0.002
    jitter: Any = 0.1
    channel_jitter: Any = True
    seed: Any = 0
    metadata: Any = True


def make_overlay(**keywords: Any) -> Any:
    """Return (with_artifact, truth) TrialSets that differ only by the artifact.

    With ``channel_jitter`` the artifact size varies independently per trial and channel;
    without it, it varies per trial only (a spatially fixed artifact, as from one
    stimulation site).

    Keywords: the fields of :class:`OverlayDesign`.
    """

    design = OverlayDesign(**keywords)
    rng = np.random.default_rng(design.seed)
    times = times_axis(design.sfreq)
    gains = rng.uniform(0.5, 1.5, design.n_channels)
    clean = (
        rng.normal(0, design.noise, (design.n_trials, design.n_channels, len(times)))
        + gains[None, :, None]
        * response(times, design.onset, design.amplitude, design.width)[None, None]
    )
    art = artifact_shape(
        times, design.sfreq, design.art_amplitude, design.art_tail, design.art_tau
    )
    scale = rng.uniform(
        1 - design.jitter,
        1 + design.jitter,
        (design.n_trials, design.n_channels if design.channel_jitter else 1, 1),
    )
    spatial = rng.uniform(0.3, 1.0, (1, design.n_channels, 1))
    noisy = clean + scale * spatial * art[None, None]
    meta = None
    if design.metadata:
        meta = pd.DataFrame(
            {
                "stim_site": ["C1-C2"] * design.n_trials,
                "stim_current_mA": [4.0] * design.n_trials,
                "stim_polarity": ["biphasic_cathodic_first"] * design.n_trials,
            }
        )
    names = [f"ch{i:02d}" for i in range(design.n_channels)]

    def make(data):
        return TrialSet(
            data=data,
            times=times,
            sfreq=design.sfreq,
            ch_names=names,
            metadata=None if meta is None else meta.copy(),
            event_times=np.arange(design.n_trials) * 5.0,
        )

    return make(noisy), make(clean)
