# SPDX-License-Identifier: Apache-2.0
"""REQ-064 with real artifact templates: ``subspace`` on synthetic EEG overlaid with the
measured artifacts of real recordings.

Two sets of real templates, read from OpenNeuro (segments cached in the evoparam data
folder; runs only with EVOPARAM_NETWORK_TESTS=1):

- TMS-EEG, the modality of the published method: ds002094, the 20 subjects of task
  ``tmseeg1`` (30 scalp channels at 5 000 Hz, DC). For each subject, 20 s are read from
  1 s before the first ``R128`` marker; the template of each channel is the median over the
  pulses of the signal on the ``tep`` preset's artifact window (-2 to 12 ms) minus its level
  from -30 to -10 ms.
- Intracranial SPES: ds004080, the 23 runs at 2048 Hz of the events validation
  (``tests/events/test_ds004080_events.py``), 32 channels spread evenly over the recorded
  ones, window -2 to 8 ms.

Synthetic EEG as in ``test_subspace.py``: 30 trials of spatially correlated noise (5 uV)
with a response (onset 12 ms) of random topography; each trial receives the template times
a gain in [0.9, 1.1]. The criterion, signal-to-interference improvement >= 30 dB on the
artifact window and up to 50 ms, is checked on every subject or run.

Measured (both versions on the same overlays, CHANGELOG "REQ-064"):

- the default of ``remove_artifact`` (directions above the whitened noise level,
  ``n_components="noise"``; projection of the window only, ``subspace_extent="window"``):
  TMS 59.4 to 60.7 dB, SPES 48.6 to 55.0 dB, every subject and run above 30 dB;
- the specification's version (95 % of the whitened window energy, at most 6 directions,
  every sample projected): TMS median 1.6 dB (0.2 to 4.1), SPES median 9.3 dB (5.0 to
  18.4). A real template has a different time course on every channel, so its spatial rank
  is high, while one whitened direction already holds 95 % of the energy. Projecting every
  sample away from enough directions reaches 30 dB (45 dB on TMS) only by removing most of
  the signal (relative error 0.95 to 1.20 on the 12 to 50 ms window, against 0.02 to 0.03 for the
  default).
"""

import os
import re
from typing import Any

import numpy as np
import pytest

from evoparam.artifact import compare_methods
from evoparam.data import TrialSet

from .helpers import response, times_axis

# Signal-to-interference improvement (dB) required (REQ-064), and runs of ds004080.
MIN_GAIN_DB = 30
N_RUNS = 23

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
        reason="set EVOPARAM_NETWORK_TESTS=1 to read OpenNeuro",
    ),
]

TMS_SUBJECTS = [
    "283",
    "711",
    "718",
    "728",
    "734",
    "735",
    "739",
    "743",
    "749",
    "759",
    "761",
    "764",
    "766",
    "767",
    "768",
    "774",
    "779",
    "780",
    "783",
    "786",
]
TMS_WINDOW = (-0.002, 0.012)
SPES_WINDOW = (-0.002, 0.008)
SPAN = 20.0

###################################################################################################
###################################################################################################


def _tms_segment(subject: Any) -> Any:
    """20 s of ds002094 task tmseeg1 from 1 s before the first pulse (cached)."""

    from evoparam.io import openneuro_url, read_bids_events, read_brainvision_segment
    from evoparam.io._fetch import data_folder

    cache = os.path.join(data_folder(), "artifact-validation", "ds002094")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"sub-{subject}_task-tmseeg1_{SPAN:g}s.npz")
    if not os.path.exists(path):
        stem = f"sub-{subject}/eeg/sub-{subject}_task-tmseeg1"
        events = read_bids_events(openneuro_url("ds002094", stem + "_events.tsv"))
        onsets = np.array([float(ev["onset"]) for ev in events if ev.get("value") == "R128"])
        tmin = max(onsets.min() - 1.0, 0.0)
        data, sfreq, names, t0 = read_brainvision_segment(
            openneuro_url("ds002094", stem + "_eeg.vhdr"), tmin, tmin + SPAN
        )
        keep = (onsets >= t0 + 0.5) & (onsets <= t0 + data.shape[1] / sfreq - 0.5)
        np.savez_compressed(
            path,
            data=data.astype(np.float32),
            sfreq=sfreq,
            names=np.array(names),
            t0=t0,
            onsets=onsets[keep],
        )
    seg = np.load(path)
    return (
        seg["data"].astype(float),
        float(seg["sfreq"]),
        [str(n) for n in seg["names"]],
        float(seg["t0"]),
        seg["onsets"],
    )


def _pulse_artifacts(data: Any, sfreq: Any, t0: Any, onsets: Any, window: Any) -> Any:
    """Per pulse: the signal on ``window`` minus its level from -30 to -10 ms."""

    lo, hi = int(round(window[0] * sfreq)), int(round(window[1] * sfreq))
    pre = int(round(0.01 * sfreq))
    out = []
    for onset in onsets:
        i0 = int(round((onset - t0) * sfreq))
        level = np.median(data[:, i0 - 3 * pre : i0 - pre], axis=1, keepdims=True)
        out.append(data[:, i0 + lo : i0 + hi + 1] - level)
    return np.array(out)


def tms_template(subject: Any) -> Any:
    data, sfreq, names, t0, onsets = _tms_segment(subject)
    return np.median(_pulse_artifacts(data, sfreq, t0, onsets, TMS_WINDOW), axis=0), sfreq


def spes_template(stem: Any, n_channels: Any = 32) -> Any:
    from evoparam.tests.events.test_ds004080_events import _segment

    data, sfreq, names, t0, onsets, _ = _segment(stem)
    keep = [
        i
        for i, n in enumerate(names)
        if re.match(r"^[A-Za-z]+\d+$", n)
        and not n.upper().startswith(("MKR", "ECG", "EMG", "EOG"))
    ]
    arts = _pulse_artifacts(data[keep], sfreq, t0, onsets, SPES_WINDOW)
    finite = np.flatnonzero(np.isfinite(arts).all(axis=(0, 2)))
    pick = finite[np.linspace(0, len(finite) - 1, n_channels).round().astype(int)]
    return np.median(arts[:, pick], axis=0), sfreq


def sir_improvement(
    template: Any, sfreq: Any, window: Any, seed: Any = 0, n_trials: Any = 30, **options: Any
) -> Any:
    """Overlay the template on synthetic EEG and return the SIR improvement of subspace
    (``options``: keyword arguments of ``remove_artifact``)."""

    rng = np.random.default_rng(seed)
    n_channels = template.shape[0]
    times = times_axis(sfreq)
    mixing = rng.normal(0, 1, (n_channels, n_channels)) / np.sqrt(n_channels)
    eeg = np.einsum("ij,kjt->kit", mixing, rng.normal(0, 5, (n_trials, n_channels, len(times))))
    topo = rng.normal(0, 1, n_channels)
    clean = eeg + topo[None, :, None] * response(times, 0.012, -20.0)[None, None]
    start = int(np.searchsorted(times, window[0] - 1e-12))
    noisy = clean.copy()
    noisy[:, :, start : start + template.shape[1]] += (
        rng.uniform(0.9, 1.1, (n_trials, 1, 1)) * template[None]
    )
    names = [f"E{i}" for i in range(n_channels)]

    def make(d):
        return TrialSet(data=d, times=times, sfreq=sfreq, ch_names=names)

    table = compare_methods(
        make(noisy),
        [{"method": "subspace", "window": window, **options}],
        truth=make(clean),
        window_eval=(window[1], 0.05),
    )
    return float(table["sir_improvement_db"].iloc[0])


SPEC_VERSION = {"n_components": "energy", "subspace_extent": "all"}
# The default must beat the specification's version by more than this (dB) on every overlay.
MIN_MARGIN_DB = 20


def _spes_runs() -> Any:
    from evoparam.tests.events.test_ds004080_events import RUNS, _segment

    return [stem for stem in RUNS if abs(_segment(stem)[1] - 2048.0) <= 1]


def test_req064_real_tms_templates() -> None:

    gains = [sir_improvement(*tms_template(subject), TMS_WINDOW) for subject in TMS_SUBJECTS]
    assert min(gains) >= MIN_GAIN_DB, np.round(gains, 1)


def test_req064_real_spes_templates() -> None:

    runs = _spes_runs()
    assert len(runs) == N_RUNS
    gains = [sir_improvement(*spes_template(stem), SPES_WINDOW) for stem in runs]
    assert min(gains) >= MIN_GAIN_DB, np.round(gains, 1)


def _compare_versions(templates: Any, window: Any) -> Any:
    """SIR improvement of the default and of the specification's version on each overlay."""

    return np.array(
        [
            (sir_improvement(*t, window), sir_improvement(*t, window, **SPEC_VERSION))
            for t in templates
        ]
    )


def test_req064_real_tms_templates_default_beats_specification_version() -> None:
    """REQ-064 is met by the default (test above). The specification's version is kept as
    an option (``n_components="energy"``, ``subspace_extent="all"``): on every real TMS
    template it stays far below 30 dB (measured median 1.6 dB, 0.2 to 4.1), which is why
    the default departs from it (spec-floor rule: a departure must do measurably better;
    CHANGELOG "REQ-064")."""

    gains = _compare_versions([tms_template(s) for s in TMS_SUBJECTS], TMS_WINDOW)
    assert (gains[:, 0] >= MIN_GAIN_DB).all(), np.round(gains, 1)
    assert (gains[:, 0] > gains[:, 1] + MIN_MARGIN_DB).all(), np.round(gains, 1)


def test_req064_real_spes_templates_default_beats_specification_version() -> None:
    """As above on the 23 SPES runs of ds004080 (specification's version: median 9.3 dB,
    5.0 to 18.4)."""

    gains = _compare_versions([spes_template(stem) for stem in _spes_runs()], SPES_WINDOW)
    assert (gains[:, 0] >= MIN_GAIN_DB).all(), np.round(gains, 1)
    assert (gains[:, 0] > gains[:, 1] + MIN_MARGIN_DB).all(), np.round(gains, 1)
