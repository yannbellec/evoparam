# SPDX-License-Identifier: Apache-2.0
"""TrialSet and Evoked built from arrays; averaging, selection, saving."""

import numpy as np

from evoparam.data import Evoked, TrialSet, load_object


def build() -> Evoked:
    times = np.arange(-100, 400) / 1000.0
    data = np.zeros((20, 3, times.size))
    trials = TrialSet(data=data, times=times, sfreq=1000.0, ch_names=["A", "B", "C"])
    picked: TrialSet = trials.select(picks=["A", "B"], tmin=-0.05, tmax=0.3)
    return picked.average()


def round_trip(evoked: Evoked, path: str) -> Evoked:
    saved = evoked.save(path)
    loaded = load_object(saved)
    assert isinstance(loaded, Evoked)
    return loaded
