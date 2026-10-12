# SPDX-License-Identifier: Apache-2.0
"""Matching-pursuit decomposition of each pulse: line noise, evoked response, artifact.

Specified only from the public abstract and claim summary of the published single-pulse
SEEG method and of the Washington University pre-grant publication US 2023/0240618 A1
(see the specification, section 13): the dictionary holds sinusoids at the line
frequency and its harmonics, Gaussian atoms for the evoked response and decaying
exponentials for the artifact; a greedy decomposition of the residual picks one atom at
a time, and only the artifact atoms are subtracted.
"""

import numpy as np

from evoparam._typing import Any, FloatArray, NDArray
from evoparam.errors import SettingsError
from evoparam.errors import warn as _warn

__all__ = []

_WARNED = False

# Dictionary grid.
N_HARMONICS = 3
GAUSS_WIDTHS = (0.002, 0.005, 0.01, 0.02, 0.05)  # SD of the response atoms (s)
GAUSS_STEP = 0.005  # spacing of their centres (s)
N_DECAYS = 12  # exponential time constants
DECAY_RANGE = (0.0002, 0.05)  # s
DECAY_START_STEP = 0.001  # spacing of their onsets (s)


def warn_patent() -> None:
    global _WARNED  # noqa: PLW0603  (once-only warning flag, reset by the tests)
    if not _WARNED:
        _warn(
            "method 'matching_pursuit' follows a published method that is the "
            "subject of patent publication US 2023/0240618 A1 (Washington "
            "University); check the licensing note in the specification, section "
            "13, before commercial use.",
            UserWarning,
            stacklevel=3,
        )
        _WARNED = True


def dictionary(rel: Any, window_end: Any, line_freq: Any) -> tuple[FloatArray, NDArray[Any]]:
    """Unit-norm atoms on the time axis ``rel`` (s from the pulse), with their kinds."""
    atoms: Any
    atoms, kinds = [], []
    for h in range(1, N_HARMONICS + 1):
        for func in (np.sin, np.cos):
            atoms.append(func(2 * np.pi * h * line_freq * rel))
            kinds.append("line")
    for width in GAUSS_WIDTHS:
        for center in np.arange(window_end, rel[-1], GAUSS_STEP):
            atoms.append(np.exp(-0.5 * ((rel - center) / width) ** 2))
            kinds.append("response")
    starts = np.r_[rel[0], np.arange(0.0, window_end + 1e-12, DECAY_START_STEP)]
    for start in np.unique(np.round(starts, 9)):
        for tau in np.geomspace(*DECAY_RANGE, N_DECAYS):
            atoms.append(np.where(rel >= start, np.exp(-(rel - start) / tau), 0.0))
            kinds.append("artifact")
    atoms = np.array(atoms)
    norms = np.linalg.norm(atoms, axis=1)
    keep = norms > 0
    return atoms[keep] / norms[keep, None], np.array(kinds)[keep]


def pursuit_clean(
    series: FloatArray,
    rel: FloatArray,
    window_end: float,
    *,
    line_freq: float = 50.0,
    n_iter: int = 20,
    tol: float = 0.01,
) -> tuple[Any, FloatArray]:
    """Greedy decomposition of every series; the artifact atoms are subtracted.

    Parameters
    ----------
    series : 2d array [n_series, n_samples]
    rel : 1d array
        Time of each sample relative to the pulse (s).
    window_end : float
        End of the artifact window (s); response atoms start there.
    line_freq : float
    n_iter : int
        Largest number of atoms per series.
    tol : float
        Stop when an atom removes less than this fraction of the series energy.

    Returns
    -------
    cleaned : 2d array
    artifact : 2d array
        The subtracted artifact part.
    """
    if n_iter < 1:
        raise SettingsError("n_iter must be at least 1.")
    atoms, kinds = dictionary(rel, window_end, line_freq)
    is_artifact = kinds == "artifact"
    gram = atoms @ atoms.T
    corr = series @ atoms.T  # correlation of the residual with each atom
    energy = np.sum(series**2, axis=1)
    coefs = np.zeros(corr.shape)
    active = np.ones(len(series), dtype=bool)
    rows = np.arange(len(series))
    for _ in range(n_iter):
        pick = np.argmax(np.abs(corr), axis=1)
        value = corr[rows, pick]
        gain = value**2
        active &= gain > tol * np.maximum(energy, 1e-300)
        if not active.any():
            break
        value = np.where(active, value, 0.0)
        np.add.at(coefs, (rows, pick), value)
        corr -= value[:, None] * gram[pick]
    artifact = coefs[:, is_artifact] @ atoms[is_artifact]
    return series - artifact, artifact
