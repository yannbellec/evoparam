"""Electrode layouts and stimulation-artifact waveforms for the simulators.

The stimulation artifact of an electrical pulse is modelled as the potential of
the stimulating current in a resistive medium (a current source and sink at the
two stimulating contacts, potential ``∝ 1/r+ − 1/r−``) times the recorded pulse
shape: the rectangular current pulse (biphasic or monophasic) seen through the
amplifier's anti-aliasing filter, followed by a double-exponential recovery tail.
Sub-sample pulses are resolved on a grid finer than the sampling period.
"""

import numpy as np
from scipy import signal as sp_signal

from evoparam.errors import SettingsError

__all__ = ["seeg_layout", "grid_layout", "line_layout", "scalp_layout", "dipole_gain",
           "pulse_shape", "tail_shape", "PULSE_SHAPES"]

# Contact spacing (mm): sEEG shafts, ECoG grids, DBS leads.
SEEG_SPACING = 3.5
GRID_SPACING = 10.0
SHAFT_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"

# Recorded pulse shapes (scaled by the potential of each channel).
PULSE_SHAPES = ("biphasic", "monophasic")


###################################################################################################
## Layouts

def seeg_layout(n_channels, contacts_per_shaft=12):
    """Contacts on parallel sEEG shafts: names ``A01``…, coordinates (mm), shaft names."""

    names, coords, groups = [], [], []
    for index in range(n_channels):
        shaft, contact = divmod(index, contacts_per_shaft)
        letter = SHAFT_LETTERS[shaft % len(SHAFT_LETTERS)] * (1 + shaft // len(SHAFT_LETTERS))
        names.append(f"{letter}{contact + 1:02d}")
        groups.append(letter)
        coords.append((15.0 * (shaft % 3), 15.0 * (shaft // 3), -SEEG_SPACING * contact))
    return names, np.asarray(coords, dtype=float), groups


def grid_layout(n_channels, n_columns=8):
    """Contacts of an ECoG grid (10 mm spacing): names ``G01``…, coordinates (mm)."""

    names, coords = [], []
    for index in range(n_channels):
        row, column = divmod(index, n_columns)
        names.append(f"G{index + 1:02d}")
        coords.append((GRID_SPACING * column, GRID_SPACING * row, 0.0))
    return names, np.asarray(coords, dtype=float), ["G"] * n_channels


def line_layout(n_channels, spacing, prefix, depth=True):
    """Contacts on a line (laminar probe, DBS lead, cochlear or spinal array)."""

    names = [f"{prefix}{index:02d}" for index in range(n_channels)]
    position = spacing * np.arange(n_channels, dtype=float)
    coords = np.zeros((n_channels, 3))
    coords[:, 2 if depth else 0] = -position if depth else position
    return names, coords, [prefix] * n_channels


# 10-20 positions by row (y) and by number (x); odd numbers on the left.
_SCALP_ROWS = {"FP": 0.92, "AF": 0.75, "F": 0.55, "FC": 0.3, "C": 0.0, "T": 0.0, "CP": -0.3,
               "TP": -0.3, "P": -0.55, "PO": -0.75, "O": -0.92}
# Midline and central channels first, so that small montages are centred on the vertex.
_SCALP_NAMES = ("Cz", "Pz", "Fz", "C3", "C4", "P3", "P4", "F3", "F4", "Oz", "O1", "O2", "FC1",
                "FC2", "CP1", "CP2", "T7", "T8", "P7", "P8", "F7", "F8", "FC5", "FC6", "CP5",
                "CP6", "Fp1", "Fp2", "TP9", "TP10", "PO9", "PO10")


def _scalp_xy(name):
    letters = name.rstrip("0123456789z").upper()
    rest = name[len(letters):]
    y = _SCALP_ROWS[letters]
    if rest.lower() == "z":
        return 0.0, y
    number = int(rest)
    x = {1: 0.18, 2: 0.18, 3: 0.4, 4: 0.4, 5: 0.62, 6: 0.62, 7: 0.85, 8: 0.85,
         9: 0.95, 10: 0.95}[number]
    radius = max(1.0, float(np.hypot(x, y)))          # the outer ring stays on the disc
    return (-x if number % 2 else x) / radius, y / radius


def scalp_layout(n_channels):
    """Scalp EEG channels: 10-20 names and positions on the unit disc (up to 32), then a
    Fibonacci spiral of ``E33``… for larger montages. Coordinates in units of the head
    radius (z = 0)."""

    names, coords = [], []
    for index in range(n_channels):
        if index < len(_SCALP_NAMES):
            name = _SCALP_NAMES[index]
            x, y = _scalp_xy(name)
        else:
            name = f"E{index + 1:02d}"
            radius = np.sqrt((index + 0.5) / n_channels)
            angle = index * np.pi * (3.0 - np.sqrt(5.0))
            x, y = radius * np.cos(angle), radius * np.sin(angle)
        names.append(name)
        coords.append((x, y, 0.0))
    return names, np.asarray(coords, dtype=float), ["scalp"] * n_channels


###################################################################################################
## Potentials and waveforms

def dipole_gain(coords, source, sink=None, radius=1.0):
    """Potential of a current source (and sink) at each contact, ``r0/r+ − r0/r−``.

    Distances below ``radius`` (mm, about the contact size) are clamped to it, so
    the stimulating contacts get the largest values (±1).
    """

    coords = np.asarray(coords, dtype=float)
    plus = np.maximum(np.linalg.norm(coords - np.asarray(source, dtype=float), axis=1), radius)
    out = radius / plus
    if sink is not None:
        minus = np.maximum(np.linalg.norm(coords - np.asarray(sink, dtype=float), axis=1), radius)
        out = out - radius / minus
    return out


def _box_samples(start, stop, grid, step):
    """Fraction of each fine bin [grid, grid + step) inside [start, stop)."""

    return np.clip((np.minimum(grid + step, stop) - np.maximum(grid, start)) / step, 0.0, 1.0)


def pulse_shape(times, sfreq, *, pulse_width_us, kind="biphasic", polarity=-1, onset=0.0,
                cutoff_fraction=0.25, filter_order=4, n_after=64):
    """Recorded shape of a rectangular current pulse of unit height.

    The current (first phase of sign ``polarity`` lasting ``pulse_width_us``,
    then, for a biphasic pulse, the opposite phase) is integrated on a grid at
    least 16 times finer than the pulse, passed through a causal Butterworth
    low-pass at ``cutoff_fraction × sfreq`` (the amplifier's anti-aliasing
    filter) and read at the sample times.

    Returns
    -------
    ndarray, shape (n_times,)
    """

    if kind not in PULSE_SHAPES:
        raise SettingsError(f"pulse kind must be one of {PULSE_SHAPES}, got {kind!r}.")
    width = float(pulse_width_us) * 1e-6
    if not width > 0:
        raise SettingsError(f"pulse_width_us must be positive, got {pulse_width_us}.")
    times = np.asarray(times, dtype=float)
    out = np.zeros(times.size)
    first = int(np.searchsorted(times, onset - 1e-12)) - 4
    first = max(first, 0)
    last = min(times.size, first + 8 + n_after + int(np.ceil(2 * width * sfreq)))
    if last - first < 2:
        return out
    factor = max(4, int(np.ceil(16.0 / (width * sfreq))))
    step = 1.0 / (sfreq * factor)
    grid = times[first] + step * np.arange((last - first) * factor)
    current = polarity * _box_samples(onset, onset + width, grid, step)
    if kind == "biphasic":
        current = current - polarity * _box_samples(onset + width, onset + 2 * width, grid, step)
    sos = sp_signal.butter(filter_order, cutoff_fraction * sfreq, fs=sfreq * factor,
                           output="sos")
    filtered = sp_signal.sosfilt(sos, current)
    out[first:last] = filtered[::factor][:last - first]
    return out


def tail_shape(times, sfreq, *, start, tau1, tau2, weight=0.7, cutoff_fraction=0.25):
    """Double-exponential recovery ``w e^(-u/tau1) + (1-w) e^(-u/tau2)`` from ``start``.

    It rises with the time constant of the anti-aliasing filter
    (``1 / (2π cutoff)``), so it starts at 0 instead of with a step.
    """

    if not (tau1 > 0 and tau2 > 0 and 0 <= weight <= 1):
        raise SettingsError("tail: tau1 and tau2 must be positive and weight in [0, 1].")
    u = np.asarray(times, dtype=float) - start
    after = u > 0
    u = np.where(after, u, 0.0)
    rise = 1.0 - np.exp(-u * 2.0 * np.pi * cutoff_fraction * sfreq)
    decay = weight * np.exp(-u / tau1) + (1.0 - weight) * np.exp(-u / tau2)
    return np.where(after, rise * decay, 0.0)
