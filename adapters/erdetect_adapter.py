# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Yann V. Bellec. See LICENSE in this folder.
"""ER-detect on one T1 or T2 task input of the evoparam benchmark.

Usage::

    python erdetect_adapter.py <task input .json> <output .tsv>

ER-detect (van Blooijs et al. 2023, GPL-3.0, ``pip install erdetect``) detects the N1 of
an averaged CCEP: the most negative peak 9-100 ms after the pulse whose amplitude exceeds
3.4 times the standard deviation of the baseline (-1 to -0.1 s), with the tool's default
configuration. This script averages the trials of the input, passes the average to the
tool's detection function and writes:

- T1: one row per non-stimulated channel, ``responsive`` = the tool's decision,
  ``p_value`` = n/a (the tool gives none), ``score`` = -(peak amplitude) / baseline SD
  (the tool's own criterion, for the ROC);
- T2: one row per channel the tool detects, ``latency`` (s) and ``amplitude`` (as the tool
  returns it).

NOT VERIFIED against an installed ER-detect: the call ``ieeg_detect_er(data,
stim_onset_index, sampling_rate)`` on a (channels, conditions, samples) array returning
(peak indices, peak amplitudes) follows the tool's documentation and must be checked with
the installed version before a result is published. ``ERDETECT_FUNCTION``
(``module:function``) overrides the entry point.
"""

import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _taskio import read_task, scored_channels, times, write_tsv  # noqa: E402

ENTRY = os.environ.get("ERDETECT_FUNCTION", "erdetect.core.detection:ieeg_detect_er")
BASELINE = (-1.0, -0.1)


def _detector():
    module, _, name = ENTRY.partition(":")
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as err:
        sys.exit(f"erdetect_adapter: cannot load {ENTRY} ({err}); install ER-detect in the "
                 "environment that runs this script (pip install erdetect).")


def main(argv):
    if len(argv) != 2:
        sys.exit("usage: erdetect_adapter.py <task input .json> <output .tsv>")
    input_path, output_path = argv
    descriptor, arrays = read_task(input_path)
    task = descriptor["task"]
    if task not in ("T1", "T2"):
        sys.exit(f"erdetect_adapter: task {task} is not answered (T1, T2 only).")
    data = arrays["data"]                                  # trials x channels x samples
    average = np.nanmean(data, axis=0)
    t = times(descriptor, average.shape[-1])
    onset = int(np.argmin(np.abs(t)))
    detect = _detector()
    indices, amplitudes = detect(average[:, None, :], onset, float(descriptor["sfreq"]))
    indices = np.asarray(indices, dtype=float).reshape(len(average), -1)[:, 0]
    amplitudes = np.asarray(amplitudes, dtype=float).reshape(len(average), -1)[:, 0]
    base = (t >= BASELINE[0]) & (t <= BASELINE[1])
    sd = np.std(average[:, base], axis=1, ddof=1)
    rows = []
    for i, channel in scored_channels(descriptor):
        detected = bool(np.isfinite(indices[i]) and np.isfinite(amplitudes[i]))
        if task == "T1":
            score = -amplitudes[i] / sd[i] if detected and sd[i] > 0 else None
            rows.append({"site": descriptor["site"], "channel": channel,
                         "responsive": int(detected), "p_value": None, "score": score})
        elif detected:
            k = int(indices[i])
            rows.append({"site": descriptor["site"], "channel": channel,
                         "latency": float(t[k]), "amplitude": float(amplitudes[i])})
    write_tsv(output_path, task, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
