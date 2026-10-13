# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Yann V. Bellec. See LICENSE in this folder.
"""Canonical response parameterization (CRP) on one T1 or T2 task input.

Usage::

    python crp_adapter.py <task input .json> <output .tsv>

CRP (Miller et al. 2023, PLoS Comput Biol; MATLAB scripts, GPL) finds, for each channel,
the response duration over which the single trials share a canonical shape, and tests the
projection of the trials on that shape. This script writes the trials of each
non-stimulated channel (0 to 1 s after the pulse, samples x trials) to a MATLAB file, runs
``crp_wrapper.m`` in MATLAB (or GNU Octave) as a separate program, reads its results back
and writes:

- T1: ``responsive`` = p < 0.05 for the projection test at the response duration,
  ``p_value`` = that p-value, ``score`` = its t-value;
- T2: for responsive channels, ``latency`` and ``amplitude`` of the most negative point of
  the canonical shape scaled by the mean projection weight, 9-100 ms after the pulse.

Environment: ``CRP_ENGINE`` = ``matlab`` (default) or ``octave``; ``CRP_PATH`` = folder of
``CRP_method.m``. Needs SciPy for the MATLAB files.

NOT VERIFIED against the CRP scripts: the call ``[crp_parms, crp_projs] =
CRP_method(V, t_win)`` and the output fields read by ``crp_wrapper.m`` (``C``, ``al``,
``tR``, ``t_value_tR``, ``p_value_tR``) follow the paper and its code as documented, and
must be checked with the scripts before a result is published.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _taskio import read_task, scored_channels, times, write_tsv  # noqa: E402

ALPHA = 0.05
WINDOW = (0.0, 1.0)
N1_WINDOW = (0.009, 0.1)


def _engine_command(script):
    engine = os.environ.get("CRP_ENGINE", "matlab")
    paths = [HERE] + [p for p in [os.environ.get("CRP_PATH")] if p]
    setup = "".join(f"addpath('{p}'); " for p in paths)
    if engine == "octave":
        return ["octave", "--no-gui", "--quiet", "--eval", setup + script]
    return ["matlab", "-batch", setup + script]


def main(argv):
    if len(argv) != 2:
        sys.exit("usage: crp_adapter.py <task input .json> <output .tsv>")
    from scipy.io import loadmat, savemat

    input_path, output_path = argv
    descriptor, arrays = read_task(input_path)
    task = descriptor["task"]
    if task not in ("T1", "T2"):
        sys.exit(f"crp_adapter: task {task} is not answered (T1, T2 only).")
    data = arrays["data"]                                  # trials x channels x samples
    t = times(descriptor, data.shape[-1])
    keep = (t >= WINDOW[0]) & (t <= WINDOW[1])
    channels = scored_channels(descriptor)
    with tempfile.TemporaryDirectory(prefix="crp-") as tmp:
        mat_in, mat_out = os.path.join(tmp, "in.mat"), os.path.join(tmp, "out.mat")
        # V: samples x trials x channels, as CRP_method takes one samples x trials matrix
        V = np.nan_to_num(data[:, [i for i, _ in channels]][:, :, keep]).transpose(2, 0, 1)
        savemat(mat_in, {"V": V, "t_win": t[keep]})
        command = _engine_command(f"crp_wrapper('{mat_in}', '{mat_out}')")
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not os.path.isfile(mat_out):
            sys.exit(f"crp_adapter: {command[0]} failed ({proc.returncode}): "
                     f"{(proc.stderr or proc.stdout)[-2000:]}")
        out = loadmat(mat_out)
    p_values = np.ravel(out["p_value"]).astype(float)
    t_values = np.ravel(out["t_value"]).astype(float)
    shapes = np.atleast_2d(out["shape"]).astype(float)     # channels x samples (0 to tR)
    t_shape = t[keep]
    rows = []
    for k, (_, channel) in enumerate(channels):
        responsive = bool(np.isfinite(p_values[k]) and p_values[k] < ALPHA)
        if task == "T1":
            rows.append({"site": descriptor["site"], "channel": channel,
                         "responsive": int(responsive),
                         "p_value": p_values[k] if np.isfinite(p_values[k]) else None,
                         "score": t_values[k] if np.isfinite(t_values[k]) else None})
            continue
        if not responsive:
            continue
        shape = shapes[k, :len(t_shape)]
        ts = t_shape[:len(shape)]
        inside = (ts >= N1_WINDOW[0]) & (ts <= N1_WINDOW[1]) & np.isfinite(shape)
        if not inside.any():
            continue
        j = np.flatnonzero(inside)[np.argmin(shape[inside])]
        rows.append({"site": descriptor["site"], "channel": channel,
                     "latency": float(t_shape[j]), "amplitude": float(shape[j])})
    write_tsv(output_path, task, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
