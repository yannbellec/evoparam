# SPDX-License-Identifier: Apache-2.0
"""Layer 3, ds004977 (spec §7.3, REQ-372): the channels left out of the adjusted common
average by the authors' CARLA code and by evoparam's ``car_adaptive``, on the same data.

The authors' code (Huang et al. 2024, https://github.com/hharveygit/CARLA_JNM, GPL-3.0) is
run as an external comparator in MATLAB; it is not part of evoparam. Give its folder in
``CARLA_JNM`` and MATLAB in ``MATLAB`` (default
``/Applications/MATLAB_R2026a.app/bin/matlab``). As in the authors' real-data script, the
channels are the good sEEG channels outside two contacts of the stimulated pair, the epochs
are those of the uncensored site of one run per subject (the runs of REQ-081), and the
sensitive optimum is used (``rng('default')``). The channel lists of the authors' code are
written to ``validation/layer3/carla_ds004977.json``, which the test
``evoparam/tests/prep/test_carla_ds004977.py`` compares with evoparam (Jaccard index of the
left-out channels, at least 0.8 per site).
"""

import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "layer3", "carla_ds004977.json")


def neighbours(site, n=2):
    """Contacts within ``n`` of each contact of the pair, on the same lead."""

    out = set()
    for ch in site.split("-"):
        match = re.match(r"^([A-Za-z]+)(\d+)$", ch.upper())
        lead, contact = match.group(1), int(match.group(2))
        out |= {f"{lead}{k}" for k in range(contact - n, contact + n + 1)}
    return out


def channels(trials, site):
    near = neighbours(site)
    keep = ~trials.ch_info["bad"].to_numpy(bool) & \
        np.array([name.upper() not in near for name in trials.ch_names]) & \
        np.isfinite(trials.data).all(axis=(0, 2))
    return np.flatnonzero(keep)


def run_matlab(trials, index, carla, matlab):
    from scipy.io import loadmat, savemat

    with tempfile.TemporaryDirectory() as work:
        mat, res = os.path.join(work, "in.mat"), os.path.join(work, "out.mat")
        savemat(mat, {"V": np.transpose(trials.data[:, index], (1, 2, 0)),
                      "tt": trials.times[None], "srate": trials.sfreq})
        command = (f"addpath('{os.path.join(carla, 'functions')}'); load('{mat}'); "
                   f"rng('default'); [~,~,s]=CARLA(tt,V,srate,true); used=s.chsUsed; "
                   f"save('{res}','used');")
        subprocess.run([matlab, "-batch", command], check=True, capture_output=True)
        used = np.ravel(loadmat(res)["used"]).astype(int) - 1
    return sorted(trials.ch_names[index[i]] for i in used)


def main():
    sys.path.insert(0, os.path.dirname(HERE))
    from evoparam.tests.prep.test_reference_ds004977 import RUNS, load_trials

    carla = os.environ["CARLA_JNM"]
    matlab = os.environ.get("MATLAB", "/Applications/MATLAB_R2026a.app/bin/matlab")
    sites = []
    for subject, run, site in RUNS:
        trials = load_trials(subject, run, site)
        index = channels(trials, site)
        sites.append({"subject": subject, "run": run, "site": site,
                      "channels": [trials.ch_names[i] for i in index],
                      "carla_used": run_matlab(trials, index, carla, matlab)})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fobj:
        json.dump({"comparator": "CARLA_JNM (Huang et al. 2024), CARLA(tt, V, srate, true)",
                   "dataset": "ds004977", "sites": sites}, fobj, indent=1)
    print(OUT)


if __name__ == "__main__":
    main()
