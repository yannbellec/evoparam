# SPDX-License-Identifier: Apache-2.0
"""Layer 3, ds004977 (spec §7.3, REQ-372): channels left out of the adjusted common average.

The channels left out by the authors' CARLA code and by evoparam's ``car_adaptive`` are
compared on the same data.

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

The file also records, for the layer-3 comparison records of REQ-374
(``validation/layer3_report.py``): the commit of CARLA_JNM that was run
(``comparator_commit``), the dataset version, and evoparam's own selection on the same
channels (``evoparam_used``: ``car_adaptive_select``, 100 bootstrap resamples, seed 0, as in
the test). ``--evoparam-only`` recomputes evoparam's selection into the existing file without
running MATLAB.
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
    """Return the indices of the channels compared: good, finite, and not next to the site."""
    near = neighbours(site)
    keep = (
        ~trials.ch_info["bad"].to_numpy(bool)
        & np.array([name.upper() not in near for name in trials.ch_names])
        & np.isfinite(trials.data).all(axis=(0, 2))
    )
    return np.flatnonzero(keep)


def run_matlab(trials, index, carla, matlab):
    """Return the channels kept in the common average by the authors' CARLA code (MATLAB)."""
    from scipy.io import loadmat, savemat

    with tempfile.TemporaryDirectory() as work:
        mat, res = os.path.join(work, "in.mat"), os.path.join(work, "out.mat")
        savemat(
            mat,
            {
                "V": np.transpose(trials.data[:, index], (1, 2, 0)),
                "tt": trials.times[None],
                "srate": trials.sfreq,
            },
        )
        command = (
            f"addpath('{os.path.join(carla, 'functions')}'); load('{mat}'); "
            f"rng('default'); [~,~,s]=CARLA(tt,V,srate,true); used=s.chsUsed; "
            f"save('{res}','used');"
        )
        subprocess.run([matlab, "-batch", command], check=True, capture_output=True)
        used = np.ravel(loadmat(res)["used"]).astype(int) - 1
    return sorted(trials.ch_names[index[i]] for i in used)


def evoparam_used(trials, index):
    """Return the channels kept in the common average by evoparam.

    ``car_adaptive_select`` with 100 bootstrap resamples and seed 0, the settings of the
    REQ-372 test.
    """
    from evoparam.prep.reference import car_adaptive_select

    chosen = car_adaptive_select(trials.data[:, index], trials.times, n_boot=100, random_state=0)
    return sorted(trials.ch_names[index[i]] for i in chosen["members"])


EVOPARAM_SELECTION = {
    "function": "evoparam.prep.reference.car_adaptive_select",
    "n_boot": 100,
    "random_state": 0,
}


def evoparam_only():
    """Recompute evoparam's selection into the existing file (no MATLAB)."""
    sys.path.insert(0, os.path.dirname(HERE))
    from evoparam import __version__
    from evoparam.tests.prep.test_reference_ds004977 import load_trials

    with open(OUT) as fobj:
        document = json.load(fobj)
    for site in document["sites"]:
        trials = load_trials(site["subject"], site["run"], site["site"])
        index = np.array([trials.ch_names.index(c) for c in site["channels"]])
        site["evoparam_used"] = evoparam_used(trials, index)
    document["evoparam_selection"] = dict(EVOPARAM_SELECTION, version=__version__)
    document.setdefault("data", {"dataset": "ds004977", "version": "1.2.0"})
    with open(OUT, "w") as fobj:
        json.dump(document, fobj, indent=1)
    print(OUT)


def main():
    """Run both selections on every site and write the result file."""
    sys.path.insert(0, os.path.dirname(HERE))
    from evoparam import __version__
    from evoparam.tests.prep.test_reference_ds004977 import RUNS, load_trials

    if "--evoparam-only" in sys.argv[1:]:
        evoparam_only()
        return
    carla = os.environ["CARLA_JNM"]
    matlab = os.environ.get("MATLAB", "/Applications/MATLAB_R2026a.app/bin/matlab")
    sites = []
    for subject, run, site in RUNS:
        trials = load_trials(subject, run, site)
        index = channels(trials, site)
        sites.append(
            {
                "subject": subject,
                "run": run,
                "site": site,
                "channels": [trials.ch_names[i] for i in index],
                "carla_used": run_matlab(trials, index, carla, matlab),
                "evoparam_used": evoparam_used(trials, index),
            }
        )
    commit = subprocess.run(
        ["git", "-C", carla, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fobj:
        json.dump(
            {
                "comparator": "CARLA_JNM (Huang et al. 2024), CARLA(tt, V, srate, true)",
                "comparator_commit": commit,
                "dataset": "ds004977",
                "data": {"dataset": "ds004977", "version": "1.2.0"},
                "evoparam_selection": dict(EVOPARAM_SELECTION, version=__version__),
                "sites": sites,
            },
            fobj,
            indent=1,
        )
    print(OUT)


if __name__ == "__main__":
    main()
