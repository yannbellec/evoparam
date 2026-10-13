# SPDX-License-Identifier: Apache-2.0
"""rodent_ep on Allen Visual Coding Neuropixels LFP (REQ-278); runs only with
EVOPARAM_NETWORK_TESTS=1.

DANDI 000021 (Allen Institute, CC-BY 4.0): the LFP of one probe (1 250 Hz) is read remotely by byte
ranges with ``read_nwb`` around the 150 full-field flashes of the session file; the epochs are
cached in the evoparam data folder. Channels in visual cortex (``VIS*``) with a significant
existence test must carry the labelled visual evoked complex (``P1`` or ``N1``, named by sign, or
``late``) in at least 90 % of cases. Measured on 2026-10-08: session 739448407, probes 757968489,
757968491 and 757968495: 16/17, 17/17, 16/16; session 732592105, probe 733744649: 20/20.
"""

import json
import os
import pickle
import re
import warnings
from typing import Any

import pytest

# Trials of the Allen session read, the fewest channels with a response, and the fraction
# of them labelled in the expected area.
N_TRIALS = 150
MIN_SIGNIFICANT = 10
MIN_LABELLED = 0.9

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
        reason="set EVOPARAM_NETWORK_TESTS=1 to read DANDI",
    ),
]

SESSION, PROBE = "739448407", "757968489"
API = "https://api.dandiarchive.org/api"


def _s3(asset: Any) -> Any:
    from evoparam.io._fetch import fetch_bytes

    meta = json.loads(fetch_bytes(f"{API}/assets/{asset}/"))
    return [u for u in meta["contentUrl"] if "s3" in u][0]


def _trials() -> Any:
    import h5py
    import pandas as pd
    import remfile

    from evoparam.io import read_nwb
    from evoparam.io._fetch import data_folder, fetch_bytes

    cache = os.path.join(data_folder(), "allen-validation", f"{SESSION}_{PROBE}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as fobj:
            return pickle.load(fobj)
    listing = json.loads(
        fetch_bytes(
            f"{API}/dandisets/000021/versions/draft/assets/?page_size=50&glob=*ses-{SESSION}*"
        )
    )
    ids = {}
    for asset in listing["results"]:
        match = re.search(r"probe-(\d+)", asset["path"])
        ids[match.group(1) if match else "session"] = asset["asset_id"]
    with h5py.File(remfile.File(_s3(ids["session"])), "r") as session:
        flashes = pd.DataFrame(
            {"start_time": session["intervals/flashes_presentations/start_time"][:]}
        )
    url = _s3(ids[PROBE])
    with h5py.File(remfile.File(url), "r") as probe:
        name = [k for k in probe["acquisition"] if k.endswith("_lfp_data")][0]
    trials = read_nwb(url, electrical_series=name, events=flashes, tmin=-0.1, tmax=0.5)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as fobj:
        pickle.dump(trials, fobj)
    return trials


def test_req278_visual_evoked_complex_on_visual_cortex() -> None:

    from evoparam.existence import test_response
    from evoparam.models import fit
    from evoparam.prep import baseline_correct

    trials = _trials()
    assert trials.n_trials == N_TRIALS
    assert abs(trials.sfreq - 1250) < 1
    visual = [
        n
        for n, loc in zip(trials.ch_names, trials.ch_info["location"], strict=False)
        if str(loc).startswith("VIS")
    ]
    sub = baseline_correct(trials.select(picks=visual), (-0.1, -0.005), check=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test = test_response(
            sub,
            window=(0.005, 0.3),
            baseline=(-0.1, -0.005),
            n_permutations=500,
            random_state=0,
            by=(),
        )
        significant = [
            n for n, s in zip(test.settings["channels"], test.significant, strict=False) if s
        ]
        group = fit(sub, "rodent_ep", picks=significant)
    labelled = [
        any(c.label in ("P1", "N1", "late") for c in group.get(ch).components)
        for ch in significant
    ]
    assert len(significant) >= MIN_SIGNIFICANT
    assert sum(labelled) / len(significant) >= MIN_LABELLED
