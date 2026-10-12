# SPDX-License-Identifier: Apache-2.0
"""Adjusted common average on OpenNeuro ds004977 (REQ-081, real-data part).

ds004977 (Mayo Clinic, sEEG, 4800 Hz, MEF 3.0) is the dataset of the adjusted common
average paper (Huang et al., 2024). Its recordings are censored with zeros outside one
stimulation site per run; one run per subject is used, with the 12 pulses of its
uncensored site, read with :func:`evoparam.io.read_bids` (remote MEF 3.0, sEEG channels
of channels.tsv, only the blocks under the epochs). Epochs are cached in the evoparam data
folder (float32, 20 to 40 MB per run; the first run of the test takes about 9 min, then
about 2 min). Runs only with EVOPARAM_NETWORK_TESTS=1.

Cross-channel R² of a channel: mean, over the other channels, of the squared Pearson
correlation between the trial averages of the two channels in the response window
(10-300 ms). Bad channels and the stimulated contacts are left out. A common average
that contains responsive channels imprints their response on every channel, which raises
this R²; REQ-081 asks that ``car_adaptive`` gives a lower R² than ``car`` (one-sided
Wilcoxon signed-rank test over channels, p < 0.01).
"""

import os
from typing import Any

import numpy as np
import pytest

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
        reason="set EVOPARAM_NETWORK_TESTS=1 to read OpenNeuro",
    ),
]

# (subject, run, uncensored stimulation site)
RUNS = [
    ("1", "01", "ROC11-ROC12"),
    ("2", "12", "RZ4-RZ5"),
    ("3", "03", "LA2-LA3"),
    ("4", "02", "RO3-RO4"),
]
EPOCH = (-0.2, 0.5)
WINDOW = (0.01, 0.3)

###################################################################################################
###################################################################################################


def load_trials(subject: Any, run: Any, site: Any) -> Any:
    """Epochs of the 12 pulses of ``site`` (cached), as a TrialSet."""

    pd = pytest.importorskip("pandas")
    from evoparam.data import TrialSet
    from evoparam.io import read_bids
    from evoparam.io._fetch import data_folder

    cache = os.path.join(data_folder(), "reference-validation", "ds004977")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"sub-{subject}_run-{run}_{site}.npz")
    if not os.path.exists(path):
        trials = read_bids(
            "ds004977",
            subject,
            session="ieeg01",
            task="ccep",
            run=run,
            remote=True,
            picks="seeg",
            epoch=EPOCH,
            select={"stim_site": site},
        )
        np.savez_compressed(
            path,
            data=trials.data.astype(np.float32),
            times=trials.times,
            sfreq=trials.sfreq,
            names=np.array(trials.ch_names),
            bad=trials.ch_info["bad"].to_numpy(dtype=bool),
            sites=trials.metadata["stim_site"].to_numpy(dtype=str),
        )
    saved = np.load(path)
    names = [str(name) for name in saved["names"]]
    return TrialSet(
        data=saved["data"].astype(float),
        times=saved["times"],
        sfreq=float(saved["sfreq"]),
        ch_names=names,
        metadata=pd.DataFrame({"stim_site": saved["sites"].astype(str)}),
        ch_info=pd.DataFrame({"name": names, "bad": saved["bad"]}),
    )


def cross_channel_r2(trials: Any, keep: Any) -> Any:
    """Mean squared correlation of each kept channel's trial average with the others."""

    window = (trials.times >= WINDOW[0]) & (trials.times <= WINDOW[1])
    average = trials.data[:, keep][:, :, window].mean(axis=0)
    r2 = np.corrcoef(average) ** 2
    np.fill_diagonal(r2, np.nan)
    return np.nanmean(r2, axis=1)


def compare(trials: Any) -> Any:
    """Cross-channel R² after ``car`` and after ``car_adaptive``, the kept channels and
    the size of the adjusted average."""

    from evoparam.prep import rereference

    stim = {part for site in trials.metadata["stim_site"] for part in site.split("-")}
    keep = ~trials.ch_info["bad"].to_numpy(dtype=bool) & np.array(
        [name not in stim for name in trials.ch_names]
    )
    keep &= np.isfinite(trials.data).all(axis=(0, 2))
    car = rereference(trials, "car")
    adaptive = rereference(trials, "car_adaptive", random_state=0)
    n_star = adaptive.provenance.steps[-1]["settings"]["n_star"]
    return cross_channel_r2(car, keep), cross_channel_r2(adaptive, keep), keep, n_star


@pytest.fixture(scope="module")
def comparisons() -> Any:
    return [compare(load_trials(*run)) for run in RUNS]


def test_reads_the_uncensored_epochs() -> None:

    trials = load_trials(*RUNS[0])
    assert trials.sfreq == 4800.0
    assert trials.data.shape[0] == 12
    assert trials.data.shape[1] > 200
    assert np.isfinite(trials.data).all()
    # Uncensored: the channels are not constant
    assert (trials.data.std(axis=-1) > 0).all(axis=0).mean() > 0.95


def test_req081_car_adaptive_lowers_cross_channel_r2(comparisons: Any) -> None:

    from scipy.stats import wilcoxon

    # Over the channels of the four runs (710 channels: median R² 0.26 with car, 0.18 with
    # car_adaptive, p ~ 1e-27 at the time of writing), and in each run (p < 1e-4)
    car = np.concatenate([c[0] for c in comparisons])
    adaptive = np.concatenate([c[1] for c in comparisons])
    assert np.median(adaptive) < np.median(car)
    assert wilcoxon(adaptive, car, alternative="less").pvalue < 0.01
    for run, (car, adaptive, _, n_star) in zip(RUNS, comparisons, strict=False):
        assert wilcoxon(adaptive, car, alternative="less").pvalue < 0.01, (run, n_star)
