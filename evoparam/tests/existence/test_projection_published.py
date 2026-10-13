# SPDX-License-Identifier: Apache-2.0
"""Cross-projection test against the published values (REQ-134).

The published numbers are those of the canonical response parameterisation paper (Miller et
al., 2023, PLOS Computational Biology 19(5):e1011105, doi:10.1371/journal.pcbi.1011105),
computed on its public example data: the ``sampledata`` folder of
https://github.com/kaijmiller/crp_scripts (commit f6efb4e, the data and code statement of
the paper; copyleft code, data "for use without restriction (other than attribution)").
Each file holds the trials of one stimulation site recorded on one channel (one channel
pair). Only the data are read, at test time; the code is neither imported nor copied.

Published values used (window t1 = 15 ms to t2 = 1 s, as in the paper):

- response duration: Fig. 3, ``t2 = 80ms = tau_R``; Fig. 9, ``tau_R = 0.23s`` with all
  trials and 0.28 s once trial #6 is removed (the values are printed to 10 ms, so the
  rounding of the publication is itself up to 5 ms);
- significance: Fig. 5, the four top examples significant (p << 1e-16), the bottom one not
  (p = 0.37); Fig. 6, A significant, B not, C significant (early component), E significant
  (20 uV offset), F not (offset corrected).

The test on these data runs with EVOPARAM_NETWORK_TESTS=1 (19 MB from
raw.githubusercontent.com, cached in the evoparam data folder). REQ-134 names ds003708 as
the example dataset; no per-channel-pair duration or significance has been published for
it, which the strict expected failure below states.
"""

import os
import pathlib
import warnings
from typing import Any

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).parents[3]
COMMIT = "f6efb4efc6ad6f421603974130a8e09ff2d3e7bd"
URL = "https://raw.githubusercontent.com/kaijmiller/crp_scripts/{}/sampledata/{}.mat"
SHA256 = {
    "Fig3_sampledata": "e0930f8baeb4a92b1930bac7c30292199b801efcf211b6c33b3a0b6628a64be9",
    "Fig5_sampledata_1": "02556e3dfce07922878618d34e1e853687eda8ae2710f7e7acfcfeb9f5f4a0e1",
    "Fig5_sampledata_2": "905cc2cd2a9ffbbe60d5e471a3d38a6bfb723fcb28892d843d6cbfbe36af2a5d",
    "Fig5_sampledata_3": "7de9ec155ff7439a04d4b73e3c604f502f460b0566d7ccad5ef0cb912a8c7470",
    "Fig5_sampledata_4": "199f4d1ead23b42adc9d1c57fe54b398d4ac13050f9ab56343855e84467cd9f8",
    "Fig5_sampledata_5": "a4bf1634c2b1de3628ebab9e0329f7db6a69677f221bd411e9da4cd6d8194142",
    "Fig6_a_significant_example": (
        "b25c3a84392f7b95af6b30047bb17c2ffa29989537d37751ad44895fcc30820c"
    ),
    "Fig6_b_insignificant_example": (
        "64e282715e47c39deb3352956574660c735c09e5030b5e750550a03dab3f2373"
    ),
    "Fig6_c_earlycomponent": "0af6a8842da9a57e9e7bc8adbe516bede960e73526a44a1f32badb458bf3e7d8",
    "Fig6_e_20muV_offset": "8b01ab781ba00264e4b424bf0e26c13d2503c6352fdec9d361f5991ecd69e8fa",
    "Fig6_f_offset_corrected": "16f75d8f906f0acb5687d6f9c467ceb3e955b8cd9cae7361be39c7398542fdfc",
    "Fig9_eeg_data_zeropad": "caabb03d39eced643fad33c5353a87d071172b7124aca2a6eebf4d16301f967d",
}
WINDOW = (0.015, 1.0)
# (file, trials removed (0-based), published tau_R in s)
DURATIONS = [
    ("Fig3_sampledata", (), 0.080),
    ("Fig9_eeg_data_zeropad", (), 0.23),
    ("Fig9_eeg_data_zeropad", (5,), 0.28),
]
# (file, published decision)
DECISIONS = [
    ("Fig5_sampledata_1", True),
    ("Fig5_sampledata_2", True),
    ("Fig5_sampledata_3", True),
    ("Fig5_sampledata_4", True),
    ("Fig5_sampledata_5", False),
    ("Fig6_a_significant_example", True),
    ("Fig6_b_insignificant_example", False),
    ("Fig6_c_earlycomponent", True),
    ("Fig6_e_20muV_offset", True),
    ("Fig6_f_offset_corrected", False),
]

needs_network = pytest.mark.skipif(
    os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
    reason="set EVOPARAM_NETWORK_TESTS=1 to read the paper's data",
)


def _trials(name: Any, drop: Any = ()) -> Any:
    """The trials of one example file (downloaded once, checked by SHA-256)."""

    from scipy.io import loadmat

    from evoparam.data import TrialSet
    from evoparam.io._fetch import data_folder, download, sha256_file

    folder = pathlib.Path(data_folder()) / "reference-validation" / "crp_scripts"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.mat"
    if not path.exists() or sha256_file(path) != SHA256[name]:
        download(URL.format(COMMIT, name), str(path))
    assert sha256_file(path) == SHA256[name], name
    mat = loadmat(path)
    data = np.asarray(mat["data"], float)  # times x trials
    times = np.asarray(mat["t"], float).ravel()
    keep = [k for k in range(data.shape[1]) if k not in drop]
    return TrialSet(
        data=data[:, keep].T[:, None, :],
        times=times,
        sfreq=1.0 / np.mean(np.diff(times)),
        ch_names=["recording"],
    )


def _projection(ts: Any, window: Any = WINDOW, **kwargs: Any) -> Any:
    from evoparam.existence import test_response

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_response(
            ts,
            method="projection",
            window=window,
            baseline=(float(ts.times[0]), 0.0),
            by=(),
            correction="none",
            alpha=0.05,
            random_state=0,
            allow_artifact_overlap=True,
            **kwargs,
        )


@pytest.mark.network
@needs_network
@pytest.mark.parametrize(("name", "drop", "published"), DURATIONS)
def test_req134_published_durations_of_the_example_data(
    name: Any, drop: Any, published: Any
) -> None:
    """REQ-134: the response duration of ``method="projection"`` is within 5 ms of the
    duration published for the paper's example data."""

    test = _projection(_trials(name, drop))
    assert abs(test.offset[0] - published) <= 0.005, (name, drop, test.offset[0])


@pytest.mark.network
@needs_network
def test_req134_published_significance_of_the_example_data() -> None:
    """REQ-134: the significance decisions of ``method="projection"`` with the paper's set of
    projections (``projection_pairs="half"``) are those published for the paper's example
    channel pairs on at least 95 % of them. The default set (``"disjoint"``, each trial in
    one pair) is kept because the paper's set rejects pure noise at 0.33 to 0.44 at
    alpha = 0.05 (``test_req134_pairs_of_the_paper_are_anticonservative``); with it the
    agreement depends on the random pairing (0.8 to 1.0 over seeds 0 to 9, CHANGELOG)."""

    agree = [
        bool(_projection(_trials(name), projection_pairs="half").significant[0]) == published
        for name, published in DECISIONS
    ]
    assert np.mean(agree) >= 0.95, [d for d, a in zip(DECISIONS, agree, strict=True) if not a]


@pytest.mark.network
@needs_network
@pytest.mark.xfail(
    strict=True,
    reason="REQ-134 with the default set of projections "
    "(projection_pairs='disjoint'): the decisions agree with the published ones "
    "on 0.8 to 1.0 of the 10 example pairs depending on the random pairing, "
    "mean 0.89 over seeds 0 to 9, below 0.95",
)
def test_req134_published_significance_with_the_default_pairs() -> None:
    """REQ-134 read with the defaults: the mean agreement over ten random pairings is at least
    95 %."""

    from evoparam.existence import test_response

    trials = {name: _trials(name) for name, _ in DECISIONS}
    agreement = []
    for seed in range(10):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agree = [
                bool(
                    test_response(
                        trials[name],
                        method="projection",
                        window=WINDOW,
                        baseline=(float(trials[name].times[0]), 0.0),
                        by=(),
                        correction="none",
                        alpha=0.05,
                        random_state=seed,
                        allow_artifact_overlap=True,
                    ).significant[0]
                )
                == published
                for name, published in DECISIONS
            ]
        agreement.append(np.mean(agree))
    assert np.mean(agreement) >= 0.95, agreement


def test_req134_reviewed_profile_is_kept_as_an_option() -> None:
    """REQ-134: the reading of §5.6 item 2 as reviewed (the mean projection divided by the
    square root of the window length) stays available, and the default is the published
    reading. On a polyphasic response (a large early peak at 25 ms, then a smaller
    deflection lasting to 200 ms) only the published reading finds the end; the reviewed
    one stops after the first peak."""

    from evoparam.tests.existence.helpers import gaussian, noise_trials, plateau, trialset

    rng = np.random.default_rng(0)
    response = 6.0 * gaussian(0.025, 0.01) + 1.0 * plateau(0.05, 0.2)
    data = noise_trials(rng, 15, 3) + response[None, None]
    published = _projection(trialset(data), window=(0.015, 0.5))
    reviewed = _projection(trialset(data), window=(0.015, 0.5), projection_profile="normalised")
    assert published.settings["projection_profile"] == "published"
    assert reviewed.settings["projection_profile"] == "normalised"
    assert np.all(np.abs(published.offset - 0.2) <= 0.005)
    assert np.all(reviewed.offset < 0.04)
    assert np.all(published.significant)
    assert np.all(reviewed.significant)


def test_req134_pairs_of_the_paper_are_anticonservative() -> None:
    """REQ-134: on pure white noise (10 trials, 200 channels) the paper's set of projections
    rejects far more often than alpha = 0.05 (0.43 measured), the default set (each trial in
    one pair) much less (0.145, above alpha by the choice of the response end at the peak of
    the profile); the default stays the one whose values are independent under the null."""

    from evoparam.tests.existence.helpers import noise_trials, trialset

    data = noise_trials(np.random.default_rng(3), 10, 200)
    half = _projection(trialset(data), window=(0.009, 0.5), projection_pairs="half")
    disjoint = _projection(trialset(data), window=(0.009, 0.5))
    assert disjoint.settings["projection_pairs"] == "disjoint"
    assert np.mean(half.significant) > 0.3
    assert np.mean(disjoint.significant) < 0.2


@pytest.mark.xfail(
    strict=True,
    reason="REQ-134 names ds003708 (OpenNeuro, the basis "
    "profile curve paper's data) as the example dataset: no per-channel-pair "
    "response duration or significance of the cross-projection method has "
    "been published for ds003708 (not in the cross-projection paper, its "
    "supporting information, its code repository, or the ds003708 deposit "
    "and its derivatives, checked 2026-10-11); the paper's published values "
    "are on its own example files, tested above",
)
def test_req134_ds003708_published_values() -> None:
    """REQ-134 on ds003708: needs the published per-channel-pair values as a comparator
    file, ``validation/layer3/crp_ds003708.json``."""

    assert (ROOT / "validation" / "layer3" / "crp_ds003708.json").exists()
