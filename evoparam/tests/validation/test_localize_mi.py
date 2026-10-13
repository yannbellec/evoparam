# SPDX-License-Identifier: Apache-2.0
"""Existence test on the scalp signal of Localize-MI (REQ-373).

Localize-MI (Mikulan et al., 2020, Scientific Data 7:127, doi:10.1038/s41597-020-0467-x):
61 sessions of 7 patients, single biphasic pulses of 0.5 ms at 0.1 to 5 mA through pairs of
sEEG contacts, recorded with 256-channel scalp EEG at 8 kHz. The data are on G-Node GIN
(https://gin.g-node.org/ezemikulan/Localize-MI, doi:10.12751/g-node.1cc1ae) and EBRAINS
(doi:10.25493/NXN2-05W); the authors' code is https://github.com/iTCf/mikulan_et_al_2020.

What REQ-373 needs and what is published (checked 2026-10-11):

- "stimulations that the dataset authors report as producing a scalp response": the paper
  reports no such list (its validation is source localisation of the stimulation artifact,
  2 to 12 mm from the stimulated contacts); the BIDS metadata of a session holds the
  stimulated pair and the intensity only (``trial_type`` of ``*_epochs.tsv``, written by
  ``fx_bids.export_data_bids`` of the authors' code).
- "sham/low-intensity trials (where available)": there is no sham condition; the lowest
  intensity is 0.1 mA (22 sessions), which still produces the stimulation artifact.
- the scalp signal after the pulse: the authors' conversion code crops the shared epochs at
  +10 ms (``epo.crop(None, 0.01)``; epochs from -0.25 s at 8 kHz, ``load_bids``), so the
  deposit, as the code writes it, holds the artifact and no later scalp response.

The hosts of the data (gin.g-node.org, data-proxy.ebrains.eu) are refused by the network
policy of the build machine; the second test reads the header of one epochs file and runs
with EVOPARAM_NETWORK_TESTS=1 on a machine that reaches GIN.
"""

import ast
import inspect
import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parents[3]
REFERENCE = ROOT / "validation" / "layer3" / "localize_mi_responses.json"
GIN = "https://gin.g-node.org/ezemikulan/Localize-MI/raw/master/"
EPOCHS = "derivatives/epochs/sub-01/eeg/sub-01_task-seegstim_run-01_epochs.npy"
SFREQ, TMIN = 8000.0, -0.25  # the authors' load_bids (fx_bids.py)


@pytest.mark.xfail(
    strict=True,
    reason="REQ-373: the Localize-MI authors report no list of "
    "stimulations producing a scalp response, and the dataset has no sham "
    "trials (Mikulan et al. 2020; the BIDS metadata hold the stimulated pair "
    "and intensity only), so the comparator "
    "validation/layer3/localize_mi_responses.json cannot be built from "
    "published values",
)
def test_req373_scalp_responses_reported_by_the_authors() -> None:
    """REQ-373: the comparator (per session: stimulated pair, intensity, the authors'
    report of a scalp response, sham or not) exists."""

    assert REFERENCE.exists()


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("EVOPARAM_NETWORK_TESTS") != "1",
    reason="set EVOPARAM_NETWORK_TESTS=1 to read Localize-MI on GIN",
)
@pytest.mark.xfail(
    strict=True,
    reason="REQ-373: the authors' conversion code crops the shared "
    "epochs at +10 ms after the pulse (fx_bids.export_data_bids, "
    "github.com/iTCf/mikulan_et_al_2020), shorter than the existence window; "
    "an unexpected pass means the deposit holds longer epochs",
)
def test_req373_deposit_holds_the_scalp_response_window() -> None:
    """REQ-373: the shared epochs reach the end of the default existence window of
    ``test_response``, so that a scalp response after the stimulation artifact can be
    tested (read from the header of the first epochs file of sub-01)."""

    from evoparam.errors import ReaderError
    from evoparam.existence import test_response
    from evoparam.io._fetch import RemoteFile

    window = inspect.signature(test_response).parameters["window"].default
    try:
        with RemoteFile(GIN + EPOCHS, chunk_size=1 << 12) as remote:
            head = remote.read(0, 512)
    except ReaderError as exc:
        pytest.skip(f"GIN is not reachable from this machine: {exc}")
    if not head.startswith(b"\x93NUMPY"):
        pytest.skip("GIN did not serve the content of the epochs file (annexed pointer?)")
    length = int.from_bytes(head[8:10], "little")
    header = ast.literal_eval(head[10 : 10 + length].decode("latin1"))
    n_times = header["shape"][-1]
    last = TMIN + (n_times - 1) / SFREQ
    assert last >= window[1], (header["shape"], last)
