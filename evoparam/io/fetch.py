# SPDX-License-Identifier: Apache-2.0
"""Fetch files from OpenNeuro, the open archive of BIDS datasets."""

import os
from collections.abc import Iterator
from typing import cast

from evoparam._typing import Any
from evoparam.errors import ReaderError
from evoparam.io._fetch import RemoteFile, data_folder, download, sha256_file

__all__ = [
    "OPENNEURO_URL",
    "openneuro_url",
    "fetch_openneuro_file",
    "fetch_erpcore",
    "RemoteFile",
    "ERPCORE_NODES",
]

###################################################################################################
###################################################################################################

OPENNEURO_URL = "https://s3.amazonaws.com/openneuro.org/"


def openneuro_url(dataset: str, path: str) -> str:
    """Return the direct address of a file in an OpenNeuro dataset.

    Parameters
    ----------
    dataset : str
        Dataset identifier, for example 'ds004080'.
    path : str
        Path of the file inside the dataset, for example
        'sub-ccepAgeUMCU01/ses-1/ieeg/sub-ccepAgeUMCU01_ses-1_electrodes.tsv'.

    Returns
    -------
    url : str
    """
    return OPENNEURO_URL + dataset + "/" + path.lstrip("/")


def _registry_files(dataset: Any) -> Any:
    """``files`` (path -> SHA-256) of a dataset of the registry, or None."""
    from evoparam.io.registry import _load_registry

    for entry in _load_registry()["datasets"]:
        if entry["id"].lower() == dataset.lower():
            return entry.get("files") or {}
    return None


def _registry_sha256(dataset: Any, path: Any) -> Any:
    """SHA-256 recorded for this file in the dataset registry, if any."""
    from evoparam.io.registry import _load_registry

    for entry in _load_registry()["datasets"]:
        if entry["id"].lower() == dataset.lower():
            return (entry.get("files") or {}).get(path.lstrip("/"))
    return None


def fetch_openneuro_file(
    dataset: str,
    path: str,
    folder: "str | os.PathLike[str] | None" = "data",
    *,
    sha256: str | None = None,
) -> str:
    """Download one file of an OpenNeuro dataset, if not already present.

    Parameters
    ----------
    dataset : str
        Dataset identifier, for example 'ds004080'.
    path : str
        Path of the file inside the dataset.
    folder : str, path or None, default: 'data'
        Local folder; the dataset structure is kept below it. None: the evoparam data
        folder (``$EVOPARAM_DATA``, else ``~/.cache/evoparam``), as the other fetchers.
    sha256 : str, optional
        Expected SHA-256 of the file. When not given, the value recorded in the
        dataset registry for this file is used, if there is one. A file that does
        not match, newly downloaded or already present, is deleted.

    Returns
    -------
    local_path : str
        Path of the downloaded file.

    Raises
    ------
    ReaderError
        If the download fails, if ``EVOPARAM_OFFLINE=1`` and the file is not present,
        or if the file does not match the expected SHA-256.
    """
    if folder is None:
        folder = data_folder()
    local_path = os.path.join(folder, dataset, *path.lstrip("/").split("/"))
    expected = sha256 or _registry_sha256(dataset, path)

    if os.path.isfile(local_path):
        found: Any = sha256_file(local_path) if expected else None
    else:
        found = download(openneuro_url(dataset, path), local_path)

    if expected and found.lower() != expected.lower():
        os.remove(local_path)
        raise ReaderError(
            f"{local_path}: SHA-256 {found} does not match the expected {expected}; "
            "the file was deleted."
        )

    return local_path


# OSF project of ERP CORE (Kappenman et al. 2021, DOI 10.18115/D5JW4R): one node per
# component, each with the folder "<component> All Data and Scripts" holding one folder per
# participant
ERPCORE_NODES = {
    "P3": "etdkz",
    "N170": "pfde9",
    "N400": "29xpq",
    "MMN": "5q4xs",
    "N2pc": "yefrq",
    "LRP": "28e6c",
    "ERN": "q6gwp",
}
ERPCORE_SUFFIXES = ("_epoch_interp_ar.set", "_epoch_interp_ar.fdt")


def _osf_pages(url: Any) -> Iterator[Any]:
    """Yield the items of every page of an OSF API listing."""
    import json

    from evoparam.io._fetch import fetch_bytes

    while url:
        page = json.loads(fetch_bytes(url))
        yield from page["data"]
        url = page["links"].get("next")


def fetch_erpcore(component: str, participant: int = 1, folder: str | None = None) -> str:
    """Fetch the epoched data of one ERP CORE participant and component.

    EEGLAB ``.set`` and ``.fdt`` after the published pipeline: ICA, interpolation, artifact flags.

    Files already present are checked, new ones downloaded from OSF; both against the
    SHA-256 of the dataset registry when it records them (``erp-core``,
    ``<component>/<participant>/<file>``), else against the SHA-256 published by OSF.

    Parameters
    ----------
    component : str
        ``"P3"``, ``"N170"``, ``"N400"``, ``"MMN"``, ``"N2pc"``, ``"LRP"`` or ``"ERN"``.
    participant : int
        Participant number (1 to 40).
    folder : str, optional
        Default ``$EVOPARAM_ERPCORE`` or ``~/.cache/evoparam/erpcore``.

    Returns
    -------
    str
        Path of the ``.set`` file (read it with ``mne.read_epochs_eeglab``).

    Raises
    ------
    ReaderError
        Unknown component, download failure, offline without the files, or a checksum
        mismatch (the file is deleted).
    """
    from evoparam.errors import ReaderError
    from evoparam.io._fetch import _check_online

    if component not in ERPCORE_NODES:
        raise ReaderError(f"ERP CORE component must be one of {sorted(ERPCORE_NODES)}.")
    folder = folder or os.environ.get(
        "EVOPARAM_ERPCORE", os.path.expanduser("~/.cache/evoparam/erpcore")
    )
    import glob

    target = os.path.join(folder, component, str(int(participant)))
    prefix = f"{component}/{int(participant)}/"
    registry = {
        key[len(prefix) :]: value
        for key, value in (_registry_files("erp-core") or {}).items()
        if key.startswith(prefix)
    }

    def local(suffix: Any) -> Any:
        """Return the downloaded file whose name ends with ``suffix`` (None when absent)."""
        found = glob.glob(os.path.join(target, f"*{suffix}"))
        return found[0] if found else None

    published = {}
    if any(local(suffix) is None for suffix in ERPCORE_SUFFIXES) or not all(
        any(n.endswith(suffix) for n in registry) for suffix in ERPCORE_SUFFIXES
    ):
        _check_online("https://api.osf.io")
        node = ERPCORE_NODES[component]
        top = list(_osf_pages(f"https://api.osf.io/v2/nodes/{node}/files/osfstorage/"))
        data = next(f for f in top if f["attributes"]["name"].endswith("All Data and Scripts"))
        subjects = _osf_pages(
            data["relationships"]["files"]["links"]["related"]["href"] + "?page[size]=100"
        )
        subject = next((s for s in subjects if s["attributes"]["name"] == str(participant)), None)
        if subject is None:
            raise ReaderError(f"ERP CORE {component}: no participant {participant} on OSF.")
        for item in _osf_pages(
            subject["relationships"]["files"]["links"]["related"]["href"] + "?page[size]=100"
        ):
            name = item["attributes"]["name"]
            if name.endswith(ERPCORE_SUFFIXES):
                published[name] = (
                    (item["attributes"].get("extra") or {}).get("hashes") or {}
                ).get("sha256")
                path = os.path.join(target, name)
                if not os.path.isfile(path):
                    download(item["links"]["download"], path)
    out = []
    for suffix in ERPCORE_SUFFIXES:
        path = local(suffix)
        if path is None:
            raise ReaderError(f"ERP CORE {component}/{participant}: no *{suffix} on OSF.")
        name = os.path.basename(path)
        want = registry.get(name) or published.get(name)
        if want:
            found = sha256_file(path)
            if found.lower() != want.lower():
                os.remove(path)
                raise ReaderError(
                    f"{path}: SHA-256 {found} does not match the expected "
                    f"{want}; the file was deleted."
                )
        out.append(path)
    return cast("str", out[0])
