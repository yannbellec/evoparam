"""Fetch files from OpenNeuro, the open archive of BIDS datasets."""

import os

from evoparam.errors import ReaderError
from evoparam.io._fetch import RemoteFile, download, sha256_file

__all__ = ["OPENNEURO_URL", "openneuro_url", "fetch_openneuro_file", "RemoteFile"]

###################################################################################################
###################################################################################################

OPENNEURO_URL = 'https://s3.amazonaws.com/openneuro.org/'


def openneuro_url(dataset, path):
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

    return OPENNEURO_URL + dataset + '/' + path.lstrip('/')


def _registry_sha256(dataset, path):
    """SHA-256 recorded for this file in the dataset registry, if any."""

    from evoparam.io.registry import _load_registry
    for entry in _load_registry()['datasets']:
        if entry['id'].lower() == dataset.lower():
            return (entry.get('files') or {}).get(path.lstrip('/'))
    return None


def fetch_openneuro_file(dataset, path, folder='data', *, sha256=None):
    """Download one file of an OpenNeuro dataset, if not already present.

    Parameters
    ----------
    dataset : str
        Dataset identifier, for example 'ds004080'.
    path : str
        Path of the file inside the dataset.
    folder : str, optional, default: 'data'
        Local folder; the dataset structure is kept below it.
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

    local_path = os.path.join(folder, dataset, *path.lstrip('/').split('/'))
    expected = sha256 or _registry_sha256(dataset, path)

    if os.path.isfile(local_path):
        found = sha256_file(local_path) if expected else None
    else:
        found = download(openneuro_url(dataset, path), local_path)

    if expected and found.lower() != expected.lower():
        os.remove(local_path)
        raise ReaderError('{}: SHA-256 {} does not match the expected {}; the file was '
                          'deleted.'.format(local_path, found, expected))

    return local_path
