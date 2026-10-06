"""Fetch files from OpenNeuro, the open archive of BIDS datasets."""

import os
from urllib.request import urlretrieve

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


def fetch_openneuro_file(dataset, path, folder='data'):
    """Download one file of an OpenNeuro dataset, if not already present.

    Parameters
    ----------
    dataset : str
        Dataset identifier, for example 'ds004080'.
    path : str
        Path of the file inside the dataset.
    folder : str, optional, default: 'data'
        Local folder; the dataset structure is kept below it.

    Returns
    -------
    local_path : str
        Path of the downloaded file.
    """

    local_path = os.path.join(folder, dataset, *path.split('/'))
    if not os.path.isfile(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        urlretrieve(openneuro_url(dataset, path), local_path)

    return local_path
