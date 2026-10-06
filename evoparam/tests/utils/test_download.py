"""Test functions for evoparam.utils.download."""

import os
import shutil
import pathlib

import numpy as np
from pytest import raises

from evoparam.utils.download import *

###################################################################################################
###################################################################################################

TEST_FOLDER = 'test_data'

def clean_up_downloads():

    shutil.rmtree(TEST_FOLDER)

def make_source(tmp_path):
    """Make a local 'remote' folder with a data file, and return its file:// address."""

    np.save(tmp_path / 'times.npy', np.arange(10))

    return pathlib.Path(tmp_path).as_uri() + '/'

###################################################################################################
###################################################################################################

def test_check_data_folder():

    check_data_folder(TEST_FOLDER)
    assert os.path.isdir(TEST_FOLDER)

def test_check_data_file(tmp_path):

    filename = 'times.npy'

    check_data_file(filename, TEST_FOLDER, url=make_source(tmp_path))
    assert os.path.isfile(os.path.join(TEST_FOLDER, filename))

    clean_up_downloads()

def test_check_data_file_no_url():

    check_data_folder(TEST_FOLDER)
    with raises(ValueError):
        check_data_file('missing.npy', TEST_FOLDER, url=None)

    clean_up_downloads()

def test_fetch_evoparam_data(tmp_path):

    filename = 'times.npy'

    fetch_evoparam_data(filename, folder=TEST_FOLDER, url=make_source(tmp_path))
    assert os.path.isfile(os.path.join(TEST_FOLDER, filename))

    clean_up_downloads()

def test_load_evoparam_data(tmp_path):

    filename = 'times.npy'

    data = load_evoparam_data(filename, folder=TEST_FOLDER, url=make_source(tmp_path))
    assert isinstance(data, np.ndarray)

    clean_up_downloads()
