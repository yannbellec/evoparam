"""Test functions for evoparam.utils.io."""

import numpy as np

from evoparam.core.items import OBJ_DESC
from evoparam.objs import evoparam, evoparamGroup

from evoparam.tests.settings import TEST_DATA_PATH

from evoparam.utils.io import *

###################################################################################################
###################################################################################################

def test_load_evoparam():

    file_name = 'test_evoparam_all'

    tfm = load_evoparam(file_name, TEST_DATA_PATH)

    assert isinstance(tfm, evoparam)

    # Check that all elements get loaded
    for result in OBJ_DESC['results']:
        assert not np.all(np.isnan(getattr(tfm, result)))
    for setting in OBJ_DESC['settings']:
        assert getattr(tfm, setting) is not None
    for data in OBJ_DESC['data']:
        assert getattr(tfm, data) is not None
    for meta_dat in OBJ_DESC['meta_data']:
        assert getattr(tfm, meta_dat) is not None

def test_load_evoparamgroup():

    file_name = 'test_evoparamgroup_all'
    tfg = load_evoparamGroup(file_name, TEST_DATA_PATH)

    assert isinstance(tfg, evoparamGroup)

    # Check that all elements get loaded
    assert len(tfg.group_results) > 0
    for setting in OBJ_DESC['settings']:
        assert getattr(tfg, setting) is not None
    assert tfg.signals is not None
    for meta_dat in OBJ_DESC['meta_data']:
        assert getattr(tfg, meta_dat) is not None
