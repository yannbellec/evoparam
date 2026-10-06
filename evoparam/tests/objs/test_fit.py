"""Tests for evoparam.objs.fit, including the evoparam object and it's methods.

NOTES
-----
The tests here are not strong tests for accuracy.
They serve rather as 'smoke tests', for if anything fails completely.
"""

import numpy as np
from pytest import raises

from evoparam import evoparam
from evoparam.core.items import OBJ_DESC, PEAK_INDS
from evoparam.core.errors import FitError
from evoparam.core.utils import group_three
from evoparam.core.modutils import safe_import
from evoparam.core.errors import DataError, NoDataError, InconsistentDataError
from evoparam.sim import gen_time_vector, simulate_erp
from evoparam.data import evoparamSettings, evoparamMetaData, evoparamResults

pd = safe_import('pandas')

from evoparam.tests.settings import TEST_DATA_PATH
from evoparam.tests.tutils import get_tfm, plot_test, default_params

from evoparam.objs.fit import *

###################################################################################################
###################################################################################################

def test_evoparam():
    """Check evoparam object initializes properly."""

    assert evoparam(verbose=False)

def test_evoparam_has_data(tfm):
    """Test the has_data property attribute, with and without model fits."""

    assert tfm.has_data

    ntfm = evoparam()
    assert not ntfm.has_data

def test_evoparam_has_model(tfm):
    """Test the has_model property attribute, with and without model fits."""
    
    assert tfm.has_model

    ntfm = evoparam()
    assert not ntfm.has_model

def test_evoparam_n_peaks(tfm):
    """Test the n_peaks property attribute."""

    assert tfm.n_peaks_

def test_evoparam_fit():
    """Test evoparam fit."""

    time_range, erp_params, nlv = default_params()
    np.random.seed(4)
    xs, ys = simulate_erp(time_range, erp_params, nlv)

    tfm = evoparam(verbose=False, max_n_peaks=3)
    tfm.fit(xs, ys, time_range=[0, time_range[1]])

    # Check model results - gaussian parameters
    for ii, gauss in enumerate(group_three(erp_params)):
        assert np.allclose(gauss, tfm.gaussian_params_[ii, :3], 
                           [2.0, 0.5, 1.0])
        assert np.isnan(tfm.shape_params_[ii, 3])

def test_evoparam_fit_noise():
    """Test evoparam fit on noisy data, to make sure nothing breaks."""

    time_range, erp_params, _ = default_params()
    nlv = 0.3

    xs, ys = simulate_erp(time_range, erp_params, nlv)

    tfm = evoparam(max_n_peaks=2, verbose=False)
    tfm.fit(xs, ys, time_range=[0, time_range[1]])

    # No accuracy checking here - just checking that it ran
    assert tfm.has_model

def test_evoparam_fit_skew():
    """Test evoparam fit on skewed data with peak_mode='skewed_gaussian'."""

    time_range, _, nlv = default_params()
    erp_params_d = [0.2, 1, .1]
    for skew in [-2, 0, 2]:
        erp_params = np.concatenate([erp_params_d, [skew]])
        np.random.seed(24)
        xs, ys = simulate_erp(time_range, erp_params, nlv, 
                              peak_mode='skewed_gaussian')

        tfm = evoparam(verbose=False, max_n_peaks=1, peak_mode='skewed_gaussian')
        tfm._max_n_iters = 2
        tfm.fit(xs, ys, time_range=[0, time_range[1]])

        # Check model results - gaussian parameters
        assert tfm.has_model
        assert np.allclose(erp_params, tfm.shape_params_[0,:4], [2.0, 1.0, 1.0, 2.0])

def test_evoparam_fit_measures():
    """Test goodness of fit & error metrics, post model fitting."""

    tfm = evoparam(verbose=False)

    # Hack fake data with known properties: total error magnitude 2
    tfm.signal = np.array([1, 2, 3, 4, 5])
    tfm._peak_fit = np.array([1, 2, 5, 4, 5])

    # Check default goodness of fit and error measures
    tfm._calc_r_squared()
    assert np.isclose(tfm.r_squared_, 0.75757575)
    tfm._calc_error()
    assert np.isclose(tfm.error_, 0.4)

    # Check with alternative error fit approach
    tfm._calc_error(metric='MSE')
    assert np.isclose(tfm.error_, 0.8)
    tfm._calc_error(metric='RMSE')
    assert np.isclose(tfm.error_, np.sqrt(0.8))
    with raises(ValueError):
        tfm._calc_error(metric='BAD')

def test_evoparam_checks():
    """Test various checks, errors and edge cases in evoparam.
    This tests all the input checking done in `_prepare_data`.
    """
    xs, ys = simulate_erp(*default_params())

    tfm = evoparam(verbose=False)

    ## Check checks & errors done in `_prepare_data`

    # Check wrong data type error
    with raises(DataError):
        tfm.fit(list(xs), list(ys))

    # Check dimension error
    with raises(DataError):
        tfm.fit(xs, np.reshape(ys, [1, len(ys)]))

    # Check shape mismatch error
    with raises(InconsistentDataError):
        tfm.fit(xs[:-1], ys)

    # Check complex inputs error
    with raises(DataError):
        tfm.fit(xs, ys.astype('complex'))

    # Check time_range cropping 
    tfm.fit(xs, ys, [1, 2])

    # Check error for `check_times` - for if there is non-evenly spaced time values
    with raises(DataError):
        tfm.fit(np.array([1, 2, 4]), np.array([1, 2, 3]))

    ## Check errors & errors done in `fit`

    # Check fit, and string report model error (no data / model fit)
    tfm = evoparam(verbose=False)
    with raises(NoDataError):
        tfm.fit()

def test_evoparam_load():
    """Test load into evoparam. Note: loads files from test_core_io."""

    # Test loading just results
    tfm = evoparam(verbose=False)
    file_name_res = 'test_evoparam_res'
    tfm.load(file_name_res, TEST_DATA_PATH, regenerate=False)
    # Check that result attributes get filled
    for result in OBJ_DESC['results']:
        assert not np.all(np.isnan(getattr(tfm, result)))
    # Test that settings and data are None
    #   Except for aperiodic mode, which can be inferred from the data
    for setting in OBJ_DESC['settings']:
        if setting != 'aperiodic_mode':
            assert getattr(tfm, setting) is None
    assert getattr(tfm, 'signal') is None

    # Test loading just settings
    tfm = evoparam(verbose=False)
    file_name_set = 'test_evoparam_set'
    tfm.load(file_name_set, TEST_DATA_PATH)
    for setting in OBJ_DESC['settings']:
        assert getattr(tfm, setting) is not None
    # Test that results and data are None
    for result in OBJ_DESC['results']:
        assert np.all(np.isnan(getattr(tfm, result)))
    assert tfm.signal is None

    # Test loading just data
    tfm = evoparam(verbose=False)
    file_name_dat = 'test_evoparam_dat'
    tfm.load(file_name_dat, TEST_DATA_PATH, regenerate=False)
    assert tfm.signal is not None
    # Test that settings and results are None
    for setting in OBJ_DESC['settings']:
        assert getattr(tfm, setting) is None
    for result in OBJ_DESC['results']:
        assert np.all(np.isnan(getattr(tfm, result)))

    # Test loading all elements
    tfm = evoparam(verbose=False)
    file_name_all = 'test_evoparam_all'
    tfm.load(file_name_all, TEST_DATA_PATH)
    for result in OBJ_DESC['results']:
        assert not np.all(np.isnan(getattr(tfm, result)))
    for setting in OBJ_DESC['settings']:
        assert getattr(tfm, setting) is not None
    for data in OBJ_DESC['data']:
        assert getattr(tfm, data) is not None
    for meta_dat in OBJ_DESC['meta_data']:
        assert getattr(tfm, meta_dat) is not None

def test_add_data():
    """Tests method to add data to evoparam objects."""

    # This test uses it's own evoparam object, to not add stuff to the global one
    tfm = get_tfm()

    # Test data for adding
    time, signal = np.array([1, 2, 3]), np.array([10, 10, 10])

    # Test adding data
    tfm.add_data(time, signal)
    assert tfm.has_data
    assert np.all(tfm.time == time)
    assert np.all(tfm.signal == signal)

    # Test that prior data does not get cleared, when requesting not to clear
    tfm._reset_data_results(True, True, True)
    tfm.add_results(evoparamResults(
        gaussian_params=np.asarray([[0.101,  1.810, 0.029],
                                    [0.198, -1.399, 0.047]]), 
        shape_params=np.asarray([[0.064, 0.04, 0.024, 0.625, 0.977, 0.971, 0.982],
                                [0.103, 0.041, 0.062, 0.398, 0.956, 0.964, 0.947]]), 
        peak_indices=np.asarray([[562, 602, 626], [657, 698, 760]]),
        r_squared=0.97, error=0.01, adj_r_squared=0.95))

    tfm.add_data(time, signal, clear_results=False)
    assert tfm.has_data
    assert tfm.has_model

    # Test that prior data does get cleared, when requesting to clear
    tfm._reset_data_results(True, True, True)
    tfm.add_data(time, signal, clear_results=True)
    assert tfm.has_data
    assert not tfm.has_model

def test_add_settings():
    """Tests method to add settings to evoparam objects."""

    # This test uses it's own evoparam object, to not add stuff to the global one
    tfm = get_tfm()

    # Test adding settings
    evoparam_settings = evoparamSettings([1, 4], 6, 0, 2, "gaussian", 0.75, 500, 0.5)
    tfm.add_settings(evoparam_settings)
    for setting in OBJ_DESC['settings']:
        assert getattr(tfm, setting) == getattr(evoparam_settings, setting)

def test_add_meta_data():
    """Tests method to add meta data to evoparam objects."""

    # This test uses it's own evoparam object, to not add stuff to the global one
    tfm = get_tfm()

    # Test adding meta data
    evoparam_meta_data = evoparamMetaData([3, 40], 0.5, (-0.5,0), 1)
    tfm.add_meta_data(evoparam_meta_data)
    for meta_dat in OBJ_DESC['meta_data']:
        assert getattr(tfm, meta_dat) == getattr(evoparam_meta_data, meta_dat)

def test_add_results():
    """Tests method to add results to evoparam objects."""

    # This test uses it's own evoparam object, to not add stuff to the global one
    tfm = get_tfm()

    # Test adding results
    evoparam_results = evoparamResults(
        gaussian_params=np.asarray([[0.10, 1.8, 0.02], [0.19, -1.39, 0.047]]), 
        shape_params=np.asarray([[0.064, 0.04, 0.024, 0.625, 0.97, 0.97, 0.98],
                                  [0.103, 0.041, 0.062, 0.39, 0.951, 0.964, 0.947]]), 
        peak_indices=np.asarray([[562, 602, 626],  [657, 698, 760]]),
        r_squared=0.99, error=0.01, adj_r_squared=0.95)

    tfm.add_results(evoparam_results)
    assert tfm.has_model
    for setting in OBJ_DESC['results']:
        assert np.sum(getattr(tfm, setting) != getattr(evoparam_results, setting.strip('_'))) == 0

def test_obj_gets(tfm):
    """Tests methods that return evoparam data objects.

    Checks: get_settings, get_meta_data, get_results
    """

    settings = tfm.get_settings()
    assert isinstance(settings, evoparamSettings)
    meta_data = tfm.get_meta_data()
    assert isinstance(meta_data, evoparamMetaData)
    results = tfm.get_results()
    assert isinstance(results, evoparamResults)

def test_get_params(tfm):
    """Test the get_params method."""

    for dname in ['shape','shape_params',
                  'error', 'r_squared', 'gaussian_params', 'gaussian']:
        assert np.any(tfm.get_params(dname))

        if dname == 'gaussian_params' or dname == 'gaussian':
            for dtype in ['MN','HT','SD']:
                assert np.any(tfm.get_params(dname, dtype))

        if dname == 'shape_params' or dname == 'shape':
            for dtype in ['latency', 'amplitude', 'width', 'fwhm', 'rise_time', 'decay_time', 'symmetry',
            'sharpness', 'sharpness_rise', 'sharpness_decay']:
                assert np.any(tfm.get_params(dname, dtype))

def test_copy():
    """Test copy evoparam method."""

    tfm = evoparam(verbose=False)
    ntfm = tfm.copy()

    assert tfm != ntfm

def test_evoparam_prints(tfm):
    """Test methods that print (alias and pass through methods).

    Checks: print_settings, print_results, print_report_issue.
    """

    tfm.print_settings()
    tfm.print_results()
    tfm.print_report_issue()

@plot_test
def test_evoparam_plot(tfm, skip_if_no_mpl):
    """Check the alias to plot evoparam."""

    tfm.plot()

def test_evoparam_resets():
    """Check that all relevant data is cleared in the reset method."""

    # Note: uses it's own tfm, to not clear the global one
    tfm = get_tfm()

    tfm._reset_data_results(True, True, True)
    tfm._reset_internal_settings()

    for data in ['data', 'model_components']:
        for field in OBJ_DESC[data]:
            assert getattr(tfm, field) is None
    for field in OBJ_DESC['results']:
        assert np.all(np.isnan(getattr(tfm, field)))
    assert tfm.time is None and tfm._peak_fit is None

def test_evoparam_report(skip_if_no_mpl):
    """Check that running the top level model method runs."""

    tfm = evoparam(verbose=False)
    tfm.report(*simulate_erp(*default_params()))

    assert tfm

def test_evoparam_fit_failure():
    """Test evoparam fit failures."""

    ## Induce a runtime error, and check it runs through
    tfm = evoparam(verbose=False)
    tfm.maxfev = 5

    tfm.fit(*simulate_erp(*default_params()))

    # Check after failing out of fit, all results are reset
    for result in OBJ_DESC['results']:
        assert np.all(np.isnan(getattr(tfm, result)))

    ## Monkey patch to check errors in general
    #  This mimics the main fit-failure, without requiring bad data / waiting for it to fail.
    tfm = evoparam(verbose=False)
    def raise_runtime_error(*args, **kwargs):
        raise FitError('Test-MonkeyPatch')
    tfm._fit_peaks = raise_runtime_error

    # Run a evoparam fit - this should raise an error, but continue in try/except
    tfm.fit(*simulate_erp(*default_params()))

    # Check after failing out of fit, all results are reset
    for result in OBJ_DESC['results']:
        assert np.all(np.isnan(getattr(tfm, result)))

def test_evoparam_debug():
    """Test evoparam in debug mode, including with fit failures."""

    tfm = evoparam(verbose=False)
    tfm.maxfev = 5

    tfm.set_debug_mode(True)
    assert tfm._debug is True

    with raises(FitError):
        tfm.fit(*simulate_erp(*default_params()))

def test_evoparam_check_data():
    """Test evoparam in with check data mode turned off, including with NaN data."""

    tfm = evoparam(verbose=False)

    tfm.set_check_data_mode(False)
    assert tfm._check_data is False

    # Add data, with check data turned off
    #   In check data mode, adding data with NaN should run
    freqs = gen_time_vector([3, 50], 0.5)
    powers = np.ones_like(freqs) * np.nan
    tfm.add_data(freqs, powers)
    assert tfm.has_data

    # Model fitting should execute, but return a null model fit, given the NaNs, without failing
    tfm.fit()
    assert not tfm.has_model

def test_evoparam_to_df(tfm, tbands, skip_if_no_pandas):

    df1 = tfm.to_df(2)
    assert isinstance(df1, pd.Series)
    df2 = tfm.to_df(tbands)
    assert isinstance(df2, pd.Series)

def test_evoparam_get_filtered_results(tfm):
    # test array output
    res = tfm.get_filtered_results(tfm.time_range, select_highest=True, 
                                   threshold=None, thresh_param='amplitude', 
                                   attribute='shape_params', extract_param=False, 
                                   dict_format=False)
    assert type(res) == np.ndarray # check data type
    assert res.shape == (1, len(PEAK_INDS)) # match number of shape params

    # test dict output
    res_dict = tfm.get_filtered_results(tfm.time_range, select_highest=True, 
                                        threshold=None, thresh_param='amplitude', 
                                        attribute='shape_params', 
                                        extract_param=False, dict_format=True)
    assert type(res_dict) == dict # check data type
    highest = np.argmax(np.abs(tfm.shape_params_[:, 1]))
    for param in PEAK_INDS: # check all results are as expected
        if param=='skew': continue # skip NaN
        assert res_dict[param] == tfm.shape_params_[highest][PEAK_INDS[param]]