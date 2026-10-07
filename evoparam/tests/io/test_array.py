"""from_array, read dispatch and the light core (REQ-020, REQ-025)."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from evoparam.data import Evoked, TrialSet
from evoparam.errors import DataError, ReaderError
from evoparam.io import from_array, read
from evoparam.tests.io.helpers import write_brainvision

###################################################################################################
###################################################################################################

def test_from_array_shapes():

    rng = np.random.default_rng(0)
    one = from_array(rng.normal(size=(3, 50)), 100.)
    assert one.data.shape == (1, 3, 50) and one.ch_names == ['CH001', 'CH002', 'CH003']
    assert np.allclose(one.times, np.arange(50) / 100.) and one.units == 'uV'
    one.validate()

    data = rng.normal(size=(4, 2, 30)).astype(np.float32)
    trials = from_array(data, 1000., times=np.arange(-10, 20) / 1000., ch_names=['a', 'b'],
                        units='mV', species='mouse', modality='lfp_ep')
    assert trials.data.dtype == np.float64 and np.allclose(trials.data, data)
    assert trials.times[10] == 0.0 and trials.units == 'mV' and trials.species == 'mouse'
    assert trials.provenance.source['reader'] == 'evoparam.io.from_array'
    assert trials.provenance.steps[-1]['function'] == 'evoparam.io.from_array'
    assert trials.provenance.data_hash is not None

    # Integer data are accepted and converted
    assert from_array(np.ones((2, 5), dtype=int), 10.).data.dtype == np.float64


def test_from_array_tables():

    pd = pytest.importorskip('pandas')
    data = np.zeros((3, 2, 10))
    trials = from_array(data, 100., metadata={'condition' : ['a', 'b', 'a'],
                                              'stim_current_mA' : [1.0, 2.0, 3.0]},
                        ch_info=pd.DataFrame({'name' : ['CH001', 'CH002'], 'type' : ['ecog'] * 2}))
    assert list(trials.metadata['condition']) == ['a', 'b', 'a']
    assert list(trials.ch_info['type']) == ['ecog', 'ecog']
    assert trials.average(by='condition')[0].n_trials == 2


def test_from_array_errors():

    with pytest.raises(DataError, match='dimensions'):
        from_array(np.zeros(10), 100.)
    with pytest.raises(DataError, match='times'):
        from_array(np.zeros((2, 10)), 100., times=np.arange(9))
    with pytest.raises(DataError, match='ch_names'):
        from_array(np.zeros((2, 10)), 100., ch_names=['a'])
    with pytest.raises(DataError, match='sfreq'):
        from_array(np.zeros((2, 10)), -1.)
    with pytest.raises(DataError, match='sfreq'):
        from_array(np.zeros((2, 10)), 100., times=np.arange(10) / 50.)
    with pytest.raises(DataError, match='units'):
        from_array(np.zeros((2, 10)), 100., units='furlong')
    with pytest.raises(DataError, match='numeric'):
        from_array(np.array([['a', 'b']]), 100.)
    pytest.importorskip('pandas')
    with pytest.raises(DataError, match='metadata'):
        from_array(np.zeros((3, 2, 10)), 100., metadata={'condition' : ['a', 'b']})


def test_read_dispatch(tmp_path):

    data = np.arange(2 * 3 * 20, dtype=float).reshape(2, 3, 20)
    trials = read(data, sfreq=100.)
    assert np.array_equal(trials.data, data)
    assert read(trials) is trials

    evoked = Evoked(data=data[0], times=np.arange(20) / 100., sfreq=100.,
                    ch_names=['a', 'b', 'c'])
    assert read(evoked).n_trials == 1

    # A BrainVision file: the recording as one trial, or epochs around onsets
    continuous = np.arange(3)[:, None] * 1000. + np.arange(2000)[None, :]
    vhdr = write_brainvision(tmp_path, 'rec', continuous, 100., ['C1', 'C2', 'C3'])
    whole = read(vhdr)
    assert whole.data.shape == (1, 3, 2000) and whole.times[0] == 0.0
    assert whole.provenance.source['sha256'] is not None
    part = read(vhdr, tmin=2.0, tmax=3.0, picks=['C2'])
    assert part.data.shape == (1, 1, 100) and part.times[0] == 2.0
    assert np.array_equal(part.data[0, 0], continuous[1, 200:300])

    epochs = read(vhdr, onsets=[1.0, 5.0, 19.9], epoch=(-0.1, 0.2))
    assert epochs.n_trials == 2 and epochs.provenance.steps[-1]['kept'] == [True, True, False]
    assert np.array_equal(epochs.data[1, 0], continuous[0, 490:520])

    # Saved container
    path = str(tmp_path / 'trials.evoparam')
    trials.save(path)
    loaded = read(path)
    assert isinstance(loaded, TrialSet) and np.array_equal(loaded.data, data)

    with pytest.raises(ReaderError, match='format'):
        read(str(tmp_path / 'unknown.xyz'))
    with pytest.raises(ReaderError, match='NWB'):
        read(str(tmp_path / 'file.nwb'))
    with pytest.raises(ReaderError, match='Cannot read'):
        read(object())


def test_light_core_in_a_subprocess():
    """REQ-025: without pandas and the reader extras, from_array works and the readers raise
    ImportError naming the extra."""

    code = textwrap.dedent('''
        import sys
        for name in ("pandas", "pyarrow", "mne", "edfio", "pyedflib", "pynwb", "neo"):
            sys.modules[name] = None
        import numpy as np
        import evoparam
        from evoparam.io import from_array, epoch, read_mne, read_edf, read_bids, list_datasets
        trials = from_array(np.zeros((2, 3, 50)), 100.)
        trials.validate()
        assert trials.metadata is None
        trials = epoch(np.zeros((3, 500)), 100., [1.0, 2.0], -0.1, 0.2)
        assert trials.n_trials == 2
        assert trials.provenance.steps[-1]["overlaps_previous"] == [False, False]
        for func, args, extra in ((read_mne, (None,), "mne"), (read_edf, ("x.edf",), "edf"),
                                  (read_bids, ("root", "01"), "data"), (list_datasets, (), "data")):
            try:
                func(*args)
            except ImportError as exc:
                assert 'pip install "evoparam[{}]"'.format(extra) in str(exc), str(exc)
            else:
                raise AssertionError(func.__name__ + " did not raise ImportError")
        print("ok")
    ''')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            timeout=300)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith('ok')
