"""read_mne (REQ-020); runs only when MNE-Python is installed (extra "mne")."""

import numpy as np
import pytest

mne = pytest.importorskip('mne')
pd = pytest.importorskip('pandas')

from evoparam.errors import DataError
from evoparam.io import read, read_mne

###################################################################################################
###################################################################################################

def make_epochs():
    rng = np.random.default_rng(0)
    info = mne.create_info(['E1', 'E2', 'E3'], 1000., ch_types=['ecog', 'ecog', 'seeg'])
    data = rng.normal(0, 1e-5, (4, 3, 300))                       # volts
    events = np.column_stack([np.arange(4) * 2000 + 1000, np.zeros(4, int), np.ones(4, int)])
    metadata = pd.DataFrame({'condition' : ['a', 'b', 'a', 'b']})
    epochs = mne.EpochsArray(data, info, events=events, tmin=-0.1, metadata=metadata,
                             verbose=False)
    return epochs, data


def test_epochs_to_trialset():

    epochs, data = make_epochs()
    trials = read_mne(epochs)
    trials.validate()
    assert trials.units == 'uV' and trials.ch_names == ['E1', 'E2', 'E3']
    assert np.allclose(trials.data, data * 1e6, rtol=1e-12, atol=0)
    assert np.allclose(trials.times, epochs.times) and trials.sfreq == 1000.
    assert list(trials.metadata['condition']) == ['a', 'b', 'a', 'b']
    assert list(trials.ch_info['type']) == ['ecog', 'ecog', 'seeg']
    assert np.allclose(trials.event_times, [1.0, 3.0, 5.0, 7.0])
    assert np.allclose(read_mne(epochs, units='mV').data, data * 1e3, rtol=1e-12, atol=0)
    assert read(epochs).n_trials == 4


def test_evoked_and_mixed_units():

    epochs, data = make_epochs()
    evoked = epochs.average()
    trials = read_mne(evoked)
    assert trials.n_trials == 1
    assert np.allclose(trials.data[0], data.mean(axis=0) * 1e6, rtol=1e-10, atol=0)

    info = mne.create_info(['E1', 'M1'], 1000., ch_types=['eeg', 'mag'])
    mixed = mne.EpochsArray(np.zeros((1, 2, 10)), info, verbose=False)
    with pytest.raises(DataError, match='do not share one unit'):
        read_mne(mixed)
