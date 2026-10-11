# SPDX-License-Identifier: Apache-2.0
"""read_bids on a synthetic mini BIDS dataset (REQ-020)."""

import hashlib
import os

import numpy as np
import pytest

pd = pytest.importorskip('pandas')

from evoparam.data import TrialSet
from evoparam.errors import DataError, EventError, ReaderError
from evoparam.io import read, read_bids
from evoparam.tests.io.helpers import (CH_NAMES, SFREQ, STIM_ONSETS, signal, write_brainvision,
                                       make_bids)

###################################################################################################
###################################################################################################

def expected_epochs(onsets, channels=(0, 1, 2), tmin=-0.5, tmax=1.0):
    full = signal(3, 60000)
    offsets = np.arange(int(round(tmin * SFREQ)), int(round(tmax * SFREQ)))
    return np.stack([full[list(channels)][:, int(round(onset * SFREQ)) + offsets]
                     for onset in onsets])


def test_read_bids_local(bids_root):

    trials = read_bids(bids_root, 'sub-01', task='spes', run='01')
    assert isinstance(trials, TrialSet)
    trials.validate()

    # 16 stimulation events, the last one too close to the end of the recording
    kept_onsets = STIM_ONSETS[:-1]
    assert trials.n_trials == 15 and trials.ch_names == CH_NAMES
    assert trials.sfreq == SFREQ and trials.units == 'uV' and trials.modality == 'spes'
    assert np.allclose(trials.times[[0, -1]], [-0.5, 0.999])
    assert np.array_equal(trials.data, expected_epochs(kept_onsets))
    assert np.allclose(trials.event_times, kept_onsets)
    kept = trials.provenance.steps[-1]['runs'][0]['kept']
    assert kept == [True] * 15 + [False]

    # Metadata: standard columns, original columns, overlap flag
    meta = trials.metadata
    assert list(meta['stim_site'][:2]) == ['A01-A02', 'A02-A03']
    assert np.allclose(meta['stim_current_mA'], 4.0)
    assert np.allclose(meta['stim_pulse_width_us'], 200.0)
    assert set(meta['stim_polarity']) == {'monophasic_cathodic'}      # from the ieeg.json text
    assert list(meta['subject'].unique()) == ['01'] and list(meta['run'].unique()) == ['01']
    assert list(meta['session'].unique()) == ['n/a'] and list(meta['task'].unique()) == ['spes']
    assert meta['trial'].dtype == np.int64
    assert 'electrical_stimulation_site' in meta.columns and 'condition' in meta.columns
    assert meta['overlaps_previous'].tolist() == [False] * 14 + [True]   # 41.8 s after 41 s
    assert meta.attrs['units']['stim_current_mA'] == 'mA'

    # Channel table from channels.tsv and electrodes.tsv (cm → mm)
    info = trials.ch_info
    assert list(info['name']) == CH_NAMES and set(info['type']) == {'ecog'}
    assert info['bad'].tolist() == [False, False, True]
    assert info['reference'].tolist() == ['monopolar:G2'] * 3
    assert np.allclose(info['x'], [0.0, 10.0, 20.0]) and set(info['coord_space']) == {'ACPC'}
    assert info['stim_contact'].tolist() == [True, True, True]


def test_provenance_source(bids_root):

    trials = read_bids(bids_root, '01', task='spes')
    source = trials.provenance.source
    data_file = os.path.join(bids_root, 'sub-01', 'ieeg', 'sub-01_task-spes_run-01_ieeg.eeg')
    with open(data_file, 'rb') as file:
        assert source['sha256'] == hashlib.sha256(file.read()).hexdigest()
    assert source['path'] == data_file and source['format'] == 'brainvision'
    assert source['reader'] == 'evoparam.io.read_bids'
    assert source['doi'] == '10.0000/mini'
    assert source['bids']['subject'] == '01' and source['bytes_read'] == 0
    step = trials.provenance.steps[-1]
    assert step['function'] == 'evoparam.io.read_bids'
    assert step['settings']['epoch'] == [-0.5, 1.0]
    assert 'unit_inferred' in ' '.join(step['runs'][0]['harmonization']['warnings'])

    # Same bytes, same source hash; data hash set
    again = read_bids(bids_root, '01', task='spes')
    assert again.provenance.source['sha256'] == source['sha256']
    assert again.provenance.data_hash == trials.provenance.data_hash is not None


def test_select_picks_and_epoch(bids_root):

    trials = read_bids(bids_root, '01', task='spes', select={'stim_site': 'A02_A03'},
                       picks=['A03', 'A01'], epoch=(-0.1, 0.2))
    onsets = STIM_ONSETS[1:14:2]                     # A02-A03; 59.9 s is dropped
    assert trials.n_trials == len(onsets) and trials.ch_names == ['A03', 'A01']
    assert np.array_equal(trials.data, expected_epochs(onsets, (2, 0), -0.1, 0.2))
    assert list(trials.ch_info['name']) == ['A03', 'A01']

    by_query = read_bids(bids_root, '01', task='spes', select='stim_site == "A02-A03"')
    assert by_query.n_trials == trials.n_trials
    by_type = read_bids(bids_root, '01', task='spes', picks='ecog')
    assert by_type.ch_names == CH_NAMES


def test_second_subject_with_session_and_unit_strings(bids_root):

    trials = read_bids(bids_root, 'sub-02', session='1', task='spes', run='01')
    meta = trials.metadata
    assert list(meta['stim_site'][:2]) == ['A01-A02', 'A02-A03']      # '_' and ' ' normalised
    assert np.allclose(meta['stim_current_mA'], 4.0)
    assert np.allclose(meta['stim_pulse_width_us'], 200.0)
    assert set(meta['stim_polarity']) == {'biphasic_cathodic_first'}
    assert list(meta['session'].unique()) == ['1']

    # Without session=: the session folders are searched
    assert read_bids(bids_root, '02', task='spes').n_trials == trials.n_trials


def test_events_argument(bids_root, tmp_path):

    events = pd.DataFrame({'onset' : [5.0, 20.0], 'trial_type' : ['x', 'y']})
    trials = read_bids(bids_root, '01', task='spes', events=events)
    assert trials.n_trials == 2 and list(trials.metadata['condition']) == ['x', 'y']
    assert trials.modality is None

    path = tmp_path / 'other_events.tsv'
    path.write_text('onset\tduration\ttrial_type\n10.0\t0\tA01-A02\n', encoding='utf-8')
    trials = read_bids(bids_root, '01', task='spes', events=str(path))
    assert trials.n_trials == 1 and trials.metadata['stim_site'][0] == 'A01-A02'


def test_save_load_round_trip(bids_root, tmp_path):

    trials = read_bids(bids_root, '01', task='spes')
    trials.save(str(tmp_path / 'trials.evoparam'))
    loaded = TrialSet.load(str(tmp_path / 'trials.evoparam'))
    assert np.allclose(loaded.data, trials.data, rtol=1e-12, atol=0)
    pd.testing.assert_frame_equal(loaded.metadata, trials.metadata)
    pd.testing.assert_frame_equal(loaded.ch_info, trials.ch_info)
    assert loaded.provenance.source == trials.provenance.source
    assert loaded == trials
    assert read(str(tmp_path / 'trials.evoparam')).n_trials == trials.n_trials


def test_read_dispatch_on_bids_folder(bids_root):

    trials = read(bids_root, subject='01', task='spes')
    assert trials.n_trials == 15


def test_several_runs_are_concatenated(tmp_path):

    root = make_bids(tmp_path / 'two_runs')
    folder = os.path.join(root, 'sub-01', 'ieeg')
    for suffix in ('_ieeg.vhdr', '_ieeg.eeg', '_events.tsv', '_channels.tsv', '_ieeg.json'):
        src = os.path.join(folder, 'sub-01_task-spes_run-01' + suffix)
        dst = os.path.join(folder, 'sub-01_task-spes_run-02' + suffix)
        with open(src, 'rb') as file:
            content = file.read()
        if suffix == '_ieeg.vhdr':
            content = content.replace(b'run-01', b'run-02')
        with open(dst, 'wb') as file:
            file.write(content)
    trials = read_bids(root, '01', task='spes')
    assert trials.n_trials == 30
    assert list(trials.metadata['run'].unique()) == ['01', '02']
    assert len(trials.provenance.source['runs']) == 2

    # A run with other channels cannot be concatenated
    write_brainvision(folder, 'sub-01_task-spes_run-02_ieeg', signal(2, 60000), SFREQ,
                      ['A01', 'A02'])
    os.remove(os.path.join(folder, 'sub-01_task-spes_run-02_channels.tsv'))
    with pytest.raises(DataError, match='different channels'):
        read_bids(root, '01', task='spes')


def test_errors(bids_root, tmp_path):

    with pytest.raises(ReaderError, match='not found'):
        read_bids(tmp_path / 'nowhere', '01')
    with pytest.raises(ReaderError, match='No ieeg recording'):
        read_bids(bids_root, '03')
    with pytest.raises(EventError, match='matches the selection'):
        read_bids(bids_root, '01', task='spes', select={'stim_site' : 'A09-A10'})
    with pytest.raises(EventError, match='no column'):
        read_bids(bids_root, '01', task='spes', select={'nope' : 1})
    with pytest.raises(EventError, match='fits inside'):
        read_bids(bids_root, '01', task='spes', epoch=(-100.0, 100.0))

    # channels.tsv that does not match the data file
    root = make_bids(tmp_path / 'bad_channels')
    path = os.path.join(root, 'sub-01', 'ieeg', 'sub-01_task-spes_run-01_channels.tsv')
    text = open(path, encoding='utf-8').read().replace('A03', 'B07')
    with open(path, 'w', encoding='utf-8') as file:
        file.write(text)
    with pytest.raises(DataError, match='does not match the data file'):
        read_bids(root, '01', task='spes')
