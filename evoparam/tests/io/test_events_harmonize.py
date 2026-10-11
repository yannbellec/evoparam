# SPDX-License-Identifier: Apache-2.0
"""Harmonisation of stimulation events and the validator report (REQ-022)."""

import io

import numpy as np
import pytest

pd = pytest.importorskip('pandas')

from evoparam.errors import DataError
from evoparam.io import (harmonize_stim_events, validate_stimulation_events,
                         select_stimulation_events, HarmonizationReport, ValidationReport)

###################################################################################################
###################################################################################################

def table(text):
    return pd.read_csv(io.StringIO(text), sep='\t', na_values=['n/a'], keep_default_na=False)


# Shaped like the events.tsv of ds004080: currents in A and pulse widths in s, without Units.
UTRECHT = table(
    'onset\tduration\ttrial_type\tsub_type\telectrodes_involved_onset\telectrical_stimulation_type'
    '\telectrical_stimulation_site\telectrical_stimulation_current'
    '\telectrical_stimulation_frequency\telectrical_stimulation_pulsewidth\tnotes\n'
    '507.0\t0.23\tartefact\tn/a\tall\tn/a\tn/a\tn/a\tn/a\tn/a\tn/a\n'
    '2881.78\t0.001\telectrical_stimulation\tSPESclin\tn/a\tmonophasic\tPT01-PT02\t0.008\t0.2\t0.001\tn/a\n'
    '2886.78\t0.001\telectrical_stimulation\tSPESclin\tn/a\tmonophasic\tPT01-PT02\t0.008\t0.2\t0.001\tn/a\n'
    '2950.10\t0.001\telectrical_stimulation\tSPESclin\tn/a\tmonophasic\tPT03-PT02\t0.004\t0.2\t0.001\tn/a\n')


class _no_warning:
    """Context manager failing on any warning (harmonisation reports, it does not warn)."""

    def __enter__(self):
        import warnings
        self._catch = warnings.catch_warnings()
        self._catch.__enter__()
        warnings.simplefilter('error')

    def __exit__(self, *exc):
        self._catch.__exit__(*exc)


def test_utrecht_columns():

    with _no_warning():
        out, report = harmonize_stim_events(UTRECHT)
    stim = out[out['trial_type'] == 'electrical_stimulation']

    # Every stimulation row has site, current and pulse width (REQ-022)
    assert (stim['stim_site'] != 'n/a').all()
    assert stim['stim_current_mA'].notna().all() and stim['stim_pulse_width_us'].notna().all()
    assert list(stim['stim_current_mA']) == pytest.approx([8.0, 8.0, 4.0])
    assert list(stim['stim_pulse_width_us']) == pytest.approx([1000.0] * 3)
    assert list(stim['stim_frequency_hz']) == pytest.approx([0.2] * 3)
    assert list(stim['stim_charge_uC']) == pytest.approx([8.0, 8.0, 4.0])
    assert list(stim['stim_site']) == ['PT01-PT02', 'PT01-PT02', 'PT03-PT02']

    # Units inferred, with a warning; polarity unknown (phase order not stated)
    assert any(line.startswith('unit_inferred: electrical_stimulation_current')
               for line in report.warnings)
    assert any(line.startswith('unit_inferred: electrical_stimulation_pulsewidth')
               for line in report.warnings)
    assert report.units['stim_current_mA'] == 'inferred:A'
    assert set(stim['stim_polarity']) == {'unknown'}
    assert list(stim['evoparam_stim_type_raw']) == ['monophasic'] * 3

    # Artefact row: missing values, condition copied from trial_type, originals kept
    assert out.loc[0, 'stim_site'] == 'n/a' and np.isnan(out.loc[0, 'stim_current_mA'])
    assert list(out['condition']) == list(out['trial_type'])
    assert 'electrical_stimulation_site' in out.columns and 'notes' in out.columns
    assert report.mapping['stim_site'] == 'electrical_stimulation_site'
    assert report.mapping['stim_n_pulses'] is None and 'stim_n_pulses' in report.missing
    assert out.attrs['units']['stim_current_mA'] == 'mA'
    assert isinstance(report, HarmonizationReport) and report.n_rows == 4


def test_sidecar_units_and_unit_strings():

    events = pd.DataFrame({
        'onset' : [1.0, 2.0, 3.0, 4.0, 5.0],
        'trial_type' : ['electrical_stimulation'] * 5,
        'electrical_stimulation_site' : ['A1 - A2', 'A1_A2', 'A1,A2', 'A1 A2', 'n/a'],
        'stim_current' : ['4.0 mA', '6mA', '4000 µA', 'n/a', '0.5 A'],
        'pulse_width' : ['1000 µs', '1ms', '200 us', '0.2 ms', 'n/a'],
        'frequency' : ['0.2 Hz', 'n/a', '1', '1 kHz', 'n/a'],
        'n_pulses' : [1, 1, 'n/a', 2, 1],
    })
    out, report = harmonize_stim_events(events)
    assert list(out['stim_site']) == ['A1-A2'] * 4 + ['n/a']
    assert np.allclose(out['stim_current_mA'], [4.0, 6.0, 4.0, np.nan, 500.0], equal_nan=True)
    assert np.allclose(out['stim_pulse_width_us'], [1000.0, 1000.0, 200.0, 200.0, np.nan],
                       equal_nan=True)
    assert np.allclose(out['stim_frequency_hz'], [0.2, np.nan, 1.0, 1000.0, np.nan],
                       equal_nan=True)
    assert out['stim_n_pulses'].tolist()[:2] == [1, 1] and out['stim_n_pulses'].isna()[2]
    assert report.units['stim_current_mA'] == 'inline'
    assert not any(line.startswith('unit_inferred: stim_current') for line in report.warnings)

    # Units from the sidecar
    events = pd.DataFrame({'onset' : [1.0], 'trial_type' : ['electrical_stimulation'],
                           'current' : [3000.0], 'pulsewidth' : [0.3]})
    sidecar = {'current' : {'Units' : 'uA'}, 'PulseWidth' : {'Units' : 'ms'},
               'pulsewidth' : {'units' : 'ms'}}
    out, report = harmonize_stim_events(events, sidecar)
    assert out.loc[0, 'stim_current_mA'] == pytest.approx(3.0)
    assert out.loc[0, 'stim_pulse_width_us'] == pytest.approx(300.0)
    assert report.units == {'stim_current_mA' : 'sidecar:uA', 'stim_pulse_width_us' : 'sidecar:ms'}


def test_unknown_units_and_unparsable_values():

    events = pd.DataFrame({'onset' : [1.0, 2.0], 'current' : ['4 furlongs', 'high'],
                           'trial_type' : ['stim', 'stim']})
    out, report = harmonize_stim_events(events)
    assert out['stim_current_mA'].isna().all()
    assert any(line.startswith('unit_unknown') for line in report.warnings)
    assert any(line.startswith('unparsable') for line in report.warnings)

    out, report = harmonize_stim_events(events.assign(current=[1.0, 2.0]),
                                        {'current' : {'Units' : 'parsecs'}})
    assert out['stim_current_mA'].isna().all()
    assert any('parsecs' in line for line in report.warnings)


def test_pair_pattern_in_trial_type():

    events = pd.DataFrame({'onset' : [1.0, 2.0, 3.0], 'trial_type' : ['PT01-PT02', 'rest', 'C3-C4']})
    out, report = harmonize_stim_events(events)
    assert list(out['stim_site']) == ['PT01-PT02', 'n/a', 'C3-C4']
    assert report.mapping['stim_site'] == 'trial_type'
    assert any(line.startswith('site_from_trial_type') for line in report.warnings)
    assert list(out['condition']) == ['PT01-PT02', 'rest', 'C3-C4']


def test_electrodes_involved_onset_only_for_stimulation_rows():

    events = pd.DataFrame({'onset' : [1.0, 2.0], 'trial_type' : ['artefact', 'stimulation'],
                           'electrodes_involved_onset' : ['all', 'C1;C2']})
    out, _ = harmonize_stim_events(events)
    assert list(out['stim_site']) == ['n/a', 'C1;C2']


def test_polarity_vocabulary():

    events = pd.DataFrame({'onset' : np.arange(6.0),
                           'stim_type' : ['biphasic, cathodic first', 'Biphasic anodic-first',
                                          'monophasic cathodic', 'alternating', 'monophasic',
                                          'n/a']})
    out, report = harmonize_stim_events(events)
    assert list(out['stim_polarity']) == ['biphasic_cathodic_first', 'biphasic_anodic_first',
                                          'monophasic_cathodic', 'alternating', 'unknown', 'n/a']

    # The phase order may come from ElectricalStimulationParameters
    out, _ = harmonize_stim_events(events.iloc[[4]],
                                   {'ElectricalStimulationParameters' : 'anodic pulses'})
    assert list(out['stim_polarity']) == ['monophasic_anodic']


def test_user_mapping_overrides_the_table():

    events = pd.DataFrame({'onset' : [1.0], 'current' : [0.008], 'my_ma' : [2.5],
                           'trial_type' : ['electrical_stimulation']})
    out, report = harmonize_stim_events(events, {'my_ma' : {'Units' : 'mA'}},
                                        mapping={'my_ma' : 'stim_current_mA'})
    assert out.loc[0, 'stim_current_mA'] == 2.5
    assert report.mapping['stim_current_mA'] == 'my_ma'
    assert report.user_mapping == {'my_ma' : 'stim_current_mA'}

    with pytest.raises(DataError, match='unknown target'):
        harmonize_stim_events(events, mapping={'my_ma' : 'stim_voltage'})
    with pytest.raises(DataError, match='not in the events table'):
        harmonize_stim_events(events, mapping={'nope' : 'stim_current_mA'})


def test_harmonising_twice_keeps_values():

    once, _ = harmonize_stim_events(UTRECHT)
    standard = once.drop(columns=[c for c in UTRECHT.columns if c not in ('onset', 'trial_type')])
    twice, report = harmonize_stim_events(standard)
    assert np.allclose(twice['stim_current_mA'], once['stim_current_mA'], equal_nan=True)
    assert report.units['stim_current_mA'] == 'native'


def test_validator_reports_missing_fields():

    # Only trial_type: everything is missing, nothing raises
    minimal = pd.DataFrame({'onset' : [1.0, 2.0], 'duration' : [0.001, 0.001],
                            'trial_type' : ['electrical_stimulation'] * 2})
    report = validate_stimulation_events(minimal)
    assert isinstance(report, ValidationReport)
    assert report.missing == ['stim_site', 'stim_current_mA', 'stim_pulse_width_us',
                              'stim_polarity', 'stim_frequency_hz', 'stim_n_pulses']
    assert report.n_rows == 2 and report.n_sites == 0 and not report.complete

    # Utrecht shape: the artefact row is not checked; polarity and pulse count are missing
    report = validate_stimulation_events(UTRECHT)
    assert report.n_rows == 3 and report.sites == ['PT01-PT02', 'PT03-PT02']
    assert report.missing == ['stim_polarity', 'stim_n_pulses']
    assert any(line.startswith('unit_inferred') for line in report.warnings)

    # Partially filled field
    partial = UTRECHT.copy()
    partial.loc[2, 'electrical_stimulation_current'] = np.nan
    report = validate_stimulation_events(partial)
    assert any(line.startswith('partial: stim_current_mA is missing in 1 of 3')
               for line in report.warnings)
    assert report.to_dict()['n_sites'] == 2


def test_select_stimulation_events_on_tables():

    out, _ = harmonize_stim_events(UTRECHT)
    assert list(select_stimulation_events(out).index) == [1, 2, 3]
    assert list(select_stimulation_events(out, site='PT03_PT02').index) == [3]
    assert list(select_stimulation_events(out, current=8.0).index) == [1, 2]
    assert len(select_stimulation_events(out, trial_type=None)) == 4

    # Inherited list form
    events = out.to_dict('records')
    assert len(select_stimulation_events(events, site='PT01-PT02')) == 2
