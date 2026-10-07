"""Writers of small BrainVision files and BIDS datasets, and a local HTTP server (io tests)."""

import contextlib
import http.server
import json
import os
import re
import threading

import numpy as np

###################################################################################################
###################################################################################################

SFREQ = 1000.
DURATION = 60.
CH_NAMES = ['A01', 'A02', 'A03']

# 20 events: 16 stimulations (one 0.8 s after the previous one, one too close to the end
# of the 60 s recording) and 4 artefacts.
STIM_ONSETS = [2.0 + 3 * k for k in range(14)] + [41.8, 59.9]
ARTEFACT_ONSETS = [50.0, 52.0, 54.0, 56.0]
SITES = ['A01-A02', 'A02-A03']


def signal(n_channels, n_samples):
    """Known samples (µV): channel c at sample n is c * 1e5 + n."""

    return np.arange(n_channels)[:, None] * 1e5 + np.arange(n_samples)[None, :]


def write_brainvision(folder, stem, data, sfreq, ch_names, fmt='IEEE_FLOAT_32',
                      orientation='MULTIPLEXED', resolutions=None, units=None):
    """Write a BrainVision recording (data in µV, [n_channels, n_times]); return the .vhdr path."""

    folder = str(folder)
    n_channels = len(ch_names)
    resolutions = [0.5] * n_channels if resolutions is None else list(resolutions)
    units = ['µV'] * n_channels if units is None else list(units)
    lines = ['Brain Vision Data Exchange Header File Version 1.0', '; test file', '',
             '[Common Infos]', 'Codepage=UTF-8', 'DataFile={}.eeg'.format(stem),
             'MarkerFile={}.vmrk'.format(stem), 'DataFormat=BINARY',
             'DataOrientation={}'.format(orientation),
             'NumberOfChannels={}'.format(n_channels),
             'SamplingInterval={}'.format(1e6 / sfreq), '', '[Binary Infos]',
             'BinaryFormat={}'.format(fmt), '', '[Channel Infos]']
    lines += ['Ch{}={},,{},{}'.format(ind + 1, name.replace(',', '\\1'), res, unit)
              for ind, (name, res, unit) in enumerate(zip(ch_names, resolutions, units))]
    with open(os.path.join(folder, stem + '.vhdr'), 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines) + '\n')

    # Stored value = value in the channel unit / resolution
    to_unit = np.array([{'µV' : 1., 'mV' : 1e-3, 'V' : 1e-6}.get(unit, 1.) for unit in units])
    stored = np.asarray(data, dtype=float) * to_unit[:, None] / np.array(resolutions)[:, None]
    dtype = {'IEEE_FLOAT_32' : '<f4', 'INT_16' : '<i2', 'INT_32' : '<i4'}[fmt]
    if dtype != '<f4':
        stored = np.round(stored)
    stored = stored.astype(dtype)
    stored = stored.T if orientation == 'MULTIPLEXED' else stored
    stored.tofile(os.path.join(folder, stem + '.eeg'))

    return os.path.join(folder, stem + '.vhdr')


def _tsv(path, header, rows):
    with open(path, 'w', encoding='utf-8') as file:
        file.write('\t'.join(header) + '\n')
        for row in rows:
            file.write('\t'.join(str(value) for value in row) + '\n')


def make_bids(root):
    """Mini BIDS-iEEG dataset: 2 subjects, 3 channels, 20 events per run.

    sub-01 (no session): current in A with Units in the root events.json, pulse width
    in s without units, sites 'A-B'. sub-02 (session 1): currents and pulse widths as
    strings with units ("4.0 mA", "200 µs"), sites written with other separators, and
    run-level sidecars.
    """

    root = str(root)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, 'dataset_description.json'), 'w', encoding='utf-8') as file:
        json.dump({'Name' : 'mini', 'BIDSVersion' : '1.9.0', 'DatasetDOI' : '10.0000/mini'}, file)
    with open(os.path.join(root, 'events.json'), 'w', encoding='utf-8') as file:
        json.dump({'electrical_stimulation_current' : {'Description' : 'current',
                                                       'Units' : 'A'}}, file)

    n_samples = int(DURATION * SFREQ)
    data = signal(len(CH_NAMES), n_samples)
    for subject, session in (('01', None), ('02', '1')):
        folder = os.path.join(root, 'sub-' + subject, *(['ses-' + session] if session else []),
                              'ieeg')
        os.makedirs(folder)
        stem = 'sub-{}{}_task-spes_run-01'.format(subject, '_ses-' + session if session else '')
        write_brainvision(folder, stem + '_ieeg', data, SFREQ, CH_NAMES)
        _tsv(os.path.join(folder, stem + '_channels.tsv'),
             ['name', 'type', 'units', 'low_cutoff', 'high_cutoff', 'reference', 'group', 'status'],
             [[name, 'ECOG', 'µV', 'n/a', 'n/a', 'G2', 'grid', 'bad' if name == 'A03' else 'good']
              for name in CH_NAMES])
        with open(os.path.join(folder, stem + '_ieeg.json'), 'w', encoding='utf-8') as file:
            json.dump({'TaskName' : 'spes', 'SamplingFrequency' : SFREQ,
                       'ElectricalStimulation' : True,
                       'ElectricalStimulationParameters' : 'cathodic pulses'}, file)

        rows = []
        for ind, onset in enumerate(STIM_ONSETS):
            site = SITES[ind % 2]
            if subject == '01':
                rows.append([onset, 0.001, 'electrical_stimulation', site, 0.004, 0.0002,
                             'monophasic'])
            else:
                site = site.replace('-', ' ' if ind % 2 else '_')
                rows.append([onset, 0.001, 'electrical_stimulation', site, '4.0 mA',
                             '200 µs' if ind % 2 else '0.2ms', 'biphasic'])
        for onset in ARTEFACT_ONSETS:
            rows.append([onset, 0.5, 'artefact', 'n/a', 'n/a', 'n/a', 'n/a'])
        rows.sort(key=lambda row: row[0])
        _tsv(os.path.join(folder, stem + '_events.tsv'),
             ['onset', 'duration', 'trial_type', 'electrical_stimulation_site',
              'electrical_stimulation_current', 'electrical_stimulation_pulsewidth',
              'electrical_stimulation_type'], rows)
        if subject == '02':
            # Run-level sidecar: overrides the Units of the root events.json
            with open(os.path.join(folder, stem + '_events.json'), 'w', encoding='utf-8') as file:
                json.dump({'electrical_stimulation_current' : {'Units' : 'mA'}}, file)
        else:
            _tsv(os.path.join(folder, 'sub-01_electrodes.tsv'), ['name', 'x', 'y', 'z'],
                 [[name, 1.0 * ind, 2.0, 3.0] for ind, name in enumerate(CH_NAMES)])
            with open(os.path.join(folder, 'sub-01_coordsystem.json'), 'w',
                      encoding='utf-8') as file:
                json.dump({'iEEGCoordinateSystem' : 'ACPC', 'iEEGCoordinateUnits' : 'cm'}, file)
    return root


###################################################################################################
## Local HTTP server

class _Stats:
    def __init__(self):
        self.requests = []           # (method, path, range header or None, status)
        self.bytes_sent = 0
        self.failures_left = 0


def _handler(directory, stats, honour_range):

    class Handler(http.server.BaseHTTPRequestHandler):

        def log_message(self, *args):
            pass

        def _path(self):
            rel = self.path.split('?')[0].lstrip('/')
            return os.path.join(directory, *rel.split('/'))

        def _fail(self):
            if stats.failures_left > 0:
                stats.failures_left -= 1
                stats.requests.append((self.command, self.path, None, 503))
                self.send_error(503)
                return True
            return False

        def do_HEAD(self):
            if self._fail():
                return
            path = self._path()
            if not os.path.isfile(path):
                stats.requests.append(('HEAD', self.path, None, 404))
                self.send_error(404)
                return
            stats.requests.append(('HEAD', self.path, None, 200))
            self.send_response(200)
            self.send_header('Content-Length', str(os.path.getsize(path)))
            self.end_headers()

        def do_GET(self):
            if self._fail():
                return
            path = self._path()
            byte_range = self.headers.get('Range')
            if not os.path.isfile(path):
                stats.requests.append(('GET', self.path, byte_range, 404))
                self.send_error(404)
                return
            size = os.path.getsize(path)
            match = re.match(r'bytes=(\d+)-(\d*)$', byte_range or '')
            if honour_range and match:
                start = int(match.group(1))
                stop = min(int(match.group(2)) if match.group(2) else size - 1, size - 1)
                if start >= size:
                    stats.requests.append(('GET', self.path, byte_range, 416))
                    self.send_error(416)
                    return
                with open(path, 'rb') as file:
                    file.seek(start)
                    body = file.read(stop - start + 1)
                self.send_response(206)
                self.send_header('Content-Range', 'bytes {}-{}/{}'.format(start, stop, size))
                status = 206
            else:
                with open(path, 'rb') as file:
                    body = file.read()
                self.send_response(200)
                status = 200
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            stats.bytes_sent += len(body)
            stats.requests.append(('GET', self.path, byte_range, status))

    return Handler


@contextlib.contextmanager
def http_server(directory, honour_range=True, failures=0):
    """Serve ``directory`` on 127.0.0.1; yield (base URL, stats).

    With ``honour_range=False`` the server ignores Range headers and answers 200 with the
    whole file. ``failures`` first requests are answered 503.
    """

    stats = _Stats()
    stats.failures_left = failures
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0),
                                             _handler(str(directory), stats, honour_range))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield 'http://127.0.0.1:{}/'.format(server.server_address[1]), stats
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
