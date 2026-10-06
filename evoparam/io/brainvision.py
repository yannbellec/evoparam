"""Read BrainVision recordings (.vhdr / .eeg), from a local file or a remote address.

BrainVision is a common format in BIDS iEEG datasets (for example OpenNeuro ds004080).
Only a time segment is read: for a remote file, only the needed bytes are downloaded,
using an HTTP range request, so a few seconds can be read from a recording of several GB.
"""

import os
import re
from configparser import ConfigParser
from urllib.request import Request, urlopen

import numpy as np

###################################################################################################
###################################################################################################

_DTYPES = {'IEEE_FLOAT_32' : '<f4', 'INT_16' : '<i2', 'INT_32' : '<i4'}


def _is_url(path):
    return path.startswith(('http://', 'https://'))


def _read_text(path):
    if _is_url(path):
        with urlopen(path) as response:
            return response.read().decode('utf-8', errors='replace')
    with open(path, encoding='utf-8', errors='replace') as file:
        return file.read()


def read_brainvision_header(vhdr):
    """Read the header of a BrainVision recording.

    Parameters
    ----------
    vhdr : str
        Path or URL of the .vhdr header file.

    Returns
    -------
    header : dict
        With keys 'data_file' (path or URL of the .eeg file), 'n_channels', 'sfreq' (Hz),
        'dtype' (numpy type string), 'ch_names' (list of str) and 'resolutions'
        (1d array, the factor that converts stored values to physical units, per channel).

    Raises
    ------
    ValueError
        If the file is not stored as multiplexed binary data.
    """

    text = _read_text(vhdr)

    # The header is an INI file, with a first line that is not a section: drop it
    text = text[text.index('['):]
    parser = ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    parser.read_string(text)

    common = parser['Common Infos']
    if common.get('DataFormat', 'BINARY').upper() != 'BINARY' or \
        common.get('DataOrientation', 'MULTIPLEXED').upper() != 'MULTIPLEXED':
        raise ValueError('Only multiplexed binary BrainVision files are supported.')

    n_channels = int(common['NumberOfChannels'])
    sfreq = 1e6 / float(common['SamplingInterval'])
    binary_format = parser['Binary Infos'].get('BinaryFormat', 'INT_16').upper()
    if binary_format not in _DTYPES:
        raise ValueError('Binary format {} not supported.'.format(binary_format))

    ch_names, resolutions = [], []
    channels = parser['Channel Infos']
    for ind in range(1, n_channels + 1):
        fields = re.split(r'(?<!\\),', channels['Ch{}'.format(ind)])
        ch_names.append(fields[0].replace('\\1', ','))
        resolutions.append(float(fields[2]) if len(fields) > 2 and fields[2] else 1.0)

    data_file = common['DataFile']
    base = vhdr.rsplit('/', 1)[0] if _is_url(vhdr) else os.path.dirname(vhdr)
    data_file = base + '/' + data_file if _is_url(vhdr) else os.path.join(base, data_file)

    return {'data_file' : data_file, 'n_channels' : n_channels, 'sfreq' : sfreq,
            'dtype' : _DTYPES[binary_format], 'ch_names' : ch_names,
            'resolutions' : np.array(resolutions)}


def read_brainvision_segment(vhdr, tmin, tmax, picks=None):
    """Read a time segment of a BrainVision recording.

    Parameters
    ----------
    vhdr : str
        Path or URL of the .vhdr header file.
    tmin, tmax : float
        Start and end of the segment, in seconds from the start of the recording.
    picks : list of str, optional
        Names of the channels to return. If None, returns all channels.

    Returns
    -------
    data : 2d array
        Signal values, as [n_channels, n_times], in the units of the recording.
    sfreq : float
        Sampling frequency, in Hz.
    ch_names : list of str
        Channel names, in the order of `data`.
    t0 : float
        Time of the first returned sample, in seconds from the start of the recording.
    """

    header = read_brainvision_header(vhdr)
    sfreq, n_channels = header['sfreq'], header['n_channels']
    itemsize = np.dtype(header['dtype']).itemsize

    first = max(int(np.ceil(tmin * sfreq)), 0)
    n_times = int(np.floor(tmax * sfreq)) - first + 1
    start, n_bytes = first * n_channels * itemsize, n_times * n_channels * itemsize

    if _is_url(header['data_file']):
        request = Request(header['data_file'],
                          headers={'Range' : 'bytes={}-{}'.format(start, start + n_bytes - 1)})
        with urlopen(request) as response:
            raw = response.read()
    else:
        with open(header['data_file'], 'rb') as file:
            file.seek(start)
            raw = file.read(n_bytes)

    n_times = len(raw) // (n_channels * itemsize)
    data = np.frombuffer(raw[:n_times * n_channels * itemsize], dtype=header['dtype'])
    data = data.reshape(n_times, n_channels).T * header['resolutions'][:, None]

    ch_names = header['ch_names']
    if picks is not None:
        inds = [ch_names.index(name) for name in picks]
        data, ch_names = data[inds], [ch_names[ind] for ind in inds]

    return data, sfreq, ch_names, first / sfreq
