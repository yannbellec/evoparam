# SPDX-License-Identifier: Apache-2.0
"""Read BrainVision recordings (.vhdr / .eeg), from a local file or a remote address.

BrainVision is a common format in BIDS iEEG datasets (for example OpenNeuro ds004080).
Only the needed samples are read: for a remote file, only the needed bytes are
downloaded, using HTTP range requests, so a few seconds can be read from a recording
of several GB. Binary data stored as INT_16, INT_32 or IEEE_FLOAT_32, multiplexed
or vectorized, are supported; the resolution of each channel is applied and every
voltage unit is converted to µV.
"""

import os
import re
from configparser import ConfigParser
from configparser import Error as ConfigError

import numpy as np

from evoparam.errors import DataError, ReaderError
from evoparam.errors import warn as _warn
from evoparam.io._fetch import RemoteFile, fetch_bytes, is_url, sha256_file

__all__ = [
    "read_brainvision",
    "read_brainvision_header",
    "read_brainvision_segment",
    "write_brainvision",
]

###################################################################################################
###################################################################################################

_DTYPES = {"IEEE_FLOAT_32": "<f4", "INT_16": "<i2", "INT_32": "<i4"}

# Factor from each voltage unit to µV. The micro sign (U+00B5) and the Greek mu
# (U+03BC) are both used in headers; the default unit of BrainVision is µV.
_TO_UV = {"µV": 1.0, "μV": 1.0, "uV": 1.0, "mV": 1e3, "V": 1e6, "nV": 1e-3}


def _decode(content):
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def _read_text(path, log=None):
    if is_url(path):
        return _decode(fetch_bytes(path, log=log))
    try:
        with open(path, "rb") as file:
            return _decode(file.read())
    except OSError as exc:
        raise ReaderError(f"Cannot read {path}: {exc}") from exc


def _unique_names(names):
    """Suffix duplicated channel names with -1, -2, ..., with a warning."""
    counts = {name: names.count(name) for name in names}
    duplicated = sorted(name for name, count in counts.items() if count > 1)
    if not duplicated:
        return list(names)
    seen, out = {}, []
    for name in names:
        if counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            out.append(f"{name}-{seen[name]}")
        else:
            out.append(name)
    _warn(
        "Duplicated channel names renamed with a suffix -1, -2, ...: {}.".format(
            ", ".join(duplicated)
        ),
        UserWarning,
        stacklevel=3,
    )
    return out


def read_brainvision_header(vhdr, *, log=None):
    """Read the header of a BrainVision recording.

    Parameters
    ----------
    vhdr : str
        Path or URL of the .vhdr header file.
    log : TransferLog, optional
        Receives the bytes transferred, for a remote header.

    Returns
    -------
    header : dict
        With keys 'data_file' (path or URL of the .eeg file), 'n_channels', 'sfreq' (Hz),
        'dtype' (numpy type string), 'ch_names' (list of str), 'resolutions'
        (1d array, the factor that converts stored values to the channel unit),
        'units' (unit of each channel, as written), 'to_uv' (1d array, factor from the
        channel unit to µV; 1 for channels whose unit is not a voltage),
        'orientation' ('MULTIPLEXED' or 'VECTORIZED') and 'n_samples'
        (from the DataPoints field, or None).

    Raises
    ------
    ReaderError
        If the header cannot be read, or the data are not binary, or the binary
        format or orientation is not supported.
    """
    text = _read_text(vhdr, log=log)

    # The header is an INI file, with a first line that is not a section: drop it
    if "[" not in text:
        raise ReaderError(f"{vhdr} is not a BrainVision header.")
    text = text[text.index("[") :]
    parser = ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    try:
        parser.read_string(text)
        common = parser["Common Infos"]
        channels = parser["Channel Infos"]
    except (ConfigError, KeyError) as exc:
        raise ReaderError(f"{vhdr}: malformed BrainVision header ({exc}).") from exc

    if common.get("DataFormat", "BINARY").upper() != "BINARY":
        raise ReaderError(f"{vhdr}: only binary BrainVision data are supported.")
    orientation = common.get("DataOrientation", "MULTIPLEXED").upper()
    if orientation not in ("MULTIPLEXED", "VECTORIZED"):
        raise ReaderError(f"{vhdr}: data orientation {orientation} is not supported.")

    n_channels = int(common["NumberOfChannels"])
    sfreq = 1e6 / float(common["SamplingInterval"])
    binary = parser["Binary Infos"] if parser.has_section("Binary Infos") else {}
    binary_format = binary.get("BinaryFormat", "INT_16").upper()
    if binary_format not in _DTYPES:
        raise ReaderError(f"{vhdr}: binary format {binary_format} is not supported.")

    ch_names, resolutions, units = [], [], []
    for ind in range(1, n_channels + 1):
        key = f"Ch{ind}"
        if key not in channels:
            raise ReaderError(f"{vhdr}: channel {key} is not described.")
        fields = re.split(r"(?<!\\),", channels[key])
        ch_names.append(fields[0].replace("\\1", ","))
        resolutions.append(float(fields[2]) if len(fields) > 2 and fields[2].strip() else 1.0)
        unit = fields[3].strip() if len(fields) > 3 and fields[3].strip() else "µV"
        units.append("µV" if unit == "�V" else unit)
    to_uv = np.array([_TO_UV.get(unit, 1.0) for unit in units])

    data_file = common["DataFile"]
    base = vhdr.rsplit("/", 1)[0] if is_url(vhdr) else os.path.dirname(vhdr)
    data_file = base + "/" + data_file if is_url(vhdr) else os.path.join(base, data_file)
    n_samples = int(common["DataPoints"]) if common.get("DataPoints", "").strip() else None

    return {
        "data_file": data_file,
        "n_channels": n_channels,
        "sfreq": sfreq,
        "dtype": _DTYPES[binary_format],
        "ch_names": _unique_names(ch_names),
        "resolutions": np.array(resolutions),
        "units": units,
        "to_uv": to_uv,
        "orientation": orientation,
        "n_samples": n_samples,
    }


class _LocalFile:
    """Local counterpart of RemoteFile: ``read(offset, nbytes)`` and ``size``."""

    def __init__(self, path):
        if not os.path.isfile(path):
            raise ReaderError(f"Data file not found: {path}")
        self.path = path
        self.size = os.path.getsize(path)
        self.bytes_read = 0
        self._file = open(path, "rb")

    def read(self, offset, nbytes):
        self._file.seek(offset)
        out = self._file.read(nbytes)
        self.bytes_read += len(out)
        return out

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _open_data(header, remote=None, *, chunk_size=1 << 20, cache_dir=None, log=None):
    """Open the binary data file of a header, locally or through range requests."""
    path = header["data_file"]
    if remote is None:
        remote = is_url(path)
    if remote and not is_url(path):
        raise ReaderError(f"remote=True needs an http(s) address, got {path}.")
    if not remote and is_url(path):
        raise ReaderError(f"remote=False but {path} is an http(s) address.")
    if remote:
        return RemoteFile(path, chunk_size=chunk_size, cache_dir=cache_dir, log=log)
    return _LocalFile(path)


def _n_samples(header, reader):
    """Number of samples per channel, from the file size."""
    itemsize = np.dtype(header["dtype"]).itemsize
    from_size = reader.size // (header["n_channels"] * itemsize)
    return from_size if header["n_samples"] is None else min(header["n_samples"], from_size)


def _scaling(header, ch_index):
    return header["resolutions"][ch_index] * header["to_uv"][ch_index]


def _read_windows(reader, header, starts, n_times, ch_index, n_samples):
    """Read windows of ``n_times`` samples starting at ``starts``.

    Returns
    -------
    data : 3d array
        As [n_windows, n_selected_channels, n_times], in µV, float64.
    """
    dtype, n_channels = np.dtype(header["dtype"]), header["n_channels"]
    itemsize = dtype.itemsize
    starts = np.asarray(starts, dtype=np.int64)
    ch_index = np.asarray(ch_index, dtype=int)
    out = np.empty((starts.size, ch_index.size, n_times))
    scale = _scaling(header, ch_index)[:, None]

    if header["orientation"] == "MULTIPLEXED":
        frame = n_channels * itemsize
        for ind, start in enumerate(starts):
            raw = reader.read(int(start) * frame, n_times * frame)
            if len(raw) != n_times * frame:
                raise ReaderError(
                    "{}: the file ends before sample {}.".format(
                        header["data_file"], int(start) + n_times
                    )
                )
            block = np.frombuffer(raw, dtype=dtype).reshape(n_times, n_channels)
            out[ind] = block[:, ch_index].T * scale
    else:
        # Vectorized: all samples of channel 1, then of channel 2, ...; read channel by
        # channel so that the windows of one channel share the cached chunks.
        for pos, channel in enumerate(ch_index):
            for ind, start in enumerate(starts):
                offset = (int(channel) * n_samples + int(start)) * itemsize
                raw = reader.read(offset, n_times * itemsize)
                if len(raw) != n_times * itemsize:
                    raise ReaderError(
                        "{}: the file ends before sample {}.".format(
                            header["data_file"], int(start) + n_times
                        )
                    )
                out[ind, pos] = np.frombuffer(raw, dtype=dtype) * scale[pos, 0]
    return out


def _pick(ch_names, picks):
    """Indices of the picked channels, by name or index, in the given order."""
    if picks is None:
        return list(range(len(ch_names)))
    if isinstance(picks, (str, int, np.integer)):
        picks = [picks]
    lookup = {name: ind for ind, name in enumerate(ch_names)}
    out = []
    for pick in picks:
        if isinstance(pick, str):
            if pick not in lookup:
                raise DataError(f"picks: unknown channel {pick!r}.")
            out.append(lookup[pick])
        elif isinstance(pick, (int, np.integer)) and -len(ch_names) <= pick < len(ch_names):
            out.append(int(pick) % len(ch_names))
        else:
            raise DataError(f"picks: cannot select a channel with {pick!r}.")
    if not out:
        raise DataError("picks: the selection is empty.")
    return out


def _warn_units(header, ch_index):
    """Warn about converted and non-voltage channel units among the picked channels."""
    units = [header["units"][ind] for ind in ch_index]
    names = [header["ch_names"][ind] for ind in ch_index]
    converted = [
        name
        for name, unit in zip(names, units, strict=False)
        if unit in _TO_UV and _TO_UV[unit] != 1.0
    ]
    other = [name for name, unit in zip(names, units, strict=False) if unit not in _TO_UV]
    if converted:
        _warn(
            "Channels in another voltage unit were converted to µV: {}.".format(
                ", ".join(converted)
            ),
            UserWarning,
            stacklevel=3,
        )
    if other:
        _warn(
            "Channels whose unit is not a voltage were left unconverted: {}.".format(
                ", ".join(other)
            ),
            UserWarning,
            stacklevel=3,
        )


def _segment(vhdr, header, first, stop, picks, remote, cache_dir=None, hash_file=False):
    """Samples ``first <= n < stop`` (clipped to the recording), in µV, and the record of
    the bytes read (the SHA-256 of a local data file only when ``hash_file``).
    """
    ch_index = _pick(header["ch_names"], picks)
    _warn_units(header, ch_index)
    with _open_data(header, remote, cache_dir=cache_dir) as reader:
        n_samples = _n_samples(header, reader)
        first, stop = max(int(first), 0), min(int(stop), n_samples)
        if stop <= first:
            raise ReaderError(
                f"{vhdr}: the requested time window is outside the recording ({n_samples} samples)."
            )
        data = _read_windows(reader, header, [first], stop - first, ch_index, n_samples)[0]
        if isinstance(reader, RemoteFile):
            source = {
                "sha256": reader.sha256(),
                "bytes_read": reader.bytes_read,
                "ranges": reader.ranges,
                "full_download": reader.full_download,
            }
        else:
            source = {
                "sha256": sha256_file(header["data_file"]) if hash_file else None,
                "bytes_read": reader.bytes_read,
            }
    source.update({"path": header["data_file"], "format": "brainvision"})
    return data, header["sfreq"], [header["ch_names"][ind] for ind in ch_index], first, source


def _sample(time, sfreq):
    """First sample at or after ``time`` (s), robust to rounding of the product."""
    return int(np.ceil(np.round(time * sfreq, 6)))


def read_brainvision(vhdr, *, tmin=None, tmax=None, picks=None, remote=None):
    """Read a BrainVision recording, or a time window of it.

    Parameters
    ----------
    vhdr : str
        Path or URL of the .vhdr header file.
    tmin, tmax : float, optional
        Window in seconds from the start of the recording; ``tmin`` included,
        ``tmax`` excluded. Default: the whole recording.
    picks : list of str or int, optional
        Channels to return, by name or index, in this order. Default: all.
    remote : bool, optional
        Read through HTTP range requests. Default: True when ``vhdr`` is an
        http(s) address.

    Returns
    -------
    data : 2d array
        Signal values in µV, as [n_channels, n_times].
    sfreq : float
        Sampling frequency, in Hz.
    ch_names : list of str
        Channel names, in the order of `data`.
    t0 : float
        Time of the first returned sample, in seconds from the start of the recording.

    Raises
    ------
    ReaderError
        Unreadable or unsupported file, window outside the recording.
    DataError
        Unknown channel in ``picks``.
    """
    data, sfreq, ch_names, t0, _ = _read_with_source(
        vhdr, tmin, tmax, picks, remote, hash_file=False
    )

    return data, sfreq, ch_names, t0


def _read_with_source(vhdr, tmin=None, tmax=None, picks=None, remote=None, hash_file=True):
    """read_brainvision, plus the record of the bytes read (sha256, bytes_read, ranges)."""
    header = read_brainvision_header(vhdr)
    sfreq = header["sfreq"]
    first = 0 if tmin is None else _sample(tmin, sfreq)
    stop = np.iinfo(np.int64).max if tmax is None else _sample(tmax, sfreq)
    data, sfreq, ch_names, first, source = _segment(
        vhdr, header, first, stop, picks, remote, hash_file=hash_file
    )

    return data, sfreq, ch_names, first / sfreq, source


def read_brainvision_segment(vhdr, tmin, tmax, picks=None):
    """Read a time segment of a BrainVision recording.

    Parameters
    ----------
    vhdr : str
        Path or URL of the .vhdr header file.
    tmin, tmax : float
        Start and end of the segment, in seconds from the start of the recording;
        both ends are included.
    picks : list of str, optional
        Names of the channels to return. If None, returns all channels.

    Returns
    -------
    data : 2d array
        Signal values, as [n_channels, n_times], in µV.
    sfreq : float
        Sampling frequency, in Hz.
    ch_names : list of str
        Channel names, in the order of `data`.
    t0 : float
        Time of the first returned sample, in seconds from the start of the recording.
    """
    header = read_brainvision_header(vhdr)
    sfreq = header["sfreq"]
    first = max(int(np.ceil(tmin * sfreq)), 0)
    stop = int(np.floor(tmax * sfreq)) + 1
    data, sfreq, ch_names, first, _ = _segment(vhdr, header, first, stop, picks, None)

    return data, sfreq, ch_names, first / sfreq


def write_brainvision(
    vhdr, data, sfreq, ch_names, *, markers=None, resolution=0.1, fmt="IEEE_FLOAT_32"
):
    """Write a BrainVision recording (``.vhdr``, ``.vmrk``, ``.eeg``; multiplexed).

    Parameters
    ----------
    vhdr : str
        Path of the header to write; the data and marker files take the same stem.
    data : ndarray, shape (n_channels, n_times)
        Samples in µV.
    sfreq : float
    ch_names : list of str
    markers : list of (sample, description), optional
        Stimulus markers (sample index from 0) written to the ``.vmrk`` file.
    resolution : float
        µV per stored unit (``IEEE_FLOAT_32`` keeps the values; ``INT_16`` rounds).
    fmt : {"IEEE_FLOAT_32", "INT_16"}

    Returns
    -------
    str
        The ``.vhdr`` path.
    """
    vhdr = str(vhdr)
    folder, name = os.path.split(vhdr)
    stem = name[:-5] if name.endswith(".vhdr") else name
    data = np.asarray(data, dtype=float)
    lines = [
        "Brain Vision Data Exchange Header File Version 1.0",
        "; written by evoparam",
        "",
        "[Common Infos]",
        "Codepage=UTF-8",
        f"DataFile={stem}.eeg",
        f"MarkerFile={stem}.vmrk",
        "DataFormat=BINARY",
        "DataOrientation=MULTIPLEXED",
        f"NumberOfChannels={len(ch_names)}",
        f"SamplingInterval={1e6 / float(sfreq)!r}",
        "",
        "[Binary Infos]",
        f"BinaryFormat={fmt}",
        "",
        "[Channel Infos]",
    ]
    lines += [
        f"Ch{i + 1}={ch.replace(',', chr(92) + '1')},,{resolution!r},µV"
        for i, ch in enumerate(ch_names)
    ]
    with open(os.path.join(folder, stem + ".vhdr"), "w", encoding="utf-8") as fobj:
        fobj.write("\n".join(lines) + "\n")
    marks = [
        "Brain Vision Data Exchange Marker File, Version 1.0",
        "",
        "[Common Infos]",
        "Codepage=UTF-8",
        f"DataFile={stem}.eeg",
        "",
        "[Marker Infos]",
        "Mk1=New Segment,,1,1,0",
    ]
    for k, (sample, description) in enumerate(markers or [], start=2):
        marks.append(f"Mk{k}=Stimulus,{description},{int(sample) + 1},1,0")
    with open(os.path.join(folder, stem + ".vmrk"), "w", encoding="utf-8") as fobj:
        fobj.write("\n".join(marks) + "\n")
    stored = data / resolution
    dtype = {"IEEE_FLOAT_32": "<f4", "INT_16": "<i2"}[fmt]
    if dtype != "<f4":
        stored = np.round(stored)
    stored.T.astype(dtype).tofile(os.path.join(folder, stem + ".eeg"))
    return os.path.join(folder, stem + ".vhdr")
