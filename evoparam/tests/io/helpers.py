# SPDX-License-Identifier: Apache-2.0
"""Writers of small BrainVision files, MEF 3.0 sessions and BIDS datasets, and a local HTTP
server (io tests)."""

import contextlib
import http.server
import json
import os
import re
import threading
from urllib.parse import parse_qs, unquote, urlsplit
from xml.sax.saxutils import escape

import numpy as np

###################################################################################################
###################################################################################################

SFREQ = 1000.0
DURATION = 60.0
CH_NAMES = ["A01", "A02", "A03"]

# 20 events: 16 stimulations (one 0.8 s after the previous one, one too close to the end
# of the 60 s recording) and 4 artefacts.
STIM_ONSETS = [2.0 + 3 * k for k in range(14)] + [41.8, 59.9]
ARTEFACT_ONSETS = [50.0, 52.0, 54.0, 56.0]
SITES = ["A01-A02", "A02-A03"]


def signal(n_channels, n_samples):
    """Known samples (µV): channel c at sample n is c * 1e5 + n."""

    return np.arange(n_channels)[:, None] * 1e5 + np.arange(n_samples)[None, :]


def write_brainvision(
    folder,
    stem,
    data,
    sfreq,
    ch_names,
    fmt="IEEE_FLOAT_32",
    orientation="MULTIPLEXED",
    resolutions=None,
    units=None,
):
    """Write a BrainVision recording (data in µV, [n_channels, n_times]); return the .vhdr path."""

    folder = str(folder)
    n_channels = len(ch_names)
    resolutions = [0.5] * n_channels if resolutions is None else list(resolutions)
    units = ["µV"] * n_channels if units is None else list(units)
    lines = [
        "Brain Vision Data Exchange Header File Version 1.0",
        "; test file",
        "",
        "[Common Infos]",
        "Codepage=UTF-8",
        "DataFile={}.eeg".format(stem),
        "MarkerFile={}.vmrk".format(stem),
        "DataFormat=BINARY",
        "DataOrientation={}".format(orientation),
        "NumberOfChannels={}".format(n_channels),
        "SamplingInterval={}".format(1e6 / sfreq),
        "",
        "[Binary Infos]",
        "BinaryFormat={}".format(fmt),
        "",
        "[Channel Infos]",
    ]
    lines += [
        "Ch{}={},,{},{}".format(ind + 1, name.replace(",", "\\1"), res, unit)
        for ind, (name, res, unit) in enumerate(zip(ch_names, resolutions, units))
    ]
    with open(os.path.join(folder, stem + ".vhdr"), "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    # Stored value = value in the channel unit / resolution
    to_unit = np.array([{"µV": 1.0, "mV": 1e-3, "V": 1e-6}.get(unit, 1.0) for unit in units])
    stored = np.asarray(data, dtype=float) * to_unit[:, None] / np.array(resolutions)[:, None]
    dtype = {"IEEE_FLOAT_32": "<f4", "INT_16": "<i2", "INT_32": "<i4"}[fmt]
    if dtype != "<f4":
        stored = np.round(stored)
    stored = stored.astype(dtype)
    stored = stored.T if orientation == "MULTIPLEXED" else stored
    stored.tofile(os.path.join(folder, stem + ".eeg"))

    return os.path.join(folder, stem + ".vhdr")


def _tsv(path, header, rows):
    with open(path, "w", encoding="utf-8") as file:
        file.write("\t".join(header) + "\n")
        for row in rows:
            file.write("\t".join(str(value) for value in row) + "\n")


def make_bids(root):
    """Mini BIDS-iEEG dataset: 2 subjects, 3 channels, 20 events per run.

    sub-01 (no session): current in A with Units in the root events.json, pulse width
    in s without units, sites 'A-B'. sub-02 (session 1): currents and pulse widths as
    strings with units ("4.0 mA", "200 µs"), sites written with other separators, and
    run-level sidecars.
    """

    root = str(root)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "dataset_description.json"), "w", encoding="utf-8") as file:
        json.dump({"Name": "mini", "BIDSVersion": "1.9.0", "DatasetDOI": "10.0000/mini"}, file)
    with open(os.path.join(root, "events.json"), "w", encoding="utf-8") as file:
        json.dump(
            {"electrical_stimulation_current": {"Description": "current", "Units": "A"}}, file
        )

    n_samples = int(DURATION * SFREQ)
    data = signal(len(CH_NAMES), n_samples)
    for subject, session in (("01", None), ("02", "1")):
        folder = os.path.join(
            root, "sub-" + subject, *(["ses-" + session] if session else []), "ieeg"
        )
        os.makedirs(folder)
        stem = "sub-{}{}_task-spes_run-01".format(subject, "_ses-" + session if session else "")
        write_brainvision(folder, stem + "_ieeg", data, SFREQ, CH_NAMES)
        _tsv(
            os.path.join(folder, stem + "_channels.tsv"),
            ["name", "type", "units", "low_cutoff", "high_cutoff", "reference", "group", "status"],
            [
                [
                    name,
                    "ECOG",
                    "µV",
                    "n/a",
                    "n/a",
                    "G2",
                    "grid",
                    "bad" if name == "A03" else "good",
                ]
                for name in CH_NAMES
            ],
        )
        with open(os.path.join(folder, stem + "_ieeg.json"), "w", encoding="utf-8") as file:
            json.dump(
                {
                    "TaskName": "spes",
                    "SamplingFrequency": SFREQ,
                    "ElectricalStimulation": True,
                    "ElectricalStimulationParameters": "cathodic pulses",
                },
                file,
            )

        rows = []
        for ind, onset in enumerate(STIM_ONSETS):
            site = SITES[ind % 2]
            if subject == "01":
                rows.append(
                    [onset, 0.001, "electrical_stimulation", site, 0.004, 0.0002, "monophasic"]
                )
            else:
                site = site.replace("-", " " if ind % 2 else "_")
                rows.append(
                    [
                        onset,
                        0.001,
                        "electrical_stimulation",
                        site,
                        "4.0 mA",
                        "200 µs" if ind % 2 else "0.2ms",
                        "biphasic",
                    ]
                )
        for onset in ARTEFACT_ONSETS:
            rows.append([onset, 0.5, "artefact", "n/a", "n/a", "n/a", "n/a"])
        rows.sort(key=lambda row: row[0])
        _tsv(
            os.path.join(folder, stem + "_events.tsv"),
            [
                "onset",
                "duration",
                "trial_type",
                "electrical_stimulation_site",
                "electrical_stimulation_current",
                "electrical_stimulation_pulsewidth",
                "electrical_stimulation_type",
            ],
            rows,
        )
        if subject == "02":
            # Run-level sidecar: overrides the Units of the root events.json
            with open(os.path.join(folder, stem + "_events.json"), "w", encoding="utf-8") as file:
                json.dump({"electrical_stimulation_current": {"Units": "mA"}}, file)
        else:
            _tsv(
                os.path.join(folder, "sub-01_electrodes.tsv"),
                ["name", "x", "y", "z"],
                [[name, 1.0 * ind, 2.0, 3.0] for ind, name in enumerate(CH_NAMES)],
            )
            with open(
                os.path.join(folder, "sub-01_coordsystem.json"), "w", encoding="utf-8"
            ) as file:
                json.dump({"iEEGCoordinateSystem": "ACPC", "iEEGCoordinateUnits": "cm"}, file)
    return root


###################################################################################################
## MEF 3.0 writer (synthetic sessions for the MEF reader tests)
##
## Written from the MEF 3.0 specification and the reference C library (meflib), independently
## of evoparam.io.mef: byte-by-byte CRC, the range encoder of RED_encode_exec, and the file
## layouts (universal header, .tmet, .tidx, .tdat).

MEF_NAN = -(1 << 31)
MEF_NEG_INF = -(1 << 31) + 1
MEF_POS_INF = (1 << 31) - 1
_MEF_UNITS = {"microvolts": 1.0, "uV": 1.0, "millivolts": 1e3, "mV": 1e3, "volts": 1e6}


def _mef_crc_table():
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xEB31D82E if crc & 1 else crc >> 1
        table.append(crc)
    return table


_MEF_CRC_TABLE = _mef_crc_table()


def mef_crc(data):
    """CRC of meflib (Koopman polynomial, start 0xFFFFFFFF), byte by byte."""

    crc = 0xFFFFFFFF
    for byte in bytes(data):
        crc = (crc >> 8) ^ _MEF_CRC_TABLE[(crc ^ byte) & 0xFF]
    return crc


def mef_round(values):
    """meflib RED_round: half away from zero, clipped to the infinity codes."""

    values = np.asarray(values, dtype=float)
    out = np.where(values >= 0, values + 0.5, values - 0.5)
    return np.trunc(np.clip(out, MEF_NEG_INF, MEF_POS_INF)).astype(np.int64)


def red_encode(
    samples, start_time, *, discontinuity=False, scale_factor=1.0, detrend=False, encrypted=False
):
    """One RED block (bytes, 8-byte aligned) of integer samples, as meflib's RED_encode_exec."""

    import struct

    x = np.asarray(samples, dtype=np.int64)
    n = x.size
    slope = intercept = 0.0
    if detrend:
        c = np.arange(1, n + 1, dtype=float)
        sx, sxx = n * (n + 1) / 2.0, n * (n + 1) * (2 * n + 1) / 6.0
        sy, sxy = float(np.sum(x)), float(np.sum(x * c))
        mx, my = sx / n, sy / n
        slope = float(np.float32(((sx * my) - sxy) / ((sx * mx) - sxx)))
        intercept = float(np.float32(my - slope * mx))
        x = mef_round(x - slope * c - intercept)
    if scale_factor > 1.0:
        x = mef_round(x / scale_factor)

    # Differences: the first sample as 4 bytes, then a byte per difference, or the
    # keysample flag -128 and 4 bytes when the difference exceeds +-127
    diffs = bytearray(struct.pack("<i", int(x[0])))
    for prev, cur in zip(x[:-1].tolist(), x[1:].tolist()):
        step = cur - prev
        if -127 <= step <= 127:
            diffs.append(step & 0xFF)
        else:
            diffs.append(0x80)
            diffs += struct.pack("<i", int(cur))

    # Statistics, scaled to fit in one byte
    counts = np.bincount(np.frombuffer(bytes(diffs), dtype=np.uint8), minlength=256)
    if counts.max() > 255:
        scale = 254.999999999 / counts.max()
        stats = [int(np.ceil(count * scale)) if count else 0 for count in counts.tolist()]
    else:
        stats = counts.tolist()
    cum = [0]
    for count in stats:
        cum.append(cum[-1] + count)
    total = cum[256]

    # Range encoder (meflib constants)
    top, bottom, carry, mask = 0x80000000, 0x800000, 0x7F800000, 0xFFFFFFFF
    low, rng, out_byte, underflow, out = 0, top, 0, 0, []

    def normalise(low, rng, out_byte, underflow):
        while rng <= bottom:
            if low < carry:
                out.append(out_byte)
                out.extend([0xFF] * underflow)
                underflow = 0
                out_byte = (low >> 23) & 0xFF
            elif low & top:
                out.append((out_byte + 1) & 0xFF)
                out.extend([0] * underflow)
                underflow = 0
                out_byte = (low >> 23) & 0xFF
            else:
                underflow += 1
            rng = (rng << 8) & mask
            low = (low << 8) & 0x7FFFFFFF
        return low, rng, out_byte, underflow

    for symbol in diffs:
        low, rng, out_byte, underflow = normalise(low, rng, out_byte, underflow)
        per = rng // total
        below = per * cum[symbol]
        low = (low + below) & mask
        rng = per * stats[symbol] if symbol < 255 else rng - below
    low, rng, out_byte, underflow = normalise(low, rng, out_byte, underflow)
    last = (low >> 23) + 1
    if last > 0xFF:
        out.append((out_byte + 1) & 0xFF)
        out.extend([0] * underflow)
    else:
        out.append(out_byte)
        out.extend([0xFF] * underflow)
    out += [last & 0xFF, 0]
    compressed = bytes(out[1:])  # the first output byte is junk

    flags = (1 if discontinuity else 0) | (2 if encrypted else 0)
    size = 304 + len(compressed)
    size += (8 - size % 8) % 8
    head = struct.pack(
        "<IB3x8xfffIIIq",
        0,
        flags,
        slope,
        intercept,
        float(scale_factor),
        len(diffs) + 1,
        n,
        size,
        int(start_time),
    ) + bytes(stats)
    block = bytearray(head + compressed + b"~" * (size - 304 - len(compressed)))
    block[0:4] = struct.pack("<I", mef_crc(block[4:]))
    return bytes(block)


def _universal_header(kind, body, *, start, end, n_entries, max_entry, segment, channel, session):
    import struct

    head = bytearray(1024)
    head[8:13] = kind.encode("ascii") + b"\0"
    head[13], head[14], head[15] = 3, 0, 1
    struct.pack_into("<qqqqi", head, 16, start, end, n_entries, max_entry, segment)
    for offset, text in ((52, channel), (308, session)):
        raw = text.encode("utf-8")[:255]
        head[offset : offset + len(raw)] = raw
    head[836:852] = os.urandom(16)  # file UUID
    struct.pack_into("<I", head, 4, mef_crc(body))
    struct.pack_into("<I", head, 0, mef_crc(head[4:]))
    return bytes(head) + bytes(body)


def write_mef(
    folder,
    session,
    data,
    sfreq,
    ch_names,
    *,
    factor=0.1,
    units="microvolts",
    block_samples=500,
    start_time=1_600_000_000_000_000,
    segment_starts=(),
    gaps=None,
    flag_gaps=True,
    scale_factor=1.0,
    detrend=False,
    encrypt=None,
):
    """Write a MEF 3.0 session ``<folder>/<session>.mefd``; return its path.

    Parameters
    ----------
    data : 2d array
        Values in µV, [n_channels, n_times]; NaN and +-inf are stored as the RED codes.
    factor, units : float, str
        Units conversion factor and description written in the metadata: the stored
        integer is ``round(value in units / factor)``.
    block_samples : int
        Samples per block (blocks also end at segment starts and gaps).
    segment_starts : sequence of int
        Samples where a new segment starts.
    gaps : dict, optional
        ``{sample: seconds}``: a recording gap of that length before the sample (the
        block that starts there is flagged as a discontinuity when ``flag_gaps``).
    scale_factor, detrend
        Lossy RED options of meflib, applied to every block.
    encrypt : {"metadata", "data"}, optional
        Mark the metadata section 2, or the blocks, as encrypted (level 1).
    """

    import struct

    data = np.atleast_2d(np.asarray(data, dtype=float))
    n_channels, n_times = data.shape
    gaps = dict(gaps or {})
    bounds = sorted(
        {0, n_times} | set(segment_starts) | set(gaps) | set(range(0, n_times, block_samples))
    )
    seg_bounds = sorted({0, n_times} | set(segment_starts))
    offsets_us = np.cumsum([gaps.get(s, 0.0) for s in range(n_times)]) * 1e6
    times = start_time + np.round(np.arange(n_times) * 1e6 / sfreq + offsets_us).astype(np.int64)
    root = os.path.join(str(folder), session + ".mefd")
    for ch, name in enumerate(ch_names):
        stored = data[ch] / _MEF_UNITS.get(units, 1.0) / factor
        values = np.where(np.isfinite(stored), mef_round(np.nan_to_num(stored)), 0)
        values = np.where(np.isnan(data[ch]), MEF_NAN, values)
        values = np.where(np.isposinf(data[ch]), MEF_POS_INF, values)
        values = np.where(np.isneginf(data[ch]), MEF_NEG_INF, values)
        for seg, (seg_first, seg_stop) in enumerate(zip(seg_bounds[:-1], seg_bounds[1:])):
            folder_seg = os.path.join(root, name + ".timd", "{}-{:06d}.segd".format(name, seg))
            os.makedirs(folder_seg, exist_ok=True)
            stem = os.path.join(folder_seg, "{}-{:06d}".format(name, seg))
            starts = [b for b in bounds if seg_first <= b < seg_stop]
            tdat, index = bytearray(), bytearray()
            for first, stop in zip(starts, starts[1:] + [seg_stop]):
                block = red_encode(
                    values[first:stop],
                    times[first],
                    discontinuity=first == 0 or (flag_gaps and first in gaps),
                    scale_factor=scale_factor,
                    detrend=detrend,
                    encrypted=encrypt == "data",
                )
                part = values[first:stop]
                index += struct.pack(
                    "<qqqIIii4xB3x8x",
                    1024 + len(tdat),
                    times[first],
                    first - seg_first,
                    stop - first,
                    len(block),
                    int(part.max()),
                    int(part.min()),
                    block[4],
                )
                tdat += block
            end = int(times[seg_stop - 1] + 1e6 / sfreq)
            common = dict(
                start=int(times[seg_first]), end=end, segment=seg, channel=name, session=session
            )
            n_blocks = len(starts)
            with open(stem + ".tdat", "wb") as file:
                file.write(
                    _universal_header(
                        "tdat", tdat, n_entries=n_blocks, max_entry=block_samples, **common
                    )
                )
            with open(stem + ".tidx", "wb") as file:
                file.write(
                    _universal_header("tidx", index, n_entries=n_blocks, max_entry=56, **common)
                )
            meta = bytearray(16384 - 1024)
            meta[0] = 1 if encrypt == "metadata" else 0  # section 2 encryption
            meta[1] = 0  # section 3 encryption
            body = 8720 - 1024
            struct.pack_into("<dddddd", meta, body, sfreq, 0.0, -1.0, -1.0, 60.0, factor)
            meta[8768 - 1024 : 8768 - 1024 + len(units)] = units.encode("utf-8")
            struct.pack_into("<qqq", meta, 8912 - 1024, seg_first, seg_stop - seg_first, n_blocks)
            with open(stem + ".tmet", "wb") as file:
                file.write(_universal_header("tmet", meta, n_entries=1, max_entry=16384, **common))
    return root


def corrupt_byte(path, offset):
    """Flip the bits of one byte of a file."""

    with open(path, "r+b") as file:
        file.seek(offset)
        value = file.read(1)[0]
        file.seek(offset)
        file.write(bytes([value ^ 0xFF]))


def make_mef_bids(root):
    """The mini BIDS dataset of :func:`make_bids`, with MEF 3.0 recordings instead of
    BrainVision (same signal, events and sidecars)."""

    root = make_bids(root)
    n_samples = int(DURATION * SFREQ)
    for vhdr in sorted(glob_files(root, ".vhdr")):
        folder, name = os.path.split(vhdr)
        stem = name[: -len(".vhdr")]
        write_mef(
            folder,
            stem,
            signal(len(CH_NAMES), n_samples),
            SFREQ,
            CH_NAMES,
            factor=0.5,
            block_samples=1000,
        )
        for ext in (".vhdr", ".eeg"):
            os.remove(os.path.join(folder, stem + ext))
    return root


def glob_files(root, suffix):
    out = []
    for folder, _, names in os.walk(str(root)):
        out += [os.path.join(folder, name) for name in names if name.endswith(suffix)]
    return out


###################################################################################################
## Local HTTP server


class _Stats:
    def __init__(self):
        self.requests = []  # (method, path, range header or None, status)
        self.bytes_sent = 0
        self.failures_left = 0


def _handler(directory, stats, honour_range):

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _path(self):
            rel = unquote(self.path.split("?")[0].lstrip("/"))
            return os.path.join(directory, *rel.split("/"))

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
                stats.requests.append(("HEAD", self.path, None, 404))
                self.send_error(404)
                return
            stats.requests.append(("HEAD", self.path, None, 200))
            self.send_response(200)
            self.send_header("Content-Length", str(os.path.getsize(path)))
            self.end_headers()

        def _listing(self):
            """S3 ListObjectsV2 answer for ``?list-type=2&prefix=...`` (one page)."""

            query = parse_qs(urlsplit(self.path).query)
            prefix = query.get("prefix", [""])[0]
            keys = []
            for folder, _, names in os.walk(directory):
                for name in names:
                    rel = os.path.relpath(os.path.join(folder, name), directory)
                    rel = rel.replace(os.sep, "/")
                    if rel.startswith(prefix):
                        keys.append((rel, os.path.getsize(os.path.join(folder, name))))
            items = "".join(
                "<Contents><Key>{}</Key><Size>{}</Size></Contents>".format(escape(key), size)
                for key, size in sorted(keys)
            )
            body = (
                (
                    '<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns='
                    '"http://s3.amazonaws.com/doc/2006-03-01/"><Prefix>{}</Prefix><KeyCount>{}'
                    "</KeyCount><IsTruncated>false</IsTruncated>{}</ListBucketResult>"
                )
                .format(escape(prefix), len(keys), items)
                .encode("utf-8")
            )
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            stats.bytes_sent += len(body)
            stats.requests.append(("GET", self.path, None, 200))

        def do_GET(self):
            if self._fail():
                return
            if "list-type=2" in self.path.partition("?")[2]:
                self._listing()
                return
            path = self._path()
            byte_range = self.headers.get("Range")
            if not os.path.isfile(path):
                stats.requests.append(("GET", self.path, byte_range, 404))
                self.send_error(404)
                return
            size = os.path.getsize(path)
            match = re.match(r"bytes=(\d+)-(\d*)$", byte_range or "")
            if honour_range and match:
                start = int(match.group(1))
                stop = min(int(match.group(2)) if match.group(2) else size - 1, size - 1)
                if start >= size:
                    stats.requests.append(("GET", self.path, byte_range, 416))
                    self.send_error(416)
                    return
                with open(path, "rb") as file:
                    file.seek(start)
                    body = file.read(stop - start + 1)
                self.send_response(206)
                self.send_header("Content-Range", "bytes {}-{}/{}".format(start, stop, size))
                status = 206
            else:
                with open(path, "rb") as file:
                    body = file.read()
                self.send_response(200)
                status = 200
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            stats.bytes_sent += len(body)
            stats.requests.append(("GET", self.path, byte_range, status))

    return Handler


@contextlib.contextmanager
def http_server(directory, honour_range=True, failures=0):
    """Serve ``directory`` on 127.0.0.1; yield (base URL, stats).

    With ``honour_range=False`` the server ignores Range headers and answers 200 with the
    whole file. ``failures`` first requests are answered 503.
    """

    stats = _Stats()
    stats.failures_left = failures
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), _handler(str(directory), stats, honour_range)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:{}/".format(server.server_address[1]), stats
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
