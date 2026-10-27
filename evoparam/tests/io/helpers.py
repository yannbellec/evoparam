# SPDX-License-Identifier: Apache-2.0
"""Writers of small BrainVision files, MEF 3.0 sessions and BIDS datasets, and a local HTTP
server (io tests)."""

import contextlib
import http.server
import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Any
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


def signal(n_channels: Any, n_samples: Any) -> Any:
    """Known samples (µV): channel c at sample n is c * 1e5 + n."""

    return np.arange(n_channels)[:, None] * 1e5 + np.arange(n_samples)[None, :]


@dataclass(frozen=True)
class BrainVisionLayout:
    """The keywords of :func:`write_brainvision`: format, orientation, resolutions, units."""

    fmt: Any = "IEEE_FLOAT_32"
    orientation: Any = "MULTIPLEXED"
    resolutions: Any = None
    units: Any = None


def write_brainvision(
    folder: Any, stem: Any, data: Any, sfreq: Any, ch_names: Any, **keywords: Any
) -> Any:
    """Write a BrainVision recording (data in µV, [n_channels, n_times]); return the .vhdr path.

    Keywords: the fields of :class:`BrainVisionLayout`.
    """

    layout = BrainVisionLayout(**keywords)
    resolutions = layout.resolutions
    units = layout.units
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
        f"DataFile={stem}.eeg",
        f"MarkerFile={stem}.vmrk",
        "DataFormat=BINARY",
        f"DataOrientation={layout.orientation}",
        f"NumberOfChannels={n_channels}",
        f"SamplingInterval={1e6 / sfreq}",
        "",
        "[Binary Infos]",
        f"BinaryFormat={layout.fmt}",
        "",
        "[Channel Infos]",
    ]
    lines += [
        "Ch{}={},,{},{}".format(ind + 1, name.replace(",", "\\1"), res, unit)
        for ind, (name, res, unit) in enumerate(zip(ch_names, resolutions, units, strict=False))
    ]
    with open(os.path.join(folder, stem + ".vhdr"), "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    # Stored value = value in the channel unit / resolution
    to_unit = np.array([{"µV": 1.0, "mV": 1e-3, "V": 1e-6}.get(unit, 1.0) for unit in units])
    stored = np.asarray(data, dtype=float) * to_unit[:, None] / np.array(resolutions)[:, None]
    dtype = {"IEEE_FLOAT_32": "<f4", "INT_16": "<i2", "INT_32": "<i4"}[layout.fmt]
    if dtype != "<f4":
        stored = np.round(stored)
    stored = stored.astype(dtype)
    stored = stored.T if layout.orientation == "MULTIPLEXED" else stored
    stored.tofile(os.path.join(folder, stem + ".eeg"))

    return os.path.join(folder, stem + ".vhdr")


def _tsv(path: Any, header: Any, rows: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write("\t".join(header) + "\n")
        for row in rows:
            file.write("\t".join(str(value) for value in row) + "\n")


def make_bids(root: Any) -> Any:
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
## MEF 3.0 writer, written from the MEF 3.0 spec and meflib, independently of evoparam.io.mef

MEF_NAN = -(1 << 31)
MEF_NEG_INF = -(1 << 31) + 1
MEF_POS_INF = (1 << 31) - 1
_MEF_UNITS = {"microvolts": 1.0, "uV": 1.0, "millivolts": 1e3, "mV": 1e3, "volts": 1e6}
# Largest difference stored in one byte by RED, and the largest byte value.
_RED_MAX_STEP = 127
_BYTE_MAX = 0xFF


def _mef_crc_table() -> Any:
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xEB31D82E if crc & 1 else crc >> 1
        table.append(crc)
    return table


_MEF_CRC_TABLE = _mef_crc_table()


def mef_crc(data: Any) -> Any:
    """CRC of meflib (Koopman polynomial, start 0xFFFFFFFF), byte by byte."""

    crc = 0xFFFFFFFF
    for byte in bytes(data):
        crc = (crc >> 8) ^ _MEF_CRC_TABLE[(crc ^ byte) & 0xFF]
    return crc


def mef_round(values: Any) -> Any:
    """meflib RED_round: half away from zero, clipped to the infinity codes."""

    values = np.asarray(values, dtype=float)
    out = np.where(values >= 0, values + 0.5, values - 0.5)
    return np.trunc(np.clip(out, MEF_NEG_INF, MEF_POS_INF)).astype(np.int64)


def _red_detrend(x: Any) -> tuple[Any, float, float]:
    """meflib's RED detrending: the samples less their float32 least-squares line, rounded,
    with the slope and intercept of the line."""

    n = x.size
    c = np.arange(1, n + 1, dtype=float)
    sx, sxx = n * (n + 1) / 2.0, n * (n + 1) * (2 * n + 1) / 6.0
    sy, sxy = float(np.sum(x)), float(np.sum(x * c))
    mx, my = sx / n, sy / n
    slope = float(np.float32(((sx * my) - sxy) / ((sx * mx) - sxx)))
    intercept = float(np.float32(my - slope * mx))
    return mef_round(x - slope * c - intercept), slope, intercept


def _red_differences(x: Any) -> bytearray:
    """The first sample as 4 bytes, then a byte per difference, or the keysample flag -128 and
    4 bytes when the difference exceeds +-127."""

    import struct

    diffs = bytearray(struct.pack("<i", int(x[0])))
    for prev, cur in zip(x[:-1].tolist(), x[1:].tolist(), strict=False):
        step = cur - prev
        if -_RED_MAX_STEP <= step <= _RED_MAX_STEP:
            diffs.append(step & 0xFF)
        else:
            diffs.append(0x80)
            diffs += struct.pack("<i", int(cur))
    return diffs


def _red_statistics(diffs: Any) -> list[int]:
    """The count of each byte value of ``diffs``, scaled to fit in one byte."""

    counts = np.bincount(np.frombuffer(bytes(diffs), dtype=np.uint8), minlength=256)
    if counts.max() > _BYTE_MAX:
        scale = 254.999999999 / counts.max()
        return [int(np.ceil(count * scale)) if count else 0 for count in counts.tolist()]
    return counts.tolist()


def _range_encode(diffs: Any, stats: Any) -> bytes:
    """meflib's range encoder of ``diffs`` with the byte statistics ``stats``."""

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
        rng = per * stats[symbol] if symbol < _BYTE_MAX else rng - below
    low, rng, out_byte, underflow = normalise(low, rng, out_byte, underflow)
    last = (low >> 23) + 1
    if last > _BYTE_MAX:
        out.append((out_byte + 1) & 0xFF)
        out.extend([0] * underflow)
    else:
        out.append(out_byte)
        out.extend([0xFF] * underflow)
    out += [last & 0xFF, 0]
    return bytes(out[1:])  # the first output byte is junk


def red_encode(
    samples: Any,
    start_time: Any,
    *,
    discontinuity: Any = False,
    scale_factor: Any = 1.0,
    detrend: Any = False,
    encrypted: Any = False,
) -> Any:
    """One RED block (bytes, 8-byte aligned) of integer samples, as meflib's RED_encode_exec."""

    import struct

    x = np.asarray(samples, dtype=np.int64)
    n = x.size
    slope = intercept = 0.0
    if detrend:
        x, slope, intercept = _red_detrend(x)
    if scale_factor > 1.0:
        x = mef_round(x / scale_factor)
    diffs = _red_differences(x)
    stats = _red_statistics(diffs)
    compressed = _range_encode(diffs, stats)

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


def _universal_header(
    kind: Any, body: Any, segment: Any, *, n_entries: Any, max_entry: Any
) -> Any:
    """The universal header of a segment file, followed by ``body``.

    ``segment`` holds what the files of one segment share: its ``start`` and ``end`` times,
    its number ``segment``, and the ``channel`` and ``session`` names.
    """
    import struct

    head = bytearray(1024)
    head[8:13] = kind.encode("ascii") + b"\0"
    head[13], head[14], head[15] = 3, 0, 1
    struct.pack_into(
        "<qqqqi",
        head,
        16,
        segment["start"],
        segment["end"],
        n_entries,
        max_entry,
        segment["segment"],
    )
    for offset, text in ((52, segment["channel"]), (308, segment["session"])):
        raw = text.encode("utf-8")[:255]
        head[offset : offset + len(raw)] = raw
    head[836:852] = os.urandom(16)  # file UUID
    struct.pack_into("<I", head, 4, mef_crc(body))
    struct.pack_into("<I", head, 0, mef_crc(head[4:]))
    return bytes(head) + bytes(body)


@dataclass(frozen=True)
class MefOptions:
    """The keywords of :func:`write_mef`: stored values, layout, gaps and RED options."""

    factor: Any = 0.1
    units: Any = "microvolts"
    block_samples: Any = 500
    start_time: Any = 1_600_000_000_000_000
    segment_starts: Any = ()
    gaps: Any = None
    flag_gaps: Any = True
    scale_factor: Any = 1.0
    detrend: Any = False
    encrypt: Any = None


def write_mef(
    folder: Any, session: Any, data: Any, sfreq: Any, ch_names: Any, **keywords: Any
) -> Any:
    """Write a MEF 3.0 session ``<folder>/<session>.mefd``; return its path.

    The keywords (``factor`` to ``encrypt``) are the fields of :class:`MefOptions`.

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

    options = MefOptions(**keywords)
    gaps = options.gaps
    import struct

    data = np.atleast_2d(np.asarray(data, dtype=float))
    n_channels, n_times = data.shape
    gaps = dict(gaps or {})
    bounds = sorted(
        {0, n_times}
        | set(options.segment_starts)
        | set(gaps)
        | set(range(0, n_times, options.block_samples))
    )
    seg_bounds = sorted({0, n_times} | set(options.segment_starts))
    offsets_us = np.cumsum([gaps.get(s, 0.0) for s in range(n_times)]) * 1e6
    times = options.start_time + np.round(np.arange(n_times) * 1e6 / sfreq + offsets_us).astype(
        np.int64
    )
    root = os.path.join(str(folder), session + ".mefd")
    for ch, name in enumerate(ch_names):
        stored = data[ch] / _MEF_UNITS.get(options.units, 1.0) / options.factor
        values = np.where(np.isfinite(stored), mef_round(np.nan_to_num(stored)), 0)
        values = np.where(np.isnan(data[ch]), MEF_NAN, values)
        values = np.where(np.isposinf(data[ch]), MEF_POS_INF, values)
        values = np.where(np.isneginf(data[ch]), MEF_NEG_INF, values)
        for seg, (seg_first, seg_stop) in enumerate(
            zip(seg_bounds[:-1], seg_bounds[1:], strict=False)
        ):
            folder_seg = os.path.join(root, name + ".timd", f"{name}-{seg:06d}.segd")
            os.makedirs(folder_seg, exist_ok=True)
            stem = os.path.join(folder_seg, f"{name}-{seg:06d}")
            starts = [b for b in bounds if seg_first <= b < seg_stop]
            tdat, index = bytearray(), bytearray()
            for first, stop in zip(starts, starts[1:] + [seg_stop], strict=False):
                block = red_encode(
                    values[first:stop],
                    times[first],
                    discontinuity=first == 0 or (options.flag_gaps and first in gaps),
                    scale_factor=options.scale_factor,
                    detrend=options.detrend,
                    encrypted=options.encrypt == "data",
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
            common = {
                "start": int(times[seg_first]),
                "end": end,
                "segment": seg,
                "channel": name,
                "session": session,
            }
            n_blocks = len(starts)
            with open(stem + ".tdat", "wb") as file:
                file.write(
                    _universal_header(
                        "tdat", tdat, common, n_entries=n_blocks, max_entry=options.block_samples
                    )
                )
            with open(stem + ".tidx", "wb") as file:
                file.write(
                    _universal_header("tidx", index, common, n_entries=n_blocks, max_entry=56)
                )
            meta = bytearray(16384 - 1024)
            meta[0] = 1 if options.encrypt == "metadata" else 0  # section 2 encryption
            meta[1] = 0  # section 3 encryption
            body = 8720 - 1024
            struct.pack_into("<dddddd", meta, body, sfreq, 0.0, -1.0, -1.0, 60.0, options.factor)
            meta[8768 - 1024 : 8768 - 1024 + len(options.units)] = options.units.encode("utf-8")
            struct.pack_into("<qqq", meta, 8912 - 1024, seg_first, seg_stop - seg_first, n_blocks)
            with open(stem + ".tmet", "wb") as file:
                file.write(_universal_header("tmet", meta, common, n_entries=1, max_entry=16384))
    return root


def corrupt_byte(path: Any, offset: Any) -> None:
    """Flip the bits of one byte of a file."""

    with open(path, "r+b") as file:
        file.seek(offset)
        value = file.read(1)[0]
        file.seek(offset)
        file.write(bytes([value ^ 0xFF]))


def make_mef_bids(root: Any) -> Any:
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


def glob_files(root: Any, suffix: Any) -> Any:
    out = []
    for folder, _, names in os.walk(str(root)):
        out += [os.path.join(folder, name) for name in names if name.endswith(suffix)]
    return out


###################################################################################################
## Local HTTP server


class _Stats:
    def __init__(self) -> None:
        self.requests = []  # (method, path, range header or None, status)
        self.bytes_sent = 0
        self.failures_left = 0


class _Handler(http.server.BaseHTTPRequestHandler):
    """Answers of the test server; :func:`_handler` sets the served ``root``, the ``stats``
    and ``honour_range``."""

    root: Any = None
    stats: Any = None
    honour_range: Any = True

    def log_message(self, *args):
        pass

    def _path(self):
        rel = unquote(self.path.split("?")[0].lstrip("/"))
        return os.path.join(self.root, *rel.split("/"))

    def _fail(self):
        if self.stats.failures_left > 0:
            self.stats.failures_left -= 1
            self.stats.requests.append((self.command, self.path, None, 503))
            self.send_error(503)
            return True
        return False

    def do_HEAD(self):
        if self._fail():
            return
        path = self._path()
        if not os.path.isfile(path):
            self.stats.requests.append(("HEAD", self.path, None, 404))
            self.send_error(404)
            return
        self.stats.requests.append(("HEAD", self.path, None, 200))
        self.send_response(200)
        self.send_header("Content-Length", str(os.path.getsize(path)))
        self.end_headers()

    def _listing(self):
        """S3 ListObjectsV2 answer for ``?list-type=2&prefix=...`` (one page)."""

        query = parse_qs(urlsplit(self.path).query)
        prefix = query.get("prefix", [""])[0]
        keys = []
        for folder, _, names in os.walk(self.root):
            for name in names:
                rel = os.path.relpath(os.path.join(folder, name), self.root)
                rel = rel.replace(os.sep, "/")
                if rel.startswith(prefix):
                    keys.append((rel, os.path.getsize(os.path.join(folder, name))))
        items = "".join(
            f"<Contents><Key>{escape(key)}</Key><Size>{size}</Size></Contents>"
            for key, size in sorted(keys)
        )
        body = (
            '<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns='
            f'"http://s3.amazonaws.com/doc/2006-03-01/"><Prefix>{escape(prefix)}</Prefix><KeyCount>{len(keys)}'
            f"</KeyCount><IsTruncated>false</IsTruncated>{items}</ListBucketResult>"
        ).encode()
        # recorded before the body is sent: the client may go on as soon as it has the body
        self.stats.bytes_sent += len(body)
        self.stats.requests.append(("GET", self.path, None, 200))
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self, path, byte_range):
        """The bytes of ``path`` to send, the status and the Content-Range header.

        The whole file (200) unless the server honours a valid Range header (206); no bytes
        (416) when the range starts after the end of the file.
        """

        size = os.path.getsize(path)
        match = re.match(r"bytes=(\d+)-(\d*)$", byte_range or "")
        if not (self.honour_range and match):
            with open(path, "rb") as file:
                return file.read(), 200, None
        start = int(match.group(1))
        stop = min(int(match.group(2)) if match.group(2) else size - 1, size - 1)
        if start >= size:
            return None, 416, None
        with open(path, "rb") as file:
            file.seek(start)
            return file.read(stop - start + 1), 206, f"bytes {start}-{stop}/{size}"

    def do_GET(self):
        if self._fail():
            return
        if "list-type=2" in self.path.partition("?")[2]:
            self._listing()
            return
        path = self._path()
        byte_range = self.headers.get("Range")
        if not os.path.isfile(path):
            self.stats.requests.append(("GET", self.path, byte_range, 404))
            self.send_error(404)
            return
        body, status, content_range = self._read(path, byte_range)
        if body is None:
            self.stats.requests.append(("GET", self.path, byte_range, status))
            self.send_error(status)
            return
        # recorded before the body is sent: once the client has read the body it goes on,
        # and a record made after the write could land among its next requests
        self.stats.bytes_sent += len(body)
        self.stats.requests.append(("GET", self.path, byte_range, status))
        self.send_response(status)
        if content_range is not None:
            self.send_header("Content-Range", content_range)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _handler(directory: Any, stats: Any, honour_range: Any) -> Any:
    """A request handler class serving ``directory`` and recording into ``stats``."""

    return type(
        "Handler", (_Handler,), {"root": directory, "stats": stats, "honour_range": honour_range}
    )


@contextlib.contextmanager
def http_server(directory: Any, honour_range: Any = True, failures: Any = 0) -> Any:
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
        yield f"http://127.0.0.1:{server.server_address[1]}/", stats
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
