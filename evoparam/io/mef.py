# SPDX-License-Identifier: Apache-2.0
"""Read MEF 3.0 sessions (Multiscale Electrophysiology Format), locally or remotely.

MEF 3.0 is the format of the Mayo Clinic iEEG recordings on OpenNeuro (ds003708,
ds004457, ds004696, ds004977: ``*_ieeg.mefd`` folders). A session folder (``.mefd``)
holds one folder per time-series channel (``<name>.timd``), split in segments
(``<name>-000000.segd``) of three files, each starting with a 1024-byte universal
header:

- ``.tmet``, the metadata: sampling frequency, units conversion factor and units,
  number of samples and blocks, encryption of the metadata sections;
- ``.tidx``, the index: one 56-byte entry per block (file offset, start time in µs,
  first sample, number of samples, size, extrema, flags);
- ``.tdat``, the data: blocks of samples compressed by range-encoded differences
  (RED), each with a 304-byte header (CRC, flags, scale factor, start time, symbol
  statistics).

The reader is written from the public format specification and the reference C
library of the Mayo Systems Electrophysiology Lab (meflib); it needs numpy only.
Only the blocks that cover the requested samples are read and decoded; for a remote
session only their bytes are downloaded, by HTTP range requests, with the metadata
and index files of the channels read. The blocks of all the channels are decoded
together, one symbol per step for every block (numpy). Block and header CRCs are
checked (``check_crc``). Lossy blocks (scale factor, detrending) are restored as
meflib does.

Time axis: sample 0 is the earliest block start time of the channels read. Within a
run of contiguous blocks the samples follow their sample numbers, whatever the jitter of
the block time stamps (the onsets of the BIDS events.tsv of these datasets are sample
numbers divided by the sampling frequency); a block flagged as a discontinuity, or whose
start time jumps forward by more than 1 s, starts a new run, placed after the previous
one by the time gap between them. Samples that no block covers (gaps) are NaN, and so
are samples stored with the reserved value ``RED_NAN``;
``RED_NEGATIVE_INFINITY`` and ``RED_POSITIVE_INFINITY`` become -inf and +inf.
Values are converted to µV with the units conversion factor and the units
description of each channel.

Not supported: encrypted sessions (metadata section 2 or data blocks encrypted with a
password raise :class:`~evoparam.errors.ReaderError`; an encrypted section 3, which
holds only the subject identity and the recording time offset, is ignored), records
(``.rdat`` / ``.ridx``: annotations), video channels and big-endian files.
"""

import concurrent.futures
import hashlib
import os
import re
import struct
from urllib.parse import quote, unquote
from xml.etree import ElementTree

import numpy as np

from evoparam.errors import DataError, ReaderError
from evoparam.errors import warn as _warn
from evoparam.io._fetch import RemoteFile, TransferLog, fetch_bytes, is_url
from evoparam.io.brainvision import _pick, _sample

__all__ = ["read_mef", "read_mef_header"]

###################################################################################################
###################################################################################################

UNIVERSAL_HEADER_BYTES = 1024
METADATA_BYTES = 16384
BLOCK_HEADER_BYTES = 304

RED_NAN = -(1 << 31)
RED_NEGATIVE_INFINITY = -(1 << 31) + 1
RED_POSITIVE_INFINITY = (1 << 31) - 1
UUTC_NO_ENTRY = -(1 << 63)

DISCONTINUITY = 1  # bit 0 of the block flags
ENCRYPTION = 2 | 4  # bits 1 and 2: level 1 or level 2 encryption

# One entry of the .tidx file (56 bytes, little endian)
INDEX_DTYPE = np.dtype(
    [
        ("file_offset", "<i8"),
        ("start_time", "<i8"),
        ("start_sample", "<i8"),
        ("n_samples", "<u4"),
        ("block_bytes", "<u4"),
        ("maximum", "<i4"),
        ("minimum", "<i4"),
        ("protected", "V4"),
        ("flags", "u1"),
        ("red_protected", "V3"),
        ("discretionary", "V8"),
    ]
)

# Header of a RED block (304 bytes)
BLOCK_DTYPE = np.dtype(
    [
        ("crc", "<u4"),
        ("flags", "u1"),
        ("protected", "V3"),
        ("discretionary", "V8"),
        ("detrend_slope", "<f4"),
        ("detrend_intercept", "<f4"),
        ("scale_factor", "<f4"),
        ("difference_bytes", "<u4"),
        ("n_samples", "<u4"),
        ("block_bytes", "<u4"),
        ("start_time", "<i8"),
        ("statistics", "u1", (256,)),
    ]
)

# Units description -> factor to µV. Descriptions are free text: "microvolts" (Mayo), "uV", ...
_TO_UV = {
    "µv": 1.0,
    "μv": 1.0,
    "uv": 1.0,
    "microvolt": 1.0,
    "microvolts": 1.0,
    "mv": 1e3,
    "millivolt": 1e3,
    "millivolts": 1e3,
    "v": 1e6,
    "volt": 1e6,
    "volts": 1e6,
    "nv": 1e-3,
    "nanovolt": 1e-3,
    "nanovolts": 1e-3,
}

# Remote reads: bytes per cached chunk of a .tdat file (the bytes read beyond the blocks
# are less than two chunks per file), and parallel HTTP requests.
_CHUNK = 1 << 12
_WORKERS = 16


###################################################################################################
## CRC (Koopman 32-bit polynomial, as meflib: initial value 0xFFFFFFFF, no final XOR)

_KOOPMAN32 = 0xEB31D82E
_CRC_SPAN = 256
_CRC = {}


def _crc_tables():
    """Byte table, tables of a byte followed by k zero bytes, and the operator that
    appends ``_CRC_SPAN`` zero bytes to a CRC state (one table per state byte).
    """
    if not _CRC:
        table = np.arange(256, dtype=np.uint64)
        for _ in range(8):
            table = np.where(table & 1, (table >> 1) ^ _KOOPMAN32, table >> 1)
        table = table.astype(np.uint32)

        def zero_byte(state):
            return (state >> np.uint32(8)) ^ table[state & np.uint32(0xFF)]

        dist = np.empty((_CRC_SPAN, 256), dtype=np.uint32)
        dist[0] = table
        for k in range(1, _CRC_SPAN):
            dist[k] = zero_byte(dist[k - 1])
        span = np.empty((4, 256), dtype=np.uint32)
        for j in range(4):
            state = np.arange(256, dtype=np.uint32) << np.uint32(8 * j)
            for _ in range(_CRC_SPAN):
                state = zero_byte(state)
            span[j] = state
        _CRC.update(table=table, dist=dist, span=span)
    return _CRC


def crc32_koopman(buf, starts, lengths):
    """CRCs of several byte ranges of a buffer, computed together (numpy).

    The CRC is linear: the contribution of each byte depends only on its value and on
    the number of bytes after it. Ranges are cut in spans of 256 bytes aligned on their
    end; the spans are reduced by XOR and chained, and the initial value 0xFFFFFFFF is
    applied by inverting the first four bytes (reflected CRC).

    Parameters
    ----------
    buf : 1d array of uint8
    starts, lengths : 1d arrays of int
        Ranges ``buf[start:start + length]``, each of at least 4 bytes.

    Returns
    -------
    crc : 1d array of uint32
    """
    tables = _crc_tables()
    starts = np.asarray(starts, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    if starts.size == 0:
        return np.zeros(0, dtype=np.uint32)
    if lengths.min() < 4:
        raise ReaderError("MEF: CRC of a range shorter than 4 bytes.")
    first = np.cumsum(lengths) - lengths
    within = np.arange(int(lengths.sum()), dtype=np.int64) - np.repeat(first, lengths)
    data = buf[np.repeat(starts, lengths) + within]
    for j in range(4):
        data[first + j] ^= 0xFF
    dist = np.repeat(lengths, lengths) - 1 - within
    local = dist % _CRC_SPAN
    contrib = tables["dist"][local, data]
    bounds = np.flatnonzero((within == 0) | (local == _CRC_SPAN - 1))
    reduced = np.bitwise_xor.reduceat(contrib, bounds)
    block = np.repeat(np.arange(lengths.size), lengths)[bounds]
    span_index = (dist // _CRC_SPAN)[bounds]
    width = int(span_index.max()) + 1
    spans = np.zeros((lengths.size, width), dtype=np.uint32)
    spans[block, width - 1 - span_index] = reduced
    span = tables["span"]
    state = np.zeros(lengths.size, dtype=np.uint32)
    for col in range(width):
        state = (
            span[0][state & 0xFF]
            ^ span[1][(state >> 8) & 0xFF]
            ^ span[2][(state >> 16) & 0xFF]
            ^ span[3][state >> 24]
            ^ spans[:, col]
        )
    return state


def _crc_bytes(content):
    buf = np.frombuffer(content, dtype=np.uint8)
    return int(crc32_koopman(buf, [0], [buf.size])[0])


###################################################################################################
## RED decoding


def _red_round(values):
    """Meflib RED_round: half away from zero, clipped to the infinity codes."""
    out = np.where(values >= 0, values + 0.5, values - 0.5)
    out = np.clip(out, RED_NEGATIVE_INFINITY, RED_POSITIVE_INFINITY)
    return np.trunc(out).astype(np.int64)


def red_decode(buf, starts, headers):
    """Decode RED blocks of a buffer, all blocks at once.

    The range decoder of meflib is run one symbol per step for every block (blocks
    sorted by length, so the active blocks are a prefix), and each decoded symbol
    updates the sample being rebuilt: a difference (-127 ... 127) adds to the
    previous sample; the flag -128 announces a keysample whose 4 little-endian
    bytes follow.

    Parameters
    ----------
    buf : 1d array of uint8
        Bytes holding the blocks, with at least 8 bytes after the last one.
    starts : 1d array of int
        Offset of each block in ``buf``.
    headers : structured array of BLOCK_DTYPE
        The block headers.

    Returns
    -------
    values : 1d array of int32
        Samples of the blocks, one block after the other.
    offsets : 1d array of int
        Index of the first sample of each block in ``values``.
    """
    n_blocks = len(starts)
    n_samples = headers["n_samples"].astype(np.int64)
    offsets = np.cumsum(n_samples) - n_samples
    values = np.zeros(int(n_samples.sum()), dtype=np.int32)
    # Coded symbols: difference_bytes counts the implicit first keysample flag
    n_symbols = np.where(n_samples > 0, headers["difference_bytes"].astype(np.int64) - 1, 0)
    if n_blocks == 0 or n_symbols.max() <= 0:
        return values, offsets
    counts = headers["statistics"].astype(np.int64)
    cumulative = np.zeros((n_blocks, 257), dtype=np.int64)
    np.cumsum(counts, axis=1, out=cumulative[:, 1:])
    totals = cumulative[:, 256]
    if np.any((totals == 0) & (n_symbols > 0)):
        raise ReaderError("MEF: a RED block has no symbol statistics (corrupted block).")
    # Symbol of each cumulative count, all blocks in one table
    lookup = np.repeat(np.tile(np.arange(256, dtype=np.int64), n_blocks), counts.ravel())
    base = np.cumsum(totals) - totals

    order = np.argsort(-n_symbols, kind="stable")
    sorted_symbols = n_symbols[order]
    n_steps = int(sorted_symbols[0])
    # active[step]: number of blocks that still have a symbol at this step
    active = n_blocks - np.searchsorted(sorted_symbols[::-1], np.arange(n_steps), side="right")

    pos = np.asarray(starts, dtype=np.int64)[order] + BLOCK_HEADER_BYTES
    in_byte = buf[pos].astype(np.int64)
    pos = pos + 1
    low = in_byte >> 1
    rng = np.full(n_blocks, 1 << 7, dtype=np.int64)
    total = totals[order]
    base = base[order]
    cum_row, count_row = order * 257, order * 256
    cum_flat, count_flat = cumulative.ravel(), counts.ravel()
    out_offset = offsets[order]
    written = np.zeros(n_blocks, dtype=np.int64)
    remaining = np.full(n_blocks, 4, dtype=np.int64)  # bytes left of a keysample
    key = np.zeros(n_blocks, dtype=np.int64)
    current = np.zeros(n_blocks, dtype=np.int64)
    bottom, mask32 = 1 << 23, 0xFFFFFFFF

    for step in range(n_steps):
        k = int(active[step])
        lo, rg, ib, ps = low[:k], rng[:k], in_byte[:k], pos[:k]
        # Renormalise: shift in one byte while the range is at most 2**23
        need = rg <= bottom
        while need.any():
            idx = np.flatnonzero(need)
            nxt = buf[ps[idx]].astype(np.int64)
            lo[idx] = (((lo[idx] << 8) | ((ib[idx] << 7) & 0xFF)) & mask32) | (nxt >> 1)
            ib[idx] = nxt
            ps[idx] += 1
            rg[idx] <<= 8
            need = rg <= bottom
        tk = total[:k]
        per_count = rg // tk
        count = np.minimum(lo // per_count, tk - 1)
        symbol = lookup[base[:k] + count]
        below = per_count * cum_flat[cum_row[:k] + symbol]
        lo -= below
        rg[:] = np.where(symbol == 255, rg - below, per_count * count_flat[count_row[:k] + symbol])
        # Rebuild the samples
        rem, ky, cur = remaining[:k], key[:k], current[:k]
        payload = rem > 0
        ky[:] = np.where(payload, ky | (symbol << ((4 - rem) << 3)), 0)
        rem[:] = np.where(payload, rem - 1, (symbol == 0x80) << 2)
        cur[:] = np.where(payload, ky - ((ky >> 31) << 32), cur + ((symbol ^ 0x80) - 0x80))
        done = np.flatnonzero(rem == 0)
        values[out_offset[done] + written[done]] = cur[done]
        written[done] += 1

    if not np.array_equal(written, n_samples[order]):
        raise ReaderError(
            "MEF: a RED block decodes to a wrong number of samples (corrupted block)."
        )

    # Lossy blocks: unscale, then add the trend back (meflib RED_unscale, RED_retrend)
    lossy = np.flatnonzero(
        (headers["scale_factor"] > 1.0)
        | (headers["detrend_slope"] != 0)
        | (headers["detrend_intercept"] != 0)
    )
    for ind in lossy:
        part = slice(offsets[ind], offsets[ind] + n_samples[ind])
        block = values[part].astype(np.float64)
        scale = float(headers["scale_factor"][ind])
        if scale > 1.0:
            block = _red_round(block * scale).astype(np.float64)
        slope = float(headers["detrend_slope"][ind])
        intercept = float(headers["detrend_intercept"][ind])
        if slope != 0 or intercept != 0:
            block = _red_round(block + slope * np.arange(1, block.size + 1) + intercept)
        values[part] = block
    return values, offsets


###################################################################################################
## Files


def _text(content, offset, nbytes):
    return content[offset : offset + nbytes].split(b"\0", 1)[0].decode("utf-8", "replace").strip()


def _universal_header(content, where, kind, check_crc):
    """Fields of the universal header; checks type, version, byte order and CRCs."""
    if len(content) < UNIVERSAL_HEADER_BYTES:
        raise ReaderError(f"{where}: truncated MEF file ({len(content)} bytes).")
    found = content[8:12].decode("ascii", "replace")
    if found != kind:
        raise ReaderError(f"{where}: not a MEF 3.0 {kind} file (type {found!r}).")
    major, minor, order = content[13], content[14], content[15]
    if major != 3:
        raise ReaderError(f"{where}: MEF version {major}.{minor}; only MEF 3 is supported.")
    if order != 1:
        raise ReaderError(f"{where}: big-endian MEF files are not supported.")
    header_crc, body_crc = struct.unpack_from("<II", content, 0)
    if check_crc:
        if header_crc and _crc_bytes(content[4:UNIVERSAL_HEADER_BYTES]) != header_crc:
            raise ReaderError(
                f"{where}: the CRC of the universal header does not match (corrupted "
                "file); check_crc=False reads it anyway."
            )
        if (
            body_crc
            and len(content) > UNIVERSAL_HEADER_BYTES + 4
            and _crc_bytes(content[UNIVERSAL_HEADER_BYTES:]) != body_crc
        ):
            raise ReaderError(
                f"{where}: the CRC of the file body does not match (corrupted or "
                "truncated file); check_crc=False reads it anyway."
            )
    start, end, n_entries, max_entry = struct.unpack_from("<qqqq", content, 16)
    return {
        "start_time": start,
        "end_time": end,
        "n_entries": n_entries,
        "segment": struct.unpack_from("<i", content, 48)[0],
        "channel": _text(content, 52, 256),
        "session": _text(content, 308, 256),
    }


def _encrypted(where, what, level):
    return ReaderError(
        f"{where}: {what} is encrypted (level {level}). evoparam reads unencrypted MEF 3.0 only; "
        "decrypt the session with the MEF library and its password first."
    )


def parse_metadata(content, where, check_crc=True):
    """Fields of a time-series metadata file (.tmet) used by the reader."""
    header = _universal_header(content, where, "tmet", check_crc)
    if len(content) < METADATA_BYTES:
        raise ReaderError(f"{where}: truncated metadata file ({len(content)} bytes).")
    section_2, section_3 = struct.unpack_from("<bb", content, 1024)
    if section_2 > 0:
        raise _encrypted(where, "the metadata (section 2)", section_2)
    (sfreq,) = struct.unpack_from("<d", content, 8720)
    if not (np.isfinite(sfreq) and sfreq > 0):
        raise ReaderError(f"{where}: no valid sampling frequency ({sfreq}).")
    (factor,) = struct.unpack_from("<d", content, 8760)
    start_sample, n_samples, n_blocks = struct.unpack_from("<qqq", content, 8912)
    return {
        "channel": header["channel"] or None,
        "session": header["session"],
        "segment": header["segment"],
        "start_time": header["start_time"],
        "sfreq": float(sfreq),
        "factor": float(factor),
        "units": _text(content, 8768, 128),
        "start_sample": max(int(start_sample), 0),
        "n_samples": int(n_samples),
        "n_blocks": int(n_blocks),
        "section_3_encrypted": section_3 > 0,
    }


def parse_index(content, where, check_crc=True):
    """Block index of a .tidx file (structured array of INDEX_DTYPE)."""
    header = _universal_header(content, where, "tidx", check_crc)
    body = len(content) - UNIVERSAL_HEADER_BYTES
    n_entries = body // INDEX_DTYPE.itemsize
    if header["n_entries"] >= 0:
        if header["n_entries"] > n_entries:
            raise ReaderError(
                "{}: truncated index ({} of {} entries).".format(
                    where, n_entries, header["n_entries"]
                )
            )
        n_entries = header["n_entries"]
    index = np.frombuffer(
        content, dtype=INDEX_DTYPE, count=n_entries, offset=UNIVERSAL_HEADER_BYTES
    ).copy()
    if np.any(index["flags"] & ENCRYPTION):
        raise _encrypted(where, "the data (RED blocks)", 1 if np.any(index["flags"] & 2) else 2)
    if np.any(index["start_time"] == UUTC_NO_ENTRY) or np.any(index["file_offset"] < 0):
        raise ReaderError(f"{where}: index entries without start time or file offset.")
    return index


###################################################################################################
## Session layout


def _natural_key(name):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _bids_channels_tsv(path):
    """Address of the BIDS channels.tsv next to a ``*_<datatype>.mefd`` session, or None."""
    stripped = path.rstrip("/")
    sibling = re.sub(r"_[A-Za-z]+\.mefd$", "_channels.tsv", stripped)
    return None if sibling == stripped else sibling


def _names_from_tsv(content):
    lines = content.decode("utf-8", "replace").splitlines()
    if not lines:
        return None
    header = lines[0].split("\t")
    if "name" not in header:
        return None
    col = header.index("name")
    return [line.split("\t")[col] for line in lines[1:] if line.strip()]


def _s3_listing(url):
    """(listing address, key prefix) of an S3 folder address, or None."""
    match = re.match(r"^(https?://s3[.-]([a-z0-9-]+[.-])?amazonaws\.com)/([^/?#]+)/([^?#]+)$", url)
    if match:
        return f"{match.group(1)}/{match.group(3)}", unquote(match.group(4))
    match = re.match(r"^(https?://[^/]+\.s3[.-]([a-z0-9-]+[.-])?amazonaws\.com)/([^?#]+)$", url)
    if match:
        return match.group(1), unquote(match.group(3))
    return None


def _list_remote(url, log):
    """Relative paths and sizes of the files of a remote folder (S3 listing), or None."""
    found = _s3_listing(url)
    if found is None:
        return None
    base, prefix = found
    prefix = prefix.rstrip("/") + "/"
    files, token = {}, None
    while True:
        query = "?list-type=2&prefix=" + quote(prefix, safe="")
        if token:
            query += "&continuation-token=" + quote(token, safe="")
        content = fetch_bytes(base + query, optional=True, log=log)
        if content is None:
            return None
        try:
            root = ElementTree.fromstring(content)
        except ElementTree.ParseError:
            return None
        ns = root.tag[: root.tag.index("}") + 1] if root.tag.startswith("{") else ""
        for item in root.iter(ns + "Contents"):
            key = item.findtext(ns + "Key") or ""
            if key.startswith(prefix):
                files[key[len(prefix) :]] = int(item.findtext(ns + "Size") or 0)
        if (root.findtext(ns + "IsTruncated") or "").lower() != "true":
            break
        token = root.findtext(ns + "NextContinuationToken")
        if not token:
            break
    return files


def _layout_from_paths(paths):
    """{channel: [segment numbers]} from paths relative to the session folder."""
    out = {}
    for rel in paths:
        match = re.match(r"^([^/]+)\.timd/\1-(\d{6})\.segd/\1-\2\.tmet$", rel)
        if match:
            out.setdefault(match.group(1), []).append(int(match.group(2)))
    return {name: sorted(segs) for name, segs in out.items()}


class _Session:
    """Files of a MEF session, on disk or behind an http(s) address."""

    def __init__(self, path, remote, log):
        path = os.fspath(path)
        if remote is None:
            remote = is_url(path)
        if remote and not is_url(path):
            raise ReaderError(f"remote=True needs an http(s) address, got {path}.")
        if not remote and is_url(path):
            raise ReaderError(f"remote=False but {path} is an http(s) address.")
        self.remote = bool(remote)
        self.root = path.rstrip("/") if self.remote else os.path.abspath(path)
        self.log = log
        if not self.remote and not os.path.isdir(self.root):
            raise ReaderError(f"MEF session not found: {self.root}")

    def rel(self, channel, segment, ext):
        seg = f"{channel}-{segment:06d}"
        return f"{channel}.timd/{seg}.segd/{seg}.{ext}"

    def location(self, rel):
        if self.remote:
            return self.root + "/" + "/".join(quote(part) for part in rel.split("/"))
        return os.path.join(self.root, *rel.split("/"))

    def read(self, rel, optional=False, log=None):
        """Whole content of a (small) file, or None for a missing optional file."""
        if self.remote:
            return fetch_bytes(self.location(rel), optional=optional, log=log)
        path = self.location(rel)
        if not os.path.isfile(path):
            if optional:
                return None
            raise ReaderError(f"MEF file not found: {path}")
        with open(path, "rb") as file:
            return file.read()

    def layout(self, hint=None):
        """{channel: [segment numbers] or None (unknown, probe)} in reading order."""
        if not self.remote:
            channels = {}
            for entry in os.listdir(self.root):
                folder = os.path.join(self.root, entry)
                if not (entry.endswith(".timd") and os.path.isdir(folder)):
                    continue
                name = entry[:-5]
                segs = []
                for sub in os.listdir(folder):
                    match = re.match(rf"^{re.escape(name)}-(\d{{6}})\.segd$", sub)
                    if match and os.path.isdir(os.path.join(folder, sub)):
                        segs.append(int(match.group(1)))
                channels[name] = sorted(segs)
        else:
            files = _list_remote(self.root, self.log)
            if files is not None:
                channels = _layout_from_paths(files)
            else:
                names = hint
                if names is None:
                    tsv = _bids_channels_tsv(self.root)
                    content = fetch_bytes(tsv, optional=True, log=self.log) if tsv else None
                    names = _names_from_tsv(content) if content is not None else None
                if names is None:
                    raise ReaderError(
                        f"{self.root}: a remote folder cannot be listed and no BIDS "
                        "channels.tsv was found next to it; give the channel "
                        "names (picks= with names)."
                    )
                channels = dict.fromkeys(names)
        if not channels:
            raise ReaderError(
                f"{self.root}: no time-series channel (.timd folder) in the MEF session."
            )
        order = [name for name in (hint or []) if name in channels]
        order += sorted((name for name in channels if name not in set(order)), key=_natural_key)
        return {name: channels[name] for name in order}


def _channel_names(path, *, remote=None, hint=None, log=None):
    """Names of the time-series channels of a session (ordered by ``hint``, then naturally)."""
    return list(_Session(path, remote, log).layout(hint))


def _read_channel(session, name, segments, check_crc):
    """Metadata and block index of one channel; returns (channel dict, bytes transferred)."""
    log = TransferLog()
    parts, number = [], 0
    probe = segments is None
    numbers = segments if not probe else [0]
    digest = []
    while numbers:
        number = numbers.pop(0)
        rel = session.rel(name, number, "tmet")
        content = session.read(rel, optional=probe and number > 0, log=log)
        if content is None:
            break
        where = session.location(rel)
        meta = parse_metadata(content, where, check_crc)
        rel_index = session.rel(name, number, "tidx")
        index_content = session.read(rel_index, log=log)
        index = parse_index(index_content, session.location(rel_index), check_crc)
        digest.append(f"{rel}:{hashlib.sha256(content).hexdigest()}")
        digest.append(f"{rel_index}:{hashlib.sha256(index_content).hexdigest()}")
        parts.append((number, meta, index))
        if probe:
            numbers = [number + 1]
    if not parts:
        raise ReaderError(f"{session.root}: channel {name} has no segment.")

    sfreqs = {meta["sfreq"] for _, meta, _ in parts}
    if len(sfreqs) != 1:
        raise ReaderError(
            f"{session.root}: the segments of channel {name} have different sampling frequencies {sorted(sfreqs)}."
        )
    meta = parts[0][1]
    blocks, segs = [], []
    for pos, (number, seg_meta, index) in enumerate(parts):
        segs.append(
            {
                "number": number,
                "tdat": session.rel(name, number, "tdat"),
                "start_sample": seg_meta["start_sample"],
            }
        )
        table = np.zeros(
            len(index),
            dtype=[
                ("segment", "i4"),
                ("offset", "i8"),
                ("time", "i8"),
                ("sample", "i8"),
                ("n", "i8"),
                ("nbytes", "i8"),
                ("flags", "u1"),
                ("maximum", "i4"),
                ("minimum", "i4"),
            ],
        )
        table["segment"] = pos
        table["offset"] = index["file_offset"]
        table["time"] = index["start_time"]
        table["sample"] = seg_meta["start_sample"] + index["start_sample"]
        table["n"] = index["n_samples"]
        table["nbytes"] = index["block_bytes"]
        table["flags"] = index["flags"]
        table["maximum"] = index["maximum"]
        table["minimum"] = index["minimum"]
        blocks.append(table)
    blocks = np.concatenate(blocks)
    blocks = blocks[blocks["n"] > 0]
    blocks = blocks[np.argsort(blocks["sample"], kind="stable")]

    units = meta["units"]
    factor = meta["factor"]
    if not np.isfinite(factor) or factor == 0:
        _warn(
            f"MEF channel {name}: no units conversion factor; the stored integers are used.",
            UserWarning,
            stacklevel=4,
        )
        factor = 1.0
    to_uv = _TO_UV.get(units.strip().lower().replace(" ", ""))
    channel = {
        "name": name,
        "sfreq": meta["sfreq"],
        "units": units,
        "factor": factor,
        "to_uv": 1.0 if to_uv is None else to_uv,
        "voltage": to_uv is not None,
        "section_3_encrypted": any(m["section_3_encrypted"] for _, m, _ in parts),
        "session": meta["session"],
        "segments": segs,
        "blocks": blocks,
        "digest": digest,
    }
    return channel, log.bytes_read


# An unflagged jump of the block start times longer than this (s) is a gap too
_UNFLAGGED_GAP = 1.0


def _positions(blocks, sfreq, t0):
    """Sample of each block on the session grid (sample 0 at time ``t0``, µs).

    Blocks follow their sample numbers within a run of contiguous blocks: the time stamps
    of real recordings jitter and drift by tens of samples (OpenNeuro ds003708, ds004457),
    while the onsets of their events.tsv are sample numbers divided by the sampling
    frequency. A run starts at a block flagged as a discontinuity, or whose start time
    jumps forward by more than 1 s; it is placed after the previous run by the time gap
    between them (the samples of the gap are NaN).
    """
    n = len(blocks)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    period = 1e6 / sfreq
    times, samples, counts = blocks["time"], blocks["sample"], blocks["n"]
    # Gap before each block: its start time minus the end time of the previous block
    gap = np.zeros(n)
    gap[1:] = (times[1:] - times[:-1]) / period - counts[:-1]
    heads = (blocks["flags"] & DISCONTINUITY).astype(bool) | (gap * period > _UNFLAGGED_GAP * 1e6)
    heads[0] = True
    pos = np.empty(n, dtype=np.int64)
    head_list = np.flatnonzero(heads)
    for k, head in enumerate(head_list):
        stop = head_list[k + 1] if k + 1 < len(head_list) else n
        if head == 0:
            start = int(np.round((times[0] - t0) / period))
        else:
            start = int(pos[head - 1] + counts[head - 1]) + max(int(np.round(gap[head])), 0)
        pos[head:stop] = start + (samples[head:stop] - samples[head])
    return pos


def read_mef_header(path, *, channels=None, remote=None, check_crc=True, log=None):
    """Read the metadata and block indices of a MEF 3.0 session.

    Parameters
    ----------
    path : str or path-like
        The session folder (``.mefd``), or its http(s) address.
    channels : list of str, optional
        Channels to read, in this order. Default: every time-series channel, in the
        order of the BIDS ``channels.tsv`` next to the session when there is one, else
        in natural order (``A2`` before ``A10``).
    remote : bool, optional
        Read through HTTP. Default: True when ``path`` is an http(s) address. A remote
        folder is listed with the S3 listing API (OpenNeuro); on other servers the
        channel names come from ``channels`` or from the BIDS ``channels.tsv``.
    check_crc : bool
        Check the CRCs of the universal headers and of the metadata and index files.
    log : TransferLog, optional
        Receives the bytes transferred.

    Returns
    -------
    header : dict
        Keys ``path``, ``remote``, ``session`` (session name), ``ch_names``,
        ``channels`` (one dict per channel: ``name``, ``sfreq``, ``units``, ``factor``
        (units conversion factor), ``to_uv``, ``segments``, ``blocks`` (structured array:
        segment, file offset, start time in µs, first sample, number of samples, bytes,
        flags, extrema) and ``pos`` (first sample of each block on the session grid)),
        ``sfreq`` (common sampling frequency, or None when the channels differ),
        ``start_time`` (µs, time of sample 0), ``n_samples`` (length of the grid at the
        common frequency) and ``section_3_encrypted``.

    Raises
    ------
    ReaderError
        Missing or corrupted files, unsupported or encrypted session, unknown channel.
    """
    path = os.fspath(path)
    own_log = TransferLog()
    session = _Session(path, remote, own_log)
    hint = None if channels is None else [str(name) for name in channels]
    tsv = _bids_channels_tsv(session.root) if hint is None else None
    if tsv and session.remote:
        content = fetch_bytes(tsv, optional=True, log=own_log)
        hint = _names_from_tsv(content) if content is not None else None
    elif tsv and os.path.isfile(tsv):
        with open(tsv, "rb") as file:
            hint = _names_from_tsv(file.read())
    layout = session.layout(hint)
    if channels is not None:
        missing = [name for name in hint if name not in layout]
        if missing:
            raise ReaderError(
                "{}: no channel {} in the MEF session.".format(
                    session.root, ", ".join(repr(name) for name in missing[:10])
                )
            )
        layout = {name: layout[name] for name in hint}

    jobs = [(name, None if segs is None else list(segs)) for name, segs in layout.items()]
    if session.remote and len(jobs) > 1:
        with concurrent.futures.ThreadPoolExecutor(min(_WORKERS, len(jobs))) as pool:
            results = list(
                pool.map(lambda job: _read_channel(session, job[0], job[1], check_crc), jobs)
            )
    else:
        results = [_read_channel(session, name, segs, check_crc) for name, segs in jobs]
    for _, nbytes in results:
        own_log.bytes_read += nbytes
    chans = [channel for channel, _ in results]

    times = [int(ch["blocks"]["time"][0]) for ch in chans if len(ch["blocks"])]
    t0 = min(times) if times else 0
    sfreqs = {ch["sfreq"] for ch in chans}
    sfreq = sfreqs.pop() if len(sfreqs) == 1 else None
    n_samples = 0
    for ch in chans:
        ch["pos"] = _positions(ch["blocks"], ch["sfreq"], t0)
        if sfreq is not None and len(ch["pos"]):
            n_samples = max(n_samples, int(np.max(ch["pos"] + ch["blocks"]["n"])))
    if log is not None:
        log.add(session.root, own_log.bytes_read)
    return {
        "path": session.root,
        "remote": session.remote,
        "session": chans[0]["session"],
        "ch_names": [ch["name"] for ch in chans],
        "channels": chans,
        "sfreq": sfreq,
        "start_time": t0,
        "n_samples": n_samples if sfreq is not None else None,
        "section_3_encrypted": any(ch["section_3_encrypted"] for ch in chans),
        "bytes_read": own_log.bytes_read,
    }


###################################################################################################
## Data


def _runs_of(blocks, need):
    """Contiguous byte runs of the needed blocks: list of (segment, offset, nbytes, [blocks])."""
    runs = []
    for ind in np.flatnonzero(need):
        seg, off, nbytes = (
            int(blocks["segment"][ind]),
            int(blocks["offset"][ind]),
            int(blocks["nbytes"][ind]),
        )
        if runs and runs[-1][0] == seg and runs[-1][1] + runs[-1][2] == off:
            runs[-1][2] += nbytes
            runs[-1][3].append(int(ind))
        else:
            runs.append([seg, off, nbytes, [int(ind)]])
    return runs


def _fetch_runs(session, rel, runs):
    """Bytes of the runs of one .tdat file; returns (list of bytes, digest lines, network bytes)."""
    out, lines = [], []
    if session.remote:
        reader = RemoteFile(session.location(rel), chunk_size=_CHUNK, max_chunks=1 << 16)
        read = reader.read
    else:
        reader = open(session.location(rel), "rb")

        def read(offset, nbytes):
            reader.seek(offset)
            return reader.read(nbytes)

    try:
        for run in runs:
            offset, nbytes = run[1], run[2]
            content = read(offset, nbytes)
            if len(content) != nbytes:
                raise ReaderError(
                    f"{session.location(rel)}: the file ends before byte {offset + nbytes} (truncated data file)."
                )
            out.append(content)
            lines.append(f"{rel}:{offset}-{offset + nbytes}:{hashlib.sha256(content).hexdigest()}")
    finally:
        reader.close()
    return out, lines, reader.bytes_read if session.remote else 0


def _convert(values, factor):
    """int32 samples -> float64 in µV, with the NaN and infinity codes."""
    out = values.astype(np.float64) * factor
    out[values == RED_NAN] = np.nan
    out[values == RED_NEGATIVE_INFINITY] = -np.inf
    out[values == RED_POSITIVE_INFINITY] = np.inf
    return out


def _read_grid(header, ch_index, starts, n_times, *, check_crc=True, log=None):
    """Windows ``[start, start + n_times)`` of the session grid.

    Constant blocks (equal extrema in the index, the censored parts of a recording for
    example) are not decoded; without ``check_crc`` they are not even read.

    Returns
    -------
    data : 3d array
        As [n_windows, n_channels, n_times], in µV (NaN where no block covers a sample).
    source : dict
        ``sha256`` of the bytes read (metadata, index and data ranges), ``data_bytes_read``
        (bytes of data transferred over the network) and ``blocks_read``.
    """
    session = _Session(header["path"], header["remote"], None)
    starts = np.atleast_1d(np.asarray(starts, dtype=np.int64))
    n_times = int(n_times)
    stops = starts + n_times
    out = np.full((starts.size, len(ch_index), n_times), np.nan)

    # Blocks that cover a window; data runs grouped per file
    pieces = []  # (col, channel index, first sample, n, values or None, value)
    files = {}
    for col, ch in enumerate(ch_index):
        channel = header["channels"][ch]
        blocks, pos = channel["blocks"], channel["pos"]
        if len(blocks) == 0:
            continue
        need = np.zeros(len(blocks), dtype=bool)
        for lo in range(0, starts.size, 64):
            need |= np.any(
                (pos[None, :] < stops[lo : lo + 64, None])
                & (pos[None, :] + blocks["n"][None, :] > starts[lo : lo + 64, None]),
                axis=0,
            )
        constant = need & (blocks["maximum"] == blocks["minimum"]) & (blocks["maximum"] != RED_NAN)
        for ind in np.flatnonzero(constant):
            pieces.append(
                (col, ch, int(pos[ind]), int(blocks["n"][ind]), None, int(blocks["maximum"][ind]))
            )
        for seg, offset, nbytes, members in _runs_of(
            blocks, need if check_crc else need & ~constant
        ):
            rel = channel["segments"][seg]["tdat"]
            files.setdefault(rel, []).append((seg, offset, nbytes, members, col, ch))

    # Read the runs (remote files in parallel)
    jobs = list(files.items())
    if session.remote and len(jobs) > 1:
        with concurrent.futures.ThreadPoolExecutor(min(_WORKERS, len(jobs))) as pool:
            fetched = list(pool.map(lambda job: _fetch_runs(session, job[0], job[1]), jobs))
    else:
        fetched = [_fetch_runs(session, rel, runs) for rel, runs in jobs]

    chunks, lines, network, cursor = [], [], 0, 0
    block_starts, info = [], []  # info: col, ch, block, n, bytes, constant
    for (rel, runs), (contents, file_lines, nbytes) in zip(jobs, fetched, strict=False):
        lines += file_lines
        network += nbytes
        for (_, offset, _, members, col, ch), content in zip(runs, contents, strict=False):
            blocks = header["channels"][ch]["blocks"]
            for ind in members:
                block_starts.append(cursor + int(blocks["offset"][ind]) - offset)
                info.append(
                    (
                        col,
                        ch,
                        ind,
                        int(blocks["n"][ind]),
                        int(blocks["nbytes"][ind]),
                        bool(
                            blocks["maximum"][ind] == blocks["minimum"][ind]
                            and blocks["maximum"][ind] != RED_NAN
                        ),
                    )
                )
            chunks += [content, bytes(8)]
            cursor += len(content) + 8
    buf = np.frombuffer(b"".join(chunks) + bytes(8), dtype=np.uint8)
    block_starts = np.asarray(block_starts, dtype=np.int64)

    if block_starts.size:
        table = np.array([item[3:] for item in info], dtype=np.int64).reshape(-1, 3)
        if np.any(table[:, 1] < BLOCK_HEADER_BYTES) or np.any(
            block_starts + table[:, 1] > buf.size
        ):
            raise ReaderError(
                "{}: a data block is shorter than its header (corrupted index).".format(
                    header["path"]
                )
            )
        rows = buf[block_starts[:, None] + np.arange(BLOCK_HEADER_BYTES)]
        headers = np.frombuffer(rows.tobytes(), dtype=BLOCK_DTYPE)

        def fail(which, message):
            col, ch, ind = info[int(which)][:3]
            raise ReaderError(
                "{}: block {} of channel {} {}".format(
                    header["path"], ind, header["ch_names"][ch], message
                )
            )

        bad = np.flatnonzero(
            (headers["n_samples"] != table[:, 0]) | (headers["block_bytes"] != table[:, 1])
        )
        if bad.size:
            fail(bad[0], "does not match its index entry (corrupted file).")
        if np.any(headers["flags"] & ENCRYPTION):
            raise _encrypted(
                header["path"], "the data (RED blocks)", 1 if np.any(headers["flags"] & 2) else 2
            )
        if check_crc:
            has = np.flatnonzero(headers["crc"] != 0)
            crc = crc32_koopman(buf, block_starts[has] + 4, table[has, 1] - 4)
            wrong = has[crc != headers["crc"][has]]
            if wrong.size:
                fail(
                    wrong[0],
                    f"has a wrong CRC ({wrong.size} corrupted blocks); check_crc=False reads "
                    "them anyway.",
                )
        decode = np.flatnonzero(table[:, 2] == 0)
        values, offsets = red_decode(buf, block_starts[decode], headers[decode])
        for slot, which in enumerate(decode):
            col, ch, ind, n = info[which][:4]
            pieces.append(
                (
                    col,
                    ch,
                    int(header["channels"][ch]["pos"][ind]),
                    n,
                    values[offsets[slot] : offsets[slot] + n],
                    None,
                )
            )

    # Place the samples on the windows
    order = np.argsort(starts, kind="stable")
    sorted_starts = starts[order]
    for col, ch, pos, n, values, constant in pieces:
        first = np.searchsorted(sorted_starts, pos - n_times, side="right")
        last = np.searchsorted(sorted_starts, pos + n, side="left")
        if first >= last:
            continue
        channel = header["channels"][ch]
        block = _convert(
            np.full(n, constant, dtype=np.int32) if values is None else values,
            channel["factor"] * channel["to_uv"],
        )
        for win in order[first:last]:
            lo, hi = max(pos, int(starts[win])), min(pos + n, int(stops[win]))
            if hi > lo:
                out[win, col, lo - starts[win] : hi - starts[win]] = block[lo - pos : hi - pos]

    meta_lines = []
    for ch in sorted(set(ch_index)):
        meta_lines += header["channels"][ch]["digest"]
    digest = hashlib.sha256("\n".join(sorted(meta_lines + lines)).encode("utf-8")).hexdigest()
    if log is not None and network:
        log.add(header["path"], network)
    return out, {
        "sha256": digest,
        "data_bytes_read": network,
        "blocks_read": int(block_starts.size),
    }


###################################################################################################
## Public readers


def _pick_mef(header, picks):
    index = _pick(header["ch_names"], picks)
    rates = {header["channels"][ind]["sfreq"] for ind in index}
    if len(rates) != 1:
        raise DataError(
            f"read_mef: the picked channels have different sampling frequencies {sorted(rates)}; "
            "pick channels of one frequency."
        )
    return index, rates.pop()


def _warn_units(header, ch_index):
    other = [header["ch_names"][ind] for ind in ch_index if not header["channels"][ind]["voltage"]]
    if other:
        _warn(
            "MEF channels whose unit is not a voltage were left unconverted: {}.".format(
                ", ".join(other)
            ),
            UserWarning,
            stacklevel=4,
        )


def _grid_length(header, ch_index):
    length = 0
    for ind in ch_index:
        channel = header["channels"][ind]
        if len(channel["pos"]):
            length = max(length, int(np.max(channel["pos"] + channel["blocks"]["n"])))
    return length


def _read_with_source(path, tmin=None, tmax=None, picks=None, remote=None, check_crc=True):
    """read_mef, plus the record of what was read (path, sha256, bytes read, ...)."""
    log = TransferLog()
    names = None
    if picks is not None:
        items = [picks] if isinstance(picks, str) else list(picks)
        if items and all(isinstance(item, str) for item in items):
            names = items  # read the metadata of these channels only
    header = read_mef_header(path, channels=names, remote=remote, check_crc=check_crc, log=log)
    ch_index, sfreq = _pick_mef(header, picks)
    _warn_units(header, ch_index)
    n_samples = _grid_length(header, ch_index)
    first = 0 if tmin is None else max(_sample(tmin, sfreq), 0)
    stop = n_samples if tmax is None else min(_sample(tmax, sfreq), n_samples)
    if stop <= first:
        raise ReaderError(
            "{}: the requested time window is outside the recording ({} samples at {} Hz).".format(
                header["path"], n_samples, sfreq
            )
        )
    data, record = _read_grid(
        header, ch_index, [first], stop - first, check_crc=check_crc, log=log
    )
    source = {
        "path": header["path"],
        "format": "mef",
        "sha256": record["sha256"],
        "bytes_read": log.bytes_read,
        "data_bytes_read": record["data_bytes_read"],
        "blocks_read": record["blocks_read"],
        "remote": header["remote"],
        "start_time_us": header["start_time"],
        "units": sorted({header["channels"][ind]["units"] for ind in ch_index}),
        "units_conversion_factor": sorted({header["channels"][ind]["factor"] for ind in ch_index}),
    }
    return data[0], sfreq, [header["ch_names"][ind] for ind in ch_index], first / sfreq, source


def read_mef(path, *, tmin=None, tmax=None, picks=None, remote=None, check_crc=True):
    """Read a MEF 3.0 session, or a time window of it.

    Parameters
    ----------
    path : str or path-like
        The session folder (``.mefd``), or its http(s) address.
    tmin, tmax : float, optional
        Window in seconds from sample 0 (the earliest block of the channels read);
        ``tmin`` included, ``tmax`` excluded. Default: the whole recording.
    picks : list of str or int, optional
        Channels by name or index, in this order. With names, only the metadata of these
        channels is read. The picked channels must share one sampling frequency.
    remote : bool, optional
        Read through HTTP range requests. Default: True when ``path`` is an http(s)
        address.
    check_crc : bool
        Check the CRCs of the headers, of the metadata and index files and of every
        block read; a mismatch raises a ReaderError.

    Returns
    -------
    data : 2d array
        Signal values in µV (channels whose unit is not a voltage: unconverted, with a
        warning), as [n_channels, n_times]; NaN where no data was recorded.
    sfreq : float
        Sampling frequency, in Hz.
    ch_names : list of str
    t0 : float
        Time of the first returned sample, in seconds from sample 0.

    Raises
    ------
    ReaderError
        Missing, corrupted, encrypted or unsupported files; window outside the recording.
    DataError
        Unknown channel in ``picks``; picked channels of different sampling frequencies.

    Examples
    --------
    ``data, sfreq, names, t0 = read_mef("sub-1_task-ccep_run-01_ieeg.mefd", tmin=10, tmax=20)``
    """
    data, sfreq, ch_names, t0, _ = _read_with_source(path, tmin, tmax, picks, remote, check_crc)
    return data, sfreq, ch_names, t0
