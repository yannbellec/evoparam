"""HTTP access: whole-file downloads and partial reads by byte ranges.

This is the only module of the package that opens network connections. Every
request logs its URL at INFO level, honours ``EVOPARAM_OFFLINE=1`` (a
:class:`~evoparam.errors.ReaderError` is raised instead of connecting) and is
retried on transient failures with a backoff of 2, 4, 8 and 16 seconds (and, up to ten
times after 0.5 s, on the spurious S3 answer "404 NoSuchBucket").
"""

import hashlib
import http.client
import logging
import os
import time
from collections import OrderedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from evoparam.errors import ReaderError, warn as _warn

__all__ = ["RemoteFile", "is_url", "fetch_bytes", "download", "data_folder", "sha256_file",
           "TransferLog"]

logger = logging.getLogger(__name__)

BACKOFF = (2.0, 4.0, 8.0, 16.0)
TRANSIENT_STATUS = (408, 429, 500, 502, 503, 504)
SPURIOUS_RETRIES = 10
SPURIOUS_DELAY = 0.5
_TRANSIENT_ERRORS = (URLError, TimeoutError, ConnectionError, http.client.HTTPException)

# Module-level so that tests can replace it and run without waiting.
_sleep = time.sleep


def is_url(path):
    """True when ``path`` is an http(s) address."""

    return isinstance(path, str) and path.startswith(("http://", "https://"))


def data_folder():
    """Folder for downloads: ``EVOPARAM_DATA``, or ``~/.cache/evoparam``."""

    return os.environ.get("EVOPARAM_DATA") or os.path.join(os.path.expanduser("~"), ".cache",
                                                            "evoparam")


def sha256_file(path, block=1 << 20):
    """SHA-256 of a local file, read block by block."""

    digest = hashlib.sha256()
    with open(path, "rb") as fobj:
        for part in iter(lambda: fobj.read(block), b""):
            digest.update(part)
    return digest.hexdigest()


def _check_online(url):
    if os.environ.get("EVOPARAM_OFFLINE") == "1":
        raise ReaderError(f"EVOPARAM_OFFLINE=1: not connecting to {url}.")


class TransferLog:
    """Running count of the bytes transferred, shared by the reads of one call."""

    def __init__(self):
        self.bytes_read = 0
        self.files = []

    def add(self, url, nbytes):
        self.bytes_read += int(nbytes)
        self.files.append(url)


def _spurious_not_found(exc):
    """True for a 404 whose body is the S3 error ``NoSuchBucket``: S3 (or a proxy in front
    of it) intermittently answers so for a bucket that exists, and the same request then
    succeeds. A missing key (``NoSuchKey``) or any other 404 is not retried."""

    if exc.code != 404:
        return False
    try:
        body = exc.read(4096) or b""
    except (OSError, ValueError, AttributeError):
        return False
    return b"<Code>NoSuchBucket</Code>" in body


def _with_retries(url, attempt_once, retries=len(BACKOFF)):
    """Run ``attempt_once()`` and retry transient HTTP and network failures (a spurious S3
    404 up to SPURIOUS_RETRIES more times, after SPURIOUS_DELAY s, without using the
    backoff)."""

    _check_online(url)
    attempt = spurious = 0
    while True:
        try:
            return attempt_once()
        except HTTPError as exc:
            if _spurious_not_found(exc) and spurious < SPURIOUS_RETRIES:
                spurious += 1
                logger.info("Retrying %s after a spurious 404 (NoSuchBucket)", url)
                _sleep(SPURIOUS_DELAY)
                continue
            if exc.code not in TRANSIENT_STATUS or attempt == retries:
                raise ReaderError(f"HTTP {exc.code} ({exc.reason}) for {url}.") from exc
            error = exc
        except _TRANSIENT_ERRORS as exc:
            if attempt == retries:
                raise ReaderError(f"Cannot read {url}: {exc}.") from exc
            error = exc
        delay = BACKOFF[min(attempt, len(BACKOFF) - 1)]
        logger.warning("Retrying %s in %g s after: %s", url, delay, error)
        _sleep(delay)
        attempt += 1


def fetch_bytes(url, *, optional=False, timeout=30.0, log=None):
    """Download a small file whole and return its content.

    Parameters
    ----------
    url : str
        Address of the file.
    optional : bool
        If True, return None when the server answers 404 instead of raising.
    timeout : float
        Seconds to wait for the server.
    log : TransferLog, optional
        Receives the number of bytes transferred.

    Returns
    -------
    content : bytes or None
    """

    def once():
        logger.info("GET %s", url)
        with urlopen(Request(url), timeout=timeout) as response:
            return response.read()

    try:
        content = _with_retries(url, once)
    except ReaderError as exc:
        cause = exc.__cause__
        if optional and isinstance(cause, HTTPError) and cause.code in (403, 404):
            return None
        raise
    if log is not None:
        log.add(url, len(content))
    return content


def download(url, path, *, timeout=30.0, log=None):
    """Download ``url`` to ``path`` (written to a temporary name, then renamed).

    Returns
    -------
    sha256 : str
        SHA-256 of the downloaded file.
    """

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    partial = path + ".part"

    def once():
        logger.info("GET %s", url)
        digest = hashlib.sha256()
        with urlopen(Request(url), timeout=timeout) as response, open(partial, "wb") as fobj:
            for block in iter(lambda: response.read(1 << 20), b""):
                fobj.write(block)
                digest.update(block)
        return digest.hexdigest()

    try:
        sha256 = _with_retries(url, once)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    os.replace(partial, path)
    if log is not None:
        log.add(url, os.path.getsize(path))
    return sha256


class RemoteFile:
    """Read parts of a remote file through HTTP range requests.

    Bytes are fetched in aligned chunks kept in a least-recently-used cache, so
    reading neighbouring windows costs one request. Consecutive missing chunks
    are fetched with one request. Transient failures are retried four times,
    after 2, 4, 8 and 16 seconds. When the server ignores the ``Range`` header
    and answers ``200`` with the whole file, the file is saved to ``cache_dir``
    with a one-line warning giving its size, and later reads use the local copy.

    Parameters
    ----------
    url : str
        Address of the file.
    chunk_size : int
        Bytes per cached chunk (default 8 MiB).
    timeout : float
        Seconds to wait for the server.
    max_chunks : int
        Chunks kept in the cache (default 32).
    cache_dir : str, optional
        Where a whole-file fallback download is saved; default
        :func:`data_folder` ``/remote``.
    log : TransferLog, optional
        Receives the number of bytes transferred.

    Attributes
    ----------
    bytes_read : int
        Bytes transferred over the network by this object.
    full_download : bool
        True when the server ignored ``Range`` and the whole file was fetched.

    Examples
    --------
    ``with RemoteFile(url) as remote: header = remote.read(0, 1024)``
    """

    def __init__(self, url, *, chunk_size=8 << 20, timeout=30.0, max_chunks=32, cache_dir=None,
                 log=None):
        if not is_url(url):
            raise ReaderError(f"RemoteFile needs an http(s) address, got {url!r}.")
        if int(chunk_size) < 1 or int(max_chunks) < 1:
            raise ReaderError("RemoteFile: chunk_size and max_chunks must be positive.")
        self.url = url
        self.chunk_size = int(chunk_size)
        self.timeout = float(timeout)
        self.max_chunks = int(max_chunks)
        self.cache_dir = cache_dir
        self.log = log
        self.bytes_read = 0
        self.full_download = False
        self._size = None
        self._cache = OrderedDict()
        self._digests = {}                # chunk index -> (start, stop, sha256 of the chunk)
        self._local = None
        self._local_sha256 = None

    # Context manager
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        """Drop the cached chunks."""

        self._cache.clear()

    def __repr__(self):
        return f"<RemoteFile {self.url} | {self.bytes_read} bytes read>"

    def _count(self, nbytes):
        self.bytes_read += int(nbytes)
        if self.log is not None:
            self.log.add(self.url, nbytes)

    @property
    def size(self):
        """Size of the remote file in bytes (one HEAD request the first time)."""

        if self._size is None:
            def once():
                logger.info("HEAD %s", self.url)
                with urlopen(Request(self.url, method="HEAD"), timeout=self.timeout) as response:
                    return response.headers.get("Content-Length")
            length = _with_retries(self.url, once)
            if length is None:
                raise ReaderError(f"The server gives no size for {self.url}.")
            self._size = int(length)
        return self._size

    @property
    def ranges(self):
        """Byte ranges ``[start, stop)`` transferred so far, merged and sorted."""

        if self.full_download:
            return [[0, self._size]]
        merged = []
        for start, stop, _ in sorted(self._digests.values()):
            if merged and merged[-1][1] == start:
                merged[-1][1] = stop
            else:
                merged.append([start, stop])
        return merged

    def sha256(self):
        """SHA-256 of the bytes read.

        After a whole-file download this is the SHA-256 of the file. Otherwise it is
        the SHA-256 of the text ``"<start>-<stop>:<sha256 of the chunk>"`` (one line
        per chunk, in increasing offset), so it depends only on which bytes were read,
        not on the order or grouping of the requests.
        """

        if self.full_download:
            return self._local_sha256
        lines = [f"{start}-{stop}:{digest}"
                 for start, stop, digest in sorted(self._digests.values())]
        return hashlib.sha256("\n".join(lines).encode("ascii")).hexdigest()

    def read(self, offset, nbytes):
        """Return ``nbytes`` bytes from ``offset`` (fewer at the end of the file).

        Raises
        ------
        ReaderError
            If the server refuses the request or the connection fails after the retries.
        """

        offset, nbytes = int(offset), int(nbytes)
        if offset < 0 or nbytes < 0:
            raise ReaderError(f"RemoteFile.read: invalid range ({offset}, {nbytes}).")
        if nbytes == 0:
            return b""
        if self._local is not None:
            return self._read_local(offset, nbytes)
        stop = offset + nbytes if self._size is None else min(offset + nbytes, self._size)
        if stop <= offset:
            return b""
        first, last = offset // self.chunk_size, (stop - 1) // self.chunk_size
        # Take the cached chunks first: fetching the others may evict them.
        got, missing = {}, []
        for index in range(first, last + 1):
            if index in self._cache:
                self._cache.move_to_end(index)
                got[index] = self._cache[index]
            else:
                missing.append(index)
        for run in _runs(missing):
            fetched = self._fetch_chunks(run[0], run[-1])
            if fetched is None:                                     # whole-file fallback
                return self._read_local(offset, nbytes)
            got.update(fetched)
        parts = []
        for index in range(first, last + 1):
            if index not in got:                                    # past the end of the file
                break
            parts.append(got[index])
        joined = b"".join(parts)
        start = offset - first * self.chunk_size
        return joined[start:start + (stop - offset)]

    def _read_local(self, offset, nbytes):
        with open(self._local, "rb") as fobj:
            fobj.seek(offset)
            return fobj.read(nbytes)

    def _fetch_chunks(self, first, last):
        start = first * self.chunk_size
        stop = (last + 1) * self.chunk_size
        if self._size is not None:
            stop = min(stop, self._size)
        request = Request(self.url, headers={"Range": f"bytes={start}-{stop - 1}"})

        def once():
            logger.info("GET %s bytes=%d-%d", self.url, start, stop - 1)
            with urlopen(request, timeout=self.timeout) as response:
                if response.status == 200:
                    self._download_whole(response)
                    return None
                if response.status != 206:
                    raise ReaderError(f"Unexpected HTTP {response.status} for {self.url}.")
                self._check_content_range(response.headers.get("Content-Range"), start)
                body = response.read()
            # The size may have been learnt from this reply: a range that ends past the
            # end of the file is answered up to the end
            expected = min(stop, self._size) - start if self._size is not None else None
            if expected is not None and len(body) != expected:
                raise http.client.IncompleteRead(body, expected - len(body))
            return body

        body = _with_retries(self.url, once)
        if body is None:
            return None
        self._count(len(body))
        out = {}
        for index in range(first, last + 1):
            lo = (index - first) * self.chunk_size
            chunk = body[lo:lo + self.chunk_size]
            if not chunk:
                break
            chunk_start = index * self.chunk_size
            self._digests[index] = (chunk_start, chunk_start + len(chunk),
                                    hashlib.sha256(chunk).hexdigest())
            out[index] = chunk
            self._cache[index] = chunk
            self._cache.move_to_end(index)
            while len(self._cache) > self.max_chunks:
                self._cache.popitem(last=False)
        return out

    def _check_content_range(self, header, start):
        # Format: "bytes <first>-<last>/<total>"
        if not header or not header.startswith("bytes "):
            raise ReaderError(f"Range reply without a valid Content-Range from {self.url}.")
        span, _, total = header[6:].partition("/")
        first = int(span.split("-")[0])
        if first != start:
            raise ReaderError(f"The server returned bytes from {first} instead of {start} "
                              f"for {self.url}.")
        if total and total != "*":
            self._size = int(total)

    def _download_whole(self, response):
        folder = self.cache_dir or os.path.join(data_folder(), "remote")
        os.makedirs(folder, exist_ok=True)
        name = hashlib.sha256(self.url.encode("utf-8")).hexdigest()[:16] + "_" + \
            os.path.basename(self.url.split("?")[0])
        path = os.path.join(folder, name)
        length = response.headers.get("Content-Length")
        size_text = f"{int(length) / 1e6:.1f} MB" if length else "unknown size"
        _warn(f"The server ignored the byte-range request; downloading the whole file "
              f"({size_text}) to {path}.", UserWarning, stacklevel=4)
        digest = hashlib.sha256()
        partial = path + ".part"
        total = 0
        try:
            with open(partial, "wb") as fobj:
                for block in iter(lambda: response.read(1 << 20), b""):
                    fobj.write(block)
                    digest.update(block)
                    total += len(block)
        except BaseException:
            if os.path.exists(partial):
                os.remove(partial)
            raise
        os.replace(partial, path)
        self._count(total)
        self._size = total
        self._local = path
        self._local_sha256 = digest.hexdigest()
        self.full_download = True
        self._cache.clear()


def _runs(indices):
    """Split sorted integers into runs of consecutive values."""

    runs = []
    for index in indices:
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs
