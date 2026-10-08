"""Tests for evoparam.io.fetch."""

import hashlib
import os

import pytest

import evoparam.io.fetch as fetch
from evoparam.errors import ReaderError
from evoparam.io.fetch import openneuro_url, fetch_openneuro_file
from evoparam.tests.io.helpers import http_server

###################################################################################################
###################################################################################################

def test_openneuro_url():

    url = openneuro_url('ds004080', '/sub-01/ses-1/ieeg/file.tsv')
    assert url == 'https://s3.amazonaws.com/openneuro.org/ds004080/sub-01/ses-1/ieeg/file.tsv'


def test_fetch_openneuro_file_with_sha256(tmp_path, monkeypatch):

    served = tmp_path / 'served' / 'dsTEST' / 'sub-01'
    served.mkdir(parents=True)
    content = b'onset\tduration\n1.0\t0.1\n'
    (served / 'events.tsv').write_bytes(content)
    good = hashlib.sha256(content).hexdigest()
    folder = str(tmp_path / 'downloads')

    with http_server(tmp_path / 'served') as (base, stats):
        monkeypatch.setattr(fetch, 'OPENNEURO_URL', base)
        path = fetch_openneuro_file('dsTEST', 'sub-01/events.tsv', folder, sha256=good)
        assert path == os.path.join(folder, 'dsTEST', 'sub-01', 'events.tsv')
        assert open(path, 'rb').read() == content

        # Present: not downloaded again, but checked
        n_requests = len(stats.requests)
        assert fetch_openneuro_file('dsTEST', 'sub-01/events.tsv', folder, sha256=good) == path
        assert len(stats.requests) == n_requests

        # A file that does not match is deleted
        with open(path, 'ab') as file:
            file.write(b'x')
        with pytest.raises(ReaderError, match='does not match'):
            fetch_openneuro_file('dsTEST', 'sub-01/events.tsv', folder, sha256=good)
        assert not os.path.exists(path)

        with pytest.raises(ReaderError, match='does not match'):
            fetch_openneuro_file('dsTEST', 'sub-01/events.tsv', folder, sha256='0' * 64)
        assert not os.path.exists(path)

        with pytest.raises(ReaderError, match='HTTP 404'):
            fetch_openneuro_file('dsTEST', 'sub-01/missing.tsv', folder)
        assert not os.path.exists(os.path.join(folder, 'dsTEST', 'sub-01', 'missing.tsv.part'))


def test_offline_mode(tmp_path, monkeypatch):

    monkeypatch.setenv('EVOPARAM_OFFLINE', '1')
    with pytest.raises(ReaderError, match='EVOPARAM_OFFLINE'):
        fetch_openneuro_file('ds004080', 'dataset_description.json', str(tmp_path))


def test_spurious_s3_not_found_is_retried(monkeypatch):

    import io
    from urllib.error import HTTPError

    import evoparam.io._fetch as fetch_module

    delays = []
    monkeypatch.setattr(fetch_module, '_sleep', delays.append)
    body = b'<?xml version="1.0"?><Error><Code>NoSuchBucket</Code></Error>'
    answers = [HTTPError('u', 404, 'Not Found', {}, io.BytesIO(body)), b'data']

    def once():
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    assert fetch_module._with_retries('u', once) == b'data' and delays == [0.5]
    # A missing key is not retried
    missing = b'<?xml version="1.0"?><Error><Code>NoSuchKey</Code></Error>'

    def gone():
        raise HTTPError('u', 404, 'Not Found', {}, io.BytesIO(missing))

    with pytest.raises(ReaderError, match='HTTP 404'):
        fetch_module._with_retries('u', gone)
    assert delays == [0.5]
