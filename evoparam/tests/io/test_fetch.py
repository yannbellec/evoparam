"""Tests for evoparam.io.fetch."""

from evoparam.io.fetch import openneuro_url

###################################################################################################
###################################################################################################

def test_openneuro_url():

    url = openneuro_url('ds004080', '/sub-01/ses-1/ieeg/file.tsv')
    assert url == 'https://s3.amazonaws.com/openneuro.org/ds004080/sub-01/ses-1/ieeg/file.tsv'
