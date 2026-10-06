"""Read stimulation events from BIDS events files (events.tsv)."""

import csv
import os
from urllib.request import urlopen

###################################################################################################
###################################################################################################

def read_bids_events(events_tsv):
    """Read a BIDS events.tsv file.

    Parameters
    ----------
    events_tsv : str
        Path or URL of the events.tsv file.

    Returns
    -------
    events : list of dict
        One dict per row, with column names as keys. 'onset' and 'duration' are converted
        to float; the BIDS value 'n/a' becomes None.
    """

    if events_tsv.startswith(('http://', 'https://')):
        with urlopen(events_tsv) as response:
            lines = response.read().decode('utf-8').splitlines()
    else:
        with open(events_tsv, encoding='utf-8') as file:
            lines = file.read().splitlines()

    events = []
    for row in csv.DictReader(lines, delimiter='\t'):
        row = {key : (None if val == 'n/a' else val) for key, val in row.items()}
        for key in ('onset', 'duration'):
            if row.get(key) is not None:
                row[key] = float(row[key])
        events.append(row)

    return events


def select_stimulation_events(events, site=None, trial_type='electrical_stimulation'):
    """Select stimulation events, optionally for one stimulation site.

    Parameters
    ----------
    events : list of dict
        Events, as returned by `read_bids_events`.
    site : str, optional
        Stimulation site, as written in the 'electrical_stimulation_site' column
        (for example 'PT01-PT02'). If None, keeps all sites.
    trial_type : str, optional, default: 'electrical_stimulation'
        Value of the 'trial_type' column that marks a stimulation.

    Returns
    -------
    selected : list of dict
        The selected events, in their original order.
    """

    return [event for event in events if event.get('trial_type') == trial_type and
            (site is None or event.get('electrical_stimulation_site') == site)]
