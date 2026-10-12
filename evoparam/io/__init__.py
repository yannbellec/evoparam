# SPDX-License-Identifier: Apache-2.0
"""Input / output: build TrialSets from arrays, files and BIDS datasets, locally or remotely.

- Arrays and files: :func:`from_array`, :func:`read` (dispatch), :func:`epoch`,
  :func:`read_brainvision`, :func:`read_mef` (MEF 3.0, local or remote),
  :func:`read_mne` (extra ``mne``), :func:`read_edf` (extra ``edf``).
- BIDS: :func:`read_bids` (local, or remote by HTTP byte ranges), with the stimulation
  events harmonised into the standard metadata columns (:func:`harmonize_stim_events`,
  :func:`validate_stimulation_events`), and the dataset-level validator of the stimulation
  metadata (:func:`validate_stim_metadata`, :func:`format_stim_report`).
- Remote files: :class:`RemoteFile`, :func:`openneuro_url`, :func:`fetch_openneuro_file`.
- Dataset registry: :func:`list_datasets`, :func:`dataset_info`.
"""

from .brainvision import (
    read_brainvision,
    read_brainvision_header,
    read_brainvision_segment,
    write_brainvision,
)
from .bids import read_bids, read_bids_events, select_stimulation_events
from ._validate import format_stim_report, validate_stim_metadata
from ._harmonize import (
    harmonize_stim_events,
    HarmonizationReport,
    validate_stimulation_events,
    ValidationReport,
)
from .epochs import epoch, epoch_data
from .mef import read_mef, read_mef_header
from .fetch import OPENNEURO_URL, RemoteFile, fetch_erpcore, fetch_openneuro_file, openneuro_url
from .readers import from_array, read, read_edf, read_events_edf, read_mne
from .registry import dataset_info, list_datasets
from .derivatives import read_bids_derivatives, write_bids_derivatives
from .nwb import read_nwb, read_nwb_results, write_nwb
from .neo import read_neo
