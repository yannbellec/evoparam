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

from ._harmonize import HarmonizationReport as HarmonizationReport
from ._harmonize import ValidationReport as ValidationReport
from ._harmonize import harmonize_stim_events as harmonize_stim_events
from ._harmonize import validate_stimulation_events as validate_stimulation_events
from ._validate import format_stim_report as format_stim_report
from ._validate import validate_stim_metadata as validate_stim_metadata
from .bids import read_bids as read_bids
from .bids import read_bids_events as read_bids_events
from .bids import select_stimulation_events as select_stimulation_events
from .brainvision import read_brainvision as read_brainvision
from .brainvision import read_brainvision_header as read_brainvision_header
from .brainvision import read_brainvision_segment as read_brainvision_segment
from .brainvision import write_brainvision as write_brainvision
from .derivatives import read_bids_derivatives as read_bids_derivatives
from .derivatives import write_bids_derivatives as write_bids_derivatives
from .epochs import epoch as epoch
from .epochs import epoch_data as epoch_data
from .fetch import OPENNEURO_URL as OPENNEURO_URL
from .fetch import RemoteFile as RemoteFile
from .fetch import fetch_erpcore as fetch_erpcore
from .fetch import fetch_openneuro_file as fetch_openneuro_file
from .fetch import openneuro_url as openneuro_url
from .mef import read_mef as read_mef
from .mef import read_mef_header as read_mef_header
from .neo import read_neo as read_neo
from .nwb import read_nwb as read_nwb
from .nwb import read_nwb_results as read_nwb_results
from .nwb import write_nwb as write_nwb
from .readers import from_array as from_array
from .readers import read as read
from .readers import read_edf as read_edf
from .readers import read_events_edf as read_events_edf
from .readers import read_mne as read_mne
from .registry import dataset_info as dataset_info
from .registry import list_datasets as list_datasets
