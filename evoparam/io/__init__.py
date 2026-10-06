"""Input / output: read recordings and stimulation events, and cut epochs around events."""

from .brainvision import read_brainvision_header, read_brainvision_segment
from .bids import read_bids_events, select_stimulation_events
from .epochs import epoch_data
from .fetch import fetch_openneuro_file, openneuro_url
