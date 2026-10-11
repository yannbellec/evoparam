# SPDX-License-Identifier: Apache-2.0
"""
From a raw recording to CCEP peak parameters
============================================

This example runs the full evoparam chain on public data, without downloading the
whole recording:

1. io     : read the stimulation events, then stream 50 seconds of one ECoG run
            (OpenNeuro ds004080, Utrecht CCEP dataset, patient sub-ccepAgeUMCU01).
2. events : find the stimulation onsets from the signal alone, and compare them
            to the onsets written in the events file.
3. artifact: remove the stimulation artifact by template subtraction, and score each
            trial.
4. models : re-reference, fit each channel's average with the ccep preset, and print
            the latency, amplitude and uncertainty of the N1.

Data: van Blooijs et al., OpenNeuro ds004080 (CC0). Only about 55 MB are read.
"""

import numpy as np

from evoparam.io import (openneuro_url, read_bids_events, select_stimulation_events,
                         read_brainvision_segment, epoch_data, from_array)
from evoparam.events import detect_pulses, match_onsets
from evoparam.artifact import remove_artifact
import evoparam
from evoparam.prep import rereference

###################################################################################################
# Settings

DATASET = 'ds004080'
RUN = ('sub-ccepAgeUMCU01/ses-1/ieeg/'
       'sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448')
SITE = 'PT01-PT02'                  # stimulated electrode pair
EPOCH = (-0.5, 1.0)                 # seconds around each pulse
ARTIFACT = (-0.002, 0.010)          # artifact window, in seconds
BASELINE = [-0.5, -0.05]            # where to measure the noise level

###################################################################################################
# 1. io: events, then a segment of the recording around them

events = select_stimulation_events(
    read_bids_events(openneuro_url(DATASET, RUN + '_events.tsv')), site=SITE)
onsets = np.array([event['onset'] for event in events])
print('{} pulses on {} (events file)'.format(len(onsets), SITE))

data, sfreq, ch_names, t0 = read_brainvision_segment(
    openneuro_url(DATASET, RUN + '_ieeg.vhdr'), onsets[0] + EPOCH[0] - 1, onsets[-1] + EPOCH[1] + 1)
print('Read {} channels x {:.1f} s at {:.0f} Hz'.format(len(ch_names), data.shape[1] / sfreq, sfreq))

# Keep the grid channels, minus the two stimulated contacts (they carry the largest artifact)
picks = [ind for ind, name in enumerate(ch_names)
         if name.startswith('PT') and name not in SITE.split('-')]
data, ch_names = data[picks], [ch_names[ind] for ind in picks]

###################################################################################################
# 2. events: onsets from the signal alone

detected = detect_pulses(data, sfreq, t0=t0, ch_names=ch_names)
check = match_onsets(detected["onset_time"], onsets, tolerance=0.01)
print('Detection: {} / {} pulses found, {} false alarms, mean error {:.1f} ms'.format(
    check['hits'], len(onsets), check['false_alarms'],
    1000 * np.mean(check['errors']) if check['hits'] else np.nan))

###################################################################################################
# 3. artifact: cut epochs around the pulses, remove the artifact

times, epochs, kept = epoch_data(data, sfreq, onsets, *EPOCH, t0=t0)
epochs = epochs - epochs[..., (times >= BASELINE[0]) & (times <= BASELINE[1])].mean(axis=-1, keepdims=True)

# Subtract the median artifact of the 10 pulses, scaled to each pulse and channel
trials = from_array(epochs, sfreq=sfreq, times=times, ch_names=ch_names,
                    metadata={'stim_site': [SITE] * len(epochs)})
cleaned = remove_artifact(trials, method='template', window=ARTIFACT, baseline=BASELINE)
epochs = cleaned.cleaned.data
print('Artifact removed (template); median trial quality {:.2f}'.format(np.median(cleaned.quality)))

###################################################################################################
# 4. models: re-reference (adjusted common average), then fit each channel's average with
#    the ccep preset: a recovery-tail background plus components chosen by model comparison

cleaned_trials = rereference(cleaned.cleaned, 'car_adaptive', random_state=0)
fits = evoparam.fit(cleaned_trials, 'ccep', baseline=(-0.5, -0.05))

print('\nN1 (negative component, 9-50 ms) per channel:')
print('{:8} {:>12} {:>16} {:>12}'.format('channel', 'latency (ms)', 'amplitude (µV)', '± (ms)'))
for name in ch_names:
    fit = fits.get(name)
    n1 = [c for c in fit.components if c.label == 'N1']
    if n1:
        comp = n1[0]
        print('{:8} {:12.1f} {:16.1f} {:12.1f}'.format(name, 1000 * comp.latency, comp.amplitude,
                                                      1000 * comp.se['latency']))
