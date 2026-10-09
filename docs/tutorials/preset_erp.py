"""
Event-related potentials: the ``erp`` preset on ERP CORE
========================================================

ERP CORE (Kappenman et al. 2021, CC-BY 4.0) gives, for 40 participants, the epochs of seven
classic paradigms after a published pipeline. This tutorial takes participant 1 and three
components: the N170 (faces, PO8), the P3 (rare targets, Pz) and the MMN (deviants, FCz).
For each, the ``erp`` preset fits the average of the condition carrying the component, the
model is put next to the classical measures of the published measurement window (peak,
mean amplitude, 50 % area latency), and the two conditions are contrasted with the classical
mean amplitude.

Data: :func:`evoparam.io.fetch_erpcore` downloads the EEGLAB files from OSF (DOI
10.18115/D5JW4R) and checks them against the SHA-256 of the dataset registry. Epochs flagged
by the published artifact detection are left out, as in the published analysis.

How often the model finds the component, over the 40 participants, is measured by the
layer-3 validation (``validation/erpcore.py``; ``validation/layer3/erpcore_parent.json``):
read those numbers before trusting a single participant's fit.
"""

# %%
# Read one participant and component
# ----------------------------------

import re
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat

import evoparam
from evoparam import presets
from evoparam.data import Evoked
from evoparam.io import fetch_erpcore
from evoparam.models import classical_measures

import mne

mne.set_log_level("ERROR")

# component -> channel, bin of the condition carrying it, other bin, measurement window,
# polarity (published ERP CORE analysis)
DESIGNS = {"N170": ("PO8", 1, 2, (0.110, 0.150), -1),
           "P3": ("Pz", 1, 2, (0.300, 0.600), 1),
           "MMN": ("FCz", 1, 2, (0.125, 0.225), -1)}


def averages(component):
    """Averages of the two conditions at the component's channel (µV), flagged epochs out."""

    channel, plus, minus, _, _ = DESIGNS[component]
    path = fetch_erpcore(component, participant=1)
    epochs = mne.read_epochs_eeglab(path)
    eeg = loadmat(path, squeeze_me=True, struct_as_record=False)["EEG"]
    flags = np.atleast_1d(getattr(eeg.reject, "rejmanual", np.zeros(0))).astype(bool)
    keep = ~flags if flags.size == len(epochs) else np.ones(len(epochs), bool)
    inverse = {code: name for name, code in epochs.event_id.items()}
    bins = [set(map(int, re.findall(r"B(\d+)", inverse[c].split("(")[0])))
            for c in epochs.events[:, 2]]
    data = epochs.get_data(picks=[channel])[:, 0] * 1e6
    out = {}
    for name, number in (("carrying", plus), ("other", minus)):
        rows = [i for i, b in enumerate(bins) if number in b and keep[i]]
        out[name] = (data[rows].mean(axis=0), len(rows))
    return epochs.times, out


# %%
# Model and classical measures
# ----------------------------

rows, groups = [], {}
for component, (channel, _, _, window, polarity) in DESIGNS.items():
    times, waves = averages(component)
    sfreq = 1.0 / np.median(np.diff(times))
    wave, n = waves["carrying"]
    # the label window is the measurement window widened by 30 ms on each side
    preset = presets.derive("erp", name=f"erp_{component}",
                            component_labels={component: (window[0] - 0.03,
                                                          window[1] + 0.03)},
                            label_polarity={component: polarity})
    evoked = Evoked(data=wave[None], times=times, sfreq=sfreq, ch_names=[channel])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = evoparam.fit(evoked, preset, existence=False, random_state=0)
    groups[component] = group
    fit = group.fits[0]
    comp = next((c for c in fit.components if c.label == component), None)
    classical = classical_measures(times, wave, window=window,
                                   polarity="negative" if polarity < 0 else "positive")
    other = classical_measures(times, waves["other"][0], window=window,
                               polarity="negative" if polarity < 0 else "positive")
    rows.append({"component": component, "channel": channel, "n_trials": n,
                 "status": fit.status,
                 "model_latency_ms": 1e3 * comp.latency if comp else np.nan,
                 "model_amplitude": comp.amplitude if comp else np.nan,
                 "classical_peak_latency_ms": 1e3 * classical["peak_latency"],
                 "classical_50%_area_ms": 1e3 * classical["fractional_area_latency"],
                 "mean_amplitude": classical["mean_amplitude"],
                 "mean_amplitude_other": other["mean_amplitude"]})
table = pd.DataFrame(rows)
print(table.round(2).to_string(index=False))

# %%
# Condition contrast
# ------------------
# The component is larger in the condition that carries it, as the paradigms are designed.

table["contrast_uV"] = table["mean_amplitude"] - table["mean_amplitude_other"]
print(table[["component", "contrast_uV"]].round(2).to_string(index=False))

# %%
# Over the 40 participants
# ------------------------

import json
import os

summary = os.path.join(os.path.dirname(os.path.abspath(evoparam.__file__)), os.pardir,
                       "validation", "layer3", "erpcore_parent.json")
if os.path.isfile(summary):
    with open(summary, encoding="utf-8") as fobj:
        print(pd.DataFrame(json.load(fobj)["summary"]).round(3).to_string(index=False))

# %%
# Methods paragraph (P3)
# ----------------------

print(evoparam.reports.methods_text(groups["P3"]))
