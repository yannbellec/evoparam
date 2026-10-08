"""Write a simulated TrialSet as a small BIDS dataset with its ground truth.

The trials are laid end to end as one continuous BrainVision recording (one event per
trial at the time of its onset), with ``events.tsv``, ``channels.tsv``, the sidecar, the
dataset description, ``participants.tsv`` and the truth table in
``derivatives/truth/sub-<label>/<datatype>/<stem>_desc-truth_responses.tsv``.
"""

import json
import os

import numpy as np

from evoparam.errors import SettingsError
from evoparam.io import write_brainvision

__all__ = ["write_bids"]

DATATYPES = {"spes": "ieeg", "lfp_ep": "ieeg", "erna": "ieeg", "ecap": "ieeg"}


def _tsv(path, header, rows):
    with open(path, "w", encoding="utf-8") as fobj:
        fobj.write("\t".join(header) + "\n")
        for row in rows:
            fobj.write("\t".join("n/a" if v is None or (isinstance(v, float) and np.isnan(v))
                                 else str(v) for v in row) + "\n")


def write_bids(ts, truth, root, *, subject="01", task="sim", datatype=None):
    """Write ``ts`` (TrialSet) and ``truth`` (GroundTruth or DataFrame) as BIDS.

    Returns the path of the ``.vhdr`` file.
    """

    datatype = datatype or DATATYPES.get(ts.modality or "", "eeg")
    if datatype not in ("ieeg", "eeg"):
        raise SettingsError("datatype must be 'ieeg' or 'eeg'.")
    os.makedirs(root, exist_ok=True)
    folder = os.path.join(root, f"sub-{subject}", datatype)
    os.makedirs(folder, exist_ok=True)
    stem = f"sub-{subject}_task-{task}"
    n_trials, n_channels, n_times = ts.data.shape
    data = np.nan_to_num(ts.data).transpose(1, 0, 2).reshape(n_channels, n_trials * n_times)
    first = int(np.argmin(np.abs(ts.times)))
    onsets = [(k * n_times + first) / ts.sfreq for k in range(n_trials)]
    vhdr = write_brainvision(os.path.join(folder, f"{stem}_{datatype}.vhdr"), data, ts.sfreq,
                             ts.ch_names,
                             markers=[(k * n_times + first, "S  1") for k in range(n_trials)])
    meta = ts.metadata
    rows = []
    for k, onset in enumerate(onsets):
        row = [repr(round(onset, 9)), "0.001", "electrical_stimulation" if datatype == "ieeg"
               else "stimulus"]
        for column in ("stim_site", "stim_current_mA", "stim_pulse_width_us"):
            row.append(meta[column].iloc[k] if meta is not None and column in meta.columns
                       else None)
        rows.append(row)
    _tsv(os.path.join(folder, f"{stem}_events.tsv"),
         ["onset", "duration", "trial_type", "electrical_stimulation_site",
          "electrical_stimulation_current", "electrical_stimulation_pulsewidth"], rows)
    with open(os.path.join(folder, f"{stem}_events.json"), "w", encoding="utf-8") as fobj:
        json.dump({"electrical_stimulation_current": {"Units": "mA"},
                   "electrical_stimulation_pulsewidth": {"Units": "us"}}, fobj, indent=1)
    _tsv(os.path.join(folder, f"{stem}_channels.tsv"),
         ["name", "type", "units", "low_cutoff", "high_cutoff", "status"],
         [[name, "ECOG" if datatype == "ieeg" else "EEG", "uV", "n/a", "n/a", "good"]
          for name in ts.ch_names])
    sidecar = {"TaskName": task, "SamplingFrequency": float(ts.sfreq),
               "PowerLineFrequency": 50, "SoftwareFilters": "n/a"}
    if datatype == "ieeg":
        # iEEG requires the reference and the electrode positions (simulated contacts on a
        # line, 5 mm apart, in an arbitrary coordinate system).
        sidecar.update({"iEEGReference": "n/a", "ElectricalStimulation": True,
                        "ElectricalStimulationParameters": "simulated by evoparam"})
        info = ts.ch_info
        coords = info is not None and all(c in info.columns for c in ("x", "y", "z"))
        _tsv(os.path.join(folder, f"sub-{subject}_electrodes.tsv"),
             ["name", "x", "y", "z", "size"],
             [[name] + ([float(info[c].iloc[i]) for c in ("x", "y", "z")] if coords
                        else [5.0 * i, 0.0, 0.0]) + [None]
              for i, name in enumerate(ts.ch_names)])
        with open(os.path.join(folder, f"sub-{subject}_coordsystem.json"), "w",
                  encoding="utf-8") as fobj:
            json.dump({"iEEGCoordinateSystem": "Other", "iEEGCoordinateUnits": "mm",
                       "iEEGCoordinateSystemDescription": "simulated contact positions"},
                      fobj, indent=1)
    else:
        sidecar.update({"EEGReference": "n/a"})
    with open(os.path.join(folder, f"{stem}_{datatype}.json"), "w", encoding="utf-8") as fobj:
        json.dump(sidecar, fobj, indent=1)
    with open(os.path.join(root, "dataset_description.json"), "w", encoding="utf-8") as fobj:
        json.dump({"Name": "evoparam simulation", "BIDSVersion": "1.9.0",
                   "DatasetType": "raw", "Authors": ["evoparam"],
                   "GeneratedBy": [{"Name": "evoparam.sim"}]}, fobj, indent=1)
    with open(os.path.join(root, "README"), "w", encoding="utf-8") as fobj:
        fobj.write("Synthetic dataset written by evoparam (evoparam.sim.bids.write_bids); "
                   "the ground truth of every trial is in derivatives/truth.\n")
    participants = os.path.join(root, "participants.tsv")
    existing = []
    if os.path.exists(participants):
        with open(participants, encoding="utf-8") as fobj:
            existing = [line.split("\t")[0] for line in fobj.read().splitlines()[1:]]
    if f"sub-{subject}" not in existing:
        _tsv(participants, ["participant_id"], [[p] for p in existing + [f"sub-{subject}"]])
    table = truth.to_dataframe() if hasattr(truth, "to_dataframe") else truth
    truth_folder = os.path.join(root, "derivatives", "truth", f"sub-{subject}", datatype)
    os.makedirs(truth_folder, exist_ok=True)
    with open(os.path.join(root, "derivatives", "truth", "dataset_description.json"), "w",
              encoding="utf-8") as fobj:
        json.dump({"Name": "evoparam simulation truth", "BIDSVersion": "1.9.0",
                   "DatasetType": "derivative",
                   "GeneratedBy": [{"Name": "evoparam.sim"}]}, fobj, indent=1)
    table.to_csv(os.path.join(truth_folder, f"{stem}_desc-truth_responses.tsv"), sep="\t",
                 index=False, na_rep="n/a")
    return vhdr
