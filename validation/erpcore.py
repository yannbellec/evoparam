"""Layer 3, ERP CORE (spec §7.3, REQ-274): the component of each paradigm on its canonical
channel, model latency against the 50 % area latency of the same average.

ERP CORE (Kappenman et al. 2021, NeuroImage; OSF DOI 10.18115/D5JW4R; CC-BY 4.0) gives,
for 40 participants and seven components, the epochs after the published pipeline
(``*_epoch_interp_ar.set``: ICA correction, interpolation, artifact flags). For each
participant the published difference wave is built from the bins of the published bin
descriptor files, on the published channel(s); the published measurement window gives the
50 % area latency (area of the component's polarity only). The model latency compared with
it is the same measure on the fitted component's curve (its peak latency is reported too: a
right-skewed P3 peaks about 20 ms before its 50 % area). evoparam fits the difference
wave with the ``erp`` preset derived for the paradigm (its label window is the measurement
window; the fit window is the preset's, 50 to 700 ms, for the stimulus-locked paradigms,
and 200 ms around the measurement window for the response-locked LRP and ERN).

Data: ``EVOPARAM_ERPCORE`` (default ``~/.cache/evoparam/erpcore``) holding
``<component>/<participant>/*_epoch_interp_ar.set`` (and ``.fdt``), downloaded from OSF.
Needs the extra ``mne`` (EEGLAB reader). Run ``python validation/erpcore.py``.
"""

import glob
import json
import os
import re
import sys
import warnings

import numpy as np

ROOT = os.environ.get("EVOPARAM_ERPCORE", os.path.expanduser("~/.cache/evoparam/erpcore"))

# component -> (bins of the difference: (plus, minus) or "lateral", channels, measurement
# window (s), polarity, lateral bins (left, right) and channel pair (left, right))
DESIGNS = {
    "P3": {"plus": 1, "minus": 2, "channel": "Pz", "window": (0.300, 0.600), "polarity": 1},
    "N170": {"plus": 1, "minus": 2, "channel": "PO8", "window": (0.110, 0.150),
             "polarity": -1},
    "N400": {"plus": 1, "minus": 2, "channel": "CPz", "window": (0.300, 0.500),
             "polarity": -1},
    "MMN": {"plus": 1, "minus": 2, "channel": "FCz", "window": (0.125, 0.225), "polarity": -1},
    "N2pc": {"lateral": (1, 2), "pair": ("PO7", "PO8"), "window": (0.200, 0.275),
             "polarity": -1},
    "LRP": {"lateral": (1, 2), "pair": ("C3", "C4"), "window": (-0.100, 0.000),
            "polarity": -1},
    "ERN": {"plus": 1, "minus": 2, "channel": "FCz", "window": (0.000, 0.100), "polarity": -1},
}
TOLERANCE = 0.010
LABEL_MARGIN = 0.030


def _bins(name):
    """Bin numbers of an ERPLAB epoch label, for example 'B1(55)/201' -> {1}."""

    head = name.split("(")[0]
    return {int(b) for b in re.findall(r"B(\d+)", head)}


def load(component, participant):
    """(times, data [epochs, channels, times] in µV, channel names, bins per epoch) without
    the epochs flagged by the published artifact detection."""

    import mne
    from scipy.io import loadmat

    files = glob.glob(os.path.join(ROOT, component, str(participant), "*_epoch_interp_ar.set"))
    if not files:
        raise FileNotFoundError(f"{component}/{participant}: no epoched file under {ROOT}")
    mne.set_log_level("ERROR")
    epochs = mne.read_epochs_eeglab(files[0])
    flags = np.zeros(len(epochs), bool)
    try:
        eeg = loadmat(files[0], squeeze_me=True, struct_as_record=False)["EEG"]
        manual = np.atleast_1d(getattr(eeg.reject, "rejmanual", np.zeros(0)))
        if manual.size == len(epochs):
            flags = manual.astype(bool)
    except (KeyError, AttributeError, NotImplementedError):
        pass
    inverse = {v: k for k, v in epochs.event_id.items()}
    labels = [inverse[code] for code in epochs.events[:, 2]]
    data = epochs.get_data() * 1e6
    keep = ~flags
    return epochs.times, data[keep], list(epochs.ch_names), [_bins(l) for l, k in
                                                            zip(labels, keep) if k]


def noise_reference(component, data, names, bins, wave="difference"):
    """The plus-minus reference of the wave (trials of alternating signs averaged, then
    combined as the wave is): its noise, without the response."""

    from evoparam.models.engine import plus_minus

    design = DESIGNS[component]

    def pm(b, ch):
        rows = [i for i, s in enumerate(bins) if b in s]
        return plus_minus(data[rows][:, names.index(ch)])

    if "lateral" in design:
        left, right = design["lateral"]
        ch_l, ch_r = design["pair"]
        parts = [pm(left, ch_r), pm(left, ch_l), pm(right, ch_l), pm(right, ch_r)]
        if any(p is None for p in parts):
            return None
        return 0.5 * ((parts[0] - parts[1]) + (parts[2] - parts[3]))
    plus = pm(design["plus"], design["channel"])
    if wave == "parent" or plus is None:
        return plus
    minus = pm(design["minus"], design["channel"])
    return None if minus is None else plus - minus


def difference_wave(component, times, data, names, bins, wave="difference"):
    """The published difference wave (``wave="difference"``) or the parent wave of the
    condition carrying the component (``"parent"``: rare, faces, unrelated, deviants,
    incorrect; the lateralised N2pc and LRP are differences by definition)."""

    design = DESIGNS[component]
    def mean_of(b, ch):
        rows = [i for i, s in enumerate(bins) if b in s]
        return data[rows][:, names.index(ch)].mean(axis=0), len(rows)
    if "lateral" in design:
        left, right = design["lateral"]
        ch_l, ch_r = design["pair"]
        # contralateral minus ipsilateral, averaged over the two sides
        a, n1 = mean_of(left, ch_r)
        b, _ = mean_of(left, ch_l)
        c, n2 = mean_of(right, ch_l)
        d, _ = mean_of(right, ch_r)
        return 0.5 * ((a - b) + (c - d)), n1 + n2
    plus, n_plus = mean_of(design["plus"], design["channel"])
    if wave == "parent":
        return plus, n_plus
    minus, _ = mean_of(design["minus"], design["channel"])
    return plus - minus, n_plus


def area_latency(times, wave, window, polarity, fraction=0.5):
    """Fractional (50 %) area latency of the component's polarity in the window."""

    inside = (times >= window[0]) & (times <= window[1])
    part = np.clip(polarity * wave[inside], 0, None)
    total = part.sum()
    if total <= 0:
        return np.nan
    cumulative = np.cumsum(part)
    return float(times[inside][np.searchsorted(cumulative, fraction * total)])


def fit_component(component, times, wave, noise_ref=None):
    """The fit of the wave; with ``noise_ref`` (the plus-minus reference), the noise model
    comes from it when the engine's "auto" rule would take it (short baseline), as for a
    fit of the trials."""

    from evoparam import presets
    from evoparam.models import ResponseModel
    from evoparam.models import engine

    design = DESIGNS[component]
    lo, hi = design["window"]
    base = presets.get("erp")
    if lo >= base.fit_window[0]:
        # stimulus-locked: the preset's fit window (50 to 700 ms), which keeps the flat
        # stretch after the baseline that pins the constant background
        fit_window = (base.fit_window[0], min(base.fit_window[1], times[-1] - 0.01))
        baseline = base.baseline
    else:
        # response-locked (LRP, ERN): the 200 ms before the measurement window and after it
        fit_window = (max(lo - 0.2, times[0] + 0.01), min(hi + 0.2, times[-1] - 0.01))
        baseline = (times[0], fit_window[0])
    # the label window: the measurement window widened by LABEL_MARGIN on each side (the
    # measurement window is set for a mean amplitude; a fitted peak can lie just outside)
    label = (lo - LABEL_MARGIN, hi + LABEL_MARGIN)
    preset = presets.derive("erp", name=f"erp_{component}", fit_window=fit_window,
                            baseline=baseline, component_labels={component: label},
                            label_polarity={component: design["polarity"]})
    model = ResponseModel(preset)
    if noise_ref is not None and not engine.use_plus_minus(model.settings(), times, wave):
        noise_ref = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit_waveform(times, wave, noise_ref=noise_ref)
    comps = [c for c in result.components if c.label == component]
    return (comps[0] if comps else None), result


def run(components=None, participants=None, wave="difference"):
    rows = []
    for component in components or DESIGNS:
        folders = sorted(int(os.path.basename(p)) for p in
                         glob.glob(os.path.join(ROOT, component, "*")) if
                         os.path.basename(p).isdigit())
        for participant in participants or folders:
            try:
                times, data, names, bins = load(component, participant)
            except (FileNotFoundError, ValueError):  # missing or incomplete download
                continue
            signal, n = difference_wave(component, times, data, names, bins, wave)
            design = DESIGNS[component]
            if not np.all(np.isfinite(signal)):
                # a bin left without trials by the published artifact rejection: the
                # participant has no difference wave (excluded, as in the published analysis)
                rows.append({"component": component, "participant": participant, "n_trials": n,
                             "excluded": True})
                continue
            reference = area_latency(times, signal, design["window"], design["polarity"])
            comp, result = fit_component(component, times, signal,
                                         noise_reference(component, data, names, bins, wave))
            peak = comp.latency if comp is not None else np.nan
            latency = np.nan
            if comp is not None:
                from evoparam.models import evaluate_fit

                k = result.components.index(comp)
                curve = evaluate_fit(result, times, components=[k], background=False)
                latency = area_latency(times, curve, design["window"], design["polarity"])
            rows.append({"component": component, "participant": participant, "n_trials": n,
                         "wave": wave, "area_latency": reference, "model_latency": latency,
                         "model_peak_latency": peak, "labelled": comp is not None,
                         "excluded": False,
                         "status": result.status,
                         "noise_reference": result.diagnostics.get("noise_reference"),
                         "within_10ms": bool(comp is not None and
                                             abs(latency - reference) <= TOLERANCE)})
    return rows


def summary(rows):
    import pandas as pd

    table = pd.DataFrame(rows)
    excluded = table.groupby("component")["excluded"].sum()
    table = table[~table["excluded"].astype(bool)]
    out = table.groupby("component").agg(n=("participant", "count"),
                                         labelled=("labelled", "mean"),
                                         within_10ms=("within_10ms", "mean"))
    diff = (table["model_latency"] - table["area_latency"]) * 1e3
    out["median_diff_ms"] = diff.groupby(table["component"]).median()
    out["excluded"] = excluded
    return out


if __name__ == "__main__":
    wave = "parent" if "--parent" in sys.argv else "difference"
    rows = run([a for a in sys.argv[1:] if not a.startswith("--")] or None, wave=wave)
    table = summary(rows)
    print(table.to_string())
    out = os.path.join(os.path.dirname(__file__), "layer3", f"erpcore_{wave}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fobj:
        json.dump({"rows": rows, "summary": table.reset_index().to_dict("records")}, fobj,
                  indent=1, default=float)
