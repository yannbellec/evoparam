# SPDX-License-Identifier: Apache-2.0
"""The ``tep`` preset on OpenNeuro ds002094, single-pulse TMS-EEG (REQ-275).

Dataset: ds002094 v1.0.0 (licence CC0, in ``dataset_description.json``), 20 healthy
participants, 600 single TMS pulses each over the right motor cortex at 120 % of the resting
motor threshold, 30 scalp EEG channels at 5 000 Hz (DC recording, reference POz), markers
``R128`` at the pulses. Task ``tmseeg1``, followed by ``tmseeg2`` for sub-728, sub-766 and
sub-780, as the dataset README says. Runs only with EVOPARAM_NETWORK_TESTS=1.

Two cleaning variants, both fixed before their pass rates over the subjects were computed.
Variant A is the preset's chain alone; it was fixed first (sub-283, first in alphabetical
order, was used to check that the steps run and to look at its averages; its largest
responses in the N45-N100 window were at FP1 and FP2, the blink reflex, and at T8, the
decay and muscle artifact under the coil, which motivated the channel rule of step 6).
Variant B adds the ICA cleaning of the published TMS-EEG pipelines; it was fixed after
the variant A counts of 10 subjects had been seen with the earlier ``tep`` shapes (gabor
included, removed in S15 part 2: N100 8 of 10, N45 3 of 10), and before any count of the
current preset or of variant B.

1. Epochs ``preset.epoch`` = (-1, 1) s around every ``R128`` marker, read from the
   recording with :func:`evoparam.io.read_bids` (each data file, 0.6 to 2.4 GB, is
   downloaded, read and deleted; the epochs are cached in the evoparam data folder as
   float32 after step 2, about 120 MB per subject, and the averages after step 5). Trials
   removed as the dataset README asks (sub-767: trials 1 and 2; sub-774: after 556;
   sub-780: 401 to 405, counted over tmseeg1 then tmseeg2), and epochs that contain a
   recording discontinuity (a ``New Segment`` marker) are dropped.
2. ``remove_artifact(method="interpolate", window=preset.artifact_window)`` at 5 000 Hz
   (linear, the package default), then :func:`evoparam.prep.resample` to 1 000 Hz.
   Variant B only, the cleaning of TESA (Rogasch et al. 2017, [LIT-artifact-9]),
   re-implemented here from the paper with its default automatic thresholds:
   2a. each epoch is demeaned over its whole length (the recording is DC-coupled; this
       step of the published pipeline was missing from the first run of variant B, whose
       ICA then mixed offsets of thousands of microvolts into the data: 5 of 20 subjects,
       a broken run, not a result);
   2b. ICA round 1 (symmetric FastICA, tanh, on the unmasked samples of all trials):
       components whose trial average has a mean absolute value from 11 to 30 ms more than
       8 times its mean over the epoch (TMS-evoked muscle) are removed;
   2c. zero-phase Butterworth band-pass 1-100 Hz (order 4) and a 60 Hz notch (power line
       of the recording site, sidecar ``PowerLineFrequency``);
   2d. ICA round 2: removed are TMS-evoked muscle (as 2b), blinks (mean z-score of the
       mixing weights at FP1 and FP2 above 2.5 in absolute value), horizontal eye movements
       (z-scores at F7 and F8 of opposite signs, both above 2), electrode noise (a z-score
       above 4) and persistent muscle (slope of log power against log frequency from 30 to
       100 Hz, 57-63 Hz left out, above -0.31).
3. ``remove_artifact(method="subspace", window=preset.artifact_window,
   baseline=preset.baseline)``: the second stage of the preset's ``interpolate+subspace``.
4. ``rereference(scheme="car")``, the preset's average reference over the 30 channels.
5. ``baseline_correct(preset.baseline)``, then the mean over trials (no trial rejection).
6. Channel of maximal response: the largest root mean square of the average from the start
   of the N45 window to the end of the N100 window of the preset (38 to 140 ms). Variant A:
   among the 17 channels that are not on the outer ring of the montage (FP1, FP2, F7, F8,
   T7, T8, P7, P8, O1, O2, IZ, TP9, TP10), where the eye, blink-reflex, temporal and neck
   muscle activity dominates a TMS-EEG recording that has had no ICA. Variant B: among all
   30 channels (the literal rule; ICA has removed those sources). The other channel set is
   reported for each variant as well.
7. ``ResponseModel("tep").fit_group`` on that channel's average, with every preset value
   (fit window 12 to 300 ms, baseline, shapes, background, widths, labels).
8. Criterion per subject: the fit has a component labelled ``N45`` (negative, 38 to 55 ms)
   and one labelled ``N100`` (negative, 80 to 140 ms). REQ-275 asks for both on at least
   80 % of the subjects; it is judged on variant B, the published cleaning, and the level
   of the preset's chain alone (variant A) is tested as well.

Preset ``tep`` 1.1 (background ``constant`` and ``reject_cancelling``, instead of the double
exponential background of 1.0): chosen on 8 development subjects (sub-283, which had served
for variant A, and 7 drawn with ``numpy.random.default_rng(275)`` from the 19 others:
sub-739, 749, 759, 764, 766, 767, 774) before the 12 others were fitted with it; the
counts of both groups are kept apart in the CHANGELOG (REQ-275, "Preset tep 1.1"). The
method above (cleaning, channel rule, label windows) is unchanged.

The methods text check (REQ-275, second part) needs no data and always runs.
"""

import dataclasses
import os
import warnings

import numpy as np
import pytest

from evoparam import presets

NETWORK = os.environ.get("EVOPARAM_NETWORK_TESTS") == "1"

DATASET = "ds002094"
SUBJECTS = ["283", "711", "718", "728", "734", "735", "739", "743", "749", "759", "761",
            "764", "766", "767", "768", "774", "779", "780", "783", "786"]
RUNS = {"728": ("tmseeg1", "tmseeg2"), "766": ("tmseeg1", "tmseeg2"),
        "780": ("tmseeg1", "tmseeg2")}
# Trials (1-based, over the runs in order) removed from the EEG by the dataset README.
README_DROP = {"767": lambda n: n in (1, 2), "774": lambda n: n > 556,
               "780": lambda n: 401 <= n <= 405}
OUTER_RING = ("FP1", "FP2", "F7", "F8", "T7", "T8", "P7", "P8", "O1", "O2", "IZ", "TP9",
              "TP10")
SFREQ = 1000.0
CACHE_VERSION = 1
CHUNK = 100

###################################################################################################
###################################################################################################

def _cache_dir():
    from evoparam.io._fetch import data_folder

    folder = os.path.join(data_folder(), "tep-validation", DATASET)
    os.makedirs(folder, exist_ok=True)
    return folder


def _pulses(root, subject):
    """Kept pulses: (task, events.tsv row) in recording order."""

    from evoparam.io import read_bids_events

    preset = presets.get("tep")
    kept, number = [], 0
    for task in RUNS.get(subject, ("tmseeg1",)):
        stem = f"sub-{subject}/eeg/sub-{subject}_task-{task}"
        events = read_bids_events(os.path.join(root, stem + "_events.tsv"))
        breaks = np.array([ev["onset"] for ev in events if ev.get("type") == "New Segment"])
        for row, ev in enumerate(events):
            if ev.get("value") != "R128":
                continue
            number += 1
            if subject in README_DROP and README_DROP[subject](number):
                continue
            lo, hi = ev["onset"] + preset.epoch[0], ev["onset"] + preset.epoch[1]
            if np.any((breaks > lo) & (breaks < hi)):
                continue
            kept.append((task, row))
    return kept


def _download(subject, folder):
    from evoparam.io import fetch_openneuro_file

    files = ["dataset_description.json"]
    for task in RUNS.get(subject, ("tmseeg1",)):
        stem = f"sub-{subject}/eeg/sub-{subject}_task-{task}"
        files += [stem + suffix for suffix in ("_events.tsv", "_channels.tsv", "_eeg.json",
                                               "_eeg.vhdr", "_eeg.vmrk", "_eeg.eeg")]
    return [fetch_openneuro_file(DATASET, f, folder=folder) for f in files]


def load_epochs(subject):
    """Steps 1-2 of the method (cached): epochs after interpolation, at 1 000 Hz."""

    pd = pytest.importorskip("pandas")
    from evoparam.artifact import remove_artifact
    from evoparam.data import TrialSet
    from evoparam.io import read_bids
    from evoparam.prep import resample

    preset = presets.get("tep")
    path = os.path.join(_cache_dir(), f"sub-{subject}_v{CACHE_VERSION}.npz")
    if not os.path.exists(path):
        folder = os.path.join(_cache_dir(), "raw")
        try:
            _download(subject, folder)
            root = os.path.join(folder, DATASET)
            pulses = _pulses(root, subject)
            parts, rows = [], []
            for task in RUNS.get(subject, ("tmseeg1",)):
                mine = [row for t, row in pulses if t == task]
                for start in range(0, len(mine), CHUNK):
                    chunk = mine[start:start + CHUNK]
                    ts = read_bids(root, subject, task=task, datatype="eeg",
                                   epoch=preset.epoch, select=f"trial in {chunk}")
                    assert list(ts.metadata["trial"]) == chunk
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ts = remove_artifact(ts, method="interpolate",
                                             window=preset.artifact_window,
                                             quality=False).cleaned
                    parts.append(resample(ts, SFREQ))
                    rows += [(task, row) for row in chunk]
        finally:
            eeg = os.path.join(folder, DATASET, f"sub-{subject}", "eeg")
            for name in os.listdir(eeg) if os.path.isdir(eeg) else []:
                if name.endswith((".eeg", ".part")):
                    os.remove(os.path.join(eeg, name))
        np.savez_compressed(path, data=np.concatenate([p.data for p in parts]).astype(np.float32),
                            masks=np.concatenate([p.masks for p in parts]),
                            times=parts[0].times, sfreq=parts[0].sfreq,
                            names=np.array(parts[0].ch_names),
                            task=np.array([t for t, _ in rows]),
                            row=np.array([r for _, r in rows]))
    saved = np.load(path)
    return TrialSet(data=saved["data"].astype(float), times=saved["times"],
                    sfreq=float(saved["sfreq"]), ch_names=[str(n) for n in saved["names"]],
                    masks=saved["masks"],
                    metadata=pd.DataFrame({"task": saved["task"].astype(str),
                                           "trial": saved["row"]}))


def _preset_stages(ts):
    """Steps 3-5: subspace stage, average reference, baseline, mean over trials."""

    from evoparam.artifact import remove_artifact
    from evoparam.prep import baseline_correct, rereference

    preset = presets.get("tep")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = remove_artifact(ts, method="subspace", window=preset.artifact_window,
                             baseline=preset.baseline, quality=False).cleaned
        ts = rereference(ts, "car")
        ts = baseline_correct(ts, preset.baseline, check=False)
    return ts.average()


def fastica(x, *, seed=0, max_iter=1000, tol=1e-4):
    """Symmetric FastICA with the tanh non-linearity (Hyvarinen 1999), the algorithm of the
    published TMS-EEG pipelines; ``x``: channels x samples. Returns unmixing (k x channels)
    and mixing (channels x k) matrices for the mean-removed data."""

    x = x - x.mean(axis=1, keepdims=True)
    values, vectors = np.linalg.eigh(x @ x.T / x.shape[1])
    keep = values > values.max() * 1e-10
    whiten = (vectors[:, keep] / np.sqrt(values[keep])).T
    z = whiten @ x
    k = z.shape[0]

    def decorrelate(w):
        vals, vecs = np.linalg.eigh(w @ w.T)
        return (vecs / np.sqrt(vals)) @ vecs.T @ w

    w = decorrelate(np.random.default_rng(seed).standard_normal((k, k)))
    for _ in range(max_iter):
        g = np.tanh(w @ z)
        new = g @ z.T / z.shape[1] - np.diag((1 - g ** 2).mean(axis=1)) @ w
        new = decorrelate(new)
        done = np.max(np.abs(np.abs(np.einsum("ij,ij->i", new, w)) - 1)) < tol
        w = new
        if done:
            break
    unmixing = w @ whiten
    return unmixing, np.linalg.pinv(unmixing)


def _z(column):
    return (column - column.mean()) / column.std()


def ica_round(ts, *, second):
    """One ICA round of the published TESA pipeline (Rogasch et al. 2017, its default
    automatic thresholds), fitted on the unmasked samples of every trial; the components
    classified as artifacts are removed from every sample. Returns the TrialSet and the
    classification counts."""

    names = [n.upper() for n in ts.ch_names]
    n_trials, n_channels, n_times = ts.data.shape
    flat = ts.data.transpose(1, 0, 2).reshape(n_channels, -1)
    used = ~ts.masks.any(axis=1).reshape(-1) if ts.masks is not None else slice(None)
    unmixing, mixing = fastica(flat[:, used])
    mean = flat[:, used].mean(axis=1, keepdims=True)
    act = (unmixing @ (flat - mean)).reshape(-1, n_trials, n_times)
    average = act.mean(axis=1)
    early = (ts.times >= 0.011) & (ts.times <= 0.030)
    ratio = np.abs(average[:, early]).mean(axis=1) / np.abs(average).mean(axis=1)
    reasons = {"tms_muscle": ratio > 8.0}
    if second:
        z = np.array([_z(mixing[:, j]) for j in range(mixing.shape[1])])
        fp = [names.index(c) for c in ("FP1", "FP2")]
        f7, f8 = names.index("F7"), names.index("F8")
        reasons["blink"] = np.abs(z[:, fp].mean(axis=1)) > 2.5
        reasons["eye_movement"] = ((z[:, f7] * z[:, f8] < 0)
                                   & (np.minimum(np.abs(z[:, f7]), np.abs(z[:, f8])) > 2.0))
        reasons["electrode_noise"] = np.abs(z).max(axis=1) > 4.0
        from scipy.signal import periodogram
        freqs, power = periodogram(act, fs=ts.sfreq, window="hann", axis=-1)
        power = power.mean(axis=1)
        band = (freqs >= 30) & (freqs <= 100) & ((freqs < 57) | (freqs > 63))
        slope = np.array([np.polyfit(np.log10(freqs[band]), np.log10(p[band]), 1)[0]
                          for p in power])
        reasons["persistent_muscle"] = slope > -0.31
    remove = np.any(list(reasons.values()), axis=0)
    cleaned = flat - mixing[:, remove] @ (unmixing[remove] @ (flat - mean))
    data = cleaned.reshape(n_channels, n_trials, n_times).transpose(1, 0, 2)
    counts = {key: int(value.sum()) for key, value in reasons.items()}
    counts["removed"] = int(remove.sum())
    return dataclasses.replace(ts, data=data), counts


def _ica_stages(ts):
    """Variant B, steps 2a-2d: demean, ICA round 1, 1-100 Hz band-pass and 60 Hz notch, ICA
    round 2."""

    from evoparam.prep import baseline_correct, highpass, lowpass, notch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = baseline_correct(ts, (ts.times[0], ts.times[-1]), check=False)
    ts, first = ica_round(ts, second=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = notch(lowpass(highpass(ts, 1.0), 100.0), 60.0, harmonics=1)
    ts, second = ica_round(ts, second=True)
    return ts, {"round1": first, "round2": second}


def average(subject, variant="A"):
    """Average of one subject after variant A (preset chain) or B (A plus the published ICA
    cleaning), cached with the epochs."""

    from evoparam.data import Evoked

    path = os.path.join(_cache_dir(), f"sub-{subject}_avg{variant}_v{CACHE_VERSION}.npz")
    if not os.path.exists(path):
        ts = load_epochs(subject)
        info = {}
        if variant == "B":
            ts, info = _ica_stages(ts)
        evoked = _preset_stages(ts)
        np.savez(path, data=evoked.data, sem=evoked.sem, masks=evoked.masks,
                 n_trials=evoked.n_trials, times=evoked.times, sfreq=evoked.sfreq,
                 names=np.array(evoked.ch_names), info=repr(info))
    saved = np.load(path)
    n_trials = saved["n_trials"]
    return Evoked(data=saved["data"], times=saved["times"], sfreq=float(saved["sfreq"]),
                  ch_names=[str(n) for n in saved["names"]],
                  n_trials=int(n_trials) if n_trials.ndim == 0 else n_trials,
                  sem=saved["sem"], masks=saved["masks"]), str(saved["info"])


def judge(evoked, channels=None):
    """Steps 6-8 on one subject: channel, fit, and whether N45 and N100 are labelled."""

    from evoparam.models import ResponseModel

    preset = presets.get("tep")
    names = list(evoked.ch_names)
    if channels is None:
        channels = [n for n in names if n.upper() not in OUTER_RING]
    window = (preset.component_labels["N45"][0], preset.component_labels["N100"][1])
    inside = (evoked.times >= window[0]) & (evoked.times <= window[1])
    rms = {n: float(np.sqrt(np.nanmean(evoked.data[names.index(n), inside] ** 2)))
           for n in channels}
    channel = max(rms, key=rms.get)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = ResponseModel("tep").fit_group(evoked, picks=[channel])
    fit = group.fits[0]
    labels = {c.label for c in fit.components}
    return {"channel": channel, "status": fit.status, "N45": "N45" in labels,
            "N100": "N100" in labels,
            "components": [(c.label, round(c.latency * 1000, 1), round(c.amplitude, 2))
                           for c in fit.components]}


def evaluate(subjects=SUBJECTS):
    """Per subject and variant: the judged fit on the 17 inner channels and on all 30."""

    out = {}
    for subject in subjects:
        out[subject] = {}
        for variant in ("A", "B"):
            evoked, info = average(subject, variant)
            out[subject][variant + "_inner"] = judge(evoked)
            out[subject][variant + "_all"] = judge(evoked, list(evoked.ch_names))
            out[subject][variant + "_info"] = info
        out[subject]["n_trials"] = int(np.max(evoked.n_trials))
    return out


def rates(results, which):
    """Fractions of subjects with N45, with N100 and with both."""

    n = len(results)
    n45 = sum(r[which]["N45"] for r in results.values()) / n
    n100 = sum(r[which]["N100"] for r in results.values()) / n
    both = sum(r[which]["N45"] and r[which]["N100"] for r in results.values()) / n
    return n45, n100, both


@pytest.fixture(scope="module")
def results():
    return evaluate()


###################################################################################################
###################################################################################################

def test_req275_methods_text_names_the_sensory_confound():

    from evoparam.models import methods_text, resolve_settings

    note = resolve_settings("tep").preset_note
    assert "coil click" in note and "scalp sensation" in note
    assert "auditory" in note and "somatosensory" in note
    times = np.arange(-0.2, 0.4, 0.001)
    y = -5 * np.exp(-0.5 * ((times - 0.1) / 0.02) ** 2)
    from evoparam.models import fit_response
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_response(times, y, window=(0.012, 0.3), baseline=(-0.2, 0.0), preset="tep")
    text = methods_text(fit)
    assert "coil click" in text and "somatosensory" in text
    # Other presets carry no TMS note.
    assert "coil click" not in resolve_settings("ccep").preset_note


# Measured on the 20 subjects (ds002094 1.0.0, evoparam 0.1.0.dev0, step S15): subjects with a
# labelled N45 / N100 / both on the channel of maximal response.
#   variant A (preset chain), 17 inner channels:  11 / 13 / 9
#   variant A, all 30 channels:                    8 /  8 / 4
#   variant B (with ICA), all 30 channels:          9 / 11 / 6   <- REQ-275 criterion
#   variant B, 17 inner channels:                   8 / 16 / 6
# Measured again (Python 3.13, numpy 2.5.3), development 8 + held-out 12 = total:
#                         preset tep 1.0 (3b222bb)   preset tep 1.1
#   A, 17 inner channels:  9 / 12 / 8               4/6/3 + 5/9/5 = 9 / 15 / 8
#   A, all 30 channels:    5 /  7 / 3               4/4/2 + 2/5/2 = 6 / 9 / 4
#   B, all 30 channels:    7 / 11 / 5               6/7/6 + 4/4/2 = 10 / 11 / 8  <- criterion
#   B, 17 inner channels:  7 / 14 / 5               6/7/6 + 6/10/6 = 12 / 17 / 12
needs_data = [pytest.mark.network,
              pytest.mark.skipif(not NETWORK, reason="set EVOPARAM_NETWORK_TESTS=1 to read "
                                 "OpenNeuro")]


def _network(test):
    for mark in reversed(needs_data):
        test = mark(test)
    return test


@_network
@pytest.mark.xfail(strict=False, reason="REQ-275 not met: with the published ICA cleaning, "
                   "N45 and N100 are both labelled on the channel of maximal response for 8 of "
                   "20 subjects (N45 10, N100 11; preset tep 1.1; 5 of 20 with tep 1.0), 8 of "
                   "20 with the preset's chain alone and the 17 inner channels. On 10 of the "
                   "20 subjects the largest response after ICA is on TP9, TP10, IZ or O1, where "
                   "the fronto-central N100 appears inverted under the average reference (the "
                   "largest fitted component from 80 to 140 ms is positive on 9 of them): both "
                   "labels on 1 of these 10, on 7 of the other 10; on the 17 inner channels 12 "
                   "of 20. See CHANGELOG, REQ-275 'Preset tep 1.1', and step S15 part 3.")
def test_req275_n45_and_n100_on_the_channel_of_maximal_response(results):

    n45, n100, both = rates(results, "B_all")
    assert both >= 0.8


@_network
@pytest.mark.xfail(strict=True, reason="floor of step S15 not reached (Python 3.13, numpy "
                   "2.5.3, preset tep 1.1): N45 / N100 / both on the 17 inner channels with the "
                   "preset chain 9 / 15 / 8 (floor 11 / 15 / 9; N100 met); with ICA on all 30 "
                   "channels 10 / 11 / 8 and on the 17 inner channels 12 / 17 / 12 (floors "
                   "raised to these levels) are met. Preset tep 1.0 at 3b222bb: 9 / 12 / 8, "
                   "7 / 11 / 5, 7 / 14 / 5 (CHANGELOG).")
def test_req275_level_reached_with_the_preset_chain(results):
    """Floor of the level reached (the counts measured in S15, raised where preset tep 1.1
    reached more): a regression guard, not the criterion of REQ-275."""

    n = len(results)
    assert n == 20
    n45, n100, both = rates(results, "A_inner")
    assert round(n45 * n) >= 11 and round(n100 * n) >= 15 and round(both * n) >= 9
    n45, n100, both = rates(results, "B_all")
    assert round(n45 * n) >= 10 and round(n100 * n) >= 11 and round(both * n) >= 8
    n45, n100, both = rates(results, "B_inner")
    assert round(n45 * n) >= 12 and round(n100 * n) >= 17 and round(both * n) >= 12


def test_fastica_recovers_mixed_sources():

    rng = np.random.default_rng(1)
    t = np.arange(20000) / 1000.0
    sources = np.vstack([np.sign(np.sin(2 * np.pi * 3 * t)), np.sin(2 * np.pi * 7 * t),
                         rng.laplace(size=t.size)])
    mixing = rng.standard_normal((5, 3))
    unmixing, estimated = fastica(mixing @ sources)
    assert unmixing.shape == (3, 5) and estimated.shape == (5, 3)
    recovered = unmixing @ (mixing @ sources)
    corr = np.abs(np.corrcoef(np.vstack([sources, recovered]))[:3, 3:])
    assert np.all(corr.max(axis=1) > 0.99)
