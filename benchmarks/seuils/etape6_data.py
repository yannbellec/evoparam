"""Build the per-subject caches of the step-6 benchmarks (version 2).

CCEP (ds004080, the 30 patients of ds004080_v2_subjects.tsv): for every good non-stimulated
channel, the trial average, the plus-minus average (noise without response), and the averages
of the odd and of the even trials; -0.6 to 0.6 s.

ERP CORE (40 subjects x 7 components): the published difference wave, its plus-minus noise,
the difference waves of the odd and of the even trials (split within each bin), and, for the
stimulus-locked components, the single trials of the condition carrying the component at the
measurement channel (for the trial-by-trial benchmark).

TEP (ds002094, 20 subjects): the average after the published cleaning (variant B of the TEP
validation) at the channel of largest response from 38 to 140 ms.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "validation"))
import etape6_common as C  # noqa: E402

SPAN = (-0.6, 0.6)


def plus_minus(trials):
    n = (len(trials) // 2) * 2
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    return (trials[:n] * signs[:, None]).mean(axis=0)


def ccep():
    out_dir = os.path.join(C.WORK, "ccep")
    os.makedirs(out_dir, exist_ok=True)
    table = pd.read_csv(os.path.join(HERE, "ds004080_v2_subjects.tsv"), sep="\t")
    for _, row in table[table["status"] == "ok"].iterrows():
        subject = row["subject"][len("sub-") :]
        path = os.path.join(out_dir, f"{subject}.npz")
        if os.path.exists(path):
            continue
        z = np.load(
            f"{C.CACHE}/n1-validation/ds004080/sub-{subject}_run-{int(row['run']):06d}_{row['pair']}.npz"
        )
        times = z["times"]
        keep = (times >= SPAN[0]) & (times <= SPAN[1])
        data = z["data"][:, :, keep].astype(float)  # trials x channels x times
        good = np.all(np.isfinite(data), axis=(0, 2))
        data = data[:, good]
        names = np.array([str(n) for n in z["names"]])[good]
        trials = data.transpose(1, 0, 2)  # channels x trials x times
        np.savez_compressed(
            path,
            times=times[keep],
            sfreq=float(z["sfreq"]),
            names=names,
            n_trials=data.shape[0],
            avg=trials.mean(axis=1),
            pm=np.stack([plus_minus(tr) for tr in trials]),
            odd=trials[:, 0::2].mean(axis=1),
            even=trials[:, 1::2].mean(axis=1),
        )
        print("ccep", subject, data.shape, flush=True)


def erp():
    import erpcore

    out_dir = os.path.join(C.WORK, "erp")
    os.makedirs(out_dir, exist_ok=True)
    for comp, design in erpcore.DESIGNS.items():
        folders = sorted(
            int(os.path.basename(p))
            for p in glob.glob(os.path.join(erpcore.ROOT, comp, "*"))
            if os.path.basename(p).isdigit()
        )
        for part in folders:
            path = os.path.join(out_dir, f"{comp}_{part}.npz")
            if os.path.exists(path):
                continue
            try:
                times, data, names, bins = erpcore.load(comp, part)
            except (FileNotFoundError, ValueError):
                continue
            wave, n = erpcore.difference_wave(comp, times, data, names, bins)
            if not np.all(np.isfinite(wave)):
                continue
            ref = erpcore.noise_reference(comp, data, names, bins)
            # odd / even trials, counted within each bin, so both halves keep the design
            halves = []
            for parity in (0, 1):
                keep = np.zeros(len(bins), bool)
                for b in {b for s in bins for b in s}:
                    rows = [i for i, s in enumerate(bins) if b in s]
                    keep[rows[parity::2]] = True
                sub = [bins[i] for i in np.flatnonzero(keep)]
                w, _ = erpcore.difference_wave(comp, times, data[keep], names, sub)
                halves.append(w)
            extra = {}
            if "plus" in design:
                rows = [i for i, s in enumerate(bins) if design["plus"] in s]
                extra["trials"] = data[rows][:, names.index(design["channel"])].astype(np.float32)
            np.savez_compressed(
                path,
                times=times,
                wave=wave,
                n=n,
                ref=ref if ref is not None else np.full_like(wave, np.nan),
                odd=halves[0],
                even=halves[1],
                **extra,
            )
            print("erp", comp, part, flush=True)


def tep():
    out_dir = os.path.join(C.WORK, "tep")
    os.makedirs(out_dir, exist_ok=True)
    for src in sorted(glob.glob(f"{C.CACHE}/tep-validation/ds002094/sub-*_avgB_v1.npz")):
        subject = os.path.basename(src).split("_")[0]
        z = np.load(src)
        times, data = z["times"], z["data"]
        inside = (times >= 0.038) & (times <= 0.140)
        ch = int(np.argmax(np.sqrt(np.nanmean(data[:, inside] ** 2, axis=1))))
        np.savez_compressed(
            os.path.join(out_dir, f"{subject}.npz"),
            times=times,
            wave=data[ch].astype(float),
            channel=str(z["names"][ch]),
            n_trials=int(z["n_trials"]),
        )
    print("tep done", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["ccep", "erp", "tep"]:
        globals()[name]()
