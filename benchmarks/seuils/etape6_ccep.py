"""Step-6 sweeps on CCEP (ds004080, 30 patients): one setting at a time, preset ``ccep``.

For every value of a setting and every patient:

- 60 semi-simulated waves: 0 to 3 components of known shape, latency, width and amplitude
  (and, in half of them, an exponential background from 9 ms) added to the plus-minus average
  of a random channel; amplitudes are 2 to 20 times the SD of that noise (log-uniform);
- the plus-minus average of every channel (real noise, no response: all components are false);
- the trial average of every channel, and the averages of its odd and of its even trials
  (real data: number of components, N1 present, agreement of the two halves).

The records (one per wave) are written to ~/.cache/evoparam/seuils/results/ccep_<setting>.pkl.
"""

import functools
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import etape6_common as C  # noqa: E402

PRESET = "ccep"
BASELINE = (-0.5, -0.02)
FIT_WINDOW = (0.009, 0.5)
N_SYNTH = 60
TEMPLATES = (  # (latency range s, FWHM range s, sign): N1, a positive deflection, N2
    ((0.015, 0.045), (0.006, 0.030), -1),
    ((0.045, 0.090), (0.015, 0.050), 1),
    ((0.100, 0.300), (0.040, 0.200), -1),
)
N1_WINDOW = (0.009, 0.05)

SWEEPS = {
    "shapes": (
        [("gaussian",), ("skewed_gaussian",), ("skewed_gaussian", "gaussian")],
        2,
        lambda v: {"shapes": v},
    ),
    "background": (["none", "constant", "linear", "exponential"], 3, lambda v: {"background": v}),
    "max_components": (list(range(1, 11)), 5, lambda v: {"max_components": v}),
    "max_fwhm": (
        [0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
        6,
        lambda v: {"max_fwhm": v},
    ),
    "delta": (
        [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            6.0,
            8.0,
            10.0,
            12.0,
            15.0,
            20.0,
            25.0,
            30.0,
            40.0,
            60.0,
            "calibrated",
        ],
        7,
        lambda v: {"delta": v},
    ),
    "detection_threshold": (
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
        3,
        lambda v: {"detection_threshold": v},
    ),
    "overlap_c": (
        [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0],
        4,
        lambda v: {"overlap_c": v},
    ),
    "reject_cancelling": ([False, True], 0, lambda v: {"reject_cancelling": v}),
}


def subjects():
    import glob

    return sorted(
        os.path.basename(p)[:-4] for p in glob.glob(os.path.join(C.WORK, "ccep", "*.npz"))
    )


@functools.lru_cache(maxsize=4)
def load(subject):
    z = np.load(os.path.join(C.WORK, "ccep", f"{subject}.npz"))
    return {k: z[k] for k in z.files}


@functools.lru_cache(maxsize=4)
def synthetic(subject):
    """The N_SYNTH semi-simulated waves of a patient (identical for every setting)."""
    d = load(subject)
    t = d["times"]
    rng = np.random.default_rng(_seed(subject))
    win = (t >= FIT_WINDOW[0]) & (t <= FIT_WINDOW[1])
    out = []
    for _ in range(N_SYNTH):
        ch = int(rng.integers(len(d["names"])))
        noise = d["pm"][ch]
        sd = float(np.std(noise[win]))
        k = int(rng.integers(0, 4))
        which = sorted(rng.choice(len(TEMPLATES), size=k, replace=False))
        y = noise.copy()
        truth = []
        for j in which:
            lat, fw, sign = TEMPLATES[j]
            comp = C.draw_component(rng, lat, fw, sign)
            amp = sd * float(np.exp(rng.uniform(np.log(2.0), np.log(20.0))))
            y += sign * amp * C.curve_of(comp)(t)
            f = C.true_features(comp)
            truth.append({**f, "sign": sign, "amplitude": amp, "template": j})
        background = rng.random() < 0.5
        if background:
            a = rng.choice([-1, 1]) * rng.uniform(2.0, 10.0) * sd
            tau = rng.uniform(0.010, 0.050)
            y += np.where(t >= FIT_WINDOW[0], a * np.exp(-(t - FIT_WINDOW[0]) / tau), 0.0)
        out.append({"y": y, "truth": truth, "channel": ch, "background": background, "sd": sd})
    return out


def _seed(subject):
    import zlib

    return zlib.crc32(f"ccep/{subject}".encode())


def has_n1(found):
    return any(N1_WINDOW[0] <= c["latency"] <= N1_WINDOW[1] and c["amplitude"] < 0 for c in found)


def n1_latency(found):
    lats = [
        c["latency"]
        for c in found
        if N1_WINDOW[0] <= c["latency"] <= N1_WINDOW[1] and c["amplitude"] < 0
    ]
    return min(lats) if lats else np.nan


def run_task(task):
    sweep, k, subject = task
    values, _, make = SWEEPS[sweep]
    ov = make(values[k])
    d = load(subject)
    t = d["times"]
    recs = []
    for i, s in enumerate(synthetic(subject)):
        r = C.fit(t, s["y"], PRESET, ov, baseline=BASELINE)
        found = [c for c in r["components"] if FIT_WINDOW[0] <= c["latency"] <= FIT_WINDOW[1]]
        sc = C.score_wave(s["truth"], found)
        recs.append(
            {
                "kind": "synth",
                "i": i,
                "background": s["background"],
                "error": "error" in r,
                "seconds": r["seconds"],
                **sc,
            }
        )
    for ch in range(len(d["names"])):
        r = C.fit(t, d["pm"][ch], PRESET, ov, baseline=BASELINE)
        recs.append(
            {
                "kind": "null",
                "i": ch,
                "n_found": len(r["components"]),
                "n1": has_n1(r["components"]),
                "error": "error" in r,
                "seconds": r["seconds"],
            }
        )
        r = C.fit(t, d["avg"][ch], PRESET, ov, baseline=BASELINE)
        ro = C.fit(t, d["odd"][ch], PRESET, ov, baseline=BASELINE)
        re_ = C.fit(t, d["even"][ch], PRESET, ov, baseline=BASELINE)
        recs.append(
            {
                "kind": "real",
                "i": ch,
                "n_found": len(r["components"]),
                "n1": has_n1(r["components"]),
                "r2": r.get("r2", np.nan),
                "error": "error" in r,
                "seconds": r["seconds"],
                "n1_odd": has_n1(ro["components"]),
                "n1_even": has_n1(re_["components"]),
                "lat_odd": n1_latency(ro["components"]),
                "lat_even": n1_latency(re_["components"]),
                "n_odd": len(ro["components"]),
                "n_even": len(re_["components"]),
            }
        )
    for r in recs:
        r.update({"sweep": sweep, "value_index": k, "subject": subject})
    return recs


def main(sweeps):
    os.makedirs(os.path.join(C.WORK, "results"), exist_ok=True)
    subs = subjects()
    with Pool(9) as pool:
        for sweep in sweeps:
            path = os.path.join(C.WORK, "results", f"ccep_{sweep}.pkl")
            if os.path.exists(path):
                continue
            values = SWEEPS[sweep][0]
            tasks = [(sweep, k, s) for s in subs for k in range(len(values))]
            recs = [r for chunk in pool.imap_unordered(run_task, tasks) for r in chunk]
            with open(path, "wb") as f:
                pickle.dump({"values": values, "proposed": SWEEPS[sweep][1], "records": recs}, f)
            print("ccep", sweep, len(recs), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(SWEEPS))
