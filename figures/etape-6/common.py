"""Shared loaders and the fit worker of the step-6 threshold figures."""

import glob
import os
import re
import time
import warnings

import numpy as np

CACHE = os.path.expanduser("~/.cache/evoparam")
HERE = os.path.dirname(os.path.abspath(__file__))


def ccep_waves(n_strong=10, n_random=10, seed=0):
    """Trial averages of CCEP runs: (run, channel, times, wave, baseline)."""
    rng = np.random.default_rng(seed)
    out = []
    files = sorted(glob.glob(f"{CACHE}/n1-validation/ds004080/*.npz")) + sorted(
        glob.glob(f"{CACHE}/reference-validation/ds004977/*.npz")
    )
    for path in files:
        z = np.load(path)
        name = os.path.basename(path)[:-4]
        stim = set(re.split(r"[_-]", name)[-2:])
        names = [str(n) for n in z["names"]]
        bad = z["bad"] if "bad" in z.files else np.zeros(len(names), bool)
        times = z["times"]
        avg = np.nanmean(z["data"], axis=0).astype(float)
        base = (max(-0.5, times[0] + 0.01), -0.02)
        inb = (times >= base[0]) & (times <= base[1])
        inr = (times >= 0.01) & (times <= 0.1)
        ok = [
            i
            for i, n in enumerate(names)
            if n not in stim and not bad[i] and np.all(np.isfinite(avg[i]))
        ]
        score = np.array(
            [
                np.sqrt(np.mean((avg[i, inr] - avg[i, inb].mean()) ** 2))
                / (avg[i, inb].std() + 1e-12)
                for i in ok
            ]
        )
        order = np.argsort(score)[::-1]
        strong = [ok[j] for j in order[:n_strong]]
        rest = [ok[j] for j in order[n_strong:]]
        rand = list(rng.choice(rest, size=min(n_random, len(rest)), replace=False))
        subject = name.split("_")[0]
        for i in strong + rand:
            out.append(
                {
                    "run": name,
                    "subject": subject,
                    "channel": names[i],
                    "strong": i in strong,
                    "times": times,
                    "wave": avg[i] - avg[i, inb].mean(),
                    "baseline": base,
                }
            )
    return out


def components(result):
    return [
        {
            "shape": c.shape,
            "latency": float(c.latency),
            "amplitude": float(c.amplitude),
            "fwhm": float(c.fwhm),
            "onset": float(c.onset),
            "offset": float(c.offset),
            "label": c.label,
            "area": float(c.area),
        }
        for c in result.components
    ]


def fit_job(job):
    """job: dict(key, times, wave, preset, overrides, window, baseline, noise_ref, curve)."""
    from evoparam.models import ResponseModel, evaluate_fit

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = ResponseModel(job["preset"], random_state=0, **job.get("overrides", {}))
            res = model.fit_waveform(
                job["times"],
                job["wave"],
                window=job.get("window"),
                baseline=job.get("baseline"),
                noise_ref=job.get("noise_ref"),
            )
        except Exception as err:  # noqa: BLE001
            return {"key": job["key"], "error": repr(err)}
    out = {
        "key": job["key"],
        "status": res.status,
        "components": components(res),
        "seconds": time.perf_counter() - t0,
    }
    if job.get("curve"):
        try:
            out["curve"] = np.asarray(evaluate_fit(res, job["times"]), float)
        except Exception:  # noqa: BLE001
            pass
    if res.gof is not None:
        out["r2"] = float(getattr(res.gof, "r2", np.nan))
    return out
