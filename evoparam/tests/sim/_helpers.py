# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the tests of evoparam.sim."""

from typing import Any

import numpy as np

from evoparam.data import Provenance, TrialSet
from evoparam.shapes import get_shape

# Small configurations of every generator, fast enough for the test suite.
SMALL = {
    "erp": {"n_trials": 12, "n_channels": 3},
    "ccep": {"n_trials": 6, "n_channels": 14, "p_response": 0.6, "tmax": 0.5},
    "erna": {"n_pulses": 4, "n_trials": 5},
    "tep": {"n_trials": 6, "n_channels": 8, "sfreq": 2000.0},
    "ecap": {"n_trials": 8, "n_channels": 5},
    "animal": {"n_trials": 10, "n_channels": 8},
}


def measured_noise_sd(ts: Any, truth: Any) -> Any:
    """Per-trial SD of the noise on the baseline (t < 0), averaged over trials, per channel.

    The noise is recovered from the data: data minus the noise-free signal and the
    artifact.
    """

    noise = ts.data - truth.signal
    if truth.artifact_data is not None:
        noise = noise - truth.artifact_data
    base = ts.times < 0
    return np.std(noise[:, :, base], axis=-1).mean(axis=0)


def measured_snr(ts: Any, truth: Any, index: Any) -> Any:
    """(measured, requested) SNR of component ``index``, averaged over the trials holding it.

    The peak is measured on the component's curve on a grid 20 times finer than the
    samples, independently of the features stored in the truth.
    """

    comp = truth.components[index]
    sd = measured_noise_sd(ts, truth)[truth.ch_names.index(comp["channel"])]
    fine = np.linspace(ts.times[0], ts.times[-1], 20 * ts.times.size)
    curves = get_shape(comp["shape"]).evaluate(fine, truth.trial_params(index))
    held = truth.present[:, index]
    peak = abs(comp["amplitude"]) * np.max(np.abs(curves), axis=1) * truth.scale[:, index]
    measured = np.mean(peak[held]) / sd
    requested = comp["snr"] * np.mean(truth.scale[held, index])
    return measured, requested


def real_baselines(
    n_trials: Any = 8,
    n_channels: Any = 3,
    sfreq: Any = 500.0,
    tmin: Any = -2.0,
    tmax: Any = 0.5,
    seed: Any = 0,
    dataset: Any = "ds004080",
    units: Any = "uV",
) -> Any:
    """A TrialSet that looks like epochs read from a registry dataset.

    The signal is synthetic (no network in the tests); the provenance names the dataset
    and the BIDS entities, and the metadata and event times are those a reader fills.
    """

    import pandas as pd

    rng = np.random.default_rng(seed)
    times = np.arange(int(round(tmin * sfreq)), int(round(tmax * sfreq))) / sfreq
    data = 15.0 * rng.standard_normal((n_trials, n_channels, times.size))
    data += np.cumsum(rng.standard_normal(data.shape), axis=-1) * 0.5
    meta = pd.DataFrame(
        {
            "trial": np.arange(n_trials, dtype=np.int64),
            "subject": ["ccepAgeUMCU01"] * n_trials,
            "run": ["021448"] * n_trials,
        }
    )
    source = {
        "path": "sub-ccepAgeUMCU01_run-021448_ieeg.vhdr",
        "format": "bids",
        "dataset": dataset,
        "bids": {"subject": "ccepAgeUMCU01", "session": "1", "run": "021448"},
    }
    return TrialSet(
        data=data,
        times=times,
        sfreq=sfreq,
        ch_names=[f"R{i:02d}" for i in range(n_channels)],
        units=units,
        metadata=meta,
        event_times=10.0 + 5.0 * np.arange(n_trials),
        provenance=Provenance(source=source),
    )


def fits_from_truth(
    truth: Any,
    *,
    latency_error: Any = 0.0,
    amplitude_factor: Any = 1.0,
    ci_halfwidth: Any = None,
    extra: Any = None,
    drop: Any = (),
) -> Any:
    """A GroupFit whose components are the true ones (component level), optionally moved.

    ``extra`` maps (channel, condition) to the latencies of false components;
    ``drop`` lists (channel, label) pairs left out of the fits.
    """

    from evoparam.data import Component, GroupFit, ResponseFit

    table = truth.to_dataframe()
    by_condition = len(set(truth.conditions)) > 1
    fits, channels, conditions = [], [], []
    for channel in truth.ch_names:
        for condition in dict.fromkeys(truth.conditions):
            rows = table[(table["channel"] == channel) & (table["condition"] == condition)]
            comps = []
            for row in rows.to_dict("records"):
                if (channel, row["label"]) in drop:
                    continue
                names = [
                    c[6:] for c in table.columns if c.startswith("param_") and np.isfinite(row[c])
                ]
                ci = None
                if ci_halfwidth is not None:
                    centre = row["latency"] + latency_error
                    ci = {"latency": (centre - ci_halfwidth, centre + ci_halfwidth)}
                comps.append(
                    Component(
                        shape=row["shape"],
                        params=[row[f"param_{n}"] for n in names],
                        param_names=names,
                        latency=row["latency"] + latency_error,
                        amplitude=row["amplitude"] * amplitude_factor,
                        polarity=row["polarity"],
                        fwhm=row["fwhm"],
                        onset=row["onset"],
                        offset=row["offset"],
                        area=row["area"] * amplitude_factor,
                        rise_time=row["rise_time"],
                        decay_time=row["decay_time"],
                        symmetry=row["symmetry"],
                        sharpness=row["sharpness"],
                        label=row["label"],
                        ci=ci,
                    )
                )
            for latency in (extra or {}).get((channel, condition), []):
                comps.append(
                    Component(
                        shape="gaussian",
                        params=[latency, 0.004],
                        param_names=["mu", "sigma"],
                        latency=latency,
                        amplitude=1.0,
                        polarity=1,
                        fwhm=0.0094,
                        onset=latency - 0.008,
                        offset=latency + 0.008,
                        area=0.01,
                        rise_time=0.008,
                        decay_time=0.008,
                        symmetry=0.0,
                        sharpness=1.0,
                        label="extra",
                    )
                )
            fits.append(
                ResponseFit(
                    components=comps,
                    background={},
                    gof=None,
                    status="ok",
                    message="",
                    window=(truth.times[0], truth.times[-1]),
                    baseline=(truth.times[0], 0.0),
                    settings={},
                )
            )
            channels.append(channel)
            conditions.append(condition)
    return GroupFit.from_fits(fits, channels, conditions if by_condition else None)
