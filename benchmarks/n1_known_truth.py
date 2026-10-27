# SPDX-License-Identifier: Apache-2.0
"""Known-truth N1 bench (DECISIONS §5.13): fake N1s injected into real ds004080 pre-stimulus noise.

Each usable patient (``n1_known_truth_subjects.tsv``, status "ok") gives the -2..0 s
pre-stimulus trials of one stimulation pair; half of its contacts get a Gaussian N1
(latency 15-45 ms, FWHM 5-15 ms, 20-500 uV log-uniform, trial jitter SD 2 ms). The
published rule (registry ``vanblooijs2023``, k = 3.4, 9-50 ms) and the ``ccep`` preset
(``curve_detection``) are run on every contact. Writes ``results/n1_known_truth.csv`` and
prints the summary of §5.13. Usage: ``python benchmarks/n1_known_truth.py [n_patients]``.
"""

# ruff: noqa: UP031, C408
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from evoparam.data import TrialSet
from evoparam.existence.rules import apply_rule, resolve_rule
from evoparam.models import ResponseModel, curve_detection
from evoparam.tests.models import test_engine_ccep as t

warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 51
BASELINE = (-1.0, -0.1)
rng = np.random.default_rng(20261020)
subs = pd.read_csv(f"{HERE}/n1_known_truth_subjects.tsv", sep="\t", dtype=str)
subs = subs[subs.status == "ok"]
out = []
for s in list(subs.itertuples())[:N]:
    t0 = time.time()
    ts = t._pair_trials(s.subject[4:], s.run, s.pair)
    keep = ts.times < 0
    x = np.asarray(ts.data)[..., keep] * 1.0
    times = ts.times[keep] + 1.0
    n_tr, n_ch, _ = x.shape
    inj = np.zeros(n_ch, bool)
    inj[rng.permutation(n_ch)[: n_ch // 2]] = True
    lat = rng.uniform(0.015, 0.045, n_ch)
    sigma = rng.uniform(0.005, 0.015, n_ch) / 2.3548
    amp = np.exp(rng.uniform(np.log(20), np.log(500), n_ch))
    for c in np.flatnonzero(inj):
        jit = rng.normal(0, 0.002, n_tr)[:, None]
        x[:, c] -= amp[c] * np.exp(-0.5 * ((times[None] - lat[c] - jit) / sigma[c]) ** 2)
    raw = x.copy()
    base = (times > BASELINE[0]) & (times < BASELINE[1])
    x -= np.median(x[..., base], axis=-1, keepdims=True)
    avg = x.mean(axis=0)
    avg -= np.median(avg[:, base], axis=-1, keepdims=True)
    sd = avg[:, base].std(axis=1)
    _, _, over = t.FRAMES["paper"]
    spec = resolve_rule(
        t.RULE_NAMES["paper"],
        baseline=BASELINE,
        **{**over, "k": 3.4, "window": (t.N1_START, t.N1_END)},
    )
    r = apply_rule(avg, times, ts.sfreq, spec, "uV")
    rule = np.asarray(r["significant"], bool)
    rule_lat = np.asarray(r["onset"], float)
    # the rule's score: depth of its deepest strict local minimum in 9-50 ms, in baseline SDs
    w = (times >= t.N1_START) & (times <= t.N1_END)
    a = avg[:, w]
    loc = np.zeros_like(a, bool)
    loc[:, 1:-1] = (a[:, 1:-1] < a[:, :-2]) & (a[:, 1:-1] < a[:, 2:])
    rz = np.where(loc.any(1), -np.where(loc, a, np.inf).min(1) / sd, 0.0)
    tsn = TrialSet(
        data=raw, times=times, sfreq=ts.sfreq, ch_names=ts.ch_names, metadata=ts.metadata
    )
    trials, _ = t._preset_input(tsn)
    fits = ResponseModel("ccep").fit_group(trials).fits
    cd = [curve_detection(f, (0.009, 0.05)) for f in fits]
    for c in range(n_ch):
        out.append(
            dict(
                subject=s.subject,
                contact=c,
                injected=bool(inj[c]),
                latency=lat[c],
                amplitude_uV=amp[c],
                snr=amp[c] / sd[c],
                rule=bool(rule[c]),
                rule_lat=rule_lat[c],
                evoparam=bool(cd[c]["detected"]),
                evoparam_lat=cd[c]["latency"],
                depth=cd[c]["depth"],
                rule_z=rz[c],
            )
        )
    d = pd.DataFrame([o for o in out if o["subject"] == s.subject])
    print(
        s.subject,
        n_ch,
        "sens rule %.0f%% evoparam %.0f%% | FP rule %.1f%% evoparam %.1f%% (%.0fs)"
        % (
            100 * d[d.injected].rule.mean(),
            100 * d[d.injected].evoparam.mean(),
            100 * d[~d.injected].rule.mean(),
            100 * d[~d.injected].evoparam.mean(),
            time.time() - t0,
        ),
        flush=True,
    )
    pd.DataFrame(out).to_csv(f"{HERE}/results/n1_known_truth.csv", index=False)

d = pd.DataFrame(out)
inj = d[d.injected]
no = d[~d.injected]
print("%d patients, %d contacts with an N1, %d without" % (d.subject.nunique(), len(inj), len(no)))
print(
    "defaults: rule %.1f %% found, %.1f %% invented; ccep %.1f %%, %.1f %%"
    % (
        100 * inj.rule.mean(),
        100 * no.rule.mean(),
        100 * inj.evoparam.mean(),
        100 * no.evoparam.mean(),
    )
)
for fp in (0.01, 0.05):
    found = [(inj[c] > np.quantile(no[c], 1 - fp)).mean() for c in ("depth", "rule_z")]
    print(
        "at %.0f %% invented: ccep %.1f %%, rule %.1f %%"
        % (100 * fp, 100 * found[0], 100 * found[1])
    )
