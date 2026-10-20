"""How k is decided: known N1s injected into real pre-stimulus noise of every ds004080 patient.

For each patient, the -2..0 s part of every trial (before the pulse: no response, no
artifact) is relabelled as a -1..+1 s epoch. Half of the contacts (fixed seed) receive a
known negative N1 (Gaussian, latency 15-45 ms, FWHM 5-15 ms, peak 20-500 uV log-uniform,
+-2 ms jitter per trial); the other half stay pure noise. The registry reading of the
published rule (gating reading, D8) is applied for every k. Output: per patient and k, the
% of injected N1s found (sensitivity) and the % of noise-only contacts flagged (false
positives), plus the per-contact table for sensitivity by amplitude.
"""
import os
import sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evoparam", "tests", "models"))
import test_engine_ccep as t
from evoparam.existence.rules import apply_rule, resolve_rule
os.chdir(os.path.dirname(os.path.abspath(__file__)))

KS = np.round(np.arange(1.0, 10.01, 0.1), 2)
BASELINE = (-1.0, -0.1)
rng = np.random.default_rng(20261020)
subs = pd.read_csv("ds004080_subjects.tsv", sep="\t", dtype=str)
subs = subs[subs.status == "ok"]
rows, contacts = [], []
for s in subs.itertuples():
    ts = t._pair_trials(s.subject[4:], s.run, s.pair)
    keep = ts.times < 0
    x = np.asarray(ts.data)[..., keep] * 1.0
    times = ts.times[keep] + 1.0  # -2..0 s becomes -1..+1 s
    n_tr, n_ch, _ = x.shape
    inj = np.zeros(n_ch, bool)
    inj[rng.permutation(n_ch)[: n_ch // 2]] = True
    lat = rng.uniform(0.015, 0.045, n_ch)
    sigma = rng.uniform(0.005, 0.015, n_ch) / 2.3548
    amp = np.exp(rng.uniform(np.log(20), np.log(500), n_ch))
    for c in np.flatnonzero(inj):
        jit = rng.normal(0, 0.002, n_tr)[:, None]
        x[:, c] -= amp[c] * np.exp(-0.5 * ((times[None] - lat[c] - jit) / sigma[c]) ** 2)
    base = (times > BASELINE[0]) & (times < BASELINE[1])
    x -= np.median(x[..., base], axis=-1, keepdims=True)
    avg = x.mean(axis=0)
    avg -= np.median(avg[:, base], axis=-1, keepdims=True)
    sd = avg[:, base].std(axis=1)
    _, _, over = t.FRAMES["paper"]
    for k in KS:
        spec = resolve_rule(t.RULE_NAMES["paper"], baseline=BASELINE, **{**over, "k": float(k), "window": (t.N1_START, t.N1_END)})
        sig = np.asarray(apply_rule(avg, times, ts.sfreq, spec, "uV")["significant"], bool)
        rows.append(dict(subject=s.subject, k=k, n_injected=int(inj.sum()), n_noise=int((~inj).sum()),
                         sensitivity=100 * sig[inj].mean(), false_positive=100 * sig[~inj].mean()))
        contacts += [dict(subject=s.subject, k=k, contact=c, injected=bool(inj[c]), amplitude_uV=amp[c],
                          snr=amp[c] / sd[c], found=bool(sig[c])) for c in range(n_ch)]
    print(s.subject, n_ch, "contacts", n_tr, "trials", flush=True)
pd.DataFrame(rows).to_csv("plotK_calibration.csv", index=False)
pd.DataFrame(contacts).to_csv("plotK_contacts.csv.gz", index=False)
print("done")
