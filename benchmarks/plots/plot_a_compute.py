"""Plot A data: rule threshold k (x SD of baseline) -> % of contacts with an N1, per patient."""
import os
import sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evoparam", "tests", "models"))
import test_engine_ccep as t
from evoparam.existence.rules import apply_rule, resolve_rule
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # inputs and CSVs live next to the script

KS = np.round(np.arange(1.0, 10.01, 0.1), 2)
subs = pd.read_csv("ds004080_subjects.tsv", sep="\t", dtype=str)
subs = subs[subs.status == "ok"]
rows = []
for s in subs.itertuples():
    ts = t._pair_trials(s.subject[4:], s.run, s.pair)
    avg = t._rule_average(ts)
    times = ts.times
    for reading in ("paper", "code"):
        baseline, samples, over = t.FRAMES[reading]
        shift = samples / ts.sfreq
        for k in KS:
            spec = resolve_rule(t.RULE_NAMES[reading], baseline=baseline, **{**over, "k": float(k), "window": (t.N1_START, t.N1_END)})
            out = apply_rule(avg, times + shift, ts.sfreq, spec, "uV")
            sig = np.asarray(out["significant"], bool)
            rows.append(dict(subject=s.subject, reading=reading, k=k, n_channels=len(sig),
                             n_n1=int(sig.sum()), pct=100 * sig.mean(),
                             median_latency_ms=float(np.nanmedian((out["onset"] - shift)[sig]) * 1000) if sig.any() else np.nan))
    print(s.subject, len(ts.ch_names), "channels", len(ts.data), "trials", flush=True)
pd.DataFrame(rows).to_csv("plotA_rule_threshold.csv", index=False)
print("done")
