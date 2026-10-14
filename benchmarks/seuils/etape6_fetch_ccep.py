"""Fetch the ds004080 patients of the step-6 benchmarks with the rule of validation/ds004080_subjects.txt.

Rule: subjects in alphabetical order; for each, the first run (file-name order of the OpenNeuro listing) recorded at
2048 Hz with a stimulation pair of >= 10 trials; that run's first such pair. Epochs -2 to 3 s,
good non-stimulated ECoG channels, raw monopolar data (same cache as the N1 validation).
"""

import io
import re
import json
import os
import sys

import pandas as pd

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "evoparam", "tests", "models"
    ),
)
import test_engine_ccep as t  # noqa: E402
from evoparam.io._fetch import fetch_bytes  # noqa: E402

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 30
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds004080_v2_subjects.tsv")
parts = pd.read_csv(io.BytesIO(fetch_bytes(f"{t.OPENNEURO}/participants.tsv")), sep="\t")
subjects = sorted(parts["participant_id"])
done = (
    pd.read_csv(OUT, sep="\t")
    if os.path.exists(OUT)
    else pd.DataFrame(columns=["subject", "run", "pair", "status"])
)
rows = done.to_dict("records")
seen = set(done["subject"])
ok = sum(r["status"] == "ok" for r in rows)
for subject in subjects:
    if ok >= TARGET:
        break
    if subject in seen:
        continue
    status, run, pair = "no qualifying run", "", ""
    try:
        listing = fetch_bytes(
            f"https://s3.amazonaws.com/openneuro.org?list-type=2&prefix=ds004080/{subject}/ses-1/ieeg/"
        ).decode()
        keys = sorted(re.findall(r"<Key>([^<]*SPESclin[^<]*_ieeg\.json)</Key>", listing))
        for key in keys:
            stem = os.path.basename(key).rsplit("_ieeg", 1)[0]
            side = json.loads(fetch_bytes(f"{t.OPENNEURO}/{subject}/ses-1/ieeg/{stem}_ieeg.json"))
            if round(float(side.get("SamplingFrequency", 0))) != 2048:
                continue
            try:
                pair = t._first_pair(subject, stem)
            except StopIteration:
                continue
            run = stem.split("_run-")[1]
            t._pair_trials(subject[len("sub-") :], run, pair)
            status = "ok"
            break
    except Exception as err:  # noqa: BLE001
        status = f"error: {err!r}"[:200]
    rows.append({"subject": subject, "run": run, "pair": pair, "status": status})
    ok += status == "ok"
    pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)
    print(subject, run, pair, status, f"({ok} ok)", flush=True)
print("done", ok)
