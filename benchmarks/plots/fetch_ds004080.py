"""Fetch, for every ds004080 participant, the first 2048 Hz SPESclin run with a pair of >= 10 trials."""
import io, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evoparam", "tests", "models"))
import test_engine_ccep as t
from evoparam.io._fetch import fetch_bytes
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # inputs and CSVs live next to the script

OUT = "ds004080_subjects.tsv"
parts = pd.read_csv(io.BytesIO(fetch_bytes(f"{t.OPENNEURO}/participants.tsv")), sep="\t")

def one(subject):
    try:
        listing = fetch_bytes(f"https://s3.amazonaws.com/openneuro.org?list-type=2&prefix=ds004080/{subject}/ses-1/ieeg/").decode()
        for key in sorted(re.findall(r"<Key>([^<]*SPESclin[^<]*_ieeg\.json)</Key>", listing)):
            stem = os.path.basename(key).rsplit("_ieeg", 1)[0]
            side = json.loads(fetch_bytes(f"{t.OPENNEURO}/{subject}/ses-1/ieeg/{stem}_ieeg.json"))
            if round(float(side.get("SamplingFrequency", 0))) != 2048:
                continue
            try:
                pair = t._first_pair(subject, stem)
            except StopIteration:
                continue
            run = stem.split("_run-")[1]
            t._pair_trials(subject[4:], run, pair)
            return dict(subject=subject, run=run, pair=pair, status="ok")
        return dict(subject=subject, run="", pair="", status="no qualifying run")
    except Exception as err:
        return dict(subject=subject, run="", pair="", status=f"error: {err!r}"[:200])

rows = []
with ThreadPoolExecutor(4) as ex:
    for r in ex.map(one, sorted(parts["participant_id"])):
        rows.append(r); print(r["subject"], r["status"], flush=True)
        pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)
print("done", sum(r["status"] == "ok" for r in rows))
