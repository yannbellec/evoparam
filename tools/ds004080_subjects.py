"""Write validation/ds004080_subjects.txt: the ds004080 subjects of the layer-3 validation,
chosen by the rule of spec §7.3 before any result is seen (REQ-371).

Rule: the first five subjects in alphabetical order of BIDS subject label that have at least
one run with >= 10 trials per stimulation pair at 2 048 Hz, plus sub-ccepAgeUMCU01 (the
example of spec §1). The example is not one of the five (owner decision D14: six subjects,
the first five qualifying subjects other than sub-ccepAgeUMCU01, then sub-ccepAgeUMCU01 with
its first qualifying run). The file records the dataset version and the SHA-256 of
participants.tsv. Changing the list needs a changelog entry stating why.

Run ``python tools/ds004080_subjects.py`` (reads OpenNeuro's S3 bucket: the listing, then the
``_ieeg.json`` and ``_events.tsv`` of each subject's runs until five subjects qualify).
"""

import hashlib
import io
import json
import pathlib
import sys

ROOT = "https://s3.amazonaws.com/openneuro.org/ds004080"
VERSION = "1.2.4"
EXAMPLE = "sub-ccepAgeUMCU01"
N_SUBJECTS = 5
MIN_TRIALS = 10
SFREQ = 2048.0


def _runs(listing, subject):
    return sorted(path[:-len("_events.tsv")] for path in listing
                  if path.startswith(subject + "/") and "/ieeg/" in path
                  and path.endswith("_events.tsv"))


def qualifies(listing, subject, fetch):
    """(True, run) for the first run of the subject meeting the rule, else (False, None)."""

    import pandas as pd

    for stem in _runs(listing, subject):
        sidecar = stem + "_ieeg.json"
        if sidecar not in listing:
            continue
        info = json.loads(fetch(f"{ROOT}/{sidecar}"))
        if abs(float(info.get("SamplingFrequency", 0)) - SFREQ) > 1e-3:
            continue
        events = pd.read_csv(io.BytesIO(fetch(f"{ROOT}/{stem}_events.tsv")), sep="\t")
        stim = events[events["trial_type"].astype(str).str.contains("stimulation")]
        if "electrical_stimulation_site" not in stim:
            continue
        counts = stim["electrical_stimulation_site"].value_counts()
        if len(counts) and counts.max() >= MIN_TRIALS:
            return True, stem.split("/")[-1]
    return False, None


def select(fetch=None, listing=None):
    from evoparam.io._fetch import fetch_bytes
    from evoparam.io.mef import _list_remote

    fetch = fetch or (lambda url: fetch_bytes(url))
    listing = listing if listing is not None else _list_remote(ROOT, None)
    subjects = sorted({path.split("/")[0] for path in listing if path.startswith("sub-")})
    chosen = []
    for subject in subjects:
        if subject == EXAMPLE:
            continue
        ok, run = qualifies(listing, subject, fetch)
        if ok:
            chosen.append((subject, run))
        if len(chosen) == N_SUBJECTS:
            break
    ok, run = qualifies(listing, EXAMPLE, fetch)
    chosen.append((EXAMPLE, run if ok else "example of spec §1"))
    participants = fetch(f"{ROOT}/participants.tsv")
    return chosen, hashlib.sha256(participants).hexdigest()


def main(path="validation/ds004080_subjects.txt"):
    chosen, sha = select()
    lines = [f"# ds004080 v{VERSION}; participants.tsv sha256 {sha}",
             f"# rule: first {N_SUBJECTS} subjects (alphabetical) with a run of >= {MIN_TRIALS} "
             f"trials per stimulation pair at {SFREQ:g} Hz other than {EXAMPLE}, plus {EXAMPLE} "
             "(spec §7.3; six subjects, owner decision D14)",
             "# subject\tfirst qualifying run"]
    lines += [f"{subject}\t{run}" for subject, run in chosen]
    pathlib.Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return chosen


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
