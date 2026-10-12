# SPDX-License-Identifier: Apache-2.0
"""Reference values of the bundled fixture fit (REQ-295 (ii), spec §7.4 analysis 5).

``fit_fixture_ccep.json`` holds the fit of a fixed simulated CCEP (``simulate_ccep``,
``seed=0``, 4 channels, 10 trials, no artifact) with the ``ccep`` preset and
``random_state=0``: every component parameter, every ``gof`` field and the background
parameters. It was produced on the reference machine named in the file. The CI matrix
(Linux, macOS, Windows) compares a fresh fit with it at ``rtol=1e-9``
(``tests/test_determinism.py``); ``validation/robustness.py`` does the same (analysis 5).

Regenerate with ``python -m evoparam.tests.reference`` only when a change of the numerical
output is intended and recorded in the CHANGELOG (with the algorithm version bump of
REQ-399).
"""

import json
import os
import platform
import warnings

import numpy as np

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fit_fixture_ccep.json")
RTOL = 1e-9


def fixture_values():
    """{name: float} of the fixture fit (component parameters, gof, background)."""

    from evoparam.models import fit
    from evoparam.sim import simulate_ccep

    ts, _ = simulate_ccep(n_trials=10, n_channels=4, p_response=1.0, seed=0, artifact=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = fit(ts, "ccep", random_state=0)
    out = {}
    for fit_ in group.fits:
        channel = (
            fit_.provenance.source.get("channel")
            if isinstance(getattr(fit_.provenance, "source", None), dict)
            else None
        )
        prefix = channel or str(group.fits.index(fit_))
        for k, value in vars(fit_.gof).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[f"{prefix}/gof/{k}"] = float(value)
        for j, comp in enumerate(fit_.components):
            for k in ("latency", "amplitude", "fwhm", "area", "onset", "offset"):
                out[f"{prefix}/c{j}/{k}"] = float(getattr(comp, k))
            for k, value in zip(comp.param_names, comp.params, strict=False):
                out[f"{prefix}/c{j}/param/{k}"] = float(value)
        out[f"{prefix}/n_components"] = float(len(fit_.components))
    return out


def compare_fixture(path=PATH, rtol=RTOL):
    """Fresh fixture fit against the reference file: match, largest relative difference,
    keys that differ."""

    with open(path, encoding="utf-8") as fobj:
        reference = json.load(fobj)
    values = fixture_values()
    ref = reference["values"]
    differ, worst = [], 0.0
    for key in sorted(set(ref) | set(values)):
        a, b = ref.get(key), values.get(key)
        if a is None or b is None:
            differ.append(key)
            worst = float("inf")
            continue
        if not (np.isfinite(a) and np.isfinite(b)):
            if not (np.isnan(a) and np.isnan(b)) and a != b:
                differ.append(key)
            continue
        rel = abs(a - b) / max(abs(a), 1e-300) if a != b else 0.0
        worst = max(worst, rel)
        if rel > rtol:
            differ.append(key)
    from evoparam.version import ALGORITHM_VERSIONS

    return {
        "match": not differ,
        "max_rel_diff": worst,
        "rtol": rtol,
        "n_values": len(ref),
        "differ": differ[:20],
        "reference_machine": reference.get("machine"),
        "algorithm_versions_changed": reference.get("algorithm_versions")
        != dict(ALGORITHM_VERSIONS),
    }


def write(path=PATH):
    from evoparam.version import ALGORITHM_VERSIONS, __version__

    values = fixture_values()
    doc = {
        "fixture": "simulate_ccep(n_trials=10, n_channels=4, p_response=1.0, seed=0, "
        "artifact=False); evoparam.models.fit(ts, 'ccep', random_state=0)",
        "package_version": __version__,
        "algorithm_versions": dict(ALGORITHM_VERSIONS),
        "machine": f"{platform.platform()} {platform.machine()}, numpy "
        f"{np.__version__}, python {platform.python_version()}",
        "values": {k: (v if np.isfinite(v) else None) for k, v in values.items()},
    }
    with open(path, "w", encoding="utf-8") as fobj:
        json.dump(doc, fobj, indent=1, sort_keys=True)
        fobj.write("\n")
