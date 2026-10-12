# SPDX-License-Identifier: Apache-2.0
"""Cancelling pairs: two components of opposite signs at almost the same latency, with large
amplitudes that cancel, describing one deflection (engine defect of build step S7).

Reproduced on clean ECAP simulations (no artifact) with the presets' own simulator: the
``ecap_ci`` preset on contact E01 of ``simulate_ecap(kind="cochlear")`` (seed 0: -313 and
+215 uV at 0.588 and 0.602 ms for a true N1 of -137 uV at 0.489 ms and a P2 of +111 uV at
0.752 ms) and the ``ecap`` preset on contact E02 of ``simulate_ecap(kind="spinal")``.
The engine now refuses a model in which two opposite-sign components overlap (latencies
closer than ``overlap_c`` times the sum of their sigmas, the overlap of spec §5.5.3 step 5):
such a pair is not identifiable.
"""

import warnings

import numpy as np
import pytest

from evoparam.models import fit
from evoparam.shapes import FWHM_PER_SIGMA

CASES = [
    ("cochlear", "ecap_ci", "E01", 0),
    ("cochlear", "ecap_ci", "E01", 3),
    ("cochlear", "ecap_ci", "E01", 4),
    ("spinal", "ecap", "E02", 0),
]


def _case(kind, preset, channel, seed, **settings):
    from evoparam.sim import simulate_ecap

    ts, truth = simulate_ecap(40, kind=kind, artifact=False, seed=seed, tmax=0.0035)
    table = truth.to_dataframe()
    table = table[(table["channel"] == channel) & (table["role"] == "target")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit(ts, preset, picks=[channel], **settings).fits[0]
    return result, table.drop_duplicates("component")


def _cancelling(result, overlap_c=0.5):
    comps = result.components
    for i, a in enumerate(comps):
        for b in comps[i + 1 :]:
            close = abs(a.latency - b.latency) < overlap_c * (a.fwhm + b.fwhm) / FWHM_PER_SIGMA
            if a.polarity != b.polarity and close:
                return True
    return False


@pytest.mark.parametrize(("kind", "preset", "channel", "seed"), CASES)
def test_no_cancelling_pair_on_clean_ecap(kind, preset, channel, seed):
    """No overlapping opposite-sign pair, and no amplitude inflated beyond 1.5 times the
    largest true amplitude of the channel."""

    result, truth = _case(kind, preset, channel, seed)
    assert result.status == "ok"
    assert not _cancelling(result), [(c.latency, c.amplitude) for c in result.components]
    largest = np.max(np.abs(truth["amplitude"]))
    assert all(abs(c.amplitude) <= 1.5 * largest for c in result.components)


def test_the_defect_is_reproduced_without_the_rule():
    """With the rule off (``reject_cancelling=False``), no reseeding (``reseed=False``, the
    engine's default repair, which the ``ecap_ci`` preset runs on every fit) and origin/main's
    free latencies (``latency_order="sorted"``) the defect of the first case comes back: the
    test above detects it. (The ordered parameterisation alone removes this case.)"""

    result, truth = _case(*CASES[0], reject_cancelling=False, reseed=False, latency_order="sorted")
    assert _cancelling(result)
    assert max(abs(c.amplitude) for c in result.components) > 2 * np.max(
        np.abs(truth["amplitude"])
    )
