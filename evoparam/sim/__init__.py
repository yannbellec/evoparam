# SPDX-License-Identifier: Apache-2.0
"""Simulators with ground truth, noise models and scorers (spec §5.12).

Six generators (:func:`simulate_erp`, :func:`simulate_ccep`, :func:`simulate_erna`,
:func:`simulate_tep`, :func:`simulate_ecap`, :func:`simulate_animal_ep`) return a
:class:`~evoparam.data.TrialSet` and the :class:`GroundTruth` of what was put in
it; :func:`simulate_bursts` adds induced oscillatory bursts to noise (the
time-frequency task of the benchmark, spec §8.1); :func:`make_noise` draws the
noise models; :func:`gen_response` makes one example response in one line;
:func:`inject` adds components to real recordings;
:func:`match_components` and :func:`score_fits` compare fits with the truth.

This module never imports the fitting engine, so the truth cannot be shaped by
the estimator (REQ-254).
"""

from ._truth import TRUTH_COLUMNS, GroundTruth
from .bursts import burst_waveform, simulate_bursts
from .components import (
    CCEP_DELAYED,
    CCEP_N1,
    CCEP_N2,
    ERP_COMPONENTS,
    ROLES,
    TEP_COMPONENTS,
    ComponentSpec,
    shape_params,
)
from .generators import (
    GENERATORS,
    simulate_animal_ep,
    simulate_ccep,
    simulate_ecap,
    simulate_erna,
    simulate_erp,
    simulate_tep,
)
from .inject import inject
from .noise import ALLOWED_LICENSES, NOISE_KINDS, make_noise
from .score import MATCH_COLUMNS, SCORE_COLUMNS, match_components, score_fits
from .simple import gen_response

__all__ = [
    "gen_response",
    "simulate_erp",
    "simulate_ccep",
    "simulate_erna",
    "simulate_tep",
    "simulate_ecap",
    "simulate_animal_ep",
    "simulate_bursts",
    "burst_waveform",
    "GENERATORS",
    "GroundTruth",
    "TRUTH_COLUMNS",
    "make_noise",
    "NOISE_KINDS",
    "ALLOWED_LICENSES",
    "inject",
    "match_components",
    "score_fits",
    "MATCH_COLUMNS",
    "SCORE_COLUMNS",
    "ComponentSpec",
    "ERP_COMPONENTS",
    "TEP_COMPONENTS",
    "CCEP_N1",
    "CCEP_N2",
    "CCEP_DELAYED",
    "ROLES",
    "shape_params",
]
