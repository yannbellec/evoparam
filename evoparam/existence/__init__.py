# SPDX-License-Identifier: Apache-2.0
"""Existence: is there a response, and how long does it last? (spec §5.6)"""

from .api import (
    HUMAN_KAPPA,
    METHODS,
    attach_test,
    compare_detectors,
    detectors,
    existence_detector,
    fit,
    list_rules,
    methods_text,
    response_duration,
    test_response,
    train_discordance,
)
from .corrections import CORRECTIONS, correct
from .rules import load_rules, resolve_rule
from .sequential import OnlineExistence, SequentialSignFlip, alpha_spending, sign_flip_test

__all__ = [
    "test_response",
    "existence_detector",
    "response_duration",
    "compare_detectors",
    "list_rules",
    "train_discordance",
    "methods_text",
    "detectors",
    "fit",
    "attach_test",
    "correct",
    "load_rules",
    "resolve_rule",
    "sign_flip_test",
    "SequentialSignFlip",
    "OnlineExistence",
    "alpha_spending",
    "METHODS",
    "CORRECTIONS",
    "HUMAN_KAPPA",
]
