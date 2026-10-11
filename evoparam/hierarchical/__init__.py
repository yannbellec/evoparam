# SPDX-License-Identifier: Apache-2.0
"""Hierarchical models: trial -> channel -> subject with uncertainty (spec §5.8)."""

from .api import ENGINES, LEVELS, compare_hierarchical, fit_hierarchical, posterior_summary

__all__ = ["fit_hierarchical", "posterior_summary", "compare_hierarchical", "ENGINES", "LEVELS"]
