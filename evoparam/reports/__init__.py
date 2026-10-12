# SPDX-License-Identifier: Apache-2.0
"""Reports: methods text, machine-readable JSON, quality summary, HTML, Markdown and PDF,
reproduction of a saved result, and the preset pipeline (spec §5.13, §9.12).

>>> from evoparam import reports
>>> "evoparam" in reports.citation()
True
"""

from ._documents import RESEARCH_USE, html, markdown, pdf, qc
from ._export import from_json, schema, to_json, validate_json
from ._reproduce import ReproduceReport, migrate, reproduce
from ._text import citation, methods_text
from .pipeline import (
    REFERENCE_PIPELINES,
    PipelineResult,
    reference_input,
    run_pipeline,
    run_reference_pipeline,
)

__all__ = [
    "methods_text",
    "citation",
    "to_json",
    "from_json",
    "validate_json",
    "schema",
    "qc",
    "html",
    "markdown",
    "pdf",
    "RESEARCH_USE",
    "reproduce",
    "ReproduceReport",
    "migrate",
    "run_pipeline",
    "PipelineResult",
    "REFERENCE_PIPELINES",
    "reference_input",
    "run_reference_pipeline",
]
