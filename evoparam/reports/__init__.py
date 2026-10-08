"""Reports: methods text, machine-readable JSON, quality summary, HTML, Markdown and PDF
(spec §5.13).

>>> from evoparam import reports
>>> "evoparam" in reports.citation()
True
"""

from ._documents import RESEARCH_USE, html, markdown, pdf, qc
from ._export import from_json, schema, to_json, validate_json
from ._reproduce import ReproduceReport, reproduce
from ._text import citation, methods_text

__all__ = ["methods_text", "citation", "to_json", "from_json", "validate_json", "schema",
           "qc", "html", "markdown", "pdf", "RESEARCH_USE", "reproduce", "ReproduceReport"]
