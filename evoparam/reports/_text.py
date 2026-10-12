# SPDX-License-Identifier: Apache-2.0
"""Methods paragraphs from provenance (spec §5.13, report 1; REQ-338, REQ-340) and the
package citation (REQ-346).

The paragraph is built by :mod:`evoparam.data._methods` (so that ``ResponseFit``,
``GroupFit`` and the other results answer ``.methods_text()`` themselves) from
``provenance.steps`` and the result's settings only, with frozen templates: it is
deterministic, a saved result gives the same paragraph, and it holds no path, date or
duration. The texts of the reference pipelines are frozen in
``evoparam/tests/reports/methods/`` (``tools/freeze_methods_text.py``).
"""

from evoparam.data._citation import citation_bibtex, citation_text, read_citation
from evoparam.data._methods import methods_text as _methods_text

__all__ = ["methods_text", "citation"]


def _provenance(obj):
    """The provenance of a result, or of a table of :mod:`evoparam.stats` (or of the
    reference-sensitivity report), which keeps it in ``attrs["provenance"]``.
    """
    from evoparam.data import Provenance

    found = getattr(obj, "provenance", None)
    if found is not None and not callable(found):
        return found
    attrs = getattr(obj, "attrs", None)
    record = attrs.get("provenance") if isinstance(attrs, dict) else None
    if isinstance(record, Provenance):
        return record
    if isinstance(record, dict):
        return Provenance.from_dict(record)
    return None


class _Merged:
    """Provenance-like view of several results: the steps of the first, then the steps of
    the others that it does not hold.
    """

    def __init__(self, parts):
        from evoparam.data import Provenance

        main = _provenance(parts[0]) or Provenance()
        steps = list(main.steps)
        keys = [(s["function"], repr(s.get("settings"))) for s in steps]
        for part in parts[1:]:
            for step in getattr(_provenance(part), "steps", []) or []:
                key = (step["function"], repr(step.get("settings")))
                if key not in keys:
                    keys.append(key)
                    steps.append(step)
        self.provenance = Provenance(
            package_version=main.package_version,
            created=main.created,
            source=main.source,
            steps=steps,
            environment={},
            random_state=main.random_state,
        )
        settings = getattr(parts[0], "settings", None)
        self.settings = settings if isinstance(settings, dict) else {}


def methods_text(result, *, cite=True):
    """Methods paragraph of a result, of a :class:`PipelineResult` or of a list of results.

    One sentence per processing step recorded in the provenance (events, artifact,
    preparation, fit, existence, trials, hierarchical, time-frequency, statistics), every
    setting changed from the preset, the package version, the random seed, and (with
    ``cite``) the citation of the package and of the published methods used. The tables of
    :mod:`evoparam.stats` (and the reference-sensitivity report of a pipeline) carry their
    step in ``attrs["provenance"]`` and are accepted too.
    """
    if type(result).__name__ == "PipelineResult":
        parts = [
            p
            for p in (
                result.group,
                result.trial_fits,
                *result.artifact,
                getattr(result, "reference_report", None),
            )
            if p is not None
        ]
        return _methods_text(_Merged(parts), cite=cite)
    if isinstance(result, (list, tuple)):
        parts = [p for p in result if p is not None]
        main = [p for p in parts if type(p).__name__ in ("GroupFit", "ResponseFit")]
        ordered = main[:1] + [p for p in parts if p is not (main[0] if main else None)]
        return _methods_text(_Merged(ordered), cite=cite)
    if getattr(result, "provenance", None) is None and _provenance(result) is not None:
        return _methods_text(_Merged([result]), cite=cite)
    return _methods_text(result, cite=cite)


def citation(bibtex=False):
    """The package citation, read from ``CITATION.cff`` (the single source, REQ-346); a
    BibTeX ``@software`` entry with ``bibtex=True``.
    """
    cff = read_citation()
    return citation_bibtex(cff) if bibtex else citation_text(cff)
