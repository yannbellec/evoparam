"""Methods paragraphs from provenance (spec §5.13, report 1; REQ-338).

The text is built from ``provenance.steps`` and the result's own settings only, with frozen
templates, so that it is deterministic and a saved result gives the same paragraph.
"""

__all__ = ["methods_text", "citation"]


def _ms(window):
    if window is None:
        return "none"
    return f"{float(window[0]) * 1e3:g} to {float(window[1]) * 1e3:g} ms"


def _read(step):
    s = step["settings"]
    what = s.get("root") or s.get("path") or s.get("source") or "the recording"
    return f"Data were read from {what} with {step['function'].rsplit('.', 1)[-1]}."


def _artifact(step):
    s = step["settings"]
    extra = ""
    if s.get("method") == "template":
        extra = f" (template '{s.get('template', 'global')}', grouped by {s.get('group_by')})"
    elif s.get("method") == "interpolate":
        extra = f" ({s.get('interpolation', 'linear')} interpolation)"
    return (f"The stimulation artifact was removed with the '{s.get('method')}' method over "
            f"{_ms(s.get('window'))}{extra}; replaced samples were masked.")


def _baseline(step):
    s = step["settings"]
    return f"Each trial was baseline-corrected ({s.get('mode', 'mean')} of {_ms(s.get('baseline'))})."


def _reference(step):
    s = step["settings"]
    return f"Channels were re-referenced ('{s.get('scheme')}' scheme)."


def _average(step):
    s = step["settings"]
    robust = s.get("robust")
    how = f"a {robust} average" if robust else "the mean"
    n = s.get("n_trials")
    return f"Trials were averaged ({how}{f', {n} trials' if n is not None else ''})."


def _events(step):
    s = step["settings"]
    return (f"Stimulation pulses were detected from the signal "
            f"({', '.join(f'{k} = {v}' for k, v in sorted(s.items()) if v is not None)}).")


TEMPLATES = {
    "remove_artifact": _artifact,
    "baseline_correct": _baseline,
    "rereference": _reference,
    "average": _average,
    "detect_pulses": _events,
    "detect_events": _events,
}
SKIPPED = {"select", "crop", "copy", "fit_waveform", "fit_group", "fit", "fit_trials",
           "test_response", "tf_fit", "fit_hierarchical", "migrate"}


def _generic(step):
    s = step["settings"]
    shown = ", ".join(f"{k} = {v}" for k, v in sorted(s.items())
                      if v is not None and not isinstance(v, (dict, list)))
    return f"{step['function']} was applied{f' ({shown})' if shown else ''}."


def _step_sentence(step):
    name = step["function"].rsplit(".", 1)[-1]
    if name in SKIPPED:
        return None
    if name.startswith("read"):
        return _read(step)
    return TEMPLATES.get(name, _generic)(step)


def _result_paragraph(result):
    """The paragraph of the final step, from the module that made the result."""

    from evoparam.data import GroupFit, ResponseFit, ResponseTest, TFFit

    if isinstance(result, GroupFit):
        fits = [f for f in result.fits if f.status == "ok"] or result.fits
        if not fits:
            return ""
        from evoparam.models import methods_text as model_text

        text = model_text(fits[0])
        statuses = [f.status for f in result.fits]
        counts = ", ".join(f"{statuses.count(s)} {s}" for s in sorted(set(statuses)))
        return f"{text} Channels: {len(statuses)} ({counts})."
    if isinstance(result, ResponseFit):
        from evoparam.models import methods_text as model_text

        return model_text(result)
    if isinstance(result, ResponseTest):
        from evoparam.existence import methods_text as test_text

        return test_text(result)
    if isinstance(result, TFFit):
        return result.methods_text()
    if hasattr(result, "methods_text"):
        try:
            return result.methods_text()
        except Exception:  # a result without the settings its text needs
            return ""
    return ""


def citation(bibtex=False):
    """The package citation, read from ``CITATION.cff`` (the single source, REQ-346)."""

    from importlib import resources

    text = resources.files("evoparam").joinpath("CITATION.cff").read_text("utf-8")
    fields, authors, current = {}, [], None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- family-names:"):
            current = {"family": stripped.split(":", 1)[1].strip().strip('"')}
            authors.append(current)
        elif stripped.startswith("given-names:") and current is not None:
            current["given"] = stripped.split(":", 1)[1].strip().strip('"')
        elif ":" in line and not line.startswith(" "):
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip().strip('"')
    names = ", ".join(f"{a.get('family', '')} {a.get('given', '')[:1]}".strip() for a in authors)
    version = fields.get("version", "")
    title = fields.get("title", "evoparam")
    doi = fields.get("doi")
    url = fields.get("repository-code", "")
    if bibtex:
        key = (authors[0].get("family", "evoparam") if authors else "evoparam").lower()
        people = " and ".join(f"{a.get('family', '')}, {a.get('given', '')}" for a in authors)
        lines = [f"@software{{{key}_evoparam,", f"  title = {{{title}}},",
                 f"  author = {{{people}}},", f"  version = {{{version}}},"]
        if doi:
            lines.append(f"  doi = {{{doi}}},")
        if url:
            lines.append(f"  url = {{{url}}},")
        lines.append("}")
        return "\n".join(lines)
    out = f"{names}. {title} (version {version})."
    if doi:
        out += f" https://doi.org/{doi}"
    elif url:
        out += f" {url}"
    return out


def methods_text(result, *, cite=True):
    """Methods text of a result: one sentence per recorded processing step, the paragraph
    of the final step, the seed, and the citation."""

    provenance = getattr(result, "provenance", None)
    steps = list(getattr(provenance, "steps", []) or [])
    sentences = [s for s in (_step_sentence(step) for step in steps) if s]
    final = _result_paragraph(result)
    parts = sentences + ([final] if final else [])
    seed = getattr(provenance, "random_state", None)
    if seed is not None:
        parts.append(f"Random seed: {seed}.")
    if cite:
        parts.append(f"Software: {citation()}")
    return " ".join(parts)

