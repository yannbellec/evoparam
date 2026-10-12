# SPDX-License-Identifier: Apache-2.0
"""Human-readable reports: quality summary, Markdown, self-contained HTML, PDF."""

import base64
import html as _html
import io
import json
from collections import Counter

import numpy as np

from evoparam.data import GroupFit, ResponseFit, ResponseTest, TFFit
from evoparam.version import __version__

from ._text import methods_text

__all__ = ["qc", "markdown", "html", "pdf", "site_pages"]

RESEARCH_USE = (
    "For research use only. evoparam is not a medical device and its outputs are "
    "not intended for diagnosis or treatment decisions."
)


def _fits(result):
    if isinstance(result, GroupFit):
        return [
            (ch, cond, result.fits[int(pos)])
            for ch, cond, pos in result.index[["channel", "condition", "position"]].itertuples(
                index=False
            )
        ]
    if isinstance(result, ResponseFit):
        return [("", "", result)]
    return []


def qc(result, *, trials=None, artifact=None, trial_fits=None):
    """Quality summary: the table a reviewer asks for.

    Rows ``(quantity, value)``: number of fits by status, warnings by type, ill-conditioned
    fits, and, when given, the masked fraction of ``trials`` (TrialSet), the artifact
    quality of ``artifact`` (ArtifactResult: median, minimum, trials below 0.5) and the
    untrusted trials of ``trial_fits`` (TrialFitSet).
    """

    import pandas as pd

    rows = []
    fits = _fits(result)
    if fits:
        statuses = Counter(f.status for _, _, f in fits)
        for status in sorted(statuses):
            rows.append(("fits: " + status, statuses[status]))
        warnings = Counter(w.split(":")[0] for _, _, f in fits for w in f.warnings)
        for name in sorted(warnings):
            rows.append(("warning: " + name, warnings[name]))
        rows.append(
            ("ill-conditioned fits", sum("ill_conditioned" in f.warnings for _, _, f in fits))
        )
    if isinstance(result, ResponseTest):
        rows.append(("significant channels", int(np.sum(result.significant))))
        rows.append(("channels tested", int(len(result.significant))))
    if trials is not None and trials.masks is not None:
        rows.append(("masked fraction", float(trials.masks.mean())))
    if artifact is not None:
        quality = np.asarray(artifact.quality, float)
        summary = artifact.quality_summary()
        rows += [
            ("artifact quality, median", summary["median"]),
            ("artifact quality, 5th percentile", summary["p05"]),
            ("artifact quality, minimum", float(np.nanmin(quality))),
            ("artifact quality below 0.5", int(np.sum(quality < 0.5))),
        ]
    if trial_fits is not None and len(trial_fits.table):
        trusted = trial_fits.table["trusted"].astype(bool)
        rows += [
            ("trials fitted", int(len(trusted))),
            ("trials not trusted", int((~trusted).sum())),
        ]
    table = pd.DataFrame(rows, columns=["quantity", "value"])
    return table


def _summary_table(result):
    import pandas as pd

    if isinstance(result, (GroupFit, ResponseFit)):
        table = result.to_dataframe()
        keep = [
            c
            for c in (
                "channel",
                "condition",
                "label",
                "latency",
                "amplitude",
                "fwhm",
                "status",
                "existence_p",
            )
            if c in table.columns
        ]
        return table[keep]
    if isinstance(result, ResponseTest):
        names = result.settings.get("channels") or list(range(len(result.statistic)))
        return pd.DataFrame(
            {
                "channel": names,
                "statistic": result.statistic,
                "p_value": result.p_value,
                "significant": result.significant,
                "onset": result.onset,
                "offset": result.offset,
            }
        )
    if isinstance(result, TFFit):
        return result.zones if result.zones is not None else pd.DataFrame()
    if hasattr(result, "to_dataframe"):
        return result.to_dataframe()
    return pd.DataFrame()


def _figures(result, data=None, max_channels=24):
    """(title, Figure) pairs for the report."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
    except ImportError:
        return []
    from evoparam import plots

    out = []
    if isinstance(result, GroupFit):
        out.append(("All channels", plots.plot_group(result, data, max_channels=64)))
        for ch, cond, fit in _fits(result)[:max_channels]:
            if fit.components:
                out.append(
                    (
                        f"{ch} {cond}".strip(),
                        plots.plot_fit(fit, data, channel=ch if data is not None else None),
                    )
                )
    elif isinstance(result, ResponseFit):
        out.append(("Fit", plots.plot_fit(result)))
    elif isinstance(result, ResponseTest):
        out.append(("Existence test", plots.plot_existence(result)))
    elif isinstance(result, TFFit):
        out.append(("Time-frequency", plots.plot_tf(result)))
    return out


def _png(fig):
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=90)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def markdown(result, *, data=None):
    """Markdown report: methods text, summary table, quality summary, provenance steps."""

    lines = [
        f"# evoparam report ({type(result).__name__})",
        "",
        RESEARCH_USE,
        "",
        "## Methods",
        "",
        methods_text(result),
        "",
        "## Summary",
        "",
    ]
    table = _summary_table(result)
    lines += [
        table.to_markdown(index=False) if _have_tabulate() else table.to_string(),
        "",
        "## Quality",
        "",
    ]
    quality = qc(result)
    lines += [
        quality.to_markdown(index=False) if _have_tabulate() else quality.to_string(),
        "",
        "## Provenance",
        "",
    ]
    for step in getattr(getattr(result, "provenance", None), "steps", []) or []:
        lines.append(f"- `{step['function']}` (evoparam {step.get('version')})")
    return "\n".join(lines) + "\n"


def _have_tabulate():
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return False
    return True


def html(result, path=None, *, data=None, max_channels=24):
    """One self-contained HTML file (no network access needed: figures are inline PNG):
    summary table, figures, quality summary, warnings, methods text and provenance."""

    table = _summary_table(result)
    quality = qc(result)
    warnings = sorted({w for _, _, f in _fits(result) for w in f.warnings})
    figures = _figures(result, data, max_channels)
    provenance = getattr(result, "provenance", None)
    steps = getattr(provenance, "steps", []) or []
    esc = _html.escape
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>evoparam report: {esc(type(result).__name__)}</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;"
        "padding:0 1em;color:#222}table{border-collapse:collapse;font-size:12px}"
        "td,th{border:1px solid #ccc;padding:3px 6px;text-align:right}"
        "th{background:#f2f2f2}.note{color:#666;font-size:13px}"
        "img{max-width:100%}pre{white-space:pre-wrap;font-size:11px;background:#f7f7f7;"
        "padding:1em}</style></head><body>",
        f"<h1>evoparam report: {esc(type(result).__name__)}</h1>",
        f"<p class='note'>evoparam {esc(__version__)}. {esc(RESEARCH_USE)}</p>",
        "<h2>Methods</h2>",
        f"<p>{esc(methods_text(result))}</p>",
        "<h2>Summary</h2>",
        table.to_html(index=False, float_format=lambda v: f"{v:.4g}", na_rep=""),
        "<h2>Quality</h2>",
        quality.to_html(index=False),
    ]
    if warnings:
        parts += ["<h2>Warnings</h2><ul>"] + [f"<li>{esc(w)}</li>" for w in warnings] + ["</ul>"]
    if figures:
        parts.append("<h2>Figures</h2>")
        for title, fig in figures:
            parts.append(
                f"<h3>{esc(title)}</h3><img alt='{esc(title)}' "
                f"src='data:image/png;base64,{_png(fig)}'>"
            )
    record = {
        "steps": steps,
        "source": getattr(provenance, "source", {}),
        "environment": getattr(provenance, "environment", {}),
    }
    parts += [
        "<h2>Provenance</h2>",
        f"<pre>{esc(json.dumps(record, indent=1, default=str))}</pre>",
        "</body></html>",
    ]
    text = "\n".join(parts)
    if path is None:
        return text
    with open(path, "w", encoding="utf-8") as fobj:
        fobj.write(text)
    return str(path)


def _sites(values, column="stim_site"):
    """Stimulation sites of a TrialSet (its metadata column), in order of appearance."""

    meta = getattr(values, "metadata", None)
    if meta is None or column not in meta.columns:
        return []
    return [
        str(v)
        for v in dict.fromkeys(meta[column].astype(str))
        if v not in ("", "n/a", "nan", "None")
    ]


def site_pages(result, data=None, *, site_column="stim_site"):
    """The pages of :func:`pdf`: ``(site, [(channel, condition, fit position), ...])`` per
    stimulation site of a GroupFit.

    The site of a fit is its condition when the GroupFit was fitted by stimulation site
    (``evoparam.fit(trials, by="stim_site")``, or ``by`` with more columns: the part of the
    label that is a site of ``data``); with a GroupFit fitted on all trials together and
    ``data`` holding one site, that site. A GroupFit that pooled the trials of several
    sites has one page for them all, named so (fit by site for one page per site).
    """

    known = _sites(data, site_column) if data is not None else []
    pages = {}
    for channel, cond, pos in result.index[["channel", "condition", "position"]].itertuples(
        index=False
    ):
        cond = str(cond)
        # the part of the label that is a site of the data, else the whole label (by=site)
        site = next((p for p in cond.split("/") if p in known), cond) if cond else None
        if site is None:
            site = (
                known[0]
                if len(known) == 1
                else ("all stimulation sites pooled" if len(known) > 1 else "all trials")
            )
        pages.setdefault(site, []).append((str(channel), cond, int(pos)))
    return list(pages.items())


def _site_data(data, site, site_column, single_condition):
    """The trials behind one page: the site's trials, or all of them when the fits used all
    trials (one condition); None when they cannot be told apart."""

    if data is None:
        return None
    if site in _sites(data, site_column) and hasattr(data, "select"):
        rows = [i for i, v in enumerate(data.metadata[site_column].astype(str)) if v == site]
        return data.select(trials=rows)
    return data if single_condition else None


def pdf(result, path, *, data=None, max_channels=24, site_column="stim_site"):
    """PDF report (needs matplotlib): a text page (methods, quality), then one page per
    stimulation site (spec §5.13), then the per-channel figures.

    A site page holds every channel of that site: the data (when ``data`` is given) and the
    model, for a :class:`GroupFit` (see :func:`site_pages` for how the site of a fit is
    found); the statistic over time with its null band for a :class:`ResponseTest` split
    by site (``by=("stim_site",)``, the default of ``test_response``).
    """

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from evoparam import plots

    with PdfPages(path) as pages:
        fig = plt.figure(figsize=(8.27, 11.69))
        text = (
            f"evoparam {__version__} report: {type(result).__name__}\n\n{RESEARCH_USE}\n\n"
            f"Methods\n{methods_text(result)}\n\nQuality\n{qc(result).to_string(index=False)}"
        )
        fig.text(0.07, 0.95, _wrap(text, 95), va="top", family="monospace", fontsize=7)
        pages.savefig(fig)
        plt.close(fig)
        if isinstance(result, GroupFit):
            for site, members in site_pages(result, data, site_column=site_column):
                conds = {c for _, c, _ in members}
                # several conditions of one site: one panel per channel and condition
                names = [f"{n} {c}" if len(conds) > 1 else n for n, c, _ in members]
                sub = GroupFit.from_fits([result.fits[pos] for _, _, pos in members], names)
                site_data = (
                    None
                    if len(conds) > 1
                    else _site_data(
                        data, site, site_column, len(set(result.index["condition"])) == 1
                    )
                )
                fig = plots.plot_group(sub, site_data, max_channels=max(64, len(members)))
                fig.suptitle(f"stimulation site: {site}")
                pages.savefig(fig)
                plt.close(fig)
        elif isinstance(result, ResponseTest):
            groups = result.settings.get("groups") or [""]
            for g, name in enumerate(groups):
                fig = plots.plot_existence(result, group=g, max_channels=64)
                fig.suptitle(f"stimulation site: {name or '(all trials)'}", fontsize=9)
                pages.savefig(fig)
                plt.close(fig)
        for title, fig in _figures(result, data, max_channels)[1:]:
            fig.suptitle(title)
            pages.savefig(fig)
            plt.close(fig)
    return str(path)


def _wrap(text, width):
    import textwrap

    out = []
    for paragraph in text.split("\n"):
        out.append(textwrap.fill(paragraph, width) if paragraph else "")
    return "\n".join(out)
