"""Minimal 16:9 slides: one per section of "evoparam - first results and decisions to make".

Each slide: the step, what it does, and one card per setting with its possible values; the
starting value is highlighted. Rendered to PNG with headless Chrome.
"""

import html
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Line icons (24 x 24 viewBox, stroke only)
ICONS = {
    "file": '<path d="M6 2h9l5 5v15H6z"/><path d="M15 2v5h5"/><path d="M8 15l2-3 2 4 2-6 2 5"/>',
    "bolt": '<path d="M13 2L4 14h7l-1 8 9-12h-7z"/>',
    "eraser": '<path d="M3 17l9-9 7 7-6 6H8z"/><path d="M8 21h13"/><path d="M9 11l7 7"/>',
    "sliders": '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="M8 12l3 3 5-6"/>',
    "bumps": '<path d="M2 18h20"/><path d="M3 18c2 0 2-10 4-10s2 10 4 10 2-6 3-6 2 6 3 6 2-3 4-3"/>',
    "gauge": '<path d="M4 18a8 8 0 1 1 16 0"/><path d="M12 18l4-6"/><path d="M4 18h16"/>',
    "funnel": '<path d="M3 4h18l-7 8v6l-4 2v-8z"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
    "window": '<path d="M7 4H4v16h3"/><path d="M17 4h3v16h-3"/><path d="M9 12h6"/>',
    "wand": '<path d="M4 20L16 8"/><path d="M15 4v2M19 8h2M18 5l1-1M13 7l4 4"/>',
    "ruler": '<path d="M3 15h18"/><path d="M6 15v-3M10 15v-4M14 15v-3M18 15v-4"/>',
    "wave": '<path d="M2 12c2-6 4-6 6 0s4 6 6 0 4-6 6 0"/>',
    "waves2": '<path d="M2 9c2-4 4-4 6 0s4 4 6 0 4-4 6 0"/><path d="M2 16c2-4 4-4 6 0s4 4 6 0 4-4 6 0" opacity="0.5"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>',
    "trash": '<path d="M4 7h16"/><path d="M6 7l1 13h10l1-13"/><path d="M10 4h4"/><path d="M10 11v6M14 11v6"/>',
    "shield": '<path d="M12 3l8 3v6c0 5-4 8-8 9-4-1-8-4-8-9V6z"/><path d="M12 8v5M12 16v0"/>',
    "shuffle": '<path d="M3 7h4l10 10h4"/><path d="M3 17h4l3-3M14 10l3-3h4"/><path d="M18 4l3 3-3 3M18 14l3 3-3 3"/>',
    "grid": '<rect x="4" y="4" width="16" height="16"/><path d="M4 10h16M4 16h16M10 4v16M16 4v16"/>',
    "flip": '<path d="M12 3v18"/><path d="M8 7l-4 5 4 5z"/><path d="M16 7l4 5-4 5z"/>',
    "book": '<path d="M4 4h7a2 2 0 0 1 2 2v14a2 2 0 0 0-2-2H4z"/><path d="M20 4h-7a2 2 0 0 0-2 2v14a2 2 0 0 1 2-2h7z"/>',
    "bell": '<path d="M2 19h20"/><path d="M4 19c4 0 5-13 8-13s4 13 8 13"/>',
    "slope": '<path d="M3 19h18"/><path d="M3 6c4 0 6 10 18 12"/>',
    "bounds": '<path d="M3 12h18"/><path d="M7 8l-4 4 4 4M17 8l4 4-4 4"/><path d="M12 6v12"/>',
    "scale": '<path d="M12 4v16M6 20h12"/><path d="M4 8h16"/><path d="M4 8l-2 5h4zM20 8l-2 5h4z"/>',
    "search": '<circle cx="10" cy="10" r="6"/><path d="M15 15l6 6"/>',
    "merge": '<path d="M6 3v5c0 4 6 5 6 9v4"/><path d="M18 3v5c0 4-6 5-6 9"/>',
    "plusminus": '<path d="M12 4v8M8 8h8"/><path d="M8 18h8"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/>',
    "flag": '<path d="M5 21V4"/><path d="M5 4h11l-2 4 2 4H5"/>',
    "crosshair": '<circle cx="12" cy="12" r="7"/><path d="M12 2v6M12 16v6M2 12h6M16 12h6"/>',
    "coil": '<circle cx="9" cy="12" r="5"/><circle cx="15" cy="12" r="5"/>',
    "stack": '<rect x="4" y="4" width="14" height="4"/><rect x="6" y="10" width="14" height="4"/><rect x="4" y="16" width="14" height="4"/>',
}


import numpy as np  # noqa: E402

W, H = 660, 210
INK, SOFT, ACC, FILL = "#1f2933", "#c3cad3", "#1f4fd1", "#e8eefc"


def _path(x, y):
    return "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y, strict=True))


def _xy(t, y, lo=-1.0, hi=1.0, pad=14):
    x = pad + (t - t[0]) / (t[-1] - t[0]) * (W - 2 * pad)
    yy = H - pad - (y - lo) / (hi - lo) * (H - 2 * pad)
    return x, yy


def _x(c, pad=14):
    return pad + c * (W - 2 * pad)


def _bump(t, mu, sd):
    return np.exp(-0.5 * ((t - mu) / sd) ** 2)


def schematic(kind):
    rng = np.random.default_rng(3)
    t = np.linspace(0, 1, 600)
    parts = []
    if kind == "pulses":
        y = rng.normal(0, 0.05, t.size)
        for c in (0.12, 0.37, 0.62, 0.87):
            y += 1.6 * np.exp(-np.abs(t - c) / 0.004) * np.sign(t - c + 1e-9)
        y += 0.35 * np.exp(-np.abs(t - 0.5) / 0.003)
        x, yy = _xy(t, y, -1.8, 1.8)
        thr = _xy(np.array([0, 1]), np.array([1.0, 1.0]), -1.8, 1.8)
        parts.append(f'<path d="{_path(x, yy)}" stroke="{INK}" stroke-width="1.6" fill="none"/>')
        parts.append(
            f'<path d="{_path(*thr)}" stroke="{ACC}" stroke-width="2" stroke-dasharray="7 6"/>'
        )
        parts.append(
            f'<text x="{W - 14}" y="{thr[1][0] - 8:.0f}" text-anchor="end" font-size="17" fill="{ACC}">threshold</text>'
        )
        for c in (0.12, 0.37, 0.62, 0.87):
            cx = _x(c)
            parts.append(
                f'<path d="M{cx - 8:.0f},12 L{cx + 8:.0f},12 L{cx:.0f},24 z" fill="{ACC}"/>'
            )
        bx = _x(0.5)
        parts.append(
            f'<text x="{bx:.0f}" y="{H - 8}" text-anchor="middle" font-size="15" fill="#8a94a0">small jump: ignored</text>'
        )
    elif kind == "artifact":
        y = (
            0.55 * _bump(t, 0.42, 0.05) * -1
            + 0.3 * _bump(t, 0.62, 0.1)
            + rng.normal(0, 0.02, t.size)
        )
        art = 3.0 * np.exp(-np.abs(t - 0.25) / 0.012) * np.sign(t - 0.25 + 1e-9)
        x, yy = _xy(t, y + art, -1.6, 1.6)
        a0, a1 = _x(0.23), _x(0.31)
        parts.append(
            f'<rect x="{a0:.0f}" y="8" width="{a1 - a0:.0f}" height="{H - 16}" fill="{FILL}"/>'
        )
        parts.append(f'<path d="{_path(x, yy)}" stroke="{SOFT}" stroke-width="1.6" fill="none"/>')
        keep = (t < 0.23) | (t > 0.31)
        x2, y2 = _xy(t, y, -1.6, 1.6)
        parts.append(f'<path d="{_path(x2, y2)}" stroke="{INK}" stroke-width="1.8" fill="none"/>')
        parts.append(
            f'<text x="{(a0 + a1) / 2:.0f}" y="{H - 10}" text-anchor="middle" font-size="16" fill="{ACC}">masked</text>'
        )
        parts.append(
            f'<text x="{W - 14}" y="30" text-anchor="end" font-size="16" fill="#8a94a0">grey: raw · black: repaired</text>'
        )
    elif kind == "prepare":
        y = rng.normal(0, 0.06, t.size) + 0.7 * _bump(t, 0.6, 0.06) - 0.4 * _bump(t, 0.75, 0.08)
        x, yy = _xy(t, y, -1, 1)
        b0, b1 = _x(0.02), _x(0.38)
        s0 = _xy(np.array([0.0, 1.0]), np.array([0.12, -0.12]), -1, 1)[1]
        parts.append(
            f'<rect x="{b0:.0f}" y="{s0[0]:.0f}" width="{b1 - b0:.0f}" height="{s0[1] - s0[0]:.0f}" fill="{FILL}"/>'
        )
        z = _xy(np.array([0, 1]), np.array([0, 0]), -1, 1)
        parts.append(
            f'<path d="{_path(*z)}" stroke="{SOFT}" stroke-width="1.2" stroke-dasharray="4 5"/>'
        )
        parts.append(f'<path d="{_path(x, yy)}" stroke="{INK}" stroke-width="1.6" fill="none"/>')
        st = _x(0.4)
        parts.append(
            f'<path d="M{st:.0f},16 L{st:.0f},{H - 16}" stroke="{ACC}" stroke-width="2"/>'
        )
        parts.append(f'<text x="{st + 8:.0f}" y="30" font-size="16" fill="{ACC}">stimulus</text>')
        parts.append(
            f'<text x="{b0:.0f}" y="{H - 12}" font-size="16" fill="{ACC}">baseline: zero and noise level</text>'
        )
    elif kind == "test":
        band = 0.22
        y = 0.8 * _bump(t, 0.55, 0.07) + np.convolve(
            rng.normal(0, 0.12, t.size), np.ones(8) / 8, "same"
        )
        lo = _xy(t, np.full(t.size, band), -1, 1)
        hi = _xy(t, np.full(t.size, -band), -1, 1)
        parts.append(
            f'<rect x="14" y="{lo[1][0]:.0f}" width="{W - 28}" height="{hi[1][0] - lo[1][0]:.0f}" fill="#eef0f3"/>'
        )
        x, yy = _xy(t, y, -1, 1)
        parts.append(f'<path d="{_path(x, yy)}" stroke="{INK}" stroke-width="1.8" fill="none"/>')
        for k in range(3):
            n = np.convolve(rng.normal(0, 0.2, t.size), np.ones(8) / 8, "same")
            xn, yn = _xy(t, n, -1, 1)
            parts.append(
                f'<path d="{_path(xn, yn)}" stroke="{SOFT}" stroke-width="1" fill="none"/>'
            )
        py = _xy(np.array([0.0, 1.0]), np.array([0.85, 0.85]), -1, 1)[1][0]
        parts.append(
            f'<text x="{_x(0.64):.0f}" y="{py - 4:.0f}" text-anchor="start" font-size="16" fill="{ACC}">beyond chance → response</text>'
        )
        parts.append(
            f'<text x="18" y="{H - 10}" font-size="16" fill="#8a94a0">grey band: what random sign flips give</text>'
        )
    elif kind == "model":
        bg = 0.9 * np.exp(-t / 0.25) - 0.2
        b1 = -0.9 * _bump(t, 0.2, 0.04)
        b2 = 0.5 * _bump(t, 0.38, 0.06)
        b3 = -0.45 * _bump(t, 0.65, 0.12)
        y = bg + b1 + b2 + b3 + rng.normal(0, 0.025, t.size)
        x, yy = _xy(t, y, -1.2, 1.0)
        parts.append(f'<path d="{_path(x, yy)}" stroke="{INK}" stroke-width="1.8" fill="none"/>')
        xb, ybb = _xy(t, bg, -1.2, 1.0)
        parts.append(
            f'<path d="{_path(xb, ybb)}" stroke="{SOFT}" stroke-width="2" stroke-dasharray="7 6" fill="none"/>'
        )
        for comp in (b1, b2, b3):
            xc, yc = _xy(t, comp - 0.0, -1.2, 1.0)
            parts.append(
                f'<path d="{_path(xc, yc)}" stroke="{ACC}" stroke-width="2" fill="none" opacity="0.85"/>'
            )
        parts.append(
            f'<text x="{W - 14}" y="28" text-anchor="end" font-size="16" fill="#8a94a0">black: signal · grey: background · blue: bumps</text>'
        )
    elif kind == "measures":
        tt = np.linspace(0, 1, 800)
        z = (tt - 0.42) / 0.11
        from math import erf

        yb = np.exp(-0.5 * z**2) * (1 + np.array([erf(2.2 * v / np.sqrt(2)) for v in z]))
        yb = yb / yb.max()
        x, yy = _xy(tt, yb, -0.15, 1.15)
        parts.append(
            f'<path d="{_path(x, yy)} L{x[-1]:.0f},{_xy(tt, np.zeros(tt.size), -0.15, 1.15)[1][0]:.0f} L{x[0]:.0f},{_xy(tt, np.zeros(tt.size), -0.15, 1.15)[1][0]:.0f} z" fill="{FILL}" stroke="none"/>'
        )
        parts.append(f'<path d="{_path(x, yy)}" stroke="{INK}" stroke-width="2" fill="none"/>')
        i_pk = int(np.argmax(yb))
        cum = np.cumsum(yb)
        i_ar = int(np.searchsorted(cum, cum[-1] / 2))
        above = np.flatnonzero(yb >= 0.1)
        marks = [
            (above[0], "onset 10 %", "#8a94a0"),
            (i_pk, "peak", "#8a94a0"),
            (i_ar, "50 % area", ACC),
            (above[-1], "offset 10 %", "#8a94a0"),
        ]
        for k, (i, lab, col) in enumerate(marks):
            xm = x[i]
            parts.append(
                f'<path d="M{xm:.0f},{yy[i]:.0f} L{xm:.0f},{H - 30}" stroke="{col}" stroke-width="1.6" stroke-dasharray="4 4"/>'
            )
            if lab == "peak":
                parts.append(
                    f'<text x="{xm - 6:.0f}" y="{yy[i] - 4:.0f}" text-anchor="end" font-size="15" fill="{col}">peak</text>'
                )
            elif lab == "50 % area":
                parts.append(
                    f'<text x="{xm + 6:.0f}" y="{yy[i] + 14:.0f}" font-size="15" fill="{col}">50 % area</text>'
                )
            else:
                parts.append(
                    f'<text x="{xm:.0f}" y="{H - 8}" text-anchor="middle" font-size="15" fill="{col}">{lab}</text>'
                )
    return f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{"".join(parts)}</svg>'


def icon(name, size=40, color="currentColor"):
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>'
    )


VERDICT = {
    "ok": ("Supported by tests", "#1e7e4f", "#e7f5ee"),
    "q": ("Questioned by tests", "#b45309", "#fdf3e4"),
    "none": ("No measurable effect", "#5b6470", "#eef0f3"),
    "inc": ("Inconclusive (too few subjects)", "#5b6470", "#eef0f3"),
}

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{width:1920px;height:1080px;background:#fff;color:#1f2933;font-family:-apple-system,'Helvetica Neue',Helvetica,Arial,sans-serif;
  padding:70px 90px 60px;display:flex;flex-direction:column}
.kicker{font-size:22px;letter-spacing:.14em;text-transform:uppercase;color:#8a94a0;margin-bottom:12px}
.head{display:flex;align-items:center;gap:26px}
.badge{width:84px;height:84px;border-radius:22px;background:#f1f4f8;display:flex;align-items:center;justify-content:center;color:#1f2933}
h1{font-size:58px;font-weight:650;letter-spacing:-.01em}
.lead{font-size:26px;color:#52606d;margin:18px 0 0;line-height:1.38}
.top{display:flex;gap:50px;align-items:flex-end;margin-bottom:34px}
.topl{flex:1}
.sketch{flex:0 0 660px;border:1.5px solid #eef1f5;border-radius:18px;background:#fbfcfd}
.grid{display:grid;gap:22px;flex:1;align-content:center}
.card{min-height:230px}
.card{border:1.5px solid #e4e8ee;border-radius:20px;padding:24px 26px;display:flex;flex-direction:column;gap:12px}
.ctop{display:flex;align-items:center;gap:14px;color:#1f2933}
.cname{font-size:25px;font-weight:620}
.cdesc{font-size:19px;color:#5b6470;line-height:1.35}
.chips{display:flex;flex-wrap:wrap;gap:9px;margin-top:auto;padding-top:8px}
.chip{font-size:18px;padding:7px 14px;border-radius:999px;border:1.5px solid #d5dbe3;color:#5b6470}
.chip.on{background:#1f4fd1;border-color:#1f4fd1;color:#fff;font-weight:600}
.verdict{font-size:16px;font-weight:600;padding:5px 12px;border-radius:8px;align-self:flex-start}
.foot{display:flex;justify-content:space-between;color:#9aa5b1;font-size:18px;margin-top:22px}
.legend{display:flex;gap:18px;align-items:center}
.dot{display:inline-block;width:14px;height:14px;border-radius:50%;background:#1f4fd1;margin-right:7px;vertical-align:-1px}
.flow{display:flex;align-items:stretch;gap:0;margin-top:90px}
.step{flex:1;border:1.5px solid #e4e8ee;border-radius:22px;padding:30px 24px;display:flex;flex-direction:column;gap:14px;align-items:flex-start}
.num{font-size:20px;color:#8a94a0;font-weight:600}
.sname{font-size:30px;font-weight:640}
.sdesc{font-size:20px;color:#5b6470;line-height:1.35}
.arrow{width:46px;display:flex;align-items:center;justify-content:center;color:#b8c1cc;font-size:34px}
.note{margin-top:70px;font-size:24px;color:#52606d;line-height:1.45;border-left:4px solid #1f4fd1;padding-left:20px}
"""


def card(ic, name, desc, options, start, verdict=None):
    starts = start if isinstance(start, (list, tuple)) else (start,)
    chips = "".join(
        f'<span class="chip{" on" if o in starts else ""}">{html.escape(o)}</span>'
        for o in options
    )
    v = ""
    if verdict:
        txt, fg, bg = VERDICT[verdict[0]]
        detail = f" · {html.escape(verdict[1])}" if len(verdict) > 1 else ""
        v = f'<span class="verdict" style="color:{fg};background:{bg}">{txt}{detail}</span>'
    return (
        f'<div class="card"><div class="ctop">{icon(ic, 34)}<span class="cname">{html.escape(name)}</span></div>'
        f'<div class="cdesc">{html.escape(desc)}</div>{v}<div class="chips">{chips}</div></div>'
    )


def page(kicker, ic, title, lead, cards, cols, footer_extra="", sketch=None):
    body = (
        f'<div class="top"><div class="topl"><div class="kicker">{kicker}</div><div class="head"><div class="badge">{icon(ic, 48)}</div>'
        f'<h1>{html.escape(title)}</h1></div><div class="lead">{html.escape(lead)}</div></div>'
        f'<div class="sketch">{schematic(sketch) if sketch else ""}</div></div>'
        f'<div class="grid" style="grid-template-columns:repeat({cols},1fr)">{"".join(cards)}</div>'
        f'<div class="foot"><span class="legend"><span><span class="dot"></span>starting value</span>{footer_extra}</span>'
        f"<span>evoparam · first results and decisions</span></div>"
    )
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"


def overview():
    steps = [
        ("file", "Read", "Load the recording and its events."),
        ("bolt", "Find the stimulations", "Locate each pulse from the artifact it leaves."),
        ("eraser", "Remove the artifact", "Clean the few ms around each pulse."),
        ("sliders", "Prepare", "Baseline, noise level, reference, bad trials."),
        ("check", "Test for a response", "Is there a response, or only noise?"),
        ("bumps", "Describe the response", "Fit bumps: latency, amplitude, width."),
    ]
    parts = []
    for i, (ic, n, d) in enumerate(steps, 1):
        parts.append(
            f'<div class="step">{icon(ic, 46)}<span class="num">STEP {i}</span><span class="sname">{n}</span>'
            f'<span class="sdesc">{d}</span></div>'
        )
        if i < len(steps):
            parts.append('<div class="arrow">›</div>')
    body = (
        f'<div class="kicker">evoparam · overview</div><div class="head"><div class="badge">{icon("bumps", 48)}</div>'
        f"<h1>The foundations: six steps</h1></div>"
        f'<div class="lead">From a raw recording to a short list of numbers per response. The foundations are in place; '
        f'this is not yet a validated tool.</div><div class="flow">{"".join(parts)}</div>'
        f'<div class="note">Steps 2 to 6 each hold settings that change the result a lot. The next slides list them, '
        f"with the possible values and the value we start from.</div>"
        f'<div class="foot" style="margin-top:auto"><span></span><span>evoparam · first results and decisions</span></div>'
    )
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"


SLIDES = {
    "0-overview": overview,
    "2-find-the-stimulations": lambda: page(
        "Step 2",
        "bolt",
        "Find the stimulations",
        "Each pulse leaves a sharp jump in the signal. We score that jump against the noise and keep the jumps that are large and repeat.",
        [
            card(
                "gauge",
                "Detection threshold",
                "How many times the noise a jump must reach to count as a pulse.",
                ["z > 10", "z > 15", "z > 20", "z > 30"],
                "z > 20",
            ),
            card(
                "funnel",
                "Size gate",
                "A candidate must reach this share of the size of the clear pulses.",
                ["none", "10 %", "25 %", "50 %"],
                "25 %",
            ),
            card(
                "layers",
                "Fallback score",
                "If the median over channels finds nothing, use the channels that move most.",
                ["none", "90th percentile", "maximum"],
                "90th percentile",
            ),
        ],
        3,
        sketch="pulses",
    ),
    "3-remove-the-artifact": lambda: page(
        "Step 3",
        "eraser",
        "Remove the artifact",
        "Right after the pulse the signal is dominated by the stimulation itself. We mask that window and repair it.",
        [
            card(
                "window",
                "Masked window (CCEP)",
                "Time around the electrical pulse that is masked.",
                ["shorter", "−2 to 8 ms", "longer"],
                "−2 to 8 ms",
            ),
            card(
                "window",
                "Masked window (TMS)",
                "Time around the magnetic pulse that is masked.",
                ["shorter", "−2 to 12 ms", "longer"],
                "−2 to 12 ms",
            ),
            card(
                "wand",
                "Repair method",
                "How the masked window is filled or the artifact subtracted.",
                [
                    "set to zero",
                    "interpolation",
                    "template + tail (CCEP)",
                    "interpolation + subspace (TMS)",
                    "ICA",
                ],
                ["template + tail (CCEP)", "interpolation + subspace (TMS)"],
            ),
        ],
        3,
        sketch="artifact",
    ),
    "4-prepare": lambda: page(
        "Step 4",
        "sliders",
        "Prepare",
        "Set the zero, measure the noise, choose the electrical reference and flag bad trials. The noise level drives every later test.",
        [
            card(
                "ruler",
                "Baseline window",
                "Quiet period before the stimulus, used for the zero and the noise.",
                ["shorter", "CCEP −500 to −20 ms", "ERP −200 to 0 ms", "longer"],
                ["CCEP −500 to −20 ms", "ERP −200 to 0 ms"],
            ),
            card(
                "wave",
                "Noise source",
                "Where the noise level is measured.",
                ["baseline only", "auto: baseline or plus-minus"],
                "auto: baseline or plus-minus",
            ),
            card(
                "waves2",
                "Noise model",
                "Noise remembers its past; correct for it or not.",
                ["none", "effective n only", "AR whitening"],
                "AR whitening",
            ),
            card(
                "target",
                "Electrical reference",
                "What every electrode is measured against.",
                ["bipolar", "common average (TEP)", "adaptive common average (CCEP)"],
                ["common average (TEP)", "adaptive common average (CCEP)"],
            ),
            card(
                "trash",
                "Trial rejection",
                "Flag a trial if it is far from the others (z > 5), poor (quality < 0.5) or > 20 % masked; at most 30 % removed.",
                ["softer", "z 5 · 0.5 · 20 % · max 30 %", "stricter"],
                "z 5 · 0.5 · 20 % · max 30 %",
            ),
        ],
        3,
        sketch="prepare",
    ),
    "5-test-for-a-response": lambda: page(
        "Step 5",
        "check",
        "Test for a response",
        "Before describing a response, check that it exists. We build chance by flipping the sign of trials at random and compare.",
        [
            card(
                "shield",
                "False-alarm risk α",
                "Accepted chance of declaring a response that is only noise.",
                ["1 %", "5 %", "10 %"],
                "5 %",
            ),
            card(
                "shuffle",
                "Random sign flips",
                "How many times chance is rebuilt; more gives a more precise p-value.",
                ["1 000", "5 000", "10 000"],
                "1 000",
            ),
            card(
                "grid",
                "Many-electrode correction",
                "Keeps false discoveries low when many electrodes are tested.",
                ["none", "FDR", "Bonferroni"],
                "FDR",
            ),
            card(
                "flip",
                "Projection test",
                "Null of the profile peak; more reliable with few trials.",
                ["half splits", "disjoint pairs", "sign flips (≤ 4 096)"],
                "sign flips (≤ 4 096)",
            ),
            card(
                "book",
                "Classic N1 rule (reference)",
                "Published rule used only for comparison.",
                ["peak > 3.4 SD, 9–100 ms", "other cut-offs"],
                "peak > 3.4 SD, 9–100 ms",
            ),
        ],
        3,
        sketch="test",
    ),
    "6a-describe-the-model": lambda: page(
        "Step 6 · part 1 — the model",
        "bumps",
        "Describe the response",
        "The response is written as a background plus a few bumps. These settings decide which bumps the model may use and how many it keeps.",
        [
            card(
                "bell",
                "Bump shapes",
                "Symmetric bell, skewed bell, or both.",
                ["Gaussian", "skewed", "both"],
                "both",
                ("ok",),
            ),
            card(
                "slope",
                "Background",
                "Slow part under the bumps.",
                ["none", "constant (ERP, TEP)", "linear", "exponential (CCEP)"],
                ["constant (ERP, TEP)", "exponential (CCEP)"],
                ("ok", "tested on CCEP"),
            ),
            card(
                "bounds",
                "Width and number limits",
                "Widest bump allowed; most bumps allowed.",
                ["≤ 0.1 s", "0.25 s · 6 bumps", "0.5 s · 10 bumps"],
                "0.25 s · 6 bumps",
                ("ok", "plateau"),
            ),
            card(
                "scale",
                "Keep one more bump: ΔBIC",
                "A bump is kept if the score drops by at least this much. The key setting.",
                ["3", "8", "10", "20", "calibrated"],
                "10",
                ("ok", "plateau 3–10"),
            ),
            card(
                "search",
                "Starting candidate",
                "Size a residual peak needs before a bump is tried (× noise).",
                ["1 σ", "2 σ", "3 σ"],
                "2 σ",
                ("ok", "fastest without loss"),
            ),
            card(
                "merge",
                "Merge close bumps",
                "Merge two bumps closer than c × (σ₁ + σ₂).",
                ["0", "0.5", "2"],
                "0.5",
                ("none",),
            ),
            card(
                "plusminus",
                "Cancelling pairs",
                "Reject a + and a − bump that cancel each other.",
                ["keep (CCEP)", "reject (ERP, TEP)"],
                ["keep (CCEP)", "reject (ERP, TEP)"],
                ("ok", "ERP; no effect in CCEP"),
            ),
        ],
        4,
        "<span>Verdicts: version-2 tests (known truth, subject-level statistics, held-out half)</span>",
        sketch="model",
    ),
    "6b-describe-the-measures": lambda: page(
        "Step 6 · part 2 — the measures",
        "clock",
        "Describe the response",
        "Once the bumps are fitted, these settings decide how latency, onset and duration are read, and where components are looked for.",
        [
            card(
                "clock",
                "Latency definition",
                "When a bump happens.",
                ["raw peak", "model peak", "50 % of the model area"],
                "50 % of the model area",
                ("ok",),
            ),
            card(
                "flag",
                "Onset and offset",
                "First and last time the bump reaches this share of its height.",
                ["5 %", "10 %", "20 %", "50 %"],
                "10 %",
                ("q", "20–50 % more accurate"),
            ),
            card(
                "crosshair",
                "Targeted ERP measure",
                "Window margin, parts per component, tolerance for a match.",
                ["1 part", "3 parts", "margin ± 25 ms", "tolerance ± 10 ms", "tolerance ± 25 ms"],
                ["3 parts", "margin ± 25 ms", "tolerance ± 10 ms"],
                ("q", "parts ok; ± 10 ms too strict"),
            ),
            card(
                "coil",
                "TEP windows",
                "Where N45 and N100 are looked for after a TMS pulse.",
                ["narrow", "N45 38–55 · N100 80–140 ms", "wide"],
                "N45 38–55 · N100 80–140 ms",
                ("inc",),
            ),
            card(
                "stack",
                "Trial by trial",
                "Largest shift a bump may take in one trial (× its width).",
                ["0.1", "0.25 (intracranial)", "0.5 (scalp)", "1", "2"],
                "0.5 (scalp)",
                ("q", "noise-dominated at any bound"),
            ),
        ],
        3,
        "<span>Verdicts: version-2 tests (known truth, subject-level statistics, held-out half)</span>",
        sketch="measures",
    ),
}


def main():
    out = os.path.join(HERE, "png")
    os.makedirs(out, exist_ok=True)
    for name, make in SLIDES.items():
        path = os.path.join(HERE, f"{name}.html")
        with open(path, "w") as f:
            f.write(make())
        png = os.path.join(out, f"{name}.png")
        subprocess.run(
            [
                CHROME,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=1",
                f"--screenshot={png}",
                "--window-size=1920,1080",
                f"file://{path}",
            ],
            check=True,
            capture_output=True,
        )
        print("saved", png)


if __name__ == "__main__":
    main()
