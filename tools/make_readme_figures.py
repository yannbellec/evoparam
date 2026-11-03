# SPDX-License-Identifier: Apache-2.0
"""Draw the README figures into ``figures/readme`` (animation, anatomy, 3D surface).

Run: ``python tools/make_readme_figures.py``. Needs the ``plot`` extra and Pillow.
"""

import pathlib
import struct
import warnings

import numpy as np

import evoparam

OUT = pathlib.Path(__file__).resolve().parents[1] / "figures" / "readme"
XMIN, XMAX, FIRST_BUMP = -60, 400, 2
BLUE, ORANGE, GREEN, PINK, GREY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7F7F7F"


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "evoparam",
        }
    )
    import matplotlib.pyplot as plt
    from matplotlib import patheffects

    return plt, [patheffects.withStroke(linewidth=3, foreground="white")]


def _pieces():
    t, y = evoparam.gen_response([(25, -150, 20), (150, -80, 120)], seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = evoparam.fit_response(t, y, preset="ccep")
    start = int(np.searchsorted(t, fit.window[0] - 1e-9))
    tw = t[start : start + len(fit.model)]
    bg = fit.background
    background = bg["c0"] + bg["c1"] * np.exp(-(tw - bg["t_ref"]) / bg["tau"])
    bumps = [
        c.amplitude * np.exp(-0.5 * ((tw - c.params[0]) / c.params[1]) ** 2)
        for c in fit.components
    ]
    return t * 1000, y, tw * 1000, background, bumps, fit


def anatomy(path):
    """Draw the parts of one response."""
    plt, halo = _mpl()
    t, y, tw, background, bumps, fit = _pieces()
    fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    ax.plot(t, y, color=GREY, lw=0.7, alpha=0.6, label="what we record")
    ax.plot(tw, background, color="0.3", lw=1.2, ls="--", path_effects=halo, label="slow drift")
    floor = background.copy()
    parts = zip(bumps, fit.components, (BLUE, GREEN), (14, 30), strict=True)
    for bump, comp, colour, shift in parts:
        top = floor + bump
        ax.fill_between(tw, floor, top, color=colour, alpha=0.25, lw=0)
        x = comp.latency * 1000
        base = np.interp(x, tw, floor)
        floor = top
        ax.annotate(
            "",
            xy=(x, base + comp.amplitude),
            xytext=(x, base),
            arrowprops={"arrowstyle": "->", "color": colour, "lw": 1},
        )
        ax.text(
            x + shift,
            base + comp.amplitude * 0.55,
            f"{comp.label}\nwhen: {x:.0f} ms\nhow big: {comp.amplitude:.0f} uV\n"
            f"how wide: {comp.fwhm * 1000:.0f} ms",
            fontsize=8,
            color=colour,
            va="top",
            path_effects=halo,
        )
    ax.plot(tw, fit.model, color="black", lw=1, path_effects=halo, label="evoparam model")
    ax.axvline(0, color=ORANGE, lw=0.8)
    ax.text(2, ax.get_ylim()[1] * 0.9, "pulse", color=ORANGE, fontsize=8)
    ax.set_xlim(XMIN, XMAX)
    ax.set_xlabel("time after the pulse (ms)")
    ax.set_ylabel("voltage (uV)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(path, metadata={"Date": None, "Creator": None})
    plt.close(fig)


def animation(path):
    """Animate the fit step by step."""
    plt, halo = _mpl()
    from matplotlib.animation import FuncAnimation, PillowWriter

    t, y, tw, background, bumps, fit = _pieces()
    steps = [
        "1. The brain answers a pulse. The line is noisy.",
        "2. evoparam finds the slow drift.",
        "3. It finds the first bump: N1.",
        "4. It finds the second bump: N2.",
        "5. Drift + bumps = the model. Each bump gets numbers.",
    ]
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=90, layout="constrained")
    hold = 14
    frames = [(s, k) for s in range(len(steps)) for k in range(hold)]
    shown = y[(t > XMIN) & (t < XMAX)]
    lo, hi = shown.min() - 30, shown.max() + 40

    def draw(frame):
        step, k = frame
        ax.clear()
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("time after the pulse (ms)")
        ax.set_ylabel("voltage (uV)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(0, color=ORANGE, lw=0.8)
        n = len(t) if step else int(len(t) * min(1.0, (k + 1) / hold * 1.4))
        ax.plot(t[:n], y[:n], color=GREY, lw=0.7, alpha=0.7)
        grow = min(1.0, (k + 1) / (hold * 0.6))
        if step >= 1:
            m = len(tw) if step > 1 else max(2, int(len(tw) * grow))
            ax.plot(tw[:m], background[:m], color="0.3", lw=1.3, ls="--", path_effects=halo)
        floor = background.copy()
        for i, (bump, colour) in enumerate(zip(bumps, (BLUE, GREEN), strict=True)):
            if step >= 2 + i:
                g = 1.0 if step > 2 + i else grow
                ax.fill_between(tw, floor, floor + g * bump, color=colour, alpha=0.3, lw=0)
                floor = floor + g * bump
        if step >= FIRST_BUMP:
            ax.plot(tw, floor, color="black", lw=0.6, alpha=0.5)
        if step == len(steps) - 1:
            ax.plot(tw, fit.model, color="black", lw=1.1, path_effects=halo)
            for comp, colour in zip(fit.components, (BLUE, GREEN), strict=True):
                x = comp.latency * 1000
                ax.text(
                    x + 8,
                    np.interp(x, tw, fit.model) - 8,
                    f"{comp.label}: {x:.0f} ms, {comp.amplitude:.0f} uV",
                    fontsize=9,
                    color=colour,
                    path_effects=halo,
                )
        ax.set_title(steps[step], fontsize=10, loc="left")

    anim = FuncAnimation(fig, draw, frames=frames)
    anim.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)


def surface(path):
    """Write a binary STL surface: electrodes x time x voltage."""
    rng = np.random.default_rng(0)
    channels, times = 28, np.linspace(0, 300, 90)
    z = np.zeros((channels, times.size))
    for c in range(channels):
        dist = c / (channels - 1)
        n1 = (-150 * np.exp(-3 * dist)) * np.exp(-0.5 * ((times - (18 + 25 * dist)) / 6) ** 2)
        n2 = (-70 * np.exp(-1.5 * dist)) * np.exp(-0.5 * ((times - (140 + 40 * dist)) / 45) ** 2)
        z[c] = n1 + n2 + rng.normal(0, 2, times.size)
    x = times / 3.0
    yy = np.arange(channels) * 3.0
    zz = z / 3.0
    tris = []
    for i in range(channels - 1):
        for j in range(times.size - 1):
            p = [
                (x[j], yy[i], zz[i, j]),
                (x[j + 1], yy[i], zz[i, j + 1]),
                (x[j + 1], yy[i + 1], zz[i + 1, j + 1]),
                (x[j], yy[i + 1], zz[i + 1, j]),
            ]
            tris += [(p[0], p[1], p[2]), (p[0], p[2], p[3])]
    with open(path, "wb") as fobj:
        fobj.write(b"evoparam response surface".ljust(80, b" "))
        fobj.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            n = np.cross(np.subtract(b, a), np.subtract(c, a))
            n = n / (np.linalg.norm(n) or 1.0)
            fobj.write(struct.pack("<12fH", *n, *a, *b, *c, 0))


BANNER = """<svg xmlns="http://www.w3.org/2000/svg" width="900" height="260" viewBox="0 0 900 260">
<style>
.bg{{fill:{bg}}} .t{{font:700 64px 'Segoe UI',Helvetica,Arial,sans-serif;fill:{fg}}}
.s{{font:400 19px 'Segoe UI',Helvetica,Arial,sans-serif;fill:{muted}}}
.l{{font:600 14px 'Segoe UI',Helvetica,Arial,sans-serif;fill:{accent};letter-spacing:1.5px}}
.k{{font:600 15px 'Segoe UI',Helvetica,Arial,sans-serif}}
.grid{{stroke:{grid};stroke-width:1}}
.raw{{fill:none;stroke:{muted};stroke-width:1;opacity:.55}}
.fit{{fill:none;stroke:{blue};stroke-width:2.6;stroke-linecap:round;stroke-dasharray:{length};
stroke-dashoffset:{length};animation:draw 7s ease-in-out infinite}}
.n1,.n2{{opacity:0;animation:pop 7s ease-in-out infinite}} .n2{{animation-delay:.5s}}
.pulse{{stroke:{orange};stroke-width:1.5}}
@keyframes draw{{0%{{stroke-dashoffset:{length}}}45%,85%{{stroke-dashoffset:0}}
100%{{stroke-dashoffset:0;opacity:0}}}}
@keyframes pop{{0%,40%{{opacity:0}}50%,85%{{opacity:1}}100%{{opacity:0}}}}
@media (prefers-reduced-motion:reduce){{.fit{{animation:none;stroke-dashoffset:0}}
.n1,.n2{{animation:none;opacity:1}}}}
</style>
<rect class="bg" width="900" height="260" rx="14"/>
<line class="grid" x1="430" y1="{zero}" x2="870" y2="{zero}"/>
<line class="pulse" x1="{x0}" y1="40" x2="{x0}" y2="225"/>
<path class="raw" d="{raw}"/>
<path class="fit" d="{model}"/>
<g class="n1"><circle cx="{n1x}" cy="{n1y}" r="4" fill="{blue}"/>
<text class="k" x="{n1tx}" y="{n1y}" fill="{blue}">N1</text></g>
<g class="n2"><circle cx="{n2x}" cy="{n2y}" r="4" fill="{green}"/>
<text class="k" x="{n2x}" y="{n2ty}" fill="{green}" text-anchor="middle">N2</text></g>
<text class="t" x="40" y="112">evoparam</text>
<text class="s" x="42" y="148">Brain answers, described in a few numbers.</text>
<text class="l" x="42" y="200">NEURAL SIGNAL PROCESSING AND MODELING LAB</text>
<text class="s" x="42" y="226" style="font-size:15px">PI: Rachel June Smith</text>
</svg>
"""

THEMES = {
    "light": {
        "bg": "#FFFFFF",
        "fg": "#1F2328",
        "muted": "#59636E",
        "grid": "#D1D9E0",
        "accent": "#0072B2",
        "blue": "#0072B2",
        "green": "#009E73",
        "orange": "#D55E00",
    },
    "dark": {
        "bg": "#0D1117",
        "fg": "#F0F6FC",
        "muted": "#9198A1",
        "grid": "#30363D",
        "accent": "#56B4E9",
        "blue": "#56B4E9",
        "green": "#3FCF9F",
        "orange": "#E69F00",
    },
}


def banner(folder):
    """Write the animated banner, light and dark."""
    t, y, tw, _, _, fit = _pieces()

    def sx(ms):
        return 440 + (ms + 40) / 440 * 420

    def sy(uv):
        return 70 - uv * 0.85

    keep = (t > XMIN + 20) & (t < XMAX)
    raw = "M" + " L".join(
        f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(t[keep][::3], y[keep][::3], strict=True)
    )
    pts = [(sx(a), sy(b)) for a, b in zip(tw[::2], fit.model[::2], strict=True)]
    model = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts)
    length = (
        int(sum(np.hypot(*np.subtract(p, q)) for p, q in zip(pts[1:], pts[:-1], strict=True))) + 10
    )
    n1, n2 = fit.components[:2]
    n1x, n1y = sx(n1.latency * 1000), sy(np.interp(n1.latency * 1000, tw, fit.model))
    n2x, n2y = sx(n2.latency * 1000), sy(np.interp(n2.latency * 1000, tw, fit.model))
    for name, theme in THEMES.items():
        text = BANNER.format(
            raw=raw,
            model=model,
            length=length,
            zero=f"{sy(0):.1f}",
            x0=f"{sx(0):.1f}",
            n1x=f"{n1x:.1f}",
            n1y=f"{n1y:.1f}",
            n1tx=f"{n1x + 10:.1f}",
            n2x=f"{n2x:.1f}",
            n2y=f"{n2y:.1f}",
            n2ty=f"{n2y + 24:.1f}",
            **theme,
        )
        (folder / f"banner-{name}.svg").write_text(text, encoding="utf-8")


def main():
    """Draw every README figure."""
    OUT.mkdir(parents=True, exist_ok=True)
    banner(OUT)
    anatomy(OUT / "anatomy.svg")
    animation(OUT / "fit.gif")
    surface(OUT / "response_surface.stl")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
