"""Implementation of :mod:`evoparam.plots`."""

import numpy as np

from evoparam.data import Evoked, GroupFit, ResponseFit, TrialSet
from evoparam.errors import SettingsError
from evoparam.utils._optional import require

__all__ = []

# Okabe and Ito (2008), colour-blind safe.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442",
           "#000000")
DATA_COLOR, MODEL_COLOR, BG_COLOR, RESID_COLOR = "#000000", "#0072B2", "#999999", "#E69F00"


def _plt():
    mpl = require("matplotlib", extra="plot", caller="evoparam.plots")
    import matplotlib.pyplot as plt

    del mpl
    return plt


def _figure_ax(ax, figsize=(6.0, 3.5)):
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        return fig, ax
    return ax.figure, ax


def _evoked(data, channel):
    """(times, waveform, mask) of one channel of an Evoked or TrialSet (its average)."""

    if isinstance(data, TrialSet):
        data = data.average()
    if not isinstance(data, Evoked):
        raise SettingsError("data must be an Evoked or a TrialSet.")
    index = data.ch_names.index(channel) if isinstance(channel, str) else int(channel or 0)
    mask = None if data.masks is None else data.masks[index]
    return data.times, data.data[index], mask


def _window_times(fit):
    n = len(fit.model) if fit.model is not None else 0
    return np.linspace(fit.window[0], fit.window[1], n) if n else np.zeros(0)


def plot_fit(fit, data=None, *, channel=None, ax=None, components=True, background=True,
             residual=True, intervals=True):
    """Data, model, components, background and residual of one :class:`ResponseFit`.

    Parameters
    ----------
    fit : ResponseFit
    data : Evoked or TrialSet, optional
        The waveform that was fitted (``channel`` names it); without it, the data are
        rebuilt in the fit window as model + residual.
    intervals : bool
        Shade the latency interval of each component (its ``ci["latency"]``).
    """

    from evoparam.models import evaluate_fit

    if not isinstance(fit, ResponseFit):
        raise SettingsError("plot_fit expects a ResponseFit (use plot_group for a GroupFit).")
    fig, ax = _figure_ax(ax)
    if data is not None:
        times, y, mask = _evoked(data, channel)
        ax.plot(times * 1e3, y, color=DATA_COLOR, lw=0.8, label="data")
        if mask is not None and mask.any():
            edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
            for start, stop in zip(edges[::2], edges[1::2]):
                ax.axvspan(times[start] * 1e3, times[stop - 1] * 1e3, color="#dddddd", lw=0,
                           label="masked" if start == edges[0] else None)
        grid = times[(times >= fit.window[0]) & (times <= fit.window[1])]
    else:
        grid = _window_times(fit)
        if fit.model is not None and fit.residual is not None:
            ax.plot(grid * 1e3, fit.model + fit.residual, color=DATA_COLOR, lw=0.8,
                    label="data (fit window)")
    if fit.components or fit.background:
        ax.plot(grid * 1e3, evaluate_fit(fit, grid), color=MODEL_COLOR, lw=1.8, label="model",
                zorder=5)
    if background and fit.background:
        bg = evaluate_fit(fit, grid, components=[])
        ax.plot(grid * 1e3, bg, color=BG_COLOR, lw=1.0, ls="--", label="background")
    if components:
        for k, comp in enumerate(fit.components):
            curve = evaluate_fit(fit, grid, components=[k], background=False)
            color = PALETTE[(k + 1) % len(PALETTE)]
            ax.plot(grid * 1e3, curve, color=color, lw=1.0, ls=":", zorder=4,
                    label=comp.label or f"C{k + 1}")
            ci = (comp.ci or {}).get("latency")
            if intervals and ci is not None and np.all(np.isfinite(ci)):
                ax.axvspan(ci[0] * 1e3, ci[1] * 1e3, color=color, alpha=0.15, lw=0)
    # the residual is drawn below the data (offset), so that it does not hide them
    lines = [line.get_ydata() for line in ax.get_lines()]
    low = min((np.nanmin(v) for v in lines if len(v)), default=0.0)
    high = max((np.nanmax(v) for v in lines if len(v)), default=1.0)
    shift = low - 0.6 * (high - low)
    if residual and fit.residual is not None and data is None:
        ax.plot(_window_times(fit) * 1e3, fit.residual + shift, color=RESID_COLOR, lw=0.6,
                label="residual (offset)")
    elif residual and data is not None and (fit.components or fit.background):
        times, y, _ = _evoked(data, channel)
        inside = (times >= fit.window[0]) & (times <= fit.window[1])
        ax.plot(grid * 1e3, y[inside] - evaluate_fit(fit, grid) + shift, color=RESID_COLOR,
                lw=0.6, label="residual (offset)")
    ax.axvline(0, color="#000000", lw=0.5)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("amplitude")
    ax.set_title(f"{fit.status}: {len(fit.components)} component(s)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    return fig


def plot_group(group_fit, data=None, *, kind="grid", ax=None, axes=None, max_channels=64,
               coords=None, parameter="amplitude", component=0):
    """Every channel of a :class:`GroupFit`.

    ``kind="grid"``: one small panel per channel (data and model); ``"shaft"``: the channels
    of each electrode (name prefix) stacked with an offset; ``"brain"``: a scatter of the
    channel ``coords`` (array n_channels x 2 or 3, or ``data.ch_info`` x, y) coloured by
    ``parameter`` of component ``component``.
    """

    from evoparam.models import evaluate_fit
    from evoparam.utils._names import contact_group

    if not isinstance(group_fit, GroupFit):
        raise SettingsError("plot_group expects a GroupFit.")
    plt = _plt()
    names = list(group_fit.index["channel"])[:max_channels]
    fits = [group_fit.fits[int(p)] for p in group_fit.index["position"]][:max_channels]
    if kind == "grid":
        n = len(fits)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        if axes is None:
            fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 1.3 * rows),
                                     sharex=True, layout="constrained", squeeze=False)
        else:
            fig = np.ravel(axes)[0].figure
        flat = np.ravel(axes)
        for a, name, fit in zip(flat, names, fits):
            if data is not None:
                times, y, _ = _evoked(data, name)
                a.plot(times * 1e3, y, color=DATA_COLOR, lw=0.5)
                grid = times[(times >= fit.window[0]) & (times <= fit.window[1])]
            else:
                grid = _window_times(fit)
                if fit.model is not None and fit.residual is not None:
                    a.plot(grid * 1e3, fit.model + fit.residual, color=DATA_COLOR, lw=0.5)
            if fit.components or fit.background:
                a.plot(grid * 1e3, evaluate_fit(fit, grid), color=MODEL_COLOR, lw=1.0)
            label = ",".join(c.label or "" for c in fit.components)
            a.set_title(f"{name} {label}", fontsize=6)
            a.tick_params(labelsize=5)
        for a in flat[len(fits):]:
            a.axis("off")
        return fig
    if kind == "shaft":
        fig, ax = _figure_ax(ax, figsize=(6.0, 8.0))
        groups = {}
        for name, fit in zip(names, fits):
            groups.setdefault(contact_group(name), []).append((name, fit))
        offset = 0.0
        ticks, labels = [], []
        span = max((float(np.ptp(f.model)) for f in fits if f.model is not None
                    and len(f.model)), default=1.0) or 1.0
        for g, (shaft, members) in enumerate(groups.items()):
            for name, fit in members:
                grid = _window_times(fit)
                if fit.model is not None and len(fit.model):
                    ax.plot(grid * 1e3, fit.model - offset, color=PALETTE[g % len(PALETTE)],
                            lw=0.8)
                ticks.append(-offset)
                labels.append(name)
                offset += span
        ax.set_yticks(ticks, labels, fontsize=6)
        ax.set_xlabel("time (ms)")
        return fig
    if kind == "brain":
        if coords is None and data is not None and getattr(data, "ch_info", None) is not None:
            info = data.ch_info
            if {"x", "y"} <= set(info.columns):
                coords = info[["x", "y"]].to_numpy(float)[:len(names)]
        if coords is None:
            raise SettingsError("kind='brain' needs coords= (n_channels x 2 or 3) or a data "
                                "argument whose ch_info has x and y.")
        coords = np.asarray(coords, float)
        values = []
        for fit in fits:
            comps = fit.components
            values.append(getattr(comps[component], parameter) if len(comps) > component
                          else np.nan)
        fig, ax = _figure_ax(ax, figsize=(5.0, 4.5))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap="cividis", s=30,
                             edgecolors="#333333", linewidths=0.4)
        fig.colorbar(scatter, ax=ax, label=f"{parameter} (component {component})")
        ax.set_aspect("equal")
        return fig
    raise SettingsError("kind must be 'grid', 'shaft' or 'brain'.")


def plot_trials(trial_fit, trials=None, *, channel=None, axes=None):
    """Trial-by-trial results: the trials sorted by latency (with ``trials``, a TrialSet)
    and the latency histogram, for one :class:`TrialFit` or a channel of a TrialFitSet."""

    plt = _plt()
    fit = trial_fit
    if hasattr(trial_fit, "fits"):
        fit = trial_fit.get(channel) if channel is not None else trial_fit.fits[0]
    latency = np.asarray(fit.latency, float)
    if latency.ndim > 1:
        latency = latency[:, 0]
    order = np.argsort(np.nan_to_num(latency, nan=np.inf))
    n_panels = 2 if trials is not None else 1
    if axes is None:
        fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.5),
                                 layout="constrained", squeeze=False)
        axes = axes[0]
    else:
        axes = np.ravel(axes)
        fig = axes[0].figure
    if trials is not None:
        index = trials.ch_names.index(fit.channel)
        image = trials.data[:, index][order]
        limit = np.nanpercentile(np.abs(image), 99) or 1.0
        axes[0].imshow(image, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit,
                       extent=[trials.times[0] * 1e3, trials.times[-1] * 1e3, len(order), 0])
        axes[0].plot(latency[order] * 1e3, np.arange(len(order)) + 0.5, color="#000000",
                     lw=0.8)
        axes[0].set_xlabel("time (ms)")
        axes[0].set_ylabel("trial (sorted by latency)")
    hist_ax = axes[-1]
    trusted = np.asarray(fit.trusted, bool)
    hist_ax.hist(latency[trusted] * 1e3, bins=20, color=PALETTE[0], alpha=0.8, label="trusted")
    if (~trusted).any():
        hist_ax.hist(latency[~trusted] * 1e3, bins=20, color=PALETTE[1], alpha=0.6,
                     label="not trusted")
    hist_ax.set_xlabel("latency (ms)")
    hist_ax.set_ylabel("trials")
    hist_ax.legend(fontsize=7, frameon=False)
    hist_ax.set_title(f"{fit.channel}", fontsize=9)
    return fig


def plot_existence(test, *, ax=None, channels=None):
    """Existence test per channel: the statistic (significant channels filled) and the
    response window from onset to offset."""

    fig, ax = _figure_ax(ax, figsize=(6.0, 0.25 * max(len(test.statistic), 8) + 1))
    names = channels or test.settings.get("channels") or \
        [f"ch{i}" for i in range(len(test.statistic))]
    y = np.arange(len(test.statistic))
    sig = np.asarray(test.significant, bool)
    ax.barh(y, test.statistic, color=np.where(sig, PALETTE[0], "#cccccc"))
    ax.set_yticks(y, names, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel(f"statistic ({test.method})")
    twin = ax.twiny()
    for i in np.flatnonzero(sig):
        if np.isfinite(test.onset[i]) and np.isfinite(test.offset[i]):
            twin.plot([test.onset[i] * 1e3, test.offset[i] * 1e3], [i, i], color=PALETTE[1],
                      lw=2)
    twin.set_xlabel("response window (ms)", color=PALETTE[1])
    ax.set_title(f"{int(sig.sum())} of {len(sig)} channels significant "
                 f"(alpha {test.alpha:g}, {test.correction})", fontsize=9)
    return fig


def plot_tf(tf, *, row=0, axes=None, maps=("total_norm", "footprint_norm", "residual_norm")):
    """Time-frequency maps of one row of a :class:`TFFit`, with the zones outlined and the
    significant pixels contoured."""

    plt = _plt()
    present = [m for m in maps if m in tf.maps]
    if axes is None:
        fig, axes = plt.subplots(1, len(present), figsize=(3.6 * len(present), 3.2),
                                 layout="constrained", squeeze=False)
        axes = axes[0]
    else:
        axes = np.ravel(axes)
        fig = axes[0].figure
    from matplotlib.patches import Rectangle

    extent = [tf.times[0] * 1e3, tf.times[-1] * 1e3, 0, len(tf.freqs)]
    for a, name in zip(axes, present):
        image = tf.maps[name][row]
        limit = np.nanpercentile(np.abs(image), 99) if np.isfinite(image).any() else 1.0
        shown = a.imshow(image, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-limit,
                         vmax=limit, extent=extent)
        if name == "total_norm" and tf.significant[row].any():
            a.contour(tf.significant[row].astype(float), levels=[0.5], colors="#000000",
                      linewidths=0.6, extent=extent)
        zones = tf.zones
        if zones is not None and len(zones) and {"t_start", "f_low"} <= set(zones.columns):
            for _, z in zones.iterrows():
                lo = np.searchsorted(tf.freqs, z["f_low"])
                hi = np.searchsorted(tf.freqs, z["f_high"], side="right")
                a.add_patch(Rectangle((z["t_start"] * 1e3, lo),
                                      (z["t_end"] - z["t_start"]) * 1e3, hi - lo, fill=False,
                                      edgecolor=PALETTE[4], lw=1.0))
        ticks = np.linspace(0, len(tf.freqs) - 1, 5).astype(int)
        a.set_yticks(ticks + 0.5, [f"{tf.freqs[t]:.0f}" for t in ticks])
        a.set_xlabel("time (ms)")
        a.set_ylabel("frequency (Hz)")
        a.set_title(name, fontsize=9)
        fig.colorbar(shown, ax=a, shrink=0.8)
    return fig


def plot_shapes(basis, *, ax=None):
    """The shared shapes of a :class:`ShapeBasis`."""

    fig, ax = _figure_ax(ax)
    for k, curve in enumerate(basis.curves):
        stability = basis.stability[k] if k < len(basis.stability) else np.nan
        ax.plot(basis.times * 1e3, curve, color=PALETTE[k % len(PALETTE)],
                label=f"shape {k + 1} (stability {stability:.2f})")
    ax.set_xlabel("time (ms)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"{basis.n_shapes} shared shape(s), {basis.method}", fontsize=9)
    return fig


def plot_artifact(result, raw=None, *, channel=0, axes=None):
    """Artifact removal: before (``raw``, a TrialSet) and after for one channel, the
    template, and the per-trial quality."""

    plt = _plt()
    cleaned = result.cleaned
    index = cleaned.ch_names.index(channel) if isinstance(channel, str) else int(channel)
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), layout="constrained")
    else:
        fig = np.ravel(axes)[0].figure
    a, b = np.ravel(axes)[:2]
    times = cleaned.times * 1e3
    if raw is not None:
        a.plot(times, np.nanmean(raw.data[:, index], axis=0), color=BG_COLOR, lw=0.8,
               label="before")
    a.plot(times, np.nanmean(cleaned.data[:, index], axis=0), color=MODEL_COLOR, lw=1.0,
           label="after")
    a.axvspan(result.window[0] * 1e3, result.window[1] * 1e3, color="#dddddd", lw=0,
              label="artifact window")
    if result.template is not None:
        template = np.asarray(result.template)
        if template.ndim >= 2 and template.shape[-1] == len(times):
            a.plot(times, template.reshape(-1, len(times))[min(index, len(template) - 1)],
                   color=RESID_COLOR, lw=0.8, ls="--", label="template")
    a.set_xlabel("time (ms)")
    a.legend(fontsize=7, frameon=False)
    a.set_title(f"{result.method}: {cleaned.ch_names[index]}", fontsize=9)
    quality = np.asarray(result.quality, float)
    if quality.ndim > 1:
        quality = np.nanmean(quality, axis=tuple(range(1, quality.ndim)))
    b.plot(quality, "o", ms=3, color=PALETTE[0])
    b.set_xlabel("trial")
    b.set_ylabel("quality")
    return fig


def plot_compare(fits, data=None, *, channel=None, labels=None, ax=None):
    """The models of several fits of the same waveform, over the data."""

    from evoparam.models import evaluate_fit

    fig, ax = _figure_ax(ax)
    if data is not None:
        times, y, _ = _evoked(data, channel)
        ax.plot(times * 1e3, y, color=DATA_COLOR, lw=0.8, label="data")
    for k, fit in enumerate(fits):
        grid = _window_times(fit) if data is None else \
            times[(times >= fit.window[0]) & (times <= fit.window[1])]
        name = labels[k] if labels else f"fit {k + 1} ({len(fit.components)} comp.)"
        ax.plot(grid * 1e3, evaluate_fit(fit, grid), color=PALETTE[k % len(PALETTE)],
                lw=1.2, label=name)
    ax.set_xlabel("time (ms)")
    ax.legend(fontsize=7, frameon=False)
    return fig
