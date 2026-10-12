# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape estimates of single pulses: frozen templates, shifted and scaled.

The reference minimal estimator of the online mode (spec §10.6): the components of each
channel's fit are frozen; per pulse and channel the free parameters are shifts (grid of
whole samples over ``+/- max_shift``, refined by parabolic interpolation, then the
components evaluated at the refined shifts) and scales (least squares), with the linear
terms of the channel's background re-estimated (its time constants stay at the channel
values). ``mode="common"`` is the method of spec §10.6, one shift and one scale for the
whole template; the default ``"shift_scale"`` gives each component its own shift and scale,
the ``"shift_scale"`` trial model of :func:`evoparam.trials.fit_trials` solved on a grid
instead of by an iterative optimiser. With a single component the modes are the same model.

Everything that does not depend on the pulse (the shifted components, the background bases,
the inverse normal matrices of every common shift and the products of every pair of shifted
components) is computed once, so that a pulse costs a few matrix products.
"""

import numpy as np

from evoparam.data import GroupFit, ResponseFit, TrialSet
from evoparam.errors import DataError, SettingsError
from evoparam.models.component_shapes import get_background
from evoparam.utils._interp import interpolate_window

from .api import _amplitude_of, _shape_for

__all__ = ["FixedShapeModel"]

# Anchor samples on each side of the artifact window (the default of remove_artifact).
INTERP_BUFFER = 3
# A component is evaluated where, at some shift within max_shift, it exceeds this fraction
# of its peak (and two samples beyond); elsewhere it is below float64 resolution.
TAIL = 1e-16


def _offset(t, sfreq, side):
    """Sample offset of the first sample at or after ``t`` (``side="start"``), or of the last
    at or before it (``"stop"``), with the comparison of ``times >= t`` on ``k / sfreq``.
    """
    k = int(np.ceil(t * sfreq)) if side == "start" else int(np.floor(t * sfreq))
    if side == "start":
        while (k - 1) / sfreq >= t:
            k -= 1
        while k / sfreq < t:
            k += 1
    else:
        while (k + 1) / sfreq <= t:
            k += 1
        while k / sfreq > t:
            k -= 1
    return k


class _Channel:
    """The frozen template of one channel."""

    def __init__(self, fit, tw):
        settings = fit.settings or {}
        self.shapes, self.params, self.locs, self.amps = [], [], [], []
        self.labels, self.latencies, self.amplitudes = [], [], []
        for k, comp in enumerate(fit.components):
            shape = _shape_for(comp, settings)
            self.shapes.append(shape)
            self.params.append(np.array(comp.params, float))
            self.locs.append(shape.param_names.index(shape.latency_param))
            self.amps.append(float(_amplitude_of(comp)))
            self.labels.append(comp.label or str(k))
            self.latencies.append(float(comp.latency))
            self.amplitudes.append(float(comp.amplitude))
        bg = fit.background or {"name": "none"}
        background = get_background(bg["name"])
        if getattr(background, "n_linear", 0):
            nl = np.array([bg[n] for n in getattr(background, "nonlinear_names", ())], float)
            self.basis = np.atleast_2d(background.basis(tw, nl if len(nl) else None))
        else:
            self.basis = np.zeros((0, len(tw)))
        self.support = np.zeros(len(tw), bool)
        for comp in fit.components:
            self.support |= (tw >= comp.onset - comp.fwhm) & (tw <= comp.offset + comp.fwhm)
        self.key = tuple(
            (type(s).__name__, repr(s), p.tobytes(), a)
            for s, p, a in zip(self.shapes, self.params, self.amps)
        )
        self.spans = [slice(None)] * len(self.shapes)

    def curves(self, tw, shifts, component=None):
        """The template (or one of its components) moved by each of ``shifts`` (s): array
        [n_shifts, n_times].
        """
        out = np.zeros((len(shifts), len(tw)))
        keep = range(len(self.shapes)) if component is None else [component]
        for k in keep:
            batch = np.tile(self.params[k], (len(shifts), 1))
            batch[:, self.locs[k]] += shifts
            span = self.spans[k]
            curve = np.asarray(self.shapes[k].evaluate(tw[span], batch))
            out[:, span] += self.amps[k] * curve.reshape(len(shifts), -1)
        return out

    def set_spans(self, tw, grid):
        """Samples where each component matters at some shift of ``grid`` (see TAIL)."""
        for k in range(len(self.shapes)):
            self.spans[k] = slice(None)
            size = np.abs(self.curves(tw, grid, component=k)).max(axis=0)
            above = np.flatnonzero(size > TAIL * size.max())
            if len(above):
                self.spans[k] = slice(max(int(above[0]) - 2, 0), min(int(above[-1]) + 3, len(tw)))

    def bounds(self, max_shift):
        """Largest shift of each component moved on its own: ``max_shift``, at most half the
        distance to the nearest other component (as in ``fit_trials``).
        """
        out = np.full(len(self.shapes), float(max_shift))
        for k, lat in enumerate(self.latencies):
            gaps = [abs(lat - other) for j, other in enumerate(self.latencies) if j != k]
            if gaps:
                out[k] = min(out[k], 0.5 * min(gaps))
        return out


def _parabolic(profile, best):
    """Sub-sample offset of the maximum of ``profile`` [n, n_points] around ``best`` [n]."""
    n, n_points = profile.shape
    rows = np.arange(n)
    frac = np.zeros(n)
    inner = (best > 0) & (best < n_points - 1)
    if inner.any():
        left = profile[rows[inner], best[inner] - 1]
        mid = profile[rows[inner], best[inner]]
        right = profile[rows[inner], best[inner] + 1]
        with np.errstate(invalid="ignore"):
            curv = left - 2 * mid + right
            safe = np.isfinite(left) & np.isfinite(right) & (curv < 0)
            frac[inner] = np.where(safe, 0.5 * (left - right) / np.where(safe, curv, -1.0), 0.0)
    return np.clip(frac, -0.5, 0.5)


def _solve(gram, rhs):
    """Batched least-squares coefficients of the normal equations ``gram @ x = rhs``."""
    try:
        return np.linalg.solve(gram, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return np.matmul(np.linalg.pinv(gram, hermitian=True), rhs[..., None])[..., 0]


class FixedShapeModel:
    """Frozen channel templates, fitted to single pulses by shifts and scales.

    Parameters
    ----------
    template : ResponseFit or GroupFit
        The channel fit(s) of an offline analysis. A ``ResponseFit`` is used for every
        channel; a ``GroupFit`` gives each channel its own fit (channels without one, or whose
        fit is not ``"ok"`` or has no component, are left out: see ``ch_names``).
    sfreq : float
        Sampling frequency (Hz).
    ch_names : list of str
        Channels of the data passed to :meth:`fit`, in order.
    window : (float, float), optional
        Fit window relative to the pulse onset (s); by default the window of the channel
        fits, which must then be the same for every channel.
    artifact_window : (float, float), optional
        Replaced by the core ``"interpolate"`` method of spec §5.3 (a straight line between
        the means of the 3 samples on each side) before the fit and the projections. When it
        overlaps the fit window, the data passed to :meth:`fit` start 3 samples before it
        (``segment_offsets``).
    max_shift : float
        Largest shift of the template (s).
    mode : {"shift_scale", "scale", "common"}
        Free parameters of a channel with several components. ``"common"`` (spec §10.6): one
        shift and one scale for the whole template. ``"scale"``: one shift, one scale per
        component. ``"shift_scale"`` (default): a shift and a scale per component, each
        shift within ``max_shift`` and half the distance to the nearest other component, as
        the ``"shift_scale"`` trial model of :func:`evoparam.trials.fit_trials`. The
        per-component shifts are found by coordinate ascent on the one-sample grid from the
        best common shift, then refined by parabolic interpolation one at a time. With a
        single component the three modes are the same model.

    The baseline segment, whose projection on the template is the per-pulse reference of
    the existence statistics, has as many samples as the window and ends just before the
    window or the artifact window, whichever starts first.

    Attributes
    ----------
    ch_names : list of str
        The modelled channels; ``picks`` are their indices in the input ``ch_names``.
    window_offsets, baseline_offsets : (int, int)
        First and last sample of the window and of the baseline, relative to the onset.
    labels, latencies, amplitudes : list of lists
        Per channel, the label, latency (s) and amplitude of each frozen component.
    """

    MODES = ("shift_scale", "scale", "common")
    MAX_SWEEPS = 20

    def __init__(
        self,
        template,
        *,
        sfreq,
        ch_names,
        window=None,
        artifact_window=None,
        max_shift=0.005,
        mode="shift_scale",
    ):
        if not sfreq > 0:
            raise SettingsError("sfreq must be positive.")
        if not max_shift >= 0:
            raise SettingsError("max_shift must be zero or positive.")
        if mode not in self.MODES:
            raise SettingsError(f"mode must be one of {self.MODES}, got {mode!r}.")
        self.mode = mode
        self.sfreq = float(sfreq)
        names = [str(name) for name in ch_names]
        fits = self._channel_fits(template, names)
        if not fits:
            raise DataError(
                "FixedShapeModel: no channel has a usable fit (status 'ok' and at "
                "least one component)."
            )
        if window is None:
            windows = {tuple(fit.window) for _, fit in fits}
            if len(windows) > 1:
                raise SettingsError("The channel fits have different windows; give window=.")
            window = windows.pop()
        window = (float(window[0]), float(window[1]))
        i0, i1 = _offset(window[0], sfreq, "start"), _offset(window[1], sfreq, "stop")
        if i1 - i0 < 3:
            raise SettingsError("window must hold at least 4 samples.")
        self.window = window
        self.window_offsets = (i0, i1)
        self.tw = np.arange(i0, i1 + 1) / self.sfreq
        n_w = len(self.tw)
        self.weights = np.ones(n_w)
        start, stop, base_end = i0, i1, i0
        self.artifact_window = None
        if artifact_window is not None:
            a0 = _offset(artifact_window[0], sfreq, "start")
            a1 = _offset(artifact_window[1], sfreq, "stop")
            if a1 < a0:
                raise SettingsError("artifact_window holds no sample.")
            base_end = min(i0, a0)
            if a0 <= i1 and a1 >= i0:
                # The interpolation changes window samples: the data reach its anchors
                self.artifact_window = (float(artifact_window[0]), float(artifact_window[1]))
                start, stop = min(i0, a0 - INTERP_BUFFER), max(i1, a1 + INTERP_BUFFER)
        self.segment_offsets = (start, stop)
        self._segment_times = np.arange(start, stop + 1) / self.sfreq
        self._in_window = slice(i0 - start, i1 - start + 1)
        b0 = base_end - n_w
        self.baseline_offsets = (b0, b0 + n_w - 1)
        self.max_shift = float(max_shift)
        self.n_grid = int(np.floor(self.max_shift * self.sfreq + 1e-9))
        self.grid = np.arange(-self.n_grid, self.n_grid + 1) / self.sfreq

        self.picks = np.array([names.index(name) for name, _ in fits])
        self.ch_names = [name for name, _ in fits]
        self._channels = [_Channel(fit, self.tw) for _, fit in fits]
        spans = {}
        for ch in self._channels:
            if ch.key not in spans:
                ch.set_spans(self.tw, self.grid)
                spans[ch.key] = ch.spans
            ch.spans = spans[ch.key]
        self.labels = [ch.labels for ch in self._channels]
        self.latencies = [ch.latencies for ch in self._channels]
        self.amplitudes = [ch.amplitudes for ch in self._channels]
        self._prepare()

    @staticmethod
    def _channel_fits(template, names):
        if isinstance(template, ResponseFit):
            if template.status != "ok" or not template.components:
                return []
            return [(name, template) for name in names]
        if not isinstance(template, GroupFit):
            raise DataError("template must be a ResponseFit or a GroupFit.")
        out = []
        for name in names:
            n_fits = int((template.index["channel"] == name).sum())
            if n_fits > 1:
                raise DataError(
                    f"The GroupFit has {n_fits} fits (conditions) for channel "
                    f"{name!r}; give a GroupFit with one fit per channel."
                )
            if n_fits == 0:
                continue
            fit = template.get(name)
            if fit.status == "ok" and fit.components:
                out.append((name, fit))
        return out

    def _prepare(self):
        """Shifted templates, bases and inverse normal matrices of every grid shift."""
        n_ch, n_w = len(self._channels), len(self.tw)
        n_b = max(ch.basis.shape[0] for ch in self._channels)
        if any(ch.basis.shape[0] != n_b for ch in self._channels):
            raise SettingsError(
                "The channel fits use backgrounds with different numbers of linear terms."
            )
        k_max = max(len(labels) for labels in self.labels)
        # Columns of the design: the whole template, or one per component (zero beyond a
        # channel's components, whose coefficient a unit diagonal term keeps at zero)
        q = 1 if self.mode == "common" or k_max == 1 else k_max
        self.n_columns = q
        self.n_linear = q + n_b
        self._pad = np.zeros((n_ch, q))
        bound = np.full((n_ch, q), self.max_shift)
        for c, ch in enumerate(self._channels):
            if q > 1:
                self._pad[c, len(ch.shapes) :] = 1.0
                bound[c] = 0.0
                bound[c, : len(ch.shapes)] = (
                    ch.bounds(self.max_shift) if self.mode == "shift_scale" else self.max_shift
                )
        self._bound = bound
        steps = np.minimum(np.floor(bound * self.sfreq + 1e-9).astype(int), self.n_grid)
        self._allowed = np.abs(np.arange(len(self.grid)) - self.n_grid) <= steps[..., None]
        groups, self._group_of = {}, np.empty(n_ch, int)
        for c, ch in enumerate(self._channels):
            self._group_of[c] = groups.setdefault(ch.key, len(groups))
        self._groups = [np.flatnonzero(self._group_of == g) for g in range(len(groups))]
        self._templates = []  # [q, n_shifts, n_times]
        for idx in self._groups:
            ch = self._channels[idx[0]]
            cols = np.zeros((q, len(self.grid), n_w))
            if q == 1:
                cols[0] = ch.curves(self.tw, self.grid)
            else:
                for k in range(len(ch.shapes)):
                    cols[k] = ch.curves(self.tw, self.grid, component=k)
            self._templates.append(cols)
        # Arrays that are the same for every channel (a ResponseFit template) are kept once,
        # with a leading axis of length 1, so that a pulse reads them from the cache
        basis = np.stack([ch.basis for ch in self._channels])
        support = np.stack([ch.support for ch in self._channels])
        self._shared = bool(
            np.array_equal(basis, np.broadcast_to(basis[:1], basis.shape))
            and np.array_equal(support, np.broadcast_to(support[:1], support.shape))
        )
        self._basis = basis[:1] if self._shared else basis
        self._support = support[:1] if self._shared else support
        self._keep = (self._support & (self.weights > 0)).astype(float)
        self._basis_gram = np.matmul(self._basis * self.weights, np.swapaxes(self._basis, 1, 2))
        clean = np.broadcast_to(self.weights, (n_ch, n_w))
        self._inverse = self._normal_inverse(clean)
        self._descent = self.mode == "shift_scale" and q > 1
        if self._descent:
            self._tables = self._pair_tables(clean)
        # Existence projection: the unshifted template, unit norm over the weighted samples
        centre = (
            np.stack(
                [
                    self._templates[self._group_of[c]][:, self.n_grid].sum(axis=0)
                    for c in range(n_ch)
                ]
            )
            * self.weights
        )
        norm = np.sqrt(np.einsum("ct,ct->c", centre, centre))
        self._unit = centre / np.where(norm > 0, norm, 1.0)[:, None]
        self._lat0 = np.full((n_ch, k_max), np.nan)
        self._amp0 = np.full((n_ch, k_max), np.nan)
        for c in range(n_ch):
            self._lat0[c, : len(self.latencies[c])] = self.latencies[c]
            self._amp0[c, : len(self.amplitudes[c])] = self.amplitudes[c]

    def _normal_inverse(self, weights, channels=None):
        """Inverse normal matrices [n_channels, n_shifts, m, m] of the common shifts, for
        per-channel weights.
        """
        channels = np.arange(len(self._channels)) if channels is None else channels
        q, diag = self.n_columns, np.arange(self.n_columns)
        out = np.empty((len(channels), len(self.grid), self.n_linear, self.n_linear))
        for i, c in enumerate(channels):
            t = self._templates[self._group_of[c]]  # [q, n_shifts, n_times]
            b = self._basis[0 if self._shared else c]  # [n_b, n_times]
            w = weights[i]
            gram = np.empty((len(self.grid), self.n_linear, self.n_linear))
            gram[:, :q, :q] = np.einsum("ajt,bjt->jab", t * w, t)
            gram[:, diag, diag] += self._pad[c]
            cross = np.einsum("ajt,bt->jab", t * w, b)  # [n_shifts, q, n_b]
            gram[:, :q, q:] = cross
            gram[:, q:, :q] = np.swapaxes(cross, 1, 2)
            gram[:, q:, q:] = (b * w) @ b.T
            out[i] = np.linalg.pinv(gram, hermitian=True)
        return out

    def _pair_tables(self, weights, channels=None):
        """Products of the shifted components, of every pair of shifts, for per-channel
        weights: ``[n, q, q, n_shifts, n_shifts]``, with the basis ``[n, q, n_shifts, n_b]``
        and the basis Gram matrices ``[n, n_b, n_b]``.
        """
        channels = np.arange(len(self._channels)) if channels is None else channels
        q, n_s = self.n_columns, len(self.grid)
        n_b = self.n_linear - q
        pairs = np.empty((len(channels), q, q, n_s, n_s))
        cross = np.empty((len(channels), q, n_s, n_b))
        gram = np.empty((len(channels), n_b, n_b))
        cache = {}
        for i, c in enumerate(channels):
            t = self._templates[self._group_of[c]]
            b = self._basis[0 if self._shared else c]
            w = weights[i]
            key = (self._group_of[c], w.tobytes())
            if key not in cache:
                cache[key] = np.einsum("ait,bjt->abij", t * w, t)
            pairs[i] = cache[key]
            cross[i] = (t * w) @ b.T
            gram[i] = (b * w) @ b.T
        return pairs, cross, gram

    def _basis_dot(self, x):
        """Products of each channel's rows ``x`` [n_channels, ..., n_times] with its basis."""
        if self._shared:
            return x @ self._basis[0].T
        if x.ndim == 2:
            return np.matmul(self._basis, x[:, :, None])[..., 0]
        return np.matmul(x, np.swapaxes(self._basis, 1, 2))

    def _ascend(self, start, r, rb, tables):
        """Per-component shifts by coordinate ascent on the grid from the common shift
        ``start``: grid indices [n_channels, q] and the explained sum of squares of every
        grid shift of each component, the others at their final shift.

        A step moves one component over the grid, the others fixed: the explained sum of
        squares is that of the other columns plus the gain of the moved one,
        ``(r_k - c' A^-1 r_o)^2 / (g_k - c' A^-1 c)`` (partitioned least squares, with ``A``
        and ``r_o`` the normal equations of the other columns and ``c`` their products with
        the moved one).
        """
        pairs, cross, bb = tables
        n_ch, q, n_s = r.shape
        n_b = self.n_linear - q
        n_o = q - 1
        rows = np.arange(n_ch)[:, None]
        candidates = np.arange(n_s)
        hi = np.where(self._allowed, candidates, -1).max(axis=2)
        lo = np.where(self._allowed, candidates, n_s).min(axis=2)
        current = np.clip(start[:, None], lo, hi)
        comps = np.arange(q)
        own = pairs[:, comps, comps][..., candidates, candidates] + self._pad[..., None]
        profile = np.empty((n_ch, q, n_s))
        gram = np.empty((n_ch, n_o + n_b, n_o + n_b))
        rhs = np.empty((n_ch, n_o + n_b))
        products = np.empty((n_ch, n_s, n_o + n_b))
        diag = np.arange(n_o)
        if n_b:
            gram[:, n_o:, n_o:] = bb
            rhs[:, n_o:] = rb
        # A component is evaluated again only when another one has moved since its last step
        stale = np.ones(q, bool)
        for _ in range(self.MAX_SWEEPS):
            if not stale.any():
                break
            for k in range(q):
                if not stale[k]:
                    continue
                others = np.delete(comps, k)
                j = current[:, others]  # [n_ch, n_o]
                gram[:, :n_o, :n_o] = pairs[
                    rows[:, :, None],
                    others[None, :, None],
                    others[None, None, :],
                    j[:, :, None],
                    j[:, None, :],
                ]
                gram[:, diag, diag] += self._pad[:, others]
                rhs[:, :n_o] = r[rows, others[None], j]
                products[..., :n_o] = pairs[
                    rows[:, :, None],
                    k,
                    others[None, None, :],
                    candidates[None, :, None],
                    j[:, None, :],
                ]
                if n_b:
                    tb = cross[rows, others[None], j]  # [n_ch, n_o, n_b]
                    gram[:, :n_o, n_o:] = tb
                    gram[:, n_o:, :n_o] = np.swapaxes(tb, 1, 2)
                    products[..., n_o:] = cross[:, k]
                try:
                    inverse = np.linalg.inv(gram)
                except np.linalg.LinAlgError:
                    inverse = np.linalg.pinv(gram, hermitian=True)
                u = np.einsum("cij,cj->ci", inverse, rhs)
                num = r[:, k] - np.einsum("csi,ci->cs", products, u)
                den = own[:, k] - np.einsum("csi,csi->cs", products @ inverse, products)
                ok = den > 1e-12 * np.abs(own[:, k])
                gain = np.where(ok, num * num / np.where(ok, den, 1.0), 0.0)
                explained = np.einsum("ci,ci->c", rhs, u)[:, None] + gain
                explained = np.where(self._allowed[:, k], explained, -np.inf)
                profile[:, k] = explained
                best = np.argmax(explained, axis=1)
                stale[k] = False
                if (best != current[:, k]).any():
                    current[:, k] = best
                    stale[others] = True
        return current, profile

    ###############################################################################################
    ## Fitting

    def fit(self, window_data, baseline_data=None):
        """Fit one pulse.

        Parameters
        ----------
        window_data : array [n_channels, n_segment]
            The samples ``segment_offsets`` (the fit window, and the anchors of the
            artifact interpolation when it overlaps the window) of the modelled channels
            (``ch_names``, in order). Non-finite samples are left out of the fit.
        baseline_data : array [n_channels, n_window], optional
            The baseline segment of the same pulse, for ``baseline_projection``.

        Returns
        -------
        dict of arrays
            ``shift`` (s) and ``scale`` (1 = the channel template) of each component,
            ``amplitude`` and ``latency`` (all [n_channels, n_components], NaN beyond a
            channel's components); per channel: ``r2`` (over the samples around the
            components), ``at_bound`` (a shift is at its bound), ``n_valid`` (samples used),
            ``projection`` and ``baseline_projection`` (the window and the baseline on the
            unit-norm unshifted template; NaN without ``baseline_data``) and ``finite`` (no
            left-out sample).
        """
        y = np.asarray(window_data, float)
        n_ch, n_w = len(self._channels), len(self.tw)
        q, n_s = self.n_columns, len(self.grid)
        n_seg = len(self._segment_times)
        if y.shape != (n_ch, n_seg):
            raise DataError(f"window_data must be [{n_ch}, {n_seg}], got {list(y.shape)}.")
        if self.artifact_window is not None:
            y, _ = interpolate_window(
                self._segment_times, y, self.artifact_window, INTERP_BUFFER, "linear"
            )
        y = y[:, self._in_window]
        finite = np.isfinite(y)
        all_finite = finite.all(axis=1)
        y = np.where(finite, y, 0.0)
        clean = bool(all_finite.all())
        if clean:
            weights = self.weights[None]
            inverse = self._inverse
        else:
            weights = np.where(finite, self.weights, 0.0)
            inverse = self._inverse.copy()
            bad = np.flatnonzero(~all_finite)
            inverse[bad] = self._normal_inverse(weights[bad], bad)
        yw = y * weights

        # Grid: explained sum of squares of every common shift
        r = np.empty((n_ch, q, n_s))
        for g, idx in enumerate(self._groups):
            r[idx] = (yw[idx] @ self._templates[g].reshape(q * n_s, n_w).T).reshape(-1, q, n_s)
        rb = self._basis_dot(yw) if self.n_linear > q else np.zeros((n_ch, 0))
        rhs = np.empty((n_ch, n_s, self.n_linear))
        rhs[..., :q] = np.swapaxes(r, 1, 2)
        rhs[..., q:] = rb[:, None]
        beta = np.matmul(inverse, rhs[..., None])[..., 0]
        explained = np.einsum("cjm,cjm->cj", rhs, beta)
        best = np.argmax(explained, axis=1)
        if self._descent:
            tables = self._tables
            if not clean:
                tables = tuple(t.copy() for t in tables)
                fresh = self._pair_tables(weights[bad], bad)
                for full, part in zip(tables, fresh):
                    full[bad] = part
            best_k, profile = self._ascend(best, r, rb, tables)
            frac = _parabolic(profile.reshape(n_ch * q, n_s), best_k.reshape(-1))
            shift = (best_k - self.n_grid + frac.reshape(n_ch, q)) / self.sfreq
        else:
            frac = _parabolic(explained, best)
            shift = np.repeat(((best - self.n_grid + frac) / self.sfreq)[:, None], q, axis=1)
        shift = np.clip(shift, -self._bound, self._bound)

        # Refined shifts: the components evaluated there, and their least-squares scales
        curve = np.zeros((n_ch, q, n_w))
        for g, idx in enumerate(self._groups):
            ch = self._channels[idx[0]]
            if q == 1:
                curve[idx, 0] = ch.curves(self.tw, shift[idx, 0])
            else:
                for k in range(len(ch.shapes)):
                    curve[idx, k] = ch.curves(self.tw, shift[idx, k], component=k)
        cw = curve * weights[:, None]
        diag = np.arange(q)
        gram = np.empty((n_ch, self.n_linear, self.n_linear))
        gram[:, :q, :q] = np.einsum("cat,cbt->cab", cw, curve)
        gram[:, diag, diag] += self._pad
        target = np.empty((n_ch, self.n_linear))
        target[:, :q] = np.einsum("cat,ct->ca", cw, y)
        if self.n_linear > q:
            cross = self._basis_dot(cw)  # [n_ch, q, n_b]
            gram[:, :q, q:] = cross
            gram[:, q:, :q] = np.swapaxes(cross, 1, 2)
            gram[:, q:, q:] = (
                self._basis_gram
                if clean
                else np.matmul(self._basis * weights[:, None, :], np.swapaxes(self._basis, 1, 2))
            )
            target[:, q:] = rb
        coef = _solve(gram, target)
        fitted = np.einsum("ca,cat->ct", coef[:, :q], curve)
        if self.n_linear > q:
            fitted += (
                coef[:, q:] @ self._basis[0]
                if self._shared
                else np.matmul(coef[:, None, q:], self._basis)[:, 0]
            )
        if clean and self._shared:
            keep = self._keep[0]
            count = np.full(n_ch, keep.sum())
            resid = y - fitted
            rss = (resid * resid) @ keep
            sum_y = y @ keep
            sum_y2 = (y * y) @ keep
        else:
            keep = np.broadcast_to(((weights > 0) & self._support).astype(float), (n_ch, n_w))
            count = keep.sum(axis=1)
            resid = y - fitted
            rss = np.einsum("ct,ct->c", keep, resid * resid)
            sum_y = np.einsum("ct,ct->c", keep, y)
            sum_y2 = np.einsum("ct,ct->c", keep, y * y)
        with np.errstate(divide="ignore", invalid="ignore"):
            tss = sum_y2 - sum_y**2 / count
            r2 = np.where(tss > 0, 1 - rss / tss, np.nan)
        absent = np.isnan(self._lat0)
        k_max = absent.shape[1]
        at_bound = (np.abs(shift) >= self._bound - 1e-12) & (self._pad == 0)
        if q == 1:
            shift = np.repeat(shift, k_max, axis=1)
            scale = np.repeat(coef[:, :1], k_max, axis=1)
        else:
            scale = coef[:, :q].copy()
        shift = np.where(absent, np.nan, shift)
        scale = np.where(absent, np.nan, scale)
        projection = np.einsum("ct,ct->c", self._unit if clean else self._unit * finite, y)
        if baseline_data is None:
            base = np.full(n_ch, np.nan)
        else:
            b = np.asarray(baseline_data, float)
            if b.shape != (n_ch, n_w):
                raise DataError(f"baseline_data must be [{n_ch}, {n_w}].")
            base = np.einsum("ct,ct->c", self._unit, b)
        return {
            "shift": shift,
            "scale": scale,
            "amplitude": self._amp0 * scale,
            "latency": self._lat0 + shift,
            "r2": r2,
            "at_bound": at_bound.any(axis=1),
            "n_valid": np.broadcast_to((weights > 0).sum(axis=1), n_ch).copy(),
            "projection": projection,
            "baseline_projection": base,
            "finite": all_finite,
        }

    def fit_trialset(self, ts):
        """Fit every trial of a TrialSet (onset at t = 0, the sampling frequency of the model).

        Returns the dict of :meth:`fit` with a leading trial axis.
        """
        if not isinstance(ts, TrialSet):
            raise DataError("fit_trialset expects a TrialSet.")
        if abs(ts.sfreq - self.sfreq) > 1e-9 * self.sfreq:
            raise DataError("The TrialSet and the model have different sampling frequencies.")
        zero = int(np.argmin(np.abs(ts.times)))
        if abs(ts.times[zero]) > 0.5 / self.sfreq:
            raise DataError("The TrialSet has no sample at t = 0.")
        index = [ts.ch_names.index(name) for name in self.ch_names]
        i0, i1 = self.segment_offsets
        b0, b1 = self.baseline_offsets
        if zero + i0 < 0 or zero + i1 >= ts.n_times:
            raise DataError(
                "The fit window (with the anchors of the artifact interpolation) "
                "is outside the epochs."
            )
        has_base = zero + b0 >= 0
        data = ts.data[:, index]
        if ts.masks is not None:
            data = np.where(ts.masks[:, index], np.nan, data)
        results = [
            self.fit(
                trial[:, zero + i0 : zero + i1 + 1],
                trial[:, zero + b0 : zero + b1 + 1] if has_base else None,
            )
            for trial in data
        ]
        return {key: np.stack([res[key] for res in results]) for key in results[0]}
