# SPDX-License-Identifier: Apache-2.0
"""Acceptance threshold of a new component calibrated on the noise model of the fit.

The rule of spec §5.5.4 accepts a component when BIC falls by ``delta``. Whether a fall of
that size is unusual on pure noise depends on what the fit can search over (the window
length, the range of widths, the whitening filter) and on how well the noise spectrum is
known; a fixed ``delta`` gives different false-response rates for different noises
(on the simulations of REQ-112: 1.9 % of white-noise windows and 5 to 6 % of AR(1)
rho = 0.9 windows at ``delta = 0``, CHANGELOG "Calibrated acceptance"). With
``alpha`` set, the threshold is instead derived from the noise model of each fit:

1. **Look-elsewhere (scan) level.** On pure noise, the deviance gained by the best
   component is the maximum over latency and width of ``Z(mu, s)^2``, ``Z`` the
   projection of the whitened noise on the whitened, normalised template (a smooth
   Gaussian random field: the nuisance parameters exist only under the alternative, the
   case where the chi-square asymptotics of a likelihood-ratio test fail, Davies 1987).
   Its tail is approximated by the expected Euler characteristic of the excursion set
   (Adler 1981; Worsley 1994; Siegmund and Worsley 1995, "Testing for a signal with
   unknown location and scale in a stationary Gaussian random field"):
   ``P(max |Z| >= u) ~ 2 [Phi(-u) + L1 rho1(u) + L2 rho2(u)]`` with
   ``rho1 = exp(-u^2/2) / (2 pi)``, ``rho2 = u exp(-u^2/2) / (2 pi)^(3/2)`` and the
   Lipschitz-Killing curvatures ``L1``, ``L2`` of the (latency, log width) rectangle in
   the metric of the field, computed numerically from the whitened Gaussian templates of
   this fit (the background projected out). ``c0 = u^2`` with ``P = alpha``.
2. **Uncertainty of the noise spectrum.** The whitening filter is estimated on the
   baseline; an error of its spectrum in the band of a component scales that component's
   null deviance by ``exp(eps)``, ``eps`` approximately normal with variance ``v`` given
   by the asymptotic covariance of the autoregressive coefficients (``sigma_e^2
   Gamma_p^{-1} / n``) through the gradient of the log spectrum, weighted by the
   component's whitened power spectrum minus the flat spectrum of the residual (the
   criterion's own variance estimate absorbs a uniform error). The threshold that keeps
   the tail of ``exp(eps) chi^2`` at the level of ``c0`` (saddle point of
   ``exp(-c e^{-eps}/2 - eps^2/(2 v))``) is ``c = 2 e* exp(e*) / v`` with
   ``e* = sqrt(1 + c0 v) - 1`` (``c -> c0 + c0^2 v / 4`` for small ``v``). On slow,
   strongly coloured noise (AR(1) rho = 0.9) the low-frequency spectrum is known to about
   40 %, and broad components need a larger deviance; on white noise ``v`` is about zero.

3. **Change of the noise spectrum over time.** The asymptotic covariance holds for a
   stationary noise; a recording whose spectrum drifts (the pre-stimulus windows of
   ds004080) has, between the baseline and the window, spectral differences larger than
   the sampling error. The baseline is cut into ``SPECTRAL_SEGMENTS`` consecutive parts
   (three: each about as long as the window on ds004080), an AR model of the same order is
   fitted on each, and the variance of their log spectra along the component's direction
   (the same weighting as in 2) beyond their own sampling error (``m`` times that of the
   whole baseline) is added to ``v`` (a moment estimate, never negative). On a stationary
   noise the excess is the positive part of a scaled chi-square with ``m - 1`` degrees of
   freedom: about 1.1 times term 2 on average for ``m = 3``; on ds004080 it is what keeps the
   pre-stimulus windows silent (CHANGELOG "Calibrated acceptance: the change of the noise
   spectrum").

A component with ``p`` parameters (amplitude included) is accepted when its deviance
``n ln(RSS_K / RSS_{K+1})`` is at least ``c + (p - 3) ln n``: the scan level covers a
location, a width and an amplitude, and every further shape parameter pays the BIC price
``ln n``. In the units of the criterion (BIC) this is a fall of at least
``delta_eff = c - 3 ln n``.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from evoparam._typing import Any, NDArray

__all__ = ["Calibration", "scan_level"]

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
# widths sampled (log scale) for the curvatures of the scan field
N_WIDTHS = 48
# parameters of the reference scan: location, width and amplitude
P_SCAN = 3
# parts of the baseline whose AR spectra measure how much the noise spectrum changes over
# time (the window is about one part long on ds004080: 1.5 s of baseline, 0.48 s window)
SPECTRAL_SEGMENTS = 3
# fewest parts with which a spread can be measured
MIN_SEGMENTS = 2
# fewest samples per AR coefficient in a part
SEGMENT_SAMPLES_PER_ORDER = 10
# the bracket of the scan level stops growing here (u = 50: a tail far below any alpha)
MAX_LEVEL = 50.0


def _ec(u: float, l1: float, l2: float) -> float:
    e = np.exp(-0.5 * u * u)
    return float(2.0 * (ndtr(-u) + l1 * e / (2 * np.pi) + l2 * u * e / (2 * np.pi) ** 1.5))


def scan_level(alpha: float, l1: float, l2: float) -> float:
    """Return ``u^2`` with ``P(max |Z| >= u) = alpha`` (expected Euler characteristic)."""
    hi = 2.0
    while _ec(hi, l1, l2) > alpha and hi < MAX_LEVEL:
        hi *= 1.5
    return float(brentq(lambda u: _ec(u, l1, l2) - alpha, 0.0, hi, xtol=1e-10)) ** 2


def _curvatures(white: Any, tw: Any, others: Any, fwhm_min: float, fwhm_max: float) -> Any:
    """Lipschitz-Killing curvatures (L1, L2) of the Gaussian scan field of one fit."""
    n = len(tw)
    dt = float(np.median(np.diff(tw))) if n > 1 else 1.0
    t = (np.asarray(tw, float) - tw[0]) / dt  # samples
    mu0 = 0.5 * (t[0] + t[-1])
    s_lo = np.log(fwhm_min / dt / FWHM_PER_SIGMA)
    s_hi = np.log(max(fwhm_max, fwhm_min * 1.0001) / dt / FWHM_PER_SIGMA)
    grid = np.linspace(s_lo, s_hi, N_WIDTHS)
    h = 1e-3
    cols = []
    for s in grid:
        sig = np.exp(s)
        for mu, sg in (
            (mu0, sig),
            (mu0 + h, sig),
            (mu0 - h, sig),
            (mu0, sig * np.exp(h)),
            (mu0, sig * np.exp(-h)),
        ):
            cols.append(np.exp(-0.5 * ((t - mu) / sg) ** 2))
    w = white(np.column_stack(cols))
    if others is not None and others.shape[1]:
        q, _ = np.linalg.qr(others)
        w = w - q @ (q.T @ w)
    w = w.reshape(w.shape[0], N_WIDTHS, 5)
    det, lmm, lss = np.zeros(N_WIDTHS), np.zeros(N_WIDTHS), np.zeros(N_WIDTHS)
    for j in range(N_WIDTHS):
        g = w[:, j, 0]
        norm = np.linalg.norm(g)
        if not norm > 0:
            continue
        gh = g / norm
        dmu = (w[:, j, 1] - w[:, j, 2]) / (2 * h)
        dsg = (w[:, j, 3] - w[:, j, 4]) / (2 * h)
        dmu = (dmu - gh * (gh @ dmu)) / norm
        dsg = (dsg - gh * (gh @ dsg)) / norm
        a, b, c = dmu @ dmu, dmu @ dsg, dsg @ dsg
        det[j] = np.sqrt(max(a * c - b * b, 0.0))
        lmm[j], lss[j] = np.sqrt(a), np.sqrt(c)
    ds = grid[1] - grid[0]
    span = float(t[-1] - t[0]) + 1.0
    l2 = span * float(np.trapezoid(det, dx=ds))
    l1 = 0.5 * (span * (lmm[0] + lmm[-1]) + 2.0 * float(np.trapezoid(lss, dx=ds)))
    return l1, l2


def _segment_filters(z: Any, p: int, n_seg: int) -> list[NDArray[Any]]:
    """AR filters ``[1, -phi]`` of order ``p`` fitted on ``n_seg`` consecutive parts of ``z``.

    Least squares on each part (as ``engine.ar_model``); empty when a part is too short for
    the order (fewer than ``SEGMENT_SAMPLES_PER_ORDER`` samples per coefficient).
    """
    z = np.asarray(z, float)
    if p == 0 or n_seg < MIN_SEGMENTS or len(z) < n_seg * SEGMENT_SAMPLES_PER_ORDER * p:
        return []
    out = []
    for raw in np.array_split(z, n_seg):
        part = raw - raw.mean()
        lagged = np.column_stack([part[p - k : len(part) - k] for k in range(1, p + 1)])
        phi = np.linalg.lstsq(lagged, part[p:], rcond=None)[0]
        out.append(np.r_[1.0, -phi])
    return out


def _spectral_cov(a: Any, innovation: float, acov: Any, n_base: int) -> Any:
    """Asymptotic covariance of the AR coefficients (standardised units), or None."""
    p = len(a) - 1
    if p == 0 or acov is None or len(acov) < p or n_base <= p:
        return None
    gamma = np.array([[acov[abs(i - j)] for j in range(p)] for i in range(p)])
    try:
        inv = np.linalg.inv(gamma)
    except np.linalg.LinAlgError:
        return None
    return innovation**2 * inv / n_base


class Calibration:
    """Calibrated acceptance threshold of one fit (see the module docstring)."""

    def __init__(  # noqa: PLR0913, PLR0917  (the noise model of one fit, from the engine)
        self,
        alpha: float,
        white: Any,
        tw: Any,
        others: Any,
        fwhm_min: float,
        fwhm_max: float,
        n_crit: float,
        a: Any,
        innovation: float,
        acov: Any,
        n_base: int,
        baseline: Any = None,
    ) -> None:
        self.alpha = float(alpha)
        self.white = white
        self.l1, self.l2 = _curvatures(white, tw, others, fwhm_min, fwhm_max)
        self.c0 = scan_level(self.alpha, self.l1, self.l2)
        self.ln_n = float(np.log(max(n_crit, P_SCAN + 2.0)))
        self.a = np.asarray(a, float)
        self.cov = _spectral_cov(self.a, innovation, acov, n_base)
        self.segments = (
            _segment_filters(baseline, len(self.a) - 1, SPECTRAL_SEGMENTS)
            if baseline is not None and self.cov is not None
            else []
        )
        self._terms: dict[int, tuple[NDArray[Any], NDArray[Any], NDArray[Any] | None]] = {}
        self.delta0 = self.c0 - P_SCAN * self.ln_n

    def _spectral_terms(self, n: int) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any] | None]:
        """Return the frequency weights of a one-sided FFT and the gradient of the spectrum.

        The FFT has length ``>= 2 n``; the gradient is that of the log spectrum with respect
        to the AR coefficients at its frequencies.
        """
        n_fft = 1 << int(np.ceil(np.log2(max(2 * n, 64))))
        known = self._terms.get(n_fft)
        if known is None:
            n_freq = n_fft // 2 + 1
            weight = np.full(n_freq, 2.0)
            weight[0] = weight[-1] = 1.0
            omega = 2 * np.pi * np.arange(n_freq) / n_fft
            lags = np.arange(1, len(self.a))
            phase = np.exp(-1j * np.outer(omega, lags))
            a_w = self.a[0] + phase @ self.a[1:]
            # d log S / d phi_k = 2 Re(exp(-i w k) / A(w)), A(w) = sum_j a_j exp(-i w j)
            grad = 2.0 * np.real(phase / a_w[:, None])
            spread = None
            if self.segments:
                # log spectra of the parts (up to a constant), centred on their mean
                logs = np.array(
                    [-2.0 * np.log(np.abs(f[0] + phase @ f[1:]) + 1e-300) for f in self.segments]
                )
                spread = logs - logs.mean(axis=0)
            known = self._terms[n_fft] = (weight, grad, spread)
        return known

    def _variance(self, q: NDArray[Any]) -> float:
        """Variance of the log scale error of the deviance along whitened direction ``q``."""
        if self.cov is None:
            return 0.0
        rows = getattr(self.white, "rows", None)
        x = q[rows] if rows is not None else q
        if not np.any(x):
            return 0.0
        weight, grad, spread = self._spectral_terms(len(x))
        n_fft = 2 * (len(weight) - 1)
        power = np.abs(np.fft.rfft(x, n_fft)) ** 2 * weight
        total = float(power.sum())
        if not (np.isfinite(total) and total > 0):
            return 0.0  # a direction too small to carry power: no spectral weighting
        # the component's share of each frequency minus the flat share of the residual
        # (the criterion's own variance estimate absorbs an error common to all frequencies)
        share = power / total - weight / weight.sum()
        c = share @ grad
        v = float(c @ self.cov @ c)
        if spread is not None and np.isfinite(v):
            # the change of the spectrum over time: the spread of the parts' spectra along
            # this direction beyond their own sampling error (each part holds 1/m of the
            # baseline: m times the variance of the whole-baseline estimate)
            m = len(spread)
            eps = spread @ share
            between = float(eps @ eps) / (m - 1)
            v += max(between - m * v, 0.0)
        return v if np.isfinite(v) and v > 0 else 0.0

    @staticmethod
    def inflate(c0: float, v: float) -> float:
        """Deviance whose tail under ``exp(eps) chi^2``, ``eps ~ N(0, v)``, is that of ``c0``."""
        if v <= 0:
            return c0
        e = np.sqrt(1.0 + c0 * v) - 1.0
        return float(2.0 * e * np.exp(e) / v)

    def deviance_needed(self, q: NDArray[Any]) -> float:
        """Deviance a component along whitened direction ``q`` needs (scan level, widened)."""
        return self.inflate(self.c0, self._variance(q))

    def required(self, design_w: NDArray[Any], k: int) -> float:
        """Fall of the criterion needed to accept component ``k`` of a whitened design."""
        q = design_w[:, k]
        others = np.delete(design_w, k, axis=1)
        if others.shape[1]:
            qq, _ = np.linalg.qr(others)
            q = q - qq @ (qq.T @ q)
        # deviance needed: c + (p - 3) ln n; BIC already charges p ln n
        return self.inflate(self.c0, self._variance(q)) - P_SCAN * self.ln_n
