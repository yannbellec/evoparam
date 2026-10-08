"""Linear time-frequency transforms: generalised Morse and Morlet wavelets, multitaper, STFT.

Every transform is a set of complex kernels, one per frequency (and per taper for the
multitaper transform), applied by multiplication in the frequency domain on a zero-padded
grid. The coefficients are therefore linear in the signal, which is what makes the
footprint/residual decomposition of :mod:`evoparam.tf.api` exact.

Gain convention: a real sinusoid of amplitude ``A`` at the kernel's frequency gives a
coefficient of modulus ``A`` (analytic kernels with a peak gain of 2 at that frequency).
"""

import numpy as np
from scipy.fft import fft, ifft, next_fast_len

from evoparam.errors import SettingsError

__all__ = ["TRANSFORMS", "Kernels", "make_kernels", "apply_kernels", "morse_spectrum",
           "morse_peak", "log_freqs", "morse_reference_modulus"]

TRANSFORMS = ("morse", "morlet", "multitaper", "stft")
# Fraction of a kernel's energy that defines its temporal support (cone of influence and
# widening of the artifact mask).
SUPPORT_ENERGY = 0.99


def log_freqs(fmin, fmax, n):
    """``n`` frequencies on a log grid from ``fmin`` to ``fmax`` (Hz)."""

    return np.geomspace(float(fmin), float(fmax), int(n))


def morse_peak(gamma, beta):
    """Peak (radian) frequency of the generalised Morse wavelet: ``(beta / gamma)**(1/gamma)``."""

    return (beta / gamma) ** (1.0 / gamma)


def morse_spectrum(omega, gamma, beta):
    """Generalised Morse wavelet in the frequency domain (Lilly and Olhede 2012).

    ``Psi(omega) = 2 * a * omega**beta * exp(-omega**gamma)`` for ``omega > 0`` and 0
    otherwise, with ``a = (e * gamma / beta)**(beta / gamma)`` so that the peak value is 2.
    ``omega`` is dimensionless (radians per unit scale).
    """

    omega = np.asarray(omega, float)
    out = np.zeros_like(omega)
    pos = omega > 0
    w = omega[pos]
    # log form avoids overflow of omega**beta for large beta
    log_a = (beta / gamma) * (1.0 + np.log(gamma) - np.log(beta))
    out[pos] = 2.0 * np.exp(log_a + beta * np.log(w) - w ** gamma)
    return out


class Kernels:
    """Kernel spectra grouped by FFT length, and temporal half-supports (samples).

    ``groups`` is a list of (n_fft, frequency indices, spectra (n_tapers, n_group, n_fft)):
    each frequency is applied on the shortest padded grid that keeps its valid pixels (whose
    kernel support lies inside the epoch) free of circular wrap-around.
    """

    def __init__(self, groups, half_support, freqs, transform, n_tapers, params):
        self.groups = groups
        self.half_support = half_support
        self.freqs = freqs
        self.transform = transform
        self.n_tapers = n_tapers
        self.params = params

    def spectrum(self, j):
        """Spectra (n_tapers, n_fft) of frequency ``j`` on its grid."""

        for _, idx, spectra in self.groups:
            where = np.flatnonzero(np.asarray(idx) == j)
            if where.size:
                return spectra[:, where[0]]
        raise IndexError(j)


def _half_support(kernel_time):
    """Half-width (samples) around lag 0 holding SUPPORT_ENERGY of a centred kernel's energy.

    ``kernel_time`` is circular with lag 0 at index 0 (as returned by an inverse FFT).
    """

    energy = np.abs(kernel_time) ** 2
    n = len(energy)
    lags = np.minimum(np.arange(n), n - np.arange(n))
    order = np.argsort(lags, kind="stable")
    cum = np.cumsum(energy[order])
    total = cum[-1]
    if total <= 0:
        return 0
    idx = int(np.searchsorted(cum, SUPPORT_ENERGY * total))
    return int(lags[order][min(idx, n - 1)])


def _centred_spectrum(h, n_fft):
    """Spectrum of a centred time-domain kernel on an ``n_fft`` grid."""

    m = len(h)
    buf = np.zeros(n_fft, complex)
    centre = m // 2
    buf[:m - centre] = h[centre:]
    buf[n_fft - centre:] = h[:centre]
    return fft(buf)


def make_kernels(transform, freqs, sfreq, n_times, *, gamma=3.0, p2=30.0, n_cycles=7.0,
                 time_bandwidth=4.0, stft_window=0.2):
    """Kernels of one transform for a frequency grid.

    Parameters
    ----------
    transform : {"morse", "morlet", "multitaper", "stft"}
    freqs : ndarray
        Centre frequencies (Hz).
    sfreq : float
    n_times : int
        Length of the signals the kernels will be applied to.
    gamma, p2 : float
        Morse parameters; ``beta = p2 / gamma`` (defaults: gamma 3, beta 10).
    n_cycles : float or ndarray
        Morlet: number of cycles (``sigma_t = n_cycles / (2 pi f)``); multitaper: window
        length in cycles (``T = n_cycles / f``).
    time_bandwidth : float
        Multitaper time-bandwidth product ``2 NW``; ``K = floor(time_bandwidth - 1)`` tapers.
    stft_window : float
        STFT Hann window length (s).
    """

    if transform not in TRANSFORMS:
        raise SettingsError(f"transform must be one of {TRANSFORMS}, not {transform!r}.")
    freqs = np.asarray(freqs, float)
    cycles = np.broadcast_to(np.asarray(n_cycles, float), freqs.shape)
    params = {"transform": transform}
    if transform == "morse":
        beta = p2 / gamma
        peak = morse_peak(gamma, beta)
        params.update(gamma=float(gamma), beta=float(beta), p2=float(p2))

        def spectrum(j, n_fft):
            nu = np.fft.fftfreq(n_fft, 1.0 / sfreq)
            return morse_spectrum(nu * peak / freqs[j], gamma, beta)[None]
    elif transform == "morlet":
        params.update(n_cycles=cycles.tolist())

        def spectrum(j, n_fft):
            nu = np.fft.fftfreq(n_fft, 1.0 / sfreq)
            sigma_t = cycles[j] / (2 * np.pi * freqs[j])
            return np.where(nu > 0, 2.0 * np.exp(-0.5 * (2 * np.pi * (nu - freqs[j])
                                                         * sigma_t) ** 2), 0.0)[None]
    else:
        if transform == "multitaper":
            from scipy.signal.windows import dpss

            nw = time_bandwidth / 2.0
            n_tapers = max(1, int(np.floor(time_bandwidth - 1)))
            lengths = np.maximum(np.round(cycles / freqs * sfreq).astype(int), 3)
            params.update(time_bandwidth=float(time_bandwidth), n_tapers=n_tapers,
                          n_cycles=cycles.tolist())

            def time_kernels(j):
                m = lengths[j]
                tapers = np.atleast_2d(dpss(m, nw, n_tapers))
                t = (np.arange(m) - m // 2) / sfreq
                return 2.0 / np.abs(tapers[0].sum()) * tapers * np.exp(2j * np.pi * freqs[j] * t)
        else:
            from scipy.signal.windows import hann

            m = max(int(round(stft_window * sfreq)), 3)
            win = hann(m, sym=True)
            params.update(stft_window=float(stft_window))

            def time_kernels(j):
                t = (np.arange(m) - m // 2) / sfreq
                return (2.0 / win.sum() * win * np.exp(2j * np.pi * freqs[j] * t))[None]

        def spectrum(j, n_fft):
            return np.stack([_centred_spectrum(h, n_fft) for h in time_kernels(j)])

    # support on a generous grid, then each frequency on the shortest safe grid
    probe = next_fast_len(4 * n_times)
    half = np.array([_half_support(ifft(spectrum(j, probe)[0])) for j in range(len(freqs))])
    lengths_fft = [next_fast_len(int(n_times + 2 * min(h, n_times)) + 1) for h in half]
    groups = []
    for n_fft in sorted(set(lengths_fft)):
        idx = [j for j, n in enumerate(lengths_fft) if n == n_fft]
        spectra = np.stack([spectrum(j, n_fft) for j in idx], axis=1)
        groups.append((n_fft, idx, spectra))
    n_tapers = groups[0][2].shape[0]
    return Kernels(groups, half, freqs, transform, n_tapers, params)


def apply_kernels(kernels, x, crop=None):
    """Complex coefficients of signals ``x`` (..., n_times): (n_tapers, ..., n_freqs, n_out),
    on the samples ``crop`` (a slice; all samples by default)."""

    x = np.asarray(x, float)
    n = x.shape[-1]
    crop = slice(0, n) if crop is None else crop
    n_out = len(range(*crop.indices(n)))
    out = np.empty((kernels.n_tapers,) + x.shape[:-1] + (len(kernels.freqs), n_out), complex)
    for n_fft, idx, spectra in kernels.groups:
        spec = fft(x, n_fft, axis=-1)
        rows = slice(idx[0], idx[-1] + 1) if idx == list(range(idx[0], idx[-1] + 1)) else idx
        for k in range(kernels.n_tapers):
            out[k][..., rows, :] = ifft(spec[..., None, :] * spectra[k], axis=-1)[..., crop]
    return out


def morse_reference_modulus(freq_signal, freq_kernel, gamma, beta):
    """Analytic modulus of the Morse coefficient of a unit cosine at ``freq_signal`` (Hz),
    for the kernel centred at ``freq_kernel``: half the wavelet's gain at that frequency."""

    peak = morse_peak(gamma, beta)
    return 0.5 * float(morse_spectrum(np.array([freq_signal * peak / freq_kernel]),
                                      gamma, beta)[0])

