# SPDX-License-Identifier: Apache-2.0
"""Noise that does not fill its band (engine defect found with REQ-375).

A recording whose noise stops well below the Nyquist frequency (upsampled data, an
anti-alias or hardware low-pass well below it) gives an AR noise model with an almost empty
band, and a whitening that amplifies that band by orders of magnitude: on ds004080 run
021448 (recorded at 512 Hz) upsampled to 1024 Hz the ``ccep`` fit found an N1 on 54 of 100
channels instead of 14. The engine now detects the empty band (``engine.noise_stride``) and
judges the fit on every k-th sample, whitened at that rate (``engine.StridedWhitener``), on
the phase that holds the recorded samples (``engine.noise_phase``); a noise that fills its
band takes the previous path unchanged.
"""

from typing import Any

import numpy as np
import pytest
from scipy.signal import resample_poly

from evoparam.models import engine as eng

from .helpers import noise

# Stride at which the fit judges a noise band-limited to half the Nyquist frequency.
HALF_RATE_STRIDE = 2
# N1 latency window (s) of the CCEP preset.
N1_START = 0.009
N1_END = 0.05


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9, 0.99])
@pytest.mark.parametrize("n", [100, 400, 1000])
def test_a_noise_that_fills_its_band_has_stride_one(rho: Any, n: Any) -> None:

    rng = np.random.default_rng(int(rho * 100) + n)
    for _ in range(20):
        assert eng.noise_stride(noise(rng, rho, n=n)) == 1


@pytest.mark.parametrize("factor", [2, 3, 4])
@pytest.mark.parametrize("rho", [0.0, 0.9, 0.99])
def test_upsampled_noise_has_the_stride_of_its_upsampling(factor: Any, rho: Any) -> None:

    rng = np.random.default_rng(factor)
    for _ in range(10):
        z = resample_poly(noise(rng, rho, n=300), factor, 1)
        assert eng.noise_stride(z) == factor


@pytest.mark.parametrize(
    ("sfreq", "cutoff"),
    [
        (256.0, 30.0),
        (256.0, 40.0),
        (1024.0, 30.0),
        (1024.0, 40.0),
        (2048.0, 300.0),
        (2048.0, 500.0),
    ],
)
def test_a_gentle_low_pass_keeps_stride_one(sfreq: Any, cutoff: Any) -> None:
    """A zero-phase Butterworth low-pass of order 2 (the gentler ERPLAB setting) leaves a
    band that is low but not empty: the previous path. Its fits were never better strided
    (CHANGELOG)."""

    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(int(sfreq + cutoff))
    sos = butter(2, cutoff, fs=sfreq, output="sos")
    for _ in range(20):
        z = sosfiltfilt(sos, rng.normal(size=int(sfreq) + 400))[200:-200]
        assert eng.noise_stride(z) == 1


@pytest.mark.parametrize("cutoff", [30.0, 40.0])
def test_a_steep_low_pass_well_below_nyquist_is_strided(cutoff: Any) -> None:
    """Order 4 at 30 or 40 Hz on 256 Hz data (ERP CORE-like): the band above 64 Hz is all
    but empty, and the fit is judged at the rate the noise occupies."""

    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(int(cutoff))
    sos = butter(4, cutoff, fs=256.0, output="sos")
    for _ in range(20):
        z = sosfiltfilt(sos, rng.normal(size=656))[200:-200]
        assert eng.noise_stride(z) >= HALF_RATE_STRIDE


def test_a_short_noise_has_stride_one() -> None:

    z = resample_poly(noise(np.random.default_rng(0), 0.5, n=10), 2, 1)
    assert eng.noise_stride(z) == 1


@pytest.mark.parametrize("factor", [2, 3, 4])
@pytest.mark.parametrize("rho", [0.0, 0.9])
def test_the_phase_is_the_one_that_holds_the_recorded_samples(factor: Any, rho: Any) -> None:
    """An upsampled noise cropped by any number of samples, with a masked stretch: the
    phase chosen is the one of the samples the recording kept."""

    rng = np.random.default_rng(10 * factor + int(10 * rho))
    for crop in range(2 * factor):
        z = resample_poly(noise(rng, rho, n=300), factor, 1)[crop:]
        keep = np.ones(len(z), bool)
        keep[100:117] = False
        index = np.flatnonzero(keep)
        assert eng.noise_phase(z[keep], index, factor) == (-crop) % factor


def test_a_short_noise_has_phase_zero() -> None:

    z = resample_poly(noise(np.random.default_rng(0), 0.5, n=20), 2, 1)[1:]
    assert eng.noise_phase(z, np.arange(len(z)), 2) == 0


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("phase", [0, 1])
def test_the_strided_whitening_is_the_whitening_at_the_lower_rate(exact: Any, phase: Any) -> None:
    """On the kept rows, the strided whitening of a full-rate residual equals the ordinary
    whitening of its every other sample; the other rows are zero."""

    rng = np.random.default_rng(3)
    a, innovation, acov = eng.ar_model(noise(rng, 0.9, n=300), return_acov=True)
    acov = acov if exact else None
    n = 201
    x = rng.normal(size=(n, 3))
    scale = rng.uniform(0.5, 2.0, n)
    valid = np.ones(n, bool)
    valid[[40, 41, 120]] = False
    strided = eng.StridedWhitener(scale, a, innovation, valid, 2, phase=phase, acov=acov)
    low = eng.Whitener(scale[phase::2], a, innovation, valid[phase::2], acov=acov)
    full = strided(x)
    assert np.array_equal(full[phase::2], low(x[phase::2]))
    assert not full[1 - phase :: 2].any()
    assert strided.n == low.n
    assert strided.p == low.p
    assert np.array_equal(strided.into(x.copy(), np.empty_like(x)), full)
    assert np.array_equal(strided(x[:, 0]), full[:, 0])
    if exact:
        # the innovation rows (the evidence of a component) are the same
        innovations = np.sum(low(x[phase::2])[low.p :] ** 2)
        assert np.sum(full[strided.skip :] ** 2) == pytest.approx(innovations)


def _n1(group: Any) -> Any:
    return [
        any(N1_START <= c.latency <= N1_END and c.amplitude < 0 for c in f.components)
        for f in group.fits
    ], sum(len(f.components) for f in group.fits)


def test_an_upsampled_ccep_is_fitted_as_at_its_recorded_rate(monkeypatch: Any) -> None:
    """A simulated CCEP recorded at 512 Hz and upsampled to 1024 Hz: the fit is judged on
    every other sample and finds what it finds at 512 Hz; without the stride the defect
    comes back (2.75 times as many components on this case)."""

    import warnings

    from evoparam import fit, sim
    from evoparam.prep import resample

    ts, _ = sim.simulate_ccep(10, n_channels=8, sfreq=512.0, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        up = resample(ts, 1024.0)
        native = fit(ts, "ccep", existence=False, random_state=0)
        strided = fit(up, "ccep", existence=False, random_state=0)
        monkeypatch.setattr(eng, "MAX_STRIDE", 1)
        plain = fit(up, "ccep", existence=False, random_state=0)
    assert all(f.diagnostics["noise_stride"] == 1 for f in native.fits)
    assert all(f.diagnostics["noise_stride"] == HALF_RATE_STRIDE for f in strided.fits)
    assert all("band_limited_noise:2" in f.warnings for f in strided.fits)
    (n1_native, k_native), (n1_strided, k_strided) = _n1(native), _n1(strided)
    assert n1_strided == n1_native
    assert k_strided <= 1.25 * k_native
    assert _n1(plain)[1] >= 2 * k_native
