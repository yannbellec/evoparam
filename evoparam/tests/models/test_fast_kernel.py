# SPDX-License-Identifier: Apache-2.0
"""The compiled fit of ``models/_fast.py`` (extra ``fast``, numba) against the NumPy path.

REQ-115 work: the kernel fits the structures it covers (Gaussian and skewed-Gaussian
components; exponential, constant, linear or no background; white, AR and strided
whitening; with and without prior rows) to the optimum of the NumPy path within the
solver tolerance, and every other case takes the NumPy path unchanged: an uncovered
structure, a design the kernel refuses (not clearly of full rank), and a machine without
numba.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.special import erfc

from evoparam.models import _fast, fit_response
from evoparam.models import engine as eng
from evoparam.shapes import get_background, get_shape

from .helpers import TIMES, gauss, noise

# Fit window (s) of the kernel tests.
WINDOW_START = 0.02
WINDOW_END = 0.6
# Solver tolerance under which the iterations converge tightly.
TIGHT_TOLERANCE = 1e-6

SFREQ = 2048.0
REQ115_TIMES = (np.arange(2048) - 1022) / SFREQ
SETTINGS = SimpleNamespace(loss="linear", max_iter=2000)


def _kernel() -> Any:
    pytest.importorskip("numba")
    kernel = _fast.kernel()
    assert kernel is not None
    return kernel


def _structure(
    shapes: Any, background: Any, whitening: Any, prior: Any, ordered: Any = True
) -> Any:
    rng = np.random.default_rng(3)
    tw = TIMES[(TIMES >= WINDOW_START) & (TIMES <= WINDOW_END)]
    y = gauss(0.1, -5, 0.03, tw) + gauss(0.3, 3, 0.06, tw) + 0.4 + noise(rng, 0.8, n=len(tw))
    valid = np.ones(len(tw), bool)
    scale = np.full(len(tw), 0.7)
    if whitening == "white":
        white = eng.Whitener(scale, np.array([1.0]), 1.0, valid)
    elif whitening == "ar":
        a, innovation, acov = eng.ar_model(noise(rng, 0.8, n=400), return_acov=True)
        white = eng.Whitener(scale, a, innovation, valid, acov=acov)
    else:
        a, innovation, acov = eng.ar_model(noise(rng, 0.8, n=200), return_acov=True)
        white = eng.StridedWhitener(scale, a, innovation, valid, 2, phase=1, acov=acov)
    return eng._Structure(
        [get_shape(name) for name in shapes],
        get_background(background),
        tw,
        white,
        y,
        prior=2.0 if prior else None,
        scale=30.0 if prior else None,
        edge=0.03,
        level=0.1,
        ordered=ordered,
        memo={},
    )


def _start(struct: Any) -> Any:
    starts = {"gaussian": [0.3, 0.025], "skewed_gaussian": [0.1, 0.012, 1.5]}
    theta = np.concatenate([starts[shape.name] for shape in struct.shapes] + [np.zeros(0)])
    return np.r_[theta, [0.08] * struct.n_bg]


def _both_paths(monkeypatch: Any, struct: Any, theta0: Any, tol: Any) -> tuple[Any, Any]:
    monkeypatch.setattr(eng, "_FAST", "off")
    reference = eng._optimise(struct, theta0, SETTINGS, None, tol)
    monkeypatch.setattr(eng, "_FAST", "auto")
    compiled = eng._optimise(struct, theta0, SETTINGS, None, tol)
    return reference, compiled


###################################################################################################
###################################################################################################


def test_vectorised_exp_within_one_ulp_of_numpy() -> None:

    _kernel()
    rng = np.random.default_rng(0)
    x = np.r_[
        -rng.uniform(0, 1, 2000),
        -rng.uniform(0, 50, 2000),
        -rng.uniform(0, 708, 2000),
        rng.uniform(0, 5, 200),
        0.0,
        -708.39,
    ]
    out = np.empty_like(x)
    _fast._vexp(x, out, np.empty(len(x), np.int64))
    np.testing.assert_allclose(out, np.exp(x), rtol=4.5e-16, atol=0)
    # below ln(2^-1022) the result is 0 (NumPy's subnormals, at most 2.2e-308)
    low = np.array([-708.4, -720.0, -745.2, -800.0, -np.inf])
    out = np.empty_like(low)
    _fast._vexp(low, out, np.empty(len(low), np.int64))
    assert np.all(out == 0.0)


def test_cephes_erfc_matches_scipy() -> None:

    _kernel()
    rng = np.random.default_rng(1)
    u = np.r_[rng.uniform(-6, 6, 4000), rng.uniform(-30, 30, 2000), 0.0, 1.0, -1.0, 8.0, -8.0]
    out, tmp = np.empty_like(u), np.empty_like(u)
    _fast._verfc(u, out, tmp, np.empty(len(u), np.int64))
    np.testing.assert_allclose(out, erfc(u), rtol=2e-15, atol=1e-300)


@pytest.mark.parametrize("background", ["exponential", "constant", "linear", "none"])
@pytest.mark.parametrize("whitening", ["white", "ar", "strided"])
@pytest.mark.parametrize("prior", [True, False])
def test_compiled_optimum_is_the_numpy_one(
    monkeypatch: Any, background: Any, whitening: Any, prior: Any
) -> None:

    _kernel()
    struct = _structure(["skewed_gaussian", "gaussian"], background, whitening, prior)
    assert eng._compiled_args(struct) is not None
    for tol in (1e-8, 1e-4):
        reference, compiled = _both_paths(monkeypatch, struct, _start(struct), tol)
        assert reference[3]
        assert compiled[3]
        # the same optimum: within the solver tolerance where the iterations converge
        # tightly, and on the same path at the loose tolerance
        rtol = 1e-6 if tol < TIGHT_TOLERANCE else 1e-3
        np.testing.assert_allclose(compiled[0], reference[0], rtol=rtol, atol=1e-9)
        np.testing.assert_allclose(compiled[1], reference[1], rtol=rtol, atol=1e-6)
        rss_ref, rss = np.sum(reference[2] ** 2), np.sum(compiled[2] ** 2)
        assert abs(rss - rss_ref) <= (1e-9 if tol < TIGHT_TOLERANCE else 1e-4) * rss_ref


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_compiled_fit_is_the_numpy_fit(monkeypatch: Any, seed: Any) -> None:
    """The REQ-115 waveform: the same components, parameters within 1e-6 (as P4)."""

    _kernel()
    rng = np.random.default_rng(seed)
    y = (
        -100 * np.exp(-0.5 * ((REQ115_TIMES - 0.025) / 0.006) ** 2)
        - 60 * np.exp(-0.5 * ((REQ115_TIMES - 0.15) / 0.04) ** 2)
        + rng.normal(0, 5, len(REQ115_TIMES))
    )
    fits = {}
    for mode in ("off", "auto"):
        monkeypatch.setattr(eng, "_FAST", mode)
        fits[mode] = fit_response(
            REQ115_TIMES, y, window=(0.009, 0.5), baseline=(-0.5, -0.02), preset="ccep"
        )
    reference, compiled = fits["off"], fits["auto"]
    assert compiled.status == reference.status
    assert compiled.warnings == reference.warnings
    assert [c.shape for c in compiled.components] == [c.shape for c in reference.components]
    for new, old in zip(compiled.components, reference.components, strict=True):
        np.testing.assert_allclose(new.params, old.params, rtol=1e-6, atol=1e-10)
        for name in ("latency", "amplitude", "fwhm", "onset", "offset", "area"):
            np.testing.assert_allclose(getattr(new, name), getattr(old, name), rtol=1e-6)


def test_rank_deficient_design_takes_the_numpy_path(monkeypatch: Any) -> None:
    """Two equal Gaussians and no prior rows (they would make the design of full rank): the
    kernel stops (FALLBACK) and the NumPy path fits from the start, with its SVD route; the
    result is the NumPy one exactly."""

    kernel = _kernel()
    struct = _structure(["gaussian", "gaussian"], "exponential", "ar", False, ordered=False)
    theta0 = np.r_[0.1, 0.02, 0.1, 0.02, 0.08]
    spec = eng._compiled_args(struct)
    assert spec is not None
    lower, upper = struct.bounds()
    status = kernel((theta0, lower, upper), np.zeros(0, np.int64), spec, 1e-8, 100)[0]
    assert status == _fast.FALLBACK
    reference, compiled = _both_paths(monkeypatch, struct, theta0, 1e-8)
    for a, b in zip(reference, compiled, strict=True):
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b


def test_uncovered_structure_takes_the_numpy_path() -> None:

    struct = _structure(["skewed_gaussian", "exgaussian"], "exponential", "ar", True)
    assert eng._compiled_args(struct) is None
    struct = _structure(["gaussian"], "double_exponential", "ar", True)
    assert eng._compiled_args(struct) is None


def test_without_numba_the_fit_is_the_numpy_fit(monkeypatch: Any) -> None:

    rng = np.random.default_rng(4)
    y = -80 * np.exp(-0.5 * ((REQ115_TIMES - 0.03) / 0.008) ** 2) + rng.normal(
        0, 5, len(REQ115_TIMES)
    )

    def fitted() -> Any:
        return fit_response(
            REQ115_TIMES, y, window=(0.009, 0.5), baseline=(-0.5, -0.02), preset="ccep"
        )

    monkeypatch.setattr(eng, "_FAST", "off")
    reference = fitted()
    monkeypatch.setattr(eng, "_FAST", "auto")
    monkeypatch.setattr(_fast, "kernel", lambda: None)  # what kernel() returns without numba
    without = fitted()
    assert [c.shape for c in without.components] == [c.shape for c in reference.components]
    for new, old in zip(without.components, reference.components, strict=True):
        np.testing.assert_array_equal(new.params, old.params)
        assert new.latency == old.latency
        assert new.amplitude == old.amplitude


def test_a_numba_that_does_not_match_numpy_falls_back(monkeypatch: Any) -> None:
    """A numba installed for another numpy fails with an error other than ImportError: the
    engine takes the NumPy path, with one warning (Colab after pip --force-reinstall)."""

    import evoparam.utils

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Numba needs NumPy 2.2 or less")

    monkeypatch.setattr(evoparam.utils, "require", broken)
    monkeypatch.setattr(_fast, "_COMPILED", {})
    with pytest.warns(RuntimeWarning, match="NumPy path"):
        assert _fast.kernel() is None
    assert _fast.kernel() is None
