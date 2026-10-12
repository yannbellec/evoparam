# SPDX-License-Identifier: Apache-2.0
"""Design columns of the engine and their derivatives, sharing their intermediate results.

The optimiser evaluates the model at a parameter vector and, when it accepts the step, the
Jacobian at the same vector. The shapes' public ``evaluate`` and ``jacobian`` check their
inputs and recompute the same exponentials and error functions; here the built-in shapes
the presets use (Gaussian, skewed Gaussian, exponential background) are evaluated once,
with the formulas of :mod:`evoparam.shapes` written in the same order of operations (the
same floating-point results), and the intermediate arrays are kept for the derivatives.
Any other shape or background goes through its public methods.
"""

import numpy as np
from scipy.special import erfc

from evoparam.shapes.backgrounds import ExponentialBackground
from evoparam.shapes.components import Gaussian, SkewedGaussian, gaussian_factor, skew_mode

__all__ = ["component", "component_derivatives", "background", "background_derivatives"]

_SQRT2 = np.sqrt(2.0)
_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)


def component(shape, tw, params):
    """``shape.evaluate(tw, params)`` and what :func:`component_derivatives` needs."""
    kind = type(shape)
    if kind is Gaussian:
        z = (tw - params[0]) / params[1]
        s = gaussian_factor(z, lambda x: np.exp(-0.5 * x**2))
        return s, (z, s)
    if kind is SkewedGaussian:
        alpha = params[2:3]
        _, c, dc = skew_mode(alpha)
        z = (tw - params[0]) / params[1]
        numerator = gaussian_factor(z, lambda x: np.exp(-0.5 * x**2) * erfc(-alpha * x / _SQRT2))
        return numerator / c, (z, c, dc, numerator)
    return shape.evaluate(tw, params), None


def component_derivatives(shape, tw, params, aux):
    """``shape.jacobian(tw, params)`` (n x p) from the ``aux`` of :func:`component`."""
    kind = type(shape)
    if kind is Gaussian and aux is not None:
        z, s = aux
        sigma = params[1]
        return np.stack([s * z / sigma, s * z**2 / sigma], axis=-1)
    if kind is SkewedGaussian and aux is not None:
        z, c, dc, numerator = aux
        omega, alpha = params[1:2], params[2:3]
        gauss = _SQRT_2_OVER_PI * gaussian_factor(
            z, lambda x: np.exp(-0.5 * x**2 * (1.0 + alpha**2))
        )
        d_z = -z * numerator + alpha * gauss
        d_alpha = z * gauss
        return np.stack(
            [-d_z / (omega * c), -d_z * z / (omega * c), d_alpha / c - numerator * dc / c**2],
            axis=-1,
        )
    return shape.jacobian(tw, params)


def background(bg, tw, nonlinear, out):
    """``bg.basis(tw, nonlinear)`` (n_linear x n) written into ``out``, and what the
    derivatives need.
    """
    if type(bg) is ExponentialBackground:
        lag = tw - bg._t_ref(tw)
        decay = np.exp(-lag / nonlinear[0:1])
        out[0] = 1.0
        out[1] = decay  # out is whitened in place afterwards: a copy
        return (lag, decay)
    out[...] = np.atleast_2d(bg.basis(tw, nonlinear))
    return None


def background_derivatives(bg, tw, nonlinear, coefs, aux):
    """``einsum("l,lnj->nj", coefs, bg.basis_jacobian(tw, nonlinear))``: the derivative of
    the background with the linear coefficients ``coefs`` (n x n_nonlinear).
    """
    if type(bg) is ExponentialBackground and aux is not None:
        lag, decay = aux
        tau = nonlinear[0:1]
        d_decay = decay * lag / tau**2
        return (coefs[1] * d_decay)[:, None]
    dj = bg.basis_jacobian(tw, nonlinear)
    return np.einsum("l,lnj->nj", coefs, dj)
