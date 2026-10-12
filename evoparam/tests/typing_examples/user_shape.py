# SPDX-License-Identifier: Apache-2.0
"""A user-defined shape that satisfies the ComponentShape protocol without inheriting
from any evoparam class, and its registration."""

from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from evoparam.shapes import ComponentShape, register_shape

Array = npt.NDArray[np.float64]


class Triangle:
    name: str = "triangle"
    param_names: tuple = ("centre", "half_width")
    n_params: int = 2
    latency_param: str = "centre"
    kind: ClassVar[str] = "component"

    def evaluate(self, times: Array, params: Array) -> Array:
        centre, half = params[..., 0:1], params[..., 1:2]
        return np.clip(1.0 - np.abs(times - centre) / half, 0.0, None)

    def jacobian(self, times: Array, params: Array) -> Array:
        centre, half = params[..., 0:1], params[..., 1:2]
        inside = np.abs(times - centre) < half
        d_centre = np.where(inside, np.sign(times - centre) / half, 0.0)
        d_half = np.where(inside, np.abs(times - centre) / half**2, 0.0)
        return np.stack([d_centre, d_half], axis=-1)

    def default_bounds(self, times: Array, data_range: Any = None) -> tuple[Array, Array]:  # noqa: ARG002  (shape protocol: signature shared by every shape)
        return (np.array([times[0], 1e-3]), np.array([times[-1], 0.2]))

    def initial_guess(self, times: Array, waveform: Array, peak_index: int) -> Array:  # noqa: ARG002  (shape protocol: signature shared by every shape)
        return np.array([times[peak_index], 0.01])

    def features(self, times: Array, params: Array, amplitude: float = 1.0) -> Any:  # noqa: ARG002  (shape protocol: signature shared by every shape)
        return None


shape: ComponentShape = Triangle()


def register() -> None:
    register_shape(shape, overwrite=True)
