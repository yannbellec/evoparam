# SPDX-License-Identifier: Apache-2.0
"""The time-axis helpers of ``evoparam.shapes._base``: the window cache, the fine feature
grid and the search on that grid without building it (REQ-298, coverage of the shapes)."""

import math

import numpy as np

from evoparam.shapes import _base


def test_window_info_cache_is_bounded_and_exact() -> None:
    _base._WINDOW_CACHE.clear()
    for k in range(_base._WINDOW_CACHE_SIZE + 5):
        times = np.arange(10) / (1000.0 + k)
        start, end, length, sfreq = _base.window_info(times)
        assert (start, end) == (0.0, float(times[-1]))
        assert math.isclose(length, float(times[-1]))
        assert math.isclose(sfreq, 1000.0 + k)
        assert len(_base._WINDOW_CACHE) <= _base._WINDOW_CACHE_SIZE
    # a cached axis gives the same answer
    times = np.arange(10) / 1000.0
    assert _base.window_info(times) == _base.window_info(times.copy())


def test_feature_grid_is_ten_times_finer_around_the_window() -> None:
    times = np.arange(0, 0.1, 0.001)
    grid, step, bounds = _base.feature_grid(times, 1000.0, (0.02, 0.06))
    assert math.isclose(step, 1e-4)
    assert bounds == (0.02, 0.06)
    assert math.isclose(grid[0], 0.02 - 0.04)
    assert grid[-1] >= 0.06 + 0.04 - 1e-12
    origin, step2, n_points, _ = _base._grid_spec(times, 1000.0, (0.02, 0.06))
    assert np.array_equal(grid, origin + step2 * np.arange(n_points, dtype=np.float64))


def test_grid_search_matches_numpy_searchsorted() -> None:
    times = np.arange(0, 0.3, 1 / 2048)
    origin, step, n_points, _ = _base._grid_spec(times, 2048.0, (0.012, 0.3))
    grid = origin + step * np.arange(n_points, dtype=np.float64)
    rng = np.random.default_rng(0)
    on_grid = grid[rng.integers(0, n_points, 400)]
    # values on the grid points, a hair off them, outside the grid and non-finite
    values = np.r_[
        on_grid,
        np.nextafter(on_grid, np.inf),
        np.nextafter(on_grid, -np.inf),
        rng.uniform(grid[0] - 1, grid[-1] + 1, 400),
        [-np.inf, np.inf, np.nan, grid[0], grid[-1]],
    ]
    for side, right in (("left", False), ("right", True)):
        expected = np.searchsorted(grid, values, side=side)
        found = [_base._grid_search(origin, step, n_points, float(v), right) for v in values]
        assert np.array_equal(found, expected), side
