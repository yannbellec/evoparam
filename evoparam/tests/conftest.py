"""Configuration file for pytest."""

import importlib.util


def pytest_configure(config):
    if importlib.util.find_spec("matplotlib") is not None:
        import matplotlib

        matplotlib.use("agg")
