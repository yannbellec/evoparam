# SPDX-License-Identifier: Apache-2.0
"""Utilities shared by the modules of evoparam.

Optional imports, names, interpolation, logging, deprecations.
"""

from ._deprecation import EvoparamDeprecationWarning
from ._estimator import NotFittedError
from ._logging import set_log_file, set_log_level, verbose
from ._optional import require

__all__ = [
    "EvoparamDeprecationWarning",
    "NotFittedError",
    "require",
    "set_log_file",
    "set_log_level",
    "verbose",
]
