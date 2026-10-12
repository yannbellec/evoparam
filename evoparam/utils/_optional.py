# SPDX-License-Identifier: Apache-2.0
"""Import optional dependencies, or raise an ImportError that names the extra to install.

Every package other than numpy and scipy is optional. Functions that need one
call :func:`require` inside their body, so that ``import evoparam`` works with
the two hard dependencies only.
"""

import importlib
import re
from importlib import metadata

__all__ = ["require"]

# Import name -> distribution name, when they differ.
_DISTRIBUTIONS = {"sklearn": "scikit-learn", "yaml": "PyYAML"}


def _version_tuple(text):
    parts = []
    for piece in re.split(r"[.+-]", str(text)):
        match = re.match(r"\d+", piece)
        if not match:
            break
        parts.append(int(match.group()))
    return tuple(parts)


def _installed_version(module, name):
    version = getattr(module, "__version__", None)
    if version is None:
        try:
            version = metadata.version(_DISTRIBUTIONS.get(name, name))
        except metadata.PackageNotFoundError:
            return None
    return str(version)


def require(module, *, extra, min_version=None, caller=None):
    """Import an optional dependency, or raise ImportError with the pip command to run.

    Parameters
    ----------
    module : str
        Import name of the package, for example ``"mne"``.
    extra : str
        Name of the evoparam extra that installs it, for example ``"mne"``.
    min_version : str, optional
        Lowest supported version. An older installed version raises ImportError.
    caller : str, optional
        Name of the function that needs the package, quoted in the message.

    Returns
    -------
    module : ModuleType
        The imported package.

    Raises
    ------
    ImportError
        If the package is missing or too old. The message names the package, the
        version floor and the command ``pip install "evoparam[<extra>]"``.
    """

    who = caller or "This function"
    floor = f" (>= {min_version})" if min_version else ""
    hint = f'Install it with:  pip install "evoparam[{extra}]"'
    try:
        imported = importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(f"{who} needs the optional package '{module}'{floor}.\n{hint}") from exc
    if min_version is not None:
        found = _installed_version(imported, module.split(".")[0])
        if found is not None and _version_tuple(found) < _version_tuple(min_version):
            raise ImportError(
                f"{who} needs the optional package '{module}'{floor}; "
                f"version {found} is installed.\n{hint}"
            )
    return imported
