"""Contact names: electrode group and contact number."""

import re

__all__ = ["contact_group", "contact_number"]

_PATTERN = re.compile(r"^(.*?)[\s_\-]*(\d+)$")


def contact_group(name):
    """Electrode group guessed from a contact name: the name without its trailing number.

    For example ``"PT01"`` -> ``"PT"``, ``"LA'12"`` -> ``"LA'"``, ``"Grid_3"`` -> ``"Grid"``;
    a name without a trailing number is its own group.
    """

    match = _PATTERN.match(str(name).strip())
    return match.group(1) if match and match.group(1) else str(name)


def contact_number(name):
    """Trailing contact number of a name (``"PT01"`` -> 1), or None."""

    match = _PATTERN.match(str(name).strip())
    return int(match.group(2)) if match and match.group(1) else None
