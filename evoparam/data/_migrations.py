# SPDX-License-Identifier: Apache-2.0
"""Migrations between major versions of the ``.evoparam`` container schema.

``MIGRATIONS`` maps ``(from_major, to_major)`` to a function that receives the
decoded ``structure.json`` dictionary and returns the migrated dictionary. It
is empty while only schema major 1 exists.
"""

MIGRATIONS = {}


def migrate(structure, from_major, to_major):
    """Apply the registered migrations one major version at a time.

    Returns the migrated structure and the list of ``(from, to)`` pairs applied.
    Raises ``KeyError`` when a step has no registered migration.
    """
    applied = []
    major = from_major
    while major < to_major:
        structure = MIGRATIONS[(major, major + 1)](structure)
        applied.append((major, major + 1))
        major += 1
    return structure, applied
