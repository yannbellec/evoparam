# SPDX-License-Identifier: Apache-2.0
"""Registry of deprecated public names (spec §9.3, REQ-293; REQ-394).

``REGISTRY = {old_name: (new_name, since, remove_in)}``: ``old_name`` is the name under the
package root (``evoparam.<old_name>``), ``new_name`` the fully qualified replacement
(``"evoparam.module:attribute"``), ``since`` the version that deprecated it and
``remove_in`` the minor version that removes it, two minor versions later.
``evoparam/tests/test_deprecations.py`` checks every entry.

No public name has been renamed or removed since the first public release (there has been
none yet: no public step before the HIIE decision, RELEASING.md), so the registry is empty.
"""

REGISTRY: dict[str, tuple[str, str, str]] = {}
