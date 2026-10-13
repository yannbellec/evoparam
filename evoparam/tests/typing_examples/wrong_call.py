# SPDX-License-Identifier: Apache-2.0
"""Deliberately wrong: the self-test returns a float, not a string."""

import evoparam

label: str = evoparam.selftest()
