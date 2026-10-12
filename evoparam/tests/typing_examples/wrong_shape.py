# SPDX-License-Identifier: Apache-2.0
"""Deliberately wrong: this class lacks ``jacobian`` and gives ``n_params`` as a string, so
it does not satisfy ComponentShape; mypy must report the assignment below."""

from evoparam.shapes import ComponentShape


class Broken:
    name: str = "broken"
    param_names: tuple = ("a",)
    n_params: str = "one"
    latency_param: str = "a"

    def evaluate(self, times: object, params: object) -> object:  # noqa: ARG002  (shape protocol: signature shared by every shape)
        return times


shape: ComponentShape = Broken()
