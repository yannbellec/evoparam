# SPDX-License-Identifier: Apache-2.0
"""scikit-learn conventions without scikit-learn (spec §10.4).

``BaseEstimatorMixin`` gives ``get_params``/``set_params`` (with nested ``step__param``
names), ``__repr__``, ``__sklearn_is_fitted__`` and, when scikit-learn is importable,
``__sklearn_tags__``; ``TransformerMixin`` gives ``fit_transform`` and a no-op
``set_output`` so that the estimators work in ``sklearn.pipeline.Pipeline``.
"""

import inspect
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from ._optional import require

__all__ = [
    "BaseEstimatorMixin",
    "TransformerMixin",
    "NotFittedError",
    "check_is_fitted",
    "validate_array",
]

_E = TypeVar("_E", bound="BaseEstimatorMixin")
_T = TypeVar("_T")


class NotFittedError(ValueError, AttributeError):
    """Raised when an estimator is used before ``fit`` (spec REQ-320).

    When scikit-learn has been imported, the error raised is also an instance of
    scikit-learn's ``NotFittedError``. Code that catches scikit-learn's class has imported
    scikit-learn, so the check at raise time is enough, and ``import evoparam`` never imports
    scikit-learn (light core, REQ-010).
    """


_BOTH: dict[type, type[Exception]] = {}


def _not_fitted_class() -> type[Exception]:
    module = sys.modules.get("sklearn.exceptions")
    base = getattr(module, "NotFittedError", None)
    if base is None or issubclass(NotFittedError, base):
        return NotFittedError
    if base not in _BOTH:
        _BOTH[base] = type(
            "NotFittedError",
            (base, NotFittedError),
            {"__module__": __name__, "__doc__": NotFittedError.__doc__},
        )
    return _BOTH[base]


def check_is_fitted(estimator: Any) -> None:
    if not estimator.__sklearn_is_fitted__():
        raise _not_fitted_class()(
            f"This {type(estimator).__name__} instance is not fitted yet; call fit first."
        )


class BaseEstimatorMixin:
    """Parameters are the keyword arguments of ``__init__``, stored unchanged."""

    @classmethod
    def _get_param_names(cls) -> list[str]:
        signature = inspect.signature(cls.__init__)
        return sorted(
            p.name
            for p in signature.parameters.values()
            if p.name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        out = {}
        for name in self._get_param_names():
            value = getattr(self, name)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                out.update({f"{name}__{k}": v for k, v in value.get_params().items()})
            out[name] = value
        return out

    def set_params(self: _E, **params: Any) -> _E:
        from evoparam.errors import SettingsError

        valid = self._get_param_names()
        nested: dict[str, dict[str, Any]] = {}
        for key, value in params.items():
            name, _, sub = key.partition("__")
            if name not in valid:
                raise SettingsError(
                    f"Invalid parameter {name!r} for {type(self).__name__}; valid: {valid}."
                )
            if sub:
                nested.setdefault(name, {})[sub] = value
            else:
                setattr(self, name, value)
        for name, values in nested.items():
            getattr(self, name).set_params(**values)
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return any(k.endswith("_") and not k.startswith("_") for k in vars(self))

    def __sklearn_tags__(self) -> Any:
        try:
            require(
                "sklearn",
                extra="sklearn",
                caller="evoparam.utils._estimator.BaseEstimatorMixin.__sklearn_tags__",
            )
            from sklearn.base import BaseEstimator
            from sklearn.utils import Tags, TargetTags  # noqa: F401
        except ImportError:  # pragma: no cover - only with scikit-learn
            return None
        from sklearn.utils import TransformerTags

        tags = BaseEstimator.__sklearn_tags__(self)
        tags.target_tags.required = False
        tags.non_deterministic = False
        if hasattr(self, "transform"):
            tags.transformer_tags = TransformerTags()
        return tags

    def __repr__(self) -> str:
        defaults = {
            p.name: p.default for p in inspect.signature(type(self).__init__).parameters.values()
        }
        shown = [
            f"{k}={getattr(self, k)!r}"
            for k in self._get_param_names()
            if getattr(self, k) is not defaults.get(k) and getattr(self, k) != defaults.get(k)
        ]
        return f"{type(self).__name__}({', '.join(shown)})"


class TransformerMixin:
    if TYPE_CHECKING:  # provided by the estimator that mixes this class in
        fit: Callable[..., Any]

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:  # noqa: N803  (scikit-learn API: X)
        return self.fit(X, y, **fit_params).transform(X)

    def set_output(self: _T, *, transform: str | None = None) -> _T:  # noqa: ARG002  (scikit-learn API)
        return self


def validate_array(
    estimator: Any, x: Any, *, reset: bool, ndim: int = 2, allow_nan: bool = False
) -> Any:
    """Check an array input the way scikit-learn does (its messages when it is installed).

    Dense, real, finite, at least 2-D with at least 2 time points, and, after ``fit``, as many
    features (last axis) as seen in ``fit``. Sets ``n_features_in_`` when ``reset``.
    """
    import numpy as np

    try:
        require("sklearn", extra="sklearn", caller="evoparam.utils._estimator.validate_array")
        from sklearn.utils.validation import check_array
    except ImportError:
        check_array = None
    if check_array is not None:
        x = check_array(
            x,
            accept_sparse=False,
            ensure_all_finite=not allow_nan,
            ensure_2d=True,
            allow_nd=ndim > 2,
            dtype="numeric",
        )
    else:
        if hasattr(x, "toarray"):
            raise TypeError("A sparse matrix was passed, but dense data is required.")
        x = np.asarray(x)
        if np.iscomplexobj(x):
            raise ValueError("Complex data not supported.")
        x = x.astype(float)
        if x.ndim < 2:
            raise ValueError("Expected 2D array, got 1D array instead. Reshape your data.")
        if not allow_nan and not np.all(np.isfinite(x)):
            raise ValueError("Input contains NaN or infinity.")
    n_features = int(x.shape[-1])
    if not reset and n_features != getattr(estimator, "n_features_in_", n_features):
        raise ValueError(
            f"X has {n_features} features, but {type(estimator).__name__} is "
            f"expecting {estimator.n_features_in_} features as input."
        )
    if n_features < 2:
        raise ValueError(
            f"Found array with {n_features} feature(s) (shape={x.shape}) while a "
            f"minimum of 2 is required (time points)."
        )
    if reset:
        estimator.n_features_in_ = n_features
    if x.ndim > ndim:
        raise ValueError(f"Expected at most {ndim}-D input, got {x.ndim}-D.")
    return x
