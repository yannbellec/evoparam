"""scikit-learn conventions without scikit-learn (spec §10.4).

``BaseEstimatorMixin`` gives ``get_params``/``set_params`` (with nested ``step__param``
names), ``__repr__``, ``__sklearn_is_fitted__`` and, when scikit-learn is importable,
``__sklearn_tags__``; ``TransformerMixin`` gives ``fit_transform`` and a no-op
``set_output`` so that the estimators work in ``sklearn.pipeline.Pipeline``.
"""

import inspect

__all__ = ["BaseEstimatorMixin", "TransformerMixin", "NotFittedError", "check_is_fitted"]


def _sklearn_not_fitted():
    try:
        from sklearn.exceptions import NotFittedError as SkNotFitted
    except ImportError:
        return None
    return SkNotFitted


_base = _sklearn_not_fitted()


class NotFittedError(*((_base,) if _base is not None else (ValueError, AttributeError))):
    """Raised when an estimator is used before ``fit`` (scikit-learn's class when it is
    installed; else a subclass of ValueError and AttributeError)."""


def check_is_fitted(estimator):
    if not estimator.__sklearn_is_fitted__():
        raise NotFittedError(f"This {type(estimator).__name__} instance is not fitted yet; "
                             f"call fit first.")


class BaseEstimatorMixin:
    """Parameters are the keyword arguments of ``__init__``, stored unchanged."""

    @classmethod
    def _get_param_names(cls):
        signature = inspect.signature(cls.__init__)
        return sorted(p.name for p in signature.parameters.values()
                      if p.name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL))

    def get_params(self, deep=True):
        out = {}
        for name in self._get_param_names():
            value = getattr(self, name)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                out.update({f"{name}__{k}": v for k, v in value.get_params().items()})
            out[name] = value
        return out

    def set_params(self, **params):
        from evoparam.errors import SettingsError

        valid = self._get_param_names()
        nested = {}
        for key, value in params.items():
            name, _, sub = key.partition("__")
            if name not in valid:
                raise SettingsError(f"Invalid parameter {name!r} for {type(self).__name__}; "
                                    f"valid: {valid}.")
            if sub:
                nested.setdefault(name, {})[sub] = value
            else:
                setattr(self, name, value)
        for name, values in nested.items():
            getattr(self, name).set_params(**values)
        return self

    def __sklearn_is_fitted__(self):
        return any(k.endswith("_") and not k.startswith("_") for k in vars(self))

    def __sklearn_tags__(self):
        try:
            from sklearn.utils import Tags, TargetTags  # noqa: F401
            from sklearn.base import BaseEstimator
        except ImportError:  # pragma: no cover - only with scikit-learn
            return None
        tags = BaseEstimator.__sklearn_tags__(self)
        tags.target_tags.required = False
        tags.non_deterministic = False
        return tags

    def __repr__(self):
        defaults = {p.name: p.default for p in
                    inspect.signature(type(self).__init__).parameters.values()}
        shown = [f"{k}={getattr(self, k)!r}" for k in self._get_param_names()
                 if getattr(self, k) is not defaults.get(k) and
                 getattr(self, k) != defaults.get(k)]
        return f"{type(self).__name__}({', '.join(shown)})"


class TransformerMixin:
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

    def set_output(self, *, transform=None):
        return self
