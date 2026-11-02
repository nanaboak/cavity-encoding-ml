"""Models from any machine learning library."""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np


class FunctionModel:
    """Wrap two functions so that any library can be used.

    Parameters
    ----------
    fit_fn : callable
        ``fit_fn(X, y)`` trains a model and returns it (or any object that
        ``predict_fn`` understands).
    predict_fn : callable
        ``predict_fn(model, X)`` returns the predictions as an array.

    Example
    -------
    >>> fm = FunctionModel(lambda X, y: float(y.mean()), lambda m, X: [m] * len(X))
    """

    def __init__(self, fit_fn: Callable, predict_fn: Callable):
        self.fit_fn = fit_fn
        self.predict_fn = predict_fn
        self._model = None

    def fit(self, X, y):
        self._model = self.fit_fn(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("FunctionModel is not fitted.")
        return np.asarray(self.predict_fn(self._model, np.asarray(X, dtype=float)), dtype=float).ravel()

    def __copy__(self):
        return FunctionModel(self.fit_fn, self.predict_fn)

    def __deepcopy__(self, memo):
        return FunctionModel(self.fit_fn, self.predict_fn)


def make_model(model):
    """Return a fresh, unfitted copy of ``model``.

    ``model`` can be an object with ``fit`` and ``predict`` (scikit-learn, XGBoost,
    LightGBM, CatBoost and others), a :class:`FunctionModel`, or a function without
    arguments that returns such an object.
    """
    if hasattr(model, "fit") and hasattr(model, "predict"):
        try:
            from sklearn.base import clone

            return clone(model)
        except Exception:
            return copy.deepcopy(model)
    if callable(model):
        fresh = model()
        if not (hasattr(fresh, "fit") and hasattr(fresh, "predict")):
            raise TypeError("The model factory must return an object with fit and predict.")
        return fresh
    raise TypeError(
        "model must have fit and predict methods, be a FunctionModel, or be a factory function."
    )
