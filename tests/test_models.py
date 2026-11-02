import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from cavity_encoding_ml import FunctionModel, make_model


def test_clone_sklearn_model():
    m = RandomForestRegressor(n_estimators=3, random_state=0)
    c = make_model(m)
    assert c is not m and c.n_estimators == 3


def test_factory():
    assert isinstance(make_model(lambda: LinearRegression()), LinearRegression)


def test_factory_must_return_model():
    with pytest.raises(TypeError):
        make_model(lambda: 5)


def test_bad_object():
    with pytest.raises(TypeError):
        make_model(42)


def test_function_model_roundtrip():
    fm = FunctionModel(lambda X, y: float(y.mean()), lambda m, X: np.full(len(X), m))
    fm.fit(np.zeros((4, 2)), [1, 2, 3, 4])
    assert fm.predict(np.zeros((3, 2))).tolist() == [2.5, 2.5, 2.5]


def test_function_model_not_fitted():
    with pytest.raises(RuntimeError):
        FunctionModel(lambda X, y: 0, lambda m, X: X).predict(np.zeros((1, 1)))


def test_function_model_copy_is_unfitted():
    fm = FunctionModel(lambda X, y: float(y.mean()), lambda m, X: np.full(len(X), m))
    fm.fit(np.zeros((2, 1)), [1, 3])
    fresh = make_model(fm)
    with pytest.raises(RuntimeError):
        fresh.predict(np.zeros((1, 1)))


def test_non_sklearn_class_is_deepcopied():
    class Mean:
        def fit(self, X, y):
            self.m = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self.m)

    a = Mean().fit(np.zeros((2, 1)), [1, 3])
    b = make_model(a)
    assert b is not a
