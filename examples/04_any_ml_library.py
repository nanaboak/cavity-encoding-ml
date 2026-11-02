"""Any machine learning library works.

1. Anything with fit and predict (scikit-learn, XGBoost, LightGBM, CatBoost).
2. A function that returns such a model.
3. FunctionModel for code that has no fit and predict, such as a PyTorch loop.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from cavity_encoding_ml import FunctionModel, validate_cavity_encoding
from cavity_encoding_ml.datasets import load_tip_example, tip_example_config

df, cfg = load_tip_example(), tip_example_config()

# 1. A model object. For XGBoost: from xgboost import XGBRegressor; model = XGBRegressor()
v1 = validate_cavity_encoding(df, cfg, HistGradientBoostingRegressor(random_state=0))

# 2. A factory function
v2 = validate_cavity_encoding(df, cfg, lambda: Ridge(alpha=1.0))


# 3. Two plain functions. Replace the body with your own PyTorch or TensorFlow code.
def fit(X, y):
    X1 = np.c_[X, np.ones(len(X))]
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def predict(weights, X):
    return np.c_[X, np.ones(len(X))] @ weights


v3 = validate_cavity_encoding(df, cfg, FunctionModel(fit, predict))

for name, v in [("HistGradientBoosting", v1), ("Ridge", v2), ("FunctionModel", v3)]:
    print(f"{name:<22} blind {1000 * v.blind.rmse:6.2f} um   one-hot {1000 * v.encoded.rmse:6.2f} um")
