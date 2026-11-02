"""What happens when ordinary K-fold is used on multi-cavity data."""

import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from cavity_encoding_ml import LeakageWarning, ShotGroupKFold, validate_cavity_encoding
from cavity_encoding_ml.datasets import load_tip_example, tip_example_config

df, cfg = load_tip_example(), tip_example_config()
model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=1)

good = validate_cavity_encoding(df, cfg, model, cv=ShotGroupKFold(4, 3, 42))

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    bad = validate_cavity_encoding(df, cfg, model, cv=KFold(4, shuffle=True, random_state=42))

print("Alert raised:", any(issubclass(w.category, LeakageWarning) for w in caught))
print(f"Shot-grouped folds : RMSE {1000 * good.blind.rmse:.2f} um, siblings in training {100 * good.leakage_share:.0f} %")
print(f"Ordinary K-fold    : RMSE {1000 * bad.blind.rmse:.2f} um, siblings in training {100 * bad.leakage_share:.0f} %")
print("The second number looks better but it is not real. It is leakage.")
