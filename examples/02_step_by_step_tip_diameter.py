"""All four steps one after the other, with comments for beginners."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from cavity_encoding_ml import cavity_diagnostic, decide_from, validate_cavity_encoding
from cavity_encoding_ml.datasets import load_tip_example, tip_example_config

df = load_tip_example()          # one row per part
config = tip_example_config()    # tells the package which column is which

# Step 1. Do the cavities differ?
diagnostic = cavity_diagnostic(df[config.response], df[config.shot], df[config.cavity])
print(diagnostic.report("mm"))
if not diagnostic.passed:
    raise SystemExit("No cavity effect. Leave the cavity label out.")

# Step 2 and 3. Encode the label (one-hot) and validate on shot-grouped folds.
model = make_pipeline(MinMaxScaler(), RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1))
validation = validate_cavity_encoding(df, config, model, encoding="onehot", response_unit="mm")
print()
print(validation.report())

# Step 4. Is the gain larger than the fold-to-fold jitter?
print()
print(decide_from(diagnostic, validation).report())
