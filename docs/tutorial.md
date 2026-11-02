# Tutorial

The tutorial uses the synthetic tip diameter data that ships with the package. The scripts in `examples/` contain the same code.

## 1. Load the data and describe it

```python
from cavity_encoding_ml.datasets import load_tip_example, tip_example_config

df = load_tip_example()        # 384 parts, 96 shots, four cavities
config = tip_example_config()
```

`CavityConfig` tells the package which column is the shot, the cavity and the response, which columns are machine features (`shared`), and which columns hold the sensor reading of each cavity (`sensors`). For your own data you write it yourself, see the README.

Save and reuse a configuration with `config.to_json("config.json")` and `CavityConfig.from_json("config.json")`.

## 2. Run the diagnostic

```python
from cavity_encoding_ml import cavity_diagnostic
d = cavity_diagnostic(df.Value, df.ID_Zk, df.CavityNumber)
print(d.report("mm"))
```

Look at `d.passed`, `d.F`, `d.partial_eta2` and `d.separation`. The step passes when the F test is significant (default 0.05) and the partial eta squared is at least 0.01.

## 3. Choose the encoding

One-hot is the default. `choose_encoding(n_cavities)` suggests target encoding for molds with more than 12 cavities. `compare_encodings(df, config, model)` runs several encodings on the same folds.

## 4. Validate

```python
from sklearn.ensemble import RandomForestRegressor
from cavity_encoding_ml import validate_cavity_encoding

v = validate_cavity_encoding(df, config, RandomForestRegressor(random_state=0), encoding="onehot")
print(v.report())
```

The package trains a fresh copy of your model three times on identical folds: the cavity-blind baseline (R0), the cavity readings without the label (R1) and the encoded label. This shows whether the label adds anything beyond the cavity sensors.

The default splitter is `ShotGroupKFold(n_splits=4, n_repeats=3, random_state=42)`. Pass `cv=` to use another one. If it separates the parts of a shot, a `LeakageWarning` is raised.

## 5. Decide

```python
from cavity_encoding_ml import decide_from
print(decide_from(d, v).report())
```

or run everything with `CavityWorkflow(config, model).run(df)`.

## 6. Use the encoding in your own model

```python
from cavity_encoding_ml import CavityMapper, ShotGroupKFold

for train, test in ShotGroupKFold(4).split(df, groups=df.ID_Zk):
    mapper = CavityMapper(config, "onehot")            # strict mode is on
    X_train = mapper.fit_transform(df.iloc[train], df.Value.iloc[train])
    X_test = mapper.transform(df.iloc[test])           # raises if a shot leaks
```

The mapper is a scikit-learn style transformer. `mapper.get_feature_names_out()` lists the columns.

## 7. Data that were screened for outliers

The original study removed outliers with a Box-Cox transform and a per-cavity MAD rule. `screen_outliers(df.Value, df.CavityNumber)` returns a boolean mask of parts to keep.
