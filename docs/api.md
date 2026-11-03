# API reference

All names can be imported from `cavity_encoding_ml`.

## Step 1

**`cavity_diagnostic(response, shot, cavity, alpha=0.05, min_partial_eta2=0.01) -> DiagnosticResult`**
Fields: `F`, `df1`, `df2`, `p_value`, `partial_eta2`, `sigma_blind`, `sigma_noise`, `separation`, `offsets`, `passed`, `r2_max_blind`, `r2_max_aware`. Methods: `report(unit="")`, `to_dict()`.

## Step 2

**`CavityEncoder(method="onehot", smoothing=10.0, cavities=None)`** with `fit(cavity, y=None)`, `transform(cavity)`, `get_feature_names_out()`. Methods are `"onehot"`, `"target"` and `"integer"`.

**`choose_encoding(n_cavities, max_onehot=12) -> str`**

**`CavityConfig(shot, cavity, response, shared=[], sensors={}, cavities=None)`** with `to_json`, `from_json`, `to_dict`, `from_dict`, `required_columns`.

**`CavityMapper(config, representation="R3", smoothing=10.0, strict=True)`** is a transformer for a DataFrame. Representations are `R0` to `R4` or `blind`, `readings`, `integer`, `onehot`, `target`. In strict mode `transform` raises `LeakageError` for shots that were used in `fit`.

**`REPRESENTATIONS`** is a dict that describes R0 to R4.

## Step 3

**`ShotGroupKFold(n_splits=4, n_repeats=1, random_state=42)`** keeps all parts of a shot in the same fold.

**`RepeatedRowKFold(n_splits=4, n_repeats=1, random_state=42)`** is ordinary K-fold on parts, only for leakage measurements.

**`sibling_share(groups, splits) -> float`** share of test parts that have a shot sibling in training.

**`LeakageWarning`**, **`LeakageError`**

**`validate_cavity_encoding(data, config, model, encoding="onehot", cv=None, smoothing=10.0, response_unit="", warn=True) -> ValidationResult`**
`arms` chooses which models are trained: any of `"blind"`, `"readings"` and `"encoded"`, default all three; `("encoded",)` cross-validates only the model with the label. `ValidationResult` has `arms["blind" | "readings" | "encoded"]`, `jitter`, `leakage_share`, `grouped`, `gain(arm)` and `report()`. Each `ArmResult` has `rmse`, `mae`, `r2`, `fold_rmse`, `oof`, `jitter`.

**`compare_encodings(data, config, model, encodings=(...), **kwargs)`**

**`leakage_audit(data, config, model, n_splits=4, n_repeats=3, random_state=42)`** returns `sibling_share`, `rmse_grouped`, `rmse_kfold` and `understatement`.

**`FunctionModel(fit_fn, predict_fn)`**, **`make_model(model)`**

## Step 4

**`decide(separation, jitter, margin=2.0, diagnostic_passed=True) -> Decision`**, **`decide_from(diagnostic, validation, margin=2.0)`**
`Decision.verdict` is `"use_cavity"`, `"adds_noise"` or `"no_cavity_effect"`.

## Workflow and tools

**`CavityWorkflow(config, model, encoding="onehot", margin=2.0, cv=None, audit_kfold=False, alpha=0.05, min_partial_eta2=0.01, response_unit="", always_validate=False, arms=("blind", "encoded"))`** with `run(data) -> WorkflowResult` and `CavityWorkflow.from_config(path, model)`.

**`screen_outliers(response, cavity, threshold=3.5, boxcox=True)`** returns a mask of parts to keep.

**`cavity_encoding_ml.plotting.plot_decision(result)`** needs matplotlib.

**`cavity_encoding_ml.datasets.load_tip_example()`**, **`tip_example_config()`**

## Command line

```
cavity-encoding-ml example [--encoding onehot|target|integer|auto]
cavity-encoding-ml diagnose data.csv --shot ID --cavity CAV --response Y
cavity-encoding-ml run data.csv --config config.json [--encoding ...] [--margin 2] [--readings] [--audit]
```
