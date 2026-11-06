# Changelog

All notable changes are listed here. The project follows semantic versioning.

## 0.3.0 - 2026-11-04

Reports that explain themselves.

- Step 3 ends with a `Result:` paragraph in plain language. It states how much the cavity sensor readings (R1) and the cavity label each lower the error, and whether the gain comes mainly from the sensors or from the label. Whether the gain is large enough to rely on stays the job of Step 4.
- Step 3 confirms when the folds are free of leakage, and the table now also shows R².
- Step 4 starts with a clear `Recommendation:` and explains the reason and what to do next. It also mentions the gain measured in the run.
- `CavityWorkflow` trains R1 again by default (`arms=("blind", "readings", "encoded")`), so the gain of the sensors and of the label can be told apart. `arms=("blind", "encoded")` keeps the faster setting of 0.2.0.
- New `ValidationResult.interpretation()` and an optional `gain` argument for `decide`.
- Command line: `--no-readings` skips R1. `--readings` is kept and has no effect.

## 0.2.0 - 2026-11-03

Leaner validation. Nothing is removed and old calls still work.

- `validate_cavity_encoding` has a new argument `arms`. It chooses which models are trained: `"blind"` (R0), `"readings"` (R1) and `"encoded"`. The default is still all three. `arms=("encoded",)` cross-validates only the model with the cavity label, which is the quick way to use the library as feature engineering.
- `CavityWorkflow` has the same argument, with the default `("blind", "encoded")`. The decision only needs these two, so R1 is no longer trained unless you ask for it. The report no longer contains the R1 line by default.
- The ordinary K-fold leakage audit stays optional and off (`audit_kfold=False`). The shot-grouped folds still report the sibling share without training anything extra.
- Command line: new flag `--readings` for `run`.
- `ValidationResult.jitter` and `gain` return `nan` when the blind model was not run.
- `scripts/reproduce_paper3.py` and the `example` command still train all three models, so the Paper 3 numbers are unchanged.

## 0.1.0 - 2026-11-02

First public release.

- Step 1: `cavity_diagnostic` with the F test, the two noise floors and the floor separation.
- Step 2: `CavityEncoder` and `CavityMapper` with one-hot (default), target and integer encoding.
- Step 3: `ShotGroupKFold`, the `LeakageWarning` alert and `validate_cavity_encoding`.
- Step 4: `decide` compares the floor separation with the fold-to-fold jitter.
- `CavityWorkflow` runs the four steps, and a command-line interface is included.
- Works with any model that offers `fit` and `predict`, and with `FunctionModel` for other libraries.
- Verification scripts and synthetic tip diameter data.
