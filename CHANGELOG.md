# Changelog

All notable changes are listed here. The project follows semantic versioning.

## 0.1.0 - 2026-11-02

First public release.

- Step 1: `cavity_diagnostic` with the F test, the two noise floors and the floor separation.
- Step 2: `CavityEncoder` and `CavityMapper` with one-hot (default), target and integer encoding.
- Step 3: `ShotGroupKFold`, the `LeakageWarning` alert and `validate_cavity_encoding`.
- Step 4: `decide` compares the floor separation with the fold-to-fold jitter.
- `CavityWorkflow` runs the four steps, and a command-line interface is included.
- Works with any model that offers `fit` and `predict`, and with `FunctionModel` for other libraries.
- Verification scripts and synthetic tip diameter data.
