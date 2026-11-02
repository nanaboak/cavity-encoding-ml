# cavity-encoding-ml

Cavity encoding and leakage-safe validation for machine learning on multi-cavity injection molding data.

[![Tests](https://github.com/nanaboak/cavity-encoding-ml/actions/workflows/tests.yml/badge.svg)](https://github.com/nanaboak/cavity-encoding-ml/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A multi-cavity mold forms several parts in every machine cycle (a *shot*). When you predict part quality from process data, three questions come up.

1. Do the cavities differ enough that the cavity label should enter the model?
2. How should the label be encoded?
3. How do you validate the model when all parts of a shot share the same machine settings?

`cavity-encoding-ml` answers them in four steps. It implements the method of Boakye, Hafizh, Lohar and Abdul Majeed (accepted, *Journal of Intelligent Manufacturing*, 2026) and works with any machine learning library that offers `fit` and `predict`.

| Step | What the package does |
|---|---|
| 1. Diagnostic | F test of the cavity effect. It also reports the *floor separation*, the largest RMSE reduction that cavity information can deliver. |
| 2. Encoding | One-hot (default), target or integer encoding of the cavity label, refitted inside every fold. |
| 3. Validation | Shot-grouped folds. A `LeakageWarning` is raised when ordinary K-fold separates the parts of one shot. |
| 4. Decision | Compares the floor separation with the fold-to-fold jitter of the cavity-blind model. If the jitter is similar or larger, it reports that the cavity information adds noise. |

## Install

```bash
pip install cavity-encoding-ml
```

Before the release is on PyPI you can install from GitHub:

```bash
pip install "git+https://github.com/nanaboak/cavity-encoding-ml.git"
```

Python 3.9 or newer. The core needs only NumPy, pandas, SciPy and scikit-learn. Figures need `pip install matplotlib`.

## Try it in one line

```bash
cavity-encoding-ml example
```

This runs the four steps on synthetic tip diameter data that is shipped with the package.

## Quick start

```python
from sklearn.ensemble import RandomForestRegressor
from cavity_encoding_ml import CavityConfig, CavityWorkflow
import pandas as pd

df = pd.read_csv("my_parts.csv")          # one row per part

config = CavityConfig(
    shot="ShotID",                         # column that identifies the machine cycle
    cavity="CavityNumber",                 # column with the cavity number of the part
    response="TipDiameter",                # column with the measured characteristic
    shared=["InjPressure", "CoolFlow"],    # machine features, equal for all parts of a shot
    sensors={                              # one cavity sensor type, one column per cavity
        "Max": ["Cav1Max", "Cav2Max", "Cav3Max", "Cav4Max"],
    },
    cavities=[1, 2, 3, 4],
)

result = CavityWorkflow(config, RandomForestRegressor(random_state=0), audit_kfold=True).run(df)
print(result.report())
print(result.verdict)                      # "use_cavity", "adds_noise" or "no_cavity_effect"
```

If your data have no cavity sensors, leave `sensors` empty. The package then compares the model with and without the cavity label alone.

## Any machine learning library

```python
from cavity_encoding_ml import validate_cavity_encoding, FunctionModel

validate_cavity_encoding(df, config, XGBRegressor())          # any object with fit and predict
validate_cavity_encoding(df, config, lambda: LGBMRegressor()) # or a function that returns one
validate_cavity_encoding(df, config, FunctionModel(my_fit, my_predict))  # PyTorch, TensorFlow, JAX
```

## What is in this repository

| Folder or file | Content |
|---|---|
| `src/cavity_encoding_ml/` | The package |
| `tests/` | 131 automatic tests |
| `docs/` | Beginner guide, tutorial, method and API reference |
| `examples/` | Six small scripts that run as they are |
| `scripts/` | Verification against the original code, reproduction on private data, synthetic data, figure |
| `paper/` | The JOSS paper (`paper.md`, `paper.bib`, `figure1.png`) |

## Verification

`python scripts/verify_paper3_pipeline.py` compares the package with a literal copy of the fold and representation code of the original study. For the random forest the pooled RMSE is exactly equal. For gradient boosting, support vector regression and the multilayer perceptron it agrees to machine precision. On the confidential tip diameter and tensile strength data, the F test reproduces F(3, 285) = 326.4 and F(3, 1513) = 11.4, and the decision matches the original study. The production data are not part of this repository. `scripts/reproduce_paper3.py` repeats the analysis if you have the files.

## Cite

Use the button **Cite this repository** on GitHub, which reads `CITATION.cff`, and cite the original study.

## Contribute and license

See `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`. The package is released under the MIT license.
