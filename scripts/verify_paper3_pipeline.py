"""Check that the library agrees with the code of the original study.

The functions below are a literal copy of the fold generator, the representation
builder, the target encoder and the model pipeline of the training script of the
original study (Boakye et al., Journal of Intelligent Manufacturing). They are
used here for one purpose only. The pooled out-of-fold RMSE of that code is
compared with the RMSE of ``cavity-encoding-ml`` for the same data, the same
feature set, the same folds and the same learner.

Run from the repository root:

    python scripts/verify_paper3_pipeline.py

The difference is printed for each learner and representation. It must be below
1e-12, which is floating-point rounding (it is exactly 0.0 for the random forest).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

TE_SMOOTHING = 10.0
RANDOM_STATE = 42
TOLERANCE = 1e-12  # differences of this size are floating-point rounding


# ---- literal copy from the original training script -----------------------
def make_cv_splits(scheme, groups, n_splits=4, n_repeats=3, seed=RANDOM_STATE):
    n = len(groups)
    idx_all = np.arange(n)
    splits = []
    if scheme == "grouped":
        uniq = np.unique(groups)
        k = int(max(2, min(n_splits, len(uniq))))
        for r in range(n_repeats):
            rng = np.random.RandomState(seed + 1000 * r)
            shuffled = uniq.copy()
            rng.shuffle(shuffled)
            chunks = np.array_split(shuffled, k)
            for f, chunk in enumerate(chunks):
                test_mask = np.isin(groups, chunk)
                splits.append((r, f, idx_all[~test_mask], idx_all[test_mask]))
    elif scheme == "kfold":
        k = int(max(2, min(n_splits, n)))
        for r in range(n_repeats):
            kf = KFold(n_splits=k, shuffle=True, random_state=seed + 1000 * r)
            for f, (tr, te) in enumerate(kf.split(idx_all)):
                splits.append((r, f, tr, te))
    return splits


class CavityTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cav_col=-1, smoothing=TE_SMOOTHING):
        self.cav_col = cav_col
        self.smoothing = smoothing

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        cav = X[:, self.cav_col]
        self.prior_ = float(np.mean(y))
        self.mapping_ = {}
        for c in np.unique(cav):
            m = cav == c
            n_c = int(m.sum())
            mean_c = float(np.mean(y[m]))
            self.mapping_[float(c)] = (n_c * mean_c + self.smoothing * self.prior_) / (n_c + self.smoothing)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        cav = X[:, self.cav_col]
        X[:, self.cav_col] = np.array([self.mapping_.get(float(c), self.prior_) for c in cav])
        return X


def build_representation(rep, base, shared_names, cavity_names, mean_map, cavity_numbers):
    shared = base[shared_names].values.astype(float)
    own = base[cavity_names].values.astype(float)
    avg = base[[mean_map[f] for f in cavity_names]].values.astype(float)
    cav = base["__cav"].values.astype(float)
    if rep == "R0":
        X = np.hstack([shared, avg])
    elif rep == "R1":
        X = np.hstack([shared, own])
    elif rep == "R2":
        X = np.hstack([shared, own, cav.reshape(-1, 1)])
    elif rep == "R3":
        onehot = np.zeros((len(base), len(cavity_numbers)))
        for j, c in enumerate(cavity_numbers):
            onehot[:, j] = (cav == c).astype(float)
        X = np.hstack([shared, own, onehot])
    elif rep == "R4":
        X = np.hstack([shared, own, cav.reshape(-1, 1)])
    return X, base["__y"].values.astype(float), base["__shot"].values


def base_model(name):
    """Baseline settings of the original study (random_state = 42)."""
    if name == "RF":
        return RandomForestRegressor(n_estimators=200, max_features="sqrt", max_depth=15,
                                     random_state=RANDOM_STATE, n_jobs=1)
    if name == "GB":
        return GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4,
                                         random_state=RANDOM_STATE)
    if name == "SVR":
        return SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")
    if name == "MLP":
        return MLPRegressor(hidden_layer_sizes=(100, 50), activation="relu", solver="adam",
                            max_iter=1000, early_stopping=True, validation_fraction=0.1,
                            n_iter_no_change=20, alpha=1e-4, random_state=RANDOM_STATE)
    raise ValueError(name)


def pipeline(rep, model="RF"):
    steps = []
    if rep == "R4":
        steps.append(("target_encode", CavityTargetEncoder(cav_col=-1, smoothing=TE_SMOOTHING)))
    steps.append(("scaler", MinMaxScaler()))
    steps.append(("model", base_model(model)))
    est = Pipeline(steps)
    if model in ("SVR", "MLP"):  # the original code scales the target for these two
        est = TransformedTargetRegressor(regressor=est, transformer=MinMaxScaler())
    return est


def original_rmse(df, cfg, rep, model="RF"):
    """Pooled out-of-fold RMSE computed with the original code."""
    cavity_numbers = list(cfg.cavities)
    names = list(cfg.sensors)
    base = pd.DataFrame({"__shot": df[cfg.shot].astype(str).values,
                         "__cav": pd.to_numeric(df[cfg.cavity]).values,
                         "__y": pd.to_numeric(df[cfg.response]).values})
    for c in cfg.shared:
        base[c] = df[c].values.astype(float)
    idx = (base["__cav"].astype(int) - cavity_numbers[0]).values
    rows = np.arange(len(base))
    mean_map, cavity_names = {}, []
    for n in names:
        src = df[list(cfg.sensors[n])].values.astype(float)
        base[f"CavityUnique{n}"] = src[rows, idx]
        base[f"CavityMean{n}"] = src.mean(axis=1)
        mean_map[f"CavityUnique{n}"] = f"CavityMean{n}"
        cavity_names.append(f"CavityUnique{n}")
    X, y, groups = build_representation(rep, base, list(cfg.shared), cavity_names, mean_map, cavity_numbers)
    oof_sum, oof_cnt = np.zeros(len(y)), np.zeros(len(y))
    for _, _, tr, te in make_cv_splits("grouped", groups):
        est = pipeline(rep, model)
        est.fit(X[tr], y[tr])
        oof_sum[te] += est.predict(X[te])
        oof_cnt[te] += 1
    oof = oof_sum / np.maximum(oof_cnt, 1)
    return float(np.sqrt(np.mean((y - oof) ** 2)))


# ---- comparison -----------------------------------------------------------
def library_rmse(df, cfg, rep, model="RF"):
    from cavity_encoding_ml import CavityMapper, ShotGroupKFold, make_model

    y = df[cfg.response].to_numpy(dtype=float)
    est0 = pipeline("R1", model)  # same scaler and learner, without the target encoder
    oof_sum, oof_cnt = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in ShotGroupKFold(4, 3, 42).split(df, y, df[cfg.shot].to_numpy()):
        m = CavityMapper(cfg, rep, smoothing=TE_SMOOTHING, strict=False)
        Xtr = m.fit_transform(df.iloc[tr], y[tr])
        Xte = m.transform(df.iloc[te])
        est = make_model(est0)
        est.fit(Xtr, y[tr])
        oof_sum[te] += est.predict(Xte)
        oof_cnt[te] += 1
    oof = oof_sum / np.maximum(oof_cnt, 1)
    return float(np.sqrt(np.mean((y - oof) ** 2)))


def compare(df, cfg, reps=("R0", "R1", "R2", "R3", "R4"), model="RF"):
    out = {}
    for rep in reps:
        a, b = original_rmse(df, cfg, rep, model), library_rmse(df, cfg, rep, model)
        out[rep] = {"original": a, "library": b, "difference": abs(a - b)}
    return out


def main():
    sys.path.insert(0, "src")
    from cavity_encoding_ml.datasets import load_tip_example, tip_example_config

    ok = True
    for model in ("RF", "GB", "SVR", "MLP"):
        res = compare(load_tip_example(), tip_example_config(), model=model)
        print(f"\nModel {model}")
        print(f"{'rep':<5}{'original code':>16}{'cavity-encoding-ml':>20}{'difference':>14}")
        for rep, r in res.items():
            print(f"{rep:<5}{r['original']:>16.8f}{r['library']:>20.8f}{r['difference']:>14.1e}")
        ok = ok and all(r["difference"] < TOLERANCE for r in res.values())
    print("\nResult:", "identical to machine precision" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
