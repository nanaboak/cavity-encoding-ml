"""Step 3 (part 2). Validation of the cavity-blind, cavity-reading and encoded models."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .encoding import choose_encoding
from .mapper import CavityConfig, CavityMapper
from .models import make_model
from .splitting import RepeatedRowKFold, ShotGroupKFold, check_splits, sibling_share

_R2 = "R\u00b2"
_ENC_NAME = {"onehot": "one-hot", "target": "target-encoded", "integer": "integer"}


def _wrap(text: str, width: int = 92) -> str:
    return textwrap.fill(text, width=width, initial_indent="  ", subsequent_indent="  ")


_ENC_TO_REP = {"integer": "R2", "onehot": "R3", "target": "R4"}
_ARM_REP = {"blind": "R0", "readings": "R1", "encoded": None}


@dataclass
class ArmResult:
    """Cross-validated accuracy of one representation."""

    name: str
    representation: str
    fold_rmse: np.ndarray
    rmse: float
    mae: float
    r2: float
    oof: np.ndarray
    n_inputs: int
    feature_names: List[str]

    @property
    def fold_mean(self) -> float:
        return float(np.mean(self.fold_rmse))

    @property
    def fold_std(self) -> float:
        return float(np.std(self.fold_rmse, ddof=1)) if len(self.fold_rmse) > 1 else float("nan")

    @property
    def jitter(self) -> float:
        """Standard deviation of the fold RMSE divided by its mean."""
        return self.fold_std / self.fold_mean


@dataclass
class ValidationResult:
    """Result of :func:`validate_cavity_encoding`."""

    encoding: str
    arms: Dict[str, ArmResult]
    n_folds: int
    leakage_share: float
    grouped: bool
    response_unit: str = ""

    @property
    def blind(self) -> ArmResult:
        return self.arms["blind"]

    @property
    def readings(self) -> ArmResult:
        return self.arms["readings"]

    @property
    def encoded(self) -> ArmResult:
        return self.arms["encoded"]

    @property
    def jitter(self) -> float:
        """Fold-to-fold jitter of the cavity-blind model (nan when the blind arm was not run)."""
        return self.blind.jitter if "blind" in self.arms else float("nan")

    def gain(self, arm: str = "encoded") -> float:
        """Relative RMSE reduction of ``arm`` against the cavity-blind model (nan without it)."""
        if "blind" not in self.arms:
            return float("nan")
        return float(1.0 - self.arms[arm].rmse / self.blind.rmse)

    def interpretation(self) -> List[str]:
        """Plain-language reading of the validation result, one sentence per entry."""
        u = f" {self.response_unit}" if self.response_unit else ""
        enc = _ENC_NAME.get(self.encoding, self.encoding)
        if "encoded" not in self.arms:
            return []
        e = self.encoded
        if "blind" not in self.arms:
            return [
                f"The model with the {enc} cavity label reaches an RMSE of {e.rmse:.4g}{u} and an "
                f"R\u00b2 of {e.r2:.2f} on shots it has not seen during training.",
                "Run it together with the cavity-blind model (arms blind and encoded) to find out "
                "whether the label is responsible for this accuracy.",
            ]
        out = []
        total = self.gain("encoded")
        has_r1 = "readings" in self.arms
        if has_r1:
            g1 = self.gain("readings")
            extra = float(1.0 - e.rmse / self.readings.rmse)
            if g1 > 0:
                out.append(f"The cavity sensor readings alone lower the error by {100 * g1:.1f} % "
                           "compared with the cavity-blind model.")
            else:
                out.append(f"The cavity sensor readings alone do not help, they raise the error by "
                           f"{100 * -g1:.1f} %.")
            out.append(f"Adding the {enc} cavity label {'lowers' if extra > 0 else 'raises'} the "
                       f"error by a further {100 * abs(extra):.1f} %, which gives "
                       f"{100 * total:+.1f} % in total against the cavity-blind model.")
        else:
            out.append(f"The {enc} cavity label {'lowers' if total > 0 else 'raises'} the error by "
                       f"{100 * abs(total):.1f} % compared with the cavity-blind model.")
        if total > 0 and has_r1:
            if extra <= 0:
                out.append("In this run the gain comes from the sensor readings, and the label "
                           "adds nothing on top of them.")
            elif g1 <= 0:
                out.append("In this run the gain comes from the label alone, which points to "
                           "cavity differences, such as gate distance, cooling or wear, that the "
                           "sensors do not record.")
            elif extra > g1:
                out.append("In this run most of the gain comes from the label, which points to "
                           "cavity differences that the sensors do not record.")
            else:
                out.append("In this run most of the gain comes from the sensor readings, and the "
                           "label adds a smaller extra step.")
        if total > 0:
            out.append("Step 4 checks whether a gain of this kind is large enough to rely on.")
        else:
            out.append("Cavity information does not improve on the cavity-blind model in this "
                       "run.")
        return out

    def report(self) -> str:
        u = f" {self.response_unit}" if self.response_unit else ""
        lines = [
            "STEP 3  Validation",
            f"  folds {self.n_folds}, parts of a test shot also in training: {100 * self.leakage_share:.1f} %",
        ]
        if self.grouped:
            lines.append(_wrap("No leakage. Every shot stays in one fold, so the errors below are "
                               "honest estimates for new shots."))
        else:
            lines.append(
                "  ALERT. The folds are not grouped by shot, so the RMSE below is too optimistic."
                " Use ShotGroupKFold for a multi-cavity mold."
            )
        show_gain = "blind" in self.arms
        head = f"  {'model input':<34}{'RMSE' + u:>14}{_R2:>8}"
        lines.append(head + (f"{'gain vs blind':>16}" if show_gain else ""))
        labels = {
            "blind": "R0 cavity-blind",
            "readings": "R1 cavity readings, no label",
            "encoded": f"{self.encoded.representation} + {self.encoding} label" if "encoded" in self.arms else "",
        }
        for key in ("blind", "readings", "encoded"):
            if key not in self.arms:
                continue
            a = self.arms[key]
            g = "" if (key == "blind" or not show_gain) else f"{100 * self.gain(key):+.1f} %"
            lines.append(f"  {labels[key]:<34}{a.rmse:>14.4g}{a.r2:>8.2f}{g:>16}")
        if show_gain:
            lines.append(
                f"  fold-to-fold jitter of the blind model: {100 * self.jitter:.1f} % "
                f"(SD {self.blind.fold_std:.3g} / mean {self.blind.fold_mean:.3g})"
            )
        text = self.interpretation()
        if text:
            lines.append(_wrap("Result: " + " ".join(text)))
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.report()


def _is_grouped(share: float) -> bool:
    return share == 0.0


def _run_arm(df, y, config, rep, model, splits, smoothing, name):
    n = len(df)
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    fold_rmse = []
    names: List[str] = []
    n_inputs = 0
    for tr, te in splits:
        mapper = CavityMapper(config, rep, smoothing=smoothing, strict=False)
        Xtr = mapper.fit_transform(df.iloc[tr], y[tr])
        Xte = mapper.transform(df.iloc[te])
        names, n_inputs = list(mapper.feature_names_), Xtr.shape[1]
        est = make_model(model)
        est.fit(Xtr, y[tr])
        pred = np.asarray(est.predict(Xte), dtype=float).ravel()
        fold_rmse.append(float(np.sqrt(np.mean((y[te] - pred) ** 2))))
        oof_sum[te] += pred
        oof_cnt[te] += 1
    ok = oof_cnt > 0
    oof = np.where(ok, oof_sum / np.maximum(oof_cnt, 1), np.nan)
    resid = y[ok] - oof[ok]
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    r2 = float(1.0 - np.sum(resid**2) / np.sum((y[ok] - y[ok].mean()) ** 2))
    return ArmResult(name, rep, np.array(fold_rmse), rmse, mae, r2, oof, n_inputs, names)


def validate_cavity_encoding(
    data: pd.DataFrame,
    config: CavityConfig,
    model,
    encoding: str = "onehot",
    cv=None,
    smoothing: float = 10.0,
    response_unit: str = "",
    warn: bool = True,
    arms=("blind", "readings", "encoded"),
) -> ValidationResult:
    """Validate a model with and without the cavity label on identical folds.

    A fresh copy of ``model`` is trained for the cavity-blind baseline (R0), for
    the cavity readings without the label (R1) and for the encoded label. This
    separates the value of the label from the value of cavity-level sensors.

    Parameters
    ----------
    data : DataFrame
        One row per part.
    config : CavityConfig
    model : object with ``fit``/``predict``, ``FunctionModel`` or factory
    encoding : {"onehot", "target", "integer", "auto"}
    cv : splitter, optional
        Any object with ``split(X, y, groups)``. Default ``ShotGroupKFold(4, 3, 42)``.
        A split that puts parts of one shot on both sides triggers a
        ``LeakageWarning``.
    arms : tuple of str
        Which models to train. Any of ``"blind"`` (R0), ``"readings"`` (R1) and
        ``"encoded"``. Default all three, as in version 0.1. Use
        ``("blind", "encoded")`` to compare with and without the label, or
        ``("encoded",)`` to only cross-validate the model with the encoded label.
    """
    if encoding == "auto":
        encoding = choose_encoding(len(config.cavities or np.unique(data[config.cavity])))
    if encoding not in _ENC_TO_REP:
        raise ValueError(f"encoding must be one of {list(_ENC_TO_REP)} or 'auto'.")
    wanted = (arms,) if isinstance(arms, str) else tuple(arms)
    unknown = [a for a in wanted if a not in _ARM_REP]
    if unknown or not wanted:
        raise ValueError(f"arms must be a non-empty subset of {list(_ARM_REP)}, got {list(wanted)}.")
    df = data.reset_index(drop=True)
    y = df[config.response].to_numpy(dtype=float)
    groups = df[config.shot].to_numpy()
    cv = cv if cv is not None else ShotGroupKFold(4, 3, 42)
    splits = list(cv.split(df, y, groups))
    share = check_splits(groups, splits) if warn else sibling_share(groups, splits)
    rep_of = dict(_ARM_REP, encoded=_ENC_TO_REP[encoding])
    results = {
        key: _run_arm(df, y, config, rep_of[key], model, splits, smoothing, key)
        for key in _ARM_REP
        if key in wanted
    }
    return ValidationResult(
        encoding=encoding,
        arms=results,
        n_folds=len(splits),
        leakage_share=share,
        grouped=_is_grouped(share),
        response_unit=response_unit,
    )


def compare_encodings(data, config, model, encodings=("onehot", "target", "integer"), **kwargs):
    """Run :func:`validate_cavity_encoding` for several encodings.

    Returns a dict ``{encoding: ValidationResult}``.
    """
    return {e: validate_cavity_encoding(data, config, model, encoding=e, **kwargs) for e in encodings}


def leakage_audit(data, config, model, n_splits: int = 4, n_repeats: int = 3, random_state: int = 42):
    """Measure how much ordinary K-fold flatters the cavity-blind model.

    Returns a dict with the sibling share of ordinary K-fold, the RMSE under
    shot-grouped and under ordinary K-fold, and the understatement
    ``(rmse_grouped - rmse_kfold) / rmse_grouped``.
    """
    import warnings

    df = data.reset_index(drop=True)
    y = df[config.response].to_numpy(dtype=float)
    groups = df[config.shot].to_numpy()
    g_splits = list(ShotGroupKFold(n_splits, n_repeats, random_state).split(df, y, groups))
    k_splits = list(RepeatedRowKFold(n_splits, n_repeats, random_state).split(df, y, groups))
    g = _run_arm(df, y, config, "R0", model, g_splits, 10.0, "blind")
    k = _run_arm(df, y, config, "R0", model, k_splits, 10.0, "blind")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        share = sibling_share(groups, k_splits)
    return {
        "sibling_share": share,
        "rmse_grouped": g.rmse,
        "rmse_kfold": k.rmse,
        "understatement": float((g.rmse - k.rmse) / g.rmse),
    }
