"""Layout of the cavity data and the five cavity representations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .encoding import CavityEncoder
from .splitting import LeakageError

#: Cavity representations, coded as in the original study.
REPRESENTATIONS = {
    "R0": "Cavity-blind baseline: machine features and the shot average of each cavity sensor",
    "R1": "Cavity readings: machine features and the sensors of the producing cavity",
    "R2": "Integer label: R1 and the cavity number as one numeric column",
    "R3": "One-hot label (default): R1 and one binary column per cavity",
    "R4": "Target-encoded label: R1 and the smoothed cavity mean of the response",
}

_ALIASES = {
    "blind": "R0",
    "readings": "R1",
    "integer": "R2",
    "onehot": "R3",
    "target": "R4",
}


def resolve_representation(name: str) -> str:
    key = _ALIASES.get(str(name).lower(), str(name).upper())
    if key not in REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation {name!r}. Use one of {list(REPRESENTATIONS)} "
            f"or {list(_ALIASES)}."
        )
    return key


@dataclass
class CavityConfig:
    """Describes where shot, cavity, response and sensors sit in the data table.

    Parameters
    ----------
    shot : str
        Column that identifies the machine cycle (shot).
    cavity : str
        Column with the cavity number of each part.
    response : str
        Column with the measured quality characteristic.
    shared : list of str
        Machine features that are the same for all parts of a shot.
    sensors : dict
        One entry per cavity sensor type. The value is the list of columns that
        hold the reading of cavity 1, cavity 2, and so on. Example:
        ``{"Max": ["P1Max", "P2Max", "P3Max", "P4Max"]}``. May be empty.
    cavities : list of int, optional
        Cavity numbers in the same order as the sensor columns. Default 1 to k.
    """

    shot: str
    cavity: str
    response: str
    shared: List[str] = field(default_factory=list)
    sensors: Dict[str, List[str]] = field(default_factory=dict)
    cavities: Optional[List[int]] = None

    def __post_init__(self):
        if self.cavities is None and self.sensors:
            k = len(next(iter(self.sensors.values())))
            self.cavities = list(range(1, k + 1))
        for name, cols in self.sensors.items():
            if self.cavities is not None and len(cols) != len(self.cavities):
                raise ValueError(
                    f"Sensor '{name}' has {len(cols)} columns but there are "
                    f"{len(self.cavities)} cavities."
                )

    def required_columns(self) -> List[str]:
        cols = [self.shot, self.cavity, self.response] + list(self.shared)
        for c in self.sensors.values():
            cols += list(c)
        return list(dict.fromkeys(cols))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CavityConfig":
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> "CavityConfig":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)


class CavityMapper(BaseEstimator, TransformerMixin):
    """Builds a cavity representation from a table of parts.

    Parameters
    ----------
    config : CavityConfig
    representation : str
        ``"R0"`` to ``"R4"`` or ``"blind"``, ``"readings"``, ``"integer"``,
        ``"onehot"`` (default ``"R3"``) and ``"target"``.
    smoothing : float
        Smoothing weight of the target encoding (R4).
    strict : bool
        When True (default), ``transform`` raises ``LeakageError`` if a part of a
        shot that was used in ``fit`` is transformed. This makes a leaky split fail
        instead of silently inflating the accuracy.
    """

    def __init__(self, config=None, representation: str = "R3", smoothing: float = 10.0, strict: bool = True):
        self.config = config
        self.representation = representation
        self.smoothing = smoothing
        self.strict = strict

    # ------------------------------------------------------------------
    def _check(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CavityMapper expects a pandas DataFrame.")
        cfg = self.config
        needed = [cfg.shot, cfg.cavity] + list(cfg.shared) + [c for v in cfg.sensors.values() for c in v]
        missing = [c for c in needed if c not in X.columns]
        if missing:
            raise KeyError(f"Columns missing from the data: {missing}")

    def _cavity_index(self, cav):
        order = list(self.config.cavities)
        lookup = {float(c): i for i, c in enumerate(order)}
        try:
            return np.array([lookup[float(c)] for c in cav])
        except KeyError as exc:
            raise ValueError(f"Cavity {exc.args[0]} is not in config.cavities {order}.") from None

    def _build(self, X: pd.DataFrame):
        cfg = self.config
        rep = self.rep_
        shared = X[list(cfg.shared)].to_numpy(dtype=float) if cfg.shared else np.empty((len(X), 0))
        names = list(cfg.shared)
        cav = X[cfg.cavity].to_numpy(dtype=float)
        if rep == "R0":
            blocks = [shared]
            for sname, cols in cfg.sensors.items():
                blocks.append(X[list(cols)].to_numpy(dtype=float).mean(axis=1, keepdims=True))
                names.append(f"{sname}_shot_mean")
            return np.hstack(blocks), names
        idx = self._cavity_index(cav) if cfg.sensors else None
        blocks = [shared]
        for sname, cols in cfg.sensors.items():
            vals = X[list(cols)].to_numpy(dtype=float)
            blocks.append(vals[np.arange(len(X)), idx].reshape(-1, 1))
            names.append(f"{sname}_own")
        if rep != "R1":
            enc = self.encoder_.transform(cav)
            blocks.append(enc)
            names += list(self.encoder_.get_feature_names_out())
        return np.hstack(blocks), names

    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        self._check(X)
        self.rep_ = resolve_representation(self.representation)
        self.fit_shots_ = set(X[self.config.shot].astype(str))
        if self.rep_ in ("R2", "R3", "R4"):
            method = {"R2": "integer", "R3": "onehot", "R4": "target"}[self.rep_]
            self.encoder_ = CavityEncoder(
                method=method, smoothing=self.smoothing, cavities=self.config.cavities
            ).fit(X[self.config.cavity].to_numpy(), y)
        _, self.feature_names_ = self._build(X.iloc[:2] if len(X) > 1 else X)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        Z, self.feature_names_ = self._build(X)
        return Z

    def transform(self, X):
        self._check(X)
        if self.strict:
            overlap = self.fit_shots_ & set(X[self.config.shot].astype(str))
            if overlap:
                raise LeakageError(
                    f"{len(overlap)} shot(s) in the data to transform were also used in fit(). "
                    "Parts of one shot share the machine state, so this would inflate the "
                    "accuracy. Use ShotGroupKFold, or set strict=False if this is intended."
                )
        Z, self.feature_names_ = self._build(X)
        return Z

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)
