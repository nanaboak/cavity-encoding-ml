"""Step 2. Encoding of the cavity label."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

ENCODINGS = ("onehot", "target", "integer")


def choose_encoding(n_cavities: int, max_onehot: int = 12) -> str:
    """Suggest an encoding for a mold with ``n_cavities`` cavities.

    One-hot encoding is the default because it has no fitted parameters. Target
    encoding keeps the input narrow for molds with many cavities, so it is
    suggested above ``max_onehot`` cavities.
    """
    return "onehot" if n_cavities <= max_onehot else "target"


class CavityEncoder(BaseEstimator, TransformerMixin):
    """Encode a cavity label as model input.

    Parameters
    ----------
    method : {"onehot", "target", "integer"}
        ``onehot`` gives one binary column per cavity (default). ``target`` gives
        one column with the smoothed cavity mean of the response, fitted on the
        training rows only. ``integer`` gives the cavity number as one numeric
        column. The integer code imposes an order that has no physical meaning.
    smoothing : float
        Empirical Bayes smoothing weight of the target encoding.
    cavities : sequence, optional
        Fixed list of cavity numbers. Needed when a training set may miss a cavity.
    """

    def __init__(self, method: str = "onehot", smoothing: float = 10.0, cavities=None):
        self.method = method
        self.smoothing = smoothing
        self.cavities = cavities

    def fit(self, cavity, y=None):
        if self.method not in ENCODINGS:
            raise ValueError(f"method must be one of {ENCODINGS}, got {self.method!r}.")
        cav = np.asarray(cavity).astype(float).ravel()
        self.categories_ = (
            np.array(sorted(float(c) for c in self.cavities))
            if self.cavities is not None
            else np.unique(cav)
        )
        if self.method == "target":
            if y is None:
                raise ValueError("Target encoding needs the response y in fit().")
            yv = np.asarray(y, dtype=float).ravel()
            self.prior_ = float(np.mean(yv))
            self.mapping_ = {}
            for c in np.unique(cav):
                m = cav == c
                n_c = int(m.sum())
                mean_c = float(np.mean(yv[m]))
                self.mapping_[float(c)] = (n_c * mean_c + self.smoothing * self.prior_) / (
                    n_c + self.smoothing
                )
        return self

    def transform(self, cavity):
        cav = np.asarray(cavity).astype(float).ravel()
        if self.method == "onehot":
            return (cav.reshape(-1, 1) == self.categories_.reshape(1, -1)).astype(float)
        if self.method == "target":
            return np.array([self.mapping_.get(float(c), self.prior_) for c in cav]).reshape(-1, 1)
        return cav.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        if self.method == "onehot":
            return np.array([f"Cavity_{int(c) if float(c).is_integer() else c}" for c in self.categories_])
        if self.method == "target":
            return np.array(["CavityTargetEncoded"])
        return np.array(["CavityID_integer"])
