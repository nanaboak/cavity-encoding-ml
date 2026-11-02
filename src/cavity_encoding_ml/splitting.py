"""Step 3 (part 1). Shot-grouped folds and the leakage check."""

from __future__ import annotations

import warnings

import numpy as np


class LeakageWarning(UserWarning):
    """Parts of one shot are split between training and test data."""


class LeakageError(RuntimeError):
    """Raised by a strict ``CavityMapper`` when shot siblings leak into the test data."""


class ShotGroupKFold:
    """Repeated K-fold in which all parts of a shot stay in the same fold.

    The shots, not the parts, are shuffled and divided into ``n_splits`` groups.
    Each repeat uses its own seed ``random_state + 1000 * repeat``.
    The splitter follows the scikit-learn splitter interface.

    Parameters
    ----------
    n_splits : int
        Number of folds (default 4).
    n_repeats : int
        Number of repeats with new shuffles (default 1).
    random_state : int
        Base seed (default 42).
    """

    def __init__(self, n_splits: int = 4, n_repeats: int = 1, random_state: int = 42):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return int(self.n_splits * self.n_repeats)

    def split(self, X=None, y=None, groups=None):
        if groups is None:
            raise ValueError("ShotGroupKFold needs the shot identifiers as `groups`.")
        g = np.asarray(groups).astype(str)  # shots are compared as text
        uniq = np.unique(g)
        k = int(max(2, min(self.n_splits, len(uniq))))
        idx = np.arange(len(g))
        for r in range(self.n_repeats):
            rng = np.random.RandomState(self.random_state + 1000 * r)
            shuffled = uniq.copy()
            rng.shuffle(shuffled)
            for chunk in np.array_split(shuffled, k):
                test = np.isin(g, chunk)
                yield idx[~test], idx[test]


class RepeatedRowKFold:
    """Ordinary repeated K-fold on single parts (ignores shots).

    Shown here only so the leakage of ordinary K-fold can be measured. It uses the
    seed ``random_state + 1000 * repeat`` for each repeat.
    """

    def __init__(self, n_splits: int = 4, n_repeats: int = 1, random_state: int = 42):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return int(self.n_splits * self.n_repeats)

    def split(self, X=None, y=None, groups=None):
        from sklearn.model_selection import KFold

        n = len(X) if X is not None else len(groups)
        idx = np.arange(n)
        for r in range(self.n_repeats):
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state + 1000 * r)
            for tr, te in kf.split(idx):
                yield tr, te


def sibling_share(groups, splits) -> float:
    """Share of test parts whose shot also has a part in the training data.

    The value is 0 for a shot-grouped split. Ordinary K-fold on a four-cavity mold
    gives about 0.98, which means almost every test part has a sibling in training.
    """
    g = np.asarray(groups).astype(str)
    hit, total = 0, 0
    for tr, te in splits:
        train_shots = set(g[tr])
        hit += sum(1 for s in g[te] if s in train_shots)
        total += len(te)
    return float(hit / total) if total else 0.0


def check_splits(groups, splits, stacklevel: int = 3) -> float:
    """Return the sibling share and warn when it is above zero."""
    share = sibling_share(groups, splits)
    if share > 0:
        warnings.warn(
            f"{100 * share:.1f} % of the test parts have a part of the same shot in the "
            "training data. The error estimate will look better than it is. "
            "Prefer ShotGroupKFold for multi-cavity molds.",
            LeakageWarning,
            stacklevel=stacklevel,
        )
    return share
