"""Optional outlier screening used in the original study."""

from __future__ import annotations

import numpy as np
from scipy import stats

MAD_CONSTANT = 0.6745


def screen_outliers(response, cavity, threshold: float = 3.5, boxcox: bool = True):
    """Flag outliers with the per-cavity MAD modified z-score.

    The response is transformed with a Box-Cox transform (shifted so that its
    minimum is 1) when ``boxcox`` is True. For each cavity the modified
    z-score ``0.6745 (x - median) / MAD`` is computed and parts with
    ``|z| > threshold`` are flagged (Iglewicz & Hoaglin, 1993).

    Returns
    -------
    keep : numpy.ndarray of bool
        True for parts to keep.
    """
    y = np.asarray(response, dtype=float)
    c = np.asarray(cavity)
    if boxcox:
        yt, _ = stats.boxcox(y + (1.0 - np.nanmin(y)))
    else:
        yt = y.copy()
    z = np.zeros(len(y))
    for lev in np.unique(c):
        m = c == lev
        med = np.median(yt[m])
        mad = np.median(np.abs(yt[m] - med))
        if mad <= 0:
            sd = np.std(yt[m], ddof=1)
            mad = MAD_CONSTANT * sd if sd > 0 else 1e-12
        z[m] = MAD_CONSTANT * (yt[m] - med) / mad
    return np.abs(z) <= threshold
