"""Step 1. The cavity-effect diagnostic (F test and noise floors).

For a response measured on every cavity of every shot, two within-shot noise
floors are computed. ``sigma_blind`` is the pooled within-shot standard deviation
of the raw response. ``sigma_noise`` is the same quantity after the systematic
cavity offsets have been removed. For a balanced layout with S shots and k cavities

    F = (S - 1) (sigma_blind^2 - sigma_noise^2) / sigma_noise^2

with k - 1 and (k - 1)(S - 1) degrees of freedom. This module uses the exact
within-shot least-squares form, which gives the same value for balanced data and
stays valid when single parts are missing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DiagnosticResult:
    """Result of :func:`cavity_diagnostic`."""

    n_parts: int
    n_shots: int
    n_cavities: int
    sigma_total: float
    sigma_blind: float
    sigma_noise: float
    offsets: Dict[int, float]
    F: float
    df1: int
    df2: int
    p_value: float
    partial_eta2: float
    separation: float
    alpha: float
    min_partial_eta2: float
    passed: bool
    r2_max_blind: float
    r2_max_aware: float
    F_from_floors: float = field(default=float("nan"))

    def to_dict(self) -> dict:
        out = asdict(self)
        out["offsets"] = {int(k): float(v) for k, v in self.offsets.items()}
        return out

    def report(self, unit: str = "") -> str:
        u = f" {unit}" if unit else ""
        offs = "  ".join(f"C{c}={o:+.4g}" for c, o in self.offsets.items())
        verdict = (
            "PASSED. The cavities differ systematically, so the cavity label may help."
            if self.passed
            else "NOT PASSED. No cavity effect of useful size was found, so leave the cavity label out."
        )
        p_txt = "<0.001" if self.p_value < 0.001 else f"{self.p_value:.4f}"
        return "\n".join(
            [
                "STEP 1  Cavity-effect diagnostic",
                f"  parts {self.n_parts}, shots {self.n_shots}, cavities {self.n_cavities}",
                f"  sigma_blind (within shot)      {self.sigma_blind:.4g}{u}",
                f"  sigma_noise (offsets removed)  {self.sigma_noise:.4g}{u}",
                f"  cavity offsets                 {offs}",
                f"  F({self.df1}, {self.df2}) = {self.F:.1f}, p {p_txt}, partial eta squared {self.partial_eta2:.3f}",
                f"  floor separation 1 - sigma_noise/sigma_blind = {100 * self.separation:.1f} %",
                f"  Result: {verdict}",
            ]
        )

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.report()


def _pooled_ss(values: np.ndarray, shots: np.ndarray):
    """Pooled within-shot sum of squares and degrees of freedom."""
    frame = pd.DataFrame({"v": values, "s": shots})
    grp = frame.groupby("s")["v"]
    n = grp.transform("size").to_numpy()
    centered = values - grp.transform("mean").to_numpy()
    ok = n >= 2
    ss = float(np.sum(centered[ok] ** 2))
    dof = int(ok.sum() - frame.loc[ok, "s"].nunique())
    return ss, dof


def cavity_diagnostic(
    response,
    shot,
    cavity,
    alpha: float = 0.05,
    min_partial_eta2: float = 0.01,
) -> DiagnosticResult:
    """Test whether the cavities of a multi-cavity mold differ systematically.

    Parameters
    ----------
    response : array-like
        Measured quality characteristic, one value per part.
    shot : array-like
        Shot (machine cycle) identifier of every part.
    cavity : array-like
        Cavity number of every part.
    alpha : float
        Significance level of the F test (default 0.05).
    min_partial_eta2 : float
        Smallest partial eta squared that counts as a useful effect. The default
        0.01 is the conventional lower bound of a small effect (Cohen, 1988).

    Returns
    -------
    DiagnosticResult
    """
    y = np.asarray(response, dtype=float)
    s = np.asarray(shot).astype(str)
    c = np.asarray(cavity)
    if not (len(y) == len(s) == len(c)):
        raise ValueError("response, shot and cavity must have the same length.")
    ok = np.isfinite(y) & ~pd.isna(c)
    y, s, c = y[ok], s[ok], c[ok]
    if len(y) < 8:
        raise ValueError("At least 8 parts are needed for the diagnostic.")
    cav_levels = np.unique(c)
    k = len(cav_levels)
    S = len(np.unique(s))
    if k < 2:
        raise ValueError("The diagnostic needs at least two cavities.")

    grand = float(y.mean())
    var_total = float(np.var(y, ddof=1))

    ss_blind, dof_within = _pooled_ss(y, s)
    if dof_within <= 0:
        raise ValueError("Every shot holds a single part, so no within-shot spread exists.")
    sigma_blind = float(np.sqrt(ss_blind / dof_within))

    offsets = {}
    adj = y.copy()
    for lev in cav_levels:
        m = c == lev
        off = float(y[m].mean() - grand)
        offsets[lev.item() if hasattr(lev, "item") else lev] = off
        adj[m] = y[m] - off
    ss_noise, _ = _pooled_ss(adj, s)
    sigma_noise = float(np.sqrt(ss_noise / dof_within))

    # exact within-shot least squares F test
    frame = pd.DataFrame({"y": y, "s": s, "c": c})
    y_dm = (frame["y"] - frame.groupby("s")["y"].transform("mean")).to_numpy()
    dummies = pd.get_dummies(frame["c"], drop_first=True).astype(float)
    xdm = (dummies - dummies.groupby(frame["s"].to_numpy()).transform("mean")).to_numpy()
    ss_reduced = float(np.sum(y_dm**2))
    beta, *_ = np.linalg.lstsq(xdm, y_dm, rcond=None)
    ss_full = float(np.sum((y_dm - xdm @ beta) ** 2))
    df1 = k - 1
    df2 = int(len(y) - S - (k - 1))
    F = ((ss_reduced - ss_full) / df1) / (ss_full / df2)
    p = float(stats.f.sf(F, df1, df2))
    eta2 = float((ss_reduced - ss_full) / ss_reduced)
    f_floors = (S - 1) * (sigma_blind**2 - sigma_noise**2) / sigma_noise**2

    passed = bool(p < alpha and eta2 >= min_partial_eta2)
    return DiagnosticResult(
        n_parts=int(len(y)),
        n_shots=int(S),
        n_cavities=int(k),
        sigma_total=float(np.sqrt(var_total)),
        sigma_blind=sigma_blind,
        sigma_noise=sigma_noise,
        offsets=offsets,
        F=float(F),
        df1=int(df1),
        df2=int(df2),
        p_value=p,
        partial_eta2=eta2,
        separation=float(1.0 - sigma_noise / sigma_blind),
        alpha=float(alpha),
        min_partial_eta2=float(min_partial_eta2),
        passed=passed,
        r2_max_blind=float(1.0 - sigma_blind**2 / var_total),
        r2_max_aware=float(1.0 - sigma_noise**2 / var_total),
        F_from_floors=float(f_floors),
    )
