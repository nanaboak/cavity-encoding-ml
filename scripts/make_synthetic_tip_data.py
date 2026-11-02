"""Create the synthetic tip-diameter example data.

The values are generated, not measured. They copy the variance structure that the
original study reports for the tip diameter of a four-cavity mold: 96 shots, a
within-shot spread of about 10.4 um, about 4.9 um after the cavity offsets are
removed, and cavity offsets of about +1.4, -10.5, +11.6 and -2.5 um.

Run from the repository root:

    python scripts/make_synthetic_tip_data.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

SEED = 2026
N_SHOTS = 96
OFFSETS_UM = np.array([1.4, -10.5, 11.6, -2.5])  # cavity offsets in micrometres
NOISE_UM = 4.94  # part-to-part noise after the offsets are removed
SHOT_PREDICTABLE_UM = 7.5  # shot effect explained by machine features
SHOT_UNEXPLAINED_UM = 12.8  # shot effect that no feature explains
MEAN_MM = 90.2554


def ar1(rng, n, phi=0.6):
    x = np.zeros(n)
    e = rng.normal(size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + np.sqrt(1 - phi**2) * e[i]
    return x


def main(out_path: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    z = np.column_stack([ar1(rng, N_SHOTS) for _ in range(5)])  # five machine features
    shared = pd.DataFrame(
        {
            "ActFlwCoolingUnit3": 6.6 + 0.45 * z[:, 0],
            "ActFlwCoolingUnit1": 9.8 + 0.60 * z[:, 1],
            "ActTimPlastification": 16.6 + 0.15 * z[:, 2],
            "ActFlwCoolingUnit5": 7.2 + 0.50 * z[:, 3],
            "ActPrsInjectionMax": 1078.0 + 6.0 * z[:, 4],
        }
    )
    w = np.array([0.55, 0.45, -0.40, 0.35, 0.45])
    w = w / np.linalg.norm(w)
    shot_effect = SHOT_PREDICTABLE_UM * (z @ w) + rng.normal(0, SHOT_UNEXPLAINED_UM, N_SHOTS)
    shot_ids = 706000 + np.cumsum(rng.integers(900, 1500, N_SHOTS))

    rows = []
    for s in range(N_SHOTS):
        eps = rng.normal(0, 1, 4)  # standardised part noise of the four cavities
        # cavity sensors: cavity differences, a shot-level part and a part-level part
        cav_gain = np.array([1.0, 1.0, 1.0, 1.0])
        p_max = 615.0 * cav_gain + 6.0 * z[s, 4] + 4.0 * eps + rng.normal(0, 5.0, 4)
        p_sw = 410.0 * cav_gain + 4.0 * z[s, 0] + 3.0 * eps + rng.normal(0, 4.0, 4)
        for c in range(4):
            noise = NOISE_UM * (0.5 * eps[c] + np.sqrt(1 - 0.25) * rng.normal())
            value = MEAN_MM + (shot_effect[s] + OFFSETS_UM[c] + noise) / 1000.0
            row = {"ID_Zk": int(shot_ids[s]), "CavityNumber": c + 1, "Value": round(value, 3)}
            row.update({k: round(float(v[s]), 3) for k, v in shared.items()})
            for i in range(4):
                row[f"ActPrsCavity{i + 1}Max"] = round(float(p_max[i]), 1)
            for i in range(4):
                row[f"ActPrsCavity{i + 1}SwitchOver"] = round(float(p_sw[i]), 1)
            rows.append(row)
    df = pd.DataFrame(rows)
    cols = (
        ["ID_Zk", "CavityNumber", "Value"]
        + list(shared.columns)
        + [f"ActPrsCavity{i}Max" for i in range(1, 5)]
        + [f"ActPrsCavity{i}SwitchOver" for i in range(1, 5)]
    )
    df = df[cols]
    if out_path is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_path = os.path.join(root, "src", "cavity_encoding_ml", "data", "tip_example.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows, {df['ID_Zk'].nunique()} shots, {len(df.columns)} columns)")
    return df


if __name__ == "__main__":
    main()
