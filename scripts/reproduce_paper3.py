"""Reproduce the tip diameter and tensile strength results on your own data.

The production data of the original study are confidential and are not part of
this repository. If you have the two training files, run

    python scripts/reproduce_paper3.py --tip path/to/Tip_TrainingData.csv \
                                       --tensile path/to/Tensile_TrainingData.csv

Both files need the columns ``Value``, ``CavityNumber`` and ``ID_Zk``. The feature sets
are the deployment sets of the original study. A random forest with the baseline
settings of that study is used. The script prints the F test, the validation, the
leakage audit and the decision for each characteristic.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

sys.path.insert(0, "src")

from cavity_encoding_ml import CavityConfig, CavityWorkflow, screen_outliers  # noqa: E402
from cavity_encoding_ml.cli import _default_model  # noqa: E402

SENSORS = {
    "Max": [f"ActPrsCavity{i}Max" for i in range(1, 5)],
    "SwitchOver": [f"ActPrsCavity{i}SwitchOver" for i in range(1, 5)],
}
SETS = {
    "Tip": dict(
        unit="mm",
        shared=["ActFlwCoolingUnit3", "ActFlwCoolingUnit1", "ActTimPlastification",
                "ActFlwCoolingUnit5", "ActPrsInjectionMax"],
    ),
    "Tensile": dict(unit="N", shared=["ActPrsInjectionMax", "ActVolMaterialCushion"]),
}


def run(name, path):
    df = pd.read_csv(path)
    keep = screen_outliers(df["Value"], df["CavityNumber"])
    df = df[keep].reset_index(drop=True)
    print(f"\n{'#' * 70}\n{name}: {len(df)} parts after the outlier screen ({(~keep).sum()} removed)\n{'#' * 70}")
    cfg = CavityConfig(shot="ID_Zk", cavity="CavityNumber", response="Value",
                       shared=SETS[name]["shared"], sensors=SENSORS, cavities=[1, 2, 3, 4])
    res = CavityWorkflow(cfg, _default_model(), audit_kfold=True, response_unit=SETS[name]["unit"]).run(df)
    print(res.report(SETS[name]["unit"]))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tip")
    ap.add_argument("--tensile")
    args = ap.parse_args()
    if not (args.tip or args.tensile):
        ap.error("give at least one of --tip and --tensile")
    if args.tip:
        run("Tip", args.tip)
    if args.tensile:
        run("Tensile", args.tensile)


if __name__ == "__main__":
    main()
