"""Bundled synthetic example data."""

from __future__ import annotations

import os

import pandas as pd

from .mapper import CavityConfig

_PATH = os.path.join(os.path.dirname(__file__), "data", "tip_example.csv")


def load_tip_example() -> pd.DataFrame:
    """Synthetic tip-diameter data of a four-cavity mold (96 shots, 384 parts).

    The values are generated, not measured. They copy the variance structure of
    the tip diameter of the original study. The unit of ``Value`` is mm.
    """
    return pd.read_csv(_PATH)


def tip_example_config() -> CavityConfig:
    """The :class:`CavityConfig` that matches :func:`load_tip_example`."""
    return CavityConfig(
        shot="ID_Zk",
        cavity="CavityNumber",
        response="Value",
        shared=[
            "ActFlwCoolingUnit3",
            "ActFlwCoolingUnit1",
            "ActTimPlastification",
            "ActFlwCoolingUnit5",
            "ActPrsInjectionMax",
        ],
        sensors={
            "Max": [f"ActPrsCavity{i}Max" for i in range(1, 5)],
            "SwitchOver": [f"ActPrsCavity{i}SwitchOver" for i in range(1, 5)],
        },
        cavities=[1, 2, 3, 4],
    )
