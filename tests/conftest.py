import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from cavity_encoding_ml import CavityConfig


@pytest.fixture(scope="session")
def toy():
    """Small four-cavity data set with a clear cavity effect."""
    rng = np.random.default_rng(0)
    S, offs = 40, np.array([2.0, -3.0, 4.0, -3.0])
    rows = []
    for s in range(S):
        x1, x2 = rng.normal(size=2)
        shot_eff = 3 * x1 + rng.normal(0, 3)
        p = 100 + 2 * x2 + rng.normal(0, 1, 4)
        for c in range(4):
            rows.append(
                dict(shot=1000 + s, cav=c + 1, y=50 + shot_eff + offs[c] + rng.normal(0, 0.5),
                     x1=x1, x2=x2, p1=p[0], p2=p[1], p3=p[2], p4=p[3])
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def toy_cfg():
    return CavityConfig(shot="shot", cavity="cav", response="y", shared=["x1", "x2"],
                        sensors={"P": ["p1", "p2", "p3", "p4"]}, cavities=[1, 2, 3, 4])


@pytest.fixture(scope="session")
def noeffect():
    """Data without any cavity effect."""
    rng = np.random.default_rng(1)
    rows = []
    for s in range(40):
        eff = rng.normal(0, 3)
        for c in range(4):
            rows.append(dict(shot=s, cav=c + 1, y=eff + rng.normal(0, 1), x1=float(s % 7),
                             p1=0.0, p2=0.0, p3=0.0, p4=0.0))
    return pd.DataFrame(rows)


@pytest.fixture
def ridge():
    return Ridge(alpha=1.0)
