"""A case like the tensile strength of the original study.

The cavities differ in a statistically detectable way, but the difference is tiny
compared with the part-to-part scatter. The package reports that the cavity
information adds noise.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from cavity_encoding_ml import CavityConfig, CavityWorkflow

rng = np.random.default_rng(3)
S = 500
offsets = np.array([-3.6, -20.6, 9.1, 15.1])           # small against a scatter of about 100
rows = []
for s in range(S):
    x = rng.normal(size=2)
    shot_effect = 60 * x[0] + rng.normal(0, 90)
    p = 100 + rng.normal(0, 5, 4)
    for c in range(4):
        rows.append(dict(shot=s, cavity=c + 1, y=15000 + shot_effect + offsets[c] + rng.normal(0, 100),
                         x1=x[0], x2=x[1], p1=p[0], p2=p[1], p3=p[2], p4=p[3]))
df = pd.DataFrame(rows)

cfg = CavityConfig("shot", "cavity", "y", shared=["x1", "x2"], sensors={"P": ["p1", "p2", "p3", "p4"]})
result = CavityWorkflow(cfg, RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=1)).run(df)
print(result.report())
