"""The guided workflow in one call."""

from sklearn.ensemble import RandomForestRegressor

from cavity_encoding_ml import CavityWorkflow
from cavity_encoding_ml.datasets import load_tip_example, tip_example_config

workflow = CavityWorkflow(
    tip_example_config(),
    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1),
    encoding="onehot",     # or "target", "integer", "auto"
    # arms=("blind", "readings", "encoded"),   # also train R1, the readings without the label
    # audit_kfold=True,                         # also show how much ordinary K-fold would flatter the model
    response_unit="mm",
)
result = workflow.run(load_tip_example())
print(result.report("mm"))
print("\nVerdict:", result.verdict)
