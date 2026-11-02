"""Step 1 alone: do the cavities differ systematically?

Only the response, the shot number and the cavity number are needed.
"""

from cavity_encoding_ml import cavity_diagnostic
from cavity_encoding_ml.datasets import load_tip_example

df = load_tip_example()
result = cavity_diagnostic(df["Value"], df["ID_Zk"], df["CavityNumber"])
print(result.report("mm"))
print("\nPassed:", result.passed)
