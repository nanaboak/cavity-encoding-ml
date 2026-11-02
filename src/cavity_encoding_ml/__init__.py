"""cavity-encoding-ml: cavity encoding and leakage-safe validation for multi-cavity molds.

The package follows four steps.

1. ``cavity_diagnostic``: do the cavities differ systematically (F test)?
2. ``CavityMapper`` / ``CavityEncoder``: encode the cavity label (one-hot by default).
3. ``validate_cavity_encoding``: validate on shot-grouped folds and warn on leakage.
4. ``decide``: compare the floor separation with the fold-to-fold jitter.

``CavityWorkflow`` runs all four steps in one call.
"""

from .decision import Decision, decide, decide_from
from .diagnostic import DiagnosticResult, cavity_diagnostic
from .encoding import CavityEncoder, choose_encoding
from .mapper import REPRESENTATIONS, CavityConfig, CavityMapper
from .models import FunctionModel, make_model
from .preprocessing import screen_outliers
from .splitting import (
    LeakageError,
    LeakageWarning,
    RepeatedRowKFold,
    ShotGroupKFold,
    sibling_share,
)
from .validation import (
    ArmResult,
    ValidationResult,
    compare_encodings,
    leakage_audit,
    validate_cavity_encoding,
)
from .workflow import CavityWorkflow, WorkflowResult

__version__ = "0.1.0"

__all__ = [
    "ArmResult",
    "CavityConfig",
    "CavityEncoder",
    "CavityMapper",
    "CavityWorkflow",
    "Decision",
    "DiagnosticResult",
    "FunctionModel",
    "LeakageError",
    "LeakageWarning",
    "REPRESENTATIONS",
    "RepeatedRowKFold",
    "ShotGroupKFold",
    "ValidationResult",
    "WorkflowResult",
    "cavity_diagnostic",
    "choose_encoding",
    "compare_encodings",
    "decide",
    "decide_from",
    "leakage_audit",
    "make_model",
    "screen_outliers",
    "sibling_share",
    "validate_cavity_encoding",
    "__version__",
]
