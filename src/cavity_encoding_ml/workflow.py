"""The guided workflow: diagnostic, encoding, validation and decision in one call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .decision import DEFAULT_MARGIN, Decision, decide_from
from .diagnostic import DiagnosticResult, cavity_diagnostic
from .encoding import choose_encoding
from .mapper import CavityConfig
from .validation import ValidationResult, leakage_audit, validate_cavity_encoding


@dataclass
class WorkflowResult:
    """All results of one workflow run."""

    diagnostic: DiagnosticResult
    validation: Optional[ValidationResult]
    decision: Decision
    audit: Optional[dict] = None
    encoding: str = "onehot"

    @property
    def verdict(self) -> str:
        return self.decision.verdict

    def report(self, unit: str = "") -> str:
        parts = ["cavity-encoding-ml report", "=" * 60, self.diagnostic.report(unit)]
        if self.validation is None:
            parts.append("STEP 2-3  Skipped, because the diagnostic did not pass.")
        else:
            parts.append(f"STEP 2  Encoding: {self.encoding}")
            parts.append(self.validation.report())
            if self.audit is not None:
                a = self.audit
                parts.append(
                    "  Leakage audit of ordinary K-fold: "
                    f"{100 * a['sibling_share']:.1f} % of the test parts had a sibling in training, "
                    f"and the RMSE of the blind model was understated by {100 * a['understatement']:.1f} %."
                )
        parts.append(self.decision.report())
        return "\n".join(parts)

    def to_dict(self) -> dict:
        out = {"diagnostic": self.diagnostic.to_dict(), "decision": self.decision.to_dict(), "encoding": self.encoding}
        if self.validation is not None:
            v = self.validation
            out["validation"] = {
                "n_folds": v.n_folds,
                "leakage_share": v.leakage_share,
                "jitter": v.jitter,
                "rmse": {k: a.rmse for k, a in v.arms.items()},
            }
        if self.audit is not None:
            out["audit"] = self.audit
        return out

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.report()


class CavityWorkflow:
    """Run the four steps for one quality characteristic.

    Parameters
    ----------
    config : CavityConfig
    model : object with fit/predict, FunctionModel or factory
    encoding : {"onehot", "target", "integer", "auto"}
        One-hot is the default.
    margin : float
        The cavity label is called noise when the floor separation is below
        ``margin`` times the fold-to-fold jitter (default 2).
    cv : splitter, optional
        Default ``ShotGroupKFold(4, 3, 42)``.
    audit_kfold : bool
        Also measure how much ordinary K-fold would flatter the blind model.
    alpha, min_partial_eta2 : float
        Settings of the F test.
    response_unit : str
        Unit text for the report.
    arms : tuple of str
        Models trained for the comparison. Default ``("blind", "readings", "encoded")``,
        which shows the gain of the cavity sensor readings (R1) and of the label separately.
        ``("blind", "encoded")`` is all the decision needs and saves one training run.
    """

    def __init__(
        self,
        config: CavityConfig,
        model,
        encoding: str = "onehot",
        margin: float = DEFAULT_MARGIN,
        cv=None,
        audit_kfold: bool = False,
        alpha: float = 0.05,
        min_partial_eta2: float = 0.01,
        response_unit: str = "",
        always_validate: bool = False,
        arms=("blind", "readings", "encoded"),
    ):
        self.config = config
        self.model = model
        self.encoding = encoding
        self.margin = margin
        self.cv = cv
        self.audit_kfold = audit_kfold
        self.alpha = alpha
        self.min_partial_eta2 = min_partial_eta2
        self.response_unit = response_unit
        self.always_validate = always_validate
        self.arms = (arms,) if isinstance(arms, str) else tuple(arms)
        if "blind" not in self.arms or "encoded" not in self.arms:
            raise ValueError(
                "The decision compares the encoded model with the cavity-blind model, so arms "
                "must contain 'blind' and 'encoded'. 'readings' is optional. To only "
                "train one model with the label, use validate_cavity_encoding(arms=('encoded',))."
            )

    @classmethod
    def from_config(cls, path: str, model, **kwargs) -> "CavityWorkflow":
        """Build the workflow from a JSON file written by ``CavityConfig.to_json``."""
        return cls(CavityConfig.from_json(path), model, **kwargs)

    def run(self, data: pd.DataFrame) -> WorkflowResult:
        cfg = self.config
        diag = cavity_diagnostic(
            data[cfg.response], data[cfg.shot], data[cfg.cavity], self.alpha, self.min_partial_eta2
        )
        encoding = self.encoding
        if encoding == "auto":
            encoding = choose_encoding(diag.n_cavities)
        validation = None
        audit = None
        if diag.passed or self.always_validate:
            validation = validate_cavity_encoding(
                data, cfg, self.model, encoding=encoding, cv=self.cv, response_unit=self.response_unit,
                arms=self.arms,
            )
            if self.audit_kfold:
                audit = leakage_audit(data, cfg, self.model)
        if validation is None:
            from .decision import decide

            decision = decide(diag.separation, float("nan"), self.margin, diagnostic_passed=False)
        else:
            decision = decide_from(diag, validation, self.margin)
        return WorkflowResult(diag, validation, decision, audit, encoding)
