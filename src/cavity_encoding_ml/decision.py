"""Step 4. Compare the floor separation with the fold-to-fold jitter."""

from __future__ import annotations

from dataclasses import asdict, dataclass

DEFAULT_MARGIN = 2.0


@dataclass
class Decision:
    """Outcome of the decision rule."""

    verdict: str  # "use_cavity", "adds_noise" or "no_cavity_effect"
    separation: float
    jitter: float
    ratio: float
    margin: float
    diagnostic_passed: bool
    message: str

    def to_dict(self) -> dict:
        return asdict(self)

    def report(self) -> str:
        return "STEP 4  Decision\n  " + self.message.replace("\n", "\n  ")

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.report()


def decide(separation: float, jitter: float, margin: float = DEFAULT_MARGIN, diagnostic_passed: bool = True) -> Decision:
    """Decide whether the cavity information is worth keeping.

    The floor separation ``1 - sigma_noise / sigma_blind`` is the largest relative
    RMSE reduction that cavity information can deliver. The jitter is the standard
    deviation of the fold RMSE of the cavity-blind model divided by its mean. When
    the separation is smaller than ``margin`` times the jitter, a gain of that size
    cannot be told apart from the fold-to-fold variation, so the cavity information
    adds noise.
    """
    ratio = float(separation / jitter) if jitter > 0 else float("inf")
    sep_txt, jit_txt = f"{100 * separation:.1f} %", f"{100 * jitter:.1f} %"
    if not diagnostic_passed:
        verdict = "no_cavity_effect"
        msg = (
            "The cavity-effect test did not pass. The cavities do not differ enough to matter.\n"
            "Leave the cavity label out."
        )
    elif separation < margin * jitter:
        verdict = "adds_noise"
        msg = (
            f"Cavity information adds noise and is not useful, leave the cavity label out.\n"
            f"The floor separation of {sep_txt} is below {margin:g} times the fold-to-fold jitter "
            f"of {jit_txt} (ratio {ratio:.1f}). A gain this small cannot be separated from the "
            f"spread between folds."
        )
    else:
        verdict = "use_cavity"
        msg = (
            f"Cavity information is worth keeping, use the encoded cavity label.\n"
            f"The floor separation of {sep_txt} is {ratio:.1f} times the fold-to-fold jitter "
            f"of {jit_txt}, which is above the margin of {margin:g}."
        )
    return Decision(
        verdict=verdict,
        separation=float(separation),
        jitter=float(jitter),
        ratio=ratio,
        margin=float(margin),
        diagnostic_passed=bool(diagnostic_passed),
        message=msg,
    )


def decide_from(diagnostic, validation, margin: float = DEFAULT_MARGIN) -> Decision:
    """Apply :func:`decide` to a diagnostic result and a validation result."""
    return decide(
        separation=diagnostic.separation,
        jitter=validation.jitter,
        margin=margin,
        diagnostic_passed=diagnostic.passed,
    )
