"""Step 4. Compare the floor separation with the fold-to-fold jitter."""

from __future__ import annotations

import textwrap
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
    gain: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)

    def report(self) -> str:
        paras = self.message.split("\n")
        body = "\n".join(
            textwrap.fill(p, width=92, initial_indent="  ", subsequent_indent="  ") for p in paras
        )
        return "STEP 4  Decision\n" + body

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.report()


def decide(
    separation: float,
    jitter: float,
    margin: float = DEFAULT_MARGIN,
    diagnostic_passed: bool = True,
    gain: float = float("nan"),
) -> Decision:
    """Decide whether the cavity information is worth keeping.

    The floor separation ``1 - sigma_noise / sigma_blind`` is the largest relative
    RMSE reduction that cavity information can deliver. The jitter is the standard
    deviation of the fold RMSE of the cavity-blind model divided by its mean. When
    the separation is smaller than ``margin`` times the jitter, a gain of that size
    cannot be told apart from the fold-to-fold variation, so the cavity information
    adds noise. ``gain`` is the measured RMSE reduction of the encoded model and is
    only used to make the message more specific.
    """
    ratio = float(separation / jitter) if jitter > 0 else float("inf")
    sep_txt, jit_txt = f"{100 * separation:.1f} %", f"{100 * jitter:.1f} %"
    has_gain = gain == gain  # not nan
    gain_txt = f"{100 * gain:.1f} %" if has_gain else ""
    if not diagnostic_passed:
        verdict = "no_cavity_effect"
        msg = (
            "Recommendation: leave the cavity label out.\n"
            "The cavity-effect test did not pass. The cavities do not differ enough to matter, "
            "so one model without the label serves all cavities. Keep the shot-grouped "
            "validation, because the parts of one shot still share the same machine state."
        )
    elif separation < margin * jitter:
        verdict = "adds_noise"
        msg = (
            "Recommendation: leave the cavity label out. Cavity information adds noise and is "
            "not useful here.\n"
            f"The cavities do differ (the F test passed), but the largest improvement they allow, "
            f"the floor separation of {sep_txt}, is below {margin:g} times the fold-to-fold jitter "
            f"of {jit_txt} (ratio {ratio:.1f}). A gain this small cannot be separated from the "
            "spread between folds, and the extra label columns make the model more complex without "
            "a reliable benefit."
        )
        if has_gain and gain > 0:
            msg += f" The gain of {gain_txt} seen in this run lies within that spread."
        msg += (" Use the cavity-blind model, or keep the cavity sensor readings if they "
                "clearly helped in Step 3.")
    else:
        verdict = "use_cavity"
        msg = (
            "Recommendation: keep the cavity label in the model. Cavity information is worth "
            "keeping.\n"
            f"The cavities differ systematically, and the largest improvement they allow, the "
            f"floor separation of {sep_txt}, is {ratio:.1f} times the fold-to-fold jitter of "
            f"{jit_txt}, above the margin of {margin:g}. A gain from the label is therefore real "
            "and does not depend on which shots happen to land in which fold."
        )
        if has_gain and gain > 0:
            msg += f" In this run the label lowered the error by {gain_txt}."
        elif has_gain:
            msg += (" In this run, however, the label did not lower the error. Check the "
                    "features, the model or the encoding before relying on it.")
    return Decision(
        verdict=verdict,
        separation=float(separation),
        jitter=float(jitter),
        ratio=ratio,
        margin=float(margin),
        diagnostic_passed=bool(diagnostic_passed),
        message=msg,
        gain=float(gain),
    )


def decide_from(diagnostic, validation, margin: float = DEFAULT_MARGIN) -> Decision:
    """Apply :func:`decide` to a diagnostic result and a validation result."""
    gain = validation.gain() if "encoded" in validation.arms else float("nan")
    return decide(
        separation=diagnostic.separation,
        jitter=validation.jitter,
        margin=margin,
        diagnostic_passed=diagnostic.passed,
        gain=gain,
    )
