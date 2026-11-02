"""Optional figure of the decision (needs matplotlib)."""

from __future__ import annotations


def plot_decision(result, ax=None):
    """Bar chart of the floor separation against the fold-to-fold jitter.

    ``result`` is a :class:`WorkflowResult`. Returns the matplotlib axes.
    """
    import matplotlib.pyplot as plt

    d = result.decision
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3.0))
    vals = [100 * d.separation, 100 * d.jitter]
    bars = ax.bar(["floor separation", "fold-to-fold jitter"], vals, color=["#2166AC", "#999999"])
    ax.axhline(100 * d.margin * d.jitter, color="#B2182B", ls="--", lw=1)
    ax.text(1.45, 100 * d.margin * d.jitter, f"{d.margin:g} x jitter", color="#B2182B", ha="right", va="bottom", fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f} %", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("RMSE (%)")
    ax.set_title(d.verdict.replace("_", " "))
    return ax
