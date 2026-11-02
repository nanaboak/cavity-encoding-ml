"""Figure 1 of the JOSS paper, drawn from the verified results of the
tip-diameter and tensile-strength data (Paper 3 production data)."""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, BOX = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#f3f2ee"
RES = {"Tip diameter": dict(F=326.4, sep=52.5, jit=8.9, verdict="use cavity label"),
       "Tensile strength": dict(F=11.4, sep=1.1, jit=3.7, verdict="label adds noise")}


def box(ax, xy, w, h, title, body):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                facecolor=BOX, edgecolor="none"))
    ax.text(xy[0] + 0.03, xy[1] + h - 0.035, title, ha="left", va="top", fontsize=8.2,
            color=INK, fontweight="bold")
    ax.text(xy[0] + 0.03, xy[1] + h - 0.095, body, ha="left", va="top", fontsize=6.8,
            color=INK2, linespacing=1.15)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, color=MUTED, lw=1.0))


fig = plt.figure(figsize=(7.2, 3.3))
ax = fig.add_axes([0.0, 0.0, 0.47, 1.0])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.02, 0.975, "(a)", fontsize=9, color=INK, fontweight="bold", va="top")
w, h, x0 = 0.60, 0.175, 0.06
ys = [0.755, 0.525, 0.295, 0.065]
steps = [("1  Diagnostic", "F test of cavity within shot,\nnoise floors and floor separation"),
         ("2  Encoding", "one-hot (default), target or\ninteger, refitted per fold"),
         ("3  Validation", "shot-grouped folds, alert for\nsibling leakage under K-fold"),
         ("4  Decision", "floor separation against the\nfold-to-fold jitter")]
for y, (t, b) in zip(ys, steps):
    box(ax, (x0, y), w, h, t, b)
for y1, y2 in zip(ys[:-1], ys[1:]):
    arrow(ax, (x0 + w / 2, y1), (x0 + w / 2, y2 + h))
for y, txt in [(ys[0], "no usable\neffect"), (ys[3], "separation\n< 2 × jitter")]:
    arrow(ax, (x0 + w, y + h / 2), (x0 + w + 0.1, y + h / 2))
    ax.text(x0 + w + 0.11, y + h / 2, txt + ",\nleave the\nlabel out", fontsize=6.8,
            color=INK2, va="center", ha="left", linespacing=1.2)

ax2 = fig.add_axes([0.60, 0.20, 0.38, 0.68])
names = list(RES)
yk, hgt = [1.0, 0.0], 0.32
sep = [RES[n]["sep"] for n in names]; jit = [RES[n]["jit"] for n in names]
ax2.barh([y + hgt / 2 + 0.02 for y in yk], sep, height=hgt, color=BLUE, label="floor separation")
ax2.barh([y - hgt / 2 - 0.02 for y in yk], jit, height=hgt, color=ORANGE, label="fold-to-fold jitter")
for y, s, j in zip(yk, sep, jit):
    ax2.text(s + 1, y + hgt / 2 + 0.02, f"{s:.1f}%", va="center", fontsize=7.2, color=INK)
    ax2.text(j + 1, y - hgt / 2 - 0.02, f"{j:.1f}%", va="center", fontsize=7.2, color=INK)
ax2.set_yticks(yk)
ax2.set_yticklabels([f"{n}\nF = {RES[n]['F']:.1f}" for n in names], fontsize=7.6, color=INK)
ax2.set_xlabel("Percent of RMSE", fontsize=7.8, color=INK2)
ax2.set_xlim(0, max(sep + jit) * 1.25); ax2.set_ylim(-0.6, 1.6)
for s_ in ("top", "right"):
    ax2.spines[s_].set_visible(False)
for s_ in ("left", "bottom"):
    ax2.spines[s_].set_color(MUTED)
ax2.tick_params(labelsize=7.2, colors=INK2)
ax2.grid(axis="x", color=GRID, linewidth=0.6); ax2.set_axisbelow(True)
ax2.legend(frameon=False, fontsize=7.0, loc="lower right", labelcolor=INK2)
fig.text(0.53, 0.975, "(b)", fontsize=9, color=INK, fontweight="bold", va="top")
for y, n in zip(yk, names):
    ax2.text(ax2.get_xlim()[1], y + 0.42, RES[n]["verdict"], ha="right", va="bottom",
             fontsize=7.0, color=INK2, style="italic")
import os
fig.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figure1.png"), dpi=300)
print("saved")
