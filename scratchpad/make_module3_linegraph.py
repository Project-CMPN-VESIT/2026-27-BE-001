"""
Line graph: false-positive / false-alarm rate vs. number of independent
entities N. Every point below is a real measured value from
scratchpad/sweep_fpr_vs_n.py (see its two completed background runs).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BLUE = "#2a78d6"
ORANGE = "#eb6834"
ORANGE_LIGHT = "#f5b79a"
INK = "#0b0b0b"
MUTED = "#6b6a66"
GRID = "#e3e2dc"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

# Granger-VAR: scene-level (>=1 flagged pair in an N-entity independent scene)
granger_n = [2, 4, 8, 12, 16, 24, 32, 44]
granger_scene_raw = [0.100, 0.475, 0.963, 1.000, 1.000, 1.000, 1.000, 1.000]
granger_scene_corrected = [0.050, 0.125, 0.0741, 0.000, 0.000, 0.000, 0.1667, 0.000]

# Production engine: geometry-verified independent scenes (no interaction possible
# beyond N=12 within the projection's reliable region -- see script docstring)
engine_n = [2, 4, 6, 8, 10, 12]
engine_false_alarm = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

fig, ax = plt.subplots(figsize=(5.6, 3.6))

ax.plot(granger_n, granger_scene_raw, color=ORANGE_LIGHT, linewidth=2,
        marker="o", markersize=6, markeredgecolor="white", markeredgewidth=0.8,
        label="Granger-VAR, raw ($p<0.05$)", zorder=3)
ax.plot(granger_n, granger_scene_corrected, color=ORANGE, linewidth=2,
        marker="o", markersize=6, markeredgecolor="white", markeredgewidth=0.8,
        label="Granger-VAR, FDR-corrected", zorder=3)
ax.plot(engine_n, engine_false_alarm, color=BLUE, linewidth=2,
        marker="s", markersize=6, markeredgecolor="white", markeredgewidth=0.8,
        label="Production engine", zorder=4)

ax.axhline(0.05, color=MUTED, linewidth=1, linestyle="--", zorder=1,
           label="5% expected by chance")

ax.annotate("engine verified up to N=12\n(geometric limit of the\nreliable-projection region)",
            xy=(12, 0.0), xytext=(13.5, 0.62),
            fontsize=7.5, color=MUTED, ha="left",
            arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))

ax.set_xscale("log")
ax.set_xticks(granger_n)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_xlabel("Number of independent entities in the scene ($N$)")
ax.set_ylabel("False-alarm rate (scene contains ≥1 flagged edge)")
ax.set_ylim(-0.03, 1.08)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

ax.legend(loc="upper left", frameon=False, fontsize=8.7)
ax.set_title("False-Alarm Rate vs. Scene Complexity\n(every point independently measured)", fontsize=10)

fig.tight_layout()
fig.savefig("/Users/preetika/Downloads/chart_module3_fpr_vs_n.png", dpi=300, bbox_inches="tight")
fig.savefig("/Users/preetika/Capstone/chart_module3_fpr_vs_n.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote chart_module3_fpr_vs_n.png to Downloads and Capstone")
