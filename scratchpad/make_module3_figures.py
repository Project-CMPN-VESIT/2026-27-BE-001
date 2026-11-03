"""
Generates the two Module 3 (causal discovery) comparison figures for the
IEEE paper, from the real 1500-trial benchmark numbers (benchmark_synthetic.py)
and the real-event FDR illustration (compare_baselines.py).

Palette: validated categorical pair from the dataviz skill's reference
palette (fixed order, colorblind-safe): slot 1 blue = Production Engine,
slot 2 orange = Granger-VAR. Same mapping used in every figure.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

BLUE = "#2a78d6"      # Production Engine
ORANGE = "#eb6834"    # Granger-VAR (corrected / final)
ORANGE_LIGHT = "#f5b79a"  # Granger-VAR (raw / uncorrected) -- same hue, lighter step
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
    "svg.fonttype": "none",
})


def bar_label(ax, rects, fmt="{:.1%}", color=INK, dy=3):
    for r in rects:
        h = r.get_height()
        ax.annotate(fmt.format(h), (r.get_x() + r.get_width() / 2, h),
                    xytext=(0, dy), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=color)


# ── Figure 1: Accuracy & reliability comparison ──────────────────────────────
fig, ax = plt.subplots(figsize=(6.6, 3.4))

groups = ["Self-Detection\n(engine only)", "Attribution\n(given incident)",
          "End-to-End\n(engine only)", "False-Alarm Rate\n(null class, lower better)"]
engine_vals = [0.997, 0.931, 0.928, 0.000]
granger_vals = [None, 1.000, None, 0.148]

x = np.arange(len(groups))
w = 0.36

engine_bars = ax.bar(x - w / 2, engine_vals, width=w, color=BLUE, label="Production Engine",
                      zorder=3)
granger_x = [xi + w / 2 for xi, gv in zip(x, granger_vals) if gv is not None]
granger_y = [gv for gv in granger_vals if gv is not None]
granger_bars = ax.bar(granger_x, granger_y, width=w, color=ORANGE, label="Granger-VAR (FDR-corrected)",
                       zorder=3)

bar_label(ax, engine_bars)
bar_label(ax, granger_bars)

# N/A markers where Granger has no equivalent capability
for xi, gv in zip(x, granger_vals):
    if gv is None:
        ax.text(xi + w / 2, 0.02, "N/A", ha="center", va="bottom",
                 fontsize=8.5, color=MUTED, style="italic")

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=8.7)
ax.set_ylabel("Rate")
ax.set_ylim(0, 1.12)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False, fontsize=9)
ax.set_title("Module 3: Accuracy & False-Alarm Comparison\n(1,500 synthetic trials: 750 collision / 750 null)",
              fontsize=10, pad=34)
fig.tight_layout()
fig.savefig("/Users/preetika/Downloads/chart_module3_accuracy.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ── Figure 2: Runtime cost comparison (log scale) ────────────────────────────
fig, ax = plt.subplots(figsize=(4.6, 3.2))

rt_groups = ["Collision-class\nclip", "Null-class\nclip"]
rt_engine = [0.940, 0.091]
rt_granger = [0.039, 0.011]

x2 = np.arange(len(rt_groups))
b1 = ax.bar(x2 - w / 2, rt_engine, width=w, color=BLUE, label="Production Engine", zorder=3)
b2 = ax.bar(x2 + w / 2, rt_granger, width=w, color=ORANGE, label="Granger-VAR", zorder=3)
bar_label(ax, b1, fmt="{:.3f}s")
bar_label(ax, b2, fmt="{:.3f}s")

ax.set_yscale("log")
ax.set_ylabel("Mean runtime per trial (s, log scale)")
ax.set_xticks(x2)
ax.set_xticklabels(rt_groups)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.26), ncol=1, frameon=False, fontsize=9)
ax.set_title("Module 3: Compute Cost per Trial", fontsize=10, pad=38)
fig.tight_layout()
fig.savefig("/Users/preetika/Downloads/chart_module3_runtime.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ── Figure 3: Effect of multiple-comparisons correction ─────────────────────
fig, ax = plt.subplots(figsize=(4.6, 3.2))

fdr_groups = ["Synthetic null class\n(750 trials, trial-level)", "Real 44-object scene\n(118 pairs, pair-level)"]
raw_vals = [0.236, 0.458]      # trial-level raw false-alarm rate (750 trials) / real-scene pair-level raw (54/118)
corr_vals = [0.148, 0.212]     # trial-level FDR-corrected (matches paper text) / real-scene pair-level FDR-corrected (25/118)

x3 = np.arange(len(fdr_groups))
r1 = ax.bar(x3 - w / 2, raw_vals, width=w, color=ORANGE_LIGHT, label="Raw ($p<0.05$, uncorrected)", zorder=3)
r2 = ax.bar(x3 + w / 2, corr_vals, width=w, color=ORANGE, label="Benjamini--Hochberg corrected", zorder=3)
bar_label(ax, r1)
bar_label(ax, r2)

ax.axhline(0.05, color=MUTED, linewidth=1, linestyle="--", zorder=2)
ax.text(1.42, 0.062, "5% expected\nby chance", fontsize=7.5, color=MUTED, ha="left")

ax.set_ylabel("False-positive rate")
ax.set_ylim(0, 0.62)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xticks(x3)
ax.set_xticklabels(fdr_groups, fontsize=8.7)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=1, frameon=False, fontsize=8.5)
ax.set_title("Granger-VAR: Multiple-Comparisons Effect", fontsize=10, pad=44)
fig.tight_layout()
fig.savefig("/Users/preetika/Downloads/chart_module3_fdr_effect.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("wrote 3 figures to /Users/preetika/Downloads/")
