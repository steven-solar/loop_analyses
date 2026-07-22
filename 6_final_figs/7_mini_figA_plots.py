import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Data
# -------------------------
counts = {
    "ESC": 51898 - 95,
    "EpiLC": 51898 - 225,
    "d4c7PGCLC": 51898 - 2910,
    "GSC": 51898 - 1364,
}

# -------------------------
# Load cell line colors
# Expects columns like: cell_line\tcolor
# -------------------------
cl_colors = pd.read_csv("../data/cl_colors.tsv", sep="\t")

# Adjust these column names if necessary
color_dict = dict(zip(cl_colors.iloc[:, 0], cl_colors.iloc[:, 1]))

colors = [color_dict[c] for c in counts]

# -------------------------
# Plot settings
# -------------------------
plt.rcParams.update({
    'svg.fonttype':  'none', 
    "font.size": 6,
    "axes.titlesize": 7,
    "axes.labelsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.25,
})

fig, ax = plt.subplots(figsize=(3.0, 2.2))

bars = ax.bar(
    counts.keys(),
    counts.values(),
    color=colors,
    edgecolor="black",
    linewidth=0.25,
)

# Annotate counts
ymax = max(counts.values())
for bar, val in zip(bars, counts.values()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + ymax * 0.01,
        f"n={val:,}",
        ha="center",
        va="bottom",
        fontsize=6,
    )

ax.set_ylabel("Non-NaN loops")
ax.set_title("Non-NaN loop counts")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(width=0.25, length=2)

ax.set_ylim(0, ymax * 1.08)

plt.tight_layout()
plt.savefig("non_nan_loop_counts.svg", format="svg", dpi=300)
plt.close()



import numpy as np
import matplotlib.pyplot as plt

# Publication-style settings
plt.rcParams.update({
    "font.size": 6,
    "axes.titlesize": 7,
    "axes.labelsize": 6,
    "axes.linewidth": 0.25,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
})

# Colors
blue = "#0072B2"
red = "#D55E00"

# Distance axis
s = np.linspace(0.05, 10, 600)

# Background interaction probability
bg = 1 / (s + 0.3) ** 1.3

# Total interaction probability (background + loop)
peak_center = 3.7
peak_width = 0.18
peak_height = 0.8
loop = peak_height * np.exp(-(s - peak_center) ** 2 / (2 * peak_width ** 2))
total = bg * 1.05 + loop

fig, ax = plt.subplots(figsize=(1.0, 1.3))

mask = total > bg
ax.fill_between(
    s,
    bg,
    total,
    where=mask,
    facecolor="none",
    hatch="////",
    edgecolor="0.3",   # hatch color
    linewidth=0.0,
    zorder=1,
)

# Curves
ax.plot(s, bg, color=blue, lw=0.1, label="Background")
ax.plot(s, total, color=red, lw=0.1, label="Total")

# Dashed guides
left = peak_center - 0.45
right = peak_center + 0.45
for x in [left, right]:
    ax.vlines(
        x,
        0,
        np.interp(x, s, bg),
        color="0.5",
        lw=0.1,
        ls=(0, (4, 3)),
    )



# Annotation
ax.annotate(
    "looping\nprobability",
    xy=(peak_center, np.interp(peak_center, s, total) * 0.75),
    xytext=(4.8, 0.8),
    fontsize=6,
    arrowprops=dict(arrowstyle="-", lw=0.1),
)

# Styling
ax.set_xlabel("genomic separation")
ax.set_ylabel("interaction probability")
ax.set_xlim(0, 10)
ax.set_ylim(0, 1.05)

# Remove ticks for a schematic look
ax.set_xticks([])
ax.set_yticks([])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Simple legend
leg = ax.legend(
    frameon=False,
    loc="lower left",
    bbox_to_anchor=(-0.02, -0.42),
    fontsize=6,
    handlelength=1.2,
)
for line in leg.get_lines():
    line.set_linewidth(0.25)

plt.savefig("toy_loop_probability.svg")
plt.show()