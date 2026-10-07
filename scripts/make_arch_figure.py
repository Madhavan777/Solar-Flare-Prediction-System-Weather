"""Draw the architecture flowchart used on the review deck's Technical Approach slide.

    python scripts/make_arch_figure.py

Writes figures/extra/architecture_flow.png. The frozen figures/fig4_1_architecture.png
is left alone: this is a redraw for the slide, with the branch routed as a bus so no
two connectors cross, every arrowhead landing on a box edge rather than on its label,
and all text in black Times New Roman to match the deck.
"""

from __future__ import annotations

import itertools
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = pathlib.Path(__file__).resolve().parent.parent / "figures/extra/architecture_flow.png"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.color"] = "black"

W, H = 100.0, 111.0  # drawing units; the figure is sized so 1 pt here is 1 pt on the slide
MAIN_W, BRANCH_W = 84.0, 40.0
HALF, BRANCH_HALF = 3.3, 4.8  # box half-heights
GAP, BUS_GAP = 2.4, 5.0  # vertical clearance between boxes, and around the branch
CX, LX, RX = 50.0, 28.0, 72.0

TITLE_PT, SUB_PT = 8.5, 7.0

TOP = 106.5  # centre of the first box
STEP = 2 * HALF + GAP
FORK = HALF + BUS_GAP + BRANCH_HALF  # main box to branch box, and back

# (title, [sub-lines]); the y positions are derived below so the spacing stays even
MAIN_T = [
    ("Historical SWAN-SF Active-Region Data", ["SDO/HMI SHARP parameters, 12-minute cadence"]),
    ("Data Validation and Quality Checks", ["timestamps, QUALITY flag, missingness"]),
    ("12-Hour Observation Window", ["past data only, one active region"]),
    ("Leakage-Safe Feature Construction", ["24 SHARP parameters × 6 statistics = 144 features"]),
    ("24-Hour Prediction Window", ["y = 1 if the largest flare is M or X, otherwise y = 0"]),
    (
        "Temporal Train / Validation / Test Split",
        ["SWAN-SF partitions P1–P3 / P4 / P5, no shuffling"],
    ),
]
LOWER_T = [
    (
        "Evaluation and Model Selection on Validation (P4)",
        ["Precision  ·  Recall  ·  F1  ·  PR-AUC  ·  ROC-AUC  ·  TSS"],
    ),
    ("Locked Test-Set Evaluation (P5)", ["scored once, after every choice was already fixed"]),
    ("Risk Assessment — Low / Moderate / High", ["threshold fixed on validation, never on test"]),
    ("Alert Generation and Web Dashboard", []),
]
MAIN = [(TOP - i * STEP, t, s) for i, (t, s) in enumerate(MAIN_T)]
BRANCH_Y = MAIN[-1][0] - FORK
LOWER = [(BRANCH_Y - FORK - i * STEP, t, s) for i, (t, s) in enumerate(LOWER_T)]
BRANCHES = [
    (LX, "Logistic Regression", ["Iteration 1: point-in-time", "Iteration 2: temporal features"]),
    (RX, "Random Forest", ["Iteration 2 refinement,", "temporal features"]),
]

fig, ax = plt.subplots(figsize=(4.38, 4.86), dpi=300)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)


def draw_box(cx, cy, w, half, title, subs, lw=0.9):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - half),
            w,
            half * 2,
            boxstyle="round,pad=0,rounding_size=1.4",
            linewidth=lw,
            edgecolor="black",
            facecolor="white",
            zorder=3,
        )
    )
    lines = [(title, TITLE_PT, "bold")] + [(s, SUB_PT, "normal") for s in subs]
    # Stack the lines about the centre, in the units this axis is drawn in.
    step = 2.55
    top = cy + (len(lines) - 1) * step / 2
    for i, (text, pt, weight) in enumerate(lines):
        ax.text(
            cx,
            top - i * step,
            text,
            ha="center",
            va="center",
            fontsize=pt,
            fontweight=weight,
            color="black",
            zorder=4,
        )


def arrow(x0, y0, x1, y1, head=True):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops={
            "arrowstyle": "-|>" if head else "-",
            "color": "black",
            "linewidth": 0.9,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 7,
        },
        zorder=2,
    )


for cy, title, subs in MAIN:
    draw_box(CX, cy, MAIN_W, HALF, title, subs)
for cy, title, subs in LOWER:
    draw_box(CX, cy, MAIN_W, HALF, title, subs, lw=1.4 if "Locked" in title else 0.9)
for cx, title, subs in BRANCHES:
    draw_box(cx, BRANCH_Y, BRANCH_W, BRANCH_HALF, title, subs)

# Straight connectors down the main column.
for (y0, _, _), (y1, _, _) in itertools.pairwise(MAIN):
    arrow(CX, y0 - HALF, CX, y1 + HALF)
for (y0, _, _), (y1, _, _) in itertools.pairwise(LOWER):
    arrow(CX, y0 - HALF, CX, y1 + HALF)

# The split fans out through a horizontal bus, so nothing crosses.
split_y = MAIN[-1][0] - HALF
bus_down = BRANCH_Y + BRANCH_HALF + 1.9
arrow(CX, split_y, CX, bus_down, head=False)
arrow(LX, bus_down, RX, bus_down, head=False)
for x in (LX, RX):
    arrow(x, bus_down, x, BRANCH_Y + BRANCH_HALF)

# ... and the two branches rejoin through a second bus.
bus_up = BRANCH_Y - BRANCH_HALF - 1.9
for x in (LX, RX):
    arrow(x, BRANCH_Y - BRANCH_HALF, x, bus_up, head=False)
arrow(LX, bus_up, RX, bus_up, head=False)
arrow(CX, bus_up, CX, LOWER[0][0] + HALF)

ax.text(
    CX,
    LOWER[-1][0] - HALF - 3.0,
    "NOAA GOES current monitoring is shown in the dashboard for context only; "
    "it is not a model input.",
    ha="center",
    va="bottom",
    fontsize=6.0,
    style="italic",
    color="black",
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300, facecolor="white")
print(f"wrote {OUT.relative_to(OUT.parent.parent.parent)}")
