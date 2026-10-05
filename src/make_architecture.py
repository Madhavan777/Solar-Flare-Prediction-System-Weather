"""Vector-quality (matplotlib) system architecture diagram, replacing the blurry
raster figure in the original draft. Reflects what was actually built:
SWAN-SF -> leakage-safe temporal feature extraction -> LR/RF -> selection on
validation -> risk assessment/alert -> dashboard, with NOAA GOES kept as a
separate monitoring branch (not fed into the trained model).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

NAVY, LIGHT, GREY, GREEN = "#1f3a5f", "#eaf1fb", "#6b7280", "#2e7d32"

fig, ax = plt.subplots(figsize=(10.5, 8.6))
ax.set_xlim(0, 10.5); ax.set_ylim(0, 13.2); ax.axis("off")


def box(cx, cy, w, h, text, fc=LIGHT, ec=NAVY, fs=9.3, tcolor="black"):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.06,rounding_size=0.08",
                       linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tcolor, linespacing=1.35,
            fontweight="bold" if fs > 9 else "normal")
    return (cx, cy, w, h)


def arrow(b1, b2, side1="bottom", side2="top", **kw):
    x1, y1, w1, h1 = b1; x2, y2, w2, h2 = b2
    pts = {"bottom": (x1, y1 - h1 / 2), "top": (x1, y1 + h1 / 2), "left": (x1 - w1 / 2, y1), "right": (x1 + w1 / 2, y1)}
    p1 = pts[side1]
    pts2 = {"bottom": (x2, y2 - h2 / 2), "top": (x2, y2 + h2 / 2), "left": (x2 - w2 / 2, y2), "right": (x2 + w2 / 2, y2)}
    p2 = pts2[side2]
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, linewidth=1.3,
                                 color=kw.get("color", NAVY), connectionstyle=kw.get("conn", "arc3,rad=0.0")))


cx = 5.1
b1 = box(cx, 12.35, 6.6, 0.72, "Historical SWAN-SF Solar Active-Region Data\n(SDO/HMI SHARP parameters, 12-min cadence)")
b2 = box(cx, 10.9, 6.6, 0.72, "Data Validation & Quality Checks\n(timestamps, QUALITY flag, missingness)")
b3 = box(cx, 9.8, 6.6, 0.72, "12 h Observation Window\n(past data only, per active region)")
b4 = box(cx, 8.7, 6.6, 0.72, "Leakage-Safe Feature Construction\n24 SHARP parameters × 6 statistics = 144 features")
b5 = box(cx, 7.6, 6.6, 0.72, "24 h Prediction Window → Target Label\ny=1 if max flare ∈ {M,X}; y=0 otherwise")
b6 = box(cx, 6.5, 6.6, 0.72, "Temporal Train / Validation / Test Split\nby SWAN-SF partition (P1–P3 / P4 / P5), no shuffling")

blr = box(3.15, 5.3, 3.0, 0.72, "Logistic Regression\n(Iteration 1: point-in-time →\nIteration 2: temporal features)")
brf = box(7.05, 5.3, 3.0, 0.72, "Random Forest\n(Iteration 2 refinement,\ntemporal features)")

b7 = box(cx, 4.1, 6.6, 0.72, "Evaluation & Model Selection on Validation (P4)\nPrecision / Recall / F1 / PR-AUC / ROC-AUC / TSS", fc="#fff3e0", ec="#b06a00")
b8 = box(cx, 3.0, 6.6, 0.6, "Locked Test-Set Evaluation (P5) — evaluated once", fc="#fff3e0", ec="#b06a00")
b9 = box(cx, 1.95, 6.6, 0.62, "Risk Assessment (Low / Moderate / High)\nthreshold fixed on validation, not test")
b10 = box(cx, 0.95, 6.6, 0.6, "Alert Generation → Web Dashboard", fc=NAVY, ec=NAVY, tcolor="white")

bnoaa = box(9.35, 1.95, 2.0, 1.0, "NOAA GOES\nCurrent Monitoring\n(context only,\nnot a model input)", fc="#f4f4f4", ec=GREY, fs=8)

for a, b in [(b1, b2), (b2, b3), (b3, b4), (b4, b5), (b5, b6)]:
    arrow(a, b)
arrow(b6, blr, side1="bottom", side2="top", conn="arc3,rad=-0.15")
arrow(b6, brf, side1="bottom", side2="top", conn="arc3,rad=0.15")
arrow(blr, b7, side1="bottom", side2="top", conn="arc3,rad=0.15")
arrow(brf, b7, side1="bottom", side2="top", conn="arc3,rad=-0.15")
arrow(b7, b8); arrow(b8, b9); arrow(b9, b10)
arrow(bnoaa, b10, side1="left", side2="right", conn="arc3,rad=0.15", color=GREY)

ax.text(0.15, 12.95, "Figure 4.1 — Proposed / Implemented End-to-End Solar Flare Forecasting Architecture",
        fontsize=9.5, style="italic", color="#333")

legend_elems = [Line2D([0], [0], color=NAVY, lw=1.3, label="Data / model pipeline"),
                Line2D([0], [0], color=GREY, lw=1.3, label="Contextual (non-ML) monitoring")]
ax.legend(handles=legend_elems, loc="lower left", bbox_to_anchor=(0.0, -0.04), fontsize=7.5, frameon=False, ncol=2)

plt.tight_layout()
plt.savefig("figures/fig4_1_architecture.png", dpi=220, facecolor="white")
print("done")
