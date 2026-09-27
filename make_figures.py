"""Generate the genuine result figures from the trained model's real predictions.
No numbers here are invented; everything is read from results/*.json/*.csv.gz.
"""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
import joblib
from common import SHARP, STATS

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "figure.dpi": 200})
NAVY, RED, GREY, GREEN = "#1f3a5f", "#b3312c", "#6b7280", "#2e7d32"

sel = json.load(open("results/selection.json"))["selected"]
thr = json.load(open("results/selection.json"))["alert_threshold"]
test = json.load(open("results/test_results.json"))
val = json.load(open("results/validation_results.json"))

# ---------- Figure 6.1 Confusion matrix (final model, locked test partition P5) ----------
dp = pd.read_csv("results/test_predictions_selected.csv.gz")
y, p = dp.y.to_numpy(), dp.p.to_numpy()
cm = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1])
fig, ax = plt.subplots(figsize=(5.2, 4.6))
im = ax.imshow(cm, cmap="Blues")
labels = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{labels[i][j]}\n{cm[i,j]:,}\n({cm[i,j]/cm.sum():.1%})", ha="center", va="center",
                fontsize=11, color="white" if cm[i, j] > cm.max() * 0.5 else "black")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["No M/X (0)", "M/X (1)"]); ax.set_yticklabels(["No M/X (0)", "M/X (1)"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — Final Model\non Locked Test Partition (P5, N={cm.sum():,})")
plt.tight_layout(); plt.savefig("figures/fig6_1_confusion_matrix.png"); plt.close()

# ---------- Figure 6.2 Iteration comparison chart ----------
rows = [
    ("Iteration 1\n(LR, point-in-time)", val["I1_LR_last"]["val"], test["I1_LR_last"]),
    ("Iteration 2\n(RF, temporal)", val["I2_RF_leaf50_d16"]["val"], test["I2_RF_leaf50_d16"]),
    ("Final\n(LR, temporal, selected)", val[sel]["val"], test[sel]),
]
metrics_show = ["precision", "recall", "f1", "tss"]
labels_show = ["Precision", "Recall", "F1-score", "TSS"]
x = np.arange(len(metrics_show)); w = 0.25
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for i, (name, vres, tres) in enumerate(rows):
    vals = [tres[m] for m in metrics_show]
    ax.bar(x + (i - 1) * w, vals, w, label=name.replace("\n", " "),
           color=[NAVY, GREY, GREEN][i])
ax.set_xticks(x); ax.set_xticklabels(labels_show)
ax.set_ylim(0, 1.05); ax.set_ylabel("Score (test partition P5)")
ax.set_title("Model Performance Across Iterations (Test Partition)")
ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("figures/fig6_2_iteration_comparison.png"); plt.close()

# ---------- Figure 6.3 Precision-Recall curve (final model, test) ----------
prec, rec, _ = precision_recall_curve(y, p)
fig, ax = plt.subplots(figsize=(5.4, 4.6))
ax.plot(rec, prec, color=NAVY, lw=2)
ax.axhline(y.mean(), color=GREY, ls="--", lw=1, label=f"No-skill baseline (P={y.mean():.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title(f"Precision-Recall Curve — Final Model (Test)\nPR-AUC = {test[sel]['pr_auc']:.3f}")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("figures/fig6_3_pr_curve.png"); plt.close()

# ---------- Figure 6.4 ROC curve ----------
fpr, tpr, _ = roc_curve(y, p)
fig, ax = plt.subplots(figsize=(5.4, 4.6))
ax.plot(fpr, tpr, color=NAVY, lw=2, label=f"Final model (AUC={test[sel]['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="Chance")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Final Model (Test Partition)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("figures/fig6_4_roc_curve.png"); plt.close()

# ---------- Figure 6.5 Logistic Regression coefficient visualization (final model) ----------
bundle = joblib.load(f"models/{sel}.joblib")
model, cols = bundle["model"], bundle["cols"]
coef = model.named_steps["clf"].coef_.ravel()
order = np.argsort(np.abs(coef))[::-1][:15]
fig, ax = plt.subplots(figsize=(7.5, 5.5))
names = [cols[i] for i in order]
vals = coef[order]
colors = [RED if v > 0 else NAVY for v in vals]
ax.barh(range(len(order)), vals[::-1], color=[c for c in colors[::-1]])
ax.set_yticks(range(len(order))); ax.set_yticklabels(names[::-1], fontsize=8)
ax.set_xlabel("Standardized coefficient (log-odds of M/X)")
ax.set_title("Top 15 Logistic Regression Coefficients — Final Model")
ax.axvline(0, color="black", lw=0.8)
plt.tight_layout(); plt.savefig("figures/fig6_5_lr_coefficients.png"); plt.close()

print("selected model:", sel)
print("top coefficients:", [cols[i] for i in order[:8]])
print("done")
