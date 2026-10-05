"""Skill metrics for a rare-event binary forecast.

The function :func:`metrics` is a faithful reimplementation of
``src/train_eval.py:metrics`` and returns the identical keys and values; it is
what the published numbers in ``results/`` were computed with. :func:`extended`
adds the quantities the original did not store, under unambiguous names.

Naming, fixed once and used everywhere (docs/DECISIONS.md D-07):

``tss``
    True Skill Statistic, recall − FPR. Also called Hanssen–Kuipers
    discriminant or Peirce's skill score. 0 for a no-skill forecast, 1 for a
    perfect one, and insensitive to the class ratio.
``hss2``
    Heidke Skill Score in its HSS2 form, the expression used in
    ``train_eval.py``. Written "HSS (HSS2 form)" on first use in prose.
``far`` — false-alarm **ratio**
    FP / (TP + FP) = 1 − precision. The fraction of raised alerts that were
    wrong. **0.8384** for the selected model on P5.
``fpr`` — false-alarm **rate**
    FP / (FP + TN). The fraction of quiet windows that triggered an alert.
    **0.0633** for the selected model on P5.

The ratio and the rate are never both called "FAR", and never substituted for
one another: they differ by more than a factor of thirteen here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)

__all__ = ["GLOSSARY", "best_threshold", "counts", "extended", "metrics"]

GLOSSARY: dict[str, str] = {
    "precision": "TP/(TP+FP) — of the windows we alerted on, the share that really flared.",
    "recall": "TP/(TP+FN) — of the windows that really flared, the share we alerted on. "
    "Also the probability of detection (POD) or hit rate.",
    "f1": "Harmonic mean of precision and recall.",
    "tss": "Recall − FPR. True Skill Statistic (Hanssen–Kuipers). "
    "Unaffected by the base rate, which is why it is the headline metric here.",
    "hss2": "Heidke Skill Score, HSS2 form: 2(TP·TN − FN·FP) / "
    "[(TP+FN)(FN+TN) + (TP+FP)(FP+TN)]. Skill against a random forecast "
    "that preserves the marginal totals; it is depressed by low precision.",
    "far": "FP/(TP+FP) — false-alarm RATIO, i.e. 1 − precision. The share of "
    "alerts that were false.",
    "fpr": "FP/(FP+TN) — false-alarm RATE. The share of quiet windows that "
    "triggered an alert. Not the same thing as the ratio.",
    "accuracy": "(TP+TN)/N. Misleading here: always predicting 'no flare' scores "
    "98.69% on P5 and detects nothing.",
    "pr_auc": "Area under the precision-recall curve (average precision). The "
    "threshold-free summary that respects the 1.31% base rate.",
    "roc_auc": "Area under the ROC curve. Optimistic under heavy imbalance "
    "because TN dominates the denominator of FPR.",
    "alert_rate": "(TP+FP)/N — the share of all windows on which an alert is raised.",
    "alert_rate_ratio": "alert_rate divided by the base rate — how many times more "
    "often the system alerts than flares actually occur.",
    "base_rate": "(TP+FN)/N — the observed share of positive windows (climatology).",
    "brier": "Mean squared error of the predicted probabilities. Low values need "
    "calibrated probabilities, which these are not (see the model card). "
    "None when the scores are not probabilities.",
    "brier_undefined_reason": "Why the Brier score could not be computed, or None.",
}


def counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(tn, fp, fn, tp)`` for binary labels, with both classes forced."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, Any]:
    """Compute the frozen metric set at decision threshold ``thr``.

    Byte-for-byte equivalent to ``src/train_eval.py:metrics``: same keys, same
    order, same values. Verified against ``results/validation_results.json`` and
    ``results/test_results.json`` for all seven candidates to better than 1e-9.

    Args:
        y: true binary labels.
        p: predicted probability of the positive class.
        thr: decision threshold; an alert is raised when ``p >= thr``.
    """
    yhat = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = counts(y, yhat)
    rec = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    p_o = tp + fn
    hss2 = 2 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    return dict(
        threshold=float(thr),
        precision=float(precision_score(y, yhat, zero_division=0)),
        recall=float(rec),
        f1=float(f1_score(y, yhat, zero_division=0)),
        tss=float(rec - fpr),
        hss2=float(hss2),
        far=float(fp / (tp + fp) if tp + fp else 0.0),
        accuracy=float(accuracy_score(y, yhat)),
        pr_auc=float(average_precision_score(y, p)),
        roc_auc=float(roc_auc_score(y, p)),
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        tn=int(tn),
        n=len(y),
        positives=int(p_o),
    )


def extended(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, Any]:
    """:func:`metrics` plus the quantities the frozen set omits.

    Adds the false-alarm **rate** (distinct from the stored ``far`` ratio), the
    alert rate and its ratio to the base rate, the base rate itself, and the
    Brier score. None of these change any published number; they are reported
    alongside it.

    ``brier`` is only defined for genuine probabilities. When ``p`` is a decision
    score outside [0, 1] — as it is for the single-feature baseline, which
    thresholds a raw SHARP value — ``brier`` is ``None`` and
    ``brier_undefined_reason`` explains why. The ranking metrics (``pr_auc``,
    ``roc_auc``) remain meaningful for any monotone score.
    """
    out = metrics(y, p, thr)
    n, tp, fp, fn, tn = out["n"], out["tp"], out["fp"], out["fn"], out["tn"]
    base = (tp + fn) / n if n else 0.0
    alert_rate = (tp + fp) / n if n else 0.0
    out.update(
        fpr=float(fp / (fp + tn)) if fp + tn else 0.0,
        base_rate=float(base),
        alert_rate=float(alert_rate),
        alert_rate_ratio=float(alert_rate / base) if base else float("nan"),
    )

    values = np.asarray(p, dtype=float)
    if values.size and float(values.min()) >= 0.0 and float(values.max()) <= 1.0:
        out["brier"] = float(brier_score_loss(y, p))
        out["brier_undefined_reason"] = None
    else:
        out["brier"] = None
        out["brier_undefined_reason"] = (
            f"scores span [{values.min():.6g}, {values.max():.6g}], which is outside "
            f"[0, 1]; the Brier score is only defined for probabilities"
        )
    return out


def best_threshold(y: np.ndarray, p: np.ndarray, criterion: str = "tss") -> float:
    """Return the threshold maximising ``criterion`` on a 2,000-point grid.

    A faithful reimplementation of ``src/train_eval.py:best_threshold``: the grid
    is the unique values of the 0.5 → 0.9999 quantiles of ``p``, and ties are
    broken by taking the first (lowest) maximiser, because the comparison is a
    strict ``>``.

    This function must only ever be given **validation** predictions. Both
    published thresholds were fixed on P4 before P5 was scored; recomputing them
    on P5 would invalidate the protocol.

    Args:
        y: true binary labels.
        p: predicted probabilities.
        criterion: ``"tss"`` or ``"f1"``.
    """
    if criterion not in ("tss", "f1"):
        raise ValueError(f"criterion must be 'tss' or 'f1', got {criterion!r}")
    y = np.asarray(y)
    p = np.asarray(p)
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.9999, 2000)))
    best = (-1.0, 0.5)
    for t in grid:
        yhat = p >= t
        tp = int(np.sum(yhat & (y == 1)))
        fn = int(np.sum(~yhat & (y == 1)))
        fp = int(np.sum(yhat & (y == 0)))
        tn = int(np.sum(~yhat & (y == 0)))
        if criterion == "tss":
            s = tp / (tp + fn) - fp / (fp + tn)
        else:
            s = 2 * tp / (2 * tp + fp + fn)
        if s > best[0]:
            best = (s, float(t))
    return best[1]
