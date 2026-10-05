"""Trivial baselines, so the model's skill can be read against something.

A headline TSS of 0.853 means nothing on its own. Two reference points make it
interpretable:

**Climatology.** Always forecast the training base rate. It is the honest
"no-skill" forecast: perfect calibration, zero discrimination. TSS 0, ROC-AUC
0.5, PR-AUC equal to the base rate — and it detects no flares at all while
scoring 98.69 % accuracy, which is why accuracy is not used as a headline here.

**Best single SHARP feature.** One feature, one threshold, both chosen on the
training and validation partitions only. If a single number did almost as well
as 144 features and a fitted model, the project would not be worth much; this
quantifies how much the full pipeline adds.

Both are selected without ever looking at P5, exactly like the real model, and
then scored on P5 once.
"""

from __future__ import annotations

import numpy as np

from .. import data, models
from ..metrics import extended
from . import _common

__all__ = ["best_cut", "run"]


def best_cut(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Find the threshold maximising TSS for a one-dimensional score.

    Exact and O(n log n): the scores are sorted descending, running hit and
    false-alarm counts are accumulated, and TSS is read off at every tie-group
    boundary, which is the complete set of distinct decision rules of the form
    ``alert if score >= t``.

    Args:
        scores: a finite score per window; higher means more flare-like.
        y: true binary labels.

    Returns:
        ``(best_tss, threshold)``. The rule is ``alert if score >= threshold``.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y)
    positives = float(y.sum())
    negatives = float(len(y) - y.sum())
    if positives == 0 or negatives == 0:
        return float("nan"), float("nan")

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_y = y[order]
    hits = np.cumsum(sorted_y)
    false_alarms = np.cumsum(1 - sorted_y)
    # Only cuts at the end of a run of equal scores are realisable.
    boundary = np.empty(len(sorted_scores), dtype=bool)
    boundary[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    boundary[-1] = True

    tss = hits / positives - false_alarms / negatives
    tss_at_boundary = np.where(boundary, tss, -np.inf)
    k = int(np.argmax(tss_at_boundary))
    return float(tss_at_boundary[k]), float(sorted_scores[k])


def run() -> dict:
    """Fit and score both baselines, and write the comparison out."""
    X, meta = data.load()
    train, validation, test = data.split_masks(meta)
    y = meta["y"].to_numpy()
    threshold, _ = models.thresholds()

    frozen_frame = _common.load_test_frame()
    frozen = extended(frozen_frame["y"].to_numpy(), frozen_frame["p"].to_numpy(), threshold)

    train_base = float(y[train].mean())
    test_base = float(y[test].mean())

    # --- baseline 1: climatology --------------------------------------------
    constant = np.full(int(test.sum()), train_base)
    climatology = extended(y[test], constant, threshold)
    climatology.update(
        description=f"Always forecast the training base rate, p = {train_base:.6f}, for "
        f"every window. Never exceeds the alert threshold, so it raises no "
        f"alerts and detects nothing.",
        fitted_on="training partitions P1-P3 (the base rate only)",
    )

    # --- baseline 2: best single feature ------------------------------------
    medians = X.loc[train].median()
    filled = X.fillna(medians)

    candidates = []
    for column in X.columns:
        values = filled[column].to_numpy()
        for direction in (1.0, -1.0):
            tss, cut = best_cut(direction * values[validation], y[validation])
            if np.isfinite(tss):
                candidates.append((tss, column, direction, cut))
    candidates.sort(reverse=True)
    best_tss_val, best_column, best_direction, best_cut_value = candidates[0]

    test_scores = best_direction * filled[best_column].to_numpy()[test]
    single = extended(y[test], test_scores, best_cut_value)
    rule = (
        f"alert if {best_column} "
        f"{'>=' if best_direction > 0 else '<='} "
        f"{best_direction * best_cut_value:.6g}"
    )
    single.update(
        description=f"One feature, one threshold. Rule: {rule}.",
        feature=best_column,
        direction="higher is more flare-like" if best_direction > 0 else "lower is more flare-like",
        rule=rule,
        validation_tss=float(best_tss_val),
        fitted_on="median imputation from P1-P3; feature and threshold chosen on P4",
        caveat="pr_auc and roc_auc here are computed on a raw feature value, not a "
        "probability, so only their ranking interpretation is meaningful.",
    )

    runners_up = [
        {
            "feature": column,
            "direction": "higher" if direction > 0 else "lower",
            "validation_tss": float(tss),
        }
        for tss, column, direction, _ in candidates[:8]
    ]

    payload = {
        "analysis": "trivial baselines, for context",
        "selection_discipline": (
            "Both baselines were chosen using the training and validation partitions "
            "only - imputation medians from P1-P3, feature and threshold from P4 - and "
            "then scored once on P5, the same discipline the real model followed."
        ),
        "test_base_rate": test_base,
        "train_base_rate": train_base,
        "baselines": {
            "climatology_constant": climatology,
            "best_single_feature": single,
        },
        "selected_model": {
            "key": models.selected_key(),
            **{
                k: frozen[k]
                for k in (
                    "tss",
                    "pr_auc",
                    "roc_auc",
                    "precision",
                    "recall",
                    "f1",
                    "fpr",
                    "far",
                    "alert_rate",
                    "accuracy",
                )
            },
        },
        "improvement_over_best_single_feature": {
            "tss": float(frozen["tss"] - single["tss"]),
            "pr_auc": float(frozen["pr_auc"] - single["pr_auc"]),
            "precision_at_similar_recall": (
                "compare precision and recall directly in the table above; the single "
                "feature and the model do not sit at the same recall"
            ),
        },
        "single_feature_runners_up_by_validation_tss": runners_up,
        "n_single_feature_rules_searched": len(candidates),
    }
    _common.write("baselines", payload)
    return payload
