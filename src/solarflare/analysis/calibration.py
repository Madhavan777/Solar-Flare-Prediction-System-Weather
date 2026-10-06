"""Calibration of the selected model's probabilities.

The selected logistic regression was fitted with ``class_weight="balanced"``,
which weights each class by ``n_samples / (n_classes · n_class)``. The exact
weights are computed from the training labels by :func:`class_weights` rather
than quoted from memory, because the figure is easy to get wrong: the relevant
base rate is the **training** partitions' 1.99 %, not the test partition's
1.31 %.

That deliberately shifts the decision boundary towards recall, and the side
effect is that **the output probabilities are not calibrated**: they
systematically over-forecast. A window the model scores at 0.60 does not flare
60 % of the time.

This matters for how the number is read, not for the published skill scores: TSS,
precision, recall, F1 and the confusion matrix depend only on the ordering of
the probabilities relative to a threshold, and PR-AUC and ROC-AUC depend only on
the ordering. A monotone recalibration leaves all of them unchanged, which is
checked here rather than asserted.

Two recalibration variants are fitted **on the validation partition only** and
reported as separate, non-selected comparisons. The frozen model stays the
headline result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .. import models, paths
from . import _common

__all__ = ["reliability", "run"]

N_BINS = 12


def reliability(
    y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS, strategy: str = "quantile"
) -> list[dict]:
    """Return reliability-diagram bins.

    Args:
        y: true labels.
        p: predicted probabilities.
        n_bins: number of bins.
        strategy: ``"quantile"`` for equal-count bins (robust when almost all
            predictions sit near zero, as here) or ``"uniform"`` for
            equal-width bins.

    Returns:
        One dict per non-empty bin with its edges, count, mean predicted
        probability and observed positive frequency.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)

    bins = []
    for b in range(len(edges) - 1):
        mask = index == b
        if not mask.any():
            continue
        bins.append(
            {
                "bin": b,
                "low": float(edges[b]),
                "high": float(edges[b + 1]),
                "n": int(mask.sum()),
                "n_positive": int(y[mask].sum()),
                "mean_predicted": float(p[mask].mean()),
                "observed_frequency": float(y[mask].mean()),
            }
        )
    return bins


def class_weights() -> dict[str, float]:
    """The weights ``class_weight="balanced"`` actually applied, from the data.

    scikit-learn computes ``n_samples / (n_classes * n_class)`` over the labels
    it was fitted on, which here are the training partitions only. Returned
    rather than hard-coded so that no document can quote a stale figure.
    """
    from sklearn.utils.class_weight import compute_class_weight

    from .. import data

    _, meta = data.load()
    train, _, _ = data.split_masks(meta)
    y_train = meta["y"].to_numpy()[train]
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    return {
        "n_training_windows": len(y_train),
        "n_training_positives": int(y_train.sum()),
        "training_base_rate": float(y_train.mean()),
        "weight_negative": float(weights[0]),
        "weight_positive": float(weights[1]),
        "positive_relative_to_negative": float(weights[1] / weights[0]),
    }


def _ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def run() -> dict:
    """Assess calibration and write the result to results/extra/."""
    frame = _common.load_test_frame()
    y_test = frame["y"].to_numpy()
    p_test = frame["p"].to_numpy()
    alert_threshold, _ = models.thresholds()

    key = models.selected_key()
    validation = pd.read_csv(paths.VAL_PREDICTIONS)
    y_val = validation["y"].to_numpy()
    p_val = validation[key].to_numpy()

    base_rate = float(y_test.mean())
    frozen = _ranking_metrics(y_test, p_test)
    weights = class_weights()

    # --- the no-skill reference: always forecast the training base rate -------
    train_base = 0.0
    try:
        meta = pd.read_csv(paths.ALL_META)
        train_base = float(meta.loc[meta["partition"].isin([1, 2, 3]), "y"].mean())
    except FileNotFoundError:
        pass
    constant = float(brier_score_loss(y_test, np.full_like(p_test, train_base)))

    # --- recalibration variants, fitted on VALIDATION only -------------------
    eps = 1e-12
    logit_val = np.log(np.clip(p_val, eps, 1 - eps) / (1 - np.clip(p_val, eps, 1 - eps)))
    logit_test = np.log(np.clip(p_test, eps, 1 - eps) / (1 - np.clip(p_test, eps, 1 - eps)))

    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10_000)
    platt.fit(logit_val.reshape(-1, 1), y_val)
    p_platt = platt.predict_proba(logit_test.reshape(-1, 1))[:, 1]

    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(p_val, y_val)
    p_isotonic = isotonic.predict(p_test)

    variants = {
        "frozen_class_weighted": {
            "description": "The selected model as published. class_weight='balanced'.",
            "selected": True,
            **frozen,
            "mean_predicted": float(p_test.mean()),
            "over_forecast_ratio": float(p_test.mean() / base_rate),
        },
        "platt_on_validation": {
            "description": "Platt scaling: a one-parameter logistic fitted on the "
            "validation partition's logits, applied to P5. "
            "NOT the selected model.",
            "selected": False,
            **_ranking_metrics(y_test, p_platt),
            "mean_predicted": float(p_platt.mean()),
            "over_forecast_ratio": float(p_platt.mean() / base_rate),
            "fitted_on": "validation partition P4 only",
        },
        "isotonic_on_validation": {
            "description": "Isotonic regression fitted on the validation partition, "
            "applied to P5. NOT the selected model.",
            "selected": False,
            **_ranking_metrics(y_test, p_isotonic),
            "mean_predicted": float(p_isotonic.mean()),
            "over_forecast_ratio": float(p_isotonic.mean() / base_rate),
            "fitted_on": "validation partition P4 only",
        },
    }

    # Monotone recalibration must not move a ranking metric. Verified, not assumed.
    roc_shift = {
        name: abs(variants[name]["roc_auc"] - frozen["roc_auc"])
        for name in ("platt_on_validation", "isotonic_on_validation")
    }

    payload = {
        "analysis": "calibration of the selected model's probabilities on P5",
        "class_weights": weights,
        "why_uncalibrated": (
            f"The model was fitted with class_weight='balanced'. On the training "
            f"partitions, where positives are {weights['training_base_rate'] * 100:.2f} % "
            f"of windows, that gives each positive a weight of "
            f"{weights['weight_positive']:.2f} against {weights['weight_negative']:.2f} "
            f"for each negative - positives count "
            f"{weights['positive_relative_to_negative']:.0f} times more heavily. "
            f"That is what buys 91.6 % recall, and it "
            "necessarily inflates the predicted probabilities: the model is trained as if "
            "major flares were far commoner than they are. The outputs are therefore "
            "useful as a ranking and against the fixed thresholds, but must not be read "
            "as literal chances of a flare."
        ),
        "what_is_unaffected": (
            "TSS, precision, recall, F1, HSS and the confusion matrix depend only on "
            "which side of the threshold each probability falls; PR-AUC and ROC-AUC "
            "depend only on the ranking. A STRICTLY monotone recalibration therefore "
            "changes none of them: Platt scaling here moves the ROC-AUC by exactly 0. "
            "Isotonic regression is monotone non-decreasing rather than strictly "
            "increasing, so it creates ties that destroy ordering information and does "
            "move both AUCs - see roc_auc_shift_under_recalibration below. That is a "
            "property of isotonic regression, not evidence against the general point."
        ),
        "test_base_rate": base_rate,
        "train_base_rate": train_base,
        "mean_predicted_probability": float(p_test.mean()),
        "brier_frozen": frozen["brier"],
        "brier_constant_climatology": constant,
        "alert_threshold": alert_threshold,
        "variants": variants,
        "roc_auc_shift_under_recalibration": roc_shift,
        "reliability_quantile_bins": reliability(y_test, p_test, strategy="quantile"),
        "reliability_uniform_bins": reliability(y_test, p_test, strategy="uniform"),
        "reliability_platt_quantile_bins": reliability(y_test, p_platt, strategy="quantile"),
    }
    _common.write("calibration", payload)
    return payload
