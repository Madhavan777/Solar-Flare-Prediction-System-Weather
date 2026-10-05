"""Loading the frozen pipelines and explaining their predictions exactly.

The seven candidate bundles live in ``models/`` as joblib dicts
``{"model": sklearn.pipeline.Pipeline, "cols": [feature names]}``. They are read
only; nothing in this module writes to ``models/``.

The explanation path here is **exact**, not an approximation: for the selected
logistic-regression pipeline the per-feature contributions plus the intercept
reproduce the model's own logit to floating-point precision, and therefore the
probability. :func:`contributions` is what the dashboard's bars are built from,
and ``tests/test_contributions.py`` asserts the identity.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import paths

if str(paths.SRC) not in sys.path:
    sys.path.insert(0, str(paths.SRC))
from common import ALL_COLS, LAST_COLS, slog

__all__ = [
    "CANDIDATE_KEYS",
    "Bundle",
    "contributions",
    "export_lr_parameters",
    "load_bundle",
    "load_selection",
    "risk_level",
    "selected_key",
    "standardised",
    "thresholds",
]

#: The seven candidates, in the order ``src/train_eval.py`` defines them.
CANDIDATE_KEYS: tuple[str, ...] = (
    "I1_LR_last",
    "I2_LR_temporal_C0.01",
    "I2_LR_temporal_C0.1",
    "I2_LR_temporal_C1",
    "I2_RF_leaf5",
    "I2_RF_leaf20",
    "I2_RF_leaf50_d16",
)

#: Human-readable labels, used in figures and the dashboard.
CANDIDATE_LABELS: dict[str, str] = {
    "I1_LR_last": "LR, point-in-time (24 last values), C=1.0",
    "I2_LR_temporal_C0.01": "LR, temporal (144 features), C=0.01",
    "I2_LR_temporal_C0.1": "LR, temporal (144 features), C=0.1",
    "I2_LR_temporal_C1": "LR, temporal (144 features), C=1.0",
    "I2_RF_leaf5": "RF, 300 trees, min_samples_leaf=5",
    "I2_RF_leaf20": "RF, 300 trees, min_samples_leaf=20",
    "I2_RF_leaf50_d16": "RF, 300 trees, min_samples_leaf=50, max_depth=16",
}


@dataclass(frozen=True)
class Bundle:
    """A loaded candidate: its key, pipeline and feature column list."""

    key: str
    model: Pipeline
    cols: list[str]

    @property
    def is_logistic(self) -> bool:
        """True if the final estimator is a logistic regression."""
        return type(self.model.named_steps["clf"]).__name__ == "LogisticRegression"

    @property
    def uses_slog(self) -> bool:
        """True if the pipeline applies the fixed signed-log transform first."""
        return "slog" in self.model.named_steps

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Positive-class probabilities for ``X``, columns selected and ordered."""
        return self.model.predict_proba(X[self.cols])[:, 1]


def load_bundle(key: str) -> Bundle:
    """Load one candidate by key.

    Args:
        key: one of :data:`CANDIDATE_KEYS`.

    Raises:
        FileNotFoundError: if the ``.joblib`` is missing.
        ValueError: if the stored feature list does not match ``common``.
    """
    path = paths.MODELS / f"{key}.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"{paths.relative(path)} not found")
    raw = joblib.load(path)
    cols = list(raw["cols"])
    expected = LAST_COLS if key == "I1_LR_last" else ALL_COLS
    if cols != expected:
        raise ValueError(
            f"{key}: stored feature list differs from common."
            f"{'LAST_COLS' if key == 'I1_LR_last' else 'ALL_COLS'}"
        )
    return Bundle(key=key, model=raw["model"], cols=cols)


def load_selection() -> dict[str, Any]:
    """Return the frozen ``results/selection.json`` as a dict."""
    return json.loads(paths.SELECTION_JSON.read_text(encoding="utf-8"))


def selected_key() -> str:
    """Return the key of the selected final model (``I2_LR_temporal_C0.01``)."""
    return str(load_selection()["selected"])


def thresholds() -> tuple[float, float]:
    """Return ``(alert_threshold, high_threshold)``, both fixed on validation.

    The alert threshold maximises validation TSS; the high-risk threshold
    maximises validation F1. Neither was ever tuned on the test partition.
    """
    selection = load_selection()
    return float(selection["alert_threshold"]), float(selection["high_threshold"])


def risk_level(p: float, alert: float, high: float) -> str:
    """Map a probability to ``"LOW"``, ``"MODERATE"`` or ``"HIGH"``.

    ``LOW`` for ``p < alert``, ``MODERATE`` for ``alert <= p < high``,
    ``HIGH`` for ``p >= high``.
    """
    if p >= high:
        return "HIGH"
    if p >= alert:
        return "MODERATE"
    return "LOW"


def standardised(bundle: Bundle, X: pd.DataFrame) -> np.ndarray:
    """Return the feature matrix as the final estimator actually sees it.

    Applies, in order, exactly the fitted steps that precede the classifier:
    the fixed signed-log transform (if present), median imputation using the
    imputer's learned ``statistics_``, and standardisation using the scaler's
    learned ``mean_`` and ``scale_``. All three were fitted on P1–P3 only.

    Args:
        bundle: a loaded candidate.
        X: raw feature frame; the bundle's own columns are selected in order.

    Returns:
        A float64 array of shape ``(len(X), len(bundle.cols))``.
    """
    values = X[bundle.cols].to_numpy(dtype=np.float64)
    values = np.where(np.isfinite(values), values, np.nan)
    steps = bundle.model.named_steps
    if "slog" in steps:
        values = slog(values)
    if "impute" in steps:
        stats = steps["impute"].statistics_
        values = np.where(np.isnan(values), stats[None, :], values)
    if "scale" in steps:
        scaler = steps["scale"]
        values = (values - scaler.mean_[None, :]) / scaler.scale_[None, :]
    return values


def contributions(bundle: Bundle, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact per-feature contributions to the log-odds, for a linear model.

    For logistic regression the logit decomposes with no residual::

        logit(p) = intercept + sum_j coef_j * standardised_x_j

    so contribution *j* is ``coef_j * standardised_x_j`` — a signed number in
    log-odds units that says how far feature *j* pushed this particular window
    away from the intercept. Summing the contributions and adding the intercept
    returns the model's own logit exactly (to ~1e-12), which is asserted in the
    test suite.

    Args:
        bundle: a loaded candidate whose final estimator is a
            ``LogisticRegression``.
        X: raw feature frame.

    Returns:
        ``(contrib, logit, intercept)`` where ``contrib`` has shape
        ``(len(X), n_features)``, ``logit`` has shape ``(len(X),)`` and
        ``intercept`` is a scalar array.

    Raises:
        TypeError: if the final estimator is not linear.
    """
    if not bundle.is_logistic:
        raise TypeError(
            f"{bundle.key} is a {type(bundle.model.named_steps['clf']).__name__}; "
            f"exact additive contributions are only defined for the linear model."
        )
    clf = bundle.model.named_steps["clf"]
    coef = clf.coef_.ravel()
    intercept = clf.intercept_.ravel()[0]
    z = standardised(bundle, X)
    contrib = z * coef[None, :]
    logit = contrib.sum(axis=1) + intercept
    return contrib, logit, np.asarray(intercept)


def export_lr_parameters(bundle: Bundle | None = None) -> dict[str, Any]:
    """Export every fitted parameter needed to re-run the model in JavaScript.

    The browser-side "Live inference" view reimplements the pipeline from this
    dictionary. Exporting the real fitted arrays — rather than hard-coding
    anything — is what makes the parity test meaningful: the JavaScript path and
    scikit-learn must agree to better than 1e-6 on P5 rows.

    Returns:
        A JSON-serialisable dict with the feature and SHARP parameter lists, the
        statistics order, the ``slog`` flag, the imputer medians, the scaler
        centre and scale, the coefficients, the intercept, and both thresholds.
    """
    bundle = bundle or load_bundle(selected_key())
    if not bundle.is_logistic:
        raise TypeError(f"{bundle.key} is not a logistic-regression pipeline")
    steps = bundle.model.named_steps
    clf = steps["clf"]
    alert, high = thresholds()
    from common import SHARP, STATS  # local import: keeps the module list in one place

    return {
        "model_key": bundle.key,
        "model_name": "Logistic Regression (temporal features, C=0.01)",
        "note": "Fitted parameters exported verbatim from models/"
        f"{bundle.key}.joblib. Preprocessing was fitted on P1-P3 only.",
        "sharp_parameters": list(SHARP),
        "statistics": list(STATS),
        "cols": list(bundle.cols),
        "apply_slog": bool(bundle.uses_slog),
        "slog": "sign(x) * log10(1 + abs(x))",
        "impute_median": [float(v) for v in steps["impute"].statistics_],
        "scale_mean": [float(v) for v in steps["scale"].mean_],
        "scale_std": [float(v) for v in steps["scale"].scale_],
        "coef": [float(v) for v in clf.coef_.ravel()],
        "intercept": float(clf.intercept_.ravel()[0]),
        "alert_threshold": alert,
        "high_threshold": high,
        "cadence_hours": 0.2,
        "expected_rows": 60,
    }
