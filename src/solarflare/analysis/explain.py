"""Exact per-window explanations for the selected linear model.

For logistic regression the log-odds decompose additively with no residual::

    logit(p) = intercept + Σ_j  coef_j · standardised_x_j

so "feature *j* contributed +0.42 to the log-odds of this forecast" is an exact
statement, not an attribution heuristic. There is no sampling, no surrogate
model and nothing to tune — which is a real advantage of having selected a
linear model, and worth saying out loud next to the explainability view.

This module records the identity holding across a large sample of P5 windows.
``tests/test_contributions.py`` asserts it as a unit test.
"""

from __future__ import annotations

import numpy as np

from .. import data, models
from . import _common

__all__ = ["run", "top_contributions"]

SAMPLE = 2000


def top_contributions(bundle: models.Bundle, row, k: int = 8) -> list[dict]:
    """Return the ``k`` largest-magnitude contributions for one window.

    Args:
        bundle: the selected logistic-regression bundle.
        row: a one-row frame of raw features.
        k: how many contributions to return.

    Returns:
        One dict per feature with its raw value, standardised value, coefficient
        and signed contribution in log-odds units, largest magnitude first.
    """
    contrib, _, _ = models.contributions(bundle, row)
    z = models.standardised(bundle, row)
    coef = bundle.model.named_steps["clf"].coef_.ravel()
    values = row[bundle.cols].to_numpy(dtype=float)[0]
    order = np.argsort(np.abs(contrib[0]))[::-1][:k]
    return [
        {
            "feature": bundle.cols[i],
            "raw_value": None if not np.isfinite(values[i]) else float(values[i]),
            "imputed": not bool(np.isfinite(values[i])),
            "standardised_value": float(z[0, i]),
            "coefficient": float(coef[i]),
            "contribution_logodds": float(contrib[0, i]),
        }
        for i in order
    ]


def run() -> dict:
    """Verify the additive identity on a P5 sample and write the evidence out."""
    X, meta = data.load()
    _, _, test = data.split_masks(meta)
    bundle = models.load_bundle(models.selected_key())

    rng = np.random.default_rng(_common.RESAMPLE_SEED)
    test_index = np.where(test)[0]
    sample = rng.choice(test_index, size=min(SAMPLE, len(test_index)), replace=False)
    subset = X.iloc[sample]

    contrib, logit, intercept = models.contributions(bundle, subset)
    reconstructed = 1.0 / (1.0 + np.exp(-logit))
    actual = bundle.predict_proba(subset)

    logit_actual = np.log(actual / (1 - actual))
    max_logit_error = float(np.max(np.abs(logit - logit_actual)))
    max_probability_error = float(np.max(np.abs(reconstructed - actual)))

    payload = {
        "analysis": "exactness of the per-window linear explanation",
        "identity": "logit(p) = intercept + sum_j coef_j * standardised_x_j",
        "why_it_is_exact": (
            "The pipeline is a fixed signed-log transform, then median imputation with "
            "medians fitted on P1-P3, then standardisation with a centre and scale "
            "fitted on P1-P3, then a linear model. Every step before the classifier is "
            "an affine or fixed elementwise map, so the classifier's input is known "
            "exactly and its logit is a plain dot product. Nothing is approximated, "
            "unlike permutation importance or a surrogate explainer."
        ),
        "n_windows_checked": len(sample),
        "seed": _common.RESAMPLE_SEED,
        "intercept": float(intercept),
        "max_abs_logit_error": max_logit_error,
        "max_abs_probability_error": max_probability_error,
        "n_features": int(contrib.shape[1]),
        "mean_abs_contribution_by_rank": [
            float(v) for v in np.sort(np.abs(contrib), axis=1)[:, ::-1].mean(axis=0)[:12]
        ],
        "note": "The two error figures above are floating-point round-off, not "
        "approximation error. tests/test_contributions.py asserts the same "
        "identity to 1e-9.",
    }
    _common.write("explanation_identity", payload)
    return payload
