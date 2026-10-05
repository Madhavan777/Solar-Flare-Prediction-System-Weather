"""Operating-point table and threshold sweep on the locked test partition.

The two published thresholds were fixed on the validation partition before P5
was scored. The sweep below shows what *would* have happened at other
thresholds. It is shown for understanding the trade-off, and it is explicitly
**not** a menu to pick from after the fact: choosing a threshold using this
table would be selecting on the test set, which is exactly what the protocol
forbids.
"""

from __future__ import annotations

import numpy as np

from .. import models
from ..metrics import extended
from . import _common

__all__ = ["run", "sweep"]

#: Number of points on the threshold sweep used by the dashboard slider.
SWEEP_POINTS = 400

_FIELDS = (
    "threshold",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "f1",
    "tss",
    "hss2",
    "far",
    "fpr",
    "accuracy",
    "alert_rate",
    "alert_rate_ratio",
)


def sweep(
    y: np.ndarray, p: np.ndarray, points: int = SWEEP_POINTS, include: tuple[float, ...] = ()
) -> list[dict]:
    """Evaluate the frozen predictions across a grid of thresholds.

    The grid is the unique quantiles of ``p`` from the 50th to the 99.99th
    percentile, which concentrates the points where the decision actually
    changes: on P5 over 90 % of probabilities sit below 0.1.

    Args:
        y: true labels.
        p: predicted probabilities.
        points: size of the quantile grid.
        include: thresholds to add to the grid exactly. The two published
            thresholds are passed here so that the sweep contains the real
            published operating point rather than a nearby quantile — otherwise
            the slider would report a confusion matrix that disagrees with
            ``results/test_results.json``.

    Returns:
        One dict per threshold, sorted ascending, with the full confusion matrix
        and metric set.
    """
    grid = np.quantile(p, np.linspace(0.5, 0.9999, points))
    if include:
        grid = np.concatenate([grid, np.asarray(include, dtype=float)])
    grid = np.unique(grid)
    rows = []
    for threshold in grid:
        computed = extended(y, p, float(threshold))
        rows.append({field: computed[field] for field in _FIELDS})
    return rows


def run() -> dict:
    """Build the operating-point table and sweep, and write them out."""
    frame = _common.load_test_frame()
    y = frame["y"].to_numpy()
    p = frame["p"].to_numpy()
    alert_threshold, high_threshold = models.thresholds()

    named = {
        "alert_threshold_validation_tss": {
            "threshold": alert_threshold,
            "chosen_on": "validation partition P4, maximising TSS",
            "is_published_operating_point": True,
            **{k: v for k, v in extended(y, p, alert_threshold).items() if k in _FIELDS},
        },
        "high_risk_threshold_validation_f1": {
            "threshold": high_threshold,
            "chosen_on": "validation partition P4, maximising F1",
            "is_published_operating_point": True,
            **{k: v for k, v in extended(y, p, high_threshold).items() if k in _FIELDS},
        },
        "naive_half": {
            "threshold": 0.5,
            "chosen_on": "nothing - the default 0.5 cut, shown for contrast",
            "is_published_operating_point": False,
            **{k: v for k, v in extended(y, p, 0.5).items() if k in _FIELDS},
        },
    }

    grid = sweep(y, p)
    best_on_test = max(grid, key=lambda r: r["tss"])

    payload = {
        "analysis": "operating points and threshold sweep on P5",
        "warning": (
            "Exploratory. The headline results use the validation-fixed thresholds "
            "below. The best-TSS point on this test sweep is reported only to show how "
            "little was lost by fixing the threshold on validation instead - it was "
            "never available to the selection procedure."
        ),
        "risk_bands": {
            "LOW": f"p < {alert_threshold:.4f}",
            "MODERATE": f"{alert_threshold:.4f} <= p < {high_threshold:.4f}",
            "HIGH": f"p >= {high_threshold:.4f}",
        },
        "named_operating_points": named,
        "best_tss_on_test_posthoc": {
            **best_on_test,
            "note": "Post-hoc only. Not used, not published as the operating point.",
            "tss_cost_of_fixing_threshold_on_validation": float(
                best_on_test["tss"] - named["alert_threshold_validation_tss"]["tss"]
            ),
        },
        "sweep_points": len(grid),
        "sweep": grid,
    }
    _common.write("operating_points", payload)
    return payload
