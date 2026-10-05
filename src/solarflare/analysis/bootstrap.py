"""Active-region block bootstrap on the locked test partition.

Why blocks and not rows: consecutive windows of one HARP region are one hour
apart 98.60 % of the time, so adjacent 12-hour windows share 11 of their 12
hours of observations. Resampling individual windows would treat those
near-duplicates as independent draws and produce confidence intervals that are
far too narrow. Resampling **whole active regions** keeps each region's windows
together, which is the unit that is plausibly independent here.

The frozen threshold is applied throughout; nothing is re-tuned. This is a
statement about how precisely the P5 score is measured, not a new result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import models
from . import _common

__all__ = ["block_bootstrap", "run"]

#: The metrics that get a confidence interval.
METRICS = (
    "recall",
    "precision",
    "f1",
    "tss",
    "pr_auc",
    "roc_auc",
    "fpr",
    "far",
    "alert_rate",
    "hss2",
)


def block_bootstrap(
    frame: pd.DataFrame, threshold: float, resamples: int = 1000, seed: int = _common.RESAMPLE_SEED
) -> pd.DataFrame:
    """Resample HARP regions with replacement and recompute the metrics.

    Each resample draws ``n_regions`` regions with replacement from the regions
    present in ``frame`` and concatenates all of their windows, so the resampled
    set has a varying size and a varying number of positives — which is the
    point: a partition containing a few very active regions is genuinely less
    informative than the raw window count suggests.

    Resamples in which the positive or negative class is absent are skipped and
    counted, since precision, recall and the AUCs are undefined there.

    Args:
        frame: P5 windows with ``harp``, ``y`` and ``p`` columns.
        threshold: the frozen alert threshold.
        resamples: number of bootstrap resamples.
        seed: RNG seed.

    Returns:
        One row per usable resample, with a column per metric.
    """
    from ..metrics import extended

    rng = np.random.default_rng(seed)
    y = frame["y"].to_numpy()
    p = frame["p"].to_numpy()

    regions = frame["harp"].to_numpy()
    order = np.argsort(regions, kind="stable")
    sorted_regions = regions[order]
    unique, starts = np.unique(sorted_regions, return_index=True)
    blocks = np.split(order, starts[1:])
    n_regions = len(unique)

    rows, skipped = [], 0
    for _ in range(resamples):
        picks = rng.integers(0, n_regions, size=n_regions)
        index = np.concatenate([blocks[i] for i in picks])
        y_b, p_b = y[index], p[index]
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            skipped += 1
            continue
        row = extended(y_b, p_b, threshold)
        row["n_windows"] = len(index)
        rows.append({k: row[k] for k in (*METRICS, "n_windows", "positives")})

    out = pd.DataFrame(rows)
    out.attrs["skipped"] = skipped
    out.attrs["n_regions"] = n_regions
    return out


def run(resamples: int = 1000) -> dict:
    """Compute 95 % block-bootstrap intervals and write them to results/extra/.

    Returns the payload that was written.
    """
    from ..metrics import extended

    frame = _common.load_test_frame()
    alert_threshold, _ = models.thresholds()
    point = extended(frame["y"].to_numpy(), frame["p"].to_numpy(), alert_threshold)

    draws = block_bootstrap(frame, alert_threshold, resamples=resamples)
    intervals = {}
    for metric in METRICS:
        values = draws[metric].to_numpy()
        intervals[metric] = {
            "point_estimate": float(point[metric]),
            "ci95_low": float(np.percentile(values, 2.5)),
            "ci95_high": float(np.percentile(values, 97.5)),
            "bootstrap_mean": float(values.mean()),
            "bootstrap_sd": float(values.std(ddof=1)),
        }

    payload = {
        "analysis": "active-region block bootstrap on the locked test partition P5",
        "method": (
            f"{resamples} resamples, seed {_common.RESAMPLE_SEED}. Each resample draws "
            f"{draws.attrs['n_regions']} HARP regions with replacement from the "
            f"{draws.attrs['n_regions']} regions in P5 and keeps all of each region's "
            f"windows. Whole regions are the resampling unit because consecutive windows "
            f"of one region overlap by 11 of their 12 hours, so windows are not "
            f"independent. The frozen alert threshold is applied unchanged; nothing is "
            f"re-tuned on P5."
        ),
        "threshold": alert_threshold,
        "n_resamples_requested": resamples,
        "n_resamples_used": len(draws),
        "n_resamples_skipped_degenerate": int(draws.attrs["skipped"]),
        "n_regions": int(draws.attrs["n_regions"]),
        "resampled_window_count": {
            "min": int(draws["n_windows"].min()),
            "median": int(draws["n_windows"].median()),
            "max": int(draws["n_windows"].max()),
        },
        "resampled_positive_count": {
            "min": int(draws["positives"].min()),
            "median": int(draws["positives"].median()),
            "max": int(draws["positives"].max()),
        },
        "intervals": intervals,
    }
    _common.write("bootstrap", payload)
    return payload
