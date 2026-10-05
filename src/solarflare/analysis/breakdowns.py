"""Where the frozen model succeeds and fails on P5, broken down.

Four cuts, all at the frozen alert threshold:

1. **M-class vs X-class.** P5 contains 971 M windows and only **19 X windows**,
   so the X-class recall is estimated from 19 cases and its confidence interval
   is very wide. The count is printed next to every X figure.
2. **By calendar year.** P5 spans 2015-03 to 2018-08, across the declining phase
   of solar cycle 24, so the base rate falls sharply over the partition.
3. **By HARP region.** A handful of regions generate a large share of the false
   alarms — a magnetically complex region that never flares produces hundreds of
   consecutive alerted windows.
4. **By missingness.** Whether windows with more missing SHARP values are
   handled worse by the median imputer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import models
from ..metrics import extended
from . import _common

__all__ = ["run"]

_KEEP = (
    "n",
    "positives",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "f1",
    "tss",
    "fpr",
    "far",
    "alert_rate",
    "base_rate",
)


def _group_metrics(frame: pd.DataFrame, threshold: float) -> dict:
    """Metric set for one subgroup, or a reason it cannot be computed."""
    y = frame["y"].to_numpy()
    p = frame["p"].to_numpy()
    if len(frame) == 0:
        return {"n": 0, "note": "empty group"}
    if y.sum() == 0:
        alerts = int((p >= threshold).sum())
        return {
            "n": len(frame),
            "positives": 0,
            "fp": alerts,
            "tn": int(len(frame) - alerts),
            "alert_rate": float(alerts / len(frame)),
            "note": "no positives in this group: recall, precision and the AUCs are undefined",
        }
    if y.sum() == len(y):
        return {
            "n": len(frame),
            "positives": int(y.sum()),
            "note": "no negatives in this group: FPR, TSS and the AUCs are undefined",
        }
    out = extended(y, p, threshold)
    return {k: out[k] for k in _KEEP}


def run() -> dict:
    """Compute all four breakdowns and write them to results/extra/."""
    frame = _common.load_test_frame()
    threshold, _ = models.thresholds()
    y = frame["y"].to_numpy()
    positives = frame[frame["y"] == 1]

    # --- 1. by GOES class ----------------------------------------------------
    by_class = {}
    for letter in ("M", "X"):
        subset = positives[positives["goes_letter"] == letter]
        detected = int((subset["p"] >= threshold).sum())
        n = len(subset)
        recall = detected / n if n else float("nan")
        # Exact binomial (Wilson) interval, so the X-class uncertainty is visible.
        if n:
            z = 1.959963984540054
            centre = (recall + z * z / (2 * n)) / (1 + z * z / n)
            half = (z / (1 + z * z / n)) * np.sqrt(recall * (1 - recall) / n + z * z / (4 * n * n))
            low, high = max(0.0, centre - half), min(1.0, centre + half)
        else:
            low = high = float("nan")
        by_class[letter] = {
            "n_positive_windows": n,
            "detected": detected,
            "missed": n - detected,
            "recall": recall,
            "recall_wilson_ci95": [float(low), float(high)],
            "small_sample_warning": (
                (
                    f"only {n} {letter}-class windows in P5 - this recall is estimated from "
                    f"{n} cases and the interval is correspondingly wide"
                )
                if n < 50
                else None
            ),
        }

    # --- 2. by calendar year -------------------------------------------------
    by_year = {
        str(year): _group_metrics(group, threshold)
        for year, group in frame.groupby("year", sort=True)
    }

    # --- 3. by HARP region ---------------------------------------------------
    per_region = (
        frame.assign(is_fp=frame["outcome"] == "FP", is_tp=frame["outcome"] == "TP")
        .groupby("harp")
        .agg(
            n_windows=("y", "size"),
            n_positive=("y", "sum"),
            n_false_alarms=("is_fp", "sum"),
            n_hits=("is_tp", "sum"),
            max_p=("p", "max"),
            mean_p=("p", "mean"),
        )
        .sort_values("n_false_alarms", ascending=False)
    )
    total_fp = int(per_region["n_false_alarms"].sum())
    top_fp = per_region.head(15).reset_index()
    cumulative = per_region["n_false_alarms"].cumsum()
    regions_for_half = int((cumulative < total_fp / 2).sum() + 1)

    by_region = {
        "n_regions_in_p5": len(per_region),
        "total_false_alarms": total_fp,
        "regions_producing_half_the_false_alarms": regions_for_half,
        "share_of_regions_producing_half": float(regions_for_half / len(per_region)),
        "regions_with_zero_false_alarms": int((per_region["n_false_alarms"] == 0).sum()),
        "top_15_false_alarm_regions": [
            {
                "harp": int(row.harp),
                "n_windows": int(row.n_windows),
                "n_positive_windows": int(row.n_positive),
                "n_false_alarms": int(row.n_false_alarms),
                "n_hits": int(row.n_hits),
                "max_probability": float(row.max_p),
                "mean_probability": float(row.mean_p),
            }
            for row in top_fp.itertuples()
        ],
    }

    # --- 4. by missingness ---------------------------------------------------
    bands = pd.cut(
        frame["nan_frac"],
        bins=[-0.001, 0.0, 0.01, 0.05, 0.25, 1.0],
        labels=["none (0%)", "0-1%", "1-5%", "5-25%", ">25%"],
    )
    by_missingness = {
        str(band): _group_metrics(group, threshold)
        for band, group in frame.groupby(bands, observed=True)
    }

    payload = {
        "analysis": "breakdowns of the frozen model on P5, at the frozen alert threshold",
        "threshold": threshold,
        "overall": {k: extended(y, frame["p"].to_numpy(), threshold)[k] for k in _KEEP},
        "by_goes_class": by_class,
        "by_calendar_year": by_year,
        "by_harp_region": by_region,
        "by_missingness": by_missingness,
        "notes": {
            "x_class": "P5 has 19 X-class windows against 971 M-class. Any statement "
            "about X-class performance rests on 19 cases.",
            "year": "P5 covers 2015-03-07 to 2018-08-17, the declining phase of solar "
            "cycle 24, so the positive base rate drops steeply across the "
            "partition. Precision falls with it, which is arithmetic rather "
            "than model degradation.",
            "region": "False alarms are concentrated: a magnetically complex region "
            "that never produces an M or X flare yields a long run of "
            "consecutive alerted windows, because consecutive windows share "
            "11 of their 12 hours of data.",
        },
    }
    _common.write("breakdowns", payload)
    return payload
