"""Error gallery: the model's worst mistakes on P5, named.

Two lists, both at the frozen alert threshold:

* the **most confident false alarms** — quiet windows the model scored highest;
* the **worst misses** — genuine M or X pre-flare windows the model scored
  lowest, including the single lowest-probability miss.

Naming the failures is the point. Every entry carries the SWAN-SF file name, so
an examiner can pull the original window out of the tarball and check it.
"""

from __future__ import annotations

import pandas as pd

from .. import models
from . import _common

__all__ = ["run"]

N_SHOW = 20


def _entries(frame: pd.DataFrame) -> list[dict]:
    """Render selected windows as JSON-friendly records."""
    alert_threshold, high_threshold = models.thresholds()
    rows = []
    for row in frame.itertuples():
        rows.append(
            {
                "file": row.file,
                "folder": row.folder,
                "harp": int(row.harp),
                "flare_class": row.flare_class,
                "goes_letter": row.goes_letter,
                "role": row.role if isinstance(row.role, str) else "",
                "window_start": row.start,
                "window_end_cutoff": row.end,
                "y": int(row.y),
                "probability": float(row.p),
                "risk": models.risk_level(float(row.p), alert_threshold, high_threshold),
                "outcome": row.outcome,
                "nan_frac": float(row.nan_frac),
                "bad_quality_frac": float(row.bad_quality_frac),
            }
        )
    return rows


def run() -> dict:
    """Build the error gallery and write it to results/extra/."""
    frame = _common.load_test_frame()
    threshold, _ = models.thresholds()

    false_alarms = frame[frame["outcome"] == "FP"].sort_values("p", ascending=False)
    misses = frame[frame["outcome"] == "FN"].sort_values("p", ascending=True)
    hits = frame[frame["outcome"] == "TP"]

    # How many distinct regions and how many distinct flare events do the misses cover?
    missed_regions = sorted(int(h) for h in misses["harp"].unique())
    missed_classes = misses["goes_letter"].value_counts().to_dict()

    # Runs of consecutive false alarms within one region, to show that 4,707 false
    # alarms are not 4,707 independent mistakes.
    runs = []
    for harp, group in frame[frame["y"] == 0].sort_values(["harp", "start"]).groupby("harp"):
        flags = (group["outcome"] == "FP").to_numpy()
        length = 0
        for flag in flags:
            length = length + 1 if flag else 0
            if length:
                runs.append((int(harp), length))
    longest_runs: dict[int, int] = {}
    for harp, length in runs:
        longest_runs[harp] = max(longest_runs.get(harp, 0), length)
    top_runs = sorted(longest_runs.items(), key=lambda kv: -kv[1])[:10]

    payload = {
        "analysis": "error gallery for the frozen model on P5",
        "threshold": threshold,
        "counts": {
            "true_positives": int((frame["outcome"] == "TP").sum()),
            "false_positives": int((frame["outcome"] == "FP").sum()),
            "false_negatives": int((frame["outcome"] == "FN").sum()),
            "true_negatives": int((frame["outcome"] == "TN").sum()),
        },
        "misses": {
            "n": len(misses),
            "n_distinct_harp_regions": len(missed_regions),
            "harp_regions": missed_regions,
            "by_goes_class": missed_classes,
            "lowest_probability": float(misses["p"].iloc[0]) if len(misses) else None,
            "highest_probability": float(misses["p"].iloc[-1]) if len(misses) else None,
            "worst_misses": _entries(misses.head(N_SHOW)),
            "note": "These are windows whose active region did produce an M or X flare "
            "within 24 h of the cutoff, and which the model scored below the "
            "alert threshold. They are the failures that matter most "
            "operationally.",
        },
        "false_alarms": {
            "n": len(false_alarms),
            "n_distinct_harp_regions": int(false_alarms["harp"].nunique()),
            "highest_probability": float(false_alarms["p"].iloc[0]) if len(false_alarms) else None,
            "n_above_high_risk_threshold": int((false_alarms["risk"] == "HIGH").sum()),
            "most_confident": _entries(false_alarms.head(N_SHOW)),
            "longest_consecutive_runs": [
                {"harp": harp, "consecutive_false_alarm_windows": length}
                for harp, length in top_runs
            ],
            "note": "Because consecutive windows of one region are one hour apart and "
            "share 11 of their 12 hours of data, a single mis-read region "
            "produces a long run of false alarms. The run lengths above show "
            "that the 4,707 false alarms are far fewer than 4,707 independent "
            "errors.",
        },
        "hits_for_contrast": {
            "n": len(hits),
            "median_probability": float(hits["p"].median()) if len(hits) else None,
            "n_above_high_risk_threshold": int((hits["risk"] == "HIGH").sum()),
            "least_confident_hits": _entries(hits.sort_values("p").head(10)),
        },
        "missingness_of_errors": {
            "mean_nan_frac_misses": float(misses["nan_frac"].mean()) if len(misses) else None,
            "mean_nan_frac_false_alarms": float(false_alarms["nan_frac"].mean()),
            "mean_nan_frac_all": float(frame["nan_frac"].mean()),
        },
    }
    _common.write("error_gallery", payload)
    return payload
