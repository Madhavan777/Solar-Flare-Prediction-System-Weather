"""Shared helpers for the post-hoc analyses.

Everything in ``solarflare.analysis`` is computed **after** the locked test
partition was unlocked, so none of it informed the choice of model or threshold.
:data:`POSTHOC` is embedded in every artefact these modules write, so a table or
figure lifted out of this repository still says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import models, paths

__all__ = ["POSTHOC", "RESAMPLE_SEED", "load_test_frame", "write"]

POSTHOC = (
    "Post-hoc analysis. Computed on the locked test partition P5 after model "
    "selection was complete and written to results/selection.json. It did not "
    "influence the choice of model, features or thresholds, and none of the "
    "frozen numbers in results/ depend on it."
)

#: Seed for every resampling procedure here, matching the project seed.
RESAMPLE_SEED = 42


def load_test_frame() -> pd.DataFrame:
    """Return one row per P5 window with its metadata and frozen prediction.

    Joins ``results/test_predictions_selected.csv.gz`` (whose ``row`` column
    indexes the merged metadata) onto ``data/all_meta.csv.gz``, and adds:

    ``p``
        the selected model's probability, read from the frozen file — not
        recomputed, so the analyses cannot drift from the published numbers.
    ``risk``
        LOW / MODERATE / HIGH from the two validation-fixed thresholds.
    ``alert``
        whether ``p >= alert_threshold``.
    ``outcome``
        TP, FP, FN or TN at the frozen alert threshold.
    ``year``
        calendar year of the window's last timestamp (the prediction cutoff).
    ``harp``
        the HARP region number — SWAN-SF's ``ar`` field. Not a NOAA AR number.

    Raises:
        FileNotFoundError: if the merged metadata is absent.
        ValueError: if the join does not line up with the published row count.
    """
    meta = pd.read_csv(paths.ALL_META)
    stored = pd.read_csv(paths.TEST_PREDICTIONS)
    frame = meta.iloc[stored["row"].to_numpy()].reset_index(drop=True)
    if not np.array_equal(frame["y"].to_numpy(), stored["y"].to_numpy()):
        raise ValueError(
            "labels in test_predictions_selected.csv.gz do not match all_meta.csv.gz; "
            "the merged data may have been rebuilt in a different order"
        )
    if len(frame) != 75365:
        raise ValueError(f"expected 75,365 test windows, joined {len(frame)}")

    alert_threshold, high_threshold = models.thresholds()
    frame["p"] = stored["p"].to_numpy()
    frame["harp"] = frame["ar"]
    frame["alert"] = frame["p"] >= alert_threshold
    frame["risk"] = [models.risk_level(p, alert_threshold, high_threshold) for p in frame["p"]]
    frame["outcome"] = np.select(
        [
            (frame["y"] == 1) & frame["alert"],
            (frame["y"] == 0) & frame["alert"],
            (frame["y"] == 1) & ~frame["alert"],
        ],
        ["TP", "FP", "FN"],
        default="TN",
    )
    frame["year"] = pd.to_datetime(frame["last_ts"]).dt.year
    return frame


def write(name: str, payload: dict[str, Any]) -> Path:
    """Write one analysis result to ``results/extra/<name>.json``.

    The payload is stamped with :data:`POSTHOC` under the key ``posthoc`` before
    being written, and the destination is checked against the frozen set.
    """
    paths.RESULTS_EXTRA.mkdir(parents=True, exist_ok=True)
    path = paths.assert_not_frozen(paths.RESULTS_EXTRA / f"{name}.json")
    body = {"posthoc": POSTHOC, **payload}
    path.write_text(json.dumps(body, indent=1, default=str), encoding="utf-8")
    return path
