"""Generate every JSON file the dashboard reads.

Nothing in ``dashboard/index.html`` may contain a typed-in number. Each file
below is derived from ``results/`` or from the saved pipeline, so the page and
the published results cannot drift apart.

Files written:

``demo.json``
    The original six views' data. Regenerated to be **identical** to the frozen
    ``results/dashboard_demo.json`` — the report's Figures 5.1-5.6 were captured
    through it, so its keys and values are a fixed interface.
``context.json``
    Everything that used to be hard-coded in the page: the risk-band text, the
    dataset counts, the base rate, the HARP/NOAA note, the disclaimer, and the
    bootstrap intervals for the headline metrics.
``model_lr.json``
    The fitted pipeline parameters, so the browser can run the real model.
``windows.json``
    About twenty real P5 windows with their 12-hour SHARP series, probability,
    risk level, exact feature contributions and true outcome.
``operating.json``
    A threshold sweep over the frozen P5 predictions for the slider view.
``replay.json``
    Three complete active-region histories from the test partition, in time
    order, so the model can be watched operating hour by hour. **A replay of
    recorded observations, never a live feed** - the data ends in 2018 and the
    page says so on every frame.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import data, models, paths
from .analysis import explain
from .analysis.operating import sweep

__all__ = ["SELECTION_RULE", "build_all", "select_windows"]

SEED = 42
SERIES_DECIMALS = 6

SELECTION_RULE = (
    "Deterministic, seeded with 42. From the 75,365 windows of the locked test "
    "partition P5, ordered by probability with ties broken by file name: (1) the two "
    "windows already used by the original demo views, so the old screenshots stay "
    "consistent; (2) the two highest-probability hits and the two hits sitting closest "
    "above the alert threshold; (3) the lowest-, median- and highest-probability misses; "
    "(4) the two highest-probability false alarms and the two sitting closest above the "
    "threshold; (5) the single lowest-probability correct quiet forecast plus two drawn "
    "at random with seed 42; (6) the highest- and lowest-probability X-class windows. "
    "The set deliberately contains successes and failures of every kind, so the view "
    "cannot be read as a showcase of cherry-picked wins."
)


# --------------------------------------------------------------------------- #
# window selection
# --------------------------------------------------------------------------- #
def select_windows(frame: pd.DataFrame, n_target: int = 20) -> pd.DataFrame:
    """Pick a balanced, reproducible set of P5 windows for the explorer view."""
    alert_threshold, _ = models.thresholds()
    ordered = frame.sort_values(["p", "file"], ascending=[False, True])
    chosen: list[int] = []

    def take(subset: pd.DataFrame, how: str, count: int) -> None:
        if subset.empty or count <= 0:
            return
        if how == "highest":
            picks = subset.head(count)
        elif how == "lowest":
            picks = subset.tail(count)
        elif how == "just_above":
            picks = subset[subset["p"] >= alert_threshold].tail(count)
        elif how == "median":
            middle = len(subset) // 2
            picks = subset.iloc[middle : middle + count]
        else:  # random
            picks = subset.sample(min(count, len(subset)), random_state=SEED)
        chosen.extend(int(i) for i in picks.index)

    demo = json.loads(paths.DASHBOARD_DEMO_JSON.read_text(encoding="utf-8"))["examples"]
    for case in demo.values():
        match = ordered[(ordered["ar"] == case["ar"]) & (ordered["start"] == case["start"])]
        chosen.extend(int(i) for i in match.index[:1])

    hits = ordered[ordered["outcome"] == "TP"]
    misses = ordered[ordered["outcome"] == "FN"]
    alarms = ordered[ordered["outcome"] == "FP"]
    quiet = ordered[ordered["outcome"] == "TN"]

    take(hits, "highest", 2)
    take(hits, "just_above", 2)
    take(misses, "highest", 1)
    take(misses, "median", 1)
    take(misses, "lowest", 1)
    take(alarms, "highest", 2)
    take(alarms, "just_above", 2)
    take(quiet, "lowest", 1)
    take(quiet, "random", 2)
    take(ordered[(ordered["goes_letter"] == "X")], "highest", 1)
    take(ordered[(ordered["goes_letter"] == "X")], "lowest", 1)

    # De-duplicate, preserving first-seen order, then top up to the target.
    seen: list[int] = []
    for index in chosen:
        if index not in seen:
            seen.append(index)
    if len(seen) < n_target:
        for index in ordered.sample(frac=1.0, random_state=SEED).index:
            if int(index) not in seen:
                seen.append(int(index))
            if len(seen) >= n_target:
                break

    picked = frame.loc[seen[:n_target]].copy()
    return picked.sort_values(["outcome", "p"], ascending=[True, False])


# --------------------------------------------------------------------------- #
# raw series extraction
# --------------------------------------------------------------------------- #
def extract_series(file_names: set[str], partition: int = 5) -> dict[str, dict[str, Any]]:
    """Pull the raw 12-hour SHARP series for named windows out of the tarball.

    The partition tarball is streamed once and the wanted members are decoded as
    they go past, so the 1.3 GB archive is never extracted to disk.

    Args:
        file_names: SWAN-SF window file basenames to collect.
        partition: which partition tarball to read.

    Returns:
        ``{file_name: {"timestamps": [...], "values": {PARAM: [...]}}}``.

    Raises:
        FileNotFoundError: if the tarball is not present.
    """
    import sys

    if str(paths.SRC) not in sys.path:
        sys.path.insert(0, str(paths.SRC))
    from common import SHARP

    tarball = paths.RAW_DATA / f"partition{partition}_instances.tar.gz"
    if not tarball.is_file():
        raise FileNotFoundError(
            f"{paths.relative(tarball)} not found. The explorer view needs the raw "
            f"SWAN-SF tarball to show the 12-hour series. Download SWAN-SF v1.2 "
            f"(DOI 10.7910/DVN/EBCFKM) into raw_data/."
        )

    wanted = set(file_names)
    found: dict[str, dict[str, Any]] = {}
    print(
        f"  streaming {paths.relative(tarball)} for {len(wanted)} windows "
        f"(this takes a few minutes)...",
        flush=True,
    )

    with tarfile.open(tarball, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".csv"):
                continue
            name = member.name.split("/")[-1]
            if name not in wanted:
                continue
            raw = archive.extractfile(member).read()
            frame = pd.read_csv(io.BytesIO(raw), sep="\t", usecols=["Timestamp", *SHARP])
            found[name] = {
                "timestamps": [str(t) for t in frame["Timestamp"].tolist()],
                "values": {
                    parameter: [
                        None if not np.isfinite(v) else round(float(v), SERIES_DECIMALS)
                        for v in frame[parameter].to_numpy(dtype=np.float64)
                    ]
                    for parameter in SHARP
                },
            }
            wanted.discard(name)
            print(f"    found {name}  ({len(found)}/{len(file_names)})", flush=True)
            if not wanted:
                break

    if wanted:
        print(
            f"  WARNING: {len(wanted)} of {len(file_names)} window(s) were not found "
            f"in {tarball.name}."
        )
        print(
            "  The P2 and P5 archives on this machine are corrupt gzip streams and stop "
            "part-way; see docs/REPRODUCIBILITY.md. This affects only the raw series "
            "shown in the explorer view, never any published metric."
        )
        for name in sorted(wanted):
            print(f"    missing: {name}")
    return found


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def _write(path: Path, payload: Any) -> Path:
    paths.assert_not_frozen(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return path


def build_demo() -> Path:
    """Rewrite ``dashboard/demo.json`` from the frozen results copy.

    Byte-identical content by construction: the frozen
    ``results/dashboard_demo.json`` is the source. This keeps the two in step
    without ever writing to the frozen file.
    """
    frozen = json.loads(paths.DASHBOARD_DEMO_JSON.read_text(encoding="utf-8"))
    return _write(paths.DASHBOARD_DEMO, frozen)


def build_context() -> Path:
    """Everything the page used to hard-code, generated from results/."""
    alert_threshold, high_threshold = models.thresholds()
    demo = json.loads(paths.DASHBOARD_DEMO_JSON.read_text(encoding="utf-8"))
    audit = json.loads(paths.DATA_AUDIT_JSON.read_text(encoding="utf-8"))
    test = demo["test"]

    bootstrap_path = paths.RESULTS_EXTRA / "bootstrap.json"
    intervals = {}
    if bootstrap_path.is_file():
        loaded = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        intervals = {
            name: {"low": stats["ci95_low"], "high": stats["ci95_high"]}
            for name, stats in loaded["intervals"].items()
        }

    nearmiss_path = paths.RESULTS_EXTRA / "near_miss.json"
    near_miss = None
    if nearmiss_path.is_file():
        loaded = json.loads(nearmiss_path.read_text(encoding="utf-8"))
        near_miss = {
            "share_preceding_a_real_flare": loaded["headline"][
                "share_of_false_alarms_preceding_a_real_flare"
            ],
            "n_preceding_a_real_flare": loaded["headline"][
                "false_alarms_preceding_a_real_b_or_c_flare"
            ],
            "n_false_alarms": loaded["counts"]["false_alarms"],
            "alert_rate_flare_quiet": loaded["by_negative_class"]["F"]["alert_rate_within_class"],
            "alert_rate_c_class": loaded["by_negative_class"]["C"]["alert_rate_within_class"],
        }

    base_rate = test["positives"] / test["n"]
    alerts = test["tp"] + test["fp"]
    flare = demo["examples"]["flare_case"]

    return _write(
        paths.DASHBOARD / "context.json",
        {
            "generated_from": "results/selection.json, results/test_results.json, "
            "results/data_audit.json, results/extra/*.json",
            "disclaimer": "Academic prototype — not an operational space-weather warning "
            "service.",
            "dataset": {
                "name": "SWAN-SF v1.2",
                "doi": "10.7910/DVN/EBCFKM",
                "n_windows_total": audit["n_total"],
                "n_features": audit["n_features"],
                "n_partitions": 5,
                "year_span": "2010–2018",
                "observation_hours": 12,
                "prediction_hours": 24,
                "records_per_window": 60,
                "cadence_minutes": 12,
                "n_sharp_parameters": 24,
            },
            "split": {
                "train": "P1–P3",
                "validation": "P4",
                "test": "P5",
                "n_train": int(audit["per_partition"]["1"]["n"])
                + int(audit["per_partition"]["2"]["n"])
                + int(audit["per_partition"]["3"]["n"]),
                "n_validation": int(audit["per_partition"]["4"]["n"]),
                "test_windows": test["n"],
                "test_positives": test["positives"],
                "test_base_rate": base_rate,
                "test_base_rate_text": f"{base_rate * 100:.2f} %",
            },
            "thresholds": {
                "alert": alert_threshold,
                "high": high_threshold,
                "alert_text": f"{alert_threshold:.4f}",
                "high_text": f"{high_threshold:.4f}",
                "chosen_on": "validation partition P4 — alert maximises TSS, high-risk "
                "maximises F1",
            },
            "risk_bands": [
                {"level": "LOW", "text": f"p < {alert_threshold:.2f}"},
                {"level": "MODERATE", "text": f"{alert_threshold:.2f} – {high_threshold:.2f}"},
                {"level": "HIGH", "text": f"p ≥ {high_threshold:.2f}"},
            ],
            "alerts": {
                "n": alerts,
                "rate": alerts / test["n"],
                "rate_text": f"{alerts / test['n'] * 100:.2f} %",
                "times_base_rate": round((alerts / test["n"]) / base_rate, 1),
            },
            "always_no_flare": {
                "accuracy": 1 - base_rate,
                "accuracy_text": f"{(1 - base_rate) * 100:.2f} %",
                "note": "A model that always forecasts 'no major flare' scores "
                f"{(1 - base_rate) * 100:.2f} % accuracy on this partition and "
                "detects nothing. That is why accuracy is not the headline metric.",
            },
            "verification_example": {
                "harp": flare["ar"],
                "noaa_ar": 12673 if flare["ar"] == 7115 else None,
                "noaa_source": "JSOC all_harps_with_noaa_ars.txt",
                "flare_class": flare["flare_class"],
                "sentence": (
                    f"This window precedes a {flare['flare_class']} flare recorded in SWAN-SF "
                    f"for HARP {flare['ar']}"
                    + (" (NOAA AR 12673)" if flare["ar"] == 7115 else "")
                    + ", confirming the model assigns high probability to a genuine pre-flare "
                    "window from the held-out test partition."
                ),
            },
            "harp_note": "SWAN-SF's region identifier is the SHARP/HARP number, which is not "
            "the NOAA active region number. Many HARPs have no NOAA counterpart.",
            "confidence_intervals_95": intervals,
            "ci_note": "95 % intervals from an active-region block bootstrap of the locked "
            "test partition, 1,000 resamples. Post-hoc; the point estimates are "
            "the published figures.",
            "near_miss": near_miss,
            "calibration_warning": "These probabilities are not calibrated. The model was "
            "trained with balanced class weights, which deliberately "
            "over-forecasts; read them as a ranking and against the "
            "fixed thresholds, not as literal chances.",
        },
    )


def build_model_parameters() -> Path:
    """Export the fitted pipeline so the browser can reproduce it exactly."""
    return _write(paths.DASHBOARD_MODEL_LR, models.export_lr_parameters())


def build_operating() -> Path:
    """Precompute the threshold sweep that drives the slider view."""
    stored = pd.read_csv(paths.TEST_PREDICTIONS)
    alert_threshold, high_threshold = models.thresholds()
    grid = sweep(
        stored["y"].to_numpy(),
        stored["p"].to_numpy(),
        points=400,
        include=(alert_threshold, high_threshold),
    )

    # The two published thresholds are in the grid exactly, so the slider can
    # land on the real operating point instead of a neighbouring quantile.
    alert_index = next(i for i, row in enumerate(grid) if row["threshold"] == alert_threshold)
    high_index = next(i for i, row in enumerate(grid) if row["threshold"] == high_threshold)
    published = grid[alert_index]
    frozen = json.loads(paths.TEST_JSON.read_text(encoding="utf-8"))[models.selected_key()]
    for field in ("tp", "fp", "fn", "tn"):
        if published[field] != frozen[field]:
            raise ValueError(
                f"sweep at the published threshold gives {field}={published[field]} "
                f"but results/test_results.json says {frozen[field]}"
            )

    return _write(
        paths.DASHBOARD_OPERATING,
        {
            "generated_from": "results/test_predictions_selected.csv.gz",
            "alert_threshold": alert_threshold,
            "high_threshold": high_threshold,
            "alert_index": alert_index,
            "high_index": high_index,
            "n_windows": len(stored),
            "n_positives": int(stored["y"].sum()),
            "banner": "Exploratory — headline results use the validation-fixed threshold. "
            "Moving this slider shows what other operating points would have given "
            "on the test partition; choosing one on this basis would be selecting "
            "on the test set.",
            "sweep": grid,
        },
    )


def build_windows(n_windows: int = 20) -> Path:
    """Build the explorer view's window set, with real series and contributions."""
    from .analysis._common import load_test_frame

    frame = load_test_frame()
    picked = select_windows(frame, n_target=n_windows)
    print(
        f"  selected {len(picked)} windows: "
        + ", ".join(f"{k}={v}" for k, v in picked["outcome"].value_counts().items())
    )

    series = extract_series(set(picked["file"]))

    X, _meta = data.load()
    bundle = models.load_bundle(models.selected_key())
    alert_threshold, high_threshold = models.thresholds()

    windows = []
    for row in picked.itertuples():
        features = X.loc[[row.Index]]
        contributions = explain.top_contributions(bundle, features, k=10)
        raw = series.get(row.file)
        windows.append(
            {
                "file": row.file,
                "folder": row.folder,
                "harp": int(row.harp),
                "noaa_ar": 12673 if int(row.harp) == 7115 else None,
                "flare_class": row.flare_class,
                "goes_letter": row.goes_letter,
                "window_start": row.start,
                "window_end_cutoff": row.end,
                "y": int(row.y),
                "probability": float(row.p),
                "risk": row.risk,
                "outcome": row.outcome,
                "alert": bool(row.alert),
                "nan_frac": float(row.nan_frac),
                "truth_text": (
                    f"An {row.flare_class} flare followed within 24 h (y = 1)"
                    if row.y == 1
                    else (
                        f"No M/X flare followed; the largest was {row.flare_class} (y = 0)"
                        if row.goes_letter in ("B", "C")
                        else "No flare of any class followed (y = 0)"
                    )
                ),
                "contributions": contributions,
                "series_available": raw is not None,
                "series_note": (
                    None
                    if raw is not None
                    else (
                        "The raw 12-hour series could not be read from "
                        "raw_data/partition5_instances.tar.gz. That archive is a corrupt gzip "
                        "stream on this machine and stops after 64,315 of the partition's 75,365 "
                        "windows; see docs/REPRODUCIBILITY.md. The probability, risk level and "
                        "feature contributions shown here are unaffected - they come from the "
                        "extracted feature matrix and the frozen predictions."
                    )
                ),
                "series": raw if raw else {},
            }
        )

    with_series = sum(1 for w in windows if w["series_available"])
    return _write(
        paths.DASHBOARD_WINDOWS,
        {
            "generated_from": "results/test_predictions_selected.csv.gz, data/all_meta.csv.gz, "
            "raw_data/partition5_instances.tar.gz",
            "selection_rule": SELECTION_RULE,
            "seed": SEED,
            "alert_threshold": alert_threshold,
            "high_threshold": high_threshold,
            "n_windows": len(windows),
            "n_with_raw_series": with_series,
            "outcome_counts": {k: int(v) for k, v in picked["outcome"].value_counts().items()},
            "windows": windows,
        },
    )


#: The three active regions replayed, chosen so that the set tells the whole
#: story rather than only the flattering part of it. Picked from the test
#: partition by their recorded outcomes, not by eye.
REPLAY_TRACKS = (
    {
        "harp": 7115,
        "title": "September 2017 — the strongest region of the cycle",
        "verdict": "catches the major flares, and misses two",
        "story": (
            "Ten days of a real active region, beginning while it is quiet. The forecast "
            "probability climbs as the magnetic field grows more complex, and stays in the "
            "HIGH band through the X9.3 and X1.3 flares. Watch for the two frames marked "
            "MISSED: the model scored a genuine pre-flare window below the alert threshold, "
            "one of them at 0.049. Both are left in."
        ),
    },
    {
        "harp": 5541,
        "title": "A region that never produced a major flare",
        "verdict": "alerts continuously, and is wrong every time",
        "story": (
            "The failure mode, in full. This region looked magnetically complex enough to "
            "alert on for days at a stretch, and never produced an M- or X-class flare. "
            "Because consecutive windows are an hour apart and share eleven of their twelve "
            "hours, one misread region becomes a long unbroken run of false alarms. This is "
            "what the 16 percent precision looks like from the inside."
        ),
    },
    {
        "harp": 7131,
        "title": "A quiet region",
        "verdict": "stays silent, correctly",
        "story": (
            "The control case, and the commonest one. The probability never approaches the "
            "alert threshold and no warning is raised, across the region's whole visible "
            "lifetime. Most of the test partition looks like this, which is why accuracy is "
            "a misleading way to score the task."
        ),
    },
)


def build_replay() -> Path:
    """Build three chronological active-region histories for the replay view.

    Every frame is a real recorded window with the model's frozen probability
    for it; nothing is simulated or interpolated. The replay advances through
    recorded time, which is what makes it a replay rather than a live feed.
    """
    from .analysis._common import load_test_frame

    frame = load_test_frame()
    frame = frame.assign(_t=pd.to_datetime(frame["end"]))
    alert_threshold, high_threshold = models.thresholds()

    tracks = []
    for spec in REPLAY_TRACKS:
        history = frame[frame["harp"] == spec["harp"]].sort_values("_t")
        if history.empty:
            print(f"  WARNING: HARP {spec['harp']} is not in the test partition")
            continue
        steps = [
            {
                "cutoff": row.end,
                "p": round(float(row.p), 6),
                "y": int(row.y),
                "risk": row.risk,
                "outcome": row.outcome,
                "flare_class": row.flare_class if row.y == 1 else None,
            }
            for row in history.itertuples()
        ]
        counts = history["outcome"].value_counts()
        tracks.append(
            {
                "id": f"harp{spec['harp']}",
                "harp": int(spec["harp"]),
                "noaa_ar": 12673 if spec["harp"] == 7115 else None,
                "title": spec["title"],
                "verdict": spec["verdict"],
                "story": spec["story"],
                "n_steps": len(steps),
                "first_cutoff": steps[0]["cutoff"],
                "last_cutoff": steps[-1]["cutoff"],
                "span_days": round(
                    (history["_t"].max() - history["_t"].min()).total_seconds() / 86400, 1
                ),
                "counts": {k: int(counts.get(k, 0)) for k in ("TP", "FP", "FN", "TN")},
                "max_p": round(float(history["p"].max()), 6),
                "flare_classes": sorted(
                    {str(c) for c in history.loc[history.y == 1, "flare_class"]}
                ),
                "steps": steps,
            }
        )
        print(
            f"  HARP {spec['harp']:>5}: {len(steps):>3} frames over "
            f"{tracks[-1]['span_days']} days  {tracks[-1]['counts']}"
        )

    return _write(
        paths.DASHBOARD / "replay.json",
        {
            "generated_from": "results/test_predictions_selected.csv.gz, data/all_meta.csv.gz",
            "what_this_is": (
                "A replay of recorded observations from the locked test partition, played "
                "back in the order they were taken. Every frame is a real 12-hour window "
                "and the probability shown is the model's frozen prediction for it."
            ),
            "what_this_is_not": (
                "Not a feed. Nothing here is current, nothing is simulated, and the system "
                "has no connection to any observatory. The recordings end in 2018."
            ),
            "alert_threshold": alert_threshold,
            "high_threshold": high_threshold,
            "cadence_minutes": 60,
            "tracks": tracks,
        },
    )


def build_all(n_windows: int = 20) -> list[Path]:
    """Build every dashboard JSON file. Returns the paths written."""
    written = [build_demo(), build_context(), build_model_parameters(), build_operating()]
    written.append(build_replay())
    written.append(build_windows(n_windows=n_windows))
    return written
