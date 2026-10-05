"""Loading, merging and auditing the SWAN-SF feature matrix.

The merged matrix (``data/all_X.npz``) and window metadata
(``data/all_meta.csv.gz``) are produced from the per-partition chunks in
``data/features/``. They are not in version control: 190 MB of derived data.
:func:`merge_chunks` rebuilds them deterministically in about four minutes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import paths

# ``common`` is the top-level module the saved pipelines unpickle against.
if str(paths.SRC) not in sys.path:
    sys.path.insert(0, str(paths.SRC))
from common import ALL_COLS, TEST_P, TRAIN_P, VAL_P

__all__ = [
    "SPLIT_OF_PARTITION",
    "audit",
    "compare_audit",
    "load",
    "merge_chunks",
    "split_masks",
    "window_gap_stats",
]

#: The frozen partition-to-split mapping. Chronological, never shuffled.
SPLIT_OF_PARTITION: dict[int, str] = {
    **dict.fromkeys(TRAIN_P, "train"),
    **dict.fromkeys(VAL_P, "validation"),
    **dict.fromkeys(TEST_P, "test"),
}


def check_raw_archives(partitions: tuple[int, ...] = (1, 2, 3, 4, 5)) -> dict[int, str]:
    """Verify that each raw partition tarball is a complete gzip stream.

    A corrupt or partial download decompresses cleanly up to the damage and then
    raises, and ``tarfile`` in streaming mode stops quietly at that point — so a
    re-extraction would silently produce a short dataset rather than failing.
    This is not hypothetical: on the machine these results were verified on, the
    P2 and P5 archives are corrupt, and P5 yields only 64,315 of its 75,365
    windows. See docs/REPRODUCIBILITY.md.

    The already-extracted features in ``data/features/`` are unaffected; they
    were produced while the archives were intact, which the window counts in
    ``results/data_audit.json`` confirm.

    Returns:
        ``{partition: "ok" | "missing" | "<error description>"}``.
    """
    import gzip

    status: dict[int, str] = {}
    for partition in partitions:
        path = paths.RAW_DATA / f"partition{partition}_instances.tar.gz"
        if not path.is_file():
            status[partition] = "missing"
            continue
        try:
            with gzip.open(path, "rb") as handle:
                while handle.read(8 << 20):
                    pass
            status[partition] = "ok"
        except Exception as exc:
            status[partition] = f"{type(exc).__name__}: {exc}"
    return status


def _chunk_files() -> list[Path]:
    """Return the per-partition feature chunks in their canonical merge order."""
    files: list[Path] = []
    for partition in range(1, 6):
        files.extend(sorted(paths.FEATURE_CHUNKS.glob(f"P{partition}_c*_X.npz")))
    return files


def merge_chunks(force: bool = False, verbose: bool = True) -> tuple[Path, Path]:
    """Build ``data/all_X.npz`` and ``data/all_meta.csv.gz`` from the chunks.

    The merge order is ``sorted(glob("P{1..5}_c*_X.npz"))``, which is the order
    ``src/merge_audit.py`` used and therefore the order the ``row`` indices in
    ``results/test_predictions_selected.csv.gz`` refer to.

    Args:
        force: overwrite the merged files if they already exist. Off by default
            so that a stray run cannot silently replace them.
        verbose: print per-chunk shapes.

    Returns:
        The paths of the two written files.

    Raises:
        FileNotFoundError: if no chunks are present.
        FileExistsError: if the outputs exist and ``force`` is false.
    """
    files = _chunk_files()
    if not files:
        raise FileNotFoundError(
            f"no feature chunks in {paths.relative(paths.FEATURE_CHUNKS)}. "
            f"Run 'python -m solarflare features' first (needs raw_data/)."
        )
    if not force and (paths.ALL_X.exists() or paths.ALL_META.exists()):
        raise FileExistsError(
            f"{paths.relative(paths.ALL_X)} or {paths.relative(paths.ALL_META)} already "
            f"exists. Pass --force to rebuild."
        )

    blocks, metas, cols_ref = [], [], None
    for path in files:
        loaded = np.load(path, allow_pickle=False)
        cols = list(loaded["cols"])
        if cols_ref is None:
            cols_ref = cols
        elif cols != cols_ref:
            raise ValueError(f"feature column names differ in {path.name}")
        blocks.append(loaded["X"])
        partition = int(path.name[1])
        frame = pd.read_csv(path.with_name(path.name.replace("_X.npz", "_meta.csv")))
        frame["partition"] = partition
        metas.append(frame)
        if verbose:
            print(f"  {path.name:16s} {loaded['X'].shape}")

    X = np.vstack(blocks)
    meta = pd.concat(metas, ignore_index=True)
    if len(X) != len(meta):
        raise ValueError(f"feature rows ({len(X)}) and metadata rows ({len(meta)}) disagree")
    if list(cols_ref) != ALL_COLS:
        raise ValueError("chunk column names do not match common.ALL_COLS")

    paths.DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths.assert_not_frozen(paths.ALL_X), X=X, cols=np.array(cols_ref))
    meta.to_csv(paths.assert_not_frozen(paths.ALL_META), index=False)
    if verbose:
        print(f"merged X={X.shape} meta={meta.shape}")
        print(f"wrote {paths.relative(paths.ALL_X)} ({paths.ALL_X.stat().st_size:,} B)")
        print(f"wrote {paths.relative(paths.ALL_META)} ({paths.ALL_META.stat().st_size:,} B)")
    return paths.ALL_X, paths.ALL_META


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the merged feature matrix and window metadata.

    Non-finite feature values are replaced with NaN exactly as
    ``src/train_eval.py`` does, so that the pipelines' median imputer sees the
    same input it was fitted against.

    Returns:
        ``(X, meta)`` — a 331,185 × 144 float64 frame whose columns are
        ``common.ALL_COLS``, and the matching one-row-per-window metadata.

    Raises:
        FileNotFoundError: if the merged files are absent.
    """
    if not paths.ALL_X.exists() or not paths.ALL_META.exists():
        raise FileNotFoundError(
            f"{paths.relative(paths.ALL_X)} / {paths.relative(paths.ALL_META)} not found. "
            f"Rebuild them with 'python -m solarflare audit'."
        )
    loaded = np.load(paths.ALL_X)
    X = pd.DataFrame(loaded["X"].astype(np.float64), columns=list(loaded["cols"]))
    if list(X.columns) != ALL_COLS:
        raise ValueError("stored feature columns do not match common.ALL_COLS")
    X = X.replace([np.inf, -np.inf], np.nan)
    meta = pd.read_csv(paths.ALL_META)
    if len(X) != len(meta):
        raise ValueError(f"feature rows ({len(X)}) and metadata rows ({len(meta)}) disagree")
    return X, meta


def split_masks(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean ``(train, validation, test)`` masks over ``meta``'s rows.

    Train is P1–P3, validation is P4, test is P5 — fixed, chronological, and
    never shuffled, because consecutive windows of one active region overlap by
    11 of their 12 hours.
    """
    return tuple(  # type: ignore[return-value]
        meta["partition"].isin(group).to_numpy() for group in (TRAIN_P, VAL_P, TEST_P)
    )


def audit(X: np.ndarray | pd.DataFrame, meta: pd.DataFrame) -> dict[str, Any]:
    """Recompute the exploratory data audit.

    Identical in logic to ``src/merge_audit.py``, so the result can be compared
    key-by-key with the frozen ``results/data_audit.json``. Returns a plain dict
    with the same keys and ordering.
    """
    values = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    meta = meta.copy()
    meta["s"] = pd.to_datetime(meta["start"])
    meta["e"] = pd.to_datetime(meta["end"])

    out: dict[str, Any] = {}
    out["n_total"] = len(meta)
    out["n_features"] = int(values.shape[1])
    out["per_partition"] = (
        meta.groupby("partition")
        .agg(
            n=("y", "size"),
            pos=("y", "sum"),
            s_min=("s", "min"),
            e_max=("e", "max"),
            n_ar=("ar", "nunique"),
        )
        .astype(str)
        .to_dict("index")
    )
    out["letter_by_folder"] = pd.crosstab(meta["goes_letter"], meta["folder"]).to_dict()
    out["letter_by_partition"] = pd.crosstab(meta["partition"], meta["goes_letter"]).to_dict(
        "index"
    )
    out["role_counts"] = meta["role"].fillna("").replace("", "none(FQ)").value_counts().to_dict()
    out["n_rows_dist"] = meta["n_rows"].value_counts().head(5).to_dict()
    out["window_hours"] = (
        ((meta["e"] - meta["s"]).dt.total_seconds() / 3600)
        .round(2)
        .value_counts()
        .head(5)
        .to_dict()
    )
    out["nan_frac_mean"] = float(meta["nan_frac"].mean())
    out["frac_windows_with_any_nan"] = float((meta["nan_frac"] > 0).mean())
    out["bad_quality_frac_mean"] = float(meta["bad_quality_frac"].mean())
    out["feature_nan_frac"] = float(np.isnan(values).mean())
    regions = {p: set(meta.loc[meta.partition == p, "ar"]) for p in range(1, 6)}
    out["shared_ar"] = {
        f"P{a}-P{b}": len(regions[a] & regions[b]) for a in range(1, 6) for b in range(a + 1, 6)
    }
    return out


def compare_audit(
    recomputed: dict[str, Any], reference: dict[str, Any], tol: float = 1e-12
) -> list[str]:
    """Compare two audit dicts recursively; return a list of mismatch messages.

    Both sides are first round-tripped through JSON with ``default=str``, which
    is exactly what ``src/merge_audit.py`` does when it writes the file. Without
    that step a freshly computed audit would carry pandas' integer group keys
    while the stored reference carries JSON's string keys, and the comparison
    would be meaningless (or raise).

    Values are then compared as floats when either side is a float, otherwise as
    strings. An empty list means the audit reproduces.
    """
    recomputed = json.loads(json.dumps(recomputed, default=str))
    reference = json.loads(json.dumps(reference, default=str))

    def walk(a: Any, b: Any, path: str = "") -> list[str]:
        bad: list[str] = []
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b), key=str):
                if key not in a:
                    bad.append(f"{path}/{key}: present in recomputed only")
                elif key not in b:
                    bad.append(f"{path}/{key}: present in reference only")
                else:
                    bad += walk(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, float) or isinstance(b, float):
            try:
                if abs(float(a) - float(b)) > tol:
                    bad.append(f"{path}: {a} != {b}")
            except (TypeError, ValueError):
                bad.append(f"{path}: {a!r} != {b!r}")
        elif str(a) != str(b):
            bad.append(f"{path}: {a!r} != {b!r}")
        return bad

    return walk(recomputed, reference)


def window_gap_stats(meta: pd.DataFrame, by: str = "ar", time: str = "start") -> dict[str, Any]:
    """Quantify how closely consecutive windows of one active region are spaced.

    This is the statistic behind the claim that consecutive same-region windows
    are one hour apart, and the quantitative basis for refusing a random split:
    a one-hour step between 12-hour windows means adjacent windows share 11 of
    their 12 hours of observations.

    Definition, stated explicitly: group the windows by ``by``, sort each group
    by ``time``, and difference consecutive timestamps. A group of *k* windows
    contributes *k* − 1 gaps, so the total is
    ``len(meta) - meta[by].nunique()``.

    Args:
        meta: window metadata.
        by: grouping column, normally ``"ar"`` (the HARP region number).
        time: ``"start"`` or ``"end"``.

    Returns:
        A dict with the group and gap counts, the share of gaps that are exactly
        one hour, the minimum, maximum and median gap, and the six commonest
        gap values in hours.
    """
    frame = meta.copy()
    frame["_t"] = pd.to_datetime(frame[time])
    grouped = frame.sort_values([by, "_t"]).groupby(by, sort=False)["_t"]
    gaps = grouped.diff().dropna().dt.total_seconds() / 3600.0
    return {
        "group_by": by,
        "ordered_by": time,
        "n_windows": len(frame),
        "n_groups": int(grouped.ngroups),
        "n_gaps": len(gaps),
        "n_gaps_exactly_1h": int(np.isclose(gaps, 1.0).sum()),
        "frac_exactly_1h": float(np.isclose(gaps, 1.0).mean()),
        "frac_le_1h": float((gaps <= 1.0 + 1e-9).mean()),
        "frac_le_2h": float((gaps <= 2.0 + 1e-9).mean()),
        "min_gap_h": float(gaps.min()),
        "max_gap_h": float(gaps.max()),
        "median_gap_h": float(gaps.median()),
        "commonest_gaps_h": {
            str(k): int(v) for k, v in gaps.round(2).value_counts().head(6).items()
        },
    }
