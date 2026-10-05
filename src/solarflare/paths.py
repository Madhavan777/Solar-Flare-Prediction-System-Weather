"""Project paths.

Every path in this project is derived from one anchor so that nothing is
hard-coded and the repository can live anywhere, including under a directory
whose name contains spaces.

The anchor is the file ``models/I2_LR_temporal_C0.01.joblib`` — the selected
final model, which must exist in any usable checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The file that identifies a project root.
ANCHOR = Path("models") / "I2_LR_temporal_C0.01.joblib"


def find_root(start: Path | str | None = None) -> Path:
    """Return the project root directory.

    Resolution order:

    1. the ``SOLARFLARE_ROOT`` environment variable, if set;
    2. if ``start`` is given, the nearest ancestor of ``start`` — and nothing
       else, so an explicit search cannot silently succeed somewhere the caller
       did not ask about;
    3. otherwise, the nearest ancestor of this file, then of the current
       working directory.

    Raises:
        FileNotFoundError: if no candidate directory contains the anchor.
    """
    env = os.environ.get("SOLARFLARE_ROOT")
    if env:
        candidate = Path(env).expanduser().resolve()
        if (candidate / ANCHOR).is_file():
            return candidate
        raise FileNotFoundError(f"SOLARFLARE_ROOT={candidate} does not contain {ANCHOR.as_posix()}")

    bases = [Path(start).resolve()] if start is not None else [Path(__file__).resolve(), Path.cwd()]

    for base in bases:
        for directory in (base, *base.parents):
            if (directory / ANCHOR).is_file():
                return directory

    searched = " or ".join(str(b) for b in bases)
    raise FileNotFoundError(
        f"could not locate the project root: no ancestor of {searched} contains "
        f"{ANCHOR.as_posix()}. Set SOLARFLARE_ROOT to the project directory."
    )


ROOT = find_root()

# --- frozen inputs and outputs: read, never written by this package ----------
SRC = ROOT / "src"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RAW_DATA = ROOT / "raw_data"

# --- data ---------------------------------------------------------------------
DATA = ROOT / "data"
FEATURE_CHUNKS = DATA / "features"
ALL_X = DATA / "all_X.npz"
ALL_META = DATA / "all_meta.csv.gz"

# --- frozen result files ------------------------------------------------------
SELECTION_JSON = RESULTS / "selection.json"
VALIDATION_JSON = RESULTS / "validation_results.json"
TEST_JSON = RESULTS / "test_results.json"
DATA_AUDIT_JSON = RESULTS / "data_audit.json"
TRAIN_LOG = RESULTS / "train_log.txt"
VAL_PREDICTIONS = RESULTS / "val_predictions.csv.gz"
TEST_PREDICTIONS = RESULTS / "test_predictions_selected.csv.gz"
TEST_PREDICTIONS_I1 = RESULTS / "test_predictions_I1.csv.gz"
DASHBOARD_DEMO_JSON = RESULTS / "dashboard_demo.json"

# --- additive outputs: everything this package writes goes under here --------
RESULTS_EXTRA = RESULTS / "extra"
FIGURES_EXTRA = FIGURES / "extra"

# --- dashboard ----------------------------------------------------------------
DASHBOARD = ROOT / "dashboard"
DASHBOARD_DEMO = DASHBOARD / "demo.json"
DASHBOARD_MODEL_LR = DASHBOARD / "model_lr.json"
DASHBOARD_WINDOWS = DASHBOARD / "windows.json"
DASHBOARD_OPERATING = DASHBOARD / "operating.json"

# --- documentation ------------------------------------------------------------
DOCS = ROOT / "docs"
SCREENSHOTS = DOCS / "screenshots"

#: Paths that must never be modified. Guarded by :func:`assert_not_frozen`.
FROZEN: tuple[Path, ...] = (
    SELECTION_JSON,
    VALIDATION_JSON,
    TEST_JSON,
    DATA_AUDIT_JSON,
    TRAIN_LOG,
    VAL_PREDICTIONS,
    TEST_PREDICTIONS,
    TEST_PREDICTIONS_I1,
    DASHBOARD_DEMO_JSON,
    *sorted(MODELS.glob("*.joblib")),
    *sorted(FIGURES.glob("fig*.png")),
)


def assert_not_frozen(path: Path) -> Path:
    """Return ``path``, or raise if it is one of the frozen artefacts.

    Call this immediately before opening any file for writing. The frozen set
    covers the seven trained pipelines, every file in ``results/``, and the
    report's figures in ``figures/``.

    Raises:
        PermissionError: if ``path`` resolves to a frozen artefact.
    """
    resolved = path.resolve()
    for frozen in FROZEN:
        if resolved == frozen.resolve():
            raise PermissionError(
                f"refusing to write {path.name}: it is a frozen artefact. "
                f"Additive output belongs in results/extra/ or figures/extra/."
            )
    return path


def ensure_output_dirs() -> None:
    """Create the additive output directories if they do not exist."""
    for directory in (RESULTS_EXTRA, FIGURES_EXTRA, SCREENSHOTS):
        directory.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    """Format ``path`` relative to the project root, with forward slashes."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()
