"""Post-hoc analyses of the frozen model.

Everything here runs **after** the locked test partition was scored and
``results/selection.json`` was written. None of it influenced the choice of
model, features or thresholds, and no published number depends on it. Each
artefact written to ``results/extra/`` carries that statement in its own
``posthoc`` field so it stays attached if the table or figure travels.

Run them all with::

    python -m solarflare analysis

then build the figures with::

    python -m solarflare figures
"""

from __future__ import annotations

import time
import traceback

from .. import paths

__all__ = ["ANALYSES", "run_all", "write_results_md"]

#: name -> (module attribute path, one-line description)
ANALYSES: dict[str, str] = {
    "bootstrap": "95 % confidence intervals by resampling whole active regions",
    "calibration": "reliability of the probabilities, and two validation-only recalibrations",
    "operating": "operating-point table and threshold sweep",
    "breakdowns": "performance by GOES class, year, HARP region and missingness",
    "nearmiss": "what actually followed each false alarm",
    "baselines": "climatology and best-single-feature references",
    "ablation": "point-in-time vs temporal, and statistic-family ablation",
    "errors": "the worst misses and the most confident false alarms",
    "explain": "exactness of the per-window linear explanation",
}


def run_all(resamples: int = 1000, only: str | None = None) -> int:
    """Run the analyses in order and write ``results/extra/RESULTS.md``.

    Args:
        resamples: bootstrap resamples to use.
        only: comma-separated subset of :data:`ANALYSES` keys.

    Returns:
        0 if every analysis succeeded, 1 otherwise.
    """
    from . import (
        ablation,
        baselines,
        bootstrap,
        breakdowns,
        calibration,
        errors,
        explain,
        nearmiss,
        operating,
    )

    runners = {
        "bootstrap": lambda: bootstrap.run(resamples=resamples),
        "calibration": calibration.run,
        "operating": operating.run,
        "breakdowns": breakdowns.run,
        "nearmiss": nearmiss.run,
        "baselines": baselines.run,
        "ablation": ablation.run,
        "errors": errors.run,
        "explain": explain.run,
    }

    wanted = [k.strip() for k in only.split(",")] if only else list(runners)
    unknown = [k for k in wanted if k not in runners]
    if unknown:
        print(f"unknown analyses: {', '.join(unknown)}. " f"Choose from: {', '.join(runners)}")
        return 1

    failures: list[str] = []
    for name in wanted:
        print(f"\n-> {name}: {ANALYSES[name]}", flush=True)
        started = time.time()
        try:
            runners[name]()
            print(f"   done in {time.time() - started:.1f}s " f"-> results/extra/{_filename(name)}")
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"   FAILED after {time.time() - started:.1f}s: " f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

    if not failures:
        write_results_md()

    print()
    if failures:
        print(f"{len(failures)} analysis/analyses failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"all {len(wanted)} analyses complete. " f"Next: python -m solarflare figures")
    return 0


def _filename(name: str) -> str:
    """Map an analysis key to the JSON file it writes."""
    return {
        "operating": "operating_points.json",
        "errors": "error_gallery.json",
        "explain": "explanation_identity.json",
        "nearmiss": "near_miss.json",
    }.get(name, f"{name}.json")


def write_results_md() -> None:
    """Assemble ``results/extra/RESULTS.md`` from the JSON artefacts.

    Deferred import so that a missing analysis cannot break ``run_all``.
    """
    from .report import build

    path = build()
    print(f"\nwrote {paths.relative(path)}")
