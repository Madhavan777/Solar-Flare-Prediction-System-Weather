"""Shared figure style for the additive figures in ``figures/extra/``.

The palette and dpi match the report's existing figures so that new and old
material sit together, but nothing here writes to ``figures/fig*.png`` — those
are frozen and embedded in the finished report.

Every figure produced through :func:`finish` carries a post-hoc stamp, because
all of it is computed after the test partition was unlocked.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import paths

__all__ = [
    "AMBER",
    "GREEN",
    "GREY",
    "NAVY",
    "POSTHOC_NOTE",
    "RED",
    "apply_style",
    "figure",
    "finish",
]

#: Matching src/make_figures.py so the additive figures are visually consistent.
NAVY = "#1f3a5f"
RED = "#b3312c"
GREY = "#6b7280"
GREEN = "#2e7d32"
AMBER = "#b8860b"

POSTHOC_NOTE = "Post-hoc analysis — not used for model selection"


def apply_style() -> None:
    """Set the rcParams used by every additive figure."""
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": False,
        }
    )


def figure(width: float = 7.0, height: float = 4.6, **kwargs):
    """Create a styled figure sized to read at A4 text width.

    Returns:
        The ``(figure, axes)`` pair from ``plt.subplots``.
    """
    apply_style()
    return plt.subplots(figsize=(width, height), **kwargs)


def finish(fig, name: str, posthoc: bool = True, directory: Path | None = None) -> Path:
    """Stamp, tighten and save a figure into ``figures/extra/``.

    Args:
        fig: the matplotlib figure.
        name: file name stem; ``.png`` is appended.
        posthoc: add the "not used for model selection" footer. Leave this on
            for anything computed on the test partition.
        directory: override the output directory.

    Returns:
        The path written.
    """
    out_dir = directory or paths.FIGURES_EXTRA
    out_dir.mkdir(parents=True, exist_ok=True)
    if posthoc:
        fig.text(0.005, 0.005, POSTHOC_NOTE, fontsize=7, color=GREY, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.02, 1, 1) if posthoc else None)
    path = paths.assert_not_frozen(out_dir / f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    return path
