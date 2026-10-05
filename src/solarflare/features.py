"""Window feature extraction.

One SWAN-SF file is one 12-hour observation window: 60 records at a 12-minute
cadence, spanning 11.8 h from the first to the last timestamp. This module turns
such a window into the fixed 144-dimensional vector the models consume, as
``<PARAM>__<stat>`` for each of the 24 SHARP parameters and six statistics.

This is a refactor of ``src/extract_features.py:features`` with identical
behaviour, written as a pure function so it can be unit-tested on synthetic
windows and mirrored in JavaScript for the browser demo.

**Leakage properties, enforced by the tests:**

* only rows *inside* the window are read — there is no access to any later
  record, and no column outside :data:`common.SHARP` is touched;
* the label is never an input: it comes from the SWAN-SF folder name
  (``FL`` / ``NF``), which encodes the largest flare in the 24 h *after* the
  window's last timestamp;
* the time axis is the record index × 0.2 h, so the slope is a per-hour trend
  that depends on position within the window only.
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from . import paths

if str(paths.SRC) not in sys.path:
    sys.path.insert(0, str(paths.SRC))
from common import ALL_COLS, EXCLUDED, SHARP, STATS

__all__ = [
    "CADENCE_HOURS",
    "FILENAME_RE",
    "excluded_columns",
    "feature_frame",
    "parse_window_filename",
    "window_features",
]

#: Hours between consecutive records (12 minutes).
CADENCE_HOURS = 0.2

#: SWAN-SF window file names, e.g. ``X1.3@1234:Primary_ar7115_s2017-09-06T10:00:00_e...csv``.
FILENAME_RE = re.compile(
    r"^(?P<cls>[^@_]+)(?:@(?P<fid>\d+):(?P<role>Primary|Secondary))?"
    r"_ar(?P<ar>\d+)_s(?P<s>[^_]+)_e(?P<e>[^_]+)\.csv$"
)


def excluded_columns() -> list[str]:
    """Return every SWAN-SF column deliberately kept out of the predictors.

    Flattens :data:`common.EXCLUDED`: the flare-derived label columns, the GOES
    X-ray measurements, the position and geometry metadata, and the quality
    flags (which are audited but never used to filter or predict).
    """
    return [column for group in EXCLUDED.values() for column in group]


def window_features(window: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, float]:
    """Summarise one observation window as a 144-vector.

    For each of the 24 SHARP parameters, six statistics are computed over the
    **finite** values of that parameter inside the window:

    ``last``
        the last finite value — the value at the prediction cutoff. If the final
        record is missing, this is the latest value that is not.
    ``mean``, ``min``, ``max``
        over finite values.
    ``std``
        population standard deviation (``ddof=0``).
    ``slope``
        least-squares linear trend per hour, fitted on the finite points only,
        against a time axis of ``index × 0.2`` hours. ``NaN`` unless at least two
        finite points exist and they span more than an instant.

    A parameter that is entirely missing yields ``NaN`` for all six of its
    statistics; the fitted median imputer fills those at predict time.

    Args:
        window: a frame containing the 24 :data:`common.SHARP` columns (extra
            columns are ignored), or an array already ordered as
            ``(n_records, 24)``.

    Returns:
        ``(features, nan_fraction)`` — a 144-vector ordered as
        :data:`common.ALL_COLS`, and the fraction of the ``n_records × 24``
        SHARP cells that were not finite.

    Raises:
        KeyError: if a frame is missing one of the SHARP columns.
        ValueError: if an array does not have exactly 24 columns.
    """
    if isinstance(window, pd.DataFrame):
        values = window[SHARP].to_numpy(dtype=np.float64)
    else:
        values = np.asarray(window, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(SHARP):
            raise ValueError(
                f"expected an array of shape (n_records, {len(SHARP)}), got {values.shape}"
            )

    time = np.arange(values.shape[0], dtype=np.float64) * CADENCE_HOURS
    out = np.full((len(SHARP), len(STATS)), np.nan)
    with np.errstate(all="ignore"):
        for j in range(values.shape[1]):
            column = values[:, j]
            ok = np.isfinite(column)
            if ok.sum() == 0:
                continue
            finite, finite_t = column[ok], time[ok]
            out[j, 0] = finite[-1]
            out[j, 1] = finite.mean()
            out[j, 2] = finite.std()
            out[j, 3] = finite.min()
            out[j, 4] = finite.max()
            if ok.sum() >= 2 and np.ptp(finite_t) > 0:
                out[j, 5] = np.polyfit(finite_t, finite, 1)[0]

    nan_fraction = float(np.mean(~np.isfinite(values))) if values.size else float("nan")
    return out.ravel(), nan_fraction


def feature_frame(window: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """:func:`window_features` as a one-row frame with named columns."""
    vector, _ = window_features(window)
    return pd.DataFrame([vector], columns=ALL_COLS)


def parse_window_filename(name: str) -> dict[str, str | int | None]:
    """Parse a SWAN-SF window file name.

    Returns the flare class, the GOES letter, the flare id and role if present,
    the HARP region number (SWAN-SF's ``ar`` field is the HARP number, not the
    NOAA active region number), and the window's start and end timestamps.

    Raises:
        ValueError: if the name does not match the SWAN-SF pattern.
    """
    match = FILENAME_RE.match(name)
    if match is None:
        raise ValueError(f"not a SWAN-SF window file name: {name!r}")
    cls = match["cls"]
    return {
        "flare_class": cls,
        "goes_letter": cls[0],
        "flare_id": int(match["fid"]) if match["fid"] else None,
        "role": match["role"] or "",
        "harp": int(match["ar"]),
        "start": match["s"],
        "end": match["e"],
    }
