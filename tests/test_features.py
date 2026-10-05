"""Unit tests for the window feature extractor, on tiny synthetic windows.

These pin down every edge case the real data contains: constant series, exact
linear trends, scattered missing values, and columns that are missing entirely.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from solarflare import features, paths

sys.path.insert(0, str(paths.SRC))
from common import ALL_COLS, SHARP, STATS

LAST, MEAN, STD, MIN, MAX, SLOPE = range(6)


def make_window(column_values: dict[str, list[float]], n: int = 60) -> pd.DataFrame:
    """Build a window where named SHARP columns take given values, rest are zero."""
    frame = pd.DataFrame({name: np.zeros(n) for name in SHARP})
    for name, values in column_values.items():
        frame[name] = values
    return frame


def stat_of(vector: np.ndarray, parameter: str, statistic: str) -> float:
    """Pull one statistic of one parameter out of the flat 144-vector."""
    return float(vector[ALL_COLS.index(f"{parameter}__{statistic}")])


def test_output_shape_and_order():
    """144 values, ordered parameter-major then statistic."""
    vector, _ = features.window_features(make_window({}))
    assert vector.shape == (144,)
    assert len(ALL_COLS) == 144
    # The flat layout must be parameter-major: index = 6*param + stat.
    assert ALL_COLS[0] == f"{SHARP[0]}__{STATS[0]}"
    assert ALL_COLS[6] == f"{SHARP[1]}__{STATS[0]}"


def test_constant_series():
    """A constant column has zero std, zero slope, and equal last/mean/min/max."""
    vector, nan_fraction = features.window_features(make_window({"TOTUSJH": [7.5] * 60}))
    assert stat_of(vector, "TOTUSJH", "last") == 7.5
    assert stat_of(vector, "TOTUSJH", "mean") == 7.5
    assert stat_of(vector, "TOTUSJH", "min") == 7.5
    assert stat_of(vector, "TOTUSJH", "max") == 7.5
    assert stat_of(vector, "TOTUSJH", "std") == pytest.approx(0.0, abs=1e-12)
    assert stat_of(vector, "TOTUSJH", "slope") == pytest.approx(0.0, abs=1e-9)
    assert nan_fraction == 0.0


def test_exact_linear_trend_gives_exact_slope():
    """A ramp of +3 units per record is +15 units per hour at a 0.2 h cadence."""
    values = [3.0 * i for i in range(60)]
    vector, _ = features.window_features(make_window({"USFLUX": values}))
    assert stat_of(vector, "USFLUX", "slope") == pytest.approx(3.0 / 0.2, rel=1e-9)
    assert stat_of(vector, "USFLUX", "last") == pytest.approx(177.0)
    assert stat_of(vector, "USFLUX", "min") == 0.0
    assert stat_of(vector, "USFLUX", "max") == pytest.approx(177.0)


def test_std_is_population_not_sample():
    """ddof=0, matching numpy's default and the original extractor."""
    values = [1.0, 2.0, 3.0, 4.0]
    vector, _ = features.window_features(make_window({"R_VALUE": values}, n=4))
    assert stat_of(vector, "R_VALUE", "std") == pytest.approx(np.std(values, ddof=0))
    assert stat_of(vector, "R_VALUE", "std") != pytest.approx(np.std(values, ddof=1))


def test_last_is_the_last_finite_value_not_the_last_row():
    """When the final records are missing, 'last' falls back to the latest finite one."""
    values = [1.0] * 58 + [99.0, np.nan]
    vector, _ = features.window_features(make_window({"TOTBSQ": values}))
    assert stat_of(vector, "TOTBSQ", "last") == 99.0


def test_scattered_missing_values_are_skipped_not_zero_filled():
    """Statistics use the finite points only; NaN is not treated as 0."""
    values = [np.nan, 10.0, np.nan, 20.0, np.nan, 30.0]
    vector, nan_fraction = features.window_features(make_window({"MEANPOT": values}, n=6))
    assert stat_of(vector, "MEANPOT", "mean") == pytest.approx(20.0)
    assert stat_of(vector, "MEANPOT", "min") == 10.0
    assert stat_of(vector, "MEANPOT", "max") == 30.0
    assert stat_of(vector, "MEANPOT", "last") == 30.0
    # 3 of 6 x 24 cells are missing.
    assert nan_fraction == pytest.approx(3 / (6 * len(SHARP)))


def test_slope_over_irregular_finite_points_uses_real_times():
    """A gap in the series must not compress the time axis."""
    # Values at indices 0 and 5 only: rise of 10 over 5 records = 1.0 h.
    values = [0.0, np.nan, np.nan, np.nan, np.nan, 10.0]
    vector, _ = features.window_features(make_window({"TOTPOT": values}, n=6))
    assert stat_of(vector, "TOTPOT", "slope") == pytest.approx(10.0, rel=1e-9)


def test_all_nan_column_yields_nan_for_all_six_statistics():
    """An entirely missing parameter produces NaN, left for the imputer."""
    vector, nan_fraction = features.window_features(make_window({"MEANALP": [np.nan] * 60}))
    for statistic in STATS:
        assert np.isnan(stat_of(vector, "MEANALP", statistic))
    assert nan_fraction == pytest.approx(60 / (60 * len(SHARP)))
    # Every other column is unaffected.
    assert stat_of(vector, "TOTUSJH", "mean") == 0.0


def test_single_finite_point_gives_nan_slope_but_real_values():
    """A slope needs two points; the other statistics do not."""
    values = [np.nan] * 59 + [5.0]
    vector, _ = features.window_features(make_window({"EPSZ": values}))
    assert stat_of(vector, "EPSZ", "last") == 5.0
    assert stat_of(vector, "EPSZ", "mean") == 5.0
    assert stat_of(vector, "EPSZ", "std") == pytest.approx(0.0)
    assert np.isnan(stat_of(vector, "EPSZ", "slope"))


def test_negative_values_are_preserved():
    """SHARP force terms are genuinely negative; nothing may clamp them."""
    values = list(np.linspace(-500.0, -100.0, 60))
    vector, _ = features.window_features(make_window({"TOTFZ": values}))
    assert stat_of(vector, "TOTFZ", "min") == pytest.approx(-500.0)
    assert stat_of(vector, "TOTFZ", "max") == pytest.approx(-100.0)
    assert stat_of(vector, "TOTFZ", "slope") > 0


def test_array_input_matches_frame_input():
    """Passing a bare (60, 24) array is equivalent to passing a frame."""
    rng = np.random.default_rng(7)
    values = rng.normal(size=(60, len(SHARP)))
    frame = pd.DataFrame(values, columns=SHARP)
    from_frame, _ = features.window_features(frame)
    from_array, _ = features.window_features(values)
    np.testing.assert_array_equal(from_frame, from_array)


def test_array_with_wrong_width_is_rejected():
    with pytest.raises(ValueError, match="expected an array of shape"):
        features.window_features(np.zeros((60, 7)))


def test_missing_sharp_column_is_rejected():
    frame = pd.DataFrame({name: np.zeros(60) for name in SHARP[:-1]})
    with pytest.raises(KeyError):
        features.window_features(frame)


def test_feature_frame_has_the_canonical_columns():
    frame = features.feature_frame(make_window({}))
    assert list(frame.columns) == ALL_COLS
    assert len(frame) == 1


# --------------------------------------------------------------------------- #
# file name parsing
# --------------------------------------------------------------------------- #
def test_parse_flare_window_filename():
    parsed = features.parse_window_filename(
        "X1.3@1234:Primary_ar7115_s2017-09-06T10:00:00_e2017-09-06T21:48:00.csv"
    )
    assert parsed["flare_class"] == "X1.3"
    assert parsed["goes_letter"] == "X"
    assert parsed["flare_id"] == 1234
    assert parsed["role"] == "Primary"
    assert parsed["harp"] == 7115
    assert parsed["start"] == "2017-09-06T10:00:00"
    assert parsed["end"] == "2017-09-06T21:48:00"


def test_parse_flare_quiet_window_filename():
    parsed = features.parse_window_filename(
        "FQ_ar5413_s2015-04-02T09:24:00_e2015-04-02T21:12:00.csv"
    )
    assert parsed["flare_class"] == "FQ"
    assert parsed["flare_id"] is None
    assert parsed["role"] == ""
    assert parsed["harp"] == 5413


def test_parse_rejects_a_non_swansf_name():
    with pytest.raises(ValueError, match="not a SWAN-SF window file name"):
        features.parse_window_filename("something_else.csv")


def test_cadence_matches_the_documented_window():
    """60 records at 0.2 h spans 11.8 h from the first to the last timestamp."""
    assert features.CADENCE_HOURS == 0.2
    assert pytest.approx(11.8) == (60 - 1) * features.CADENCE_HOURS
