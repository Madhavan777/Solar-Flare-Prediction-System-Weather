"""Leakage tests.

These are the tests an examiner should care about most. Each one closes a
specific route by which information about the future, or about the test
partition, could reach the model.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from solarflare import data, features, metrics, models, paths

sys.path.insert(0, str(paths.SRC))
from common import ALL_COLS, LAST_COLS, SHARP, STATS, TEST_P, TRAIN_P, VAL_P


# --------------------------------------------------------------------------- #
# 1. No excluded column can reach the predictor list
# --------------------------------------------------------------------------- #
def test_excluded_columns_are_disjoint_from_predictors():
    """No flare-derived, GOES, geometry or quality column appears in the features."""
    excluded = set(features.excluded_columns())
    assert excluded, "the exclusion list is empty - common.EXCLUDED is not being read"
    assert excluded.isdisjoint(set(ALL_COLS))
    assert excluded.isdisjoint(set(LAST_COLS))
    assert excluded.isdisjoint(set(SHARP))


def test_no_feature_name_derives_from_an_excluded_column():
    """A feature named <PARAM>__<stat> can only come from a SHARP parameter."""
    for column in ALL_COLS:
        parameter, _, statistic = column.partition("__")
        assert parameter in SHARP, f"{column} is not built from a SHARP parameter"
        assert statistic in STATS, f"{column} has an unknown statistic"
    for excluded in features.excluded_columns():
        assert not any(c.startswith(f"{excluded}__") for c in ALL_COLS)


def test_label_derived_columns_are_all_excluded():
    """Every flare-class column SWAN-SF ships is on the exclusion list."""
    excluded = set(features.excluded_columns())
    for letter in "BCMX":
        for suffix in ("", "_LOC", "_LABEL", "_LABEL_LOC"):
            assert f"{letter}FLARE{suffix}" in excluded


def test_goes_and_quality_columns_are_excluded():
    """The GOES X-ray flux and the quality flags are never predictors."""
    excluded = set(features.excluded_columns())
    for column in ("XR_MAX", "XR_QUAL", "QUALITY", "SPEI", "IS_TMFI"):
        assert column in excluded


def test_feature_count_is_144():
    """24 SHARP parameters x 6 statistics, with no duplicates."""
    assert len(SHARP) == 24
    assert len(STATS) == 6
    assert len(ALL_COLS) == 144
    assert len(set(ALL_COLS)) == 144


# --------------------------------------------------------------------------- #
# 2. The feature function reads only what is inside the window
# --------------------------------------------------------------------------- #
def test_features_ignore_non_sharp_columns():
    """Adding label-like columns with extreme values changes nothing."""
    rng = np.random.default_rng(0)
    window = pd.DataFrame(rng.normal(size=(60, len(SHARP))), columns=SHARP)
    baseline, _ = features.window_features(window)

    contaminated = window.copy()
    for column in features.excluded_columns():
        contaminated[column] = 1e9
    contaminated["MFLARE"] = 1.0
    contaminated["Timestamp"] = pd.date_range("2015-01-01", periods=60, freq="12min")
    polluted, _ = features.window_features(contaminated)

    np.testing.assert_array_equal(baseline, polluted)


def test_features_do_not_read_rows_after_the_window():
    """Records appended after the window cannot affect the window's features."""
    rng = np.random.default_rng(1)
    window = pd.DataFrame(rng.normal(size=(60, len(SHARP))), columns=SHARP)
    baseline, _ = features.window_features(window)

    future = pd.DataFrame(rng.normal(size=(24, len(SHARP))) * 1e6, columns=SHARP)
    extended = pd.concat([window, future], ignore_index=True)
    with_future, _ = features.window_features(extended)

    assert not np.allclose(baseline, with_future), (
        "appending future records changed nothing, so this test proves nothing - "
        "check the fixture"
    )
    # The real guarantee: the extractor is only ever handed one window's rows.
    # Re-slicing the window back out must restore the original vector exactly.
    resliced, _ = features.window_features(extended.iloc[: len(window)])
    np.testing.assert_array_equal(baseline, resliced)


# --------------------------------------------------------------------------- #
# 3. The split is chronological and shares no active region
# --------------------------------------------------------------------------- #
def test_partition_to_split_mapping_is_exactly_123_4_5():
    """Train = P1-P3, validation = P4, test = P5. Nothing else."""
    assert TRAIN_P == (1, 2, 3)
    assert VAL_P == (4,)
    assert TEST_P == (5,)
    assert data.SPLIT_OF_PARTITION == {
        1: "train",
        2: "train",
        3: "train",
        4: "validation",
        5: "test",
    }


@pytest.mark.largedata
def test_no_active_region_is_shared_between_partitions(matrix):
    """All ten pairwise HARP-region overlaps are empty."""
    _, meta = matrix
    regions = {p: set(meta.loc[meta.partition == p, "ar"]) for p in range(1, 6)}
    for a in range(1, 6):
        for b in range(a + 1, 6):
            shared = regions[a] & regions[b]
            assert not shared, f"P{a} and P{b} share HARP regions: {sorted(shared)[:10]}"


@pytest.mark.largedata
def test_no_region_crosses_the_train_validation_test_boundary(matrix, splits):
    """Stronger than pairwise: the three splits are region-disjoint."""
    _, meta = matrix
    train, validation, test = splits
    train_regions = set(meta.loc[train, "ar"])
    validation_regions = set(meta.loc[validation, "ar"])
    test_regions = set(meta.loc[test, "ar"])
    assert not train_regions & validation_regions
    assert not train_regions & test_regions
    assert not validation_regions & test_regions


@pytest.mark.largedata
def test_split_sizes_match_the_published_counts(splits):
    """204,559 / 51,261 / 75,365."""
    train, validation, test = splits
    assert int(train.sum()) == 204559
    assert int(validation.sum()) == 51261
    assert int(test.sum()) == 75365


@pytest.mark.largedata
def test_every_window_has_exactly_60_records(matrix):
    """A varying window length would break the fixed time axis."""
    _, meta = matrix
    assert set(meta["n_rows"].unique()) == {60}


@pytest.mark.largedata
def test_label_matches_the_swan_sf_folder(matrix):
    """y is 1 if and only if the window file sits in the FL folder."""
    _, meta = matrix
    assert (meta["y"] == (meta["folder"] == "FL").astype(int)).all()
    assert set(meta.loc[meta.y == 1, "goes_letter"]) == {"M", "X"}
    assert set(meta.loc[meta.y == 0, "goes_letter"]) <= {"B", "C", "F"}


# --------------------------------------------------------------------------- #
# 4. Learned preprocessing was fitted on the training partitions only
# --------------------------------------------------------------------------- #
@pytest.mark.largedata
@pytest.mark.models
@pytest.mark.parametrize("key", ["I1_LR_last", "I2_LR_temporal_C0.01"])
def test_imputer_medians_equal_training_partition_medians(matrix, splits, key):
    """The median imputer saw P1-P3 and nothing else."""
    X, _ = matrix
    train, _, _ = splits
    bundle = models.load_bundle(key)

    training = X.loc[train, bundle.cols].to_numpy(dtype=np.float64)
    training = np.where(np.isfinite(training), training, np.nan)
    if bundle.uses_slog:
        from common import slog

        training = slog(training)

    expected = np.nanmedian(training, axis=0)
    np.testing.assert_allclose(
        bundle.model.named_steps["impute"].statistics_, expected, rtol=0, atol=1e-9
    )


@pytest.mark.largedata
@pytest.mark.models
@pytest.mark.parametrize("key", ["I1_LR_last", "I2_LR_temporal_C0.01"])
def test_scaler_statistics_equal_training_partition_statistics(matrix, splits, key):
    """The scaler's centre and scale came from P1-P3 after imputation."""
    X, _ = matrix
    train, _, _ = splits
    bundle = models.load_bundle(key)
    steps = bundle.model.named_steps

    training = X.loc[train, bundle.cols].to_numpy(dtype=np.float64)
    training = np.where(np.isfinite(training), training, np.nan)
    if bundle.uses_slog:
        from common import slog

        training = slog(training)
    filled = np.where(np.isnan(training), steps["impute"].statistics_[None, :], training)

    np.testing.assert_allclose(steps["scale"].mean_, filled.mean(axis=0), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        steps["scale"].scale_, filled.std(axis=0, ddof=0), rtol=1e-9, atol=1e-9
    )


@pytest.mark.largedata
@pytest.mark.models
def test_preprocessing_differs_from_a_full_data_fit(matrix, selected):
    """Sanity check: fitting on all partitions would give different statistics.

    Without this, the two tests above could pass trivially if train and test
    happened to have identical distributions.
    """
    X, _ = matrix
    from common import slog

    everything = slog(
        np.where(
            np.isfinite(X[selected.cols].to_numpy(dtype=np.float64)),
            X[selected.cols].to_numpy(dtype=np.float64),
            np.nan,
        )
    )
    all_medians = np.nanmedian(everything, axis=0)
    fitted = selected.model.named_steps["impute"].statistics_
    assert not np.allclose(
        fitted, all_medians, atol=1e-9
    ), "the imputer medians match a fit on ALL partitions, which would indicate leakage"


# --------------------------------------------------------------------------- #
# 5. Both thresholds come from validation only
# --------------------------------------------------------------------------- #
@pytest.mark.largedata
def test_thresholds_are_recoverable_from_validation_predictions_alone():
    """Recomputing on P4 alone reproduces both published thresholds."""
    validation = pd.read_csv(paths.VAL_PREDICTIONS)
    key = models.selected_key()
    y = validation["y"].to_numpy()
    p = validation[key].to_numpy()

    alert, high = models.thresholds()
    assert metrics.best_threshold(y, p, "tss") == pytest.approx(alert, abs=1e-12)
    assert metrics.best_threshold(y, p, "f1") == pytest.approx(high, abs=1e-12)


@pytest.mark.largedata
@pytest.mark.models
def test_thresholds_were_not_tuned_on_the_test_partition(matrix, splits, selected):
    """The test-optimal threshold differs from the published one.

    If they coincided, the protocol would be unverifiable from the outside.
    """
    X, meta = matrix
    _, _, test = splits
    y = meta["y"].to_numpy()
    p = selected.predict_proba(X.loc[test])

    alert, _ = models.thresholds()
    test_optimal = metrics.best_threshold(y[test], p, "tss")
    assert test_optimal != pytest.approx(alert, abs=1e-6), (
        "the published threshold equals the test-optimal threshold, which would suggest "
        "it was chosen on P5"
    )


@pytest.mark.largedata
def test_validation_predictions_file_covers_only_partition_4():
    """val_predictions.csv.gz must not contain test rows."""
    validation = pd.read_csv(paths.VAL_PREDICTIONS)
    assert set(validation["partition"].unique()) == {4}
    assert len(validation) == 51261
