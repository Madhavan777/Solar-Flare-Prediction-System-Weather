"""Regression tests: the published numbers must not move.

If any of these fail, something that was supposed to be frozen has changed.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from solarflare import metrics, models, paths, verify


def test_selection_json_is_unchanged():
    """The selected model, the rule and both thresholds, to full precision."""
    selection = json.loads(paths.SELECTION_JSON.read_text(encoding="utf-8"))
    assert selection["selected"] == "I2_LR_temporal_C0.01"
    assert selection["rule"] == "max validation TSS (tie-break: validation PR-AUC)"
    assert selection["alert_threshold"] == pytest.approx(0.544546643935129, abs=1e-15)
    assert selection["high_threshold"] == pytest.approx(0.9672703856605697, abs=1e-15)


def test_all_seven_candidates_are_present():
    for key in models.CANDIDATE_KEYS:
        assert (paths.MODELS / f"{key}.joblib").is_file(), f"{key}.joblib is missing"
    stored = json.loads(paths.VALIDATION_JSON.read_text(encoding="utf-8"))
    assert set(stored) == set(models.CANDIDATE_KEYS)
    stored_test = json.loads(paths.TEST_JSON.read_text(encoding="utf-8"))
    assert set(stored_test) == set(models.CANDIDATE_KEYS)


def test_frozen_test_results_match_the_published_table():
    """Every candidate's test-partition figures, rounded as published."""
    expected = {
        "I1_LR_last": (0.867, 0.462, 0.126, 0.955, 0.223, 0.980),
        "I2_LR_temporal_C0.01": (0.853, 0.489, 0.162, 0.916, 0.275, 0.979),
        "I2_LR_temporal_C0.1": (0.861, 0.479, 0.149, 0.931, 0.257, 0.978),
        "I2_LR_temporal_C1": (0.863, 0.472, 0.154, 0.931, 0.264, 0.978),
        "I2_RF_leaf5": (0.863, 0.352, 0.122, 0.954, 0.217, 0.971),
        "I2_RF_leaf20": (0.860, 0.386, 0.114, 0.960, 0.204, 0.978),
        "I2_RF_leaf50_d16": (0.865, 0.399, 0.123, 0.957, 0.217, 0.980),
    }
    stored = json.loads(paths.TEST_JSON.read_text(encoding="utf-8"))
    for key, (tss, pr, precision, recall, f1, roc) in expected.items():
        row = stored[key]
        assert round(row["tss"], 3) == tss, key
        assert round(row["pr_auc"], 3) == pr, key
        assert round(row["precision"], 3) == precision, key
        assert round(row["recall"], 3) == recall, key
        assert round(row["f1"], 3) == f1, key
        assert round(row["roc_auc"], 3) == roc, key
        assert row["n"] == 75365, key
        assert row["positives"] == 990, key


def test_frozen_validation_tss_matches_the_published_table():
    expected = {
        "I1_LR_last": (0.846, 0.482),
        "I2_LR_temporal_C0.01": (0.862, 0.500),
        "I2_LR_temporal_C0.1": (0.860, 0.484),
        "I2_LR_temporal_C1": (0.853, 0.465),
        "I2_RF_leaf5": (0.854, 0.233),
        "I2_RF_leaf20": (0.852, 0.368),
        "I2_RF_leaf50_d16": (0.848, 0.471),
    }
    stored = json.loads(paths.VALIDATION_JSON.read_text(encoding="utf-8"))
    for key, (tss, pr) in expected.items():
        assert round(stored[key]["val"]["tss"], 3) == tss, key
        assert round(stored[key]["val"]["pr_auc"], 3) == pr, key


def test_selection_rule_applied_to_frozen_validation_picks_the_selected_model():
    """Max validation TSS at 3 dp, tie-break validation PR-AUC."""
    stored = json.loads(paths.VALIDATION_JSON.read_text(encoding="utf-8"))
    picked = max(
        stored, key=lambda k: (round(stored[k]["val"]["tss"], 3), stored[k]["val"]["pr_auc"])
    )
    assert picked == "I2_LR_temporal_C0.01"


def test_selected_model_is_not_the_best_on_test():
    """Documented honestly: several candidates score higher test TSS.

    This is the expected consequence of selecting on validation only. If it ever
    stopped being true, the project's central claim would need re-examining.
    """
    stored = json.loads(paths.TEST_JSON.read_text(encoding="utf-8"))
    selected = stored["I2_LR_temporal_C0.01"]["tss"]
    better = [k for k, v in stored.items() if v["tss"] > selected]
    assert better, "no candidate beats the selected model on test TSS any more"
    assert "I2_LR_temporal_C0.01" not in better


def test_stored_predictions_reproduce_the_frozen_confusion_matrix():
    """From results/ alone, with no model loading and no feature matrix."""
    stored = pd.read_csv(paths.TEST_PREDICTIONS)
    alert, _ = models.thresholds()
    got = metrics.metrics(stored["y"].to_numpy(), stored["p"].to_numpy(), alert)
    assert {k: got[k] for k in ("tp", "fp", "fn", "tn")} == verify.FROZEN_CONFUSION
    for name, want in verify.FROZEN_HEADLINE.items():
        assert round(got[name], 3) == want, name


def test_test_partition_shape_and_base_rate():
    stored = pd.read_csv(paths.TEST_PREDICTIONS)
    assert len(stored) == 75365
    assert int(stored["y"].sum()) == 990
    assert round(float(stored["y"].mean()), 4) == 0.0131


def test_always_no_flare_scores_high_accuracy_and_detects_nothing():
    """The sentence that must accompany every accuracy figure, as a test."""
    stored = pd.read_csv(paths.TEST_PREDICTIONS)
    y = stored["y"].to_numpy()
    never = np.zeros_like(y)
    accuracy = float((never == y).mean())
    assert round(accuracy * 100, 2) == 98.69
    assert never.sum() == 0


def test_alert_counts_match_the_published_figures():
    """5,614 alerts on 75,365 windows: 7.45 %, about 5.7x the base rate."""
    stored = pd.read_csv(paths.TEST_PREDICTIONS)
    alert, _ = models.thresholds()
    alerts = int((stored["p"] >= alert).sum())
    assert alerts == 5614
    assert round(alerts / len(stored) * 100, 2) == 7.45
    base = float(stored["y"].mean())
    assert round((alerts / len(stored)) / base, 1) == 5.7


def test_false_alarm_ratio_and_rate_are_different_numbers():
    """FAR = FP/(TP+FP) = 0.838; FPR = FP/(FP+TN) = 0.063. Never interchangeable."""
    stored = pd.read_csv(paths.TEST_PREDICTIONS)
    alert, _ = models.thresholds()
    got = metrics.extended(stored["y"].to_numpy(), stored["p"].to_numpy(), alert)
    assert round(got["far"], 3) == 0.838
    assert round(got["fpr"], 4) == 0.0633
    assert got["far"] / got["fpr"] > 10


@pytest.mark.largedata
@pytest.mark.models
@pytest.mark.slow
def test_full_reproduction_passes(matrix):
    """The whole verification suite, end to end."""
    X, meta = matrix
    report = verify.verify_audit(X, meta).merge(verify.verify_models(X, meta))
    assert report.ok, "\n".join(report.failures)
    assert report.checks > 300
