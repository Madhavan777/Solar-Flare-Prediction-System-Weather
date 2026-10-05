"""Tests for the metric definitions.

Written against hand-computed confusion matrices so the definitions themselves
are pinned, not just their agreement with sklearn.
"""

from __future__ import annotations

import numpy as np
import pytest

from solarflare import metrics


@pytest.fixture
def toy():
    """A 10-window problem: 4 positives, 6 negatives, threshold 0.5.

    At p >= 0.5:  TP = 3, FN = 1, FP = 2, TN = 4.
    """
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.6, 0.4, 0.7, 0.55, 0.3, 0.2, 0.1, 0.05])
    return y, p


def test_counts_are_as_expected(toy):
    y, p = toy
    tn, fp, fn, tp = metrics.counts(y, (p >= 0.5).astype(int))
    assert (tp, fp, fn, tn) == (3, 2, 1, 4)


def test_core_metrics_against_hand_computation(toy):
    y, p = toy
    got = metrics.metrics(y, p, 0.5)
    assert (got["tp"], got["fp"], got["fn"], got["tn"]) == (3, 2, 1, 4)
    assert got["precision"] == pytest.approx(3 / 5)
    assert got["recall"] == pytest.approx(3 / 4)
    assert got["f1"] == pytest.approx(2 * 3 / (2 * 3 + 2 + 1))
    assert got["accuracy"] == pytest.approx(7 / 10)
    # TSS = recall - FPR = 0.75 - 2/6
    assert got["tss"] == pytest.approx(0.75 - 2 / 6)
    # FAR is the ratio FP/(TP+FP) = 2/5, i.e. 1 - precision
    assert got["far"] == pytest.approx(2 / 5)
    assert got["far"] == pytest.approx(1 - got["precision"])


def test_extended_adds_the_false_alarm_rate_distinctly(toy):
    y, p = toy
    got = metrics.extended(y, p, 0.5)
    assert got["fpr"] == pytest.approx(2 / 6)
    assert got["far"] == pytest.approx(2 / 5)
    assert got["fpr"] != pytest.approx(got["far"])
    assert got["base_rate"] == pytest.approx(0.4)
    assert got["alert_rate"] == pytest.approx(0.5)
    assert got["alert_rate_ratio"] == pytest.approx(0.5 / 0.4)


def test_tss_is_zero_for_a_constant_forecast():
    """Any forecast that alerts on everything, or nothing, has zero skill."""
    y = np.array([1] * 10 + [0] * 990)
    always = np.ones(1000)
    never = np.zeros(1000)
    assert metrics.metrics(y, always, 0.5)["tss"] == pytest.approx(0.0)
    assert metrics.metrics(y, never, 0.5)["tss"] == pytest.approx(0.0)


def test_tss_is_one_for_a_perfect_forecast():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.1, 0.2])
    assert metrics.metrics(y, p, 0.5)["tss"] == pytest.approx(1.0)


def test_tss_is_insensitive_to_the_class_ratio():
    """The property that makes TSS the headline metric under 1.3 % positives."""
    rng = np.random.default_rng(0)
    recall, fpr = 0.9, 0.1

    def build(n_pos: int, n_neg: int):
        y = np.r_[np.ones(n_pos), np.zeros(n_neg)]
        p = np.r_[
            np.where(rng.random(n_pos) < recall, 0.9, 0.1),
            np.where(rng.random(n_neg) < fpr, 0.9, 0.1),
        ]
        return y, p

    balanced = metrics.metrics(*build(20000, 20000), 0.5)
    skewed = metrics.metrics(*build(500, 40000), 0.5)
    assert balanced["tss"] == pytest.approx(skewed["tss"], abs=0.02)
    # Precision, by contrast, collapses.
    assert balanced["precision"] - skewed["precision"] > 0.4


def test_hss2_formula(toy):
    """2(TP·TN - FN·FP) / [(TP+FN)(FN+TN) + (TP+FP)(FP+TN)]."""
    y, p = toy
    tp, fp, fn, tn = 3, 2, 1, 4
    expected = 2 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    assert metrics.metrics(y, p, 0.5)["hss2"] == pytest.approx(expected)


def test_pr_auc_of_a_random_forecast_approaches_the_base_rate():
    rng = np.random.default_rng(3)
    y = rng.random(50000) < 0.0131
    p = rng.random(50000)
    got = metrics.metrics(y.astype(int), p, 0.5)
    assert got["pr_auc"] == pytest.approx(0.0131, abs=0.004)
    assert got["roc_auc"] == pytest.approx(0.5, abs=0.02)


def test_best_threshold_maximises_tss_on_a_separable_problem():
    y = np.array([0] * 50 + [1] * 50)
    p = np.r_[np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)]
    threshold = metrics.best_threshold(y, p, "tss")
    got = metrics.metrics(y, p, threshold)
    assert got["tss"] == pytest.approx(1.0)


def test_best_threshold_rejects_an_unknown_criterion():
    with pytest.raises(ValueError, match="must be 'tss' or 'f1'"):
        metrics.best_threshold(np.array([0, 1]), np.array([0.1, 0.9]), "accuracy")


def test_best_threshold_tss_and_f1_disagree_under_imbalance():
    """Why there are two thresholds: F1 wants precision, TSS does not."""
    rng = np.random.default_rng(11)
    y = (rng.random(20000) < 0.02).astype(int)
    p = np.clip(0.08 * y + rng.random(20000) * 0.9, 0, 1)
    tss_threshold = metrics.best_threshold(y, p, "tss")
    f1_threshold = metrics.best_threshold(y, p, "f1")
    assert f1_threshold > tss_threshold


def test_glossary_covers_every_reported_metric(toy):
    """Nothing can be reported without a definition on record."""
    y, p = toy
    for name in metrics.extended(y, p, 0.5):
        if name in ("threshold", "tp", "fp", "fn", "tn", "n", "positives"):
            continue
        assert name in metrics.GLOSSARY, f"{name} has no glossary entry"
