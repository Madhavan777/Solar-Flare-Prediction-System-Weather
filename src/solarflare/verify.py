"""Reproduction checks for the frozen results.

Recomputes every published validation and test number from the saved pipelines
and the feature matrix, and compares against ``results/validation_results.json``,
``results/test_results.json``, ``results/selection.json`` and
``results/data_audit.json``.

This module is the project's main evidence: it demonstrates that the shipped
artefacts and the shipped code still agree, without retraining anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import data, metrics, models, paths

__all__ = [
    "FROZEN_CONFUSION",
    "FROZEN_HEADLINE",
    "Report",
    "verify_all",
    "verify_audit",
    "verify_models",
]

#: The selected model's confusion matrix on the locked test partition, as
#: published. Any deviation means something is broken.
FROZEN_CONFUSION: dict[str, int] = {"tp": 907, "fp": 4707, "fn": 83, "tn": 69668}

#: Headline test-partition figures of the selected model, rounded as published.
FROZEN_HEADLINE: dict[str, float] = {
    "tss": 0.853,
    "pr_auc": 0.489,
    "roc_auc": 0.979,
    "precision": 0.162,
    "recall": 0.916,
    "f1": 0.275,
    "hss2": 0.258,
    "far": 0.838,
    "accuracy": 0.936,
}

#: Largest magnitude standardised coefficients of the selected model, 4 dp.
FROZEN_TOP_COEFFICIENTS: list[tuple[str, float]] = [
    ("R_VALUE__std", -1.5714),
    ("TOTUSJH__max", 1.1227),
    ("TOTUSJH__last", 1.0100),
    ("SHRGT45__min", 1.0098),
    ("R_VALUE__max", 0.8373),
    ("SHRGT45__last", 0.7854),
]

TOL_METRIC = 1e-6
TOL_THRESHOLD = 1e-12
TOL_PROBABILITY = 1e-9


@dataclass
class Report:
    """The outcome of a verification run."""

    checks: int = 0
    failures: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True if every check passed."""
        return not self.failures

    def check(self, condition: bool, message: str) -> bool:
        """Record one check; append ``message`` to failures when it fails."""
        self.checks += 1
        if not condition:
            self.failures.append(message)
        return condition

    def close(self, a: float, b: float, tol: float, message: str) -> bool:
        """Record a numeric closeness check."""
        return self.check(abs(float(a) - float(b)) <= tol, f"{message}: {a!r} vs {b!r}")

    def say(self, line: str) -> None:
        """Record a line of human-readable output."""
        self.lines.append(line)

    def merge(self, other: Report) -> Report:
        """Fold another report into this one."""
        self.checks += other.checks
        self.failures += other.failures
        self.lines += other.lines
        self.details.update(other.details)
        return self

    def summary(self) -> str:
        """Return the full output followed by a verdict line."""
        body = "\n".join(self.lines)
        if self.ok:
            return f"{body}\n\nPASS - {self.checks} checks, no mismatches."
        detail = "\n".join(f"  - {f}" for f in self.failures)
        return f"{body}\n\nFAIL - {len(self.failures)} of {self.checks} checks failed:\n{detail}"


def verify_models(X: pd.DataFrame | None = None, meta: pd.DataFrame | None = None) -> Report:
    """Recompute all seven candidates' validation and test metrics and compare.

    Checks, in order:

    1. every metric in ``val``, ``val_at_0_5`` and ``test`` for all seven
       candidates, to :data:`TOL_METRIC`;
    2. the selection rule, re-applied to the recomputed validation scores,
       re-picks the model named in ``selection.json``;
    3. both thresholds, recomputed from validation probabilities only, match
       ``selection.json`` to :data:`TOL_THRESHOLD`;
    4. the selected model's P5 confusion matrix equals
       :data:`FROZEN_CONFUSION` exactly;
    5. ``results/test_predictions_selected.csv.gz`` has the expected row
       indices, labels and probabilities;
    6. the top-six standardised coefficients match the published list.

    Args:
        X: feature matrix; loaded if omitted.
        meta: window metadata; loaded if omitted.
    """
    report = Report()
    if X is None or meta is None:
        X, meta = data.load()
    train, val, test = data.split_masks(meta)
    y = meta["y"].to_numpy()

    report.say(
        f"X {X.shape}  train={train.sum()} val={val.sum()} test={test.sum()}  "
        f"positives={int(y.sum())}"
    )
    report.check(int(train.sum()) == 204559, f"train size {int(train.sum())} != 204559")
    report.check(int(val.sum()) == 51261, f"validation size {int(val.sum())} != 51261")
    report.check(int(test.sum()) == 75365, f"test size {int(test.sum())} != 75365")

    val_ref = json.loads(paths.VALIDATION_JSON.read_text(encoding="utf-8"))
    test_ref = json.loads(paths.TEST_JSON.read_text(encoding="utf-8"))
    selection = models.load_selection()

    recomputed_val: dict[str, dict[str, Any]] = {}
    predictions_val: dict[str, np.ndarray] = {}

    for key in models.CANDIDATE_KEYS:
        report.check(key in val_ref, f"{key} missing from validation_results.json")
        if key not in val_ref:
            continue
        bundle = models.load_bundle(key)
        thr = float(val_ref[key]["val"]["threshold"])

        p_val = bundle.predict_proba(X.loc[val])
        p_test = bundle.predict_proba(X.loc[test])
        predictions_val[key] = p_val

        got_val = metrics.metrics(y[val], p_val, thr)
        got_val_half = metrics.metrics(y[val], p_val, 0.5)
        got_test = metrics.metrics(y[test], p_test, thr)
        recomputed_val[key] = {"val": got_val, "val_at_0_5": got_val_half}

        for block, got in (("val", got_val), ("val_at_0_5", got_val_half)):
            for name, value in got.items():
                report.close(value, val_ref[key][block][name], TOL_METRIC, f"{key}.{block}.{name}")
        for name, value in got_test.items():
            report.close(value, test_ref[key][name], TOL_METRIC, f"{key}.test.{name}")

        report.say(
            f"{key:22s} val TSS={got_val['tss']:.6f} PR={got_val['pr_auc']:.6f} | "
            f"test TSS={got_test['tss']:.6f} PR={got_test['pr_auc']:.6f} "
            f"P={got_test['precision']:.4f} R={got_test['recall']:.4f}"
        )

    # 2 — the selection rule, re-applied
    picked = max(
        recomputed_val,
        key=lambda k: (
            round(recomputed_val[k]["val"]["tss"], 3),
            recomputed_val[k]["val"]["pr_auc"],
        ),
    )
    report.say(f"\nselection rule re-picks : {picked}")
    report.check(
        picked == selection["selected"],
        f"selection rule picks {picked}, selection.json says {selection['selected']}",
    )

    # 3 — thresholds, from validation only
    alert = metrics.best_threshold(y[val], predictions_val[picked], "tss")
    high = metrics.best_threshold(y[val], predictions_val[picked], "f1")
    report.say(f"alert threshold         : {alert!r}  (stored {selection['alert_threshold']!r})")
    report.say(f"high-risk threshold     : {high!r}  (stored {selection['high_threshold']!r})")
    report.close(alert, selection["alert_threshold"], TOL_THRESHOLD, "alert_threshold")
    report.close(high, selection["high_threshold"], TOL_THRESHOLD, "high_threshold")

    # 4 — the exact confusion matrix
    selected = models.load_bundle(picked)
    p_test_sel = selected.predict_proba(X.loc[test])
    got = metrics.metrics(y[test], p_test_sel, float(selection["alert_threshold"]))
    confusion = {k: got[k] for k in ("tp", "fp", "fn", "tn")}
    report.say(f"P5 confusion matrix     : {confusion}  expected {FROZEN_CONFUSION}")
    report.check(
        confusion == FROZEN_CONFUSION, f"confusion matrix {confusion} != {FROZEN_CONFUSION}"
    )
    for name, want in FROZEN_HEADLINE.items():
        report.close(round(got[name], 3), want, 1e-9, f"headline {name} (3 dp)")

    # 5 — the stored prediction file
    stored = pd.read_csv(paths.TEST_PREDICTIONS)
    report.check(
        np.array_equal(stored["row"].to_numpy(), np.where(test)[0]),
        "test_predictions_selected.csv.gz: row index != np.where(partition == 5)",
    )
    report.check(
        np.array_equal(stored["y"].to_numpy(), y[test]),
        "test_predictions_selected.csv.gz: y column mismatch",
    )
    max_diff = float(np.max(np.abs(stored["p"].to_numpy() - p_test_sel)))
    report.say(f"stored vs recomputed P5 probabilities: max abs diff = {max_diff:.3e}")
    report.check(max_diff <= TOL_PROBABILITY, f"stored P5 probabilities differ by {max_diff:.3e}")

    # 6 — coefficients
    coef = selected.model.named_steps["clf"].coef_.ravel()
    order = np.argsort(np.abs(coef))[::-1][:6]
    top = [(selected.cols[i], round(float(coef[i]), 4)) for i in order]
    report.say(f"top-6 coefficients      : {top}")
    for (got_name, got_value), (want_name, want_value) in zip(
        top, FROZEN_TOP_COEFFICIENTS, strict=True
    ):
        report.check(
            got_name == want_name, f"coefficient order: got {got_name}, expected {want_name}"
        )
        report.close(got_value, want_value, 5e-4, f"coefficient {want_name}")

    report.details["recomputed_val"] = recomputed_val
    report.details["selected"] = picked
    return report


def verify_audit(X: pd.DataFrame | None = None, meta: pd.DataFrame | None = None) -> Report:
    """Recompute the data audit and compare it with ``results/data_audit.json``."""
    report = Report()
    if X is None or meta is None:
        X, meta = data.load()
    recomputed = data.audit(X, meta)
    reference = json.loads(paths.DATA_AUDIT_JSON.read_text(encoding="utf-8"))
    mismatches = data.compare_audit(recomputed, reference)
    report.checks += 1
    if mismatches:
        report.failures.extend(f"data_audit{m}" for m in mismatches)
        report.say(f"data audit: {len(mismatches)} mismatch(es)")
    else:
        report.say(
            "data audit reproduces results/data_audit.json exactly " "(every key; floats to 1e-12)"
        )
    report.details["audit"] = recomputed
    return report


def verify_all() -> Report:
    """Run every reproduction check and return a single merged report."""
    X, meta = data.load()
    report = verify_audit(X, meta)
    report.say("")
    return report.merge(verify_models(X, meta))
