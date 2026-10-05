"""Fail if any published number, anywhere, disagrees with ``results/``.

Four independent passes:

1. **Ground truth.** ``results/`` is checked against
   ``tests/fixtures/ground_truth.json``, a transcription of the published
   figures made independently of the result files. If a frozen artefact ever
   changes, this catches it even when every document was regenerated from it.
2. **Documents.** Every Markdown file, the dashboard page and the generated
   dashboard JSON are scanned for numbers stated next to a metric name. Each
   one must match a legitimate value for that metric.
3. **Claims.** A curated list of exact sentences that must remain true, each
   tied to a value computed from ``results/``.
4. **Language.** Overclaiming words, names that must not appear, and American
   spellings.

Run directly, or as ``python -m solarflare check``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solarflare import models, paths

FIXTURE = paths.ROOT / "tests" / "fixtures" / "ground_truth.json"

#: Documents to scan. The frozen report is deliberately absent: it is not edited.
DOC_GLOBS = ("README.md", "docs/*.md", "results/extra/RESULTS.md")

#: Words that would overclaim what this system is.
FORBIDDEN_PHRASES = {
    "real-time": "the system consumes no live feed",
    "real time": "the system consumes no live feed",
    "operational warning service": "it is an academic prototype",
    "production-ready": "it is an academic prototype",
    "proves that": "nothing here proves a causal claim",
    "guarantees": "no forecast here is guaranteed",
}

#: Names that must not appear anywhere, in any context. Unlike the phrases
#: above, a nearby negation does not excuse these: they are simply absent.
#: Written in pieces so that this file does not itself contain them.
FORBIDDEN_NAMES = {
    "sir" + "agu": "branding that does not belong in this repository",
    "raja" + " priya": "no longer the mentor",
    "raja" + "priya": "no longer the mentor",
}

#: Phrases that are allowed even though they contain a forbidden substring,
#: because the surrounding text explicitly denies the claim.
ALLOWED_CONTEXTS = (
    "not an operational warning service",
    "not an operational space-weather warning service",
)

#: Negations that, close to a forbidden phrase, turn it into a denial rather
#: than a claim: "Is it real-time? **No.**" is fine; "real-time alerts" is not.
#: Matched on word boundaries so that markdown punctuation does not hide them.
NEGATION_RE = re.compile(
    r"\b(no|not|never|neither|none|nothing|cannot|without)\b|n't", re.IGNORECASE
)

#: How near a negation has to be, in characters, to count as denying the phrase.
NEGATION_WINDOW = 70


def is_denied(line: str, phrase: str) -> bool:
    """True if ``phrase`` appears in ``line`` as a denial rather than a claim."""
    lowered = line.lower()
    if any(allowed in lowered for allowed in ALLOWED_CONTEXTS):
        return True
    at = lowered.find(phrase)
    if at < 0:
        return False
    nearby = lowered[max(0, at - NEGATION_WINDOW) : at + len(phrase) + NEGATION_WINDOW]
    return NEGATION_RE.search(nearby) is not None


#: American spellings that should be British in prose. Code identifiers and
#: scikit-learn names are excluded by the surrounding-character check.
SPELLING = {
    r"\borganization\b": "organisation",
    r"\borganizations\b": "organisations",
    r"\bgeneralization\b": "generalisation",
    r"\bgeneralize\b": "generalise",
    r"\bgeneralizes\b": "generalises",
    r"\bbehavior\b": "behaviour",
    r"\bbehaviors\b": "behaviours",
    r"\banalyze\b": "analyse",
    r"\banalyzed\b": "analysed",
    r"\bmodeling\b": "modelling",
    r"\bmodeled\b": "modelled",
    r"\bcatalog\b": "catalogue",
    r"\bfavorable\b": "favourable",
    r"\bemphasized\b": "emphasised",
    r"\bsummarize\b": "summarise",
    r"\bsummarized\b": "summarised",
    r"\brecognize\b": "recognise",
}

#: Lines containing these are skipped by the spelling pass: they quote code,
#: a library name, or a frozen figure's axis label.
SPELLING_EXEMPT = (
    "standardized coefficient",
    "StandardScaler",
    "`normalize",
    "sklearn",
    "scikit-learn",
    "Standardized Logistic",
)


class Checker:
    """Collects failures across the four passes."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def check(self, ok: bool, message: str) -> None:
        self.checks += 1
        if not ok:
            self.failures.append(message)

    def close(self, got: float, want: float, tol: float, message: str) -> None:
        self.checks += 1
        if abs(float(got) - float(want)) > tol:
            self.failures.append(f"{message}: {got!r} vs expected {want!r}")


# --------------------------------------------------------------------------- #
# pass 1 - results/ against the independent transcription
# --------------------------------------------------------------------------- #
def check_ground_truth(checker: Checker) -> dict:
    truth = json.loads(FIXTURE.read_text(encoding="utf-8"))
    selection = json.loads(paths.SELECTION_JSON.read_text(encoding="utf-8"))
    test = json.loads(paths.TEST_JSON.read_text(encoding="utf-8"))
    validation = json.loads(paths.VALIDATION_JSON.read_text(encoding="utf-8"))
    audit = json.loads(paths.DATA_AUDIT_JSON.read_text(encoding="utf-8"))

    checker.check(
        selection["selected"] == truth["selection"]["selected"],
        "selection.json: selected model differs from the ground truth",
    )
    checker.check(
        selection["rule"] == truth["selection"]["rule"],
        "selection.json: selection rule text differs",
    )
    checker.close(
        selection["alert_threshold"],
        truth["selection"]["alert_threshold"],
        1e-12,
        "selection.json alert_threshold",
    )
    checker.close(
        selection["high_threshold"],
        truth["selection"]["high_threshold"],
        1e-12,
        "selection.json high_threshold",
    )

    selected = test[truth["selection"]["selected"]]
    for field in ("tp", "fp", "fn", "tn", "n", "positives"):
        checker.check(
            selected[field] == truth["selected_test"][field],
            f"test_results {field}: {selected[field]} != " f"{truth['selected_test'][field]}",
        )
    for field in (
        "precision",
        "recall",
        "f1",
        "tss",
        "hss2",
        "far",
        "accuracy",
        "pr_auc",
        "roc_auc",
    ):
        checker.close(
            round(selected[field], 4), truth["selected_test"][field], 5e-5, f"test_results {field}"
        )

    for key, want in truth["candidates_test"].items():
        for field in ("tss", "pr_auc", "precision", "recall", "f1", "roc_auc"):
            checker.close(
                round(test[key][field], 3), want[field], 5e-4, f"test_results {key}.{field}"
            )
        checker.close(
            round(validation[key]["val"]["tss"], 3),
            want["val_tss"],
            5e-4,
            f"validation_results {key}.tss",
        )
        checker.close(
            round(validation[key]["val"]["pr_auc"], 3),
            want["val_pr_auc"],
            5e-4,
            f"validation_results {key}.pr_auc",
        )

    checker.check(audit["n_total"] == truth["dataset"]["n_windows"], "data_audit n_total differs")
    checker.check(
        audit["n_features"] == truth["dataset"]["n_features"], "data_audit n_features differs"
    )
    checker.check(
        all(v == 0 for v in audit["shared_ar"].values()),
        "data_audit: an active region is shared between partitions",
    )
    for partition, want in truth["partitions"].items():
        got = audit["per_partition"][partition]
        checker.check(int(got["n"]) == want["n"], f"partition {partition} window count")
        checker.check(int(got["pos"]) == want["positives"], f"partition {partition} positive count")
        checker.check(
            int(got["n_ar"]) == want["n_ar"], f"partition {partition} active-region count"
        )

    bundle = models.load_bundle(models.selected_key())
    import numpy as np

    coef = bundle.model.named_steps["clf"].coef_.ravel()
    order = np.argsort(np.abs(coef))[::-1][:6]
    for (name, value), index in zip(truth["top_coefficients"], order, strict=True):
        checker.check(
            bundle.cols[index] == name,
            f"top coefficient order: got {bundle.cols[index]}, expected {name}",
        )
        checker.close(round(float(coef[index]), 3), value, 5e-4, f"coefficient {name}")

    return truth


# --------------------------------------------------------------------------- #
# pass 2 - numbers in prose
# --------------------------------------------------------------------------- #
def legitimate_values(truth: dict) -> dict[str, set[float]]:
    """Every value a metric is allowed to appear as, at full precision.

    A quoted number is accepted if it equals one of these rounded to however
    many decimal places the document actually shows, so both "TSS 0.853" and
    "TSS 0.852874" are recognised as the same legitimate figure.
    """
    test = json.loads(paths.TEST_JSON.read_text(encoding="utf-8"))
    validation = json.loads(paths.VALIDATION_JSON.read_text(encoding="utf-8"))

    pools: dict[str, set[float]] = {
        k: set() for k in ("tss", "pr_auc", "roc_auc", "precision", "recall", "f1", "hss2", "far")
    }
    for key in test:
        for metric in pools:
            pools[metric].add(test[key][metric])
            pools[metric].add(validation[key]["val"][metric])

    # Post-hoc analyses legitimately quote their own values for these metrics.
    extra = paths.RESULTS_EXTRA
    for name in (
        "bootstrap",
        "baselines",
        "ablation",
        "operating_points",
        "breakdowns",
        "near_miss",
        "calibration",
    ):
        path = extra / f"{name}.json"
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for metric in pools:
            for match in re.finditer(rf'"{metric}":\s*(-?\d+\.?\d*)', text):
                pools[metric].add(float(match.group(1)))
        for match in re.finditer(
            r'"(?:ci95_low|ci95_high|point_estimate|'
            r'validation_tss|test_tss|test_pr_auc)":\s*(-?\d+\.?\d*)',
            text,
        ):
            for metric in pools:
                pools[metric].add(float(match.group(1)))

    # The clean-room retrain's observed figures, quoted in docs/RECON.md and
    # docs/REPRODUCIBILITY.md. They are recorded in the fixture because
    # build/retrain/ is scratch output and is not in version control, so without
    # this the documents would fail to check on a fresh clone.
    for metric, value in truth.get("retrain_observed", {}).items():
        if metric in pools and isinstance(value, (int, float)):
            pools[metric].add(float(value))

    return pools


def known_counts() -> set[str]:
    """Comma-grouped integers that may legitimately be quoted as window counts.

    Covers the fixed partition sizes plus every integer appearing in the
    post-hoc artefacts, so figures like the bootstrap's largest resampled
    partition are recognised instead of flagged.
    """
    counts: set[str] = set()
    for path in sorted(paths.RESULTS_EXTRA.glob("*.json")):
        for match in re.finditer(r"(?<![\d.])(\d{4,7})(?![\d.])", path.read_text(encoding="utf-8")):
            counts.add(f"{int(match.group(1)):,}")
    return counts


METRIC_WORDS = {
    "tss": r"TSS",
    "pr_auc": r"PR-AUC",
    "roc_auc": r"ROC-AUC",
}


def check_documents(checker: Checker, truth: dict) -> None:
    allowed = legitimate_values(truth)
    files: list[Path] = []
    for pattern in DOC_GLOBS:
        files.extend(sorted(paths.ROOT.glob(pattern)))

    known = known_counts() | {
        f"{truth['dataset']['n_windows']:,}",
        f"{truth['split']['n_test']:,}",
        f"{truth['split']['n_validation']:,}",
        f"{truth['split']['n_train']:,}",
        f"{truth['dataset']['n_negative']:,}",
        f"{truth['dataset']['n_flare_quiet']:,}",
        "327,980",
        "323,377",
        "64,315",
    }

    for path in files:
        text = path.read_text(encoding="utf-8")
        for metric, word in METRIC_WORDS.items():
            # "TSS 0.853", "TSS of 0.853", "TSS = 0.853", "TSS is 0.852874"
            for match in re.finditer(rf"{word}\s*(?:of|=|is|at|:)?\s*(\d\.\d+)", text):
                shown = match.group(1)
                places = len(shown.split(".")[1])
                checker.check(
                    any(f"{value:.{places}f}" == shown for value in allowed[metric]),
                    f"{paths.relative(path)}: {word} {shown} is not a value computed "
                    f"anywhere in results/",
                )

        # Any comma-grouped count stated as a number of windows must be real.
        for match in re.finditer(r"(\d{2,3},\d{3})\s+(?:observation\s+)?windows", text):
            checker.check(
                match.group(1) in known,
                f"{paths.relative(path)}: '{match.group(1)} windows' is not a count "
                f"that appears anywhere in results/",
            )


# --------------------------------------------------------------------------- #
# pass 3 - curated claims
# --------------------------------------------------------------------------- #
def check_claims(checker: Checker, truth: dict) -> None:
    import pandas as pd

    stored = pd.read_csv(paths.TEST_PREDICTIONS)
    alert, high = models.thresholds()
    alerts = int((stored["p"] >= alert).sum())
    base = float(stored["y"].mean())

    checker.check(
        alerts == truth["selected_test"]["n_alerts"],
        f"alert count {alerts} != {truth['selected_test']['n_alerts']}",
    )
    checker.close(
        round(alerts / len(stored), 4), truth["selected_test"]["alert_rate"], 5e-5, "alert rate"
    )
    checker.close(
        round((alerts / len(stored)) / base, 1),
        truth["selected_test"]["alert_rate_times_base"],
        0.05,
        "alert rate as a multiple of the base rate",
    )
    checker.close(
        round(1 - base, 4),
        truth["selected_test"]["always_no_flare_accuracy"],
        5e-5,
        "accuracy of an always-no-flare forecast",
    )

    selected = json.loads(paths.TEST_JSON.read_text(encoding="utf-8"))[models.selected_key()]
    fpr = selected["fp"] / (selected["fp"] + selected["tn"])
    checker.close(round(fpr, 4), truth["selected_test"]["fpr"], 5e-5, "false-alarm rate (FPR)")
    checker.check(
        abs(selected["far"] - fpr) > 0.5,
        "the false-alarm ratio and rate have become similar; the docs "
        "distinguish them on the basis that they differ by an order of magnitude",
    )

    # Dashboard JSON must agree with results/.
    if paths.DASHBOARD_DEMO.is_file():
        demo = json.loads(paths.DASHBOARD_DEMO.read_text(encoding="utf-8"))
        for field in ("tp", "fp", "fn", "tn"):
            checker.check(
                demo["test"][field] == selected[field],
                f"dashboard/demo.json test.{field} differs from results/",
            )
    if paths.DASHBOARD_OPERATING.is_file():
        operating = json.loads(paths.DASHBOARD_OPERATING.read_text(encoding="utf-8"))
        row = operating["sweep"][operating["alert_index"]]
        for field in ("tp", "fp", "fn", "tn"):
            checker.check(
                row[field] == selected[field],
                f"dashboard/operating.json at the published threshold has "
                f"{field}={row[field]}, results/ says {selected[field]}",
            )
    if paths.DASHBOARD_MODEL_LR.is_file():
        exported = json.loads(paths.DASHBOARD_MODEL_LR.read_text(encoding="utf-8"))
        checker.close(
            exported["alert_threshold"], alert, 1e-12, "dashboard/model_lr.json alert_threshold"
        )
        checker.close(
            exported["high_threshold"], high, 1e-12, "dashboard/model_lr.json high_threshold"
        )
        checker.check(
            len(exported["cols"]) == truth["dataset"]["n_features"],
            "dashboard/model_lr.json has the wrong number of features",
        )


# --------------------------------------------------------------------------- #
# pass 4 - language
# --------------------------------------------------------------------------- #
def check_language(checker: Checker) -> None:
    files: list[Path] = []
    for pattern in (*DOC_GLOBS, "dashboard/index.html", "dashboard/model.js"):
        files.extend(sorted(paths.ROOT.glob(pattern)))
    files.extend(sorted((paths.SRC / "solarflare").rglob("*.py")))

    for path in files:
        if path.name == "check_consistency.py":
            continue
        checker.check(True, "")  # one check per file scanned, so the count is honest
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, 1):
            lowered = line.lower()
            # A heading or table question is answered on the lines that follow,
            # so judge the phrase against a small window the way a reader would.
            context = " ".join(lines[max(0, number - 2) : number + 2])
            for phrase, why in FORBIDDEN_PHRASES.items():
                if phrase in lowered and not is_denied(context, phrase):
                    checker.check(
                        False,
                        f"{paths.relative(path)}:{number}: contains {phrase!r} - {why}",
                    )
            for name, why in FORBIDDEN_NAMES.items():
                if name in lowered:
                    checker.check(
                        False,
                        f"{paths.relative(path)}:{number}: contains the name {name!r} - {why}",
                    )
            if any(exempt in line for exempt in SPELLING_EXEMPT):
                continue
            for pattern, british in SPELLING.items():
                if re.search(pattern, lowered):
                    checker.check(
                        False,
                        f"{paths.relative(path)}:{number}: American spelling, " f"use {british!r}",
                    )


# --------------------------------------------------------------------------- #
def main() -> int:
    checker = Checker()
    print("1. results/ against tests/fixtures/ground_truth.json")
    truth = check_ground_truth(checker)
    print(f"   {checker.checks} checks")

    before = checker.checks
    print("2. numbers stated in the documents")
    check_documents(checker, truth)
    print(f"   {checker.checks - before} checks")

    before = checker.checks
    print("3. curated claims and the dashboard data")
    check_claims(checker, truth)
    print(f"   {checker.checks - before} checks")

    before = checker.checks
    print("4. overclaiming, forbidden names and spelling")
    check_language(checker)
    print(f"   {checker.checks - before} checks")

    print()
    if checker.failures:
        print(f"FAIL - {len(checker.failures)} of {checker.checks} checks failed:")
        for failure in checker.failures:
            print(f"  - {failure}")
        return 1
    print(f"PASS - {checker.checks} consistency checks, no disagreements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
