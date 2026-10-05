"""Data-contract tests for the result and dashboard JSON files.

``src/shoot.py`` and ``dashboard/index.html`` consume these files by key. The
keys and the view ids are therefore a published interface: the report's Figures
5.1–5.6 were captured through them, so breaking one silently invalidates a
figure in a frozen document.
"""

from __future__ import annotations

import json

import pytest

from solarflare import models, paths

# Keys src/shoot.py and the original six dashboard views depend on.
DEMO_REQUIRED = {
    "model_key": str,
    "model_name": str,
    "alert_threshold": float,
    "high_threshold": float,
    "test": dict,
    "examples": dict,
    "top_features": list,
}
TEST_BLOCK_REQUIRED = (
    "threshold",
    "precision",
    "recall",
    "f1",
    "tss",
    "hss2",
    "far",
    "accuracy",
    "pr_auc",
    "roc_auc",
    "tp",
    "fp",
    "fn",
    "tn",
    "n",
    "positives",
)
#: The six original view ids. shoot.py calls window.showView(name) for each.
ORIGINAL_VIEWS = ("overview", "predict", "result", "risk", "explain", "eval")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# results/*.json
# --------------------------------------------------------------------------- #
def test_selection_json_schema():
    selection = load(paths.SELECTION_JSON)
    assert set(selection) >= {"selected", "rule", "alert_threshold", "high_threshold", "decided_at"}
    assert isinstance(selection["alert_threshold"], float)
    assert 0.0 < selection["alert_threshold"] < selection["high_threshold"] < 1.0


@pytest.mark.parametrize("path", [paths.TEST_JSON, paths.VALIDATION_JSON])
def test_result_json_covers_all_candidates(path):
    stored = load(path)
    assert set(stored) == set(models.CANDIDATE_KEYS)


def test_test_results_blocks_are_complete():
    stored = load(paths.TEST_JSON)
    for key, block in stored.items():
        missing = [f for f in TEST_BLOCK_REQUIRED if f not in block]
        assert not missing, f"{key} is missing {missing}"
        assert block["tp"] + block["fn"] == block["positives"], key
        assert block["tp"] + block["fp"] + block["fn"] + block["tn"] == block["n"], key


def test_validation_results_have_both_threshold_blocks():
    stored = load(paths.VALIDATION_JSON)
    for key, block in stored.items():
        assert {"iteration", "family", "description", "n_features", "val", "val_at_0_5"} <= set(
            block
        ), key
        assert block["val_at_0_5"]["threshold"] == 0.5, key
        assert block["n_features"] in (24, 144), key


def test_data_audit_json_schema():
    audit = load(paths.DATA_AUDIT_JSON)
    assert audit["n_total"] == 331185
    assert audit["n_features"] == 144
    assert set(audit["per_partition"]) == {"1", "2", "3", "4", "5"}
    assert len(audit["shared_ar"]) == 10
    assert all(v == 0 for v in audit["shared_ar"].values())


# --------------------------------------------------------------------------- #
# dashboard/demo.json — the contract src/shoot.py relies on
# --------------------------------------------------------------------------- #
def test_demo_json_exists_in_both_places():
    assert paths.DASHBOARD_DEMO.is_file()
    assert paths.DASHBOARD_DEMO_JSON.is_file()


def test_demo_json_has_every_required_key():
    demo = load(paths.DASHBOARD_DEMO)
    for key, kind in DEMO_REQUIRED.items():
        assert key in demo, f"demo.json is missing {key}"
        assert isinstance(demo[key], kind), f"demo.json[{key}] should be {kind.__name__}"


def test_demo_json_test_block_is_complete():
    demo = load(paths.DASHBOARD_DEMO)
    for field in TEST_BLOCK_REQUIRED:
        assert field in demo["test"], f"demo.json['test'] is missing {field}"


def test_demo_json_examples_shape():
    examples = load(paths.DASHBOARD_DEMO)["examples"]
    assert set(examples) == {"flare_case", "quiet_case"}
    for name, case in examples.items():
        assert {"ar", "start", "end", "probability"} <= set(case), name
        assert isinstance(case["ar"], int)
        assert 0.0 <= case["probability"] <= 1.0
    assert "flare_class" in examples["flare_case"]


def test_demo_json_top_features_shape():
    features = load(paths.DASHBOARD_DEMO)["top_features"]
    assert len(features) >= 5
    for entry in features:
        assert set(entry) == {"name", "weight"}
        assert isinstance(entry["weight"], float)


def test_demo_json_agrees_with_the_frozen_results():
    """Nothing in the dashboard's data may contradict results/."""
    demo = load(paths.DASHBOARD_DEMO)
    frozen = load(paths.TEST_JSON)[demo["model_key"]]
    alert, high = models.thresholds()

    assert demo["model_key"] == models.selected_key()
    assert demo["alert_threshold"] == pytest.approx(alert, abs=5e-5)
    assert demo["high_threshold"] == pytest.approx(high, abs=5e-5)
    for field in TEST_BLOCK_REQUIRED:
        assert demo["test"][field] == pytest.approx(frozen[field], rel=1e-12), field


def test_demo_json_matches_results_copy():
    """dashboard/demo.json and results/dashboard_demo.json must not diverge."""
    assert load(paths.DASHBOARD_DEMO) == load(paths.DASHBOARD_DEMO_JSON)


# --------------------------------------------------------------------------- #
# the dashboard page itself
# --------------------------------------------------------------------------- #
def test_dashboard_html_keeps_the_six_original_view_ids():
    """shoot.py drives these ids to regenerate the report's Figures 5.1-5.6."""
    html = (paths.DASHBOARD / "index.html").read_text(encoding="utf-8")
    for view in ORIGINAL_VIEWS:
        assert f'id="v-{view}"' in html, f"view id v-{view} has gone"
        assert f'data-v="{view}"' in html, f"nav button for {view} has gone"
    assert "window.showView" in html


def test_dashboard_declares_the_prototype_disclaimer():
    html = (paths.DASHBOARD / "index.html").read_text(encoding="utf-8")
    assert "not an operational space-weather warning service" in html


def test_dashboard_has_no_external_resource_references():
    """It must work offline at the review, with no CDN and no network calls."""
    html = (paths.DASHBOARD / "index.html").read_text(encoding="utf-8")
    for needle in ("http://", "https://", "//cdn", "integrity="):
        assert needle not in html, f"dashboard/index.html references {needle}"


# --------------------------------------------------------------------------- #
# the generated dashboard data added in phase 3
# --------------------------------------------------------------------------- #
def test_model_lr_json_contract():
    """The browser-side pipeline reads exactly these fields."""
    payload = load(paths.DASHBOARD_MODEL_LR)
    required = {
        "model_key",
        "cols",
        "apply_slog",
        "impute_median",
        "scale_mean",
        "scale_std",
        "coef",
        "intercept",
        "alert_threshold",
        "high_threshold",
        "sharp_parameters",
        "statistics",
        "cadence_hours",
        "expected_rows",
    }
    assert required <= set(payload)
    n = len(payload["cols"])
    assert n == 144
    for field in ("impute_median", "scale_mean", "scale_std", "coef"):
        assert len(payload[field]) == n, field
    assert payload["apply_slog"] is True
    assert len(payload["sharp_parameters"]) == 24
    assert len(payload["statistics"]) == 6
    assert payload["expected_rows"] == 60
    assert payload["cadence_hours"] == 0.2
    assert all(s > 0 for s in payload["scale_std"]), "a zero scale would divide by zero"


def test_model_lr_json_matches_the_saved_pipeline():
    """The exported parameters are the real fitted ones, not a copy that drifted."""
    payload = load(paths.DASHBOARD_MODEL_LR)
    bundle = models.load_bundle(models.selected_key())
    steps = bundle.model.named_steps
    assert payload["cols"] == list(bundle.cols)
    assert payload["coef"] == pytest.approx(list(steps["clf"].coef_.ravel()), rel=1e-12, abs=1e-15)
    assert payload["intercept"] == pytest.approx(
        float(steps["clf"].intercept_.ravel()[0]), rel=1e-12
    )
    assert payload["impute_median"] == pytest.approx(list(steps["impute"].statistics_), rel=1e-12)
    assert payload["scale_mean"] == pytest.approx(list(steps["scale"].mean_), rel=1e-12)
    assert payload["scale_std"] == pytest.approx(list(steps["scale"].scale_), rel=1e-12)


def test_windows_json_contract():
    """The explorer view's window set."""
    payload = load(paths.DASHBOARD_WINDOWS)
    assert {"selection_rule", "seed", "windows"} <= set(payload)
    windows = payload["windows"]
    assert len(windows) >= 15
    outcomes = {w["outcome"] for w in windows}
    assert {
        "TP",
        "FP",
        "FN",
        "TN",
    } <= outcomes, f"the explorer must include successes and failures; got {outcomes}"
    for window in windows:
        assert {
            "file",
            "harp",
            "y",
            "probability",
            "risk",
            "outcome",
            "series",
            "series_available",
            "contributions",
            "window_start",
            "window_end_cutoff",
        } <= set(window)
        assert 0.0 <= window["probability"] <= 1.0
        assert window["risk"] in ("LOW", "MODERATE", "HIGH")
        assert window["outcome"] in ("TP", "FP", "FN", "TN")
        assert len(window["contributions"]) >= 5
        # A window may legitimately lack its raw series: the P5 archive on the
        # machine these results were verified on is a corrupt gzip stream and
        # stops after 64,315 of 75,365 windows (docs/REPRODUCIBILITY.md). When
        # that happens the window must say so rather than show an empty panel.
        if window["series_available"]:
            assert window["series"], "series_available is true but the series is empty"
            assert window["series"]["values"], "the series carries no parameter columns"
        else:
            assert window["series_note"], "a window without its series must explain why"


def test_most_explorer_windows_carry_their_raw_series():
    """A mostly-empty explorer would make the view pointless."""
    payload = load(paths.DASHBOARD_WINDOWS)
    with_series = sum(1 for w in payload["windows"] if w["series_available"])
    assert with_series >= 0.75 * len(payload["windows"]), (
        f"only {with_series} of {len(payload['windows'])} explorer windows have their "
        f"raw 12-hour series"
    )
    assert payload["n_with_raw_series"] == with_series


def test_operating_json_contract():
    """The threshold slider's precomputed grid."""
    payload = load(paths.DASHBOARD_OPERATING)
    assert {"alert_threshold", "high_threshold", "banner", "sweep"} <= set(payload)
    assert "Exploratory" in payload["banner"]
    sweep = payload["sweep"]
    assert len(sweep) >= 100
    for row in sweep[:5]:
        assert {
            "threshold",
            "tp",
            "fp",
            "fn",
            "tn",
            "precision",
            "recall",
            "tss",
            "far",
            "fpr",
            "alert_rate",
        } <= set(row)
    thresholds = [row["threshold"] for row in sweep]
    assert thresholds == sorted(thresholds), "the sweep must be sorted by threshold"
    for row in sweep:
        assert row["tp"] + row["fp"] + row["fn"] + row["tn"] == 75365
