"""Tests for the project's own machinery: paths, the frozen-file guard, the
environment check, the CLI surface, and the compatibility shim the pickles need.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from solarflare import cli, env, models, paths


# --------------------------------------------------------------------------- #
# the compatibility shim
# --------------------------------------------------------------------------- #
def test_common_is_importable_as_a_top_level_module():
    """The saved pipelines embed FunctionTransformer(common.slog).

    Unpickling resolves it by the name ``common``, so that module must stay
    importable at top level however the package is laid out.
    """
    import common

    assert hasattr(common, "slog")
    assert common.slog(0.0) == 0.0
    assert common.slog(9.0) == pytest.approx(1.0)
    assert common.slog(-9.0) == pytest.approx(-1.0)


def test_slog_is_odd_and_monotone():
    import numpy as np

    import common

    x = np.array([-1e6, -1e3, -1.0, -0.1, 0.0, 0.1, 1.0, 1e3, 1e6])
    y = common.slog(x)
    np.testing.assert_allclose(y, -common.slog(-x), atol=1e-12)
    assert np.all(np.diff(y) > 0)


@pytest.mark.models
def test_every_saved_pipeline_unpickles():
    for key in models.CANDIDATE_KEYS:
        bundle = models.load_bundle(key)
        assert bundle.key == key
        assert len(bundle.cols) in (24, 144)
        assert hasattr(bundle.model, "predict_proba")


@pytest.mark.models
def test_pipeline_step_structure_is_as_documented():
    selected = models.load_bundle("I2_LR_temporal_C0.01")
    assert list(selected.model.named_steps) == ["slog", "impute", "scale", "clf"]
    assert selected.is_logistic
    assert selected.uses_slog
    assert selected.model.named_steps["clf"].C == 0.01
    assert selected.model.named_steps["clf"].class_weight == "balanced"
    assert selected.model.named_steps["clf"].random_state == 42

    baseline = models.load_bundle("I1_LR_last")
    assert list(baseline.model.named_steps) == ["impute", "scale", "clf"]
    assert not baseline.uses_slog


@pytest.mark.models
def test_random_forest_hyperparameters_are_as_documented():
    for key, leaf, depth in (
        ("I2_RF_leaf5", 5, None),
        ("I2_RF_leaf20", 20, None),
        ("I2_RF_leaf50_d16", 50, 16),
    ):
        forest = models.load_bundle(key).model.named_steps["clf"]
        assert forest.n_estimators == 300
        assert forest.min_samples_leaf == leaf
        assert forest.max_depth == depth
        assert forest.max_features == "sqrt"
        assert forest.class_weight == "balanced_subsample"
        assert forest.random_state == 42


# --------------------------------------------------------------------------- #
# paths and the frozen guard
# --------------------------------------------------------------------------- #
def test_project_root_is_discovered_by_its_anchor():
    assert (paths.ROOT / paths.ANCHOR).is_file()
    assert paths.find_root() == paths.ROOT


def test_root_discovery_fails_loudly_outside_a_project(tmp_path):
    with pytest.raises(FileNotFoundError, match="could not locate the project root"):
        paths.find_root(tmp_path)


def test_frozen_set_covers_models_results_and_report_figures():
    frozen = {p.name for p in paths.FROZEN}
    for key in models.CANDIDATE_KEYS:
        assert f"{key}.joblib" in frozen
    for name in (
        "selection.json",
        "validation_results.json",
        "test_results.json",
        "data_audit.json",
        "train_log.txt",
        "test_predictions_selected.csv.gz",
    ):
        assert name in frozen
    assert any(n.startswith("fig6_") for n in frozen)
    assert any(n.startswith("fig5_") for n in frozen)


@pytest.mark.parametrize(
    "target",
    [
        paths.SELECTION_JSON,
        paths.TEST_JSON,
        paths.DATA_AUDIT_JSON,
        paths.TEST_PREDICTIONS,
        paths.MODELS / "I2_LR_temporal_C0.01.joblib",
        paths.FIGURES / "fig6_1_confusion_matrix.png",
    ],
)
def test_frozen_files_cannot_be_opened_for_writing(target):
    with pytest.raises(PermissionError, match="frozen artefact"):
        paths.assert_not_frozen(target)


@pytest.mark.parametrize(
    "target",
    [
        paths.RESULTS_EXTRA / "anything.json",
        paths.FIGURES_EXTRA / "anything.png",
        paths.DASHBOARD_MODEL_LR,
    ],
)
def test_additive_outputs_are_allowed(target):
    assert paths.assert_not_frozen(target) == target


def test_relative_formats_with_forward_slashes():
    assert paths.relative(paths.SELECTION_JSON) == "results/selection.json"
    assert paths.relative(paths.MODELS / "x.joblib") == "models/x.joblib"


# --------------------------------------------------------------------------- #
# environment check
# --------------------------------------------------------------------------- #
def test_environment_check_passes_here():
    env.check()


def test_scikit_learn_is_the_required_version():
    assert env.installed("scikit-learn") == env.REQUIRED["scikit-learn"] == "1.8.0"


def test_environment_report_lists_every_tracked_package():
    report = env.report()
    assert "python" in report
    for package in (*env.REQUIRED, *env.EXPECTED):
        assert package in report


def test_environment_check_raises_on_a_missing_requirement(monkeypatch):
    monkeypatch.setattr(env, "REQUIRED", {"not-a-real-package": "9.9.9"})
    with pytest.raises(env.EnvironmentMismatch, match="not installed"):
        env.check()


def test_environment_check_raises_on_a_wrong_version(monkeypatch):
    monkeypatch.setattr(env, "REQUIRED", {"scikit-learn": "0.0.1"})
    with pytest.raises(env.EnvironmentMismatch, match="will not reproduce"):
        env.check()


# --------------------------------------------------------------------------- #
# the CLI surface
# --------------------------------------------------------------------------- #
def test_every_documented_command_exists():
    parser = cli.build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    commands = set()
    for action in actions:
        commands |= set(action.choices)
    for expected in (
        "audit",
        "features",
        "train",
        "evaluate",
        "figures",
        "demo-data",
        "dashboard",
        "test",
        "all",
        "env",
        "analysis",
        "check",
        "screenshots",
    ):
        assert expected in commands, f"the CLI has no '{expected}' command"


def test_cli_with_no_command_prints_help_and_returns_2():
    assert cli.main([]) == 2


def test_cli_help_runs_as_a_subprocess():
    done = subprocess.run(
        [sys.executable, "-m", "solarflare", "--help"],
        cwd=paths.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0
    assert "not an operational warning service" in done.stdout


def test_cli_env_command_returns_zero():
    import argparse

    assert cli.cmd_env(argparse.Namespace(strict=False)) == 0


# --------------------------------------------------------------------------- #
# repository hygiene
# --------------------------------------------------------------------------- #
def test_gitignore_excludes_the_large_inputs():
    text = (paths.ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("raw_data/", "data/", ".venv/", "reference/"):
        assert pattern in text, f".gitignore does not exclude {pattern}"


def test_no_committed_file_exceeds_25_mb():
    """Models are 5-14 MB and ship deliberately; nothing may be larger."""
    skip = {".venv", "raw_data", "data", "reference", ".git", "build", "__pycache__"}
    offenders = []
    for path in paths.ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip for part in path.relative_to(paths.ROOT).parts):
            continue
        size = path.stat().st_size
        if size > 25 * 1024 * 1024:
            offenders.append(f"{paths.relative(path)} ({size / 1e6:.1f} MB)")
    assert not offenders, "files over 25 MB: " + ", ".join(offenders)


def test_no_file_is_double_encoded():
    """Catch UTF-8 text that has been round-tripped through a single-byte code page.

    A tool that reads a UTF-8 file as cp1252 and writes it back as UTF-8 turns
    every non-ASCII character into a mojibake sequence: an em dash becomes
    "a-circumflex, euro, right-double-quote". It is easy to do by accident on
    Windows, it does not raise, and the damage only surfaces later in generated
    output - so it is worth a test rather than an eye.
    """
    markers = ("â€", "â\u0080", "Ã©", "Â ")
    skip_dirs = {
        ".venv",
        ".git",
        "raw_data",
        "data",
        "reference",
        "build",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
    }
    suffixes = {
        ".py",
        ".md",
        ".html",
        ".js",
        ".json",
        ".toml",
        ".cff",
        ".txt",
        ".yml",
        ".yaml",
        ".cfg",
        ".ini",
    }
    offenders = []
    for path in paths.ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if any(part in skip_dirs for part in path.relative_to(paths.ROOT).parts):
            continue
        if path.name == "test_project.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            offenders.append(f"{paths.relative(path)} is not valid UTF-8")
            continue
        for marker in markers:
            if marker in text:
                offenders.append(f"{paths.relative(path)} contains mojibake {marker!r}")
                break
    assert not offenders, "; ".join(offenders)


def test_requirements_pin_scikit_learn_exactly():
    text = (paths.ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "scikit-learn==1.8.0" in text


def test_license_and_citation_exist():
    assert (paths.ROOT / "LICENSE").is_file()
    assert (paths.ROOT / "CITATION.cff").is_file()
    citation = (paths.ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "10.7910/DVN/EBCFKM" in citation


def test_no_retired_branding_or_former_mentor_reference_anywhere():
    """Two names that must not appear in this repository.

    The needles are assembled from pieces so that this test file does not
    itself contain the strings it forbids, and so does not have to exempt
    itself from its own scan.
    """
    forbidden = ("sir" + "agu", "raja" + " priya", "raja" + "priya")
    skip_dirs = {
        ".venv",
        ".git",
        "raw_data",
        "data",
        "reference",
        "build",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
    }
    text_suffixes = {
        ".py",
        ".md",
        ".html",
        ".json",
        ".yml",
        ".yaml",
        ".toml",
        ".txt",
        ".cff",
        ".cfg",
        ".ini",
        ".js",
        ".css",
    }
    offenders = []
    for path in paths.ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        if any(part in skip_dirs for part in path.relative_to(paths.ROOT).parts):
            continue
        lowered = path.read_text(encoding="utf-8", errors="ignore").lower()
        for needle in forbidden:
            if needle in lowered:
                offenders.append(f"{paths.relative(path)} contains {needle!r}")
    assert not offenders, "; ".join(offenders)
