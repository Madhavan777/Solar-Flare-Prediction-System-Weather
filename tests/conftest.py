"""Shared fixtures.

Tests that need the 190 MB feature matrix or the trained pipelines are marked
``largedata`` / ``models`` and **skip cleanly** when those files are absent, so
the fast suite runs in CI on a fresh clone with no data.
"""

from __future__ import annotations

import pytest

from solarflare import data, models, paths


def pytest_collection_modifyitems(config, items):
    """Skip data- and model-dependent tests when their inputs are missing."""
    no_data = not (paths.ALL_X.exists() and paths.ALL_META.exists())
    no_models = not (paths.MODELS / "I2_LR_temporal_C0.01.joblib").exists()
    skip_data = pytest.mark.skip(
        reason=f"needs {paths.relative(paths.ALL_X)} and {paths.relative(paths.ALL_META)}; "
        f"rebuild with 'python -m solarflare audit'"
    )
    skip_models = pytest.mark.skip(reason="needs the trained pipelines in models/")
    for item in items:
        if no_data and "largedata" in item.keywords:
            item.add_marker(skip_data)
        if no_models and "models" in item.keywords:
            item.add_marker(skip_models)


@pytest.fixture(scope="session")
def matrix():
    """The merged feature matrix and window metadata, loaded once."""
    return data.load()


@pytest.fixture(scope="session")
def splits(matrix):
    """Boolean ``(train, validation, test)`` masks."""
    _, meta = matrix
    return data.split_masks(meta)


@pytest.fixture(scope="session")
def selected():
    """The selected final model's bundle."""
    return models.load_bundle(models.selected_key())


@pytest.fixture(scope="session")
def thresholds():
    """``(alert_threshold, high_threshold)`` from the frozen selection."""
    return models.thresholds()
