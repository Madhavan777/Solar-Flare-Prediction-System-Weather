"""The per-window explanation must be exact, not approximate.

For the selected linear pipeline the log-odds decompose with no residual::

    logit(p) = intercept + Î£_j coef_j Â· standardised_x_j

These tests assert that identity, which is what licenses the dashboard to say
"this feature contributed +0.42 to the log-odds of this forecast".
"""

from __future__ import annotations

import numpy as np
import pytest

from solarflare import models
from solarflare.analysis import explain

pytestmark = [pytest.mark.largedata, pytest.mark.models]


@pytest.fixture(scope="module")
def sample(matrix):
    """A deterministic 500-window sample of the test partition."""
    from solarflare import data

    X, meta = matrix
    _, _, test = data.split_masks(meta)
    rng = np.random.default_rng(42)
    index = rng.choice(np.where(test)[0], size=500, replace=False)
    return X.iloc[index]


def test_contributions_plus_intercept_equal_the_models_logit(selected, sample):
    """The additive identity, to 1e-9."""
    _contrib, logit, _intercept = models.contributions(selected, sample)
    probability = selected.predict_proba(sample)
    model_logit = np.log(probability / (1 - probability))
    assert np.max(np.abs(logit - model_logit)) < 1e-9


def test_reconstructed_probability_matches_predict_proba(selected, sample):
    """Applying the sigmoid to the reconstructed logit returns the real output."""
    _, logit, _ = models.contributions(selected, sample)
    reconstructed = 1.0 / (1.0 + np.exp(-logit))
    assert np.max(np.abs(reconstructed - selected.predict_proba(sample))) < 1e-9


def test_contribution_matrix_shape(selected, sample):
    contrib, logit, _ = models.contributions(selected, sample)
    assert contrib.shape == (len(sample), 144)
    assert logit.shape == (len(sample),)


def test_standardised_values_match_the_pipelines_own_transform(selected, sample):
    """Our re-implementation of the preprocessing equals the fitted steps."""
    ours = models.standardised(selected, sample)
    steps = selected.model.named_steps
    theirs = sample[selected.cols]
    for name in ("slog", "impute", "scale"):
        theirs = steps[name].transform(theirs)
    np.testing.assert_allclose(ours, np.asarray(theirs), rtol=1e-12, atol=1e-12)


def test_standardised_training_columns_are_centred_and_unit_scale(selected, matrix, splits):
    """Sanity check on the scaler: P1-P3 standardises to mean 0, std 1."""
    X, _ = matrix
    train, _, _ = splits
    z = models.standardised(selected, X.loc[train])
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(z.std(axis=0, ddof=0), 1.0, atol=1e-8)


def test_imputed_features_are_flagged_in_the_explanation(selected, matrix):
    """A window with a fully missing parameter reports that feature as imputed."""
    X, _meta = matrix
    missing = X[X["MEANALP__mean"].isna()]
    if missing.empty:
        pytest.skip("no window in the matrix has a fully missing MEANALP")
    row = missing.iloc[[0]]
    top = explain.top_contributions(selected, row, k=144)
    by_name = {entry["feature"]: entry for entry in top}
    assert by_name["MEANALP__mean"]["imputed"] is True
    assert by_name["MEANALP__mean"]["raw_value"] is None
    assert np.isfinite(by_name["MEANALP__mean"]["standardised_value"])


def test_top_contributions_are_sorted_by_absolute_magnitude(selected, sample):
    top = explain.top_contributions(selected, sample.iloc[[0]], k=10)
    magnitudes = [abs(entry["contribution_logodds"]) for entry in top]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert len(top) == 10


def test_contribution_equals_coefficient_times_standardised_value(selected, sample):
    """Spot-check the definition itself."""
    row = sample.iloc[[0]]
    top = explain.top_contributions(selected, row, k=5)
    for entry in top:
        expected = entry["coefficient"] * entry["standardised_value"]
        assert entry["contribution_logodds"] == pytest.approx(expected, abs=1e-12)


def test_random_forest_contributions_are_refused():
    """Exact additive contributions are only defined for the linear model."""
    forest = models.load_bundle("I2_RF_leaf50_d16")
    assert not forest.is_logistic
    with pytest.raises(TypeError, match="only defined for the linear model"):
        models.contributions(forest, None)
