"""The browser must compute the same numbers as scikit-learn.

The dashboard's "Live inference" view reimplements the fitted pipeline in
JavaScript. That is only defensible if it is checked, so these tests run the
real ``dashboard/model.js`` in Node against the real Python path:

* **probabilities** — at least 1,000 precomputed P5 feature rows through the
  JavaScript pipeline versus ``predict_proba`` from the saved joblib model;
* **feature extraction** — at least 200 real SWAN-SF windows through the
  JavaScript extractor versus ``solarflare.features.window_features``.

Both must agree to better than 1e-6. The tests skip cleanly when Node is not
installed.
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from solarflare import features, models, paths

NODE = shutil.which("node")
FIXTURE = paths.ROOT / "tests" / "fixtures" / "parity_windows.json.gz"
TOLERANCE = 1e-6

pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


def run_node(script: str, payload: dict, tmp_path: Path) -> dict:
    """Execute a Node script with a JSON payload, and return its JSON output."""
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    script_path = tmp_path / "driver.js"
    script_path.write_text(script, encoding="utf-8")

    done = subprocess.run(
        [NODE, str(script_path), str(payload_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed ({done.returncode}):\n{done.stderr}")
    return json.loads(done.stdout)


MODEL_JS = (paths.DASHBOARD / "model.js").as_posix()


def test_model_js_exists_and_exports_the_api(tmp_path):
    """The dashboard and these tests must load the same file."""
    assert (paths.DASHBOARD / "model.js").is_file()
    out = run_node(
        textwrap.dedent(
            f"""
            const m = require({MODEL_JS!r});
            console.log(JSON.stringify({{
              keys: Object.keys(m).sort(),
              slog0: m.slog(0), slog9: m.slog(9), slogNeg9: m.slog(-9)
            }}));
        """
        ),
        {},
        tmp_path,
    )
    assert out["keys"] == ["parseWindowCSV", "predict", "slog", "windowFeatures"]
    assert out["slog0"] == 0.0
    assert out["slog9"] == pytest.approx(1.0)
    assert out["slogNeg9"] == pytest.approx(-1.0)


def test_javascript_slog_matches_python(tmp_path):
    """The fixed transform, over a wide dynamic range including negatives."""
    import common

    values = [-1e9, -1e6, -1234.5, -1.0, -1e-6, 0.0, 1e-6, 1.0, 1234.5, 1e6, 1e9]
    out = run_node(
        textwrap.dedent(
            f"""
            const m = require({MODEL_JS!r});
            const p = require(process.argv[2]);
            console.log(JSON.stringify(p.values.map(m.slog)));
        """
        ),
        {"values": values},
        tmp_path,
    )
    expected = common.slog(np.array(values))
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.largedata
@pytest.mark.models
@pytest.mark.slow
def test_javascript_probabilities_match_sklearn_on_1000_test_rows(tmp_path, matrix, splits):
    """>= 1,000 real P5 feature rows, JavaScript pipeline versus scikit-learn."""
    X, _ = matrix
    _, _, test = splits
    model_json = json.loads(paths.DASHBOARD_MODEL_LR.read_text(encoding="utf-8"))
    bundle = models.load_bundle(models.selected_key())

    rng = np.random.default_rng(42)
    index = rng.choice(np.where(test)[0], size=1000, replace=False)
    subset = X.iloc[index][bundle.cols]

    rows = [
        [None if not np.isfinite(v) else float(v) for v in row]
        for row in subset.to_numpy(dtype=np.float64)
    ]

    out = run_node(
        textwrap.dedent(
            f"""
            const m = require({MODEL_JS!r});
            const p = require(process.argv[2]);
            const out = p.rows.map(r => {{
              const feats = r.map(v => (v === null ? NaN : v));
              const res = m.predict(feats, p.model);
              return [res.probability, res.logit];
            }});
            console.log(JSON.stringify(out));
        """
        ),
        {"rows": rows, "model": model_json},
        tmp_path,
    )

    js_probability = np.array([r[0] for r in out])
    js_logit = np.array([r[1] for r in out])
    sk_probability = bundle.predict_proba(subset)

    max_diff = float(np.max(np.abs(js_probability - sk_probability)))
    assert max_diff < TOLERANCE, (
        f"JavaScript and scikit-learn probabilities differ by up to {max_diff:.3e} "
        f"on {len(subset)} P5 rows"
    )

    # The logit path must agree too, which is a stricter check at the extremes
    # where the sigmoid saturates and probabilities would agree trivially.
    _, py_logit, _ = models.contributions(bundle, subset)
    assert float(np.max(np.abs(js_logit - py_logit))) < 1e-6


@pytest.mark.largedata
@pytest.mark.models
def test_javascript_probabilities_match_on_saturated_extremes(tmp_path, matrix, splits):
    """The most and least confident P5 windows, where the sigmoid saturates."""
    X, _ = matrix
    _, _, test = splits
    bundle = models.load_bundle(models.selected_key())
    model_json = json.loads(paths.DASHBOARD_MODEL_LR.read_text(encoding="utf-8"))

    probability = bundle.predict_proba(X.loc[test])
    order = np.argsort(probability)
    test_index = np.where(test)[0]
    picked = np.concatenate([test_index[order[:50]], test_index[order[-50:]]])
    subset = X.iloc[picked][bundle.cols]

    rows = [
        [None if not np.isfinite(v) else float(v) for v in row]
        for row in subset.to_numpy(dtype=np.float64)
    ]
    out = run_node(
        textwrap.dedent(
            f"""
            const m = require({MODEL_JS!r});
            const p = require(process.argv[2]);
            console.log(JSON.stringify(p.rows.map(r =>
              m.predict(r.map(v => (v === null ? NaN : v)), p.model).probability)));
        """
        ),
        {"rows": rows, "model": model_json},
        tmp_path,
    )
    np.testing.assert_allclose(out, bundle.predict_proba(subset), rtol=0, atol=TOLERANCE)


@pytest.mark.skipif(not FIXTURE.is_file(), reason="run scripts/build_parity_fixture.py first")
@pytest.mark.slow
def test_javascript_feature_extraction_matches_python_on_200_real_windows(tmp_path):
    """>= 200 genuine SWAN-SF windows through both extractors."""
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        fixture = json.load(handle)
    windows = fixture["windows"]
    assert len(windows) >= 200, f"the fixture holds only {len(windows)} windows"

    model_json = json.loads(paths.DASHBOARD_MODEL_LR.read_text(encoding="utf-8"))
    out = run_node(
        textwrap.dedent(
            f"""
            const m = require({MODEL_JS!r});
            const p = require(process.argv[2]);
            const res = p.windows.map(w => m.windowFeatures(w.columns, p.model));
            console.log(JSON.stringify(res));
        """
        ),
        {"windows": windows, "model": model_json},
        tmp_path,
    )

    worst = 0.0
    worst_where = ""
    nan_mismatches = []
    for i, window in enumerate(windows):
        frame = {
            name: [np.nan if v is None else v for v in values]
            for name, values in window["columns"].items()
        }
        import pandas as pd

        python_features, _ = features.window_features(pd.DataFrame(frame))
        js_features = np.array(
            [np.nan if v is None else float(v) for v in out[i]], dtype=np.float64
        )

        python_nan = np.isnan(python_features)
        js_nan = np.isnan(js_features)
        if not np.array_equal(python_nan, js_nan):
            differing = np.where(python_nan != js_nan)[0][:3]
            nan_mismatches.append(f"{window['file']} at feature indices {differing.tolist()}")
            continue

        both = ~python_nan
        if both.any():
            scale = np.maximum(np.abs(python_features[both]), 1.0)
            diff = float(np.max(np.abs(js_features[both] - python_features[both]) / scale))
            if diff > worst:
                worst, worst_where = diff, window["file"]

    assert (
        not nan_mismatches
    ), "JavaScript and Python disagree about which features are missing: " + "; ".join(
        nan_mismatches[:5]
    )
    assert worst < TOLERANCE, (
        f"feature extraction differs by up to {worst:.3e} (relative), worst window "
        f"{worst_where}"
    )


@pytest.mark.skipif(not FIXTURE.is_file(), reason="run scripts/build_parity_fixture.py first")
def test_parity_fixture_contains_missing_values(tmp_path):
    """The fixture must exercise the NaN paths, or it proves little."""
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        fixture = json.load(handle)
    with_missing = sum(
        1
        for w in fixture["windows"]
        if any(v is None for col in w["columns"].values() for v in col)
    )
    assert with_missing > 0, (
        "no window in the parity fixture has a missing value, so the imputation and "
        "finite-point handling are untested"
    )


def test_javascript_handles_synthetic_edge_cases(tmp_path):
    """Constant series, exact ramps, all-NaN columns and single finite points."""
    import pandas as pd

    import common

    rng = np.random.default_rng(3)
    cases = []
    for i in range(60):
        frame = pd.DataFrame(rng.normal(scale=10 ** (i % 7), size=(60, 24)), columns=common.SHARP)
        if i % 3 == 0:
            frame.loc[:, "TOTUSJH"] = 7.5  # constant
        if i % 4 == 0:
            frame.loc[:, "USFLUX"] = np.arange(60) * 3.0  # exact ramp
        if i % 5 == 0:
            frame.loc[:, "MEANALP"] = np.nan  # entirely missing
        if i % 7 == 0:
            frame.loc[:58, "EPSZ"] = np.nan  # one finite point
        if i % 11 == 0:
            frame.iloc[-1, :] = np.nan  # final record missing
        cases.append(frame)

    model_json = json.loads(paths.DASHBOARD_MODEL_LR.read_text(encoding="utf-8"))
    payload = {
        "windows": [
            {
                "columns": {
                    name: [None if not np.isfinite(v) else float(v) for v in frame[name]]
                    for name in common.SHARP
                }
            }
            for frame in cases
        ],
        "model": model_json,
    }
    out = run_node(
        textwrap.dedent(
            f"""
            const m = require({MODEL_JS!r});
            const p = require(process.argv[2]);
            console.log(JSON.stringify(
              p.windows.map(w => m.windowFeatures(w.columns, p.model))));
        """
        ),
        payload,
        tmp_path,
    )

    for i, frame in enumerate(cases):
        python_features, _ = features.window_features(frame)
        js_features = np.array(
            [np.nan if v is None else float(v) for v in out[i]], dtype=np.float64
        )
        np.testing.assert_array_equal(
            np.isnan(python_features),
            np.isnan(js_features),
            err_msg=f"NaN pattern differs on synthetic case {i}",
        )
        both = ~np.isnan(python_features)
        scale = np.maximum(np.abs(python_features[both]), 1.0)
        np.testing.assert_allclose(
            js_features[both] / scale,
            python_features[both] / scale,
            rtol=0,
            atol=TOLERANCE,
            err_msg=f"feature values differ on synthetic case {i}",
        )


def test_javascript_csv_parser_reads_a_tab_separated_window(tmp_path):
    """The live view's parser, on a window rebuilt from the fixture."""
    if not FIXTURE.is_file():
        pytest.skip("parity fixture not built")
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        fixture = json.load(handle)
    window = fixture["windows"][0]
    parameters = fixture["sharp_parameters"]

    lines = ["Timestamp\t" + "\t".join(parameters)]
    for i in range(window["n_rows"]):
        cells = []
        for name in parameters:
            v = window["columns"][name][i]
            cells.append("" if v is None else repr(v))
        lines.append(f"2015-01-01T{i:02d}:00:00\t" + "\t".join(cells))
    text = "\n".join(lines)

    model_json = json.loads(paths.DASHBOARD_MODEL_LR.read_text(encoding="utf-8"))
    out = run_node(
        textwrap.dedent(
            f"""
            const m = require({MODEL_JS!r});
            const p = require(process.argv[2]);
            const parsed = m.parseWindowCSV(p.text, p.model);
            console.log(JSON.stringify({{
              rows: parsed.rows,
              features: m.windowFeatures(parsed.columns, p.model)
            }}));
        """
        ),
        {"text": text, "model": model_json},
        tmp_path,
    )
    assert out["rows"] == window["n_rows"]

    import pandas as pd

    frame = pd.DataFrame(
        {name: [np.nan if v is None else v for v in window["columns"][name]] for name in parameters}
    )
    python_features, _ = features.window_features(frame)
    js_features = np.array([np.nan if v is None else float(v) for v in out["features"]])
    np.testing.assert_array_equal(np.isnan(python_features), np.isnan(js_features))
    both = ~np.isnan(python_features)
    scale = np.maximum(np.abs(python_features[both]), 1.0)
    np.testing.assert_allclose(
        js_features[both] / scale, python_features[both] / scale, rtol=0, atol=TOLERANCE
    )
