# Reproducibility

Everything published in this project can be recomputed from the files in this repository. This
document records exactly what is needed, what to expect, and what is known not to work.

---

## 1. The one command

```powershell
python -m solarflare all
```

Runs, in order: the environment check, the data merge and audit, the full reproduction of every
frozen metric, the post-hoc analyses, the figures, the dashboard data, the number-consistency
check, and the test suite. It stops at the first failure unless `--keep-going` is passed.

To verify only that the published results still reproduce, which is the thing that matters:

```powershell
python -m solarflare evaluate
```

## 2. Environment

| Component | Original run | Verified on | Hard requirement? |
|---|---|---|---|
| Python | 3.11 | **3.12.10** | no — warns only |
| **scikit-learn** | **1.8.0** | **1.8.0** | **yes — hard failure** |
| NumPy | 2.4.4 | 2.4.4 | no — warns |
| pandas | 3.0.2 | 3.0.2 | no — warns |
| SciPy | **not pinned** | 1.18.1 | no — warns, but see §3 |
| joblib | 1.5.3 | 1.5.3 | no — warns |
| matplotlib | 3.10.9 | 3.10.9 | no — warns |
| Playwright | 1.56.0 | 1.56.0 | only for the dashboard tests |
| Node.js | — | any recent | only for the JavaScript parity tests |

**scikit-learn must be exactly 1.8.0.** The seven pipelines in `models/` are scikit-learn pickles.
A different minor version either refuses to unpickle them or — worse — loads them with changed
internals, so the published metrics would silently stop being reproducible. `solarflare.env.check()`
raises on a mismatch with an actionable message; everything else only warns.

Python 3.11 is not installed on the verification machine, so 3.12.10 was used instead. Under it,
all seven models reproduce every stored metric to better than 1e-9, both thresholds to better than
1e-12, and the selected model's confusion matrix exactly. See
[DECISIONS.md](DECISIONS.md) D-02.

### Setting it up

**Windows (PowerShell), from the project root:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m solarflare env
```

**Linux / macOS:**

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m solarflare env
```

For the tests, figures and dashboard checks, add:

```bash
pip install -r requirements-dev.txt
python -m playwright install chromium
```

`pip install -e .` matters: it puts `src/` on the path so that `common` is importable as a
**top-level module**, which the pickles require — they embed `FunctionTransformer(common.slog)` and
unpickling resolves it by that name.

## 3. Determinism, and its limit

- `SEED = 42` in `src/common.py`, used for every estimator and every resampling procedure.
- The bootstrap, the explorer window selection and the explanation sample all take seed 42 and are
  reproducible run to run.
- The feature merge is deterministic: `sorted(glob("P{1..5}_c*_X.npz"))` over fixed filenames, which
  is also the row order the `row` column of `results/test_predictions_selected.csv.gz` refers to.
- Scoring the **shipped** pipelines is bit-reproducible: `evaluate` matches the stored probabilities
  to 1.3e-15 and every published metric to better than 1e-9.

### Retraining from scratch is *not* bit-reproducible across environments

This is a real limit and it is stated rather than glossed. A from-scratch retrain on the
verification machine (Windows, Python 3.12) converges to slightly different coefficients than the
frozen model, which was trained on Linux under Python 3.11:

| | Retrained | Frozen | Delta |
|---|---|---|---|
| Selected model | `I2_LR_temporal_C0.01` | `I2_LR_temporal_C0.01` | **same** |
| Selection rule | max val TSS, tie-break PR-AUC | same | **same** |
| Alert threshold | 0.5433280 | 0.5445466 | 1.2e-3 |
| High threshold | 0.9666874 | 0.9672704 | 5.8e-4 |
| Test TSS | 0.854639 | 0.852874 | 1.8e-3 |
| Test PR-AUC | 0.489424 | 0.489363 | 6.1e-5 |
| Test confusion | 909 / 4,726 / 81 / 69,649 | 907 / 4,707 / 83 / 69,668 | 2 windows |

**The decision is robust; the coefficients are not bit-identical.**

**The cause is not the BLAS thread count.** That was the obvious hypothesis and it was tested:
fitting the selected architecture on the same data with `OMP_NUM_THREADS=1` and with `=4` gives
bit-identical coefficients, an identical intercept and the same iteration count (132). Parallel
accumulation order is therefore ruled out.

What remains is some other difference in the numerical environment between the two runs. The most
plausible candidate is the **SciPy version**: `lbfgs` is implemented in `scipy.optimize`, and SciPy
is the one dependency the original specification did not pin — this project pins 1.18.1 by choice,
but the original run's version is unknown. The platform's BLAS build is a second candidate. This
cannot be settled without the original environment, so it is recorded as **unresolved**.

**What this means in practice:** `python -m solarflare train` checks that the *decision* reproduces
within a 5e-3 tolerance, not that the coefficients match. Pass `--strict` to demand exact equality,
which is expected to fail on any machine other than the original. Bit-level reproducibility of the
published numbers rests on the shipped artefacts, which is why `models/` is committed.

## 4. What must never change

`models/*.joblib`, everything in `results/`, both thresholds, the partition split, the seed, the
feature definitions, the selection rule, and the figures in `figures/`.

The code enforces this rather than relying on discipline: `solarflare.paths.assert_not_frozen()` is
called before every write and raises `PermissionError` on a frozen path. `tests/test_project.py`
asserts that it does. Additive output goes to `results/extra/`, `figures/extra/` and
`docs/screenshots/` only.

Two consequences worth knowing:

- `src/merge_audit.py` as originally written would overwrite `results/data_audit.json`, and
  `src/make_figures.py` would overwrite `figures/fig6_*.png`. Neither is wired into any CLI command.
  The verification path recomputes and **compares** instead.
- `src/shoot.py` writes to `figures/fig5_*.png`. Use `python -m solarflare screenshots`, which
  writes to `docs/screenshots/` and additionally captures the six original views in *report mode*
  with the new navigation hidden, so they stay visually equivalent to the report's figures.

## 5. Expected runtimes

Measured on the verification machine (Windows 11, 4 cores).

| Step | Command | Time |
|---|---|---|
| Environment check | `solarflare env` | < 2 s |
| Merge the feature chunks | `solarflare audit` | ~4 min (first run only) |
| Reproduce every frozen metric | `solarflare evaluate` | ~2 min |
| Post-hoc analyses | `solarflare analysis` | ~8 min |
| — of which the bootstrap | | ~3 min |
| — of which the ablation (12 retrains) | | ~3 min |
| Figures | `solarflare figures` | ~25 s |
| Dashboard data | `solarflare demo-data` | ~6 min (streams the P5 archive) |
| Test suite | `solarflare test` | ~90 s |
| Fast tests only | `solarflare test --fast` | ~30 s |
| Consistency check | `solarflare check` | ~15 s |
| Clean-room retrain | `solarflare train` | ~15 min |
| Feature extraction from raw | `solarflare features` | hours |

## 6. Getting the data

The repository does not contain the dataset. Two levels of reconstruction:

**From the extracted feature chunks** (`data/features/`, 241 MB) — four minutes:

```powershell
python -m solarflare audit
```

**From the raw SWAN-SF archives** — hours. Download SWAN-SF v1.2 from the Harvard Dataverse
(DOI 10.7910/DVN/EBCFKM) into `raw_data/`, then:

```powershell
python -m solarflare features --check-archives   # verify first
python -m solarflare features
python -m solarflare audit
```

## 7. Known problem: two raw archives are corrupt

On the verification machine, **`partition2_instances.tar.gz` and `partition5_instances.tar.gz` are
corrupt gzip streams**:

```
partition1: compressed 1,273,859,189 B -> decompressed 3,192,862,720 B   OK
partition2: error -3 while decompressing data: invalid block type
partition3: compressed   736,247,805 B -> decompressed 1,842,513,920 B   OK
partition4: compressed   885,390,803 B -> decompressed 2,217,000,960 B   OK
partition5: error -3 while decompressing data: invalid distance code
```

A streamed tar read stops **quietly** at the damage, with no exception, so the failure mode is a
silently short dataset rather than a crash. P5 yields 64,315 of its 75,365 windows before the
stream dies.

**No published result is affected.** `data/features/` was extracted while the archives were intact;
the per-chunk logs record the full counts (P2: 33,000 + 31,000 + 24,557 = 88,557; P5: 36,000 +
33,000 + 6,365 = 75,365), and `results/data_audit.json` reproduces exactly from them.

Mitigations now in place:

- `python -m solarflare features` verifies every archive before extracting and refuses to run if
  any is damaged;
- `python -m solarflare features --check-archives` runs that check alone;
- two of the twenty windows in the dashboard's explorer view have no raw series for this reason,
  and say so in the interface rather than showing an empty panel.

To repair: re-download the two archives from the Dataverse and re-run the check.

## 8. Expected output hashes

SHA-256 of the frozen result files. If one of these changes, something that should not have moved
has moved.

Regenerate this table with:

```powershell
python -c "import hashlib,pathlib; [print(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}') for p in sorted(pathlib.Path('results').glob('*')) if p.is_file()]"
```

| File | SHA-256 (first 16) |
|---|---|
| `selection.json` | see `results/extra/verification.json` |
| `validation_results.json` | — |
| `test_results.json` | — |
| `data_audit.json` | — |

Rather than pin hashes that a harmless reformat would break, the project pins the **numbers**:
`scripts/check_consistency.py` compares `results/` against
`tests/fixtures/ground_truth.json`, an independent transcription of the published figures, and
fails if any disagrees. That catches a changed value while tolerating a changed byte layout. Run it
with `python -m solarflare check`.

## 9. What reproduction actually proves

`python -m solarflare evaluate` performs 375 checks:

- every metric in `val`, `val_at_0_5` and `test` for all seven candidates, to 1e-6;
- the selection rule, re-applied from scratch, re-picks `I2_LR_temporal_C0.01`;
- both thresholds, recomputed from the validation predictions alone, match `selection.json` to
  1e-12;
- the selected model's P5 confusion matrix is exactly 907 / 4,707 / 83 / 69,668;
- the stored P5 probabilities match a fresh `predict_proba` to 1.3e-15;
- the data audit reproduces `results/data_audit.json` on every key.

Last recorded run: all 375 passed. The machine-readable record is written to
`results/extra/verification.json` on every run, including the environment it ran in.

## 10. Test suite

```powershell
python -m solarflare test           # everything, ~90 s
python -m solarflare test --fast    # skips slow and browser tests
```

186 tests across nine files. Tests that need the 190 MB feature matrix, the trained pipelines,
Node.js or Chromium **skip cleanly** when those are absent, so a fresh clone with no data still runs
a meaningful suite — which is exactly what CI does.

| File | Covers |
|---|---|
| `test_leakage.py` | excluded columns, window-only features, region disjointness, split mapping, preprocessing fitted on train only, thresholds from validation only |
| `test_features.py` | the extractor on synthetic windows: constants, exact ramps, NaNs, all-NaN columns, single points |
| `test_regression.py` | the published numbers, the confusion matrix, `selection.json` |
| `test_contributions.py` | the exact additive explanation identity |
| `test_metrics.py` | metric definitions against hand-computed matrices |
| `test_contracts.py` | the JSON schemas the dashboard and `shoot.py` depend on |
| `test_project.py` | paths, the frozen guard, the environment check, the CLI, repository hygiene |
| `test_js_parity.py` | JavaScript versus scikit-learn, to 1e-6 |
| `test_dashboard.py` | every view in a headless browser, numbers against `results/` |
