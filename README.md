# Solar Flare Prediction & Space Weather Alert System

**Forecasting major solar flares from magnetogram data, with the protocol you would want a
forecaster to follow.** Given 12 hours of SDO/HMI SHARP measurements for one solar active region,
this system estimates the probability that the region will produce a GOES M- or X-class flare in
the next 24 hours. It is built on the SWAN-SF benchmark — 331,185 observation windows spanning
2010–2018 — under a leakage-safe protocol in which the train, validation and test partitions share
no active region, every learned preprocessing step is fitted on the training partitions alone, and
both the model and its decision threshold were fixed on validation **before the test partition was
scored once**.

> ### Academic prototype — not an operational space-weather warning service
> Historical data only. No live feed, no NOAA/GOES monitoring, no operational validation. It is
> recall-oriented by design: it detects 91.6 % of major flares, and 84 % of the alerts it raises are
> not followed by one. Both numbers are real and both are reported everywhere.

---

## Headline results

On the **locked test partition P5** — 75,365 windows, 990 positive (1.31 %), scored once at the
validation-fixed threshold:

| Metric | Value | 95 % interval |
|---|---|---|
| **TSS** (true skill statistic) | **0.8529** | [0.751, 0.915] |
| **PR-AUC** | **0.4894** | [0.267, 0.657] |
| Recall (probability of detection) | 0.9162 | [0.812, 0.976] |
| **Precision** | **0.1616** | [0.094, 0.237] |
| ROC-AUC | 0.9790 | [0.964, 0.989] |
| F1 | 0.2747 | [0.168, 0.378] |
| HSS (HSS2 form) | 0.2581 | [0.157, 0.357] |
| False-alarm **ratio** FP/(TP+FP) | 0.8384 | [0.763, 0.906] |
| False-alarm **rate** FP/(FP+TN) | 0.0633 | [0.046, 0.082] |

**Confusion matrix: TP 907 · FP 4,707 · FN 83 · TN 69,668.** Accuracy 0.9364 — and a model that
always forecasts "no major flare" scores **98.69 %** accuracy while detecting nothing, which is why
accuracy is not the headline here.

Intervals come from resampling whole active regions 1,000 times (seed 42), because consecutive
windows of one region are one hour apart 98.60 % of the time and so share 11 of their 12 hours of
data. **They are wide enough to matter:** the TSS interval spans 0.164 while the entire spread
across all seven candidate models is 0.014 — the candidates are statistically indistinguishable on
this partition.

## Three things this project is honest about

**1. A single feature matches the model on TSS.** The rule `TOTUSJH__max ≥ 1442.37`, chosen on
validation and scored once on P5, reaches TSS 0.8514 against the model's 0.8529, and beats it on
precision and F1. The model's real advantage is ranking quality: PR-AUC 0.4894 versus 0.3720, a
32 % relative improvement. The defensible claim is that the pipeline buys calibrated ranking and
exact explainability, **not** a higher headline TSS than a classical threshold.

**2. Most "false alarms" are not false.** Of the 4,707 false alarms, **77.8 % precede a real B- or
C-class flare**; only 22.2 % land on a genuinely flare-quiet window. The model alerts on 49.8 % of
C-class windows but 1.67 % of quiet ones, and its median probability rises monotonically across
F → B → C → M → X despite only ever seeing a binary target. The errors sit at the magnitude
boundary, not scattered over the quiet Sun. This does not make 16 % precision acceptable — a
C-class flare is not an M-class flare — but it is a different failure from guessing.

**3. The candidate search was narrow.** An ablation found several smaller variants with *higher*
validation TSS than the selected model (a 24-feature last-value variant reaches 0.8744 against
0.8622). They were not candidates, so the selection was honest, but the project should not claim
144 temporal features were necessary.

## Architecture

```
  SWAN-SF window                 feature extraction              fitted pipeline
  60 records x 24 SHARP   ->   24 params x 6 statistics   ->   slog -> median impute
  12 h, 12-min cadence         = 144 features                  -> standardise -> logistic
                                                                   (fitted on P1-P3 only)
                                              |
                                              v
                              probability -> threshold -> LOW / MODERATE / HIGH
                                             (both fixed on P4)
```

| Split | Partitions | Windows | Role |
|---|---|---|---|
| Train | P1–P3 | 204,559 | fit the model and all preprocessing |
| Validation | P4 | 51,261 | choose the model **and** both thresholds |
| Test | P5 | 75,365 | scored **once**, after `selection.json` was written |

Selected model: **`I2_LR_temporal_C0.01`** — L2 logistic regression, 144 temporal features,
`class_weight="balanced"`, C=0.01, seed 42. Selection rule, fixed in advance: maximum validation
TSS to 3 decimals, tie-break validation PR-AUC.

Thresholds, both from validation only: **alert 0.5445** (maximises TSS), **high-risk 0.9673**
(maximises F1).

## Quick start

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m solarflare evaluate
```

### Linux / macOS

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m solarflare evaluate
```

`evaluate` recomputes every published number from the saved models and compares — 375 checks, about
two minutes. **scikit-learn must be exactly 1.8.0**; the saved pipelines are scikit-learn pickles
and will not reproduce under another minor version. The runtime check fails loudly if it is wrong.

### See the demo

Fastest route — press **Ctrl + Alt + S** anywhere in Windows. That hotkey is attached to a desktop
shortcut which starts the server and opens the browser. Install or change it with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1
```

`-Hotkey "CTRL+ALT+F"` picks a different combination and `-Remove` deletes the shortcut. Windows
only honours a shortcut's hotkey while the `.lnk` stays on the Desktop. Double-clicking
**`Open Dashboard.bat`** does the same thing without a hotkey.

Or start it by hand:

```powershell
python -m solarflare dashboard
```

Then open **http://localhost:8791/index.html**. It runs entirely offline — no network, no CDN. Press
**▶ 3-minute tour** for the guided walkthrough, and use the address bar's `#explore`, `#live` or
`#operate` to link straight to a view.

## Commands

Every command is `python -m solarflare <name>`; `make <name>` works too where `make` is available.

| Command | What it does | Time |
|---|---|---|
| `env` | report the runtime and whether the models can be loaded | < 2 s |
| `audit` | merge the feature chunks, verify the data audit | ~4 min |
| `evaluate` | **recompute every frozen metric and compare** | ~2 min |
| `analysis` | the post-hoc evidence: intervals, calibration, baselines, ablation, errors | ~8 min |
| `figures` | build `figures/extra/` | ~25 s |
| `demo-data` | regenerate every JSON the dashboard reads | ~6 min |
| `dashboard` | serve the dashboard on localhost | — |
| `screenshots` | capture every view to `docs/screenshots/` | ~20 s |
| `test` | the pytest suite (`--fast` skips slow and browser tests) | ~90 s |
| `check` | verify every documented number against `results/` | ~15 s |
| `train` | clean-room retrain into a scratch directory | ~15 min |
| `features` | extract features from the raw tarballs | hours |
| `all` | everything above, in order | ~20 min |

## Reproducing every table and figure

| Artefact | Command |
|---|---|
| The headline table in this README | `python -m solarflare evaluate` |
| The 95 % intervals | `python -m solarflare analysis` → `results/extra/bootstrap.json` |
| Every post-hoc table | `python -m solarflare analysis` → `results/extra/RESULTS.md` |
| `figures/extra/*.png` | `python -m solarflare figures` |
| The dashboard's data | `python -m solarflare demo-data` |
| `docs/screenshots/*.png` | `python -m solarflare screenshots` |
| That the docs agree with `results/` | `python scripts/check_consistency.py` |

The report's own figures (`figures/fig*.png`) and everything in `results/` and `models/` are
**frozen**. The code refuses to write to them: `solarflare.paths.assert_not_frozen()` is called
before every write and raises. Additive output goes to `results/extra/`, `figures/extra/` and
`docs/screenshots/`.

## Folder map

| Path | Contents |
|---|---|
| `src/solarflare/` | the package: data, features, metrics, models, verification, CLI |
| `src/solarflare/analysis/` | the nine post-hoc analyses and their report generator |
| `src/common.py`, `src/train_eval.py`, … | the original pipeline scripts, unmodified |
| `models/` | the seven trained pipelines (28 MB) — **frozen** |
| `results/` | every published metric, prediction file and log — **frozen** |
| `results/extra/` | post-hoc analyses, all labelled as such |
| `figures/` | the report's figures — **frozen** |
| `figures/extra/` | the eight additive figures |
| `dashboard/` | the static single-page demo, its generated JSON, and `model.js` |
| `tests/` | 189 tests across nine files |
| `docs/` | the documents listed below |
| `scripts/` | the consistency checker and the parity-fixture builder |
| `data/`, `raw_data/` | not in git; see [docs/DATA_CARD.md](docs/DATA_CARD.md) |

## Documentation

| Document | Read it for |
|---|---|
| [docs/PROJECT_EXPLAINED.md](docs/PROJECT_EXPLAINED.md) | the whole project from scratch, for a reader with no background |
| [docs/FEATURES.md](docs/FEATURES.md) | what every view does, in plain English, and how to describe it |
| [docs/JUDGE_DEMO_SCRIPT.md](docs/JUDGE_DEMO_SCRIPT.md) | the timed three-minute walkthrough |
| [docs/JUDGES_QA.md](docs/JUDGES_QA.md) | fifteen hard questions, each answered with a file or a number |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | intended use, out-of-scope use, metrics with intervals, calibration, failure modes |
| [docs/DATA_CARD.md](docs/DATA_CARD.md) | SWAN-SF provenance, the partition table, the leakage audit, missingness |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | versions, seeds, runtimes, known problems |
| [docs/DECISIONS.md](docs/DECISIONS.md) | every non-obvious choice, with the reason |
| [docs/RECON.md](docs/RECON.md) | the reproduction evidence |
| [results/extra/RESULTS.md](results/extra/RESULTS.md) | all nine post-hoc analyses in full |

## Honest limitations

- **Not operational.** No live feed, no GOES monitoring, no latency or availability engineering, no
  operational validation. It would be irresponsible to act on its output.
- **Not calibrated.** Probabilities over-forecast by about 7.3× because of the balanced class
  weights. Read them as a ranking and against the two fixed thresholds, never as literal chances.
- **Not validated outside 2010–2018 or outside HMI SHARP data.** Solar cycle 24 declined across the
  test partition; nothing here shows the model transfers to cycle 25.
- **Not causal.** The coefficients are correlations in a standardised feature space. That total
  unsigned current helicity and the R-value dominate is a plausibility check against the
  literature, not evidence of mechanism.
- **Wide intervals.** 75,365 test windows come from only 758 regions and 990 positives. The PR-AUC
  interval is [0.267, 0.657].
- **No X-class claim.** X-class recall is 19/19, but on 19 cases — interval [0.832, 1.000].
- **Two raw archives on the verification machine are corrupt** (P2 and P5). No published result is
  affected, but re-extraction from them would silently produce a short dataset. See
  [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) §7.

## Credits

**Madhavan G** (Register No. 2104251040518) — pipeline owner: SWAN-SF acquisition, feature
extraction, leakage audit, partition split design, LR/RF experiments, model selection, test
evaluation, result figures.

**Sriram Sivakumar** (Register No. 2104251040971) — applied half: exploratory analysis,
risk-scoring and alert logic, web dashboard, documentation and reproducibility checks.

Contribution recorded as 50/50.

Project-Based Learning project, Machine Learning course, B.E. Computer Science and Engineering,
**Chennai Institute of Technology (Autonomous)**, Chennai – 69, affiliated to Anna University,
Chennai. Academic year 2026–2027.

Department of CSE — Mentor and Project Co-ordinator: **R. Poornima Lakshmi, M.E.**, Assistant
Professor. HOD: **Dr. S. Pavithra, M.E., Ph.D.** Dean: **Dr. V. Srinivasa Rao, M.E., Ph.D.**
Class advisors: **Dr. G. Irin Loretta, M.E.** and **S.E. Neela Kandan, M.Tech.**

## Citing this work

See [CITATION.cff](CITATION.cff). If you use the data, cite SWAN-SF separately:

> Angryk, R. A., Martens, P. C., Aydin, B., Kempton, D., Mahajan, S. S., Basodi, S., Ahmadzadeh, A.,
> Cai, X., Filali Boubrahimi, S., Hamdi, S. M., Schuh, M. A., and Georgoulis, M. K. (2020).
> Multivariate time series dataset for space weather data analytics. *Scientific Data* **7**, 227.
> doi:10.1038/s41597-020-0548-x · Dataset DOI: 10.7910/DVN/EBCFKM

## Licence

MIT — see [LICENSE](LICENSE). The licence covers this repository's code, documentation and figures.
**It does not cover the SWAN-SF dataset**, which is distributed by its authors through the Harvard
Dataverse under that repository's terms. Nothing here redistributes the dataset.
