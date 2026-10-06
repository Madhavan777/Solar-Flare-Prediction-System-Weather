# Model card — Solar Flare Prediction & Space Weather Alert System

**Model:** `I2_LR_temporal_C0.01` — L2-regularised logistic regression on 144 temporal SHARP
features.
**Version:** 1.0.0, selected 2026-09-27, frozen.
**Licence:** MIT (code). The SWAN-SF data is separately licensed; see [DATA_CARD.md](DATA_CARD.md).

> **This is an academic prototype, not an operational space-weather warning service.** It is
> trained and evaluated only on historical SWAN-SF data from 2010-05-01 to 2018-08-17, it consumes
> no live feed of any kind, and it has never been validated in an operational setting.

---

## 1. What it does

Given 12 hours of SDO/HMI SHARP magnetogram summaries for one solar active region (60 records at a
12-minute cadence, spanning 11.8 h from the first to the last timestamp), it estimates the
probability that the region will produce a **major flare — GOES class M or X — in the following
24 hours**, measured from the last timestamp of the window.

The probability is turned into one of three risk levels using two thresholds that were fixed on the
validation partition before the test partition was ever scored:

| Level | Condition |
|---|---|
| LOW | p < 0.5445 |
| MODERATE | 0.5445 ≤ p < 0.9673 |
| HIGH | p ≥ 0.9673 |

## 2. Intended use

- Teaching and demonstrating a leakage-safe supervised learning pipeline on a severely imbalanced,
  temporally structured scientific dataset.
- Reproducing and inspecting the published results in this repository.
- Exploring the precision/recall trade-off of a rare-event forecast.

## 3. Out-of-scope use

Do not use this model for any of the following. Each is a limitation of the evidence, not a legal
disclaimer.

- **Operational or safety-critical space-weather forecasting.** No live data path exists, no
  service-level behaviour has been established, and no operational validation has been done.
- **Forecasting outside 2010–2018 or outside SDO/HMI SHARP data.** Solar cycle 24 declined across
  the test partition and the model has never been applied to cycle 25. Nothing here establishes
  that it transfers.
- **Reading the output as a literal probability.** See §7 — it over-forecasts by about 7.3×.
- **Any causal or physical claim.** The coefficients are correlations in a standardised feature
  space. That the strongest weights fall on total unsigned current helicity and the R-value is a
  plausibility check against the flare-physics literature, not evidence of mechanism.
- **Forecasting individual flare timing, magnitude, or location.** The target is binary and
  window-level: M-or-X versus not, within 24 hours.

## 4. Architecture

A single scikit-learn `Pipeline`, fitted end to end on the training partitions only:

```
signed-log  ->  median imputation  ->  standardisation  ->  logistic regression
  (fixed)        (fitted P1-P3)        (fitted P1-P3)       (C=0.01, balanced)
```

- `slog(x) = sign(x)·log₁₀(1+|x|)` — a **fixed** transform, not learned, so it cannot carry
  information across the split. It compresses the SHARP parameters' enormous dynamic range
  (USFLUX spans ~10²², TOTPOT ~10²³) while preserving sign.
- Median imputation and standardisation are fitted inside the pipeline on P1–P3 only. Tests assert
  that the imputer's `statistics_` and the scaler's `mean_`/`scale_` equal those recomputed from
  the training partitions, and that they differ from an all-data fit.
- `class_weight="balanced"` weights each class by `n_samples / (n_classes · n_class)`. On the
  training partitions, where positives are **1.99 %** of windows, that is a weight of **25.07** per
  positive against **0.51** per negative — positives count about **49×** more heavily. This is what buys
  the recall, and it is also why the probabilities are not calibrated.
- `random_state=42`, `max_iter=5000`.

**Features:** 24 SHARP parameters × 6 statistics (`last`, `mean`, `std` with ddof=0, `min`, `max`,
`slope` as a least-squares trend per hour), computed over the finite values inside the window only.
Full definitions and the exclusion list are in [DATA_CARD.md](DATA_CARD.md).

## 5. How it was selected

Seven candidates were trained — one point-in-time logistic regression, three temporal logistic
regressions at different regularisation strengths, and three random forests. The rule was fixed
before any of them was scored on the test partition:

> maximum validation TSS rounded to 3 decimals, tie-break on validation PR-AUC.

The winner, `I2_LR_temporal_C0.01`, reached validation TSS 0.862 and validation PR-AUC 0.500. Its
decision threshold was then chosen on validation alone, as the TSS-maximising point of a 2,000-point
quantile grid. **Only then was the test partition scored, once.**

`python -m solarflare evaluate` re-applies that rule from scratch and re-derives both thresholds
from the validation predictions, reproducing `results/selection.json` to better than 1e-12.

## 6. Performance on the locked test partition (P5)

75,365 windows, 990 positive (1.31 %). Scored once, at the validation-fixed threshold.

| Metric | Value | 95 % interval |
|---|---|---|
| **TSS** | **0.8529** | [0.751, 0.915] |
| **PR-AUC** | **0.4894** | [0.267, 0.657] |
| ROC-AUC | 0.9790 | [0.964, 0.989] |
| Recall (probability of detection) | 0.9162 | [0.812, 0.976] |
| **Precision** | **0.1616** | [0.094, 0.237] |
| F1 | 0.2747 | [0.168, 0.378] |
| HSS (HSS2 form) | 0.2581 | [0.157, 0.357] |
| False-alarm **ratio** FP/(TP+FP) | 0.8384 | [0.763, 0.906] |
| False-alarm **rate** FP/(FP+TN) | 0.0633 | [0.046, 0.082] |
| Accuracy | 0.9364 | — |

Confusion matrix: **TP 907, FP 4,707, FN 83, TN 69,668.**

Intervals come from resampling whole active regions, 1,000 times, seed 42
(`results/extra/bootstrap.json`). Whole regions are the unit because consecutive windows of one
region are one hour apart 98.60 % of the time and so share 11 of their 12 hours of observations.

**Read the accuracy figure with this sentence attached:** a model that always forecasts "no major
flare" scores **98.69 %** accuracy on this partition and detects nothing at all.

**The intervals are wide enough to matter.** The TSS interval spans 0.164, while the entire spread
of test TSS across all seven candidates is 0.014. The candidates are statistically indistinguishable
on this partition, which is why selecting on validation was the only defensible procedure.

## 7. Calibration — the probabilities over-forecast

**The output is not a calibrated probability.** Mean predicted probability on P5 is 0.0964 against
an actual base rate of 0.0131: the model over-forecasts by about **7.3×**. A window scored 0.60 does
not flare 60 % of the time. Brier score 0.0500, against 0.0130 for a constant forecast at the
training base rate — a constant forecast beats the model on Brier score while detecting nothing,
which is why Brier score is not used to choose a flare forecaster.

This is a direct and expected consequence of `class_weight="balanced"`: the model is trained as if
major flares were far commoner than they are, which is what shifts the boundary towards recall.

**What it does not affect:** TSS, precision, recall, F1, HSS and the confusion matrix depend only on
which side of the threshold each probability falls; PR-AUC and ROC-AUC depend only on the ranking.
Platt scaling fitted on the validation partition reduces the Brier score to 0.0090 and moves the
ROC-AUC by exactly 0.0 — verified, not assumed, in `results/extra/calibration.json`. The recalibrated
variants are reported as separate comparisons and are **not** the selected model.

Treat the number as a ranking and against the two fixed thresholds, never as a literal chance.

## 8. Failure modes

Measured, not speculated. Full detail in `results/extra/error_gallery.json` and
`results/extra/near_miss.json`.

**The 83 misses.** All 83 are M-class; the model missed no X-class window (19 of 19 detected). They
fall in just 9 HARP regions. The lowest probability assigned to a genuine pre-flare window was
0.0493 — confidently wrong, not merely uncertain. That window is in HARP 7115, the same region as
the showcase X1.3 example.

**The 4,707 false alarms, in context.** They involve 95 of the 758 regions in P5, and 15 regions
(2.0 %) produce half of them; 663 regions produce none. Because consecutive windows overlap by 11 of
their 12 hours, one mis-read region yields a long run — the longest is 211 consecutive alerted
windows in HARP 5541. So 4,707 false alarms are nothing like 4,707 independent errors.

**What actually followed those false alarms.** 77.8 % precede a genuine B- or C-class flare, and only
22.2 % land on a window that was truly flare-quiet. The model alerts on 49.8 % of C-class windows but
just 1.67 % of flare-quiet ones, and its median predicted probability rises monotonically across
F → B → C → M → X even though it only ever saw a binary target. The errors are concentrated at the
M-class magnitude boundary rather than scattered over the quiet Sun.

**This does not make the precision acceptable.** A C-class flare is not an M-class flare, and an
operator told "major flare likely" before a C-class event has still been misinformed. The published
precision of 0.1616 is correct and is not revised by that analysis.

**Missingness.** Performance is stable up to about 5 % missing values and degrades beyond it
(recall 0.667 in the 5–25 % band, from a small sample). 10.43 % of all windows contain at least one
missing SHARP value.

**Calendar drift.** Precision is stable across 2015–2017 (0.164–0.165) while the base rate falls as
cycle 24 declines. P5's final year, 2018, contains **no positive windows at all**, so no skill
metric is defined there.

## 9. What the model is actually worth

Honest accounting, from `results/extra/baselines.json`:

| | Climatology | Best single feature | This model |
|---|---|---|---|
| Rule | constant p = 0.0199 | `TOTUSJH__max ≥ 1442.37` | LR on 144 features |
| TSS | 0.0000 | 0.8514 | 0.8529 |
| Precision | 0.0000 | 0.1667 | 0.1616 |
| F1 | 0.0000 | 0.2819 | 0.2747 |
| PR-AUC | 0.0131 | 0.3720 | **0.4894** |

A single threshold on one SHARP parameter — chosen on validation, scored once on P5 — **matches the
model on TSS and beats it on precision and F1**. The model's advantage is in ranking quality across
all thresholds: PR-AUC 0.4894 against 0.3720, a 32 % relative improvement. That is what supports the
second, much stricter HIGH-risk band, where precision reaches 0.58.

The defensible claim is the narrow one: the pipeline buys calibrated ranking and exact
explainability, not a higher headline TSS than a classical single-parameter threshold.

A statistic-family ablation (`results/extra/ablation.json`) goes further: several smaller variants
reach a **higher** validation TSS than the selected model — a 24-feature last-value variant reaches
0.8744 against 0.8622. They were not candidates, so the selection was honest, but the candidate
search was narrow and the project should not claim that 144 features were necessary.

## 10. Explainability

Because the pipeline is a fixed transform, an imputer, a scaler and a linear model, the per-window
explanation is **exact**:

```
logit(p) = intercept + Σⱼ coefⱼ · standardised_xⱼ
```

There is no surrogate model, no sampling and nothing to tune. Across 2,000 random P5 windows the
reconstructed logit matches the model's own to 1.95e-14 (`results/extra/explanation_identity.json`),
and `tests/test_contributions.py` asserts the identity to 1e-9.

Largest standardised coefficients: `R_VALUE__std` −1.571, `TOTUSJH__max` +1.123, `TOTUSJH__last`
+1.010, `SHRGT45__min` +1.010, `R_VALUE__max` +0.837, `SHRGT45__last` +0.785. Total unsigned current
helicity and the R-value are established flare-productivity proxies in the literature, so the
ranking is physically plausible — as a sanity check only, never as a causal claim.

## 11. Ethical and operational caveats

- **Asymmetric costs are assumed, not measured.** The model is recall-oriented because missing a
  major flare is treated as worse than a false alarm. No cost-loss analysis was performed, so the
  operating point is a defensible default rather than an optimised decision.
- **Alert fatigue is a real risk of this design.** Alerting on 7.45 % of windows, 5.7× the base
  rate, with 84 % of alerts not followed by a major flare, would plausibly be ignored in practice.
- **It must never be presented as a NOAA/SWPC product or substitute.** No live GOES monitoring is
  implemented and none is a model input.
- **Deployment would need operational validation, live-data engineering, monitoring for drift
  across the solar cycle, and recalibration** — none of which exists here.

## 12. Reproducing these numbers

```powershell
python -m solarflare evaluate    # recomputes every figure above and compares
python -m solarflare analysis    # the post-hoc evidence, including the intervals
python scripts/check_consistency.py
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for versions, seeds, runtimes and file hashes.

## 13. Credits

Madhavan G (Register No. 2104251040518) and Sriram Sivakumar (Register No. 2104251040971),
B.E. Computer Science and Engineering, Chennai Institute of Technology (Autonomous), Chennai – 69,
affiliated to Anna University, Chennai. Project-Based Learning project, Machine Learning course,
academic year 2026–2027. Mentor and Project Co-ordinator: R. Poornima Lakshmi, M.E., Assistant
Professor, Department of CSE.
