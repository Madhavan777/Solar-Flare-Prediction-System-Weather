# Questions a panel will ask, and the evidence for each answer

Every answer points to a file or a number you can open on the spot. Nothing here is rhetorical:
if an answer would be weak, it says so.

---

## 1. Your precision is 16 %. Isn't the model useless?

**Short answer.** No, but the honest framing is not "16 % precision is fine" — it is "the errors are
concentrated at the magnitude boundary, and the operating point was a deliberate choice".

Three pieces of evidence, in order of strength:

**(a) 77.8 % of the "false alarms" precede a real flare.** Of 4,707 false alarms, 2,871 (61.0 %)
precede a C-class flare and 792 (16.8 %) a B-class flare. Only 1,044 (22.2 %) land on a genuinely
flare-quiet window. The model alerts on **49.8 %** of C-class windows but only **1.67 %** of
flare-quiet ones — a factor of thirty. Its median predicted probability rises monotonically across
F (0.000) → B (0.110) → C (0.542) → M (0.936) → X (1.000), even though it only ever saw a binary
target. *Evidence:* `results/extra/near_miss.json`, `figures/extra/extra5_near_miss.png`.

**(b) The errors are concentrated, not scattered.** 15 of P5's 758 regions (2.0 %) produce half of
all false alarms and 663 regions produce none. Because consecutive windows overlap by 11 of their 12
hours, one mis-read region gives a long run — the longest is 211 consecutive alerted windows in
HARP 5541. So 4,707 false alarms are nothing like 4,707 independent mistakes.
*Evidence:* `results/extra/error_gallery.json`.

**(c) It is a deliberate trade.** The threshold maximises TSS on validation, which for a rare,
high-consequence event means favouring recall. Raising the threshold to the HIGH-risk band lifts
precision to **0.58** at recall 0.32. Both points are on the dashboard's Operating Point view.

**What this does not excuse.** A C-class flare is not an M-class flare. An operator told "major
flare likely" before a C-class event has still been misinformed, and alert fatigue is a genuine
risk of this design. The published precision of 0.1616 is correct and is not revised by any of the
above.

---

## 2. Why did logistic regression win when a random forest has higher test TSS?

Because **the differences are statistically meaningless, and selection never used the test set.**

Three random forests and the point-in-time baseline score test TSS 0.860–0.867 against the selected
model's 0.853. But the 95 % interval on TSS, from resampling whole active regions, is
**[0.751, 0.915]** — **0.164 wide**. The entire spread across all seven candidates is **0.014**,
about twelve times narrower than the interval around any one of them. The candidates are
statistically indistinguishable on this partition.

Selection used validation evidence only, where the selected model had the highest TSS (0.862) and
the best PR-AUC (0.500). On the test partition it also has the **best PR-AUC of all seven**
(0.489 versus 0.352–0.479), which is the metric that respects the 1.31 % base rate.

Had we picked the random forest after seeing P5, we would have been selecting on the test set and
the whole protocol would be worthless. *Evidence:* `results/extra/bootstrap.json`,
`results/test_results.json`.

---

## 3. How do you know there is no data leakage?

Five mechanisms, each with a test that fails if it breaks. Run `python -m solarflare test`.

1. **No label-derived column can reach the model.** `BFLARE`/`CFLARE`/`MFLARE`/`XFLARE` and their
   `_LOC`/`_LABEL` variants, the GOES X-ray columns, the position and geometry metadata and the
   quality flags are all excluded in `src/common.py`. `test_excluded_columns_are_disjoint_from_predictors`
   asserts the exclusion set and the 144 predictors share nothing.
2. **The label comes from the folder name**, never from a column that is also a predictor.
   `test_label_matches_the_swan_sf_folder` asserts `y == (folder == "FL")` for all 331,185 windows.
3. **Features read only rows inside the window.** `test_features_ignore_non_sharp_columns` adds
   label-like columns at 1e9 and shows the feature vector is unchanged.
4. **No active region crosses the split.** All ten pairwise HARP intersections are empty
   (`results/data_audit.json`); `test_no_region_crosses_the_train_validation_test_boundary` asserts
   it for train/validation/test directly.
5. **All learned preprocessing was fitted on P1–P3 only.**
   `test_imputer_medians_equal_training_partition_medians` and
   `test_scaler_statistics_equal_training_partition_statistics` recompute the imputer's
   `statistics_` and the scaler's `mean_`/`scale_` from the training partitions and compare to the
   fitted values. A companion test asserts they **differ** from an all-data fit, so the check cannot
   pass vacuously.

Plus: both thresholds are recomputed from the validation predictions alone and match
`selection.json` to 1e-12, and a test asserts the published threshold is **not** equal to the
test-optimal one — if it were, the protocol would be unverifiable from outside.

---

## 4. Why not a random train/test split?

Because consecutive windows of one active region are one hour apart **98.60 %** of the time, while
each window covers 12 hours. Adjacent windows therefore share **11 of their 12 hours** of
observations and are near-duplicates.

Precisely: grouping the 331,185 windows by HARP and differencing consecutive start times gives
327,980 gaps, of which 323,377 are exactly 1.00 h. Minimum 1.00 h, maximum 171.00 h, median 1.00 h.

A random split would scatter near-duplicate windows across train and test and inflate every skill
score — the model would partly be retrieving windows it had already seen. The chronological,
region-disjoint partition split is what prevents it.

*Evidence:* [DATA_CARD.md](DATA_CARD.md) §6, `results/extra/window_gaps.json`.

*Honest footnote:* we did not run the random-split experiment to quantify the inflation. The
argument is structural and quantified by the overlap statistic, not demonstrated empirically.

---

## 5. Why PR-AUC and TSS rather than accuracy?

**Accuracy is uninformative at a 1.31 % base rate.** A model that always forecasts "no major flare"
scores **98.69 %** accuracy on the test partition and detects nothing. Our model scores 93.64 % —
*lower* than the useless one — while detecting 91.6 % of major flares.

- **TSS** = recall − false-alarm rate. It is insensitive to the class ratio, which is why it is the
  standard skill score in flare forecasting; a no-skill forecast scores 0 whatever the balance.
- **PR-AUC** summarises the precision/recall trade-off across all thresholds and, unlike ROC-AUC,
  respects the base rate. ROC-AUC looks flattering here (0.979) because true negatives dominate its
  denominator.

`tests/test_metrics.py::test_tss_is_insensitive_to_the_class_ratio` demonstrates the property, and
`test_always_no_flare_scores_high_accuracy_and_detects_nothing` pins the 98.69 % figure as a test.

---

## 6. What do the baselines show? Does the machine learning earn its place?

This is the question with the most uncomfortable answer, so here it is first.

| | Climatology | Best single feature | This model |
|---|---|---|---|
| Rule | constant p = 0.0199 | `TOTUSJH__max ≥ 1442.37` | LR on 144 features |
| TSS | 0.0000 | **0.8514** | 0.8529 |
| Precision | 0.0000 | **0.1667** | 0.1616 |
| F1 | 0.0000 | **0.2819** | 0.2747 |
| PR-AUC | 0.0131 | 0.3720 | **0.4894** |

**On TSS, a single threshold on one SHARP parameter matches the model**, and it beats the model on
precision and F1. The single feature was chosen honestly — searched over 288 one-feature rules on
the validation partition, then scored once on P5.

**Where the model genuinely wins is ranking quality:** PR-AUC 0.4894 versus 0.3720, a 32 % relative
improvement. That matters because the rest of the system depends on it — the HIGH-risk band
(precision 0.58) needs probabilities ordered well across the whole range, a raw feature value is not
a probability and cannot be recalibrated or banded, and PR-AUC is the metric on which the selected
model beats every other candidate including all three random forests.

So: **the pipeline's value is calibrated ranking and exact explainability, not a higher headline
TSS than a classical single-parameter threshold.** Anyone citing TSS 0.853 as proof that machine
learning was necessary here is overclaiming.

*Evidence:* `results/extra/baselines.json`, `figures/extra/extra6_baselines.png`.

---

## 7. Is this real-time? Could it run operationally?

**No, and it is labelled as such in the interface, the README and the model card.**

It is trained and evaluated only on historical SWAN-SF data from 2010-05-01 to 2018-08-17. It
consumes no live feed. **NOAA GOES monitoring is not implemented and is not a model input.** Nothing
in the repository connects to any external service; the dashboard runs entirely offline from static
JSON.

What an operational version would need, none of which exists here: a live SHARP ingestion path,
latency and availability engineering, monitoring for drift across the solar cycle, recalibration,
and operational validation against an existing service. Listed as future scope in the report, not
as work done.

---

## 8. What would you do with GOES data and deep learning?

Future scope, explicitly **not implemented** — no numbers are claimed for any of it.

- **GOES X-ray flux** is in SWAN-SF as `XR_MAX`/`XR_QUAL` and was deliberately excluded as a
  predictor: it measures flare activity directly and overlaps the prediction window. Using it
  safely would need a strictly causal formulation reading only pre-cutoff flux.
- **Sequence models** (LSTM, temporal convolutions, transformers) would consume the 60 × 24 series
  directly instead of 144 hand-built summaries. Worth trying — though note that the ablation here
  shows the representation is already highly redundant, so the gain is not obvious.
- **Per-event evaluation and a cost-loss analysis** would say more about operational value than any
  further metric tuning.

We would keep the same protocol: partition split, selection on validation, test scored once.

---

## 9. How were the seven models chosen between, and could you have cheated?

The rule was written down before the test partition was scored:

> maximum validation TSS rounded to 3 decimals, tie-break on validation PR-AUC.

`results/selection.json` records the choice, the rule and the timestamp `2026-09-27 07:23:39`.
`src/train_eval.py` writes that file **before** it loads any model for test scoring — the ordering is
visible in the source, not merely asserted.

You can re-derive it: `python -m solarflare evaluate` re-applies the rule to recomputed validation
scores and independently re-picks `I2_LR_temporal_C0.01`, and re-derives both thresholds from the
validation predictions alone, matching to 1e-12.

**Could we have cheated?** Not without leaving a trace. The threshold is not the test-optimal one —
the best threshold on the P5 sweep is 0.5008 with TSS 0.8652, against our 0.5445 with TSS 0.8529.
Fixing the threshold on validation **cost us 0.0123 TSS**. A project that had tuned on the test set
would not have left that on the table, and the selected model is not even the best of the seven on
test TSS.

---

## 10. What does "locked test partition" mean, and how can you prove it was scored once?

It means P5 was not looked at in any way — no metric, no plot, no threshold search — until the
model and threshold were chosen and written to `results/selection.json`.

Four independent traces:

1. **Code ordering.** In `src/train_eval.py` the selection block writes `selection.json` and
   `validation_results.json`, and only the block *after* it loads each model and scores P5.
2. **The threshold is suboptimal on test** (0.0123 TSS left on the table), as above.
3. **The selected model is not the best on test TSS** — four other candidates score higher. A
   test-informed choice would not have produced that.
4. **Everything computed on P5 since then is labelled.** Every file in `results/extra/` carries a
   `posthoc` field stating that it was computed after selection and did not influence it.

What this cannot prove is that nobody ever glanced at P5 — no protocol can. What it does show is
that every decision which *could* have been contaminated demonstrably was not.

---

## 11. Your probabilities — are they calibrated?

**No, and deliberately not.** The model over-forecasts by about **7.3×**: mean predicted probability
0.0964 against a base rate of 0.0131. Brier score 0.0500, worse than the 0.0130 of a constant
forecast that detects nothing.

The cause is `class_weight="balanced"`. On the training partitions, where positives are 1.99 % of
windows, it gives each positive a weight of 25.07 against 0.51 for each negative — positives count
about 49× more heavily.
That is exactly what buys 91.6 % recall.

It does not affect any published metric: TSS, precision, recall, F1, HSS and the confusion matrix
depend only on which side of the threshold a probability falls, and PR-AUC and ROC-AUC depend only
on the ranking. Platt scaling fitted on validation cuts the Brier score to 0.0090 and moves ROC-AUC
by **exactly 0.0** — checked, not assumed. The recalibrated variants are reported separately and are
not the selected model.

*Evidence:* `results/extra/calibration.json`, `figures/extra/extra2_calibration_reliability.png`.

---

## 12. The X-class recall is 100 %. Isn't that suspicious?

It is **19 of 19** X-class windows detected, with a 95 % Wilson interval of **[0.832, 1.000]**.

Nineteen cases is a small sample and the interval says so. We report it with the count attached
every time and never quote it as a headline. The M-class recall, on 971 windows, is 0.9145 with a
much tighter interval of [0.895, 0.931] — that is the figure to trust.

It is at least physically unsurprising: X-class flares come from the largest, most magnetically
complex regions, which are the easiest for any method to flag.

*Evidence:* `results/extra/breakdowns.json`.

---

## 13. Does the dashboard really run the model, or is it a mock-up?

It really runs it. The "Live inference" view reimplements the fitted pipeline in JavaScript from
parameters exported verbatim out of the joblib file, and **this is tested, not asserted**:

- probabilities on **1,000 random P5 feature rows**, JavaScript versus scikit-learn: max absolute
  difference **< 1e-6**;
- feature extraction on **220 real SWAN-SF windows**: agreement to < 1e-6, including the NaN
  patterns;
- plus synthetic edge cases — constant series, exact ramps, all-NaN columns, single finite points.

`tests/test_js_parity.py`, 8 tests, all passing. The browser and the test run the identical file,
`dashboard/model.js`.

The other views are generated from `results/` as well: `tests/test_dashboard.py` drives every view
in a headless browser and asserts the displayed numbers equal `results/test_results.json`.

---

## 14. What is the single biggest weakness of this project?

Two, stated without hedging.

**The candidate search was narrow.** A statistic-family ablation found several variants with
*higher* validation TSS than the selected model — a 24-feature last-value variant reaches 0.8744
against 0.8622. They were not candidates, so the selection was honest, but a wider search under the
same protocol would probably have selected a smaller model. The project should not claim the 144
temporal features were necessary. *Evidence:* `results/extra/ablation.json`.

**The test partition is smaller than it looks.** 75,365 windows sound like a lot, but they come from
758 regions and only 990 positive windows, concentrated in a handful of active regions. The block
bootstrap makes the consequence explicit: the PR-AUC interval is [0.267, 0.657] around a point
estimate of 0.489. Any single-figure result from this project should be read with that width in
mind.

---

## 15. Who did what?

Madhavan G (2104251040518) — pipeline owner: SWAN-SF acquisition, feature extraction, leakage audit,
partition split design, the LR and RF experiments, model selection and test evaluation, result
figures.

Sriram Sivakumar (2104251040971) — applied half: exploratory analysis, risk-scoring and alert logic,
the web dashboard, documentation and reproducibility checks.

Contribution recorded as 50/50.
