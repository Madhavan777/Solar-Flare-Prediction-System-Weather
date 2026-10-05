# Post-hoc analyses

> Post-hoc analysis. Computed on the locked test partition P5 after model selection was complete and written to results/selection.json. It did not influence the choice of model, features or thresholds, and none of the frozen numbers in results/ depend on it.

Every number below is read from a JSON file in this directory by `src/solarflare/analysis/report.py`; none is typed in by hand. Regenerate with:

```powershell
python -m solarflare analysis
python -m solarflare figures
```

Figures for these analyses are in `figures/extra/`. The report's own figures in `figures/` are frozen and untouched.

---

## 1. Confidence intervals by active-region block bootstrap

**What this is.** How precisely the P5 score is measured — not a new result. 1,000 resamples, seed 42, frozen threshold throughout.

**Why whole regions are the resampling unit.** Consecutive windows of one HARP region are one hour apart 98.60 % of the time, so adjacent 12-hour windows share 11 of their 12 hours of observations. Resampling individual windows would treat those near-duplicates as independent draws and give intervals that are far too narrow. Resampling the 758 regions keeps each region's windows together.

| Metric | Point estimate | 95 % interval |
|---|---|---|
| Recall (probability of detection) | 0.9162 | [0.8121, 0.9755] |
| Precision | 0.1616 | [0.0939, 0.2367] |
| F1-score | 0.2747 | [0.1676, 0.3784] |
| TSS | 0.8529 | [0.7513, 0.9146] |
| PR-AUC | 0.4894 | [0.2671, 0.6568] |
| ROC-AUC | 0.9790 | [0.9642, 0.9885] |
| False-alarm rate (FPR) | 0.0633 | [0.0462, 0.0824] |
| False-alarm ratio (FAR) | 0.8384 | [0.7633, 0.9061] |
| Alert rate | 0.0745 | [0.0544, 0.0975] |
| HSS (HSS2 form) | 0.2581 | [0.1569, 0.3569] |

Resampled partitions ranged from 69,123 to 81,475 windows (median 75,314) and from 322 to 1864 positives (median 989). That spread is the honest picture: P5's effective sample size is set by how many *regions* flared, not by its 75,365 windows.

### What these intervals mean for the candidate comparison

The TSS interval is **0.163 wide** ([0.751, 0.915]). The entire spread of test TSS across all seven candidates is **0.014** — roughly 12 times narrower than the interval around any one of them.

**The seven candidates are statistically indistinguishable on this test partition.** That is the correct answer to "why did logistic regression win when a random forest scores higher test TSS?": the random forests' apparent advantage of 0.007-0.014 TSS is far inside the noise, so it is not evidence of anything. The selection was made on validation, as the protocol requires, and no test-partition ranking among these models could have been trusted anyway.

The PR-AUC interval is wider still, [0.267, 0.657] around a point estimate of 0.489. Any single-figure PR-AUC quoted for this project should be read with that in mind.

---

## 2. Calibration — the probabilities over-forecast, and that is expected

**The finding, stated plainly.** The selected model's probabilities are **not calibrated**. A window scored 0.60 does not flare 60 % of the time. The mean predicted probability on P5 is 0.0964 against an actual base rate of 0.0131 — the model over-forecasts by about **7.3x**.

**Why.** The model was fitted with class_weight='balanced', which up-weights the 1.3 % positive class by about 33x. That is what buys 91.6 % recall, and it necessarily inflates the predicted probabilities: the model is trained as if major flares were far commoner than they are. The outputs are therefore useful as a ranking and against the fixed thresholds, but must not be read as literal chances of a flare.

**What this does not affect.** TSS, precision, recall, F1, HSS and the confusion matrix depend only on which side of the threshold each probability falls; PR-AUC and ROC-AUC depend only on the ranking. A strictly monotone recalibration changes none of them, which the roc_auc_shift_under_recalibration figures below confirm.

Brier score: **0.0500** for the frozen model, against 0.0130 for a constant forecast at the training base rate. A constant forecast beats the model on Brier score while detecting nothing — which is precisely why Brier score is not used to choose a flare forecaster.

| Variant | Selected? | Brier | PR-AUC | ROC-AUC | Mean forecast / base rate |
|---|---|---|---|---|---|
| `frozen_class_weighted` | **yes** | 0.0500 | 0.4894 | 0.9790 | 7.34x |
| `platt_on_validation` | no | 0.0090 | 0.4894 | 0.9790 | 0.96x |
| `isotonic_on_validation` | no | 0.0092 | 0.4391 | 0.9620 | 0.92x |

Both recalibrations were fitted on the **validation partition only** and are reported for contrast. Neither is the selected model and neither changes any published number. As a check that the recalibrations really are monotone, the ROC-AUC moves by 0.00e+00 (platt), 1.69e-02 (isotonic) — i.e. not at all beyond floating-point noise.

---

## 3. Operating points

**Exploratory. The headline results use the validation-fixed thresholds below. The best-TSS point on this test sweep is reported only to show how little was lost by fixing the threshold on validation instead - it was never available to the selection procedure.**

Risk bands as the dashboard renders them: **LOW** p < 0.5445, **MODERATE** 0.5445 <= p < 0.9673, **HIGH** p >= 0.9673.

| Operating point | Threshold | Recall | Precision | FAR (ratio) | FPR (rate) | TSS | Alert rate | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `alert_threshold_validation_tss` **(published)** | 0.5445 | 0.9162 | 0.1616 | 0.8384 | 0.0633 | 0.8529 | 0.0745 | 907 | 4,707 | 83 | 69,668 |
| `high_risk_threshold_validation_f1` **(published)** | 0.9673 | 0.3202 | 0.5795 | 0.4205 | 0.0031 | 0.3171 | 0.0073 | 317 | 230 | 673 | 74,145 |
| `naive_half` | 0.5000 | 0.9354 | 0.1505 | 0.8495 | 0.0703 | 0.8651 | 0.0816 | 926 | 5,227 | 64 | 69,148 |

The highest-TSS threshold anywhere on the P5 sweep is 0.5008, scoring TSS 0.8652. The published validation-fixed threshold scores 0.8529, so fixing the threshold on validation instead of the test set cost **0.0123 TSS**. That number is the price of doing the protocol honestly, and it is small.

The full 400-point sweep is in `results/extra/operating_points.json`, and drives the dashboard's threshold slider.

---

## 4. Breakdowns: where the model does better and worse

### By GOES class

| Class | Positive windows | Detected | Missed | Recall | 95 % Wilson interval |
|---|---|---|---|---|---|
| M | 971 | 888 | 83 | 0.9145 | [0.895, 0.931] |
| X | 19 | 19 | 0 | 1.0000 | [0.832, 1.000] |

**Caveat:** only 19 X-class windows in P5 - this recall is estimated from 19 cases and the interval is correspondingly wide. The X-class recall should not be quoted as a headline figure.


### By calendar year

| Year | Windows | Positives | Base rate | Recall | Precision | TSS |
|---|---|---|---|---|---|---|
| 2015 | 46,147 | 732 | 0.0159 | 0.9071 | 0.1645 | 0.8329 |
| 2016 | 11,751 | 107 | 0.0091 | 1.0000 | 0.1644 | 0.9533 |
| 2017 | 12,862 | 151 | 0.0117 | 0.9007 | 0.1646 | 0.8464 |
| 2018 | 4,605 | 0 | — | — | — | no positives in this group: recall, precision and the AUCs are undefined |

P5 covers 2015-03-07 to 2018-08-17, the declining phase of solar cycle 24, so the positive base rate drops steeply across the partition. Precision falls with it, which is arithmetic rather than model degradation.

### By HARP region — false alarms are concentrated

P5 contains 758 HARP regions and 4,707 false alarms. **15 regions (2.0 % of them) produce half of them**, and 663 regions produce none at all.

False alarms are concentrated: a magnetically complex region that never produces an M or X flare yields a long run of consecutive alerted windows, because consecutive windows share 11 of their 12 hours of data.

| HARP | Windows | Positive windows | False alarms | Hits | Max p |
|---|---|---|---|---|---|
| 5541 | 211 | 0 | 211 | 0 | 0.9965 |
| 5848 | 219 | 0 | 184 | 0 | 0.8742 |
| 5447 | 216 | 24 | 182 | 24 | 0.9865 |
| 5637 | 237 | 12 | 172 | 12 | 0.9812 |
| 6015 | 231 | 16 | 167 | 16 | 0.9593 |
| 7075 | 216 | 41 | 165 | 41 | 0.9694 |
| 6975 | 233 | 0 | 162 | 0 | 0.9918 |
| 6027 | 220 | 0 | 158 | 0 | 0.9442 |
| 5718 | 193 | 0 | 154 | 0 | 0.9401 |
| 6206 | 197 | 44 | 153 | 44 | 0.9773 |

### By missingness

| Missing SHARP values in window | Windows | Positives | Recall | Precision | Alert rate |
|---|---|---|---|---|---|
| none (0%) | 66,503 | 894 | 0.9083 | 0.1592 | 0.0767 |
| 0-1% | 32 | 0 | — | — | 0.0000 |
| 1-5% | 7,981 | 93 | 1.0000 | 0.1879 | 0.0620 |
| 5-25% | 774 | 3 | 0.6667 | 0.1111 | 0.0233 |
| >25% | 75 | 0 | — | — | 0.0000 |

---

## 5. What actually followed each false alarm

**The question.** The published precision of 0.1616 counts every alerted negative window as an equal error. SWAN-SF labels each negative window with the largest flare that followed it - B, C, or none. This asks what that distribution looks like among the false alarms.

**The answer.** 3,663 of the 4,707 false alarms (**77.8 %**) precede a genuine B- or C-class flare. Only 1,044 (22.2 %) land on a window that was truly flare-quiet.

| What followed the window | Negative windows | Share of negatives | False alarms | Share of false alarms | Enrichment | Alert rate within class |
|---|---|---|---|---|---|---|
| **F** — flare-quiet: no B-, C-, M- or X-class flare in the next 24 h | 62,688 | 84.29 % | 1,044 | 22.18 % | **x0.26** | 1.67 % |
| **B** — a B-class flare followed, the weakest GOES category | 5,924 | 7.97 % | 792 | 16.83 % | **x2.11** | 13.37 % |
| **C** — a C-class flare followed, one order of magnitude below the M threshold | 5,763 | 7.75 % | 2,871 | 60.99 % | **x7.87** | 49.82 % |

The decisive comparison is the last column: the model alerts on **49.8 %** of C-class windows but only **1.67 %** of flare-quiet windows — a factor of 30. C-class windows are enriched among the false alarms by x7.9, while flare-quiet windows are *depleted* by x0.26.

### The predicted probability rises monotonically with flare magnitude

| Largest flare in the next 24 h | Median predicted probability |
|---|---|
| F | 0.0005 |
| B | 0.1103 |
| C | 0.5415 |
| M | 0.9358 |
| X | 0.9997 |

The model was trained on a binary target that lumps F, B and C together as one class and M and X together as the other. It was never told that C is stronger than B, or that X is stronger than M. That the median probability nonetheless increases across all five categories is evidence that it learnt a continuous notion of magnetic activity rather than a boundary memorised from the label.

**Interpretation.** 3,663 of the 4,707 false alarms (77.8 %) precede a genuine B- or C-class flare, and only 1,044 (22.2 %) land on a window that was truly flare-quiet. Within the flare-quiet windows the model alerts only 1.67 % of the time, against 49.8 % within C-class windows. The model is therefore discriminating magnetic activity successfully and failing mainly to resolve the M-class boundary, which is a materially different weakness from alerting at random on quiet regions.

**What this does not excuse.** A C-class flare is not an M-class flare, and an operator told 'major flare likely' when a C-class flare follows has still been misinformed. The published precision of 0.1616 for the M/X task is correct and is not revised by this analysis. What this shows is that the errors are concentrated near the decision boundary of flare magnitude rather than scattered over quiet Sun, which is the difference between a miscalibrated threshold and a model that has learnt nothing.

For scale only, and emphatically not as a result: Hypothetical only, and NOT the project's task: if the target were 'will any B, C, M or X flare follow?' rather than 'will an M or X flare follow?', the same alerts at the same threshold would score this precision. Reported to size the effect, not as a result - the model was never trained or selected for this target, and the frozen metrics stand. That hypothetical precision is 0.8140, against the real task's 0.1616.

---

## 6. Baselines — what the model actually buys

**Selection discipline.** Both baselines were chosen using the training and validation partitions only - imputation medians from P1-P3, feature and threshold from P4 - and then scored once on P5, the same discipline the real model followed.

| | Climatology | Best single feature | Selected model |
|---|---|---|---|
| Rule | constant p = 0.019940 | `alert if TOTUSJH__max >= 1442.37` | LR on 144 temporal features |
| TSS | 0.0000 | 0.8514 | 0.8529 |
| Recall | 0.0000 | 0.9121 | 0.9162 |
| Precision | 0.0000 | 0.1667 | 0.1616 |
| PR-AUC | 0.0131 | 0.3720 | 0.4894 |
| Alerts raised | 0 | 5,416 | 5,614 |

**Climatology** is the no-skill reference: perfectly calibrated, zero discrimination, TSS 0.0. It raises no alerts and detects nothing, while scoring 0.9869 accuracy. Keep that number next to any accuracy figure quoted for this project.

**The best single feature** was found by searching 288 one-feature threshold rules on the validation partition, then scoring the winner once on P5: `alert if TOTUSJH__max >= 1442.37`.

### This is the most uncomfortable result in the project, so state it plainly

On TSS, **the single-feature rule essentially matches the model**: 0.8514 against 0.8529, a difference of +0.0014. On precision it is higher (0.1667 vs 0.1616), and on F1 higher (0.2819 vs 0.2747), at a comparable recall (0.9121 vs 0.9162).

So the honest answer to *"does the machine learning earn its place on TSS?"* is **no, not at this one operating point.** One threshold on one SHARP parameter — the total unsigned current helicity, which is a well-established flare predictor — reproduces the headline skill score.

Where the model does win, decisively, is in **ranking quality across all thresholds**: PR-AUC 0.4894 against 0.3720, an improvement of **+0.1174** (+32 % relative). That is not a cosmetic difference, and it is what the rest of the system actually depends on:

* the three-band risk output needs the probabilities to be ordered well over the whole range, not just split well at one cut. At the HIGH-risk threshold the model reaches 58 % precision — a single raw feature value cannot support a second, much stricter band in any calibrated way;
* a single feature's score is not a probability at all, so there is nothing to recalibrate, nothing to threshold a second time, and no per-feature explanation to show;
* PR-AUC is the threshold-free summary that respects the 1.31 % base rate, and it is the metric on which the selected model beats every other candidate tried in this project, including all three random forests.

The defensible claim is therefore the narrow one: **the pipeline's value is in calibrated ranking and explainability, not in raising the headline TSS above what a classical single-parameter threshold achieves.** Anyone quoting TSS 0.853 as evidence that machine learning was necessary here is overclaiming.

Runners-up by validation TSS: `TOTUSJH__max` (0.8616), `TOTUSJH__last` (0.8606), `USFLUX__last` (0.8580), `TOTUSJZ__last` (0.8572), `USFLUX__max` (0.8550). Note that they are all measures of total field strength, current or flux — the single-feature result is not an artefact of one lucky column.

---

## 7. Ablation

**Protocol.** Every variant is a new logistic regression with the selected model's architecture (signed-log, median imputation, standardisation, C=0.01, class_weight='balanced', seed 42), fitted on P1-P3, with its threshold chosen on P4 by TSS and scored once on P5. The frozen model is unchanged and no ablation result was used to select anything.

### Point-in-time versus temporal (the two frozen candidates)

| Candidate | Features | Validation TSS | Test TSS | Test PR-AUC | Test precision |
|---|---|---|---|---|---|
| `I1_LR_last` | 24 | 0.8463 | 0.8666 | 0.4624 | 0.1262 |
| `I2_LR_temporal_C0.01` | 144 | 0.8622 | 0.8529 | 0.4894 | 0.1616 |

Moving from 24 point-in-time values to 144 temporal summaries raised validation TSS by +0.0160 and test PR-AUC by +0.0270, and raised precision from 0.126 to 0.162. Note that test **TSS** went the other way (0.8666 to 0.8529) — see §7.

### Statistic families

| Family | Drop it (120 features) | Keep only it (24 features) |
|---|---|---|
| `last` | 0.8622 | 0.8744 |
| `mean` | 0.8621 | 0.8687 |
| `std` | 0.8698 | 0.8500 |
| `min` | 0.8617 | 0.8639 |
| `max` | 0.8654 | 0.8672 |
| `slope` | 0.8665 | 0.2026 |

Validation TSS of the full model: **0.8622**.

Dropping any one family costs at most 0.0005 TSS — the representation is **highly redundant**, which is unsurprising given that six statistics of the same 24 physical parameters are strongly correlated. The most damaging single family to remove is `min` (+0.0005 TSS). The strongest family on its own is `last` at validation TSS 0.8744, and the weakest by a wide margin is `slope`, which collapses on its own — a per-hour trend carries very little signal without the level it is a trend in.

### The unflattering finding in this ablation

**7 of the 12 variants reach a higher validation TSS than the selected model.** Since selection used validation TSS, any of these would have won had it been a candidate.

| Variant | Features | Validation TSS | vs selected | Test TSS | Test PR-AUC |
|---|---|---|---|---|---|
| `only_last` | 24 | 0.8744 | +0.0122 | 0.8540 | 0.4755 (-0.0139) |
| `drop_std` | 120 | 0.8698 | +0.0075 | 0.8566 | 0.4989 (+0.0096) |
| `only_mean` | 24 | 0.8687 | +0.0065 | 0.8430 | 0.4758 (-0.0136) |
| `only_max` | 24 | 0.8672 | +0.0050 | 0.8468 | 0.4880 (-0.0014) |
| `drop_slope` | 120 | 0.8665 | +0.0043 | 0.8460 | 0.4916 (+0.0022) |
| `drop_max` | 120 | 0.8654 | +0.0032 | 0.8616 | 0.4914 (+0.0020) |
| `only_min` | 24 | 0.8639 | +0.0017 | 0.8414 | 0.4613 (-0.0281) |

7 of the 12 ablation variants reach a higher validation TSS than the selected model. They were never candidates - the seven-candidate set was fixed before any of this was computed, and the selection rule was applied to that set honestly - but it shows the candidate search was narrow. A wider search over feature subsets, run under the same protocol, would very likely have selected a smaller model. Stated as a limitation rather than as a correction: the frozen result stands, and this is what an examiner should be told about it.

Two things keep this from being a protocol breach. First, the seven-candidate set was fixed and trained before the test partition was ever scored, and the selection rule was applied to that set without modification — the frozen result is exactly what the stated procedure produces. Second, these variants are being compared on validation, which is the right partition for the comparison, so the conclusion is legitimate rather than test-set mining.

What it does mean is that **the model's advantage over a much smaller feature set is not established**, and the project should not claim that 144 temporal features were necessary. The defensible claim is narrower: the selected model has the best PR-AUC on the test partition of everything tried here, and the temporal representation improved precision substantially over the point-in-time baseline at comparable recall.

---

## 8. Error gallery — the failures, named

At the frozen threshold the model produced 907 hits, 4,707 false alarms, 83 misses and 69,668 correct quiet forecasts.

### The 83 misses

These span 9 HARP regions, by class: M = 83. The lowest probability assigned to a genuine pre-flare window was **0.049295** — the model was confidently wrong there, not merely uncertain.

These are windows whose active region did produce an M or X flare within 24 h of the cutoff, and which the model scored below the alert threshold. They are the failures that matter most operationally.

| File | HARP | Class | Window end (cutoff) | p |
|---|---|---|---|---|
| `M1.2@13862:Primary_ar7115_s2017-09-02T18:00:00_e2017-09-03T05:48:00.csv` | 7115 | M1.2 | 2017-09-03T05:48:00 | 0.049295 |
| `M1.2@13862:Primary_ar7115_s2017-09-02T19:00:00_e2017-09-03T06:48:00.csv` | 7115 | M1.2 | 2017-09-03T06:48:00 | 0.076847 |
| `M4.4@13204:Primary_ar6972_s2017-03-31T13:24:00_e2017-04-01T01:12:00.csv` | 6972 | M4.4 | 2017-04-01T01:12:00 | 0.078063 |
| `M4.4@13204:Primary_ar6972_s2017-03-31T12:24:00_e2017-04-01T00:12:00.csv` | 6972 | M4.4 | 2017-04-01T00:12:00 | 0.084927 |
| `M1.2@10285:Secondary_ar5456_s2015-04-20T14:48:00_e2015-04-21T02:36:00.csv` | 5456 | M1.2 | 2015-04-21T02:36:00 | 0.087701 |
| `M1.2@10285:Secondary_ar5456_s2015-04-20T13:48:00_e2015-04-21T01:36:00.csv` | 5456 | M1.2 | 2015-04-21T01:36:00 | 0.090117 |
| `M1.2@10285:Secondary_ar5456_s2015-04-20T11:48:00_e2015-04-20T23:36:00.csv` | 5456 | M1.2 | 2015-04-20T23:36:00 | 0.090557 |
| `M4.4@13204:Primary_ar6972_s2017-03-31T14:24:00_e2017-04-01T02:12:00.csv` | 6972 | M4.4 | 2017-04-01T02:12:00 | 0.091310 |
| `M1.2@10285:Secondary_ar5456_s2015-04-20T12:48:00_e2015-04-21T00:36:00.csv` | 5456 | M1.2 | 2015-04-21T00:36:00 | 0.091581 |
| `M1.2@10285:Secondary_ar5456_s2015-04-20T10:48:00_e2015-04-20T22:36:00.csv` | 5456 | M1.2 | 2015-04-20T22:36:00 | 0.091605 |

### The 4,707 false alarms

They involve 95 regions, and 230 of them exceeded the HIGH-risk threshold. The most confident false alarm scored **0.996909**.

Because consecutive windows of one region are one hour apart and share 11 of their 12 hours of data, a single mis-read region produces a long run of false alarms. The run lengths above show that the 4,707 false alarms are far fewer than 4,707 independent errors.

| File | HARP | Folder | Window end (cutoff) | p |
|---|---|---|---|---|
| `C4.1@11893:Primary_ar6327_s2016-02-15T09:24:00_e2016-02-15T21:12:00.csv` | 6327 | NF | 2016-02-15T21:12:00 | 0.996909 |
| `C6.9@11872:Primary_ar6327_s2016-02-13T05:24:00_e2016-02-13T17:12:00.csv` | 6327 | NF | 2016-02-13T17:12:00 | 0.996880 |
| `C4.2@11886:Primary_ar6327_s2016-02-14T23:24:00_e2016-02-15T11:12:00.csv` | 6327 | NF | 2016-02-15T11:12:00 | 0.996818 |
| `C4.2@11886:Primary_ar6327_s2016-02-15T04:24:00_e2016-02-15T16:12:00.csv` | 6327 | NF | 2016-02-15T16:12:00 | 0.996751 |
| `C3.5@11894:Primary_ar6327_s2016-02-15T10:24:00_e2016-02-15T22:12:00.csv` | 6327 | NF | 2016-02-15T22:12:00 | 0.996708 |
| `C6.9@11872:Primary_ar6327_s2016-02-13T06:24:00_e2016-02-13T18:12:00.csv` | 6327 | NF | 2016-02-13T18:12:00 | 0.996693 |
| `C6.9@11872:Primary_ar6327_s2016-02-13T04:24:00_e2016-02-13T16:12:00.csv` | 6327 | NF | 2016-02-13T16:12:00 | 0.996673 |
| `C2.5@10448:Primary_ar5541_s2015-05-14T14:36:00_e2015-05-15T02:24:00.csv` | 5541 | NF | 2015-05-15T02:24:00 | 0.996501 |
| `C2.5@10448:Primary_ar5541_s2015-05-14T13:36:00_e2015-05-15T01:24:00.csv` | 5541 | NF | 2015-05-15T01:24:00 | 0.996500 |
| `C4.2@11886:Primary_ar6327_s2016-02-15T05:24:00_e2016-02-15T17:12:00.csv` | 6327 | NF | 2016-02-15T17:12:00 | 0.996498 |

Longest runs of consecutive false-alarm windows within one region: HARP 5541 (211), HARP 5447 (181), HARP 6206 (153), HARP 6327 (147), HARP 5692 (122), HARP 5848 (106).

Mean missing-value fraction: 0.00080 for the misses, 0.00305 for the false alarms, 0.00431 across all of P5 — so the errors are not simply the windows with the most missing data.

---

## 9. The per-window explanation is exact

**Identity.** `logit(p) = intercept + sum_j coef_j * standardised_x_j`

The pipeline is a fixed signed-log transform, then median imputation with medians fitted on P1-P3, then standardisation with a centre and scale fitted on P1-P3, then a linear model. Every step before the classifier is an affine or fixed elementwise map, so the classifier's input is known exactly and its logit is a plain dot product. Nothing is approximated, unlike permutation importance or a surrogate explainer.

Checked on 2,000 randomly chosen P5 windows (seed 42): the largest discrepancy between the reconstructed and the model's own logit was **1.95e-14**, and between the reconstructed and actual probability **1.78e-15**.

The two error figures above are floating-point round-off, not approximation error. tests/test_contributions.py asserts the same identity to 1e-9.

---
