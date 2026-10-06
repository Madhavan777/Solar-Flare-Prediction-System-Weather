# The project, explained

Written to be read start to finish by someone who has not seen the code. No background in solar
physics or machine learning is assumed; terms are defined where they first appear.

For the shorter versions: [FEATURES.md](FEATURES.md) is the view-by-view tour,
[JUDGE_DEMO_SCRIPT.md](JUDGE_DEMO_SCRIPT.md) is the timed walkthrough, and
[JUDGES_QA.md](JUDGES_QA.md) is the list of hard questions with evidence.

---

## 1. The problem, in plain terms

The Sun periodically releases a burst of radiation called a **solar flare**. The large ones matter:
they disturb satellites, high-frequency radio, GPS accuracy and, in severe cases, power
distribution. Flares are graded by X-ray brightness on a letter scale — B, C, M, X — where each
letter is ten times stronger than the last. **M and X are the ones that cause trouble.**

Flares come from **active regions**: patches of intensely twisted magnetic field on the Sun's
surface, visible as sunspot groups. NASA's Solar Dynamics Observatory carries an instrument, HMI,
that measures the magnetic field across the Sun's visible surface, and the measurements for each
active region change in characteristic ways before it erupts.

So the question this project answers is:

> Given the last twelve hours of magnetic measurements for one active region, how likely is that
> region to produce an M- or X-class flare in the next twenty-four hours?

## 2. What the system actually does

One **observation window** is twelve hours of recordings for a single active region: 60 readings
taken every 12 minutes, each reading containing 24 different measures of the magnetic field.

Those 60 × 24 numbers are condensed into **144 summary numbers** — for each of the 24 measures,
six descriptions of how it behaved: its final value, its average, how much it varied, its lowest
and highest points, and whether it was trending up or down.

Those 144 numbers go into a trained model, which returns a single probability. That probability is
compared with a fixed cut-off to produce **LOW**, **MODERATE** or **HIGH** risk.

```
12 hours of readings  ->  144 summary numbers  ->  model  ->  probability  ->  risk level
     (60 x 24)                                                                (LOW/MOD/HIGH)
```

## 3. The data

**SWAN-SF**, a published benchmark dataset built from NASA SDO observations and distributed through
the Harvard Dataverse. It contains **331,185 observation windows** covering **2010 to 2018**, from
3,205 distinct active regions.

Major flares are rare. Only **6,234 windows** — about 1.9 % — were followed by an M- or X-class
flare. That imbalance drives almost every design decision that follows, and it is the single most
important thing to understand about the problem.

The label comes from which folder SWAN-SF puts each recording in, never from a measurement the
model can see. Full provenance is in [DATA_CARD.md](DATA_CARD.md).

## 4. The hard part: not fooling yourself

This is where most of the engineering effort went, and it is what distinguishes a careful project
from a lucky one.

**The trap.** Consecutive recordings of the same active region are taken one hour apart, but each
one covers twelve hours. Two neighbouring recordings therefore share eleven of their twelve hours —
they are nearly the same observation twice. We measured how often: **98.60 %** of consecutive
recordings from one region are exactly one hour apart.

If the data were split randomly into training and testing sets, near-identical recordings would land
on both sides. The model would be tested on data it had effectively already seen, and every score
would be flattering and wrong.

**What we did instead.** The dataset is divided into five chronological partitions, and we split by
partition rather than at random:

| Partitions 1–3 | Partition 4 | Partition 5 |
|---|---|---|
| **Training** — the model learns here | **Validation** — the model and its cut-off are chosen here | **Test** — scored once, at the end |
| 204,559 windows | 51,261 windows | 75,365 windows |

No active region appears in more than one partition — all ten pairwise overlaps are zero. So no
recording's near-twin ever crosses the split.

**Five further precautions**, each with an automated test that fails if it is ever broken:

1. Every column that reveals the answer is excluded — the flare-class columns, the X-ray
   measurements, and the position metadata. Only magnetic-field measurements are used.
2. The feature calculation reads only rows inside the window. It cannot see the future.
3. Averages used to fill in missing values, and the scaling applied to each measurement, are
   calculated from the **training partitions alone** — with a companion test proving they differ
   from what an all-data calculation would give, so the check cannot pass vacuously.
4. Both decision cut-offs were fixed on the validation partition before the test partition was ever
   scored.
5. Partition 5 was scored **once**, after the choice of model had been written to disk.

## 5. Choosing the model

Seven candidates were trained: four logistic regressions and three random forests, differing in how
much information they used and how strongly they were regularised.

The selection rule was written down **before** any of them touched the test data:

> pick the model with the highest skill score on the validation partition; break ties on
> precision-recall area.

The winner was a **logistic regression** using all 144 summary numbers with strong regularisation.

It is a deliberately simple model, and that brought a real benefit: because it is linear, the
contribution of each measurement to each individual forecast can be calculated **exactly**. The
contributions plus a constant reproduce the model's own output to fourteen decimal places. More
elaborate models would have needed an approximate explanation method.

## 6. The results

On the 75,365 test recordings, scored once:

| | Value | What it means |
|---|---|---|
| **Recall** | **0.9162** | It caught 92 % of the major flares that happened |
| **Precision** | **0.1616** | Only 16 % of its alerts were followed by a major flare |
| **TSS** | **0.8529** | Skill score: 0 is useless, 1 is perfect |
| **PR-AUC** | **0.4894** | Quality of its ranking, against a 0.013 floor for guessing |
| Accuracy | 0.9364 | **Misleading — see below** |

Of 990 recordings that preceded a major flare, it caught **907** and missed **83**. It raised
**4,707** alerts that were not followed by one.

**Two different "false alarm" numbers, which are easy to confuse.** They differ by more than a
factor of thirteen, so saying the wrong one is a visible error:

| | Formula | Value | In words |
|---|---|---|---|
| False-alarm **ratio** (FAR) | FP / (TP + FP) | **0.8384** | Of the alerts raised, 84 % were not followed by a major flare. This is 1 − precision. |
| False-alarm **rate** (FPR) | FP / (FP + TN) | **0.0633** | Of the quiet recordings, 6 % wrongly triggered an alert. |

The high number describes the alerts; the low number describes the quiet Sun. Both are true and they
answer different questions. The project never uses "FAR" for the rate.

**Why accuracy is the wrong measure.** A system that always says "no flare" scores **98.69 %**
accuracy on this data and detects nothing at all. Our model scores *lower* on accuracy than that
useless system, while catching 92 % of the flares. Quote the two numbers together or not at all.

**How confident are these numbers?** Re-running the evaluation 1,000 times on resampled active
regions gives a 95 % interval of **[0.751, 0.915]** for the skill score. That interval is wide —
wider than the gap between all seven candidate models — which means those models are statistically
indistinguishable on this data. It is the honest reason the logistic regression was chosen on
validation rather than by comparing test scores.

## 7. Being honest about the weaknesses

These are stated in the dashboard, in the model card, and should be stated aloud.

**Most of the alerts are wrong — but not randomly wrong.** Of the 4,707 false alarms, **77.8 %**
were followed by a real B- or C-class flare — a genuine flare, just below the M threshold. Only
22 % landed on a genuinely quiet recording. The model alerts on 49.8 % of C-class recordings but
only 1.67 % of flare-quiet ones, and its average score rises steadily across the flare classes even
though it was only ever told "major" or "not major". It is detecting magnetic activity correctly
and failing to pin down the magnitude boundary. **That does not make 16 % precision acceptable** —
a C-class flare is not an M-class flare — but it is a different failure from guessing.

**A single measurement nearly matches the whole model.** One threshold on one magnetic measurement,
chosen with the same discipline, scores 0.8514 on the skill statistic against the model's 0.8529,
and beats it on precision. The model's genuine advantage is in ranking quality — 0.4894 against
0.3720 — which is what supports the second, stricter HIGH-risk band. The defensible claim is that
the pipeline buys calibrated ranking and exact explanation, **not** a higher headline score than a
classical threshold.

**The search was narrow.** A later experiment found simpler variants that score *higher* on the
validation partition than the selected model. They were not candidates, so the selection was
honest, but the project should not claim that all 144 numbers were necessary.

**The probabilities are not literal chances.** During training, each recording that preceded a major
flare was counted about 49 times more heavily than one that did not, to stop the model simply
learning to always say "no". That is what buys the 92 % recall, and the side effect is that the
model over-forecasts by about 7.3 times. Read the number as a ranking and against the fixed
cut-offs, never as a percentage chance.

**It is not operational.** No live feed, no connection to any observatory, no validation beyond 2018
or beyond this one instrument. It is a prototype demonstrating a method.

## 8. The engineering behind it, and what each part buys

The modelling was finished before this work began. What follows is the engineering that makes the
result *checkable* — which, for a project being examined, is most of what separates a believable
claim from an assertion.

| # | What was built | What it achieves |
|---|---|---|
| 1 | **One command-line entry point.** `python -m solarflare <cmd>` covers every operation: `env`, `audit`, `evaluate`, `train`, `analysis`, `figures`, `demo-data`, `dashboard`, `screenshots`, `test`, `check`, `all`. | Anyone can reproduce any part of the project without reading the source. There is one supported path, so instructions cannot rot. |
| 2 | **A frozen-artefact guard.** `assert_not_frozen()` runs before every file write and raises on anything under `models/`, `results/` or the report's figures. | The published evidence cannot be destroyed by accident. Two of the original scripts would have overwritten it; now the attempt fails loudly instead of succeeding quietly. |
| 3 | **Full reproduction verification.** `solarflare evaluate` reloads each trained model, re-scores validation and test, re-applies the selection rule from scratch and re-derives both thresholds — 375 checks. | Turns "the results are correct" from a claim into something an examiner can run in two minutes. Every published figure matches to better than one part in a billion; the stored probabilities to 1.3 × 10⁻¹⁵. |
| 4 | **Nine post-hoc analyses**, each stamped "not used for selection" inside its own output file. | Supplies the confidence intervals, calibration evidence, baselines and error analysis the headline numbers need — while making it impossible to mistake them for selection evidence, even if a table is lifted out of the repository. |
| 5 | **186 automated tests**, including leakage tests that fail if an excluded column reaches the model, if a region crosses the split, or if preprocessing is fitted on anything but the training partitions. | The protocol is enforced by machinery rather than by memory. One test deliberately proves the preprocessing statistics *differ* from an all-data fit, so the check cannot pass vacuously. |
| 6 | **An automated consistency checker**, 392 checks in seven passes, comparing every number in the documentation and the dashboard against the stored results. | Documentation cannot drift away from the evidence. It has already caught real errors, including a class-weight figure that was wrong in six places. |
| 7 | **A pinned, reproducible environment.** scikit-learn 1.8.0 is a hard requirement with a runtime check; Docker, CI, pre-commit, ruff and black are configured. | The saved models are scikit-learn pickles and will not reproduce under another minor version. The check fails immediately with an actionable message instead of producing subtly wrong numbers. |
| 8 | **A dashboard generated entirely from the results.** Not one metric is typed into the page; the browser-side model is parity-tested against scikit-learn to better than 10⁻⁶. | The demonstration cannot contradict the report, because it holds no independent copy of the numbers. |
| 9 | **Report mode.** The views added after the report was frozen are hidden on demand. | Re-running the screenshot step still reproduces the report's Figures 5.1–5.6 byte-identically, so the document and the live system stay in agreement. |
| 10 | **Examiner-facing documentation** — model card, data card, reproducibility notes, decision log, this explainer, a timed demo script and fifteen answered questions. | Every hard question has an answer that points at a file or a number, prepared in advance rather than improvised. |

Each of these is verifiable: run `python -m solarflare test` and `python scripts/check_consistency.py`.

## 9. What the dashboard may and may not claim

The dashboard is a demonstration of a trained model on recorded data. It is held to the same honesty
standard as the written report, and a test fails the build if the page drifts from it.

**It may say:** that it scores recorded observations from 2010–2018; that the model runs in the
browser on a window you supply; that the Replay view steps through real stored predictions in the
order the observations were taken; that it caught 907 of 990 major flares and raised 4,707 false
alarms.

**It may not say, and does not:** that it watches the Sun; that anything is real-time, live or
current; that it is an operational or production warning service; that it uses NOAA or GOES
monitoring; that its probabilities are literal chances; or that any finding is causal.

The word "live" appears in one place — the **Live Inference** view — where it means the model
computes in your browser as you watch. It is live *computation*, not live *data*. If anyone reads it
the other way, correct them immediately; `docs/FEATURES.md` gives the wording.

## 10. How to see it for yourself

Press **Ctrl + Alt + S**, or run `python -m solarflare dashboard` and open
`http://localhost:8791/index.html`.

The **Replay** view is the quickest way to understand the system: ten days of a real active region
in September 2017, played back in the order the observations were recorded. The forecast climbs
from near zero as the magnetic field grows more complex, and holds in the high-risk band through the
X9.3 and X1.3 flares — with two frames marked MISSED, where a genuine pre-flare recording scored
below the cut-off. Those are left in on purpose.

## 11. Can the numbers be trusted?

Everything in this document can be recomputed:

```powershell
python -m solarflare evaluate     # 375 checks against the stored results
python -m solarflare test         # the full test suite
python scripts/check_consistency.py
```

`evaluate` reloads each trained model, re-scores the validation and test partitions, re-applies the
selection rule from scratch, and re-derives both cut-offs. It reproduces every published figure to
better than one part in a billion, and the confusion matrix exactly.

The consistency checker goes further: it compares every number printed in these documents and shown
in the dashboard against the stored results, and fails if any disagrees. It also checks for
overclaiming language. So the documentation cannot quietly drift away from the evidence.

## 12. Credits

**Madhavan G** (2104251040518) — data acquisition, feature extraction, the leakage audit, the split
design, the model experiments, selection and test evaluation, and the result figures.

**Sriram Sivakumar** (2104251040971) — exploratory analysis, the risk and alert logic, the web
dashboard, documentation and reproducibility checks.

Contribution recorded as 50/50. Project-Based Learning project, Machine Learning course,
B.E. Computer Science and Engineering, Chennai Institute of Technology (Autonomous), Chennai – 69,
affiliated to Anna University, Chennai. Academic year 2026–2027.

Mentor and Project Co-ordinator: **R. Poornima Lakshmi, M.E.**, Assistant Professor, Department of
CSE.
