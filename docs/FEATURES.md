# What the dashboard does, view by view

A guide for presenting the demonstrator. Pair it with
[JUDGE_DEMO_SCRIPT.md](JUDGE_DEMO_SCRIPT.md) for the timed walkthrough and
[JUDGES_QA.md](JUDGES_QA.md) for the hard questions.

---

## The project in five lines

The Sun occasionally throws out a burst of radiation called a solar flare, and the big ones
disrupt satellites, radio communication, GPS and power grids. Before a flare happens, the magnetic
field in that part of the Sun behaves in a measurable way. This project reads twelve hours of those
magnetic measurements for one active region and estimates how likely that region is to produce a
major flare in the next twenty-four hours. It learned the pattern from eight years of historical
observations, and it catches about nine out of every ten major flares — at the cost of raising a
fair number of alerts that come to nothing.

---

## The one thing to be clear about: "live" does not mean live data

The **Live Inference** view runs the trained model **in the browser, as you watch**. That is live
*computation*. It is **not** a live feed of the Sun.

This system has never been connected to any live data source. It is trained and evaluated only on
historical SWAN-SF observations from 2010-05-01 to 2018-08-17. There is no NOAA or GOES monitoring,
no ingestion path, and no operational validation.

If anyone hears "live" and pictures a monitoring console, correct it immediately:

> Live inference means the real trained model executes in the browser on a window you hand it. It is
> live computation, not live data. This system has never been connected to a live feed.

| Do not say | Say instead |
|---|---|
| real-time, monitoring, tracking | historical, retrospective, offline |
| operational, warning service | academic prototype, demonstrator |
| "it watches the Sun" | "it scores a recorded observation window" |

---

## What the thing actually is

A **static, offline, single-page demonstrator**. No server logic, no database, no network request of
any kind — enforced by a test that fails if an external reference appears in the page.

Everything on screen is generated from `results/` by `python -m solarflare demo-data`. Not one
metric, threshold or count is typed into the HTML. That is the claim worth making: **the dashboard
cannot contradict the published results, because it holds no independent copy of them.**

---

## The nine views

### 1. Overview
The problem, the dataset, the 12 h → 24 h forecast framing, the pipeline and the selected model.
Carries the "Academic Prototype" badge from the first screen.

### 2. Forecast
The observation window being scored: the HARP region, the window start, the prediction cutoff and
the 24 h window that follows, plus the three counts that define the representation — 24 SHARP
parameters, 60 records, 144 engineered features.

**Run Forecast** scores that window with the real browser-side model and opens the result.

### 3. Prediction
The probability as a gauge, the model's classification, the decision threshold that produced it,
and the ground truth shown for verification. The side card names the flare that actually followed.

### 4. Risk & Alert
The three bands — LOW, MODERATE, HIGH — generated from the two thresholds in
`results/selection.json`, with the alert wording and the prototype caveat.

### 5. Explainability
The largest standardised coefficients as signed bars.

**The point to make:** because the final model is linear, these contributions are *exact*. They are
not an approximation like a surrogate explainer or permutation importance. The contributions plus
the intercept reproduce the model's own log-odds to fourteen decimal places.

### 6. Model Evaluation
Six headline metrics, **each with its 95 % confidence interval**, and the confusion matrix on the
locked test partition. The notes panel states the precision plainly and carries the finding about
what actually followed the false alarms.

### 7. Explore Windows — the strongest view
Twenty real windows from the locked test partition, chosen by a seeded, documented rule that
deliberately includes the failures: hits, misses, false alarms and correct quiet forecasts
together. Each shows the raw 12-hour SHARP series, the exact per-feature contributions, and the
true outcome **hidden behind a Reveal button** so the forecast can be read first.

**Say this out loud:** the selection rule is seeded and written into the page, and the misses are
there on purpose. Showing the failures earns more credit than any metric.

### 8. Live Inference
Load a SWAN-SF window file and the browser runs the full pipeline — signed-log, median imputation,
standardisation, logistic regression — from parameters exported out of the joblib file.

It is checked rather than asserted: the browser agrees with scikit-learn to better than 1e-6 on
1,000 real test-partition rows and on 220 real windows, plus synthetic edge cases. A built-in
example means the view works even with no file to hand.

### 9. Operating Point
A threshold slider over the real test-partition predictions, with the confusion matrix updating as
it moves, the validation-fixed point marked, and an "Exploratory" banner.

**The line to deliver:** fixing the threshold on validation rather than picking the best point on
this curve cost 0.0123 TSS, and that was left on the table deliberately.

---

## Supporting features

| Feature | Why it matters on the day |
|---|---|
| **3-minute guided tour** | Seven timed steps with captions, matching the demo script. Removes any risk of fumbling navigation in front of the panel. |
| **Ctrl + Alt + S** | Opens the dashboard from anywhere in Windows — no terminal visible to the examiners. |
| **Works offline** | No CDN, no external request, asserted by a test. It still works if the venue's network fails. |
| **Deep links** | `#explore`, `#live`, `#operate` open a view directly, and a refresh does not lose your place. |
| **Prints cleanly** | Black on white, one view per page, if an examiner wants paper. |
| **Keyboard and screen-reader support** | Visible focus throughout, tab semantics, labelled controls, AA contrast. |
| **Report mode** | Hides the three added views so the six original ones still reproduce the report's Figures 5.1-5.6 exactly. |

---

## If asked whether it could track the Sun live

> No, and that is a deliberate boundary rather than an unfinished feature. The GOES X-ray flux is in
> the dataset and we excluded it as a predictor, because it measures flare activity directly and
> overlaps the window we are forecasting — using it would be leakage. A live version would need a
> SHARP ingestion path, monitoring for drift across the solar cycle, recalibration and operational
> validation against an existing service. None of that exists here, and the model card says so.

That answer turns the gap into evidence of discipline rather than an omission.

---

## Running it

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1   # once
```

Then **Ctrl + Alt + S**, or double-click `Open Dashboard.bat`, or:

```powershell
python -m solarflare dashboard
```

and open `http://localhost:8791/index.html`.
