# The dashboard, in full

Everything the site does, how to drive it, and what each control is for. Pair it with
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

## 1. Opening it

Any one of these:

```powershell
python -m solarflare dashboard
```

then open the `localhost:8791/index.html` address it prints. Or, on Windows, double-click
**Open Dashboard.bat**, which starts the server, waits for the port to answer and opens the browser
for you. Or press **Ctrl + Alt + S** from anywhere, once the shortcut is installed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1   # once only
```

> **Do not open `dashboard/index.html` by double-clicking it.** Over the `file:` protocol the
> browser refuses to let a page read its own data files, so every view comes up empty. The page
> detects this and says so, but the fix is to serve the folder rather than open the file.

---

## 2. How to use it — the five-minute route

This is the order that makes the project legible fastest. Nothing you click can change a result;
there is no way to break it.

| | Do this | What you are looking at |
|---|---|---|
| **1** | Read the **Start** page | What the system is, in plain words, and the four numbers that matter |
| **2** | Open **Replay**, press **▶ Play** | Ten days of a real active region. The dial climbs and turns red *before* the flare |
| **3** | Open **Explore**, click any tile, press **Reveal the ground truth** | A real case — including ones it got wrong. Predict first, then check |
| **4** | Open **Live**, press **Use a built-in example** | The real model running in your browser, with its working shown |
| **5** | Open **Model Evaluation**, then **Operating** | The full report card, then drag the cut-off to see the trade-off move |

If you would rather be driven, press **▶ 3-minute tour** on the Start page and it does all of this
with captions.

**The one slide to linger on is step 3.** Clicking a false alarm and revealing it is what persuades
a panel that the numbers are honest.

---

## 3. What is on every screen

| Element | What it does |
|---|---|
| **The mark, top left** | A solar disc with an eruption sweeping off its limb. Decorative. |
| **The navigation** | Eleven tabs. The current one is highlighted; the address bar follows it, so a reload keeps your place. |
| **The view cue** | A one-line box under each title saying what to press — or saying plainly *"Just read this one"* when there is nothing to press. Someone hunting for a button that does not exist assumes the page is broken. |
| **▶ 3-minute tour** | Drives the whole demonstration with captions. |
| **The footer** | Repeats the prototype disclaimer on every view, so no screenshot can be taken out of context. |

---

## 4. The eleven views

### 1. Start — the landing page

Written for someone who has never heard of the project. It carries:

- the headline question, *"Will this patch of the Sun erupt in the next 24 hours?"*, and a
  plain-English paragraph under it;
- **how it works, in three steps** — Observe (60 readings over 12 hours), Summarise & score
  (144 numbers into one probability), Decide (compare with a fixed cut-off);
- **four headline numbers**: 92 % of real major flares caught, 16 % of alerts followed by one,
  0.853 skill score, and the 1.31 % base rate that makes the first two meaningful;
- **How to use this site** — the four steps in §2, with the three-minute estimate stated;
- **eight feature cards**, each labelled *Interactive — open →* or *Read →* so the distinction is
  visible before clicking rather than after;
- **Being straight about it** — two columns, what this is and what it is not, side by side.

Three buttons at the top: take the tour, watch it run on a real flare, or jump to the results.

### 2. Overview

The problem, the dataset, the 12 h → 24 h forecast framing, the pipeline and the selected model.
Carries the "Academic Prototype" badge.

### 3. Forecast

The observation window being scored: the HARP region, the window start, the prediction cutoff and
the 24 h window that follows, plus the three counts that define the representation — **24** SHARP
parameters, **60** records, **144** engineered features.

**Run Forecast** scores that window with the real browser-side model and opens the result.

### 4. Prediction

The probability as a gauge, the model's classification, the decision threshold that produced it,
and the ground truth shown for verification. The side card names the flare that actually followed.

### 5. Risk & Alert

The three bands — LOW, MODERATE, HIGH — generated from the two thresholds in
`results/selection.json`, with the alert wording and the prototype caveat.

### 6. Explainability

The largest standardised coefficients as signed bars, red pushing the score up and blue pulling it
down.

**The point to make:** because the final model is linear, these contributions are *exact*. They are
not an approximation like a surrogate explainer or permutation importance. The contributions plus
the intercept reproduce the model's own log-odds to fourteen decimal places.

### 7. Model Evaluation

Six headline metrics, **each with its 95 % confidence interval**, and the confusion matrix on the
locked test partition. The notes panel states the precision plainly and carries the finding about
what actually followed the false alarms.

### 8. Explore — the strongest view

Twenty real windows from the locked test partition, chosen by a seeded, documented rule that
deliberately includes the failures: **7 correct quiet forecasts, 5 hits, 4 misses and 4 false
alarms**. Each tile shows the region, the probability and the risk band; the **outcome tag is shown,
but the true flare stays hidden** until you press **Reveal the ground truth**.

Clicking a tile opens the detail panel: the raw 12-hour SHARP series as sparklines, the exact
per-feature contributions, and the window's timing.

**Eighteen of the twenty** carry their raw series. The other two say why they cannot — the P5
archive on this machine is a corrupt gzip stream (see [REPRODUCIBILITY.md](REPRODUCIBILITY.md) §7)
— rather than showing an empty panel. Their probability and contributions are unaffected, because
those come from the extracted feature matrix, not the archive.

**Say this out loud:** the selection rule is seeded and written into the page, and the misses are
there on purpose. Showing the failures earns more credit than any metric.

### 9. Live

Three ways in: drop a SWAN-SF window `.csv` on the panel, **Choose a file**, or **Use a built-in
example**. The browser then runs the full pipeline — signed-log, median imputation,
standardisation, logistic regression — from parameters exported out of the joblib file.

It is checked rather than asserted: the browser agrees with scikit-learn to better than 1e-6 on
1,000 real test-partition rows and on 220 real windows, plus synthetic edge cases.

The built-in example is `dashboard/example_window.json`, **this view's own file**, holding one real
P5 window. It used to be rebuilt out of the explorer's 650 KB `windows.json`, which meant a missing
explorer file silently disabled this button too; the two are now independent.

Expected input: a header row, the 24 SHARP columns, 60 records. A different row count still runs,
with a warning that it is not a standard window.

### 10. Operating

A **402-point threshold sweep** over the real test-partition predictions, with the confusion matrix
and the six metrics updating as the slider moves. The validation-fixed operating point is marked on
the track, as is the high-risk threshold, and **Reset** returns to the real setting. An
"Exploratory" banner states that nothing here changes the published result.

**The line to deliver:** fixing the threshold on validation rather than picking the best point on
this curve cost 0.0123 TSS, and that was left on the table deliberately.

### 11. Replay

Three real active regions played back hour by hour at a 60-minute cadence, chosen so the set tells
the whole story rather than the flattering part of it:

| Region | Span | Steps | Outcome | What you see |
|---|---|---|---|---|
| **HARP 7115** (NOAA AR 12673) | 2017-08-30 → 09-08 | 101 | 55 hits, 2 misses, 44 correct quiet | The strongest region of the cycle. The dial climbs and stays red through the X9.3 and X1.3 flares. **Two frames are marked MISSED** and left in. |
| **HARP 5541** | 2015-05-07 → 05-17 | 211 | 211 false alarms | A region that never produced a major flare. The model alerts continuously and is wrong every single time. |
| **HARP 7131** | 2017-09-11 → 09-21 | 238 | 238 correct quiet | A quiet region. The model stays silent, correctly. |

**▶ Play** runs about fifteen seconds; the scrubber jumps to any frame; **Restart** returns to the
beginning. Each frame shows the dial, the risk band, the timestamp and a verdict line naming the
flare that followed, if any.

A banner states plainly that this is recorded data played back in the order it was taken — every
frame is a real 12-hour window and the probability shown is the model's frozen prediction for it.

**Show the 5541 track too.** Spending ten seconds on the region it gets wrong every time is braver
than showing only 7115, and it reads better.

---

## 5. Supporting features

| Feature | Why it matters on the day |
|---|---|
| **3-minute guided tour** | Timed steps with captions, matching the demo script. Removes any risk of fumbling navigation in front of the panel. |
| **Per-view cues** | Every view opens with one line telling the reader what to press, or saying there is nothing to press. |
| **Ctrl + Alt + S** | Opens the dashboard from anywhere in Windows — no terminal visible to the examiners. |
| **Works offline** | No CDN, no external request, asserted by a test. It still works if the venue's network fails. |
| **Deep links** | `#start`, `#overview`, `#predict`, `#result`, `#risk`, `#explain`, `#eval`, `#explore`, `#live`, `#operate`, `#replay` open a view directly, and a refresh does not lose your place. |
| **Prints cleanly** | Black on white, one view per page, if an examiner wants paper. |
| **Keyboard and screen-reader support** | Visible focus throughout, tab semantics, labelled controls, AA contrast. |
| **Degrades honestly** | A data file that cannot be read costs one view, never the page. The view names the file and the command that regenerates it, and offers **Try again**. |
| **Never served stale** | The server sends `Cache-Control: no-store`, so a regenerated file is never masked by a cached copy. |
| **Report mode** | Hides the five added views, the per-view cues and the new mark, so the six original views still reproduce the report's Figures 5.1-5.6 exactly. |

---

## 6. The one thing to be clear about: "live" does not mean live data

The **Live** view runs the trained model **in the browser, as you watch**. That is live
*computation*. It is **not** a live feed of the Sun.

This system has never been connected to any live data source. It is trained and evaluated only on
historical SWAN-SF observations from 2010-05-01 to 2018-08-17. There is no NOAA or GOES monitoring,
no ingestion path, and no operational validation. The Replay view is recorded data, not a feed.

If anyone hears "live" and pictures a monitoring console, correct it immediately:

> Live inference means the real trained model executes in the browser on a window you hand it. It is
> live computation, not live data. This system has never been connected to a live feed.

| Do not say | Say instead |
|---|---|
| real-time, monitoring, tracking | historical, retrospective, offline |
| operational, warning service | academic prototype, demonstrator |
| "it watches the Sun" | "it scores a recorded observation window" |

---

## 7. What the thing actually is

A **static, offline, single-page demonstrator**. No server logic, no database, no network request of
any kind — enforced by a test that fails if an external reference appears in the page.

Everything on screen is generated from `results/` by `python -m solarflare demo-data`. Not one
metric, threshold or count is typed into the HTML. That is the claim worth making: **the dashboard
cannot contradict the published results, because it holds no independent copy of them.**

---

## 8. If asked whether it could track the Sun live

> No, and that is a deliberate boundary rather than an unfinished feature. The GOES X-ray flux is in
> the dataset and we excluded it as a predictor, because it measures flare activity directly and
> overlaps the window we are forecasting — using it would be leakage. A live version would need a
> SHARP ingestion path, monitoring for drift across the solar cycle, recalibration and operational
> validation against an existing service. None of that exists here, and the model card says so.

That answer turns the gap into evidence of discipline rather than an omission.

---

## 9. If something looks wrong on the day

| What you see | What to do |
|---|---|
| A view says a `.json` file could not be loaded | Press **Try again**. It is usually a tab left open or a server that was still starting. |
| Still failing after that | `python -m solarflare demo-data`, then reload. |
| Every view is empty, with a red banner about the `file:` protocol | The page was opened from disk. Serve it instead — see §1. |
| The page looks out of date | Hard-reload once with **Ctrl + F5**. |
| The server will not start | Something is already on the port; it names the next free one. |

The server also prints a warning naming any missing data file and the view it costs, so the
terminal tells you before the browser does.
