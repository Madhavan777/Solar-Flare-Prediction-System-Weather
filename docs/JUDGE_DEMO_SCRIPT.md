# Three-minute demonstration script

A timed walkthrough for the review panel. The dashboard has a **"▶ 3-minute tour"** button that
drives exactly these nine steps in order, with the captions on screen — use it if you would rather
not navigate manually, and press **Auto-play** to advance hands-free.

---

## Before the panel arrives

**Press Ctrl + Alt + S.** That starts the server and opens the dashboard in one keystroke, from
anywhere in Windows — no terminal in front of the examiners. (Equivalently, double-click
`Open Dashboard.bat` in the project folder.)

If the hotkey is not registered on the machine you are presenting from:

```powershell
cd "C:\Users\MG-Laptop\Claude\Projects\Solar Flare Detection System"
powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1
```

Or start it by hand:

```powershell
.\.venv\Scripts\python.exe -m solarflare dashboard
```

Then open **http://localhost:8791/index.html**. It runs entirely offline — no network, no CDN.

Checklist:

- [ ] the page loads with no red bar across the top;
- [ ] the Overview view shows "SWAN-SF" and "12h → 24h";
- [ ] the Explore view lists 20 windows;
- [ ] the Live view's "Use a built-in example" button returns a probability;
- [ ] the Replay view's **▶ Play** advances the dial;
- [ ] have `results/extra/RESULTS.md` open in a second window for follow-up questions.

If a view reports a missing file, press its **Try again** button first — it is usually a tab left
open. If that does not clear it, `python -m solarflare demo-data` regenerates every JSON the page
reads.

---

## The script

### 0:00 — 0:15 · What this is *(Start)*

> "Before the detail: this reads twelve hours of magnetic measurements from one patch of the Sun
> and estimates whether that region will produce a major flare in the next twenty-four hours. It
> learned from 331,185 recorded observations between 2010 and 2018. Everything here runs on
> recorded data — nothing is connected to the Sun."

Point at: the headline question and the four numbers beneath it.

### 0:15 — 0:45 · Watch it work *(Replay)*

Press **▶ Play** on the HARP 7115 track and let it run while you talk.

> "This is a real active region, September 2017, played back hour by hour. Every frame is a real
> twelve-hour window and the probability is the model's stored prediction for it. Watch the dial:
> it climbs and crosses the alert threshold *before* the X9.3 flare, and stays high through it. Two
> frames are marked MISSED — genuine pre-flare windows it scored too low. They are left in."

**Then switch to the HARP 5541 track** for ten seconds.

> "And here is the same model on a region that never produced a major flare. It alerts continuously
> and is wrong every single time — 211 false alarms. That is the cost of the recall you just saw."

Showing 5541 is braver than showing only 7115, and it lands better with a panel.

### 0:45 — 1:10 · The problem *(Overview)*

> "The task is to forecast whether a solar active region will produce a major flare — GOES class M
> or X — in the 24 hours after a 12-hour observation window of magnetogram data. We use SWAN-SF:
> 331,185 windows, five partitions, 2010 to 2018. Major flares are rare, 1.3 % of the test
> partition, and that imbalance shapes every decision that follows."

Point at: the three cards — dataset, 12h → 24h, and the **Academic Prototype** badge.

### 1:10 — 1:30 · The protocol *(stay on Overview)*

> "The split is by SWAN-SF partition, chronological, never shuffled: partitions 1 to 3 train,
> partition 4 validates, partition 5 is the locked test set. That matters because consecutive
> windows of one active region are one hour apart 98.6 % of the time, so adjacent 12-hour windows
> share eleven of their twelve hours. A random split would put near-duplicates on both sides and
> inflate everything. No active region appears in more than one partition — all ten pairwise
> overlaps are zero."

Point at: the **Pipeline** card, the "P1–P3 train / P4 val / P5 test" pill.

### 1:30 — 2:00 · The result, with its uncertainty *(Model Evaluation)*

> "On the locked test partition the model reaches a true skill statistic of 0.853 and a PR-AUC of
> 0.489, detecting 91.6 % of major flares. The intervals under each figure come from resampling
> whole active regions, because windows are not independent.
>
> Now the number we do not hide: precision is 0.162. Of the alerts we raise, 84 % are not followed
> by a major flare. And note the accuracy — 93.6 %. A model that always says 'no flare' scores
> 98.69 % and detects nothing, which is exactly why accuracy is not our headline metric."

Point at: the six KPI tiles with their 95 % intervals, then the confusion matrix — 907 hits, 83
misses, 4,707 false alarms.

**If asked about precision here, give the 30-second version:** 77.8 % of those false alarms precede
a real B- or C-class flare; only 22 % land on a genuinely quiet window. Full answer in
[JUDGES_QA.md](JUDGES_QA.md) §1.

### 2:00 — 2:20 · Why the threshold is honest *(Operating)*

> "This slider shows every operating point available on the test partition. Our threshold was fixed
> on the validation partition before this data was ever scored. The best threshold on this curve
> would have given 0.865 — fixing it honestly cost us 0.012 TSS. We left that on the table, and
> that is the point: choosing from this curve would be selecting on the test set."

Point at: the orange marker, then drag the slider and let them watch recall and precision trade off.
Press **Reset** — the confusion matrix returns to exactly 907 / 4,707 / 83 / 69,668.

### 2:20 — 2:45 · Successes and failures together *(Explore)*

> "Twenty real windows from the test partition, chosen by a seeded rule that deliberately includes
> the misses and the false alarms, not just the wins. Each shows the raw twelve-hour SHARP series,
> and the exact per-feature contributions to the log-odds — exact, because the model is linear, so
> the contributions plus the intercept reproduce the logit to fourteen decimal places."

Click a window with outcome **FN**, scroll to the contributions, then click **Reveal the ground
truth**.

> "This one is a genuine pre-flare window the model scored below threshold. We show the failures
> because an examiner will find them anyway."

### 2:45 — 3:00 · It is the real model *(Live)*

> "This runs the actual fitted pipeline in the browser — signed-log, median imputation,
> standardisation, logistic regression — from parameters exported out of the joblib file. It is
> tested against scikit-learn on a thousand test windows and agrees to better than one part in a
> million."

Click **Use a built-in example**. Let the probability and contribution bars appear.

### 3:00 — 3:15 · What it is not *(Risk & Alert)*

> "Finally, the honest framing. This is an academic prototype, not an operational warning service.
> It uses historical data only, it has no live feed, it was never validated beyond 2018 or beyond
> HMI SHARP data, and its probabilities are deliberately over-forecast because of the balanced
> class weights. Those are limitations we state, not ones you have to find."

Point at: the footer disclaimer, visible on every view.

---

## Likely follow-ups, with the one-line answer

| Question | One line | Full answer |
|---|---|---|
| Why is precision so low? | 78 % of false alarms precede a real B/C flare; only 22 % are truly quiet. | [QA §1](JUDGES_QA.md) |
| Why LR when RF scores higher on test? | The 95 % TSS interval is 0.164 wide; the spread across all seven is 0.014. They are indistinguishable. | [QA §2](JUDGES_QA.md) |
| How do you know there is no leakage? | Five mechanisms, each with a failing test. `python -m solarflare test`. | [QA §3](JUDGES_QA.md) |
| Why not a random split? | Adjacent windows share 11 of 12 hours, 98.6 % of the time. | [QA §4](JUDGES_QA.md) |
| Is accuracy not better? | Always-no-flare scores 98.69 % and detects nothing. | [QA §5](JUDGES_QA.md) |
| Does ML beat a simple rule? | On TSS, no — one feature matches it. On PR-AUC we gain 32 %. | [QA §6](JUDGES_QA.md) |
| Is it real-time? | No. No live feed, no GOES monitoring, historical data only. | [QA §7](JUDGES_QA.md) |
| Prove the test set was used once. | The threshold is suboptimal on test by 0.012 TSS, and four candidates beat the selected one there. | [QA §10](JUDGES_QA.md) |
| Are the probabilities calibrated? | No — they over-forecast by 7.3×, by design. It changes no published metric. | [QA §11](JUDGES_QA.md) |
| Biggest weakness? | The candidate search was narrow; the ablation finds smaller variants with higher validation TSS. | [QA §14](JUDGES_QA.md) |

---

## Two things to avoid saying

- **Never "real-time" or "operational".** It is neither, and the documentation says so; contradicting
  your own model card in front of the panel is the worst possible outcome.
- **Never quote accuracy without the companion sentence.** "93.6 % accurate" invites the reply that
  doing nothing scores 98.7 %. Say both numbers together, every time.

## If something breaks live

| Symptom | Fix |
|---|---|
| A view says a `.json` file could not be loaded | Press its **Try again** button. A tab left open or a server still starting is the usual cause, and one press clears both |
| Still failing after that | `python -m solarflare demo-data`, then reload |
| Every view is empty, red banner mentioning `file:` | The page was opened from disk by double-clicking. Serve it instead: `python -m solarflare dashboard` |
| The page looks out of date | Hard-reload once with **Ctrl + F5** |
| Port 8791 already in use | It names the next free port. Or `python -m solarflare dashboard --port 8795` |
| Ctrl + Alt + S does nothing | The shortcut must be on the Desktop. Re-run `scripts\install_shortcut.ps1` |
| You lose your place after a refresh | You will not — the view is in the URL. Every view deep-links, `#start` through `#replay` |
| Asked to prove reproduction on the spot | `python -m solarflare evaluate` — 375 checks, about two minutes |

The server also prints a warning naming any missing data file and the view it costs, so the
terminal tells you before the browser does.
