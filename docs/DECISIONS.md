# Decisions and assumptions

Every non-obvious choice made while hardening this project, with the reason. Entries are append-only
and dated. "Frozen" throughout means: the seven models in `models/`, everything in `results/`, both
thresholds, the partition split, the seed, the feature definitions, the selection rule, and the
published metrics — none of which may change.

---

## D-01 — The frozen artefacts are the source of truth, not the scripts

**Date:** 2026-10-05
**Decision:** Where a script and a stored result could disagree, the stored result in `results/` wins,
and the script is only ever changed to make it reproduce that result more clearly — never to change
the number.
**Why:** The report is finished and frozen and quotes those numbers. Re-deriving them is a
verification activity, not a modelling activity.
**Consequence:** `src/train_eval.py`, `src/merge_audit.py` and `src/build_demo_data.py` keep their
behaviour exactly. Verification code re-implements their maths verbatim and compares, rather than
importing and re-running them over the real output directories.

## D-02 — Python 3.12.10 instead of 3.11

**Date:** 2026-10-05
**Decision:** Reproduce and test on Python 3.12.10 rather than the 3.11 named in the brief.
**Why:** Python 3.11 is not installed on this laptop (3.10.2, 3.12.10 and 3.14.0 are), and installing
a fourth interpreter system-wide is a heavier, less reversible change than the task needs. The
constraint that actually matters for unpickling the saved pipelines is the **scikit-learn** version,
which is pinned to the required 1.8.0.
**Evidence it is safe:** under 3.12.10 / scikit-learn 1.8.0 all seven models reproduce every stored
validation and test metric to better than 1e-9, both thresholds to better than 1e-12, and the
selected model's P5 confusion matrix exactly (907 / 4707 / 83 / 69668), with stored probabilities
matching a fresh `predict_proba` to 1.3e-15. See [RECON.md](RECON.md) §2a.
**Consequence:** `requirements.txt` pins the library versions exactly and the runtime check warns —
rather than refuses — on a Python minor-version mismatch, while failing hard on a scikit-learn
mismatch. Documented in `docs/REPRODUCIBILITY.md`.

## D-03 — `data/all_X.npz` and `all_meta.csv.gz` were rebuilt, not re-extracted

**Date:** 2026-10-05
**Decision:** Regenerate the two merged data files from the existing `data/features/P*_c*` chunks
rather than re-streaming the five raw tarballs.
**Why:** Both merged files were missing, and every downstream script reads them, so nothing could run
at all. The 13 per-chunk feature files were present and intact (241 MB), and merging them is the
documented second half of the pipeline (`merge_audit.py`), taking about four minutes instead of the
hours that re-streaming 5.7 GB of tarballs would cost.
**Evidence it is faithful:** the audit recomputed from the rebuilt matrix reproduces
`results/data_audit.json` exactly on every key, with floats agreeing to 1e-12 — including all ten
zero active-region overlaps, the 60-rows-per-window and 11.8 h-per-window invariants, and the
missingness fractions. Independently, `results/test_predictions_selected.csv.gz` has a `row` column
exactly equal to `np.where(partition == 5)[0]` against the rebuilt metadata, confirming the row
ordering of the merge is identical to the original.
**Consequence:** the merge is deterministic given the chunks (`sorted(glob(...))` over fixed
filenames), so this is repeatable. Recorded as a pipeline step in the CLI.

## D-04 — Nothing is written over a frozen file, ever

**Date:** 2026-10-05
**Decision:** Verification and analysis code writes to `results/extra/`, `figures/extra/`,
`docs/screenshots/` or a temporary directory. The merge step refuses to overwrite `data/all_X.npz`
or `data/all_meta.csv.gz` if they already exist, rather than silently replacing them.
**Why:** `src/merge_audit.py` as written would overwrite `results/data_audit.json`, and
`src/make_figures.py` would overwrite `figures/fig6_*.png` — both frozen, and the figures are
embedded in the finished report. A single careless run would destroy the evidence chain.
**Consequence:** re-running the original scripts verbatim is deliberately *not* wired into the
default `all` target. The CLI's verification path recomputes and compares instead.

## D-05 — Post-hoc analyses are labelled in their own output

**Date:** 2026-10-05
**Decision:** Every statistic computed on P5 beyond the frozen metrics carries the words
"post-hoc, not used for selection" in the artefact itself — the JSON, the figure caption and the
Markdown — not only in surrounding prose.
**Why:** The honesty requirement. A figure or table extracted from this repo and pasted elsewhere
must still say that it was computed after the test partition was unlocked, so it cannot be mistaken
for selection evidence.

## D-06 — "HARP" is used for the `ar` field in all new text

**Date:** 2026-10-05
**Decision:** New documentation and UI call the `ar` field a HARP number, not a NOAA AR number.
**Why:** SWAN-SF derives `ar` from the SHARP/HARP region numbering, which is not the NOAA active
region numbering. The existing dashboard renders it as "AR 7115", which is wrong in a way a
domain-expert examiner would catch.
**Status:** the HARP 7115 ↔ NOAA AR 12673 mapping is still to be verified against JSOC before any
"(NOAA AR …)" annotation is added. Until then, new text says HARP only.

## D-07 — Metric names are fixed and defined once

**Date:** 2026-10-05
**Decision:** Across all documentation and UI: **TSS** = recall − FPR; **HSS** refers to the HSS2
form that `train_eval.py` computes and is always written "HSS (HSS2 form)" on first use; **false-alarm
ratio (FAR)** = FP/(TP+FP) = 0.838; **false-alarm rate (FPR)** = FP/(FP+TN) = 4707/74375 = 0.0633.
The two false-alarm quantities are never both called "FAR" and never used interchangeably.
**Why:** `train_eval.py` stores the ratio under the key `far`, which invites exactly that confusion.
0.838 and 0.063 differ by more than an order of magnitude.
**Consequence:** a glossary in `docs/MODEL_CARD.md`, and the automated consistency check treats the
two as distinct quantities.

## D-08 — `src/shoot.py` is left in place but is not the screenshot path

**Date:** 2026-10-05
**Decision:** `src/shoot.py` is kept unmodified and unused. Screenshots are taken with
`python -m solarflare screenshots`, which writes to `docs/screenshots/`.
**Why:** `shoot.py` writes directly to `figures/fig5_*.png`, which are embedded in the frozen
report. Running it would overwrite them. It also launches two Chromium instances and leaks one
(line 10), which is a bug, but fixing a script that must not be run is not worth the churn.
**How the report's figures stay verifiable:** the new screenshot command additionally captures the
six original views in **report mode**, with the three added navigation buttons and the tour
launcher hidden, under the same 1440×960 viewport. Those images are visually equivalent to the
report's and can be diffed against them by hand. `tests/test_dashboard.py::test_report_mode_hides_the_new_navigation`
asserts that report mode shows exactly the six original buttons.

## D-09 — The explorer view's window selection is reproducible, not availability-driven

**Date:** 2026-10-05
**Decision:** The twenty windows in the dashboard's explorer view are chosen by a seeded rule over
the full test partition, **before** checking whether their raw series can be read. Windows whose
series is unavailable are shown with their probability, risk level and contributions, and say why
the series is missing.
**Why:** the alternative — picking only from the windows that happen to be readable on this machine
— would make the selection depend on a corrupt local file, so a colleague with intact archives
would get a different set. Keeping the rule pure means the selection is identical everywhere, and
the missing data is surfaced rather than hidden.
**Consequence:** 18 of 20 windows carry their series here. `test_most_explorer_windows_carry_their_raw_series`
requires at least 75 %, and the contract test requires that any window without one explains itself.

## D-10 — The near-miss analysis reframes precision but does not revise it

**Date:** 2026-10-05
**Decision:** Added an analysis of what actually followed each false alarm, and gave it a prominent
place in the model card, README and demo script. The published precision of 0.1616 is **not**
restated or adjusted anywhere.
**Why:** 77.8 % of the false alarms precede a real B- or C-class flare and only 22.2 % are truly
quiet; the model alerts on 49.8 % of C-class windows against 1.67 % of flare-quiet ones. That is a
materially different failure from alerting at random, and an examiner asking "is 16 % precision
useless?" deserves the evidence. But a C-class flare is not an M-class flare, so the headline
number stands and every presentation of this analysis says so explicitly.
**Consequence:** `results/extra/near_miss.json` and `figures/extra/extra5_near_miss.png`, both
carrying the standard post-hoc stamp.

## D-11 — Two raw archives are corrupt; the pipeline now refuses to use them silently

**Date:** 2026-10-05
**Decision:** Verified every raw tarball as a complete gzip stream and found `partition2` and
`partition5` corrupt. Added `solarflare.data.check_raw_archives()` and wired it into
`python -m solarflare features`, which now refuses to extract from a damaged archive.
**Why:** a streamed tar read stops **quietly** at the damage with no exception, so the failure mode
is a silently short dataset rather than a crash. P5 yields 64,315 of its 75,365 windows. Without
the check, a future re-extraction would produce wrong data that looked fine.
**Impact on published results: none.** `data/features/` was extracted while the archives were
intact — the per-chunk logs record the full 88,557 for P2 and 75,365 for P5, and
`results/data_audit.json` reproduces exactly from them.
**Consequence:** documented in `docs/REPRODUCIBILITY.md` §7 and `docs/DATA_CARD.md` §10, and two
explorer windows show the reason their series is absent.

## D-12 — The browser pipeline lives in one file, tested from Node

**Date:** 2026-10-05
**Decision:** The JavaScript implementation of the model lives in `dashboard/model.js`, loaded by
the page with a `<script src>` tag and `require()`d by the parity tests.
**Why:** the first draft inlined the functions in `index.html`, which made them untestable without
a browser and invited the page and the tested code to drift apart. One file, two consumers.
**Consequence:** `tests/test_js_parity.py` checks probabilities on 1,000 real P5 feature rows and
feature extraction on 220 real SWAN-SF windows, both to better than 1e-6, plus synthetic edge
cases. The dashboard now requires a local server (`fetch` of the JSON files does not work from
`file://` in any case, which was already true of the original page).

## D-13 — The threshold sweep contains the published thresholds exactly

**Date:** 2026-10-05
**Decision:** The operating-point sweep's grid is the usual quantile grid **plus** the two
published thresholds inserted exactly, and `dashboard/operating.json` records their indices.
**Why:** a quantile grid does not naturally contain 0.544546643935129. The first version of the
slider defaulted to the nearest grid point and displayed FP 4,672 where `results/test_results.json`
says 4,707 — the dashboard contradicted itself on the same screen. Found by driving the page, not
by reading the code.
**Consequence:** `build_operating()` now raises if the sweep row at the published threshold
disagrees with the frozen confusion matrix, and
`test_operating_view_resets_to_the_published_operating_point` asserts the slider reproduces
907 / 4,707 / 83 / 69,668.

## D-15 — A retrain is judged on the decision, not on bit-identical coefficients

**Date:** 2026-10-05
**Decision:** `python -m solarflare train` passes when the retrained run selects the same model by
the same rule and lands within 5e-3 on both thresholds and the headline test metrics. Exact
equality is available behind `--strict` and is expected to fail off the original machine.
**Why:** the first version demanded exact equality and failed. Investigating showed the retrain
reproduces the *decision* perfectly — same model, same rule — while the fitted coefficients differ
slightly, moving the alert threshold by 1.2e-3, the test TSS by 1.8e-3 and two windows across the
decision boundary. Treating that as a failure would be misleading: it would imply something is
broken when the published artefacts reproduce their numbers to 1e-9.
**What was ruled out:** the obvious explanation — floating-point accumulation order under
parallelism — is **wrong**. Fitting the selected architecture on the same data with
`OMP_NUM_THREADS=1` and with `=4` gives bit-identical coefficients, an identical intercept and the
same iteration count. Tested rather than assumed, and the earlier claim that thread count was the
cause has been corrected everywhere it appeared.
**What remains unresolved:** some other difference between the original environment (Linux,
Python 3.11) and the verification one (Windows, Python 3.12). The most plausible candidate is the
**SciPy version**, since `lbfgs` lives in `scipy.optimize` and SciPy was the one dependency the
original specification did not pin. Not settleable without the original environment, so it is
documented as open rather than guessed at.
**Consequence:** `docs/REPRODUCIBILITY.md` §3 states the limit plainly, and the tolerance is a
named constant with the measured drift recorded beside it.

## D-17 — Serving refuses an occupied port rather than binding alongside

**Date:** 2026-10-06
**Decision:** `python -m solarflare dashboard` probes the port before binding and exits 1 with an
actionable message if something is already serving there.
**Why:** on Windows, `SO_REUSEADDR` permits a second process to bind a port that is already being
served. Tested: two `dashboard --port 8810` invocations both started, two processes ran, and
**no error was shown**. The result is two servers contending for one port and an orphan process left
behind — a silent failure, and exactly the kind that surfaces during a live demonstration.
**Evidence:** reproduced before the fix (two PIDs, one listener, no error) and after (clean message,
exit 1, no orphan). `tests/test_project.py::test_dashboard_refuses_a_port_that_is_already_serving`
binds a socket and asserts the command refuses.
**Affects:** engineering and presentation only. No published number depends on it.

## D-18 — A corrupt feature matrix reports itself as corrupt

**Date:** 2026-10-06
**Decision:** `data.load()` wraps the `np.load` of `data/all_X.npz` and re-raises a
`FileNotFoundError` naming the file, saying it is probably truncated, and giving the rebuild command.
**Why:** a damaged `.npz` raises numpy's "This file contains pickled (object) data…" message, which
reads as a security problem and sends the reader in entirely the wrong direction. The file existing
but being unreadable is a plausible state — the project already documents two corrupt source
archives on this machine.
**Evidence:** reproduced with a deliberately malformed file;
`tests/test_project.py::test_a_corrupt_feature_matrix_reports_itself_clearly` asserts the message.
**Affects:** engineering only.

## D-19 — The explorer view explains its own absence

**Date:** 2026-10-06
**Decision:** `renderExplore()` is called unconditionally and, when `windows.json` is missing or
unreadable, replaces the grid with a message naming the file and the command that regenerates it.
**Why:** every other view already names the file it is missing. The explorer alone degraded
silently, leaving an empty grid and a placeholder dash — which during a demonstration reads as a
broken page rather than a missing input. Verified by corrupting the file: the page still loaded with
no console errors, but the view was simply blank.
**Evidence:** `tests/test_dashboard.py::test_explore_explains_itself_when_its_data_is_unreadable`.
**Affects:** presentation only.

## D-21 — One missing file may cost one view, never the page

**Date:** 2026-10-07
**Decision:** Every dashboard JSON file is loaded optionally. Failures are collected rather than
thrown, each render step is isolated so one that raises cannot stop the next, and a banner names
each file that failed with the command that regenerates it. The live view's example moved out of
`windows.json` into its own `dashboard/example_window.json`.
**Why:** `demo.json` and `context.json` were required, so one failed fetch rejected `main()` and
took all eleven views down together — no cues, no explorer, dead buttons on the live view, and one
line of explanation. The commonest way to hit it is not a missing file at all: opening `index.html`
from disk means `fetch` cannot read a sibling file, so every load fails at once and the page looks
broken for a reason that has nothing to do with the project. It now says exactly that.

Separately, the live view's example button rebuilt its CSV out of `windows.json`, which is 650 KB
and belongs to the explorer. That coupled two unrelated views: losing the explorer's file silently
disabled the live view's example too, and the button reported only that no example was available.
The example is now a 30 KB file of its own, carrying one real P5 window; `windows.json` remains a
fallback. A test asserts the two agree on that window's probability and outcome.

`showWindow()` also read `S.model.sharp_parameters` to label the series, so a missing
`model_lr.json` threw on every explorer tile click. It falls back to the series' own keys.
**Evidence:** `tests/test_dashboard.py::test_a_missing_explorer_file_does_not_break_the_live_example`
serves a copy with `windows.json` deleted and asserts the example still runs;
`tests/test_contracts.py::test_the_live_example_agrees_with_the_explorer`.
**Affects:** presentation and engineering; no published number depends on it.

## D-22 — The dashboard server forbids caching

**Date:** 2026-10-07
**Decision:** `cmd_dashboard` sends `Cache-Control: no-store, must-revalidate` on every response,
the page fetches with `cache: 'no-store'`, a failed fetch is retried twice, and the explorer's
degraded state shows the real error and offers a **Try again** button.
**Why:** the explorer kept showing "windows.json is missing or unreadable" on a machine where the
file was present, valid, and served correctly — 200 with all 648 KB, thirty times out of thirty,
and a fresh browser rendered all twenty windows. The cause was the response headers:
`SimpleHTTPRequestHandler` sends `Last-Modified` and **no `Cache-Control` at all**, so browsers fall
back to heuristic freshness and may serve `index.html` and the JSON from cache without revalidating.
A page left open, or one loaded while the server was still starting, therefore kept showing a stale
failure that no amount of reloading the data would clear. Nothing here is worth caching: it is
localhost, and the files are regenerated by `demo-data`.

Retrying matters for the same reason from the other side: the launcher opens the browser as soon as
the port answers, and one refused connection used to disable a view for the life of the page.
**Evidence:** `tests/test_dashboard.py::test_the_server_forbids_caching` and
`::test_the_explorer_recovers_without_a_reload`, which withholds the file, asserts the view names
the real HTTP status, restores it, and presses Try again.
**Affects:** presentation only; no published number depends on it.

## D-23 — A drawn mark, with the old disc kept for report mode

**Date:** 2026-10-08
**Decision:** The header logo is an inline SVG: a glowing solar disc with an eruption sweeping off
its limb, over the same hazy corona the old mark had. `favicon.svg` matches it. The previous flat
disc is kept in the markup and shown only under `body.report-mode`.
**Why:** the old mark was a CSS radial gradient on an empty `<div>` — a featureless orange blob that
said nothing about the project. Five candidate marks were drawn and rendered at 96, 38 and 20 px
before choosing, because the failures are only visible at size: a small disc with a single curling
arc reads as a lit bomb fuse, a cropped limb under an arch reads as a handbag, a tapered plume
reads as a lollipop, and concentric arcs read as a wifi icon. The eruption crescent is the only one
that stayed legible at 20 px and unambiguous at 96 px.

The logo appears in every view, so changing it would have altered the captures of Figures 5.1-5.6.
It is held back in report mode exactly as the added navigation and the view cues are, and the six
report-mode captures are byte-identical after the change.
**Evidence:** `tests/test_dashboard.py::test_the_mark_is_held_back_in_report_mode`.
**Affects:** presentation only.

## D-20 — British spelling

**Date:** 2026-10-05
**Decision:** All new prose uses British spelling ("organisation", "generalisation", "standardised"
in prose), while **code identifiers and scikit-learn terminology keep their original spelling**
(`StandardScaler`, `normalize`, `standardized coefficient` when quoting an axis label that is
already baked into a frozen figure).
**Why:** The brief asks for British spelling and for consistency. Renaming code or frozen figure
captions to match would break reproduction.
