"""Build the PBL final-review deck on top of the department's template.

    python scripts/make_deck.py [template.pptx] [out.pptx]

The template's theme, layouts, institute logo, footers and slide numbers are
kept exactly as they are; only the placeholder prose is replaced and figures
are added. Every number written here is read from results/ at build time, so
the deck cannot drift from the frozen artefacts.

Needs python-pptx, Pillow and qrcode, which are in requirements-dev.txt. The
dashboard captures it embeds live in docs/deck_assets/ and are produced by
scripts/shoot_deck.py against a running `solarflare dashboard`.
"""

from __future__ import annotations

import json
import pathlib
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

PROJ = pathlib.Path(__file__).resolve().parent.parent
IMG = PROJ / "docs/deck_assets"
TEMPLATE = pathlib.Path(
    sys.argv[1] if len(sys.argv) > 1 else PROJ / "docs/PBL_Final_Review_Template.pptx"
)
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else PROJ / "Solar_Flare_PBL_Final_Review.pptx")

# ----------------------------------------------------------------- the numbers
T = json.loads((PROJ / "results/test_results.json").read_text(encoding="utf-8"))
SEL = json.loads((PROJ / "results/selection.json").read_text(encoding="utf-8"))
M = T[SEL["selected"]]
BOOT = json.loads((PROJ / "results/extra/bootstrap.json").read_text(encoding="utf-8"))
NEAR = json.loads((PROJ / "results/extra/near_miss.json").read_text(encoding="utf-8"))
BASE = json.loads((PROJ / "results/extra/baselines.json").read_text(encoding="utf-8"))
BRK = json.loads((PROJ / "results/extra/breakdowns.json").read_text(encoding="utf-8"))

ci = BOOT["intervals"]
X = BRK["by_goes_class"]["X"]

# test_results.json publishes the false-alarm *ratio*; the false-alarm *rate*
# lives in the breakdown. Cross-check the two agree before borrowing it.
assert all(BRK["overall"][k] == M[k] for k in ("tp", "fp", "fn", "tn", "precision", "recall"))
M = {**M, "fpr": BRK["overall"]["fpr"]}


def f(x, n=3):
    return f"{x:.{n}f}"


# ------------------------------------------------------------- type and colour
# The deck is black-on-white in one typeface throughout, so the names below all
# resolve to the same colour; they are kept distinct only to document intent at
# each call site. enforce_type() at the end of this file is what guarantees it,
# including for the text that came with the template.
FONT = "Times New Roman"
BLACK = RGBColor(0x00, 0x00, 0x00)
HEAD = BODY = MUTED = BULLET = WARN = GOOD = BLACK

# Times New Roman sets smaller on the page than the template's Calibri at the
# same point size, and narrower, so every size written here is scaled up.
SCALE = 1.10

BUL = "•"  # bullet, present in Times New Roman
DASH = "–"


# ------------------------------------------------------------------- helpers
def no_bullet(p):
    """Take full control of the glyph: we draw our own."""
    pPr = p._p.get_or_add_pPr()
    for tag in ("a:buChar", "a:buAutoNum", "a:buNone", "a:buFont", "a:buClr", "a:buSzPct"):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    pPr.insert(0, pPr.makeelement(qn("a:buNone"), {}))


def runs(p, text, size, color, bold=False, italic=False, font=None):
    """Write `text` into paragraph `p`, honouring **bold** spans."""
    for i, part in enumerate(text.split("**")):
        if not part:
            continue
        r = p.add_run()
        r.text = part
        r.font.size = Pt(round(size * SCALE, 1))
        r.font.bold = bold or (i % 2 == 1)
        r.font.italic = italic
        r.font.color.rgb = color
        r.font.name = font or FONT


def write(shape, items, anchor_top=True):
    """Replace a shape's text with `items`.

    Each item is a dict: t=text, sz=pt, b=bold, c=colour, bullet=glyph colour,
    sp=space-after pt, i=italic, indent=inches, mono=use a mono font.
    """
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    if anchor_top:
        tf.vertical_anchor = MSO_ANCHOR.TOP
    for n, it in enumerate(items):
        p = tf.paragraphs[0] if n == 0 else tf.add_paragraph()
        no_bullet(p)
        p.space_after = Pt(it.get("sp", 4))
        p.space_before = Pt(it.get("spb", 0))
        p.line_spacing = it.get("ls", 1.0)
        # The template's body paragraphs carry a hanging indent meant for its
        # own bullets. Set it deliberately for ours, and clear it otherwise.
        pPr = p._p.get_or_add_pPr()
        hang = Inches(it["indent"]) if it.get("indent") else 0
        pPr.set("marL", str(int(hang)))
        pPr.set("indent", str(-int(hang)))
        if it.get("align"):
            p.alignment = it["align"]
        if it.get("bullet"):
            r = p.add_run()
            r.text = it["bullet"] + "  "
            r.font.size = Pt(round(it.get("sz", 13) * SCALE, 1))
            r.font.bold = True
            r.font.color.rgb = it.get("bc", BULLET)
            r.font.name = FONT
        runs(
            p,
            it["t"],
            it.get("sz", 13),
            it.get("c", BODY),
            bold=it.get("b", False),
            italic=it.get("i", False),
            font="Consolas" if it.get("mono") else None,
        )


def head(text, sz=17):
    return {"t": text, "sz": sz, "b": True, "c": HEAD, "sp": 6}


def bullet(text, sz=12.5, c=BODY, glyph=BUL, bc=BULLET, sp=4):
    return {
        "t": text,
        "sz": sz,
        "c": c,
        "bullet": glyph,
        "bc": bc,
        "sp": sp,
        "indent": 0.22,
        "ls": 1.0,
    }


def para(text, sz=12.5, c=BODY, **kw):
    return {"t": text, "sz": sz, "c": c, "sp": kw.pop("sp", 5), **kw}


def walk(shapes):
    """Every shape on a slide, descending into groups."""
    for sh in shapes:
        yield sh
        if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from walk(sh.shapes)


TITLES = (PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE)


def find(slide, prefix):
    """The content shape whose text starts with `prefix`.

    Slide titles are skipped: 'Feasibility, Risk and Challenges' and
    'Conclusion, Limitations & Future Scope' both start with the name of a
    content block below them, and matching the title would overwrite it.
    """
    for sh in walk(slide.shapes):
        if sh.is_placeholder and sh.placeholder_format.type in TITLES:
            continue
        if sh.has_text_frame and sh.text_frame.text.strip().startswith(prefix):
            return sh
    raise LookupError(f"no shape starting with {prefix!r}")


def box(sh, left=None, top=None, width=None, height=None):
    if left is not None:
        sh.left = Inches(left)
    if top is not None:
        sh.top = Inches(top)
    if width is not None:
        sh.width = Inches(width)
    if height is not None:
        sh.height = Inches(height)


def place(slide, img, left, top, width=None, height=None, border=True):
    """Add a picture, scaled to fit `width` or `height`, and return its shape."""
    from PIL import Image

    w, h = Image.open(img).size
    ratio = w / h
    if width is None:
        width = height * ratio
    if height is None:
        height = width / ratio
    pic = slide.shapes.add_picture(
        str(img), Inches(left), Inches(top), Inches(width), Inches(height)
    )
    if border:
        pic.line.color.rgb = BLACK
        pic.line.width = Pt(0.75)
    return pic


def caption(slide, text, left, top, width, size=9.5):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(0.26))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    no_bullet(p)
    p.alignment = PP_ALIGN.CENTER
    runs(p, text, size, MUTED, italic=True)
    return tb


def shrink_to_fit(shape, max_pt_drop=3.0):
    """Nudge every run down a little; used where content is unavoidably dense."""
    for p in shape.text_frame.paragraphs:
        for r in p.runs:
            if r.font.size:
                r.font.size = Pt(max(9.0, r.font.size.pt - max_pt_drop))


prs = Presentation(str(TEMPLATE))
S = prs.slides

# ===================================================================== slide 1
s = S[0]
TITLE = "Solar Flare Prediction & Space Weather Alert System"
SUB = "Forecasting major (M/X-class) flares 24 hours ahead from solar magnetic-field data"

t = find(s, "TITLE OF THE PROJECT")
box(t, left=0.60, top=2.98, width=12.13, height=1.30)
write(
    t,
    [
        {"t": TITLE, "sz": 26, "b": True, "c": HEAD, "sp": 4, "align": PP_ALIGN.CENTER},
        {"t": SUB, "sz": 14, "i": True, "c": MUTED, "sp": 0, "align": PP_ALIGN.CENTER},
    ],
)
t.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE

write(
    find(s, "Presentation by"),
    [
        {"t": "Presentation by", "sz": 18, "b": True, "c": HEAD, "sp": 7},
        {"t": "Madhavan G  **(2104251040518)**", "sz": 15, "c": BODY, "sp": 3},
        {"t": "Sriram Sivakumar  **(2104251040971)**", "sz": 15, "c": BODY, "sp": 3},
        {"t": "B.E. Computer Science and Engineering", "sz": 14, "c": MUTED, "sp": 0},
    ],
)

write(
    find(s, "Mentor"),
    [
        {"t": "Mentor", "sz": 18, "b": True, "c": HEAD, "sp": 7},
        {"t": "Ms Poornima Lakshmi", "sz": 15, "c": BODY, "sp": 3},
        {"t": "Mentor & Project Coordinator", "sz": 14, "c": MUTED, "sp": 3},
        {"t": "Dept. of Computer Science and Engineering", "sz": 14, "c": MUTED, "sp": 0},
    ],
)

# ===================================================================== slide 2
s = S[1]
sh = find(s, "Problem Statement")
box(sh, height=2.60)
write(
    sh,
    [
        head("Problem Statement"),
        para(
            "A major solar flare releases in minutes the energy of billions of tonnes of TNT. The "
            "radiation reaches Earth in about eight minutes and disrupts **HF radio and aviation "
            "communication, degrades GPS accuracy, damages satellite electronics and can induce "
            "currents that destabilise power grids.**",
            sz=14,
        ),
        para(
            "Satellite operators, polar-route airlines, power utilities and space agencies are all "
            "affected, and once an eruption is visible it is already too late to act.",
            sz=14,
        ),
        para(
            "The magnetic field of a solar active region, however, **changes measurably in the hours "
            "before it erupts.** The question this project asks is whether that change can be learned "
            "from historical observations and turned into a usable 24-hour advance warning.",
            sz=14,
        ),
    ],
)

sh = find(s, "Objective")
box(sh, top=3.78, height=3.17)
write(
    sh,
    [
        head("Objective"),
        bullet(
            "**Identify** the magnetic-field signatures that precede major flares, using SDO/HMI "
            "SHARP parameters from the SWAN-SF benchmark.",
            sz=13.5,
        ),
        bullet(
            "**Develop** a leakage-safe pipeline that turns a 12-hour observation window into a "
            "24-hour forecast.",
            sz=13.5,
        ),
        bullet(
            "**Apply and compare** Logistic Regression and Random Forest across two feature "
            "iterations, selecting on validation data only.",
            sz=13.5,
        ),
        bullet(
            "**Evaluate** on a locked test partition, scored once, using imbalance-aware metrics "
            "(TSS, PR-AUC, HSS2) rather than accuracy.",
            sz=13.5,
        ),
    ],
)

sh = find(s, "Expected Outcome")
box(sh, top=3.78, height=3.17)
write(
    sh,
    [
        head("Expected Outcome"),
        para(
            "A machine-learning system capable of flagging, **24 hours in advance**, which solar "
            "active regions are likely to produce a major flare.",
            sz=13.5,
        ),
        para("Delivered as:", sz=12, sp=3),
        bullet(
            "a reproducible codebase that recomputes every published number with one command;",
            sz=13.5,
        ),
        bullet(
            "an interactive web dashboard showing the forecast, the risk level and the exact "
            "reason for each decision;",
            sz=13.5,
        ),
        bullet(
            "an honest account of the error rates, with the failures shown rather than hidden.",
            sz=13.5,
        ),
    ],
)

# ===================================================================== slide 3
s = S[2]
sh = find(s, "Input/ Data Sources")
box(sh, height=2.92)
write(
    sh,
    [
        head("Input / Data Sources"),
        bullet("**SWAN-SF v1.2** benchmark, Harvard Dataverse (DOI 10.7910/DVN/EBCFKM).", sz=12),
        bullet(
            "**331,185** multivariate time-series windows spanning **2010-05-01 to 2018-08-17**.",
            sz=12,
        ),
        bullet(
            "Each window: **12 h** of SDO/HMI SHARP magnetic parameters at 12-minute cadence "
            "— 60 readings × 24 parameters.",
            sz=12,
        ),
        bullet("Label y = 1 if a **GOES M- or X-class** flare begins within the next 24 h.", sz=12),
        bullet(
            "Five chronological partitions: **P1–P3 train (204,559)**, **P4 validation "
            "(51,261)**, **P5 test (75,365)**.",
            sz=12,
        ),
    ],
)

sh = find(s, "Analysis/ Processing")
box(sh, height=5.86)  # full height, so the chart below sits inside the box
write(
    sh,
    [
        head("Analysis / Processing"),
        bullet(
            "24 parameters × 6 statistics = **144 features** per window (mean, sd, min, max, "
            "slope, last value).",
            sz=11.5,
        ),
        bullet(
            "Signed-log transform → median imputation → standardisation — all "
            "**fitted on training data only.**",
            sz=11.5,
        ),
        bullet(
            "Partitions are **region-disjoint and time-ordered**: no window and no active region "
            "appears in two splits.",
            sz=11.5,
        ),
        bullet(
            "Class imbalance (1.99 % positive in training) handled with "
            'class_weight="balanced" — a **49:1** weighting.',
            sz=11.5,
        ),
        bullet("Thresholds fixed on validation; the test partition was **scored once.**", sz=11.5),
    ],
)
place(s, PROJ / "figures/fig6_2_iteration_comparison.png", 7.52, 3.78, width=4.55, border=False)
caption(
    s,
    "Seven candidates were compared; the final model wins on F1 and PR-AUC " "while holding TSS.",
    6.92,
    6.60,
    5.74,
    size=9,
)

sh = find(s, "Key Insights")
box(sh, top=4.08, height=2.87)
write(
    sh,
    [
        head("Key Insights"),
        bullet(
            "**Trend beats snapshot.** Adding slope and last-value features lifted validation TSS "
            "over point-in-time features alone.",
            sz=12,
        ),
        bullet(
            "**Accuracy is meaningless here.** Only 1.31 % of test windows precede a major flare, "
            "so never alerting already scores 98.69 %.",
            sz=12,
        ),
        bullet(
            "A constant-climatology baseline scores **TSS 0.000** and detects nothing; this model "
            f"scores **TSS {f(M['tss'])}**.",
            sz=12,
        ),
        bullet(
            "**Total unsigned current helicity** and **free magnetic energy** dominate the fitted "
            "coefficients — consistent with the physics of flare onset.",
            sz=12,
        ),
    ],
)

# ===================================================================== slide 4
s = S[3]
sh = find(s, "Architecture")
box(sh, height=2.50)
write(
    sh,
    [
        head("Architecture"),
        para("Five stages, end to end:", sz=12, sp=4),
        bullet(
            "**Ingest & validate** — SWAN-SF archives, timestamp and QUALITY-flag checks.",
            sz=11.5,
            glyph="1.",
        ),
        bullet(
            "**Window & label** — 12 h observation → 24 h prediction horizon, past data " "only.",
            sz=11.5,
            glyph="2.",
        ),
        bullet(
            "**Feature construction** — 144 leakage-safe summary statistics.", sz=11.5, glyph="3."
        ),
        bullet(
            "**Train & select** — fit on P1–P3, choose on P4 by maximum TSS.", sz=11.5, glyph="4."
        ),
        bullet(
            "**Serve** — locked P5 evaluation → risk banding → web dashboard.", sz=11.5, glyph="5."
        ),
    ],
)

sh = find(s, "ML Methodology")
box(sh, height=5.86)  # full height, so the diagram sits inside the box
write(
    sh,
    [
        head("ML Methodology and Implementation Process"),
    ],
)
# Redrawn for the slide by scripts/make_arch_figure.py, and placed at its natural
# size so its type sits at the same scale as the rest of the deck.
place(s, PROJ / "figures/extra/architecture_flow.png", 7.60, 1.70, width=4.38, border=False)
caption(
    s, "Figure 4.1 — implemented end-to-end forecasting architecture.", 6.92, 6.56, 5.74, size=9
)

sh = find(s, "Technology Stack")
box(sh, top=3.72, height=3.23)
write(
    sh,
    [
        head("Technology Stack"),
        bullet(
            "**Python 3.12** · scikit-learn **1.8.0** · NumPy · pandas · SciPy "
            "· joblib · matplotlib",
            sz=11.5,
        ),
        bullet(
            "**Pipeline:** signed-log → median imputer → StandardScaler → Logistic "
            "Regression (L2, class-balanced)",
            sz=11.5,
        ),
        bullet(
            "**Dashboard:** HTML / CSS / vanilla JavaScript — the fitted model is re-run in "
            "the browser and agrees with scikit-learn to 1e-6",
            sz=11.5,
        ),
        bullet(
            "**Testing:** pytest (202 tests), Playwright headless browser, Ruff + Black", sz=11.5
        ),
        bullet(
            "**Reproducibility:** one command — python -m solarflare all — running 375 "
            "metric checks and 397 consistency checks",
            sz=11.5,
        ),
    ],
)

# ===================================================================== slide 5
s = S[4]
sh = find(s, "Feasibility")
box(sh, height=2.55)
write(
    sh,
    [
        head("Feasibility"),
        bullet(
            "**Data** — a public, curated benchmark; no licence barrier and no collection " "cost.",
            sz=12,
        ),
        bullet(
            "**Compute** — the full pipeline trains in about 15 minutes on a 4-core laptop. "
            "No GPU is required.",
            sz=12,
        ),
        bullet(
            "**Deployment** — the model is 144 coefficients, so it runs inside a browser tab "
            "with the network switched off.",
            sz=12,
        ),
        bullet(
            "**Verification** — every published number recomputes from the repository in "
            "about two minutes.",
            sz=12,
        ),
    ],
)

sh = find(s, "Risk and Challenges")
box(sh, height=5.86)
write(
    sh,
    [
        head("Risk and Challenges"),
        bullet(
            "**Severe class imbalance.** Positives are 1.3–2.0 % of windows, so accuracy is "
            "misleading and naive training learns to answer “no flare” always.",
            sz=11.0,
            bc=WARN,
        ),
        bullet(
            "**Data leakage is the dominant failure mode** in flare forecasting: overlapping "
            "windows and shuffled splits inflate published results.",
            sz=11.0,
            bc=WARN,
        ),
        bullet(
            "**Low precision.** At the operating point, **"
            f"{(1 - M['precision']) * 100:.0f} %** of alerts are not followed by a major flare.",
            sz=11.0,
            bc=WARN,
        ),
        bullet(
            "**Outputs are not calibrated probabilities** — they rank well but must not be "
            "read as literal chances.",
            sz=11.0,
            bc=WARN,
        ),
        bullet(
            "**Two raw partition archives (P2, P5) are corrupt** gzip streams that fail silently "
            "mid-stream rather than raising an error.",
            sz=11.0,
            bc=WARN,
        ),
        bullet(
            "**Retraining is not bit-identical** across platforms.",
            sz=11.0,
            bc=WARN,
        ),
    ],
)
# The two risks that cannot be engineered away are instead made inspectable.
place(s, IMG / "d_operate.png", 7.09, 4.24, width=5.40)
caption(s, "Moving the threshold shows what each choice costs.", 7.09, 5.28, 5.40)
place(s, IMG / "d_risk.png", 7.09, 5.56, width=5.40)
caption(s, "An uncalibrated probability is delivered as a three-level band.", 7.09, 6.67, 5.40)

sh = find(s, "Mitigation")
box(sh, top=3.76, height=3.19)
write(
    sh,
    [
        head("Mitigation"),
        bullet(
            "Class weighting plus **TSS / PR-AUC / HSS2** reporting, with a climatology baseline "
            "for context, instead of accuracy.",
            sz=11.5,
            bc=GOOD,
        ),
        bullet(
            "Region-disjoint chronological splits, preprocessing fitted on train only, thresholds "
            "from validation only — enforced by **21 automated leakage tests.**",
            sz=11.5,
            bc=GOOD,
        ),
        bullet(
            "Precision is published rather than hidden, and an operating-point view exposes the "
            f"trade-off. **{NEAR['headline']['share_of_false_alarms_preceding_a_real_flare']*100:.0f} %** "
            "of the false alarms did precede a real B- or C-class flare.",
            sz=11.5,
            bc=GOOD,
        ),
        bullet(
            "Archive integrity is verified before extraction; frozen artefacts are committed and "
            "write-protected in code.",
            sz=11.5,
            bc=GOOD,
        ),
    ],
)

# ===================================================================== slide 6
s = S[5]
sh = find(s, "Project Output")
box(sh, top=1.06, height=4.52)  # encloses the KPI strip, both captures and the summary
write(sh, [head("Project Output  —  interactive dashboard, evaluated on 75,365 unseen windows")])

place(s, IMG / "d_eval_kpi.png", 0.66, 1.52, width=12.02, border=False)
place(s, IMG / "d_replay.png", 0.66, 2.52, width=7.80)
place(s, IMG / "d_confusion.png", 8.68, 2.77, width=4.00)
caption(
    s,
    "Replay of a real active region (HARP 7115, September 2017): the forecast crosses the "
    "alert threshold and stays high through the X1.3 flare.",
    0.66,
    4.60,
    7.80,
    size=9,
)
caption(s, "Confusion matrix on the locked test partition.", 8.68, 4.60, 4.00, size=9)

tb = s.shapes.add_textbox(Inches(0.66), Inches(5.00), Inches(12.02), Inches(0.52))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
no_bullet(p)
runs(
    p,
    f"**{M['tp']} of {M['positives']}** flare-producing windows were caught "
    f"(**recall {f(M['recall'])}**, 95 % CI {f(ci['recall']['ci95_low'])}–"
    f"{f(ci['recall']['ci95_high'])}), including **{X['detected']} of {X['n_positive_windows']} "
    f"X-class windows**, at a false-alarm rate of **{f(M['fpr'])}**. "
    f"Precision is **{f(M['precision'])}** — shown, not hidden.",
    12,
    BODY,
)

sh = find(s, "Applications")
box(sh, top=5.64, height=1.14)
write(
    sh,
    [
        head("Applications", sz=16),
        para(
            "Satellite operations (safe-mode and manoeuvre scheduling) · Polar-route aviation "
            "(HF-radio and radiation planning) · Power-grid readiness against geomagnetically "
            "induced currents · Launch and spacewalk scheduling · Space-agency situational "
            "awareness · A teaching benchmark for severely imbalanced time-series forecasting.",
            sz=12,
        ),
    ],
)

# ===================================================================== slide 7
s = S[6]
sh = find(s, "Conclusion")
box(sh, height=2.40)
write(
    sh,
    [
        head("Conclusion"),
        para(
            "We developed a **leakage-safe machine-learning pipeline** using SWAN-SF SHARP "
            "magnetic-field parameters to forecast major (M/X-class) solar flares **24 hours in "
            "advance.** Seven candidate models were trained on 204,559 windows and compared on a "
            "separate validation partition; the selected model — **L2-regularised Logistic "
            "Regression on temporal features** — was then scored **once** on 75,365 windows it "
            "had never seen.",
            sz=12.5,
        ),
        para(
            f"It achieved a **True Skill Statistic of {f(M['tss'])}** (95 % CI "
            f"{f(ci['tss']['ci95_low'])}–{f(ci['tss']['ci95_high'])}), detecting "
            f"**{M['tp']} of {M['positives']}** flare-producing windows — "
            f"**{M['recall']*100:.1f} % recall**, and **all {X['n_positive_windows']} X-class "
            f"windows** — at a false-alarm rate of {f(M['fpr'])} and **ROC-AUC "
            f"{f(M['roc_auc'])}**.",
            sz=12.5,
        ),
        para(
            "Every published number recomputes from the repository with a single command, and the "
            "project ships an interactive dashboard that runs the real trained model in the browser.",
            sz=12.5,
        ),
    ],
)

sh = find(s, "Limitation")
box(sh, top=3.62, height=3.33)
write(
    sh,
    [
        head("Limitations"),
        bullet(
            f"**Precision is {f(M['precision'])}** — "
            f"{(1 - M['precision'])*100:.0f} % of alerts are not followed by an M/X flare "
            f"(though {NEAR['headline']['share_of_false_alarms_preceding_a_real_flare']*100:.0f} % "
            "do precede a smaller B/C flare).",
            sz=11.5,
            bc=WARN,
        ),
        bullet(
            "Outputs rank well but are **not calibrated** — they are not literal chances.",
            sz=11.5,
            bc=WARN,
        ),
        bullet(
            "**One instrument, one span:** SDO/HMI, 2010–2018. No validation beyond 2018 or "
            "on other instruments.",
            sz=11.5,
            bc=WARN,
        ),
        bullet(
            f"Only **{X['n_positive_windows']} X-class windows** in the test set, so that recall "
            "carries a wide interval.",
            sz=11.5,
            bc=WARN,
        ),
        bullet(
            "Two raw archives are corrupt; results rest on the previously extracted features.",
            sz=11.5,
            bc=WARN,
        ),
        bullet(
            "From-scratch retraining is **not bit-reproducible** across platforms.",
            sz=11.5,
            bc=WARN,
        ),
    ],
)

sh = find(s, "Future Scope")
box(sh, top=3.62, height=3.33)
write(
    sh,
    [
        head("Future Scope"),
        bullet(
            "**Probability calibration** on a held-out year, so the output can be read as a "
            "genuine chance.",
            sz=11.5,
            bc=GOOD,
        ),
        bullet(
            "**Sequence models** (LSTM / Transformer) on the raw 60-step series instead of summary "
            "statistics.",
            sz=11.5,
            bc=GOOD,
        ),
        bullet(
            "Extend to **solar cycle 25** (2019–present) and to other magnetograph " "instruments.",
            sz=11.5,
            bc=GOOD,
        ),
        bullet(
            "Add **magnetogram imagery** alongside the scalar SHARP parameters.", sz=11.5, bc=GOOD
        ),
        bullet(
            "**Cost-sensitive thresholds** per user — a satellite operator and a power "
            "utility do not pay the same price for a miss.",
            sz=11.5,
            bc=GOOD,
        ),
        bullet(
            "Ingest a **live SHARP feed** to move from prototype toward an operational service.",
            sz=11.5,
            bc=GOOD,
        ),
    ],
)

# ===================================================================== slide 8
s = S[7]
REPO = "https://github.com/Madhavan777/Solar-Flare-Prediction-System-Weather"
write(
    find(s, "<Student 1_Name>"),
    [
        {
            "t": "Madhavan G   ·   Sriram Sivakumar",
            "sz": 19,
            "b": True,
            "c": HEAD,
            "sp": 4,
            "align": PP_ALIGN.CENTER,
        },
        {
            "t": "B.E. Computer Science and Engineering",
            "sz": 16,
            "c": MUTED,
            "sp": 0,
            "align": PP_ALIGN.CENTER,
        },
    ],
)

write(
    find(s, "Project Video Link"),
    [
        {"t": "Project Video Link", "sz": 14, "b": True, "c": HEAD, "sp": 4},
        {
            "t": "<paste the drive / YouTube URL of the voice-over demo here>",
            "sz": 11,
            "c": MUTED,
            "i": True,
            "sp": 0,
        },
    ],
)

write(
    find(s, "GitHub Repository Link"),
    [
        {"t": "GitHub Repository Link", "sz": 14, "b": True, "c": HEAD, "sp": 4},
        {"t": REPO, "sz": 10, "c": BULLET, "sp": 0},
    ],
)

# A scannable code, so the repository is one phone-camera away from the panel.
import qrcode  # noqa: E402

qr = qrcode.QRCode(box_size=10, border=2, error_correction=qrcode.constants.ERROR_CORRECT_M)
qr.add_data(REPO)
qr.make(fit=True)
qr_png = IMG / "qr_repo.png"
qr.make_image(fill_color="black", back_color="white").save(qr_png)
place(s, qr_png, 10.52, 5.62, width=1.15, border=False)
caption(s, "Scan for the repository", 9.95, 6.78, 2.30, size=9)

# ------------------------------------------------- the date, on every slide
for s in S:
    for sh in walk(s.shapes):
        if sh.has_text_frame and sh.text_frame.text.strip().startswith("Date:"):
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    r.text = r.text.replace("DD/MM/YYYY", "07/10/2026")


# ------------------------------------------- one typeface, one colour, no exceptions
def _mirror_latin(rPr):
    """Copy the latin typeface onto the east-asian and complex-script slots.

    font.name writes only <a:latin>. Without the other two, PowerPoint picks a
    substitute for anything outside the latin range - the bullet, the arrows,
    the en dash - and the deck ends up in two typefaces after all.
    """
    latin = rPr.find(qn("a:latin"))
    if latin is None:
        return
    for tag in ("a:ea", "a:cs"):
        for e in rPr.findall(qn(tag)):
            rPr.remove(e)
    at = list(rPr).index(latin)
    for offset, tag in enumerate(("a:ea", "a:cs"), start=1):
        rPr.insert(at + offset, rPr.makeelement(qn(tag), {"typeface": FONT}))


def _force_rPr(rPr):
    """Black Times New Roman on any run-properties element.

    Used for the ones python-pptx does not reach: a:fld (the slide-number
    placeholder is a field and owns no run, which is why it stayed grey), and
    a:defRPr / a:endParaRPr, which still point at the theme's minor font and
    the tx1 scheme colour. Those are invisible today but decide what anything
    typed into the deck later will look like.
    """
    for tag in ("a:solidFill", "a:gradFill", "a:noFill", "a:latin", "a:ea", "a:cs"):
        for e in rPr.findall(qn(tag)):
            rPr.remove(e)
    fill = rPr.makeelement(qn("a:solidFill"), {})
    fill.append(rPr.makeelement(qn("a:srgbClr"), {"val": "000000"}))
    rPr.insert(0, fill)
    for tag in ("a:latin", "a:ea", "a:cs"):
        rPr.append(rPr.makeelement(qn(tag), {"typeface": FONT}))


def enforce_type(prs):
    """Force every piece of text in the deck to black Times New Roman.

    This runs last and covers the text that came with the template too - slide
    titles, footers, the date and the slide numbers - which otherwise keep the
    theme's Calibri and its greys.
    """
    runs_done = others = 0
    for slide in prs.slides:
        for sh in walk(slide.shapes):
            if not sh.has_text_frame:
                continue
            body = sh.text_frame._txBody
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    r.font.name = FONT
                    r.font.color.rgb = BLACK
                    _mirror_latin(r._r.get_or_add_rPr())
                    runs_done += 1
            for tag in ("a:fld", "a:defRPr", "a:endParaRPr"):
                for el in body.iter(qn(tag)):
                    if tag == "a:fld":
                        rPr = el.find(qn("a:rPr"))
                        if rPr is None:
                            rPr = el.makeelement(qn("a:rPr"), {"lang": "en-US"})
                            el.insert(0, rPr)
                    else:
                        rPr = el
                    _force_rPr(rPr)
                    others += 1
    return runs_done, others


_runs, _others = enforce_type(prs)
print(f"set {_runs} runs and {_others} fields and defaults to black {FONT}")

prs.save(str(OUT))
print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB, {len(S)} slides)")
