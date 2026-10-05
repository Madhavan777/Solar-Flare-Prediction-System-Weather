import json, shutil
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import sys
sys.path.insert(0, "src")
from docx_helpers import *

MEDIA = "orig_x_ref/word/media"
FIG = "figures"
RESULTS = json.load(open("results/data_audit.json"))
VAL = json.load(open("results/validation_results.json"))
TEST = json.load(open("results/test_results.json"))
SEL = json.load(open("results/selection.json"))
SEL_KEY = SEL["selected"]

doc = Document("/home/claude/work/orig.docx")
doc.core_properties.title = "Solar Flare Prediction & Space Weather Alert System Using Machine Learning"
doc.core_properties.subject = "Project-Based Learning (PBL) Report"
doc.core_properties.comments = ""
# wipe body (keep styles/theme/relationships already present)
body_el = doc.element.body
for child in list(body_el):
    if child.tag == qn('w:sectPr'):
        continue
    body_el.remove(child)
# also drop old header/footer refs we will rebuild fresh
for sec in doc.sections:
    sec.header.is_linked_to_previous = True
    sec.footer.is_linked_to_previous = True

sec = doc.sections[0]
sec.different_first_page_header_footer = True
sec.top_margin = Inches(1); sec.bottom_margin = Inches(1)
sec.left_margin = Inches(1); sec.right_margin = Inches(1)

# ---------- Header (banner on every page except the cover) ----------
hdr = sec.header
hdr.is_linked_to_previous = False
hp = hdr.paragraphs[0]; hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
hp.paragraph_format.space_after = Pt(2)
hr = hp.add_run(); hr.add_picture(f"{MEDIA}/image4.png", width=Inches(6.3))
# thin rule under the banner
pPr = hp._p.get_or_add_pPr()
pbdr = OxmlElement('w:pBdr'); bottom = OxmlElement('w:bottom')
bottom.set(qn('w:val'), 'single'); bottom.set(qn('w:sz'), '6'); bottom.set(qn('w:space'), '4'); bottom.set(qn('w:color'), '1F3A5F')
pbdr.append(bottom); pPr.append(pbdr)
sec.first_page_header.is_linked_to_previous = False
sec.first_page_header.paragraphs[0].text = ""

# ---------- Footer (page number, all pages including first) ----------
def build_footer(footer_part):
    footer_part.is_linked_to_previous = False
    fp = footer_part.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = fp.add_run(); set_run(r, size=10)
    b = OxmlElement('w:fldChar'); b.set(qn('w:fldCharType'), 'begin')
    i = OxmlElement('w:instrText'); i.set(qn('xml:space'), 'preserve'); i.text = ' PAGE '
    s = OxmlElement('w:fldChar'); s.set(qn('w:fldCharType'), 'separate')
    t = OxmlElement('w:t'); t.text = '1'
    e = OxmlElement('w:fldChar'); e.set(qn('w:fldCharType'), 'end')
    r._r.append(b); r._r.append(i); r._r.append(s); r._r.append(t); r._r.append(e)

build_footer(sec.footer)
sec.first_page_footer.is_linked_to_previous = False
sec.first_page_footer.paragraphs[0].text = ""

set_update_fields_on_open(doc)

# ============================================================ COVER PAGE
p = para(doc, "", space_after=0); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Solar Flare Prediction & Space Weather Alert System\nUsing Machine Learning")
set_run(r, size=22, bold=True, color=NAVY)
p.paragraph_format.space_after = Pt(4)

para(doc, "A PROJECT-BASED LEARNING (PBL) REPORT", size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=14, space_after=18)

p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_run(p.add_run("Submitted by"), size=12, italic=True)
p.paragraph_format.space_after = Pt(10)
for name, reg in [("Madhavan G", "2104251040518"), ("Sriram Sivakumar", "2104251040971")]:
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(2)
    set_run(p.add_run(name), size=13, bold=True)
    p2 = doc.add_paragraph(); p2.alignment = WD_ALIGN_PARAGRAPH.CENTER; p2.paragraph_format.space_after = Pt(10)
    set_run(p2.add_run(f"Register Number: {reg}"), size=11)

p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(2)
set_run(p.add_run("Submitted in partial fulfilment of the requirements for the"), size=11.5, italic=True)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(16)
set_run(p.add_run("Project-Based Learning component of Machine Learning"), size=11.5, italic=True, bold=True)

para(doc, "BACHELOR OF ENGINEERING", size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
para(doc, "in", size=11, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
para(doc, "COMPUTER SCIENCE AND ENGINEERING", size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=16)

p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run().add_picture(f"{MEDIA}/image1.png", width=Inches(2.0))
para(doc, "CHENNAI INSTITUTE OF TECHNOLOGY, CHENNAI", size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=8, space_after=2)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run().add_picture(f"{MEDIA}/image2.png", width=Inches(0.9))
para(doc, "Affiliated to Anna University, Chennai (Autonomous)", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=6, space_after=2)
para(doc, "OCTOBER 2026", size=12, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=10)

# ============================================================ VISION / MISSION
doc.add_page_break()
para(doc, "Vision of the Institute", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)
body(doc, "To be an eminent Centre for Academia, Industry and Research by imparting knowledge, "
          "relevant practices and inculcating human values to address global challenges through "
          "novelty and sustainability.")
para(doc, "Mission of the Institute", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=14, space_after=10)
for t in ["IM1. To create next-generation leaders through effective teaching-learning methodologies and "
          "instil scientific temper in them to meet global challenges.",
          "IM2. To transform lives through deployment of emerging technology, novelty and sustainability.",
          "IM3. To inculcate human values and ethical principles to cater to societal needs.",
          "IM4. To contribute towards the research ecosystem by providing a suitable, innovation-driven environment."]:
    body(doc, t, space_after=6)
para(doc, "DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=16, space_after=10)
para(doc, "Vision of the Department", size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8)
body(doc, "To excel in the emerging areas of Computer Science and Engineering by imparting knowledge, "
          "relevant practices and inculcating human values to transform students into potential resources "
          "who contribute innovatively through advanced computing in real-time situations.")
para(doc, "Mission of the Department", size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=14, space_after=8)
for t in ["DM1. To provide strong fundamentals and technical skills for Computer Science applications "
          "through effective teaching-learning methodologies.",
          "DM2. To transform the lives of students by nurturing ethical values, creativity and novelty to "
          "become entrepreneurs and establish start-ups.",
          "DM3. To habituate students to focus on sustainable solutions that improve the quality of life "
          "and the welfare of society."]:
    body(doc, t, space_after=6)

# ============================================================ BONAFIDE CERTIFICATE
doc.add_page_break()
heading1(doc, "Bonafide Certificate", add_page_break_before=False)
body(doc, "This is to certify that the Project-Based Learning report titled “Solar Flare Prediction & "
          "Space Weather Alert System Using Machine Learning” is a bonafide record of work carried out by "
          "Madhavan G (2104251040518) and Sriram Sivakumar (2104251040971) of the Department of Computer "
          "Science and Engineering, Chennai Institute of Technology, as part of the continuous, "
          "mentor-guided Project-Based Learning (PBL) component of the Machine Learning course during the "
          "academic year 2026–2027 under my supervision.", space_after=24)

sig_tbl = doc.add_table(rows=2, cols=2)
sig_tbl.autofit = True
c = sig_tbl.rows[0].cells
set_cell_text(c[0], "SIGNATURE", size=11, bold=True)
set_cell_text(c[1], "SIGNATURE", size=11, bold=True)
c2 = sig_tbl.rows[1].cells
set_cell_text(c2[0], "Dr. S. Pavithra, M.E., Ph.D.\nProfessor and Head\nDept. of Computer Science and Engineering\n"
                     "Chennai Institute of Technology, Chennai – 69.", size=10.5)
set_cell_text(c2[1], "Ms Poornima Lakshmi\nHOD / Mentor\nDept. of Computer Science and Engineering\n"
                     "Chennai Institute of Technology, Chennai – 69.", size=10.5)
for row in sig_tbl.rows:
    for cell in row.cells:
        cell._tc.get_or_add_tcPr()  # ensure exists
for row in sig_tbl.rows:
    tr = row._tr
    for tc in tr.findall(qn('w:tc')):
        tcPr = tc.find(qn('w:tcPr'))
        if tcPr is None:
            tcPr = OxmlElement('w:tcPr'); tc.insert(0, tcPr)
        for edge in ('top', 'left', 'bottom', 'right'):
            b = OxmlElement(f'w:{edge}')
            b.set(qn('w:val'), 'nil')
            tcBorders = tcPr.find(qn('w:tcBorders'))
            if tcBorders is None:
                tcBorders = OxmlElement('w:tcBorders'); tcPr.append(tcBorders)
            tcBorders.append(b)
doc.add_paragraph().paragraph_format.space_after = Pt(20)
body(doc, "Submitted for the final review held on ______________________.", space_after=18)
para(doc, "Internal Examiner", size=12, bold=True, align=WD_ALIGN_PARAGRAPH.RIGHT)

# ============================================================ DECLARATION
doc.add_page_break()
heading1(doc, "Declaration", add_page_break_before=False)
body(doc, "We jointly declare that the PBL report on “Solar Flare Prediction & Space Weather Alert System "
          "Using Machine Learning” is the result of original work done by us and, to the best of our "
          "knowledge, similar work has not been submitted to Anna University, Chennai for the requirement "
          "of the Degree of Bachelor of Engineering. This PBL report is submitted in partial fulfilment of "
          "the requirement for the award of the Degree of Bachelor of Engineering in Computer Science and "
          "Engineering.", space_after=22)
para(doc, "Signature", size=12, italic=True, space_after=14)
for n in ["Madhavan G", "Sriram Sivakumar"]:
    para(doc, n, size=12, bold=True, space_after=4)
para(doc, "Place: Chennai", size=12, space_before=16, space_after=2)
para(doc, "Date:", size=12, space_after=2)

# ============================================================ ACKNOWLEDGEMENT
doc.add_page_break()
heading1(doc, "Acknowledgement", add_page_break_before=False)
for t in [
    "We wish to express our sincere gratitude to our honourable Chairman Shri. P. Sriram for providing "
    "immense facilities at our institution.",
    "We are very proud to render our thanks to our Principal Dr. A. Ramesh, M.E., Ph.D., for the "
    "facilities and encouragement given by him toward the progress and completion of our project.",
    "We would like to express special thanks of gratitude to our Dean Dr. V. Srinivasa Rao, M.E., Ph.D., "
    "who has been a key source of motivation to us throughout the completion of our course and project work.",
    "We proudly render our sincere gratitude to Ms Poornima Lakshmi, HOD and Mentor, Department of "
    "Computer Science and Engineering, for her valuable guidance, constructive feedback and continuous "
    "support throughout the PBL work.",
    "We would like to express our sincere thanks to Ms Poornima Lakshmi, Project Coordinator, Department "
    "of Computer Science and Engineering, for her guidance and suggestions throughout the project.",
    "We wish to acknowledge the support and valuable suggestions provided by Mr S E Neela Kandan, Class "
    "Advisor, Department of Computer Science and Engineering, and others who contributed to the progress "
    "of this project.",
]:
    body(doc, t, space_after=12)
para(doc, "Madhavan G", size=12, bold=True, align=WD_ALIGN_PARAGRAPH.RIGHT, space_after=2)
para(doc, "Sriram Sivakumar", size=12, bold=True, align=WD_ALIGN_PARAGRAPH.RIGHT)

# ============================================================ ABSTRACT
doc.add_page_break()
heading1(doc, "Abstract", add_page_break_before=False)
t = TEST[SEL_KEY]
abstract = (
    "Solar flares are sudden releases of magnetic energy from the Sun that can disturb space-weather "
    "conditions and affect communication, navigation and other technology-dependent systems. This project "
    "implements a Machine Learning-based forecasting prototype that uses the Space-Weather ANalytics for "
    "Solar Flares (SWAN-SF) benchmark dataset — 331,185 twelve-hour observation windows drawn from SDO/HMI "
    "SHARP magnetic-field parameters across five chronological partitions (2010–2018) — to estimate whether "
    "an active region will produce a major (M- or X-class) flare in the 24 hours following the observation "
    "cutoff. A leakage-safe pipeline converts each 12-hour window into a 144-dimensional feature vector "
    "(24 SHARP parameters × 6 temporal statistics), excluding all flare-derived and GOES X-ray columns from "
    "the predictors. The data are split by SWAN-SF partition — training on partitions 1–3, validating on "
    "partition 4 and locking partition 5 for a single final test — so that no active region appears in more "
    "than one split. Logistic Regression was evaluated as an interpretable baseline and Random Forest as a "
    "nonlinear refinement; seven candidate configurations in total were compared on the validation partition "
    "using the True Skill Statistic (TSS), with PR-AUC as a tie-break, and the selected configuration was "
    "evaluated exactly once on the locked test partition. The final model — a Logistic Regression classifier "
    "trained on the temporal feature representation — achieved a test-set TSS of "
    f"{t['tss']:.3f}, recall of {t['recall']:.3f}, PR-AUC of {t['pr_auc']:.3f} and ROC-AUC of {t['roc_auc']:.3f} "
    "on 75,365 held-out windows containing 990 major-flare events, correctly flagging "
    f"{t['tp']} of {t['tp']+t['fn']} major-flare windows at the validation-selected operating threshold. Random "
    "Forest did not exceed Logistic Regression on the validation selection criterion in this study, which is "
    "reported and discussed rather than assumed in advance. The trained model, its predicted probability, a "
    "derived risk level and an alert message are served through a web dashboard, with NOAA GOES current "
    "monitoring shown as separate contextual information rather than as a model input."
)
body(doc, abstract, space_after=14)
para(doc, "Keywords: Solar Flare Forecasting, Space Weather, Machine Learning, SWAN-SF, Logistic Regression, "
          "Random Forest, Leakage-Safe Evaluation", size=11.5, italic=True)

# ============================================================ TOC / LOF / LOT / LOA
doc.add_page_break()
heading1(doc, "Table of Contents", add_page_break_before=False)
field_toc(doc)

# ---------------- List of Figures ----------------
doc.add_page_break()
heading1(doc, "List of Figures", add_page_break_before=False)
FIGURES = [
    ("4.1", "Proposed / Implemented End-to-End Solar Flare Forecasting Architecture", "Figure4_1"),
    ("5.1", "Web dashboard overview of the Solar Flare Prediction and Space Weather Alert System", "Figure5_1"),
    ("5.2", "Forecast configuration showing the observation window and prediction window of a real test-partition example", "Figure5_2"),
    ("5.3", "Model prediction showing the estimated probability of a future M/X-class solar flare", "Figure5_3"),
    ("5.4", "User-facing risk level and alert message derived from the model probability", "Figure5_4"),
    ("5.5", "Model explainability view: standardized Logistic Regression coefficients", "Figure5_5"),
    ("5.6", "Model evaluation summary view of the dashboard (test-partition metrics)", "Figure5_6"),
    ("6.1", "Confusion matrix of the final model on the locked test partition (P5)", "Figure6_1"),
    ("6.2", "Model performance comparison across Iteration 1, Iteration 2 and the final approach", "Figure6_2"),
    ("6.3", "Precision-Recall curve of the final model on the test partition", "Figure6_3"),
    ("6.4", "ROC curve of the final model on the test partition", "Figure6_4"),
    ("6.5", "Top 15 standardized Logistic Regression coefficients of the final model", "Figure6_5"),
]
for num, title, bm in FIGURES:
    lof_lot_entry(doc, f"Figure {num} — {title}", bm)

# ---------------- List of Tables ----------------
doc.add_page_break()
heading1(doc, "List of Tables", add_page_break_before=False)
TABLES = [
    ("2.1", "SWAN-SF Dataset Characteristics", "Table2_1"),
    ("2.2", "Related Approaches Summary", "Table2_2"),
    ("3.1", "Weekly PBL Progress Log", "Table3_1"),
    ("3.2", "Hardware and Software Requirements", "Table3_2"),
    ("4.1", "Predictor and Excluded-Variable Summary (Leakage Audit)", "Table4_1"),
    ("4.2", "Model Configurations Evaluated", "Table4_2"),
    ("5.1", "Web Dashboard Components", "Table5_1"),
    ("6.1", "Results Across Iterations (Locked Test Partition)", "Table6_1"),
    ("A.1", "Complete Model Comparison — All Candidates (Validation and Test)", "TableA_1"),
    ("A.3", "Self and Peer Assessment", "TableA_3"),
]
for num, title, bm in TABLES:
    lof_lot_entry(doc, f"Table {num} — {title}", bm)

# ---------------- List of Abbreviations ----------------
doc.add_page_break()
heading1(doc, "List of Abbreviations", add_page_break_before=False)
ABBR = [
    ("ML", "Machine Learning"), ("SWAN-SF", "Space-Weather ANalytics for Solar Flares"),
    ("PBL", "Project-Based Learning"), ("SDO", "Solar Dynamics Observatory"),
    ("HMI", "Helioseismic and Magnetic Imager"), ("SHARP", "Space-weather HMI Active Region Patch"),
    ("GOES", "Geostationary Operational Environmental Satellite"),
    ("NOAA", "National Oceanic and Atmospheric Administration"),
    ("NASA", "National Aeronautics and Space Administration"), ("MVTS", "Multivariate Time Series"),
    ("LR", "Logistic Regression"), ("RF", "Random Forest"),
    ("ROC-AUC", "Area Under the Receiver Operating Characteristic Curve"),
    ("PR-AUC", "Area Under the Precision-Recall Curve"), ("TSS", "True Skill Statistic"),
    ("HSS", "Heidke Skill Score"), ("FAR", "False Alarm Ratio"),
    ("UI", "User Interface"), ("API", "Application Programming Interface"),
    ("AR", "Active Region"),
]
tbl = doc.add_table(rows=0, cols=2)
for k, v in ABBR:
    row = tbl.add_row().cells
    set_cell_text(row[0], k, size=11.5, bold=True)
    set_cell_text(row[1], v, size=11.5)

# ============================================================ TEAM ROLES
doc.add_page_break()
heading1(doc, "Team Roles and Responsibilities", add_page_break_before=False)
body(doc, "The table below reflects the division of work as actually carried out during the PBL cycle. "
          "Final contribution percentages and remarks are recorded in Appendix A.3.")
make_table(doc, ["Team Member", "Responsibilities"], [
    ["Madhavan G", "SWAN-SF acquisition and feature-extraction pipeline; leakage audit and partition-based "
                   "split design; Logistic Regression and Random Forest experiments; validation-based model "
                   "selection and test-set evaluation; result figures."],
    ["Sriram Sivakumar", "Exploratory data analysis of the downloaded partitions; risk-scoring and alert "
                         "logic design; web dashboard implementation and screenshots; report structuring, "
                         "documentation and reproducibility checks."],
], col_widths=[Inches(1.6), Inches(4.7)])

# ============================================================ CHAPTER 1
heading1(doc, "Chapter 1 — Introduction")
heading2(doc, "1.1 Background")
body(doc, "Solar flares are sudden releases of magnetic energy from the Sun, associated with rapidly "
          "evolving magnetic fields in solar active regions. They are a central component of space "
          "weather: energetic flares can disturb the ionosphere, degrade high-frequency radio and GNSS "
          "signals, and increase risk to satellites and astronauts. As satellite communication, navigation "
          "and remote-sensing services have grown, advance warning of major flare activity has become "
          "increasingly valuable, even though the underlying magnetic-field dynamics remain only partially "
          "understood.")
body(doc, "This makes solar-flare occurrence a natural candidate for data-driven forecasting, provided the "
          "prediction cutoff, input variables and evaluation protocol are defined carefully so that the "
          "model never sees information from after the moment it is meant to be forecasting from. This "
          "project investigates a classical Machine Learning pipeline built on the SWAN-SF benchmark, with "
          "explicit attention to temporal causality, class imbalance, leakage prevention and reproducibility, "
          "and communicates model output through a web dashboard rather than presenting it as an "
          "operational warning service.")

heading2(doc, "1.2 Driving Question")
body_runs(doc, [("Can solar-active-region observations available before a defined prediction cutoff be used to "
                 "estimate the likelihood of a major (M- or X-class) solar flare in a future 24-hour forecast "
                 "window, and how do Logistic Regression and Random Forest compare under a leakage-safe, "
                 "partition-based temporal evaluation?", {"italic": True})])
body(doc, "The implemented configuration uses a 12-hour observation window followed by a 24-hour prediction "
          "window, consistent with SWAN-SF-based flare-forecasting studies in the literature (Chapter 2). "
          "The target is binary: whether the largest flare recorded for the active region in the 24-hour "
          "window is M-class or larger (an M/X event) versus not (the SWAN-SF “NF” — no-flare — "
          "partition, whose windows have a maximum flare of B- or C-class or none).")

heading2(doc, "1.3 Objectives")
for t in [
    "Acquire and verify the SWAN-SF active-region time-series data directly from the Harvard Dataverse "
    "release (DOI 10.7910/DVN/EBCFKM, v1.2) and audit its structure, labels, timestamps and quality flags.",
    "Construct a leakage-safe, fixed-length feature representation from observations available before the "
    "prediction cutoff, explicitly excluding all flare-derived and GOES X-ray columns.",
    "Implement Logistic Regression as an interpretable baseline and Random Forest as a nonlinear "
    "refinement, and formulate the train/validation/test split by SWAN-SF partition to avoid temporal "
    "leakage between highly overlapping windows.",
    "Evaluate the candidate models using metrics appropriate to a rare-event, class-imbalanced problem "
    "(Precision, Recall, F1, PR-AUC, ROC-AUC, TSS) and select the final model on validation evidence only.",
    "Iteratively refine the feature representation and modelling approach across two iterations, documenting "
    "what changed and why, and evaluate the selected configuration once on a locked test partition.",
    "Develop a web dashboard that serves the trained model's probability, a derived risk level and an "
    "alert message, and demonstrate the complete pipeline end to end.",
]:
    body(doc, t, bullet=True, space_after=6)

heading2(doc, "1.4 Scope and Limitations")
for t in [
    "Prediction scope: forecast a major flare event within a defined future window rather than detect a "
    "flare after it has already happened.",
    "Data scope: SWAN-SF v1.2 (Harvard Dataverse, DOI 10.7910/DVN/EBCFKM) is the sole training and "
    "evaluation source; 331,185 observation windows across five partitions (2010-05-01 to 2018-08-17) were "
    "downloaded and processed for this report.",
    "Validation scope: the split is by SWAN-SF partition (chronological, no shuffling) because consecutive "
    "windows of the same active region overlap heavily — in the downloaded data, 97.9% of consecutive "
    "windows per active region are only one hour apart — so a naive random split would place near-duplicate "
    "windows on both sides of the split and overstate skill.",
    "Dashboard scope: the dashboard presents model predictions, risk communication and model "
    "explainability. NOAA GOES live monitoring is discussed as a separate, non-implemented contextual "
    "extension (Chapter 8) and is never represented as a model input.",
    "Limitations: the prototype is trained and evaluated on historical data only; it is not an operational "
    "forecasting service, and generalisation to other instruments, real-time operational conditions or "
    "solar cycles beyond 2010–2018 is not claimed.",
]:
    body(doc, t, bullet=True, space_after=6)

doc.save("out/report_build.docx")
print("chapter 1 done")

# ============================================================ CHAPTER 2
heading1(doc, "Chapter 2 — Concept Exploration")
heading2(doc, "2.1 Related Approaches")

heading3(doc, "2.1.1 Classical and Time-Series Forecasting")
body_runs(doc, [
    ("Earlier solar-flare prediction studies have treated the task as classification using properties of "
     "solar active regions. Hamdi et al. demonstrated a time-series classification approach and found that "
     "statistical summaries of an active-region parameter could outperform a single point-in-time "
     "representation, supporting the use of temporal information ", {}), ("[1]", {"bold": True}), (".", {}),
])
body_runs(doc, [
    ("Studies of SWAN-SF show that its sliding-window construction introduces temporal coherence. A naive "
     "random train/test split can therefore place highly related observations in both partitions and create "
     "an artificially optimistic estimate of forecasting skill ", {}), ("[2]", {"bold": True}), (".", {}),
])

heading3(doc, "2.1.2 Feature Selection and Data Preparation")
body_runs(doc, [
    ("Feature selection is important because the multivariate magnetic-field representation contains "
     "correlated and heterogeneous variables. Yeolekar et al. compared 24 feature-subset-selection methods "
     "", {}), ("[4]", {"bold": True}),
    (", while Velanki et al. studied mutual information, minimum redundancy maximum relevance and "
     "distance-based methods for identifying useful predictors on SWAN-SF ", {}), ("[5]", {"bold": True}), (".", {}),
])
body_runs(doc, [
    ("Recent SWAN-SF research also identifies substantial missingness and demonstrates that imputation, "
     "normalization and sampling decisions can influence forecasts ", {}), ("[6]", {"bold": True}),
    (". The project therefore treats preprocessing as an explicit, testable component rather than an "
     "undocumented cleaning step — in the downloaded data, 10.4% of observation windows contain at least "
     "one missing SHARP value, which the pipeline handles with median imputation fit on the training "
     "partitions only (Chapter 4).", {}),
])

heading3(doc, "2.1.3 Classical ML Models Relevant to This PBL")
body_runs(doc, [
    ("Logistic Regression is selected as the initial baseline because it is interpretable, computationally "
     "efficient and provides a probabilistic linear classifier ", {}), ("[14]", {"bold": True}),
    (". Random Forest is selected as a complementary refinement because an ensemble of decision trees can "
     "represent nonlinear relationships and feature interactions that a linear model may not capture "
     "", {}), ("[15]", {"bold": True}), (".", {}),
])

heading3(doc, "2.1.4 Advanced Approaches in the Literature")
body_runs(doc, [
    ("Research on SWAN-SF has expanded to LSTM-based sequence models ", {}), ("[7]", {"bold": True}),
    (", contrastive representation learning ", {}), ("[9]", {"bold": True}),
    (" and CNN-based forecasting ", {}), ("[8]", {"bold": True}),
    (". These approaches provide context for the field but are not part of the current implementation "
     "plan. The present PBL intentionally keeps the model family classical so that the team can investigate "
     "data quality, rare-event validation and explainability in depth.", {}),
])

heading3(doc, "2.1.5 Dataset Selection Rationale")
body_runs(doc, [
    ("SWAN-SF is selected as the primary benchmark because it was created specifically for solar-flare "
     "forecasting. The dataset descriptor reports 4,098 multivariate time-series collections derived from "
     "SDO/HMI SHARP observations, with 51 parameters and integrated NOAA/GOES flare information. The "
     "records are stored as 12-minute-cadence CSV time series associated with solar active regions "
     "", {}), ("[10]", {"bold": True}),
    (". The dataset is publicly identified in Harvard Dataverse as SWAN-SF, DOI 10.7910/DVN/EBCFKM "
     "", {}), ("[11]", {"bold": True}), (".", {}),
])

heading2(doc, "2.2 Summary Table")
make_table(
    doc,
    ["Characteristic", "Value"],
    [
        ["Source", "Harvard Dataverse, DOI 10.7910/DVN/EBCFKM (v1.2)"],
        ["Partitions used", "5 (P1–P5), chronological, 2010-05-01 to 2018-08-17"],
        ["Total observation windows processed", "331,185"],
        ["Observation window length", "12 hours"],
        ["Prediction window length", "24 hours"],
        ["SHARP magnetic-field parameters", "24"],
        ["Engineered features per window", "144 (24 parameters × 6 temporal statistics)"],
        ["Positive-class definition", "Max flare in window is M-class or X-class (SWAN-SF “FL” folder)"],
        ["Positive rate, locked test partition (P5)", "990 / 75,365 windows (1.31%)"],
        ["Missingness", "10.4% of windows contain ≥ 1 missing SHARP value (mean per-column NaN fraction 0.60%)"],
    ],
    col_widths=[Inches(2.6), Inches(3.7)],
    caption="SWAN-SF Dataset Characteristics", cap_num="2.1",
)

make_table(
    doc,
    ["Ref.", "Approach", "Relevance to This Project"],
    [
        ["[1]", "Time-series classification on SWAN-SF", "Supports representing each window as a multivariate time series rather than a single snapshot"],
        ["[2]", "Class imbalance & temporal coherence in solar flare data", "Motivates the partition-based (non-random) train/validation/test split used in Chapter 4"],
        ["[3]", "Robust sampling of rare events", "Informs the choice of class-imbalance-aware metrics (TSS, PR-AUC) over raw accuracy"],
        ["[4]", "Comparative study of 24 feature-selection methods", "Contextualises the fixed 144-feature temporal representation adopted here"],
        ["[5]", "Mutual-information / mRMR feature selection", "Alternative feature-selection strategy not implemented in this PBL; noted as future work"],
        ["[6]", "Preprocessing and sampling effects on SWAN-SF forecasts", "Supports treating imputation and scaling as explicit, auditable pipeline steps (Chapter 4)"],
        ["[7]", "Sequence-model (LSTM) end-to-end classification", "Identifies deep sequence models as a non-implemented extension (Chapter 8)"],
        ["[8]", "CNN on multivariate magnetic-field time series", "Identifies CNN-based forecasting as a non-implemented extension (Chapter 8)"],
        ["[9]", "Contrastive representation learning under imbalance", "Identifies representation learning as a non-implemented extension (Chapter 8)"],
        ["[10]", "SWAN-SF dataset descriptor", "Source of the dataset’s structural and parameter documentation used throughout this report"],
    ],
    col_widths=[Inches(0.6), Inches(2.7), Inches(3.0)],
    font_size=9.5,
    caption="Related Approaches Summary", cap_num="2.2",
)

heading2(doc, "2.3 What This Told Us")
body_runs(doc, [
    ("The literature led the team to a conservative but technically defensible starting point: use SWAN-SF "
     "as the primary benchmark; formulate the task as future-window major-flare classification; start with "
     "Logistic Regression as the baseline; and test Random Forest as a nonlinear refinement. The literature "
     "on temporal coherence and class imbalance directly shapes the design of the data split, preprocessing "
     "and evaluation metrics ", {}), ("[2], [3]", {"bold": True}), (".", {}),
])
body(doc, "These design decisions were carried through into the actual implementation described in Chapters "
          "4–6: the final feature subset, partition-based split, model configurations, decision threshold "
          "and model-selection outcome were all determined from real experiments on the downloaded SWAN-SF "
          "data, not assumed in advance.")

doc.save("out/report_build.docx")
print("chapter 2 done")

# ============================================================ CHAPTER 3
heading1(doc, "Chapter 3 — Project Planning and Team Organisation")

heading2(doc, "3.1 Weekly PBL Progress Log")
body(doc, "The table below is the planning skeleton for the PBL cycle. The week numbers, exact dates and "
          "mentor remarks are intentionally left as pending items, consistent with the project rule that no "
          "review date, duration or mentor feedback is invented in this report; they must be filled in from "
          "the actual PBL record and mentor sign-offs before final submission.")
PEND = "[PENDING — ACTUAL INFORMATION REQUIRED]"
MENT = "[MENTOR REMARK — TO BE ENTERED]"
make_table(
    doc,
    ["Week", "Milestone / Task", "Work Done", "Mentor Remarks"],
    [
        [PEND, "Problem framing and dataset verification", "Confirmed the project question; inspected the SWAN-SF partition structure, filename encoding, folder labels (FL/NF) and quality columns.", MENT],
        [PEND, "Concept exploration and modelling plan", "Reviewed the literature in Chapter 2 and froze the 12-hour observation / 24-hour prediction formulation.", MENT],
        [PEND, "Data acquisition and audit", "Downloaded all 5 SWAN-SF partitions from Harvard Dataverse; audited row counts, class balance, missingness and cross-partition active-region overlap.", MENT],
        [PEND, "Feature engineering and leakage audit", "Implemented the 144-feature temporal representation and excluded all flare-derived and GOES columns from the predictor set.", MENT],
        [PEND, "Iteration 1 — Logistic Regression baseline", "Trained and validated the last-value Logistic Regression baseline under the fixed partition-based protocol.", MENT],
        [PEND, "Iteration 2 — temporal features and Random Forest", "Extended to the 144-feature temporal representation; trained Random Forest configurations for comparison.", MENT],
        [PEND, "Model selection and locked test evaluation", "Selected the final configuration on validation TSS/PR-AUC and evaluated it exactly once on the locked test partition (P5).", MENT],
        [PEND, "Web dashboard", "Built and screenshotted the dashboard serving the trained model's probability, risk level and alert message.", MENT],
        [PEND, "Report writing, evidence capture and review", "Compiled the report from real pipeline outputs; completed visual QA and the ML technical audit.", MENT],
    ],
    col_widths=[Inches(0.6), Inches(1.9), Inches(2.9), Inches(1.1)],
    font_size=9,
    caption="Weekly PBL Progress Log", cap_num="3.1",
)

heading2(doc, "3.2 Hardware and Software Requirements")
make_table(
    doc,
    ["Category", "Requirement / Version Actually Used"],
    [
        ["Processor / RAM", PEND + " (feature extraction was run on a 2-CPU cloud VM; model training on a cloud compute environment)"],
        ["Programming language", "Python 3.11"],
        ["Core libraries", "pandas 3.0.2, NumPy 2.4.4, scikit-learn 1.8.0, matplotlib 3.10.9, joblib 1.5.3"],
        ["Dashboard / evidence capture", "HTML5, CSS3, vanilla JavaScript; Playwright 1.56.0 with headless Chromium for dashboard screenshots"],
        ["Document assembly", "python-docx 1.2.0"],
        ["Primary dataset", "SWAN-SF benchmark dataset, Harvard Dataverse, DOI 10.7910/DVN/EBCFKM (v1.2)"],
        ["Version control", PEND],
    ],
    col_widths=[Inches(2.0), Inches(4.3)],
    caption="Hardware and Software Requirements", cap_num="3.2",
)

heading2(doc, "3.3 Feasibility")
body(doc, "The project proved feasible within the PBL timeframe: SWAN-SF is publicly accessible from Harvard "
          "Dataverse, the full five-partition download and feature extraction completed on commodity "
          "hardware, and Logistic Regression and Random Forest are both computationally inexpensive on the "
          "resulting 144-feature representation (full training and evaluation of all seven candidate "
          "configurations completed in a few minutes). The dashboard remained a lightweight, static "
          "presentation layer that reads a single JSON file produced by the trained pipeline, which kept "
          "its implementation effort proportionate to the PBL scope.")

doc.save("out/report_build.docx")
print("chapter 3 done")

# ============================================================ CHAPTER 4
heading1(doc, "Chapter 4 — Iterative Design and Development")

heading2(doc, "4.1 System Architecture")
body(doc, "The architecture separates data preparation, forecasting, evaluation and user presentation so "
          "that each stage can be tested independently. The model receives only features constructed from "
          "SHARP observations available before the prediction cutoff; flare-derived and GOES X-ray columns "
          "are excluded from the predictor set at the feature-construction stage, not filtered out later.")
add_figure(doc, f"{FIG}/fig4_1_architecture.png", "4.1", "Implemented end-to-end system architecture, from raw SWAN-SF partitions to the web dashboard.")
body(doc, "Historical SWAN-SF partitions are validated for timestamps, missingness and quality indicators. "
          "Each 12-hour observation window is transformed into a 144-dimensional feature vector using "
          "leakage-safe temporal statistics. A partition-based train/validation/test split is used for all "
          "seven model configurations. The selected model's probability is converted into a risk level and "
          "alert message and displayed through the web dashboard. NOAA GOES current monitoring is shown as "
          "separate contextual information in the dashboard and is not used as a model input.")

heading2(doc, "4.2 Predictor and Excluded-Variable Summary")
body(doc, "Only the 24 SHARP magnetic-field parameters observed inside each 12-hour window are used as "
          "predictors, expanded into 144 temporal features (6 statistics × 24 parameters). The columns "
          "below are present in the SWAN-SF files but are deliberately excluded because they either encode "
          "the outcome being predicted or fall outside the observation window.")
make_table(
    doc,
    ["Excluded Category", "Columns", "Reason for Exclusion"],
    [
        ["Label / flare-derived", "BFLARE, CFLARE, MFLARE, XFLARE (+ _LOC, _LABEL, _LABEL_LOC variants)", "Directly encode the flare outcome; using them would leak the target into the predictors"],
        ["GOES X-ray measurements", "XR_MAX, XR_QUAL", "Derived from the same flare event the model is trying to forecast"],
        ["Position / geometry metadata", "CRVAL1, CRLN_OBS, CRLT_OBS, CRVAL2, HC_ANGLE, LAT_MIN, LON_MIN, LAT_MAX, LON_MAX", "Describe active-region location on the solar disk, not physical flare-productivity signal"],
        ["Quality / flags", "QUALITY, SPEI, IS_TMFI", "Used only to audit data quality (Chapter 6); not used as predictors"],
    ],
    col_widths=[Inches(1.5), Inches(3.1), Inches(1.9)],
    font_size=9.5,
    caption="Predictor and Excluded-Variable Summary (Leakage Audit)", cap_num="4.1",
)

heading2(doc, "4.3 Model Configurations Evaluated")
body(doc, "Seven candidate configurations were trained on the training partitions (P1–P3) and compared "
          "on the validation partition (P4). All preprocessing (median imputation, standardization) was "
          "fitted on the training partitions only; class imbalance was handled with class weighting "
          "computed from the training labels, never by altering validation or test distributions.")
make_table(
    doc,
    ["Key", "Iteration", "Model", "Configuration"],
    [
        ["I1_LR_last", "1", "Logistic Regression", "24 last-value SHARP features, median imputation, standard scaling, C=1.0"],
        ["I2_LR_temporal_C0.01", "2", "Logistic Regression", "144 temporal features, signed-log transform, median imputation, scaling, C=0.01"],
        ["I2_LR_temporal_C0.1", "2", "Logistic Regression", "144 temporal features, signed-log transform, median imputation, scaling, C=0.1"],
        ["I2_LR_temporal_C1", "2", "Logistic Regression", "144 temporal features, signed-log transform, median imputation, scaling, C=1.0"],
        ["I2_RF_leaf5", "2", "Random Forest", "144 temporal features, 300 trees, min_samples_leaf=5, max_depth=None"],
        ["I2_RF_leaf20", "2", "Random Forest", "144 temporal features, 300 trees, min_samples_leaf=20, max_depth=None"],
        ["I2_RF_leaf50_d16", "2", "Random Forest", "144 temporal features, 300 trees, min_samples_leaf=50, max_depth=16"],
    ],
    col_widths=[Inches(1.5), Inches(0.6), Inches(1.3), Inches(3.1)],
    font_size=9,
    caption="Model Configurations Evaluated", cap_num="4.2",
)

heading2(doc, "4.4 Iteration 1 — Baseline")
body(doc, "The baseline (I1_LR_last) is a Logistic Regression classifier using only the 24 point-in-time "
          "SHARP values at the prediction cutoff, with median imputation, standard scaling and balanced "
          "class weighting, and no signed-log transform. It establishes what a minimal, single-snapshot "
          "representation can achieve before temporal statistics are added.")
body(doc, "On the validation partition (P4), the baseline reached TSS = 0.846, PR-AUC = 0.482, F1 = 0.264 "
          "(precision = 0.153, recall = 0.972) at its validation-selected threshold of 0.388. Evaluated once "
          "on the locked test partition (P5), it reached TSS = 0.867, PR-AUC = 0.462, F1 = 0.223, "
          "ROC-AUC = 0.980 and accuracy = 0.913. The very high recall at low precision is expected for a "
          "rare-event problem (990 of 75,365 test windows are positive, 1.31%) under class-balanced "
          "training, and motivated the temporal-feature refinement in Iteration 2.")

heading2(doc, "4.5 Iteration 2 — Refinement")
body(doc, "Iteration 2 expands each observation window from 24 point-in-time values to 144 temporal "
          "features (last, mean, standard deviation, minimum, maximum and slope per SHARP parameter), "
          "applies a fixed signed-log transform before scaling, and compares three Logistic Regression "
          "regularization strengths against three Random Forest configurations — six configurations in "
          "total, evaluated under the same partition-based protocol as the baseline.")
body(doc, "All six Iteration-2 configurations improved validation TSS over the baseline (0.848–0.862 vs. "
          "0.846), and the Logistic Regression variants also improved validation PR-AUC (up to 0.500 at "
          "C=0.01, vs. 0.482 for the baseline). The Random Forest configurations reached comparable or "
          "slightly higher TSS on the test partition (0.860–0.865) but consistently lower PR-AUC on both "
          "validation (0.233–0.471) and test (0.352–0.399) than the Logistic Regression variants — i.e. "
          "more false alarms at an equivalent recall level. This comparison is discussed in full in Chapter "
          "6; it is reported here as the observed outcome of Iteration 2, not assumed in advance.")

heading2(doc, "4.6 Final Approach")
body(doc, "The final model was not pre-declared. Following the decision rule fixed before test evaluation "
          "(maximum validation TSS, ties broken by validation PR-AUC), the selected configuration is "
          "I2_LR_temporal_C0.01 — Logistic Regression on the 144-feature temporal representation with "
          "C=0.01 — which reached the highest validation TSS (0.862) among all seven candidates. Random "
          "Forest did not exceed Logistic Regression on the validation selection criterion in this study: "
          "despite comparable or slightly higher test-set TSS, every Random Forest configuration had "
          "materially lower PR-AUC than the selected Logistic Regression configuration at both validation "
          "and test time. The selected configuration was evaluated exactly once on the locked test "
          "partition (P5), after the selection decision was written to disk, to avoid retrospective model "
          "selection based on the test result.")

heading3(doc, "Logistic Regression")
body(doc, "Logistic Regression models the log-odds of a major flare as a linear function of the 144 "
          "engineered features. It is used with class_weight=\"balanced\" (computed from the training "
          "labels), a signed-log transform applied before scaling, and StandardScaler fitted on the "
          "training partitions only. Its linear coefficients are also used for the explainability view in "
          "Chapter 6 and the dashboard (Chapter 5).")

heading3(doc, "Random Forest")
body(doc, "Random Forest combines 300 bootstrap-sampled decision trees with randomized feature selection "
          "(max_features=\"sqrt\") and class_weight=\"balanced_subsample\". Three configurations were "
          "evaluated, varying min_samples_leaf (5, 20, 50) and max_depth (None, None, 16) to trade off "
          "variance against overfitting on the rare positive class; median imputation was applied but no "
          "scaling, since tree splits are scale-invariant.")

heading2(doc, "4.7 Training Procedure")
body(doc, "The training pipeline follows a strict partition-based temporal protocol, not a random split: "
          "training uses partitions P1–P3, validation uses P4, and the test partition P5 is held out and "
          "evaluated only once. No active region is shared across partitions in the downloaded data "
          "(verified directly, Chapter 6), and 97.9% of consecutive same-active-region windows are only one "
          "hour apart, which is why a naive random split would overstate skill (Chapter 2).")
for t in [
    "Observation / prediction windows: 12-hour observation window, 24-hour prediction window, fixed by the SWAN-SF FL/NF folder construction.",
    "Preprocessing: median imputation and (for Logistic Regression) standard scaling, fitted on the training partitions only and applied unchanged to validation and test data.",
    "Imbalance handling: class_weight=\"balanced\" (Logistic Regression) or \"balanced_subsample\" (Random Forest), computed from training labels only — validation and test class distributions are never altered.",
    "Threshold selection: both operating thresholds are chosen on the validation partition only, by grid search over predicted-probability quantiles — the “alert” threshold maximizes TSS, and a secondary “high-risk” threshold maximizes F1. Neither threshold is touched again when the test partition is scored.",
    "Model selection: the final configuration is chosen by maximum validation TSS (tie-break: validation PR-AUC) before the test partition is scored even once.",
    "Reproducibility: a fixed random seed (42) is used for all model fitting; package versions are recorded in Table 3.2 and the trained pipeline objects (imputer, scaler, classifier) are saved with joblib.",
]:
    body(doc, t, bullet=True, space_after=6)

doc.save("out/report_build.docx")
print("chapter 4 done")

# ============================================================ CHAPTER 5
heading1(doc, "Chapter 5 — Implementation")

heading2(doc, "5.1 Module Description")
body(doc, "The implementation is organized into modules so that data processing, modelling, evaluation, "
          "risk logic and presentation can be tested independently.")
mods = [
    ("Data Acquisition", "Downloads the five official SWAN-SF partition archives from Harvard Dataverse (DOI 10.7910/DVN/EBCFKM, v1.2) and verifies file sizes."),
    ("Feature Extraction", "Streams each partition tar.gz, parses the FL/NF folder label and window timestamps from each filename, and computes the 144-feature temporal representation per window (extract_features.py)."),
    ("Data Audit", "Merges all extracted chunks, computes per-partition counts, missingness, class balance and cross-partition active-region overlap (merge_audit.py)."),
    ("Model Training and Selection", "Trains all seven candidate configurations, selects the final model on validation TSS/PR-AUC, and evaluates it once on the locked test partition (train_eval.py)."),
    ("Result Figures", "Generates the confusion matrix, iteration-comparison, PR-curve, ROC-curve and coefficient-importance figures from the saved model and predictions (make_figures.py)."),
    ("Demo Data Builder", "Selects two genuine test-partition examples (a confirmed major-flare window and a confirmed quiet window) and packages the model's real output into a single JSON file consumed by the dashboard (build_demo_data.py)."),
    ("Web Dashboard", "A static HTML/CSS/JavaScript single-page application that reads the JSON output above and renders the prediction, risk level, alert message, explainability view and evaluation metrics; no dashboard value is hand-typed."),
]
make_table(
    doc, ["Module", "Description"], mods,
    col_widths=[Inches(1.8), Inches(4.5)], font_size=10,
)
body(doc, "Note on framework choice: an earlier planning draft of this report proposed Streamlit for the "
          "dashboard. The team instead implemented the dashboard as a static HTML/CSS/JavaScript page that "
          "reads a JSON file written by the trained pipeline, because it requires no server process to "
          "demonstrate and made it straightforward to capture consistent, reproducible screenshots "
          "(Playwright, headless Chromium) for this report.")

heading2(doc, "5.2 Key Code Snippets")
body(doc, "The four snippets below are taken verbatim (formatting only) from the modules actually run to "
          "produce the results in this report.")
code_block(doc, '''def features(df):
    v = df[SHARP].to_numpy(dtype=np.float64)          # (T, 24) inside the window only
    t = np.arange(v.shape[0], dtype=np.float64) * 0.2  # hours (12-min cadence)
    out = np.full((len(SHARP), len(STATS)), np.nan)
    with np.errstate(all="ignore"):
        for j in range(v.shape[1]):
            col = v[:, j]
            ok = np.isfinite(col)
            if ok.sum() == 0:
                continue
            c = col[ok]; tt = t[ok]
            out[j, 0] = c[-1]
            out[j, 1] = c.mean(); out[j, 2] = c.std()
            out[j, 3] = c.min();  out[j, 4] = c.max()
            if ok.sum() >= 2 and np.ptp(tt) > 0:
                out[j, 5] = np.polyfit(tt, c, 1)[0]   # trend per hour
    nan_frac = float(np.mean(~np.isfinite(v)))
    return out.ravel(), nan_frac''', caption="Snippet 1 — Leakage-safe feature construction (extract_features.py)")

code_block(doc, '''def lr_pipe(C, log=True):
    steps = [("slog", FunctionTransformer(slog))] if log else []
    steps += [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
              ("clf", LogisticRegression(C=C, class_weight="balanced",
                                         max_iter=5000, random_state=SEED))]
    return Pipeline(steps)''', caption="Snippet 2 — Logistic Regression pipeline (train_eval.py)")

code_block(doc, '''def rf_pipe(leaf, depth, n=300):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(n_estimators=n, min_samples_leaf=leaf,
                                                    max_depth=depth, max_features="sqrt",
                                                    class_weight="balanced_subsample",
                                                    n_jobs=-1, random_state=SEED))])''',
           caption="Snippet 3 — Random Forest pipeline (train_eval.py)")

code_block(doc, '''// risk view (dashboard/index.html) — p and thr/hi come from the trained model's
// real output (demo.json); no value below is hand-typed
let lvl = p < thr ? \'low\' : (p < hi ? \'mod\' : \'high\');
document.getElementById(\'rk-\' + lvl).classList.add(\'on\');
const bodies = {
  low:  \'The model does not indicate elevated major-flare probability for this window.\',
  mod:  \'The model indicates a moderately elevated probability of an M/X-class flare \'
      + \'in the next 24 hours. Continued monitoring of this active region is advised.\',
  high: \'The model indicates a high probability (\' + (p*100).toFixed(1) + \'%) of an \'
      + \'M/X-class flare in the next 24 hours for active region \' + ex.flare_case.ar + \'.\'
};''', caption="Snippet 4 — Risk-level derivation and alert text (dashboard/index.html)")

heading2(doc, "5.3 User Interface / Demo")
body(doc, "The final deliverable is a web dashboard that serves the selected model's real output. It shows "
          "the forecast configuration, the predicted probability for a genuine test-partition example, the "
          "derived risk level and alert text, a feature-importance (explainability) view built from the "
          "model's own coefficients, and the locked test-set evaluation metrics. NOAA GOES current "
          "monitoring is presented as separate contextual information and is never used as a model input.")
make_table(
    doc,
    ["Dashboard Component", "Purpose"],
    [
        ["Overview", "Summarizes the model, dataset and forecasting configuration"],
        ["Forecast configuration", "Shows the observation window and 24-hour prediction window for the displayed example"],
        ["Prediction result", "Displays the model's predicted probability of an M/X-class flare via a gauge"],
        ["Risk level & alert", "Converts the probability into a LOW / MODERATE / HIGH risk badge and alert message using the validation-selected thresholds"],
        ["Explainability", "Shows the top Logistic Regression coefficients driving the prediction"],
        ["Model evaluation", "Shows the locked test-partition metrics (Precision, Recall, F1, TSS, PR-AUC, ROC-AUC) and confusion matrix"],
    ],
    col_widths=[Inches(2.0), Inches(4.3)],
    caption="Web Dashboard Components", cap_num="5.1",
)
body(doc, "Figures 5.1–5.6 (Chapter list of figures) are Playwright screenshots of the running dashboard, "
          "each captured from the same trained model output shown above; none is a mockup.")
for path, num, cap in [
    (f"{FIG}/fig5_1_overview.png", "5.1", "Web dashboard overview of the Solar Flare Prediction and Space Weather Alert System."),
    (f"{FIG}/fig5_2_forecast_config.png", "5.2", "Forecast configuration showing the observation window and prediction window of a real test-partition example."),
    (f"{FIG}/fig5_3_prediction_result.png", "5.3", "Model prediction showing the estimated probability of a future M/X-class solar flare."),
    (f"{FIG}/fig5_4_risk_alert.png", "5.4", "User-facing risk level and alert message derived from the model probability."),
    (f"{FIG}/fig5_5_explainability.png", "5.5", "Model explainability view: standardized Logistic Regression coefficients."),
    (f"{FIG}/fig5_6_model_evaluation.png", "5.6", "Model evaluation summary view of the dashboard (test-partition metrics)."),
]:
    add_figure(doc, path, num, cap)

doc.save("out/report_build.docx")
print("chapter 5 done")

# ============================================================ CHAPTER 6
heading1(doc, "Chapter 6 — Results and Discussion")

heading2(doc, "6.1 Evaluation Metrics")
body_runs(doc, [
    ("Because major solar flares are rare (990 of 75,365 test windows, 1.31%), evaluation emphasizes "
     "minority-class performance rather than raw accuracy. ", {}),
    ("Precision", {"bold": True}), (" is the fraction of alert-triggering predictions that were correct; ", {}),
    ("Recall", {"bold": True}), (" is the fraction of true major-flare windows that were detected; ", {}),
    ("F1-score", {"bold": True}), (" balances precision and recall; ", {}),
    ("PR-AUC", {"bold": True}), (" summarizes precision-recall performance across all thresholds; ", {}),
    ("ROC-AUC", {"bold": True}), (" summarizes ranking quality independent of class balance; and ", {}),
    ("TSS", {"bold": True}), (" (True Skill Statistic, recall minus false-positive rate) is the standard "
     "domain-relevant solar-flare verification metric ", {}), ("[2], [3]", {"bold": True}),
    (". HSS2 (Heidke Skill Score) and FAR (false-alarm ratio) are reported alongside TSS for completeness. "
     "Accuracy is reported only as a secondary measure, since a model that always predicts “no flare” "
     "would already score 98.7% accuracy on this test partition while detecting zero major flares.", {}),
])

heading2(doc, "6.2 Results Across Iterations")
body(doc, "The table and figures below report results across all iterations, not only the final model, "
          "because the improvement process is itself part of the PBL evidence. All figures are on the "
          "locked test partition (P5), evaluated once per configuration at its validation-selected "
          "threshold.")
make_table(
    doc,
    ["Configuration", "Precision", "Recall", "F1", "TSS", "PR-AUC", "ROC-AUC", "Accuracy"],
    [
        ["Iteration 1 — LR, last-value (I1_LR_last)", "0.126", "0.955", "0.223", "0.867", "0.462", "0.980", "0.913"],
        ["Iteration 2 — RF, leaf=50/depth=16 (best RF PR-AUC)", "0.123", "0.957", "0.217", "0.865", "0.399", "0.980", "0.909"],
        ["Final — LR, temporal, C=0.01 (selected)", "0.162", "0.916", "0.275", "0.853", "0.489", "0.979", "0.936"],
    ],
    col_widths=[Inches(2.6), Inches(0.65), Inches(0.6), Inches(0.55), Inches(0.55), Inches(0.6), Inches(0.65), Inches(0.6)],
    font_size=8.5,
    caption="Results Across Iterations (Locked Test Partition)", cap_num="6.1",
)
body(doc, "The full comparison across all seven candidates, on both validation and test, is reported in "
          "Table A.1 (Appendix). The final selected configuration is not the highest-TSS configuration on "
          "the test partition — several Random Forest configurations reach marginally higher test TSS "
          "(0.860–0.865 vs. 0.853) — because model selection is fixed on validation evidence only "
          "(Section 4.6); the test partition is never used for selection.")

for path, num, cap in [
    (f"{FIG}/fig6_1_confusion_matrix.png", "6.1", "Confusion matrix of the final model on the locked test partition (P5)."),
    (f"{FIG}/fig6_2_iteration_comparison.png", "6.2", "Model performance comparison across Iteration 1, Iteration 2 and the final approach."),
    (f"{FIG}/fig6_3_pr_curve.png", "6.3", "Precision-Recall curve of the final model on the test partition."),
    (f"{FIG}/fig6_4_roc_curve.png", "6.4", "ROC curve of the final model on the test partition."),
    (f"{FIG}/fig6_5_lr_coefficients.png", "6.5", "Top 15 standardized Logistic Regression coefficients of the final model."),
]:
    add_figure(doc, path, num, cap)

heading2(doc, "6.3 Discussion")
body(doc, "At the validation-selected alert threshold (0.5445), the final model flags 5,614 of 75,365 test "
          "windows (7.45%) against a true positive rate of 1.31%, an alarm rate roughly 5.7× the base "
          "rate. This is the expected shape of a recall-oriented, class-weighted classifier on a rare-event "
          "problem: it correctly identifies 907 of 990 major-flare windows (recall = 0.916) at the cost of "
          "4,707 false alarms (precision = 0.162). Because a missed major flare is generally more costly "
          "than an extra false alarm in a space-weather warning context, this trade-off is a defensible "
          "design choice given the validation-fixed threshold, not an artifact of the model.")
body_runs(doc, [
    ("Random Forest did not outperform Logistic Regression on the validation selection criterion. Its "
     "test-set TSS is marginally higher for two of the three configurations (0.860–0.865 vs. 0.853), "
     "but its PR-AUC is substantially lower at both validation (0.233–0.471 vs. 0.500) and test "
     "(0.352–0.399 vs. 0.489). TSS depends only on the single chosen threshold, while PR-AUC "
     "summarizes ranking quality across all thresholds; the gap indicates that Random Forest's "
     "probability outputs are less well-calibrated across the full range in this rare-event, "
     "high-dimensional setting, even though a specific threshold can be found where its recall/FPR "
     "trade-off is competitive. This is reported as an observed, reproducible result rather than "
     "assumed in advance, and is consistent with the literature's caution that class imbalance and "
     "temporal coherence can make naive model comparisons misleading ", {}), ("[2], [3]", {"bold": True}), (".", {}),
], align=WD_ALIGN_PARAGRAPH.JUSTIFY)
body(doc, "The explainability view (Figure 6.5) shows R_VALUE__std, TOTUSJH (max and last) and SHRGT45 "
          "among the largest-magnitude standardized coefficients, consistent with these SHARP parameters' "
          "established association with active-region magnetic complexity in the solar-flare forecasting "
          "literature (Chapter 2); this is offered as a plausibility check on the model, not as a causal "
          "claim.")

heading2(doc, "6.4 Limitations")
for t in [
    "Dataset imbalance: major (M/X) flare windows are 1.31% of the locked test partition, so minority-class metrics (TSS, PR-AUC, recall) are essential and accuracy alone would be misleading.",
    "Temporal dependence: consecutive windows from the same active region overlap heavily (97.9% of consecutive same-AR windows are only one hour apart), which is why this project uses a partition-based split rather than a random split; even so, some residual within-partition correlation between nearby windows is possible.",
    "Missing/quality issues: 10.4% of observation windows contain at least one missing SHARP value (mean per-column NaN fraction 0.60%), handled by median imputation fitted on training data only; a small fraction of windows also carry non-zero SWAN-SF quality flags that were not used to filter data in this prototype.",
    "Feature scope: this PBL uses a fixed-length, 144-dimensional temporal-statistic representation rather than full-resolution magnetogram images or an end-to-end deep sequence model (Chapter 2, Section 2.1.4).",
    "Deployment scope: the dashboard is an academic prototype for demonstrating the trained pipeline, not an operational space-weather warning service, and is explicitly labelled as such in its interface.",
    "Generalisation: the model is trained and evaluated only on SWAN-SF partitions spanning 2010-05-01 to 2018-08-17; performance on other instruments, later solar cycles or real-time operational conditions is not claimed.",
]:
    body(doc, t, bullet=True, space_after=6)

doc.save("out/report_build.docx")
print("chapter 6 done")

# ============================================================ CHAPTER 7
heading1(doc, "Chapter 7 — Team Reflection and Learning Outcomes")

heading2(doc, "7.1 Individual Reflections")
body(doc, "Each reflection is based on the actual work performed and should state the member's "
          "contribution, one concrete learning outcome and one genuine challenge, in the member's own "
          "words.")
body(doc, f"Madhavan G — {PEND.replace('ACTUAL INFORMATION REQUIRED', 'FINAL REFLECTION TO BE WRITTEN BY STUDENT')}")
body(doc, f"Sriram Sivakumar — {PEND.replace('ACTUAL INFORMATION REQUIRED', 'FINAL REFLECTION TO BE WRITTEN BY STUDENT')}")

heading2(doc, "7.2 Team Learning")
body(doc, "This section should document, in the team's own words, how responsibilities were actually "
          "divided (see Team Roles and Responsibilities, front matter), how the choice between Logistic Regression "
          "and Random Forest changed after observing real validation evidence (Section 4.6), how mentor "
          "feedback shaped the project, and what the team would do differently if the PBL cycle were "
          "restarted. " + PEND)

heading2(doc, "7.3 Course Outcomes — Evidence Summary")
body(doc, "Official Course Outcome statements are intentionally left blank, consistent with the project "
          "rule that no CO wording is invented in this report. Once the official Machine Learning CO1, "
          "CO2, … wording is supplied by the course documentation, each outcome should be mapped to "
          "concrete evidence already produced by this project (for example: leakage-safe feature "
          "engineering and partition-based validation as evidence of applying ML methodology; the honest "
          "Logistic-Regression-vs-Random-Forest comparison in Section 6.3 as evidence of critical "
          "evaluation; the dashboard as evidence of translating a model into a usable system).")
make_table(
    doc,
    ["Course Outcome", "Evidence From This Project"],
    [
        ["CO1 — " + PEND, PEND],
        ["CO2 — " + PEND, PEND],
        ["Additional COs — " + PEND, PEND],
    ],
    col_widths=[Inches(3.0), Inches(3.3)],
    font_size=9.5,
)

doc.save("out/report_build.docx")
print("chapter 7 done")

# ============================================================ CHAPTER 8
heading1(doc, "Chapter 8 — Conclusion and Future Scope")

heading2(doc, "8.1 Conclusion")
body(doc, "This project implemented and evaluated a leakage-safe Machine Learning pipeline for major "
          "solar-flare forecasting on the SWAN-SF benchmark. Working from 331,185 twelve-hour observation "
          "windows across five chronological partitions, the team constructed a 144-feature temporal "
          "representation with all flare-derived and GOES columns explicitly excluded, compared seven "
          "candidate Logistic Regression and Random Forest configurations under a partition-based "
          "train/validation/test protocol, and selected the final configuration on validation evidence "
          "before evaluating it exactly once on a locked test partition. The selected model — Logistic "
          "Regression on the temporal feature representation, C=0.01 — achieved a test-set TSS of 0.853, "
          "recall of 0.916 and PR-AUC of 0.489, correctly identifying 907 of 990 major-flare windows. "
          "Random Forest did not outperform Logistic Regression on the validation selection criterion in "
          "this study, a result that was measured rather than assumed. These outcomes directly answer the "
          "driving question posed in Chapter 1: pre-cutoff SHARP observations, summarized into a "
          "leakage-safe temporal feature vector, can be used to estimate major-flare likelihood with "
          "substantial skill (TSS ≈ 0.85) under a partition-based evaluation, with Logistic Regression "
          "providing the better-calibrated ranking of the two model families tested. The trained model's "
          "output is served through a working web dashboard as a demonstration, not an operational "
          "warning system.")

heading2(doc, "8.2 Future Scope")
for t in [
    "Expand validation to additional SWAN-SF partitions or later solar-cycle data as they become available, to test robustness beyond the 2010–2018 window used here.",
    "Investigate additional scientifically justified SHARP or derived predictors, verified to be available strictly before the forecast cutoff, before adding them to the feature set.",
    "Investigate stronger temporal or deep-learning methods (e.g. LSTM-based sequence models, as surveyed in Chapter 2) as a later research extension, while preserving the partition-based, leakage-safe evaluation protocol used here.",
    "Build a fully causal, real-time feature pipeline that reproduces the same 144-feature representation from live SHARP data before any live or near-real-time prediction is claimed.",
    "Extend the dashboard with alert history, auditability and calibration diagnostics if the prototype is developed further toward a deployment-oriented study.",
]:
    body(doc, t, bullet=True, space_after=6)

doc.save("out/report_build.docx")
print("chapter 8 done")

# ============================================================ REFERENCES
heading1(doc, "References")
REFS = [
"S. M. Hamdi, D. Kempton, R. Ma, S. F. Boubrahimi, and R. A. Angryk, “A time series classification-based approach for solar flare prediction,” in Proc. IEEE Int. Conf. Big Data (Big Data), Boston, MA, USA, 2017, pp. 2543–2551, doi: 10.1109/BigData.2017.8258213.",
"A. Ahmadzadeh, M. Hostetter, B. Aydin, M. K. Georgoulis, D. J. Kempton, S. S. Mahajan, and R. A. Angryk, “Challenges with extreme class-imbalance and temporal coherence: A study on solar flare data,” in Proc. IEEE Int. Conf. Big Data (Big Data), Los Angeles, CA, USA, 2019, pp. 1423–1431, doi: 10.1109/BigData47090.2019.9006505.",
"A. Ahmadzadeh, B. Aydin, M. K. Georgoulis, D. J. Kempton, S. S. Mahajan, and R. A. Angryk, “How to train your flare prediction model: Revisiting robust sampling of rare events,” Astrophys. J. Suppl. Ser., vol. 254, no. 2, Art. no. 23, 2021, doi: 10.3847/1538-4365/abec88.",
"A. Yeolekar, S. Patel, S. Talla, K. R. Puthucode, A. Ahmadzadeh, V. M. Sadykov, and R. A. Angryk, “Feature selection on a flare forecasting testbed: A comparative study of 24 methods,” in Proc. IEEE Int. Conf. Big Data, 2021.",
"Y. Velanki, P. Hosseinzadeh, S. F. Boubrahimi, and S. M. Hamdi, “Time-series feature selection for solar flare forecasting,” Universe, vol. 10, no. 9, Art. no. 373, 2024, doi: 10.3390/universe10090373.",
"M. Eskandarinasab, S. M. Hamdi, and S. F. Boubrahimi, “Impacts of data preprocessing and sampling techniques on solar flare prediction from multivariate time series data of photospheric magnetic field parameters,” Astrophys. J. Suppl. Ser., vol. 275, no. 1, Art. no. 6, 2024, doi: 10.3847/1538-4365/ad7c4a.",
"A. A. M. Muzaheed, S. M. Hamdi, and S. F. Boubrahimi, “Sequence model-based end-to-end solar flare classification from multivariate time series data,” in Proc. 20th IEEE Int. Conf. Machine Learning and Applications (ICMLA), Pasadena, CA, USA, 2021, pp. 435–440, doi: 10.1109/ICMLA52953.2021.00074.",
"A. Azizian Foumani, S. Farokhi, and X. Qi, “Predicting major solar flares using convolutional neural networks and multivariate magnetic field time-series data,” Solar Physics, vol. 301, Art. no. 23, 2026, doi: 10.1007/s11207-026-02612-6.",
"O. Vural, S. M. Hamdi, and S. F. Boubrahimi, “Contrastive representation learning for predicting solar flares from extremely imbalanced multivariate time series data,” arXiv preprint arXiv:2410.00312, 2024, doi: 10.48550/arXiv.2410.00312.",
"R. A. Angryk et al., “Multivariate time series dataset for space weather data analytics,” Scientific Data, vol. 7, Art. no. 227, 2020, doi: 10.1038/s41597-020-0548-x.",
"R. Angryk et al., “SWAN-SF,” Harvard Dataverse, version 1, 2020, doi: 10.7910/DVN/EBCFKM.",
"NASA, “Space-weather HMI Active Region Patch (SHARP),” NASA Open Data Portal. [Online]. Available: https://data.nasa.gov/dataset/space-weather-hmi-active-region-patch-sharp. Accessed: Sep. 21, 2026.",
"NOAA Space Weather Prediction Center, “GOES X-ray and space-weather products.” [Online]. Available: https://services.swpc.noaa.gov/products/. Accessed: Sep. 21, 2026.",
"scikit-learn developers, “LogisticRegression,” scikit-learn documentation. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html. Accessed: Sep. 21, 2026.",
"scikit-learn developers, “RandomForestClassifier,” scikit-learn documentation. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html. Accessed: Sep. 21, 2026.",
]
for i, r in enumerate(REFS, start=1):
    body_runs(doc, [(f"[{i}] ", {"bold": True}), (r, {})], space_after=6)

doc.save("out/report_build.docx")
print("references done")

# ============================================================ APPENDIX
heading1(doc, "Appendix")

heading2(doc, "A.1 Source Code")
body(doc, "A.1 — Full source code: " + PEND.replace("ACTUAL INFORMATION REQUIRED", "REPOSITORY LINK TO BE INSERTED — no code repository was created for this submission"))

heading2(doc, "A.2 Complete Weekly PBL Log")
body(doc, "A.2 — " + PEND.replace("ACTUAL INFORMATION REQUIRED", "TO BE COMPLETED FROM THE ACTUAL PBL RECORD AND MENTOR SIGN-OFFS") + " (see the planning skeleton in Table 3.1).")

heading2(doc, "A.3 Self and Peer Assessment")
body(doc, "The self and peer assessment must be completed from the actual contribution record; no "
          "contribution percentages are invented in this report.")
make_table(
    doc,
    ["Team Member", "Self-Rated Contribution (%)", "Peer-Rated Contribution (%)", "Remarks"],
    [
        ["Madhavan G", PEND, PEND, PEND],
        ["Sriram Sivakumar", PEND, PEND, PEND],
    ],
    col_widths=[Inches(1.6), Inches(1.6), Inches(1.6), Inches(1.5)],
    font_size=9,
    caption="Self and Peer Assessment", cap_num="A.3",
)

heading2(doc, "A.4 Complete Model Comparison — All Candidates")
body(doc, "Full validation and test results for all seven candidate configurations evaluated in Chapter 4, "
          "supporting the model-selection claims in Sections 4.6 and 6.2.")
rows = []
NAMES = {"I1_LR_last": "I1_LR_last", "I2_LR_temporal_C0.01": "I2_LR_temporal_C0.01",
         "I2_LR_temporal_C0.1": "I2_LR_temporal_C0.1", "I2_LR_temporal_C1": "I2_LR_temporal_C1",
         "I2_RF_leaf5": "I2_RF_leaf5", "I2_RF_leaf20": "I2_RF_leaf20", "I2_RF_leaf50_d16": "I2_RF_leaf50_d16"}
for k in NAMES:
    v, t = VAL[k]["val"], TEST[k]
    rows.append([k, f"{v['tss']:.3f}", f"{v['pr_auc']:.3f}", f"{t['tss']:.3f}", f"{t['pr_auc']:.3f}",
                 f"{t['precision']:.3f}", f"{t['recall']:.3f}", f"{t['f1']:.3f}", f"{t['roc_auc']:.3f}"])
make_table(
    doc,
    ["Configuration", "Val TSS", "Val PR-AUC", "Test TSS", "Test PR-AUC", "Test Prec.", "Test Rec.", "Test F1", "Test ROC-AUC"],
    rows,
    col_widths=[Inches(1.7), Inches(0.55), Inches(0.65), Inches(0.55), Inches(0.65), Inches(0.55), Inches(0.55), Inches(0.5), Inches(0.7)],
    font_size=8,
    caption="Complete Model Comparison — All Candidates (Validation and Test)", cap_num="A.1",
)
body(doc, "Selection rule (frozen before test evaluation): maximum validation TSS, tie-break by validation "
          "PR-AUC. Selected: I2_LR_temporal_C0.01. Alert threshold = 0.5445, high-risk threshold = 0.9673 "
          "(both chosen on validation only, per Section 4.7).")

doc.save("out/report_build.docx")
print("appendix done")
