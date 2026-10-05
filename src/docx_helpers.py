"""Shared helpers for assembling the PBL report with python-docx."""
import re
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.section import WD_SECTION

BLACK = RGBColor(0, 0, 0)
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
CONTENT_W = Inches(6.3)

_bm_counter = [100]


def get_style(doc, name):
    for s in doc.styles:
        if s.name.lower() == name.lower():
            return s
    raise KeyError(name)


def set_run(run, size=12, bold=False, italic=False, color=BLACK, name="Times New Roman", caps=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts'); rPr.append(rFonts)
    rFonts.set(qn('w:ascii'), name); rFonts.set(qn('w:hAnsi'), name); rFonts.set(qn('w:cs'), name)
    if caps:
        run.font.all_caps = True
    return run


def para(doc, text="", size=12, bold=False, italic=False, align=None, space_before=0, space_after=6,
         color=BLACK, line_spacing=1.15, style=None):
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    if text:
        set_run(p.add_run(text), size=size, bold=bold, italic=italic, color=color)
    return p


def add_bookmark(paragraph, name=None):
    if name is None:
        _bm_counter[0] += 1
        name = f"bm{_bm_counter[0]}"
    bm_id = str(_bm_counter[0])
    start = OxmlElement('w:bookmarkStart'); start.set(qn('w:id'), bm_id); start.set(qn('w:name'), name)
    end = OxmlElement('w:bookmarkEnd'); end.set(qn('w:id'), bm_id)
    paragraph._p.insert(0, start)
    paragraph._p.append(end)
    return name


def pageref_field(paragraph, bookmark_name):
    run = paragraph.add_run()
    set_run(run, size=11)
    fld_begin = OxmlElement('w:fldChar'); fld_begin.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText'); instr.set(qn('xml:space'), 'preserve')
    instr.text = f' PAGEREF {bookmark_name} \\h '
    fld_sep = OxmlElement('w:fldChar'); fld_sep.set(qn('w:fldCharType'), 'separate')
    fld_txt = OxmlElement('w:t'); fld_txt.text = '1'
    fld_end = OxmlElement('w:fldChar'); fld_end.set(qn('w:fldCharType'), 'end')
    run._r.append(fld_begin); run._r.append(instr); run._r.append(fld_sep); run._r.append(fld_txt); run._r.append(fld_end)


def heading1(doc, text, add_page_break_before=True):
    if add_page_break_before:
        doc.add_page_break()
    p = doc.add_paragraph(style=get_style(doc, "Heading 1"))
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(18)
    set_run(p.add_run(text), size=17, bold=True, color=NAVY, caps=True)
    add_bookmark(p)
    return p


def heading2(doc, text):
    p = doc.add_paragraph(style=get_style(doc, "Heading 2"))
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(8)
    set_run(p.add_run(text), size=13.5, bold=True, color=NAVY)
    return p


def heading3(doc, text):
    p = doc.add_paragraph(style=get_style(doc, "Heading 3"))
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    set_run(p.add_run(text), size=12.5, bold=True, italic=True, color=BLACK)
    return p


def body(doc, text, size=12, align=None, space_after=8, bullet=False, italic=False, bold=False):
    p = doc.add_paragraph()
    if bullet:
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.18)
        text = "•   " + text
    if align is not None:
        p.alignment = align
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.28
    set_run(p.add_run(text), size=size, italic=italic, bold=bold)
    return p


def body_runs(doc, parts, size=12, align=None, space_after=8):
    """parts: list of (text, kwargs) tuples for mixed formatting (e.g. citations superscript-like bold)."""
    p = doc.add_paragraph()
    p.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.28
    for text, kw in parts:
        set_run(p.add_run(text), size=kw.get("size", size), bold=kw.get("bold", False),
                italic=kw.get("italic", False), color=kw.get("color", BLACK))
    return p


def set_cell_text(cell, text, size=10.5, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT, color=BLACK, italic=False):
    cell.text = ""
    p = cell.paragraphs[0]; p.alignment = align
    p.paragraph_format.space_after = Pt(2)
    set_run(p.add_run(text), size=size, bold=bold, color=color, italic=italic)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def shade_cell(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd'); shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)


def make_table(doc, headers, rows, col_widths=None, header_fill="1F3A5F", font_size=10, caption=None, cap_num=None):
    if caption:
        add_caption(doc, "Table", cap_num, caption)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = doc.styles['Table Grid']
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False

    def _no_split(row):
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement('w:cantSplit')
        trPr.append(cs)

    _no_split(t.rows[0])
    for i, h in enumerate(headers):
        set_cell_text(t.rows[0].cells[i], h, size=font_size, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF),
                      align=WD_ALIGN_PARAGRAPH.CENTER)
        shade_cell(t.rows[0].cells[i], header_fill)
    for r_i, row in enumerate(rows):
        tr = t.add_row()
        _no_split(tr)
        cells = tr.cells
        for i, val in enumerate(row):
            set_cell_text(cells[i], str(val), size=font_size, align=WD_ALIGN_PARAGRAPH.CENTER if i > 0 else WD_ALIGN_PARAGRAPH.LEFT)
        if r_i % 2 == 1:
            for c in cells:
                shade_cell(c, "F2F5FA")
    if col_widths:
        for row in t.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = w
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


_fig_counter = {}
_tab_counter = {}


def add_caption(doc, kind, number, text, bookmark=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(12)
    r = p.add_run(f"{kind} {number} — {text}")
    set_run(r, size=10.5, bold=True, italic=True, color=NAVY)
    r._r.rPr.rFonts.set(qn('w:ascii'), "Times New Roman")
    bm = bookmark or f"{kind}{number}".replace(".", "_").replace(" ", "")
    add_bookmark(p, bm)
    return bm


def code_block(doc, code, caption=None, size=8.5):
    """A shaded, monospace, single-cell table for a short genuine code excerpt."""
    if caption:
        p = doc.add_paragraph()
        r = p.add_run(caption)
        set_run(r, size=10.5, bold=True, italic=True, color=NAVY)
        p.paragraph_format.space_before = Pt(8); p.paragraph_format.space_after = Pt(2)
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    cell = t.rows[0].cells[0]
    cell.width = CONTENT_W
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd'); shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), 'F5F6F8')
    tcPr.append(shd)
    borders = OxmlElement('w:tcBorders')
    for edge in ('top', 'left', 'bottom', 'right'):
        e = OxmlElement(f'w:{edge}'); e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), '4'); e.set(qn('w:color'), 'C9CDD3')
        borders.append(e)
    tcPr.append(borders)
    cell.text = ""
    lines = code.strip("\n").split("\n")
    for i, ln in enumerate(lines):
        p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        set_run(p.add_run(ln if ln else " "), size=size, name="Consolas", color=RGBColor(0x1A, 0x1A, 0x1A))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def add_figure(doc, path, number, caption, width=CONTENT_W):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    run = p.add_run()
    run.add_picture(path, width=width)
    return add_caption(doc, "Figure", number, caption)


def field_toc(doc):
    p = doc.add_paragraph()
    fld_begin = OxmlElement('w:fldChar'); fld_begin.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText'); instr.set(qn('xml:space'), 'preserve')
    instr.text = ' TOC \\o "1-3" \\h \\z \\u '
    fld_sep = OxmlElement('w:fldChar'); fld_sep.set(qn('w:fldCharType'), 'separate')
    txt = OxmlElement('w:t'); txt.text = "Right-click and choose “Update Field” (or press F9) to generate the Table of Contents."
    fld_end = OxmlElement('w:fldChar'); fld_end.set(qn('w:fldCharType'), 'end')
    r = p.add_run(); set_run(r, size=11, italic=True)
    r._r.append(fld_begin); r._r.append(instr); r._r.append(fld_sep); r._r.append(txt); r._r.append(fld_end)
    return p


def set_update_fields_on_open(doc):
    settings = doc.settings.element
    uf = OxmlElement('w:updateFields'); uf.set(qn('w:val'), 'true')
    settings.append(uf)


def lof_lot_entry(doc, label_text, bookmark_name):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    tab_stops = p.paragraph_format.tab_stops
    tab_stops.add_tab_stop(CONTENT_W, alignment=2, leader=3)  # right align, dotted leader
    r = p.add_run(label_text); set_run(r, size=11)
    r2 = p.add_run("\t"); set_run(r2, size=11)
    pageref_field(p, bookmark_name)
    return p
