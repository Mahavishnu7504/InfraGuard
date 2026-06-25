from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether,
)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus.flowables import HRFlowable
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics import renderPDF
from reportlab.platypus import flowables

from datetime import datetime
from functools import lru_cache
import os

# =========================================================
# COLOUR PALETTE
# =========================================================
BLUE_ACCENT  = colors.HexColor("#2563EB")
BLUE_LIGHT   = colors.HexColor("#EFF6FF")
INK          = colors.HexColor("#111827")
INK_LIGHT    = colors.HexColor("#374151")
BORDER       = colors.HexColor("#D1D5DB")
LABEL_BG     = colors.HexColor("#F3F4F6")
WHITE        = colors.white

SEV_CRITICAL = colors.HexColor("#B91C1C")
SEV_HIGH     = colors.HexColor("#C2410C")
SEV_MEDIUM   = colors.HexColor("#92400E")
SEV_LOW      = colors.HexColor("#166534")

CARD_BG      = colors.HexColor("#F8FAFC")
GOLD         = colors.HexColor("#D97706")
TEAL         = colors.HexColor("#0F766E")
PURPLE       = colors.HexColor("#7C3AED")

# KPI card colored backgrounds
CARD_GREEN   = colors.HexColor("#DCFCE7")
CARD_GREEN_B = colors.HexColor("#166534")
CARD_ORANGE  = colors.HexColor("#FFEDD5")
CARD_ORANGE_B= colors.HexColor("#C2410C")
CARD_BLUE    = colors.HexColor("#DBEAFE")
CARD_BLUE_B  = colors.HexColor("#1D4ED8")
CARD_PURPLE  = colors.HexColor("#EDE9FE")
CARD_PURPLE_B= colors.HexColor("#6D28D9")
CARD_RED     = colors.HexColor("#FEE2E2")
CARD_RED_B   = colors.HexColor("#B91C1C")
CARD_SLATE   = colors.HexColor("#F1F5F9")
CARD_SLATE_B = colors.HexColor("#475569")

# =========================================================
# PAGE GEOMETRY
# =========================================================
PAGE_W, PAGE_H = A4
MARGIN         = 50
CONTENT_W      = PAGE_W - 2 * MARGIN


# =========================================================
# TYPOGRAPHY
# =========================================================
@lru_cache(maxsize=1)
def _styles():
    base = getSampleStyleSheet()

    def S(name, **kw):
        parent = kw.pop("parent", base["Normal"])
        return ParagraphStyle(name, parent=parent, **kw)

    return {
        "header_tag": S(
            "hdr_tag",
            fontSize=9, fontName="Helvetica-Bold",
            textColor=WHITE, alignment=TA_LEFT,
        ),
        "report_title": S(
            "rpt_title",
            fontSize=22, fontName="Helvetica-Bold",
            textColor=INK, alignment=TA_CENTER,
            spaceBefore=14, spaceAfter=8, leading=28,
        ),
        "report_subtitle": S(
            "rpt_sub",
            fontSize=10, fontName="Helvetica",
            textColor=INK_LIGHT, alignment=TA_LEFT,
            spaceAfter=10, leading=14,
        ),
        "section_heading": S(
            "sec_hdr",
            fontSize=13, fontName="Helvetica-Bold",
            textColor=BLUE_ACCENT,
            spaceBefore=14, spaceAfter=6, leading=18,
        ),
        "finding_heading": S(
            "fnd_hdr",
            fontSize=10, fontName="Helvetica-Bold",
            textColor=INK,
            spaceBefore=10, spaceAfter=3, leading=14,
        ),
        "body": S(
            "body",
            fontSize=10, fontName="Helvetica",
            textColor=INK, leading=15, alignment=TA_LEFT,
        ),
        "body_justify": S(
            "body_j",
            fontSize=10, fontName="Helvetica",
            textColor=INK, leading=15, alignment=TA_JUSTIFY,
        ),
        "bullet": S(
            "bul",
            fontSize=10, fontName="Helvetica",
            textColor=INK, leading=15,
            leftIndent=16, spaceBefore=3,
        ),
        "caption": S(
            "cap",
            fontSize=9, fontName="Helvetica",
            textColor=INK_LIGHT, alignment=TA_LEFT,
            spaceBefore=4, spaceAfter=8, leading=13,
        ),
        "tbl_hdr": S(
            "tbl_hdr",
            fontSize=10, fontName="Helvetica-Bold",
            textColor=WHITE,
        ),
        "tbl_cell": S(
            "tbl_cel",
            fontSize=10, fontName="Helvetica",
            textColor=INK,
        ),
        "kv_label": S(
            "kv_lbl",
            fontSize=10, fontName="Helvetica-Bold",
            textColor=INK,
        ),
        "kv_value": S(
            "kv_val",
            fontSize=10, fontName="Helvetica",
            textColor=INK,
        ),
        "footer": S(
            "ftr",
            fontSize=8, fontName="Helvetica",
            textColor=INK_LIGHT, alignment=TA_CENTER,
        ),
        "card_title": S(
            "card_ttl",
            fontSize=8, fontName="Helvetica",
            textColor=INK_LIGHT, alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "card_value": S(
            "card_val",
            fontSize=14, fontName="Helvetica-Bold",
            textColor=INK, alignment=TA_CENTER,
        ),
        "badge": S(
            "badge",
            fontSize=9, fontName="Helvetica-Bold",
            textColor=WHITE, alignment=TA_CENTER,
        ),
        "subsection_heading": S(
            "sub_hdr",
            fontSize=11, fontName="Helvetica-Bold",
            textColor=INK,
            spaceBefore=8, spaceAfter=4, leading=15,
        ),
        "signoff_label": S(
            "so_lbl",
            fontSize=10, fontName="Helvetica-Bold",
            textColor=INK_LIGHT, alignment=TA_LEFT,
            spaceAfter=2,
        ),
        "signoff_line": S(
            "so_line",
            fontSize=10, fontName="Helvetica",
            textColor=INK, alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "chart_title": S(
            "ch_ttl",
            fontSize=10, fontName="Helvetica-Bold",
            textColor=INK, alignment=TA_CENTER,
            spaceBefore=6, spaceAfter=4,
        ),
        "revision_label": S(
            "rev_lbl",
            fontSize=8, fontName="Helvetica-Bold",
            textColor=INK_LIGHT, alignment=TA_LEFT,
        ),
        "revision_value": S(
            "rev_val",
            fontSize=8, fontName="Helvetica",
            textColor=INK_LIGHT, alignment=TA_LEFT,
        ),
    }


# =========================================================
# PRIMITIVES
# =========================================================
def _sp(h=8):
    return Spacer(1, h)


def _page_break(elements: list):
    elements.append(PageBreak())


def _sev_color(severity: str):
    return {
        "Critical": SEV_CRITICAL,
        "High":     SEV_HIGH,
        "Medium":   SEV_MEDIUM,
        "Low":      SEV_LOW,
    }.get(severity, INK_LIGHT)


def _compliance_color(text: str):
    t = text.lower()
    if "non-compliant" in t or "critical" in t:
        return SEV_CRITICAL
    if "conditional" in t:
        return SEV_MEDIUM
    if "compliant" in t:
        return SEV_LOW
    return INK_LIGHT


def _section_heading(num: int, title: str, ST: dict, elements: list):
    elements.append(Paragraph(f"{num}. {title}", ST["section_heading"]))


def _inline_img(path: str, ST: dict, elements: list, caption: str = "",
                target_width: float = None):
    if not path or not os.path.exists(path):
        return

    img_w = target_width or (CONTENT_W * 0.99)
    img_h = 3.3 * inch

    try:
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            iw, ih = im.size
            if iw > 0:
                ratio = ih / iw
                img_h = img_w * ratio
                max_h = 3.9 * inch
                if img_h > max_h:
                    img_h = max_h
                    img_w = img_h / ratio
    except Exception:
        pass

    elements.append(Image(path, width=img_w, height=img_h))
    if caption:
        elements.append(Paragraph(caption, ST["caption"]))
    elements.append(_sp(6))


def _severity_badge(severity: str, ST: dict):
    """Return a small colored badge Table for a severity level."""
    sev_upper = severity.upper()
    label_map = {
        "CRITICAL": "[ CRITICAL ]",
        "HIGH":     "[ HIGH ]",
        "MEDIUM":   "[ MEDIUM ]",
        "LOW":      "[ LOW ]",
    }
    color_map = {
        "CRITICAL": SEV_CRITICAL,
        "HIGH":     SEV_HIGH,
        "MEDIUM":   SEV_MEDIUM,
        "LOW":      SEV_LOW,
    }
    label = label_map.get(sev_upper, f"[ {sev_upper} ]")
    bg    = color_map.get(sev_upper, INK_LIGHT)

    badge = Table(
        [[Paragraph(label, ST["badge"])]],
        colWidths=[80],
    )
    badge.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("ROUNDEDCORNERS", (0, 0), (-1, -1), [3, 3, 3, 3]),
    ]))
    return badge


# =========================================================
# TITLE BLOCK  (enhanced — Phase 3)
# =========================================================
def _title_block(data: dict, ST: dict, elements: list):
    # ── Blue header bar ────────────────────────────────────
    report_id    = data.get("report_id", "INF-QA-2026-001")
    gen_date     = data.get("generated_date",
                            datetime.now().strftime("%d-%b-%Y"))
    insp_grade   = data.get("inspection_grade", "—")
    overall_risk = data.get("overall_risk", data.get("overall_status", "—"))
    exec_risk    = data.get("executive_risk_index", "—")
    audit_ready  = data.get("audit_readiness", "—")

    hdr = Table(
        [[Paragraph("INFRA GUARD AI PLATFORM", ST["header_tag"])]],
        colWidths=[CONTENT_W],
    )
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), BLUE_ACCENT),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    elements.append(hdr)
    elements.append(_sp(12))

    # ── Report title ───────────────────────────────────────
    elements.append(Paragraph(
        "Construction Quality Inspection Report",
        ST["report_title"],
    ))

    # ── Subtitle ───────────────────────────────────────────
    elements.append(Paragraph(
        "AI-assisted visual inspection summary for construction site quality "
        "assurance, housekeeping discipline, and material management review.",
        ST["report_subtitle"],
    ))
    elements.append(_sp(6))

    # ── 3-row summary table + new meta fields ──────────────
    status = data.get("overall_status", "—")
    score  = data.get("compliance_score", 0)
    total  = data.get("total_findings", 0)

    sc     = _compliance_color(status)
    sc_hex = sc.hexval()[2:]

    risk_hex = _sev_color(overall_risk).hexval()[2:]

    # Build executive risk index display
    exec_risk_str = (
        f"{exec_risk}/100" if isinstance(exec_risk, (int, float))
        else str(exec_risk)
    )

    summary_rows = [
        [Paragraph("<b>Report ID</b>",           ST["kv_label"]),
         Paragraph(report_id,                    ST["kv_value"])],
        [Paragraph("<b>Generated</b>",            ST["kv_label"]),
         Paragraph(gen_date,                     ST["kv_value"])],
        [Paragraph("<b>Overall Status</b>",       ST["kv_label"]),
         Paragraph(f'<font color="#{sc_hex}"><b>{status}</b></font>',
                   ST["kv_value"])],
        [Paragraph("<b>Compliance Score</b>",     ST["kv_label"]),
         Paragraph(f"<b>{score}/100</b>",         ST["kv_value"])],
        [Paragraph("<b>Total Findings</b>",       ST["kv_label"]),
         Paragraph(f"<b>{total}</b>",             ST["kv_value"])],
        [Paragraph("<b>Inspection Grade</b>",     ST["kv_label"]),
         Paragraph(f"<b>{insp_grade}</b>",        ST["kv_value"])],
        [Paragraph("<b>Overall Risk</b>",         ST["kv_label"]),
         Paragraph(
             f'<font color="#{risk_hex}"><b>{overall_risk}</b></font>',
             ST["kv_value"],
         )],
        [Paragraph("<b>Executive Risk Index</b>", ST["kv_label"]),
         Paragraph(f"<b>{exec_risk_str}</b>",     ST["kv_value"])],
        [Paragraph("<b>Audit Readiness</b>",      ST["kv_label"]),
         Paragraph(f"<b>{audit_ready}</b>",       ST["kv_value"])],
    ]

    col_l = CONTENT_W * 0.32
    col_r = CONTENT_W * 0.68
    summary_tbl = Table(summary_rows, colWidths=[col_l, col_r])
    summary_tbl.setStyle(TableStyle([
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("BACKGROUND",    (0, 0), (0, -1),  LABEL_BG),
        ("BACKGROUND",    (1, 0), (1, -1),  WHITE),
        ("TOPPADDING",    (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))

    pad_l = CONTENT_W * 0.08
    pad_r = CONTENT_W * 0.08
    wrapper = Table([[summary_tbl]], colWidths=[CONTENT_W])
    wrapper.setStyle(TableStyle([
        ("LEFTPADDING",   (0, 0), (-1, -1), pad_l),
        ("RIGHTPADDING",  (0, 0), (-1, -1), pad_r),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    elements.append(wrapper)
    elements.append(_sp(10))

    # ── Revision Block ─────────────────────────────────────
    revision    = data.get("revision", "1.0")
    doc_class   = data.get("document_classification",
                            "Internal Quality Assessment")
    generated_by = "InfraGuard Enterprise AI"

    rev_rows = [[
        Paragraph(f"<b>Revision:</b> {revision}", ST["revision_label"]),
        Paragraph(f"<b>Generated By:</b> {generated_by}", ST["revision_label"]),
        Paragraph(f"<b>Classification:</b> {doc_class}", ST["revision_label"]),
    ]]
    rev_tbl = Table(rev_rows, colWidths=[CONTENT_W * 0.22, CONTENT_W * 0.40, CONTENT_W * 0.38])
    rev_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#F0F4FF")),
        ("BOX",           (0, 0), (-1, -1), 0.5, BLUE_ACCENT),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(rev_tbl)
    elements.append(_sp(14))


# =========================================================
# COVER PAGE  (new — Priority 1)
# =========================================================
def _cover_page(data: dict, ST: dict, elements: list):
    """
    Large, centered title page rendered before the title block.
    Gives the report an enterprise-audit first impression
    (Project Name / Inspection ID / Date / Compliance Score /
    Grade / Generated By) ahead of the dense data pages.
    """
    project_name = data.get("project_name", "—")
    report_id    = data.get("report_id", "INF-QA-2026-001")
    gen_date     = data.get("generated_date",
                             datetime.now().strftime("%d-%b-%Y"))
    score        = data.get("compliance_score", 0)
    grade        = data.get("inspection_grade", "—")
    generated_by = "InfraGuard Enterprise AI"

    cover_brand = ParagraphStyle(
        "cover_brand", parent=ST["header_tag"],
        fontSize=13, textColor=BLUE_ACCENT, alignment=TA_CENTER,
        spaceAfter=4,
    )
    cover_title = ParagraphStyle(
        "cover_title", parent=ST["report_title"],
        fontSize=28, leading=34, spaceBefore=18, spaceAfter=10,
    )
    cover_kv_label = ParagraphStyle(
        "cover_kv_lbl", parent=ST["kv_label"],
        alignment=TA_RIGHT,
    )
    cover_kv_value = ParagraphStyle(
        "cover_kv_val", parent=ST["kv_value"],
        fontName="Helvetica-Bold",
    )

    elements.append(_sp(70))
    elements.append(Paragraph("INFRAGUARD", cover_brand))
    elements.append(HRFlowable(
        width="40%", thickness=1.2, color=BLUE_ACCENT,
        spaceBefore=2, spaceAfter=18, hAlign="CENTER",
    ))
    elements.append(Paragraph(
        "Construction Quality Inspection Report", cover_title,
    ))
    elements.append(_sp(40))

    rows = [
        ["Project Name",      project_name],
        ["Inspection ID",     report_id],
        ["Inspection Date",   gen_date],
        ["Compliance Score",  f"{score}/100"],
        ["Inspection Grade",  grade],
        ["Generated By",      generated_by],
    ]
    cover_rows = [
        [Paragraph(f"{label}", cover_kv_label),
         Paragraph(str(value), cover_kv_value)]
        for label, value in rows
    ]
    cover_tbl = Table(
        cover_rows,
        colWidths=[CONTENT_W * 0.40, CONTENT_W * 0.42],
        hAlign="CENTER",
    )
    cover_tbl.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.4, BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    # Center the key/value block on the page using an outer wrapper
    wrapper = Table([[cover_tbl]], colWidths=[CONTENT_W])
    wrapper.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(wrapper)

    elements.append(_sp(90))
    elements.append(HRFlowable(
        width="100%", thickness=0.5, color=BORDER,
        spaceBefore=0, spaceAfter=8,
    ))
    elements.append(Paragraph(
        "This document is AI-generated and intended for internal "
        "construction quality assurance review.",
        ST["footer"],
    ))
    _page_break(elements)


# =========================================================
# TABLE OF CONTENTS  (new)
# =========================================================
def _table_of_contents(data: dict, ST: dict, elements: list):
    """
    Static, numbered section listing shown right after the cover page.
    NOTE: this lists section titles in document order but does not
    include page numbers — doing that accurately would require
    switching the document build from a single doc.build() pass to a
    two-pass multiBuild() with a live TableOfContents flowable (which
    tracks page numbers via outline entries). That's a build-process
    change, not a one-line addition, so it's flagged here rather than
    silently faked with placeholder numbers.
    """
    elements.append(Paragraph("Table of Contents", ST["section_heading"]))
    elements.append(_sp(6))

    toc_items = [
        "Inspection Metadata",
        "Executive Dashboard",
        "Site Health Index",
        "Executive Risk Summary",
        "Executive Summary",
        "AI Findings",
        "Inspection Intelligence",
        "Recurring Issue Analysis",
        "Management Attention",
        "Risk Matrix",
        "Department Exposure Dashboard",
        "Site Intelligence Dashboard",
        "Image-Wise Inspection Results",
        "Audit Readiness",
        "Severity Summary",
        "Charts",
        "Compliance Matrix",
        "Finding Closure Matrix",
        "Responsible Team Matrix",
        "Executive Recommendations",
        "Follow-Up Action Matrix",
        "Conclusion",
        "Inspection Certification",
        "Executive Decision",
        "Executive Sign-Off",
        "Appendix",
    ]

    toc_style = ParagraphStyle(
        "toc_item", parent=ST["body"], spaceAfter=6,
    )
    rows = [
        [Paragraph(f"{i}.", toc_style), Paragraph(title, toc_style)]
        for i, title in enumerate(toc_items, 1)
    ]
    tbl = Table(rows, colWidths=[CONTENT_W * 0.06, CONTENT_W * 0.94])
    tbl.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(tbl)
    elements.append(_sp(8))
    elements.append(Paragraph(
        "Section order reflects this report's structure; not all "
        "sections render in every report (e.g. those depending on "
        "multi-image or recurrence data only appear when applicable).",
        ST["caption"],
    ))
    _page_break(elements)


# =========================================================
# INSPECTION METADATA SECTION  (new — Priority 2)
# =========================================================
def _inspection_metadata(data: dict, ST: dict, elements: list):
    """
    Consolidates scattered metadata (project, site, inspector,
    date, version, review status) into a single up-front section
    so reviewers don't have to hunt for it across the report.
    Values are sourced from the report_generator.py payload.
    """
    elements.append(Paragraph("Inspection Information", ST["section_heading"]))
    elements.append(_sp(4))

    project_name  = data.get("project_name", "—")
    site_location = data.get("site_location", "—")
    inspector     = data.get("inspector_name", data.get("inspector", "—"))
    gen_date      = data.get("generated_date",
                              datetime.now().strftime("%d-%b-%Y"))
    report_ver    = data.get("revision", data.get("report_version", "1.0"))
    review_status = data.get("review_status", "Pending Review")

    rows = [
        [Paragraph("<b>Project Name</b>",   ST["kv_label"]),
         Paragraph(str(project_name),        ST["kv_value"])],
        [Paragraph("<b>Site Location</b>",  ST["kv_label"]),
         Paragraph(str(site_location),       ST["kv_value"])],
        [Paragraph("<b>Inspector</b>",      ST["kv_label"]),
         Paragraph(str(inspector),           ST["kv_value"])],
        [Paragraph("<b>Inspection Date</b>", ST["kv_label"]),
         Paragraph(str(gen_date),            ST["kv_value"])],
        [Paragraph("<b>Report Version</b>", ST["kv_label"]),
         Paragraph(str(report_ver),          ST["kv_value"])],
        [Paragraph("<b>Review Status</b>",  ST["kv_label"]),
         Paragraph(str(review_status),       ST["kv_value"])],
    ]

    tbl = Table(rows, colWidths=[CONTENT_W * 0.30, CONTENT_W * 0.70])
    tbl.setStyle(TableStyle([
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("BACKGROUND",    (0, 0), (0, -1),  LABEL_BG),
        ("BACKGROUND",    (1, 0), (1, -1),  WHITE),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(tbl)
    elements.append(_sp(14))


# =========================================================
# EXECUTIVE DASHBOARD  (colored KPI cards + risk trend)
# =========================================================
def _executive_dashboard(data: dict, ST: dict, elements: list):
    elements.append(Paragraph("Executive Dashboard", ST["section_heading"]))
    elements.append(_sp(4))

    score        = data.get("compliance_score", 0)
    grade        = data.get("inspection_grade", "—")
    overall_risk = data.get("overall_risk", data.get("overall_status", "—"))
    audit_ready  = data.get("audit_readiness", "—")
    exec_risk    = data.get("executive_risk_index", "—")
    total        = data.get("total_findings", 0)

    exec_risk_num = exec_risk if isinstance(exec_risk, (int, float)) else None
    exec_risk_str = f"{exec_risk}/100" if exec_risk_num is not None else str(exec_risk)

    # Pick card color scheme based on semantics
    def _score_scheme():
        if score >= 80: return CARD_GREEN,  CARD_GREEN_B
        if score >= 50: return CARD_ORANGE, CARD_ORANGE_B
        return CARD_RED, CARD_RED_B

    def _risk_scheme(r):
        r = str(r).lower()
        if "critical" in r: return CARD_RED,    CARD_RED_B
        if "high"     in r: return CARD_ORANGE, CARD_ORANGE_B
        if "medium"   in r: return CARD_ORANGE, CARD_ORANGE_B
        return CARD_GREEN, CARD_GREEN_B

    def _audit_scheme(a):
        a = str(a).lower()
        if "not" in a or "fail" in a: return CARD_RED, CARD_RED_B
        if "conditional" in a:         return CARD_ORANGE, CARD_ORANGE_B
        return CARD_GREEN, CARD_GREEN_B

    def _exec_risk_scheme(v):
        if v is None: return CARD_SLATE, CARD_SLATE_B
        if v >= 75:  return CARD_RED,    CARD_RED_B
        if v >= 50:  return CARD_ORANGE, CARD_ORANGE_B
        return CARD_GREEN, CARD_GREEN_B

    s_bg, s_fg   = _score_scheme()
    r_bg, r_fg   = _risk_scheme(overall_risk)
    a_bg, a_fg   = _audit_scheme(audit_ready)
    e_bg, e_fg   = _exec_risk_scheme(exec_risk_num)

    def _kpi_card(title: str, value: str, bg, fg):
        title_style = ParagraphStyle(
            f"kt_{title[:4]}",
            parent=ST["card_title"],
            textColor=fg,
            fontSize=7,
        )
        value_style = ParagraphStyle(
            f"kv_{title[:4]}",
            parent=ST["card_value"],
            textColor=fg,
            fontSize=13,
        )
        card = Table(
            [[Paragraph(title, title_style)],
             [Paragraph(value, value_style)]],
            colWidths=[CONTENT_W / 6 - 6],
        )
        card.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), bg),
            ("BOX",           (0, 0), (-1, -1), 1.0, fg),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ]))
        return card

    cards = [
        _kpi_card("Compliance Score", f"{score}/100",    s_bg,         s_fg),
        _kpi_card("Inspection Grade", grade,              CARD_BLUE,    CARD_BLUE_B),
        _kpi_card("Overall Risk",     overall_risk,       r_bg,         r_fg),
        _kpi_card("Audit Readiness",  audit_ready,        a_bg,         a_fg),
        _kpi_card("Exec Risk Index",  exec_risk_str,      e_bg,         e_fg),
        _kpi_card("Total Findings",   str(total),         CARD_SLATE,   CARD_SLATE_B),
    ]

    dash_tbl = Table(
        [cards],
        colWidths=[CONTENT_W / 6] * 6,
        hAlign="LEFT",
    )
    dash_tbl.setStyle(TableStyle([
        ("LEFTPADDING",   (0, 0), (-1, -1), 3),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 3),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(dash_tbl)
    elements.append(_sp(10))

    # ── Risk Trend Indicator ──────────────────────────────
    if exec_risk_num is not None:
        if exec_risk_num >= 75:
            trend_symbol = "\u2191\u2191"   # ↑↑
            trend_label  = "Critical — Immediate Action Required"
            trend_color  = SEV_CRITICAL
        elif exec_risk_num >= 50:
            trend_symbol = "\u2191"          # ↑
            trend_label  = "Increasing — Monitor Closely"
            trend_color  = SEV_HIGH
        else:
            trend_symbol = "\u2193"          # ↓
            trend_label  = "Stable — Within Acceptable Range"
            trend_color  = SEV_LOW

        hex_tc = trend_color.hexval()[2:]
        trend_para = Paragraph(
            f'<b>Risk Trend:</b>  '
            f'<font color="#{hex_tc}"><b>{trend_symbol}  {trend_label}</b></font>'
            f'  <font color="#{CARD_SLATE_B.hexval()[2:]}">'
            f'(Executive Risk Index: {exec_risk_num}/100)</font>',
            ST["body"],
        )
        trend_tbl = Table([[trend_para]], colWidths=[CONTENT_W])
        trend_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), CARD_BG),
            ("BOX",           (0, 0), (-1, -1), 0.8, trend_color),
            ("TOPPADDING",    (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ]))
        elements.append(trend_tbl)

    elements.append(_sp(14))


# =========================================================
# SITE HEALTH INDEX  (new — Priority: visual score gauge)
# =========================================================
def _site_health_index(data: dict, ST: dict, elements: list):
    """
    Single-glance visual gauge derived from compliance_score. Distinct
    from the Compliance Score KPI card in the Executive Dashboard —
    this is a standalone, large-format block intended as the one
    number an executive reader takes away from a quick skim.
    """
    score = data.get("compliance_score", 0)
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0

    if score >= 90:
        rating, bg, fg = "Excellent", CARD_GREEN, CARD_GREEN_B
    elif score >= 75:
        rating, bg, fg = "Good", CARD_GREEN, CARD_GREEN_B
    elif score >= 50:
        rating, bg, fg = "Fair", CARD_ORANGE, CARD_ORANGE_B
    else:
        rating, bg, fg = "Poor", CARD_RED, CARD_RED_B

    label_style = ParagraphStyle(
        "shi_label", parent=ST["card_title"],
        textColor=fg, fontSize=10, alignment=TA_CENTER,
    )
    score_style = ParagraphStyle(
        "shi_score", parent=ST["card_value"],
        textColor=fg, fontSize=30, alignment=TA_CENTER,
    )
    rating_style = ParagraphStyle(
        "shi_rating", parent=ST["card_value"],
        textColor=fg, fontSize=13, alignment=TA_CENTER,
    )

    gauge = Table(
        [[Paragraph("SITE HEALTH INDEX", label_style)],
         [Paragraph(f"{score:g} / 100", score_style)],
         [Paragraph(rating, rating_style)]],
        colWidths=[CONTENT_W],
    )
    gauge.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("BOX",           (0, 0), (-1, -1), 1.2, fg),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(gauge)
    elements.append(_sp(14))


# =========================================================
# SEVERITY DISTRIBUTION SUMMARY  (new — Priority 3)
# =========================================================
def _severity_summary(data: dict, ST: dict, elements: list):
    """
    Colored severity-count cards shown ahead of the bar charts.
    Management typically scans this row before reading any chart,
    so it earns a dedicated spot rather than living inside the
    chart section.
    """
    elements.append(Paragraph("Severity Summary", ST["section_heading"]))
    elements.append(_sp(4))

    sev_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    breakdown  = data.get("severity_breakdown")
    if breakdown:
        for k in sev_counts:
            sev_counts[k] = breakdown.get(k, 0)
    else:
        for ir in data.get("image_reports", []):
            for f in ir.get("findings", []):
                s = f.get("severity", "Medium")
                if s in sev_counts:
                    sev_counts[s] += 1

    scheme = {
        "Critical": (CARD_RED,    CARD_RED_B),
        "High":     (CARD_ORANGE, CARD_ORANGE_B),
        "Medium":   (CARD_ORANGE, CARD_ORANGE_B),
        "Low":      (CARD_GREEN,  CARD_GREEN_B),
    }

    def _sev_card(label, count, bg, fg):
        title_style = ParagraphStyle(
            f"sevt_{label}", parent=ST["card_title"],
            textColor=fg, fontSize=8,
        )
        value_style = ParagraphStyle(
            f"sevv_{label}", parent=ST["card_value"],
            textColor=fg, fontSize=18,
        )
        card = Table(
            [[Paragraph(label.upper(), title_style)],
             [Paragraph(str(count), value_style)]],
            colWidths=[CONTENT_W / 4 - 6],
        )
        card.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), bg),
            ("BOX",           (0, 0), (-1, -1), 1.0, fg),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ]))
        return card

    cards = [
        _sev_card(label, sev_counts[label], *scheme[label])
        for label in ("Critical", "High", "Medium", "Low")
    ]
    row = Table([cards], colWidths=[CONTENT_W / 4] * 4)
    row.setStyle(TableStyle([
        ("LEFTPADDING",   (0, 0), (-1, -1), 3),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 3),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(row)
    elements.append(_sp(14))


# =========================================================
# EXECUTIVE RISK SUMMARY PAGE  (new — Priority 4)
# =========================================================
def _executive_risk_summary(data: dict, ST: dict, elements: list):
    """
    Stand-alone executive risk callout shown right after the
    dashboard. Pulls overall_risk / executive_risk_index /
    audit_readiness straight from the payload — these already
    exist on the dashboard cards, this just gives them a
    dedicated, larger-format page for executives skimming only
    the front matter.
    """
    elements.append(Paragraph("Executive Risk Assessment", ST["section_heading"]))
    elements.append(_sp(4))

    overall_risk = data.get("overall_risk", data.get("overall_status", "—"))
    exec_risk    = data.get("executive_risk_index", "—")
    audit_ready  = data.get("audit_readiness", "—")
    total        = data.get("total_findings", 0)

    exec_risk_str = (
        f"{exec_risk}/100" if isinstance(exec_risk, (int, float))
        else str(exec_risk)
    )

    needs_attention = str(overall_risk).lower() in ("critical", "high") or (
        isinstance(exec_risk, (int, float)) and exec_risk >= 60
    )
    attention_text  = "YES" if needs_attention else "NO"
    attention_color = SEV_CRITICAL if needs_attention else SEV_LOW

    risk_hex = _sev_color(overall_risk).hexval()[2:]
    att_hex  = attention_color.hexval()[2:]

    rows = [
        [Paragraph("<b>Overall Site Risk</b>", ST["kv_label"]),
         Paragraph(f'<font color="#{risk_hex}"><b>{overall_risk}</b></font>',
                    ST["kv_value"])],
        [Paragraph("<b>Executive Risk Index</b>", ST["kv_label"]),
         Paragraph(f"<b>{exec_risk_str}</b>", ST["kv_value"])],
        [Paragraph("<b>Audit Readiness</b>", ST["kv_label"]),
         Paragraph(f"<b>{audit_ready}</b>", ST["kv_value"])],
        [Paragraph("<b>Total Findings</b>", ST["kv_label"]),
         Paragraph(f"<b>{total}</b>", ST["kv_value"])],
        [Paragraph("<b>Immediate Attention Required</b>", ST["kv_label"]),
         Paragraph(f'<font color="#{att_hex}"><b>{attention_text}</b></font>',
                    ST["kv_value"])],
    ]

    tbl = Table(rows, colWidths=[CONTENT_W * 0.40, CONTENT_W * 0.60])
    tbl.setStyle(TableStyle([
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("BACKGROUND",    (0, 0), (0, -1),  LABEL_BG),
        ("BACKGROUND",    (1, 0), (1, -1),  WHITE),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(tbl)
    elements.append(_sp(14))


# =========================================================
# SECTION 1 — EXECUTIVE SUMMARY
# =========================================================
def _executive_summary(data: dict, ST: dict, elements: list, sec_num: int = 1):
    _section_heading(sec_num, "Executive Summary", ST, elements)

    site_summary    = data.get("site_summary",
                                data.get("executive_summary", ""))
    priority_action = data.get("priority_action",
                                "Rectify all critical findings immediately.")

    elements.append(Paragraph(site_summary, ST["body_justify"]))
    elements.append(_sp(8))
    elements.append(Paragraph(
        f"<b>Priority Action:</b> {priority_action}",
        ST["body"],
    ))
    elements.append(_sp(10))


# =========================================================
# SECTION 2 — EXECUTIVE NARRATIVE
# =========================================================
def _executive_narrative(data: dict, ST: dict, elements: list, sec_num: int):
    narrative = data.get("site_executive_narrative", "")
    if not narrative:
        return

    _section_heading(sec_num, "Executive Narrative", ST, elements)
    elements.append(Paragraph(narrative, ST["body_justify"]))
    elements.append(_sp(10))


# =========================================================
# SECTION 3 — AI FINDINGS
# =========================================================
def _ai_findings(data: dict, ST: dict, elements: list, sec_num: int = 2):
    _section_heading(sec_num, "AI Findings", ST, elements)

    bullets = data.get("ai_findings_bullets", [])
    if not bullets:
        total = data.get("total_findings", 0)
        bullets = [
            f"A total of {total} visible quality issue(s) were identified "
            f"in the inspected image.",
            "Corrective action and preventive controls are recommended to "
            "improve site discipline.",
        ]

    for b in bullets:
        elements.append(Paragraph(f"\u2022  {b}", ST["bullet"]))
    elements.append(_sp(10))


# =========================================================
# INSPECTION INTELLIGENCE  (new — Phase 1)
# =========================================================
def _inspection_intelligence(data: dict, sec_num: int, ST: dict, elements: list):
    _section_heading(sec_num, "Inspection Intelligence", ST, elements)

    dominant      = data.get("dominant_issue", "—")
    pattern       = data.get("site_pattern_analysis", "—")
    systemic      = data.get("systemic_deficiencies", "—")
    site_intel    = data.get("site_intelligence_summary", "—")
    pattern_narr  = data.get("pattern_narrative", "")

    rows = [
        ["Dominant Issue",        dominant],
        ["Pattern Analysis",      pattern],
        ["Systemic Deficiencies", systemic],
        ["Site Intelligence",     site_intel],
    ]
    if pattern_narr:
        rows.append(["Pattern Narrative", pattern_narr])

    tbl_data = [
        [Paragraph(f"<b>{r[0]}</b>", ST["kv_label"]),
         Paragraph(str(r[1]),         ST["kv_value"])]
        for r in rows
    ]

    tbl = Table(tbl_data, colWidths=[CONTENT_W * 0.30, CONTENT_W * 0.70])
    tbl.setStyle(TableStyle([
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("BACKGROUND",    (0, 0), (0, -1),  LABEL_BG),
        ("BACKGROUND",    (1, 0), (1, -1),  WHITE),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(tbl)
    elements.append(_sp(14))


# =========================================================
# RECURRING ISSUE ANALYSIS  (new — Phase 2; recurrence badge — Priority 7)
# =========================================================
def _recurrence_level(count) -> str:
    """Classify a dominant-issue occurrence count into a recurrence tier."""
    try:
        n = int(count)
    except (TypeError, ValueError):
        return "Isolated"
    if n >= 8:
        return "Systemic"
    if n >= 5:
        return "Frequent"
    if n >= 2:
        return "Recurring"
    return "Isolated"


def _recurrence_badge(level: str, ST: dict):
    """Colored badge Table for the recurrence level (mirrors _severity_badge)."""
    color_map = {
        "Isolated":  SEV_LOW,
        "Recurring": SEV_MEDIUM,
        "Frequent":  SEV_HIGH,
        "Systemic":  SEV_CRITICAL,
    }
    bg    = color_map.get(level, INK_LIGHT)
    label = f"{level.upper()} ISSUE" if level == "Systemic" else level.upper()

    badge = Table([[Paragraph(label, ST["badge"])]], colWidths=[110])
    badge.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", (0, 0), (-1, -1), [3, 3, 3, 3]),
    ]))
    return badge


def _recurring_issue_analysis(data: dict, sec_num: int, ST: dict, elements: list):
    _section_heading(sec_num, "Recurring Issue Analysis", ST, elements)

    dominant      = data.get("dominant_issue", "—")
    count         = data.get("dominant_issue_count", "—")
    narrative     = data.get("pattern_narrative",
                              "Recurring deficiencies indicate a site-wide "
                              "control weakness requiring immediate attention.")

    # ── Recurrence Level badge (Priority 7) ───────────────
    level = data.get("recurrence_level") or _recurrence_level(count)
    badge_row = Table(
        [[Paragraph("<b>Recurrence Level</b>", ST["kv_label"]),
          _recurrence_badge(level, ST)]],
        colWidths=[CONTENT_W * 0.25, CONTENT_W * 0.30],
    )
    badge_row.setStyle(TableStyle([
        ("VALIGN",  (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",   (1, 0), (1, 0),   "LEFT"),
    ]))
    elements.append(badge_row)
    elements.append(_sp(8))

    rows = [
        [Paragraph("<b>Issue</b>",       ST["kv_label"]),
         Paragraph(str(dominant),         ST["kv_value"])],
        [Paragraph("<b>Occurrences</b>", ST["kv_label"]),
         Paragraph(str(count),            ST["kv_value"])],
        [Paragraph("<b>Narrative</b>",   ST["kv_label"]),
         Paragraph(str(narrative),        ST["kv_value"])],
    ]

    tbl = Table(rows, colWidths=[CONTENT_W * 0.25, CONTENT_W * 0.75])
    tbl.setStyle(TableStyle([
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("BACKGROUND",    (0, 0), (0, -1),  LABEL_BG),
        ("BACKGROUND",    (1, 0), (1, -1),  WHITE),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(tbl)
    elements.append(_sp(14))


# =========================================================
# MANAGEMENT ATTENTION  (new — Phase 1)
# =========================================================
def _management_attention(data: dict, sec_num: int, ST: dict, elements: list):
    _section_heading(sec_num, "Management Attention Required", ST, elements)

    areas = data.get("management_attention_areas", [])
    if not areas:
        elements.append(Paragraph("No specific management attention areas identified.", ST["body"]))
        elements.append(_sp(8))
        return

    for area in areas:
        elements.append(Paragraph(f"\u2022  {area}", ST["bullet"]))
    elements.append(_sp(12))


# =========================================================
# RISK MATRIX  (new — Phase 1)
# =========================================================
def _risk_matrix(data: dict, sec_num: int, ST: dict, elements: list):
    _section_heading(sec_num, "Risk Matrix", ST, elements)

    matrix = data.get("risk_matrix", {})
    if not matrix:
        elements.append(Paragraph("No risk matrix data available.", ST["body"]))
        elements.append(_sp(8))
        return

    header = [
        Paragraph("<b>Risk Category</b>", ST["tbl_hdr"]),
        Paragraph("<b>Count</b>",         ST["tbl_hdr"]),
    ]
    rows = [header]

    for cat, cnt in matrix.items():
        c_hex = _sev_color(cat).hexval()[2:]
        rows.append([
            Paragraph(
                f'<font color="#{c_hex}"><b>{cat}</b></font>',
                ST["tbl_cell"],
            ),
            Paragraph(str(cnt), ST["tbl_cell"]),
        ])

    tbl = Table(rows, colWidths=[CONTENT_W * 0.60, CONTENT_W * 0.40])
    style_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0),  BLUE_ACCENT),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("BACKGROUND",    (0, 1), (-1, -1), WHITE),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i in range(2, len(rows), 2):
        style_cmds.append(("BACKGROUND", (0, i), (-1, i), BLUE_LIGHT))
    tbl.setStyle(TableStyle(style_cmds))
    elements.append(tbl)
    elements.append(_sp(14))


# =========================================================
# DEPARTMENT EXPOSURE DASHBOARD  (new — Priority 6)
# =========================================================
def _department_exposure(data: dict, sec_num: int, ST: dict, elements: list):
    """
    Shows finding counts by responsible department, sourced from
    analysis_service's `department_exposure`. Falls back to grouping
    findings by `department` / `responsible_team` field if the
    pre-aggregated dict isn't present, so the section still renders
    on older payloads.
    """
    _section_heading(sec_num, "Department Exposure Dashboard", ST, elements)

    exposure = data.get("department_exposure")
    if not exposure:
        exposure = {}
        for ir in data.get("image_reports", []):
            for f in ir.get("findings", []):
                dept = f.get("department", f.get("responsible_team", "Unassigned"))
                exposure[dept] = exposure.get(dept, 0) + 1

    if not exposure:
        elements.append(Paragraph(
            "No department exposure data available.", ST["body"],
        ))
        elements.append(_sp(8))
        return

    header = [
        Paragraph("<b>Department</b>", ST["tbl_hdr"]),
        Paragraph("<b>Findings</b>",   ST["tbl_hdr"]),
    ]
    rows = [header]
    for dept, cnt in exposure.items():
        rows.append([
            Paragraph(str(dept), ST["tbl_cell"]),
            Paragraph(str(cnt),  ST["tbl_cell"]),
        ])

    table_w = CONTENT_W * 0.46
    chart_w = CONTENT_W * 0.50

    tbl = Table(rows, colWidths=[table_w * 0.65, table_w * 0.35])
    style_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0),  BLUE_ACCENT),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("BACKGROUND",    (0, 1), (-1, -1), WHITE),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i in range(2, len(rows), 2):
        style_cmds.append(("BACKGROUND", (0, i), (-1, i), BLUE_LIGHT))
    tbl.setStyle(TableStyle(style_cmds))

    palette = [BLUE_ACCENT, TEAL, GOLD, PURPLE, SEV_HIGH, INK_LIGHT]
    dept_names  = list(exposure.keys())
    dept_values = [int(v) for v in exposure.values()]
    dept_colors = [palette[i % len(palette)] for i in range(len(dept_names))]

    mini_chart = _bar_chart(
        "Findings by Department",
        dept_names, dept_values, dept_colors,
        width=chart_w, height=120,
    )

    layout = Table([[tbl, mini_chart]], colWidths=[table_w, chart_w])
    layout.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    elements.append(layout)
    elements.append(_sp(14))


# =========================================================
# SECTION N — ORIGINAL INSPECTION IMAGE
# =========================================================
def _original_image(img_data: dict, sec_num: int, ST: dict, elements: list):
    _section_heading(sec_num, "Original Inspection Image", ST, elements)
    _inline_img(img_data.get("image_path", ""), ST, elements)


# =========================================================
# SECTION N+1 — AI VISUAL INSPECTION OUTPUT (enhanced)
# =========================================================
def _annotated_image(img_data: dict, sec_num: int, ST: dict, elements: list):
    _section_heading(sec_num, "AI Visual Inspection Output", ST, elements)
    elements.append(Paragraph(
        "The following image presents the AI-assisted visual inspection overlay, "
        "highlighting inferred issue regions and quality concern labels.",
        ST["body"],
    ))
    elements.append(_sp(6))
    _inline_img(img_data.get("annotated_image_path", ""), ST, elements)

    # Image-level metrics
    img_score    = img_data.get("compliance_score", "—")
    risk_rating  = img_data.get("risk_level", "—")
    finding_cnt  = img_data.get("total_findings", len(img_data.get("findings", [])))
    comp_impact  = img_data.get("compliance_status", "—")
    sev_dist     = img_data.get("severity_breakdown", {})

    sev_parts = ", ".join(
        f"{k}: {v}" for k, v in sev_dist.items() if v
    ) if sev_dist else "—"

    # Evidence Classification derived from avg confidence
    findings_list = img_data.get("findings", [])
    if findings_list:
        avg_conf = sum(f.get("confidence", 0.0) for f in findings_list) / len(findings_list)
    else:
        avg_conf = img_data.get("confidence", 0.0)

    if avg_conf >= 0.90:
        ev_quality = "Verified — High Confidence"
        ev_color   = SEV_LOW
    elif avg_conf >= 0.75:
        ev_quality = "High Confidence"
        ev_color   = SEV_LOW
    elif avg_conf >= 0.55:
        ev_quality = "Moderate Confidence"
        ev_color   = SEV_MEDIUM
    else:
        ev_quality = "Review Required — Low Confidence"
        ev_color   = SEV_HIGH

    ev_hex = ev_color.hexval()[2:]

    img_rows = [
        [Paragraph("<b>Image Score</b>",           ST["kv_label"]),
         Paragraph(f"{img_score}/100",              ST["kv_value"])],
        [Paragraph("<b>Risk Rating</b>",            ST["kv_label"]),
         Paragraph(str(risk_rating),                ST["kv_value"])],
        [Paragraph("<b>Finding Count</b>",          ST["kv_label"]),
         Paragraph(str(finding_cnt),                ST["kv_value"])],
        [Paragraph("<b>Compliance Impact</b>",      ST["kv_label"]),
         Paragraph(str(comp_impact),                ST["kv_value"])],
        [Paragraph("<b>Severity Distribution</b>",  ST["kv_label"]),
         Paragraph(sev_parts,                       ST["kv_value"])],
        [Paragraph("<b>Evidence Quality</b>",       ST["kv_label"]),
         Paragraph(
             f'<font color="#{ev_hex}"><b>{ev_quality}</b></font>'
             f'  <font color="#{CARD_SLATE_B.hexval()[2:]}">'
             f'(avg conf: {avg_conf:.2f})</font>',
             ST["kv_value"],
         )],
    ]
    tbl = Table(img_rows, colWidths=[CONTENT_W * 0.32, CONTENT_W * 0.68])
    tbl.setStyle(TableStyle([
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("BACKGROUND",    (0, 0), (0, -1),  LABEL_BG),
        ("BACKGROUND",    (1, 0), (1, -1),  WHITE),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(tbl)
    elements.append(_sp(10))


# =========================================================
# SECTION N+2 — INSPECTION FINDINGS SUMMARY TABLE (enhanced)
# =========================================================
def _findings_summary_table(img_data: dict, sec_num: int,
                             ST: dict, elements: list):
    _section_heading(sec_num, "Inspection Findings Summary", ST, elements)

    findings = img_data.get("findings", [])
    bd       = img_data.get("severity_breakdown", {})
    total    = img_data.get("total_findings", len(findings))

    # Summary card above table
    crit  = bd.get("Critical", sum(1 for f in findings if f.get("severity", "") == "Critical"))
    high  = bd.get("High",     sum(1 for f in findings if f.get("severity", "") == "High"))
    med   = bd.get("Medium",   sum(1 for f in findings if f.get("severity", "") == "Medium"))
    low   = bd.get("Low",      sum(1 for f in findings if f.get("severity", "") == "Low"))

    summary_card_rows = [[
        Paragraph(f"<b>Total Findings:</b> {total}", ST["body"]),
        Paragraph(
            f'<font color="#{SEV_CRITICAL.hexval()[2:]}"><b>Critical: {crit}</b></font>  '
            f'<font color="#{SEV_HIGH.hexval()[2:]}">High: {high}</font>  '
            f'<font color="#{SEV_MEDIUM.hexval()[2:]}">Medium: {med}</font>  '
            f'<font color="#{SEV_LOW.hexval()[2:]}">Low: {low}</font>',
            ST["body"],
        ),
    ]]
    card_tbl = Table(summary_card_rows, colWidths=[CONTENT_W * 0.28, CONTENT_W * 0.72])
    card_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), CARD_BG),
        ("BOX",           (0, 0), (-1, -1), 0.8, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(card_tbl)
    elements.append(_sp(8))

    if not findings:
        elements.append(Paragraph("No findings recorded.", ST["body"]))
        elements.append(_sp(8))
        return

    header = [
        Paragraph("<b>Issue Type</b>",   ST["tbl_hdr"]),
        Paragraph("<b>Severity</b>",     ST["tbl_hdr"]),
        Paragraph("<b>Confidence</b>",   ST["tbl_hdr"]),
        Paragraph("<b>Category</b>",     ST["tbl_hdr"]),
    ]
    rows = [header]

    for f in findings:
        sev     = f.get("severity", "Medium")
        sc      = _sev_color(sev)
        sc_hex  = sc.hexval()[2:]
        conf    = f.get("confidence", 0.0)
        rows.append([
            Paragraph(
                f.get("issue_type", "").replace("_", " ").title(),
                ST["tbl_cell"],
            ),
            Paragraph(
                f'<font color="#{sc_hex}">{sev}</font>',
                ST["tbl_cell"],
            ),
            Paragraph(f"{conf:.2f}", ST["tbl_cell"]),
            Paragraph(f.get("category", ""), ST["tbl_cell"]),
        ])

    col_w = [
        CONTENT_W * 0.28,
        CONTENT_W * 0.14,
        CONTENT_W * 0.14,
        CONTENT_W * 0.44,
    ]
    tbl = Table(rows, colWidths=col_w)

    style_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0),  BLUE_ACCENT),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("BACKGROUND",    (0, 1), (-1, -1), WHITE),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i in range(2, len(rows), 2):
        style_cmds.append(("BACKGROUND", (0, i), (-1, i), BLUE_LIGHT))

    tbl.setStyle(TableStyle(style_cmds))
    elements.append(tbl)
    elements.append(_sp(14))


# =========================================================
# HELPER — default closure timeline by severity
# =========================================================
def _default_closure(severity: str) -> str:
    return {
        "Critical": "Immediate (0–24 hrs)",
        "High":     "7 Days",
        "Medium":   "30 Days",
        "Low":      "Routine / Next Cycle",
    }.get(severity, "30 Days")


# =========================================================
# SECTION N+3 — DETAILED ISSUE GUIDANCE (enhanced)
# =========================================================
def _detailed_issue_guidance(img_data: dict, sec_num: int,
                              ST: dict, elements: list):
    _section_heading(sec_num, "Detailed Issue Guidance", ST, elements)

    findings = img_data.get("findings", [])
    if not findings:
        elements.append(Paragraph("No detailed findings recorded.", ST["body"]))
        return

    for i, f in enumerate(findings, 1):
        sev           = f.get("severity", "Medium")
        issue_lbl     = f.get("issue_type", "Unknown").replace("_", " ").title()
        finding_id    = f.get("finding_id", f"FND-{i:03d}")
        consequences  = f.get("potential_consequences", "—")
        mgmt_impact   = f.get("management_impact", "—")
        risk_category = f.get("risk_category", "—")

        # Standalone Finding ID heading (e.g. FND-001)
        elements.append(Paragraph(
            f"<b>{finding_id}</b>",
            ST["finding_heading"],
        ))

        # Finding header row: title + badge side by side
        hdr_row = Table(
            [[
                Paragraph(
                    f"<b>{i}. {issue_lbl}</b>",
                    ST["finding_heading"],
                ),
                _severity_badge(sev, ST),
            ]],
            colWidths=[CONTENT_W * 0.75, CONTENT_W * 0.25],
        )
        hdr_row.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
            ("TOPPADDING",    (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        elements.append(hdr_row)
        elements.append(_sp(4))

        field_defs = [
            ("Finding ID",             finding_id),
            ("Observation",            f.get("observation", "—")),
            ("Potential Consequences", consequences),
            ("Risk Category",          risk_category),
            ("Management Impact",      mgmt_impact),
            ("Risk Description",       f.get("risk",
                                             f.get("operational_impact", "—"))),
            ("Corrective Action",      f.get("corrective_action", "—")),
            ("Preventive Action",      f.get("preventive_action", "—")),
            ("Best Practice",          f.get("best_practice", "—")),
            ("Guideline Reference",    f.get("guideline_reference", "—")),
            ("Responsible Party",      f.get("responsible_party",
                                             "Site Supervisor")),
            ("Target Closure Date",    f.get("target_closure_date",
                                             _default_closure(f.get("severity", "Medium")))),
        ]

        for label, value in field_defs:
            elements.append(Paragraph(
                f"<b>{label}:</b> {value}",
                ST["body"],
            ))
            elements.append(_sp(3))

        elements.append(HRFlowable(
            width="100%", thickness=0.4,
            color=BORDER, spaceBefore=6, spaceAfter=8,
        ))


# =========================================================
# CHARTS — Severity / Risk / Issue Distribution (new)
# =========================================================
class _DrawingFlowable(flowables.Flowable):
    """Wraps a ReportLab Drawing so it flows in a Platypus story."""
    def __init__(self, drawing):
        super().__init__()
        self.drawing = drawing
        self.width   = drawing.width
        self.height  = drawing.height

    def draw(self):
        renderPDF.draw(self.drawing, self.canv, 0, 0)

    def wrap(self, *args):
        return self.width, self.height


def _bar_chart(
    title: str,
    labels: list,
    values: list,
    bar_colors: list,
    width: float = 160,
    height: float = 120,
) -> _DrawingFlowable:
    """Build a vertical bar chart drawing."""
    d   = Drawing(width, height)
    pad = 30

    # Title
    d.add(String(
        width / 2, height - 10,
        title,
        fontName="Helvetica-Bold", fontSize=7,
        fillColor=colors.HexColor("#111827"),
        textAnchor="middle",
    ))

    if not values or max(values, default=0) == 0:
        d.add(String(
            width / 2, height / 2,
            "No data",
            fontName="Helvetica", fontSize=7,
            fillColor=colors.HexColor("#9CA3AF"),
            textAnchor="middle",
        ))
        return _DrawingFlowable(d)

    bar_area_h = height - pad - 22
    bar_area_w = width  - 20
    x0         = 10
    y0         = 18
    n          = len(labels)
    bar_w      = max(8, (bar_area_w - (n - 1) * 4) / n)
    max_val    = max(values)

    for idx, (lbl, val, bc) in enumerate(zip(labels, values, bar_colors)):
        bx = x0 + idx * (bar_w + 4)
        bh = (val / max_val) * bar_area_h if max_val else 0
        # Bar
        d.add(Rect(bx, y0, bar_w, bh, fillColor=bc, strokeColor=None))
        # Value label
        d.add(String(
            bx + bar_w / 2, y0 + bh + 2,
            str(val),
            fontName="Helvetica-Bold", fontSize=6,
            fillColor=colors.HexColor("#111827"),
            textAnchor="middle",
        ))
        # X axis label (truncate)
        short = lbl[:6] if len(lbl) > 6 else lbl
        d.add(String(
            bx + bar_w / 2, y0 - 10,
            short,
            fontName="Helvetica", fontSize=6,
            fillColor=colors.HexColor("#374151"),
            textAnchor="middle",
        ))

    # X axis line
    d.add(Line(x0, y0, x0 + bar_area_w, y0,
               strokeColor=colors.HexColor("#D1D5DB"), strokeWidth=0.5))

    return _DrawingFlowable(d)


def _charts_section(data: dict, sec_num: int, ST: dict, elements: list):
    _section_heading(sec_num, "Issue Distribution Charts", ST, elements)

    # Collect all findings
    all_findings = []
    for ir in data.get("image_reports", []):
        all_findings.extend(ir.get("findings", []))

    # ── Chart 1: Severity Distribution ───────────────────
    sev_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for f in all_findings:
        s = f.get("severity", "Medium")
        if s in sev_counts:
            sev_counts[s] += 1

    sev_chart = _bar_chart(
        "Severity Distribution",
        list(sev_counts.keys()),
        list(sev_counts.values()),
        [SEV_CRITICAL, SEV_HIGH, SEV_MEDIUM, SEV_LOW],
        width=150, height=110,
    )

    # ── Chart 2: Risk Category ────────────────────────────
    risk_matrix = data.get("risk_matrix", {})
    if not risk_matrix:
        risk_matrix = {"Critical": sev_counts.get("Critical", 0),
                       "High":     sev_counts.get("High", 0),
                       "Medium":   sev_counts.get("Medium", 0),
                       "Low":      sev_counts.get("Low", 0)}

    risk_labels = list(risk_matrix.keys())
    risk_vals   = [int(v) for v in risk_matrix.values()]
    risk_cols   = [_sev_color(k) for k in risk_labels]

    risk_chart = _bar_chart(
        "Risk Category Distribution",
        risk_labels, risk_vals, risk_cols,
        width=150, height=110,
    )

    # ── Chart 3: Issue Type Distribution ─────────────────
    issue_counts: dict = {}
    for f in all_findings:
        it = f.get("issue_type", "Other").replace("_", " ").title()
        issue_counts[it] = issue_counts.get(it, 0) + 1

    top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:6]
    if top_issues:
        i_labels, i_vals = zip(*top_issues)
        palette = [
            colors.HexColor("#2563EB"),
            colors.HexColor("#7C3AED"),
            colors.HexColor("#0F766E"),
            colors.HexColor("#D97706"),
            colors.HexColor("#B91C1C"),
            colors.HexColor("#374151"),
        ]
        i_cols = palette[:len(i_labels)]
    else:
        i_labels, i_vals, i_cols = ["No Data"], [0], [BORDER]

    issue_chart = _bar_chart(
        "Top Issue Types",
        list(i_labels), list(i_vals), i_cols,
        width=150, height=110,
    )

    # ── Layout: 3 charts side by side ────────────────────
    chart_row = Table(
        [[sev_chart, risk_chart, issue_chart]],
        colWidths=[CONTENT_W / 3] * 3,
    )
    chart_row.setStyle(TableStyle([
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
        ("BACKGROUND",    (0, 0), (-1, -1), CARD_BG),
    ]))
    elements.append(chart_row)
    elements.append(_sp(14))


# =========================================================
# FINDING CLOSURE MATRIX  (new)
# =========================================================
def _finding_closure_matrix(data: dict, sec_num: int, ST: dict, elements: list):
    _section_heading(sec_num, "Finding Closure Matrix", ST, elements)

    all_findings = []
    for ir in data.get("image_reports", []):
        for idx, f in enumerate(ir.get("findings", []), 1):
            all_findings.append((f, idx, ir.get("image_label", "—")))

    if not all_findings:
        elements.append(Paragraph("No findings available.", ST["body"]))
        elements.append(_sp(8))
        return

    header = [
        Paragraph("<b>Finding ID</b>",   ST["tbl_hdr"]),
        Paragraph("<b>Issue</b>",         ST["tbl_hdr"]),
        Paragraph("<b>Severity</b>",      ST["tbl_hdr"]),
        Paragraph("<b>Owner</b>",         ST["tbl_hdr"]),
        Paragraph("<b>Due Date</b>",      ST["tbl_hdr"]),
        Paragraph("<b>Status</b>",        ST["tbl_hdr"]),
    ]
    rows = [header]

    for f, idx, img_lbl in all_findings:
        sev         = f.get("severity", "Medium")
        sev_hex     = _sev_color(sev).hexval()[2:]
        finding_id  = f.get("finding_id", f"F-{idx:03d}")
        issue_lbl   = f.get("issue_type", "Unknown").replace("_", " ").title()
        owner       = f.get("responsible_party", "Site Supervisor")
        due_date    = f.get("target_closure_date", _default_closure(sev))
        status      = f.get("closure_status", "Open")

        status_hex  = (
            SEV_LOW.hexval()[2:]      if status.lower() == "closed"
            else SEV_MEDIUM.hexval()[2:] if "progress" in status.lower()
            else SEV_CRITICAL.hexval()[2:]
        )

        rows.append([
            Paragraph(finding_id,                          ST["tbl_cell"]),
            Paragraph(issue_lbl,                           ST["tbl_cell"]),
            Paragraph(f'<font color="#{sev_hex}">{sev}</font>', ST["tbl_cell"]),
            Paragraph(owner,                               ST["tbl_cell"]),
            Paragraph(due_date,                            ST["tbl_cell"]),
            Paragraph(
                f'<font color="#{status_hex}"><b>{status}</b></font>',
                ST["tbl_cell"],
            ),
        ])

    col_w = [
        CONTENT_W * 0.12,
        CONTENT_W * 0.22,
        CONTENT_W * 0.12,
        CONTENT_W * 0.20,
        CONTENT_W * 0.18,
        CONTENT_W * 0.16,
    ]
    tbl = Table(rows, colWidths=col_w)
    style_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0),  BLUE_ACCENT),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("BACKGROUND",    (0, 1), (-1, -1), WHITE),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i in range(2, len(rows), 2):
        style_cmds.append(("BACKGROUND", (0, i), (-1, i), BLUE_LIGHT))
    tbl.setStyle(TableStyle(style_cmds))
    elements.append(tbl)
    elements.append(_sp(14))


# =========================================================
# RESPONSIBLE TEAM MATRIX  (new — Priority 5)
# =========================================================
def _responsible_team_matrix(data: dict, sec_num: int, ST: dict, elements: list):
    """
    Enterprise-style accountability matrix: Issue / Responsible Team /
    Priority / Target Resolution / Status. Distinct from the Finding
    Closure Matrix above (which tracks Owner/Due Date/Status per
    finding ID) — this view is organized around team ownership and
    priority rather than individual finding IDs, which is what
    management typically reviews in standup/audit meetings.
    """
    _section_heading(sec_num, "Responsible Team Matrix", ST, elements)

    all_findings = []
    for ir in data.get("image_reports", []):
        all_findings.extend(ir.get("findings", []))

    if not all_findings:
        elements.append(Paragraph("No findings available.", ST["body"]))
        elements.append(_sp(8))
        return

    header = [
        Paragraph("<b>Issue</b>",              ST["tbl_hdr"]),
        Paragraph("<b>Responsible Team</b>",   ST["tbl_hdr"]),
        Paragraph("<b>Priority</b>",           ST["tbl_hdr"]),
        Paragraph("<b>Target Resolution</b>",  ST["tbl_hdr"]),
        Paragraph("<b>Status</b>",             ST["tbl_hdr"]),
    ]
    rows = [header]

    _PRIORITY_FROM_SEV = {
        "Critical": "P1 — Immediate",
        "High":     "P2 — Urgent",
        "Medium":   "P3 — Planned",
        "Low":      "P4 — Routine",
    }

    for f in all_findings:
        sev         = f.get("severity", "Medium")
        sev_hex     = _sev_color(sev).hexval()[2:]
        issue_lbl   = f.get("issue_type", "Unknown").replace("_", " ").title()
        team        = f.get("responsible_team",
                             f.get("department",
                                   f.get("responsible_party", "Site Team")))
        priority    = f.get("priority", _PRIORITY_FROM_SEV.get(sev, "P3 — Planned"))
        target      = f.get("target_closure_date", _default_closure(sev))
        status      = f.get("closure_status", "Open")

        status_hex  = (
            SEV_LOW.hexval()[2:]      if status.lower() == "closed"
            else SEV_MEDIUM.hexval()[2:] if "progress" in status.lower()
            else SEV_CRITICAL.hexval()[2:]
        )

        rows.append([
            Paragraph(f'<font color="#{sev_hex}">{issue_lbl}</font>', ST["tbl_cell"]),
            Paragraph(str(team),                                     ST["tbl_cell"]),
            Paragraph(str(priority),                                 ST["tbl_cell"]),
            Paragraph(str(target),                                   ST["tbl_cell"]),
            Paragraph(
                f'<font color="#{status_hex}"><b>{status}</b></font>',
                ST["tbl_cell"],
            ),
        ])

    col_w = [
        CONTENT_W * 0.26,
        CONTENT_W * 0.20,
        CONTENT_W * 0.18,
        CONTENT_W * 0.20,
        CONTENT_W * 0.16,
    ]
    tbl = Table(rows, colWidths=col_w)
    style_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0),  BLUE_ACCENT),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("BACKGROUND",    (0, 1), (-1, -1), WHITE),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i in range(2, len(rows), 2):
        style_cmds.append(("BACKGROUND", (0, i), (-1, i), BLUE_LIGHT))
    tbl.setStyle(TableStyle(style_cmds))
    elements.append(tbl)
    elements.append(_sp(14))


# =========================================================
# EXECUTIVE SIGN-OFF PAGE  (new)
# =========================================================
def _executive_signoff(data: dict, ST: dict, elements: list):
    _page_break(elements)

    # Page header bar
    hdr = Table(
        [[Paragraph("INFRA GUARD AI PLATFORM — DOCUMENT SIGN-OFF", ST["header_tag"])]],
        colWidths=[CONTENT_W],
    )
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), BLUE_ACCENT),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    elements.append(hdr)
    elements.append(_sp(20))

    elements.append(Paragraph("Executive Sign-Off", ST["section_heading"]))
    elements.append(Paragraph(
        "This Construction Quality Inspection Report has been prepared by InfraGuard "
        "Enterprise AI and requires review and sign-off by authorised personnel prior "
        "to formal acceptance and distribution.",
        ST["body_justify"],
    ))
    elements.append(_sp(24))

    signoffs = data.get("signoffs", [
        {"role": "Prepared By",   "name": "", "designation": "AI Inspection System"},
        {"role": "Reviewed By",   "name": "", "designation": "Site Quality Engineer"},
        {"role": "Approved By",   "name": "", "designation": "Project Manager"},
    ])

    for so in signoffs:
        role        = so.get("role", "")
        name        = so.get("name", "")
        designation = so.get("designation", "")

        elements.append(Paragraph(role, ST["signoff_label"]))

        # Signature line
        sig_row = Table(
            [[
                Paragraph(
                    f"Name: {name or '_' * 30}",
                    ST["signoff_line"],
                ),
                Paragraph(
                    f"Designation: {designation or '_' * 20}",
                    ST["signoff_line"],
                ),
                Paragraph(
                    "Date: _____________",
                    ST["signoff_line"],
                ),
            ]],
            colWidths=[CONTENT_W * 0.38, CONTENT_W * 0.36, CONTENT_W * 0.26],
        )
        sig_row.setStyle(TableStyle([
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ]))
        elements.append(sig_row)

        # Signature box
        sig_box = Table(
            [[Paragraph("Signature:", ST["signoff_label"]), ""]],
            colWidths=[CONTENT_W * 0.20, CONTENT_W * 0.80],
        )
        sig_box.setStyle(TableStyle([
            ("BOX",           (1, 0), (1, 0), 0.8, BORDER),
            ("MINROWHEIGHT",  (0, 0), (-1, -1), 40),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(sig_box)
        elements.append(_sp(18))

    elements.append(HRFlowable(
        width="100%", thickness=0.5,
        color=BORDER, spaceBefore=8, spaceAfter=8,
    ))
    elements.append(Paragraph(
        "<b>DISCLAIMER:</b> This report is generated by InfraGuard Enterprise AI. "
        "All findings are based on visual AI analysis and must be verified by a "
        "qualified engineer before corrective actions are initiated. "
        "This document is classified as an Internal Quality Assessment.",
        ST["footer"],
    ))


# =========================================================
# COMPLIANCE MATRIX  (existing — Phase 2)
# =========================================================
def _compliance_matrix(data: dict, sec_num: int, ST: dict, elements: list):
    _section_heading(sec_num, "Compliance Matrix", ST, elements)

    all_findings = []
    for ir in data.get("image_reports", []):
        all_findings.extend(ir.get("findings", []))

    if not all_findings:
        elements.append(Paragraph("No findings available for compliance matrix.", ST["body"]))
        elements.append(_sp(8))
        return

    header = [
        Paragraph("<b>Issue</b>",         ST["tbl_hdr"]),
        Paragraph("<b>Severity</b>",      ST["tbl_hdr"]),
        Paragraph("<b>Risk Category</b>", ST["tbl_hdr"]),
    ]
    rows = [header]
    for f in all_findings:
        sev  = f.get("severity", "—")
        sev_hex = _sev_color(sev).hexval()[2:]
        rows.append([
            Paragraph(f.get("issue_type", "—").replace("_", " ").title(), ST["tbl_cell"]),
            Paragraph(f'<font color="#{sev_hex}">{sev}</font>',           ST["tbl_cell"]),
            Paragraph(f.get("risk_category", "—"),                         ST["tbl_cell"]),
        ])

    tbl = Table(rows, colWidths=[CONTENT_W * 0.44, CONTENT_W * 0.20, CONTENT_W * 0.36])
    style_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0),  BLUE_ACCENT),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("BACKGROUND",    (0, 1), (-1, -1), WHITE),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i in range(2, len(rows), 2):
        style_cmds.append(("BACKGROUND", (0, i), (-1, i), BLUE_LIGHT))
    tbl.setStyle(TableStyle(style_cmds))
    elements.append(tbl)
    elements.append(_sp(14))


# =========================================================
# AUDIT READINESS  (new — Phase 2)
# =========================================================
def _audit_readiness_section(data: dict, sec_num: int, ST: dict, elements: list):
    _section_heading(sec_num, "Audit Readiness", ST, elements)

    audit_ready  = data.get("audit_readiness", "—")
    benchmark    = data.get("compliance_benchmark", "—")
    priority     = data.get("priority_action", "—")
    status       = data.get("overall_status", "—")
    timeline     = data.get("recommended_timeline", "Within 30 days")

    rows = [
        [Paragraph("<b>Current Status</b>",       ST["kv_label"]),
         Paragraph(str(audit_ready),               ST["kv_value"])],
        [Paragraph("<b>Benchmark</b>",             ST["kv_label"]),
         Paragraph(str(benchmark),                 ST["kv_value"])],
        [Paragraph("<b>Priority Action</b>",       ST["kv_label"]),
         Paragraph(str(priority),                  ST["kv_value"])],
        [Paragraph("<b>Overall Status</b>",        ST["kv_label"]),
         Paragraph(str(status),                    ST["kv_value"])],
        [Paragraph("<b>Recommended Timeline</b>",  ST["kv_label"]),
         Paragraph(str(timeline),                  ST["kv_value"])],
    ]

    tbl = Table(rows, colWidths=[CONTENT_W * 0.32, CONTENT_W * 0.68])
    tbl.setStyle(TableStyle([
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("BACKGROUND",    (0, 0), (0, -1),  LABEL_BG),
        ("BACKGROUND",    (1, 0), (1, -1),  WHITE),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(tbl)
    elements.append(_sp(14))


# =========================================================
# FOLLOW-UP MATRIX  (new — Phase 3)
# =========================================================
def _followup_matrix(data: dict, sec_num: int, ST: dict, elements: list):
    _section_heading(sec_num, "Follow-Up Action Matrix", ST, elements)

    header = [
        Paragraph("<b>Timeline</b>", ST["tbl_hdr"]),
        Paragraph("<b>Action</b>",   ST["tbl_hdr"]),
    ]

    followup_rows = [
        ("Immediate (0–24 hrs)", "Address all Critical findings. Halt unsafe operations."),
        ("7 Days",               "Rectify all High severity findings and re-inspect."),
        ("30 Days",              "Remediate Medium severity findings; update site protocols."),
        ("Routine",              "Monitor and close Low severity findings in next inspection cycle."),
    ]

    rows = [header] + [
        [Paragraph(t, ST["tbl_cell"]), Paragraph(a, ST["tbl_cell"])]
        for t, a in followup_rows
    ]

    tbl = Table(rows, colWidths=[CONTENT_W * 0.28, CONTENT_W * 0.72])
    style_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0),  BLUE_ACCENT),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("BACKGROUND",    (0, 1), (-1, -1), WHITE),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i in range(2, len(rows), 2):
        style_cmds.append(("BACKGROUND", (0, i), (-1, i), BLUE_LIGHT))
    tbl.setStyle(TableStyle(style_cmds))
    elements.append(tbl)
    elements.append(_sp(14))


# =========================================================
# AUDIT EXPOSURE  (new — Phase 2)
# =========================================================
def _audit_exposure(data: dict, sec_num: int, ST: dict, elements: list):
    _section_heading(sec_num, "Audit Exposure Summary", ST, elements)

    exposure = data.get(
        "audit_exposure_summary",
        "Current site conditions may result in adverse audit observations. "
        "Immediate corrective measures are recommended prior to any formal audit."
    )
    elements.append(Paragraph(exposure, ST["body_justify"]))
    elements.append(_sp(14))


# =========================================================
# EXECUTIVE RECOMMENDATIONS  (new — Priority 8)
# =========================================================
def _executive_recommendations(data: dict, sec_num: int, ST: dict, elements: list):
    """
    Consolidates management-facing recommendations into one numbered
    list, sourced from report_generator's `executive_recommendations`.
    Previously these were scattered across priority_action,
    audit_readiness, and conclusion fields — this gives them one home.
    """
    _section_heading(sec_num, "Management Recommendations", ST, elements)

    recs = data.get("executive_recommendations")
    if not recs:
        recs = [
            "Address all critical structural and safety findings without delay.",
            "Improve housekeeping and material storage controls site-wide.",
            "Conduct a reinspection of flagged areas within 14 days.",
            "Increase supervisor oversight on recurring non-conformance items.",
        ]

    for i, rec in enumerate(recs, 1):
        elements.append(Paragraph(f"{i}. {rec}", ST["bullet"]))
    elements.append(_sp(12))


# =========================================================
# SITE INTELLIGENCE DASHBOARD  (new — Phase 1)
# =========================================================
def _site_intelligence_dashboard(data: dict, sec_num: int, ST: dict, elements: list):
    _section_heading(sec_num, "Site Intelligence Dashboard", ST, elements)

    analytics = data.get("analytics", {})
    dominant  = analytics.get("dominant_issue",
                               data.get("dominant_issue", "—"))
    hotspot   = analytics.get("site_hotspot", "—")
    cluster   = analytics.get("issue_cluster", "—")
    mgmt_attn = analytics.get("management_attention",
                               data.get("management_attention_areas", ["—"]))
    if isinstance(mgmt_attn, list):
        mgmt_attn = ", ".join(mgmt_attn) if mgmt_attn else "—"

    rows = [
        [Paragraph("<b>Dominant Issue</b>",          ST["kv_label"]),
         Paragraph(str(dominant),                     ST["kv_value"])],
        [Paragraph("<b>Site Hotspot</b>",             ST["kv_label"]),
         Paragraph(str(hotspot),                      ST["kv_value"])],
        [Paragraph("<b>Issue Cluster</b>",            ST["kv_label"]),
         Paragraph(str(cluster),                      ST["kv_value"])],
        [Paragraph("<b>Management Attention</b>",     ST["kv_label"]),
         Paragraph(str(mgmt_attn),                    ST["kv_value"])],
    ]

    tbl = Table(rows, colWidths=[CONTENT_W * 0.32, CONTENT_W * 0.68])
    tbl.setStyle(TableStyle([
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("BACKGROUND",    (0, 0), (0, -1),  LABEL_BG),
        ("BACKGROUND",    (1, 0), (1, -1),  WHITE),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(tbl)
    elements.append(_sp(14))


# =========================================================
# CONCLUSION  (enhanced — Phase 1)
# =========================================================
def _conclusion(data: dict, sec_num: int, ST: dict, elements: list):
    _section_heading(sec_num, "Conclusion", ST, elements)

    insp_outcome  = data.get(
        "inspection_outcome",
        data.get(
            "conclusion",
            "This AI-assisted construction quality inspection report is intended "
            "to support site supervisors, safety engineers, and project teams in "
            "identifying visible quality concerns and improving overall site discipline "
            "through structured corrective and preventive action.",
        ),
    )
    risk_outlook  = data.get("risk_outlook",   "")
    mgmt_rec      = data.get("management_recommendation", "")
    audit_outlook = data.get("audit_outlook",  "")

    elements.append(Paragraph(
        f"<b>Inspection Outcome:</b> {insp_outcome}",
        ST["body_justify"],
    ))
    elements.append(_sp(6))

    if risk_outlook:
        elements.append(Paragraph(
            f"<b>Risk Outlook:</b> {risk_outlook}",
            ST["body"],
        ))
        elements.append(_sp(4))

    if mgmt_rec:
        elements.append(Paragraph(
            f"<b>Management Recommendation:</b> {mgmt_rec}",
            ST["body"],
        ))
        elements.append(_sp(4))

    if audit_outlook:
        elements.append(Paragraph(
            f"<b>Audit Outlook:</b> {audit_outlook}",
            ST["body"],
        ))
        elements.append(_sp(4))

    elements.append(_sp(12))

    elements.append(HRFlowable(
        width="100%", thickness=0.5,
        color=BORDER, spaceBefore=4, spaceAfter=6,
    ))
    elements.append(Paragraph(
        "This report was generated by <b>InfraGuard Enterprise AI</b> — "
        "Construction Quality Intelligence Platform. "
        "All findings must be reviewed by a qualified engineer before "
        "corrective actions are initiated.",
        ST["footer"],
    ))


# =========================================================
# REPORT READINESS CERTIFICATE  (new — Priority 9)
# =========================================================
def _readiness_certificate(data: dict, ST: dict, elements: list):
    """
    Checklist-style certification block shown right before the
    sign-off page. Each line reflects a pipeline stage that has
    completed by the time the PDF is generated; this is a static
    confirmation of process completion, not a substitute for human
    sign-off (which still happens on the following page).
    """
    elements.append(Paragraph("Inspection Certification", ST["section_heading"]))
    elements.append(_sp(6))

    default_items = [
        "AI Analysis Completed",
        "Findings Reviewed",
        "Corrective Actions Generated",
        "Audit Readiness Evaluated",
        "Report Approved For Distribution",
    ]
    items = data.get("certification_items", default_items)

    check_style = ParagraphStyle(
        "cert_check", parent=ST["body"],
        textColor=SEV_LOW, fontName="Helvetica-Bold",
    )
    rows = [[Paragraph(f"\u2713  {item}", check_style)] for item in items]

    tbl = Table(rows, colWidths=[CONTENT_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), CARD_GREEN),
        ("BOX",           (0, 0), (-1, -1), 0.8, CARD_GREEN_B),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.4, colors.HexColor("#BBF7D0")),
    ]))
    elements.append(tbl)
    elements.append(_sp(14))


# =========================================================
# EXECUTIVE DECISION BLOCK  (new)
# =========================================================
def _executive_decision(data: dict, ST: dict, elements: list):
    """
    Checkbox-style decision block for the approving authority, shown
    after the readiness certificate and before formal sign-off lines.
    The pre-selected option (if any) is marked from
    data['executive_decision']; all four remain visible either way
    since this is a printed form, not a rendered decision.

    Boxes are drawn shapes (Drawing/Rect), not the ☐/☒ Unicode glyphs —
    Helvetica's font encoding doesn't include those code points, so
    ReportLab silently renders both as the same solid glyph, making a
    selected option visually indistinguishable from an unselected one.
    """
    elements.append(Paragraph("Executive Decision", ST["section_heading"]))
    elements.append(_sp(4))

    selected = (data.get("executive_decision") or "").strip().lower()

    options = [
        ("approved",                "Approved"),
        ("conditionally_approved",  "Conditionally Approved"),
        ("reinspection_required",   "Reinspection Required"),
        ("immediate_action",        "Immediate Corrective Action Required"),
    ]

    def _checkbox(is_checked: bool):
        box_size = 11
        d = Drawing(box_size + 2, box_size + 2)
        d.add(Rect(
            1, 1, box_size, box_size,
            strokeColor=INK_LIGHT, strokeWidth=1,
            fillColor=(BLUE_ACCENT if is_checked else WHITE),
        ))
        if is_checked:
            # Simple checkmark made of two short diagonal lines
            d.add(Line(2.5, 6, 5, 2.5, strokeColor=WHITE, strokeWidth=1.6))
            d.add(Line(5, 2.5, 10.5, 9.5, strokeColor=WHITE, strokeWidth=1.6))
        return d

    box_style = ParagraphStyle(
        "decision_box", parent=ST["body"], fontSize=11,
    )
    rows = []
    for key, label in options:
        is_checked = (key == selected)
        label_style = (
            ParagraphStyle("decision_box_sel", parent=box_style,
                           fontName="Helvetica-Bold")
            if is_checked else box_style
        )
        rows.append([_checkbox(is_checked), Paragraph(label, label_style)])

    tbl = Table(rows, colWidths=[24, CONTENT_W - 24])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), CARD_BG),
        ("BOX",           (0, 0), (-1, -1), 0.8, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (0, -1),  14),
        ("LEFTPADDING",   (1, 0), (1, -1),  6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.4, BORDER),
    ]))
    elements.append(KeepTogether(tbl))
    elements.append(_sp(14))


# =========================================================
# APPENDIX  (new — Priority 10)
# =========================================================
def _appendix(data: dict, ST: dict, elements: list):
    """
    Reference section closing the report: severity/issue definitions,
    risk classification methodology, and an explanation of AI
    confidence scoring. Static reference content — no dependence on
    per-report data, so it's safe to render unconditionally.
    """
    _page_break(elements)
    elements.append(Paragraph("Appendix", ST["section_heading"]))
    elements.append(_sp(6))

    elements.append(Paragraph("Severity Definitions", ST["subsection_heading"]))
    severity_defs = [
        ("Critical", "Immediate threat to safety or compliance."),
        ("High",     "Significant operational impact."),
        ("Medium",   "Requires planned corrective action."),
        ("Low",      "Routine maintenance item."),
    ]
    for label, desc in severity_defs:
        hexc = _sev_color(label).hexval()[2:]
        elements.append(Paragraph(
            f'<font color="#{hexc}"><b>{label}</b></font> — {desc}',
            ST["body"],
        ))
        elements.append(_sp(3))
    elements.append(_sp(8))

    elements.append(Paragraph("Risk Classification Methodology", ST["subsection_heading"]))
    elements.append(Paragraph(
        "Overall site risk and the Executive Risk Index are derived from "
        "a weighted combination of finding severity distribution, "
        "recurrence of issue types across inspection images, and "
        "current audit readiness status. Higher weight is given to "
        "Critical and High severity findings and to issues classified "
        "as Frequent or Systemic under the Recurring Issue Analysis.",
        ST["body_justify"],
    ))
    elements.append(_sp(8))

    elements.append(Paragraph("AI Confidence Explanation", ST["subsection_heading"]))
    elements.append(Paragraph(
        "Each finding is generated by InfraGuard's computer-vision "
        "inspection model from the submitted site imagery. Findings "
        "represent the model's assessment of visible conditions only "
        "and should be confirmed by a qualified engineer or inspector "
        "before corrective or preventive action is finalized.",
        ST["body_justify"],
    ))
    elements.append(_sp(8))

    elements.append(Paragraph("Issue Definitions", ST["subsection_heading"]))
    elements.append(Paragraph(
        "Issue types referenced throughout this report (e.g. housekeeping, "
        "material storage, structural, PPE compliance) correspond to "
        "categories defined in InfraGuard's construction quality taxonomy "
        "and are assigned automatically based on the visual content of "
        "each inspection image.",
        ST["body_justify"],
    ))


# =========================================================
# FOOTER CALLBACK  (Phase 3 — report_id + date + page X of Y)
# =========================================================
def _make_footer(report_id: str, gen_date: str):
    """Return an onPage callback for use with doc.build(onFirstPage=..., onLaterPages=...)."""
    from reportlab.lib.units import inch

    def draw_footer(canvas, doc):
        canvas.saveState()

        canvas.setFont("Helvetica-Bold", 7)
        canvas.setFillColor(INK_LIGHT)
        canvas.drawCentredString(
            PAGE_W / 2.0, MARGIN / 2 + 9,
            "Generated by InfraGuard Enterprise AI  •  Confidential Internal Document",
        )

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(INK_LIGHT)
        footer_text = f"{report_id}   |   {gen_date}   |   Page {doc.page}"
        canvas.drawCentredString(PAGE_W / 2.0, MARGIN / 2, footer_text)

        canvas.restoreState()

    return draw_footer


# =========================================================
# PRIMARY PUBLIC API
# =========================================================
def generate_quality_pdf(report_data: dict, output_path: str) -> None:
    """
    Generate a Construction Quality Inspection Report PDF.

    Parameters
    ----------
    report_data : dict
        Output of report_generator.generate_multi_image_report().
    output_path : str
        Destination file path.
    """
    report_id = report_data.get("report_id", "INF-QA-2026-001")
    gen_date  = report_data.get("generated_date",
                                 datetime.now().strftime("%d-%b-%Y"))

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=MARGIN,
        leftMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN + 10,
        # PDF Metadata (Phase 3)
        title="InfraGuard Quality Inspection Report",
        author="InfraGuard Enterprise AI",
        subject="Construction Quality Assurance",
        creator="InfraGuard Platform",
    )

    ST       = _styles()
    elements = []

    # ── Priority 1: Cover Page ─────────────────────────────
    _cover_page(report_data, ST, elements)

    # ── Table of Contents ──────────────────────────────────
    _table_of_contents(report_data, ST, elements)

    # ── Title block ───────────────────────────────────────
    _title_block(report_data, ST, elements)

    # ── Priority 2: Inspection Metadata Section ───────────
    _inspection_metadata(report_data, ST, elements)

    # ── Executive Dashboard ───────────────────────────────
    _executive_dashboard(report_data, ST, elements)

    # ── Site Health Index gauge ───────────────────────────
    _site_health_index(report_data, ST, elements)

    # ── Priority 4: Executive Risk Summary ────────────────
    _executive_risk_summary(report_data, ST, elements)

    # ── Section 1: Executive Summary ──────────────────────
    _executive_summary(report_data, ST, elements, sec_num=1)
    sec = 2

    # ── Section 2: Executive Narrative (Priority 4) ───────
    if report_data.get("site_executive_narrative"):
        _executive_narrative(report_data, ST, elements, sec_num=sec)
        sec += 1

    # ── Section: AI Findings ──────────────────────────────
    _ai_findings(report_data, ST, elements, sec_num=sec)
    sec += 1

    # ── Inspection Intelligence ───────────────────────────
    if any(report_data.get(k) for k in (
        "dominant_issue", "site_pattern_analysis",
        "systemic_deficiencies", "site_intelligence_summary",
    )):
        _inspection_intelligence(report_data, sec, ST, elements)
        sec += 1

    # ── Recurring Issue Analysis (incl. recurrence badge) ─
    if report_data.get("dominant_issue") or report_data.get("pattern_narrative"):
        _recurring_issue_analysis(report_data, sec, ST, elements)
        sec += 1

    # ── Management Attention ──────────────────────────────
    if report_data.get("management_attention_areas"):
        _management_attention(report_data, sec, ST, elements)
        sec += 1

    # ── Risk Matrix ───────────────────────────────────────
    if report_data.get("risk_matrix"):
        _risk_matrix(report_data, sec, ST, elements)
        sec += 1

    # ── Priority 6: Department Exposure Dashboard ─────────
    _department_exposure(report_data, sec, ST, elements)
    sec += 1

    # ── Site Intelligence Dashboard ───────────────────────
    if report_data.get("analytics") or report_data.get("dominant_issue"):
        _site_intelligence_dashboard(report_data, sec, ST, elements)
        sec += 1

    # ── Sections: per-image blocks ────────────────────────
    image_reports = report_data.get("image_reports", [])
    n_images      = len(image_reports)

    for idx, img in enumerate(image_reports):
        if n_images > 1:
            label = img.get("image_label", f"Image {idx + 1}")
            elements.append(Paragraph(
                f"<b>Image {idx + 1}: {label}</b>",
                ST["section_heading"],
            ))

        _original_image(img,     sec,     ST, elements)
        _annotated_image(img,    sec + 1, ST, elements)
        _findings_summary_table(img, sec + 2, ST, elements)
        _detailed_issue_guidance(img, sec + 3, ST, elements)

        if n_images > 1 and idx < n_images - 1:
            _page_break(elements)
        sec += 4

    # ── Audit Readiness ───────────────────────────────────
    _audit_readiness_section(report_data, sec, ST, elements)
    sec += 1

    # ── Priority 3: Severity Summary (shown before charts) ─
    _severity_summary(report_data, ST, elements)

    # ── Charts ────────────────────────────────────────────
    _charts_section(report_data, sec, ST, elements)
    sec += 1

    # ── Compliance Matrix ─────────────────────────────────
    _compliance_matrix(report_data, sec, ST, elements)
    sec += 1

    # ── Finding Closure Matrix ────────────────────────────
    _finding_closure_matrix(report_data, sec, ST, elements)
    sec += 1

    # ── Priority 5: Responsible Team Matrix ───────────────
    _responsible_team_matrix(report_data, sec, ST, elements)
    sec += 1

    # ── Audit Exposure ────────────────────────────────────
    if report_data.get("audit_exposure_summary"):
        _audit_exposure(report_data, sec, ST, elements)
        sec += 1

    # ── Priority 8: Executive Recommendations ─────────────
    _executive_recommendations(report_data, sec, ST, elements)
    sec += 1

    # ── Follow-Up Action Matrix ───────────────────────────
    _followup_matrix(report_data, sec, ST, elements)
    sec += 1

    # ── Conclusion ────────────────────────────────────────
    _conclusion(report_data, sec, ST, elements)

    # ── Priority 9: Report Readiness Certificate ──────────
    _readiness_certificate(report_data, ST, elements)

    # ── Executive Decision Block ──────────────────────────
    _executive_decision(report_data, ST, elements)

    # ── Executive Sign-Off Page ───────────────────────────
    _executive_signoff(report_data, ST, elements)

    # ── Priority 10: Appendix (last page) ─────────────────
    _appendix(report_data, ST, elements)

    # ── Build with footer callback ────────────────────────
    footer_cb = _make_footer(report_id, gen_date)
    doc.build(elements, onFirstPage=footer_cb, onLaterPages=footer_cb)


# =========================================================
# BACKWARD-COMPATIBLE WRAPPER  (single-image callers)
# =========================================================
def generate_quality_pdf_legacy(
    data: dict,
    image_path: str,
    annotated_image_path: str,
    output_path: str,
) -> None:
    """
    Thin shim for v1 single-image callers.
    Wraps payload into the multi-image schema and delegates.
    """
    findings = data.get("report", [])

    bd = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for f in findings:
        sev = f.get("severity", "Medium").capitalize()
        if sev in bd:
            bd[sev] += 1

    total  = len(findings)
    score  = data.get("compliance_score", 0)
    status = data.get("overall_status", "—")
    risk   = data.get("overall_risk", "—")

    issue_names = []
    seen        = set()
    for f in findings:
        it = f.get("issue_type", "")
        if it and it not in seen:
            seen.add(it)
            issue_names.append(it.replace("_", " ").lower())
    issues_str = ", ".join(issue_names) if issue_names else "various conditions"

    site_summary = (
        f"The site condition is currently {status.lower()} due to critical "
        f"visible findings involving {issues_str}."
    )

    _PRIORITY_MAP = {
        "Critical": (
            "Halt operations immediately. Engage qualified engineering personnel "
            "to assess and rectify all critical findings before work may resume."
        ),
        "High": (
            "Rectify all critical findings immediately and secure unsafe work "
            "zones if required."
        ),
        "Medium": (
            "Schedule corrective actions for all identified findings within "
            "the next operational period."
        ),
        "Low": (
            "Monitor and address findings during routine inspection and "
            "maintenance cycles."
        ),
    }
    priority_action = _PRIORITY_MAP.get(
        risk,
        "Rectify all findings as soon as practicable.",
    )

    ai_bullets = [
        f"A total of {total} visible quality issue(s) were identified in "
        f"the inspected image.",
        f"The most notable observed conditions include: {issues_str}.",
        "Corrective action and preventive controls are recommended to "
        "improve site discipline.",
    ]

    img_rpt = {
        "image_index":          1,
        "image_label":          "Image 1",
        "image_path":           image_path or "",
        "annotated_image_path": annotated_image_path or "",
        "findings":             findings,
        "total_findings":       total,
        "severity_breakdown":   bd,
        "compliance_score":     score,
        "compliance_status":    status,
        "risk_level":           risk,
        "inspection_grade":     data.get("inspection_grade", "—"),
        "corrective_actions":   list({f.get("corrective_action", "")
                                      for f in findings}),
        "preventive_actions":   list({f.get("preventive_action", "")
                                      for f in findings}),
        "best_practices":       list({f.get("best_practice", "")
                                      for f in findings}),
    }

    compat = {
        "overall_status":       status,
        "compliance_score":     score,
        "total_findings":       total,
        "site_summary":         site_summary,
        "priority_action":      priority_action,
        "ai_findings_bullets":  ai_bullets,
        "image_reports":        [img_rpt],
        "conclusion": (
            "This AI-assisted construction quality inspection report is "
            "intended to support site supervisors, safety engineers, and "
            "project teams in identifying visible quality concerns and "
            "improving overall site discipline through structured corrective "
            "and preventive action."
        ),
    }

    generate_quality_pdf(compat, output_path)