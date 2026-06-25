import logging
import os
import uuid
from collections import Counter
from datetime import datetime
from enum import Enum
from functools import lru_cache

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# =========================================================
# ENTERPRISE THRESHOLDS  (Priority 1)
# =========================================================
# Single source of truth for all score-based decisions.
# Adjust here and every downstream function updates automatically.
# =========================================================
ENTERPRISE_THRESHOLDS: dict[str, int] = {
    # Audit / readiness gates
    "audit_ready":          90,
    "conditional":          75,
    "corrective":           60,
    # Operational status gates
    "operationally_stable": 90,
    "conditionally_stable": 75,
    "corrective_attention": 60,
    # Benchmark gates
    "enterprise_grade":     90,
    "industry_acceptable":  75,
    "below_standard":       60,
    # Health rating gates
    "excellent":            90,
    "good":                 80,
    "needs_improvement":    65,
    # Grade gates
    "grade_a_plus":         95,
    "grade_a":              90,
    "grade_b":              80,
    "grade_c":              70,
    # Risk classification gates
    "risk_low":             85,
    "risk_moderate":        70,
    "risk_high":            50,
}

# =========================================================
# SEVERITY CONSTANTS  (Priority 2)
# =========================================================
SEVERITY_CRITICAL = "critical"
SEVERITY_HIGH     = "high"
SEVERITY_MEDIUM   = "medium"
SEVERITY_LOW      = "low"

# =========================================================
# ANALYTICS THRESHOLDS  (Priority 4)
# =========================================================
CONFIDENCE_ENTERPRISE = 0.85   # >= this → enterprise verified
CONFIDENCE_MODERATE   = 0.70   # >= this → moderate confidence (else review_required)

SYSTEMIC_THRESHOLD  = 7   # dominant issue count for "Systemic" recurrence
FREQUENT_THRESHOLD  = 4   # dominant issue count for "Frequent" recurrence
RECURRING_THRESHOLD = 2   # dominant issue count for "Recurring" recurrence

# =========================================================
# INSPECTION GRADE ENUM  (Priority 5)
# =========================================================

class InspectionGrade(str, Enum):
    A_PLUS = "A+"
    A      = "A"
    B      = "B"
    C      = "C"
    D      = "D"

from backend.services.quality.report_generator import (
    generate_report,
    SEVERITY_SCORE_DEDUCTIONS,
)

from backend.services.quality.llm_service import (
    generate_llm_summary,
)


# =========================================================
# ISSUE CLASSIFICATION CLUSTERS
# =========================================================
# Used by analytics_summary() to detect whether findings
# are concentrated in structural, safety, or mixed domains.
# =========================================================

_STRUCTURAL_ISSUES = {
    "surface_crack",
    "corrosion",
    "water_leakage",
    "rebar_exposure",
    "material_damage",
}

_SAFETY_ISSUES = {
    "poor_housekeeping",
    "ppe_non_compliance",
}

_ISSUE_DISPLAY_NAMES = {
    "surface_crack":      "Surface Cracking",
    "corrosion":          "Corrosion",
    "water_leakage":      "Water Leakage",
    "rebar_exposure":     "Rebar Exposure",
    "material_damage":    "Material Damage",
    "poor_housekeeping":  "Housekeeping Deficiencies",
    "ppe_non_compliance": "PPE Non-Compliance",
}


@lru_cache(maxsize=64)
def _display_name(issue_type: str) -> str:
    return _ISSUE_DISPLAY_NAMES.get(
        issue_type,
        issue_type.replace("_", " ").title(),
    )


# =========================================================
# RISK CLASSIFICATION
# =========================================================

def classify_risk(score: int, severity_counts: Counter = None) -> str:
    """
    Classify overall risk using both the numeric compliance score
    and the severity distribution.

    A score-only approach can mask a single critical finding inside
    an otherwise clean result.  Distribution-aware escalation rules:

      - Any critical finding  → risk cannot be lower than High
      - 3 + high findings     → risk cannot be lower than High
      - 1–2 high findings     → risk cannot be lower than Medium

    These rules only escalate; they never downgrade a score-derived
    classification that is already worse.
    """

    # Score-derived baseline
    if score >= ENTERPRISE_THRESHOLDS["risk_low"]:
        base = "Low"
    elif score >= ENTERPRISE_THRESHOLDS["risk_moderate"]:
        base = "Moderate"
    elif score >= ENTERPRISE_THRESHOLDS["risk_high"]:
        base = "High"
    else:
        base = "Critical"

    if not severity_counts:
        return base

    _rank = {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}
    _rev  = {v: k for k, v in _rank.items()}

    escalated_rank = _rank[base]

    critical_count = severity_counts.get(SEVERITY_CRITICAL, 0)
    high_count     = severity_counts.get(SEVERITY_HIGH,     0)

    if critical_count > 0:
        escalated_rank = max(escalated_rank, _rank["High"])

    if high_count >= 3:
        escalated_rank = max(escalated_rank, _rank["High"])
    elif high_count >= 1:
        escalated_rank = max(escalated_rank, _rank["Moderate"])

    return _rev[escalated_rank]


# =========================================================
# INSPECTION GRADE
# =========================================================

def inspection_grade(score: int) -> InspectionGrade:
    if score >= ENTERPRISE_THRESHOLDS["grade_a_plus"]:
        return InspectionGrade.A_PLUS
    if score >= ENTERPRISE_THRESHOLDS["grade_a"]:
        return InspectionGrade.A
    if score >= ENTERPRISE_THRESHOLDS["grade_b"]:
        return InspectionGrade.B
    if score >= ENTERPRISE_THRESHOLDS["grade_c"]:
        return InspectionGrade.C
    return InspectionGrade.D


# =========================================================
# OPERATIONAL STATUS
# =========================================================

def operational_status(score: int, severity_counts: Counter = None) -> str:
    """
    Returns an operational status label.  When severity counts
    are provided, critical findings override the score-based
    label to avoid misleading "Conditionally Stable" for
    inspections that contain critical structural issues.
    """

    critical_count = (severity_counts or {}).get(SEVERITY_CRITICAL, 0)

    if critical_count > 0:
        return "Critical Intervention Required"

    if score >= ENTERPRISE_THRESHOLDS["operationally_stable"]:
        return "Operationally Stable"
    if score >= ENTERPRISE_THRESHOLDS["conditionally_stable"]:
        return "Conditionally Stable"
    if score >= ENTERPRISE_THRESHOLDS["corrective_attention"]:
        return "Corrective Attention Required"
    return "Critical Intervention Required"


# =========================================================
# PRIORITY ACTION
# =========================================================

def priority_action(
    severity_counts: Counter,
    report: list = None,
) -> str:
    """
    Returns a specific priority action statement referencing
    the actual issue types found at the dominant severity level.

    Parameters
    ----------
    severity_counts : Counter
        Severity counts keyed by lowercase severity name.
    report : list, optional
        Full finding list.  When supplied, issue type names
        are extracted to produce specific rather than generic
        action text.
    """

    def _extract_issues_at_severity(sev_target: str) -> list:
        """Return unique display names for findings at sev_target."""
        if not report:
            return []
        seen   = {}
        result = []
        for item in report:
            if item.get("severity", "").lower() == sev_target:
                it = item.get("issue_type", "")
                if it and it not in seen:
                    seen[it] = True
                    result.append(_display_name(it))
        return result

    def _issue_clause(issue_names: list) -> str:
        if not issue_names:
            return ""
        if len(issue_names) == 1:
            return f" ({issue_names[0]})"
        if len(issue_names) == 2:
            return f" ({issue_names[0]} and {issue_names[1]})"
        preview = ", ".join(issue_names[:2])
        return f" ({preview}, and {len(issue_names) - 2} more)"

    critical_count = severity_counts.get(SEVERITY_CRITICAL, 0)
    high_count     = severity_counts.get(SEVERITY_HIGH,     0)
    medium_count   = severity_counts.get(SEVERITY_MEDIUM,   0)

    if critical_count > 0:
        clause = _issue_clause(_extract_issues_at_severity(SEVERITY_CRITICAL))
        return (
            f"Immediate engineering review and executive corrective enforcement "
            f"required for {critical_count} critical finding"
            f"{'s' if critical_count != 1 else ''}{clause}."
        )

    if high_count > 0:
        clause = _issue_clause(_extract_issues_at_severity(SEVERITY_HIGH))
        return (
            f"Accelerated corrective workflows required for {high_count} "
            f"high-priority finding{'s' if high_count != 1 else ''}{clause}. "
            f"Engineering supervision and preventive operational controls are recommended."
        )

    if medium_count > 0:
        clause = _issue_clause(_extract_issues_at_severity(SEVERITY_MEDIUM))
        return (
            f"Preventive maintenance and structured inspection monitoring are "
            f"recommended for {medium_count} moderate finding"
            f"{'s' if medium_count != 1 else ''}{clause}."
        )

    return (
        "No significant deviations detected. Continue standard operational "
        "inspection workflows and routine quality monitoring procedures."
    )


# =========================================================
# AUDIT READINESS
# =========================================================

def audit_readiness(
    score: int,
    severity_counts: Counter = None,
) -> str:
    """
    Determines audit readiness using both the compliance score
    and the presence of critical findings.

    Any critical finding blocks 'Audit Ready' regardless of
    score, because critical conditions represent unresolved
    safety or structural risk that no numerical average can mask.
    """

    critical_count = (severity_counts or {}).get(SEVERITY_CRITICAL, 0)
    high_count     = (severity_counts or {}).get(SEVERITY_HIGH,     0)

    if critical_count > 0:
        return (
            f"Not Audit Ready — "
            f"{critical_count} Critical Finding"
            f"{'s' if critical_count != 1 else ''} "
            f"Require Immediate Resolution"
        )

    if score >= ENTERPRISE_THRESHOLDS["audit_ready"] and high_count == 0:
        return "Audit Ready"

    if score >= ENTERPRISE_THRESHOLDS["conditional"]:
        if high_count > 0:
            return (
                f"Conditionally Audit Ready — "
                f"{high_count} High-Priority Finding"
                f"{'s' if high_count != 1 else ''} Pending Corrective Action"
            )
        return "Conditionally Audit Ready — Minor Corrective Actions Pending"

    if score >= ENTERPRISE_THRESHOLDS["corrective"]:
        return "Moderate Compliance Gaps — Corrective Plan Required Before Audit"

    return "Audit Risk Detected — Significant Remediation Required"


# =========================================================
# COMPLIANCE BENCHMARK
# =========================================================

def benchmark(score: int, severity_counts: Counter = None) -> str:
    """
    Classifies the inspection result against enterprise
    compliance benchmarks.  A critical finding presence
    prevents an 'Enterprise Grade' classification.
    """

    critical_count = (severity_counts or {}).get(SEVERITY_CRITICAL, 0)

    if score >= ENTERPRISE_THRESHOLDS["enterprise_grade"] and critical_count == 0:
        return "Enterprise Grade"

    if score >= ENTERPRISE_THRESHOLDS["enterprise_grade"] and critical_count > 0:
        return "Below Enterprise Grade — Critical Findings Present"

    if score >= ENTERPRISE_THRESHOLDS["industry_acceptable"]:
        return "Industry Acceptable"

    if score >= ENTERPRISE_THRESHOLDS["below_standard"]:
        return "Below Recommended Standard"

    return "Critical Compliance Deviation"


# =========================================================
# ANALYTICS
# =========================================================

def analytics_summary(report: list) -> dict:
    """
    Produces enriched analytics from the finding list,
    including:

    - Standard severity and category counts
    - Confidence tier breakdown (high / moderate / low)
    - Structural vs safety issue cluster detection
    - Dominant severity narrative for UI context
    """

    severity_counter    = Counter()
    categories          = Counter()
    issue_types         = Counter()
    confidence_sum      = 0.0
    conf_high           = 0   # confidence >= 0.85
    conf_moderate       = 0   # 0.70 <= confidence < 0.85
    conf_low            = 0   # confidence < 0.70

    for item in report:

        sev = item.get("severity", SEVERITY_MEDIUM).lower()
        severity_counter[sev] += 1

        cat = item.get("category", "General Construction Quality")
        categories[cat] += 1

        it = item.get("issue_type", "unknown")
        issue_types[it] += 1

        conf = float(item.get("confidence", 0))
        confidence_sum += conf

        if conf >= CONFIDENCE_ENTERPRISE:
            conf_high     += 1
        elif conf >= CONFIDENCE_MODERATE:
            conf_moderate += 1
        else:
            conf_low      += 1

    total = len(report)

    avg_confidence = round(confidence_sum / total, 2) if total > 0 else 0.0

    dominant_category = (
        categories.most_common(1)[0][0]
        if categories else
        "General Construction Quality"
    )

    # ── Cluster detection ─────────────────────────────────
    structural_count = sum(
        issue_types.get(it, 0) for it in _STRUCTURAL_ISSUES
    )
    safety_count = sum(
        issue_types.get(it, 0) for it in _SAFETY_ISSUES
    )

    if structural_count > 0 and safety_count > 0:
        issue_cluster = "Mixed — Structural and Safety Findings"
    elif structural_count > 0:
        issue_cluster = "Structural Integrity Focus"
    elif safety_count > 0:
        issue_cluster = "Workforce Safety Focus"
    else:
        issue_cluster = "General Construction Quality"

    # ── Dominant severity narrative ───────────────────────
    if severity_counter.get(SEVERITY_CRITICAL, 0) > 0:
        dominant_severity_narrative = (
            "Critical findings are present and require immediate attention."
        )
    elif severity_counter.get(SEVERITY_HIGH, 0) > 0:
        dominant_severity_narrative = (
            "High-priority findings are driving the primary risk exposure."
        )
    elif severity_counter.get(SEVERITY_MEDIUM, 0) > 0:
        dominant_severity_narrative = (
            "Moderate findings require structured corrective follow-up."
        )
    elif total > 0:
        dominant_severity_narrative = (
            "Only low-risk observations were identified."
        )
    else:
        dominant_severity_narrative = (
            "No significant findings were detected."
        )

    # ── Dominant issue analysis ───────────────────────────
    dominant_issue       = None
    dominant_issue_count = 0

    if issue_types:
        dominant_issue, dominant_issue_count = issue_types.most_common(1)[0]

    # ── Confidence profile ────────────────────────────────
    confidence_profile = {
        "enterprise_verified": conf_high,
        "moderate_confidence": conf_moderate,
        "review_required":     conf_low,
    }

    # ── Recurrence level classification ──────────────────
    if dominant_issue_count >= SYSTEMIC_THRESHOLD:
        recurrence_level = "Systemic"
    elif dominant_issue_count >= FREQUENT_THRESHOLD:
        recurrence_level = "Frequent"
    elif dominant_issue_count >= RECURRING_THRESHOLD:
        recurrence_level = "Recurring"
    else:
        recurrence_level = "Isolated"

    # ── Pattern narrative ─────────────────────────────────
    pattern_narrative = ""
    if dominant_issue_count >= 3:
        pattern_narrative = (
            f"{_display_name(dominant_issue)} represents the dominant "
            f"recurring issue across the inspection."
        )

    # ── Site risk hotspot ─────────────────────────────────
    site_hotspot = dominant_category

    # ── Management attention flag ─────────────────────────
    management_attention_required = (
        severity_counter.get(SEVERITY_CRITICAL, 0) > 0
        or severity_counter.get(SEVERITY_HIGH, 0) >= 3
    )

    return {

        # Standard counters
        "total_findings":           total,
        "critical_findings":        severity_counter.get(SEVERITY_CRITICAL, 0),
        "high_findings":            severity_counter.get(SEVERITY_HIGH,     0),
        "medium_findings":          severity_counter.get(SEVERITY_MEDIUM,   0),
        "low_findings":             severity_counter.get(SEVERITY_LOW,      0),

        # Confidence intelligence
        "average_ai_confidence":    avg_confidence,
        "high_confidence_findings": conf_high,
        "moderate_confidence_findings": conf_moderate,
        "low_confidence_findings":  conf_low,
        "confidence_profile":       confidence_profile,

        # Pattern intelligence
        "dominant_category":        dominant_category,
        "issue_cluster":            issue_cluster,
        "dominant_severity_narrative": dominant_severity_narrative,
        "dominant_issue":           dominant_issue,
        "dominant_issue_display":   (
            _display_name(dominant_issue) if dominant_issue else "None"
        ),
        "dominant_issue_count":     dominant_issue_count,
        "issue_distribution":       dict(issue_types),
        "pattern_narrative":        pattern_narrative,
        "site_hotspot":             site_hotspot,

        # Management intelligence
        "management_attention_required": management_attention_required,

        # Recurrence intelligence
        "recurrence_level":              recurrence_level,
    }



# =========================================================
# DEPARTMENT EXPOSURE SUMMARY
# =========================================================

_DEPARTMENT_MAP = {
    "surface_crack":      "Engineering",
    "rebar_exposure":     "Engineering",
    "water_leakage":      "Engineering",
    "corrosion":          "Maintenance",
    "material_damage":    "Quality",
    "poor_housekeeping":  "Operations",
    "ppe_non_compliance": "Safety",
}


def department_exposure(report: list) -> dict:
    """
    Maps each finding's issue type to a responsible department
    and returns a count per department.

    Used by dashboard analytics and future executive views to
    surface which functional teams carry the highest exposure.

    Returns
    -------
    dict  e.g. {"Engineering": 5, "Quality": 3, "Safety": 2}
    """
    exposure: Counter = Counter()

    for item in report:
        it   = item.get("issue_type", "")
        dept = _DEPARTMENT_MAP.get(it, "General")
        exposure[dept] += 1

    return dict(exposure)


# =========================================================
# MANAGEMENT ESCALATION FLAG
# =========================================================

def management_escalation_required(severity_counts: Counter) -> bool:
    """
    Returns True when the finding profile warrants executive attention.

    Escalation triggers:
      - Any critical finding present
      - Three or more high-priority findings
    """
    if severity_counts.get(SEVERITY_CRITICAL, 0) > 0:
        return True
    if severity_counts.get(SEVERITY_HIGH, 0) >= 3:
        return True
    return False


# =========================================================
# INSPECTION HEALTH RATING
# =========================================================

def inspection_health_rating(score: int) -> str:
    """
    Single-word management KPI derived from the compliance score.

    Thresholds
    ----------
    >= 90  → Excellent
    >= 80  → Good
    >= 65  → Needs Improvement
    <  65  → Critical
    """
    if score >= ENTERPRISE_THRESHOLDS["excellent"]:
        return "Excellent"
    if score >= ENTERPRISE_THRESHOLDS["good"]:
        return "Good"
    if score >= ENTERPRISE_THRESHOLDS["needs_improvement"]:
        return "Needs Improvement"
    return "Critical"


# =========================================================
# EXECUTIVE INSIGHT GENERATOR
# =========================================================

def executive_insight(analytics: dict) -> str:
    """
    Produces a single executive-facing insight sentence from the
    analytics payload already produced by analytics_summary().

    Prioritises the dominant issue when a pattern has been
    identified; falls back to a cluster-level statement otherwise.
    Singular/plural is handled correctly when exactly one issue type
    is referenced.
    """
    dominant_display = analytics.get("dominant_issue_display", "None")
    dominant_count   = analytics.get("dominant_issue_count",   0)
    issue_cluster    = analytics.get("issue_cluster",          "General Construction Quality")

    if dominant_display and dominant_display != "None" and dominant_count >= RECURRING_THRESHOLD:
        verb = "represents" if dominant_count == 1 else "represent"
        return (
            f"{dominant_display} {verb} the dominant recurring issue "
            f"across the inspected site and should be prioritised for "
            f"corrective action."
        )

    return (
        f"The inspection identified a {issue_cluster.lower()} pattern "
        f"that warrants targeted corrective review across affected work areas."
    )


# =========================================================
# SCORING ENGINE
# =========================================================

def calculate_score(report: list) -> tuple[int, Counter]:
    """
    Compute the compliance score from a processed finding list.

    Uses the same weights, confidence factor, and repeat-dampening
    logic as report_generator.compute_compliance_score() so the
    analysis panel and PDF report always show consistent scores.

    Returns
    -------
    (score: int, severity_counts: Counter)
    """

    score            = 100.0
    severity_counter = Counter()
    issue_tally      = Counter()

    for item in report:

        severity   = item.get("severity",   SEVERITY_MEDIUM).lower()
        confidence = float(item.get("confidence", 0.75))
        issue_type = item.get("issue_type", "unknown")

        severity_counter[severity] += 1

        base_penalty = SEVERITY_SCORE_DEDUCTIONS.get(severity, 5)

        # Confidence weighting: [0.65, 1.0]
        conf_factor = 0.65 + (0.35 * min(max(confidence, 0.0), 1.0))

        # Repeat dampening: > 2 occurrences of same issue → 70 %
        issue_tally[issue_type] += 1
        repeat_factor = 1.0 if issue_tally[issue_type] <= 2 else 0.70

        score -= base_penalty * conf_factor * repeat_factor

    score = max(0, min(round(score), 100))

    return score, severity_counter


# =========================================================
# EXECUTIVE RISK INDEX
# =========================================================

def executive_risk_index(
    compliance_score: int,
    severity_counts: Counter,
) -> int:
    """
    Produces a single executive-facing risk index (0-100, higher
    is worse) derived from the compliance score and weighted by
    critical/high severity counts.
    """

    risk = 100 - compliance_score

    risk += severity_counts.get(SEVERITY_CRITICAL, 0) * 10
    risk += severity_counts.get(SEVERITY_HIGH, 0) * 5

    return max(0, min(100, risk))


# =========================================================
# SITE INTELLIGENCE SUMMARY
# =========================================================

def site_intelligence_summary(
    analytics: dict,
    compliance_score: int,
) -> str:
    """
    Produces a concise executive-facing narrative combining the
    issue cluster and dominant category intelligence already
    computed by analytics_summary().
    """

    return (
        f"The inspection indicates "
        f"{analytics['issue_cluster'].lower()} "
        f"with primary exposure arising from "
        f"{analytics['dominant_category'].lower()} findings."
    )


# =========================================================
# AI FINDINGS
# =========================================================

def build_ai_findings(report: list) -> list:
    """
    Reshape the processed finding list into a concise AI
    findings structure consumed by the frontend findings panel.
    """

    findings = []

    for item in report:
        findings.append({
            "issue":      item.get("issue_type",        "unknown"),
            "label":      _display_name(item.get("issue_type", "unknown")),
            "severity":   item.get("severity",          "Medium"),
            "risk":       item.get("risk",              ""),
            "risk_category": item.get("risk_category",  "C"),
            "impact":     item.get("operational_impact", ""),
            "confidence": item.get("confidence",        0.0),
            "observation": item.get("observation",      ""),
        })

    return findings


# =========================================================
# ANNOTATED EVIDENCE IMAGE GENERATION
# =========================================================
# Renders a branded HUD-style evidence overlay onto a copy of the
# source image and persists it to disk:
#
#   - A top title bar: "InfraGuard AI — <inspection title>" with a
#     live "Detected Findings: N" counter
#   - Numbered, severity-colored bounding boxes
#     (Critical = red, High = orange, Medium = yellow, Low = green)
#   - A two-line label above each box: "{n}. {Issue}" followed by
#     "{Severity} | Conf. {confidence} | {Category}"
#   - A bottom-left severity legend matching the box colors
#   - A bottom-right capture timestamp for chain-of-custody
#
# This is purely additive: failures here are swallowed so the
# core analysis pipeline (scoring, analytics, executive summary,
# AI findings, report data) is never affected.
# =========================================================

ANNOTATED_IMAGE_OUTPUT_DIR = os.environ.get(
    "ANNOTATED_IMAGE_OUTPUT_DIR",
    os.path.join(os.getcwd(), "media", "annotated_evidence"),
)

DEFAULT_INSPECTION_TITLE = "Infrastructure Quality Inspection"

# ── Brand palette (BGR) ─────────────────────────────────────────────────────
# Title-bar background: very dark with a faint warm brownish tint
# matching the reference screenshot.
_BRAND_DARK_BG_BGR  = (18, 12, 22)        # dark warm near-black      (#160c12)
_BRAND_AMBER_BGR    = (0,  176, 255)       # findings counter / accent (#ffb000)
_BRAND_WHITE_BGR    = (245, 245, 245)
_BRAND_MUTED_BGR    = (185, 185, 185)

# ── Severity → BGR color map  (shared by boxes, labels, and legend) ──────────
_SEVERITY_COLORS_BGR = {
    "critical": (0,   0, 215),    # red     #d70000
    "high":     (0, 130, 255),    # orange  #ff8200
    "medium":   (0, 210, 255),    # yellow  #ffd200
    "low":      (60, 200,  60),   # green   #3cc83c
}
_SEVERITY_DEFAULT_COLOR_BGR = (160, 160, 160)
_SEVERITY_LEGEND_ORDER      = ["critical", "high", "medium", "low"]

_BANNER_FONT    = cv2.FONT_HERSHEY_DUPLEX
_LABEL_FONT     = cv2.FONT_HERSHEY_DUPLEX
_TIMESTAMP_FONT = cv2.FONT_HERSHEY_SIMPLEX

_TIMESTAMP_FONT_SCALE = 0.50
_TIMESTAMP_THICKNESS  = 1
_TIMESTAMP_TEXT_COLOR = _BRAND_WHITE_BGR


def _render_scale(width: int) -> float:
    """Linear scale factor relative to a 1280-px reference frame."""
    return max(0.55, min(width / 1280.0, 1.8))


def _severity_color(severity: str) -> tuple:
    """BGR color for a severity string (case-insensitive)."""
    return _SEVERITY_COLORS_BGR.get(
        (severity or "").lower(), _SEVERITY_DEFAULT_COLOR_BGR
    )


def _banner_height(height: int) -> int:
    """Title-bar pixel height, clamped for short frames."""
    return int(max(36, min(88, height * 0.075)))


def _to_cv2_image(image):
    """
    Normalises any supported image input into a BGR numpy array.
    Supported: ndarray, file path, raw bytes, file-like object, PIL image.
    Returns None when decoding fails.
    """
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, (str, os.PathLike)):
        return cv2.imread(str(image))
    if isinstance(image, (bytes, bytearray)):
        buf = np.frombuffer(image, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if hasattr(image, "read"):
        try:
            buf = np.frombuffer(image.read(), dtype=np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:
            return None
    try:
        arr = np.array(image)
        if arr.ndim == 3 and arr.shape[2] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr
    except Exception:
        return None


def _extract_bbox_coords(detection: dict):
    """
    Extracts (x1, y1, x2, y2) from a detection dict.
    Tolerates: list [x1,y1,x2,y2], dict {x1/y1/x2/y2}, {left/top/right/bottom},
    {x/y/width/height}, and alternate keys bounding_box / box.
    Returns None when no usable coordinates are found.
    """
    bbox = (
        detection.get("bbox")
        or detection.get("bounding_box")
        or detection.get("box")
    )
    if bbox is None:
        return None
    try:
        if isinstance(bbox, dict):
            if {"x1", "y1", "x2", "y2"} <= bbox.keys():
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            elif {"left", "top", "right", "bottom"} <= bbox.keys():
                x1, y1, x2, y2 = bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]
            elif {"x", "y", "width", "height"} <= bbox.keys():
                x1, y1 = bbox["x"], bbox["y"]
                x2, y2 = x1 + bbox["width"], y1 + bbox["height"]
            else:
                return None
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
        else:
            return None
        x1, y1, x2, y2 = (
            int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        )
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2
    except (TypeError, ValueError, KeyError):
        return None


def _overlay_rect(image_array, x1: int, y1: int, x2: int, y2: int,
                  color_bgr: tuple, alpha: float = 0.55) -> None:
    """
    Alpha-blends a solid filled rectangle into image_array in-place.
    Clips silently to frame boundaries so partial-OOB rects are safe.
    """
    h, w = image_array.shape[:2]
    rx1, ry1 = max(x1, 0), max(y1, 0)
    rx2, ry2 = min(x2, w), min(y2, h)
    if rx2 <= rx1 or ry2 <= ry1:
        return
    roi = image_array[ry1:ry2, rx1:rx2]
    overlay = roi.copy()
    overlay[:] = color_bgr
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


def _label_for_detection(detection: dict, report_item: dict = None) -> dict:
    """
    Builds the structured label dict consumed by _draw_box_label().
    Prefers the enriched report_item when present; falls back to raw
    detection fields.

    Returns: {issue, severity, confidence, category}
    """
    source     = report_item if report_item else detection
    issue_type = source.get("issue_type", detection.get("issue_type", "unknown"))
    severity   = str(source.get("severity", detection.get("severity", SEVERITY_MEDIUM))).lower()
    confidence = float(source.get("confidence", detection.get("confidence", 0.0)))
    category   = source.get("category", detection.get("category", ""))
    return {
        "issue":      _display_name(issue_type),
        "severity":   severity,
        "confidence": confidence,
        "category":   category,
    }


def _draw_box_label(image_array, label: dict, number: int,
                    x1: int, y1: int, x2: int, y2: int,
                    scale: float) -> None:
    """
    Renders the two-line label panel INSIDE the bounding box at its
    top edge, spanning from x1 to x2 (full box width):

      Line 1 (white, bold):    "{n}. {Issue Name}"
      Line 2 (severity color): "{Severity} | Conf. {0.00} | {Category}"

    The panel background is a dark semi-transparent rectangle anchored
    to the top-left corner of the box — identical to the reference style.
    No left accent bar is drawn.

    Parameters
    ----------
    x1, y1 : top-left corner of the bounding box
    x2, y2 : bottom-right corner of the bounding box
    scale   : _render_scale() value for this frame
    """
    h_img, w_img = image_array.shape[:2]

    # ── Font sizes ────────────────────────────────────────────────
    fs_main = max(0.38, 0.44 * scale)
    fs_sub  = max(0.28, 0.34 * scale)
    th_main = max(1, int(round(scale)))
    th_sub  = max(1, th_main)

    # ── Label strings ─────────────────────────────────────────────
    line1 = f"{number}. {label['issue']}"
    conf_str = f"{label['confidence']:.2f}"
    sev_str  = label["severity"].title()
    cat_str  = label["category"]
    meta_parts = [sev_str, f"Conf. {conf_str}"]
    if cat_str:
        meta_parts.append(cat_str)
    line2 = " | ".join(meta_parts)

    # ── Measure text ──────────────────────────────────────────────
    (w1, h1), bl1 = cv2.getTextSize(line1, _LABEL_FONT, fs_main, th_main)
    (w2, h2), bl2 = cv2.getTextSize(line2, _LABEL_FONT, fs_sub,  th_sub)

    pad_x = int(8  * scale)
    pad_y = int(6  * scale)
    gap   = int(4  * scale)

    panel_h = pad_y + h1 + gap + h2 + pad_y

    # ── Panel spans full width of the box ────────────────────────
    # Anchored at the top-left corner, inside the box.
    px1 = x1
    px2 = min(x2, w_img)
    py1 = y1
    py2 = min(y1 + panel_h, y2, h_img)   # never overflows the box or frame

    # ── Dark semi-transparent background ─────────────────────────
    _overlay_rect(image_array, px1, py1, px2, py2, _BRAND_DARK_BG_BGR, alpha=0.80)

    # ── Text ─────────────────────────────────────────────────────
    sev_color = _severity_color(label["severity"])
    text_x    = px1 + pad_x
    line1_y   = py1 + pad_y + h1
    line2_y   = line1_y + gap + h2

    cv2.putText(
        image_array, line1,
        (text_x, min(line1_y, h_img - 1)),
        _LABEL_FONT, fs_main, _BRAND_WHITE_BGR, th_main, cv2.LINE_AA,
    )
    cv2.putText(
        image_array, line2,
        (text_x, min(line2_y, h_img - 1)),
        _LABEL_FONT, fs_sub, sev_color, th_sub, cv2.LINE_AA,
    )


def _draw_title_banner(image_array, title: str, finding_count: int,
                       scale: float) -> int:
    """
    Renders the full-width title bar:

        "InfraGuard AI — {title}"          [white, left]
        "Detected Findings: {n}"           [amber, right]

    Bottom edge has an amber accent rule matching the reference.
    Returns the banner pixel height.
    """
    h_img, w_img = image_array.shape[:2]
    bh = _banner_height(h_img)

    # ── Banner background ─────────────────────────────────────────
    _overlay_rect(image_array, 0, 0, w_img, bh, _BRAND_DARK_BG_BGR, alpha=0.90)

    # ── Bottom amber accent rule ──────────────────────────────────
    rule_h = max(2, int(3 * scale))
    _overlay_rect(
        image_array, 0, bh - rule_h, w_img, bh,
        _BRAND_AMBER_BGR, alpha=0.95,
    )

    fs   = max(0.42, 0.58 * scale)
    th   = max(1, int(round(scale)))
    pad  = int(12 * scale)
    ty   = int(bh * 0.65)

    # ── Left: branded title ───────────────────────────────────────
    brand_text = f"InfraGuard AI  —  {title}"
    cv2.putText(
        image_array, brand_text,
        (pad, ty),
        _BANNER_FONT, fs, _BRAND_WHITE_BGR, th, cv2.LINE_AA,
    )

    # ── Right: findings counter (amber) ──────────────────────────
    counter_text = f"Detected Findings: {finding_count}"
    (cw, _), _ = cv2.getTextSize(counter_text, _BANNER_FONT, fs, th)
    cv2.putText(
        image_array, counter_text,
        (w_img - cw - pad, ty),
        _BANNER_FONT, fs, _BRAND_AMBER_BGR, th, cv2.LINE_AA,
    )

    return bh


def _draw_severity_legend(image_array, scale: float) -> None:
    """
    Renders the severity color legend in the bottom-left corner as a
    2-column grid matching the reference annotation style:

        ■ Critical    ■ High
        ■ Medium      ■ Low
    """
    h_img, w_img = image_array.shape[:2]

    fs      = max(0.28, 0.36 * scale)
    th      = max(1, int(round(scale * 0.85)))
    pad     = int(10 * scale)
    swatch  = int(11 * scale)
    gap_x   = int(8  * scale)   # gap between swatch and label
    col_gap = int(24 * scale)   # gap between the two columns
    row_h   = int(20 * scale)

    # Legend entries in display order, left-col then right-col
    left_col  = ["critical", "medium"]
    right_col = ["high",     "low"]
    rows      = list(zip(left_col, right_col))   # 2 rows × 2 cols

    # Measure widest label in each column
    def text_w(s):
        return cv2.getTextSize(s.title(), _LABEL_FONT, fs, th)[0][0]

    lw = max(text_w(s) for s in left_col)
    rw = max(text_w(s) for s in right_col)

    header      = "Severity Legend"
    (hw, hh), _ = cv2.getTextSize(header, _LABEL_FONT, fs, th)

    panel_w = (
        pad
        + swatch + gap_x + lw
        + col_gap
        + swatch + gap_x + rw
        + pad
    )
    panel_h = pad + hh + int(6 * scale) + row_h * len(rows) + pad

    px1 = pad
    py2 = h_img - pad
    py1 = py2 - panel_h
    px2 = px1 + panel_w

    # ── Panel background ──────────────────────────────────────────
    _overlay_rect(image_array, px1, py1, px2, py2, _BRAND_DARK_BG_BGR, alpha=0.80)

    # ── Header ───────────────────────────────────────────────────
    cv2.putText(
        image_array, header,
        (px1 + pad, py1 + pad + hh),
        _LABEL_FONT, fs, _BRAND_WHITE_BGR, th, cv2.LINE_AA,
    )

    row_start_y = py1 + pad + hh + int(6 * scale)

    # ── Left-column x-origin and right-column x-origin ───────────
    col_x = [
        px1 + pad,
        px1 + pad + swatch + gap_x + lw + col_gap,
    ]

    for row_i, (lsev, rsev) in enumerate(rows):
        ry = row_start_y + row_i * row_h

        for ci, sev in enumerate([lsev, rsev]):
            cx    = col_x[ci]
            color = _severity_color(sev)
            label = sev.title()

            # Colored swatch square
            cv2.rectangle(
                image_array,
                (cx, ry),
                (cx + swatch, ry + swatch),
                color,
                cv2.FILLED,
            )

            # Label text beside swatch
            (_, th_), _ = cv2.getTextSize(label, _LABEL_FONT, fs, th)
            cv2.putText(
                image_array, label,
                (cx + swatch + gap_x, ry + th_),
                _LABEL_FONT, fs, color, th, cv2.LINE_AA,
            )


def _draw_timestamp(image_array, timestamp_text: str, scale: float) -> None:
    """Compact timestamp badge in the bottom-right corner."""
    h_img, w_img = image_array.shape[:2]

    fs  = max(0.30, _TIMESTAMP_FONT_SCALE * scale)
    th  = _TIMESTAMP_THICKNESS
    pad = int(8 * scale)

    (tw, text_h), bl = cv2.getTextSize(
        timestamp_text, _TIMESTAMP_FONT, fs, th
    )

    rx2 = w_img - pad
    ry2 = h_img - pad
    rx1 = rx2 - tw - pad * 2
    ry1 = ry2 - text_h - bl - pad

    _overlay_rect(image_array, rx1, ry1, rx2, ry2, _BRAND_DARK_BG_BGR, alpha=0.72)
    cv2.putText(
        image_array, timestamp_text,
        (rx1 + pad, ry2 - bl - 2),
        _TIMESTAMP_FONT, fs, _TIMESTAMP_TEXT_COLOR, th, cv2.LINE_AA,
    )


def generate_annotated_evidence_image(
    image,
    detections: list,
    report: list = None,
    title: str = None,
):
    """
    Renders a reference-quality HUD evidence overlay onto a copy of the
    source image and persists it to disk as a JPEG file.

    Render order (back → front):
      1. Full-width title banner with findings counter
      2. Per-detection: subtle tinted box fill + severity-colored border
      3. Per-detection: two-line numbered label panel inside the box top
      4. Bottom-left two-column severity legend
      5. Bottom-right capture timestamp

    Parameters
    ----------
    image      : ndarray | str | PathLike | bytes | file-like | PIL.Image
    detections : raw detection dicts with bbox coordinates
    report     : enriched finding list from generate_report() (index-aligned)
    title      : inspection title for the banner; defaults to
                 DEFAULT_INSPECTION_TITLE

    Returns
    -------
    str or None — absolute path of saved JPEG, or None on failure.
    Failures are swallowed; the core analysis pipeline is never affected.
    """
    try:
        image_array = _to_cv2_image(image)
        if image_array is None:
            return None

        annotated = image_array.copy()
        h_img, w_img = annotated.shape[:2]
        scale = _render_scale(w_img)

        inspection_title   = title or DEFAULT_INSPECTION_TITLE
        finding_count      = sum(
            1 for d in (detections or [])
            if _extract_bbox_coords(d) is not None
        )
        report_by_index = (
            report if report and len(report) == len(detections) else None
        )

        # ── 1. Title banner ───────────────────────────────────────
        _draw_title_banner(annotated, inspection_title, finding_count, scale)

        # ── 2 & 3. Box fill, border, and label ───────────────────
        # Border thickness: 3 px at reference scale, minimum 2 px.
        box_thickness = max(2, int(round(3 * scale)))
        number = 0

        for idx, detection in enumerate(detections or []):
            coords = _extract_bbox_coords(detection)
            if coords is None:
                continue

            number += 1
            x1, y1, x2, y2 = coords

            report_item = report_by_index[idx] if report_by_index else None
            label       = _label_for_detection(detection, report_item)
            sev_color   = _severity_color(label["severity"])

            # Subtle tinted fill inside the box (matches reference look)
            _overlay_rect(
                annotated, x1 + 1, y1 + 1, x2 - 1, y2 - 1,
                sev_color, alpha=0.07,
            )

            # Severity-colored border
            cv2.rectangle(annotated, (x1, y1), (x2, y2), sev_color, box_thickness)

            # Two-line label panel inside the top of the box
            _draw_box_label(annotated, label, number, x1, y1, x2, y2, scale)

        # ── 4. Severity legend ────────────────────────────────────
        _draw_severity_legend(annotated, scale)

        # ── 5. Timestamp ──────────────────────────────────────────
        ts = f"Inspection Capture: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        _draw_timestamp(annotated, ts, scale)

        # ── Persist ───────────────────────────────────────────────
        os.makedirs(ANNOTATED_IMAGE_OUTPUT_DIR, exist_ok=True)
        filename    = f"annotated_{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(ANNOTATED_IMAGE_OUTPUT_DIR, filename)

        if not cv2.imwrite(output_path, annotated):
            return None

        return output_path

    except Exception:
        # Annotation is best-effort; never propagate to core pipeline.
        logger.exception("Failed to generate annotated evidence image")
        return None


def analyze_quality(
    image,
    detections: list,
) -> dict[str, object]:
    """
    Analyzes a single image against its detections.
    Called once per image by the route layer.
    Returns a self-contained per-image result dict.
    No cross-image merging occurs here.
    """

    # ── Report ────────────────────────────────────────────
    report = generate_report(detections)

    # ── Scoring ───────────────────────────────────────────
    compliance_score, severity_counts = calculate_score(report)

    # ── Analytics ─────────────────────────────────────────
    analytics = analytics_summary(report)

    # ── New analytics layer ───────────────────────────────
    dept_exposure   = department_exposure(report)
    escalation_flag = management_escalation_required(severity_counts)
    health_rating   = inspection_health_rating(compliance_score)
    insight         = executive_insight(analytics)

    # ── Executive risk index & site intelligence ──────────
    risk_index = executive_risk_index(
        compliance_score,
        severity_counts,
    )

    site_intelligence = site_intelligence_summary(
        analytics,
        compliance_score,
    )

    # ── Risk (distribution-aware) ─────────────────────────
    overall_risk = classify_risk(compliance_score, severity_counts)

    # ── Grade ─────────────────────────────────────────────
    grade = inspection_grade(compliance_score)

    # ── Status (severity-aware) ───────────────────────────
    status = operational_status(compliance_score, severity_counts)

    # ── Priority (issue-type aware) ───────────────────────
    priority = priority_action(severity_counts, report)

    # ── Audit (critical-count aware) ──────────────────────
    audit = audit_readiness(compliance_score, severity_counts)

    # ── Benchmark (critical-count aware) ──────────────────
    compliance_benchmark = benchmark(compliance_score, severity_counts)

    # ── Executive summary ─────────────────────────────────
    executive_summary = generate_llm_summary(report, compliance_score)

    # ── AI findings ───────────────────────────────────────
    ai_findings = build_ai_findings(report)

    # ── Annotated evidence image ──────────────────────────
    annotated_image_path = generate_annotated_evidence_image(
        image, detections, report
    )

    # ── Final response ────────────────────────────────────
    return {

        # Core
        "report":               report,
        "compliance_score":     compliance_score,
        "overall_risk":         overall_risk,
        "inspection_grade":     grade,
        "overall_status":       status,
        "annotated_image_path": annotated_image_path,

        # Executive
        "priority_action":   priority,
        "executive_summary": executive_summary,
        "executive_risk_index":     risk_index,
        "site_intelligence_summary": site_intelligence,

        # Enterprise analytics
        "analytics": {
            **analytics,
            "audit_readiness":          audit,
            "compliance_benchmark":     compliance_benchmark,
            "department_exposure":      dept_exposure,
            "management_escalation":    escalation_flag,
            "inspection_health_rating": health_rating,
            "executive_insight":        insight,
        },

        # AI
        "ai_findings": ai_findings,
    }