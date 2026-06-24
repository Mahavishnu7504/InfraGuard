from collections import Counter
from backend.services.quality.guideline_service import get_guidelines
from datetime import datetime
import uuid
import random

SEVERITY_SCORE_DEDUCTIONS = {
    "critical": 25,
    "high":     15,
    "medium":    7,
    "low":       3,
}
SEVERITY_ORDER = {
    "critical": 0,
    "high":     1,
    "medium":   2,
    "low":      3,
}

# =========================================================
# OBSERVATION DIVERSITY TEMPLATES
# =========================================================
# Issue types not present here fall back to the guideline's
# single static observation sentence (see build_observation).
OBSERVATION_TEMPLATES = {
    "poor_housekeeping": [
        "Housekeeping deficiencies were identified within the inspection area",
        "Material organization standards were not consistently maintained",
        "Site cleanliness controls appear to have deteriorated in this area",
    ],
    "rebar_exposure": [
        "Reinforcement elements were observed without adequate concrete cover",
        "Exposed reinforcement was identified during inspection",
        "Concrete protection of reinforcing steel appears compromised",
    ],
    "material_damage": [
        "Visible material deterioration was observed",
        "Construction materials exhibited signs of physical damage",
        "Material condition deficiencies were identified",
    ],
}

# =========================================================
# COMPLIANCE SCORING
# =========================================================
def compute_compliance_score(detections: list) -> int:
    """
    Deduct points per finding with two intelligence layers:

    1. Confidence weighting
       Penalty scales between 65 % (confidence = 0) and
       100 % (confidence = 1).  Low-confidence detections
       do not unfairly collapse the score.

    2. Repeat-issue dampening
       Once the same issue_type appears more than twice,
       each additional occurrence carries 70 % weight.
       The score already reflects the pattern; further
       repetitions are informative but not additive at
       full weight.
    """
    score             = 100.0
    issue_type_tally  = Counter()

    for item in detections:
        sev        = item.get("severity",   "medium").lower()
        confidence = float(item.get("confidence", 0.75))
        issue_type = item.get("issue_type", "unknown")

        base_penalty = SEVERITY_SCORE_DEDUCTIONS.get(sev, 5)

        # ── Confidence factor: [0.65, 1.0] ──────────────────
        conf_factor = 0.65 + (0.35 * min(max(confidence, 0.0), 1.0))

        # ── Repeat dampening ─────────────────────────────────
        issue_type_tally[issue_type] += 1
        repeat_factor = (
            1.0 if issue_type_tally[issue_type] <= 2 else 0.70
        )

        score -= base_penalty * conf_factor * repeat_factor

    return max(0, round(score))


def compliance_status(score: int) -> str:
    if score >= 85:
        return "Compliant"
    if score >= 65:
        return "Conditionally Compliant"
    if score >= 40:
        return "Non-Compliant"
    return "Critical Non-Compliance"


def risk_level_from_score(score: int) -> str:
    if score >= 85:
        return "Low"
    if score >= 65:
        return "Medium"
    if score >= 40:
        return "High"
    return "Critical"


def inspection_grade(score: int) -> str:
    if score >= 95:
        return "A+"
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 50:
        return "D"
    return "F"


# =========================================================
# SEVERITY BREAKDOWN COUNTER
# =========================================================
def severity_breakdown(detections: list) -> dict:
    counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for item in detections:
        sev = item.get("severity", "Medium").capitalize()
        if sev in counts:
            counts[sev] += 1
        else:
            counts["Medium"] += 1
    return counts


# =========================================================
# LOCATION INTELLIGENCE
# =========================================================
def classify_zone(bbox: list, image_width: int = 1000, image_height: int = 1000) -> str:
    """
    Classify a bounding box into one of five spatial zones.
    bbox is expected as [x1, y1, x2, y2] in pixel or normalised coordinates.
    If values are all <= 1.0 they are treated as normalised; otherwise as pixels.
    """
    if not bbox or len(bbox) < 4:
        return ""

    x1, y1, x2, y2 = bbox[:4]

    # Detect normalised coordinates (0–1 range)
    if all(0.0 <= v <= 1.0 for v in [x1, y1, x2, y2]):
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
    else:
        cx = (x1 + x2) / 2.0 / image_width
        cy = (y1 + y2) / 2.0 / image_height

    # Horizontal thirds: left (<0.33), centre (0.33–0.67), right (>0.67)
    # Vertical halves:   upper (<0.50), lower (>=0.50)
    if cx < 0.33:
        h = "left"
    elif cx > 0.67:
        h = "right"
    else:
        h = "centre"

    v = "upper" if cy < 0.50 else "lower"

    zone_map = {
        ("upper", "left"):   "upper-left",
        ("upper", "right"):  "upper-right",
        ("upper", "centre"): "central",
        ("lower", "left"):   "lower-left",
        ("lower", "right"):  "lower-right",
        ("lower", "centre"): "central",
    }
    return zone_map.get((v, h), "central")


_ZONE_DESCRIPTIONS = {
    "upper-left":  "upper-left structural inspection zone",
    "upper-right": "upper-right overhead work zone",
    "central":     "central work area",
    "lower-left":  "lower-left foundation and base zone",
    "lower-right": "lower-right structural work zone adjacent to active material staging areas",
}

_ZONE_CONTEXT = {
    "upper-left": (
        "in proximity to overhead structural connections and upper perimeter supports"
    ),
    "upper-right": (
        "within elevated work areas subject to increased exposure and load cycling"
    ),
    "central": (
        "across the primary operational corridor where worker traffic concentration is highest"
    ),
    "lower-left": (
        "at foundation and base-level interfaces where moisture and ground contact risk is elevated"
    ),
    "lower-right": (
        "adjacent to active material staging and plant movement corridors"
    ),
}


def zone_description(zone: str) -> str:
    return _ZONE_DESCRIPTIONS.get(zone, "work zone")


def zone_context(zone: str) -> str:
    return _ZONE_CONTEXT.get(zone, "within the inspected work area")


# =========================================================
# CONFIDENCE INTELLIGENCE
# =========================================================
def confidence_qualifier(confidence: float) -> str:
    if confidence >= 0.95:
        return (
            "Detection confidence is very high, "
            "with strong and consistent visual evidence "
            "supporting this finding."
        )
    if confidence >= 0.85:
        return (
            "Detection confidence is high, "
            "with clear visual indicators present "
            "across the inspected surface."
        )
    if confidence >= 0.70:
        return (
            "Detection confidence is moderate, "
            "with visible but partially ambiguous evidence "
            "observed during inspection."
        )
    return (
        "Detection confidence is limited; "
        "field verification by a qualified inspector "
        "is recommended before corrective action is initiated."
    )


def confidence_evidence_phrase(confidence: float) -> str:
    """
    Returns a short inline evidence phrase for embedding
    directly into observation sentences.
    """
    if confidence > 0.90:
        return "Strong visual evidence was identified with high detection confidence"
    if confidence >= 0.75:
        return "Moderate visual evidence was identified requiring verification"
    return "Limited visual evidence was identified and should be validated through manual inspection"


# =========================================================
# FINDING DENSITY INTELLIGENCE
# =========================================================
def density_phrase(total_findings: int) -> str:
    """
    Returns the appropriate density descriptor based on
    the total number of findings across the image or site.
    """
    if total_findings == 1:
        return "An isolated occurrence was identified"
    if total_findings <= 4:
        return "A localized cluster of findings was identified"
    return "Widespread site-wide deficiencies were identified"


def density_context(total_findings: int) -> str:
    """
    Returns a fuller narrative sentence describing finding density.
    """
    if total_findings == 1:
        return (
            "An isolated occurrence was identified in the inspection area, "
            "suggesting a localised deficiency rather than a systemic pattern."
        )
    if total_findings <= 4:
        return (
            f"A localised cluster of {total_findings} findings was identified, "
            "indicating a concentrated zone of quality deviation that warrants "
            "targeted corrective intervention."
        )
    return (
        f"Widespread site-wide deficiencies were identified across "
        f"{total_findings} detected findings, indicating systemic quality "
        "control failures requiring immediate management-level escalation."
    )


# =========================================================
# SEVERITY OBSERVATION CONTEXT
# =========================================================
def severity_observation_context(severity: str) -> str:
    mapping = {
        "critical": (
            "This condition is classified as critical severity "
            "and requires immediate engineering attention "
            "before operations continue."
        ),
        "high": (
            "This high-severity condition requires prompt "
            "corrective action and qualified engineering review "
            "within the current operational period."
        ),
        "medium": (
            "This moderate-severity condition should be addressed "
            "within a scheduled maintenance or corrective workflow."
        ),
        "low": (
            "This low-severity observation should be monitored "
            "and resolved during the next routine inspection cycle."
        ),
    }
    return mapping.get(severity.lower(), mapping["medium"])


# =========================================================
# ENHANCED OBSERVATION BUILDER
# =========================================================
def build_observation(
    guideline:      dict,
    confidence:     float,
    severity:       str,
    issue_type:     str = "",
    bbox:           list = None,
    total_findings: int = 1,
) -> str:
    """
    Constructs a dynamic, context-rich observation narrative
    integrating issue type, spatial zone, severity, confidence
    evidence, and finding density.
    """
    templates = OBSERVATION_TEMPLATES.get(issue_type)
    if templates:
        base = random.choice(templates)
    else:
        base = guideline["observation"]
    sev_key = severity.lower()

    # ── Location context ─────────────────────────────────
    zone     = classify_zone(bbox) if bbox else ""
    zone_str = (
        f" within the {zone_description(zone)}" if zone else ""
    )
    zone_ctx = (
        f", {zone_context(zone)}," if zone else ""
    )

    # ── Confidence evidence phrase ────────────────────────
    evidence = confidence_evidence_phrase(confidence)

    # ── Density phrase ────────────────────────────────────
    density  = density_phrase(total_findings)

    # ── Assemble by severity tier ─────────────────────────
    if sev_key == "critical":
        obs = (
            f"{severity_observation_context(severity)} "
            f"{base}{zone_str}{zone_ctx} "
            f"{evidence}. "
            f"{density}, indicating conditions that demand immediate "
            f"engineering escalation."
        )
    elif sev_key == "high":
        obs = (
            f"{base}{zone_str}. "
            f"{evidence}{zone_ctx} "
            f"{severity_observation_context(severity)} "
            f"{density}."
        )
    else:
        obs = (
            f"{base}{zone_str}. "
            f"{evidence}. "
            f"{severity_observation_context(severity)} "
            f"{density}."
        )

    return obs


# =========================================================
# OPERATIONAL IMPACT  (severity-stratified)
# =========================================================
_OPERATIONAL_IMPACTS: dict = {

    "surface_crack": {
        "critical": (
            "Structural failure risk is elevated. Load-bearing "
            "capacity may be critically compromised, creating "
            "immediate safety hazards to personnel and infrastructure."
        ),
        "high": (
            "Progressive crack propagation may accelerate structural "
            "degradation and reduce long-term infrastructure reliability "
            "if not remediated promptly."
        ),
        "medium": (
            "Continued crack progression may increase structural "
            "maintenance requirements and reduce infrastructure service "
            "life."
        ),
        "low": (
            "Minor surface cracking presents limited operational impact "
            "but may progress under sustained loading or environmental "
            "exposure."
        ),
    },

    "corrosion": {
        "critical": (
            "Material cross-section and load capacity may be critically "
            "reduced. Immediate structural assessment is required to "
            "confirm residual integrity."
        ),
        "high": (
            "Active corrosion is weakening material performance and "
            "accelerating infrastructure degradation."
        ),
        "medium": (
            "Material oxidation activity may reduce operational lifespan "
            "and structural durability over time."
        ),
        "low": (
            "Early-stage corrosion has limited immediate operational "
            "impact but should be treated to prevent progression."
        ),
    },

    "water_leakage": {
        "critical": (
            "Uncontrolled moisture ingress presents immediate risks of "
            "structural damage, electrical hazard, and rapid material "
            "deterioration."
        ),
        "high": (
            "Active moisture infiltration is accelerating infrastructure "
            "deterioration and may spread to adjacent structural components."
        ),
        "medium": (
            "Persistent leakage may progressively deteriorate structural "
            "materials and increase long-term maintenance exposure."
        ),
        "low": (
            "Minor moisture indicators have limited immediate impact "
            "but may escalate if waterproofing deficiency is not addressed."
        ),
    },

    "rebar_exposure": {
        "critical": (
            "Actively corroding exposed reinforcement is critically "
            "undermining structural integrity and concrete protection "
            "capacity."
        ),
        "high": (
            "Significant rebar exposure is accelerating corrosion "
            "propagation and reducing concrete structural performance."
        ),
        "medium": (
            "Partial reinforcement exposure may allow progressive "
            "corrosion and reduce long-term concrete durability."
        ),
        "low": (
            "Minor concrete cover deficiency has limited immediate "
            "impact but creates conditions for moisture-driven corrosion."
        ),
    },

    "material_damage": {
        "critical": (
            "Severe material damage may have critically compromised "
            "structural or operational integrity, requiring immediate "
            "engineering assessment."
        ),
        "high": (
            "Significant material deterioration is reducing infrastructure "
            "performance and may escalate if corrective action is delayed."
        ),
        "medium": (
            "Moderate material damage may negatively affect operational "
            "reliability and increase maintenance requirements."
        ),
        "low": (
            "Minor surface damage has limited operational impact but "
            "should be repaired to prevent environmental progression."
        ),
    },

    "poor_housekeeping": {
        "critical": (
            "Severely disorganised conditions are creating immediate "
            "safety hazards and indicate a breakdown in site management "
            "controls."
        ),
        "high": (
            "Significant housekeeping deficiencies are elevating worker "
            "injury risk and reducing site safety compliance performance."
        ),
        "medium": (
            "Housekeeping shortfalls are creating unnecessary operational "
            "hazards and reducing site efficiency."
        ),
        "low": (
            "Minor housekeeping issues present limited risk but should "
            "be resolved to maintain site compliance standards."
        ),
    },

    "ppe_non_compliance": {
        "critical": (
            "Systemic PPE non-compliance critically elevates worker "
            "injury risk and indicates a safety culture failure requiring "
            "immediate management intervention."
        ),
        "high": (
            "Significant PPE non-compliance is creating serious worker "
            "injury exposure and safety enforcement failures."
        ),
        "medium": (
            "PPE non-compliance in the affected area is elevating worker "
            "injury risk and may compromise the site safety record."
        ),
        "low": (
            "Isolated PPE non-compliance presents limited risk if "
            "corrective instruction is issued promptly."
        ),
    },
}

_DEFAULT_OPERATIONAL_IMPACT = {
    "critical": (
        "This critical condition may immediately compromise operational "
        "safety and infrastructure performance."
    ),
    "high": (
        "This high-severity condition may reduce infrastructure "
        "reliability and operational performance if not addressed promptly."
    ),
    "medium": (
        "Operational quality deviations may negatively affect "
        "infrastructure reliability and compliance consistency."
    ),
    "low": (
        "This low-severity condition has limited operational impact "
        "under current conditions."
    ),
}


def operational_impact(issue_type: str, severity: str = "medium") -> str:
    impact_map   = _OPERATIONAL_IMPACTS.get(issue_type, _DEFAULT_OPERATIONAL_IMPACT)
    severity_key = severity.lower()
    if severity_key not in ("critical", "high", "medium", "low"):
        severity_key = "medium"
    return impact_map.get(severity_key, impact_map.get("medium", ""))


# =========================================================
# AI METADATA
# =========================================================
def build_ai_metadata(confidence: float) -> dict:
    if confidence >= 0.90:
        level = "Enterprise Verified"
    elif confidence >= 0.80:
        level = "High Confidence"
    elif confidence >= 0.70:
        level = "Moderate Confidence"
    else:
        level = "Review Recommended"
    return {
        "confidence_level":  level,
        "analysis_engine":   "InfraGuard Enterprise AI",
        "inspection_type":   "Construction Quality Assurance",
        "detection_model":   "InfraVision-X Enterprise",
        "analysis_mode":     "Realtime Inspection Intelligence",
    }


# =========================================================
# VISUAL CONTEXT
# =========================================================
def build_visual_context(features: dict) -> str:
    return (
        f"Brightness Index: {features.get('brightness', '-')}, "
        f"Edge Density Level: {features.get('edge_density', '-')}, "
        f"Surface Texture Complexity: {features.get('texture_complexity', '-')}"
    )


# =========================================================
# REFERENCE STANDARD MAPPING
# =========================================================
_REFERENCE_STANDARDS: dict = {

    "surface_crack": (
        "IS 456:2000 — Code of Practice for Plain and Reinforced Concrete "
        "(Clause 35.3.2 — Cracking)"
    ),

    "corrosion": (
        "ISO 12944 — Corrosion Protection of Steel Structures by Protective "
        "Paint Systems"
    ),

    "water_leakage": (
        "IS 3067 — Code of Practice for General Design Considerations for "
        "Waterproofing and Damp-Proofing of Buildings"
    ),

    "rebar_exposure": (
        "IS 456:2000 — Clause 26.4 (Nominal Cover to Reinforcement)"
    ),

    "material_damage": (
        "IS 4082 — Recommendations on Stacking and Storage of Construction "
        "Materials at Site"
    ),

    "poor_housekeeping": (
        "OSHA 29 CFR 1926.25 — Housekeeping (Construction Industry)"
    ),

    "ppe_non_compliance": (
        "OSHA 29 CFR 1926.95 — Criteria for Personal Protective Equipment"
    ),
}

_DEFAULT_REFERENCE_STANDARD = (
    "InfraGuard Enterprise Quality Assurance Framework — General "
    "Construction Quality Provisions"
)


def reference_standard(issue_type: str) -> str:
    return _REFERENCE_STANDARDS.get(issue_type, _DEFAULT_REFERENCE_STANDARD)


# =========================================================
# HSE CONDITION CLASSIFICATION  (severity-stratified)
# =========================================================
_CONDITION_CLASSIFICATION: dict = {
    "critical": "Unsafe Condition — Immediate Hazard",
    "high":     "At-Risk Condition — Hazardous",
    "medium":   "Substandard Condition",
    "low":      "Minor Deviation",
}


def condition_classification(severity: str) -> str:
    severity_key = severity.lower()
    if severity_key not in _CONDITION_CLASSIFICATION:
        severity_key = "medium"
    return _CONDITION_CLASSIFICATION[severity_key]


# =========================================================
# PRIORITY LEVEL  (severity-stratified)
# =========================================================
_PRIORITY_LEVELS: dict = {
    "critical": "P1 — Immediate",
    "high":     "P2 — Urgent",
    "medium":   "P3 — Scheduled",
    "low":      "P4 — Routine",
}


def priority_level(severity: str) -> str:
    severity_key = severity.lower()
    if severity_key not in _PRIORITY_LEVELS:
        severity_key = "medium"
    return _PRIORITY_LEVELS[severity_key]


# =========================================================
# INSPECTION STATUS  (severity-stratified)
# =========================================================
_INSPECTION_STATUS: dict = {
    "critical": "Open — Immediate Action Required",
    "high":     "Open — Action Required",
    "medium":   "Open — Pending Corrective Action",
    "low":      "Open — Monitoring",
}


def inspection_status(severity: str) -> str:
    severity_key = severity.lower()
    if severity_key not in _INSPECTION_STATUS:
        severity_key = "medium"
    return _INSPECTION_STATUS[severity_key]


# =========================================================
# RESPONSIBLE TEAM  (issue-type stratified, frontend Phase 6/4)
# =========================================================
_RESPONSIBLE_TEAM_MAP: dict = {
    "surface_crack":      "Structural Engineer",
    "corrosion":          "Structural Engineer",
    "water_leakage":      "Civil Engineer",
    "rebar_exposure":     "Structural Engineer",
    "material_damage":    "Site Supervisor",
    "poor_housekeeping":  "Site Supervisor",
    "ppe_non_compliance": "Site Safety Officer",
}

_DEFAULT_RESPONSIBLE_TEAM = "Site Engineer"


def responsible_team(issue_type: str) -> str:
    """
    Maps an issue_type to the engineering/site role accountable for
    resolving it. Drives the "Responsible Team" / "Owner" columns
    consumed by GenerateReport.jsx (Phase 4 Corrective Action Matrix
    and Phase 6 Findings Intelligence).
    """
    return _RESPONSIBLE_TEAM_MAP.get(issue_type, _DEFAULT_RESPONSIBLE_TEAM)


# =========================================================
# TARGET RESOLUTION  (severity-stratified, frontend Phase 6)
# =========================================================
_TARGET_RESOLUTION: dict = {
    "critical": "Within 24 Hours",
    "high":     "Within 3 Days",
    "medium":   "Within 14 Days",
    "low":      "Within 30 Days",
}


def target_resolution(severity: str) -> str:
    severity_key = severity.lower()
    if severity_key not in _TARGET_RESOLUTION:
        severity_key = "medium"
    return _TARGET_RESOLUTION[severity_key]


# =========================================================
# SINGLE-IMAGE FINDING BUILDER
# =========================================================
# =========================================================
# DYNAMIC ROOT CAUSE INTELLIGENCE
# =========================================================
def build_dynamic_root_cause(issue_type: str, severity: str, total_findings: int) -> str:
    """
    Derive an executive-level root cause statement from severity,
    finding density, and issue type, instead of a single static
    guideline string.
    """
    if severity.lower() == "critical":
        return "Systemic management control failure"

    if total_findings >= 5:
        return "Recurring site-wide process enforcement weakness"

    if issue_type == "poor_housekeeping":
        return "Housekeeping discipline breakdown"

    if issue_type == "ppe_non_compliance":
        return "Safety awareness and supervision deficiency"

    if issue_type == "material_damage":
        return "Improper material handling and storage practices"

    return "Localized operational control gap"


def build_finding(item: dict, total_findings: int = 1) -> dict:
    issue_type = item.get("issue_type", "unknown_issue")
    confidence = float(item.get("confidence", 0.0))
    category   = item.get("category",   "General Construction Quality")
    severity   = item.get("severity",   "Medium")
    features   = item.get("features",   {})
    bbox       = item.get("bbox",       [])

    guideline   = get_guidelines(issue_type, severity, confidence)
    observation = build_observation(
        guideline,
        confidence,
        severity,
        issue_type,
        bbox=bbox,
        total_findings=total_findings,
    )

    return {
        # Identity
        "finding_id":          f"INF-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
        # Core
        "issue_type":          issue_type,
        "category":            category,
        "severity":            severity,
        "confidence":          round(confidence, 2),
        # Narrative
        "observation":         observation,
        "risk":                guideline["risk"],
        "potential_consequences": guideline.get("potential_consequences", []),
        "management_impact":   guideline.get("management_impact", ""),
        "risk_category":       guideline.get("risk_category", "C"),
        "operational_impact":  operational_impact(issue_type, severity),
        "corrective_action":   guideline["corrective_action"],
        "preventive_action":   guideline["preventive_action"],
        "best_practice":       guideline["best_practice"],
        "guideline_reference": guideline["guideline_reference"],
        # HSE classification
        "root_cause":          build_dynamic_root_cause(issue_type, severity, total_findings),
        "reference_standard":  reference_standard(issue_type),
        "condition":           condition_classification(severity),
        "priority_level":      priority_level(severity),
        "inspection_status":   inspection_status(severity),
        # Spatial / visual
        "zone":                classify_zone(bbox),
        "visual_context":      build_visual_context(features),
        "bbox":                bbox,
        # AI metadata
        "ai_metadata":         build_ai_metadata(confidence),
        # ── Frontend Phase 4 / 6 fields (GenerateReport.jsx) ──
        # risk_impact / owner / action are aliases of existing fields,
        # kept under both names so the frontend's preferred key always
        # resolves without requiring a frontend change.
        "risk_impact":         operational_impact(issue_type, severity),
        "responsible_team":    responsible_team(issue_type),
        "owner":               responsible_team(issue_type),
        "target_resolution":   target_resolution(severity),
        "action":              guideline["corrective_action"],
    }


def sort_findings(findings: list) -> list:
    return sorted(
        findings,
        key=lambda x: SEVERITY_ORDER.get(x["severity"].lower(), 99),
    )


# =========================================================
# PER-IMAGE REPORT SECTION
# =========================================================
def build_image_report(image_data: dict) -> dict:
    detections    = image_data.get("detections", [])
    total_in_img  = len(detections)

    # Pass total_findings per image so density intelligence
    # reflects the image-level finding count in observations.
    findings = sort_findings(
        [build_finding(d, total_findings=total_in_img) for d in detections]
    )

    score     = compute_compliance_score(detections)
    breakdown = severity_breakdown(detections)
    status    = compliance_status(score)
    risk      = risk_level_from_score(score)
    grade     = inspection_grade(score)

    corrective_actions = list({f["corrective_action"] for f in findings})
    preventive_actions = list({f["preventive_action"] for f in findings})
    best_practices     = list({f["best_practice"]     for f in findings})

    return {
        "image_index":           image_data.get("image_index", 1),
        "image_label":           image_data.get("image_label",
                                     f"Image {image_data.get('image_index', 1)}"),
        "location":              image_data.get("location", ""),
        "image_path":            image_data.get("image_path", ""),
        "annotated_image_path":  image_data.get("annotated_image_path", ""),
        "findings":              findings,
        "total_findings":        len(findings),
        "severity_breakdown":    breakdown,
        "compliance_score":      score,
        "compliance_status":     status,
        "risk_level":            risk,
        "inspection_grade":      grade,
        "corrective_actions":    corrective_actions,
        "preventive_actions":    preventive_actions,
        "best_practices":        best_practices,
    }


# =========================================================
# OVERALL RECOMMENDATIONS AGGREGATOR
# =========================================================
def aggregate_overall_recommendations(image_reports: list) -> dict:
    critical_actions = {}
    high_actions     = {}
    preventive       = {}
    best_practices   = set()

    for img in image_reports:
        for f in img["findings"]:
            sev        = f["severity"].lower()
            issue_type = f["issue_type"]
            ca         = f["corrective_action"]
            pa         = f["preventive_action"]
            bp         = f["best_practice"]

            if sev == "critical":
                critical_actions[issue_type] = ca
            elif sev == "high":
                if issue_type not in critical_actions:
                    high_actions[issue_type] = ca
            else:
                preventive[issue_type] = pa

            best_practices.add(bp)

    return {
        "critical_actions":   list(critical_actions.values()),
        "high_actions":       list(high_actions.values()),
        "preventive_actions": list(preventive.values()),
        "best_practices":     list(best_practices),
    }


# =========================================================
# SITE-WIDE PATTERN ANALYSIS
# =========================================================
def build_pattern_analysis(image_reports: list) -> dict:
    """
    Detect recurring issue types across all inspected images and
    produce a narrative describing site-wide pattern intelligence.

    Returns
    -------
    dict with keys:
        recurring_issues : list of {issue_type, display_name, count}
        narrative        : str
    """
    counter = Counter()
    for img in image_reports:
        for f in img["findings"]:
            counter[f["issue_type"]] += 1

    if not counter:
        return {
            "recurring_issues": [],
            "narrative": (
                "No recurring issue patterns were identified across the "
                "inspected images."
            ),
        }

    top_issues = counter.most_common(3)
    recurring_issues = [
        {
            "issue_type":   issue_type,
            "display_name": _display_name(issue_type),
            "count":        count,
        }
        for issue_type, count in top_issues
        if count > 1
    ]

    if not recurring_issues:
        return {
            "recurring_issues": [],
            "narrative": (
                "No recurring issue patterns were identified across the "
                "inspected images; findings were largely isolated occurrences."
            ),
        }

    sentences = []
    for entry in recurring_issues:
        sentences.append(
            f"{entry['display_name']} appeared in "
            f"{entry['count']} finding{'s' if entry['count'] != 1 else ''}, "
            f"indicating a recurring site-wide quality control gap."
        )

    narrative = " ".join(sentences)

    return {
        "recurring_issues": recurring_issues,
        "narrative":        narrative,
    }


# =========================================================
# MANAGEMENT ATTENTION AREAS
# =========================================================
def build_management_attention(image_reports: list) -> str:
    """
    Produce an executive-level narrative highlighting the issue
    categories that represent the greatest management and audit
    exposure across the inspection dataset.
    """
    severity_counter = Counter()
    type_counter     = Counter()

    for img in image_reports:
        for f in img["findings"]:
            sev = f["severity"].lower()
            if sev in ("critical", "high"):
                type_counter[f["issue_type"]] += 1
            severity_counter[sev] += 1

    if not type_counter:
        return (
            "No issue categories currently represent elevated management "
            "attention; the site reflects standard operational conditions."
        )

    top_types    = [t for t, _ in type_counter.most_common(2)]
    display_names = [_display_name(t) for t in top_types]

    if len(display_names) == 1:
        issue_str = display_names[0]
    else:
        issue_str = f"{display_names[0]} and {display_names[1]}"

    critical_n = severity_counter.get("critical", 0)
    high_n     = severity_counter.get("high", 0)

    narrative = (
        f"Recurring {issue_str} represent the highest audit exposure "
        f"identified during this inspection."
    )

    if critical_n > 0:
        narrative += (
            f" {critical_n} critical-severity finding"
            f"{'s' if critical_n != 1 else ''} require immediate "
            f"management escalation and engineering sign-off."
        )
    elif high_n > 0:
        narrative += (
            f" {high_n} high-severity finding{'s' if high_n != 1 else ''} "
            f"require prompt management review and resource allocation "
            f"for corrective action."
        )

    return narrative


# =========================================================
# SYSTEMIC DEFICIENCY ANALYSIS
# =========================================================
def build_systemic_deficiencies(image_reports: list) -> list:
    """
    Identify systemic (cross-image) weaknesses by mapping recurring
    issue types onto broader management system categories.

    Returns a list of distinct deficiency-area display strings,
    e.g. ["Housekeeping", "Material Storage", "Safety Compliance"].
    """
    _SYSTEM_CATEGORY_MAP = {
        "surface_crack":      "Structural Quality Control",
        "corrosion":          "Asset Preservation & Maintenance",
        "water_leakage":      "Waterproofing & Drainage Management",
        "rebar_exposure":     "Construction Quality Assurance",
        "material_damage":    "Material Storage & Handling",
        "poor_housekeeping":  "Housekeeping",
        "ppe_non_compliance": "Safety Compliance",
    }

    counter = Counter()
    for img in image_reports:
        for f in img["findings"]:
            counter[f["issue_type"]] += 1

    deficiencies = []
    for issue_type, count in counter.most_common():
        if count > 1:
            category = _SYSTEM_CATEGORY_MAP.get(
                issue_type, _display_name(issue_type)
            )
            if category not in deficiencies:
                deficiencies.append(category)

    return deficiencies


# =========================================================
# AUDIT EXPOSURE SUMMARY
# =========================================================
def build_audit_exposure_summary(image_reports: list, overall_score: int) -> str:
    """
    Produce audit-facing language describing the likely exposure
    this inspection's findings would create during an external
    quality or HSE audit.
    """
    total_findings = sum(r["total_findings"] for r in image_reports)
    risk           = risk_level_from_score(overall_score)
    status         = compliance_status(overall_score)

    if total_findings == 0:
        return (
            "No findings were identified during this inspection. Current "
            "conditions are unlikely to result in adverse observations "
            "during external quality or HSE audits."
        )

    if risk == "Critical":
        return (
            f"Current findings reflect a {status.lower()} status and are "
            f"highly likely to result in adverse observations, non-conformance "
            f"reports, and potential regulatory escalation during external "
            f"quality and HSE audits unless immediate corrective action is taken."
        )
    if risk == "High":
        return (
            f"Current findings reflect a {status.lower()} status and may "
            f"result in adverse observations during external quality and HSE "
            f"audits if corrective actions are not implemented within the "
            f"recommended timeframe."
        )
    if risk == "Medium":
        return (
            f"Current findings may result in minor adverse observations "
            f"during external quality and HSE audits. Implementing the "
            f"recommended corrective and preventive actions will reduce "
            f"audit exposure."
        )
    return (
        f"Current findings present minimal audit exposure. Maintaining "
        f"existing preventive maintenance and inspection practices should "
        f"sustain favourable outcomes during external quality and HSE audits."
    )


# =========================================================
# COMPLIANCE BENCHMARK  (frontend Phase 7)
# =========================================================
def compliance_benchmark(score: int) -> str:
    """
    Maps the overall compliance score onto a three-tier enterprise
    benchmark label consumed by the Compliance Benchmark Panel.
    """
    if score >= 90:
        return "Enterprise Grade"
    if score >= 65:
        return "Industry Acceptable"
    return "Below Standard"


# =========================================================
# AUDIT STATUS  (frontend Phase 7 — short categorical companion
# to the existing free-text `audit_readiness` narrative)
# =========================================================
def audit_status(score: int, critical_count: int) -> str:
    if critical_count > 0:
        return "Not Audit Ready"
    if score >= 85:
        return "Audit Ready"
    if score >= 65:
        return "Conditionally Audit Ready"
    return "Not Audit Ready"


# =========================================================
# OPERATIONAL STATUS  (frontend Phase 7)
# =========================================================
def operational_status(overall_risk: str) -> str:
    mapping = {
        "Critical": "Halted",
        "High":     "Restricted",
        "Medium":   "Stable — Monitoring Required",
        "Low":      "Stable",
    }
    return mapping.get(overall_risk, "Stable")


# =========================================================
# AI CONFIDENCE ANALYTICS  (frontend Phase 8)
# =========================================================
def analytics_summary(image_reports: list) -> dict:
    """
    Aggregates per-finding confidence across every image into the
    summary figures consumed by the AI Confidence Analytics panel:
        average_confidence        — mean confidence, as a whole-number %
        high_confidence_findings  — count with confidence >= 0.80
        review_required_findings  — count with confidence <  0.60
    """
    confidences = [
        f["confidence"]
        for img in image_reports
        for f in img["findings"]
    ]

    if not confidences:
        return {
            "average_confidence":       0,
            "high_confidence_findings": 0,
            "review_required_findings": 0,
        }

    average_pct = round((sum(confidences) / len(confidences)) * 100)
    high_count   = sum(1 for c in confidences if c >= 0.80)
    review_count = sum(1 for c in confidences if c < 0.60)

    return {
        "average_confidence":       average_pct,
        "high_confidence_findings": high_count,
        "review_required_findings": review_count,
    }


# =========================================================
# ISSUE DISPLAY NAMES
# =========================================================
_ISSUE_DISPLAY_NAMES = {
    "surface_crack":      "Surface Cracking",
    "corrosion":          "Corrosion",
    "water_leakage":      "Water Leakage",
    "rebar_exposure":     "Rebar Exposure",
    "material_damage":    "Material Damage",
    "poor_housekeeping":  "Housekeeping Deficiencies",
    "ppe_non_compliance": "PPE Non-Compliance",
}


def _display_name(issue_type: str) -> str:
    return _ISSUE_DISPLAY_NAMES.get(
        issue_type,
        issue_type.replace("_", " ").title(),
    )


# =========================================================
# SITE-LEVEL INTELLIGENCE HELPERS
# =========================================================
def _most_common_issue(image_reports: list) -> str:
    """Returns display name of the most frequently occurring issue type."""
    counter = Counter()
    for img in image_reports:
        for f in img["findings"]:
            counter[f["issue_type"]] += 1
    if not counter:
        return ""
    return _display_name(counter.most_common(1)[0][0])


def _most_affected_zone(image_reports: list) -> str:
    """Returns the zone label with the highest finding concentration."""
    zone_counter = Counter()
    for img in image_reports:
        for f in img["findings"]:
            z = f.get("zone", "")
            if z:
                zone_counter[z] += 1
    if not zone_counter:
        return ""
    top_zone = zone_counter.most_common(1)[0][0]
    return zone_description(top_zone)


def _highest_severity_issue(image_reports: list) -> str:
    """Returns display name of the issue with the highest severity rank."""
    best_rank = 99
    best_issue = ""
    for img in image_reports:
        for f in img["findings"]:
            rank = SEVERITY_ORDER.get(f["severity"].lower(), 99)
            if rank < best_rank:
                best_rank  = rank
                best_issue = f["issue_type"]
    return _display_name(best_issue) if best_issue else ""


def _site_condition_summary(overall_score: int, total_findings: int) -> str:
    """
    Returns a concise overall site condition narrative based on
    score and finding volume.
    """
    risk   = risk_level_from_score(overall_score)
    status = compliance_status(overall_score)

    if total_findings == 0:
        return (
            f"Overall site condition is strong with no detectable quality "
            f"deviations. The site is operating within compliance parameters "
            f"and demonstrates effective quality management practices."
        )

    density = density_phrase(total_findings).lower()

    condition_map = {
        "Critical": (
            f"{density} requiring immediate management escalation. "
            f"The site presents a {status.lower()} status ({overall_score}/100) "
            f"with critical risk exposure demanding engineering intervention "
            f"before operations resume."
        ),
        "High": (
            f"{density} indicating elevated risk exposure across the site. "
            f"The {status.lower()} status ({overall_score}/100) requires "
            f"accelerated corrective action and heightened engineering oversight."
        ),
        "Medium": (
            f"{density} reflecting moderate quality deviations that, while not "
            f"immediately critical, require structured corrective planning. "
            f"The site achieves a {status.lower()} rating ({overall_score}/100)."
        ),
        "Low": (
            f"{density} representing minor or routine deviations. "
            f"The site demonstrates a {status.lower()} compliance status "
            f"({overall_score}/100) with low residual risk."
        ),
    }
    return condition_map.get(risk, condition_map["Medium"])


# =========================================================
# SITE SUMMARY  (short plain-language status line)
# =========================================================
def build_site_summary(image_reports: list, overall_score: int) -> str:
    """
    Generates the concise one-sentence site condition statement
    shown at the top of the Executive Summary section.
    """
    status = compliance_status(overall_score)

    issue_names: list = []
    seen: set         = set()
    for img in image_reports:
        for f in img["findings"]:
            it = f["issue_type"]
            if it not in seen:
                seen.add(it)
                issue_names.append(_display_name(it).lower())

    if not issue_names:
        return (
            f"The site condition is currently {status.lower()} with no "
            f"critical findings detected during this inspection cycle."
        )

    issues_str = ", ".join(issue_names)
    return (
        f"The site condition is currently {status.lower()} due to critical "
        f"visible findings involving {issues_str}."
    )


# =========================================================
# PRIORITY ACTION  (risk-stratified)
# =========================================================
def build_priority_action(overall_risk: str) -> str:
    """Returns the bold Priority Action sentence for the report."""
    mapping = {
        "Critical": (
            "Halt operations immediately. Engage qualified engineering "
            "personnel to assess and rectify all critical findings before "
            "work may resume."
        ),
        "High": (
            "Rectify all critical findings immediately and secure unsafe "
            "work zones if required."
        ),
        "Medium": (
            "Schedule corrective actions for all identified findings within "
            "the next operational period and assign responsible personnel "
            "for each item."
        ),
        "Low": (
            "Continue standard preventive maintenance. Monitor and address "
            "any findings during routine inspection cycles."
        ),
    }
    return mapping.get(
        overall_risk,
        "Rectify all findings as soon as practicable.",
    )


# =========================================================
# AI FINDINGS BULLETS  (Section 2)
# =========================================================
def build_ai_findings_bullets(
    image_reports: list,
    total_findings: int,
) -> list:
    """
    Returns the three bullet strings rendered under Section 2:
    AI Findings.
    """
    issue_names: list = []
    seen: set         = set()
    for img in image_reports:
        for f in img["findings"]:
            it = f["issue_type"]
            if it not in seen:
                seen.add(it)
                issue_names.append(_display_name(it).lower())

    n_images   = len(image_reports)
    img_word   = "image" if n_images == 1 else f"{n_images} images"
    issues_str = ", ".join(issue_names) if issue_names else "no issues"

    return [
        (
            f"A total of {total_findings} visible quality issue(s) were "
            f"identified in the inspected {img_word}."
        ),
        (
            f"The most notable observed conditions include: {issues_str}."
            if issue_names else
            "No notable conditions were identified in the inspected image."
        ),
        (
            "Corrective action and preventive controls are recommended "
            "to improve site discipline."
        ),
    ]


# =========================================================
# EXECUTIVE SUMMARY BUILDER  (Section 1 body text)
# =========================================================
# =========================================================
# SITE EXECUTIVE NARRATIVE
# =========================================================
def build_site_executive_narrative(image_reports: list, overall_score: int) -> str:
    """
    Produce a short executive-level narrative summarizing systemic
    risk posture across the inspected site, distinct from the
    per-finding executive summary.
    """
    total_findings = sum(r["total_findings"] for r in image_reports)
    critical_count  = sum(r["severity_breakdown"].get("Critical", 0) for r in image_reports)
    high_count      = sum(r["severity_breakdown"].get("High",     0) for r in image_reports)

    issue_counter: Counter = Counter()
    for img in image_reports:
        for f in img["findings"]:
            issue_counter[f["issue_type"]] += 1
    recurring_issue_count = sum(1 for c in issue_counter.values() if c > 1)

    if total_findings == 0:
        return (
            "The inspection identified no significant quality deviations "
            "across the reviewed operational zones. Current site discipline "
            "and supervision practices appear effective and should be "
            "maintained through routine monitoring."
        )

    if recurring_issue_count > 0:
        opening = (
            "The inspection identified recurring quality deficiencies across "
            "multiple operational zones."
        )
        pattern_line = (
            "The distribution and frequency of findings indicate systemic "
            "control weaknesses rather than isolated events."
        )
    else:
        opening = (
            "The inspection identified isolated quality deviations within "
            "the reviewed operational zones."
        )
        pattern_line = (
            "Findings do not currently indicate a site-wide pattern, though "
            "continued monitoring is recommended."
        )

    if critical_count > 0:
        severity_line = (
            f"{critical_count} critical finding{'s' if critical_count != 1 else ''} "
            f"{'require' if critical_count != 1 else 'requires'} immediate management "
            f"escalation and engineering review."
        )
    elif high_count > 0:
        severity_line = (
            f"{high_count} high-priority finding{'s' if high_count != 1 else ''} "
            f"{'warrant' if high_count != 1 else 'warrants'} accelerated corrective "
            f"action within the current operational period."
        )
    else:
        severity_line = (
            "No critical or high-priority findings were identified, though "
            "continued enforcement of corrective actions is advised."
        )

    closing = (
        "Management attention should focus on strengthening site supervision, "
        "quality assurance enforcement, and corrective action tracking "
        "mechanisms."
    )

    return f"{opening} {pattern_line} {severity_line} {closing}"


def build_executive_summary(
    image_reports: list,
    overall_score: int,
) -> str:
    total_images   = len(image_reports)
    total_findings = sum(r["total_findings"] for r in image_reports)
    risk           = risk_level_from_score(overall_score)
    status         = compliance_status(overall_score)

    critical_count = sum(r["severity_breakdown"].get("Critical", 0) for r in image_reports)
    high_count     = sum(r["severity_breakdown"].get("High",     0) for r in image_reports)
    medium_count   = sum(r["severity_breakdown"].get("Medium",   0) for r in image_reports)

    issue_severity_rank: dict = {}
    issue_count: Counter      = Counter()

    for img in image_reports:
        for f in img["findings"]:
            it   = f["issue_type"]
            rank = SEVERITY_ORDER.get(f["severity"].lower(), 99)
            issue_severity_rank[it] = min(issue_severity_rank.get(it, 99), rank)
            issue_count[it] += 1

    ranked_issues = sorted(
        issue_severity_rank.keys(),
        key=lambda x: (issue_severity_rank[x], -issue_count[x]),
    )
    top_display = [_display_name(i) for i in ranked_issues[:3]]

    # ── Intelligence-enriched metadata ───────────────────
    common_issue   = _most_common_issue(image_reports)
    affected_zone  = _most_affected_zone(image_reports)
    highest_sev    = _highest_severity_issue(image_reports)
    site_condition = _site_condition_summary(overall_score, total_findings)

    if total_findings == 0:
        return (
            f"InfraGuard AI conducted a comprehensive quality inspection across "
            f"{total_images} uploaded image{'s' if total_images != 1 else ''}. "
            f"No significant construction quality deviations were identified. "
            f"The site demonstrates strong compliance alignment with a score of "
            f"{overall_score}/100 ({status}). "
            f"Routine preventive monitoring and scheduled inspection workflows are "
            f"recommended to sustain current compliance standards."
        )

    parts = [
        f"InfraGuard AI conducted a comprehensive quality inspection across "
        f"{total_images} uploaded image{'s' if total_images != 1 else ''}, "
        f"identifying {total_findings} construction quality "
        f"deviation{'s' if total_findings != 1 else ''} requiring attention."
    ]

    if critical_count > 0:
        parts.append(
            f"{critical_count} critical finding{'s were' if critical_count != 1 else ' was'} "
            f"identified, requiring immediate engineering intervention and operational "
            f"restriction to prevent escalating safety and structural risk."
        )

    if high_count > 0:
        parts.append(
            f"{high_count} high-priority deviation{'s were' if high_count != 1 else ' was'} "
            f"detected, requiring accelerated corrective action within the next "
            f"operational cycle."
        )

    if critical_count == 0 and high_count == 0 and medium_count > 0:
        parts.append(
            f"{medium_count} moderate deviation{'s were' if medium_count != 1 else ' was'} "
            f"identified, requiring structured corrective maintenance within the "
            f"next scheduled operational period."
        )

    if top_display:
        if len(top_display) == 1:
            issue_str = top_display[0]
        elif len(top_display) == 2:
            issue_str = f"{top_display[0]} and {top_display[1]}"
        else:
            issue_str = (
                f"{top_display[0]}, {top_display[1]}, and {top_display[2]}"
            )
        parts.append(
            f"The predominant issue categories identified include {issue_str}."
        )

    # ── Intelligence layer: common issue, zone, severity ─
    if common_issue:
        parts.append(
            f"The most frequently occurring finding type across the inspection "
            f"dataset is {common_issue}, indicating a recurring quality control "
            f"gap that warrants systemic corrective intervention."
        )

    if affected_zone:
        parts.append(
            f"The most significantly affected spatial area is the "
            f"{affected_zone}, where finding concentration is highest and "
            f"targeted remediation should be prioritised."
        )

    if highest_sev and highest_sev != common_issue:
        parts.append(
            f"The highest-severity condition identified during this inspection "
            f"is {highest_sev}, which must be addressed as a primary engineering "
            f"priority before lower-severity items are scheduled."
        )

    # ── Site condition summary ────────────────────────────
    parts.append(site_condition)

    if risk == "Critical":
        closing = (
            f"The overall compliance score of {overall_score}/100 reflects a "
            f"{status} status with Critical risk exposure. Immediate engineering "
            f"review, operational restrictions, and executive corrective oversight "
            f"are mandatory before resuming full operational activities."
        )
    elif risk == "High":
        closing = (
            f"The overall compliance score of {overall_score}/100 indicates a "
            f"{status} status with High risk exposure. Accelerated corrective "
            f"action planning and engineering supervision are strongly recommended "
            f"to prevent further infrastructure degradation."
        )
    elif risk == "Medium":
        closing = (
            f"The overall compliance score of {overall_score}/100 indicates a "
            f"{status} status with Medium risk exposure. Structured corrective "
            f"and preventive maintenance workflows should be initiated within "
            f"the next operational period."
        )
    else:
        closing = (
            f"The overall compliance score of {overall_score}/100 demonstrates "
            f"a {status} status with Low risk exposure. Continued adherence to "
            f"preventive maintenance schedules will sustain current compliance "
            f"performance."
        )

    parts.append(closing)

    if common_issue:
        parts.append(
            f"Recurring issue analysis indicates {common_issue} as the "
            f"dominant site-wide quality concern requiring management "
            f"attention."
        )

    return " ".join(parts)


# =========================================================
# CONCLUSION BUILDER
# =========================================================
def build_conclusion(image_reports: list, overall_score: int) -> str:
    total_findings = sum(r["total_findings"] for r in image_reports)
    risk           = risk_level_from_score(overall_score)
    status         = compliance_status(overall_score)
    grade          = inspection_grade(overall_score)
    n_images       = len(image_reports)

    critical_images = [
        r["image_label"]
        for r in image_reports
        if r["severity_breakdown"].get("Critical", 0) > 0
    ]
    high_images = [
        r["image_label"]
        for r in image_reports
        if r["severity_breakdown"].get("High", 0) > 0
           and r["image_label"] not in critical_images
    ]

    conclusion = (
        f"This InfraGuard AI inspection identified {total_findings} total quality "
        f"deviation{'s' if total_findings != 1 else ''} across {n_images} "
        f"inspected image{'s' if n_images != 1 else ''}. "
        f"The overall infrastructure quality rating is Grade {grade} with a "
        f"compliance score of {overall_score}/100 ({status}) and a {risk} risk "
        f"classification."
    )

    if critical_images:
        conclusion += (
            f" Critical conditions requiring immediate engineering intervention "
            f"were identified in: {', '.join(critical_images)}."
        )

    if high_images:
        conclusion += (
            f" High-priority deviations requiring accelerated corrective action "
            f"were identified in: {', '.join(high_images)}."
        )

    if risk == "Critical":
        conclusion += (
            " All critical corrective actions must be implemented immediately "
            "under qualified engineering supervision. Operational restrictions "
            "must remain in place until written engineering clearance is obtained. "
            "A formal follow-up inspection is required within 7 days to verify "
            "remediation effectiveness and confirm structural safety."
        )
    elif risk == "High":
        conclusion += (
            " All identified corrective actions must be implemented and verified "
            "by qualified engineering personnel within 14 days. A follow-up "
            "inspection is recommended within 7 to 14 days to confirm remediation "
            "compliance and assess any residual risk."
        )
    elif risk == "Medium":
        conclusion += (
            " Corrective and preventive actions should be implemented within the "
            "next operational period and verified by qualified site personnel. "
            "A follow-up inspection is recommended within 30 days to confirm "
            "remediation status and sustained compliance."
        )
    else:
        conclusion += (
            " Continue standard preventive maintenance and inspection workflows. "
            "A routine follow-up inspection within 90 days is recommended to "
            "verify sustained compliance performance."
        )

    return conclusion


# =========================================================
# MASTER REPORT GENERATOR  (PRIMARY PUBLIC API)
# =========================================================
def generate_multi_image_report(
    images_data:      list,
    inspection_id:    str = None,
    project_name:     str = "Construction Site Inspection",
    inspection_date:  str = None,
) -> dict:
    """
    Generate a complete enterprise inspection report across
    multiple images.

    Parameters
    ----------
    images_data : list of dict
        Each element describes one uploaded image:
            {
                "image_index":          int,
                "image_label":          str,
                "image_path":           str,
                "annotated_image_path": str,
                "detections":           list,
                "location":             str,
            }

    Returns
    -------
    dict — complete structured report consumed by
           pdf_service.generate_quality_pdf
    """
    if not inspection_id:
        inspection_id = f"IG-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    if not inspection_date:
        inspection_date = datetime.utcnow().strftime("%d %b %Y")

    image_reports     = [build_image_report(img) for img in images_data]

    all_detections    = [d for img in images_data for d in img.get("detections", [])]
    overall_score     = compute_compliance_score(all_detections)
    overall_breakdown = severity_breakdown(all_detections)
    overall_risk      = risk_level_from_score(overall_score)
    overall_status    = compliance_status(overall_score)
    overall_grade     = inspection_grade(overall_score)
    overall_recs      = aggregate_overall_recommendations(image_reports)

    executive_summary = build_executive_summary(image_reports, overall_score)
    site_executive_narrative = build_site_executive_narrative(image_reports, overall_score)
    site_summary      = build_site_summary(image_reports, overall_score)
    priority_action   = build_priority_action(overall_risk)
    conclusion        = build_conclusion(image_reports, overall_score)

    pattern_analysis = build_pattern_analysis(image_reports)

    management_attention = build_management_attention(image_reports)

    systemic_deficiencies = build_systemic_deficiencies(image_reports)

    audit_exposure_summary = build_audit_exposure_summary(
        image_reports,
        overall_score,
    )

    # ── Risk matrix (A / B / C finding counts) ────────────
    risk_matrix = {"A": 0, "B": 0, "C": 0}
    for img in image_reports:
        for f in img["findings"]:
            rc = f.get("risk_category", "C")
            if rc not in risk_matrix:
                rc = "C"
            risk_matrix[rc] += 1

    total_findings    = sum(r["total_findings"] for r in image_reports)

    ai_findings_bullets = build_ai_findings_bullets(
        image_reports, total_findings
    )

    # ── Audit readiness ───────────────────────────────────
    critical_total = overall_breakdown.get("Critical", 0)

    if overall_score >= 85 and critical_total == 0:
        audit_readiness = "Audit Ready"
    elif overall_score >= 65 and critical_total == 0:
        audit_readiness = "Conditionally Audit Ready — Remediation Required"
    elif critical_total > 0:
        audit_readiness = (
            f"Not Audit Ready — {critical_total} Critical "
            f"Finding{'s' if critical_total != 1 else ''} Require Immediate Resolution"
        )
    else:
        audit_readiness = "Not Audit Ready — Immediate Remediation Required"

    # ── Follow-up recommendation ──────────────────────────
    if overall_risk == "Critical":
        follow_up = "Mandatory follow-up inspection required within 7 days."
    elif overall_risk == "High":
        follow_up = "Follow-up inspection required within 14 days."
    elif overall_risk == "Medium":
        follow_up = "Follow-up inspection recommended within 30 days."
    else:
        follow_up = "Routine follow-up inspection within 90 days."

    # ── Frontend Phase 7 fields (GenerateReport.jsx) ───────
    benchmark      = compliance_benchmark(overall_score)
    audit_status_  = audit_status(overall_score, critical_total)
    op_status       = operational_status(overall_risk)

    # ── Frontend Phase 8 field — AI confidence analytics ───
    analytics = analytics_summary(image_reports)

    return {
        # ── Cover / Identity ──────────────────────────────
        "inspection_id":          inspection_id,
        "project_name":           project_name,
        "inspection_date":        inspection_date,
        "generated_at":           datetime.utcnow().strftime("%d %b %Y %H:%M UTC"),
        "inspection_type":        "Construction Quality Assurance",
        "processing_engine":      "InfraGuard Enterprise AI",

        # ── Report header fields ──────────────────────────
        "overall_status":         overall_status,
        "compliance_score":       overall_score,
        "total_findings":         total_findings,

        # ── Section 1: Executive Summary ──────────────────
        "site_summary":           site_summary,
        "executive_summary":      executive_summary,
        "site_executive_narrative": site_executive_narrative,
        "priority_action":        priority_action,

        # ── Section 2: AI Findings bullets ────────────────
        "ai_findings_bullets":    ai_findings_bullets,

        # ── Dashboard KPIs (legacy / downstream consumers) ─
        "total_images_inspected": len(image_reports),
        "overall_risk":           overall_risk,
        "inspection_grade":       overall_grade,
        "severity_breakdown":     overall_breakdown,

        # ── Per-Image Sections ────────────────────────────
        "image_reports":          image_reports,
        # Alias: GenerateReport.jsx reads data.images[] — keeping both
        # keys means neither side has to be the one that adapts.
        "images":                 image_reports,
        # Alias: GenerateReport.jsx reads data.total_images_processed.
        "total_images_processed": len(image_reports),

        # ── Aggregated Recommendations ────────────────────
        "overall_recommendations": overall_recs,

        # ── Site Pattern & Management Intelligence ────────
        "site_pattern_analysis":     pattern_analysis,
        "management_attention_areas": management_attention,
        "systemic_deficiencies":     systemic_deficiencies,
        "audit_exposure_summary":    audit_exposure_summary,
        "risk_matrix":               risk_matrix,

        # ── Compliance Summary ────────────────────────────
        "audit_readiness":        audit_readiness,
        "follow_up_action":       follow_up,

        # ── Frontend Phase 7 — Compliance Benchmark Panel ──
        "compliance_benchmark":   benchmark,
        "benchmark":              benchmark,
        "audit_status":           audit_status_,
        "operational_status":     op_status,

        # ── Frontend Phase 8 — AI Confidence Analytics ─────
        "analytics":              analytics,

        # ── Conclusion ────────────────────────────────────
        "conclusion":             conclusion,
    }


# =========================================================
# BACKWARD-COMPATIBLE WRAPPER  (single-image callers)
# =========================================================
def generate_report(detections: list) -> list:
    """
    Legacy single-image interface kept for backward compatibility.
    Returns a flat list of finding dicts identical to the v1 shape.
    """
    total = len(detections)
    return sort_findings([build_finding(d, total_findings=total) for d in detections])