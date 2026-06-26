from typing import List, Dict, Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)

# -----------------------------------------------------
# CLASS ID MAP (YOLO Class Mapping)
# Maps raw model class names → canonical internal names.
# Source: actual trained YOLO model classes.
# -----------------------------------------------------

CLASS_ID_MAP: Dict[int, str] = {
    0:  "Helmet",
    1:  "boots",
    2:  "glove",
    3:  "head",
    4:  "helmet",
    5:  "no helmet",
    6:  "no vest",
    7:  "person",
    8:  "vest",
    9:  "vests",
    10: "Bulldozer",
    11: "Dump Truck",
    12: "Excavator",
    13: "Grader",
    14: "Loader",
    15: "Mixer Truck",
    16: "Mobile Crane",
    17: "Roller",
    18: "crack",
}

# Canonical label normalizer.
LABEL_NORMALIZE: Dict[str, str] = {
    "Helmet":       "helmet",
    "helmet":       "helmet",
    "no helmet":    "no_helmet",
    "head":         "head",

    "vest":         "vest",
    "vests":        "vest",
    "no vest":      "no_vest",

    "boots":        "boots",
    "glove":        "gloves",

    "person":       "person",

    "Bulldozer":    "bulldozer",
    "Dump Truck":   "dump_truck",
    "Excavator":    "excavator",
    "Grader":       "grader",
    "Loader":       "loader",
    "Mixer Truck":  "mixer_truck",
    "Mobile Crane": "mobile_crane",
    "Roller":       "roller",

    "crack":        "crack",
}


def normalize_class_name(raw: str) -> str:
    """Return the canonical internal label for a raw YOLO class name."""
    return LABEL_NORMALIZE.get(raw, raw.lower())


# -----------------------------------------------------
# SHARED CONSTANTS
# Single source of truth consumed by detection_service.py,
# violation_analyzer.py, alert_service.py, analytics_service.py,
# and the React dashboard.
# -----------------------------------------------------

PPE_VIOLATIONS: Dict[str, str] = {
    "no_helmet": "HIGH",
    "no_vest":   "MEDIUM",
    "no_boots":  "LOW",
    "no_gloves": "LOW",
}

PPE_ITEM_NAMES: Dict[str, str] = {
    "no_helmet": "helmet",
    "no_vest":   "vest",
    "no_boots":  "boots",
    "no_gloves": "gloves",
}

SEVERITY_ORDER: Dict[str, int] = {
    "SAFE":     0,
    "LOW":      1,
    "MEDIUM":   2,
    "HIGH":     3,
    "CRITICAL": 4,
}


# -----------------------------------------------------
# SITE PROFILES
# -----------------------------------------------------

SITE_PROFILES: Dict[str, Dict] = {
    "construction": {
        "critical_ppe":   {"helmet", "vest"},
        "important_ppe":  {"boots", "gloves"},
        "machine_classes": {
            "bulldozer", "dump_truck", "excavator",
            "grader", "loader", "mixer_truck",
            "mobile_crane", "roller",
        },
    },
    "factory": {
        "critical_ppe":   {"helmet", "gloves"},
        "important_ppe":  {"boots"},
        "machine_classes": {"loader", "roller"},
    },
    "warehouse": {
        "critical_ppe":   {"vest"},
        "important_ppe":  {"helmet", "boots"},
        "machine_classes": {"loader", "dump_truck"},
    },
}

CRITICAL_PPE:    set = SITE_PROFILES["construction"]["critical_ppe"]
IMPORTANT_PPE:   set = SITE_PROFILES["construction"]["important_ppe"]
MACHINE_CLASSES: set = SITE_PROFILES["construction"]["machine_classes"]


# -----------------------------------------------------
# Fix 7: Dynamic danger radius per machine type
# Different machines have different safety zones.
# -----------------------------------------------------

MACHINE_DANGER_RADIUS: Dict[str, int] = {
    "roller":       120,
    "loader":       250,
    "bulldozer":    300,
    "grader":       300,
    "dump_truck":   350,
    "mixer_truck":  350,
    "mobile_crane": 400,
    "excavator":    450,
}
DEFAULT_DANGER_RADIUS = 300  # fallback for unknown machines

# Intrinsic risk weight per machine type (added to proximity score).
# Combined with distance-based scoring in detect_danger_zones.
MACHINE_RISK: Dict[str, int] = {
    "roller":       20,
    "loader":       30,
    "bulldozer":    40,
    "grader":       40,
    "dump_truck":   40,
    "mixer_truck":  40,
    "excavator":    50,
    "mobile_crane": 60,
}

DEFAULT_DANGER_ZONES = [
    {
        "name": "Heavy Equipment Zone",
        "risk": "HIGH",
        "polygon": [],          # replace with real zone coordinates when available
        "machine_types": list(SITE_PROFILES["construction"]["machine_classes"]),
    }
]

# -----------------------------------------------------
# Utility Functions
# -----------------------------------------------------

def bbox_center(box: List[float]) -> Tuple[float, float]:
    """Compute bounding box center."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def bbox_dimensions(box: List[float]) -> Tuple[float, float]:
    """Return (width, height) of a bounding box."""
    x1, y1, x2, y2 = box
    return abs(x2 - x1), abs(y2 - y1)


def euclidean_distance(
        a: Tuple[float, float],
        b: Tuple[float, float]
) -> float:
    """Compute Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# -----------------------------------------------------
# Phase 1 — Risk Score Engine
#
# Replaces the old fixed-rule lookup ("missing helmet -> HIGH") with an
# additive point system. Every hazard contributes points; the sum maps
# to a risk level via RISK_BANDS. This makes risk continuous and lets
# multiple smaller issues combine into something bigger (Combined
# Hazard Scoring) rather than each hazard being judged in isolation.
# -----------------------------------------------------

# Points contributed by each individual hazard / missing item.
HAZARD_SCORES: Dict[str, int] = {
    "missing_helmet":  40,
    "missing_vest":    30,
    "missing_boots":   20,
    "missing_gloves":  10,
    "danger_zone":      50,
    "machine_nearby":   30,
    "crack":            60,
}

# Risk level bands, evaluated low -> high. Each entry is
# (inclusive_lower_bound, label). The band whose lower bound is the
# highest value <= the score wins.
RISK_BANDS: List[Tuple[int, str]] = [
    (0,   "SAFE"),
    (21,  "LOW"),
    (41,  "MEDIUM"),
    (71,  "HIGH"),
    (91,  "CRITICAL"),
]

# Display label per risk level, used alongside the numeric score.
SEVERITY_LABELS: Dict[str, str] = {
    "SAFE":     "Safe",
    "LOW":      "Low",
    "MEDIUM":   "Medium",
    "HIGH":     "High",
    "CRITICAL": "Critical",
}

# Legacy map kept for backward compatibility with callers that used
# (numeric_score, label) tuples. New code should prefer RISK_BANDS.
SEVERITY_MAP: Dict[str, Tuple[int, str]] = {
    "SAFE":     (0,   "Safe"),
    "LOW":      (20,  "Low"),
    "MEDIUM":   (45,  "Medium"),
    "HIGH":     (70,  "High"),
    "CRITICAL": (100, "Critical"),
}

# Site-wide escalation: number of HIGH-or-above individual workers
# required to bump the whole-image risk level up to CRITICAL, even if
# no single worker individually scored as CRITICAL.
HIGH_WORKER_ESCALATION_THRESHOLD = 3


def risk_level_for_score(score: int) -> str:
    """
    Map an additive hazard score to a risk level using RISK_BANDS.

    0-20 SAFE, 21-40 LOW, 41-70 MEDIUM, 71-90 HIGH, 91+ CRITICAL.
    """
    level = RISK_BANDS[0][1]
    for lower_bound, label in RISK_BANDS:
        if score >= lower_bound:
            level = label
        else:
            break
    return level


def compute_severity(risk_level: str) -> Tuple[int, str]:
    """
    Convert a risk level string to a representative numeric score
    (the band's lower bound) and a display label.

    Kept for callers that only have a risk level string on hand
    (e.g. the crack path, which is always forced to CRITICAL).
    """
    for lower_bound, label in RISK_BANDS:
        if label == risk_level:
            return lower_bound, SEVERITY_LABELS.get(risk_level, "Safe")
    return 0, "Safe"


def risk_level_rank(risk_level: str) -> int:
    """Ordinal rank of a risk level, for comparisons (SAFE < LOW < ... < CRITICAL)."""
    for i, (_, label) in enumerate(RISK_BANDS):
        if label == risk_level:
            return i
    return 0


def max_risk_level(*levels: str) -> str:
    """Return whichever of the given risk levels ranks highest."""
    return max(levels, key=risk_level_rank) if levels else "SAFE"


# -----------------------------------------------------
# IoU Calculation
# -----------------------------------------------------

def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Intersection over Union for two bounding boxes."""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_width  = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area   = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


# -----------------------------------------------------
# Fix 9: Detection Validation
# Validate every detection before processing.
# -----------------------------------------------------

def validate_detection(det: Dict, confidence_threshold: float = 0.3) -> Tuple[bool, str]:
    """
    Validate a single detection dict has all required fields and passes
    minimum quality thresholds.

    Returns:
        (valid: bool, reason: str)
    """
    if "bbox" not in det:
        return False, "missing bbox"

    if "class_name" not in det:
        return False, "missing class_name"

    if "confidence" not in det:
        return False, "missing confidence"

    bbox = det["bbox"]
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return False, "bbox must be a 4-element list"

    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return False, f"bbox area is {area} (must be > 0)"

    if det["confidence"] < confidence_threshold:
        return False, f"confidence {det['confidence']:.2f} below threshold {confidence_threshold}"

    return True, "ok"


def filter_valid_detections(
        detections: List[Dict],
        confidence_threshold: float = 0.3
) -> List[Dict]:
    """
    Return only detections that pass validation.
    Logs every rejection with the reason.
    """
    valid = []
    for i, det in enumerate(detections):
        ok, reason = validate_detection(det, confidence_threshold)
        if ok:
            valid.append(det)
        else:
            label = det.get("class_name", "unknown")
            logger.warning("Detection #%d (%s) rejected: %s", i, label, reason)
    return valid


# -----------------------------------------------------
# Fix 1 + 4: Helmet — upper body region check
# Helmet center must fall inside the top 30% of the person bbox,
# AND IoU must exceed threshold.
# -----------------------------------------------------

def _helmet_in_upper_body(person_box: List[float], helmet_box: List[float]) -> bool:
    """
    Return True if the helmet center falls within the top 30% of person bbox.
    """
    x1, y1, x2, y2 = person_box
    height = y2 - y1
    upper_limit = y1 + height * 0.30

    hx, hy = bbox_center(helmet_box)
    return y1 <= hy <= upper_limit and x1 <= hx <= x2


# -----------------------------------------------------
# Fix 3: Vest — middle body region check
# Vest center must fall in the middle 30–70% vertically.
# -----------------------------------------------------

def _vest_in_middle_body(person_box: List[float], vest_box: List[float]) -> bool:
    """
    Return True if the vest center falls within the middle body region (30–70%).
    """
    x1, y1, x2, y2 = person_box
    height = y2 - y1
    mid_top    = y1 + height * 0.30
    mid_bottom = y1 + height * 0.70

    vx, vy = bbox_center(vest_box)
    return mid_top <= vy <= mid_bottom and x1 <= vx <= x2


# -----------------------------------------------------
# Fix 2: Boots — bottom 25% region check
# Boot center must fall in the bottom 25% of person bbox.
# -----------------------------------------------------

def _boots_in_lower_body(person_box: List[float], boot_box: List[float]) -> bool:
    """
    Return True if the boot center falls within the bottom 25% of person bbox.
    """
    x1, y1, x2, y2 = person_box
    height = y2 - y1
    foot_start = y1 + height * 0.75

    bx, by = bbox_center(boot_box)
    return by >= foot_start and x1 <= bx <= x2


# -----------------------------------------------------
# Fix 1: PPE Association — spatial region + IoU
# Each PPE type is matched using its appropriate body region
# AND an IoU overlap check. IoU-only is no longer used.
# -----------------------------------------------------

# PPE labels that require special region-aware matching.
_REGION_MATCHERS = {
    "helmet": _helmet_in_upper_body,
    "vest":   _vest_in_middle_body,
    "boots":  _boots_in_lower_body,
}

# Gloves: no strong region constraint — use IoU only with a tighter threshold.
_GLOVE_IOC_THRESHOLD = 0.05


def associate_ppe_to_person(
        person_box: List[float],
        detections: List[Dict],
        iou_threshold: float = 0.1
) -> Dict[str, bool]:
    """
    Associate PPE items with a person using body-region checks AND IoU.

    Rules:
      - helmet  → must be in top 30% of person bbox AND IoU > threshold
      - vest    → must be in middle 30–70% AND IoU > threshold
      - boots   → must be in bottom 25% AND IoU > threshold
      - gloves  → IoU > (smaller) threshold only
      - other   → IoU > threshold (machines, unknown labels are skipped)

    Returns a dict mapping each tracked PPE item to True/False, e.g.:
        {"helmet": True, "vest": False, "boots": True, "gloves": False}
    """
    detected: set = set()

    for det in detections:
        label = normalize_class_name(det["class_name"])

        # Skip non-PPE labels and persons.
        if label in {"person"} | MACHINE_CLASSES | {"crack", "no_helmet", "no_vest",
                                                      "head", "unknown"}:
            continue

        ppe_box = det["bbox"]
        iou     = compute_iou(person_box, ppe_box)

        if label in _REGION_MATCHERS:
            region_ok = _REGION_MATCHERS[label](person_box, ppe_box)
            if region_ok and iou > iou_threshold:
                detected.add(label)
        elif label == "gloves":
            if iou > _GLOVE_IOC_THRESHOLD:
                detected.add(label)
        else:
            # Unknown PPE-like label — fall back to IoU only.
            if iou > iou_threshold:
                detected.add(label)

    # Return a structured dict for all tracked items (enables UI reports
    # and detailed analytics without callers having to check set membership).
    return {item: (item in detected) for item in _COMPLIANCE_ITEMS}


# -----------------------------------------------------
# PPE Violation Detection (Phase 1: score-based)
# -----------------------------------------------------

# The four PPE items tracked for compliance %, each worth an equal
# 25% share (Worker Compliance Percentage).
_COMPLIANCE_ITEMS: Tuple[str, ...] = ("helmet", "vest", "boots", "gloves")
_COMPLIANCE_SHARE = 100 / len(_COMPLIANCE_ITEMS)  # 25% each


def compute_worker_compliance(assigned: set) -> Dict:
    """
    Return PPE compliance details for a worker.

    Returns a dict:
        percentage  – 0-100 rounded to nearest integer
        worn        – number of tracked items present
        required    – total number of tracked items
        missing     – list of items not detected

    e.g. helmet + vest present, boots + gloves missing ->
         {"percentage": 50, "worn": 2, "required": 4, "missing": ["boots", "gloves"]}
    """
    worn    = sum(1 for item in _COMPLIANCE_ITEMS if item in assigned)
    missing = [item for item in _COMPLIANCE_ITEMS if item not in assigned]
    return {
        "percentage": round(worn * _COMPLIANCE_SHARE),
        "worn":       worn,
        "required":   len(_COMPLIANCE_ITEMS),
        "missing":    missing,
    }


def detect_ppe_violations(detections: List[Dict]) -> Dict:
    """
    Evaluate per-person PPE compliance using the additive risk score
    engine (Phase 1).

    For each worker:
      - Missing helmet  -> +40
      - Missing vest    -> +30
      - Missing boots   -> +20
      - Missing gloves  -> +10
      These stack (Combined Hazard Scoring): e.g. missing both helmet
      and vest contributes 40 + 30 = 70 before any proximity/danger
      points are added later in evaluate_risk.

    class_name may be raw YOLO labels; normalization applied internally.
    """
    persons = [
        d for d in detections
        if normalize_class_name(d["class_name"]) == "person"
    ]

    report      = []
    image_score = 0

    for idx, person in enumerate(persons):
        assigned_dict = associate_ppe_to_person(person["bbox"], detections)
        # Convert to set for existing set-arithmetic (missing = required - present)
        assigned = {item for item, present in assigned_dict.items() if present}

        missing_critical  = CRITICAL_PPE  - assigned
        missing_important = IMPORTANT_PPE - assigned
        missing_all        = sorted(missing_critical | missing_important)

        # Fix 10: Structured reason list for UI and debugging.
        reasons     = []
        person_score = 0

        for item in sorted(missing_critical | missing_important):
            hazard_key = f"missing_{item}"
            points = HAZARD_SCORES.get(hazard_key, 0)
            person_score += points
            reasons.append(f"Missing {item.capitalize()} (+{points})")

        if not reasons:
            reasons.append("All required PPE detected")

        # Multiple-PPE escalation: 3 or more missing items -> +20 bonus.
        if len(missing_all) >= 3:
            person_score += 20
            reasons.append("Multiple PPE violations (+20)")

        risk       = risk_level_for_score(person_score)
        compliance = compute_worker_compliance(assigned)

        image_score = max(image_score, person_score)

        report.append({
            "person_id":         idx,
            "risk":              risk,
            "risk_score":        person_score,
            "compliance_pct":    compliance["percentage"],
            "compliance":        compliance,
            "missing":           missing_all,
            "assigned_ppe":      assigned_dict,          # full dict: {"helmet": True, ...}
            "assigned_ppe_list": sorted(assigned),       # backward-compat list
            "reasons":           reasons,
            # Keep single-string reason for backward compatibility.
            "reason":            "; ".join(reasons),
        })

    # ── Multiple Worker Escalation ──────────────────────────────────────
    # Site-wide risk isn't just "the worst single worker" — if enough
    # workers are independently at HIGH risk or above, the whole image
    # escalates to CRITICAL even though no individual worker may have
    # scored that high on their own.
    high_or_above_count = sum(
        1 for p in report if risk_level_rank(p["risk"]) >= risk_level_rank("HIGH")
    )

    image_risk = risk_level_for_score(image_score)
    if high_or_above_count >= HIGH_WORKER_ESCALATION_THRESHOLD:
        image_risk = "CRITICAL"
    elif high_or_above_count >= 1:
        image_risk = max_risk_level(image_risk, "HIGH")

    return {
        "image_risk":          image_risk,
        "image_score":         image_score,
        "high_worker_count":   high_or_above_count,
        "persons":             report,
    }


# -----------------------------------------------------
# Fix 5 + 6: Crack Detection — structural defect path
# Cracks are structural, not PPE events.
# They skip PPE logic and return HIGH risk immediately.
# -----------------------------------------------------

def detect_cracks(detections: List[Dict]) -> List[Dict]:
    """
    Return all crack detections found in the frame.
    """
    return [
        d for d in detections
        if normalize_class_name(d["class_name"]) == "crack"
    ]


# -----------------------------------------------------
# Worker–Machine Proximity Detection
# -----------------------------------------------------

def detect_vehicle_proximity(
        detections: List[Dict],
        threshold: int = 300
) -> List[Dict]:
    """
    Warn when a worker is within `threshold` pixels of any machine.
    """
    persons  = [
        d for d in detections
        if normalize_class_name(d["class_name"]) == "person"
    ]
    machines = [
        d for d in detections
        if normalize_class_name(d["class_name"]) in MACHINE_CLASSES
    ]

    violations = []

    for person in persons:
        pc = bbox_center(person["bbox"])

        for machine in machines:
            mc       = bbox_center(machine["bbox"])
            distance = euclidean_distance(pc, mc)

            if distance < threshold:
                machine_label = normalize_class_name(machine["class_name"])
                violations.append({
                    "type":     "worker_near_machine",
                    "machine":  machine_label,
                    "distance": int(distance),
                    "reason":   (
                        f"Worker is {int(distance)}px from {machine_label} "
                        f"(proximity threshold: {threshold}px)"
                    ),
                })

    return violations


# -----------------------------------------------------
# Fix 7 + 8: Danger Zone Detection — adaptive radius
# Each machine type uses its own safety radius.
# Radius is further expanded by machine physical size.
# -----------------------------------------------------

def _adaptive_danger_radius(machine_label: str, machine_box: List[float]) -> int:
    """
    Compute the effective danger radius for a machine.

    Base radius comes from MACHINE_DANGER_RADIUS per machine type.
    An additional term proportional to machine size is added so that
    physically larger machines have correspondingly larger safety zones.

    effective_radius = base_radius + 0.25 * (machine_width + machine_height)
    """
    base   = MACHINE_DANGER_RADIUS.get(machine_label, DEFAULT_DANGER_RADIUS)
    w, h   = bbox_dimensions(machine_box)
    bonus  = 0.25 * (w + h)
    return int(base + bonus)


def detect_danger_zones(detections: List[Dict]) -> List[Dict]:
    """
    Flag workers inside the adaptive danger radius of any machine.

    Proximity tiers (Change 10):
      distance < 50 px              -> CRITICAL
      distance < radius             -> HIGH
      distance < radius + 40        -> MEDIUM

    Machine intrinsic severity (MACHINE_RISK) is included in the alert
    so downstream services can weight it further.
    """
    persons  = [
        d for d in detections
        if normalize_class_name(d["class_name"]) == "person"
    ]
    machines = [
        d for d in detections
        if normalize_class_name(d["class_name"]) in MACHINE_CLASSES
    ]

    alerts = []

    for machine in machines:
        machine_label = normalize_class_name(machine["class_name"])
        mc            = bbox_center(machine["bbox"])
        radius        = _adaptive_danger_radius(machine_label, machine["bbox"])
        machine_sev   = MACHINE_RISK.get(machine_label, 30)

        for person in persons:
            pc       = bbox_center(person["bbox"])
            distance = euclidean_distance(pc, mc)

            if distance < 50:
                tier = "CRITICAL"
                reason = (
                    f"Worker entered {machine_label} safety radius "
                    f"({int(distance)}px — immediate danger)"
                )
            elif distance < radius:
                tier = "HIGH"
                reason = (
                    f"Worker inside {machine_label} danger zone "
                    f"({int(distance)}px / radius {radius}px)"
                )
            elif distance < radius + 40:
                tier = "MEDIUM"
                reason = (
                    f"Worker approaching {machine_label} danger zone "
                    f"({int(distance)}px / radius {radius}px)"
                )
            else:
                continue

            alerts.append({
                "type":           "danger_zone",
                "machine":        machine_label,
                "machine_risk":   machine_sev,
                "distance":       int(distance),
                "danger_radius":  radius,
                "tier":           tier,
                "reason":         reason,
            })

    return alerts


# -----------------------------------------------------
# Fix 6: Crack-first evaluation gate
# -----------------------------------------------------

def _build_crack_result(cracks: List[Dict]) -> Dict:
    """
    Build the risk result dict for a frame containing structural cracks.
    PPE and proximity checks are skipped entirely.

    Each crack contributes its hazard score (+60); multiple cracks
    stack additively, same as any other hazard (Combined Hazard
    Scoring), so e.g. 2 cracks = 120 -> CRITICAL.
    """
    crack_points = HAZARD_SCORES.get("crack", 60)
    score = len(cracks) * crack_points

    reasons = [
        f"Structural crack detected (+{crack_points}) (bbox: {c['bbox']})"
        for c in cracks
    ]
    risk_level = risk_level_for_score(score)
    _, severity_label = compute_severity(risk_level)

    return {
        # ── Unified contract ───────────────────────────────────────────────
        "detections":   cracks,   # only crack detections in this path
        "workers":      [],
        # ── Core risk output ───────────────────────────────────────────────
        "risk_level":   risk_level,
        "risk_score":   score,
        "severity":     severity_label,
        "crack":        True,
        "crack_count":  len(cracks),
        # Fix 10: reasons list.
        "reasons":      reasons,
        "reason":       "; ".join(reasons),
        "ppe":          {"image_risk": risk_level, "image_score": score, "persons": []},
        "proximity":    [],
        "danger_zones": [],
    }


# -----------------------------------------------------
# MAIN RISK EVALUATION
# -----------------------------------------------------

def evaluate_risk(
        detections: List[Dict],
        confidence_threshold: float = 0.3
) -> Dict:
    """
    Full risk evaluation pipeline (Phase 1: additive risk score engine).

    Evaluation order (Fix 6):
      1. Validate all detections (Fix 9).
      2. Check for structural cracks -> if found, score them directly
         and return CRITICAL/whatever band they land in (Fix 5).
      3. PPE compliance check, scored per missing item (Fix 1-4).
      4. Worker-machine proximity check (+30 per nearby machine).
      5. Danger zone check with adaptive radius (+50 per zone) (Fix 7-8).
      6. Combine hazards per worker additively (Combined Hazard
         Scoring) — e.g. missing helmet (+40) + danger zone (+50) +
         machine nearby (+30) = 120 -> CRITICAL for that worker.
      7. Escalate site-wide risk if multiple workers are independently
         HIGH or above (Multiple Worker Escalation).

    Args:
        detections:           Raw YOLO detections. Each must have
                              'class_name', 'bbox', 'confidence'.
        confidence_threshold: Minimum confidence to accept a detection.

    Returns:
        Dict with risk_level, risk_score, severity, reasons, ppe,
        proximity, danger_zones, and pipeline_stats.
    """
    # ── Fix 9: Validate every detection first ──────────────────────────────
    valid_detections = filter_valid_detections(detections, confidence_threshold)

    pipeline_stats = {
        "detections_in":      len(detections),
        "detections_valid":   len(valid_detections),
        "detections_rejected": len(detections) - len(valid_detections),
    }

    # ── Fix 5 + 6: Crack gate ──────────────────────────────────────────────
    cracks = detect_cracks(valid_detections)
    if cracks:
        result = _build_crack_result(cracks)
        result["pipeline_stats"] = pipeline_stats
        return result

    # ── PPE, proximity, danger zones ───────────────────────────────────────
    ppe       = detect_ppe_violations(valid_detections)
    proximity = detect_vehicle_proximity(valid_detections)
    danger    = detect_danger_zones(valid_detections)

    machine_nearby_points = HAZARD_SCORES.get("machine_nearby", 30)
    danger_zone_points    = HAZARD_SCORES.get("danger_zone", 50)

    # ── Combined Hazard Scoring ─────────────────────────────────────────────
    # Add each worker's own proximity/danger-zone hits on top of their
    # PPE score, so e.g. "Helmet Missing + Danger Zone + Machine Nearby"
    # stacks into one combined per-worker score instead of three
    # separately-judged events.
    #
    # detect_vehicle_proximity / detect_danger_zones don't currently
    # tag which person they belong to, so we recompute per-person
    # counts here using the same distance logic, keyed by person index
    # in ppe["persons"] order (same iteration order as detect_ppe_violations).
    persons = [
        d for d in valid_detections
        if normalize_class_name(d["class_name"]) == "person"
    ]
    machines = [
        d for d in valid_detections
        if normalize_class_name(d["class_name"]) in MACHINE_CLASSES
    ]

    for idx, person_entry in enumerate(ppe["persons"]):
        person_box = persons[idx]["bbox"]
        pc = bbox_center(person_box)

        nearby_count = 0
        in_danger_zone = False

        for machine in machines:
            machine_label = normalize_class_name(machine["class_name"])
            mc       = bbox_center(machine["bbox"])
            distance = euclidean_distance(pc, mc)

            if distance < 300:  # matches detect_vehicle_proximity default
                nearby_count += 1

            radius = _adaptive_danger_radius(machine_label, machine["bbox"])
            if distance < radius:
                in_danger_zone = True

        if nearby_count:
            added = nearby_count * machine_nearby_points
            person_entry["risk_score"] += added
            person_entry["reasons"].append(
                f"Machine nearby x{nearby_count} (+{added})"
            )
        if in_danger_zone:
            person_entry["risk_score"] += danger_zone_points
            person_entry["reasons"].append(
                f"In machine danger zone (+{danger_zone_points})"
            )

        person_entry["reason"] = "; ".join(person_entry["reasons"])
        person_entry["risk"] = risk_level_for_score(person_entry["risk_score"])

    # ── Re-derive site-wide risk now that combined scores are in ───────────
    if ppe["persons"]:
        image_score = max(p["risk_score"] for p in ppe["persons"])
    else:
        image_score = 0
        # No detected persons, but danger/proximity may still exist
        # site-wide (e.g. machine alone in frame) — fold those in too.
        if danger:
            image_score = max(image_score, danger_zone_points)
        elif proximity:
            image_score = max(image_score, machine_nearby_points)

    high_or_above_count = sum(
        1 for p in ppe["persons"]
        if risk_level_rank(p["risk"]) >= risk_level_rank("HIGH")
    )

    # Image-level escalation: multiple independently unsafe workers make
    # the whole scene more dangerous than any single worker score implies.
    if high_or_above_count >= HIGH_WORKER_ESCALATION_THRESHOLD:
        image_score += 30

    risk_level = risk_level_for_score(image_score)
    if high_or_above_count >= HIGH_WORKER_ESCALATION_THRESHOLD:
        risk_level = "CRITICAL"
    elif high_or_above_count >= 1:
        risk_level = max_risk_level(risk_level, "HIGH")

    ppe["image_risk"]        = risk_level
    ppe["image_score"]       = image_score
    ppe["high_worker_count"] = high_or_above_count

    risk_score = image_score
    _, severity_label = compute_severity(risk_level)

    # ── Fix 10: Collect all reasons for top-level explanation ──────────────
    top_reasons: List[str] = []

    for p in ppe["persons"]:
        top_reasons.extend(p.get("reasons", []))

    for prox in proximity:
        top_reasons.append(prox.get("reason", ""))

    for zone in danger:
        top_reasons.append(zone.get("reason", ""))

    if high_or_above_count >= HIGH_WORKER_ESCALATION_THRESHOLD:
        top_reasons.append(
            f"Escalated to CRITICAL: {high_or_above_count} workers at HIGH risk or above"
        )

    if not top_reasons:
        top_reasons = ["No violations detected"]

    return {
        # ── Unified contract expected by detection_service.py ──────────────
        "detections":     valid_detections,
        "workers":        ppe.get("persons", []),
        # ── Core risk output ───────────────────────────────────────────────
        "risk_level":     risk_level,
        "risk_score":     risk_score,
        "severity":       severity_label,
        "crack":          False,
        # Fix 10: structured reasons.
        "reasons":        top_reasons,
        "reason":         "; ".join(top_reasons),
        "ppe":            ppe,
        "proximity":      proximity,
        "danger_zones":   danger,
        "pipeline_stats": pipeline_stats,
    }