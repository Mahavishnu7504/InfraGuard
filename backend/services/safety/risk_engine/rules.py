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
# Severity Calculation
# -----------------------------------------------------

SEVERITY_MAP: Dict[str, Tuple[int, str]] = {
    "SAFE":   (0,   "Safe"),
    "LOW":    (25,  "Low"),
    "MEDIUM": (60,  "Medium"),
    "HIGH":   (100, "High"),
}


def compute_severity(risk_level: str) -> Tuple[int, str]:
    """Convert a risk level string to a numeric score and display label."""
    return SEVERITY_MAP.get(risk_level, (0, "Safe"))


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
) -> set:
    """
    Associate PPE items with a person using body-region checks AND IoU.

    Rules:
      - helmet  → must be in top 30% of person bbox AND IoU > threshold
      - vest    → must be in middle 30–70% AND IoU > threshold
      - boots   → must be in bottom 25% AND IoU > threshold
      - gloves  → IoU > (smaller) threshold only
      - other   → IoU > threshold (machines, unknown labels are skipped)
    """
    assigned = set()

    for det in detections:
        label = normalize_class_name(det["class_name"])

        # Skip non-PPE labels and persons.
        if label in ("person",) | MACHINE_CLASSES | {"crack", "no_helmet", "no_vest",
                                                      "head", "unknown"}:
            continue

        ppe_box = det["bbox"]
        iou     = compute_iou(person_box, ppe_box)

        if label in _REGION_MATCHERS:
            region_ok = _REGION_MATCHERS[label](person_box, ppe_box)
            if region_ok and iou > iou_threshold:
                assigned.add(label)
        elif label == "gloves":
            if iou > _GLOVE_IOC_THRESHOLD:
                assigned.add(label)
        else:
            # Unknown PPE-like label — fall back to IoU only.
            if iou > iou_threshold:
                assigned.add(label)

    return assigned


# -----------------------------------------------------
# PPE Violation Detection
# -----------------------------------------------------

def detect_ppe_violations(detections: List[Dict]) -> Dict:
    """
    Evaluate per-person PPE compliance.
    class_name may be raw YOLO labels; normalization applied internally.
    """
    persons = [
        d for d in detections
        if normalize_class_name(d["class_name"]) == "person"
    ]

    report     = []
    image_risk = "LOW"

    for idx, person in enumerate(persons):
        assigned = associate_ppe_to_person(person["bbox"], detections)

        missing_critical  = CRITICAL_PPE  - assigned
        missing_important = IMPORTANT_PPE - assigned

        # Fix 10: Structured reason list for UI and debugging.
        reasons = []

        if missing_critical:
            risk = "HIGH"
            for item in sorted(missing_critical):
                reasons.append(f"Missing critical PPE: {item}")
        elif missing_important:
            risk = "MEDIUM"
            for item in sorted(missing_important):
                reasons.append(f"Missing important PPE: {item}")
        else:
            risk = "LOW"
            reasons.append("All required PPE detected")

        if risk == "HIGH":
            image_risk = "HIGH"
        elif risk == "MEDIUM" and image_risk != "HIGH":
            image_risk = "MEDIUM"

        report.append({
            "person_id":    idx,
            "risk":         risk,
            "missing":      sorted(missing_critical | missing_important),
            "assigned_ppe": sorted(assigned),
            "reasons":      reasons,
            # Keep single-string reason for backward compatibility.
            "reason":       "; ".join(reasons),
        })

    return {
        "image_risk": image_risk,
        "persons":    report,
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
                    # Fix 10: reason string for UI.
                    "reason":   f"Worker is {int(distance)}px from {machine_label}",
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

        for person in persons:
            pc       = bbox_center(person["bbox"])
            distance = euclidean_distance(pc, mc)

            if distance < radius:
                alerts.append({
                    "type":           "danger_zone",
                    "machine":        machine_label,
                    "distance":       int(distance),
                    "danger_radius":  radius,
                    # Fix 10: reason string for UI.
                    "reason":         (
                        f"Worker is {int(distance)}px from {machine_label} "
                        f"(danger zone: {radius}px)"
                    ),
                })

    return alerts


# -----------------------------------------------------
# Fix 6: Crack-first evaluation gate
# -----------------------------------------------------

def _build_crack_result(cracks: List[Dict]) -> Dict:
    """
    Build the risk result dict for a frame containing structural cracks.
    PPE and proximity checks are skipped entirely.
    """
    reasons = [
        f"Structural crack detected (bbox: {c['bbox']})" for c in cracks
    ]
    risk_score, severity_label = compute_severity("HIGH")

    return {
        "risk_level":   "HIGH",
        "risk_score":   risk_score,
        "severity":     severity_label,
        "crack":        True,
        "crack_count":  len(cracks),
        # Fix 10: reasons list.
        "reasons":      reasons,
        "reason":       "; ".join(reasons),
        "ppe":          {"image_risk": "HIGH", "persons": []},
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
    Full risk evaluation pipeline.

    Evaluation order (Fix 6):
      1. Validate all detections (Fix 9).
      2. Check for structural cracks → if found, return HIGH immediately (Fix 5).
      3. PPE compliance check (Fix 1–4).
      4. Worker–machine proximity check.
      5. Danger zone check with adaptive radius (Fix 7–8).
      6. Combine results with structured reasons (Fix 10).

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

    risk_level = ppe["image_risk"]

    if danger:
        risk_level = "HIGH"
    elif proximity and risk_level != "HIGH":
        risk_level = "MEDIUM"

    risk_score, severity_label = compute_severity(risk_level)

    # ── Fix 10: Collect all reasons for top-level explanation ──────────────
    top_reasons: List[str] = []

    for p in ppe["persons"]:
        top_reasons.extend(p.get("reasons", []))

    for prox in proximity:
        top_reasons.append(prox.get("reason", ""))

    for zone in danger:
        top_reasons.append(zone.get("reason", ""))

    if not top_reasons:
        top_reasons = ["No violations detected"]

    return {
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