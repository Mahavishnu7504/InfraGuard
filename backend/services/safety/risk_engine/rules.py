from typing import List, Dict, Tuple
import math


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
# Converts any raw class_name from YOLO detections to a stable internal key.
# All downstream logic (PPE checks, machine checks) uses these canonical names.
LABEL_NORMALIZE: Dict[str, str] = {
    # Helmet variants
    "Helmet":       "helmet",
    "helmet":       "helmet",
    "no helmet":    "no_helmet",
    "head":         "head",          # bare head — treated as no helmet upstream

    # Vest variants
    "vest":         "vest",
    "vests":        "vest",
    "no vest":      "no_vest",

    # Other PPE
    "boots":        "boots",
    "glove":        "gloves",

    # People
    "person":       "person",

    # Heavy equipment
    "Bulldozer":    "bulldozer",
    "Dump Truck":   "dump_truck",
    "Excavator":    "excavator",
    "Grader":       "grader",
    "Loader":       "loader",
    "Mixer Truck":  "mixer_truck",
    "Mobile Crane": "mobile_crane",
    "Roller":       "roller",

    # Structural defects
    "crack":        "crack",
}


def normalize_class_name(raw: str) -> str:
    """
    Return the canonical internal label for a raw YOLO class name.
    Falls back to lowercased raw name if not in the map.
    """
    return LABEL_NORMALIZE.get(raw, raw.lower())


# -----------------------------------------------------
# SITE PROFILES
# PPE requirements per site type — single source of truth.
# NOTE: goggles removed (not detected by current model).
#       boots added (detected and enforced).
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

# Default active profile (construction site).
# Other modules import these directly as the single source of truth.
CRITICAL_PPE:   set = SITE_PROFILES["construction"]["critical_ppe"]
IMPORTANT_PPE:  set = SITE_PROFILES["construction"]["important_ppe"]
MACHINE_CLASSES: set = SITE_PROFILES["construction"]["machine_classes"]


# -----------------------------------------------------
# Utility Functions
# -----------------------------------------------------

def bbox_center(box: List[float]) -> Tuple[float, float]:
    """Compute bounding box center."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def euclidean_distance(
        a: Tuple[float, float],
        b: Tuple[float, float]
) -> float:
    """Compute Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# -----------------------------------------------------
# Severity Calculation
# Returns (score: int, label: str) tuple.
# risk_summary.py uses:  risk_score, severity = compute_severity(...)
# -----------------------------------------------------

SEVERITY_MAP: Dict[str, Tuple[int, str]] = {
    "SAFE":   (0,   "Safe"),
    "LOW":    (25,  "Low"),
    "MEDIUM": (60,  "Medium"),
    "HIGH":   (100, "High"),
}


def compute_severity(risk_level: str) -> Tuple[int, str]:
    """
    Convert a risk level string to a numeric score and a display label.

    Returns:
        (score, label)  e.g. ("HIGH") → (100, "High")
    """
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
# PPE Association
# -----------------------------------------------------

def associate_ppe_to_person(
        person_box: List[float],
        detections: List[Dict],
        iou_threshold: float = 0.1
) -> set:
    """
    Associate PPE objects with a detected person using IoU.
    Uses canonical (normalized) class names for matching.
    """
    assigned = set()

    for det in detections:
        label = normalize_class_name(det["class_name"])

        if label == "person":
            continue

        iou = compute_iou(person_box, det["bbox"])

        if iou > iou_threshold:
            assigned.add(label)

    return assigned


# -----------------------------------------------------
# PPE Violation Detection
# -----------------------------------------------------

def detect_ppe_violations(detections: List[Dict]) -> Dict:
    """
    Evaluate per-person PPE compliance.
    Detections must have 'class_name' and 'bbox' keys.
    class_name may be raw YOLO labels; normalization is applied internally.
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

        if missing_critical:
            risk   = "HIGH"
            reason = f"Missing critical PPE: {', '.join(sorted(missing_critical))}"
        elif missing_important:
            risk   = "MEDIUM"
            reason = f"Missing important PPE: {', '.join(sorted(missing_important))}"
        else:
            risk   = "LOW"
            reason = "All required PPE detected"

        # Update image-level risk
        if risk == "HIGH":
            image_risk = "HIGH"
        elif risk == "MEDIUM" and image_risk != "HIGH":
            image_risk = "MEDIUM"

        report.append({
            "person_id":    idx,
            "risk":         risk,
            "missing":      sorted(missing_critical | missing_important),
            "assigned_ppe": sorted(assigned),
            "reason":       reason,
        })

    return {
        "image_risk": image_risk,
        "persons":    report,
    }


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
                violations.append({
                    "type":     "worker_near_machine",
                    "machine":  normalize_class_name(machine["class_name"]),
                    "distance": int(distance),
                })

    return violations


# -----------------------------------------------------
# Danger Zone Detection
# -----------------------------------------------------

def detect_danger_zones(
        detections: List[Dict],
        radius: int = 350
) -> List[Dict]:
    """
    Flag workers inside the danger radius of any machine.
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
        mc = bbox_center(machine["bbox"])

        for person in persons:
            pc       = bbox_center(person["bbox"])
            distance = euclidean_distance(pc, mc)

            if distance < radius:
                alerts.append({
                    "type":     "danger_zone",
                    "machine":  normalize_class_name(machine["class_name"]),
                    "distance": int(distance),
                })

    return alerts


# -----------------------------------------------------
# MAIN RISK EVALUATION
# -----------------------------------------------------

def evaluate_risk(detections: List[Dict]) -> Dict:
    """
    Standard risk evaluation wrapper (used by backend).
    Accepts raw YOLO detections; normalization happens internally.
    """
    ppe       = detect_ppe_violations(detections)
    proximity = detect_vehicle_proximity(detections)
    danger    = detect_danger_zones(detections)

    risk_level = ppe["image_risk"]

    if danger:
        risk_level = "HIGH"
    elif proximity and risk_level != "HIGH":
        risk_level = "MEDIUM"

    risk_score, severity_label = compute_severity(risk_level)

    return {
        "risk_level":     risk_level,
        "risk_score":     risk_score,
        "severity":       severity_label,
        "ppe":            ppe,
        "proximity":      proximity,
        "danger_zones":   danger,
    }