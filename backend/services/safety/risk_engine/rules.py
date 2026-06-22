from typing import List, Dict, Tuple
import math


# -----------------------------------------------------
# CLASS ID MAP (YOLO Class Mapping)
# -----------------------------------------------------

CLASS_ID_MAP = {
    0: "person",
    1: "helmet",
    2: "vest",
    3: "goggles",
    4: "gloves",
    5: "boots",
    6: "forklift",
    7: "truck",
    8: "crack",
    9: "wire",
    10: "rod"
}


# -----------------------------------------------------
# SITE PROFILES
# -----------------------------------------------------

SITE_PROFILES = {
    "construction": {
        "critical_ppe": {"helmet", "vest"},
        "important_ppe": {"goggles", "gloves"},
        "machine_classes": {"forklift", "truck"}
    },
    "factory": {
        "critical_ppe": {"helmet", "gloves"},
        "important_ppe": {"goggles"},
        "machine_classes": {"forklift"}
    },
    "warehouse": {
        "critical_ppe": {"vest"},
        "important_ppe": {"helmet"},
        "machine_classes": {"forklift", "truck"}
    }
}

CRITICAL_PPE = {"helmet", "vest"}
IMPORTANT_PPE = {"goggles", "gloves"}
MACHINE_CLASSES = {"forklift", "truck"}


# -----------------------------------------------------
# Utility Functions
# -----------------------------------------------------

def bbox_center(box: List[float]) -> Tuple[float, float]:
    """Compute bounding box center."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Compute Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# -----------------------------------------------------
# Severity Calculation
# -----------------------------------------------------

def compute_severity(risk_level: str) -> int:
    severity_map = {
        "SAFE": 0,
        "LOW": 25,
        "MEDIUM": 60,
        "HIGH": 100
    }

    return severity_map.get(risk_level, 0)


# -----------------------------------------------------
# IoU Calculation
# -----------------------------------------------------

def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Intersection over Union for two bounding boxes."""

    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)

    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union = area_a + area_b - inter_area

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
    """

    assigned = set()

    for det in detections:

        if det["class_name"] == "person":
            continue

        iou = compute_iou(person_box, det["bbox"])

        if iou > iou_threshold:
            assigned.add(det["class_name"])

    return assigned


# -----------------------------------------------------
# PPE Violation Detection
# -----------------------------------------------------

def detect_ppe_violations(detections: List[Dict]) -> Dict:

    persons = [d for d in detections if d["class_name"] == "person"]

    report = []
    image_risk = "LOW"

    for idx, person in enumerate(persons):

        assigned = associate_ppe_to_person(person["bbox"], detections)

        missing_critical = CRITICAL_PPE - assigned
        missing_important = IMPORTANT_PPE - assigned

        if missing_critical:
            risk = "HIGH"
            reason = f"Missing critical PPE: {', '.join(missing_critical)}"

        elif missing_important:
            risk = "MEDIUM"
            reason = f"Missing important PPE: {', '.join(missing_important)}"

        else:
            risk = "LOW"
            reason = "All required PPE detected"

        # Update image-level risk
        if risk == "HIGH":
            image_risk = "HIGH"
        elif risk == "MEDIUM" and image_risk != "HIGH":
            image_risk = "MEDIUM"

        report.append({
            "person_id": idx,
            "risk": risk,
            "missing": list(missing_critical | missing_important),
            "assigned_ppe": list(assigned),
            "reason": reason
        })

    return {
        "image_risk": image_risk,
        "persons": report
    }


# -----------------------------------------------------
# Worker–Machine Proximity Detection
# -----------------------------------------------------

def detect_vehicle_proximity(
        detections: List[Dict],
        threshold: int = 300
) -> List[Dict]:

    persons = [d for d in detections if d["class_name"] == "person"]

    machines = [
        d for d in detections
        if d["class_name"] in MACHINE_CLASSES
    ]

    violations = []

    for person in persons:

        pc = bbox_center(person["bbox"])

        for machine in machines:

            mc = bbox_center(machine["bbox"])

            distance = euclidean_distance(pc, mc)

            if distance < threshold:

                violations.append({
                    "type": "worker_near_machine",
                    "machine": machine["class_name"],
                    "distance": int(distance)
                })

    return violations


# -----------------------------------------------------
# Danger Zone Detection
# -----------------------------------------------------

def detect_danger_zones(
        detections: List[Dict],
        radius: int = 350
) -> List[Dict]:

    persons = [d for d in detections if d["class_name"] == "person"]

    machines = [
        d for d in detections
        if d["class_name"] in MACHINE_CLASSES
    ]

    alerts = []

    for machine in machines:

        mc = bbox_center(machine["bbox"])

        for person in persons:

            pc = bbox_center(person["bbox"])

            distance = euclidean_distance(pc, mc)

            if distance < radius:

                alerts.append({
                    "type": "danger_zone",
                    "machine": machine["class_name"],
                    "distance": int(distance)
                })

    return alerts


# -----------------------------------------------------
# MAIN RISK EVALUATION
# -----------------------------------------------------

def evaluate_risk(detections):
    """
    Standard risk evaluation wrapper (used by backend)
    """

    ppe = detect_ppe_violations(detections)
    proximity = detect_vehicle_proximity(detections)
    danger = detect_danger_zones(detections)

    # FINAL RISK LEVEL
    risk_level = ppe["image_risk"]

    if danger:
        risk_level = "HIGH"
    elif proximity and risk_level != "HIGH":
        risk_level = "MEDIUM"

    return {
        "risk_level": risk_level,
        "ppe": ppe,
        "proximity": proximity,
        "danger_zones": danger
    }