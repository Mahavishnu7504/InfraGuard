from typing import List, Dict

from backend.services.risk_engine.rules import (
    CRITICAL_PPE,
    IMPORTANT_PPE,
    compute_iou,
    normalize_class_name,
)


def associate_ppe_to_person(
        person_box: List[float],
        ppe_boxes: List[Dict],
        iou_thresh: float = 0.1
) -> set:
    """
    Assign PPE items to a person based on IoU overlap.

    Expects each item in ppe_boxes to have:
        "class_name"  (str)  — raw or canonical YOLO label
        "bbox"        (list) — [x1, y1, x2, y2]
    """
    assigned = set()

    for ppe in ppe_boxes:
        if compute_iou(person_box, ppe["bbox"]) >= iou_thresh:
            assigned.add(normalize_class_name(ppe["class_name"]))

    return assigned


def detect_ppe_violations(detections: List[Dict]) -> Dict:
    """
    Evaluate per-person PPE compliance for a single frame.

    Expects detections as:
        [
          { "class_name": "person",  "bbox": [x1, y1, x2, y2], "track_id": "..." },
          { "class_name": "helmet",  "bbox": [x1, y1, x2, y2] },
          ...
        ]

    Returns:
        {
          "image_risk": "HIGH" | "MEDIUM" | "LOW",
          "persons": [ { per-person result }, ... ],
          "reason":  str   (only present when no persons detected)
        }
    """
    persons   = [d for d in detections if normalize_class_name(d["class_name"]) == "person"]
    ppe_items = [d for d in detections if normalize_class_name(d["class_name"]) != "person"]

    if not persons:
        return {
            "image_risk": "LOW",
            "persons":    [],
            "reason":     "No person detected",
        }

    results    = []
    image_risk = "LOW"

    for idx, person in enumerate(persons):

        assigned_ppe = associate_ppe_to_person(person["bbox"], ppe_items)

        missing_critical  = CRITICAL_PPE  - assigned_ppe
        missing_important = IMPORTANT_PPE - assigned_ppe

        if missing_critical:
            risk   = "HIGH"
            reason = f"Missing critical PPE: {', '.join(sorted(missing_critical))}"
        elif missing_important:
            risk   = "MEDIUM"
            reason = f"Missing important PPE: {', '.join(sorted(missing_important))}"
        else:
            risk   = "LOW"
            reason = "All required PPE detected"

        # Escalate image-level risk
        if risk == "HIGH":
            image_risk = "HIGH"
        elif risk == "MEDIUM" and image_risk != "HIGH":
            image_risk = "MEDIUM"

        results.append({
            "person_id":    idx,
            "track_id":     person.get("track_id"),
            "risk":         risk,
            "missing":      sorted(missing_critical | missing_important),
            "assigned_ppe": sorted(assigned_ppe),
            "reason":       reason,
        })

    return {
        "image_risk": image_risk,
        "persons":    results,
    }