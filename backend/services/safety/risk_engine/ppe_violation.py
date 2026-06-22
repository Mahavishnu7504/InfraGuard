from typing import List, Dict

# PPE categories
CRITICAL_PPE = {"helmet", "vest"}
IMPORTANT_PPE = {"goggles", "gloves"}

def iou(boxA, boxB):
    """Intersection over Union of two boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)


def associate_ppe_to_person(person_box, ppe_boxes, iou_thresh=0.1):
    """Assign PPE items to a person based on IoU"""
    assigned = set()
    for ppe in ppe_boxes:
        if iou(person_box, ppe["box"]) >= iou_thresh:
            assigned.add(ppe["class"])
    return assigned


def detect_ppe_violations(detections: List[Dict]):
    """
    detections = [
      { "class": "person", "box": [...] },
      { "class": "helmet", "box": [...] },
      ...
    ]
    """

    persons = [d for d in detections if d["class"] == "person"]
    ppe_items = [d for d in detections if d["class"] != "person"]

    results = []
    image_risk = "LOW"

    if not persons:
        return {
            "image_risk": "LOW",
            "persons": [],
            "reason": "No person detected"
        }

    for idx, person in enumerate(persons):
        assigned_ppe = associate_ppe_to_person(person["box"], ppe_items)

        missing_critical = CRITICAL_PPE - assigned_ppe
        missing_important = IMPORTANT_PPE - assigned_ppe

        if missing_critical:
            risk = "HIGH"
            reason = f"Missing critical PPE: {', '.join(missing_critical)}"
        elif missing_important:
            risk = "MEDIUM"
            reason = f"Missing important PPE: {', '.join(missing_important)}"
        else:
            risk = "LOW"
            reason = "All required PPE detected"

        results.append({
            "person_id": idx,
            "risk": risk,
            "assigned_ppe": list(assigned_ppe),
            "reason": reason
        })

        # Escalate image risk
        if risk == "HIGH":
            image_risk = "HIGH"
        elif risk == "MEDIUM" and image_risk != "HIGH":
            image_risk = "MEDIUM"

    return {
        "image_risk": image_risk,
        "persons": results
    }