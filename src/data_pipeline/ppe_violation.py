# src/data_pipeline/ppe_violation.py

CRITICAL_PPE = {"helmet", "vest"}
IMPORTANT_PPE = {"goggles", "gloves"}

RISK_PRIORITY = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2
}

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / (boxAArea + boxBArea - interArea + 1e-6)


def evaluate_person(person_box, ppe_boxes):
    detected = set()

    for cls, box in ppe_boxes:
        if iou(person_box, box) > 0.1:
            detected.add(cls)

    missing_critical = CRITICAL_PPE - detected
    missing_important = IMPORTANT_PPE - detected

    if missing_critical:
        return {
            "risk": "HIGH",
            "violations": [f"Missing critical PPE: {', '.join(missing_critical)}"]
        }

    if missing_important:
        return {
            "risk": "MEDIUM",
            "violations": [f"Missing important PPE: {', '.join(missing_important)}"]
        }

    return {
        "risk": "LOW",
        "violations": ["All required PPE detected"]
    }


def detect_ppe_violations(detections):
    persons = [d for d in detections if d["class"] == "person"]
    ppe_boxes = [(d["class"], d["box"]) for d in detections if d["class"] != "person"]

    if not persons:
        return {
            "risk": "LOW",
            "violations": ["No person detected"],
            "persons": []
        }

    person_results = []

    for idx, person in enumerate(persons):
        result = evaluate_person(person["box"], ppe_boxes)
        person_results.append({
            "person_id": idx,
            **result
        })

    final_risk = max(
        person_results,
        key=lambda r: RISK_PRIORITY[r["risk"]]
    )["risk"]

    return {
        "risk": final_risk,
        "persons": person_results
    }