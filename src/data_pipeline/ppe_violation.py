# src/data_pipeline/ppe_violation.py

CRITICAL_PPE = {"helmet", "vest"}
IMPORTANT_PPE = {"goggles", "gloves"}

def detect_ppe_violations(detections):
    """
    detections: list of dicts
    [
        {"class": "helmet", "box": [...]},
        {"class": "vest", "box": [...]}
    ]
    """

    detected = {d["class"] for d in detections}

    missing_critical = CRITICAL_PPE - detected
    missing_important = IMPORTANT_PPE - detected

    violations = []

    if missing_critical:
        risk = "HIGH"
        violations.append(
            f"Missing critical PPE: {', '.join(missing_critical)}"
        )
    elif missing_important:
        risk = "MEDIUM"
        violations.append(
            f"Missing important PPE: {', '.join(missing_important)}"
        )
    else:
        risk = "LOW"
        violations.append("All required PPE detected")

    return {
        "risk": risk,
        "violations": violations
    }