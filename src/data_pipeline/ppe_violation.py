# src/data_pipeline/ppe_violation.py

CRITICAL_PPE = {"helmet", "vest"}
IMPORTANT_PPE = {"goggles", "gloves"}

def detect_ppe_violations(detections):
    """
    detections: list of dicts
    [
      {"class": "helmet", "box": [...]},
      {"class": "person", "box": [...]}
    ]
    """

    detected = {d["class"] for d in detections}

    # ✅ No person → no risk
    if "person" not in detected:
        return {
            "risk": "LOW",
            "violations": ["No person detected"]
        }

    missing_critical = CRITICAL_PPE - detected
    missing_important = IMPORTANT_PPE - detected

    if missing_critical:
        return {
            "risk": "HIGH",
            "violations": [
                f"Missing critical PPE: {', '.join(missing_critical)}"
            ]
        }

    if missing_important:
        return {
            "risk": "MEDIUM",
            "violations": [
                f"Missing important PPE: {', '.join(missing_important)}"
            ]
        }

    return {
        "risk": "LOW",
        "violations": ["All required PPE detected"]
    }