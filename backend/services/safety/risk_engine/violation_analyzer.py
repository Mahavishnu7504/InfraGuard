from typing import List, Dict
from rules import (
    evaluate_risk,
    detect_ppe_violations,
    detect_vehicle_proximity,
    detect_danger_zones,
    compute_severity,
    CLASS_ID_MAP,
)


# -----------------------------------------------------
# PPE VIOLATION ANALYZER
# -----------------------------------------------------

def analyze_ppe(detections: List[Dict]) -> List[Dict]:
    """
    Analyze PPE compliance for all detected persons.

    Returns a list of violation records, one per person,
    including assigned PPE, missing items, risk level, and severity score.
    """

    if not detections:
        return []

    ppe_result = detect_ppe_violations(detections)
    violations = []

    for person in ppe_result["persons"]:

        severity = compute_severity(person["risk"])

        violations.append({
            "person_id": person["person_id"],
            "risk": person["risk"],
            "severity_score": severity,
            "assigned_ppe": person["assigned_ppe"],
            "missing_ppe": person["missing"],
            "reason": person["reason"],
        })

    return violations


# -----------------------------------------------------
# PROXIMITY VIOLATION ANALYZER
# -----------------------------------------------------

def analyze_proximity(detections: List[Dict], threshold: int = 300) -> List[Dict]:
    """
    Analyze worker-machine proximity violations.

    Returns a list of proximity alerts with machine type and pixel distance.
    """

    if not detections:
        return []

    raw = detect_vehicle_proximity(detections, threshold=threshold)
    violations = []

    for alert in raw:
        violations.append({
            "type": alert["type"],
            "machine": alert["machine"],
            "distance_px": alert["distance"],
            "risk": "HIGH" if alert["distance"] < threshold // 2 else "MEDIUM",
            "reason": (
                f"Worker within {alert['distance']}px of {alert['machine']} "
                f"(threshold: {threshold}px)"
            ),
        })

    return violations


# -----------------------------------------------------
# DANGER ZONE ANALYZER
# -----------------------------------------------------

def analyze_danger_zones(detections: List[Dict], radius: int = 350) -> List[Dict]:
    """
    Analyze danger zone breaches around heavy machinery.

    Returns a list of danger zone alerts with machine type and breach distance.
    """

    if not detections:
        return []

    raw = detect_danger_zones(detections, radius=radius)
    alerts = []

    for entry in raw:
        alerts.append({
            "type": entry["type"],
            "machine": entry["machine"],
            "distance_px": entry["distance"],
            "risk": "HIGH",
            "reason": (
                f"Worker entered danger zone of {entry['machine']} "
                f"at {entry['distance']}px (radius: {radius}px)"
            ),
        })

    return alerts


# -----------------------------------------------------
# FULL VIOLATION REPORT
# -----------------------------------------------------

def generate_violation_report(
    detections: List[Dict],
    image_id: str = "unknown",
    proximity_threshold: int = 300,
    danger_radius: int = 350,
) -> Dict:
    """
    Run all violation checks and return a unified report dict.

    Args:
        detections:           List of detection dicts with 'class_id', 'class_name', 'bbox', 'confidence'.
        image_id:             Identifier for the source image (filename or UUID).
        proximity_threshold:  Pixel distance threshold for worker-machine proximity.
        danger_radius:        Pixel radius for danger zone breach detection.

    Returns:
        A structured violation report with summary, per-category findings,
        overall risk level, and aggregate severity score.
    """

    ppe_violations      = analyze_ppe(detections)
    proximity_alerts    = analyze_proximity(detections, threshold=proximity_threshold)
    danger_alerts       = analyze_danger_zones(detections, radius=danger_radius)

    # Derive overall risk from evaluate_risk (single source of truth)
    risk_result         = evaluate_risk(detections)
    overall_risk        = risk_result["risk_level"]
    overall_severity    = compute_severity(overall_risk)

    total_persons       = len([d for d in detections if d["class_name"] == "person"])
    persons_at_risk     = len([v for v in ppe_violations if v["risk"] != "LOW"])

    return {
        "image_id": image_id,
        "overall_risk": overall_risk,
        "overall_severity": overall_severity,
        "summary": {
            "total_persons": total_persons,
            "persons_at_risk": persons_at_risk,
            "ppe_violations": len([v for v in ppe_violations if v["risk"] != "LOW"]),
            "proximity_alerts": len(proximity_alerts),
            "danger_zone_breaches": len(danger_alerts),
        },
        "ppe_analysis": ppe_violations,
        "proximity_analysis": proximity_alerts,
        "danger_zone_analysis": danger_alerts,
    }