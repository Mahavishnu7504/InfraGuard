from typing import List, Dict
from datetime import datetime

from backend.services.safety.risk_engine.rules import (
    evaluate_risk,
    detect_ppe_violations,
    detect_vehicle_proximity,
    detect_danger_zones,
    compute_severity,
)


def _timestamp():
    return datetime.utcnow().isoformat()


# -----------------------------------------------------
# PPE VIOLATION ANALYZER
# -----------------------------------------------------

def analyze_ppe(detections: List[Dict]) -> List[Dict]:
    """
    Analyze PPE compliance while preserving existing behavior.
    Adds tracking, event metadata and compliance-ready fields.
    """

    if not detections:
        return []

    ppe_result = detect_ppe_violations(detections)
    violations = []

    for person in ppe_result.get("persons", []):

        severity_result = compute_severity(person.get("risk", "LOW"))

        if isinstance(severity_result, tuple):
            risk_score, severity = severity_result
        else:
            risk_score, severity = None, severity_result

        violations.append({
            "event_type": "PPE",
            "timestamp": _timestamp(),
            "track_id": person.get("track_id"),
            "person_id": person.get("person_id"),
            "risk": person.get("risk"),
            "risk_score": risk_score,
            "severity_score": severity,
            "assigned_ppe": person.get("assigned_ppe", []),
            "missing_ppe": person.get("missing", []),
            "reason": person.get("reason"),
            "model_source": "infraguard",
        })

    return violations


# -----------------------------------------------------
# PROXIMITY VIOLATION ANALYZER
# -----------------------------------------------------

def analyze_proximity(detections: List[Dict], threshold: int = 300) -> List[Dict]:

    if not detections:
        return []

    raw = detect_vehicle_proximity(detections, threshold=threshold)
    violations = []

    for alert in raw:
        distance = alert.get("distance", 0)

        violations.append({
            "event_type": "Equipment",
            "timestamp": _timestamp(),
            "type": alert.get("type"),
            "machine": alert.get("machine"),
            "machine_count": 1,
            "distance_px": distance,
            "worker_distance": distance,
            "risk": "HIGH" if distance < threshold // 2 else "MEDIUM",
            "reason": (
                f"Worker within {distance}px of "
                f"{alert.get('machine')} "
                f"(threshold: {threshold}px)"
            ),
            "model_source": "infraguard",
        })

    return violations


# -----------------------------------------------------
# DANGER ZONE ANALYZER
# -----------------------------------------------------

def analyze_danger_zones(detections: List[Dict], radius: int = 350) -> List[Dict]:

    if not detections:
        return []

    raw = detect_danger_zones(detections, radius=radius)
    alerts = []

    for entry in raw:
        alerts.append({
            "event_type": "Danger Zone",
            "timestamp": _timestamp(),
            "type": entry.get("type"),
            "machine": entry.get("machine"),
            "distance_px": entry.get("distance"),
            "risk": "HIGH",
            "reason": (
                f"Worker entered danger zone of "
                f"{entry.get('machine')} "
                f"at {entry.get('distance')}px"
            ),
            "model_source": "infraguard",
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
    camera_id: str = "unknown",
) -> Dict:

    ppe_violations = analyze_ppe(detections)
    proximity_alerts = analyze_proximity(
        detections,
        threshold=proximity_threshold
    )
    danger_alerts = analyze_danger_zones(
        detections,
        radius=danger_radius
    )

    risk_result = evaluate_risk(detections)

    overall_risk = risk_result.get("risk_level", "LOW")
    severity_result = compute_severity(overall_risk)

    if isinstance(severity_result, tuple):
        risk_score, overall_severity = severity_result
    else:
        risk_score, overall_severity = None, severity_result

    total_persons = len([
        d for d in detections
        if d.get("class_name") == "person"
    ])

    compliant_workers = len([
        v for v in ppe_violations
        if v.get("risk") == "LOW"
    ])

    ppe_compliance = (
        round((compliant_workers / total_persons) * 100, 2)
        if total_persons else 100
    )

    cracks = [
        d for d in detections
        if "crack" in d.get("class_name", "").lower()
    ]

    return {
        "image_id": image_id,
        "camera_id": camera_id,

        "overall_risk": overall_risk,
        "overall_severity": overall_severity,

        "summary": {
            "workers": total_persons,
            "equipment": len(proximity_alerts),
            "cracks": len(cracks),
            "danger_zones": len(danger_alerts),
            "ppe_compliance_percentage": ppe_compliance,
            "overall_safety_score": risk_score,
            "persons_at_risk": len([
                v for v in ppe_violations
                if v.get("risk") != "LOW"
            ]),
        },

        "ppe_analysis": ppe_violations,
        "equipment_analysis": proximity_alerts,
        "proximity_analysis": proximity_alerts,
        "danger_zone_analysis": danger_alerts,

        "crack_analysis": [
            {
                "event_type": "Crack",
                "timestamp": _timestamp(),
                "class_name": d.get("class_name"),
                "confidence": d.get("confidence"),
                "model_source": "crack",
            }
            for d in cracks
        ],
    }
