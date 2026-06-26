import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from backend.services.safety.risk_engine.rules import (
    evaluate_risk,
    detect_ppe_violations,
    detect_vehicle_proximity,
    detect_danger_zones,
    compute_severity,
)


def _timestamp() -> str:
    return datetime.utcnow().isoformat()


def _violation_id() -> str:
    return str(uuid.uuid4())


def _extract_location(detection: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Pull a location out of whatever shape the detection/alert dict provides.
    Pure extraction — no inference, no computation.
    """
    if not detection:
        return None

    if "location" in detection and detection["location"] is not None:
        return detection["location"]

    if "bbox" in detection and detection["bbox"] is not None:
        return {"bbox": detection["bbox"]}

    x = detection.get("x")
    y = detection.get("y")
    if x is not None and y is not None:
        return {"x": x, "y": y}

    return None


# -----------------------------------------------------
# SEVERITY ENVELOPE (pass-through from Risk Engine)
# -----------------------------------------------------

# NOTE: These fallback tables are a temporary shim. They exist ONLY because
# compute_severity may not yet return priority/recommendation/corrective_action
# natively. They do not make any risk decision — they map an ALREADY-DECIDED
# severity label to display text. Once the risk engine returns these fields
# directly, delete this section and `_unpack_severity` should just pass them
# through unchanged.
_PRIORITY_FALLBACK = {
    "CRITICAL": 1,
    "HIGH": 2,
    "MEDIUM": 3,
    "LOW": 4,
    "SAFE": 5,
}

_RECOMMENDATION_FALLBACK = {
    "CRITICAL": "Stop work immediately and secure the area.",
    "HIGH": "Immediate corrective action required.",
    "MEDIUM": "Address the issue promptly.",
    "LOW": "Monitor and improve compliance.",
    "SAFE": "No action required.",
}


def _unpack_severity(severity_result: Any, risk_label: str) -> Dict[str, Any]:
    """
    Normalizes whatever compute_severity(risk_label) returns into a single
    consistent envelope. Supports:
      - dict with any subset of the expected keys
      - 5-tuple: (risk_score, severity, priority, recommendation, corrective_action)
      - 2-tuple: (risk_score, severity)   [legacy]
      - bare severity value                [legacy]

    No value here is computed from scratch — every field either comes
    straight from compute_severity's output, or (if absent) from the
    fallback tables above, which only re-express the SAME risk_label that
    the engine already decided on.
    """
    risk_score = None
    severity = risk_label
    priority = None
    recommendation = None
    corrective_action = None

    if isinstance(severity_result, dict):
        risk_score = severity_result.get("risk_score")
        severity = severity_result.get("severity", risk_label)
        priority = severity_result.get("priority")
        recommendation = severity_result.get("recommendation")
        corrective_action = severity_result.get("corrective_action")

    elif isinstance(severity_result, tuple):
        if len(severity_result) >= 5:
            risk_score, severity, priority, recommendation, corrective_action = severity_result[:5]
        elif len(severity_result) == 2:
            risk_score, severity = severity_result
        else:
            severity = severity_result[0] if severity_result else risk_label

    else:
        severity = severity_result if severity_result is not None else risk_label

    severity_key = (severity or risk_label or "LOW").upper()

    if priority is None:
        priority = _PRIORITY_FALLBACK.get(severity_key)

    if recommendation is None:
        recommendation = _RECOMMENDATION_FALLBACK.get(severity_key)

    if corrective_action is None:
        # No fallback invented here on purpose: corrective action is specific
        # to the violation type and is attached by each analyzer below,
        # not guessed at in this shared helper.
        corrective_action = None

    return {
        "risk_score": risk_score,
        "severity": severity,
        "priority": priority,
        "recommendation": recommendation,
        "corrective_action": corrective_action,
    }


def _compliance_status(risk_label: Optional[str]) -> str:
    """
    Derived ONLY from the risk label the engine already produced.
    SAFE or LOW -> Compliant, anything else -> Non-Compliant.
    This is a relabeling, not a new decision.
    """
    risk = (risk_label or "").upper()

    if risk in ("SAFE", "LOW"):
        return "Compliant"

    return "Non-Compliant"


def _base_event_fields(
    event_type: str,
    risk_label: str,
    camera_id: str,
    worker_id: Optional[str],
    location: Optional[Dict[str, Any]],
    corrective_action: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Shared identity + severity envelope attached to every event,
    regardless of type (PPE / Equipment / Danger Zone / Crack).
    """
    envelope = _unpack_severity(compute_severity(risk_label), risk_label)

    if corrective_action is not None:
        envelope["corrective_action"] = corrective_action

    return {
        "violation_id": _violation_id(),
        "event_type": event_type,
        "timestamp": _timestamp(),
        "camera_id": camera_id,
        "worker_id": worker_id,
        "location": location,

        "risk": risk_label,
        "risk_score": envelope["risk_score"],
        "severity": envelope["severity"],
        "priority": envelope["priority"],
        "compliance_status": _compliance_status(risk_label),
        "recommendation": envelope["recommendation"],
        "corrective_action": envelope["corrective_action"],

        "model_source": "infraguard",
        "pipeline": "Safety Pipeline",
        "category": "Safety",
    }


# -----------------------------------------------------
# PPE VIOLATION ANALYZER
# -----------------------------------------------------

def analyze_ppe(
    detections: List[Dict],
    camera_id: str = "unknown",
) -> List[Dict]:
    """
    Converts Risk Engine PPE decisions into PPE violation events.
    Does not re-judge risk; uses person.get("risk") exactly as returned.
    """

    if not detections:
        return []

    ppe_result = detect_ppe_violations(detections)
    violations = []

    for person in ppe_result.get("persons", []):
        risk_label = person.get("risk", "LOW")
        missing = person.get("missing", [])

        corrective_action = None
        if missing:
            corrective_action = "Provide " + ", ".join(missing) + " immediately."

        event = _base_event_fields(
            event_type="PPE",
            risk_label=risk_label,
            camera_id=camera_id,
            worker_id=person.get("worker_id") or person.get("person_id") or person.get("track_id"),
            location=_extract_location(person),
            corrective_action=corrective_action,
        )

        event.update({
            "track_id": person.get("track_id"),
            "person_id": person.get("person_id"),
            "assigned_ppe": person.get("assigned_ppe", []),
            "missing_ppe": missing,
            "reason": person.get("reason"),
        })

        violations.append(event)

    return violations


# -----------------------------------------------------
# PROXIMITY VIOLATION ANALYZER
# -----------------------------------------------------

def analyze_proximity(
    detections: List[Dict],
    camera_id: str = "unknown",
    threshold: int = 300,
) -> List[Dict]:
    """
    Converts Risk Engine proximity decisions into equipment-proximity events.

    IMPORTANT: `threshold` is forwarded to the risk engine as a parameter of
    ITS detection call (detect_vehicle_proximity), not used here to derive
    risk. The risk label on each alert is read directly from the engine's
    output (`alert.get("risk")`), never recomputed from distance here.
    """

    if not detections:
        return []

    raw = detect_vehicle_proximity(detections, threshold=threshold)
    violations = []

    for alert in raw:
        distance = alert.get("distance", 0)
        # Risk label comes straight from the engine. If a given engine build
        # doesn't yet attach one, default to None rather than re-deriving it
        # from distance — that recomputation is exactly what we're removing.
        risk_label = alert.get("risk")

        event = _base_event_fields(
            event_type="Equipment",
            risk_label=risk_label or "LOW",
            camera_id=camera_id,
            worker_id=alert.get("worker_id") or alert.get("track_id"),
            location=_extract_location(alert),
            corrective_action=alert.get("corrective_action")
            or (f"Maintain safe distance from {alert.get('machine')}." if alert.get("machine") else None),
        )

        event.update({
            "type": alert.get("type"),
            "machine": alert.get("machine"),
            "machine_count": 1,
            "distance_px": distance,
            "worker_distance": distance,
            "threshold_px": threshold,
            "reason": alert.get("reason") or (
                f"Worker within {distance}px of "
                f"{alert.get('machine')} "
                f"(threshold: {threshold}px)"
            ),
        })

        violations.append(event)

    return violations


# -----------------------------------------------------
# DANGER ZONE ANALYZER
# -----------------------------------------------------

def analyze_danger_zones(
    detections: List[Dict],
    camera_id: str = "unknown",
    radius: int = 350,
) -> List[Dict]:
    """
    Converts Risk Engine danger-zone decisions into events.
    Risk label is read from the engine's output when present; "HIGH" is
    used as the documented default for a confirmed danger-zone entry,
    matching the engine's own definition of that event type — not a
    distance-based recalculation.
    """

    if not detections:
        return []

    raw = detect_danger_zones(detections, radius=radius)
    alerts = []

    for entry in raw:
        risk_label = entry.get("risk", "HIGH")

        event = _base_event_fields(
            event_type="Danger Zone",
            risk_label=risk_label,
            camera_id=camera_id,
            worker_id=entry.get("worker_id") or entry.get("track_id"),
            location=_extract_location(entry),
            corrective_action=entry.get("corrective_action")
            or (f"Remove worker from danger zone of {entry.get('machine')} immediately." if entry.get("machine") else None),
        )

        event.update({
            "type": entry.get("type"),
            "machine": entry.get("machine"),
            "distance_px": entry.get("distance"),
            "radius_px": radius,
            "reason": entry.get("reason") or (
                f"Worker entered danger zone of "
                f"{entry.get('machine')} "
                f"at {entry.get('distance')}px"
            ),
        })

        alerts.append(event)

    return alerts


# -----------------------------------------------------
# CRACK / STRUCTURAL ANALYZER
# -----------------------------------------------------

def analyze_cracks(
    detections: List[Dict],
    camera_id: str = "unknown",
) -> List[Dict]:
    """
    Converts crack detections into events with the same severity envelope
    as every other event type. Risk label is read from the detection if the
    engine attaches one; otherwise left as None rather than guessed.
    """
    cracks = [d for d in detections if "crack" in d.get("class_name", "").lower()]
    events = []

    for d in cracks:
        risk_label = d.get("risk")

        event = _base_event_fields(
            event_type="Crack",
            risk_label=risk_label or "HIGH",
            camera_id=camera_id,
            worker_id=None,
            location=_extract_location(d),
        )

        event.update({
            "class_name": d.get("class_name"),
            "confidence": d.get("confidence"),
        })
        # Crack events use a dedicated model source, matching prior behavior.
        event["model_source"] = "crack"

        events.append(event)

    return events


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

    ppe_violations = analyze_ppe(detections, camera_id=camera_id)
    proximity_alerts = analyze_proximity(
        detections,
        camera_id=camera_id,
        threshold=proximity_threshold,
    )
    danger_alerts = analyze_danger_zones(
        detections,
        camera_id=camera_id,
        radius=danger_radius,
    )
    crack_events = analyze_cracks(detections, camera_id=camera_id)

    # Overall risk/severity for the whole report is read directly from the
    # risk engine's evaluate_risk output — not derived from the counts below.
    risk_result = evaluate_risk(detections)
    overall_risk = risk_result.get("risk_level", "LOW")
    overall_envelope = _unpack_severity(compute_severity(overall_risk), overall_risk)

    total_persons = len([
        d for d in detections
        if d.get("class_name") == "person"
    ])

    compliant_workers = len([
        v for v in ppe_violations
        if v.get("compliance_status") == "Compliant"
    ])

    ppe_compliance = (
        round((compliant_workers / total_persons) * 100, 2)
        if total_persons else 100
    )

    return {
        "image_id": image_id,
        "generated_at": _timestamp(),
        "camera_id": camera_id,
        "report_version": "3.0",
        "engine": "InfraGuard Enterprise AI",

        "overall_risk": overall_risk,
        "overall_risk_score": overall_envelope["risk_score"],
        "overall_severity": overall_envelope["severity"],
        "overall_priority": overall_envelope["priority"],
        "overall_recommendation": overall_envelope["recommendation"],

        "summary": {
            "workers": total_persons,
            "equipment": len([
                d for d in detections
                if d.get("class_name") not in ("person", None)
                and "crack" not in d.get("class_name", "").lower()
            ]),
            "cracks": len(crack_events),
            "danger_zones": len(danger_alerts),
            "ppe_compliance_percentage": ppe_compliance,
            "overall_safety_score": overall_envelope["risk_score"],
            "persons_at_risk": len([
                v for v in ppe_violations
                if v.get("compliance_status") != "Compliant"
            ]),
        },

        "ppe_analysis": ppe_violations,
        "equipment_analysis": proximity_alerts,
        "proximity_analysis": proximity_alerts,
        "danger_zone_analysis": danger_alerts,
        "crack_analysis": crack_events,

        # Flat list of every event across categories — convenient for
        # history/analytics consumers that want a single timeline rather
        # than four separate buckets.
        "all_events": ppe_violations + proximity_alerts + danger_alerts + crack_events,
    }