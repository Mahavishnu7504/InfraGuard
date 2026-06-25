# =========================================================
# INFRA GUARD — ENTERPRISE SAFETY INTELLIGENCE ENGINE
# detection_service.py  v2.0
# =========================================================

import uuid
import math
import time
import traceback
import numpy as np
import cv2

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict

from ai_engine.pipelines.safety_pipeline import run_safety_pipeline

# ---- Optional: swap in your real EnterpriseTracker ----
try:
    from ai_engine.tracker import EnterpriseTracker as _ET
    _tracker = _ET()
    _USE_ENTERPRISE_TRACKER = True
except ImportError:
    _tracker = None
    _USE_ENTERPRISE_TRACKER = False

# =========================================================
# CONFIGURATION
# =========================================================

MIN_CONFIDENCE: float = 0.35   # detections below this are silently dropped

# =========================================================
# LABEL NORMALISATION MAP
# =========================================================

LABEL_MAP: Dict[str, str] = {
    # helmets
    "helmet":       "helmet",
    "Helmet":       "helmet",
    "hard hat":     "helmet",
    "hardhat":      "helmet",
    # vests
    "vest":         "vest",
    "vests":        "vest",
    "safety vest":  "vest",
    # boots
    "boots":        "boots",
    "boot":         "boots",
    "safety boots": "boots",
    # gloves
    "glove":        "gloves",
    "gloves":       "gloves",
    # violations
    "no helmet":    "no_helmet",
    "no_helmet":    "no_helmet",
    "no vest":      "no_vest",
    "no_vest":      "no_vest",
    "no gloves":    "no_gloves",
    "no_gloves":    "no_gloves",
    # people
    "person":       "person",
    "worker":       "person",
    # cracks
    "crack":            "crack",
    "crack detection":  "crack",
    # equipment
    "excavator":    "excavator",
    "Excavator":    "excavator",
    "loader":       "loader",
    "Loader":       "loader",
    "bulldozer":    "bulldozer",
    "Bulldozer":    "bulldozer",
    "roller":       "roller",
    "Roller":       "roller",
    "grader":       "grader",
    "Grader":       "grader",
    "crane":        "crane",
    "Crane":        "crane",
}

EQUIPMENT_CLASSES = {"excavator", "loader", "bulldozer", "roller", "grader", "crane"}

# How close (px, bbox-center to bbox-center) a piece of gear/equipment must be
# to a person before it is considered "theirs" / "nearby" for narrative reasoning.
PPE_LINK_RADIUS_PX:       float = 180.0
EQUIPMENT_PROXIMITY_PX:   float = 260.0

# =========================================================
# DANGER ZONES
# =========================================================

DANGER_ZONES = [
    {
        "name":    "CRANE ZONE",
        "risk":    "CRITICAL",
        "polygon": [[820, 180], [1180, 180], [1240, 520], [780, 520]]
    },
    {
        "name":    "MACHINE AREA",
        "risk":    "HIGH",
        "polygon": [[120, 420], [420, 420], [420, 690], [120, 690]]
    }
]

# =========================================================
# PPE INTELLIGENCE
# =========================================================

PPE_VIOLATIONS: Dict[str, str] = {
    "no_helmet": "HIGH",
    "no_vest":   "MEDIUM",
    "no_gloves": "LOW"
}

# Maps a missing-PPE class → the friendly item name used in narrative sentences
PPE_ITEM_NAMES: Dict[str, str] = {
    "no_helmet": "helmet",
    "no_vest":   "vest",
    "no_gloves": "gloves",
    "no_boots":  "boots",
}

# Required PPE set used for "worker detected, but X missing" inference
REQUIRED_PPE: Dict[str, str] = {
    "helmet": "no_helmet",
    "vest":   "no_vest",
}

# =========================================================
# NARRATIVE TEMPLATES
# =========================================================
# "Instead of a bare label, return an operational sentence."
# These are intentionally short, declarative, and report-ready.

EQUIPMENT_REASONING_TEMPLATES: Dict[str, str] = {
    "excavator": "Excavator operating near workers.",
    "loader":    "Loader operating near workers.",
    "bulldozer": "Bulldozer operating near workers.",
    "roller":    "Roller operating near workers.",
    "grader":    "Grader operating near workers.",
    "crane":     "Crane operating near workers.",
}

EQUIPMENT_REASONING_ISOLATED: Dict[str, str] = {
    "excavator": "Excavator active on site.",
    "loader":    "Loader active on site.",
    "bulldozer": "Bulldozer active on site.",
    "roller":    "Roller active on site.",
    "grader":    "Grader active on site.",
    "crane":     "Crane active on site.",
}

CRACK_REASONING: str = "Structural crack detected. Inspection recommended."

PPE_OK_REASONING:  str = "Worker detected. All required PPE present. Low risk."
PPE_MISSING_TEMPLATE: str = "Worker detected. Missing {items}. {risk_label} risk."

# =========================================================
# RISK CONFIG
# =========================================================

RISK_SCORES: Dict[str, int] = {
    "critical": 90,
    "high":     70,
    "medium":   45,
    "low":      20
}

SEVERITY_ORDER: Dict[str, int] = {
    "LOW":      0,
    "MEDIUM":   1,
    "HIGH":     2,
    "CRITICAL": 3
}

RISK_COLORS: Dict[str, tuple] = {
    "critical": (0,   0,   255),
    "high":     (0,   80,  255),
    "medium":   (0,   215, 255),
    "low":      (0,   255, 120),
}

# Canonical class → BGR color
CLASS_COLORS: Dict[str, tuple] = {
    "person":    (0,   200, 255),
    "helmet":    (0,   255, 120),
    "vest":      (0,   180, 255),
    "boots":     (60,  255, 200),
    "gloves":    (120, 255, 180),
    "no_helmet": (0,   0,   255),
    "no_vest":   (0,   80,  255),
    "no_gloves": (0,   150, 255),
    "crack":     (0,   215, 255),
    "equipment": (180, 180, 255),
}

CONFIDENCE_TIERS = [
    (0.95, "Verified"),
    (0.90, "Excellent"),
    (0.80, "Good"),
    (0.70, "Review"),
    (0.00, "Low"),
]

# =========================================================
# DETECTION SCHEMA
# =========================================================

@dataclass
class Detection:
    """
    Canonical detection object. Every stage reads and writes this schema.
    No field is ever added or removed outside this class.
    """
    # --- Core inference output (set by normalize_labels) ---
    id:             str   = field(default_factory=lambda: str(uuid.uuid4()))
    class_name:     str   = ""          # normalised label
    raw_label:      str   = ""          # original model output, kept for debugging
    confidence:     float = 0.0
    bbox:           List  = field(default_factory=list)   # [x1, y1, x2, y2]
    model_source:   str   = ""
    timestamp:      str   = field(default_factory=lambda: datetime.utcnow().isoformat())
    camera_id:      str   = ""

    # --- Category helpers (set by normalize_labels) ---
    event_type:     str   = "Observation"   # PPE | Equipment | Crack | DangerZone | Observation
    equipment_type: Optional[str] = None    # "Excavator" etc. when class is equipment
    worker_id:      Optional[str] = None    # reserved for future person re-ID

    # --- Tracker output (set by run_tracker) ---
    tracking_id:    Optional[int] = None
    trajectory:     List          = field(default_factory=list)  # list of (x, y) tuples
    velocity:       float         = 0.0
    direction:      float         = 0.0   # degrees, 0 = right, CCW positive
    age:            int           = 0
    frames_tracked: int           = 0

    # --- Risk classification (set by classify_risk) ---
    risk:             str = "LOW"
    incident_type:    str = "Operational Observation"
    priority:         int = 1
    confidence_level: str = "Low"

    # --- Zone intelligence (set by analyze_intrusions) ---
    danger_zone:    bool          = False
    zone_name:      Optional[str] = None
    zone_level:     Optional[str] = None
    distance_to_zone: float       = -1.0  # px; -1 = not measured

    # --- Narrative intelligence (set by build_reasoning) ---
    reasoning:        str       = ""   # human-readable "why" behind the risk tag
    missing_ppe:      List[str] = field(default_factory=list)   # PPE absent on this worker
    nearby_equipment: List[str] = field(default_factory=list)   # equipment within proximity radius

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================================================
# PIPELINE
# =========================================================

def process_frame(frame, camera_id: str = "") -> Dict[str, Any]:
    """
    Main entry point.

    Pipeline:
        Frame
          ↓  run_inference        — call AI model, get raw detections
          ↓  normalize_labels     — map raw output → list[Detection]
          ↓  run_tracker          — assign tracking_id + trajectory
          ↓  classify_risk        — PPE / crack / person risk tagging
          ↓  analyze_intrusions   — danger-zone alerts
          ↓  build_reasoning      — Enterprise Intelligence: turns labels into narrative
                                    findings ("Worker detected → Helmet missing → High Risk",
                                    "Excavator operating near workers.",
                                    "Structural crack detected. Inspection recommended.")
          ↓  calculate_analytics  — aggregate risk score + findings/summary/compliance/
                                    recommendations/risk_factors/statistics
          ↓  draw_frame           — all rendering (zones → boxes → trajectories → HUD)
          ↓  Return frame + result dict
    """
    if frame is None:
        return _empty(camera_id)

    t0 = time.perf_counter()

    try:
        raw           = run_inference(frame)
        detections    = normalize_labels(raw, camera_id)
        detections    = run_tracker(detections)
        detections    = classify_risk(detections)
        alerts        = analyze_intrusions(detections)
        detections    = build_reasoning(detections)
        analytics     = calculate_analytics(detections, alerts)

        # Sort: critical → high → medium → low so important boxes render on top
        detections.sort(key=lambda d: SEVERITY_ORDER.get(d.risk.upper(), 0))

        draw_frame(frame, detections, alerts, analytics)

        processing_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "frame":       None,   # caller injects encoded frame bytes if needed
            "detections":  [d.to_dict() for d in detections],
            "alerts":      alerts,
            "zones":       DANGER_ZONES,
            "analytics":   analytics,
            "telemetry":   {"processing_ms": processing_ms, "detection_count": len(detections)},
            "ai_metadata": build_ai_metadata(processing_ms),
            "timestamp":   datetime.utcnow().isoformat(),
            "camera_id":   camera_id,
        }

    except Exception:
        print("\n" + "=" * 80)
        print("[DETECTION ERROR] process_frame() failed:")
        traceback.print_exc()
        print("=" * 80 + "\n")
        return _empty(camera_id)


# =========================================================
# STAGE 1 — INFERENCE
# =========================================================

def run_inference(frame) -> Dict[str, Any]:
    """
    Call the AI model. Returns the raw pipeline result dict unchanged.
    No schema mapping happens here — that is normalize_labels' job.
    """
    return run_safety_pipeline(frame)


# =========================================================
# STAGE 2 — NORMALIZE LABELS
# =========================================================

def normalize_labels(raw: Dict[str, Any], camera_id: str = "") -> List[Detection]:
    """
    Convert raw pipeline output → list[Detection].
    - Maps raw class names through LABEL_MAP.
    - Filters detections below MIN_CONFIDENCE.
    - Sets event_type and equipment_type.
    """
    now = datetime.utcnow().isoformat()
    detections: List[Detection] = []

    for raw_det in raw.get("detections", []):
        conf = float(raw_det.get("confidence", 0.0))
        if conf < MIN_CONFIDENCE:
            continue

        raw_label  = str(raw_det.get("class_name", ""))
        class_name = LABEL_MAP.get(raw_label, raw_label.lower().replace(" ", "_"))

        event_type     = _resolve_event_type(class_name)
        equipment_type = class_name.capitalize() if class_name in EQUIPMENT_CLASSES else None

        det = Detection(
            id             = str(uuid.uuid4()),
            class_name     = class_name,
            raw_label      = raw_label,
            confidence     = conf,
            bbox           = list(raw_det.get("bbox", [])),
            model_source   = str(raw_det.get("model_source", raw.get("model_source", ""))),
            timestamp      = now,
            camera_id      = camera_id,
            event_type     = event_type,
            equipment_type = equipment_type,
        )
        detections.append(det)

    return detections


def _resolve_event_type(class_name: str) -> str:
    if class_name in PPE_VIOLATIONS:
        return "PPE"
    if class_name in EQUIPMENT_CLASSES:
        return "Equipment"
    if class_name == "crack":
        return "Crack"
    if class_name == "person":
        return "Personnel"
    return "Observation"


# =========================================================
# STAGE 3 — TRACKER
# =========================================================

def run_tracker(detections: List[Detection]) -> List[Detection]:
    """
    Assign stable tracking IDs and update per-object trajectory history.

    If EnterpriseTracker is importable, it is used directly via update().
    Otherwise the lightweight stub assigns sequential IDs so every downstream
    stage has a non-None tracking_id to work with.

    EnterpriseTracker.update() contract expected:
        results = tracker.update(bboxes, confidences, class_names)
        each result exposes:
            .tracking_id   int
            .trajectory    List[(x,y)]
            .velocity      float  (px/frame)
            .direction     float  (degrees)
            .age           int
            .frames_tracked int
    """
    if _USE_ENTERPRISE_TRACKER and _tracker is not None:
        bboxes      = [d.bbox       for d in detections]
        confidences = [d.confidence for d in detections]
        classes     = [d.class_name for d in detections]

        try:
            results = _tracker.update(bboxes, confidences, classes)
            for det, res in zip(detections, results):
                det.tracking_id   = res.tracking_id
                det.trajectory    = list(res.trajectory)
                det.velocity      = float(getattr(res, "velocity",      0.0))
                det.direction     = float(getattr(res, "direction",     0.0))
                det.age           = int(getattr(res, "age",             0))
                det.frames_tracked = int(getattr(res, "frames_tracked", 0))
            return detections
        except Exception:
            pass   # fall through to stub on any tracker error

    # --- Stub ---
    for i, det in enumerate(detections):
        if det.tracking_id is None:
            det.tracking_id = i

    return detections


# =========================================================
# STAGE 4 — RISK CLASSIFICATION
# =========================================================

def classify_risk(detections: List[Detection]) -> List[Detection]:
    """
    Tag each detection with risk, incident_type, priority, and confidence_level.
    PPE violations are evaluated first; no rule may downgrade an already-higher risk.
    """
    for det in detections:
        label = det.class_name.lower()

        # --- PPE violations ---
        if label in PPE_VIOLATIONS:
            new_risk = PPE_VIOLATIONS[label]
            if SEVERITY_ORDER.get(new_risk, 0) >= SEVERITY_ORDER.get(det.risk, 0):
                det.risk     = new_risk
            det.priority      = max(det.priority, 3)
            det.incident_type = "PPE Non-Compliance"

        # --- Infrastructure cracks — never downgrade a PPE classification ---
        elif label == "crack":
            if SEVERITY_ORDER.get(det.risk, 0) < SEVERITY_ORDER["MEDIUM"]:
                det.risk = "MEDIUM"
            det.priority      = max(det.priority, 2)
            det.incident_type = "Infrastructure Degradation"

        # --- Equipment ---
        elif label in EQUIPMENT_CLASSES:
            det.incident_type = "Equipment Activity"
            det.priority      = max(det.priority, 1)

        # --- Person ---
        elif label == "person":
            det.incident_type = "Personnel Activity"

        # --- Confidence tier ---
        det.confidence_level = _confidence_tier(det.confidence)

    return detections


def _confidence_tier(conf: float) -> str:
    for threshold, label in CONFIDENCE_TIERS:
        if conf >= threshold:
            return label
    return "Low"


# =========================================================
# STAGE 5 — DANGER ZONE INTELLIGENCE
# =========================================================

def analyze_intrusions(detections: List[Detection]) -> List[Dict[str, Any]]:
    """
    Check each person-class detection against every danger zone.
    Zones are evaluated highest-risk first; only the first match wins.
    Mutates det.risk / det.danger_zone / det.zone_name / det.zone_level /
            det.priority / det.distance_to_zone / det.event_type in-place.
    Returns a list of alert dicts for the HUD and result payload.
    """
    alerts: List[Dict[str, Any]] = []

    sorted_zones = sorted(
        DANGER_ZONES,
        key=lambda z: SEVERITY_ORDER.get(z["risk"], 0),
        reverse=True
    )

    for det in detections:
        if "person" not in det.class_name.lower():
            continue
        if len(det.bbox) != 4:
            continue

        x1, y1, x2, y2 = det.bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        for zone in sorted_zones:
            pts = np.array(zone["polygon"], dtype=np.int32)
            dist = cv2.pointPolygonTest(pts, (float(cx), float(cy)), True)  # signed distance

            inside = dist >= 0

            if inside:
                det.risk             = zone["risk"]
                det.danger_zone      = True
                det.zone_name        = zone["name"]
                det.zone_level       = zone["risk"]
                det.distance_to_zone = 0.0
                det.priority         = 5
                det.event_type       = "DangerZone"

                alerts.append({
                    "zone":     zone["name"],
                    "severity": zone["risk"],
                    "center":   [cx, cy],
                })
                break

            else:
                # Track closest zone for analytics even if not inside
                abs_dist = abs(dist)
                if det.distance_to_zone < 0 or abs_dist < det.distance_to_zone:
                    det.distance_to_zone = round(abs_dist, 1)

    return alerts


# =========================================================
# STAGE 5.5 — ENTERPRISE INTELLIGENCE / REASONING ENGINE
# =========================================================
#
# This is the layer that turns bare labels into operational language:
#
#   Worker detected → Helmet missing → High Risk
#       instead of just "Helmet detected."
#
#   "Excavator"               → "Excavator operating near workers."
#   "Crack"                   → "Structural crack detected. Inspection recommended."
#
# It never changes risk/priority (that already happened in classify_risk /
# analyze_intrusions) — it only attaches the human-readable `reasoning`
# string, plus `missing_ppe` / `nearby_equipment` context, to each Detection.

def build_reasoning(detections: List[Detection]) -> List[Detection]:
    """
    Populates det.reasoning (and det.missing_ppe / det.nearby_equipment)
    for every detection using:
      - PPE chain reasoning   (worker ↔ missing-PPE proximity linking)
      - Equipment reasoning   (equipment ↔ nearby-worker proximity linking)
      - Crack reasoning       (static, always the same operational sentence)
      - Danger-zone reasoning (overrides everything if a worker is in-zone)
    """
    people     = [d for d in detections if d.class_name == "person"]
    violations = [d for d in detections if d.class_name in PPE_VIOLATIONS]
    equipment  = [d for d in detections if d.class_name in EQUIPMENT_CLASSES]

    for det in detections:
        label = det.class_name.lower()

        # --- Danger zone takes narrative priority over everything else ---
        if det.danger_zone and det.zone_name:
            det.reasoning = (
                f"Worker detected inside {det.zone_name.title()}. "
                f"{det.zone_level.title() if det.zone_level else 'High'} risk — "
                f"immediate distance required."
            )
            continue

        # --- PPE violation classes (model emitted "no_helmet" etc. directly) ---
        if label in PPE_VIOLATIONS:
            item = PPE_ITEM_NAMES.get(label, label.replace("no_", "").replace("_", " "))
            risk_label = PPE_VIOLATIONS[label].title()
            det.missing_ppe = [item]
            det.reasoning = (
                f"Worker detected. {item.capitalize()} missing. {risk_label} risk."
            )
            continue

        # --- Equipment: contextualize against nearby workers ---
        if label in EQUIPMENT_CLASSES:
            near_workers = _nearby(det, people, EQUIPMENT_PROXIMITY_PX)
            det.nearby_equipment = []  # equipment doesn't track "nearby equipment" on itself
            if near_workers:
                det.reasoning = EQUIPMENT_REASONING_TEMPLATES.get(
                    label, f"{label.capitalize()} operating near workers."
                )
            else:
                det.reasoning = EQUIPMENT_REASONING_ISOLATED.get(
                    label, f"{label.capitalize()} active on site."
                )
            continue

        # --- Structural cracks: fixed operational sentence ---
        if label == "crack":
            det.reasoning = CRACK_REASONING
            continue

        # --- Person: infer missing PPE from absence of positive PPE classes
        #     nearby, OR from linked no_x violation detections, then build the
        #     "Worker detected → X missing → Risk" chain. If nothing is
        #     missing, report compliance explicitly. ---
        if label == "person":
            linked_violations = _nearby(det, violations, PPE_LINK_RADIUS_PX)
            missing = sorted({
                PPE_ITEM_NAMES.get(v.class_name, v.class_name)
                for v in linked_violations
            })
            det.missing_ppe      = missing
            det.nearby_equipment = [
                e.equipment_type or e.class_name.capitalize()
                for e in _nearby(det, equipment, EQUIPMENT_PROXIMITY_PX)
            ]

            if missing:
                # Escalate to the worst risk among the linked violations,
                # mirroring classify_risk's HIGH > MEDIUM > LOW ordering.
                worst = max(
                    (v.class_name for v in linked_violations),
                    key=lambda c: SEVERITY_ORDER.get(PPE_VIOLATIONS.get(c, "LOW"), 0)
                )
                risk_label = PPE_VIOLATIONS.get(worst, "LOW").title()
                items_str  = ", ".join(missing)
                det.reasoning = PPE_MISSING_TEMPLATE.format(items=items_str, risk_label=risk_label)

                # Propagate the violation's risk onto the worker themselves —
                # this is the "Worker detected → Helmet missing → High Risk"
                # chain. Never downgrade a risk a person already holds
                # (e.g. from a danger-zone intrusion classified upstream).
                escalated_risk = PPE_VIOLATIONS.get(worst, "LOW")
                if SEVERITY_ORDER.get(escalated_risk, 0) >= SEVERITY_ORDER.get(det.risk, 0):
                    det.risk = escalated_risk
                det.priority      = max(det.priority, 3)
                det.incident_type = "PPE Non-Compliance"
            else:
                det.reasoning = PPE_OK_REASONING
            continue

        # --- Fallback for anything uncategorized ---
        if not det.reasoning:
            det.reasoning = f"{det.class_name.replace('_', ' ').capitalize()} detected."

    return detections


def _bbox_center(det: Detection) -> Optional[tuple]:
    if len(det.bbox) != 4:
        return None
    x1, y1, x2, y2 = det.bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _nearby(anchor: Detection, candidates: List[Detection], radius_px: float) -> List[Detection]:
    """Return every candidate whose bbox center lies within radius_px of anchor's center."""
    a_center = _bbox_center(anchor)
    if a_center is None:
        return []

    found = []
    for cand in candidates:
        if cand is anchor:
            continue
        c_center = _bbox_center(cand)
        if c_center is None:
            continue
        dx = a_center[0] - c_center[0]
        dy = a_center[1] - c_center[1]
        if math.hypot(dx, dy) <= radius_px:
            found.append(cand)
    return found


# =========================================================
# STAGE 6 — ANALYTICS & ENTERPRISE NARRATIVE
# =========================================================

def calculate_analytics(detections: List[Detection], alerts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    alerts = alerts or []
    score   = 0
    highest = "LOW"

    workers        = 0
    helmet_count   = 0
    vest_count     = 0
    boots_count    = 0
    gloves_count   = 0
    equipment_count = 0
    crack_count    = 0
    ppe_violations = 0
    danger_zone_count = 0

    for det in detections:
        risk = det.risk.upper()
        score += RISK_SCORES.get(risk.lower(), 10)
        if SEVERITY_ORDER.get(risk, 0) > SEVERITY_ORDER.get(highest, 0):
            highest = risk

        cn = det.class_name
        if cn == "person":          workers        += 1
        elif cn == "helmet":        helmet_count   += 1
        elif cn == "vest":          vest_count     += 1
        elif cn == "boots":         boots_count    += 1
        elif cn == "gloves":        gloves_count   += 1
        elif cn in EQUIPMENT_CLASSES: equipment_count += 1
        elif cn == "crack":         crack_count    += 1
        if cn in PPE_VIOLATIONS:    ppe_violations += 1
        if det.danger_zone:         danger_zone_count += 1

    ppe_compliance = (
        round((1 - ppe_violations / max(workers, 1)) * 100, 1)
        if workers > 0 else 100.0
    )

    # -----------------------------------------------------
    # UI READY SUMMARIES (added without changing old fields)
    # -----------------------------------------------------

    ppe_summary = {
        "helmet": "Detected" if helmet_count > 0 else "Missing",
        "vest": "Detected" if vest_count > 0 else "Missing",
        "boots": "Detected" if boots_count > 0 else "Missing",
        "gloves": "Detected" if gloves_count > 0 else "Missing",
    }

    equipment = [
        det.equipment_type or det.class_name.capitalize()
        for det in detections
        if det.class_name in EQUIPMENT_CLASSES
    ]

    # remove duplicates while preserving order
    equipment = list(dict.fromkeys(equipment))

    recommendations = []

    if ppe_summary["helmet"] == "Missing":
        recommendations.append("Provide safety helmet.")

    if ppe_summary["vest"] == "Missing":
        recommendations.append("Provide reflective vest.")

    if ppe_summary["boots"] == "Missing":
        recommendations.append("Provide safety boots.")

    if danger_zone_count > 0:
        recommendations.append("Maintain safe distance from danger zones.")

    if equipment:
        recommendations.append("Maintain safe distance from machinery.")

    # -----------------------------------------------------
    # ENTERPRISE INTELLIGENCE — NARRATIVE GENERATORS
    # -----------------------------------------------------
    # These read det.reasoning (set in build_reasoning / Stage 5.5) so the
    # whole pipeline speaks in operational language end-to-end, e.g.:
    #   Worker detected → Helmet missing → High Risk
    #   Excavator operating near workers.
    #   Structural crack detected. Inspection recommended.

    findings        = generate_findings(detections, alerts)
    risk_factors    = generate_risk_factors(detections, alerts)
    compliance      = generate_compliance(detections, workers, ppe_violations, ppe_compliance)
    statistics      = generate_statistics(detections, workers, equipment_count, crack_count,
                                           ppe_violations, danger_zone_count, highest)
    summary         = generate_summary(workers, equipment_count, crack_count, danger_zone_count,
                                        ppe_violations, highest, ppe_compliance)

    return {
        # existing analytics kept unchanged
        "workers":           workers,
        "helmet_count":      helmet_count,
        "vest_count":        vest_count,
        "boots_count":       boots_count,
        "gloves_count":      gloves_count,
        "equipment_count":   equipment_count,
        "crack_count":       crack_count,
        "ppe_violations":    ppe_violations,
        "ppe_compliance":    ppe_compliance,
        "danger_zone_count": danger_zone_count,
        "risk_score":        min(score, 100),
        "overall_risk":      highest,

        # frontend-ready fields
        "ppe_summary":       ppe_summary,
        "equipment":         equipment,
        "recommendations":   recommendations,

        # Enterprise Intelligence narrative layer
        "findings":          findings,
        "summary":           summary,
        "compliance":        compliance,
        "risk_factors":      risk_factors,
        "statistics":        statistics,
    }


# ---------------------------------------------------------
# Findings — one narrative line per noteworthy detection,
# e.g. "Worker detected. Helmet missing. High risk."
#       "Excavator operating near workers."
#       "Structural crack detected. Inspection recommended."
# ---------------------------------------------------------

def generate_findings(detections: List[Detection], alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings = []

    for det in detections:
        if not det.reasoning:
            continue
        # Skip raw violation/positive-PPE classes already folded into their
        # linked person's finding, to avoid duplicate noise in the findings feed.
        if det.class_name in PPE_VIOLATIONS:
            continue

        findings.append({
            "id":         det.id,
            "class_name": det.class_name,
            "event_type": det.event_type,
            "risk":       det.risk,
            "tracking_id": det.tracking_id,
            "finding":    det.reasoning,
        })

    # Sort: critical findings first
    findings.sort(key=lambda f: SEVERITY_ORDER.get(str(f["risk"]).upper(), 0), reverse=True)
    return findings


# ---------------------------------------------------------
# Risk factors — the distinct hazard categories present, with
# a short cause statement, used to drive a "why is risk elevated" panel.
# ---------------------------------------------------------

def generate_risk_factors(detections: List[Detection], alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    factors: List[Dict[str, Any]] = []
    seen = set()

    for det in detections:
        if det.danger_zone and "danger_zone" not in seen:
            seen.add("danger_zone")
            factors.append({
                "factor":   "Danger Zone Intrusion",
                "severity": det.zone_level or "HIGH",
                "detail":   f"Personnel detected inside {det.zone_name}.",
            })

        if det.missing_ppe and "ppe_gap" not in seen:
            seen.add("ppe_gap")
            factors.append({
                "factor":   "PPE Non-Compliance",
                "severity": "HIGH" if "helmet" in det.missing_ppe else "MEDIUM",
                "detail":   "One or more workers missing required protective equipment.",
            })

        if det.class_name in EQUIPMENT_CLASSES and det.nearby_equipment == [] and "equipment" not in seen:
            # only flag once; proximity-specific detail comes from reasoning text
            pass

        if det.class_name in EQUIPMENT_CLASSES and "equipment_proximity" not in seen:
            near = _nearby(det, [d for d in detections if d.class_name == "person"], EQUIPMENT_PROXIMITY_PX)
            if near:
                seen.add("equipment_proximity")
                factors.append({
                    "factor":   "Heavy Equipment Proximity",
                    "severity": "HIGH",
                    "detail":   f"{det.equipment_type or det.class_name.capitalize()} operating near personnel.",
                })

        if det.class_name == "crack" and "structural" not in seen:
            seen.add("structural")
            factors.append({
                "factor":   "Structural Degradation",
                "severity": "MEDIUM",
                "detail":   CRACK_REASONING,
            })

    if not factors:
        factors.append({
            "factor":   "None",
            "severity": "LOW",
            "detail":   "No elevated risk factors identified in this frame.",
        })

    return factors


# ---------------------------------------------------------
# Compliance — a focused PPE/site-compliance scorecard.
# ---------------------------------------------------------

def generate_compliance(detections: List[Detection], workers: int, ppe_violations: int,
                         ppe_compliance: float) -> Dict[str, Any]:
    compliant_workers = max(workers - ppe_violations, 0) if workers else 0

    status = "Compliant"
    if ppe_compliance < 50:
        status = "Critical Non-Compliance"
    elif ppe_compliance < 80:
        status = "Partial Compliance"
    elif ppe_compliance < 100:
        status = "Minor Gaps"

    return {
        "status":             status,
        "ppe_compliance_pct": ppe_compliance,
        "compliant_workers":  compliant_workers,
        "total_workers":      workers,
        "violations":         ppe_violations,
    }


# ---------------------------------------------------------
# Statistics — flat numeric snapshot, convenient for dashboards/exports.
# ---------------------------------------------------------

def generate_statistics(detections: List[Detection], workers: int, equipment_count: int,
                         crack_count: int, ppe_violations: int, danger_zone_count: int,
                         overall_risk: str) -> Dict[str, Any]:
    risk_breakdown = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for det in detections:
        risk_breakdown[det.risk.upper()] = risk_breakdown.get(det.risk.upper(), 0) + 1

    avg_confidence = (
        round(sum(d.confidence for d in detections) / len(detections), 3)
        if detections else 0.0
    )

    return {
        "total_detections":   len(detections),
        "workers":             workers,
        "equipment":           equipment_count,
        "structural_cracks":   crack_count,
        "ppe_violations":      ppe_violations,
        "danger_zone_events":  danger_zone_count,
        "overall_risk":        overall_risk,
        "risk_breakdown":      risk_breakdown,
        "average_confidence":  avg_confidence,
    }


# ---------------------------------------------------------
# Summary — one paragraph, plain-English, frame-level rollup.
# ---------------------------------------------------------

def generate_summary(workers: int, equipment_count: int, crack_count: int,
                      danger_zone_count: int, ppe_violations: int, overall_risk: str,
                      ppe_compliance: float) -> str:
    parts = []

    if workers == 0:
        parts.append("No personnel detected in frame.")
    else:
        parts.append(f"{workers} worker{'s' if workers != 1 else ''} detected.")

    if ppe_violations > 0:
        parts.append(f"{ppe_violations} PPE violation{'s' if ppe_violations != 1 else ''} identified "
                      f"({ppe_compliance:.0f}% compliance).")
    elif workers > 0:
        parts.append("Full PPE compliance observed.")

    if equipment_count > 0:
        parts.append(f"{equipment_count} piece{'s' if equipment_count != 1 else ''} of heavy equipment active.")

    if crack_count > 0:
        parts.append(f"{crack_count} structural crack{'s' if crack_count != 1 else ''} flagged for inspection.")

    if danger_zone_count > 0:
        parts.append(f"{danger_zone_count} danger-zone intrusion{'s' if danger_zone_count != 1 else ''} recorded.")

    parts.append(f"Overall site risk: {overall_risk.title()}.")

    return " ".join(parts)


# =========================================================
# STAGE 7 — RENDERING
# =========================================================

def draw_frame(frame, detections: List[Detection], alerts: List[Dict], analytics: Dict):
    """
    All cv2 drawing in one place, in correct layer order:
      1. danger zone fills  (background)
      2. detection boxes + labels
      3. trajectory lines
      4. HUD overlay        (foreground)
    """
    draw_danger_zones(frame, alerts)
    draw_cinematic(frame, detections)
    draw_trajectories(frame, detections)
    draw_hud(frame, detections, alerts, analytics)


def draw_danger_zones(frame, intrusion_alerts: List[Dict]):
    pulse = abs(math.sin(time.time() * 3))

    for zone in DANGER_ZONES:
        intrusion = any(a["zone"] == zone["name"] for a in intrusion_alerts)
        color     = (0, 0, 255) if intrusion else (255, 120, 0)
        alpha     = (0.12 + 0.10 * pulse) if intrusion else (0.04 + 0.04 * pulse)

        overlay = frame.copy()
        pts     = np.array(zone["polygon"], np.int32)
        cv2.fillPoly(overlay, [pts], color)
        frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.polylines(frame, [pts], True, color, 3)


def draw_cinematic(frame, detections: List[Detection]):
    """
    Render bounding boxes with:
      - Risk-derived border color
      - Alpha-blended fill
      - Label showing tracking ID + class + confidence tier
    Detections are already sorted critical-last by process_frame, so
    critical boxes are drawn on top.
    """
    for det in detections:
        if len(det.bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, det.bbox)
        risk  = det.risk.lower()
        color = RISK_COLORS.get(risk, RISK_COLORS["low"])

        # Build label:  "Worker #12  Helmet  96%"  or  "Excavator #2"
        tier  = det.confidence_level
        pct   = f"{det.confidence * 100:.0f}%"
        if det.class_name in EQUIPMENT_CLASSES:
            label = f"{det.equipment_type or det.class_name.capitalize()} #{det.tracking_id}  {pct}"
        else:
            prefix = "Worker" if det.class_name == "person" else det.class_name.replace("_", " ").title()
            label  = f"{prefix} #{det.tracking_id}  {tier}  {pct}"

        # Filled semi-transparent box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 8)
        frame[:] = cv2.addWeighted(overlay, 0.14, frame, 0.86, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        pad = 8
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - pad * 2),
            (x1 + text_w + pad * 2, y1),
            color, -1
        )
        cv2.putText(
            frame, label,
            (x1 + pad, y1 - pad),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1, cv2.LINE_AA
        )


def draw_trajectories(frame, detections: List[Detection]):
    for det in detections:
        if len(det.trajectory) < 2:
            continue

        color = RISK_COLORS.get(det.risk.lower(), RISK_COLORS["low"])
        for i in range(1, len(det.trajectory)):
            cv2.line(frame, det.trajectory[i - 1], det.trajectory[i], color, 2)


def draw_hud(frame, detections: List[Detection], intrusion_alerts: List[Dict], analytics: Dict):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (10, 15, 25), -1)
    frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(
        frame, "InfraGuard Enterprise AI Surveillance",
        (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2
    )
    cv2.putText(
        frame, f"Objects: {len(detections)}",
        (w - 260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2
    )
    cv2.putText(
        frame, f"Risk: {analytics.get('overall_risk', 'N/A')}",
        (w - 460, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2
    )
    cv2.putText(
        frame, f"PPE: {analytics.get('ppe_compliance', 100.0):.0f}%",
        (w - 620, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2
    )

    if intrusion_alerts:
        cv2.putText(
            frame, "DANGER ZONE INTRUSION",
            (w // 2 - 180, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 0, 255), 3
        )

    cv2.putText(
        frame, time.strftime("%d-%m-%Y %H:%M:%S"),
        (18, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2
    )


# =========================================================
# AI METADATA
# =========================================================

def build_ai_metadata(processing_ms: float = 0.0) -> Dict[str, Any]:
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    return {
        "engine":   "InfraGuard Enterprise AI",
        "pipeline": "Realtime Safety Intelligence",
        "models": {
            "infraguard": "InfraGuard PPE + Equipment",
            "crack":      "InfraGuard Crack Detection",
        },
        "device":          device,
        "processing_ms":   processing_ms,
        "analysis_mode":   "Operational Surveillance",
        "timestamp":       datetime.utcnow().isoformat(),
    }


# =========================================================
# EMPTY / ERROR FALLBACK
# =========================================================

def _empty(camera_id: str = "") -> Dict[str, Any]:
    return {
        "frame":       None,
        "detections":  [],
        "alerts":      [],
        "zones":       DANGER_ZONES,
        "analytics":   {
            "workers": 0, "helmet_count": 0, "vest_count": 0,
            "boots_count": 0, "gloves_count": 0, "equipment_count": 0,
            "crack_count": 0, "ppe_violations": 0, "ppe_compliance": 100.0,
            "danger_zone_count": 0, "risk_score": 0, "overall_risk": "LOW",
            "ppe_summary": {
                "helmet": "Missing",
                "vest": "Missing",
                "boots": "Missing",
                "gloves": "Missing",
            },
            "equipment": [],
            "recommendations": [],

            # Enterprise Intelligence narrative layer
            "findings":     [],
            "summary":      "No personnel detected in frame. Overall site risk: Low.",
            "compliance": {
                "status":             "Compliant",
                "ppe_compliance_pct": 100.0,
                "compliant_workers":  0,
                "total_workers":      0,
                "violations":         0,
            },
            "risk_factors": [{
                "factor":   "None",
                "severity": "LOW",
                "detail":   "No elevated risk factors identified in this frame.",
            }],
            "statistics": {
                "total_detections":  0,
                "workers":           0,
                "equipment":         0,
                "structural_cracks": 0,
                "ppe_violations":    0,
                "danger_zone_events": 0,
                "overall_risk":      "LOW",
                "risk_breakdown":    {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0},
                "average_confidence": 0.0,
            },
        },
        "telemetry":   {"processing_ms": 0.0, "detection_count": 0},
        "ai_metadata": build_ai_metadata(),
        "timestamp":   datetime.utcnow().isoformat(),
        "camera_id":   camera_id,
    }