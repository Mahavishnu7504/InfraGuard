# =========================================================
# INFRA GUARD — ENTERPRISE SAFETY INTELLIGENCE ENGINE
# detection_service.py  v3.0
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

from backend.services.safety.risk_engine.rules import (
    evaluate_risk,
    compute_severity,
    detect_ppe_violations,
    detect_vehicle_proximity,
    detect_danger_zones,
)
import backend.services.safety.risk_engine.rules as rules

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

# Persistent tracking ID counter for the stub tracker (increments across frames
# so IDs never reset even when EnterpriseTracker is unavailable).
_tracking_counter: int = 0

MIN_CONFIDENCE: float = 0.35   # detections below this are rejected (logged, not silent)

# Debug mode — enables stage counts, rejection reports, confidence analytics,
# class distribution, telemetry breakdown, and validation warnings.
# Set to False for normal production behaviour.
DETECTION_DEBUG: bool = True

# Canonical set of allowed class names after normalisation.
# Any label not in this set will be logged as an unknown label warning.
CANONICAL_CLASSES = {
    "person", "helmet", "vest", "boots", "gloves",
    "no_helmet", "no_vest", "no_gloves", "no_boots",
    "crack",
    "excavator", "loader", "bulldozer", "roller", "grader", "mobile_crane",
}

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
    "crane":        "mobile_crane",
    "Crane":        "mobile_crane",
}

EQUIPMENT_CLASSES = {"excavator", "loader", "bulldozer", "roller", "grader", "mobile_crane"}

# How close (px, bbox-center to bbox-center) equipment must be to a person
# before it is considered "nearby" for *narrative* reasoning text only.
# (The PPE-chain proximity radius used for *risk* decisions lives in rules.py.)
EQUIPMENT_PROXIMITY_PX: float = 260.0

# =========================================================
# DANGER ZONES
# =========================================================
# Single source of truth for zone geometry/severity lives in rules.py.
# detection_service.py reads it for rendering (draw_danger_zones) and for
# including in the returned payload — it never decides zone risk itself.

DANGER_ZONES = rules.DEFAULT_DANGER_ZONES

# =========================================================
# PPE INTELLIGENCE
# =========================================================
# PPE_VIOLATIONS / PPE_ITEM_NAMES are risk-engine inputs, owned by rules.py.
# detection_service.py only re-uses them for narrative text and analytics
# grouping — it never assigns risk from them directly.

PPE_VIOLATIONS: Dict[str, str] = rules.PPE_VIOLATIONS
PPE_ITEM_NAMES: Dict[str, str] = rules.PPE_ITEM_NAMES

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
    "excavator":    "Excavator operating near workers.",
    "loader":       "Loader operating near workers.",
    "bulldozer":    "Bulldozer operating near workers.",
    "roller":       "Roller operating near workers.",
    "grader":       "Grader operating near workers.",
    "mobile_crane": "Mobile crane operating near workers.",
}

EQUIPMENT_REASONING_ISOLATED: Dict[str, str] = {
    "excavator":    "Excavator active on site.",
    "loader":       "Loader active on site.",
    "bulldozer":    "Bulldozer active on site.",
    "roller":       "Roller active on site.",
    "grader":       "Grader active on site.",
    "mobile_crane": "Mobile crane active on site.",
}

CRACK_REASONING: str = "Structural crack detected. Inspection recommended."

PPE_OK_REASONING:  str = "Worker detected wearing all mandatory PPE. No immediate safety violations identified."
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

SEVERITY_ORDER = {
    "SAFE": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}

RISK_COLORS: Dict[str, tuple] = {
    "safe":     (0,   255, 0),
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

    # --- Risk classification (decided by rules.py, never by this module) ---
    risk:             str = "LOW"
    incident_type:    str = "Operational Observation"
    priority:         int = 1
    confidence_level: str = "Low"

    # --- Zone intelligence (decided by rules.py, never by this module) ---
    danger_zone:    bool          = False
    zone_name:      Optional[str] = None
    zone_level:     Optional[str] = None
    distance_to_zone: float       = -1.0  # px; -1 = not measured

    # --- Narrative intelligence (set by build_reasoning) ---
    reasoning:        str       = ""   # human-readable "why" behind the risk tag
    missing_ppe:      List[str] = field(default_factory=list)   # PPE absent on this worker
    nearby_equipment: List[str] = field(default_factory=list)   # equipment within proximity radius

    # --- Audit trail (lifecycle stages this detection has passed through) ---
    audit_trail:      List[str] = field(default_factory=list)

    # --- Risk decision trace (why this detection received its risk label) ---
    risk_decision_trace: List[str] = field(default_factory=list)

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
          ↓  validate_detections  — validate bbox / confidence / class / source / timestamp
          ↓  normalize_labels     — map raw output → list[Detection]
          ↓  run_tracker          — assign tracking_id + trajectory
          ↓  tag_confidence_level — model-quality tier from confidence (not risk)
          ↓  rules.evaluate_risk  — THE ONLY place risk is decided (PPE / crack /
                                    equipment / person baseline, PPE-chain escalation,
                                    danger-zone intrusion). Returns risk-tagged
                                    detections + zone alerts. detection_service.py
                                    never assigns det.risk itself past this point.
          ↓  build_reasoning      — Enterprise Intelligence: turns labels + the risk
                                    rules.py already decided into narrative findings
                                    ("Worker detected → Helmet missing → High Risk",
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
    stage_times: Dict[str, float] = {}
    rejected_detections: List[Dict] = []

    def _tick(label: str, t_start: float) -> float:
        """Record elapsed ms since t_start; return now."""
        t_now = time.perf_counter()
        stage_times[label] = round((t_now - t_start) * 1000, 2)
        return t_now

    try:
        # ── Stage 1: Inference ──────────────────────────────────────────────
        ts = time.perf_counter()
        raw = run_inference(frame)
        ts = _tick("inference_ms", ts)

        raw_count = len(raw.get("detections", []))

        # ── Stage 1.5: Validation ───────────────────────────────────────────
        ts = time.perf_counter()
        valid_raw, rejected_detections = validate_detections(raw.get("detections", []))
        raw["detections"] = valid_raw
        ts = _tick("validation_ms", ts)

        validated_count = len(valid_raw)

        # ── Stage 2: Normalise ──────────────────────────────────────────────
        ts = time.perf_counter()
        detections = normalize_labels(raw, camera_id)
        ts = _tick("normalization_ms", ts)

        norm_count = len(detections)

        # ── Stage 3: Tracker ────────────────────────────────────────────────
        ts = time.perf_counter()
        detections = run_tracker(detections)
        ts = _tick("tracking_ms", ts)

        tracked_count = len(detections)

        # ── Stage 3.5: Confidence tiering (not a risk decision) ──────────────
        ts = time.perf_counter()
        detections = tag_confidence_level(detections)
        ts = _tick("confidence_ms", ts)

        # ── Stage 4: Risk evaluation (rules.py is the only risk authority) ───
        ts = time.perf_counter()
        risk_result = rules.evaluate_risk([d.to_dict() for d in detections])
        alerts = risk_result.get("danger_zones", [])
        # map risk information back into detections
        for det in detections:
            for rd in risk_result.get("detections", []):
                if rd.get("id") == det.id:
                    det.risk = rd.get("risk", det.risk)
                    det.incident_type = rd.get("incident_type", det.incident_type)
                    det.priority = rd.get("priority", det.priority)
                    det.danger_zone = rd.get("danger_zone", det.danger_zone)
                    det.zone_name = rd.get("zone_name", det.zone_name)
                    det.zone_level = rd.get("zone_level", det.zone_level)
                    det.risk_decision_trace = rd.get("risk_decision_trace", det.risk_decision_trace)
                    det.audit_trail.append("risk_classified")
                    break
        ts = _tick("risk_ms", ts)

        risk_count = len(detections)

        # ── Stage 5: Reasoning ──────────────────────────────────────────────
        ts = time.perf_counter()
        detections = build_reasoning(detections)
        ts = _tick("reasoning_ms", ts)

        reasoning_count = len(detections)

        # ── Stage 6: Analytics ──────────────────────────────────────────────
        ts = time.perf_counter()
        analytics = calculate_analytics(detections, alerts)
        ts = _tick("analytics_ms", ts)

        analytics_count = len(detections)

        # ── Sort: critical last so important boxes render on top ─────────────
        detections.sort(key=lambda d: SEVERITY_ORDER.get(d.risk.upper(), 0))

        # ── Stage 7: Rendering ──────────────────────────────────────────────
        ts = time.perf_counter()
        draw_frame(frame, detections, alerts, analytics)
        ts = _tick("rendering_ms", ts)

        processing_ms = round((time.perf_counter() - t0) * 1000, 1)
        stage_times["total_ms"] = processing_ms

        # ── Empty result diagnosis ───────────────────────────────────────────
        empty_reason: Optional[str] = None
        if len(detections) == 0:
            if raw_count == 0:
                empty_reason = "Inference returned zero objects."
            elif validated_count == 0:
                empty_reason = "All detections failed validation."
            elif norm_count == 0:
                empty_reason = "All detections rejected due to confidence threshold."
            elif tracked_count == 0:
                empty_reason = "Tracker removed all detections."
            else:
                empty_reason = "Detections lost during downstream processing."
            print(f"[DETECTION] No detections returned. Reason: {empty_reason}")

        # ── Integrity report ────────────────────────────────────────────────
        integrity_report = {
            "raw_detections":  raw_count,
            "validated":       validated_count,
            "rejected":        len(rejected_detections),
            "normalized":      norm_count,
            "tracked":         tracked_count,
            "risk_classified": risk_count,
            "analytics_counted": analytics_count,
            "returned":        len(detections),
            "empty_reason":    empty_reason,
        }

        # ── Debug output ────────────────────────────────────────────────────
        if DETECTION_DEBUG:
            _print_debug_report(
                stage_times, integrity_report, rejected_detections,
                analytics, detections
            )

        return {
            "frame":       None,   # caller injects encoded frame bytes if needed
            "detections":  [d.to_dict() for d in detections],
            "alerts":      alerts,
            "zones":       DANGER_ZONES if DANGER_ZONES else [],
            "analytics":   analytics,
            "telemetry":   {
                "processing_ms":    processing_ms,
                "detection_count":  len(detections),
                "stage_times":      stage_times,
            },
            "ai_metadata":       build_ai_metadata(processing_ms),
            "integrity_report":  integrity_report,
            "rejected_detections": rejected_detections if DETECTION_DEBUG else [],
            "timestamp":         datetime.utcnow().isoformat(),
            "camera_id":         camera_id,
        }

    except Exception:
        print("\n" + "=" * 80)
        print("[DETECTION ERROR] process_frame() failed:")
        traceback.print_exc()
        print("=" * 80 + "\n")
        return _empty(camera_id)


def _print_debug_report(
    stage_times: Dict[str, float],
    integrity: Dict[str, Any],
    rejected: List[Dict],
    analytics: Dict[str, Any],
    detections: List["Detection"],
) -> None:
    """Print a structured debug report to stdout when DETECTION_DEBUG is True."""
    sep = "-" * 60
    print(f"\n{'=' * 60}")
    print("  DETECTION DEBUG REPORT")
    print(f"{'=' * 60}")

    # Stage counts / pipeline flow
    print("\n[Pipeline Flow]")
    print(f"  Raw Inference      : {integrity['raw_detections']}")
    print(f"  Validated          : {integrity['validated']}")
    print(f"  Rejected           : {integrity['rejected']}")
    print(f"  Normalized         : {integrity['normalized']}")
    print(f"  Tracked            : {integrity['tracked']}")
    print(f"  Risk Classified    : {integrity['risk_classified']}")
    print(f"  Analytics Counted  : {integrity['analytics_counted']}")
    print(f"  Returned           : {integrity['returned']}")
    if integrity["empty_reason"]:
        print(f"  ⚠ Empty Reason     : {integrity['empty_reason']}")

    # Stage timings
    print(f"\n[Stage Timings]")
    for stage, ms in stage_times.items():
        print(f"  {stage:<22}: {ms:>7.2f} ms")

    # Rejection report
    if rejected:
        print(f"\n[Rejected Detections — {len(rejected)}]")
        for r in rejected:
            print(f"  ✗ label={r.get('raw_label','?')!r:20s}  reason={r.get('reason','?')}")

    # Class distribution
    class_dist: Dict[str, int] = {}
    for det in detections:
        class_dist[det.class_name] = class_dist.get(det.class_name, 0) + 1
    if class_dist:
        print(f"\n[Class Distribution]")
        for cls, count in sorted(class_dist.items()):
            print(f"  {cls:<20}: {count}")

    # Confidence analytics
    if detections:
        confs = [d.confidence for d in detections]
        print(f"\n[Confidence Analytics]")
        print(f"  Highest  : {max(confs):.3f}")
        print(f"  Lowest   : {min(confs):.3f}")
        median_conf = sorted(confs)[len(confs) // 2]
        print(f"  Median   : {median_conf:.3f}")
        print(f"  Average  : {sum(confs)/len(confs):.3f}")
        below = sum(1 for c in confs if c < MIN_CONFIDENCE)
        print(f"  Below threshold ({MIN_CONFIDENCE}): {below}")

    # Analytics consistency warnings
    _validate_analytics(analytics)

    print(f"{'=' * 60}\n")


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
# STAGE 1.5 — DETECTION VALIDATION
# =========================================================

def validate_detections(raw_detections: List[Dict]) -> tuple:
    """
    Validate every raw detection before normalisation.
    Returns (valid_list, rejected_list).

    Each rejected entry is a dict with the original data plus a 'reason' field
    explaining why it was rejected. Invalid detections are logged and skipped —
    never silently passed downstream.
    """
    valid:    List[Dict] = []
    rejected: List[Dict] = []

    for raw_det in raw_detections:
        reason = _check_detection(raw_det)
        if reason:
            entry = {**raw_det, "reason": reason, "raw_label": raw_det.get("class_name", "<none>")}
            rejected.append(entry)
            if DETECTION_DEBUG:
                print(f"[VALIDATION] Rejected detection — {reason} | raw={raw_det}")
        else:
            valid.append(raw_det)

    return valid, rejected


def _check_detection(raw_det: Dict) -> Optional[str]:
    """
    Return a rejection reason string if the detection is invalid, else None.
    Checks: bbox, confidence, class_name, model_source, timestamp.
    """
    bbox = raw_det.get("bbox")
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return "Invalid or missing bbox"

    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return "Bbox values are not numeric"

    if x2 <= x1 or y2 <= y1:
        return "Bbox has zero or negative area"

    conf = raw_det.get("confidence")
    if conf is None:
        return "Missing confidence"
    try:
        if float(conf) < 0.0 or float(conf) > 1.0:
            return f"Confidence out of range: {conf}"
    except (TypeError, ValueError):
        return f"Confidence is not numeric: {conf!r}"

    class_name = raw_det.get("class_name")
    if not class_name or not isinstance(class_name, str) or not class_name.strip():
        return "Missing or empty class_name"

    if not raw_det.get("model_source") and not raw_det.get("source"):
        # model_source is optional in some pipelines — log but don't reject
        if DETECTION_DEBUG:
            print(f"[VALIDATION] Warning — detection missing model_source: class={class_name!r}")

    return None


# =========================================================
# STAGE 2 — NORMALIZE LABELS
# =========================================================

def normalize_labels(raw: Dict[str, Any], camera_id: str = "") -> List[Detection]:
    """
    Convert raw pipeline output → list[Detection].
    - Maps raw class names through LABEL_MAP.
    - Filters detections below MIN_CONFIDENCE (logged, not silent).
    - Validates normalised label against CANONICAL_CLASSES.
    - Sets event_type and equipment_type.
    - Starts each detection's audit_trail.
    """
    now = datetime.utcnow().isoformat()
    detections: List[Detection] = []

    for raw_det in raw.get("detections", []):
        conf = float(raw_det.get("confidence", 0.0))
        if conf < MIN_CONFIDENCE:
            if DETECTION_DEBUG:
                print(f"[NORMALIZE] Rejected — confidence {conf:.3f} < {MIN_CONFIDENCE} "
                      f"| label={raw_det.get('class_name','?')!r}")
            continue

        raw_label  = str(raw_det.get("class_name", ""))
        class_name = LABEL_MAP.get(raw_label, raw_label.lower().strip().replace(" ", "_"))

        # Label validation: warn if the normalised label is not in the canonical set
        if class_name not in CANONICAL_CLASSES:
            if DETECTION_DEBUG:
                print(f"[LABEL VALIDATION] Unknown label after normalisation: "
                      f"{class_name!r} (raw: {raw_label!r}). "
                      f"Check LABEL_MAP or CANONICAL_CLASSES.")

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
            audit_trail    = ["validated", "normalized"],
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
                det.audit_trail.append("tracked")
            return detections
        except Exception:
            pass   # fall through to stub on any tracker error

    # --- Stub: assign persistent IDs that don't reset between frames ---
    global _tracking_counter
    for det in detections:
        if det.tracking_id is None:
            _tracking_counter += 1
            det.tracking_id = _tracking_counter

    for det in detections:
        det.audit_trail.append("tracked")

    return detections


# =========================================================
# STAGE 4 — CONFIDENCE TIERING
# =========================================================
# NOTE: This is NOT risk evaluation. confidence_level is a model-quality
# label ("Verified" / "Good" / "Low" etc.) derived purely from det.confidence.
# All risk/incident_type/priority decisions live in rules.evaluate_risk().

def tag_confidence_level(detections: List[Detection]) -> List[Detection]:
    """Populate det.confidence_level from det.confidence. Never touches risk."""
    for det in detections:
        det.confidence_level = _confidence_tier(det.confidence)
        det.audit_trail.append("confidence_tagged")
    return detections


def _confidence_tier(conf: float) -> str:
    for threshold, label in CONFIDENCE_TIERS:
        if conf >= threshold:
            return label
    return "Low"


# =========================================================
# STAGE 5 — ENTERPRISE INTELLIGENCE / REASONING ENGINE
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
# It never decides risk/priority/incident_type — those were already decided
# by rules.evaluate_risk() upstream. This stage only READS det.risk and
# det.danger_zone to phrase a sentence; it attaches the human-readable
# `reasoning` string, plus `missing_ppe` / `nearby_equipment` context, to
# each Detection.

def build_reasoning(detections: List[Detection]) -> List[Detection]:
    """
    Populates det.reasoning (and det.missing_ppe / det.nearby_equipment)
    for every detection using:
      - PPE chain reasoning   (worker ↔ missing-PPE via bbox containment/IoU)
      - Equipment reasoning   (equipment ↔ nearby-worker proximity linking)
      - Crack reasoning       (static, always the same operational sentence)
      - Danger-zone reasoning (overrides everything if a worker is in-zone)

    PPE association uses bounding-box containment (PPE inside worker bbox) with
    IoU as a fallback, rather than only centre-distance proximity. This prevents
    boots/helmet from one worker being incorrectly assigned to an adjacent worker.
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
            det.audit_trail.append("reasoning_added")
            continue

        # --- PPE violation classes (model emitted "no_helmet" etc. directly) ---
        if label in PPE_VIOLATIONS:
            item = PPE_ITEM_NAMES.get(label, label.replace("no_", "").replace("_", " "))
            risk_label = PPE_VIOLATIONS[label].title()
            det.missing_ppe = [item]
            det.reasoning = (
                f"Worker detected. {item.capitalize()} missing. {risk_label} risk."
            )
            det.audit_trail.append("reasoning_added")
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
            det.audit_trail.append("reasoning_added")
            continue

        # --- Structural cracks: fixed operational sentence ---
        if label == "crack":
            det.reasoning = CRACK_REASONING
            det.audit_trail.append("reasoning_added")
            continue

        # --- Person: associate PPE using bbox containment + IoU, then fall back
        #     to proximity for violation classes not overlapping the worker bbox.
        #     Uses rules.py's own association helper so narrative linking and
        #     risk-decision linking can never disagree about whose PPE is missing.
        #
        #     This stage only READS det.risk (already decided by
        #     rules.evaluate_risk's PPE-chain rule) — it never assigns it. ---
        if label == "person":
            linked_ppe = rules.associate_ppe_to_person(
                det.bbox,
                [v.to_dict() for v in violations]
            )
            linked_violations = [
                v for v in violations
                if any(lp.get("id") == v.id for lp in linked_ppe)
            ]
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
                risk_label = det.risk.title()
                items_str  = ", ".join(missing)
                det.reasoning = PPE_MISSING_TEMPLATE.format(items=items_str, risk_label=risk_label)
            else:
                det.reasoning = PPE_OK_REASONING

            det.audit_trail.append("reasoning_added")
            continue

        # --- Fallback for anything uncategorized ---
        if not det.reasoning:
            det.reasoning = f"{det.class_name.replace('_', ' ').capitalize()} detected."
        det.audit_trail.append("reasoning_added")

    return detections


# ── Equipment/worker proximity helpers (narrative-only, not risk decisions) ──
# NOTE: PPE-to-worker association (_associate_ppe_to_worker, bbox IoU/containment)
# now lives exclusively in rules.py — build_reasoning calls rules._associate_ppe_to_worker
# directly so the narrative layer and the risk-decision layer can never disagree
# about which PPE detection belongs to which worker.

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
    highest = "SAFE"

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
        det_score = RISK_SCORES.get(risk.lower(), 10)
        if det_score > score:
            score = det_score          # base = highest single-detection score
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

    # Apply multi-hazard penalties on top of the highest base score
    if ppe_violations > 0:
        score = min(score + ppe_violations * 5, 100)
    if danger_zone_count > 0:
        score = min(score + danger_zone_count * 10, 100)
    if crack_count > 0:
        score = min(score + crack_count * 3, 100)
    score = min(score, 100)

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

    # Class distribution — count of each canonical class detected this frame
    class_distribution: Dict[str, int] = {}
    for det in detections:
        class_distribution[det.class_name] = class_distribution.get(det.class_name, 0) + 1

    # Stamp audit trail
    for det in detections:
        det.audit_trail.append("analytics_counted")

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

        # New: class distribution for dashboards and debugging
        "class_distribution": class_distribution,

        # Convenience alias for charts — mirrors statistics.risk_breakdown
        "risk_distribution":  statistics.get("risk_breakdown", {}),
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
            "id":          det.id,
            "class_name":  det.class_name,
            "event_type":  det.event_type,
            "risk":        det.risk,
            "priority":    det.priority,
            "confidence":  det.confidence,
            "tracking_id": det.tracking_id,
            "finding":     det.reasoning,
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
    risk_breakdown = {"SAFE": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for det in detections:
        risk_breakdown[det.risk.upper()] = risk_breakdown.get(det.risk.upper(), 0) + 1

    # Full confidence analytics
    if detections:
        confs  = sorted(d.confidence for d in detections)
        n      = len(confs)
        avg_c  = round(sum(confs) / n, 3)
        max_c  = round(confs[-1], 3)
        min_c  = round(confs[0], 3)
        med_c  = round(confs[n // 2], 3)
        below  = sum(1 for c in confs if c < MIN_CONFIDENCE)
    else:
        avg_c = max_c = min_c = med_c = 0.0
        below = 0

    return {
        "total_detections":   len(detections),
        "workers":             workers,
        "equipment":           equipment_count,
        "structural_cracks":   crack_count,
        "ppe_violations":      ppe_violations,
        "danger_zone_events":  danger_zone_count,
        "overall_risk":        overall_risk,
        "risk_breakdown":      risk_breakdown,
        # Expanded confidence analytics
        "average_confidence":  avg_c,
        "max_confidence":      max_c,
        "min_confidence":      min_c,
        "median_confidence":   med_c,
        "detections_below_threshold": below,
    }


# ---------------------------------------------------------
# Analytics Validation — consistency checks after generation.
# Logs warnings for logically impossible states (e.g. compliance > 100%).
# Called from the debug report; also safe to call in production if desired.
# ---------------------------------------------------------

def _validate_analytics(analytics: Dict[str, Any]) -> None:
    workers       = analytics.get("workers", 0)
    ppe_comp      = analytics.get("ppe_compliance", 100.0)
    ppe_viol      = analytics.get("ppe_violations", 0)
    helmet_count  = analytics.get("helmet_count", 0)

    if ppe_comp > 100.0:
        print(f"[ANALYTICS WARNING] ppe_compliance > 100%: {ppe_comp:.1f}")
    if ppe_comp < 0.0:
        print(f"[ANALYTICS WARNING] ppe_compliance < 0%: {ppe_comp:.1f}")
    if workers > 0 and helmet_count > workers * 2:
        print(f"[ANALYTICS WARNING] helmet_count ({helmet_count}) seems high for "
              f"{workers} worker(s).")
    stats = analytics.get("statistics", {})
    if stats:
        total = stats.get("total_detections", 0)
        rb    = stats.get("risk_breakdown", {})
        rb_sum = sum(rb.values())
        if rb_sum != total:
            print(f"[ANALYTICS WARNING] risk_breakdown sum ({rb_sum}) ≠ "
                  f"total_detections ({total}).")


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

    # Determine the primary cause for the risk level
    if overall_risk.upper() in ("HIGH", "CRITICAL"):
        if danger_zone_count > 0:
            cause = " due to danger-zone intrusion"
        elif ppe_violations > 0:
            cause = " due to PPE non-compliance"
        elif crack_count > 0:
            cause = " due to structural defects"
        else:
            cause = ""
    else:
        cause = ""

    parts.append(f"Overall site risk: {overall_risk.title()}{cause}.")

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
        polygon = zone.get("polygon")
        if not polygon or len(polygon) < 3:
            continue

        pts = np.asarray(polygon, dtype=np.int32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            continue

        intrusion = any(a["zone"] == zone["name"] for a in intrusion_alerts)
        color     = (0, 0, 255) if intrusion else (255, 120, 0)
        alpha     = (0.12 + 0.10 * pulse) if intrusion else (0.04 + 0.04 * pulse)

        overlay = frame.copy()
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

    cv2.rectangle(overlay, (0, 0), (w, 76), (10, 15, 25), -1)
    frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Row 1 — title
    cv2.putText(
        frame, "InfraGuard Enterprise AI Surveillance",
        (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2
    )
    # Row 1 — per-category counts (right side)
    cv2.putText(
        frame, f"Workers: {analytics.get('workers', 0)}",
        (w - 260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2
    )
    cv2.putText(
        frame, f"Equip: {analytics.get('equipment_count', 0)}",
        (w - 420, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 255), 2
    )
    cv2.putText(
        frame, f"Cracks: {analytics.get('crack_count', 0)}",
        (w - 570, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2
    )
    # Row 2 — risk and PPE compliance
    cv2.putText(
        frame, f"Risk: {analytics.get('overall_risk', 'N/A')}",
        (w - 260, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2
    )
    cv2.putText(
        frame, f"PPE: {analytics.get('ppe_compliance', 100.0):.0f}%",
        (w - 420, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2
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
            "danger_zone_count": 0, "risk_score": 0, "overall_risk": "SAFE",
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
            "summary":      "No personnel detected in frame. Overall site risk: Safe.",
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
                "overall_risk":      "SAFE",
                "risk_breakdown":    {"SAFE": 0, "CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0},
                "average_confidence":  0.0,
                "max_confidence":      0.0,
                "min_confidence":      0.0,
                "median_confidence":   0.0,
                "detections_below_threshold": 0,
            },
            "class_distribution": {},
        },
        "telemetry":   {
            "processing_ms":   0.0,
            "detection_count": 0,
            "stage_times":     {},
        },
        "ai_metadata":       build_ai_metadata(),
        "integrity_report":  {
            "raw_detections":    0,
            "validated":         0,
            "rejected":          0,
            "normalized":        0,
            "tracked":           0,
            "risk_classified":   0,
            "analytics_counted": 0,
            "returned":          0,
            "empty_reason":      "Frame was None.",
        },
        "rejected_detections": [],
        "timestamp":   datetime.utcnow().isoformat(),
        "camera_id":   camera_id,
    }