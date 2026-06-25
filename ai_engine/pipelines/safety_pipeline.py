import atexit
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Full, Empty
from datetime import datetime, timezone
from threading import Lock
import uuid
import time

import numpy as np

from ai_engine.core.predictor import (
    InfraGuardPredictor
)

from backend.services.safety.tracker import (
    EnterpriseTracker
)

from backend.services.safety.alert_service import (
    AlertManager
)

from backend.services.safety.event_service import (
    save_event
)

from backend.api.activity_routes import (
    add_alert
)

# =========================================
# CORE
# =========================================

predictor = InfraGuardPredictor()

# [R5 #8] Predictor warmup — eliminates first-inference latency spike
try:
    _dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    predictor.predict_frame(_dummy_frame)
    del _dummy_frame
except Exception:
    pass

tracker = EnterpriseTracker()

alert_manager = AlertManager(
    cooldown=10
)

# [R5 #4] Async executor — save_event / add_alert run off the inference thread
# [FIX] Bounded queue (max 500 pending tasks) prevents unbounded memory growth
# under sustained high-alert load (e.g. 30 FPS × 10 alerts/sec with slow DB).
# Tasks submitted when the queue is full are silently dropped (non-critical logging).
_MAX_PENDING_TASKS = 500
_task_queue: Queue = Queue(maxsize=_MAX_PENDING_TASKS)
_executor = ThreadPoolExecutor(max_workers=2)


def _submit_task(fn, *args, **kwargs):
    """Submit a fire-and-forget task, dropping it if the queue is full."""
    try:
        _task_queue.put_nowait(True)   # reserve a slot
    except Full:
        return                          # drop — logging backlog, not safety data
    def _run():
        try:
            fn(*args, **kwargs)
        finally:
            try:
                _task_queue.get_nowait()   # release the slot
            except Empty:
                pass
    _executor.submit(_run)


# [FIX #1] Prevent thread leaks on backend reload / crash
atexit.register(_executor.shutdown, wait=False)

# [R2 #8] Rolling analytics history — last 100 frames
analytics_history = deque(maxlen=100)

# [FIX #8] Global alert history — configurable size, default 5000
MAX_ALERT_HISTORY = 5000
alert_history: deque = deque(maxlen=MAX_ALERT_HISTORY)

# [R3 #9] Detection persistence — last known detections per camera
last_detections: dict = {}     # { camera_id: [det, ...] }

# [FIX] Track last-seen time per camera so stale entries can be expired.
last_seen_cameras: dict = {}   # { camera_id: float (time.time()) }

# Cameras not seen for this many seconds are removed from last_detections.
CAMERA_EXPIRY_SECONDS = 300

# [FIX] Track last-reported area per crack UUID to deduplicate growth alerts.
# Alerts fire at 20 % increments (20 %, 40 %, 60 %…) rather than every frame.
crack_growth_reported: dict = {}   # { crack_uuid: last_reported_area }

# [R5 #3] Crack registry — stable UUID tracking across frames
# { camera_id: { crack_uuid: {"bbox": [x1,y1,x2,y2], "area": int} } }
crack_registry: dict = {}

# [R5 #6] Heatmap data — last 500 detection centre points for frontend overlays
heatmap_data: deque = deque(maxlen=500)

# [FIX] Rolling FPS — smoothed over last 30 frames to reduce dashboard noise
fps_history: deque = deque(maxlen=30)

# [FIX] Throttle cleanup to once per minute instead of every frame.
# At 30 FPS × 10 cameras that saves ~17 900 unnecessary scans/min.
_CLEANUP_INTERVAL = 60   # seconds
_last_cleanup: float = 0.0
# camera streams: analytics_history, alert_history, heatmap_data,
# crack_registry, last_detections, last_seen_cameras, fps_history.
state_lock = Lock()


# =========================================
# CONFIG
# =========================================

# Class normalization — maps raw model output to canonical names
CLASS_MAP = {
    "Helmet":                                   "helmet",
    "helmet":                                   "helmet",
    "vest":                                     "vest",
    "vests":                                    "vest",
    "no helmet":                                "no_helmet",
    "no vest":                                  "no_vest",
    "crack detection - v2 2023-11-03 11-16am":  "crack",
    # Equipment — normalize casing variations
    "Bulldozer":        "bulldozer",
    "bulldozer":        "bulldozer",
    "Dump Truck":       "dump truck",
    "dump truck":       "dump truck",
    "Excavator":        "excavator",
    "excavator":        "excavator",
    "Grader":           "grader",
    "grader":           "grader",
    "Loader":           "loader",
    "loader":           "loader",
    "Roller":           "roller",
    "roller":           "roller",
    "Mobile Crane":     "mobile crane",
    "mobile crane":     "mobile crane",
}

# Per-class confidence thresholds
CLASS_THRESHOLDS = {
    "person":       0.40,
    "helmet":       0.35,
    "vest":         0.35,
    "crack":        0.25,
    "bulldozer":    0.40,
    "dump truck":   0.40,
    "excavator":    0.40,
    "grader":       0.40,
    "loader":       0.40,
    "roller":       0.40,
    "mobile crane": 0.40,
}

CONFIDENCE_THRESHOLD = 0.40  # default fallback

HIGH_RISK_CLASSES = {
    "no_helmet",
    "no_vest",
    "danger_intrusion",
    "crack",
}

# [R2 #3] Heavy equipment added to medium risk
MEDIUM_RISK_CLASSES = {
    "person",
    "bulldozer",
    "dump truck",
    "excavator",
    "grader",
    "loader",
    "roller",
    "mobile crane",
}

HEAVY_EQUIPMENT = {
    "excavator", "bulldozer", "loader", "roller", "mobile crane",
    "dump truck", "grader",  # [FIX] previously missing from equipment_count
}

# [R5 #7] Critical added as highest severity tier
SEVERITY_SCORE = {
    "critical": 150,
    "high":     100,
    "medium":    60,
    "low":       20,
}

# [R5 #4] Equipment restricted zones — configure per camera_id.
# Format: { camera_id: [[x1, y1, x2, y2], ...] }
# Example: DANGER_ZONES = {0: [[100, 100, 500, 500]]}
# Leave as {} to disable globally; override at startup or via site config.
DANGER_ZONES: dict = {}

# IoU threshold for PPE-to-worker association.
# Low because a helmet bbox is much smaller than a worker bbox.
PPE_IOU_THRESHOLD = 0.05

# IoU threshold for crack re-identification across frames.
CRACK_IOU_THRESHOLD = 0.30

# [FIX #2] Cracks not seen for this many seconds are removed from the registry.
CRACK_EXPIRY_SECONDS = 300


# =========================================
# GEOMETRY HELPERS
# =========================================

def _iou(a: list, b: list) -> float:
    """Intersection-over-Union for two [x1,y1,x2,y2] bboxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def _center_inside(inner: list, outer: list) -> bool:
    """True if the centre point of `inner` bbox falls inside `outer` bbox."""
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _boxes_intersect(a: list, b: list) -> bool:
    """True if two [x1,y1,x2,y2] bboxes share any area."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


# =========================================
# CRACK TRACKER
# =========================================

def _resolve_crack_id(camera_id, bbox: list, area: int) -> str:
    """
    [R5 #3] Stable crack UUID across frames via IoU re-identification.
    Matches incoming crack bbox against registry for this camera.
    Returns an existing UUID if IoU > threshold, otherwise mints a new one.
    Thread-safe: caller must hold state_lock.
    """
    cam_cracks = crack_registry.setdefault(camera_id, {})

    best_uuid  = None
    best_iou   = 0.0

    for c_uuid, entry in cam_cracks.items():
        score = _iou(bbox, entry["bbox"])
        if score > best_iou:
            best_iou   = score
            best_uuid  = c_uuid

    if best_uuid and best_iou >= CRACK_IOU_THRESHOLD:
        # update stored bbox/area so subsequent frames track correctly
        cam_cracks[best_uuid] = {"bbox": bbox, "area": area, "last_seen": time.time()}
        return best_uuid

    new_uuid = str(uuid.uuid4())
    cam_cracks[new_uuid] = {"bbox": bbox, "area": area, "last_seen": time.time()}
    return new_uuid


def cleanup_crack_registry():
    """
    [FIX #2] Remove stale crack entries that haven't been seen recently.
    Prevents unbounded memory growth in long-running deployments.
    Call once per frame from run_safety_pipeline.
    """
    now = time.time()
    for cam_id in list(crack_registry.keys()):
        cam_cracks = crack_registry[cam_id]
        stale = [
            c_uuid for c_uuid, entry in cam_cracks.items()
            if now - entry.get("last_seen", now) > CRACK_EXPIRY_SECONDS
        ]
        for c_uuid in stale:
            del cam_cracks[c_uuid]
            crack_growth_reported.pop(c_uuid, None)  # prevent memory leak

        # [FIX #3] Remove camera entry entirely once all its cracks have expired
        # Prevents accumulation of empty dicts over months of operation
        if not cam_cracks:
            del crack_registry[cam_id]


def cleanup_stale_cameras():
    """
    [FIX] Remove cameras from last_detections that haven't been seen recently.
    Prevents unbounded memory growth when many camera_ids cycle through the system.
    Call once per frame from run_safety_pipeline.
    """
    now = time.time()
    stale = [
        cam_id for cam_id, last_seen in last_seen_cameras.items()
        if now - last_seen > CAMERA_EXPIRY_SECONDS
    ]
    for cam_id in stale:
        last_detections.pop(cam_id, None)
        del last_seen_cameras[cam_id]
        crack_registry.pop(cam_id, None)  # remove stale camera's crack registry


# =========================================
# PIPELINE
# =========================================

def run_safety_pipeline(frame, camera_id=0):   # [R2 #5] camera_id param

    start = time.time()

    # Single timestamp for the entire frame — avoids 20-50 datetime.now() calls per frame
    frame_ts = datetime.now(timezone.utc).isoformat()

    # [FIX] Normalise camera_id to str so int 0 and str "0" map to the same key
    # across last_detections, crack_registry, alert keys, and API responses.
    camera_id = str(camera_id)

    # [FIX] Run cleanup at most once per minute (not every frame).
    global _last_cleanup
    with state_lock:
        now = time.time()
        if now - _last_cleanup >= _CLEANUP_INTERVAL:
            cleanup_crack_registry()
            # [FIX] Expire cameras not seen recently to prevent last_detections growth
            cleanup_stale_cameras()
            _last_cleanup = now
        # [FIX] Record this camera as active
        last_seen_cameras[camera_id] = time.time()
    try:
        # [R4 #6] Per-stage timing — isolates model inference cost
        predict_start = time.time()
        raw = predictor.predict_frame(frame)
        predict_ms = round((time.time() - predict_start) * 1000, 2)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "detections":   [],
            "alerts":       [],
            "critical":     0,
            "high":         0,
            "medium":       0,
            "low":          0,
            "risk":         "LOW",
            "analytics": {
                "total_objects":    0,
                "processing_ms":    0,
                "predict_ms":       0,
                "fps":              0,
                "tracker_active":   False,
                "class_counts":     {},
                "ppe_compliance": {
                    "overall":      100.0,
                    "helmet":       100.0,
                    "vest":         100.0,
                    "worker_count": 0,
                }
            },
            "error": str(e)
        }

    # =====================================
    # CONFIDENCE FILTERING
    # =====================================

    filtered = []

    for det in raw:

        conf      = float(det.get("confidence", 0))
        raw_label = det.get("class_name", "")
        canonical = CLASS_MAP.get(raw_label, raw_label.lower())
        threshold = CLASS_THRESHOLDS.get(canonical, CONFIDENCE_THRESHOLD)

        if conf < threshold:
            continue

        filtered.append(det)

    # =====================================
    # SPLIT DETECTIONS
    # =====================================

    def _canonical(d):
        return CLASS_MAP.get(d.get("class_name", ""), d.get("class_name", "").lower())

    persons     = [d for d in filtered if _canonical(d) == "person"]
    non_persons = [d for d in filtered if _canonical(d) != "person"]

    # Separate PPE items for association pass
    helmets_raw = [d for d in non_persons if _canonical(d) == "helmet"]
    vests_raw   = [d for d in non_persons if _canonical(d) == "vest"]

    persons = tracker.update(persons)

    detections = []
    alerts     = []

    # =====================================
    # PERSON DETECTIONS + PPE ASSOCIATION
    # =====================================

    for p in persons:

        bbox = p.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        worker_bbox     = [int(x1), int(y1), int(x2), int(y2)]

        confidence = float(p.get("confidence", 0))
        raw_label  = p.get("class_name", "person")
        label      = CLASS_MAP.get(raw_label, raw_label.lower())
        worker_id  = str(p.get("id", uuid.uuid4()))

        # [R5 #1] PPE association — check whether a helmet / vest bbox centre
        # falls inside this worker's bounding box (IoU alone understates overlap
        # when the PPE item is much smaller than the worker).
        has_helmet = any(
            _center_inside(h.get("bbox", [0, 0, 0, 0]), worker_bbox)
            or _iou(h.get("bbox", [0, 0, 0, 0]), worker_bbox) >= PPE_IOU_THRESHOLD
            for h in helmets_raw
            if len(h.get("bbox", [])) == 4
        )
        has_vest = any(
            _center_inside(v.get("bbox", [0, 0, 0, 0]), worker_bbox)
            or _iou(v.get("bbox", [0, 0, 0, 0]), worker_bbox) >= PPE_IOU_THRESHOLD
            for v in vests_raw
            if len(v.get("bbox", [])) == 4
        )
        ppe_compliant = has_helmet and has_vest

        # Workers missing any PPE are high risk
        risk = "high" if not ppe_compliant else calculate_risk(label, confidence)

        det = {

            "id":           worker_id,

            "class_name":   label,

            "label":        f"Worker {worker_id[:4]}",

            "bbox":         worker_bbox,

            "x":            int(x1),
            "y":            int(y1),
            "w":            int(x2 - x1),
            "h":            int(y2 - y1),

            # [R2 #4] Bounding box area
            "area":         int((x2 - x1) * (y2 - y1)),

            "confidence":   confidence,
            "risk":         risk,
            "score":        SEVERITY_SCORE[risk],
            "timestamp":    frame_ts,
            "camera_id":    camera_id,              # [R2 #5]
            "type":         "worker",
            "tracking":     True,

            # [R5 #1] Per-worker PPE association results
            "has_helmet":   has_helmet,
            "has_vest":     has_vest,
            "ppe_compliant": ppe_compliant,
        }

        detections.append(det)

        # =================================
        # SMART ALERTS
        # =================================

        # [FIX] Alert key scoped to violation type so helmet/vest alerts
        # are tracked independently and don't suppress each other.
        violation_type = []
        if not has_helmet:
            violation_type.append("helmet")
        if not has_vest:
            violation_type.append("vest")
        alert_key = (
            f"{worker_id}_{'_'.join(violation_type)}"
            if violation_type
            else worker_id
        )

        should_alert = alert_manager.should_alert(alert_key, risk)

        if should_alert:

            message = (
                f"Worker {worker_id[:4]} "
                f"{risk.upper()} risk detected"
            )

            alert = {

                "worker_id":    worker_id,
                "risk":         risk,
                "score":        SEVERITY_SCORE[risk],
                "message":      message,
                "camera_id":    camera_id,          # [R2 #5]
                "timestamp":    frame_ts
            }

            alerts.append(alert)

            # [R5 #4] Async — event logging runs off the inference thread
            _submit_task(save_event, {
                "event_type":           "PPE_ALERT",
                "risk_level":           risk.upper(),
                "camera_id":            camera_id,
                "workers":              1,
                "violating_workers":    1,
                "description":          message
            })

            _submit_task(add_alert,
                event_type="PPE Violation",
                risk=risk,
                cam_id=camera_id,
                description=message
            )

    # =====================================
    # NON-PERSON DETECTIONS
    # (helmets, vests, cracks, equipment, etc.)
    # =====================================

    for n in non_persons:

        bbox = n.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox

        confidence = float(n.get("confidence", 0))
        raw_label  = n.get("class_name", "object")
        label      = CLASS_MAP.get(raw_label, raw_label.lower())
        det_area   = int((x2 - x1) * (y2 - y1))
        det_bbox   = [int(x1), int(y1), int(x2), int(y2)]

        risk = calculate_risk(label, confidence, area=det_area)

        # [R5 #3] Crack re-identification — stable UUID across frames
        if label == "crack":
            with state_lock:
                stable_id = _resolve_crack_id(camera_id, det_bbox, det_area)
        else:
            stable_id = f"{label}_{int(x1)}_{int(y1)}"

        det = {

            "id":               stable_id,

            # [R4 #4] Stable audit UUID — kept separate from positional id
            "detection_uuid":   str(uuid.uuid4()),

            "class_name":   label,
            "label":        label,

            "bbox":         det_bbox,

            "x":            int(x1),
            "y":            int(y1),
            "w":            int(x2 - x1),
            "h":            int(y2 - y1),

            # [R2 #4] Bounding box area
            "area":         det_area,

            "confidence":   confidence,
            "risk":         risk,
            "score":        SEVERITY_SCORE[risk],
            "timestamp":    frame_ts,
            "camera_id":    camera_id,              # [R2 #5]
            "type":         "object",
            "tracking":     False,

            # [R3 #13] Crack severity — populated for crack class only
            "crack_severity": (
                classify_crack_severity(confidence, det_area)
                if label == "crack" else None
            ),
        }

        detections.append(det)

        # [R2 #1] Alert spam protection — use AlertManager for non-person too
        if risk in ("high", "critical"):

            # [R4 #3] Stable alert key — grid-rounded to suppress 5-pixel drift
            alert_key = f"{label}_{int(x1 / 50)}_{int(y1 / 50)}"

            should_alert = alert_manager.should_alert(alert_key, risk)

            if should_alert:

                alert = {

                    "type":         label,
                    "risk":         risk,
                    "score":        SEVERITY_SCORE[risk],
                    "message":      f"{label.replace('_', ' ').title()} detected",
                    "camera_id":    camera_id,      # [R2 #5]
                    "timestamp":    frame_ts
                }

                alerts.append(alert)

                # [R5 #4] Async logging
                _submit_task(save_event, {
                    "event_type":           "STRUCTURAL_ALERT",
                    "risk_level":           risk.upper(),
                    "camera_id":            camera_id,
                    "workers":              0,
                    "violating_workers":    0,
                    "description":          alert["message"]
                })

                _submit_task(add_alert,
                    event_type="Structural Defect",
                    risk=risk,
                    cam_id=camera_id,
                    description=alert["message"]
                )

        # [R4 #2] Equipment alerts — medium risk machinery alert channel
        if label in HEAVY_EQUIPMENT:

            equip_key = f"equip_{label}_{int(x1 / 50)}_{int(y1 / 50)}"

            if alert_manager.should_alert(equip_key, "medium"):

                alerts.append({
                    "type":         label,
                    "risk":         "medium",
                    "score":        SEVERITY_SCORE["medium"],
                    "message":      f"Heavy equipment detected: {label.title()}",
                    "camera_id":    camera_id,
                    "timestamp":    frame_ts
                })

        # [R5 #4] Equipment zone violation check
        # [FIX] Track which equipment bboxes are in a zone so risk can be
        # escalated to "high" on the detection itself (not just via an alert).
        in_zone = False
        for zone in DANGER_ZONES.get(camera_id, []):
            if len(zone) == 4 and label in HEAVY_EQUIPMENT and _boxes_intersect(det_bbox, zone):
                in_zone = True
                zone_key = f"zone_{label}_{int(x1 / 50)}_{int(y1 / 50)}"

                if alert_manager.should_alert(zone_key, "high"):

                    alerts.append({
                        "type":         "zone_violation",
                        "risk":         "high",
                        "score":        SEVERITY_SCORE["high"],
                        "message":      (
                            f"{label.title()} entered restricted zone "
                            f"[{zone[0]},{zone[1]},{zone[2]},{zone[3]}]"
                        ),
                        "equipment":    label,
                        "zone":         zone,
                        "camera_id":    camera_id,
                        "timestamp":    frame_ts
                    })

        # [FIX] Escalate detection risk to "high" when equipment is in a danger zone
        if in_zone and det["risk"] == "medium":
            det["risk"]  = "high"
            det["score"] = SEVERITY_SCORE["high"]

    # =====================================
    # ANALYTICS
    # =====================================

    # [R5 #7] Four-tier risk counting
    critical_count = len([d for d in detections if d["risk"] == "critical"])
    high           = len([d for d in detections if d["risk"] == "high"])
    medium         = len([d for d in detections if d["risk"] == "medium"])
    low            = len([d for d in detections if d["risk"] == "low"])

    # [R2 #7] Weighted risk score — four tiers now included
    risk_score = (
        (critical_count * 150) +
        (high           * 100) +
        (medium         *  50) +
        (low            *  10)
    )

    if risk_score >= 500:
        overall = "CRITICAL"
    elif risk_score >= 300:
        overall = "HIGH"
    elif risk_score >= 100:
        overall = "MEDIUM"
    else:
        overall = "LOW"

    class_counts = {}
    for d in detections:
        cls = d["class_name"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    # [R2 #6] FPS as wall-clock rate — [FIX] rolling average over last 30 frames
    elapsed        = time.time() - start
    instant_fps    = round(1 / elapsed, 2) if elapsed > 0 else 0.0
    with state_lock:
        fps_history.append(instant_fps)
        fps = round(sum(fps_history) / len(fps_history), 2)
    inference_time = round(elapsed * 1000, 2)

    # [R5 #1] PPE compliance derived from per-worker association flags
    worker_dets          = [d for d in detections if d.get("type") == "worker"]
    worker_count         = len(worker_dets)
    workers_with_helmet  = sum(1 for d in worker_dets if d.get("has_helmet"))
    workers_with_vest    = sum(1 for d in worker_dets if d.get("has_vest"))

    if worker_count > 0:
        # [R4 #1] Cap at 100 — prevents >100% when helmets outnumber workers
        helmet_compliance = min(100.0, round(workers_with_helmet / worker_count * 100, 1))
        vest_compliance   = min(100.0, round(workers_with_vest   / worker_count * 100, 1))
        ppe_compliance    = min(100.0, round(
            (workers_with_helmet + workers_with_vest) / (worker_count * 2) * 100,
            1
        ))
    else:
        helmet_compliance = 100.0
        vest_compliance   = 100.0
        ppe_compliance    = 100.0

    # [FIX #5] Total equipment count from class_counts
    equipment_count = sum(
        class_counts.get(eq, 0) for eq in HEAVY_EQUIPMENT
    )

    # [FIX #6] PPE compliance breakdown for dashboard widgets
    fully_compliant = sum(
        1 for d in worker_dets if d.get("has_helmet") and d.get("has_vest")
    )
    helmet_only = sum(
        1 for d in worker_dets if d.get("has_helmet") and not d.get("has_vest")
    )
    vest_only = sum(
        1 for d in worker_dets if d.get("has_vest") and not d.get("has_helmet")
    )
    non_compliant = sum(
        1 for d in worker_dets if not d.get("has_helmet") and not d.get("has_vest")
    )

    # [R3 #9] Persist last known detections for this camera (needed for crack growth below)
    with state_lock:
        prev_detections = last_detections.get(camera_id, [])
        last_detections[camera_id] = detections

    # [R4 #8] Crack growth monitoring — alert at each 20 % growth milestone.
    # [FIX] Deduplicated: uses crack_growth_reported so a static enlarged crack
    # does not re-alert every frame.  Alerts at 20 %, 40 %, 60 %… milestones.
    prev_cracks = {
        d["id"]: d for d in prev_detections
        if d.get("class_name") == "crack"
    }
    for d in detections:
        if d.get("class_name") != "crack":
            continue
        prev = prev_cracks.get(d["id"])
        if not prev or prev.get("area", 0) <= 0:
            continue
        crack_id  = d["id"]
        base_area = crack_growth_reported.get(crack_id, prev["area"])
        if base_area <= 0:
            continue
        growth = (d["area"] - base_area) / base_area
        if growth >= 0.20:
            alerts.append({
                "type":         "crack_growth",
                "risk":         "high",
                "score":        SEVERITY_SCORE["high"],
                "message":      (
                    f"Crack growth detected: "
                    f"{round(growth * 100, 1)}% area increase"
                ),
                "camera_id":    camera_id,
                "timestamp":    frame_ts
            })
            # Advance the baseline so next alert fires at the NEXT 20 % step
            crack_growth_reported[crack_id] = d["area"]

    # [R3 #15] Safety score 0–100
    # Calculated BEFORE frame_analytics so it can be stored directly in the dict.
    # Starts at 100, penalised by risk score and PPE non-compliance.
    ppe_penalty   = round((100 - ppe_compliance) * 0.4, 2)   # up to 40 pts
    risk_penalty  = min(60, round(risk_score / 10, 2))        # up to 60 pts
    safety_score  = max(0, round(100 - ppe_penalty - risk_penalty, 1))

    # [FIX #4] Count zone violations from alerts generated this frame
    # (includes crack-growth alerts now that they're appended above)
    zone_violations = sum(1 for a in alerts if a.get("type") == "zone_violation")

    frame_analytics = {

        "total_objects":    len(detections),
        "processing_ms":    inference_time,
        "predict_ms":       predict_ms,         # [R4 #6] inference-only latency
        "fps":              fps,
        "tracker_active":   True,
        "camera_health":    "ONLINE",           # [FIX #7] camera health status
        "risk_score":       risk_score,         # [R2 #7]
        "safety_score":     safety_score,       # [FIX #3] stored here, not after
        "class_counts":     class_counts,
        "ppe_compliance": {
            "overall":              ppe_compliance,
            "helmet":               helmet_compliance,
            "vest":                 vest_compliance,
            "worker_count":         worker_count,
            "workers_with_helmet":  workers_with_helmet,
            "workers_with_vest":    workers_with_vest,
            # [FIX #6] Compliance breakdown
            "fully_compliant":      fully_compliant,
            "helmet_only":          helmet_only,
            "vest_only":            vest_only,
            "non_compliant":        non_compliant,
        },
        "equipment_count":  equipment_count,    # [FIX #5]
        "zone_violations":  zone_violations,    # [FIX #4]
        "timestamp":        frame_ts,
        "camera_id":        camera_id,
    }

    # [R4 #5] Frame-level counters — useful for history trend charts
    frame_analytics["alert_count"]     = len(alerts)
    frame_analytics["detection_count"] = len(detections)

    # [R2 #8] Append to rolling history
    # [FIX] Lock shared deques for multi-camera thread safety
    with state_lock:
        analytics_history.append(frame_analytics)

        # [FIX #8] Persist ALL alerts (including crack-growth) to global history
        for alert in alerts:
            alert_history.append(alert)

        # [R5 #6] Heatmap — store centre points of every detection
        for d in detections:
            heatmap_data.append({
                "cx":        d["x"] + d["w"] // 2,
                "cy":        d["y"] + d["h"] // 2,
                "risk":      d["risk"],
                "class":     d["class_name"],
                "camera_id": camera_id,
                "timestamp": frame_ts,
            })

    return {

        "detections":   detections,
        "alerts":       alerts,

        "critical":     critical_count,         # [R5 #7]
        "high":         high,
        "medium":       medium,
        "low":          low,
        "risk":         overall,

        "safety_score": safety_score,           # [R3 #15]

        "analytics":    frame_analytics
    }


# =========================================
# RISK ENGINE
# =========================================

def calculate_risk(label: str, confidence: float, area: int = 0) -> str:

    label = label.lower()

    if label in HIGH_RISK_CLASSES:

        # [R4 bug] Crack risk uses actual severity (area + confidence)
        if label == "crack":
            severity = classify_crack_severity(confidence, area)
            # [R5 #7] Critical crack → critical tier
            if severity == "critical":
                return "critical"
            if severity == "severe":
                return "high"
            return "medium"

        return "high"

    if label in MEDIUM_RISK_CLASSES:
        return "medium"

    return "low"


# [R3 #13] Crack severity classifier
# Returns one of: "minor" | "moderate" | "severe" | "critical"
def classify_crack_severity(confidence: float, area: int) -> str:
    if confidence >= 0.85 and area >= 10_000:
        return "critical"
    if confidence >= 0.70 or area >= 6_000:
        return "severe"
    if confidence >= 0.50 or area >= 2_000:
        return "moderate"
    return "minor"


# =========================================
# ANALYTICS API
# =========================================

def get_analytics_history() -> list:
    """
    [R4 #9] Returns last 100 frames of analytics for frontend charts.
    Each entry: total_objects, processing_ms, predict_ms, fps,
    risk_score, class_counts, ppe_compliance, alert_count,
    detection_count, timestamp, camera_id.
    """
    with state_lock:
        return list(analytics_history)


def get_summary(camera_id=None) -> dict:
    """
    [R5 #5] Aggregated analytics across rolling history.

    Args:
        camera_id: Filter to a specific camera. None = all cameras.

    Returns:
        {
            avg_fps, avg_safety_score, avg_processing_ms,
            total_alerts, total_detections,
            total_workers, total_cracks,
            peak_risk_score, frames_analysed
        }
    """
    with state_lock:
        frames = [
            f for f in analytics_history
            if camera_id is None or f.get("camera_id") == camera_id
        ]

    if not frames:
        return {
            "avg_fps":              0.0,
            "avg_safety_score":     0.0,
            "avg_processing_ms":    0.0,
            "total_alerts":         0,
            "total_detections":     0,
            "total_workers":        0,
            "total_cracks":         0,
            "peak_risk_score":      0,
            "frames_analysed":      0,
        }

    n = len(frames)

    avg_fps            = round(sum(f.get("fps",            0) for f in frames) / n, 2)
    avg_processing_ms  = round(sum(f.get("processing_ms",  0) for f in frames) / n, 2)
    peak_risk_score    = max(f.get("risk_score", 0) for f in frames)
    total_alerts       = sum(f.get("alert_count",     0) for f in frames)
    total_detections   = sum(f.get("detection_count", 0) for f in frames)
    total_workers      = sum(f.get("ppe_compliance", {}).get("worker_count", 0) for f in frames)  # [FIX] robust to class rename
    total_cracks       = sum(f.get("class_counts", {}).get("crack",  0) for f in frames)

    # [FIX #3] avg_safety_score now uses the true safety_score stored per frame
    # (previously used ppe_compliance.overall which ignores risk penalties)
    safety_vals = [f.get("safety_score", 100.0) for f in frames]
    avg_safety_score = round(sum(safety_vals) / n, 1)

    return {
        "avg_fps":              avg_fps,
        "avg_safety_score":     avg_safety_score,
        "avg_processing_ms":    avg_processing_ms,
        "total_alerts":         total_alerts,
        "total_detections":     total_detections,
        "total_workers":        total_workers,
        "total_cracks":         total_cracks,
        "peak_risk_score":      peak_risk_score,
        "frames_analysed":      n,
    }


def get_heatmap_data(camera_id=None) -> list:
    """
    [R5 #6] Returns detection centre points for frontend heatmap overlays.

    Args:
        camera_id: Filter to a specific camera. None = all cameras.

    Returns:
        List of { cx, cy, risk, class, camera_id, timestamp }
    """
    with state_lock:
        if camera_id is None:
            return list(heatmap_data)
        return [p for p in heatmap_data if p.get("camera_id") == camera_id]


def get_alert_history(camera_id=None, limit: int = 100) -> list:
    """
    [FIX #8] Returns recent alerts from global alert_history deque.

    Args:
        camera_id: Filter to a specific camera. None = all cameras.
        limit:     Maximum number of alerts to return (most recent first).

    Returns:
        List of alert dicts with type, risk, score, message, camera_id, timestamp.
    """
    with state_lock:
        history = list(alert_history)
    if camera_id is not None:
        history = [a for a in history if a.get("camera_id") == camera_id]
    return history[-limit:][::-1]  # most recent first


def get_processing_breakdown(camera_id=None) -> dict:
    """
    [FIX #9] Returns average per-stage timing breakdown across rolling history.

    Shows how inference time is distributed, useful for optimization:
        predict_ms   — YOLO model inference
        processing_ms — full pipeline wall time
        overhead_ms  — tracking + alerts + analytics (processing - predict)

    Args:
        camera_id: Filter to a specific camera. None = all cameras.

    Returns:
        { predict_ms, processing_ms, overhead_ms, fps, frames_analysed }
    """
    with state_lock:
        frames = [
            f for f in analytics_history
            if camera_id is None or f.get("camera_id") == camera_id
        ]
    if not frames:
        return {
            "predict_ms":     0.0,
            "processing_ms":  0.0,
            "overhead_ms":    0.0,
            "fps":            0.0,
            "frames_analysed": 0,
        }
    n = len(frames)
    avg_predict    = round(sum(f.get("predict_ms",    0) for f in frames) / n, 2)
    avg_processing = round(sum(f.get("processing_ms", 0) for f in frames) / n, 2)
    avg_fps        = round(sum(f.get("fps",           0) for f in frames) / n, 2)
    return {
        "predict_ms":      avg_predict,
        "processing_ms":   avg_processing,
        "overhead_ms":     round(avg_processing - avg_predict, 2),
        "fps":             avg_fps,
        "frames_analysed": n,
    }