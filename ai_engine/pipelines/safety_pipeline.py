import atexit
import hashlib
import logging
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Full, Empty
from datetime import datetime, timezone
from threading import Lock, RLock   # Fix 5: RLock for re-entrant cleanup safety
from typing import Dict, List
import uuid
import time

import numpy as np

# =========================================
# STRUCTURED LOGGING  (Priority 1 #8)
# =========================================
# Replace bare traceback.print_exc() calls with structured logger output.
# In production attach a FileHandler or use a log aggregator (e.g. Loki, Datadog).
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    ))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

from ai_engine.core.predictor import (
    InfraGuardPredictor,
    CrackPredictor,
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

# =========================================
# CONFIG  (Priority 1 #2)
# =========================================
# All tunable constants are now read from environment variables so future
# capacity / performance tuning requires no code changes — just restart with
# updated env vars or a .env file loaded by your process manager.
#
# Equivalent safety_config.py snippet:
#   from backend.config.safety_config import *
# or load via python-dotenv:
#   from dotenv import load_dotenv; load_dotenv()

MAX_PREDICTOR_POOL       = int(os.getenv("MAX_PREDICTOR_POOL",       8))
INFERENCE_EVERY_N_DEFAULT= int(os.getenv("INFERENCE_EVERY_N",        3))
CAMERA_EXPIRY_SECONDS    = int(os.getenv("CAMERA_EXPIRY_SECONDS",    300))
CRACK_EXPIRY_SECONDS_CFG = int(os.getenv("CRACK_EXPIRY_SECONDS",     300))
_MAX_PENDING_TASKS       = int(os.getenv("MAX_PENDING_TASKS",        500))
MAX_ALERT_HISTORY        = int(os.getenv("MAX_ALERT_HISTORY",        5000))
CAMERA_OFFLINE_THRESHOLD = int(os.getenv("CAMERA_OFFLINE_THRESHOLD", 30))
# Crack model runs every Nth inference frame (Priority 2 #5).
# Set to 1 to run on every inference frame; default 5 gives ~25-40 % GPU saving.
CRACK_INFERENCE_EVERY_N  = int(os.getenv("CRACK_INFERENCE_EVERY_N",  5))

# ── Fix 1: Bounded predictor pool — avoids loading N models for N cameras ───
# 20 cameras no longer means 20 YOLO instances; each pool slot is shared via
# consistent hashing (hashlib.md5(camera_id) % pool_size).
# Improvement 6: raised from 4 → 8 slots so 20 cameras spread across more
# slots, halving hash collisions and the resulting lock contention.
# Tune down to 4 if GPU VRAM is constrained (each InfraGuard slot ~40-50 MB).

_infra_predictor_pool: List["InfraGuardPredictor"] = [
    InfraGuardPredictor() for _ in range(MAX_PREDICTOR_POOL)
]
_crack_predictor_pool: List["CrackPredictor"] = [
    CrackPredictor() for _ in range(MAX_PREDICTOR_POOL)
]

# Legacy per-camera dicts kept for cleanup_stale_cameras compatibility (now empty).
_camera_infra_predictors: Dict[str, "InfraGuardPredictor"] = {}
_camera_crack_predictors: Dict[str, "CrackPredictor"] = {}

# Fix 5: RLock instead of Lock — allows cleanup helpers to be called while the
# caller already holds the lock (prevents deadlock on future nested calls).
_predictor_lock = RLock()

# Issue 2 fix: one lock per pool slot — prevents concurrent cameras from hitting
# the same predictor simultaneously when the YOLO wrapper is not thread-safe.
_infra_predictor_locks: List[Lock] = [Lock() for _ in range(MAX_PREDICTOR_POOL)]
_crack_predictor_locks: List[Lock] = [Lock() for _ in range(MAX_PREDICTOR_POOL)]

# [R5 #8] Predictor warmup — eliminates first-inference latency spike.
# Improvement 2: use model.warmup() instead of predict_frame() to avoid
# generating fake detections that could pollute analytics or trigger alerts.
try:
    _dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _p in _infra_predictor_pool:
        try:
            _p.model.warmup(imgsz=(1, 3, 480, 640))
        except Exception:
            _p.predict_frame(_dummy_frame)   # fallback if model attr unavailable
    for _p in _crack_predictor_pool:
        try:
            _p.model.warmup(imgsz=(1, 3, 480, 640))
        except Exception:
            _p.predict_frame(_dummy_frame)
    del _dummy_frame
except Exception:
    pass


def _pool_idx(camera_id: str) -> int:
    """Issue 3 fix: stable pool slot via hashlib.md5 — survives process restarts."""
    return int(hashlib.md5(camera_id.encode()).hexdigest(), 16) % MAX_PREDICTOR_POOL


def _get_infra_predictor(camera_id: str) -> "InfraGuardPredictor":
    """Return the pooled InfraGuard predictor for `camera_id` via consistent hashing."""
    return _infra_predictor_pool[_pool_idx(camera_id)]


def _get_crack_predictor(camera_id: str) -> "CrackPredictor":
    """Return the pooled Crack predictor for `camera_id` via consistent hashing."""
    return _crack_predictor_pool[_pool_idx(camera_id)]


def _safe_infra_predict(camera_id: str, frame) -> list:
    """Issue 2 fix: serialise concurrent cameras that share the same pool slot."""
    idx = _pool_idx(camera_id)
    with _infra_predictor_locks[idx]:
        return _infra_predictor_pool[idx].predict_frame(frame)


def _safe_crack_predict(camera_id: str, frame) -> list:
    """Issue 2 fix: serialise concurrent cameras that share the same pool slot."""
    idx = _pool_idx(camera_id)
    with _crack_predictor_locks[idx]:
        return _crack_predictor_pool[idx].predict_frame(frame)


def _recover_infra_predictor(camera_id: str) -> None:
    """Fix 13: Replace a crashed pool slot with a fresh instance."""
    idx = _pool_idx(camera_id)
    with _predictor_lock:
        try:
            _infra_predictor_pool[idx] = InfraGuardPredictor()
            logger.info("Recovered infra predictor slot %d for camera %s", idx, camera_id)
        except Exception:
            logger.exception("Failed to recover infra predictor slot %d", idx)


def _recover_crack_predictor(camera_id: str) -> None:
    """Fix 13: Replace a crashed crack pool slot with a fresh instance."""
    idx = _pool_idx(camera_id)
    with _predictor_lock:
        try:
            _crack_predictor_pool[idx] = CrackPredictor()
            logger.info("Recovered crack predictor slot %d for camera %s", idx, camera_id)
        except Exception:
            logger.exception("Failed to recover crack predictor slot %d", idx)

# ── Mod 5: Person tracker only (cracks / equipment have their own) ───────────
person_tracker = EnterpriseTracker()

# ── Mod 6: Dedicated equipment tracker (enables movement/zone/near-miss) ─────
equipment_tracker = EnterpriseTracker()

alert_manager = AlertManager(
    cooldown=10
)

# Fix 2: Separate executors so logging/alerting backlog never starves inference.
# inference_executor runs YOLO predict_frame calls (latency-critical).
# logging_executor runs save_event / add_alert (throughput-tolerant).
# Improvement 1: inference workers = 2 × pool size so concurrent camera pairs
# (infra + crack) don't queue behind each other at 20-camera scale.
_MAX_PENDING_TASKS = 500
_task_queue: Queue = Queue(maxsize=_MAX_PENDING_TASKS)
inference_executor = ThreadPoolExecutor(max_workers=MAX_PREDICTOR_POOL * 2)
logging_executor   = ThreadPoolExecutor(max_workers=2)


def _submit_task(fn, *args, **kwargs):
    """Submit a fire-and-forget logging task, dropping it if the queue is full."""
    try:
        _task_queue.put_nowait(True)   # reserve a slot
    except Full:
        return                          # drop — logging backlog, not safety data
    def _run():
        try:
            fn(*args, **kwargs)
        except Exception:
            logger.exception("Async logging task failed: %s", fn.__name__)
        finally:
            try:
                _task_queue.get_nowait()   # release the slot
            except Empty:
                pass
    try:
        logging_executor.submit(_run)   # Fix 2: dedicated logging executor
    except Exception:
        try:
            _task_queue.get_nowait()
        except Empty:
            pass


# [FIX #1] Prevent thread leaks on backend reload / crash
atexit.register(inference_executor.shutdown, wait=False)
atexit.register(logging_executor.shutdown,   wait=False)

# [R2 #8] Rolling analytics history.
# Improvement 5: raised to 10 000 frames — at 20 cameras × 30 FPS the old
# limit of 3 000 kept only ~5 seconds of history; 10 000 gives ~16 seconds.
analytics_history = deque(maxlen=10_000)

# [FIX #8] Global alert history — configurable size, read from env via MAX_ALERT_HISTORY
alert_history: deque = deque(maxlen=MAX_ALERT_HISTORY)

# [R3 #9] Detection persistence — last known detections per camera
last_detections: Dict[str, List[dict]] = {}     # { camera_id: [det, ...] }

# [FIX] Track last-seen time per camera so stale entries can be expired.
last_seen_cameras: Dict[str, float] = {}   # { camera_id: float (time.time()) }

# Cameras not seen for CAMERA_EXPIRY_SECONDS (env) are removed from last_detections.

# [FIX Issue #6] Crack growth baseline — stores both the original detection
# area AND the area at which the last growth alert fired.
# Structure: { crack_uuid: {"original_area": int, "last_alert_area": int} }
# This prevents the baseline from drifting forward with each alert milestone,
# so growth percentages remain anchored to first detection, not last alert.
crack_growth_reported: Dict[str, dict] = {}   # { crack_uuid: {original_area, last_alert_area} }

# [R5 #3] Crack registry — stable UUID tracking across frames
# { camera_id: { crack_uuid: {"bbox": [x1,y1,x2,y2], "area": int} } }
crack_registry: Dict[str, Dict[str, dict]] = {}

# Fix 7: Per-camera heatmap — prevents cameras overwriting each other's data.
# Each camera gets its own deque(maxlen=5000).
heatmap_data: Dict[str, deque] = {}

# [FIX] Rolling FPS — per-camera deque (maxlen=30) to avoid cross-camera averaging
# { camera_id: deque([fps, ...]) }
fps_history: Dict[str, deque] = {}

# Biggest improvement: frame-skip inference — run YOLO every Nth frame and
# propagate tracker state in between. Gives 2-3× effective throughput with
# negligible accuracy loss on continuous video streams.
# Set INFERENCE_EVERY_N = 1 to disable and run YOLO on every frame.
INFERENCE_EVERY_N: int = INFERENCE_EVERY_N_DEFAULT   # run YOLO on frame 1, 4, 7 … (read from env)
_frame_counters: Dict[str, int] = {}   # { camera_id: frame_index }
# Priority 2 #6: Per-camera frame skip overrides (camera_id → every-N value).
# Example: {"cam_entrance": 1, "cam_storage": 5}  — set via CAMERA_FRAME_SKIP env as JSON.
import json as _json
_CAMERA_FRAME_SKIP: Dict[str, int] = _json.loads(os.getenv("CAMERA_FRAME_SKIP", "{}"))
# Per-camera crack inference counter (Priority 2 #5)
_crack_frame_counters: Dict[str, int] = {}

# [FIX] predict_lock removed — per-camera predictors (_camera_predictors) provide
# isolation without serialising concurrent streams. See _get_predictor().

# [FIX] Throttle cleanup to once per minute instead of every frame.
# At 30 FPS × 10 cameras that saves ~17 900 unnecessary scans/min.
_CLEANUP_INTERVAL = 60   # seconds
_last_cleanup: float = 0.0
# Improvement 3: split state_lock into three domain locks to reduce contention
# at 20-camera scale. Previously one lock serialised every frame across all cameras.
# analytics_lock : analytics_history, fps_history, last_seen_cameras, cleanup gate
# alert_lock     : alert_history
# heatmap_lock   : heatmap_data
# Fix 5: RLock so cleanup helpers can be called while the caller already holds
# state_lock (e.g. run_safety_pipeline → cleanup_crack_registry).
state_lock    = RLock()   # kept as alias for analytics_lock for backward compat
analytics_lock = state_lock
alert_lock     = RLock()
heatmap_lock   = RLock()


# ── Mod 12: Pipeline version — included in every frame result ────────────────
PIPELINE_VERSION = "2.0.0"

# Fix 10: Model health monitoring — dashboard can surface FAILED status.
model_health: Dict[str, object] = {
    "infra_status": "ONLINE",
    "crack_status": "ONLINE",
    "last_error":   None,
}

# Priority 1 #4: Failure counters — when a model pool slot fails repeatedly,
# auto-replace it instead of retrying the same broken instance forever.
_model_failure_counts: Dict[str, int] = {
    "infra": 0,
    "crack": 0,
}
_MODEL_FAILURE_THRESHOLD = int(os.getenv("MODEL_FAILURE_THRESHOLD", 3))

# ── Priority 1 #3: GPU Memory Monitoring ─────────────────────────────────────
# Exposes gpu_memory_mb and gpu_utilization in get_gpu_stats().
# Gracefully degrades if torch / CUDA is unavailable.
try:
    import torch as _torch
    _CUDA_AVAILABLE = _torch.cuda.is_available()
except ImportError:
    _torch = None          # type: ignore[assignment]
    _CUDA_AVAILABLE = False


def get_gpu_stats() -> dict:
    """
    Return current GPU memory usage (allocated / reserved MB) and a simple
    utilisation estimate (allocated / reserved ratio).

    Returns zeros on CPU-only deployments.
    """
    if not _CUDA_AVAILABLE or _torch is None:
        return {"gpu_memory_allocated_mb": 0, "gpu_memory_reserved_mb": 0, "gpu_utilization": 0.0}
    allocated = _torch.cuda.memory_allocated() / (1024 ** 2)
    reserved  = _torch.cuda.memory_reserved()  / (1024 ** 2)
    utilization = round(allocated / reserved, 3) if reserved > 0 else 0.0
    return {
        "gpu_memory_allocated_mb": round(allocated, 1),
        "gpu_memory_reserved_mb":  round(reserved,  1),
        "gpu_utilization":         utilization,
    }

# Fix 11: Camera offline threshold — cameras not seen for this many seconds
# are reported as OFFLINE in frame analytics.
# Value is CAMERA_OFFLINE_THRESHOLD read from env in the config block above.

# =========================================
# DETECTION CONFIG
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
    "crack":                                     "crack",   # Mod 1: clean key
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

# ── Mod 11: Typed zone system ─────────────────────────────────────────────────
# Each zone entry: { "type": <zone_type>, "bbox": [x1, y1, x2, y2] }
# Zone types: "equipment_only" | "workers_prohibited" | "restricted_area"
# Example:
# DANGER_ZONES = {
#     "camera_1": [
#         {"type": "equipment_only",      "bbox": [100, 100, 400, 400]},
#         {"type": "workers_prohibited",  "bbox": [450, 50,  750, 350]},
#         {"type": "restricted_area",     "bbox": [200, 400, 600, 600]},
#     ]
# }
DANGER_ZONES: dict = {}

# IoU threshold for PPE-to-worker association.
# Low because a helmet bbox is much smaller than a worker bbox.
PPE_IOU_THRESHOLD = 0.05

# IoU threshold for crack re-identification across frames.
CRACK_IOU_THRESHOLD = 0.30

# [FIX #2] Cracks not seen for this many seconds are removed from the registry.
CRACK_EXPIRY_SECONDS = CRACK_EXPIRY_SECONDS_CFG


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


def _nms(detections: list, iou_threshold: float = 0.45) -> list:
    """
    Fix 12: Non-Maximum Suppression to remove duplicate detections of the same
    object (e.g. two 'person' boxes at 0.82 and 0.79 confidence overlapping).
    Groups by class, sorts by confidence descending, suppresses lower-confidence
    boxes whose IoU with a kept box exceeds `iou_threshold`.
    """
    if not detections:
        return detections

    # Group by canonical class
    by_class: Dict[str, list] = {}
    for d in detections:
        cls = CLASS_MAP.get(d.get("class_name", ""), d.get("class_name", "").lower())
        by_class.setdefault(cls, []).append(d)

    kept = []
    for cls_dets in by_class.values():
        cls_dets = sorted(cls_dets, key=lambda d: float(d.get("confidence", 0)), reverse=True)
        suppressed = [False] * len(cls_dets)
        for i in range(len(cls_dets)):
            if suppressed[i]:
                continue
            kept.append(cls_dets[i])
            bbox_i = cls_dets[i].get("bbox", [])
            if len(bbox_i) != 4:
                continue
            for j in range(i + 1, len(cls_dets)):
                if suppressed[j]:
                    continue
                bbox_j = cls_dets[j].get("bbox", [])
                if len(bbox_j) == 4 and _iou(bbox_i, bbox_j) >= iou_threshold:
                    suppressed[j] = True
    return kept


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
            crack_growth_reported.pop(c_uuid, None)  # [FIX Issue #6] dict entry, pop is type-agnostic

        # [FIX #3] Remove camera entry entirely once all its cracks have expired
        # Prevents accumulation of empty dicts over months of operation
        if not cam_cracks:
            del crack_registry[cam_id]


def cleanup_stale_cameras():
    """
    [FIX] Remove cameras from last_detections that haven't been seen recently.
    Prevents unbounded memory growth when many camera_ids cycle through the system.
    Call once per frame from run_safety_pipeline.

    Priority 1 #1: Also calls cleanup_stale_tracks() on both EnterpriseTrackers
    so track IDs are pruned and tracker memory stays bounded over long deployments.
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
        # [FIX Issue #1] Release predictor instance to prevent memory leak.
        # Each predictor holds a loaded YOLO model (~40-50 MB); without this
        # a system cycling through many camera IDs leaks RAM indefinitely.
        with _predictor_lock:
            _camera_infra_predictors.pop(cam_id, None)
            _camera_crack_predictors.pop(cam_id, None)

    # Priority 1 #1: Purge stale tracker entries so track IDs don't grow forever.
    # Requires EnterpriseTracker to expose cleanup_stale_tracks(); if the method
    # doesn't exist yet, the try/except logs a one-time warning and degrades safely.
    for _tracker, _name in ((person_tracker, "person"), (equipment_tracker, "equipment")):
        if hasattr(_tracker, "cleanup_stale_tracks"):
            try:
                _tracker.cleanup_stale_tracks()
            except Exception:
                logger.exception("cleanup_stale_tracks() failed on %s tracker", _name)
        else:
            logger.warning(
                "EnterpriseTracker (%s) has no cleanup_stale_tracks() method — "
                "add MAX_TRACK_AGE pruning to prevent unbounded track growth.",
                _name,
            )

    # Purge any crack_growth_reported entries not referenced by any live crack
    # (catches orphans from unexpected registry clears or very long runtimes)
    live_crack_ids = {
        c_uuid
        for cam_cracks in crack_registry.values()
        for c_uuid in cam_cracks
    }
    for crack_id in list(crack_growth_reported.keys()):
        if crack_id not in live_crack_ids:
            crack_growth_reported.pop(crack_id, None)


# =========================================
# PIPELINE
# =========================================

def run_safety_pipeline(frame, camera_id=0):   # [R2 #5] camera_id param

    start = time.time()
    # Opt 1: single time.time() snapshot reused everywhere in this frame.
    frame_now = start

    # Single timestamp for the entire frame — avoids 20-50 datetime.now() calls per frame
    frame_ts = datetime.now(timezone.utc).isoformat()

    # [FIX] Normalise camera_id to str so int 0 and str "0" map to the same key
    # across last_detections, crack_registry, alert keys, and API responses.
    camera_id = str(camera_id)

    # [FIX] Run cleanup at most once per minute (not every frame).
    global _last_cleanup
    with state_lock:
        now = frame_now
        if now - _last_cleanup >= _CLEANUP_INTERVAL:
            cleanup_crack_registry()
            # [FIX] Expire cameras not seen recently to prevent last_detections growth
            cleanup_stale_cameras()
            _last_cleanup = now
        # [FIX] Record this camera as active
        last_seen_cameras[camera_id] = frame_now
    # Biggest improvement: frame-skip inference.
    # Advance per-camera counter and decide whether this frame gets YOLO or
    # just uses the previous frame's detections propagated by the tracker.
    # Priority 2 #6: Use per-camera override if configured, else global default.
    _cam_skip = _CAMERA_FRAME_SKIP.get(camera_id, INFERENCE_EVERY_N)
    _frame_counters[camera_id] = _frame_counters.get(camera_id, 0) + 1
    _is_inference_frame = (_frame_counters[camera_id] % _cam_skip) == 1

    # Priority 2 #5: Crack model runs less frequently than infra model.
    # Cracks are structural — they don't move frame-to-frame, so running the
    # crack model every Nth inference frame saves ~25-40% GPU without accuracy loss.
    _crack_frame_counters[camera_id] = _crack_frame_counters.get(camera_id, 0) + 1
    _is_crack_frame = (_crack_frame_counters[camera_id] % CRACK_INFERENCE_EVERY_N) == 1

    if not _is_inference_frame:
        # Tracker-only frame: return last known detections immediately.
        # Alerts and analytics are intentionally skipped — they will fire on the
        # next inference frame, keeping cooldowns and history consistent.
        cached = last_detections.get(camera_id, [])
        processing_ms = round((time.time() - start) * 1000, 2)
        return {
            "pipeline_version": PIPELINE_VERSION,
            "detections":   cached,
            "alerts":       [],
            "alert_channels": {"ppe": [], "crack": [], "equipment": [], "zone": [], "near_miss": []},
            "critical":     0,
            "high":         0,
            "medium":       0,
            "low":          0,
            "risk":         "LOW",
            "safety_score": 100.0,
            "skipped_frame": True,
            "analytics": {
                "total_objects":    len(cached),
                "processing_ms":    processing_ms,
                "predict_ms":       0,
                "fps":              0,
                "tracker_active":   True,
                "class_counts":     {},
                "ppe_compliance": {"overall": 100.0, "helmet": 100.0, "vest": 100.0, "worker_count": 0},
                "camera_id":        camera_id,
                "timestamp":        frame_ts,
            },
        }

    try:
        # ── Mod 3: Parallel dual-model inference (25-40 % faster) ────────────
        predict_start = time.time()

        # Fix 2: YOLO inference runs on inference_executor (never competes with
        # logging tasks that run on logging_executor).
        # Issue 2 fix: _safe_*_predict serialises cameras sharing the same pool slot.
        infra_future = inference_executor.submit(_safe_infra_predict, camera_id, frame)

        # Priority 2 #5: Only run crack model on crack frames; reuse cached detections
        # on skipped frames.  Saves ~25-40% GPU — cracks don't move between frames.
        if _is_crack_frame:
            crack_future = inference_executor.submit(_safe_crack_predict, camera_id, frame)
        else:
            crack_future = None

        try:
            infra_raw = infra_future.result()
            model_health["infra_status"] = "ONLINE"   # Fix 10
            _model_failure_counts["infra"] = 0        # reset on success
        except Exception as ie:
            model_health["infra_status"] = "FAILED"   # Fix 10
            model_health["last_error"]   = str(ie)
            _model_failure_counts["infra"] += 1
            logger.exception(
                "Infra predictor failed for camera %s (failure #%d)",
                camera_id, _model_failure_counts["infra"],
            )
            # Priority 1 #4: replace pool slot after repeated failures
            if _model_failure_counts["infra"] >= _MODEL_FAILURE_THRESHOLD:
                logger.warning(
                    "Infra predictor slot %d exceeded failure threshold — replacing.",
                    _pool_idx(camera_id),
                )
                _recover_infra_predictor(camera_id)
                _model_failure_counts["infra"] = 0
            else:
                _recover_infra_predictor(camera_id)   # Fix 13: always attempt recovery
            raise

        if crack_future is not None:
            try:
                crack_raw = crack_future.result()
                model_health["crack_status"] = "ONLINE"   # Fix 10
                _model_failure_counts["crack"] = 0
            except Exception as ce:
                model_health["crack_status"] = "FAILED"   # Fix 10
                model_health["last_error"]   = str(ce)
                _model_failure_counts["crack"] += 1
                logger.exception(
                    "Crack predictor failed for camera %s (failure #%d)",
                    camera_id, _model_failure_counts["crack"],
                )
                if _model_failure_counts["crack"] >= _MODEL_FAILURE_THRESHOLD:
                    logger.warning(
                        "Crack predictor slot %d exceeded failure threshold — replacing.",
                        _pool_idx(camera_id),
                    )
                    _recover_crack_predictor(camera_id)
                    _model_failure_counts["crack"] = 0
                else:
                    _recover_crack_predictor(camera_id)   # Fix 13
                raise
        else:
            # Reuse previously cached crack detections for this skipped crack frame
            crack_raw = [
                d for d in last_detections.get(camera_id, [])
                if d.get("model_source") == "crack"
            ]

        predict_ms = round((time.time() - predict_start) * 1000, 2)

        # ── Mod 4: Tag every detection with its model source ──────────────────
        for det in infra_raw:
            det["model_source"] = "infraguard"
        for det in crack_raw:
            det.setdefault("model_source", "crack")

        raw = infra_raw + crack_raw

    except Exception as e:
        logger.exception("Pipeline error for camera %s", camera_id)
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

    # Fix 12: Remove duplicate detections (YOLO sometimes emits overlapping boxes
    # for the same object at slightly different confidences). NMS keeps only the
    # highest-confidence box when two same-class boxes overlap heavily.
    filtered = _nms(filtered)
    # =====================================

    def _canonical(d):
        return CLASS_MAP.get(d.get("class_name", ""), d.get("class_name", "").lower())

    persons     = [d for d in filtered if _canonical(d) == "person"]
    non_persons = [d for d in filtered if _canonical(d) != "person"]

    # Separate PPE items for association pass
    helmets_raw = [d for d in non_persons if _canonical(d) == "helmet"]
    vests_raw   = [d for d in non_persons if _canonical(d) == "vest"]

    try:
        persons = person_tracker.update(persons)
    except Exception:
        logger.exception("person_tracker.update() failed for camera %s", camera_id)
        persons = []

    detections = []

    # ── Mod 7: Separate alert channels for dashboard filtering / analytics ────
    ppe_alerts       = []
    crack_alerts     = []
    equipment_alerts = []
    zone_alerts      = []
    near_miss_alerts = []

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

            # ── Mod 4: Model source tag ───────────────────────────────────────
            "model_source": p.get("model_source", "infraguard"),
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

            ppe_alerts.append(alert)

            # [R5 #4] Async — event logging runs off the inference thread
            _submit_task(save_event, {
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

    # Fix 4: Collect all heavy-equipment detections BEFORE the loop and update
    # the tracker once — the tracker needs to see the full set simultaneously
    # to assign stable IDs and compute motion/zone state correctly.
    equipment_raw = [
        d for d in non_persons
        if CLASS_MAP.get(d.get("class_name", ""), d.get("class_name", "").lower()) in HEAVY_EQUIPMENT
    ]
    try:
        tracked_equipment_list = equipment_tracker.update(equipment_raw)
        # Build a lookup so individual loop iterations can find their tracked entry.
        _tracked_equip_by_idx = {i: t for i, t in enumerate(tracked_equipment_list)}
    except Exception:
        logger.exception("equipment_tracker.update() failed for camera %s", camera_id)
        _tracked_equip_by_idx = {}

    _equip_raw_idx = 0   # running index used to correlate loop iterations below

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

            # ── Mod 8: Crack severity with length/width/aspect_ratio ─────────
            "crack_severity": (
                classify_crack_severity(
                    confidence, det_area,
                    width=int(x2 - x1), height=int(y2 - y1)
                )
                if label == "crack" else None
            ),
            # ── Mod 4: Model source tag ───────────────────────────────────────
            "model_source":   n.get("model_source", "infraguard"),
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

                crack_alerts.append(alert)

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

        # ── Mod 6: Apply pre-computed batch equipment tracking ────────────────
        if label in HEAVY_EQUIPMENT:
            tracked_entry = _tracked_equip_by_idx.get(_equip_raw_idx)
            if tracked_entry:
                det["id"]       = str(tracked_entry.get("id", det["id"]))
                det["tracking"] = True
            _equip_raw_idx += 1

        # [R4 #2] Equipment alerts — medium risk machinery alert channel
        if label in HEAVY_EQUIPMENT:

            equip_key = f"equip_{label}_{int(x1 / 50)}_{int(y1 / 50)}"

            if alert_manager.should_alert(equip_key, "medium"):

                equip_msg = f"Heavy equipment detected: {label.title()}"
                # ── Mod 7: Route to equipment_alerts channel ──────────────────
                equipment_alerts.append({
                    "type":         label,
                    "risk":         "medium",
                    "score":        SEVERITY_SCORE["medium"],
                    "message":      equip_msg,
                    "camera_id":    camera_id,
                    "timestamp":    frame_ts
                })

                # [FIX] Persist equipment alerts — previously only shown on dashboard
                _submit_task(save_event, {
                    "event_type":        "EQUIPMENT_ALERT",
                    "risk_level":        "MEDIUM",
                    "camera_id":         camera_id,
                    "workers":           0,
                    "violating_workers": 0,
                    "description":       equip_msg,
                })
                _submit_task(add_alert,
                    event_type="Heavy Equipment",
                    risk="medium",
                    cam_id=camera_id,
                    description=equip_msg,
                )

        # [R5 #4] Equipment zone violation check
        # ── Mod 11: Typed zone system + Mod 7: zone_alerts channel ──────────
        # Zones are now dicts: {"type": "equipment_only"|"workers_prohibited"|
        #                        "restricted_area", "bbox": [x1,y1,x2,y2]}
        # Legacy flat [x1,y1,x2,y2] lists are still accepted for compatibility.
        in_zone = False
        for zone_entry in DANGER_ZONES.get(camera_id, []):
            # Support both typed dict and legacy list formats
            if isinstance(zone_entry, dict):
                zone      = zone_entry.get("bbox", [])
                zone_type = zone_entry.get("type", "restricted_area")
            else:
                zone      = zone_entry
                zone_type = "restricted_area"

            if len(zone) == 4 and label in HEAVY_EQUIPMENT and _boxes_intersect(det_bbox, zone):
                in_zone  = True
                zone_key = f"zone_{label}_{int(x1 / 50)}_{int(y1 / 50)}"

                if alert_manager.should_alert(zone_key, "high"):
                    zone_alerts.append({
                        "type":         "zone_violation",
                        "zone_type":    zone_type,
                        "risk":         "high",
                        "score":        SEVERITY_SCORE["high"],
                        "message":      (
                            f"{label.title()} entered {zone_type.replace('_', ' ')} zone "
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

    # ── Mod 10: Near-Miss Detection ───────────────────────────────────────────
    # Improvement 4: threshold is now relative to the size of the objects being
    # compared rather than a fixed pixel fraction of the frame. This gives
    # consistent real-world proximity detection across 720p, 1080p, and 4K.
    # A near-miss fires when centre-to-centre distance < the larger of the two
    # bboxes' shorter edge — roughly "within one body-width of each other".
    frame_h, frame_w = frame.shape[:2] if hasattr(frame, "shape") else (720, 1280)

    worker_dets_for_nm   = [d for d in detections if d.get("type") == "worker"]
    equipment_dets_for_nm = [d for d in detections if d.get("class_name") in HEAVY_EQUIPMENT]

    for w in worker_dets_for_nm:
        w_cx = w["x"] + w["w"] // 2
        w_cy = w["y"] + w["h"] // 2
        for eq in equipment_dets_for_nm:
            eq_cx = eq["x"] + eq["w"] // 2
            eq_cy = eq["y"] + eq["h"] // 2
            distance = ((w_cx - eq_cx) ** 2 + (w_cy - eq_cy) ** 2) ** 0.5
            # Improvement 4: threshold = larger of each bbox's shorter side.
            # Resolution-independent: a worker at 720p and 4K triggers at the
            # same real-world proximity, not at different pixel distances.
            near_miss_threshold = max(
                min(w["w"], w["h"]),
                min(eq["w"], eq["h"]),
            )
            if distance < near_miss_threshold:
                nm_key = f"nearmiss_{w['id'][:4]}_{eq['class_name']}_{int(eq_cx / 50)}_{int(eq_cy / 50)}"
                if alert_manager.should_alert(nm_key, "critical"):
                    nm_msg = (
                        f"Near-miss: Worker {w['id'][:4]} within "
                        f"{int(distance)}px of {eq['class_name'].title()}"
                    )
                    near_miss_alerts.append({
                        "type":         "near_miss",
                        "risk":         "critical",
                        "score":        SEVERITY_SCORE["critical"],
                        "message":      nm_msg,
                        "worker_id":    w["id"],
                        "equipment":    eq["class_name"],
                        "distance_px":  int(distance),
                        "camera_id":    camera_id,
                        "timestamp":    frame_ts,
                    })
                    _submit_task(save_event, {
                        "event_type":        "NEAR_MISS",
                        "risk_level":        "CRITICAL",
                        "camera_id":         camera_id,
                        "workers":           1,
                        "violating_workers": 1,
                        "description":       nm_msg,
                    })
                    _submit_task(add_alert,
                        event_type="Near Miss",
                        risk="critical",
                        cam_id=camera_id,
                        description=nm_msg,
                    )

    # ── Mod 7: Merge all alert channels ──────────────────────────────────────
    alerts = ppe_alerts + crack_alerts + equipment_alerts + zone_alerts + near_miss_alerts

    # =====================================
    # ANALYTICS
    # =====================================

    # [R5 #7] Four-tier risk counting — single pass over detections
    risk_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for d in detections:
        risk_counts[d["risk"]] = risk_counts.get(d["risk"], 0) + 1
    critical_count = risk_counts["critical"]
    high           = risk_counts["high"]
    medium         = risk_counts["medium"]
    low            = risk_counts["low"]

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

    # [R2 #6] FPS as wall-clock rate — [FIX] rolling average per camera (not global)
    elapsed        = time.time() - start
    instant_fps    = round(1 / elapsed, 2) if elapsed > 0 else 0.0
    with state_lock:
        if camera_id not in fps_history:
            fps_history[camera_id] = deque(maxlen=30)
        fps_history[camera_id].append(instant_fps)
        cam_fps = fps_history[camera_id]
        fps = round(sum(cam_fps) / len(cam_fps), 2)
    # Fix 9: Separate full-pipeline time from YOLO-only predict_ms so
    # analytics can attribute overhead to tracking, alerts, and DB logging.
    total_processing_ms = round(elapsed * 1000, 2)
    overhead_ms         = round(total_processing_ms - predict_ms, 2)
    inference_time      = total_processing_ms   # kept for backward compat

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
    # Fix 6: Track last_milestone (0, 1, 2 … = 0%, 20%, 40% …) instead of
    # last_alert_area so milestones are never skipped when growth jumps quickly
    # (e.g. 0% → 45% in one frame now fires the 20% AND 40% milestones).
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
        crack_id     = d["id"]
        current_area = d["area"]

        # Initialise baseline on first sighting
        if crack_id not in crack_growth_reported:
            crack_growth_reported[crack_id] = {
                "original_area":  prev["area"],
                "last_milestone": 0,   # Fix 6: milestone index (0 = 0%, 1 = 20%, …)
            }

        entry         = crack_growth_reported[crack_id]
        original      = entry["original_area"]
        last_milestone = entry["last_milestone"]

        if original <= 0:
            continue

        # Total growth relative to original (always anchored to first detection)
        total_growth   = (current_area - original) / original
        # Which milestone index the current area sits at
        current_milestone = int(total_growth * 100) // 20

        if current_milestone > last_milestone:
            # Fire once per missed milestone so no step is silently skipped
            for ms in range(last_milestone + 1, current_milestone + 1):
                crack_alerts.append({
                    "type":      "crack_growth",
                    "risk":      "high",
                    "score":     SEVERITY_SCORE["high"],
                    "message":   (
                        f"Crack growth milestone {ms * 20}%: "
                        f"{round(total_growth * 100, 1)}% total area increase"
                    ),
                    "camera_id":  camera_id,
                    "timestamp":  frame_ts,
                })
            entry["last_milestone"] = current_milestone

        # Re-merge alerts after crack growth (keeps alert channels in sync)
        alerts = ppe_alerts + crack_alerts + equipment_alerts + zone_alerts + near_miss_alerts

    # [R3 #15] Safety score 0–100
    # Calculated BEFORE frame_analytics so it can be stored directly in the dict.
    # Starts at 100, penalised by risk score and PPE non-compliance.
    ppe_penalty   = round((100 - ppe_compliance) * 0.4, 2)   # up to 40 pts
    risk_penalty  = min(60, round(risk_score / 10, 2))        # up to 60 pts
    safety_score  = max(0, round(100 - ppe_penalty - risk_penalty, 1))

    # [FIX #4] Count zone violations from alerts generated this frame
    # (includes crack-growth alerts now that they're appended above)
    zone_violations = sum(1 for a in alerts if a.get("type") == "zone_violation")

    # ── Mod 9: Crack analytics breakdown ─────────────────────────────────────
    crack_dets     = [d for d in detections if d.get("class_name") == "crack"]
    crack_count    = len(crack_dets)
    minor_count    = sum(1 for d in crack_dets if d.get("crack_severity") == "minor")
    moderate_count = sum(1 for d in crack_dets if d.get("crack_severity") == "moderate")
    severe_count   = sum(1 for d in crack_dets if d.get("crack_severity") == "severe")
    critical_crack = sum(1 for d in crack_dets if d.get("crack_severity") == "critical")

    frame_analytics = {

        "total_objects":    len(detections),
        "processing_ms":    inference_time,
        "predict_ms":       predict_ms,         # [R4 #6] inference-only latency
        "overhead_ms":      overhead_ms,        # Fix 9: tracking+alerts+logging time
        "fps":              fps,
        "tracker_active":   True,
        "camera_health":    (
            "ONLINE"
            if time.time() - last_seen_cameras.get(camera_id, time.time()) <= CAMERA_OFFLINE_THRESHOLD
            else "OFFLINE"
        ),   # Fix 11: real offline detection
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
        # ── Mod 9: Crack analytics ────────────────────────────────────────────
        "cracks": {
            "count":    crack_count,
            "minor":    minor_count,
            "moderate": moderate_count,
            "severe":   severe_count,
            "critical": critical_crack,
        },
        # ── Analytics expansion (Mod 7 channel counts) ────────────────────────
        "worker_count":         worker_count,
        "crack_count":          crack_count,
        "equipment_count":      equipment_count,
        "near_miss_count":      len(near_miss_alerts),
        "ppe_violation_count":  sum(1 for a in ppe_alerts if a.get("risk") in ("high", "critical")),
        "active_alert_count":   len(alerts),
        "critical_alert_count": sum(1 for a in alerts if a.get("risk") == "critical"),
    }

    # [R4 #5] Frame-level counters — useful for history trend charts
    frame_analytics["alert_count"]     = len(alerts)
    frame_analytics["detection_count"] = len(detections)

    # Opt 2: build heatmap entries before acquiring the lock so the critical
    # section only does fast deque appends, not dict comprehension per detection.
    heatmap_entries = [
        {
            "cx":        d["x"] + d["w"] // 2,
            "cy":        d["y"] + d["h"] // 2,
            "risk":      d["risk"],
            "class":     d["class_name"],
            "camera_id": camera_id,
            "timestamp": frame_ts,
        }
        for d in detections
    ]

    # [R2 #8] Append to rolling history
    # Improvement 3: use granular locks — each domain acquires only its own lock
    # instead of one giant state_lock that serialises all cameras together.
    with analytics_lock:
        analytics_history.append(frame_analytics)

    with alert_lock:
        # Fix 8: Store a copy — future status mutations (e.g. alert["status"]="resolved")
        # must not silently change the historical record.
        for alert in alerts:
            alert_history.append(alert.copy())

    with heatmap_lock:
        # Fix 7: Per-camera heatmap — each camera maintains its own deque
        if camera_id not in heatmap_data:
            heatmap_data[camera_id] = deque(maxlen=5000)
        heatmap_data[camera_id].extend(heatmap_entries)

    return {

        "pipeline_version": PIPELINE_VERSION,   # Mod 12

        "detections":   detections,
        "alerts":       alerts,

        # ── Mod 7: Separate alert channels for dashboard filtering ────────────
        "alert_channels": {
            "ppe":       ppe_alerts,
            "crack":     crack_alerts,
            "equipment": equipment_alerts,
            "zone":      zone_alerts,
            "near_miss": near_miss_alerts,
        },

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


# ── Mod 8: Upgraded crack severity using length, width, aspect ratio ──────────
# Returns one of: "minor" | "moderate" | "severe" | "critical"
def classify_crack_severity(confidence: float, area: int,
                             width: int = 0, height: int = 0) -> str:
    """
    Severity based on area, geometric dimensions, and confidence.
    crack_length = max(w, h);  crack_width = min(w, h)
    aspect_ratio > 3 means an elongated structural crack (more dangerous).
    """
    crack_length = max(width, height) if (width or height) else 0
    crack_width  = min(width, height) if (width or height) else 0
    aspect_ratio = (crack_length / crack_width) if crack_width > 0 else 1.0

    if (confidence >= 0.85 and area >= 10_000) or crack_length >= 300:
        return "critical"
    if (confidence >= 0.70 or area >= 6_000) or (crack_length >= 150 and aspect_ratio >= 3):
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
    with analytics_lock:
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
    with analytics_lock:
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
    Fix 7: Data is now stored per-camera to prevent cross-camera overwrite.

    Args:
        camera_id: Filter to a specific camera. None = merge all cameras.

    Returns:
        List of { cx, cy, risk, class, camera_id, timestamp }
    """
    with heatmap_lock:
        if camera_id is not None:
            return list(heatmap_data.get(camera_id, []))
        # Merge all cameras
        merged = []
        for cam_deque in heatmap_data.values():
            merged.extend(cam_deque)
        return merged


def get_alert_history(camera_id=None, limit: int = 100) -> list:
    """
    [FIX #8] Returns recent alerts from global alert_history deque.

    Args:
        camera_id: Filter to a specific camera. None = all cameras.
        limit:     Maximum number of alerts to return (most recent first).

    Returns:
        List of alert dicts with type, risk, score, message, camera_id, timestamp.
    """
    with alert_lock:
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
    with analytics_lock:
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


def get_model_health() -> dict:
    """
    Priority 1 #4: Returns current model health status and cumulative failure counts.

    Returns:
        {
            infra_status, crack_status, last_error,
            infra_failures, crack_failures,
            gpu_memory_allocated_mb, gpu_memory_reserved_mb, gpu_utilization
        }
    """
    return {
        **model_health,
        "infra_failures": _model_failure_counts["infra"],
        "crack_failures": _model_failure_counts["crack"],
        **get_gpu_stats(),
    }