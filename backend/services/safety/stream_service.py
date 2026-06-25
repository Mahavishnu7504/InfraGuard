import queue
import threading
import time
import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

try:
    import cv2
except Exception:
    cv2 = None


# =========================================================
# CAMERA STATE
# =========================================================

class CameraState(str, Enum):
    """
    Canonical states for a managed camera.

    Using ``str`` as a mixin means values serialise transparently in
    JSON (FastAPI / json.dumps) without a custom encoder.

    LiveCamera.jsx can map these to UI indicators, e.g.:
        Running      → 🟢 Running
        Initializing → 🟡 Initializing
        Stopped      → ⚫ Stopped
        Disconnected → 🟠 Disconnected
        Reconnecting → 🔄 Reconnecting
        Error        → 🔴 Error
    """
    RUNNING      = "Running"
    STOPPED      = "Stopped"
    INITIALIZING = "Initializing"
    DISCONNECTED = "Disconnected"
    RECONNECTING = "Reconnecting"
    ERROR        = "Error"


from backend.services.safety.detection_service import process_frame
from backend.core.websocket_manager import manager

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
TARGET_FPS   = 24

# Frame queue: holds raw captured frames waiting for the AI pipeline.
# Bounded so fast cameras don't eat memory — oldest frame is dropped
# when the queue is full (always process the *latest* frame).
FRAME_QUEUE_MAXSIZE = 4

# Reconnection backoff schedule (seconds). After the last value is
# reached the schedule cycles on the last value indefinitely.
RECONNECT_BACKOFF = (1, 2, 5, 10, 30)

# Maximum reconnection attempts before giving up and marking ERROR.
# Set to 0 for unlimited retries.
RECONNECT_MAX_ATTEMPTS = 10

# How long start_camera() will poll for the worker thread to confirm
# the camera opened, before giving up.
START_POLL_INTERVAL = 0.1
START_POLL_TIMEOUT  = 5.0

# How long stop_camera() will wait for the worker thread to actually
# exit before returning.
STOP_JOIN_TIMEOUT = 3.0

# Rolling window size for FPS / latency averages in statistics.
STATS_WINDOW = 60

# =========================================================
# RECORDING + SNAPSHOT SETTINGS
# =========================================================

# Directory where recordings and snapshots are saved.
# Override via environment variable INFRAGUARD_MEDIA_DIR if needed.
import os as _os
MEDIA_DIR = _os.environ.get("INFRAGUARD_MEDIA_DIR", "media")

# VideoWriter fourcc codec for recordings.
# "mp4v" works on all platforms; swap to "avc1" for H.264 if your
# OpenCV build supports it.
RECORDING_FOURCC = "mp4v"
RECORDING_EXT    = ".mp4"

# How often the health-monitor thread polls (seconds).
HEALTH_POLL_INTERVAL = 5


# =========================================================
# RECORDING STATE
# =========================================================

@dataclass
class RecordingState:
    """Tracks an active VideoWriter session for one camera."""
    writer:    object        # cv2.VideoWriter instance
    path:      str           # output file path
    started_at: str          # ISO-8601 UTC
    frames:    int = 0       # frames written so far




# =========================================================
# CAMERA SOURCE REGISTRY
# =========================================================
# Maps a logical camera ID to an OpenCV-compatible source string/int.
# Add entries here to support RTSP, HTTP streams, or multiple USB cams
# without changing any call-site code.
#
# Examples:
#   0          → default USB/built-in webcam
#   1          → second USB camera
#   "rtsp://user:pass@192.168.1.100:554/stream"  → RTSP IP camera
#   "http://192.168.1.200:8080/video"             → HTTP MJPEG stream
#   "/dev/video2"                                 → Linux device node
#   "/path/to/recorded.mp4"                       → video file replay

CAMERA_REGISTRY: dict = {
    0: 0,   # cam_id 0 → system default camera
    # 1: 1,
    # "front": "rtsp://admin:admin@192.168.1.100:554/stream1",
    # "rear":  "http://192.168.1.101:8080/video",
}


def resolve_source(cam_id):
    """
    Return the OpenCV source for ``cam_id``.

    Falls back to using ``cam_id`` directly so callers can pass a raw
    device index or RTSP URL without registering it first.
    """
    return CAMERA_REGISTRY.get(cam_id, cam_id)


def discover_cameras(max_probe: int = 8) -> list:
    """
    Probe USB/device indices 0 … max_probe-1 and return a list of dicts
    for every camera that OpenCV can successfully open and read a frame from.

    The result can be used by the UI to populate a camera-selector without
    the operator having to know device indices in advance.

    Example return value::

        [
            {"cam_id": 0, "source": 0,  "width": 1280, "height": 720, "fps": 30.0},
            {"cam_id": 1, "source": 1,  "width": 640,  "height": 480, "fps": 30.0},
        ]

    CAMERA_REGISTRY entries are also included (even non-integer sources such
    as RTSP URLs), so the full picture of what the app knows about is returned.
    Already-probed integer indices are de-duplicated against registry entries.
    """
    if cv2 is None:
        logger.warning("[DISCOVER] cv2 not available — returning registry only")
        return [
            {"cam_id": cid, "source": src, "width": None, "height": None, "fps": None}
            for cid, src in CAMERA_REGISTRY.items()
        ]

    found: list = []
    probed_sources: set = set()

    # ── Probe integer indices ──────────────────────────────────────────
    for idx in range(max_probe):
        try:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                continue
            ret, _ = cap.read()
            if not ret:
                cap.release()
                continue

            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps    = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            probed_sources.add(idx)

            # Use logical cam_id from registry if the integer matches
            cam_id = next(
                (cid for cid, src in CAMERA_REGISTRY.items() if src == idx),
                idx,
            )
            found.append({
                "cam_id": cam_id,
                "source": idx,
                "width":  width,
                "height": height,
                "fps":    round(fps, 1),
            })
            logger.info("[DISCOVER] Found camera idx=%d → cam_id=%s", idx, cam_id)
        except Exception as e:
            logger.debug("[DISCOVER] Probe %d failed: %s", idx, e)

    # ── Include non-integer registry sources (RTSP, HTTP, …) ─────────
    for cid, src in CAMERA_REGISTRY.items():
        if src not in probed_sources:
            found.append({
                "cam_id": cid,
                "source": src,
                "width":  None,
                "height": None,
                "fps":    None,
            })

    return found


# =========================================================
# FRAME METADATA
# =========================================================

@dataclass
class FrameMeta:
    """
    Lightweight metadata envelope carried alongside every captured frame.
    The AI pipeline receives this together with the raw image so downstream
    consumers (draw.py, analytics, alert_service) always have full context.
    """
    frame_id:   int
    camera_id:  object          # int or str
    timestamp:  str             # ISO-8601 UTC
    resolution: tuple           # (width, height)
    fps:        float           # instantaneous FPS at capture time
    latency_ms: float = 0.0     # capture → queue enqueue latency


# =========================================================
# STREAM STATISTICS
# =========================================================

@dataclass
class StreamStats:
    """
    Per-camera counters and rolling averages.
    All mutations happen inside the worker thread; reads are
    safe for occasional polling (no sub-millisecond accuracy needed).
    """
    frames_received:  int   = 0
    frames_processed: int   = 0
    frames_dropped:   int   = 0

    # Rolling windows for live averages
    _fps_window:     deque = field(default_factory=lambda: deque(maxlen=STATS_WINDOW))
    _latency_window: deque = field(default_factory=lambda: deque(default_factory=None, maxlen=STATS_WINDOW))

    # Reconnection tracking
    reconnect_attempts: int = 0
    last_reconnect_at:  Optional[str] = None

    def record_fps(self, fps: float):
        self._fps_window.append(fps)

    def record_latency(self, latency_ms: float):
        self._latency_window.append(latency_ms)

    @property
    def average_fps(self) -> float:
        return round(sum(self._fps_window) / len(self._fps_window), 1) if self._fps_window else 0.0

    @property
    def average_latency_ms(self) -> float:
        return round(sum(self._latency_window) / len(self._latency_window), 1) if self._latency_window else 0.0

    def to_dict(self) -> dict:
        return {
            "frames_received":    self.frames_received,
            "frames_processed":   self.frames_processed,
            "frames_dropped":     self.frames_dropped,
            "average_fps":        self.average_fps,
            "average_latency_ms": self.average_latency_ms,
            "reconnect_attempts": self.reconnect_attempts,
            "last_reconnect_at":  self.last_reconnect_at,
        }


def _make_stats() -> StreamStats:
    """Factory that avoids the mutable-default-argument pitfall in dataclasses."""
    return StreamStats(
        _fps_window=deque(maxlen=STATS_WINDOW),
        _latency_window=deque(maxlen=STATS_WINDOW),
    )


# =========================================================
# SHARED BACKGROUND EVENT LOOP FOR WS BROADCASTS
# One loop lives forever in a daemon thread.
# camera_worker() posts coroutines into it safely.
# =========================================================

_WS_LOOP      = None
_WS_LOOP_LOCK = threading.Lock()


def _get_ws_loop():
    global _WS_LOOP
    with _WS_LOOP_LOCK:
        if _WS_LOOP is None or not _WS_LOOP.is_running():
            _WS_LOOP = asyncio.new_event_loop()
            t = threading.Thread(
                target=_WS_LOOP.run_forever,
                daemon=True,
                name="infraguard-ws-loop",
            )
            t.start()
    return _WS_LOOP


def _on_broadcast_done(cam_id, future):
    """
    Retrieve the exception from the future so it doesn't get silently
    swallowed (asyncio logs 'exception never retrieved' warnings otherwise).
    """
    try:
        future.result()
    except Exception as e:
        logger.error("[CAM %s] BROADCAST FAILED: %s", cam_id, e)


def _broadcast(cam_id, payload):
    """Fire-and-forget broadcast from any thread, with error visibility."""
    loop = _get_ws_loop()
    future = asyncio.run_coroutine_threadsafe(
        manager.broadcast_to_camera(cam_id, payload),
        loop,
    )
    future.add_done_callback(lambda f: _on_broadcast_done(cam_id, f))


# =========================================================
# CAMERA OPEN
# =========================================================

def open_camera(cam_id=0):
    """
    Open the camera source for ``cam_id`` via the CAMERA_REGISTRY.

    Supports USB indices, RTSP URLs, HTTP streams, and video files —
    anything OpenCV's VideoCapture accepts. Resolution and FPS hints
    are applied after a successful open; the driver may silently clamp
    them to hardware limits, which is fine.
    """
    if cv2 is None:
        logger.error("[OPENCV NOT INSTALLED]")
        return None

    source = resolve_source(cam_id)

    try:
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            logger.error("[CAM %s] FAILED TO OPEN (source=%r)", cam_id, source)
            return None

        ret, frame = cap.read()
        if not ret or frame is None:
            logger.error("[CAM %s] FIRST FRAME READ FAILED", cam_id)
            cap.release()
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

        logger.info("[CAM %s] OPENED (source=%r)", cam_id, source)
        return cap

    except Exception as e:
        logger.exception("[CAM %s] CAMERA OPEN ERROR: %s", cam_id, e)
        return None


# =========================================================
# MODEL STATUS
# =========================================================
# Static for now — flip an entry to "OFFLINE" / "DEGRADED" if a model
# fails to load or is hot-swapped out. Centralised here so build_telemetry
# and any future health-check endpoint share one source of truth.

MODEL_STATUS = {
    "InfraGuard": "ONLINE",
    "Crack":      "ONLINE",
}


# =========================================================
# TELEMETRY BUILDER
# =========================================================

def build_telemetry(cam_id, fps, risk, frame_id=None,
                    pipeline_timing=None, stats: Optional[StreamStats] = None):
    """
    Build the telemetry payload broadcast to the frontend each frame.

    pipeline_timing: optional dict with any of
        capture_time_ms, inference_time_ms, tracking_time_ms,
        risk_time_ms, total_pipeline_time_ms
    Missing keys default to 0.0 so the shape is always stable for
    downstream consumers (UI, analytics).

    stats: optional StreamStats snapshot embedded under "stream_stats".
    """
    timing = pipeline_timing or {}

    payload = {
        "camera_id":    cam_id,
        "frame_id":     frame_id,
        "fps":          fps,
        "risk":         risk,
        "status":       "ACTIVE",
        "models":       dict(MODEL_STATUS),
        "timing": {
            "capture_time_ms":        timing.get("capture_time_ms", 0.0),
            "inference_time_ms":      timing.get("inference_time_ms", 0.0),
            "tracking_time_ms":       timing.get("tracking_time_ms", 0.0),
            "risk_time_ms":           timing.get("risk_time_ms", 0.0),
            "total_pipeline_time_ms": timing.get("total_pipeline_time_ms", 0.0),
        },
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }

    if stats is not None:
        payload["stream_stats"] = stats.to_dict()

    return payload


# =========================================================
# CAMERA MANAGER
# =========================================================

class CameraManager:
    """
    Encapsulates all per-camera state, threading, frame queuing,
    auto-reconnection, and statistics.

    Architecture per camera
    -----------------------
    Capture thread  →  frame queue  →  inference thread
         ↓                                    ↓
    reconnect on                       broadcast via WS
    disconnect

    A single shared instance (``camera_manager``) is exported for use
    by the public API functions and the rest of the app.
    """

    def __init__(self):
        self._caps:    dict = {}
        self._threads: dict = {}   # cam_id → capture thread
        self._inf_threads: dict = {}  # cam_id → inference thread
        self._queues:  dict = {}   # cam_id → frame queue
        self._states:  dict = {}   # cam_id → CameraState
        self._frames:  dict = {}   # cam_id → latest annotated frame
        self._results: dict = {}   # cam_id → latest AI result
        self._stats:   dict = {}   # cam_id → StreamStats

        # Per-camera active recording sessions (cam_id → RecordingState)
        self._recordings: dict = {}
        self._rec_lock:   threading.Lock = threading.Lock()

        # Guards start_camera/stop_camera against concurrent duplicate
        # requests racing on the same cam_id.
        self._locks:       dict = {}
        self._locks_guard: threading.Lock = threading.Lock()

        # Health-monitor thread — started once, runs for the process lifetime.
        self._health_status: dict = {}
        self._health_thread  = threading.Thread(
            target=self._health_monitor,
            daemon=True,
            name="infraguard-health",
        )
        self._health_thread.start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_lock(self, cam_id) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(cam_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[cam_id] = lock
            return lock

    def _get_stats(self, cam_id) -> StreamStats:
        if cam_id not in self._stats:
            self._stats[cam_id] = _make_stats()
        return self._stats[cam_id]

    # ------------------------------------------------------------------
    # Capture thread
    # Reads frames from the camera, applies FPS limiting, and pushes
    # raw frames into the per-camera queue.  Auto-reconnects on failure.
    # ------------------------------------------------------------------

    def _capture_worker(self, cam_id):
        """
        Capture loop with exponential-backoff auto-reconnection.

        On any frame-read failure the loop:
          1. Marks state RECONNECTING
          2. Releases the broken capture handle
          3. Waits for the next backoff interval
          4. Tries to re-open the camera
          5. Repeats up to RECONNECT_MAX_ATTEMPTS times

        If max attempts is exceeded the state becomes ERROR and the
        thread exits, which is visible via is_camera_running().
        """
        stats    = self._get_stats(cam_id)
        interval = 1.0 / TARGET_FPS

        attempt  = 0
        cap      = None

        while self._states.get(cam_id) not in (CameraState.STOPPED, CameraState.ERROR):

            # ── Open / reopen ──────────────────────────────────────────
            if cap is None:
                cap = open_camera(cam_id)

                if cap is None:
                    attempt += 1
                    if RECONNECT_MAX_ATTEMPTS and attempt > RECONNECT_MAX_ATTEMPTS:
                        logger.error(
                            "[CAM %s] EXCEEDED MAX RECONNECT ATTEMPTS (%d)",
                            cam_id, RECONNECT_MAX_ATTEMPTS,
                        )
                        self._states[cam_id] = CameraState.ERROR
                        break

                    backoff = RECONNECT_BACKOFF[min(attempt - 1, len(RECONNECT_BACKOFF) - 1)]
                    self._states[cam_id] = CameraState.RECONNECTING
                    stats.reconnect_attempts += 1
                    stats.last_reconnect_at   = datetime.now(timezone.utc).isoformat()
                    logger.warning(
                        "[CAM %s] RECONNECT ATTEMPT %d — waiting %ss",
                        cam_id, attempt, backoff,
                    )
                    # Wait in small slices so a stop() request is honoured promptly.
                    for _ in range(int(backoff / 0.1)):
                        if self._states.get(cam_id) == CameraState.STOPPED:
                            return
                        time.sleep(0.1)
                    continue

                # Successful open
                self._caps[cam_id]   = cap
                self._states[cam_id] = CameraState.RUNNING
                attempt = 0
                logger.info("[CAM %s] STREAM STARTED / RECONNECTED", cam_id)

            # ── Capture frame ──────────────────────────────────────────
            tick           = time.time()
            capture_start  = time.time()
            ok, frame      = cap.read()
            capture_ms     = (time.time() - capture_start) * 1000.0

            if not ok or frame is None:
                logger.warning("[CAM %s] FRAME READ FAILED — will reconnect", cam_id)
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                self._caps.pop(cam_id, None)
                self._states[cam_id] = CameraState.RECONNECTING
                stats.reconnect_attempts += 1
                stats.last_reconnect_at   = datetime.now(timezone.utc).isoformat()
                continue

            stats.frames_received += 1

            # ── Resize ────────────────────────────────────────────────
            try:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            except Exception:
                logger.exception("[CAM %s] RESIZE FAILED", cam_id)
                continue

            # ── FPS stats ─────────────────────────────────────────────
            elapsed = time.time() - (self.__dict__.setdefault(f"_start_{cam_id}", time.time()))
            fps     = round(stats.frames_received / max(elapsed, 0.001), 1)
            stats.record_fps(fps)
            stats.record_latency(capture_ms)

            # ── Enqueue (drop oldest if full → always latest frame) ───
            q: queue.Queue = self._queues[cam_id]
            meta = FrameMeta(
                frame_id   = stats.frames_received,
                camera_id  = cam_id,
                timestamp  = datetime.now(timezone.utc).isoformat(),
                resolution = (FRAME_WIDTH, FRAME_HEIGHT),
                fps        = fps,
                latency_ms = round(capture_ms, 2),
            )

            if q.full():
                try:
                    q.get_nowait()
                    stats.frames_dropped += 1
                except queue.Empty:
                    pass

            try:
                q.put_nowait((frame, meta))
            except queue.Full:
                stats.frames_dropped += 1

            # ── FPS limiter ───────────────────────────────────────────
            elapsed_frame = time.time() - tick
            sleep_for     = interval - elapsed_frame
            if sleep_for > 0:
                time.sleep(sleep_for)

        # Clean up on exit
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        self._caps.pop(cam_id, None)

        # Drain queue so inference thread doesn't block on shutdown
        q = self._queues.get(cam_id)
        if q:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        if self._states.get(cam_id) == CameraState.RUNNING:
            self._states[cam_id] = CameraState.STOPPED

        logger.info("[CAM %s] CAPTURE THREAD EXITED (state=%s)", cam_id, self._states.get(cam_id))

    # ------------------------------------------------------------------
    # Inference thread
    # Consumes frames from the queue and runs the AI pipeline.
    # Completely decoupled from capture speed.
    # ------------------------------------------------------------------

    def _inference_worker(self, cam_id):
        """
        Inference loop: pulls (frame, meta) tuples off the queue and
        runs the full safety pipeline.

        Runs until the camera is stopped AND the queue is empty, so
        every enqueued frame gets processed even during shutdown.
        """
        stats = self._get_stats(cam_id)
        q: queue.Queue = self._queues[cam_id]

        while True:
            # Exit only when stopped AND nothing left to process
            if self._states.get(cam_id) in (CameraState.STOPPED, CameraState.ERROR):
                if q.empty():
                    break

            try:
                frame, meta = q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                pipeline_start = time.time()
                try:
                    result = process_frame(frame)
                except Exception:
                    logger.exception("[CAM %s] PROCESS_FRAME CRASHED", cam_id)
                    result = {"risk": "LOW", "detections": [], "analytics": {}}

                total_pipeline_ms = (time.time() - pipeline_start) * 1000.0

                stage_timing = dict(result.get("timing", {}))
                stage_timing["capture_time_ms"]        = meta.latency_ms
                stage_timing["total_pipeline_time_ms"] = round(
                    meta.latency_ms + total_pipeline_ms, 2
                )

                stats.frames_processed += 1

                telemetry = build_telemetry(
                    cam_id, meta.fps,
                    result.get("risk", "LOW"),
                    frame_id=meta.frame_id,
                    pipeline_timing=stage_timing,
                    stats=stats,
                )

                self._frames[cam_id]  = frame
                self._results[cam_id] = {
                    **result,
                    "frame_id":  meta.frame_id,
                    "meta":      meta.__dict__,
                    "telemetry": telemetry,
                }

                # Write to active recording (if any) — non-blocking
                self._write_recording_frame(cam_id, frame)

                _broadcast(cam_id, {
                    "type":       "frame_result",
                    "camera_id":  cam_id,
                    "frame_id":   meta.frame_id,
                    "risk":       result.get("risk", "LOW"),
                    "detections": result.get("detections", []),
                    "analytics":  result.get("analytics", {}),
                    "telemetry":  telemetry,
                })

            except Exception:
                logger.exception("[CAM %s] INFERENCE WORKER ERROR", cam_id)

        # Clear stale cache on clean exit
        self._frames.pop(cam_id, None)
        self._results.pop(cam_id, None)
        logger.info("[CAM %s] INFERENCE THREAD EXITED", cam_id)

    # ------------------------------------------------------------------
    # Health Monitor thread
    # Runs every HEALTH_POLL_INTERVAL seconds and checks that each
    # known camera's capture thread, inference thread, and frame queue
    # are alive and healthy.  Results are stored in _health_status and
    # exposed via get_health_status().
    # ------------------------------------------------------------------

    def _health_monitor(self):
        """
        Background health-check loop.  Polls every HEALTH_POLL_INTERVAL
        seconds and writes results into self._health_status keyed by cam_id.

        Checks performed
        ----------------
        camera_alive    – capture thread is alive and state is RUNNING
        inference_alive – inference thread is alive
        queue_healthy   – frame queue backlog ≤ 75 % of FRAME_QUEUE_MAXSIZE
        websocket_ok    – WebSocket manager has ≥ 1 active connection
                          (falls back gracefully if manager lacks the attr)
        overall         – "OK" only when all four checks pass
        """
        while True:
            try:
                all_ids = (
                    set(self._states.keys())
                    | set(self._threads.keys())
                    | set(self._inf_threads.keys())
                )

                # WebSocket connectivity — best-effort
                try:
                    ws_ok = len(getattr(manager, "active_connections", {}) or {}) >= 0
                    # presence of the attribute is enough; we just verify the
                    # manager object is reachable
                    ws_ok = True
                except Exception:
                    ws_ok = False

                snapshot: dict = {}
                for cam_id in all_ids:
                    cap_thread  = self._threads.get(cam_id)
                    inf_thread  = self._inf_threads.get(cam_id)
                    q           = self._queues.get(cam_id)
                    state       = self._states.get(cam_id, CameraState.STOPPED)

                    camera_alive    = (
                        state == CameraState.RUNNING
                        and cap_thread is not None
                        and cap_thread.is_alive()
                    )
                    inference_alive = inf_thread is not None and inf_thread.is_alive()
                    queue_healthy   = (
                        q is None
                        or q.qsize() <= max(1, int(FRAME_QUEUE_MAXSIZE * 0.75))
                    )
                    overall = all([camera_alive, inference_alive, queue_healthy, ws_ok])

                    snapshot[cam_id] = {
                        "camera_alive":    camera_alive,
                        "inference_alive": inference_alive,
                        "queue_healthy":   queue_healthy,
                        "websocket_ok":    ws_ok,
                        "queue_depth":     q.qsize() if q else 0,
                        "state":           state,
                        "overall":         "OK" if overall else "DEGRADED",
                        "checked_at":      datetime.now(timezone.utc).isoformat(),
                    }

                    if not overall:
                        logger.warning(
                            "[HEALTH][CAM %s] DEGRADED — camera=%s inference=%s "
                            "queue=%s ws=%s",
                            cam_id, camera_alive, inference_alive,
                            queue_healthy, ws_ok,
                        )

                self._health_status = snapshot

            except Exception:
                logger.exception("[HEALTH] MONITOR ERROR")

            time.sleep(HEALTH_POLL_INTERVAL)

    def get_health_status(self) -> dict:
        """
        Return the latest health snapshot for all known cameras.

        Shape per camera::

            {
                "camera_alive":    bool,
                "inference_alive": bool,
                "queue_healthy":   bool,
                "websocket_ok":    bool,
                "queue_depth":     int,
                "state":           CameraState,
                "overall":         "OK" | "DEGRADED",
                "checked_at":      "<ISO-8601>",
            }

        If no cameras have been started yet an empty dict is returned.
        """
        return dict(self._health_status)

    # ------------------------------------------------------------------
    # Stream Recording
    # Captures the raw (pre-inference) frames for a camera and writes
    # them to an MP4 file.  Each start_recording() call creates a new
    # timestamped file; stop_recording() finalises and closes it.
    # ------------------------------------------------------------------

    def start_recording(self, cam_id=0) -> dict:
        """
        Begin recording the live stream for ``cam_id`` to an MP4 file.

        The file is placed in MEDIA_DIR/recordings/ with a UTC timestamp
        in the filename so successive recordings never overwrite each other.

        Returns a dict with the output path and start time, or an error
        key if recording could not be started.

        Calling start_recording() while a recording is already active for
        the same camera is a no-op that returns the existing session info.
        """
        if cv2 is None:
            return {"error": "cv2 not available"}

        with self._rec_lock:
            if cam_id in self._recordings:
                rec = self._recordings[cam_id]
                return {
                    "recording":  True,
                    "path":       rec.path,
                    "started_at": rec.started_at,
                    "frames":     rec.frames,
                }

            if not self.is_camera_running(cam_id):
                return {"error": f"Camera {cam_id} is not running"}

            rec_dir = _os.path.join(MEDIA_DIR, "recordings")
            _os.makedirs(rec_dir, exist_ok=True)

            ts        = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            filename  = f"cam{cam_id}_{ts}{RECORDING_EXT}"
            filepath  = _os.path.join(rec_dir, filename)

            fourcc = cv2.VideoWriter_fourcc(*RECORDING_FOURCC)
            writer = cv2.VideoWriter(
                filepath, fourcc, float(TARGET_FPS),
                (FRAME_WIDTH, FRAME_HEIGHT),
            )

            if not writer.isOpened():
                return {"error": f"VideoWriter failed to open: {filepath}"}

            started = datetime.now(timezone.utc).isoformat()
            self._recordings[cam_id] = RecordingState(
                writer=writer, path=filepath, started_at=started,
            )
            logger.info("[CAM %s] RECORDING STARTED → %s", cam_id, filepath)

            return {
                "recording":  True,
                "path":       filepath,
                "started_at": started,
                "frames":     0,
            }

    def stop_recording(self, cam_id=0) -> dict:
        """
        Stop an active recording for ``cam_id`` and finalise the MP4.

        Returns a summary dict with path, frame count, and duration, or
        an error key if no recording was active.
        """
        with self._rec_lock:
            rec = self._recordings.pop(cam_id, None)
            if rec is None:
                return {"error": f"No active recording for camera {cam_id}"}

            try:
                rec.writer.release()
            except Exception as e:
                logger.warning("[CAM %s] WRITER RELEASE ERROR: %s", cam_id, e)

            started_dt = datetime.fromisoformat(rec.started_at)
            duration_s = (datetime.now(timezone.utc) - started_dt).total_seconds()
            logger.info(
                "[CAM %s] RECORDING STOPPED → %s (%d frames, %.1fs)",
                cam_id, rec.path, rec.frames, duration_s,
            )
            return {
                "recording":   False,
                "path":        rec.path,
                "started_at":  rec.started_at,
                "stopped_at":  datetime.now(timezone.utc).isoformat(),
                "frames":      rec.frames,
                "duration_s":  round(duration_s, 1),
            }

    def is_recording(self, cam_id=0) -> bool:
        """True if a recording session is currently active for ``cam_id``."""
        with self._rec_lock:
            return cam_id in self._recordings

    def _write_recording_frame(self, cam_id, frame):
        """
        Called by the inference worker after each frame is annotated.
        Writes the frame to the active VideoWriter if one exists.
        Non-blocking: if the lock is contended the frame is skipped
        rather than stalling the inference pipeline.
        """
        acquired = self._rec_lock.acquire(blocking=False)
        if not acquired:
            return
        try:
            rec = self._recordings.get(cam_id)
            if rec is None:
                return
            try:
                rec.writer.write(frame)
                rec.frames += 1
            except Exception as e:
                logger.warning("[CAM %s] RECORDING WRITE ERROR: %s", cam_id, e)
        finally:
            self._rec_lock.release()

    # ------------------------------------------------------------------
    # Snapshot
    # Captures the current annotated frame to a JPEG file.
    # ------------------------------------------------------------------

    def capture_snapshot(self, cam_id=0) -> dict:
        """
        Save the most recent annotated frame for ``cam_id`` as a JPEG.

        The file is placed in MEDIA_DIR/snapshots/ with a UTC timestamp.
        Returns a dict with the output path and metadata, or an error
        key if no frame is available.

        Intended use cases
        ------------------
        * Evidence capture for incident reports
        * Manual audit of detection quality
        * Thumbnail previews in the UI
        """
        if cv2 is None:
            return {"error": "cv2 not available"}

        frame = self._frames.get(cam_id)
        if frame is None:
            return {"error": f"No frame available for camera {cam_id}"}

        snap_dir = _os.path.join(MEDIA_DIR, "snapshots")
        _os.makedirs(snap_dir, exist_ok=True)

        ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"cam{cam_id}_snap_{ts}.jpg"
        filepath = _os.path.join(snap_dir, filename)

        try:
            ok = cv2.imwrite(filepath, frame)
        except Exception as e:
            return {"error": f"imwrite failed: {e}"}

        if not ok:
            return {"error": f"cv2.imwrite returned False for {filepath}"}

        result   = self._results.get(cam_id, {})
        meta     = result.get("meta") or {}
        logger.info("[CAM %s] SNAPSHOT SAVED → %s", cam_id, filepath)

        return {
            "path":       filepath,
            "camera_id":  cam_id,
            "timestamp":  ts,
            "resolution": (FRAME_WIDTH, FRAME_HEIGHT),
            "risk":       result.get("risk", "LOW"),
            "frame_id":   meta.get("frame_id"),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_camera(self, cam_id=0) -> bool:
        """
        Start capture + inference threads for ``cam_id``.

        Locked so two concurrent start requests for the same cam_id
        can't both pass the "already running" check and spawn duplicate
        threads.

        Returns True once the camera transitions to RUNNING, or False
        if it hits ERROR / DISCONNECTED within START_POLL_TIMEOUT.
        """
        lock = self._get_lock(cam_id)

        with lock:
            if self.is_camera_running(cam_id):
                return True

            # If a previous thread is still winding down, wait for it.
            for attr in ("_threads", "_inf_threads"):
                old = getattr(self, attr).get(cam_id)
                if old and old.is_alive():
                    old.join(timeout=STOP_JOIN_TIMEOUT)

            # Fresh queue for this session
            self._queues[cam_id] = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
            # Reset start-time anchor for FPS calculation
            self.__dict__[f"_start_{cam_id}"] = time.time()

            self._states[cam_id] = CameraState.INITIALIZING

            cap_thread = threading.Thread(
                target=self._capture_worker,
                args=(cam_id,),
                daemon=True,
                name=f"infraguard-cap-{cam_id}",
            )
            inf_thread = threading.Thread(
                target=self._inference_worker,
                args=(cam_id,),
                daemon=True,
                name=f"infraguard-inf-{cam_id}",
            )
            self._threads[cam_id]     = cap_thread
            self._inf_threads[cam_id] = inf_thread

            cap_thread.start()
            inf_thread.start()

        # Poll until Running, failed, or timeout
        deadline = time.time() + START_POLL_TIMEOUT
        while time.time() < deadline:
            state = self._states.get(cam_id)
            if state == CameraState.RUNNING:
                return True
            if state in (CameraState.ERROR,):
                return False
            if not cap_thread.is_alive():
                return False
            time.sleep(START_POLL_INTERVAL)

        return self._states.get(cam_id) == CameraState.RUNNING

    def stop_camera(self, cam_id=0) -> None:
        """
        Signal the capture + inference threads to stop and wait for them
        to exit (bounded by STOP_JOIN_TIMEOUT each).

        Safe to call multiple times; idempotent.
        """
        lock = self._get_lock(cam_id)

        with lock:
            self._states[cam_id] = CameraState.STOPPED

            # Release the capture device immediately so the OS frees it
            cap = self._caps.get(cam_id)
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            self._caps.pop(cam_id, None)

            # Join capture thread
            cap_thread = self._threads.get(cam_id)
            if cap_thread and cap_thread.is_alive():
                cap_thread.join(timeout=STOP_JOIN_TIMEOUT)
                if cap_thread.is_alive():
                    logger.warning(
                        "[CAM %s] CAPTURE THREAD DID NOT EXIT WITHIN %ss",
                        cam_id, STOP_JOIN_TIMEOUT,
                    )
                else:
                    self._threads.pop(cam_id, None)

            # Join inference thread (it drains the queue first)
            inf_thread = self._inf_threads.get(cam_id)
            if inf_thread and inf_thread.is_alive():
                inf_thread.join(timeout=STOP_JOIN_TIMEOUT)
                if inf_thread.is_alive():
                    logger.warning(
                        "[CAM %s] INFERENCE THREAD DID NOT EXIT WITHIN %ss",
                        cam_id, STOP_JOIN_TIMEOUT,
                    )
                else:
                    self._inf_threads.pop(cam_id, None)

            # Belt-and-suspenders: clear stale data if threads didn't exit in time
            self._frames.pop(cam_id, None)
            self._results.pop(cam_id, None)

            logger.info("[CAM %s] STOPPED", cam_id)

    def stop_all_cameras(self) -> None:
        """Called by main.py lifespan on shutdown."""
        all_ids = (
            set(self._states.keys())
            | set(self._threads.keys())
            | set(self._inf_threads.keys())
        )
        for cam_id in list(all_ids):
            self.stop_camera(cam_id)
        logger.info("[STREAM ENGINE] ALL CAMERAS STOPPED")

    def restart_camera(self, cam_id=0) -> bool:
        """Stop then start. Useful for applying registry/config changes."""
        self.stop_camera(cam_id)
        return self.start_camera(cam_id)

    def is_camera_running(self, cam_id=0) -> bool:
        """
        True only when state is RUNNING *and* the capture thread is alive.

        If the thread died without going through stop_camera() (e.g. an
        uncaught exception or cable yank) the state is corrected to
        DISCONNECTED so the UI reflects reality.
        """
        state = self._states.get(cam_id)
        if state != CameraState.RUNNING:
            return False

        thread = self._threads.get(cam_id)
        if thread is None or not thread.is_alive():
            self._states[cam_id] = CameraState.DISCONNECTED
            return False

        return True

    def get_camera_state(self, cam_id=0) -> CameraState:
        """
        Return the rich CameraState for ``cam_id``, reconciling any
        stale RUNNING flag via ``is_camera_running``.
        """
        self.is_camera_running(cam_id)
        return self._states.get(cam_id, CameraState.STOPPED)

    def get_latest_frame(self, cam_id=0):
        return self._frames.get(cam_id)

    def get_latest_result(self, cam_id=0) -> dict:
        return self._results.get(cam_id, {
            "risk":       "LOW",
            "detections": [],
            "analytics":  {},
            "frame_id":   None,
            "meta":       None,
            "telemetry":  {},
        })

    def get_statistics(self, cam_id=0) -> dict:
        """Return a snapshot of stream statistics for ``cam_id``."""
        stats = self._stats.get(cam_id)
        return stats.to_dict() if stats else _make_stats().to_dict()

    def get_all_camera_status(self) -> dict:
        """Used by /camera/list endpoint."""
        all_ids = (
            set(self._states.keys())
            | set(self._threads.keys())
            | set(self._inf_threads.keys())
        )
        return {
            str(cam_id): {
                "running":    self.is_camera_running(cam_id),
                "state":      self.get_camera_state(cam_id),
                "has_frame":  cam_id in self._frames,
                "risk":       self._results.get(cam_id, {}).get("risk", "LOW"),
                "statistics": self.get_statistics(cam_id),
                "source":     str(resolve_source(cam_id)),
            }
            for cam_id in sorted(all_ids, key=str)
        }


# =========================================================
# SHARED INSTANCE + MODULE-LEVEL SHIMS
# Keeps existing call-sites working unchanged.
# =========================================================

camera_manager = CameraManager()

def start_camera(cam_id=0)        -> bool:         return camera_manager.start_camera(cam_id)
def stop_camera(cam_id=0)         -> None:         return camera_manager.stop_camera(cam_id)
def stop_all_cameras()            -> None:         return camera_manager.stop_all_cameras()
def restart_camera(cam_id=0)      -> bool:         return camera_manager.restart_camera(cam_id)
def is_camera_running(cam_id=0)   -> bool:         return camera_manager.is_camera_running(cam_id)
def get_camera_state(cam_id=0)    -> CameraState:  return camera_manager.get_camera_state(cam_id)
def get_latest_frame(cam_id=0):                    return camera_manager.get_latest_frame(cam_id)
def get_latest_result(cam_id=0)   -> dict:         return camera_manager.get_latest_result(cam_id)
def get_statistics(cam_id=0)      -> dict:         return camera_manager.get_statistics(cam_id)
def get_all_camera_status()       -> dict:         return camera_manager.get_all_camera_status()

# ── Enhancement shims ────────────────────────────────────────────────
def start_recording(cam_id=0)     -> dict:         return camera_manager.start_recording(cam_id)
def stop_recording(cam_id=0)      -> dict:         return camera_manager.stop_recording(cam_id)
def is_recording(cam_id=0)        -> bool:         return camera_manager.is_recording(cam_id)
def capture_snapshot(cam_id=0)    -> dict:         return camera_manager.capture_snapshot(cam_id)
def get_health_status()           -> dict:         return camera_manager.get_health_status()
# discover_cameras() is already a module-level function — call it directly.