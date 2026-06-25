import threading
import time
import asyncio
import logging
from datetime import datetime, timezone

try:
    import cv2
except Exception:
    cv2 = None

from backend.services.safety.detection_service import process_frame
from backend.core.websocket_manager import manager

logger = logging.getLogger(__name__)

# =========================================================
# CONSTANTS
# =========================================================

FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
TARGET_FPS   = 24

# How long start_camera() will poll for the worker thread to confirm
# the camera opened, before giving up. Replaces the old blocking
# time.sleep(1) (#10) with a bounded, responsive poll loop.
START_POLL_INTERVAL = 0.1
START_POLL_TIMEOUT  = 5.0

# How long stop_camera() will wait for the worker thread to actually
# exit before returning (#6 — thread never joined).
STOP_JOIN_TIMEOUT = 3.0


# =========================================================
# SHARED BACKGROUND EVENT LOOP FOR WS BROADCASTS
# FIX: one loop lives forever in a daemon thread.
#      camera_worker() posts coroutines into it safely.
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
    FIX (#16): retrieve the exception from the future so it doesn't
    get silently swallowed (asyncio logs 'exception never retrieved'
    warnings, and real failures go unnoticed otherwise).
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

    if cv2 is None:
        logger.error("[OPENCV NOT INSTALLED]")
        return None

    try:
        cap = cv2.VideoCapture(cam_id)

        if not cap.isOpened():
            logger.error("[CAM %s] FAILED TO OPEN", cam_id)
            return None

        ret, frame = cap.read()

        if not ret:
            logger.error("[CAM %s] FIRST FRAME READ FAILED", cam_id)
            cap.release()
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

        return cap

    except Exception as e:
        logger.exception("[CAM %s] CAMERA OPEN ERROR: %s", cam_id, e)
        return None


# =========================================================
# TELEMETRY BUILDER
# =========================================================

def build_telemetry(cam_id, fps, risk):
    return {
        "camera_id": cam_id,
        "fps":       fps,
        "risk":      risk,
        "status":    "ACTIVE",
        "engine":    "InfraGuard Enterprise AI",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =========================================================
# CAMERA MANAGER
# =========================================================

class CameraManager:
    """
    Encapsulates all per-camera state that was previously spread across
    module-level dicts. A single shared instance (``camera_manager``) is
    exported for use by the public API functions and the rest of the app.
    """

    def __init__(self):
        self._caps:    dict = {}
        self._threads: dict = {}
        self._running: dict = {}
        self._frames:  dict = {}
        self._results: dict = {}

        # FIX (#15): guards start_camera/stop_camera against concurrent
        # duplicate requests racing on the same cam_id.
        self._locks:       dict = {}
        self._locks_guard: threading.Lock = threading.Lock()

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

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker(self, cam_id):
        cap = open_camera(cam_id)

        if cap is None:
            self._running[cam_id] = False
            logger.warning("[CAM %s] UNAVAILABLE — thread exiting", cam_id)
            return

        self._caps[cam_id]    = cap
        self._running[cam_id] = True
        logger.info("[CAM %s] STREAM STARTED", cam_id)

        frame_counter = 0
        start_time    = time.time()
        interval      = 1.0 / TARGET_FPS

        while self._running.get(cam_id, False):

            tick = time.time()

            ok, frame = cap.read()

            if not ok:
                logger.warning("[CAM %s] FRAME READ FAILED", cam_id)
                time.sleep(0.05)
                continue

            if frame is None:
                logger.warning("[CAM %s] FRAME IS NONE", cam_id)
                continue

            try:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

                try:
                    result = process_frame(frame)
                except Exception:
                    logger.exception("[CAM %s] PROCESS_FRAME CRASHED", cam_id)
                    result = {
                        "risk":       "LOW",
                        "detections": [],
                        "analytics":  {},
                    }

                frame_counter += 1
                elapsed = time.time() - start_time
                fps     = round(frame_counter / elapsed, 1)

                telemetry = build_telemetry(
                    cam_id, fps,
                    result.get("risk", "LOW")
                )

                self._frames[cam_id]  = frame
                self._results[cam_id] = {**result, "telemetry": telemetry}

                _broadcast(cam_id, {
                    "type":       "frame_result",
                    "camera_id":  cam_id,
                    "risk":       result.get("risk", "LOW"),
                    "detections": result.get("detections", []),
                    "analytics":  result.get("analytics", {}),
                    "telemetry":  telemetry,
                })

            except Exception:
                logger.exception("[CAM %s] PROCESS ERROR", cam_id)

            elapsed_frame = time.time() - tick
            sleep_for     = interval - elapsed_frame
            if sleep_for > 0:
                time.sleep(sleep_for)

        try:
            cap.release()
        except Exception:
            pass

        self._caps.pop(cam_id, None)
        self._running[cam_id] = False

        # FIX (#7, #8): drop cached frame/result so a stopped camera doesn't
        # keep serving stale data or holding memory indefinitely.
        self._frames.pop(cam_id, None)
        self._results.pop(cam_id, None)

        logger.info("[CAM %s] STOPPED", cam_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_camera(self, cam_id=0) -> bool:
        """
        FIX (#15): locked so two concurrent start requests for the same
        cam_id can't both pass the "already running" check and spawn
        duplicate worker threads.

        FIX (#10): replaces the blocking time.sleep(1) with a bounded
        poll loop, returning as soon as the worker thread confirms the
        camera opened (or bailing out after START_POLL_TIMEOUT).
        """
        lock = self._get_lock(cam_id)

        with lock:
            if self.is_camera_running(cam_id):
                return True

            # FIX (#6): if a previous thread for this cam_id is still
            # winding down, wait for it to fully exit before starting a
            # new one, so we never have two threads touching the same
            # cam_id's state at once.
            old_thread = self._threads.get(cam_id)
            if old_thread and old_thread.is_alive():
                old_thread.join(timeout=STOP_JOIN_TIMEOUT)

            thread = threading.Thread(
                target=self._worker,
                args=(cam_id,),
                daemon=True,
                name=f"infraguard-cam-{cam_id}",
            )
            self._threads[cam_id] = thread
            thread.start()

        deadline = time.time() + START_POLL_TIMEOUT
        while time.time() < deadline:
            if self._running.get(cam_id, False):
                return True
            if not thread.is_alive():
                # Worker exited early (e.g. camera failed to open).
                return False
            time.sleep(START_POLL_INTERVAL)

        return self._running.get(cam_id, False)

    def stop_camera(self, cam_id=0) -> None:
        """
        FIX (#15): locked alongside start_camera so a stop can't race a
        start for the same cam_id.

        FIX (#6): joins the worker thread (bounded by STOP_JOIN_TIMEOUT)
        instead of firing-and-forgetting, so callers know the camera has
        actually released its resources before responding, and thread
        references don't pile up.
        """
        lock = self._get_lock(cam_id)

        with lock:
            self._running[cam_id] = False

            cap = self._caps.get(cam_id)
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            self._caps.pop(cam_id, None)

            thread = self._threads.get(cam_id)
            if thread:
                thread.join(timeout=STOP_JOIN_TIMEOUT)
                if thread.is_alive():
                    logger.warning(
                        "[CAM %s] WARNING: worker thread did not exit within %ss",
                        cam_id, STOP_JOIN_TIMEOUT,
                    )
                else:
                    self._threads.pop(cam_id, None)

            # Belt-and-suspenders: _worker() already clears these on
            # exit, but if the thread didn't exit in time, clear the
            # cache now so no stale frame/result is served.
            self._frames.pop(cam_id, None)
            self._results.pop(cam_id, None)

            logger.info("[CAM %s] STOP REQUESTED", cam_id)

    def stop_all_cameras(self) -> None:
        """Called by main.py lifespan on shutdown."""
        for cam_id in list(self._running.keys()):
            self.stop_camera(cam_id)
        logger.info("[STREAM ENGINE] ALL CAMERAS STOPPED")

    def is_camera_running(self, cam_id=0) -> bool:
        """
        FIX (#9): a stale _running[cam_id] = True flag isn't enough —
        if the worker thread died unexpectedly (uncaught exception, camera
        yanked, etc.) without going through stop_camera(), the flag could
        still say True. Cross-check against actual thread liveness.
        """
        running_flag = self._running.get(cam_id, False)
        if not running_flag:
            return False

        thread = self._threads.get(cam_id)
        if thread is None or not thread.is_alive():
            # Thread died without cleaning up — reflect that immediately.
            self._running[cam_id] = False
            return False

        return True

    def get_latest_frame(self, cam_id=0):
        return self._frames.get(cam_id)

    def get_latest_result(self, cam_id=0) -> dict:
        return self._results.get(cam_id, {
            "risk":       "LOW",
            "detections": [],
            "analytics":  {},
            "telemetry":  {},
        })

    def get_all_camera_status(self) -> dict:
        """Used by /camera/list endpoint."""
        all_ids = set(self._running.keys()) | set(self._threads.keys())
        return {
            str(cam_id): {
                "running":   self.is_camera_running(cam_id),
                "has_frame": cam_id in self._frames,
                "risk":      self._results.get(cam_id, {}).get("risk", "LOW"),
            }
            for cam_id in sorted(all_ids)
        }


# =========================================================
# SHARED INSTANCE + MODULE-LEVEL SHIMS
# (keeps the existing call-sites working unchanged)
# =========================================================

camera_manager = CameraManager()

def start_camera(cam_id=0)        -> bool:  return camera_manager.start_camera(cam_id)
def stop_camera(cam_id=0)         -> None:  return camera_manager.stop_camera(cam_id)
def stop_all_cameras()            -> None:  return camera_manager.stop_all_cameras()
def is_camera_running(cam_id=0)   -> bool:  return camera_manager.is_camera_running(cam_id)
def get_latest_frame(cam_id=0):             return camera_manager.get_latest_frame(cam_id)
def get_latest_result(cam_id=0)   -> dict:  return camera_manager.get_latest_result(cam_id)
def get_all_camera_status()       -> dict:  return camera_manager.get_all_camera_status()