import threading
import time
import asyncio
from datetime import datetime, timezone

try:
    import cv2
except Exception:
    cv2 = None

from backend.services.safety.detection_service import process_frame
from backend.core.websocket_manager import manager

# =========================================================
# GLOBALS
# =========================================================

CAMERA_CAPS    : dict = {}
CAMERA_THREADS : dict = {}
CAMERA_RUNNING : dict = {}
LATEST_FRAMES  : dict = {}
LATEST_RESULTS : dict = {}

# FIX (#15): guards start_camera/stop_camera against concurrent
# duplicate requests racing on the same cam_id.
_CAMERA_LOCKS      : dict = {}
_CAMERA_LOCKS_GUARD = threading.Lock()

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


def _get_camera_lock(cam_id):
    with _CAMERA_LOCKS_GUARD:
        lock = _CAMERA_LOCKS.get(cam_id)
        if lock is None:
            lock = threading.Lock()
            _CAMERA_LOCKS[cam_id] = lock
        return lock


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
        print(f"[CAM {cam_id}] BROADCAST FAILED:", e)


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
        print("[OPENCV NOT INSTALLED]")
        return None

    try:
        cap = cv2.VideoCapture(cam_id)

        if not cap.isOpened():
            print(f"[CAM {cam_id}] FAILED TO OPEN")
            return None

        ret, frame = cap.read()

        if not ret:
            print(f"[CAM {cam_id}] FIRST FRAME READ FAILED")
            cap.release()
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

        return cap

    except Exception as e:
        print(f"[CAM {cam_id}] CAMERA OPEN ERROR:", e)
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
# CAMERA WORKER THREAD
# =========================================================

def camera_worker(cam_id):

    cap = open_camera(cam_id)

    if cap is None:
        CAMERA_RUNNING[cam_id] = False
        print(f"[CAM {cam_id}] UNAVAILABLE — thread exiting")
        return

    CAMERA_CAPS[cam_id]    = cap
    CAMERA_RUNNING[cam_id] = True
    print(f"[CAM {cam_id}] STREAM STARTED")

    frame_counter = 0
    start_time    = time.time()
    interval      = 1.0 / TARGET_FPS

    while CAMERA_RUNNING.get(cam_id, False):

        tick = time.time()

        ok, frame = cap.read()

        if not ok:
            print(f"[CAM {cam_id}] FRAME READ FAILED")
            time.sleep(0.05)
            continue

        if frame is None:
            print(f"[CAM {cam_id}] FRAME IS NONE")
            continue

        try:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            try:
                result = process_frame(frame)
            except Exception:
                import traceback
                print(f"[CAM {cam_id}] PROCESS_FRAME CRASHED:")
                traceback.print_exc()
                result = {
                    "risk": "LOW",
                    "detections": [],
                    "analytics": {},
                }

            frame_counter += 1
            elapsed = time.time() - start_time
            fps     = round(frame_counter / elapsed, 1)

            telemetry = build_telemetry(
                cam_id, fps,
                result.get("risk", "LOW")
            )

            LATEST_FRAMES[cam_id]  = frame
            LATEST_RESULTS[cam_id] = {**result, "telemetry": telemetry}

            _broadcast(cam_id, {
                "type":       "frame_result",
                "camera_id":  cam_id,
                "risk":       result.get("risk", "LOW"),
                "detections": result.get("detections", []),
                "analytics":  result.get("analytics", {}),
                "telemetry":  telemetry,
            })

        except Exception as e:
            import traceback
            print(f"[CAM {cam_id}] PROCESS ERROR:")
            traceback.print_exc()

        elapsed_frame = time.time() - tick
        sleep_for     = interval - elapsed_frame
        if sleep_for > 0:
            time.sleep(sleep_for)

    try:
        cap.release()
    except Exception:
        pass

    CAMERA_CAPS.pop(cam_id, None)
    CAMERA_RUNNING[cam_id] = False

    # FIX (#7, #8): drop cached frame/result so a stopped camera doesn't
    # keep serving stale data or holding memory indefinitely.
    LATEST_FRAMES.pop(cam_id, None)
    LATEST_RESULTS.pop(cam_id, None)

    print(f"[CAM {cam_id}] STOPPED")


# =========================================================
# PUBLIC API
# =========================================================

def start_camera(cam_id=0):
    """
    FIX (#15): locked so two concurrent start requests for the same
    cam_id can't both pass the "already running" check and spawn
    duplicate worker threads.

    FIX (#10): replaces the blocking time.sleep(1) with a bounded
    poll loop, returning as soon as the worker thread confirms the
    camera opened (or bailing out after START_POLL_TIMEOUT).
    """
    lock = _get_camera_lock(cam_id)

    with lock:
        if is_camera_running(cam_id):
            return True

        # FIX (#6): if a previous thread for this cam_id is still
        # winding down, wait for it to fully exit before starting a
        # new one, so we never have two threads touching the same
        # cam_id's globals at once.
        old_thread = CAMERA_THREADS.get(cam_id)
        if old_thread and old_thread.is_alive():
            old_thread.join(timeout=STOP_JOIN_TIMEOUT)

        thread = threading.Thread(
            target=camera_worker,
            args=(cam_id,),
            daemon=True,
            name=f"infraguard-cam-{cam_id}",
        )
        CAMERA_THREADS[cam_id] = thread
        thread.start()

    deadline = time.time() + START_POLL_TIMEOUT
    while time.time() < deadline:
        if CAMERA_RUNNING.get(cam_id, False):
            return True
        if not thread.is_alive():
            # Worker exited early (e.g. camera failed to open).
            return False
        time.sleep(START_POLL_INTERVAL)

    return CAMERA_RUNNING.get(cam_id, False)


def stop_camera(cam_id=0):
    """
    FIX (#15): locked alongside start_camera so a stop can't race a
    start for the same cam_id.

    FIX (#6): joins the worker thread (bounded by STOP_JOIN_TIMEOUT)
    instead of firing-and-forgetting, so callers/the caller's caller
    (e.g. the API route) know the camera has actually released its
    resources before responding, and thread references don't pile up.
    """
    lock = _get_camera_lock(cam_id)

    with lock:
        CAMERA_RUNNING[cam_id] = False

        cap = CAMERA_CAPS.get(cam_id)
        if cap:
            try:
                cap.release()
            except Exception:
                pass
        CAMERA_CAPS.pop(cam_id, None)

        thread = CAMERA_THREADS.get(cam_id)
        if thread:
            thread.join(timeout=STOP_JOIN_TIMEOUT)
            if thread.is_alive():
                print(f"[CAM {cam_id}] WARNING: worker thread did not "
                      f"exit within {STOP_JOIN_TIMEOUT}s")
            else:
                CAMERA_THREADS.pop(cam_id, None)

        # Belt-and-suspenders: camera_worker() already clears these on
        # exit, but if the thread didn't exit in time, clear the cache
        # now so no stale frame/result is served while it winds down.
        LATEST_FRAMES.pop(cam_id, None)
        LATEST_RESULTS.pop(cam_id, None)

        print(f"[CAM {cam_id}] STOP REQUESTED")


def stop_all_cameras():
    """Called by main.py lifespan on shutdown."""
    for cam_id in list(CAMERA_RUNNING.keys()):
        stop_camera(cam_id)
    print("[STREAM ENGINE] ALL CAMERAS STOPPED")


def is_camera_running(cam_id=0):
    """
    FIX (#9): a stale CAMERA_RUNNING[cam_id] = True flag isn't enough —
    if the worker thread died unexpectedly (uncaught exception, camera
    yanked, etc.) without going through stop_camera(), the flag could
    still say True. Cross-check against actual thread liveness.
    """
    running_flag = CAMERA_RUNNING.get(cam_id, False)
    if not running_flag:
        return False

    thread = CAMERA_THREADS.get(cam_id)
    if thread is None or not thread.is_alive():
        # Thread died without cleaning up — reflect that immediately.
        CAMERA_RUNNING[cam_id] = False
        return False

    return True


def get_latest_frame(cam_id=0):
    return LATEST_FRAMES.get(cam_id)


def get_latest_result(cam_id=0):
    return LATEST_RESULTS.get(cam_id, {
        "risk":       "LOW",
        "detections": [],
        "analytics":  {},
        "telemetry":  {},
    })


def get_all_camera_status():
    """Used by /camera/list endpoint."""
    all_ids = set(CAMERA_RUNNING.keys()) | set(CAMERA_THREADS.keys())
    return {
        str(cam_id): {
            "running":   is_camera_running(cam_id),
            "has_frame": cam_id in LATEST_FRAMES,
            "risk":      LATEST_RESULTS.get(cam_id, {}).get("risk", "LOW"),
        }
        for cam_id in sorted(all_ids)
    }