

import asyncio
import base64
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    Query,
)
from fastapi.responses import StreamingResponse

from backend.services.safety.detection_service import (
    process_frame
)

from backend.services.safety.stream_service import (
    start_camera,
    stop_camera,
    is_camera_running,
    get_latest_frame
)

from backend.core.websocket_manager import (
    manager
)

from backend.api.activity_routes import (
    add_alert
)

router = APIRouter()

# =========================================
# LOGGING  (Improvement 9 — API Logging)
# =========================================
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    ))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# =========================================
# CONFIG
# =========================================

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp",
}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB

PIPELINE_VERSION = "routes-2.0.0"
MODEL_LABEL = "InfraGuard + Crack (dual-model)"


# =========================================
# ERROR CATEGORIES  (Improvement 6)
# =========================================

class ApiError(Exception):
    status_code = 500
    category = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ValidationError(ApiError):
    status_code = 400
    category = "validation_error"


class InferenceError(ApiError):
    status_code = 502
    category = "inference_error"


class PipelineError(ApiError):
    status_code = 502
    category = "pipeline_error"


class SerializationError(ApiError):
    status_code = 500
    category = "serialization_error"


# =========================================
# REQUEST ID  (Improvement 2)
# =========================================

def _new_request_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"REQ-{stamp}-{uuid.uuid4().hex[:8]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# =========================================
# METRICS  (Improvement 11)
# =========================================

_metrics: Dict[str, Any] = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_processing_ms": 0.0,
    "total_inference_ms": 0.0,
    "errors_by_category": {},
}


def _record_metrics(success: bool, processing_ms: float, inference_ms: float,
                     error_category: Optional[str] = None) -> None:
    _metrics["total_requests"] += 1
    if success:
        _metrics["successful_requests"] += 1
    else:
        _metrics["failed_requests"] += 1
        if error_category:
            _metrics["errors_by_category"][error_category] = (
                _metrics["errors_by_category"].get(error_category, 0) + 1
            )
    _metrics["total_processing_ms"] += processing_ms
    _metrics["total_inference_ms"] += inference_ms


def _metrics_snapshot() -> Dict[str, Any]:
    total = _metrics["total_requests"]
    avg_processing = round(_metrics["total_processing_ms"] / total, 2) if total else 0.0
    avg_inference = round(_metrics["total_inference_ms"] / total, 2) if total else 0.0
    return {
        "total_requests": total,
        "successful_requests": _metrics["successful_requests"],
        "failed_requests": _metrics["failed_requests"],
        "success_rate": (
            round(_metrics["successful_requests"] / total, 4) if total else 0.0
        ),
        "avg_processing_ms": avg_processing,
        "avg_inference_ms": avg_inference,
        "errors_by_category": dict(_metrics["errors_by_category"]),
    }


# =========================================
# IMAGE DECODER
# =========================================

def decode_image(contents: bytes):
    arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Invalid image")

    return frame


# =========================================
# IMAGE ENCODER
# =========================================

def encode_image(frame):
    ret, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 85]
    )

    if not ret:
        raise ValueError("Image encoding failed")

    return base64.b64encode(buffer.tobytes()).decode("utf-8")


# =========================================
# REQUEST VALIDATION  (Improvement 1)
# =========================================

def _validate_upload(file: UploadFile, contents: bytes, request_id: str) -> None:
    if file is None or not file.filename:
        raise ValidationError("No file was uploaded.")

    if not contents:
        raise ValidationError("Uploaded file is empty.")

    if len(contents) > MAX_UPLOAD_BYTES:
        raise ValidationError(
            f"Uploaded file is {len(contents)} bytes, which exceeds the "
            f"{MAX_UPLOAD_BYTES} byte limit."
        )

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file extension '.{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}."
        )

    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.info(
            "[%s] unexpected content-type '%s' for filename '%s' — "
            "proceeding, decode step will be the real gate.",
            request_id, file.content_type, file.filename,
        )


# =========================================
# RESPONSE VALIDATION  (Improvement 4)
# =========================================

def _validate_service_result(result: Any, request_id: str) -> Dict[str, Any]:
    if result is None:
        raise PipelineError("Detection service returned no result (None).")

    if not isinstance(result, dict):
        raise PipelineError(
            f"Detection service returned {type(result).__name__}, expected dict."
        )

    if "detections" in result and not isinstance(result["detections"], list):
        raise PipelineError(
            f"Detection service 'detections' field was "
            f"{type(result['detections']).__name__}, expected list."
        )

    if "analytics" in result and not isinstance(result["analytics"], dict):
        raise PipelineError(
            f"Detection service 'analytics' field was "
            f"{type(result['analytics']).__name__}, expected dict."
        )

    logger.info(
        "[%s] response validation passed: %d detection(s), analytics=%s",
        request_id, len(result.get("detections", []) or []),
        "present" if result.get("analytics") else "empty",
    )
    return result


# =========================================
# DETECTION SUMMARY  (Improvement 8)
# =========================================

def _summarize_detections(detections: List[dict]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for det in detections or []:
        cls = (det.get("class_name") or "unknown").lower() if isinstance(det, dict) else "unknown"
        summary[cls] = summary.get(cls, 0) + 1
    return summary


# =========================================
# STANDARD RESPONSE SCHEMA  (Improvement 5)
# =========================================

def _build_response(
    *,
    request_id: str,
    started_at: float,
    extra: Dict[str, Any],
    warnings: Optional[List[str]] = None,
    timing_ms: Optional[Dict[str, float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = time.time()
    envelope = {
        "success": True,
        "request_id": request_id,
        "timestamp": _now_iso(),
        "warnings": warnings or [],
        "errors": [],
        "timing_ms": timing_ms or {},
        "metadata": metadata or {},
    }
    envelope.update(extra)
    envelope["timing_ms"]["total_request_ms"] = round((now - started_at) * 1000, 2)
    return envelope


def _build_error_response(
    *,
    request_id: str,
    error: ApiError,
) -> Dict[str, Any]:
    return {
        "success": False,
        "request_id": request_id,
        "timestamp": _now_iso(),
        "error_category": error.category,
        "errors": [error.message],
        "warnings": [],
    }


def _raise_http(request_id: str, error: ApiError) -> None:
    logger.warning(
        "[%s] request failed: category=%s message=%s",
        request_id, error.category, error.message,
    )
    raise HTTPException(
        status_code=error.status_code,
        detail=_build_error_response(request_id=request_id, error=error),
    )


# =========================================
# STREAM GENERATOR
# =========================================

def generate_stream(
    request: Request,
    cam_id=0
):

    logger.info("[stream %s] connected", cam_id)

    try:

        while True:

            if request.client is None:
                break

            if not is_camera_running(cam_id):
                break

            frame = get_latest_frame(cam_id)

            if frame is None:
                time.sleep(0.03)
                continue

            ret, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 72]
            )

            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes +
                b"\r\n"
            )

            time.sleep(0.03)

    except GeneratorExit:
        logger.info("[stream %s] closed", cam_id)

    except Exception as e:
        logger.exception("[stream %s] error: %s", cam_id, e)


# =========================================
# LIVE FEED
# =========================================

@router.get(
    "/camera/feed",
    summary="Live MJPEG camera feed",
    description=(
        "Streams a multipart/x-mixed-replace MJPEG feed for an already-"
        "running camera. Does NOT start the camera — call /camera/start "
        "first. Returns 404 if the camera is not currently running."
    ),
    response_description="multipart/x-mixed-replace JPEG stream",
)
async def camera_feed(
    request: Request,
    cam_id: int = 0
):
    if not is_camera_running(cam_id):
        raise HTTPException(
            status_code=404,
            detail=f"Camera {cam_id} is not running"
        )

    return StreamingResponse(
        generate_stream(request, cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# =========================================
# ENTERPRISE WEBSOCKET
# =========================================

@router.websocket("/ws/{cam_id}")
async def websocket_stream(
    websocket: WebSocket,
    cam_id: int
):

    await manager.connect(cam_id, websocket)
    logger.info("[ws %s] connected", cam_id)

    try:

        while True:
            await asyncio.sleep(30)

    except WebSocketDisconnect:
        logger.info("[ws %s] disconnected", cam_id)
        manager.disconnect(websocket)

    except Exception as e:
        logger.exception("[ws %s] error: %s", cam_id, e)
        manager.disconnect(websocket)


# =========================================
# DETECT IMAGE  (lightweight)
# =========================================

@router.post(
    "/detect",
    summary="Lightweight single-frame detection",
    description=(
        "Upload a single image (jpg/jpeg/png/bmp/webp, max "
        f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB) and run it through the "
        "safety detection pipeline. This is the lightweight endpoint — "
        "it does NOT encode/return an annotated image, to keep per-"
        "request cost low for polling or streaming callers. Use "
        "/detect-full if you need the annotated image and richer "
        "analytics payload.\n\n"
        "Always returns the standard response envelope "
        "(success, request_id, timestamp, timing_ms, metadata, "
        "warnings, errors) plus detect-specific fields "
        "(risk, detections, summary, findings, recommendations, "
        "analytics, telemetry, ai_metadata, annotated_image)."
    ),
    response_description="Standard envelope + lightweight detection payload",
)
async def detect(
    file: UploadFile = File(...),
    debug: bool = Query(
        False,
        description="If true, include a 'health' block in the response (Improvement 10).",
    ),
):
    request_id = _new_request_id()
    started_at = time.time()
    logger.info("[%s] /detect received file=%s", request_id, file.filename)

    inference_ms = 0.0

    try:
        contents = await file.read()

        _validate_upload(file, contents, request_id)

        try:
            frame = decode_image(contents)
        except ValueError as ve:
            raise ValidationError(str(ve))

        logger.info("[%s] image validated and decoded", request_id)

        predict_start = time.time()
        try:
            result = process_frame(frame)
        except Exception as ie:
            logger.exception("[%s] detection service raised", request_id)
            raise InferenceError(f"Detection service failed: {ie}")
        inference_ms = round((time.time() - predict_start) * 1000, 2)

        result = _validate_service_result(result, request_id)

        logger.info("[%s] pipeline completed in %.2f ms", request_id, inference_ms)

        risk = (result.get("risk") or "LOW").lower()
        detections = result.get("detections", []) or []
        analytics = result.get("analytics", {}) or {}

        missing_helmet = any(
            isinstance(d, dict) and d.get("has_helmet") is False for d in detections
        )
        missing_vest = any(
            isinstance(d, dict) and d.get("has_vest") is False for d in detections
        )

        try:
            if missing_helmet and missing_vest:
                add_alert("Helmet and Vest Missing", risk, 0)
            elif missing_helmet:
                add_alert("Helmet Missing", risk, 0)
            elif missing_vest:
                add_alert("Vest Missing", risk, 0)
            elif risk in ("high", "critical"):
                add_alert("High Risk Detected", risk, 0)
            else:
                add_alert("Worker Safe", "low", 0)
        except Exception:
            logger.exception("[%s] add_alert failed (non-fatal)", request_id)

        detection_summary = _summarize_detections(detections)

        response = _build_response(
            request_id=request_id,
            started_at=started_at,
            timing_ms={"inference_ms": inference_ms},
            metadata={
                "model": MODEL_LABEL,
                "pipeline_version": PIPELINE_VERSION,
                "camera": "upload",
                "image_size": f"{frame.shape[1]}x{frame.shape[0]}",
            },
            extra={
                "risk": risk,
                "detections": detections,
                "detection_summary": detection_summary,
                "summary": analytics.get("ppe_summary", {}),
                "findings": result.get("findings", []),
                "recommendations": analytics.get("recommendations", []),
                "analytics": analytics,
                "telemetry": result.get("telemetry", {}),
                "ai_metadata": result.get("ai_metadata", {}),
                "annotated_image": "",
            },
        )

        if debug:
            response["health"] = {
                "pipeline_status": "healthy",
                "model_loaded": True,
                "detection_count": len(detections),
            }

        processing_ms = response["timing_ms"]["total_request_ms"]
        _record_metrics(True, processing_ms, inference_ms)
        logger.info("[%s] /detect response sent (%.2f ms total)", request_id, processing_ms)
        return response

    except ApiError as ae:
        processing_ms = round((time.time() - started_at) * 1000, 2)
        _record_metrics(False, processing_ms, inference_ms, ae.category)
        _raise_http(request_id, ae)

    except Exception as e:
        logger.exception("[%s] unclassified error in /detect", request_id)
        processing_ms = round((time.time() - started_at) * 1000, 2)
        err = ApiError(str(e))
        _record_metrics(False, processing_ms, inference_ms, err.category)
        _raise_http(request_id, err)


# =========================================
# FULL DETECTION API
# =========================================

@router.post(
    "/detect-full",
    summary="Full single-frame detection with annotated image",
    description=(
        "Upload a single image and run it through the safety detection "
        "pipeline, returning the same standard envelope as /detect plus "
        "a base64-encoded annotated JPEG and the full analytics payload "
        "(ppe_summary, equipment, recommendations). Heavier per-request "
        "cost than /detect — use this when the caller actually needs the "
        "annotated image."
    ),
    response_description="Standard envelope + image + full analytics payload",
)
async def detect_full(
    file: UploadFile = File(...),
    debug: bool = Query(
        False,
        description="If true, include a 'health' block in the response (Improvement 10).",
    ),
):
    request_id = _new_request_id()
    started_at = time.time()
    logger.info("[%s] /detect-full received file=%s", request_id, file.filename)

    inference_ms = 0.0

    try:
        contents = await file.read()

        _validate_upload(file, contents, request_id)

        try:
            frame = decode_image(contents)
        except ValueError as ve:
            raise ValidationError(str(ve))

        logger.info("[%s] image validated and decoded", request_id)

        predict_start = time.time()
        try:
            result = process_frame(frame)
        except Exception as ie:
            logger.exception("[%s] detection service raised", request_id)
            raise InferenceError(f"Detection service failed: {ie}")
        inference_ms = round((time.time() - predict_start) * 1000, 2)

        result = _validate_service_result(result, request_id)

        logger.info("[%s] pipeline completed in %.2f ms", request_id, inference_ms)

        encode_start = time.time()
        try:
            annotated_image = encode_image(frame)
        except ValueError as se:
            raise SerializationError(str(se))
        encoding_ms = round((time.time() - encode_start) * 1000, 2)

        analytics = result.get("analytics", {}) or {}
        detections = result.get("detections", []) or []
        risk = result.get("risk", analytics.get("overall_risk", "LOW"))

        try:
            add_alert("Safety Analysis Complete", risk, 0)
        except Exception:
            logger.exception("[%s] add_alert failed (non-fatal)", request_id)

        detection_summary = _summarize_detections(detections)

        response = _build_response(
            request_id=request_id,
            started_at=started_at,
            timing_ms={
                "inference_ms": inference_ms,
                "encoding_ms": encoding_ms,
            },
            metadata={
                "model": MODEL_LABEL,
                "pipeline_version": PIPELINE_VERSION,
                "camera": "upload",
                "image_size": f"{frame.shape[1]}x{frame.shape[0]}",
            },
            extra={
                "image": annotated_image,
                "detections": detections,
                "detection_summary": detection_summary,
                "analytics": analytics,
                "ppe_summary": analytics.get("ppe_summary", {}),
                "equipment": analytics.get("equipment", []),
                "recommendations": analytics.get("recommendations", []),
                "risk": risk,
            },
        )

        if debug:
            response["health"] = {
                "pipeline_status": "healthy",
                "model_loaded": True,
                "detection_count": len(detections),
            }

        processing_ms = response["timing_ms"]["total_request_ms"]
        _record_metrics(True, processing_ms, inference_ms)
        logger.info("[%s] /detect-full response sent (%.2f ms total)", request_id, processing_ms)
        return response

    except ApiError as ae:
        processing_ms = round((time.time() - started_at) * 1000, 2)
        _record_metrics(False, processing_ms, inference_ms, ae.category)
        _raise_http(request_id, ae)

    except Exception as e:
        logger.exception("[%s] unclassified error in /detect-full", request_id)
        processing_ms = round((time.time() - started_at) * 1000, 2)
        err = ApiError(str(e))
        _record_metrics(False, processing_ms, inference_ms, err.category)
        _raise_http(request_id, err)


# =========================================
# START CAMERA
# =========================================

@router.get(
    "/camera/start",
    summary="Start a camera stream",
    description="Starts the camera identified by cam_id. Returns {'status': 'started'|'failed'}.",
)
def start(cam_id: int = 0):
    ok = start_camera(cam_id)
    logger.info("camera/start cam_id=%s ok=%s", cam_id, ok)
    return {"status": "started" if ok else "failed"}


# =========================================
# STOP CAMERA
# =========================================

@router.get(
    "/camera/stop",
    summary="Stop a camera stream",
    description="Stops the camera identified by cam_id. Returns {'status': 'stopped'}.",
)
def stop(cam_id: int = 0):
    stop_camera(cam_id)
    logger.info("camera/stop cam_id=%s", cam_id)
    return {"status": "stopped"}


# =========================================
# STATUS
# =========================================

@router.get(
    "/camera/status",
    summary="Check whether a camera is running",
    description="Returns {'running': bool} for the camera identified by cam_id.",
)
def status(cam_id: int = 0):
    return {"running": is_camera_running(cam_id)}


# =========================================
# METRICS  (Improvement 11)
# =========================================

@router.get(
    "/metrics",
    summary="Request/response metrics for this process",
    description=(
        "Lightweight in-memory counters: total/successful/failed "
        "requests, average processing/inference time, and a breakdown "
        "of failures by error category. Resets on process restart; "
        "not a substitute for a real metrics backend in a multi-"
        "process deployment, but useful for local debugging and "
        "lightweight monitoring."
    ),
)
def metrics():
    return _metrics_snapshot()