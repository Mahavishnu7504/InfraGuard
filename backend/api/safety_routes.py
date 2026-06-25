from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect
)

from fastapi.responses import StreamingResponse

import cv2
import numpy as np
import time
import asyncio
import base64

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
# IMAGE DECODER
# =========================================

def decode_image(
    contents: bytes
):

    arr = np.frombuffer(
        contents,
        np.uint8
    )

    frame = cv2.imdecode(
        arr,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        raise ValueError(
            "Invalid image"
        )

    return frame


# =========================================
# IMAGE ENCODER
# =========================================

def encode_image(frame):
    ret, buffer = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            85
        ]
    )

    if not ret:
        raise ValueError("Image encoding failed")

    return base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")


# =========================================
# STREAM GENERATOR
# =========================================

def generate_stream(
    request: Request,
    cam_id=0
):

    print(
        f"[STREAM {cam_id}] CONNECTED"
    )

    try:

        while True:

            if request.client is None:
                break

            if not is_camera_running(
                cam_id
            ):
                break

            frame = get_latest_frame(
                cam_id
            )

            if frame is None:

                time.sleep(0.03)

                continue

            ret, buffer = cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    72
                ]
            )

            if not ret:
                continue

            frame_bytes = (
                buffer.tobytes()
            )

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes +
                b"\r\n"
            )

            time.sleep(0.03)

    except GeneratorExit:

        print(
            f"[STREAM {cam_id}] CLOSED"
        )

    except Exception as e:

        print(
            "[STREAM ERROR]",
            e
        )


# =========================================
# LIVE FEED
# =========================================

@router.get("/camera/feed")
async def camera_feed(
    request: Request,
    cam_id: int = 0
):

    # FIX (#1): previously this auto-started the camera if it wasn't
    # running, which meant a stopped camera would silently restart the
    # moment the frontend's <img> tag re-requested the feed URL. The
    # feed endpoint should only ever serve an already-running camera;
    # starting a camera is the job of /camera/start.
    if not is_camera_running(
        cam_id
    ):

        raise HTTPException(
            status_code=404,
            detail=f"Camera {cam_id} is not running"
        )

    return StreamingResponse(

        generate_stream(
            request,
            cam_id
        ),

        media_type=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )


# =========================================
# ENTERPRISE WEBSOCKET
# =========================================

@router.websocket("/ws/{cam_id}")
async def websocket_stream(
    websocket: WebSocket,
    cam_id: int
):

    await manager.connect(
        cam_id,
        websocket
    )

    try:

        while True:

            # keep connection alive
            await asyncio.sleep(
                30
            )

    except WebSocketDisconnect:

        manager.disconnect(
            websocket
        )

    except Exception as e:

        print(
            "[WS ERROR]",
            e
        )

        manager.disconnect(
            websocket
        )


# =========================================
# DETECT IMAGE
# =========================================

@router.post("/detect")
async def detect(
    file: UploadFile = File(...)
):

    try:

        contents = await file.read()

        frame = decode_image(
            contents
        )

        result = process_frame(
            frame
        )

        risk = result.get(
            "risk",
            "LOW"
        ).lower()

        if risk == "high":

            add_alert(
                "Helmet Missing",
                "high",
                0
            )

        elif risk == "medium":

            add_alert(
                "Vest Missing",
                "medium",
                0
            )

        else:

            add_alert(
                "Worker Safe",
                "low",
                0
            )

        # NOTE: analytics/annotated_image are intentionally NOT computed
        # here. /detect is the lightweight endpoint (used by polling /
        # streaming callers); encoding a frame or building full analytics
        # on every call adds real per-request cost. We just surface
        # whatever process_frame already returned, defaulting missing
        # keys to empty values so the response shape is always stable.
        # If you need the heavier payload populated for real, call
        # /detect-full instead, or let me know and I'll wire frame
        # encoding into this path too.
        analytics = result.get(
            "analytics",
            {}
        )

        return {

            "success": True,

            "risk": risk,

            "detections":
                result.get(
                    "detections",
                    []
                ),

            "summary":
                analytics.get(
                    "ppe_summary",
                    {}
                ),

            "findings":
                result.get(
                    "findings",
                    []
                ),

            "recommendations":
                analytics.get(
                    "recommendations",
                    []
                ),

            "analytics":
                analytics,

            "telemetry":
                result.get(
                    "telemetry",
                    {}
                ),

            "ai_metadata":
                result.get(
                    "ai_metadata",
                    {}
                ),

            "annotated_image": ""
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================
# FULL DETECTION API
# =========================================

@router.post("/detect-full")
async def detect_full(
    file: UploadFile = File(...)
):

    try:

        contents = await file.read()

        frame = decode_image(
            contents
        )

        result = process_frame(
            frame
        )

        annotated_image = encode_image(
            frame
        )

        analytics = result.get(
            "analytics",
            {}
        )

        add_alert(
            "Safety Analysis Complete",
            result.get("risk", "LOW"),
            0
        )

        return {

            "success": True,

            "image": annotated_image,

            "detections":
                result.get(
                    "detections",
                    []
                ),

            "analytics":
                analytics,

            "ppe_summary":
                analytics.get(
                    "ppe_summary",
                    {}
                ),

            "equipment":
                analytics.get(
                    "equipment",
                    []
                ),

            "recommendations":
                analytics.get(
                    "recommendations",
                    []
                ),

            "risk":
                result.get(
                    "risk",
                    analytics.get(
                        "overall_risk",
                        "LOW"
                    )
                )
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )



# =========================================
# START CAMERA
# =========================================

@router.get("/camera/start")
def start(
    cam_id: int = 0
):

    ok = start_camera(
        cam_id
    )

    return {

        "status":
            "started"
            if ok
            else "failed"
    }


# =========================================
# STOP CAMERA
# =========================================

@router.get("/camera/stop")
def stop(
    cam_id: int = 0
):

    stop_camera(
        cam_id
    )

    return {
        "status": "stopped"
    }


# =========================================
# STATUS
# =========================================

@router.get("/camera/status")
def status(
    cam_id: int = 0
):

    return {

        "running":
            is_camera_running(
                cam_id
            )
    }