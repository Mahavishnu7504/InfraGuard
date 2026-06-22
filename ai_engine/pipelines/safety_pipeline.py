from datetime import datetime
import uuid
import time

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

tracker =  EnterpriseTracker()

alert_manager = AlertManager(
    cooldown=10
)

# =========================================
# CONFIG
# =========================================

CONFIDENCE_THRESHOLD = 0.40

HIGH_RISK_CLASSES = {
    "no_helmet",
    "no_vest",
    "danger_intrusion"
}

MEDIUM_RISK_CLASSES = {
    "person"
}

# =========================================
# PIPELINE
# =========================================

def run_safety_pipeline(frame):

    start = time.time()

    raw = predictor.predict_frame(frame)

    filtered = []

    for det in raw:

        conf = float(
            det.get(
                "confidence",
                0
            )
        )

        if conf < CONFIDENCE_THRESHOLD:
            continue

        filtered.append(det)

    persons = [

        d for d in filtered

        if d.get(
            "class_name",
            ""
        ).lower() == "person"
    ]

    non_persons = [

        d for d in filtered

        if d.get(
            "class_name",
            ""
        ).lower() != "person"
    ]

    persons = tracker.update(
        persons
    )

    detections = []
    alerts = []

    # =====================================
    # DETECTIONS
    # =====================================

    for p in persons:

        bbox = p.get(
            "bbox",
            []
        )

        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox

        confidence = float(
            p.get(
                "confidence",
                0
            )
        )

        label = (
            p.get(
                "class_name",
                "person"
            )
            .lower()
        )

        risk = calculate_risk(
            label,
            confidence
        )

        worker_id = str(
            p.get(
                "id",
                uuid.uuid4()
            )
        )

        det = {

            "id":
                worker_id,

            "class_name":
                label,

            "label":
                f"Worker {worker_id[:4]}",

            "bbox": [
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ],

            "x":
                int(x1),

            "y":
                int(y1),

            "w":
                int(x2 - x1),

            "h":
                int(y2 - y1),

            "confidence":
                confidence,

            "risk":
                risk,

            "timestamp":
                datetime.utcnow().isoformat(),

            "type":
                "worker",

            "tracking":
                True
        }

        detections.append(det)

        # =================================
        # SMART ALERTS
        # =================================

        should_alert = (
            alert_manager.should_alert(
                worker_id,
                risk
            )
        )

        if should_alert:

            message = (
                f"Worker {worker_id[:4]} "
                f"{risk.upper()} risk detected"
            )

            alert = {

                "worker_id":
                    worker_id,

                "risk":
                    risk,

                "message":
                    message,

                "timestamp":
                    datetime.utcnow().isoformat()
            }

            alerts.append(
                alert
            )

            # =================================
            # SAVE EVENTS
            # =================================

            try:
                save_event({

                    "event_type":
                        "PPE_ALERT",

                    "risk_level":
                        risk.upper(),

                    "camera_id":
                        0,

                    "workers":
                        1,

                    "violating_workers":
                        1,

                    "description":
                        message
                })
            except Exception:
                import traceback
                traceback.print_exc()

            # =================================
            # LIVE ACTIVITY
            # =================================

            try:
                add_alert(
                    event_type=
                        "PPE Violation",

                    risk=
                        risk,

                    cam_id=0,

                    description=
                        message
                )
            except Exception:
                import traceback
                traceback.print_exc()

    # =====================================
    # NON-PERSON DETECTIONS
    # (helmets, vests, cracks, equipment, etc.)
    # =====================================

    for n in non_persons:

        bbox = n.get(
            "bbox",
            []
        )

        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox

        confidence = float(
            n.get(
                "confidence",
                0
            )
        )

        label = (
            n.get(
                "class_name",
                "object"
            )
            .lower()
        )

        risk = calculate_risk(
            label,
            confidence
        )

        det = {

            "id":
                str(uuid.uuid4()),

            "class_name":
                label,

            "label":
                label,

            "bbox": [
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ],

            "x":
                int(x1),

            "y":
                int(y1),

            "w":
                int(x2 - x1),

            "h":
                int(y2 - y1),

            "confidence":
                confidence,

            "risk":
                risk,

            "timestamp":
                datetime.utcnow().isoformat(),

            "type":
                "object",

            "tracking":
                False
        }

        detections.append(det)

    # =====================================
    # ANALYTICS
    # =====================================

    high = len([
        d for d in detections
        if d["risk"] == "high"
    ])

    medium = len([
        d for d in detections
        if d["risk"] == "medium"
    ])

    low = len([
        d for d in detections
        if d["risk"] == "low"
    ])

    overall = "LOW"

    if high > 0:
        overall = "HIGH"

    elif medium > 0:
        overall = "MEDIUM"

    inference_time = round(
        (
            time.time() - start
        ) * 1000,
        2
    )

    return {

        "detections":
            detections,

        "alerts":
            alerts,

        "high":
            high,

        "medium":
            medium,

        "low":
            low,

        "risk":
            overall,

        "analytics": {

            "total_objects":
                len(detections),

            "processing_ms":
                inference_time,

            "tracker_active":
                True
        }
    }


# =========================================
# RISK ENGINE
# =========================================

def calculate_risk(
    label,
    confidence
):

    label = label.lower()

    if label in HIGH_RISK_CLASSES:
        return "high"

    if label in MEDIUM_RISK_CLASSES:

        if confidence > 0.8:
            return "medium"

    return "low"