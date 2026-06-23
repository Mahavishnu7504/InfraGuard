from fastapi import APIRouter
from datetime import datetime
import uuid

try:
    from backend.core.websocket_manager import manager
except ImportError:
    manager = None

router = APIRouter()

ALERTS = []

MAX_ALERTS = 300


# =========================================
# ADD ALERT
# =========================================

def add_alert(
    event_type="System Check",
    risk="low",
    cam_id=0,
    description=""
):

    ALERTS.insert(0, {

        "id":
            str(uuid.uuid4()),

        "event_type":
            event_type,

        "risk_level":
            risk.upper(),

        "camera_id":
            cam_id,

        "description":
            description,

        "timestamp":
            datetime.utcnow().isoformat()
    })

    del ALERTS[MAX_ALERTS:]


# =========================================
# HISTORY
# =========================================

@router.get("/latest/{cam_id}")
def latest_activity(cam_id: int):

    data = [

        x for x in ALERTS

        if x["camera_id"] == cam_id
    ]

    return data[:60]


# =========================================
# ANALYTICS
# =========================================

@router.get("/analytics/summary")
def analytics_summary():

    low = len([
        x for x in ALERTS
        if x["risk_level"] == "LOW"
    ])

    medium = len([
        x for x in ALERTS
        if x["risk_level"] == "MEDIUM"
    ])

    high = len([
        x for x in ALERTS
        if x["risk_level"] == "HIGH"
    ])

    total = (
        low +
        medium +
        high
    )

    safety_score = 100

    if total > 0:

        safety_score = round(

            (
                (
                    low * 1.0 +
                    medium * 0.7 +
                    high * 0.35
                ) / total
            ) * 100
        )

    active_connections = 0
    events_sent = 0

    if manager is not None:
        active_connections = len(
            getattr(manager, "active_connections", [])
        )
        events_sent = getattr(manager, "total_events_sent", 0)

    return {

        "low":
            low,

        "medium":
            medium,

        "high":
            high,

        "total":
            total,

        "safety_score":
            safety_score,

        "system_status":
            "ACTIVE",

        "active_connections":
            active_connections,

        "events_sent":
            events_sent,

        "last_updated":
            datetime.utcnow().isoformat()
    }


# =========================================
# ANALYTICS HEALTH
# =========================================

@router.get("/analytics/health")
def analytics_health():

    active_connections = 0
    events_sent = 0

    if manager is not None:
        active_connections = len(
            getattr(manager, "active_connections", [])
        )
        events_sent = getattr(manager, "total_events_sent", 0)

    return {

        "alerts_stored":
            len(ALERTS),

        "max_alerts":
            MAX_ALERTS,

        "websocket_connections":
            active_connections,

        "events_sent":
            events_sent,

        "status":
            "HEALTHY" if manager is not None else "DEGRADED"
    }


# =========================================
# DEMO
# =========================================

@router.get("/seed")
def seed_demo():

    add_alert(
        "Helmet Missing",
        "high",
        0,
        "Worker without helmet detected"
    )

    add_alert(
        "Danger Zone Intrusion",
        "high",
        0,
        "Worker entered restricted zone"
    )

    add_alert(
        "Vest Missing",
        "medium",
        0,
        "Worker without safety vest"
    )

    add_alert(
        "Safe Activity",
        "low",
        0,
        "Worker compliant with PPE"
    )

    return {
        "status": "seeded"
    }