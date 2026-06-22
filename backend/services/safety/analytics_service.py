# =========================================================
# INFRA GUARD — ENTERPRISE ANALYTICS ENGINE
# =========================================================

from datetime import datetime
from collections import Counter

# =========================================================
# MOCK DATA SOURCE
# =========================================================

# In production:
# these come from database,
# websocket telemetry,
# AI incident streams,
# inspection records,
# and analytics aggregation pipelines.

SAFETY_INCIDENTS = [

    {
        "type": "PPE Violation",
        "risk": "HIGH"
    },

    {
        "type": "Danger Zone Intrusion",
        "risk": "CRITICAL"
    },

    {
        "type": "Structural Crack",
        "risk": "MEDIUM"
    },

    {
        "type": "PPE Violation",
        "risk": "HIGH"
    }
]

QUALITY_INSPECTIONS = [

    {
        "compliance": 96,
        "status": "PASS"
    },

    {
        "compliance": 92,
        "status": "PASS"
    },

    {
        "compliance": 81,
        "status": "WARNING"
    }
]

# =========================================================
# SAFETY ANALYTICS
# =========================================================

def get_safety_analytics():

    total_incidents = len(
        SAFETY_INCIDENTS
    )

    risk_distribution = Counter(

        item["risk"]

        for item in SAFETY_INCIDENTS
    )

    incident_types = Counter(

        item["type"]

        for item in SAFETY_INCIDENTS
    )

    high_risk = (

        risk_distribution.get("HIGH", 0)

        +

        risk_distribution.get("CRITICAL", 0)
    )

    operational_score = max(

        100 - (high_risk * 8),

        65
    )

    return {

        "module":
            "Safety Intelligence",

        "total_incidents":
            total_incidents,

        "risk_distribution":
            dict(risk_distribution),

        "incident_types":
            dict(incident_types),

        "operational_safety_score":
            operational_score,

        "live_ai_status":
            "ACTIVE",

        "active_cameras":
            4,

        "ai_processing_fps":
            28
    }

# =========================================================
# QUALITY ANALYTICS
# =========================================================

def get_quality_analytics():

    total = len(
        QUALITY_INSPECTIONS
    )

    avg_compliance = round(

        sum(
            i["compliance"]

            for i in QUALITY_INSPECTIONS
        ) / total,

        1
    )

    passed = len([

        i

        for i in QUALITY_INSPECTIONS

        if i["status"] == "PASS"
    ])

    audit_readiness = (

        "A+"

        if avg_compliance >= 90

        else "B"
    )

    return {

        "module":
            "Quality Intelligence",

        "total_inspections":
            total,

        "average_compliance":
            avg_compliance,

        "successful_inspections":
            passed,

        "audit_readiness":
            audit_readiness,

        "ai_confidence":
            "98%",

        "report_generation":
            "OPERATIONAL"
    }

# =========================================================
# ENTERPRISE OVERVIEW
# =========================================================

def get_enterprise_overview():

    safety = get_safety_analytics()

    quality = get_quality_analytics()

    overall_health = round(

        (
            safety["operational_safety_score"]

            +

            quality["average_compliance"]
        ) / 2,

        1
    )

    return {

        "platform":
            "InfraGuard Enterprise AI",

        "timestamp":
            datetime.utcnow().isoformat(),

        "system_health":
            f"{overall_health}%",

        "modules": {

            "safety":
                safety,

            "quality":
                quality
        },

        "enterprise_status":
            "OPERATIONAL",

        "realtime_ai":
            True
    }

# =========================================================
# INCIDENT FEED
# =========================================================

def get_incident_feed():

    feed = []

    for idx, item in enumerate(
        SAFETY_INCIDENTS
    ):

        feed.append({

            "id":
                idx + 1,

            "title":
                item["type"],

            "severity":
                item["risk"],

            "timestamp":
                datetime.utcnow().isoformat(),

            "status":
                "ACTIVE"
        })

    return {

        "count":
            len(feed),

        "incidents":
            feed
    }

# =========================================================
# EXECUTIVE KPI
# =========================================================

def get_executive_kpis():

    safety = get_safety_analytics()

    quality = get_quality_analytics()

    return {

        "live_cameras":
            safety["active_cameras"],

        "active_incidents":
            safety["total_incidents"],

        "safety_score":
            safety["operational_safety_score"],

        "compliance_score":
            quality["average_compliance"],

        "ai_confidence":
            quality["ai_confidence"],

        "system_status":
            "STABLE",

        "realtime_processing":
            f"{safety['ai_processing_fps']} FPS"
    }

# =========================================================
# TELEMETRY
# =========================================================

def get_realtime_telemetry():

    return {

        "stream_engine":
            "ACTIVE",

        "websocket":
            "CONNECTED",

        "ai_pipeline":
            "RUNNING",

        "incident_monitoring":
            "ACTIVE",

        "quality_engine":
            "READY",

        "safety_engine":
            "RUNNING",

        "enterprise_mode":
            True,

        "last_updated":
            datetime.utcnow().isoformat()
    }