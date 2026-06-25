# =========================================================
# INFRA GUARD — ENTERPRISE ANALYTICS ENGINE
# =========================================================

from datetime import datetime
from collections import Counter

from backend.services.event_service import get_latest_events, get_alert_events

# =========================================================
# SAFETY ANALYTICS (HISTORICAL — last 500 events)
# =========================================================

def get_safety_analytics():

    incidents = get_latest_events(limit=500)

    total_incidents = len(incidents)

    risk_distribution = Counter(
        e.risk_level for e in incidents
    )

    incident_types = Counter(
        e.event_type for e in incidents
    )

    # NOTE: VALID_RISK_LEVELS in event_service.py is {"LOW", "MEDIUM", "HIGH"}.
    # "CRITICAL" is not a valid risk_level for this system, so it is omitted
    # here rather than silently summed in as a phantom 0.
    high_risk = risk_distribution.get("HIGH", 0)

    operational_score = max(
        100 - (high_risk * 8),
        65
    )

    return {
        "module":               "Safety Intelligence",
        "total_incidents":      total_incidents,
        "risk_distribution":    dict(risk_distribution),
        "incident_types":       dict(incident_types),
        "operational_safety_score": operational_score,
        "live_ai_status":       "ACTIVE",
        "active_cameras":       4,
        "ai_processing_fps":    28,
    }


# =========================================================
# QUALITY ANALYTICS (HISTORICAL — last 500 events)
# =========================================================

def get_quality_analytics():

    events = get_latest_events(limit=500)

    inspections = [
        e for e in events
        if e.event_type == "PPE_DETECTION"
    ]

    total = len(inspections) or 1  # avoid division by zero

    compliant = sum(
        1 for e in inspections
        if e.risk_level == "LOW"
    )

    avg_compliance = round((compliant / total) * 100, 1)

    passed = sum(
        1 for e in inspections
        if e.compliant_workers > 0 and e.violating_workers == 0
    )

    audit_readiness = "A+" if avg_compliance >= 90 else "B"

    return {
        "module":                   "Quality Intelligence",
        "total_inspections":        total,
        "average_compliance":       avg_compliance,
        "successful_inspections":   passed,
        "audit_readiness":          audit_readiness,
        "ai_confidence":            "98%",
        "report_generation":        "OPERATIONAL",
    }


# =========================================================
# LIVE / CURRENT STATE ANALYTICS
#
# Derived strictly from the most recent event(s) returned by
# get_latest_events() / get_alert_events() — no new data sources,
# no new event_service queries, no assumed schema fields beyond
# what is defined on backend.models.db_models.Event and validated
# in event_service.VALID_EVENT_TYPES / VALID_RISK_LEVELS.
# =========================================================

def _latest_event_of_type(events, event_type):
    """Return the most recent event matching event_type, or None.

    events is assumed already ordered newest-first, which is how
    get_latest_events() returns rows (ordered by desc(Event.timestamp)).
    """
    for event in events:
        if event.event_type == event_type:
            return event
    return None


def get_current_snapshot():
    """
    Live/current-state snapshot, as opposed to the historical
    aggregates in get_safety_analytics() / get_quality_analytics().

    'Current' is defined as: the most recent event overall (for
    workers/PPE/compliance/safety score), and the most recent event
    of each relevant type (for equipment/cracks), pulled from the
    same get_latest_events() feed already used elsewhere in this file.
    """

    events = get_latest_events(limit=500)

    if not events:
        return {
            "module":              "Live Snapshot",
            "has_data":            False,
            "current_workers":     0,
            "current_ppe": {
                "helmet":  0,
                "vest":    0,
                "boots":   0,
            },
            "current_equipment": {
                "machines":     0,
                "danger_zones": 0,
            },
            "current_cracks":     0,
            "current_safety_score":     None,
            "current_compliance":       None,
            "last_event_timestamp":     None,
        }

    latest = events[0]  # newest event overall, since get_latest_events
                          # orders by desc(Event.timestamp)

    # --- Current Workers / Compliance: from the most recent event
    # that actually carries worker counts (PPE_DETECTION events are
    # the ones where workers / compliant_workers / violating_workers
    # are populated per save_event()).
    ppe_event = _latest_event_of_type(events, "PPE_DETECTION") or latest

    current_workers = ppe_event.workers

    if ppe_event.workers:
        current_compliance = round(
            (ppe_event.compliant_workers / ppe_event.workers) * 100, 1
        )
    else:
        current_compliance = None

    current_ppe = {
        "helmet": ppe_event.helmet,
        "vest":   ppe_event.vest,
        "boots":  ppe_event.boots,
    }

    # --- Current Equipment: machines / danger_zones are fields on
    # every Event row (see event_service.save_event), so pull them
    # from whichever event is most recent overall.
    current_equipment = {
        "machines":     latest.machines,
        "danger_zones": latest.danger_zones,
    }

    # --- Current Cracks: from the most recent CRACK_DETECTION event,
    # since 'cracks' is only meaningfully populated on that event_type.
    crack_event = _latest_event_of_type(events, "CRACK_DETECTION")
    current_cracks = crack_event.cracks if crack_event else 0

    # --- Current Safety Score: risk_level of the single most recent
    # event, mapped the same direction as operational_safety_score
    # (HIGH/MEDIUM drag the score down; LOW keeps it at 100).
    risk_penalty = {
        "LOW":    0,
        "MEDIUM": 15,
        "HIGH":   30,
    }
    current_safety_score = 100 - risk_penalty.get(latest.risk_level, 0)

    return {
        "module":                   "Live Snapshot",
        "has_data":                 True,
        "current_workers":          current_workers,
        "current_ppe":              current_ppe,
        "current_equipment":        current_equipment,
        "current_cracks":           current_cracks,
        "current_safety_score":     current_safety_score,
        "current_compliance":       current_compliance,
        "last_event_timestamp":     latest.timestamp.isoformat(),
    }


def get_todays_alerts():
    """
    Today's Alerts: alert events (HIGH/MEDIUM risk, per
    event_service.get_alert_events) filtered down to those whose
    timestamp falls on today's UTC date.

    There is no get_events_since()/date-filtered query in
    event_service.py, so the date filter is applied here in Python
    against the rows get_alert_events() already returns, rather than
    adding a new query function to event_service.py.
    """

    alerts = get_alert_events(limit=50)

    today = datetime.utcnow().date()

    todays = [
        event for event in alerts
        if event.timestamp.date() == today
    ]

    by_risk = Counter(e.risk_level for e in todays)

    return {
        "module":          "Today's Alerts",
        "count":           len(todays),
        "by_risk_level":   dict(by_risk),
        "alerts": [
            {
                "id":        event.id,
                "title":     event.event_type,
                "severity":  event.risk_level,
                "timestamp": event.timestamp.isoformat(),
                "status":    "ACTIVE",
            }
            for event in todays
        ],
    }


# =========================================================
# ENTERPRISE OVERVIEW
# =========================================================

def get_enterprise_overview():

    safety   = get_safety_analytics()
    quality  = get_quality_analytics()
    current  = get_current_snapshot()
    alerts   = get_todays_alerts()

    overall_health = round(
        (
            safety["operational_safety_score"]
            + quality["average_compliance"]
        ) / 2,
        1
    )

    return {
        "platform":         "InfraGuard Enterprise AI",
        "timestamp":        datetime.utcnow().isoformat(),
        "system_health":    f"{overall_health}%",
        "modules": {
            "safety":   safety,
            "quality":  quality,
            "current":  current,
            "alerts_today": alerts,
        },
        "enterprise_status": "OPERATIONAL",
        "realtime_ai":       True,
    }


# =========================================================
# INCIDENT FEED
# =========================================================

def get_incident_feed():

    alerts = get_alert_events(limit=50)

    feed = [
        {
            "id":        event.id,
            "title":     event.event_type,
            "severity":  event.risk_level,
            "timestamp": event.timestamp.isoformat(),
            "status":    "ACTIVE",
        }
        for event in alerts
    ]

    return {
        "count":     len(feed),
        "incidents": feed,
    }


# =========================================================
# EXECUTIVE KPI
# =========================================================

def get_executive_kpis():

    safety   = get_safety_analytics()
    quality  = get_quality_analytics()
    current  = get_current_snapshot()
    alerts   = get_todays_alerts()

    return {
        "live_cameras":         safety["active_cameras"],
        "active_incidents":     safety["total_incidents"],
        "safety_score":         safety["operational_safety_score"],
        "compliance_score":     quality["average_compliance"],
        "ai_confidence":        quality["ai_confidence"],
        "system_status":        "STABLE",
        "realtime_processing":  f"{safety['ai_processing_fps']} FPS",

        # --- Live snapshot fields (current, not historical) ---
        "current_workers":      current["current_workers"],
        "current_ppe":          current["current_ppe"],
        "current_equipment":    current["current_equipment"],
        "current_cracks":       current["current_cracks"],
        "current_safety_score": current["current_safety_score"],
        "current_compliance":   current["current_compliance"],
        "todays_alert_count":   alerts["count"],
    }


# =========================================================
# TELEMETRY
# =========================================================

def get_realtime_telemetry():

    return {
        "stream_engine":       "ACTIVE",
        "websocket":           "CONNECTED",
        "ai_pipeline":         "RUNNING",
        "incident_monitoring": "ACTIVE",
        "quality_engine":      "READY",
        "safety_engine":       "RUNNING",
        "enterprise_mode":     True,
        "last_updated":        datetime.utcnow().isoformat(),
    }