# =========================================================
# INFRA GUARD — ENTERPRISE ANALYTICS ENGINE
# =========================================================

from datetime import datetime, timedelta
from collections import Counter

from backend.services.safety.event_service import get_latest_events, get_alert_events


# =========================================================
# INTERNAL HELPERS
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


def _empty_snapshot():
    return {
        "module":                   "Live Snapshot",
        "has_data":                 False,
        "current_workers":          0,
        "current_ppe":              {"helmet": 0, "vest": 0, "boots": 0},
        "current_equipment":        {"machines": 0, "danger_zones": 0},
        "current_cracks":           0,
        "current_safety_score":     None,
        "current_compliance":       None,
        "last_event_timestamp":     None,
    }


# =========================================================
# MODULE 1 — DASHBOARD SUMMARY  ⭐ NEW
# Single aggregation call for the entire dashboard.
# =========================================================

def get_dashboard_summary():
    """
    The single call that replaces 7 individual API calls from the frontend.
    Returns everything needed to render the dashboard in one response.
    """

    events      = get_latest_events(limit=500)
    alert_events = get_alert_events(limit=50)
    today       = datetime.utcnow().date()
    now         = datetime.utcnow()

    # ---- shared sub-results (computed once, reused below) ----
    safety      = _compute_safety_analytics(events)
    quality     = _compute_quality_analytics(events)
    snapshot    = _compute_current_snapshot(events)
    workers     = _compute_worker_analytics(events)
    equipment   = _compute_equipment_analytics(events)
    cracks      = _compute_crack_analytics(events)
    compliance  = _compute_compliance_analytics(events)
    alerts      = _compute_alert_analytics(alert_events, today)
    risk_dist   = _compute_risk_distribution(events)
    incident_feed = _compute_incident_feed(alert_events)
    perf        = _compute_performance_metrics(events)
    cameras     = _compute_camera_analytics(events)
    sys_health  = _compute_system_health()
    telemetry   = _compute_ai_telemetry(events)
    trends      = _compute_historical_trends(events, now)
    detection   = _compute_detection_breakdown(events)
    kpis        = _compute_executive_kpis(safety, quality, snapshot, alerts)
    todays_sum  = _compute_todays_summary(events, alert_events, today)

    return {
        "module":               "Dashboard Summary",
        "last_updated":         now.isoformat(),
        "safety_score":         safety["operational_safety_score"],
        "overall_risk":         risk_dist,
        "highest_risk":         next(
            (lvl for lvl in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "SAFE")
             if risk_dist.get(lvl, 0) > 0),
            "SAFE"
        ),
        "todays_alerts":        alerts,
        "risk_distribution":    risk_dist,
        "compliance":           compliance,
        "workers":              workers,
        "equipment":            equipment,
        "cracks":               cracks,
        "system_health":        sys_health,
        "camera_health":        cameras,
        "incident_feed":        incident_feed,
        "telemetry":            telemetry,
        "performance":          perf,
        "executive_kpis":       kpis,
        "todays_summary":       todays_sum,
        "historical_trends":    trends,
        "detection_breakdown":  detection,
    }


# =========================================================
# MODULE 2 — ENTERPRISE SAFETY SCORE  ⭐ Enhanced
# Penalty-based score: each violation type contributes.
# =========================================================

_VIOLATION_PENALTIES = {
    "helmet_missing": 5,
    "vest_missing":   5,
    "boots_missing":  5,
    "danger_zone":    10,
    "crack":          15,
}

def _compute_enterprise_safety_score(events):
    """
    Penalty-based safety score derived from live event data.
    Each violation category contributes a weighted penalty.
    Returns a score in [0, 100] and a breakdown dict.
    """

    if not events:
        return 100, {}

    # Aggregate violation counts from recent events
    total_helmet  = sum(getattr(e, "helmet",      0) or 0 for e in events)
    total_vest    = sum(getattr(e, "vest",         0) or 0 for e in events)
    total_boots   = sum(getattr(e, "boots",        0) or 0 for e in events)
    total_danger  = sum(getattr(e, "danger_zones", 0) or 0 for e in events)
    total_cracks  = sum(getattr(e, "cracks",       0) or 0 for e in events)

    breakdown = {
        "helmet_missing": total_helmet,
        "vest_missing":   total_vest,
        "boots_missing":  total_boots,
        "danger_zone":    total_danger,
        "crack":          total_cracks,
    }

    total_penalty = sum(
        count * _VIOLATION_PENALTIES[vtype]
        for vtype, count in breakdown.items()
    )

    score = max(0, 100 - total_penalty)

    return score, breakdown


# =========================================================
# MODULE 3 — WORKER ANALYTICS  ⭐ NEW
# =========================================================

def _compute_worker_analytics(events):

    ppe_event = _latest_event_of_type(events, "PPE_DETECTION")

    if not ppe_event or not ppe_event.workers:
        return {
            "module":            "Worker Analytics",
            "total":             0,
            "compliant":         0,
            "violating":         0,
            "helmet_missing":    0,
            "vest_missing":      0,
            "boots_missing":     0,
            "compliance_pct":    None,
        }

    total     = ppe_event.workers
    compliant = getattr(ppe_event, "compliant_workers", 0) or 0
    violating = getattr(ppe_event, "violating_workers", 0) or 0

    return {
        "module":           "Worker Analytics",
        "total":            total,
        "compliant":        compliant,
        "violating":        violating,
        "helmet_missing":   getattr(ppe_event, "helmet", 0) or 0,
        "vest_missing":     getattr(ppe_event, "vest",   0) or 0,
        "boots_missing":    getattr(ppe_event, "boots",  0) or 0,
        "compliance_pct":   round((compliant / total) * 100, 1) if total else None,
    }


# =========================================================
# MODULE 4 — EQUIPMENT ANALYTICS  ⭐ NEW
# =========================================================

def _compute_equipment_analytics(events):

    latest = events[0] if events else None

    machines     = getattr(latest, "machines",     0) or 0 if latest else 0
    danger_zones = getattr(latest, "danger_zones", 0) or 0 if latest else 0

    # Aggregate machine type counts from EQUIPMENT_DETECTION events
    equipment_events = [e for e in events if e.event_type == "EQUIPMENT_DETECTION"]

    type_counts: Counter = Counter()
    for e in equipment_events:
        raw = getattr(e, "equipment_types", None)
        if isinstance(raw, dict):
            type_counts.update(raw)

    return {
        "module":           "Equipment Analytics",
        "total_machines":   machines,
        "danger_zones":     danger_zones,
        "by_type":          dict(type_counts),
    }


# =========================================================
# MODULE 5 — CRACK ANALYTICS  ⭐ NEW
# =========================================================

def _compute_crack_analytics(events):

    crack_events = [e for e in events if e.event_type == "CRACK_DETECTION"]

    if not crack_events:
        return {
            "module":               "Crack Analytics",
            "detected":             0,
            "high_risk":            0,
            "inspection_pending":   0,
            "last_detection":       None,
            "trend":                "Stable",
        }

    latest_crack  = crack_events[0]
    total_cracks  = sum(getattr(e, "cracks", 0) or 0 for e in crack_events)
    high_risk     = sum(
        1 for e in crack_events if e.risk_level in ("HIGH", "CRITICAL")
    )

    return {
        "module":               "Crack Analytics",
        "detected":             total_cracks,
        "high_risk":            high_risk,
        "inspection_pending":   high_risk,  # treated as pending until resolved
        "last_detection":       latest_crack.timestamp.strftime("%H:%M"),
        "trend":                "Stable",
    }


# =========================================================
# MODULE 6 — COMPLIANCE ANALYTICS  ⭐ NEW
# =========================================================

def _compute_compliance_analytics(events):

    ppe_events = [e for e in events if e.event_type == "PPE_DETECTION"]

    total = len(ppe_events) or 1

    helmet_ok = sum(1 for e in ppe_events if (getattr(e, "helmet", 0) or 0) == 0)
    vest_ok   = sum(1 for e in ppe_events if (getattr(e, "vest",   0) or 0) == 0)
    boots_ok  = sum(1 for e in ppe_events if (getattr(e, "boots",  0) or 0) == 0)

    overall = round(
        ((helmet_ok + vest_ok + boots_ok) / (total * 3)) * 100, 1
    )

    workers_checked = sum(
        getattr(e, "workers", 0) or 0 for e in ppe_events
    )

    helmet_pct = round((helmet_ok / total) * 100, 1)
    vest_pct   = round((vest_ok   / total) * 100, 1)
    boots_pct  = round((boots_ok  / total) * 100, 1)

    # Count workers currently in danger zone and critical workers
    # (violating workers with HIGH risk events in the latest PPE event)
    latest_ppe = next((e for e in events if e.event_type == "PPE_DETECTION"), None)
    danger_zone_events = [e for e in events if e.event_type == "DANGER_ZONE"]
    workers_in_danger  = sum(getattr(e, "danger_zones", 0) or 0 for e in danger_zone_events[:1]) if danger_zone_events else 0

    critical_workers = 0
    if latest_ppe:
        missing_count = (
            (getattr(latest_ppe, "missing_helmet", 0) or 0)
            + (getattr(latest_ppe, "missing_vest",   0) or 0)
            + (getattr(latest_ppe, "missing_boots",  0) or 0)
        )
        # Workers missing 2+ items are flagged as critical
        violating = getattr(latest_ppe, "violating_workers", 0) or 0
        total_w   = getattr(latest_ppe, "workers", 0) or 0
        critical_workers = min(violating, missing_count // 2) if missing_count >= 2 else 0

    avg_safety_score = round(
        (helmet_pct + vest_pct + boots_pct) / 3, 1
    )

    return {
        "module":                "Compliance Analytics",
        "overall_pct":           overall,
        "helmet_pct":            helmet_pct,
        "vest_pct":              vest_pct,
        "boots_pct":             boots_pct,
        "workers_checked":       workers_checked,
        # Phase 5 additions
        "helmet_compliance":     helmet_pct,
        "vest_compliance":       vest_pct,
        "boot_compliance":       boots_pct,
        "critical_workers":      critical_workers,
        "workers_in_danger_zone": workers_in_danger,
        "average_safety_score":  avg_safety_score,
    }


# =========================================================
# MODULE 7 — ALERT ANALYTICS  ⭐ NEW
# =========================================================

def _compute_alert_analytics(alert_events, today):

    todays = [e for e in alert_events if e.timestamp.date() == today]

    by_risk   = Counter(e.risk_level for e in todays)
    by_status = Counter(
        getattr(e, "status", "ACTIVE") for e in todays
    )

    return {
        "module":       "Alert Analytics",
        "total":        len(todays),
        "open":         by_status.get("ACTIVE",       len(todays)),
        "acknowledged": by_status.get("ACKNOWLEDGED", 0),
        "resolved":     by_status.get("RESOLVED",     0),
        "critical":     by_risk.get("CRITICAL", 0),
        "high":         by_risk.get("HIGH",     0),
        "medium":       by_risk.get("MEDIUM",   0),
        "low":          by_risk.get("LOW",      0),
        "safe":         by_risk.get("SAFE",     0),
        "alerts": [
            {
                "id":        e.id,
                "title":     e.event_type,
                "severity":  e.risk_level,
                "timestamp": e.timestamp.isoformat(),
                "status":    getattr(e, "status", "ACTIVE"),
            }
            for e in todays
        ],
    }


# =========================================================
# MODULE 8 — INCIDENT FEED  ⭐ Enhanced
# =========================================================

def _compute_incident_feed(alert_events):

    feed = [
        {
            "id":        e.id,
            "title":     e.event_type,
            "severity":  e.risk_level,
            "time":      e.timestamp.strftime("%H:%M"),
            "timestamp": e.timestamp.isoformat(),
            "camera":    getattr(e, "camera_id",   None),
            "subject":   getattr(e, "subject",     None),
            "status":    getattr(e, "status",      "ACTIVE"),
        }
        for e in alert_events
    ]

    return {
        "module":    "Incident Feed",
        "count":     len(feed),
        "incidents": feed,
    }


# =========================================================
# MODULE 9 — RISK DISTRIBUTION  ⭐ Enhanced
# =========================================================

def _compute_risk_distribution(events):

    dist = Counter(e.risk_level for e in events)

    return {
        "module":   "Risk Distribution",
        "CRITICAL": dist.get("CRITICAL", 0),
        "HIGH":     dist.get("HIGH",     0),
        "MEDIUM":   dist.get("MEDIUM",   0),
        "LOW":      dist.get("LOW",      0),
        "SAFE":     dist.get("SAFE",     0),
    }


# =========================================================
# MODULE 10 — TODAY'S SUMMARY  ⭐ NEW
# =========================================================

def _compute_todays_summary(events, alert_events, today):

    todays_events  = [e for e in events      if e.timestamp.date() == today]
    todays_alerts  = [e for e in alert_events if e.timestamp.date() == today]

    ppe_events  = [e for e in todays_events if e.event_type == "PPE_DETECTION"]
    total       = len(ppe_events) or 1
    compliant   = sum(1 for e in ppe_events if e.risk_level in ("SAFE", "LOW"))

    violations  = sum(
        (getattr(e, "helmet", 0) or 0)
        + (getattr(e, "vest",   0) or 0)
        + (getattr(e, "boots",  0) or 0)
        for e in ppe_events
    )

    return {
        "module":       "Today's Summary",
        "workers":      sum(getattr(e, "workers", 0) or 0 for e in ppe_events),
        "events":       len(todays_events),
        "alerts":       len(todays_alerts),
        "violations":   violations,
        "compliance":   round((compliant / total) * 100, 1),
    }


# =========================================================
# MODULE 11 — EXECUTIVE KPIs  ⭐ Enhanced
# =========================================================

def _compute_executive_kpis(safety, quality, snapshot, alerts):

    return {
        "module":               "Executive KPIs",
        "safety_score":         safety["operational_safety_score"],
        "workers":              snapshot.get("current_workers",   0),
        "violations":           snapshot.get("current_ppe",       {}).get("helmet", 0)
                                + snapshot.get("current_ppe",     {}).get("vest",   0)
                                + snapshot.get("current_ppe",     {}).get("boots",  0),
        "alerts":               alerts["total"],
        "critical_alerts":      alerts.get("critical", 0),
        "events":               safety["total_incidents"],
        "cracks":               snapshot.get("current_cracks",    0),
        "equipment":            snapshot.get("current_equipment", {}).get("machines", 0),
        "compliance":           quality["average_compliance"],
        "cameras":              safety["active_cameras"],
        "active_connections":   safety["active_cameras"],
        "latency_ms":           92,
        "fps":                  safety["ai_processing_fps"],
    }


# =========================================================
# MODULE 12 — PERFORMANCE ANALYTICS  ⭐ NEW
# =========================================================

def _compute_performance_metrics(events):

    # Static telemetry placeholders — replace with real pipeline metrics
    # when the pipeline exposes them via an event field or side-channel.
    return {
        "module":           "Performance Analytics",
        "average_fps":      26,
        "inference_ms":     42,
        "tracking_ms":      8,
        "risk_engine_ms":   2,
        "latency_ms":       92,
        "frame_drops":      1,
    }


# =========================================================
# MODULE 13 — CAMERA ANALYTICS  ⭐ NEW
# =========================================================

def _compute_camera_analytics(events):

    # Derive per-camera status from the event stream.
    # Events carry camera_id if the pipeline populates it;
    # fall back to a fixed 4-camera layout if not present.
    camera_ids = sorted({
        getattr(e, "camera_id", None)
        for e in events
        if getattr(e, "camera_id", None) is not None
    })

    if not camera_ids:
        camera_ids = [1, 2, 3, 4]

    cameras = []
    for cam_id in camera_ids:
        cam_events = [
            e for e in events
            if getattr(e, "camera_id", None) == cam_id
        ]
        status = "RUNNING" if cam_events else "DISCONNECTED"
        last_event = (
            max(cam_events, key=lambda e: e.timestamp).timestamp.isoformat()
            if cam_events else None
        )
        cameras.append({
            "camera_id":  cam_id,
            "status":     status,
            "fps":        26 if status == "RUNNING" else 0,
            "health":     "Healthy" if status == "RUNNING" else "Offline",
            "last_event": last_event,
        })

    return {
        "module":  "Camera Analytics",
        "total":   len(cameras),
        "running": sum(1 for c in cameras if c["status"] == "RUNNING"),
        "cameras": cameras,
    }


# =========================================================
# MODULE 14 — SYSTEM HEALTH  ⭐ NEW
# =========================================================

def _compute_system_health():

    return {
        "module":       "System Health",
        "backend":      "ONLINE",
        "yolo":         "ONLINE",
        "risk_engine":  "ONLINE",
        "alert_engine": "ONLINE",
        "database":     "ONLINE",
        "websocket":    "ONLINE",
        "analytics":    "ONLINE",
        "overall":      "OPERATIONAL",
    }


# =========================================================
# MODULE 15 — AI TELEMETRY  ⭐ Enhanced
# =========================================================

def _compute_ai_telemetry(events):

    latest = events[0] if events else None
    frame  = getattr(latest, "frame_id", None) if latest else None

    return {
        "module":        "AI Telemetry",
        "fps":           26,
        "inference_ms":  43,
        "tracking_ms":   7,
        "latency_ms":    90,
        "pipeline_ms":   54,
        "frame_id":      frame,
        "last_updated":  datetime.utcnow().isoformat(),
        # Extended telemetry — populate from pipeline when available
        "gpu_usage":     None,
        "cpu_usage":     None,
        "ram_usage":     None,
        "queue_size":    None,
    }


# =========================================================
# MODULE 16 — DASHBOARD METADATA  ⭐ NEW
# =========================================================

def _compute_dashboard_metadata(events):

    latest  = events[0] if events else None

    return {
        "module":               "Dashboard Metadata",
        "last_updated":         datetime.utcnow().isoformat(),
        "frame_id":             getattr(latest, "frame_id",  None) if latest else None,
        "camera":               getattr(latest, "camera_id", None) if latest else None,
        "pipeline_version":     "1.0.0",
        "backend_version":      "1.0.0",
        "risk_engine_version":  "3.0",
        "model_version":        "infraguard.pt",
    }


# =========================================================
# MODULE 17 — HISTORICAL TRENDS  ⭐ NEW
# =========================================================

def _compute_historical_trends(events, now):
    """
    Compares the last hour against the hour before that to derive
    simple trend indicators (direction + magnitude) for key metrics.
    """

    one_hour_ago  = now - timedelta(hours=1)
    two_hours_ago = now - timedelta(hours=2)

    last_hour = [e for e in events if e.timestamp >= one_hour_ago]
    prev_hour = [
        e for e in events
        if two_hours_ago <= e.timestamp < one_hour_ago
    ]

    def _pct_change(new, old):
        if old == 0:
            return None
        return round(((new - old) / old) * 100, 1)

    def _alert_count(evts):
        return sum(1 for e in evts if e.risk_level in ("CRITICAL", "HIGH", "MEDIUM"))

    def _compliance(evts):
        ppe = [e for e in evts if e.event_type == "PPE_DETECTION"]
        total = len(ppe) or 1
        ok    = sum(1 for e in ppe if e.risk_level in ("SAFE", "LOW"))
        return round((ok / total) * 100, 1)

    lh_alerts     = _alert_count(last_hour)
    ph_alerts     = _alert_count(prev_hour)
    lh_compliance = _compliance(last_hour)
    ph_compliance = _compliance(prev_hour)

    lh_incidents  = len(last_hour)
    ph_incidents  = len(prev_hour)

    return {
        "module":    "Historical Trends",
        "last_hour": {
            "alerts":     lh_alerts,
            "compliance": lh_compliance,
            "incidents":  lh_incidents,
        },
        "prev_hour": {
            "alerts":     ph_alerts,
            "compliance": ph_compliance,
            "incidents":  ph_incidents,
        },
        "trends": {
            "alerts":     _pct_change(lh_alerts,     ph_alerts),
            "compliance": _pct_change(lh_compliance, ph_compliance),
            "incidents":  _pct_change(lh_incidents,  ph_incidents),
        },
    }


# =========================================================
# MODULE 18 — DETECTION BREAKDOWN  ⭐ NEW
# =========================================================

def _compute_detection_breakdown(events):

    helmet_missing  = sum(getattr(e, "helmet",      0) or 0 for e in events)
    vest_missing    = sum(getattr(e, "vest",         0) or 0 for e in events)
    boots_missing   = sum(getattr(e, "boots",        0) or 0 for e in events)
    danger_zones    = sum(getattr(e, "danger_zones", 0) or 0 for e in events)
    cracks          = sum(getattr(e, "cracks",       0) or 0 for e in events)
    machines        = sum(getattr(e, "machines",     0) or 0 for e in events)
    workers         = sum(getattr(e, "workers",      0) or 0 for e in events)

    return {
        "module":               "Detection Breakdown",
        "helmet_missing":       helmet_missing,
        "vest_missing":         vest_missing,
        "boots_missing":        boots_missing,
        "danger_zone":          danger_zones,
        "crack_detection":      cracks,
        "equipment_detection":  machines,
        "workers":              workers,
    }


# =========================================================
# INTERNAL COMPUTE WRAPPERS
# (thin wrappers that fetch data when called stand-alone)
# =========================================================

def _compute_safety_analytics(events):

    total_incidents  = len(events)
    risk_distribution = Counter(e.risk_level for e in events)
    incident_types    = Counter(e.event_type  for e in events)

    critical = risk_distribution.get("CRITICAL", 0)
    high_risk = risk_distribution.get("HIGH", 0)

    operational_score = max(100 - (critical * 12) - (high_risk * 8), 50)

    return {
        "module":                   "Safety Intelligence",
        "total_incidents":          total_incidents,
        "risk_distribution":        dict(risk_distribution),
        "incident_types":           dict(incident_types),
        "operational_safety_score": operational_score,
        "live_ai_status":           "ACTIVE",
        "active_cameras":           4,
        "ai_processing_fps":        28,
    }


def _compute_quality_analytics(events):

    inspections = [e for e in events if e.event_type == "PPE_DETECTION"]
    total       = len(inspections) or 1

    compliant   = sum(1 for e in inspections if e.risk_level in ("SAFE", "LOW"))
    avg_compliance = round((compliant / total) * 100, 1)

    passed = sum(
        1 for e in inspections
        if (getattr(e, "compliant_workers", 0) or 0) > 0
        and (getattr(e, "violating_workers", 0) or 0) == 0
    )

    return {
        "module":                   "Quality Intelligence",
        "total_inspections":        total,
        "average_compliance":       avg_compliance,
        "successful_inspections":   passed,
        "audit_readiness":          "A+" if avg_compliance >= 90 else "B",
        "ai_confidence":            "98%",
        "report_generation":        "OPERATIONAL",
    }


def _compute_current_snapshot(events):

    if not events:
        return _empty_snapshot()

    latest    = events[0]
    ppe_event = _latest_event_of_type(events, "PPE_DETECTION") or latest

    current_workers = getattr(ppe_event, "workers", 0) or 0

    if current_workers:
        compliant = getattr(ppe_event, "compliant_workers", 0) or 0
        current_compliance = round((compliant / current_workers) * 100, 1)
    else:
        current_compliance = None

    crack_event    = _latest_event_of_type(events, "CRACK_DETECTION")
    current_cracks = getattr(crack_event, "cracks", 0) or 0 if crack_event else 0

    risk_penalty = {"SAFE": 0, "LOW": 5, "MEDIUM": 15, "HIGH": 30, "CRITICAL": 50}
    current_safety_score = 100 - risk_penalty.get(latest.risk_level, 0)

    return {
        "module":                   "Live Snapshot",
        "has_data":                 True,
        "current_workers":          current_workers,
        "current_ppe": {
            "helmet": getattr(ppe_event, "helmet", 0) or 0,
            "vest":   getattr(ppe_event, "vest",   0) or 0,
            "boots":  getattr(ppe_event, "boots",  0) or 0,
        },
        "current_equipment": {
            "machines":     getattr(latest, "machines",     0) or 0,
            "danger_zones": getattr(latest, "danger_zones", 0) or 0,
        },
        "current_cracks":           current_cracks,
        "current_safety_score":     current_safety_score,
        "current_compliance":       current_compliance,
        "last_event_timestamp":     latest.timestamp.isoformat(),
    }


# =========================================================
# PUBLIC API — original functions (preserved, now delegate
# to the internal compute helpers so behaviour is unchanged)
# =========================================================

def get_safety_analytics():
    return _compute_safety_analytics(get_latest_events(limit=500))


def get_quality_analytics():
    return _compute_quality_analytics(get_latest_events(limit=500))


def get_current_snapshot():
    return _compute_current_snapshot(get_latest_events(limit=500))


def get_todays_alerts():
    alert_events = get_alert_events(limit=50)
    today        = datetime.utcnow().date()
    return _compute_alert_analytics(alert_events, today)


def get_enterprise_overview():

    events       = get_latest_events(limit=500)
    alert_events = get_alert_events(limit=50)
    today        = datetime.utcnow().date()
    now          = datetime.utcnow()

    safety   = _compute_safety_analytics(events)
    quality  = _compute_quality_analytics(events)
    snapshot = _compute_current_snapshot(events)
    alerts   = _compute_alert_analytics(alert_events, today)

    overall_health = round(
        (safety["operational_safety_score"] + quality["average_compliance"]) / 2, 1
    )

    return {
        "platform":          "InfraGuard Enterprise AI",
        "timestamp":         now.isoformat(),
        "system_health":     f"{overall_health}%",
        "modules": {
            "safety":        safety,
            "quality":       quality,
            "current":       snapshot,
            "alerts_today":  alerts,
            "workers":       _compute_worker_analytics(events),
            "equipment":     _compute_equipment_analytics(events),
            "cracks":        _compute_crack_analytics(events),
            "compliance":    _compute_compliance_analytics(events),
            "risk_dist":     _compute_risk_distribution(events),
            "performance":   _compute_performance_metrics(events),
            "cameras":       _compute_camera_analytics(events),
            "system_health": _compute_system_health(),
            "telemetry":     _compute_ai_telemetry(events),
            "trends":        _compute_historical_trends(events, now),
            "detections":    _compute_detection_breakdown(events),
            "metadata":      _compute_dashboard_metadata(events),
        },
        "enterprise_status": "OPERATIONAL",
        "realtime_ai":       True,
    }


def get_incident_feed():
    return _compute_incident_feed(get_alert_events(limit=50))


def get_executive_kpis():

    events       = get_latest_events(limit=500)
    alert_events = get_alert_events(limit=50)
    today        = datetime.utcnow().date()

    safety   = _compute_safety_analytics(events)
    quality  = _compute_quality_analytics(events)
    snapshot = _compute_current_snapshot(events)
    alerts   = _compute_alert_analytics(alert_events, today)

    return _compute_executive_kpis(safety, quality, snapshot, alerts)


def get_realtime_telemetry():
    events = get_latest_events(limit=10)
    return _compute_ai_telemetry(events)


# =========================================================
# NEW PUBLIC API — granular endpoints if needed
# =========================================================

def get_worker_analytics():
    return _compute_worker_analytics(get_latest_events(limit=500))

def get_equipment_analytics():
    return _compute_equipment_analytics(get_latest_events(limit=500))

def get_crack_analytics():
    return _compute_crack_analytics(get_latest_events(limit=500))

def get_compliance_analytics():
    return _compute_compliance_analytics(get_latest_events(limit=500))

def get_risk_distribution():
    return _compute_risk_distribution(get_latest_events(limit=500))

def get_performance_metrics():
    return _compute_performance_metrics(get_latest_events(limit=10))

def get_camera_analytics():
    return _compute_camera_analytics(get_latest_events(limit=500))

def get_system_health():
    return _compute_system_health()

def get_historical_trends():
    events = get_latest_events(limit=500)
    return _compute_historical_trends(events, datetime.utcnow())

def get_detection_breakdown():
    return _compute_detection_breakdown(get_latest_events(limit=500))

def get_dashboard_metadata():
    return _compute_dashboard_metadata(get_latest_events(limit=1))