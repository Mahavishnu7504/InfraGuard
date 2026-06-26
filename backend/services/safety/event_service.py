import logging
import threading
import time
from datetime import datetime, timedelta

from backend.core.database import SessionLocal
from backend.models.db_models import Event
from sqlalchemy import desc, func

logger = logging.getLogger(__name__)

VALID_EVENT_TYPES = {"PPE_DETECTION", "DANGER_ZONE", "CRACK_DETECTION"}
VALID_RISK_LEVELS = {
    "SAFE",
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
}

# Used for sorting/display priority (lower number = more urgent).
RISK_PRIORITY = {
    "CRITICAL": 1,
    "HIGH": 2,
    "MEDIUM": 3,
    "LOW": 4,
    "SAFE": 5,
}

# How long (seconds) a repeated identical event from the same camera is
# suppressed before being allowed to save again. Tune per deployment.
DUPLICATE_SUPPRESSION_WINDOW_SECONDS = 4.0


# ==========================================
# VALIDATION
# ==========================================

def validate_event_data(event_data: dict) -> list[str]:
    """Return a list of validation error messages (empty = valid)."""

    errors = []

    if not isinstance(event_data, dict):
        return ["event_data must be a dict."]

    event_type = str(event_data.get("event_type", "PPE_DETECTION")).upper()
    if event_type not in VALID_EVENT_TYPES:
        errors.append(
            f"Invalid event_type '{event_type}'. "
            f"Must be one of {VALID_EVENT_TYPES}."
        )

    risk_level = str(event_data.get("risk_level", "LOW")).upper()
    if risk_level not in VALID_RISK_LEVELS:
        errors.append(
            f"Invalid risk_level '{risk_level}'. "
            f"Must be one of {VALID_RISK_LEVELS}."
        )

    int_fields = [
        "camera_id",
        "helmet", "vest", "boots",
        "low", "medium", "high",
        "workers", "compliant_workers", "violating_workers",
        "missing_helmet", "missing_vest", "missing_boots",
        "danger_zones", "machines", "cracks",
    ]
    for field in int_fields:
        value = event_data.get(field, 0)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            errors.append(
                f"Field '{field}' must be a non-negative number "
                f"(got {value!r})."
            )

    return errors


# ==========================================
# DESCRIPTION ENGINE
# ==========================================

def infer_description(event_data: dict) -> str:

    event_type = str(
        event_data.get("event_type", "PPE_DETECTION")
    ).upper()

    if event_type == "DANGER_ZONE":
        return "Worker entered restricted zone."

    if event_type == "CRACK_DETECTION":
        return "Potential structural crack detected."

    if event_type == "PPE_DETECTION":

        missing = []

        if event_data.get("missing_helmet", 0) > 0:
            missing.append("helmet")

        if event_data.get("missing_vest", 0) > 0:
            missing.append("vest")

        if event_data.get("missing_boots", 0) > 0:
            missing.append("boots")

        if missing:
            if len(missing) == 1:
                items = missing[0]
            elif len(missing) == 2:
                items = f"{missing[0]} and {missing[1]}"
            else:
                items = f"{', '.join(missing[:-1])}, and {missing[-1]}"
            return f"Worker missing {items}."

        return "Worker compliant with PPE."

    return "AI safety event recorded."


# ==========================================
# DUPLICATE SUPPRESSION
# ==========================================
# Live camera feeds can re-fire the same detection many times per second.
# We suppress identical (camera, worker, event_type, risk_level, description)
# combos within a short rolling window so the DB only gets one row per
# "incident", instead of one row per frame. Including worker_id prevents two
# different workers on the same camera from being collapsed into one event.

_recent_events_lock = threading.Lock()
_recent_events: dict[tuple, float] = {}


def _dedup_key(event_data: dict, description: str) -> tuple:
    return (
        event_data.get("camera_id", 0),
        event_data.get("worker_id"),
        str(event_data.get("event_type", "PPE_DETECTION")).upper(),
        str(event_data.get("risk_level", "LOW")).upper(),
        description,
    )


def _is_duplicate(key: tuple, window_seconds: float) -> bool:
    """Return True (and refresh the timestamp) if this key was seen recently."""

    now = time.monotonic()

    with _recent_events_lock:
        last_seen = _recent_events.get(key)

        if last_seen is not None and (now - last_seen) < window_seconds:
            # Still within the suppression window: refresh timestamp so the
            # window keeps sliding forward while the condition persists, but
            # tell the caller this is a duplicate (don't save).
            _recent_events[key] = now
            return True

        _recent_events[key] = now
        return False


def reset_duplicate_cache():
    """Mainly useful for tests."""
    with _recent_events_lock:
        _recent_events.clear()


# ==========================================
# SAVE EVENT
# ==========================================

def save_event(event_data: dict, suppress_duplicates: bool = True):
    """Validate, optionally dedup, and persist a single event.

    Returns the saved Event row, or None if validation failed, the event
    was suppressed as a duplicate, or a DB error occurred.
    """

    errors = validate_event_data(event_data)
    if errors:
        for error in errors:
            logger.warning("[EVENT VALIDATION] %s", error)
        return None

    description = event_data.get("description") or infer_description(event_data)

    if suppress_duplicates:
        key = _dedup_key(event_data, description)
        if _is_duplicate(key, DUPLICATE_SUPPRESSION_WINDOW_SECONDS):
            logger.debug(
                "[EVENT SUPPRESSED] camera=%s type=%s desc=%r (duplicate within %.1fs)",
                event_data.get("camera_id", 0),
                str(event_data.get("event_type", "PPE_DETECTION")).upper(),
                description,
                DUPLICATE_SUPPRESSION_WINDOW_SECONDS,
            )
            return None

    db = SessionLocal()

    try:

        event_kwargs = dict(

            event_type=event_data.get("event_type", "PPE_DETECTION"),

            risk_level=event_data.get("risk_level", "LOW").upper(),

            description=description,

            camera_id=event_data.get("camera_id", 0),

            image_path=event_data.get("image_path"),

            helmet=event_data.get("helmet", 0),

            vest=event_data.get("vest", 0),

            boots=event_data.get("boots", 0),

            low=event_data.get("low", 0),

            medium=event_data.get("medium", 0),

            high=event_data.get("high", 0),

            workers=event_data.get("workers", 0),

            compliant_workers=event_data.get("compliant_workers", 0),

            violating_workers=event_data.get("violating_workers", 0),

            missing_helmet=event_data.get("missing_helmet", 0),

            missing_vest=event_data.get("missing_vest", 0),

            missing_boots=event_data.get("missing_boots", 0),

            danger_zones=event_data.get("danger_zones", 0),

            machines=event_data.get("machines", 0),

            cracks=event_data.get("cracks", 0),
        )

        # The fields below (risk_score, compliance_pct, violations, worker_id)
        # are only set if the Event model actually defines that column, so
        # this won't crash on a DB schema that hasn't been migrated yet.
        # Once the columns exist, these will start persisting automatically.
        optional_fields = {
            "worker_id": event_data.get("worker_id"),
            "risk_score": event_data.get("risk_score", 0),
            "compliance_pct": event_data.get("compliance_pct"),
            "violations": event_data.get("violations", []),
        }
        for field, value in optional_fields.items():
            if hasattr(Event, field):
                event_kwargs[field] = value

        event = Event(**event_kwargs)

        db.add(event)
        db.commit()
        db.refresh(event)

        logger.info(
            "[EVENT SAVED] id=%s type=%s risk=%s camera=%s worker=%s score=%s",
            event.id,
            event.event_type,
            event.risk_level,
            event.camera_id,
            event_data.get("worker_id"),
            event_data.get("risk_score"),
        )

        return event

    except Exception as e:

        db.rollback()
        logger.error("[EVENT SAVE ERROR] %s", e, exc_info=True)
        return None

    finally:

        db.close()


def save_events(events_data: list[dict], suppress_duplicates: bool = True):
    """Save multiple events (e.g. all detections from one frame).

    Each event is independently validated and deduped, so one bad event
    in the batch doesn't block the rest. Returns the list of successfully
    saved Event rows (skips None results from validation/dedup/errors).
    """

    saved = []

    for event_data in events_data:
        result = save_event(event_data, suppress_duplicates=suppress_duplicates)
        if result is not None:
            saved.append(result)

    logger.info(
        "[EVENT BATCH SAVED] %d/%d events persisted",
        len(saved),
        len(events_data),
    )

    return saved


# ==========================================
# HISTORY
# ==========================================

def get_latest_events(limit=100):

    db = SessionLocal()

    try:

        rows = (
            db.query(Event)
            .order_by(desc(Event.timestamp))
            .limit(limit)
            .all()
        )

        return rows

    finally:

        db.close()


def get_recent_events(limit=100):
    """Alias of get_latest_events, kept for naming consistency with
    get_events_by_* helpers below."""
    return get_latest_events(limit=limit)


# ==========================================
# ALERT EVENTS
# ==========================================

def get_alert_events(limit=50):

    db = SessionLocal()

    try:

        rows = (
            db.query(Event)
            .filter(
                Event.risk_level.in_(
                    [
                        "CRITICAL",
                        "HIGH",
                        "MEDIUM",
                    ]
                )
            )
            .order_by(desc(Event.timestamp))
            .limit(limit)
            .all()
        )

        return rows

    finally:

        db.close()


# ==========================================
# FILTERING HELPERS
# ==========================================

def get_events_by_camera(camera_id, limit=100):

    db = SessionLocal()

    try:

        rows = (
            db.query(Event)
            .filter(Event.camera_id == camera_id)
            .order_by(desc(Event.timestamp))
            .limit(limit)
            .all()
        )

        return rows

    finally:

        db.close()


def get_events_by_type(event_type: str, limit=100):

    db = SessionLocal()

    try:

        rows = (
            db.query(Event)
            .filter(Event.event_type == str(event_type).upper())
            .order_by(desc(Event.timestamp))
            .limit(limit)
            .all()
        )

        return rows

    finally:

        db.close()


# ==========================================
# PHASE 7 — RICH HISTORY FORMAT
# ==========================================

def _infer_violations(event) -> list[str]:
    """Derive a human-readable list of violations from an event row.

    If the event already has a persisted `violations` value (Change 6 in
    the review — saved directly at write time instead of reconstructed),
    use that instead of re-inferring it from the raw counters.
    """
    saved = getattr(event, "violations", None)
    if saved:
        return list(saved)

    violations = []
    if getattr(event, "missing_helmet", 0):
        violations.append("Helmet Missing")
    if getattr(event, "missing_vest", 0):
        violations.append("Vest Missing")
    if getattr(event, "missing_boots", 0):
        violations.append("Boot Missing")
    if getattr(event, "danger_zones", 0):
        violations.append("Danger Zone")
    if getattr(event, "cracks", 0):
        violations.append("Crack Detected")
    return violations


def _infer_risk_score(event) -> int:
    """Compute a 0-100 risk score from the event's violation profile.

    If the event already has a persisted `risk_score` (Change 4 — saved at
    write time from the upstream risk engine), use that instead, so this
    service's scoring stays in sync with `rules.py` rather than recomputing
    its own slightly different number.
    """
    saved = getattr(event, "risk_score", None)
    if saved:
        return min(int(saved), 100)

    penalties = {
        "missing_helmet": 25,
        "missing_vest":   20,
        "missing_boots":  15,
        "danger_zones":   30,
        "cracks":         40,
    }
    score = sum(
        penalties[field]
        for field in penalties
        if getattr(event, field, 0)
    )
    return min(score, 100)


def _infer_corrective_action(event) -> str:
    """Suggest a corrective action based on the event's violation profile."""
    if getattr(event, "risk_level", None) == "CRITICAL":
        return "Stop Work Immediately."
    if getattr(event, "danger_zones", 0):
        return "Remove worker from danger zone immediately."
    if getattr(event, "cracks", 0):
        return "Halt operations; arrange structural inspection."
    missing = []
    if getattr(event, "missing_helmet", 0):
        missing.append("helmet")
    if getattr(event, "missing_vest", 0):
        missing.append("vest")
    if getattr(event, "missing_boots", 0):
        missing.append("boots")
    if missing:
        return f"Provide PPE immediately: {', '.join(missing)}."
    return "No corrective action required."


def _compute_worker_compliance_pct(event) -> float | None:
    """Return per-worker compliance percentage from an event row."""
    workers = getattr(event, "workers", 0) or 0
    if not workers:
        return None
    compliant = getattr(event, "compliant_workers", 0) or 0
    return round((compliant / workers) * 100, 1)


def format_event_for_history(event) -> dict:
    """
    Convert a raw Event ORM row into the Phase-7 rich history dict:

        {
            "id":                int,
            "worker":            str,          # e.g. "Worker 4"
            "camera":            str,          # e.g. "Gate 2"
            "violations":        list[str],    # ["Helmet Missing", "Vest Missing"]
            "compliance_pct":    float | None, # per-worker compliance %
            "risk":              str,          # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "SAFE"
            "risk_score":        int,          # 0-100
            "priority":          int,          # 1 (most urgent) .. 5 (least urgent)
            "time":              str,          # "HH:MM:SS"
            "timestamp":         str,          # ISO-8601
            "corrective_action": str,
            "recommended_action": str,         # currently mirrors corrective_action;
                                                # intended to be supplied by
                                                # alert_service.py in the future
            "event_type":        str,
            "description":       str,
            "image_path":        str | None,
        }
    """
    worker_id  = getattr(event, "worker_id", None)
    if worker_id:
        worker_str = f"Worker {worker_id}"
    else:
        cam_id_fallback = getattr(event, "camera_id", None)
        worker_str = f"Worker {cam_id_fallback or '?'}"

    cam_id     = getattr(event, "camera_id", None)
    camera_str = f"Camera {cam_id}" if cam_id else "Unknown Camera"

    violations  = _infer_violations(event)
    risk_score  = _infer_risk_score(event)
    compliance  = getattr(event, "compliance_pct", None)
    if compliance is None:
        compliance = _compute_worker_compliance_pct(event)

    # If compliance still cannot be derived, estimate from score
    if compliance is None and risk_score is not None:
        compliance = round(max(0.0, 100.0 - risk_score), 1)

    # Promote risk to CRITICAL when score is very high
    risk = event.risk_level
    if risk_score >= 75 and risk != "CRITICAL":
        risk = "CRITICAL"

    corrective_action = _infer_corrective_action(event)

    return {
        "id":                event.id,
        "worker":            worker_str,
        "camera":            camera_str,
        "violations":        violations,
        "compliance_pct":    compliance,
        "risk":              risk,
        "risk_score":        risk_score,
        "priority":          RISK_PRIORITY.get(risk, 5),
        "time":              event.timestamp.strftime("%H:%M:%S"),
        "timestamp":         event.timestamp.isoformat(),
        "corrective_action": corrective_action,
        "recommended_action": corrective_action,
        "event_type":        event.event_type,
        "description":       event.description or "",
        "image_path":        getattr(event, "image_path", None),
    }


def get_rich_event_history(limit: int = 100) -> list[dict]:
    """
    Return the latest events formatted for the Phase-7 History view.
    Each row includes worker identity, violations list, compliance %,
    risk score, and a corrective action — replacing the old single
    `description` field.
    """
    rows = get_latest_events(limit=limit)
    return [format_event_for_history(e) for e in rows]


def get_rich_alert_history(limit: int = 50) -> list[dict]:
    """
    Same as get_rich_event_history but restricted to CRITICAL/HIGH/MEDIUM
    events (the same set exposed by get_alert_events).
    """
    rows = get_alert_events(limit=limit)
    return [format_event_for_history(e) for e in rows]


def get_events_by_risk_level(risk_level: str, limit=100):

    db = SessionLocal()

    try:

        rows = (
            db.query(Event)
            .filter(Event.risk_level == str(risk_level).upper())
            .order_by(desc(Event.timestamp))
            .limit(limit)
            .all()
        )

        return rows

    finally:

        db.close()


# ==========================================
# DATE FILTERING
# ==========================================
# Assumes Event.timestamp is a datetime column (already relied on by the
# history helpers above via desc(Event.timestamp)).

def _day_bounds(reference: datetime) -> tuple[datetime, datetime]:
    """Return (start, end) datetimes covering the calendar day of `reference`."""
    start = reference.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start, end


def get_events_for_date_range(start: datetime, end: datetime, limit=500):
    """Return events with start <= timestamp < end, most recent first.

    Both `start` and `end` must be datetime objects. Caller is responsible
    for timezone consistency with however Event.timestamp is stored.
    """

    db = SessionLocal()

    try:

        rows = (
            db.query(Event)
            .filter(Event.timestamp >= start, Event.timestamp < end)
            .order_by(desc(Event.timestamp))
            .limit(limit)
            .all()
        )

        return rows

    finally:

        db.close()


def get_events_today(limit=500):
    start, end = _day_bounds(datetime.now())
    return get_events_for_date_range(start, end, limit=limit)


def get_events_yesterday(limit=500):
    start, end = _day_bounds(datetime.now() - timedelta(days=1))
    return get_events_for_date_range(start, end, limit=limit)


def get_events_last_n_days(days: int = 7, limit=500):
    """Events from `days` ago (inclusive, start of that day) through now."""
    end = datetime.now()
    start, _ = _day_bounds(end - timedelta(days=days - 1))
    return get_events_for_date_range(start, end, limit=limit)


# ==========================================
# SEARCH
# ==========================================

def search_events(
    keyword: str = None,
    risk_level: str = None,
    camera_id=None,
    event_type: str = None,
    worker_id=None,
    limit=100,
):
    """Flexible search over events. All filters are optional and combined
    with AND. `keyword` does a case-insensitive substring match against
    the description field, and — when the corresponding columns exist on
    the Event model — also against `violations` and `corrective_action`.
    """

    db = SessionLocal()

    try:

        query = db.query(Event)

        if keyword:
            keyword_filters = [Event.description.ilike(f"%{keyword}%")]
            # `violations` may be stored as JSON/array rather than text,
            # in which case `.ilike` isn't valid SQL for that column — only
            # add it if it behaves like a string column.
            if hasattr(Event, "violations"):
                try:
                    keyword_filters.append(Event.violations.ilike(f"%{keyword}%"))
                except (AttributeError, TypeError):
                    pass
            if hasattr(Event, "corrective_action"):
                keyword_filters.append(Event.corrective_action.ilike(f"%{keyword}%"))
            query = query.filter(func.or_(*keyword_filters))

        if risk_level:
            query = query.filter(Event.risk_level == str(risk_level).upper())

        if camera_id is not None:
            query = query.filter(Event.camera_id == camera_id)

        if event_type:
            query = query.filter(Event.event_type == str(event_type).upper())

        if worker_id is not None and hasattr(Event, "worker_id"):
            query = query.filter(Event.worker_id == worker_id)

        rows = (
            query
            .order_by(desc(Event.timestamp))
            .limit(limit)
            .all()
        )

        return rows

    finally:

        db.close()


# ==========================================
# STATISTICS
# ==========================================

def _count_since(db, since: datetime) -> int:
    return (
        db.query(func.count(Event.id))
        .filter(Event.timestamp >= since)
        .scalar()
        or 0
    )


def get_risk_counts(since: datetime = None) -> dict:
    """Counts grouped by risk_level, optionally restricted to events at or
    after `since`."""

    db = SessionLocal()

    try:

        query = db.query(Event.risk_level, func.count(Event.id))

        if since is not None:
            query = query.filter(Event.timestamp >= since)

        rows = query.group_by(Event.risk_level).all()

        return {risk_level: count for risk_level, count in rows}

    finally:

        db.close()


def get_camera_counts(since: datetime = None) -> dict:
    """Counts grouped by camera_id, optionally restricted to events at or
    after `since`."""

    db = SessionLocal()

    try:

        query = db.query(Event.camera_id, func.count(Event.id))

        if since is not None:
            query = query.filter(Event.timestamp >= since)

        rows = query.group_by(Event.camera_id).all()

        return {camera_id: count for camera_id, count in rows}

    finally:

        db.close()


def get_type_counts(since: datetime = None) -> dict:
    """Counts grouped by event_type, optionally restricted to events at or
    after `since`."""

    db = SessionLocal()

    try:

        query = db.query(Event.event_type, func.count(Event.id))

        if since is not None:
            query = query.filter(Event.timestamp >= since)

        rows = query.group_by(Event.event_type).all()

        return {event_type: count for event_type, count in rows}

    finally:

        db.close()


def get_event_statistics() -> dict:
    """Single call returning the common counters dashboards/analytics need:
    today / this week / this month totals, plus breakdowns by risk level,
    camera, and event type (all-time breakdowns, not just this month).
    """

    now = datetime.now()
    today_start, _ = _day_bounds(now)
    week_start, _ = _day_bounds(now - timedelta(days=7))
    month_start, _ = _day_bounds(now - timedelta(days=30))

    db = SessionLocal()

    try:

        today_count = _count_since(db, today_start)
        week_count = _count_since(db, week_start)
        month_count = _count_since(db, month_start)

    finally:

        db.close()

    risk_counts = get_risk_counts()

    return {
        "today": today_count,
        "this_week": week_count,
        "this_month": month_count,
        "by_risk": {
            "critical": risk_counts.get("CRITICAL", 0),
            "high": risk_counts.get("HIGH", 0),
            "medium": risk_counts.get("MEDIUM", 0),
            "low": risk_counts.get("LOW", 0),
            "safe": risk_counts.get("SAFE", 0),
        },
        "by_camera": get_camera_counts(),
        "by_type": get_type_counts(),
    }


# ==========================================
# DASHBOARD SUMMARY
# ==========================================

def get_dashboard_events(latest_limit=20, alert_limit=10) -> dict:
    """One call that bundles together what a dashboard view typically
    needs, so callers don't have to issue several separate queries.
    """

    latest = get_latest_events(limit=latest_limit)
    alerts = get_alert_events(limit=alert_limit)
    risk_counts = get_risk_counts()

    db = SessionLocal()

    try:
        total_count = db.query(func.count(Event.id)).scalar() or 0
    finally:
        db.close()

    return {
        "latest_events": latest,
        "alert_events": alerts,
        "alert_count": len(alerts),
        "total_event_count": total_count,
        "risk_counts": {
            "critical": risk_counts.get("CRITICAL", 0),
            "high": risk_counts.get("HIGH", 0),
            "medium": risk_counts.get("MEDIUM", 0),
            "low": risk_counts.get("LOW", 0),
            "safe": risk_counts.get("SAFE", 0),
        },
        "last_incident": latest[0] if latest else None,
    }