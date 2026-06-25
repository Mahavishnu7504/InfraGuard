import logging
import threading
import time

from backend.core.database import SessionLocal
from backend.models.db_models import Event
from sqlalchemy import desc

logger = logging.getLogger(__name__)

VALID_EVENT_TYPES = {"PPE_DETECTION", "DANGER_ZONE", "CRACK_DETECTION"}
VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}

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
            return f"Missing {', '.join(missing)}."

        return "Worker compliant with PPE."

    return "AI safety event recorded."


# ==========================================
# DUPLICATE SUPPRESSION
# ==========================================
# Live camera feeds can re-fire the same detection many times per second.
# We suppress identical (camera, event_type, description) combos within a
# short rolling window so the DB only gets one row per "incident", instead
# of one row per frame.

_recent_events_lock = threading.Lock()
_recent_events: dict[tuple, float] = {}


def _dedup_key(event_data: dict, description: str) -> tuple:
    return (
        event_data.get("camera_id", 0),
        str(event_data.get("event_type", "PPE_DETECTION")).upper(),
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

        event = Event(

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

        db.add(event)
        db.commit()
        db.refresh(event)

        logger.info(
            "[EVENT SAVED] id=%s type=%s risk=%s camera=%s",
            event.id,
            event.event_type,
            event.risk_level,
            event.camera_id,
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
            .filter(Event.risk_level.in_(["HIGH", "MEDIUM"]))
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