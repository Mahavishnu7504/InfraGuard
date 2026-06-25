import logging

from backend.core.database import SessionLocal
from backend.models.db_models import Event
from sqlalchemy import desc

logger = logging.getLogger(__name__)

VALID_EVENT_TYPES = {"PPE_DETECTION", "DANGER_ZONE", "CRACK_DETECTION"}
VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}


# ==========================================
# VALIDATION
# ==========================================

def validate_event_data(event_data: dict) -> list[str]:
    """Return a list of validation error messages (empty = valid)."""

    errors = []

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
        if not isinstance(value, (int, float)) or value < 0:
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
# SAVE EVENT
# ==========================================

def save_event(event_data: dict):

    errors = validate_event_data(event_data)
    if errors:
        for error in errors:
            logger.warning("[EVENT VALIDATION] %s", error)
        return None

    db = SessionLocal()

    try:

        event = Event(

            event_type=event_data.get("event_type", "PPE_DETECTION"),

            risk_level=event_data.get("risk_level", "LOW").upper(),

            description=(
                event_data.get("description")
                or infer_description(event_data)
            ),

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