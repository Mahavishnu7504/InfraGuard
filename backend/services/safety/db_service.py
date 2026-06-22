# =========================================================
# INFRA GUARD — DB SERVICE
# FIX: was importing SafetyLog which doesn't exist.
#      Now correctly uses Event from db_models.
# =========================================================

from backend.core.database import SessionLocal
from backend.models.db_models import Event


def save_safety_log(data: dict):

    db = SessionLocal()

    try:

        event = Event(

            event_type=data.get(
                "event_type",
                "PPE_DETECTION"
            ),

            risk_level=data.get(
                "risk_level",
                "LOW"
            ).upper(),

            low=data.get("low", 0),
            medium=data.get("medium", 0),
            high=data.get("high", 0),

            helmet=data.get("helmet", 0),
            vest=data.get("vest", 0),
            boots=data.get("boots", 0),

            camera_id=data.get("camera_id", 0),
        )

        db.add(event)
        db.commit()

    except Exception as e:

        db.rollback()
        print("[DB SERVICE ERROR]", e)

    finally:

        db.close()