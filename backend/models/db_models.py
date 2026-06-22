from sqlalchemy import Column, Integer, String, DateTime, Text, Index
from datetime import datetime, timezone
from backend.core.database import Base


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)

    # Core info
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    event_type = Column(String, index=True)
    risk_level = Column(String, index=True)
    description = Column(Text, nullable=True)

    # Context
    camera_id = Column(Integer, default=0, index=True)
    image_path = Column(String, nullable=True)

    # PPE summary
    helmet = Column(Integer, default=0)
    vest = Column(Integer, default=0)
    boots = Column(Integer, default=0)

    # Risk counts
    low = Column(Integer, default=0)
    medium = Column(Integer, default=0)
    high = Column(Integer, default=0)

    # Worker intelligence
    workers = Column(Integer, default=0)
    compliant_workers = Column(Integer, default=0)
    violating_workers = Column(Integer, default=0)

    missing_helmet = Column(Integer, default=0)
    missing_vest = Column(Integer, default=0)
    missing_boots = Column(Integer, default=0)

    # Hazard intelligence
    danger_zones = Column(Integer, default=0)
    machines = Column(Integer, default=0)
    cracks = Column(Integer, default=0)


Index("idx_event_time_camera", Event.timestamp, Event.camera_id)
Index("idx_event_risk", Event.risk_level)
Index("idx_event_type", Event.event_type)