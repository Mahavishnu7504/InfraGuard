from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RiskLevel(Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class AlertCategory(Enum):
    PPE         = "PPE"
    CRACK       = "CRACK"
    EQUIPMENT   = "EQUIPMENT"
    DANGER_ZONE = "DANGER_ZONE"
    NEAR_MISS   = "NEAR_MISS"
    SYSTEM      = "SYSTEM"


class AlertStatus(Enum):
    ACTIVE       = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED     = "resolved"


# ---------------------------------------------------------------------------
# Priority mapping  (lower number = more urgent)
# ---------------------------------------------------------------------------

RISK_PRIORITY: Dict[RiskLevel, int] = {
    RiskLevel.CRITICAL: 0,
    RiskLevel.HIGH:     1,
    RiskLevel.MEDIUM:   2,
    RiskLevel.LOW:      3,
}

ALERTABLE_RISKS: Set[RiskLevel] = {
    RiskLevel.CRITICAL,
    RiskLevel.HIGH,
    RiskLevel.MEDIUM,
}

# Escalation ladder: after N seconds at a level, promote to next level.
ESCALATION_LADDER: Dict[RiskLevel, Optional[RiskLevel]] = {
    RiskLevel.MEDIUM:   RiskLevel.HIGH,
    RiskLevel.HIGH:     RiskLevel.CRITICAL,
    RiskLevel.CRITICAL: None,   # already at the top
    RiskLevel.LOW:      None,
}

# Default thresholds (seconds) before escalation triggers.
DEFAULT_ESCALATION_THRESHOLDS: Dict[RiskLevel, int] = {
    RiskLevel.MEDIUM: 30,
    RiskLevel.HIGH:   60,
}

# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------

# Maps (category, event_type keyword) → recommended action.
# Falls back to category-level defaults when no keyword matches.
_RECOMMENDATIONS: List[tuple[AlertCategory, str, str]] = [
    (AlertCategory.PPE,         "helmet",   "Provide a safety helmet immediately before work resumes."),
    (AlertCategory.PPE,         "vest",     "Issue a high-visibility vest before re-entering the work zone."),
    (AlertCategory.PPE,         "glove",    "Supply cut-resistant gloves before the task continues."),
    (AlertCategory.PPE,         "boot",     "Replace footwear with steel-toed safety boots."),
    (AlertCategory.PPE,         "goggle",   "Provide protective eyewear before work resumes."),
    (AlertCategory.CRACK,       "crack",    "Halt operations in this area and notify the structural engineer for assessment."),
    (AlertCategory.EQUIPMENT,   "excavat",  "Enforce minimum safe operating distance from excavator."),
    (AlertCategory.EQUIPMENT,   "crane",    "Clear the lift radius; verify load capacity before continuing."),
    (AlertCategory.EQUIPMENT,   "forklift", "Keep pedestrians 3 m clear; check aisle barriers."),
    (AlertCategory.DANGER_ZONE, "",         "Remove personnel from the danger zone immediately."),
    (AlertCategory.NEAR_MISS,   "",         "Review near-miss report with site supervisor; update hazard register."),
    (AlertCategory.SYSTEM,      "",         "Check camera feed and sensor connectivity."),
]

_CATEGORY_DEFAULTS: Dict[AlertCategory, str] = {
    AlertCategory.PPE:         "Ensure full PPE compliance before re-entering the work zone.",
    AlertCategory.CRACK:       "Isolate the affected area and arrange structural inspection.",
    AlertCategory.EQUIPMENT:   "Maintain safe distance from heavy equipment at all times.",
    AlertCategory.DANGER_ZONE: "Remove all non-essential personnel from the area.",
    AlertCategory.NEAR_MISS:   "Document the incident and review with site safety officer.",
    AlertCategory.SYSTEM:      "Check system health and camera connectivity.",
}


def get_recommended_action(category: AlertCategory, event_type: str) -> str:
    """Return the most specific recommendation for a given category + event type."""
    et_lower = event_type.lower()
    for cat, keyword, action in _RECOMMENDATIONS:
        if cat == category and keyword and keyword in et_lower:
            return action
    return _CATEGORY_DEFAULTS.get(category, "Follow site safety procedures.")


# ---------------------------------------------------------------------------
# Alert data class
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    # Identity
    alert_id:   str = field(default_factory=lambda: str(uuid.uuid4()))
    camera_id:  Optional[str] = None
    worker_id:  Optional[str] = None
    track_id:   Optional[str] = None

    # Classification
    category:   AlertCategory = AlertCategory.SYSTEM
    event_type: str = ""
    severity:   RiskLevel = RiskLevel.MEDIUM
    risk_score: float = 0.0

    # Human-readable
    title:              str = ""
    description:        str = ""
    recommended_action: str = ""

    # Lifecycle
    status:     AlertStatus = AlertStatus.ACTIVE
    image_path: Optional[str] = None

    # Timestamps (Unix epoch, float)
    created_at:      float = field(default_factory=time.time)
    updated_at:      float = field(default_factory=time.time)
    acknowledged_at: Optional[float] = None
    resolved_at:     Optional[float] = None
    last_seen_at:    float = field(default_factory=time.time)

    # Audit trail: list of {"event": str, "severity": str, "ts": float}
    history: List[dict] = field(default_factory=list)

    # Internal
    _escalation_started_at: float = field(default_factory=time.time, repr=False)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dict for API responses / WebSocket."""
        return {
            "alert_id":           self.alert_id,
            "camera_id":          self.camera_id,
            "worker_id":          self.worker_id,
            "track_id":           self.track_id,
            "category":           self.category.value,
            "event_type":         self.event_type,
            "severity":           self.severity.value,
            "risk_score":         round(self.risk_score, 3),
            "priority":           RISK_PRIORITY[self.severity],
            "title":              self.title,
            "description":        self.description,
            "recommended_action": self.recommended_action,
            "status":             self.status.value,
            "image_path":         self.image_path,
            "created_at":         self.created_at,
            "updated_at":         self.updated_at,
            "acknowledged_at":    self.acknowledged_at,
            "resolved_at":        self.resolved_at,
            "last_seen_at":       self.last_seen_at,
            "history":            self.history,
        }

    # ------------------------------------------------------------------
    def _record(self, event: str) -> None:
        self.history.append({
            "event":    event,
            "severity": self.severity.value,
            "ts":       time.time(),
        })
        self.updated_at = time.time()

    def acknowledge(self) -> None:
        if self.status == AlertStatus.ACTIVE:
            self.status = AlertStatus.ACKNOWLEDGED
            self.acknowledged_at = time.time()
            self._record("acknowledged")

    def resolve(self) -> None:
        if self.status != AlertStatus.RESOLVED:
            self.status = AlertStatus.RESOLVED
            self.resolved_at = time.time()
            self._record("resolved")

    def escalate(self, new_severity: RiskLevel) -> None:
        self._record(f"escalated from {self.severity.value} to {new_severity.value}")
        self.severity = new_severity
        self._escalation_started_at = time.time()
        self.status = AlertStatus.ACTIVE   # re-activate if it was acknowledged


# ---------------------------------------------------------------------------
# Suppression key helpers
# ---------------------------------------------------------------------------

def _suppression_key(
    camera_id: Optional[str],
    worker_id: Optional[str],
    track_id:  Optional[str],
    event_type: str,
) -> str:
    """Unique key that identifies 'the same violation still happening'."""
    return f"{camera_id}|{worker_id}|{track_id}|{event_type}"


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class AlertManager:
    

    def __init__(
        self,
        cooldown: int = 10,
        escalation_thresholds: Optional[Dict[RiskLevel, int]] = None,
        broadcast_fn: Optional[Callable] = None,
        stale_timeout: int = 120,
    ):
        self.cooldown   = cooldown
        self.stale_timeout = stale_timeout
        self.broadcast_fn  = broadcast_fn

        self.escalation_thresholds: Dict[RiskLevel, int] = {
            **DEFAULT_ESCALATION_THRESHOLDS,
            **(escalation_thresholds or {}),
        }

        # suppression_key → Alert  (only ACTIVE / ACKNOWLEDGED alerts live here)
        self._active: Dict[str, Alert] = {}

        # Full history (including resolved alerts)
        self._history: List[Alert] = []

    # ------------------------------------------------------------------
    # Backward Compatibility
    # ------------------------------------------------------------------

    def should_alert(self, key: str, risk) -> bool:
        """
        Compatibility method for older safety_pipeline.py versions.

        The new AlertManager handles alert lifecycle through
        process_detection(), but older pipeline code still calls
        should_alert(). This method preserves cooldown behaviour.
        """

        now = time.time()

        if not hasattr(self, "_legacy_last_alert"):
            self._legacy_last_alert = {}

        last = self._legacy_last_alert.get(key)

        if last is None:
            self._legacy_last_alert[key] = now
            return True

        if (now - last) >= self.cooldown:
            self._legacy_last_alert[key] = now
            return True

        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_detection(
        self,
        *,
        event_type:  str,
        category:    AlertCategory,
        severity:    RiskLevel,
        risk_score:  float        = 0.0,
        camera_id:   Optional[str] = None,
        worker_id:   Optional[str] = None,
        track_id:    Optional[str] = None,
        title:       str           = "",
        description: str           = "",
        image_path:  Optional[str] = None,
    ) -> Optional[Alert]:
        """
        Call once per detection frame.  Returns the Alert object if an alert
        was created or updated, or None for LOW-risk / suppressed detections.
        """
        if severity not in ALERTABLE_RISKS:
            return None

        key = _suppression_key(camera_id, worker_id, track_id, event_type)
        now = time.time()

        existing = self._active.get(key)

        if existing is not None:
            # Refresh last-seen; keep the alert alive.
            existing.last_seen_at = now
            existing.updated_at   = now
            if image_path:
                existing.image_path = image_path
            # Allow upgrading severity if the live detection is worse.
            if RISK_PRIORITY[severity] < RISK_PRIORITY[existing.severity]:
                existing.escalate(severity)
                self._broadcast(existing, event="severity_upgraded")
            return existing

        # Check cooldown against recently resolved alerts for the same key.
        recent = self._last_resolved(key)
        if recent is not None and (now - recent.resolved_at) < self.cooldown:
            return None

        # Create a fresh alert.
        action = get_recommended_action(category, event_type)
        alert = Alert(
            camera_id          = camera_id,
            worker_id          = worker_id,
            track_id           = track_id,
            category           = category,
            event_type         = event_type,
            severity           = severity,
            risk_score         = risk_score,
            title              = title or self._default_title(category, event_type),
            description        = description,
            recommended_action = action,
            image_path         = image_path,
        )
        alert._record("created")
        self._active[key] = alert
        self._history.append(alert)
        self._broadcast(alert, event="created")
        return alert

    def acknowledge(self, alert_id: str) -> Optional[Alert]:
        """Mark an alert as acknowledged by an operator."""
        alert = self._find_active(alert_id)
        if alert:
            alert.acknowledge()
            self._broadcast(alert, event="acknowledged")
        return alert

    def resolve(self, alert_id: str) -> Optional[Alert]:
        """Manually resolve an alert (e.g. operator confirmed PPE is now worn)."""
        alert = self._find_active(alert_id)
        if alert:
            key = self._key_for(alert)
            alert.resolve()
            self._active.pop(key, None)
            self._broadcast(alert, event="resolved")
        return alert

    def tick(self) -> None:
        """
        Call periodically (e.g. every second) to:
        - Auto-escalate long-running alerts.
        - Auto-resolve stale alerts (detection gone from scene).
        """
        now  = time.time()
        dead = []

        for key, alert in self._active.items():
            # Auto-resolve stale alerts.
            if (now - alert.last_seen_at) > self.stale_timeout:
                alert.resolve()
                dead.append(key)
                self._broadcast(alert, event="auto_resolved")
                continue

            # Escalation (only for ACTIVE alerts).
            if alert.status != AlertStatus.ACTIVE:
                continue

            next_level = ESCALATION_LADDER.get(alert.severity)
            if next_level is None:
                continue   # already CRITICAL or LOW

            threshold = self.escalation_thresholds.get(alert.severity)
            if threshold and (now - alert._escalation_started_at) >= threshold:
                alert.escalate(next_level)
                self._broadcast(alert, event="escalated")

        for key in dead:
            self._active.pop(key, None)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def active_alerts(
        self,
        camera_id: Optional[str] = None,
        category:  Optional[AlertCategory] = None,
        severity:  Optional[RiskLevel] = None,
    ) -> List[dict]:
        """Return sorted list of active alert dicts (most urgent first)."""
        results = list(self._active.values())
        if camera_id:
            results = [a for a in results if a.camera_id == camera_id]
        if category:
            results = [a for a in results if a.category == category]
        if severity:
            results = [a for a in results if a.severity == severity]
        results.sort(key=lambda a: (RISK_PRIORITY[a.severity], a.created_at))
        return [a.to_dict() for a in results]

    def alert_history(
        self,
        limit: int = 100,
        camera_id: Optional[str] = None,
    ) -> List[dict]:
        """Return recent alerts (newest first)."""
        results = list(reversed(self._history))
        if camera_id:
            results = [a for a in results if a.camera_id == camera_id]
        return [a.to_dict() for a in results[:limit]]

    @staticmethod
    def priority(risk: RiskLevel) -> int:
        """Lower value = more urgent. Use as a sort key."""
        return RISK_PRIORITY[risk]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _broadcast(self, alert: Alert, event: str = "") -> None:
        if self.broadcast_fn is None:
            return
        payload = alert.to_dict()
        payload["ws_event"] = event
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.broadcast_fn(payload))
            else:
                loop.run_until_complete(self.broadcast_fn(payload))
        except Exception:
            pass   # Never let broadcast errors crash the detection pipeline.

    def _find_active(self, alert_id: str) -> Optional[Alert]:
        for alert in self._active.values():
            if alert.alert_id == alert_id:
                return alert
        return None

    def _key_for(self, alert: Alert) -> Optional[str]:
        for k, v in self._active.items():
            if v is alert:
                return k
        return None

    def _last_resolved(self, key: str) -> Optional[Alert]:
        """Find the most recently resolved alert for this suppression key."""
        candidates = [
            a for a in self._history
            if a.status == AlertStatus.RESOLVED
            and self._key_for_history(a) == key
            and a.resolved_at is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda a: a.resolved_at)

    @staticmethod
    def _key_for_history(alert: Alert) -> str:
        return _suppression_key(
            alert.camera_id, alert.worker_id, alert.track_id, alert.event_type
        )

    @staticmethod
    def _default_title(category: AlertCategory, event_type: str) -> str:
        return event_type.replace("_", " ").title() or category.value.replace("_", " ").title()