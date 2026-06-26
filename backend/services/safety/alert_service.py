from __future__ import annotations

import logging
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RiskLevel(Enum):
    SAFE     = "safe"
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
    RiskLevel.SAFE:     4,
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
    RiskLevel.SAFE:     None,
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
    camera_name: Optional[str] = None
    worker_id:  Optional[str] = None
    track_id:   Optional[str] = None

    # Classification
    category:   AlertCategory = AlertCategory.SYSTEM
    event_type: str = ""
    severity:   RiskLevel = RiskLevel.MEDIUM
    risk_score: float = 0.0
    compliance_score: Optional[float] = None

    # Human-readable
    title:              str = ""
    description:        str = ""
    recommended_action: str = ""

    # Phase 6: worker-level enrichment
    violations:         List[str] = field(default_factory=list)   # e.g. ["Helmet Missing", "Vest Missing"]
    missing_ppe_count:  int = 0                                    # len(violations) for dashboard widgets
    nearby_equipment:   Optional[str] = None                       # e.g. "Near Excavator"
    compliance_pct:     Optional[float] = None                     # e.g. 25.0
    suggested_action:   str = ""                                   # e.g. "Stop Work"
    zone_name:          Optional[str] = None                       # e.g. "Excavator Zone"

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
    history_version: int = 1

    # Internal
    _escalation_started_at: float = field(default_factory=time.time, repr=False)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dict for API responses / WebSocket."""
        return {
            "alert_id":           self.alert_id,
            "camera_id":          self.camera_id,
            "camera_name":        self.camera_name,
            "worker_id":          self.worker_id,
            "track_id":           self.track_id,
            "category":           self.category.value,
            "event_type":         self.event_type,
            "severity":           self.severity.value,
            "risk_score":         round(self.risk_score, 3),
            "compliance_score":   round(self.compliance_score, 3) if self.compliance_score is not None else None,
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
            "history_version":    self.history_version,
            # Phase 6: worker-level detail
            "violations":         self.violations,
            "missing_ppe_count":  self.missing_ppe_count,
            "nearby_equipment":   self.nearby_equipment,
            "compliance_pct":     self.compliance_pct,
            "suggested_action":   self.suggested_action,
            "zone_name":          self.zone_name,
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
        event_type:       str,
        category:         AlertCategory,
        severity:         RiskLevel,
        risk_score:       float             = 0.0,
        camera_id:        Optional[str]     = None,
        camera_name:      Optional[str]     = None,
        worker_id:        Optional[str]     = None,
        track_id:         Optional[str]     = None,
        title:            str               = "",
        description:      str               = "",
        image_path:       Optional[str]     = None,
        # Phase 6: worker-level detail
        violations:       Optional[List[str]] = None,
        nearby_equipment: Optional[str]     = None,
        compliance_pct:   Optional[float]   = None,
        compliance_score: Optional[float]   = None,
        suggested_action: str               = "",
        zone_name:        Optional[str]     = None,
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
        resolved_violations = violations or []
        missing_ppe_count = len(resolved_violations)
        action = get_recommended_action(category, event_type)
        # Change 16: override recommended action with Stop Work for CRITICAL severity.
        if severity == RiskLevel.CRITICAL:
            action = "Stop Work immediately and remove all personnel from the hazard area."

        # Change 17: generate richer title for CRITICAL alerts.
        default_title = self._default_title(category, event_type)
        if severity == RiskLevel.CRITICAL:
            default_title = f"CRITICAL {category.value} VIOLATION"

        # Change 18: enrich description with compliance and risk score.
        enriched_description = description
        if not enriched_description:
            parts = []
            if compliance_pct is not None:
                parts.append(f"Compliance: {compliance_pct:.0f}%")
            if risk_score:
                parts.append(f"Risk Score: {risk_score:.0f}")
            enriched_description = " | ".join(parts) if parts else ""

        alert = Alert(
            camera_id          = camera_id,
            camera_name        = camera_name,
            worker_id          = worker_id,
            track_id           = track_id,
            category           = category,
            event_type         = event_type,
            severity           = severity,
            risk_score         = risk_score,
            compliance_score   = compliance_score,
            title              = title or default_title,
            description        = enriched_description,
            recommended_action = action,
            image_path         = image_path,
            # Phase 6
            violations         = resolved_violations,
            missing_ppe_count  = missing_ppe_count,
            nearby_equipment   = nearby_equipment,
            compliance_pct     = compliance_pct,
            suggested_action   = suggested_action or action,
            zone_name          = zone_name,
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
    # Enterprise Enhancement 1: Alert Statistics
    # ------------------------------------------------------------------

    def get_alert_statistics(self) -> dict:
        """
        Aggregate counts across the full alert history (active + resolved).

        Returns a flat dict suitable for powering Dashboard / Analytics pages:
            {
                "total_alerts":  int,
                "active":        int,
                "acknowledged":  int,
                "resolved":      int,
                "critical":      int,
                "high":          int,
                "medium":        int,
                "low":           int,
            }
        """
        all_alerts = self._all_alerts()

        status_counts = Counter(a.status for a in all_alerts)
        severity_counts = Counter(a.severity for a in all_alerts)

        return {
            "total_alerts":  len(all_alerts),
            "active":        status_counts.get(AlertStatus.ACTIVE, 0),
            "acknowledged":  status_counts.get(AlertStatus.ACKNOWLEDGED, 0),
            "resolved":      status_counts.get(AlertStatus.RESOLVED, 0),
            "critical":      severity_counts.get(RiskLevel.CRITICAL, 0),
            "high":          severity_counts.get(RiskLevel.HIGH, 0),
            "medium":        severity_counts.get(RiskLevel.MEDIUM, 0),
            "low":           severity_counts.get(RiskLevel.LOW, 0),
            "safe":          severity_counts.get(RiskLevel.SAFE, 0),
        }

    # ------------------------------------------------------------------
    # Enterprise Enhancement 2: Dashboard Summary
    # ------------------------------------------------------------------

    def get_dashboard_alert_summary(self) -> dict:
        """
        Compact summary tailored for a dashboard "at a glance" widget.

        Returns:
            {
                "open_alerts":           int,   # active + acknowledged
                "critical_alerts":       int,   # currently open AND critical
                "resolved_today":        int,
                "acknowledged_today":    int,
                "newest_alert":          dict | None,
            }
        """
        all_alerts = self._all_alerts()
        today_start = self._start_of_day()

        open_alerts = [
            a for a in all_alerts
            if a.status in (AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED)
        ]
        critical_open = [a for a in open_alerts if a.severity == RiskLevel.CRITICAL]

        resolved_today = [
            a for a in all_alerts
            if a.resolved_at is not None and a.resolved_at >= today_start
        ]
        acknowledged_today = [
            a for a in all_alerts
            if a.acknowledged_at is not None and a.acknowledged_at >= today_start
        ]

        newest_alert = max(all_alerts, key=lambda a: a.created_at, default=None)

        critical_alerts = [a for a in all_alerts if a.severity == RiskLevel.CRITICAL]
        highest_priority_alert = (
            min(critical_alerts, key=lambda a: a.created_at)
            if critical_alerts
            else min(
                open_alerts,
                key=lambda a: (RISK_PRIORITY[a.severity], a.created_at),
                default=None,
            )
        )

        return {
            "open_alerts":             len(open_alerts),
            "critical_alerts":         len(critical_open),
            "resolved_today":          len(resolved_today),
            "acknowledged_today":      len(acknowledged_today),
            "newest_alert":            newest_alert.to_dict() if newest_alert else None,
            "highest_priority_alert":  highest_priority_alert.to_dict() if highest_priority_alert else None,
        }

    # ------------------------------------------------------------------
    # Enterprise Enhancement 3: Alert Search
    # ------------------------------------------------------------------

    def search_alerts(
        self,
        camera_id:   Optional[str] = None,
        camera_name: Optional[str] = None,
        worker_id:   Optional[str] = None,
        event_type:  Optional[str] = None,
        risk:        Optional[RiskLevel] = None,
        category:    Optional[AlertCategory] = None,
        status:      Optional[AlertStatus] = None,
        keyword:     Optional[str] = None,
        zone_name:   Optional[str] = None,
        limit:       Optional[int] = None,
    ) -> List[dict]:
        """
        Search across the full alert history using any combination of filters.

        - camera_id / worker_id: exact match.
        - camera_name / zone_name: case-insensitive substring match.
        - event_type: case-insensitive substring match.
        - risk / category / status: exact enum match.
        - keyword: case-insensitive substring match against title,
          description, event_type, and recommended_action.
        - limit: optionally cap the number of results (newest first).
        """
        results = self._all_alerts()

        if camera_id:
            results = [a for a in results if a.camera_id == camera_id]
        if camera_name:
            cn = camera_name.lower()
            results = [a for a in results if a.camera_name and cn in a.camera_name.lower()]
        if worker_id:
            results = [a for a in results if a.worker_id == worker_id]
        if event_type:
            et = event_type.lower()
            results = [a for a in results if et in a.event_type.lower()]
        if risk:
            results = [a for a in results if a.severity == risk]
        if category:
            results = [a for a in results if a.category == category]
        if status:
            results = [a for a in results if a.status == status]
        if zone_name:
            zn = zone_name.lower()
            results = [a for a in results if a.zone_name and zn in a.zone_name.lower()]
        if keyword:
            kw = keyword.lower()
            results = [
                a for a in results
                if kw in a.title.lower()
                or kw in a.description.lower()
                or kw in a.event_type.lower()
                or kw in a.recommended_action.lower()
            ]

        results.sort(key=lambda a: a.created_at, reverse=True)
        if limit is not None:
            results = results[:limit]
        return [a.to_dict() for a in results]

    # ------------------------------------------------------------------
    # Enterprise Enhancement 4: Date Filtering
    # ------------------------------------------------------------------

    def filter_alerts_by_date(
        self,
        period:     str = "today",
        start_date: Optional[datetime] = None,
        end_date:   Optional[datetime] = None,
    ) -> List[dict]:
        """
        Filter alert history by a date window, based on created_at.

        period: one of "today", "yesterday", "last_week", "range".
          - "range" requires start_date and/or end_date (datetime objects;
            naive datetimes are treated as local time).
        """
        all_alerts = self._all_alerts()
        period_key = period.lower()

        if period_key == "today":
            lo = self._start_of_day()
            hi = lo + 86400
        elif period_key == "yesterday":
            hi = self._start_of_day()
            lo = hi - 86400
        elif period_key == "last_week":
            hi = self._start_of_day() + 86400
            lo = hi - 7 * 86400
        elif period_key == "range":
            lo = start_date.timestamp() if start_date else 0.0
            hi = end_date.timestamp() if end_date else time.time() + 1
        else:
            raise ValueError(
                f"Unknown period '{period}'. Use 'today', 'yesterday', "
                f"'last_week', or 'range'."
            )

        results = [a for a in all_alerts if lo <= a.created_at < hi]
        results.sort(key=lambda a: a.created_at, reverse=True)
        return [a.to_dict() for a in results]

    # ------------------------------------------------------------------
    # Enterprise Enhancement 5: Alert Trends
    # ------------------------------------------------------------------

    def get_alert_trends(self, days: int = 7) -> List[dict]:
        """
        Return alert counts bucketed by day for the last `days` days
        (oldest first), e.g.:
            [{"date": "2026-06-20", "day_name": "Saturday", "count": 12}, ...]
        Useful for feeding a bar/line chart directly.
        """
        all_alerts = self._all_alerts()
        today_start = self._start_of_day()

        buckets: Dict[str, int] = {}
        day_names: Dict[str, str] = {}
        for i in range(days - 1, -1, -1):
            day_start = today_start - i * 86400
            d = datetime.fromtimestamp(day_start, tz=timezone.utc).date()
            key = d.isoformat()
            buckets[key] = 0
            day_names[key] = d.strftime("%A")

        window_start = today_start - (days - 1) * 86400
        for a in all_alerts:
            if a.created_at < window_start:
                continue
            d = datetime.fromtimestamp(a.created_at, tz=timezone.utc).date()
            key = d.isoformat()
            if key in buckets:
                buckets[key] += 1

        return [
            {"date": key, "day_name": day_names[key], "count": count}
            for key, count in buckets.items()
        ]

    # ------------------------------------------------------------------
    # Enterprise Enhancement 6: Category Summary
    # ------------------------------------------------------------------

    def get_category_summary(self) -> List[dict]:
        """
        Return alert counts grouped by category, sorted highest first:
            [{"category": "PPE", "count": 18}, {"category": "DANGER_ZONE", "count": 6}, ...]
        """
        all_alerts = self._all_alerts()
        counts = Counter(a.category for a in all_alerts)
        ordered = counts.most_common()
        return [{"category": cat.value, "count": count} for cat, count in ordered]

    # ------------------------------------------------------------------
    # Enterprise Enhancement 7: Camera Summary
    # ------------------------------------------------------------------

    def get_camera_summary(self) -> List[dict]:
        """
        Return alert counts grouped by camera, sorted highest first:
            [{"camera_id": "Camera 1", "count": 12}, {"camera_id": "Camera 2", "count": 4}, ...]
        Alerts with no camera_id are grouped under "unknown".
        """
        all_alerts = self._all_alerts()
        counts = Counter(a.camera_id or "unknown" for a in all_alerts)
        ordered = counts.most_common()
        return [{"camera_id": cam, "count": count} for cam, count in ordered]

    # ------------------------------------------------------------------
    # Enterprise Enhancement 8: Alert Export
    # ------------------------------------------------------------------

    def export_alerts(
        self,
        fmt: str = "json",
        alerts: Optional[List[dict]] = None,
    ) -> str:
        """
        Export alerts to CSV or JSON. PDF export is not yet implemented and
        raises NotImplementedError so callers can show a clear "coming soon"
        message rather than failing silently.

        alerts: optionally pass a pre-filtered list of alert dicts (e.g. from
        search_alerts / filter_alerts_by_date). Defaults to full history.
        """
        import csv
        import io
        import json as json_module

        data = alerts if alerts is not None else self.alert_history(limit=len(self._history) or 1)
        fmt_key = fmt.lower()

        if fmt_key == "json":
            return json_module.dumps(data, indent=2, default=str)

        if fmt_key == "csv":
            if not data:
                return ""
            buffer = io.StringIO()
            fieldnames = list(data[0].keys())
            writer = csv.DictWriter(buffer, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                # Flatten non-scalar fields (e.g. history list) for CSV safety.
                flat_row = {
                    k: (json_module.dumps(v) if isinstance(v, (list, dict)) else v)
                    for k, v in row.items()
                }
                writer.writerow(flat_row)
            return buffer.getvalue()

        if fmt_key == "pdf":
            raise NotImplementedError(
                "PDF export is planned but not yet implemented."
            )

        raise ValueError(f"Unsupported export format '{fmt}'. Use 'csv' or 'json'.")

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
        except Exception as exc:
            logger.exception("Alert broadcast failed", exc_info=exc)

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

    def _all_alerts(self) -> List[Alert]:
        """
        Full alert set for analytics: every alert ever created.

        self._history already contains every Alert object created via
        process_detection (including ones still active), so this is simply
        an alias — kept as its own method so analytics helpers don't depend
        on _history's internal shape directly.
        """
        return list(self._history)

    @staticmethod
    def _start_of_day(ts: Optional[float] = None) -> float:
        """Unix timestamp (UTC) for the start of the day containing `ts` (default: now)."""
        moment = datetime.fromtimestamp(ts if ts is not None else time.time(), tz=timezone.utc)
        start = moment.replace(hour=0, minute=0, second=0, microsecond=0)
        return start.timestamp()