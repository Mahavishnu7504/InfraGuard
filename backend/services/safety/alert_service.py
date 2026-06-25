import time
from enum import Enum


class RiskLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


ALERTABLE_RISKS = {RiskLevel.HIGH, RiskLevel.MEDIUM}


class AlertManager:
    def __init__(self, cooldown=10):
        self.last_alert_time = {}
        self.cooldown = cooldown

    def should_alert(self, worker_id, risk: RiskLevel):
        """
        Smart alert logic:
        - Only HIGH / MEDIUM
        - Cooldown per worker
        """

        if risk not in ALERTABLE_RISKS:
            return False

        now = time.time()
        last_time = self.last_alert_time.get(worker_id)

        if last_time is None:
            self.last_alert_time[worker_id] = now
            return True

        if now - last_time > self.cooldown:
            self.last_alert_time[worker_id] = now
            return True

        return False