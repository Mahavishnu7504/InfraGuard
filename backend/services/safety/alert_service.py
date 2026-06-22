import time

class AlertManager:
    def __init__(self, cooldown=10):
        self.last_alert_time = {}
        self.cooldown = cooldown

    def should_alert(self, worker_id, risk):
        """
        Smart alert logic:
        - Only HIGH / MEDIUM
        - Cooldown per worker
        """

        if risk not in ["high", "medium"]:
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