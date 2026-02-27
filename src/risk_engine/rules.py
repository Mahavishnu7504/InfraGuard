from src.utils.logger import get_logger


class RiskEngine:
    def __init__(self):
        self.logger = get_logger("RiskEngine")

    def evaluate(self, detections):
        persons = [d for d in detections if d["class_id"] == 0]
        helmets = [d for d in detections if d["class_id"] == 1]
        vests = [d for d in detections if d["class_id"] == 2]

        risk_report = {
            "total_persons": len(persons),
            "helmet_violations": 0,
            "vest_violations": 0,
            "severity": "LOW"
        }

        if persons and not helmets:
            risk_report["helmet_violations"] = len(persons)

        if persons and not vests:
            risk_report["vest_violations"] = len(persons)

        if risk_report["helmet_violations"] > 0:
            risk_report["severity"] = "HIGH"

        self.logger.info(f"Risk evaluated: {risk_report}")

        return risk_report