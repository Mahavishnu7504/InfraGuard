class RiskEngine:
    def __init__(self, helmet_required=True, vest_required=True):
        self.helmet_required = helmet_required
        self.vest_required = vest_required

    def evaluate(self, detections):
        risks = []

        if self.helmet_required and "helmet" not in detections:
            risks.append("Helmet Missing")

        if self.vest_required and "vest" not in detections:
            risks.append("Safety Vest Missing")

        return risks