class RiskEngine:
    def __init__(self, required_ppe=None):
        self.required_ppe = required_ppe or ["helmet", "vest"]

    def evaluate(self, detections):
        risks = []
        for ppe in self.required_ppe:
            if ppe not in detections:
                risks.append(f"{ppe.capitalize()} Missing")
        return risks