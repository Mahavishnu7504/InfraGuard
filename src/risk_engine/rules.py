class RiskEngine:
    def __init__(self, required_ppe=None):
        self.required_ppe = required_ppe or ["helmet", "vest"]

    def evaluate(detections):
     persons = [d for d in detections if d["class"] == 0]
     helmets = [d for d in detections if d["class"] == 1]
     vests = [d for d in detections if d["class"] == 2]

     risks = []

     if persons and not helmets:
        risks.append("NO_HELMET")

     if persons and not vests:
        risks.append("NO_VEST")

     return risks