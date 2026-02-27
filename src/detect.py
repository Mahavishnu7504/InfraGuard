from src.inference.predictor import Predictor
from src.risk_engine.rules import RiskEngine
from src.utils.config_loader import ConfigLoader


class InfraGuardSystem:
    def __init__(self, config_path="configs/model.yaml"):
        config = ConfigLoader(config_path)
        model_path = config.get("model.path")

        self.predictor = Predictor(model_path)
        self.risk_engine = RiskEngine()

    def process_frame(self, frame):
        detections = self.predictor.predict(frame)
        risk_report = self.risk_engine.evaluate(detections)

        return {
            "detections": detections,
            "risk_report": risk_report
        }