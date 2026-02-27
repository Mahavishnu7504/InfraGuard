from src.inference.predictor import Predictor
from src.risk_engine.rules import RiskEngine
from src.utils.logger import get_logger


class InfraGuardSystem:
    """
    Central Orchestrator
    Connects:
        - Predictor (AI layer)
        - RiskEngine (Business logic)
    Returns structured system output
    """

    def __init__(self, config_path: str = "configs/model.yaml"):
        self.logger = get_logger("InfraGuardSystem")

        self.predictor = Predictor(config_path=config_path)
        self.risk_engine = RiskEngine()

        self.logger.info("InfraGuard system initialized successfully")

    def process_frame(self, frame):
        """
        Process single image/frame
        """

        detections = self.predictor.predict(frame)
        risk_report = self.risk_engine.evaluate(detections)

        response = {
            "detections": detections,
            "risk_report": risk_report
        }

        return response