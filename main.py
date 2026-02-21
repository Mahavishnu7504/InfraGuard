from src.utils.config_loader import load_config
from src.inference.predictor import Predictor
from src.risk_engine.rules import RiskEngine

model_cfg = load_config("configs/model.yaml")
deploy_cfg = load_config("configs/deployment.yaml")

predictor = Predictor(
    model_path=model_cfg["model"]["weights_path"],
    confidence=model_cfg["model"]["confidence_threshold"]
)

risk_engine = RiskEngine()

# Example usage
detections = ["person"]  # replace with real output parsing
risks = risk_engine.evaluate(detections)

print("Detected Risks:", risks)