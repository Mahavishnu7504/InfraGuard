from src.utils.config_loader import load_config
from src.inference.predictor import Predictor
from src.risk_engine.rules import RiskEngine
import os

# Load config
model_cfg = load_config("configs/model.yaml")

# Initialize predictor
predictor = Predictor(
    model_path=model_cfg["model"]["weights_path"],
    confidence=model_cfg["model"]["confidence_threshold"]
)

# Initialize risk engine
risk_engine = RiskEngine()

# ---------------- CONFIG ----------------
image_path = "data/test_images/site_1.jpg"
DEMO_MODE = True   # 🔥 turn ON/OFF here
# --------------------------------------

if DEMO_MODE:
    # Simulated detections (for PPE demo)
    detections = ['person', 'helmet', 'vest', 'person', 'helmet']
    print("DEMO MODE detections:", detections)
else:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    detections = predictor.predict(image_path)
    print("YOLO Detections:", detections)

# Evaluate safety risks
risks = risk_engine.evaluate(detections)
print("Detected Risks:", risks)