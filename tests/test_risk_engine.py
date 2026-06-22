import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from backend.services.risk_engine.rules import evaluate_risk


def test_risk_engine():

    detections = [
        {"class_name": "person", "bbox": [100, 100, 200, 300]},
        {"class_name": "helmet", "bbox": [120, 90, 180, 150]},
        {"class_name": "forklift", "bbox": [250, 100, 400, 300]}
    ]

    result = evaluate_risk(detections)

    print("Risk Evaluation:")
    print(result)


if __name__ == "__main__":
    test_risk_engine()