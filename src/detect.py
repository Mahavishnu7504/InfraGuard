from src.inference.predictor import Predictor
from src.risk_engine.rules import RiskEngine
from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader

import json
from pathlib import Path
import cv2


class InfraGuardSystem:
    """
    Industry-Grade InfraGuard Controller
    """

    def __init__(self, config_path: str = "configs/model.yaml"):
        self.logger = get_logger("InfraGuardSystem")
        self.config = ConfigLoader(config_path)

        self.model_version = self.config.get("model.version", "unknown")

        self.predictor = Predictor(config_path=config_path)
        self.risk_engine = RiskEngine()

        self.logger.info("InfraGuard system initialized successfully")

    def _annotate_image(self, frame, detections, risk_report):
        """
        Draw bounding boxes and violation info
        """

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            class_id = det["class_id"]

            if class_id == 0:
                label = "Person"
                color = (255, 0, 0)
            elif class_id == 1:
                label = "Helmet"
                color = (0, 255, 0)
            elif class_id == 2:
                label = "Vest"
                color = (0, 255, 255)
            else:
                label = "Unknown"
                color = (200, 200, 200)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Add overall severity text
        severity = risk_report["severity"]
        severity_color = (0, 255, 0) if severity == "LOW" else (0, 0, 255)

        cv2.putText(
            frame,
            f"Severity: {severity}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            severity_color,
            3,
        )

        return frame

    def process_frame(
        self,
        frame,
        image_name="result.jpg",
        save_output=False,
        save_image=False,
    ):
        """
        Full pipeline processing
        """

        detections = self.predictor.predict(frame)
        risk_report = self.risk_engine.evaluate(detections)

        response = {
            "model_version": self.model_version,
            "detections": detections,
            "risk_report": risk_report,
        }

        # Save JSON
        if save_output:
            json_dir = Path("inference/outputs/json")
            json_dir.mkdir(parents=True, exist_ok=True)

            json_path = json_dir / f"{image_name}.json"

            with open(json_path, "w") as f:
                json.dump(response, f, indent=4)

        # Save annotated image
        if save_image:
            img_dir = Path("inference/outputs/ppe_results")
            img_dir.mkdir(parents=True, exist_ok=True)

            annotated = self._annotate_image(
                frame.copy(), detections, risk_report
            )

            img_path = img_dir / image_name
            cv2.imwrite(str(img_path), annotated)

        return response