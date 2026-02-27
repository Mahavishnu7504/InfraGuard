from ultralytics import YOLO
from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader


class Predictor:
    """
    Industry-grade Predictor
    - Config driven
    - Device aware
    - Structured output
    """

    def __init__(self, config_path: str = "configs/model.yaml"):
        self.logger = get_logger("Predictor")
        self.config = ConfigLoader(config_path)

        # Load model configuration
        weights_path = self.config.get("model.paths.weights")
        device_type = self.config.get("model.device.type", "cpu")

        self.conf_threshold = self.config.get(
            "inference.confidence_threshold", 0.4
        )
        self.iou_threshold = self.config.get(
            "inference.iou_threshold", 0.5
        )
        self.max_det = self.config.get(
            "inference.max_detections", 100
        )

        # Load model
        self.model = YOLO(weights_path)

        self.logger.info(f"Model loaded from: {weights_path}")
        self.logger.info(f"Device: {device_type}")
        self.logger.info(
            f"Inference settings -> conf: {self.conf_threshold}, "
            f"iou: {self.iou_threshold}, max_det: {self.max_det}"
        )

        self.device_type = device_type

    def predict(self, frame):
        """
        Run inference on a single frame
        """

        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            device=self.device_type
        )

        detections = []

        for r in results:
            for box in r.boxes:
                detections.append({
                    "class_id": int(box.cls),
                    "confidence": float(box.conf),
                    "bbox": box.xyxy.tolist()[0]
                })

        return detections