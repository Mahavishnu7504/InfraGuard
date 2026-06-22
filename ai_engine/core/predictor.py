from ultralytics import YOLO
from pathlib import Path
import torch


class InfraGuardPredictor:
    def __init__(self, model_name="infraguard.pt"):
        # ==============================
        # PATH SETUP
        # ==============================
        project_root = Path(__file__).resolve().parents[2]
        model_path = project_root / "models" / model_name

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # ==============================
        # DEVICE SELECTION
        # ==============================
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[YOLO] Loading model on: {self.device}")

        self.model = YOLO(str(model_path))

        # ==============================
        # GPU OPTIMIZATION
        # ==============================
        if self.device == "cuda":
            self.model.to("cuda")
            self.model.fuse()  # speed boost

        print(f"[YOLO] Loaded model from: {model_path}")

    # ==============================
    # SINGLE FRAME PREDICTION
    # ==============================
    def predict_frame(self, frame):
        """
        Returns standardized detections
        """

        # 🔥 Resize for speed (balanced)
        frame = self._preprocess(frame)

        results = self.model(
            frame,
            device=self.device,
            conf=0.4,
            iou=0.5,
            verbose=False
        )

        return self._parse_results(results)

    # ==============================
    # BATCH PREDICTION (ADVANCED)
    # ==============================
    def predict_batch(self, frames):
        """
        Batch inference for performance boost
        """
        frames = [self._preprocess(f) for f in frames]

        results = self.model(
            frames,
            device=self.device,
            conf=0.4,
            iou=0.5,
            verbose=False
        )

        batch_outputs = []
        for res in results:
            batch_outputs.append(self._parse_single(res))

        return batch_outputs

    # ==============================
    # PREPROCESS
    # ==============================
    def _preprocess(self, frame):
        """
        Resize frame for faster inference
        """
        import cv2
        return cv2.resize(frame, (640, 480))

    # ==============================
    # PARSE MULTI RESULTS
    # ==============================
    def _parse_results(self, results):
        detections = []

        for result in results:
            detections.extend(self._parse_single(result))

        return detections

    # ==============================
    # PARSE SINGLE RESULT
    # ==============================
    def _parse_single(self, result):
        detections = []

        if result.boxes is None:
            return detections

        names = result.names

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "class_id": cls,
                "class_name": str(names[cls]).lower().strip(),
                "confidence": round(conf, 2)
            })

        return detections