from ultralytics import YOLO
from pathlib import Path


class QualityModel:
    def __init__(self, model_name="quality.pt"):
        root = Path(__file__).resolve().parents[2]
        model_path = root / "models" / model_name

        if not model_path.exists():
            print("[WARNING] Quality model not found, fallback mode enabled")
            self.model = None
        else:
            self.model = YOLO(str(model_path))

    def predict(self, image):
        if self.model is None:
            return []

        results = self.model(image, verbose=False)

        detections = []
        for r in results:
            if r.boxes is None:
                continue

            names = r.names

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "label": names[cls],
                    "confidence": round(conf, 2)
                })

        return detections