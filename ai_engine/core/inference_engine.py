from ultralytics import YOLO

class InferenceEngine:

    def __init__(self, model_path="models/infraguard.pt"):

        print("Loading InfraGuard model...")
        self.model = YOLO(model_path)

    def predict(self, image):

        results = self.model(image)

        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "class_id": cls,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })

        return detections