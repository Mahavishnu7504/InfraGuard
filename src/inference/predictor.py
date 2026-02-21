from ultralytics import YOLO

class Predictor:
    def __init__(self, model_path: str, confidence: float):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def predict(self, image):
        results = self.model(image, conf=self.confidence)
        return results