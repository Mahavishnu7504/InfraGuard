from ultralytics import YOLO


class YOLOPredictor:
    def __init__(self, model_path: str, conf: float = 0.25, imgsz: int = 416):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def predict(self, source: str, save: bool, project: str, name: str):
        return self.model.predict(
            source=source,
            conf=self.conf,
            imgsz=self.imgsz,
            save=save,
            project=project,
            name=name,
            exist_ok=True
        )