# src/training/train.py

from ultralytics import YOLO
from src.utils.config_loader import load_config


def main():
    # Load training configuration
    cfg = load_config("configs/training.yaml")

    # Load base YOLOv8 weights
    model = YOLO(cfg["model"]["base_weights"])

    # Start training
    model.train(
        data=cfg["data"]["dataset_yaml"],
        epochs=cfg["training"]["epochs"],
        imgsz=cfg["training"]["imgsz"],
        batch=cfg["training"]["batch"],
        device=cfg["training"]["device"],
        project=cfg["training"]["project"],
        name=cfg["training"]["name"],
        workers=cfg["training"].get("workers", 0),
        cache=False,          # avoids RAM pressure
        verbose=True
    )


if __name__ == "__main__":
    main()