# src/training/train.py

from ultralytics import YOLO
from src.utils.config_loader import load_config


def main():
    cfg = load_config("configs/training.yaml")

    # Load base YOLO weights
    model = YOLO(cfg["model"]["base_weights"])

    # Train
    model.train(
        data=cfg["data"]["dataset_yaml"],
        epochs=cfg["training"]["epochs"],
        imgsz=cfg["training"]["imgsz"],
        batch=cfg["training"]["batch"],
        device=cfg["training"]["device"],
        project=cfg["training"]["project"],
        name=cfg["training"]["name"]
    )


if __name__ == "__main__":
    main()