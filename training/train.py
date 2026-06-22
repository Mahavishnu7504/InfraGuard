from ultralytics import YOLO
from ai_engine.utils.config_loader import ConfigLoader
from ai_engine.utils.logger import get_logger


def main():

    logger = get_logger("Training")

    config = ConfigLoader("configs/training.yaml")

    base_weights = config.get("model.base_weights")
    dataset_yaml = config.get("data.dataset_yaml")

    epochs = config.get("training.epochs")
    imgsz = config.get("training.imgsz")
    batch = config.get("training.batch")
    device = config.get("training.device")

    project = config.get("training.project")
    name = config.get("training.name")

    logger.info("Starting InfraGuard Training")

    model = YOLO(base_weights)

    model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name
    )

    logger.info("Training completed")


if __name__ == "__main__":
    main()