

from ultralytics import YOLO
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger


def main():
    """
    Industry-grade training entrypoint
    - Config driven
    - Logged
    - Reproducible
    """

    logger = get_logger("Training")

    # Load training config
    config = ConfigLoader("configs/training.yaml")

    base_weights = config.get("model.base_weights")
    dataset_yaml = config.get("data.dataset_yaml")

    epochs = config.get("training.epochs")
    imgsz = config.get("training.imgsz")
    batch = config.get("training.batch")
    device = config.get("training.device")
    project = config.get("training.project")
    name = config.get("training.name")
    workers = config.get("training.workers", 0)
    resume = config.get("training.resume", False)

    logger.info("Starting InfraGuard Training")
    logger.info(f"Base weights: {base_weights}")
    logger.info(f"Dataset: {dataset_yaml}")
    logger.info(f"Epochs: {epochs}, Batch: {batch}, Img Size: {imgsz}")
    logger.info(f"Device: {device}")

    # Initialize YOLO model
    model = YOLO(base_weights)

    # Start training
    model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        workers=workers,
        resume=resume,
        cache=False,
        verbose=True,
    )

    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()