import cv2
from ai_engine.pipelines.pipeline import InfraGuardPipeline


def main():

    image_path = "data/test_images/test1.jpg"

    frame = cv2.imread(image_path)

    if frame is None:
        raise RuntimeError("Image not found")

    pipeline = InfraGuardPipeline()

    result = pipeline.process_frame(frame)

    print("\nDetections:", result["detections"])
    print("\nRisk Summary:", result["ppe_report"])


if __name__ == "__main__":
    main()