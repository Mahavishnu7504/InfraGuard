import sys
from pathlib import Path
import cv2

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add project root to Python path
sys.path.append(str(PROJECT_ROOT))

from ai_engine.pipelines.pipeline import InfraGuardPipeline


def test_pipeline():

    pipeline = InfraGuardPipeline()

    image_path = PROJECT_ROOT / "data" / "test_images" / "test2.jpg"

    frame = cv2.imread(str(image_path))

    if frame is None:
        raise RuntimeError(f"Unable to read image: {image_path}")

    result = pipeline.process_frame(frame)

    print("\n========== InfraGuard Detection ==========")

    print("\nDetections:")
    print(result["detections"])

    print("\nPPE Report:")
    print(result["ppe_report"])

    print("\nProximity Alerts:")
    print(result["proximity_alerts"])

    print("\nDanger Zone Alerts:")
    print(result["danger_zone_alerts"])

    print("\n==========================================")

    cv2.imshow("InfraGuard Test", result["output_image"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_pipeline()